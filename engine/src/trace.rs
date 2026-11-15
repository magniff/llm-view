//! [`Trace`]: a hook that records a whole generation session, indexed by
//! absolute position.
//!
//! The prefill call delivers rows for positions `0..T`; each decode step
//! delivers one more row. Because every hook knows its `pos0`, all calls fold
//! into one consistent history, as if the whole sequence had been run at once
//! (which, thanks to the KV cache, it effectively was).
//!
//! It also carries one intervention, **head ablation**: zeroing a head's
//! output before `o_proj`, so you can see what changes without it.
//!
//! [`logit_lens`] reads the residual stream at intermediate layers as if it
//! were the final layer: apply the final RMSNorm and the unembedding to it and
//! see which token it "currently" predicts.

use std::collections::HashSet;

use crate::hooks::{Hooks, Site};
use crate::model::Model;
use crate::ops::{linear_serial, n_threads, rmsnorm};
use crate::tensor::Tensor;

/// Top logits of one position, enough to rebuild the head of the distribution.
#[derive(Clone, Debug, Default)]
pub struct TopLogits {
    /// `(token id, raw logit)`, highest first.
    pub top: Vec<(u32, f32)>,
    /// log Σ exp(logits) over the whole vocabulary: `p(i) = exp(logit_i − lse)`.
    pub lse: f32,
}

pub const TOP_LOGITS: usize = 100;

#[derive(Default)]
pub struct Trace {
    /// Residual stream snapshots `[stage][pos] → [hidden]`.
    /// Stage 0 = embeddings, `2l+1` = after layer l's attention, `2l+2` = after layer l's MLP.
    pub resid: Vec<Vec<Vec<f32>>>,
    /// Attention probabilities `[layer][head][pos] → [pos + 1]` (row of the causal matrix).
    pub attn: Vec<Vec<Vec<Vec<f32>>>>,
    /// ‖each head's output slice‖ before `o_proj`: `[layer][pos][head]`.
    pub head_norm: Vec<Vec<Vec<f32>>>,
    /// ‖what attention adds to the residual‖: `[layer][pos]`.
    pub attn_write: Vec<Vec<f32>>,
    /// ‖what the MLP adds to the residual‖: `[layer][pos]`.
    pub mlp_write: Vec<Vec<f32>>,
    /// MLP hidden activations `silu(gate) * up`: `[layer][pos] → [intermediate]`.
    pub mlp_hidden: Vec<Vec<Vec<f32>>>,
    /// Output distribution at each position (predicting position + 1).
    pub logits: Vec<TopLogits>,
    /// Heads to zero out, as `(layer, head)`.
    pub ablate: HashSet<(usize, usize)>,
    head_dim: usize,
}

fn put<T: Default>(v: &mut Vec<T>, i: usize, x: T) {
    if v.len() <= i {
        v.resize_with(i + 1, T::default);
    }
    v[i] = x;
}

fn at<T: Default>(v: &mut Vec<T>, i: usize) -> &mut T {
    if v.len() <= i {
        v.resize_with(i + 1, T::default);
    }
    &mut v[i]
}

fn norm(x: &[f32]) -> f32 {
    x.iter().map(|v| v * v).sum::<f32>().sqrt()
}

impl Trace {
    pub fn new(model: &Model) -> Self {
        Self { head_dim: model.cfg.head_dim, ..Default::default() }
    }

    /// Number of positions recorded.
    pub fn len(&self) -> usize {
        self.resid.first().map_or(0, |s| s.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Forget positions `len..` (to rewind, alongside `KvCache::truncate`).
    pub fn truncate(&mut self, len: usize) {
        self.resid.iter_mut().for_each(|s| s.truncate(len));
        self.attn.iter_mut().flatten().for_each(|h| h.truncate(len));
        self.head_norm.iter_mut().for_each(|l| l.truncate(len));
        self.attn_write.iter_mut().for_each(|l| l.truncate(len));
        self.mlp_write.iter_mut().for_each(|l| l.truncate(len));
        self.mlp_hidden.iter_mut().for_each(|l| l.truncate(len));
        self.logits.truncate(len);
    }

    fn rows(&mut self, stage: usize, pos0: usize, t: &Tensor) {
        let s = at(&mut self.resid, stage);
        for r in 0..t.rows() {
            put(s, pos0 + r, t.row(r).to_vec());
        }
    }
}

impl Hooks for Trace {
    fn on(&mut self, site: &Site, t: &mut Tensor) {
        let p0 = site.pos0;
        let Some(l) = site.layer else {
            match site.name {
                "embed.out" => self.rows(0, p0, t),
                "logits" => {
                    for r in 0..t.rows() {
                        put(&mut self.logits, p0 + r, top_logits(t.row(r), TOP_LOGITS));
                    }
                }
                _ => {}
            }
            return;
        };
        match site.name {
            "resid.post_attn" => self.rows(2 * l + 1, p0, t),
            "resid.post_mlp" => self.rows(2 * l + 2, p0, t),
            "attn.probs" => {
                let (h, tn, s) = (t.shape[0], t.shape[1], t.shape[2]);
                let layer = at(&mut self.attn, l);
                for hh in 0..h {
                    let head = at(layer, hh);
                    for r in 0..tn {
                        let pos = p0 + r;
                        put(head, pos, t.data[(hh * tn + r) * s..][..pos + 1].to_vec());
                    }
                }
            }
            "attn.out" => {
                let hd = self.head_dim;
                let n_heads = t.cols() / hd;
                for r in 0..t.rows() {
                    let row = t.row_mut(r);
                    for hh in 0..n_heads {
                        if self.ablate.contains(&(l, hh)) {
                            row[hh * hd..(hh + 1) * hd].fill(0.0);
                        }
                    }
                    let norms = (0..n_heads).map(|hh| norm(&row[hh * hd..(hh + 1) * hd])).collect();
                    put(at(&mut self.head_norm, l), p0 + r, norms);
                }
            }
            "attn.o_proj" => {
                for r in 0..t.rows() {
                    put(at(&mut self.attn_write, l), p0 + r, norm(t.row(r)));
                }
            }
            "mlp.down" => {
                for r in 0..t.rows() {
                    put(at(&mut self.mlp_write, l), p0 + r, norm(t.row(r)));
                }
            }
            "mlp.hidden" => {
                for r in 0..t.rows() {
                    put(at(&mut self.mlp_hidden, l), p0 + r, t.row(r).to_vec());
                }
            }
            _ => {}
        }
    }
}

pub fn top_logits(logits: &[f32], k: usize) -> TopLogits {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let lse = max + logits.iter().map(|&x| (x - max).exp()).sum::<f32>().ln();
    let mut idx: Vec<(u32, f32)> = logits.iter().enumerate().map(|(i, &x)| (i as u32, x)).collect();
    let k = k.min(idx.len());
    let by_desc = |a: &(u32, f32), b: &(u32, f32)| b.1.total_cmp(&a.1);
    if k < idx.len() {
        idx.select_nth_unstable_by(k, by_desc);
        idx.truncate(k);
    }
    idx.sort_unstable_by(by_desc);
    TopLogits { top: idx, lse }
}

/// One cell of the logit lens.
#[derive(Clone, Debug)]
pub struct LensCell {
    /// `(token id, probability)`, highest first.
    pub top: Vec<(u32, f32)>,
    /// `(probability, rank)` of each requested token, in request order (rank 0 = top).
    pub tracked: Vec<(f32, usize)>,
    /// Entropy of the distribution in bits: ~0 = certain, ~15.6 = uniform over 49k.
    pub entropy: f32,
}

/// Decode residual-stream vectors as if each were the last layer's output.
/// `track[i]` lists tokens whose probability and rank to report for row `i`,
/// e.g. the token that actually came next, or the model's final answer, so
/// you can follow one token's fate through the layers even when it's far
/// outside the top k.
///
/// Each row costs a full vocabulary projection (49k × 576 for SmolLM2-135M), so
/// rows are split across threads, and each thread does its own matmul *and* the
/// softmax / entropy / top-k over its 49k logits.
pub fn logit_lens(model: &Model, rows: &[&[f32]], track: &[Vec<u32>], k: usize) -> Vec<LensCell> {
    let n = rows.len();
    let threads = n_threads().min(n.div_ceil(8)).max(1);
    if threads == 1 {
        return lens_rows(model, rows, track, k); // also the only option in WebAssembly
    }
    let per = n.div_ceil(threads);
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|i| (i * per, ((i + 1) * per).min(n)))
            .filter(|(lo, hi)| lo < hi)
            .map(|(lo, hi)| s.spawn(move || lens_rows(model, &rows[lo..hi], &track[lo..hi], k)))
            .collect();
        handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
    })
}

fn lens_rows(model: &Model, rows: &[&[f32]], track: &[Vec<u32>], k: usize) -> Vec<LensCell> {
    let d = model.cfg.hidden;
    let mut out = Vec::with_capacity(rows.len());
    // 32 rows × 49k vocab × 4 bytes = 6.3 MB of logits at a time; each weight row
    // is reused for all 32 while it's in cache.
    for (chunk, tchunk) in rows.chunks(32).zip(track.chunks(32)) {
        let mut x = Tensor::zeros(vec![chunk.len(), d]);
        for (i, r) in chunk.iter().enumerate() {
            x.row_mut(i).copy_from_slice(r);
        }
        let logits = linear_serial(&rmsnorm(&x, &model.final_norm, model.cfg.norm_eps), model.unembed());
        for (i, ids) in tchunk.iter().enumerate() {
            let row = logits.row(i);
            let tl = top_logits(row, k);
            let p = |x: f32| (x - tl.lse).exp();
            let entropy = -row
                .iter()
                .map(|&x| {
                    let pi = p(x);
                    if pi > 0.0 { pi * pi.log2() } else { 0.0 }
                })
                .sum::<f32>();
            let tracked = ids
                .iter()
                .map(|&id| {
                    let lt = row[id as usize];
                    (p(lt), row.iter().filter(|&&x| x > lt).count())
                })
                .collect();
            out.push(LensCell { top: tl.top.iter().map(|&(id, x)| (id, p(x))).collect(), tracked, entropy });
        }
    }
    out
}
