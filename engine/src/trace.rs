//! [`Trace`]: a hook that records the residual stream of a whole session,
//! indexed by absolute position.
//!
//! The prefill call delivers rows for positions `0..T`; each decode step
//! delivers one more row. Because every hook knows its `pos0`, all calls fold
//! into one consistent history, as if the whole sequence had been run at once
//! (which, thanks to the KV cache, it effectively was).
//!
//! Only the stream between blocks is kept (~140 KB per position). Everything
//! inside a layer is recomputed on demand from it and the KV cache, see
//! [`crate::walk`].

use crate::hooks::{Hooks, Site};
use crate::model::Model;
use crate::tensor::Tensor;

/// Top logits of one position, enough to rebuild the head of the distribution.
#[derive(Clone, Debug, Default)]
pub struct TopLogits {
    /// `(token id, raw logit)`, highest first.
    pub top: Vec<(u32, f32)>,
    /// log Σ exp(logits) over the whole vocabulary: `p(i) = exp(logit_i − lse)`.
    pub lse: f32,
}

#[derive(Default)]
pub struct Trace {
    /// Residual stream snapshots `[stage][pos] → [hidden]`.
    /// Stage 0 = embeddings, `2l+1` = after layer l's attention, `2l+2` = after layer l's MLP.
    pub resid: Vec<Vec<Vec<f32>>>,
}

impl Trace {
    pub fn new(model: &Model) -> Self {
        Self { resid: vec![vec![]; 2 * model.cfg.n_layers + 1] }
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
    }

    fn rows(&mut self, stage: usize, pos0: usize, t: &Tensor) {
        let s = &mut self.resid[stage];
        for r in 0..t.rows() {
            let pos = pos0 + r;
            if s.len() <= pos {
                s.resize_with(pos + 1, Vec::new);
            }
            s[pos] = t.row(r).to_vec();
        }
    }
}

impl Hooks for Trace {
    fn on(&mut self, site: &Site, t: &mut Tensor) {
        let p0 = site.pos0;
        match (site.name, site.layer) {
            ("embed.out", None) => self.rows(0, p0, t),
            ("resid.post_attn", Some(l)) => self.rows(2 * l + 1, p0, t),
            ("resid.post_mlp", Some(l)) => self.rows(2 * l + 2, p0, t),
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
