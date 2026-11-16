//! The Flow view: follow one position through the network in full detail.
//!
//! The [`Trace`] keeps only what every view needs for every position (residual
//! snapshots, attention maps, …). Here we need *everything* for one position
//! in one layer: q, k, v before and after RoPE, raw scores, each head's output,
//! gate/up/act/hidden of the MLP, and so on. Rather than recording all of that
//! for every position, [`layer_flow`] recomputes it on demand from two things
//! the session already holds:
//!
//! - the residual stream entering the layer at that position (from the trace);
//! - the keys and values of earlier positions (from the KV cache).
//!
//! That is exactly the input a decode step would see, so the recomputation is
//! the forward pass for that one position, and its outputs must match the
//! trace's residual snapshots (the tests check this).
//!
//! Two decompositions say *where* a write came from:
//!
//! - **source positions**: attention's write is linear in the attended values,
//!   `Δ = Σ_h Wo_h Σ_j p_hj v_j`, so it splits exactly into one term per
//!   (head, source). Projecting each term on the direction of `Δ` gives
//!   numbers that add up to ‖Δ‖: "how much of this write came from token j".
//! - **direct logit attribution**: the final logits are `E · (g ⊙ x / rms)`
//!   and `x` is a sum of writes, so with `rms` frozen at its actual value each
//!   write has an exact, additive effect on every logit.
//!
//! [`Trace`]: crate::trace::Trace

use std::collections::HashSet;

use crate::kv_cache::KvCache;
use crate::model::Model;
use crate::ops::{dot, linear, rmsnorm, rope, silu, softmax};
use crate::tensor::Tensor;

/// Everything one position computes in one layer.
pub struct LayerFlow {
    /// Residual stream entering the layer.
    pub x_in: Vec<f32>,
    /// `sqrt(mean(x²) + eps)` of `x_in`: what the attention RMSNorm divides by.
    pub attn_rms: f32,
    /// RMSNorm output, the attention's input.
    pub xn: Vec<f32>,
    /// `[H*D]` queries before and after RoPE.
    pub q_pre: Vec<f32>,
    pub q: Vec<f32>,
    /// `[KVH*D]` this position's keys (before/after RoPE) and values.
    pub k_pre: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    /// `[head][source]` scaled scores `q·k/√D` and their softmax, over sources `0..=pos`.
    pub scores: Vec<Vec<f32>>,
    pub probs: Vec<Vec<f32>>,
    pub ablated: Vec<bool>,
    /// `[H*D]` each head's weighted sum of values (zero for ablated heads).
    pub head_out: Vec<f32>,
    /// `[head] → [hidden]` each head's share of the attention write, `Wo_h · out_h`.
    pub head_write: Vec<Vec<f32>>,
    /// `[head][source]` each (head, source) term projected on the direction of
    /// `d_attn`. All of them add up to ‖d_attn‖.
    pub src_proj: Vec<Vec<f32>>,
    /// What attention adds to the residual stream.
    pub d_attn: Vec<f32>,
    pub x_mid: Vec<f32>,
    pub mlp_rms: f32,
    pub xn2: Vec<f32>,
    /// `[intermediate]` MLP internals.
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub act: Vec<f32>,
    pub hidden: Vec<f32>,
    /// ‖column i of W_down‖: how big a write neuron i makes per unit of activation.
    pub down_norm: Vec<f32>,
    /// `hidden_i · (W_down[:, i] · d̂_mlp)`: each neuron's part of ‖d_mlp‖ (they add up to it).
    pub neuron_proj: Vec<f32>,
    /// What the MLP adds to the residual stream.
    pub d_mlp: Vec<f32>,
    pub x_out: Vec<f32>,
}

pub fn rms(x: &[f32], eps: f32) -> f32 {
    (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt()
}

pub fn norm(x: &[f32]) -> f32 {
    x.iter().map(|v| v * v).sum::<f32>().sqrt()
}

fn row(v: &[f32]) -> Tensor {
    Tensor::new(vec![1, v.len()], v.to_vec())
}

/// `Wᵀ · u` for `W: [out, in]`: which input direction produces output direction `u`.
fn t_matvec(w: &Tensor, u: &[f32]) -> Vec<f32> {
    let (n_out, n_in) = (w.shape[0], w.shape[1]);
    let mut y = vec![0.0f32; n_in];
    for (r, &ur) in u.iter().enumerate().take(n_out) {
        if ur != 0.0 {
            for (yi, wi) in y.iter_mut().zip(&w.data[r * n_in..(r + 1) * n_in]) {
                *yi += ur * wi;
            }
        }
    }
    y
}

/// `v / ‖v‖`, or zeros for a zero vector.
fn unit(v: &[f32]) -> Vec<f32> {
    let n = norm(v);
    v.iter().map(|x| if n > 0.0 { x / n } else { 0.0 }).collect()
}

/// Recompute position `pos` through layer `layer`, given the residual stream
/// entering it. `cache` must hold that layer's keys and values for `0..pos`.
pub fn layer_flow(
    model: &Model,
    cache: &KvCache,
    x_in: &[f32],
    layer: usize,
    pos: usize,
    ablate: &HashSet<(usize, usize)>,
) -> LayerFlow {
    let c = &model.cfg;
    let l = &model.layers[layer];
    let (h, kvh, hd, eps) = (c.n_heads, c.n_kv_heads, c.head_dim, c.norm_eps);
    let group = h / kvh;

    // ---- attention ----
    let xn = rmsnorm(&row(x_in), &l.attn_norm, eps).data;
    let xt = row(&xn);
    let q_pre = linear(&xt, &l.wq).data;
    let k_pre = linear(&xt, &l.wk).data;
    let v = linear(&xt, &l.wv).data;
    let mut q = q_pre.clone();
    let mut k = k_pre.clone();
    for hh in 0..h {
        rope(&mut q[hh * hd..(hh + 1) * hd], pos, &model.inv_freq);
    }
    for hh in 0..kvh {
        rope(&mut k[hh * hd..(hh + 1) * hd], pos, &model.inv_freq);
    }
    // Earlier positions come from the cache; this one from what we just computed.
    let key = |sp: usize| if sp == pos { &k[..] } else { cache.k_row(layer, sp) };
    let val = |sp: usize| if sp == pos { &v[..] } else { cache.v_row(layer, sp) };

    let scale = 1.0 / (hd as f32).sqrt();
    let ablated: Vec<bool> = (0..h).map(|hh| ablate.contains(&(layer, hh))).collect();
    let mut scores = Vec::with_capacity(h);
    let mut probs = Vec::with_capacity(h);
    let mut head_out = vec![0.0f32; h * hd];
    for hh in 0..h {
        let kh = hh / group;
        let qv = &q[hh * hd..(hh + 1) * hd];
        let s: Vec<f32> = (0..=pos).map(|sp| dot(qv, &key(sp)[kh * hd..(kh + 1) * hd]) * scale).collect();
        let mut p = s.clone();
        softmax(&mut p);
        if !ablated[hh] {
            let out = &mut head_out[hh * hd..(hh + 1) * hd];
            for (sp, &w) in p.iter().enumerate() {
                for (o, vi) in out.iter_mut().zip(&val(sp)[kh * hd..(kh + 1) * hd]) {
                    *o += w * vi;
                }
            }
        }
        scores.push(s);
        probs.push(p);
    }
    let d_attn = linear(&row(&head_out), &l.wo).data;
    // Per head: the same projection with every other head's slice zeroed.
    let mut masked = Tensor::zeros(vec![h, h * hd]);
    for hh in 0..h {
        masked.row_mut(hh)[hh * hd..(hh + 1) * hd].copy_from_slice(&head_out[hh * hd..(hh + 1) * hd]);
    }
    let hw = linear(&masked, &l.wo);
    let head_write: Vec<Vec<f32>> = (0..h).map(|hh| hw.row(hh).to_vec()).collect();
    // Split by source: term (h, j) = Wo_h · p_hj v_j, projected on d̂ = p_hj · (v_j · (Wo_hᵀ d̂)).
    let back = t_matvec(&l.wo, &unit(&d_attn));
    let src_proj = (0..h)
        .map(|hh| {
            let kh = hh / group;
            let w = &back[hh * hd..(hh + 1) * hd];
            (0..=pos)
                .map(|sp| if ablated[hh] { 0.0 } else { probs[hh][sp] * dot(&val(sp)[kh * hd..(kh + 1) * hd], w) })
                .collect()
        })
        .collect();
    let x_mid: Vec<f32> = x_in.iter().zip(&d_attn).map(|(a, b)| a + b).collect();

    // ---- MLP ----
    let xn2 = rmsnorm(&row(&x_mid), &l.mlp_norm, eps).data;
    let xt2 = row(&xn2);
    let gate = linear(&xt2, &l.w_gate).data;
    let up = linear(&xt2, &l.w_up).data;
    let act: Vec<f32> = gate.iter().map(|&g| silu(g)).collect();
    let hidden: Vec<f32> = act.iter().zip(&up).map(|(a, u)| a * u).collect();
    let d_mlp = linear(&row(&hidden), &l.w_down).data;
    let x_out: Vec<f32> = x_mid.iter().zip(&d_mlp).map(|(a, b)| a + b).collect();

    let inter = c.intermediate;
    let mut down_norm = vec![0.0f32; inter];
    for r in 0..c.hidden {
        for (n, w) in down_norm.iter_mut().zip(&l.w_down.data[r * inter..(r + 1) * inter]) {
            *n += w * w;
        }
    }
    down_norm.iter_mut().for_each(|n| *n = n.sqrt());
    let back2 = t_matvec(&l.w_down, &unit(&d_mlp));
    let neuron_proj = hidden.iter().zip(&back2).map(|(a, b)| a * b).collect();

    LayerFlow {
        attn_rms: rms(x_in, eps),
        x_in: x_in.to_vec(),
        xn,
        q_pre,
        q,
        k_pre,
        k,
        v,
        scores,
        probs,
        ablated,
        head_out,
        head_write,
        src_proj,
        d_attn,
        mlp_rms: rms(&x_mid, eps),
        x_mid,
        xn2,
        gate,
        up,
        act,
        hidden,
        down_norm,
        neuron_proj,
        d_mlp,
        x_out,
    }
}

/// For one position, every layer's attention write split by source position:
/// `[layer][source]`, projected on that write's direction (each row adds up to
/// ‖write‖). Also split by head: `[layer][head]`.
///
/// Uses the trace's attention maps and the cached values, so it costs
/// O(layers × heads × positions × head_dim) and no matrix products beyond one
/// `Woᵀ · d̂` per layer.
pub fn attn_sources(
    model: &Model,
    cache: &KvCache,
    attn: &[Vec<Vec<Vec<f32>>>],
    resid: &[Vec<Vec<f32>>],
    pos: usize,
    ablate: &HashSet<(usize, usize)>,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let c = &model.cfg;
    let (h, hd) = (c.n_heads, c.head_dim);
    let group = h / c.n_kv_heads;
    let mut by_src = Vec::with_capacity(c.n_layers);
    let mut by_head = Vec::with_capacity(c.n_layers);
    for (li, l) in model.layers.iter().enumerate() {
        let d: Vec<f32> = resid[2 * li + 1][pos].iter().zip(&resid[2 * li][pos]).map(|(a, b)| a - b).collect();
        let back = t_matvec(&l.wo, &unit(&d));
        let mut src = vec![0.0f32; pos + 1];
        let mut heads = vec![0.0f32; h];
        for hh in 0..h {
            if ablate.contains(&(li, hh)) {
                continue;
            }
            let kh = hh / group;
            let w = &back[hh * hd..(hh + 1) * hd];
            for (sp, &p) in attn[li][hh][pos].iter().enumerate() {
                let x = p * dot(&cache.v_row(li, sp)[kh * hd..(kh + 1) * hd], w);
                src[sp] += x;
                heads[hh] += x;
            }
        }
        by_src.push(src);
        by_head.push(heads);
    }
    (by_src, by_head)
}

/// Direct logit attribution: each row's exact additive effect on the final
/// logits, `E · (g ⊙ v) / final_rms`, with the final RMSNorm's divisor frozen
/// at its actual value for this position. `[rows, vocab]`.
pub fn direct_logits(model: &Model, rows: &[&[f32]], final_rms: f32) -> Tensor {
    let d = model.cfg.hidden;
    let mut x = Tensor::zeros(vec![rows.len(), d]);
    for (i, r) in rows.iter().enumerate() {
        for ((o, v), g) in x.row_mut(i).iter_mut().zip(*r).zip(&model.final_norm) {
            *o = v * g / final_rms;
        }
    }
    linear(&x, model.unembed())
}

/// The same attribution, but only onto a few tokens: `[row][token]`. Cheap.
pub fn token_attribution(model: &Model, rows: &[&[f32]], final_rms: f32, tokens: &[u32]) -> Vec<Vec<f32>> {
    let e = model.unembed();
    let dirs: Vec<Vec<f32>> = tokens
        .iter()
        .map(|&t| e.row(t as usize).iter().zip(&model.final_norm).map(|(a, g)| a * g / final_rms).collect())
        .collect();
    rows.iter().map(|r| dirs.iter().map(|dv| dot(r, dv)).collect()).collect()
}

/// `(index, value)` pairs.
pub type Ranked = Vec<(u32, f32)>;

/// The `k` largest and `k` smallest entries of a row.
pub fn extremes(row: &[f32], k: usize) -> (Ranked, Ranked) {
    let mut idx: Vec<(u32, f32)> = row.iter().enumerate().map(|(i, &x)| (i as u32, x)).collect();
    let k = k.min(idx.len());
    let hi_first = |a: &(u32, f32), b: &(u32, f32)| b.1.total_cmp(&a.1);
    idx.select_nth_unstable_by(k.max(1) - 1, hi_first);
    let mut top = idx[..k].to_vec();
    top.sort_unstable_by(hi_first);
    let lo_first = |a: &(u32, f32), b: &(u32, f32)| a.1.total_cmp(&b.1);
    idx.select_nth_unstable_by(k.max(1) - 1, lo_first);
    let mut bottom = idx[..k].to_vec();
    bottom.sort_unstable_by(lo_first);
    (top, bottom)
}
