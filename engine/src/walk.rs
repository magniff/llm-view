//! The Step-by-step view: every intermediate of one layer, recomputed on demand.
//!
//! The [`Trace`] keeps only the residual stream between blocks. The view
//! needs *everything* a layer computes: q, k, v before and after RoPE, the
//! attention weights, each head's output, the MLP's gate/up/activation, and so
//! on. Rather than recording all of that for every position and layer,
//! [`layer_flow`] recomputes one position through one layer from two things
//! the session already holds:
//!
//! - the residual stream entering the layer at that position (from the trace);
//! - the keys and values of earlier positions (from the KV cache).
//!
//! That is exactly the input a decode step sees, so the recomputation *is* the
//! forward pass for that position, and its output must land exactly on the
//! trace's next snapshot (the tests check this).
//!
//! [`Trace`]: crate::trace::Trace

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
    /// `[head][source]` attention weights over sources `0..=pos`.
    pub probs: Vec<Vec<f32>>,
    /// `[H*D]` each head's weighted sum of values.
    pub head_out: Vec<f32>,
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
    /// What the MLP adds to the residual stream.
    pub d_mlp: Vec<f32>,
    pub x_out: Vec<f32>,
}

pub fn rms(x: &[f32], eps: f32) -> f32 {
    (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt()
}

fn row(v: &[f32]) -> Tensor {
    Tensor::new(vec![1, v.len()], v.to_vec())
}

/// Recompute position `pos` through layer `layer`, given the residual stream
/// entering it. `cache` must hold that layer's keys and values for `0..pos`.
pub fn layer_flow(model: &Model, cache: &KvCache, x_in: &[f32], layer: usize, pos: usize) -> LayerFlow {
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
    let mut probs = Vec::with_capacity(h);
    let mut head_out = vec![0.0f32; h * hd];
    for hh in 0..h {
        let kh = hh / group;
        let qv = &q[hh * hd..(hh + 1) * hd];
        let mut p: Vec<f32> = (0..=pos).map(|sp| dot(qv, &key(sp)[kh * hd..(kh + 1) * hd]) * scale).collect();
        softmax(&mut p);
        let out = &mut head_out[hh * hd..(hh + 1) * hd];
        for (sp, &w) in p.iter().enumerate() {
            for (o, vi) in out.iter_mut().zip(&val(sp)[kh * hd..(kh + 1) * hd]) {
                *o += w * vi;
            }
        }
        probs.push(p);
    }
    let d_attn = linear(&row(&head_out), &l.wo).data;
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

    LayerFlow {
        attn_rms: rms(x_in, eps),
        x_in: x_in.to_vec(),
        xn,
        q_pre,
        q,
        k_pre,
        k,
        v,
        probs,
        head_out,
        d_attn,
        mlp_rms: rms(&x_mid, eps),
        x_mid,
        xn2,
        gate,
        up,
        act,
        hidden,
        d_mlp,
        x_out,
    }
}

/// The output head on residual-stream rows: final RMSNorm (row `i` divided by
/// `rms[i]`, then × the learned gain), then a dot product with every token's
/// embedding. `[rows, vocab]` logits.
///
/// On the last layer's output this is the model's prediction. On an earlier
/// layer's output it is the "logit lens": what the stream would predict if the
/// remaining layers were skipped.
pub fn output_logits(model: &Model, rows: &[&[f32]], rms: &[f32]) -> Tensor {
    let d = model.cfg.hidden;
    let mut x = Tensor::zeros(vec![rows.len(), d]);
    for (i, (r, s)) in rows.iter().zip(rms).enumerate() {
        for ((o, v), g) in x.row_mut(i).iter_mut().zip(*r).zip(&model.final_norm) {
            *o = v * g / s;
        }
    }
    linear(&x, model.unembed())
}
