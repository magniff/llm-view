//! Hooks: the engine calls [`Hooks::on`] with every intermediate tensor.
//!
//! The tensor is passed `&mut`, so a hook can either just look (record it,
//! compute statistics, stream it to a UI) or *intervene* — zero an attention
//! head, patch in an activation from another run, add a steering vector.
//! Whatever the hook leaves in the tensor is what the rest of the forward pass uses.
//!
//! Hook sites, in order, with shapes (T = new tokens this call, S = total
//! positions in the cache including these, H = 9 query heads, KVH = 3 KV heads,
//! D = 64 head dim, V = 49152):
//!
//! ```text
//! embed.out                     [T, 576]
//! layer.{i}.attn_norm.out       [T, 576]
//! layer.{i}.attn.q              [T, H*D]     before RoPE
//! layer.{i}.attn.k / .v         [T, KVH*D]
//! layer.{i}.attn.q_rope         [T, H*D]
//! layer.{i}.attn.k_rope         [T, KVH*D]   this is what gets written to the KV cache
//! layer.{i}.attn.probs          [H, T, S]    rows sum to 1; future positions are 0
//! layer.{i}.attn.out            [T, H*D]     heads concatenated, before o_proj
//! layer.{i}.attn.o_proj         [T, 576]
//! layer.{i}.resid.post_attn     [T, 576]     residual stream after adding attention
//! layer.{i}.mlp_norm.out        [T, 576]
//! layer.{i}.mlp.gate / .up      [T, 1536]
//! layer.{i}.mlp.act             [T, 1536]    silu(gate)
//! layer.{i}.mlp.hidden          [T, 1536]    silu(gate) * up
//! layer.{i}.mlp.down            [T, 576]
//! layer.{i}.resid.post_mlp      [T, 576]     residual stream leaving the block
//! final_norm.out                [T, 576]
//! logits                        [T, V]
//! ```

use std::collections::HashMap;

use crate::tensor::Tensor;

/// Where in the forward pass a hook fired.
#[derive(Debug, Clone, Copy)]
pub struct Site {
    /// Transformer block index, or `None` for embed / final norm / logits.
    pub layer: Option<usize>,
    pub name: &'static str,
    /// Absolute position of the first token in this forward call
    /// (0 for a fresh prefill, `cache.len` for a decode step).
    pub pos0: usize,
}

impl Site {
    /// Canonical string key, e.g. `layer.3.attn.probs`. Matches the golden files.
    pub fn key(&self) -> String {
        match self.layer {
            Some(l) => format!("layer.{l}.{}", self.name),
            None => self.name.to_string(),
        }
    }
}

pub trait Hooks {
    fn on(&mut self, site: &Site, t: &mut Tensor);
}

/// Does nothing. Use for plain generation.
pub struct NoHooks;

impl Hooks for NoHooks {
    fn on(&mut self, _: &Site, _: &mut Tensor) {}
}

/// Records a copy of every tensor it sees (later calls overwrite earlier ones
/// with the same key).
#[derive(Default)]
pub struct Recorder {
    pub tensors: HashMap<String, Tensor>,
}

impl Hooks for Recorder {
    fn on(&mut self, site: &Site, t: &mut Tensor) {
        self.tensors.insert(site.key(), t.clone());
    }
}

/// Adapt any closure into a hook.
pub struct FnHook<F: FnMut(&Site, &mut Tensor)>(pub F);

impl<F: FnMut(&Site, &mut Tensor)> Hooks for FnHook<F> {
    fn on(&mut self, site: &Site, t: &mut Tensor) {
        (self.0)(site, t)
    }
}
