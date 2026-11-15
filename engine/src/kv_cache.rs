//! The KV cache.
//!
//! In causal attention, token t's keys and values never change once computed —
//! later tokens only *read* them. So instead of recomputing K and V for the
//! whole sequence on every step, we store them, and each decode step computes
//! K/V for just the one new token and appends it.
//!
//! Layout per layer: `k[layer]` is a flat `[max_seq, KVH * D]` buffer; row
//! `pos` holds the (post-RoPE) keys of every KV head at that position.
//!
//! Size for SmolLM2-135M: 30 layers × 2 (K,V) × 3 heads × 64 dims × 4 bytes
//! = 46 KB per token. Grouped-query attention is why this is small: with 9
//! separate KV heads (plain multi-head attention) it would be 3× larger.

pub struct KvCache {
    pub n_layers: usize,
    pub max_seq: usize,
    /// `n_kv_heads * head_dim`
    pub kv_dim: usize,
    /// Number of positions filled so far.
    pub len: usize,
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
}

impl KvCache {
    pub fn new(n_layers: usize, max_seq: usize, kv_dim: usize) -> Self {
        Self {
            n_layers,
            max_seq,
            kv_dim,
            len: 0,
            k: vec![vec![0.0; max_seq * kv_dim]; n_layers],
            v: vec![vec![0.0; max_seq * kv_dim]; n_layers],
        }
    }

    pub fn k_row(&self, layer: usize, pos: usize) -> &[f32] {
        &self.k[layer][pos * self.kv_dim..(pos + 1) * self.kv_dim]
    }

    pub fn v_row(&self, layer: usize, pos: usize) -> &[f32] {
        &self.v[layer][pos * self.kv_dim..(pos + 1) * self.kv_dim]
    }

    pub(crate) fn write(&mut self, layer: usize, pos: usize, k: &[f32], v: &[f32]) {
        let r = pos * self.kv_dim..(pos + 1) * self.kv_dim;
        self.k[layer][r.clone()].copy_from_slice(k);
        self.v[layer][r].copy_from_slice(v);
    }

    /// Bytes actually in use (filled positions only).
    pub fn bytes_used(&self) -> usize {
        self.len * self.kv_dim * self.n_layers * 2 * std::mem::size_of::<f32>()
    }

    /// Forget everything after `len` — e.g. to rewind and branch a generation.
    /// Stale data past `len` is simply overwritten later.
    pub fn truncate(&mut self, len: usize) {
        self.len = self.len.min(len);
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }
}
