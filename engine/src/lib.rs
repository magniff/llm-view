//! A from-scratch inference engine for SmolLM2-135M (Llama architecture),
//! built for *looking inside*: every intermediate tensor of the forward pass
//! is passed through a [`hooks::Hooks`] callback, which can read or modify it.
//!
//! Map of the crate, in the order data flows:
//!
//! - [`safetensors`] — parse the weight file format by hand (JSON header + raw bytes)
//! - [`tokenizer`]   — byte-level BPE: text → token ids → text
//! - [`model`]       — config, weights, and the forward pass (embed → 30 × block → logits)
//! - [`kv_cache`]    — per-layer K/V storage that makes decoding O(seq) instead of O(seq²)
//! - [`ops`]         — the handful of numeric kernels a Llama needs
//! - [`sampler`]     — logits → next token (greedy / temperature / top-k / top-p)
//! - [`hooks`]       — the observation / intervention interface
//! - [`generate`]    — the prefill + decode loop
//! - [`chat`]        — ChatML formatting and multi-turn conversations
//! - [`trace`]       — a hook that records a whole session by position, plus the logit lens
//! - [`session`]     — the visualizer's API: one session, driven by method + JSON args
//! - [`download`]    — fetch model files from the Hugging Face Hub on first run

pub mod chat;
pub mod clock;
pub mod download;
pub mod generate;
pub mod hooks;
pub mod kv_cache;
pub mod model;
pub mod ops;
pub mod safetensors;
pub mod sampler;
pub mod session;
pub mod tensor;
pub mod tokenizer;
pub mod trace;
mod unicode_tables;

pub type Error = Box<dyn std::error::Error + Send + Sync>;
pub type Result<T> = std::result::Result<T, Error>;
