//! An inspection session over the model, used by the visualizer, both by the
//! native server and by the in-browser WebAssembly build.
//!
//! A session is one token sequence, its KV cache, and a [`Trace`] of every
//! activation recorded while running it. Everything is driven through
//! [`Session::call`]: a method name plus JSON arguments in, JSON out, so any
//! transport (HTTP, a Web Worker message) can sit in front of it.

use std::collections::HashSet;

use serde_json::{json, Value};

use crate::clock::Stopwatch;
use crate::walk::{layer_flow, output_logits, rms, LayerFlow};
use crate::kv_cache::KvCache;
use crate::model::Model;
use crate::sampler::Sampler;
use crate::tokenizer::Tokenizer;
use crate::tensor::Tensor;
use crate::trace::{top_logits, TopLogits, Trace};

const DEFAULT_PROMPT: &str = "The love is";

/// One inspection session: a token sequence, its KV cache, and the trace of
/// everything the model computed for it.
pub struct Session {
    model: Model,
    tok: Tokenizer,
    name: String,
    tokens: Vec<u32>,
    n_prompt: usize,
    cache: KvCache,
    trace: Trace,
    /// Full logits for the last position: what the next token is sampled from.
    last_logits: Vec<f32>,
    sampler: Sampler,
    last_op: Value,
    specials: HashSet<u32>,
}

/// A token's text for display; bytes that are only part of a UTF-8 character are shown as `<XX>`.
fn tok_text(tok: &Tokenizer, id: u32) -> String {
    let b = tok.decode_bytes(&[id]);
    match std::str::from_utf8(&b) {
        Ok(s) => s.to_string(),
        Err(_) => b.iter().map(|x| format!("<{x:02X}>")).collect(),
    }
}

/// Round for JSON: 4 significant decimals is plenty for a picture and keeps payloads small.
/// Positions the step-by-step view shows: whole matrices, so keep it readable.
const WALK_MAX: usize = 16;

fn r4(x: f32) -> f64 {
    (x as f64 * 1e4).round() / 1e4
}

fn r4v(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| r4(x)).collect()
}

impl Session {
    fn text(&self, id: u32) -> String {
        tok_text(&self.tok, id)
    }

    /// Feed tokens through the model (appending to the cache), recording into the trace.
    fn feed(&mut self, toks: &[u32]) -> f64 {
        let t = Stopwatch::start();
        let logits = self.model.forward(toks, &mut self.cache, &mut self.trace);
        self.last_logits = logits.row(logits.rows() - 1).to_vec();
        t.ms()
    }

    fn run(&mut self, text: &str) -> Result<(), String> {
        self.run_tokens(self.tok.encode(text))
    }

    /// Start over from exact token ids (no re-tokenization).
    fn run_tokens(&mut self, toks: Vec<u32>) -> Result<(), String> {
        if let Some(&bad) = toks.iter().find(|&&t| t as usize >= self.model.cfg.vocab) {
            return Err(format!("token id {bad} out of range"));
        }
        if toks.is_empty() {
            return Err("empty prompt".into());
        }
        if toks.len() >= self.cache.max_seq {
            return Err(format!("prompt is {} tokens; the limit here is {}", toks.len(), self.cache.max_seq));
        }
        self.tokens = toks.clone();
        self.n_prompt = toks.len();
        self.cache.clear();
        self.trace.truncate(0);
        let ms = self.feed(&toks);
        self.last_op = json!({"kind": "prefill", "pos0": 0, "n": toks.len(), "ms": ms});
        Ok(())
    }

    fn step(&mut self, n: usize, force: Option<u32>) {
        let pos0 = self.tokens.len();
        let mut force = force;
        let (mut done, mut ms, mut reason) = (0, 0.0, "requested steps done");
        for _ in 0..n {
            if self.cache.len >= self.cache.max_seq {
                reason = "context full";
                break;
            }
            let next = force.take().unwrap_or_else(|| self.sampler.sample_with_history(&self.last_logits, &self.tokens));
            self.tokens.push(next);
            ms += self.feed(&[next]);
            done += 1;
            if next == self.model.cfg.eos_token || next == 0 {
                reason = "end-of-sequence token";
                break;
            }
        }
        self.last_op = json!({"kind": "decode", "pos0": pos0, "n": done, "ms": ms, "stop": reason});
    }

    /// Keep positions `0..len`. The cache and trace keep `0..len-1`, and the last
    /// kept token is re-fed so we have its full logits to sample from again.
    fn rewind(&mut self, len: usize) {
        let len = len.clamp(1, self.tokens.len());
        self.tokens.truncate(len);
        self.n_prompt = self.n_prompt.min(len);
        self.cache.truncate(len - 1);
        self.trace.truncate(len - 1);
        let ms = self.feed(&[self.tokens[len - 1]]);
        self.last_op = json!({"kind": "rewind", "pos0": len - 1, "n": 1, "ms": ms});
    }

    fn state(&self) -> Value {
        let c = &self.model.cfg;
        let tokens: Vec<Value> = self
            .tokens
            .iter()
            .enumerate()
            .map(|(i, &id)| json!({"id": id, "text": self.text(id), "gen": i >= self.n_prompt, "special": self.specials.contains(&id)}))
            .collect();
        let next = top_logits(&self.last_logits, 50);
        let bytes_per_token = c.n_layers * 2 * c.kv_dim() * 4;
        json!({
            "model": {
                "name": self.name, "params": self.model.n_params(), "n_layers": c.n_layers,
                "n_heads": c.n_heads, "n_kv_heads": c.n_kv_heads, "head_dim": c.head_dim,
                "hidden": c.hidden, "intermediate": c.intermediate, "vocab": c.vocab,
                "rope_theta": c.rope_theta, "eos": c.eos_token,
            },
            "tokens": tokens,
            "n_prompt": self.n_prompt,
            "cache": {
                "len": self.cache.len, "max_seq": self.cache.max_seq, "kv_dim": c.kv_dim(),
                "bytes": self.cache.bytes_used(), "bytes_per_token": bytes_per_token,
            },
            "last_op": self.last_op,
            "next": {
                "lse": next.lse,
                // [token, length of the repeat it would extend], for the DRY penalty
                "dry": self.sampler.dry_matches(&self.tokens),
                "top": next.top.iter().map(|&(id, l)| json!([id, self.text(id), l])).collect::<Vec<_>>(),
            },
            "sampler": {"temperature": self.sampler.temperature, "top_k": self.sampler.top_k,
                        "top_p": self.sampler.top_p, "repeat_penalty": self.sampler.repeat_penalty,
                        "dry": self.sampler.dry_multiplier, "dry_base": self.sampler.dry_base,
                        "dry_allowed_len": self.sampler.dry_allowed_len, "repeat_window": self.sampler.repeat_window},
        })
    }

    /// The step-by-step view: one layer for every position at once, as the
    /// matrices `[T, …]` the forward pass works with (rows = tokens). Only the
    /// first [`WALK_MAX`] positions: it is for reading, not for long prompts.
    fn walk_layer(&self, layer: usize) -> Result<Value, String> {
        let c = &self.model.cfg;
        if layer >= c.n_layers {
            return Err("no such layer".into());
        }
        let n = self.trace.len().min(WALK_MAX);
        let fl: Vec<_> = (0..n)
            .map(|p| layer_flow(&self.model, &self.cache, &self.trace.resid[2 * layer][p], layer, p))
            .collect();
        let rows = |f: &dyn Fn(&LayerFlow) -> &[f32]| fl.iter().map(|x| r4v(f(x))).collect::<Vec<_>>();
        let l = &self.model.layers[layer];
        Ok(json!({
            "layer": layer, "n": n, "eps": c.norm_eps,
            "attn_norm_w": r4v(&l.attn_norm), "mlp_norm_w": r4v(&l.mlp_norm),
            "x_in": rows(&|f| &f.x_in), "attn_rms": fl.iter().map(|f| r4(f.attn_rms)).collect::<Vec<_>>(), "xn": rows(&|f| &f.xn),
            "q_pre": rows(&|f| &f.q_pre), "q": rows(&|f| &f.q), "k_pre": rows(&|f| &f.k_pre), "k": rows(&|f| &f.k), "v": rows(&|f| &f.v),
            // [position][head][source 0..=position]
            "probs": fl.iter().map(|f| f.probs.iter().map(|p| r4v(p)).collect::<Vec<_>>()).collect::<Vec<_>>(),
            "head_out": rows(&|f| &f.head_out), "d_attn": rows(&|f| &f.d_attn), "x_mid": rows(&|f| &f.x_mid),
            "mlp_rms": fl.iter().map(|f| r4(f.mlp_rms)).collect::<Vec<_>>(), "xn2": rows(&|f| &f.xn2),
            "gate": rows(&|f| &f.gate), "up": rows(&|f| &f.up), "act": rows(&|f| &f.act), "hidden": rows(&|f| &f.hidden),
            "d_mlp": rows(&|f| &f.d_mlp), "x_out": rows(&|f| &f.x_out),
        }))
    }

    /// The two ends of the step-by-step view: the embedding rows, and the final
    /// norm, logits and probabilities for every position.
    /// Unembed the residual stream at `stage` for the walked positions:
    /// `(rms of each row, logits [rows, vocab])`.
    fn unembed_stage(&self, stage: usize) -> (Vec<f32>, Tensor) {
        let n = self.trace.len().min(WALK_MAX);
        let rows: Vec<&[f32]> = (0..n).map(|p| self.trace.resid[stage][p].as_slice()).collect();
        let rms: Vec<f32> = rows.iter().map(|r| rms(r, self.model.cfg.norm_eps)).collect();
        let logits = output_logits(&self.model, &rows, &rms);
        (rms, logits)
    }

    fn top_json(&self, tl: &TopLogits) -> Value {
        tl.top.iter().map(|&(id, l)| json!([id, self.text(id), r4(l), r4((l - tl.lse).exp())])).collect()
    }

    fn walk_ends(&self) -> Result<Value, String> {
        let c = &self.model.cfg;
        let n = self.trace.len().min(WALK_MAX);
        let (rms, logits) = self.unembed_stage(2 * c.n_layers);
        let out: Vec<Value> = (0..n)
            .map(|p| {
                let x = &self.trace.resid[2 * c.n_layers][p];
                let xn: Vec<f32> = x.iter().zip(&self.model.final_norm).map(|(v, g)| v * g / rms[p]).collect();
                let tl = top_logits(logits.row(p), 5);
                json!({"x": r4v(x), "rms": r4(rms[p]), "xn": r4v(&xn), "lse": r4(tl.lse), "top": self.top_json(&tl)})
            })
            .collect();
        Ok(json!({
            "n": n,
            "embed": (0..n).map(|p| r4v(&self.trace.resid[0][p])).collect::<Vec<_>>(),
            "norm_w": r4v(&self.model.final_norm),
            "out": out,
        }))
    }

    /// The logit lens on layer `layer`'s output: each row through the final norm
    /// and the unembedding, as if the layers after it were skipped. Each row
    /// also reports how the model's final answer for that row (the last layer's
    /// top token) stands here: `[id, text, probability, rank]` (rank 0 = top).
    fn walk_lens(&self, layer: usize) -> Result<Value, String> {
        let c = &self.model.cfg;
        if layer >= c.n_layers {
            return Err("no such layer".into());
        }
        let (_, fin) = self.unembed_stage(2 * c.n_layers);
        let (rms, here) = self.unembed_stage(2 * layer + 2);
        let rows: Vec<Value> = (0..rms.len())
            .map(|p| {
                let tl = top_logits(here.row(p), 5);
                let answer = top_logits(fin.row(p), 1).top[0].0;
                let row = here.row(p);
                let l = row[answer as usize];
                let rank = row.iter().filter(|&&x| x > l).count();
                json!({"rms": r4(rms[p]), "top": self.top_json(&tl), "final": [answer, self.text(answer), (l - tl.lse).exp() as f64, rank]})
            })
            .collect();
        Ok(json!({"layer": layer, "rows": rows}))
    }

    fn tokenize_view(&self, text: &str) -> Value {
        let mut words = vec![];
        let ids = self.tok.encode_traced(text, Some(&mut words));
        json!({
            "ids": ids,
            "words": words.iter().map(|w| json!({
                "word": w.word,
                "steps": w.steps.iter().map(|s| s.iter().map(|&id| json!([id, self.text(id)])).collect::<Vec<_>>()).collect::<Vec<_>>(),
            })).collect::<Vec<_>>(),
        })
    }
}

impl Session {
    /// A fresh session with room for `max_seq` positions (it keeps a full trace
    /// of each, so keep it modest), already run on a default prompt.
    pub fn new(model: Model, tok: Tokenizer, name: &str, max_seq: usize) -> Result<Self, String> {
        let specials: HashSet<u32> = tok.specials.iter().map(|&(_, id)| id).collect();
        // DRY against loops (a 135M model falls into them constantly).
        let breaker = |id: u32| specials.contains(&id) || Sampler::dry_breaker_bytes(&tok.decode_bytes(&[id]));
        // Penalties apply to words only, so answers can repeat numbers (see `Sampler::words_only`).
        let sp: Vec<u32> = specials.iter().copied().collect();
        let words = |s: Sampler| s.words_only(model.cfg.vocab, &sp, |id| tok.decode_bytes(&[id]));
        // Default: the page's "Focused" preset, the model's own top token unless DRY sees a loop.
        let sampler = words(Sampler::new(0.0, 0, 0.9, 1234).with_dry(model.cfg.vocab, breaker));
        let mut s = Session {
            cache: model.new_cache(max_seq),
            trace: Trace::new(&model),
            model,
            tok,
            name: name.to_string(),
            tokens: vec![],
            n_prompt: 0,
            last_logits: vec![],
            sampler,
            last_op: Value::Null,
            specials,
        };
        s.run(DEFAULT_PROMPT)?;
        Ok(s)
    }

    /// The whole API: `method` with JSON `args`, returning JSON.
    pub fn call(&mut self, method: &str, args: &Value) -> Result<Value, String> {
        // Absent → None; present but not a non-negative integer that fits → error.
        // (`as usize` would silently wrap 2^32 to 0 on wasm32.)
        let int = |k: &str| -> Result<Option<usize>, String> {
            match &args[k] {
                Value::Null => Ok(None),
                v => v
                    .as_u64()
                    .and_then(|x| usize::try_from(x).ok())
                    .map(Some)
                    .ok_or_else(|| format!("{method}: '{k}' must be a non-negative integer")),
            }
        };
        let need = |k: &str| int(k)?.ok_or_else(|| format!("{method}: missing integer '{k}'"));
        match method {
            "state" => Ok(self.state()),
            "run" => {
                if let Some(ids) = args["ids"].as_array() {
                    let ids = ids
                        .iter()
                        .map(|v| v.as_u64().and_then(|x| u32::try_from(x).ok()).ok_or("run: 'ids' must be token ids"))
                        .collect::<Result<Vec<_>, _>>()?;
                    self.run_tokens(ids)?;
                } else {
                    let text = args["text"].as_str().ok_or("run: missing 'text' (or 'ids')")?;
                    self.run(text)?;
                }
                Ok(self.state())
            }
            "step" => {
                // Validate everything before changing anything.
                let force = int("force")?;
                if force.is_some_and(|f| f >= self.model.cfg.vocab) {
                    return Err("token id out of range".into());
                }
                let n = int("n")?.unwrap_or(1).clamp(1, 256);
                apply_sampling(&mut self.sampler, args, method)?;
                self.step(n, force.map(|f| f as u32));
                Ok(self.state())
            }
            "rewind" => {
                self.rewind(need("len")?);
                Ok(self.state())
            }
            "walk_layer" => self.walk_layer(need("layer")?),
            "walk_ends" => self.walk_ends(),
            "walk_lens" => self.walk_lens(need("layer")?),
            "tokenize" => {
                let text = args["text"].as_str().ok_or("tokenize: missing 'text'")?;
                Ok(self.tokenize_view(text))
            }
            _ => Err(format!("unknown method '{method}'")),
        }
    }
}

/// Read optional sampler settings from `args`, rejecting values that would
/// make sampling meaningless (NaN logits) before touching the sampler.
fn apply_sampling(s: &mut Sampler, args: &Value, method: &str) -> Result<(), String> {
    let get = |k: &str| -> Result<Option<f64>, String> {
        match &args[k] {
            Value::Null => Ok(None),
            v => v.as_f64().map(Some).ok_or_else(|| format!("{method}: '{k}' must be a number")),
        }
    };
    let bad = |k: &str, why: &str| Err(format!("{method}: '{k}' {why}"));
    let t = get("temperature")?;
    let p = get("top_p")?;
    let r = get("repeat_penalty")?;
    let d = get("dry")?;
    let k = match &args["top_k"] {
        Value::Null => None,
        v => Some(v.as_u64().ok_or_else(|| format!("{method}: 'top_k' must be a non-negative integer"))?),
    };
    if t.is_some_and(|t| t < 0.0) {
        return bad("temperature", "must be >= 0");
    }
    if p.is_some_and(|p| !(0.0..=1.0).contains(&p)) {
        return bad("top_p", "must be in [0, 1]");
    }
    if r.is_some_and(|r| r <= 0.0) {
        return bad("repeat_penalty", "must be > 0");
    }
    if d.is_some_and(|d| d < 0.0) {
        return bad("dry", "must be >= 0 (0 = off)");
    }
    if let Some(t) = t { s.temperature = t as f32 }
    if let Some(p) = p { s.top_p = p as f32 }
    if let Some(r) = r { s.repeat_penalty = r as f32 }
    if let Some(d) = d { s.dry_multiplier = d as f32 }
    if let Some(k) = k { s.top_k = usize::try_from(k).unwrap_or(usize::MAX) }
    Ok(())
}
