//! An inspection session over the model, used by the visualizer, both by the
//! native server and by the in-browser WebAssembly build.
//!
//! A session owns a token sequence, its KV cache, and a [`Trace`] of every
//! activation recorded while running it. Everything is driven through
//! [`Session::call`]: a method name plus JSON arguments in, JSON out, so any
//! transport (HTTP, a Web Worker message) can sit in front of it.

use std::collections::{HashMap, HashSet};

use serde_json::{json, Value};

use crate::chat::render_single;
use crate::clock::Stopwatch;
use crate::kv_cache::KvCache;
use crate::model::Model;
use crate::sampler::Sampler;
use crate::tokenizer::Tokenizer;
use crate::trace::{logit_lens, top_logits, LensCell, Trace};

const DEFAULT_PROMPT: &str = "The capital of France is Paris. The capital of Germany is";

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
    /// Logit-lens cells already computed, by (stage, pos).
    lens: HashMap<(usize, usize), LensCell>,
    last_op: Value,
    specials: HashSet<u32>,
}

/// Round for JSON: 4 significant decimals is plenty for a picture and keeps payloads small.
fn r4(x: f32) -> f64 {
    (x as f64 * 1e4).round() / 1e4
}

fn r4v(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| r4(x)).collect()
}

impl Session {
    fn text(&self, id: u32) -> String {
        let b = self.tok.decode_bytes(&[id]);
        match std::str::from_utf8(&b) {
            Ok(s) => s.to_string(),
            // Part of a multi-byte character: show the raw bytes.
            Err(_) => b.iter().map(|x| format!("<{x:02X}>")).collect(),
        }
    }

    /// Feed tokens through the model (appending to the cache), recording into the trace.
    fn feed(&mut self, toks: &[u32]) -> f64 {
        let t = Stopwatch::start();
        let logits = self.model.forward(toks, &mut self.cache, &mut self.trace);
        self.last_logits = logits.row(logits.rows() - 1).to_vec();
        t.ms()
    }

    fn run(&mut self, text: &str, chat: bool) -> Result<(), String> {
        let toks = self.tok.encode(&if chat { render_single(None, text) } else { text.to_string() });
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
        self.lens.clear();
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
            // The lens column at the old last position now has a known "next token".
            let last = self.tokens.len() - 1;
            self.lens.retain(|&(_, p), _| p != last);
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
        self.lens.retain(|&(_, p), _| p < len - 1);
        let ms = self.feed(&[self.tokens[len - 1]]);
        self.last_op = json!({"kind": "rewind", "pos0": len - 1, "n": 1, "ms": ms});
    }

    /// Re-run the whole sequence from scratch (after changing ablations).
    fn recompute(&mut self) {
        self.cache.clear();
        self.trace.truncate(0);
        self.lens.clear();
        let toks = self.tokens.clone();
        let ms = self.feed(&toks);
        self.last_op = json!({"kind": "prefill", "pos0": 0, "n": toks.len(), "ms": ms, "note": "re-run with new ablations"});
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
        let resid_norm: Vec<Vec<f64>> = self
            .trace
            .resid
            .iter()
            .map(|s| s.iter().map(|v| r4(v.iter().map(|x| x * x).sum::<f32>().sqrt())).collect())
            .collect();
        let bytes_per_token = c.n_layers * 2 * c.kv_dim() * 4;
        let mut ablate: Vec<_> = self.trace.ablate.iter().copied().collect();
        ablate.sort();
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
                "top": next.top.iter().map(|&(id, l)| json!([id, self.text(id), l])).collect::<Vec<_>>(),
            },
            "resid_norm": resid_norm,
            "attn_write": self.trace.attn_write.iter().map(|v| r4v(v)).collect::<Vec<_>>(),
            "mlp_write": self.trace.mlp_write.iter().map(|v| r4v(v)).collect::<Vec<_>>(),
            "ablate": ablate,
            "sampler": {"temperature": self.sampler.temperature, "top_k": self.sampler.top_k,
                        "top_p": self.sampler.top_p, "repeat_penalty": self.sampler.repeat_penalty},
        })
    }

    /// Logit lens over all positions. Each cell tracks two tokens through the layers:
    /// `f` = the model's final answer at that position (the top token of the last
    /// layer), and `t` = the token that actually came next (if any).
    fn lens_view(&mut self, mid: bool) -> Value {
        let l = self.model.cfg.n_layers;
        let last = 2 * l; // stage index of the last layer's output
        let stages: Vec<usize> = if mid { (0..=last).collect() } else { (0..=l).map(|i| 2 * i).collect() };
        let n = self.trace.len();
        let next = |p: usize| self.tokens.get(p + 1).copied();

        // 1. The last layer first: its top token is the "final answer" every other layer is compared to.
        let todo: Vec<usize> = (0..n).filter(|&p| !self.lens.contains_key(&(last, p))).collect();
        if !todo.is_empty() {
            let rows: Vec<&[f32]> = todo.iter().map(|&p| self.trace.resid[last][p].as_slice()).collect();
            let track: Vec<Vec<u32>> = todo.iter().map(|&p| next(p).into_iter().collect()).collect();
            for (p, mut c) in todo.into_iter().zip(logit_lens(&self.model, &rows, &track, 5)) {
                c.tracked.insert(0, (c.top[0].1, 0)); // the final answer is its own top token
                self.lens.insert((last, p), c);
            }
        }
        let answer = |lens: &HashMap<(usize, usize), LensCell>, p: usize| lens[&(last, p)].top[0].0;

        // 2. Every other stage, tracking [final answer, actual next].
        let missing: Vec<(usize, usize)> = stages
            .iter()
            .flat_map(|&s| (0..n).map(move |p| (s, p)))
            .filter(|k| !self.lens.contains_key(k))
            .collect();
        if !missing.is_empty() {
            let rows: Vec<&[f32]> = missing.iter().map(|&(s, p)| self.trace.resid[s][p].as_slice()).collect();
            let track: Vec<Vec<u32>> = missing
                .iter()
                .map(|&(_, p)| std::iter::once(answer(&self.lens, p)).chain(next(p)).collect())
                .collect();
            let cells = logit_lens(&self.model, &rows, &track, 5);
            for (k, c) in missing.into_iter().zip(cells) {
                self.lens.insert(k, c);
            }
        }

        let label = |s: usize| match s {
            0 => "embed".to_string(),
            s if s % 2 == 1 => format!("L{} attn", s / 2),
            s => format!("L{}", s / 2 - 1),
        };
        let pr = |x: Option<&(f32, usize)>| x.map(|&(p, rank)| json!([r4(p), rank]));
        let cells: Vec<Vec<Value>> = stages
            .iter()
            .map(|&s| {
                (0..n)
                    .map(|p| {
                        let c = &self.lens[&(s, p)];
                        json!({
                            "top": c.top.iter().map(|&(id, pr)| json!([id, self.text(id), r4(pr)])).collect::<Vec<_>>(),
                            "f": pr(c.tracked.first()),
                            "t": pr(c.tracked.get(1)),
                            "h": r4(c.entropy),
                        })
                    })
                    .collect()
            })
            .collect();
        let answers: Vec<Value> = (0..n).map(|p| {
            let id = answer(&self.lens, p);
            json!([id, self.text(id)])
        }).collect();
        json!({"stages": stages.iter().map(|&s| label(s)).collect::<Vec<_>>(), "cells": cells, "answers": answers})
    }

    fn attn_view(&self, layer: usize) -> Result<Value, String> {
        let heads = self.trace.attn.get(layer).ok_or("no such layer")?;
        Ok(json!({
            "layer": layer,
            "heads": heads.iter().map(|h| h.iter().map(|row| r4v(row)).collect::<Vec<_>>()).collect::<Vec<_>>(),
            "head_norm": self.trace.head_norm[layer].iter().map(|v| r4v(v)).collect::<Vec<_>>(),
        }))
    }

    fn kv_view(&self, layer: usize) -> Result<Value, String> {
        if layer >= self.model.cfg.n_layers {
            return Err("no such layer".into());
        }
        let n = self.cache.len;
        Ok(json!({
            "layer": layer,
            "k": (0..n).map(|p| r4v(self.cache.k_row(layer, p))).collect::<Vec<_>>(),
            "v": (0..n).map(|p| r4v(self.cache.v_row(layer, p))).collect::<Vec<_>>(),
        }))
    }

    fn mlp_view(&self, layer: usize, pos: usize) -> Result<Value, String> {
        let h = self.trace.mlp_hidden.get(layer).and_then(|l| l.get(pos)).ok_or("no such layer/position")?;
        Ok(json!({"layer": layer, "pos": pos, "hidden": r4v(h)}))
    }

    fn tokenize_view(&self, text: &str, chat: bool) -> Value {
        let text = if chat { render_single(None, text) } else { text.to_string() };
        let mut words = vec![];
        let ids = self.tok.encode_traced(&text, Some(&mut words));
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
    /// A fresh session with room for `max_seq` positions, already run on the default prompt.
    pub fn new(model: Model, tok: Tokenizer, name: &str, max_seq: usize) -> Result<Self, String> {
        let specials = tok.specials.iter().map(|&(_, id)| id).collect();
        let mut sampler = Sampler::new(0.7, 0, 0.9, 1234);
        sampler.repeat_penalty = 1.1;
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
            lens: HashMap::new(),
            last_op: Value::Null,
            specials,
        };
        s.run(DEFAULT_PROMPT, false)?;
        Ok(s)
    }

    /// The whole API: `method` with JSON `args`, returning JSON.
    pub fn call(&mut self, method: &str, args: &Value) -> Result<Value, String> {
        let int = |k: &str| args[k].as_u64().map(|v| v as usize);
        let num = |k: &str| args[k].as_f64().map(|v| v as f32);
        let need = |k: &str| int(k).ok_or_else(|| format!("{method}: missing integer '{k}'"));
        match method {
            "state" => Ok(self.state()),
            "run" => {
                let text = args["text"].as_str().ok_or("run: missing 'text'")?;
                self.run(text, args["chat"].as_bool().unwrap_or(false))?;
                Ok(self.state())
            }
            "step" => {
                if let Some(t) = num("temperature") { self.sampler.temperature = t }
                if let Some(k) = int("top_k") { self.sampler.top_k = k }
                if let Some(p) = num("top_p") { self.sampler.top_p = p }
                if let Some(p) = num("repeat_penalty") { self.sampler.repeat_penalty = p }
                let force = int("force").map(|f| f as u32);
                if force.is_some_and(|f| f as usize >= self.model.cfg.vocab) {
                    return Err("token id out of range".into());
                }
                self.step(int("n").unwrap_or(1).clamp(1, 256), force);
                Ok(self.state())
            }
            "rewind" => {
                self.rewind(need("len")?);
                Ok(self.state())
            }
            "ablate" => {
                let (l, h) = (self.model.cfg.n_layers, self.model.cfg.n_heads);
                let heads: Vec<(usize, usize)> = serde_json::from_value(args["heads"].clone()).map_err(|e| e.to_string())?;
                self.trace.ablate = heads.into_iter().filter(|&(li, hi)| li < l && hi < h).collect();
                self.recompute();
                Ok(self.state())
            }
            "lens" => Ok(self.lens_view(args["mid"].as_bool().unwrap_or(false))),
            "attn" => self.attn_view(need("layer")?),
            "kv" => self.kv_view(need("layer")?),
            "mlp" => self.mlp_view(need("layer")?, int("pos").unwrap_or(0)),
            "tokenize" => {
                let text = args["text"].as_str().ok_or("tokenize: missing 'text'")?;
                Ok(self.tokenize_view(text, args["chat"].as_bool().unwrap_or(false)))
            }
            _ => Err(format!("unknown method '{method}'")),
        }
    }
}
