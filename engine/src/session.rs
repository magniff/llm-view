//! An inspection session over the model, used by the visualizer, both by the
//! native server and by the in-browser WebAssembly build.
//!
//! A session has two independent parts sharing one model:
//!
//! - **completion** (the inspector): one token sequence, its KV cache, and a
//!   [`Trace`] of every activation recorded while running it;
//! - **chat**: a multi-turn ChatML conversation with its own KV cache. It is
//!   deliberately lightweight: no trace (that costs ~0.3 MB per token), only
//!   the probability and top alternatives of each generated token.
//!
//! Everything is driven through
//! [`Session::call`]: a method name plus JSON arguments in, JSON out, so any
//! transport (HTTP, a Web Worker message) can sit in front of it.

use std::collections::{HashMap, HashSet};

use serde_json::{json, Value};

use crate::chat::{render_single, system_block, user_turn, DEFAULT_SYSTEM};
use crate::hooks::NoHooks;
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
    chat: ChatState,
}

/// Chat mode state. See the module docs.
struct ChatState {
    cache: KvCache,
    /// Every token fed into `cache`, in order.
    tokens: Vec<u32>,
    /// Tokens decided on but not fed yet (the system block before the first turn).
    pending: Vec<u32>,
    system: String,
    /// Logits after the last fed token: what the next reply token is sampled from.
    last_logits: Vec<f32>,
    /// An assistant reply has been started and not yet closed with `<|im_end|>\n`.
    assistant_open: bool,
    /// Currently producing a reply (between `chat_send` and the end of the turn).
    generating: bool,
    sampler: Sampler,
    last_turn: Value,
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

    fn run(&mut self, text: &str, chat: bool) -> Result<(), String> {
        let toks = self.tok.encode(&if chat { render_single(None, text) } else { text.to_string() });
        self.run_tokens(toks)
    }

    /// Start over from exact token ids (e.g. a chat conversation, so the
    /// inspector sees precisely the tokens chat produced, not a re-tokenization).
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
    /// A fresh session: `max_seq` positions for the inspector (which keeps a full
    /// trace, so keep it modest), `chat_ctx` positions for chat. The inspector
    /// starts out already run on a default prompt.
    pub fn new(model: Model, tok: Tokenizer, name: &str, max_seq: usize, chat_ctx: usize) -> Result<Self, String> {
        let specials = tok.specials.iter().map(|&(_, id)| id).collect();
        let mut sampler = Sampler::new(0.7, 0, 0.9, 1234);
        sampler.repeat_penalty = 1.1;
        // SmolLM2's suggested chat settings, plus a mild repetition penalty.
        let mut chat_sampler = Sampler::new(0.2, 0, 0.9, 42);
        chat_sampler.repeat_penalty = 1.1;
        let chat = ChatState {
            cache: model.new_cache(chat_ctx),
            tokens: vec![],
            pending: vec![],
            system: DEFAULT_SYSTEM.to_string(),
            last_logits: vec![],
            assistant_open: false,
            generating: false,
            sampler: chat_sampler,
            last_turn: Value::Null,
        };
        let mut s = Session {
            chat,
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
        s.chat_reset(None);
        Ok(s)
    }

    // ------------------------------------------------------------ chat

    fn chat_reset(&mut self, system: Option<&str>) {
        let c = &mut self.chat;
        if let Some(sys) = system {
            c.system = sys.to_string();
        }
        c.cache.clear();
        c.tokens.clear();
        c.pending = self.tok.encode(&system_block(&c.system));
        c.last_logits.clear();
        c.assistant_open = false;
        c.generating = false;
        c.last_turn = Value::Null;
    }

    /// Start a turn: close the previous reply if needed, feed the user's message
    /// plus `<|im_start|>assistant\n`. Only these new tokens are computed; the
    /// conversation so far is already in the KV cache.
    fn chat_send(&mut self, text: &str) -> Result<(), String> {
        let c = &mut self.chat;
        let mut feed = c.pending.clone();
        if c.assistant_open {
            // The model's own <|im_end|> was sampled but never fed (or the reply was
            // cut short), so close the turn the way the training data does.
            feed.extend(self.tok.encode("<|im_end|>\n"));
        }
        feed.extend(self.tok.encode(&user_turn(text)));
        // Leave room for at least a short reply.
        if c.cache.len + feed.len() + 8 > c.cache.max_seq {
            return Err(format!(
                "the conversation is full ({} of {} positions used); start a new chat",
                c.cache.len, c.cache.max_seq
            ));
        }
        let reused = c.cache.len;
        let t = Stopwatch::start();
        let logits = self.model.forward(&feed, &mut c.cache, &mut NoHooks);
        c.last_logits = logits.row(logits.rows() - 1).to_vec();
        c.tokens.extend_from_slice(&feed);
        c.pending.clear();
        c.assistant_open = true;
        c.generating = true;
        c.last_turn = json!({
            "prefilled": feed.len(), "reused": reused, "prefill_ms": t.ms(),
            "generated": 0, "decode_ms": 0.0, "stop": Value::Null,
        });
        Ok(())
    }

    /// Generate up to `n` reply tokens. Each comes with its probability under the
    /// model's own distribution and the top 5 alternatives.
    fn chat_step(&mut self, n: usize) -> Value {
        let eos = self.model.cfg.eos_token;
        let c = &mut self.chat;
        let t = Stopwatch::start();
        let mut out = vec![];
        let mut stop = Value::Null;
        for _ in 0..n {
            if !c.generating {
                break;
            }
            if c.cache.len >= c.cache.max_seq {
                c.generating = false;
                stop = json!("context window full");
                break;
            }
            let next = c.sampler.sample_with_history(&c.last_logits, &c.tokens);
            let top = top_logits(&c.last_logits, 5);
            let p = |l: f32| r4((l - top.lse).exp());
            let info = json!({
                "id": next,
                "text": tok_text(&self.tok, next),
                // Raw bytes too: one character can span several tokens, so the page
                // decodes the reply as a byte stream rather than token by token.
                "bytes": self.tok.decode_bytes(&[next]),
                "p": p(c.last_logits[next as usize]),
                "alts": top.top.iter().map(|&(id, l)| json!([tok_text(&self.tok, id), p(l)])).collect::<Vec<_>>(),
            });
            if next == eos || next == 0 {
                // End of turn. Not fed: chat_send closes the turn before the next message.
                c.generating = false;
                stop = json!(format!("end of turn ({})", tok_text(&self.tok, next)));
                break;
            }
            let logits = self.model.forward(&[next], &mut c.cache, &mut NoHooks);
            c.last_logits = logits.row(0).to_vec();
            c.tokens.push(next);
            out.push(info);
        }
        if let Some(lt) = c.last_turn.as_object_mut() {
            lt["generated"] = json!(lt["generated"].as_u64().unwrap_or(0) + out.len() as u64);
            lt["decode_ms"] = json!(lt["decode_ms"].as_f64().unwrap_or(0.0) + t.ms());
            if !stop.is_null() {
                lt["stop"] = stop.clone();
            }
        }
        json!({"tokens": out, "done": !c.generating, "stop": stop, "last_turn": c.last_turn, "context": self.chat_context()})
    }

    fn chat_context(&self) -> Value {
        let c = &self.chat;
        json!({"len": c.cache.len, "max_seq": c.cache.max_seq, "bytes": c.cache.bytes_used()})
    }

    fn chat_state(&self) -> Value {
        let c = &self.chat;
        let s = &c.sampler;
        json!({
            "system": c.system,
            "tokens": c.tokens.iter().map(|&id| json!({"id": id, "text": self.text(id), "special": self.specials.contains(&id)})).collect::<Vec<_>>(),
            "transcript": self.tok.decode(&c.tokens),
            "context": self.chat_context(),
            "last_turn": c.last_turn,
            "generating": c.generating,
            "sampler": {"temperature": s.temperature, "top_k": s.top_k, "top_p": s.top_p, "repeat_penalty": s.repeat_penalty},
        })
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
                    self.run(text, args["chat"].as_bool().unwrap_or(false))?;
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
            "mlp" => self.mlp_view(need("layer")?, int("pos")?.unwrap_or(0)),
            "tokenize" => {
                let text = args["text"].as_str().ok_or("tokenize: missing 'text'")?;
                Ok(self.tokenize_view(text, args["chat"].as_bool().unwrap_or(false)))
            }
            "chat_state" => Ok(self.chat_state()),
            "chat_reset" => {
                self.chat_reset(args["system"].as_str());
                Ok(self.chat_state())
            }
            "chat_send" => {
                let text = args["text"].as_str().ok_or("chat_send: missing 'text'")?;
                self.chat_send(text)?;
                Ok(self.chat_state())
            }
            "chat_step" => {
                let n = int("n")?.unwrap_or(1).clamp(1, 64);
                apply_sampling(&mut self.chat.sampler, args, method)?;
                Ok(self.chat_step(n))
            }
            "chat_stop" => {
                self.chat.generating = false;
                if let Some(lt) = self.chat.last_turn.as_object_mut() {
                    lt["stop"] = json!("stopped by user");
                }
                Ok(self.chat_state())
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
    if let Some(t) = t { s.temperature = t as f32 }
    if let Some(p) = p { s.top_p = p as f32 }
    if let Some(r) = r { s.repeat_penalty = r as f32 }
    if let Some(k) = k { s.top_k = usize::try_from(k).unwrap_or(usize::MAX) }
    Ok(())
}
