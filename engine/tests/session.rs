//! Tests for `engine::session`: the API behind the visualizer (completion
//! inspector + chat), exercised exactly the way the page uses it.

use engine::model::Model;
use engine::session::Session;
use engine::tokenizer::Tokenizer;
use serde_json::{json, Value};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
const SYSTEM: &str = "You are a helpful AI assistant named SmolLM, trained by Hugging Face";

fn tokenizer() -> Tokenizer {
    Tokenizer::from_file(format!("{ROOT}/models/smollm2-135m-instruct/tokenizer.json")).unwrap()
}

fn session() -> Session {
    let dir = format!("{ROOT}/models/smollm2-135m-instruct");
    Session::new(Model::load(&dir).unwrap(), tokenizer(), "test", 256, 512).unwrap()
}

fn ok(s: &mut Session, method: &str, args: Value) -> Value {
    s.call(method, &args).unwrap_or_else(|e| panic!("{method}: {e}"))
}

const GREEDY: fn() -> Value = || json!({"temperature": 0.0, "repeat_penalty": 1.0});

/// Generate a whole chat reply greedily; returns (token ids, stop reason).
fn reply(s: &mut Session, max: usize) -> (Vec<u32>, String) {
    let mut ids = vec![];
    loop {
        let mut args = GREEDY();
        args["n"] = json!(8);
        let r = ok(s, "chat_step", args);
        ids.extend(r["tokens"].as_array().unwrap().iter().map(|t| t["id"].as_u64().unwrap() as u32));
        if r["done"].as_bool().unwrap() || ids.len() >= max {
            return (ids, r["stop"].as_str().unwrap_or("").to_string());
        }
    }
}

fn ids_of(state: &Value) -> Vec<u32> {
    state["tokens"].as_array().unwrap().iter().map(|t| t["id"].as_u64().unwrap() as u32).collect()
}

// ---------------------------------------------------------------- chat

/// The chat API must reproduce HF's greedy replies (same golden data as tests/golden.rs).
#[test]
fn chat_reply_matches_hf() {
    let cases: Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{ROOT}/golden/chat.json")).unwrap()).unwrap();
    let mut s = session();
    for c in cases.as_array().unwrap() {
        ok(&mut s, "chat_reset", json!({}));
        ok(&mut s, "chat_send", json!({"text": c["user"]}));
        let want: Vec<u32> = c["reply_ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
        let (got, stop) = reply(&mut s, want.len());
        assert_eq!(got, want, "reply to {}", c["user"]);
        if want.len() < 48 {
            // HF stopped at <|im_end|> before its 48-token budget, so must we.
            let (more, stop2) = reply(&mut s, 1);
            assert!(more.is_empty() && (stop.contains("im_end") || stop2.contains("im_end")), "stop: {stop} / {stop2}");
        }
        // Prompt tokens are the ChatML rendering HF uses.
        let st = ok(&mut s, "chat_state", json!({}));
        let prompt: Vec<u32> = c["ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
        assert_eq!(&ids_of(&st)[..prompt.len()], &prompt[..]);
    }
}

/// Multi-turn: the cache holds exactly the ChatML transcript, each turn closes
/// the previous reply with <|im_end|>\n, and only new tokens are prefilled.
#[test]
fn chat_turns_build_chatml_and_reuse_the_cache() {
    let tok = tokenizer();
    let mut s = session();
    let mut expected = format!("<|im_start|>system\n{SYSTEM}<|im_end|>\n");
    let mut cached = 0;
    for (turn, q) in ["Hi!", "What is 2+2?", "Thanks."].into_iter().enumerate() {
        let st = ok(&mut s, "chat_send", json!({"text": q}));
        let lt = &st["last_turn"];
        assert_eq!(lt["reused"].as_u64().unwrap() as usize, cached, "turn {turn}: everything before is reused");
        let new_part = format!("{}<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n", if turn > 0 { "<|im_end|>\n" } else { "" });
        let system_part = if turn == 0 { tok.encode(&format!("<|im_start|>system\n{SYSTEM}<|im_end|>\n")).len() } else { 0 };
        assert_eq!(lt["prefilled"].as_u64().unwrap() as usize, system_part + tok.encode(&new_part).len(), "turn {turn}: only new tokens prefilled");
        if turn > 0 {
            expected.push_str("<|im_end|>\n");
        }
        expected += &format!("<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n");
        let (ids, _) = reply(&mut s, 24);
        expected += &tok.decode(&ids);
        let st = ok(&mut s, "chat_state", json!({}));
        assert_eq!(st["transcript"].as_str().unwrap(), expected);
        assert_eq!(st["context"]["len"].as_u64().unwrap() as usize, ids_of(&st).len(), "cache length == tokens fed");
        cached = ids_of(&st).len();
    }
}

/// A reply cut short (Stop) is closed with <|im_end|> on the next message, and
/// the conversation's logits equal a from-scratch run of the same tokens.
#[test]
fn chat_stop_then_send_is_consistent() {
    let mut s = session();
    ok(&mut s, "chat_send", json!({"text": "Write a long poem about the sea."}));
    let mut args = GREEDY();
    args["n"] = json!(5);
    let r = ok(&mut s, "chat_step", args);
    assert_eq!(r["tokens"].as_array().unwrap().len(), 5);
    assert!(!r["done"].as_bool().unwrap());
    let st = ok(&mut s, "chat_stop", json!({}));
    assert_eq!(st["generating"], json!(false));
    let mut args = GREEDY();
    args["n"] = json!(3);
    let r = ok(&mut s, "chat_step", args.clone());
    assert!(r["tokens"].as_array().unwrap().is_empty(), "no tokens after stop");

    ok(&mut s, "chat_send", json!({"text": "Shorter please."}));
    let st = ok(&mut s, "chat_state", json!({}));
    let t = st["transcript"].as_str().unwrap();
    let cut = t.rfind("<|im_start|>user\nShorter").unwrap();
    assert!(t[..cut].ends_with("<|im_end|>\n"), "the stopped reply was closed: {:?}", &t[cut.saturating_sub(40)..cut]);

    // Cached incremental chat == running the whole token sequence at once.
    let first = ok(&mut s, "chat_step", args)["tokens"][0]["id"].as_u64().unwrap();
    let mut fresh = session();
    let r = ok(&mut fresh, "run", json!({"ids": ids_of(&st)}));
    assert_eq!(r["next"]["top"][0][0].as_u64().unwrap(), first, "greedy next token from scratch matches chat");
}

#[test]
fn chat_rejects_overflow_without_corrupting_state() {
    let dir = format!("{ROOT}/models/smollm2-135m-instruct");
    let mut s = Session::new(Model::load(&dir).unwrap(), tokenizer(), "tiny", 64, 48).unwrap();
    let before = ok(&mut s, "chat_state", json!({}));
    let err = s.call("chat_send", &json!({"text": "word ".repeat(60)})).unwrap_err();
    assert!(err.contains("full"), "{err}");
    let after = ok(&mut s, "chat_state", json!({}));
    assert_eq!(before["tokens"], after["tokens"]);
    assert_eq!(after["context"]["len"], json!(0));
    // A short message still works afterwards, and generation stops when the cache fills.
    ok(&mut s, "chat_send", json!({"text": "Hi"}));
    let (_, stop) = reply(&mut s, 1000);
    let st = ok(&mut s, "chat_state", json!({}));
    assert!(st["context"]["len"].as_u64().unwrap() <= 48);
    assert!(!stop.is_empty());
}

#[test]
fn chat_reply_tokens_carry_probabilities_and_bytes() {
    let mut s = session();
    ok(&mut s, "chat_send", json!({"text": "Say hello."}));
    let mut args = GREEDY();
    args["n"] = json!(4);
    let r = ok(&mut s, "chat_step", args);
    for t in r["tokens"].as_array().unwrap() {
        let p = t["p"].as_f64().unwrap();
        let alts = t["alts"].as_array().unwrap();
        assert_eq!(alts.len(), 5);
        // Greedy picks the top alternative, so its probability is the top one's.
        assert!((p - alts[0][1].as_f64().unwrap()).abs() < 1e-6);
        assert!(alts.windows(2).all(|w| w[0][1].as_f64() >= w[1][1].as_f64()));
        let bytes: Vec<u8> = t["bytes"].as_array().unwrap().iter().map(|b| b.as_u64().unwrap() as u8).collect();
        assert_eq!(String::from_utf8_lossy(&bytes), t["text"].as_str().unwrap());
    }
}

// ---------------------------------------------------------------- completion inspector

/// `run {ids}` and `run {text}` agree, and ids are taken verbatim.
#[test]
fn run_by_ids_equals_run_by_text() {
    let tok = tokenizer();
    let mut s = session();
    let text = "The quick brown fox";
    let a = ok(&mut s, "run", json!({"text": text}));
    let b = ok(&mut s, "run", json!({"ids": tok.encode(text)}));
    assert_eq!(a["next"], b["next"]);
    assert_eq!(a["tokens"], b["tokens"]);
    assert!(s.call("run", &json!({"ids": [999_999]})).is_err());
    assert!(s.call("run", &json!({"ids": []})).is_err());
}

/// Rewinding and re-decoding greedily reproduces the same continuation and the
/// same views as the first time.
#[test]
fn rewind_is_deterministic() {
    let mut s = session();
    ok(&mut s, "run", json!({"text": "Once upon a time"}));
    let n0 = ids_of(&ok(&mut s, "state", json!({}))).len();
    let mut args = GREEDY();
    args["n"] = json!(6);
    let first = ok(&mut s, "step", args.clone());
    let lens1 = ok(&mut s, "lens", json!({"mid": true}));
    let attn1 = ok(&mut s, "attn", json!({"layer": 7}));

    ok(&mut s, "rewind", json!({"len": n0}));
    assert_eq!(ids_of(&ok(&mut s, "state", json!({}))).len(), n0);
    let second = ok(&mut s, "step", args);
    assert_eq!(first["tokens"], second["tokens"]);
    assert_eq!(first["next"], second["next"]);
    assert_eq!(lens1, ok(&mut s, "lens", json!({"mid": true})), "lens after rewind + redo");
    assert_eq!(attn1, ok(&mut s, "attn", json!({"layer": 7})));
}

/// The logit lens is cached per (layer, position). Computing it, stepping, and
/// computing again must equal computing it once on the final sequence.
#[test]
fn lens_cache_is_invalidated_correctly() {
    let mut s = session();
    ok(&mut s, "run", json!({"text": "The capital of France is"}));
    ok(&mut s, "lens", json!({}));
    let mut args = GREEDY();
    args["n"] = json!(3);
    ok(&mut s, "step", args);
    let incremental = ok(&mut s, "lens", json!({}));
    let ids = ids_of(&ok(&mut s, "state", json!({})));

    let mut fresh = session();
    ok(&mut fresh, "run", json!({"ids": ids}));
    let scratch = ok(&mut fresh, "lens", json!({}));
    assert_eq!(incremental["answers"], scratch["answers"]);
    // Cells: identical up to float noise between prefill and decode (summation order).
    for (ra, rb) in incremental["cells"].as_array().unwrap().iter().zip(scratch["cells"].as_array().unwrap()) {
        for (a, b) in ra.as_array().unwrap().iter().zip(rb.as_array().unwrap()) {
            assert_eq!(a["top"][0][0], b["top"][0][0]);
            assert_eq!(a["t"].is_null(), b["t"].is_null(), "actual-next known in both");
            for k in ["f", "t"] {
                if let (Some(x), Some(y)) = (a[k][0].as_f64(), b[k][0].as_f64()) {
                    assert!((x - y).abs() < 2e-3, "{k}: {x} vs {y}");
                }
            }
        }
    }
}

/// Ablating a head changes the output; clearing ablations restores it exactly.
#[test]
fn ablation_round_trips() {
    let mut s = session();
    let base = ok(&mut s, "run", json!({"text": "The capital of France is"}));
    let all: Vec<[usize; 2]> = (0..30).flat_map(|l| (0..9).map(move |h| [l, h])).collect();
    let ablated = ok(&mut s, "ablate", json!({"heads": all}));
    assert_ne!(base["next"]["top"], ablated["next"]["top"]);
    assert_eq!(ablated["ablate"].as_array().unwrap().len(), 270);
    let restored = ok(&mut s, "ablate", json!({"heads": []}));
    assert_eq!(base["next"], restored["next"]);
    // Out-of-range heads are ignored, not an error.
    let r = ok(&mut s, "ablate", json!({"heads": [[99, 0], [0, 99]]}));
    assert!(r["ablate"].as_array().unwrap().is_empty());
}

#[test]
fn bad_requests_are_errors_not_panics() {
    let mut s = session();
    for (m, a) in [
        ("nope", json!({})),
        ("run", json!({})),
        ("attn", json!({})),
        ("attn", json!({"layer": 30})),
        ("kv", json!({"layer": 1000})),
        ("mlp", json!({"layer": 0, "pos": 9999})),
        ("rewind", json!({})),
        ("step", json!({"force": 49152})),
        ("chat_send", json!({})),
        ("ablate", json!({"heads": "x"})),
        // Ids beyond u32 must not wrap around to valid ones (2^32 + 5 → 5).
        ("run", json!({"ids": [4294967301u64]})),
        ("run", json!({"ids": [1, -1]})),
        ("step", json!({"force": 4294967301u64})),
        ("attn", json!({"layer": 4294967296u64})),
        ("attn", json!({"layer": 1.5})),
        ("attn", json!({"layer": "0"})),
        // Sampler settings that would produce NaN probabilities.
        ("step", json!({"repeat_penalty": 0})),
        ("step", json!({"repeat_penalty": -1})),
        ("step", json!({"temperature": -0.5})),
        ("step", json!({"top_p": 1.5})),
        ("step", json!({"top_k": -1})),
        ("step", json!({"temperature": "hot"})),
        ("chat_step", json!({"repeat_penalty": 0})),
        ("chat_step", json!({"n": -3})),
    ] {
        assert!(s.call(m, &a).is_err(), "{m} {a} should be an error");
    }
    // A rejected request leaves the settings alone.
    let st = ok(&mut s, "state", json!({}));
    assert_eq!(st["sampler"]["repeat_penalty"].as_f64().unwrap() as f32, 1.1);
    let before = st["tokens"].as_array().unwrap().len();
    assert!(s.call("step", &json!({"n": 3, "force": 49152, "temperature": 0.0})).is_err());
    let st = ok(&mut s, "state", json!({}));
    assert_eq!(st["tokens"].as_array().unwrap().len(), before, "failed step must not generate");
    assert!(st["sampler"]["temperature"].as_f64().unwrap() > 0.0, "failed step must not change the sampler");
    // Rewind is clamped, not an error.
    ok(&mut s, "rewind", json!({"len": 0}));
    ok(&mut s, "rewind", json!({"len": 100000}));
}
