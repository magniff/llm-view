//! Tests for `engine::session`: the API behind the visualizer, exercised
//! exactly the way the page uses it.

use engine::model::Model;
use engine::session::Session;
use engine::tokenizer::Tokenizer;
use serde_json::{json, Value};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/..");

fn tokenizer() -> Tokenizer {
    Tokenizer::from_file(format!("{ROOT}/models/smollm2-135m/tokenizer.json")).unwrap()
}

fn session() -> Session {
    let dir = format!("{ROOT}/models/smollm2-135m");
    Session::new(Model::load(&dir).unwrap(), tokenizer(), "test", 256).unwrap()
}

fn ok(s: &mut Session, method: &str, args: Value) -> Value {
    s.call(method, &args).unwrap_or_else(|e| panic!("{method}: {e}"))
}

const GREEDY: fn() -> Value = || json!({"temperature": 0.0, "repeat_penalty": 1.0, "dry": 0.0}); // HF greedy: no penalties

fn ids_of(state: &Value) -> Vec<u32> {
    state["tokens"].as_array().unwrap().iter().map(|t| t["id"].as_u64().unwrap() as u32).collect()
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
    let walk1 = ok(&mut s, "walk_layer", json!({"layer": 7}));

    ok(&mut s, "rewind", json!({"len": n0}));
    assert_eq!(ids_of(&ok(&mut s, "state", json!({}))).len(), n0);
    let second = ok(&mut s, "step", args);
    assert_eq!(first["tokens"], second["tokens"]);
    assert_eq!(first["next"], second["next"]);
    assert_eq!(walk1, ok(&mut s, "walk_layer", json!({"layer": 7})), "layer internals after rewind + redo");
}

#[test]
fn bad_requests_are_errors_not_panics() {
    let mut s = session();
    let sampler0 = ok(&mut s, "state", json!({}))["sampler"].clone();
    for (m, a) in [
        ("nope", json!({})),
        ("run", json!({})),
        ("walk_layer", json!({})),
        ("walk_layer", json!({"layer": 30})),
        ("lens", json!({})),
        ("rewind", json!({})),
        ("step", json!({"force": 49152})),
        // Ids beyond u32 must not wrap around to valid ones (2^32 + 5 → 5).
        ("run", json!({"ids": [4294967301u64]})),
        ("run", json!({"ids": [1, -1]})),
        ("step", json!({"force": 4294967301u64})),
        ("walk_layer", json!({"layer": 4294967296u64})),
        ("walk_layer", json!({"layer": 1.5})),
        ("walk_layer", json!({"layer": "0"})),
        // Sampler settings that would produce NaN probabilities.
        ("step", json!({"repeat_penalty": 0})),
        ("step", json!({"repeat_penalty": -1})),
        ("step", json!({"temperature": -0.5})),
        ("step", json!({"top_p": 1.5})),
        ("step", json!({"top_k": -1})),
        ("step", json!({"temperature": "hot"})),
        ("step", json!({"dry": -1})),
        ("step", json!({"dry": "lots"})),
        ("step", json!({"n": -3})),
        ("chat_send", json!({"text": "hi"})),
    ] {
        assert!(s.call(m, &a).is_err(), "{m} {a} should be an error");
    }
    // A rejected request leaves the settings alone.
    let st = ok(&mut s, "state", json!({}));
    assert_eq!(st["sampler"], sampler0, "rejected requests must not change the sampler");
    let before = st["tokens"].as_array().unwrap().len();
    assert!(s.call("step", &json!({"n": 3, "force": 49152, "temperature": 0.5})).is_err());
    let st = ok(&mut s, "state", json!({}));
    assert_eq!(st["tokens"].as_array().unwrap().len(), before, "failed step must not generate");
    assert_eq!(st["sampler"], sampler0, "failed step must not change the sampler");
    // Rewind is clamped, not an error.
    ok(&mut s, "rewind", json!({"len": 0}));
    ok(&mut s, "rewind", json!({"len": 100000}));
}

/// The step-by-step view's matrices chain together: each layer starts where the
/// previous one ended, the first starts at the embeddings, RoPE leaves position
/// 0 alone, and every attention row is a distribution.
#[test]
fn walk_matrices_chain_together() {
    let mut s = session();
    let ends = ok(&mut s, "walk_ends", json!(null));
    let n = ends["n"].as_u64().unwrap() as usize;
    assert!(n > 2 && n <= 16);
    let l0 = ok(&mut s, "walk_layer", json!({"layer": 0}));
    let l1 = ok(&mut s, "walk_layer", json!({"layer": 1}));
    assert_eq!(l0["x_in"], ends["embed"]);
    assert_eq!(l0["x_out"], l1["x_in"]);
    assert_eq!(l0["q"][0], l0["q_pre"][0]);
    assert_ne!(l0["q"][1], l0["q_pre"][1]);
    assert_eq!(l0["q"].as_array().unwrap().len(), n);
    assert_eq!(l0["k"][0].as_array().unwrap().len(), 192);
    assert_eq!(l0["gate"][0].as_array().unwrap().len(), 1536);
    for (p, heads) in l0["probs"].as_array().unwrap().iter().enumerate() {
        for h in heads.as_array().unwrap() {
            let h = h.as_array().unwrap();
            assert_eq!(h.len(), p + 1, "position {p} sees positions 0..={p}");
            assert!((h.iter().map(|x| x.as_f64().unwrap()).sum::<f64>() - 1.0).abs() < 1e-3);
        }
    }
    assert_eq!(ends["out"][0]["top"].as_array().unwrap().len(), 5);
    assert!(s.call("walk_layer", &json!({"layer": 30})).is_err());
}

/// The step-by-step view recomputes every layer from the trace and the KV
/// cache, decode steps included; each layer's recomputed output must land
/// where the real forward pass put it (the next layer's input, from the trace),
/// and the last one must give the same prediction the model made.
#[test]
fn walk_recomputes_the_forward_pass() {
    let mut s = session();
    let mut args = GREEDY();
    args["n"] = json!(3);
    let st = ok(&mut s, "step", args); // decode steps too, not just the prefill
    let f = |v: &Value| v.as_f64().unwrap();
    let close = |a: &Value, b: &Value| {
        let (a, b) = (a.as_array().unwrap(), b.as_array().unwrap());
        let scale = a.iter().map(|x| f(x).abs()).fold(1.0, f64::max);
        a.iter().zip(b).map(|(x, y)| (f(x) - f(y)).abs()).fold(0.0, f64::max) / scale
    };
    let layers: Vec<Value> = (0..30).map(|l| ok(&mut s, "walk_layer", json!({"layer": l}))).collect();
    let ends = ok(&mut s, "walk_ends", json!(null));
    let n = ends["n"].as_u64().unwrap() as usize;
    for p in 0..n {
        for l in 0..29 {
            let d = close(&layers[l]["x_out"][p], &layers[l + 1]["x_in"][p]);
            assert!(d < 1e-3, "layer {l}, position {p}: off by {d}");
        }
        let d = close(&layers[29]["x_out"][p], &ends["out"][p]["x"]);
        assert!(d < 1e-3, "last layer, position {p}: off by {d}");
    }
    // The last row's top token is what the model predicts next.
    assert_eq!(ends["out"][n - 1]["top"][0][0], st["next"]["top"][0][0]);
}

/// The penalties must not break text that has to repeat digits: DRY used to
/// read "100 + 20" echoed from earlier as a loop and push the next "0" down
/// ("100 + 201"). Penalties apply to words only, so digits are left alone.
#[test]
fn penalties_leave_digits_alone() {
    let mut s = session();
    ok(&mut s, "run", json!({"text": "100 + 200 = 300\n100 + 200 = 300\n100 + 200 ="}));
    let st = ok(&mut s, "step", json!({"n": 4, "temperature": 0.0, "dry": 0.8, "repeat_penalty": 1.3}));
    let text: String = st["tokens"].as_array().unwrap().iter().skip(st["n_prompt"].as_u64().unwrap() as usize)
        .map(|t| t["text"].as_str().unwrap()).collect();
    assert!(text.starts_with(" 300"), "{text:?}");
}

/// The default sampler is the page's "Focused" preset: greedy, with DRY on.
#[test]
fn default_sampler_is_focused() {
    let mut s = session();
    let st = ok(&mut s, "state", json!({}));
    assert_eq!(st["sampler"]["temperature"].as_f64(), Some(0.0));
    assert!(st["sampler"]["dry"].as_f64().unwrap() > 0.0);
    assert_eq!(st["tokens"].as_array().unwrap().iter().map(|t| t["text"].as_str().unwrap()).collect::<String>(), "The love is");
}

/// The lens on the last layer is the model's own prediction; on earlier layers
/// it is a distribution too, and it reports where the final answer ranks.
#[test]
fn walk_lens_ends_at_the_prediction() {
    let mut s = session();
    let ends = ok(&mut s, "walk_ends", json!(null));
    let last = ok(&mut s, "walk_lens", json!({"layer": 29}));
    let n = ends["n"].as_u64().unwrap() as usize;
    for p in 0..n {
        assert_eq!(last["rows"][p]["top"], ends["out"][p]["top"]);
        assert_eq!(last["rows"][p]["final"][3], 0, "the final answer is the last layer's top token");
    }
    let early = ok(&mut s, "walk_lens", json!({"layer": 0}));
    let ps: f64 = early["rows"][0]["top"].as_array().unwrap().iter().map(|t| t[3].as_f64().unwrap()).sum();
    assert!(ps > 0.0 && ps <= 1.0001);
    assert!(s.call("walk_lens", &json!({"layer": 30})).is_err());
}
