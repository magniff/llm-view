//! Compare the engine against HuggingFace transformers, tensor by tensor.
//! Regenerate the golden data with `.venv/bin/python reference/dump_golden.py`.

use std::sync::OnceLock;

use engine::hooks::{NoHooks, Recorder};
use engine::model::Model;
use engine::safetensors::SafeTensors;
use engine::sampler::Sampler;
use engine::tokenizer::Tokenizer;

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/..");

fn model() -> &'static Model {
    static M: OnceLock<Model> = OnceLock::new();
    M.get_or_init(|| Model::load(format!("{ROOT}/models/smollm2-135m")).unwrap())
}

fn tokenizer() -> Tokenizer {
    Tokenizer::from_file(format!("{ROOT}/models/smollm2-135m/tokenizer.json")).unwrap()
}

fn meta() -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(format!("{ROOT}/golden/meta.json")).unwrap()).unwrap()
}

fn ids(v: &serde_json::Value) -> Vec<u32> {
    v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()
}

/// max |a-b| / max |b| — error relative to the tensor's own scale.
fn rel_err(a: &[f32], b: &[f32]) -> f32 {
    let diff = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max);
    let scale = b.iter().map(|y| y.abs()).fold(0.0, f32::max).max(1e-6);
    diff / scale
}

#[test]
fn tokenizer_matches_hf() {
    let tok = tokenizer();
    let cases: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{ROOT}/golden/tokenizer_cases.json")).unwrap()).unwrap();
    let mut failures = vec![];
    for c in cases.as_array().unwrap() {
        let text = c["text"].as_str().unwrap();
        let want = ids(&c["ids"]);
        let got = tok.encode(text);
        if got != want {
            failures.push(format!("{text:?}\n  want {want:?}\n  got  {got:?}"));
        }
        assert_eq!(tok.decode(&got), text, "round trip");
    }
    assert!(failures.is_empty(), "tokenizer mismatches:\n{}", failures.join("\n"));
}

#[test]
fn every_activation_matches_hf() {
    let m = model();
    let golden = SafeTensors::open(format!("{ROOT}/golden/trace.safetensors")).unwrap();
    let prompt = ids(&meta()["prompt_ids"]);

    let mut rec = Recorder::default();
    let mut cache = m.new_cache(prompt.len());
    m.forward(&prompt, &mut cache, &mut rec);

    let mut names: Vec<&String> = golden.tensors.keys().collect();
    names.sort();
    let mut worst = (0.0f32, String::new());
    let mut bad = vec![];
    for name in names {
        let want = golden.get(name).unwrap();
        let got = rec.tensors.get(name).unwrap_or_else(|| panic!("engine never emitted hook {name}"));
        assert_eq!(got.shape, want.shape, "{name} shape");
        let e = rel_err(&got.data, &want.data);
        if e > worst.0 {
            worst = (e, name.clone());
        }
        if e > 1e-4 {
            bad.push(format!("{name}: rel err {e:.2e}"));
        }
    }
    eprintln!("{} tensors compared, worst rel err {:.2e} at {}", golden.tensors.len(), worst.0, worst.1);
    assert!(bad.is_empty(), "mismatches:\n{}", bad.join("\n"));
}

/// Prefill + one-token-at-a-time decoding must give the same logits as
/// processing the whole sequence in one go with no history.
#[test]
fn kv_cache_matches_full_recompute() {
    let m = model();
    let seq = ids(&meta()["prompt_ids"]);
    let full = m.forward(&seq, &mut m.new_cache(seq.len()), &mut NoHooks);

    let split = 4;
    let mut cache = m.new_cache(seq.len());
    m.forward(&seq[..split], &mut cache, &mut NoHooks);
    for (i, &t) in seq.iter().enumerate().skip(split) {
        let step = m.forward(&[t], &mut cache, &mut NoHooks);
        // Bit-identical: decode (one row, `dot`) and prefill (`dot4`) share a reduction order.
        assert_eq!(step.row(0), full.row(i), "position {i}: cached decode differs from full prefill");
    }
    assert_eq!(cache.len, seq.len());
}

#[test]
fn greedy_generation_matches_hf() {
    let m = model();
    let meta = meta();
    let prompt = ids(&meta["gen_prompt_ids"]);
    let want = ids(&meta["gen_ids"]);

    let mut cache = m.new_cache(prompt.len() + want.len());
    let mut sampler = Sampler::greedy();
    let logits = m.forward(&prompt, &mut cache, &mut NoHooks);
    let mut next = sampler.sample(logits.row(logits.rows() - 1));
    let mut got = vec![];
    for _ in 0..want.len() {
        got.push(next);
        let l = m.forward(&[next], &mut cache, &mut NoHooks);
        next = sampler.sample(l.row(0));
    }
    let tok = tokenizer();
    assert_eq!(got, want, "\n got: {:?}\nwant: {:?}", tok.decode(&got), tok.decode(&want));
}

fn instruct() -> &'static Model {
    static M: OnceLock<Model> = OnceLock::new();
    M.get_or_init(|| Model::load(format!("{ROOT}/models/smollm2-135m-instruct")).unwrap())
}

/// Our ChatML rendering + tokenization must equal HF's `apply_chat_template`,
/// and the greedy reply through `generate` must match HF token for token.
#[test]
fn chat_template_and_reply_match_hf() {
    use engine::chat::render_single;
    use engine::generate::generate;
    let cases: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{ROOT}/golden/chat.json")).unwrap()).unwrap();
    let m = instruct();
    let tok = tokenizer();
    for c in cases.as_array().unwrap() {
        let text = render_single(None, c["user"].as_str().unwrap());
        assert_eq!(text, c["text"].as_str().unwrap(), "rendered template");
        let prompt = tok.encode(&text);
        assert_eq!(prompt, ids(&c["ids"]), "template token ids");

        let want = ids(&c["reply_ids"]);
        let mut cache = m.new_cache(512);
        let stats = generate(m, &tok, &mut cache, &prompt, &mut Sampler::greedy(), 48, &[m.cfg.eos_token], &mut NoHooks, |_| {});
        assert_eq!(stats.generated, want, "\n got: {:?}\nwant: {:?}", tok.decode(&stats.generated), tok.decode(&want));
    }
}

/// After two turns, the cached token stream must be exactly the ChatML
/// transcript of the conversation.
#[test]
fn conversation_transcript_is_chatml() {
    use engine::chat::{Conversation, DEFAULT_SYSTEM};
    use engine::generate::generate;
    let m = instruct();
    let tok = tokenizer();
    let mut conv = Conversation::new(m, &tok, None, 512);
    let mut expected = format!("<|im_start|>system\n{DEFAULT_SYSTEM}<|im_end|>\n");
    for (turn, q) in ["Hi!", "What is 2+2?"].into_iter().enumerate() {
        let prompt = conv.prompt_for(&tok, q);
        let s = generate(m, &tok, &mut conv.cache, &prompt, &mut Sampler::greedy(), 24, &[m.cfg.eos_token], &mut NoHooks, |_| {});
        conv.record(&tok, &prompt, &s.generated);
        if turn > 0 {
            expected.push_str("<|im_end|>\n"); // closes the previous assistant reply
        }
        expected += &format!("<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{}", tok.decode(&s.generated));
        assert_eq!(tok.decode(&conv.tokens), expected);
        assert_eq!(conv.cache.len, conv.tokens.len());
    }
}

/// Prompts generate() can't run are a no-op, not a panic: nothing generated,
/// cache untouched, `on_text` never called.
#[test]
fn generate_refuses_empty_and_oversized_prompts() {
    use engine::generate::generate;
    let m = model();
    let tok = tokenizer();
    let mut cache = m.new_cache(16);
    let mut called = false;
    let s = generate(m, &tok, &mut cache, &[], &mut Sampler::greedy(), 8, &[], &mut NoHooks, |_| called = true);
    assert!(s.generated.is_empty() && cache.len == 0 && !called);
    let s = generate(m, &tok, &mut cache, &[1; 17], &mut Sampler::greedy(), 8, &[], &mut NoHooks, |_| called = true);
    assert!(s.generated.is_empty() && cache.len == 0 && !called);
    // Exactly full: prefill fits, but there's no room to feed anything back.
    let s = generate(m, &tok, &mut cache, &[1; 16], &mut Sampler::greedy(), 8, &[], &mut NoHooks, |_| {});
    assert!(s.generated.is_empty() && cache.len == 16);
}
