//! How do sampler settings change what SmolLM2-135M-Instruct says? Runs the
//! same prompts over many seeds and measures accuracy on short questions and
//! how often long replies fall into loops. This is how the defaults were chosen.
//!
//!     cargo run --release --example sampling_eval
//!     (sample replies go to stderr; the table to stdout)

use engine::{chat::render_single, generate::generate, hooks::NoHooks, model::Model, sampler::Sampler, tokenizer::Tokenizer};
use std::collections::HashMap;

/// How many 8-token windows exactly repeat an earlier, non-overlapping one.
fn loop_score(ids: &[u32]) -> usize {
    let mut seen: HashMap<&[u32], usize> = HashMap::new();
    let mut reps = 0;
    for (i, w) in ids.windows(8).enumerate() {
        match seen.get(w) { Some(&j) if i >= j + 8 => reps += 1, Some(_) => {}, None => { seen.insert(w, i); } }
    }
    reps
}

fn main() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/smollm2-135m-instruct");
    let m = Model::load(dir).unwrap();
    let tok = Tokenizer::from_file(format!("{dir}/tokenizer.json")).unwrap();
    let specials: Vec<u32> = tok.specials.iter().map(|&(_, id)| id).collect();
    let short = [("What is the capital of France?", "Paris"), ("What is the capital of Germany?", "Berlin"),
                 ("What is 7 + 5?", "12"), ("What color is the sky on a clear day?", "blue")];
    let long = ["How was your day?", "Tell me a long story about a dragon."];
    // (name, temperature, repeat penalty, dry)
    let settings: [(&str, f32, f32, bool); 4] = [("card: T0.2 p0.9", 0.2, 1.0, false), ("card + DRY", 0.2, 1.0, true),
                                                 ("rep 1.1 (exempt)", 0.2, 1.1, false), ("DRY + rep 1.05", 0.2, 1.05, true)];
    let (m, tok, specials) = (&m, &tok, &specials);
    let rows: Vec<String> = std::thread::scope(|sc| {
        let hs: Vec<_> = settings.iter().map(|&(name, t, rp, dry)| sc.spawn(move || {
            let mk = |seed| {
                let mut s = Sampler::new(t, 0, 0.9, seed);
                s.repeat_penalty = rp;
                s.no_penalty = specials.clone();
                if dry { s = s.with_dry(m.cfg.vocab, |id| specials.contains(&id) || Sampler::dry_breaker_bytes(&tok.decode_bytes(&[id]))); }
                s
            };
            let (mut correct, mut concise, mut n) = (0, 0, 0);
            for (q, ans) in short {
                let ids = tok.encode(&render_single(None, q));
                for seed in 0..20 {
                    let mut cache = m.new_cache(300);
                    let st = generate(m, tok, &mut cache, &ids, &mut mk(seed), 120, &[m.cfg.eos_token], &mut NoHooks, |_| {});
                    n += 1;
                    if tok.decode(&st.generated).contains(ans) { correct += 1; }
                    if st.generated.len() <= 40 && st.stopped_by.is_some() { concise += 1; }
                }
            }
            let mut row = format!("{name:<18} short Qs: correct {:>3}%, concise {:>3}%", 100 * correct / n, 100 * concise / n);
            for q in long {
                let ids = tok.encode(&render_single(None, q));
                let (mut looped, mut ended, mut lens) = (0, 0, vec![]);
                for seed in 0..10 {
                    let mut cache = m.new_cache(700);
                    let st = generate(m, tok, &mut cache, &ids, &mut mk(seed), 400, &[m.cfg.eos_token], &mut NoHooks, |_| {});
                    if loop_score(&st.generated) > 10 { looped += 1; }
                    if st.stopped_by.is_some() { ended += 1; }
                    lens.push(st.generated.len());
                    if seed == 0 { eprintln!("--- [{name}] {q}\n{}\n", tok.decode(&st.generated)); }
                }
                lens.sort();
                row += &format!(" | {:?}: loops {looped}/10, ends {ended}/10, median {} tok", &q[..12], lens[5]);
            }
            row
        })).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for r in rows { println!("{r}"); }
}
