//! Chat tuning for SmolLM2-135M-Instruct: what keeps answers correct
//! (arithmetic included) while stopping open-ended prompts from rambling?
//! Compares repetition penalties (plain, and words-only), a growing bias
//! towards `<|im_end|>`, temperature, and a "be brief" system prompt.
//!
//!     cargo run --release --example chat_tuning
//!     (sample replies go to stderr; the table to stdout)

use engine::{chat::render_single, chat::DEFAULT_SYSTEM, hooks::NoHooks, model::Model, sampler::Sampler, tokenizer::Tokenizer};
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

#[derive(Clone, Copy)]
struct Setting {
    name: &'static str,
    temp: f32,
    rep: f32,
    /// Repetition penalty only for tokens containing a letter (digits, operators, punctuation exempt).
    words_only: bool,
    brief: bool,
    end_after: usize,
    end_slope: f32,
    dry: f32,
    /// Penalties look only at this reply, not at the prompt (which a correct answer often echoes).
    reply_only: bool,
    /// DRY never penalizes tokens without a letter (numbers, operators, punctuation).
    dry_words: bool,
}

const S: Setting = Setting { name: "", temp: 0.2, rep: 1.0, words_only: false, brief: false, end_after: 0, end_slope: 0.0, dry: 0.8, reply_only: false, dry_words: false };

/// One reply: (tokens, ended by <|im_end|>).
fn reply(m: &Model, tok: &Tokenizer, s: &Setting, sampler: &mut Sampler, q: &str, max: usize) -> (Vec<u32>, bool) {
    let sys = s.brief.then(|| format!("{DEFAULT_SYSTEM}. Answer briefly."));
    let prompt = tok.encode(&render_single(sys.as_deref(), q));
    let mut cache = m.new_cache(prompt.len() + max + 1);
    let mut hist = prompt.clone();
    let mut logits = m.forward(&prompt, &mut cache, &mut NoHooks).row(prompt.len() - 1).to_vec();
    let eos = m.cfg.eos_token;
    let mut out = vec![];
    while out.len() < max {
        // The longer the reply, the more likely ending it becomes.
        if s.end_slope > 0.0 && out.len() > s.end_after {
            logits[eos as usize] += s.end_slope * (out.len() - s.end_after) as f32;
        }
        let next = sampler.sample_with_history(&logits, if s.reply_only { &out } else { &hist });
        if next == eos || next == 0 {
            return (out, true);
        }
        out.push(next);
        hist.push(next);
        logits = m.forward(&[next], &mut cache, &mut NoHooks).row(0).to_vec();
    }
    (out, false)
}

fn main() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/smollm2-135m-instruct");
    let m = Model::load(dir).unwrap();
    let tok = Tokenizer::from_file(format!("{dir}/tokenizer.json")).unwrap();
    let specials: Vec<u32> = tok.specials.iter().map(|&(_, id)| id).collect();
    // Tokens with no letter in them: digits, operators, punctuation, whitespace.
    let no_letters: Vec<u32> = (0..m.cfg.vocab as u32)
        .filter(|&id| tok.decode_bytes(&[id]).iter().all(|b| b.is_ascii() && !b.is_ascii_alphabetic()))
        .collect();
    eprintln!("{} tokens contain no letter", no_letters.len());

    let arith = [
        ("What is 100 + 200?", "300"), ("compute 100 + 200", "300"), ("What is 7 + 5?", "12"), ("What is 12 * 3?", "36"),
        ("What is 45 - 17?", "28"), ("What is 15 + 27?", "42"), ("What is 9 times 8?", "72"), ("What is 1000 - 1?", "999"),
        ("What is 2 + 2?", "4"), ("What is 50 divided by 5?", "10"),
    ];
    let facts = [
        ("What is the capital of France?", "Paris"), ("What is the capital of Germany?", "Berlin"),
        ("Who wrote Romeo and Juliet?", "Shakespeare"), ("What is the largest planet in our solar system?", "Jupiter"),
        ("Repeat after me: the cat sat on the mat", "the cat sat on the mat"),
    ];
    let open = ["What is love?", "What is the meaning of life?", "What is happiness?", "Tell me about yourself.",
                "How was your day?", "Write a poem about the sea."];
    let settings = [
        Setting { name: "default: T.2 DRY.8", ..S },
        Setting { name: "greedy, no penalties", temp: 0.0, dry: 0.0, ..S },
        Setting { name: "DRYw + w1.1 + e40 T.2", dry_words: true, rep: 1.1, words_only: true, end_after: 40, end_slope: 0.05, ..S },
        Setting { name: "DRYw + w1.1 + e40 T.1", temp: 0.1, dry_words: true, rep: 1.1, words_only: true, end_after: 40, end_slope: 0.05, ..S },
        Setting { name: "DRYw + w1.1 + e40 T0", temp: 0.0, dry_words: true, rep: 1.1, words_only: true, end_after: 40, end_slope: 0.05, ..S },
        Setting { name: "DRYw + w1.15 + e40 T.2", dry_words: true, rep: 1.15, words_only: true, end_after: 40, end_slope: 0.05, ..S },
        Setting { name: "DRYw + w1.1 + e40 T.3", temp: 0.3, dry_words: true, rep: 1.1, words_only: true, end_after: 40, end_slope: 0.05, ..S },
    ];
    let (m, tok, specials, no_letters) = (&m, &tok, &specials, &no_letters);
    let seeds = 10;
    let rows: Vec<String> = std::thread::scope(|sc| {
        let hs: Vec<_> = settings.iter().map(|s| sc.spawn(move || {
            let mk = |seed| {
                let mut x = Sampler::new(s.temp, 0, 0.9, seed)
                    .with_dry(m.cfg.vocab, |id| specials.contains(&id) || Sampler::dry_breaker_bytes(&tok.decode_bytes(&[id])));
                x.dry_multiplier = s.dry;
                x.repeat_penalty = s.rep;
                x.no_penalty = specials.clone();
                if s.words_only { x.no_penalty.extend(no_letters); }
                if s.dry_words {
                    x.dry_exempt = vec![false; m.cfg.vocab];
                    for &id in no_letters { x.dry_exempt[id as usize] = true; }
                }
                x
            };
            let score = |qs: &[(&str, &str)]| {
                let (mut ok, mut n) = (0, 0);
                for &(q, ans) in qs {
                    for seed in 0..seeds {
                        let (ids, _) = reply(m, tok, s, &mut mk(seed), q, 120);
                        let text = tok.decode(&ids);
                        n += 1;
                        if text.to_lowercase().contains(&ans.to_lowercase()) { ok += 1; }
                        else if seed == 0 { eprintln!("--- [{}] WRONG {q}\n{}\n", s.name, text); }
                    }
                }
                100 * ok / n
            };
            let (a, f) = (score(&arith), score(&facts));
            let (mut lens, mut ended, mut looped, mut rep, mut n) = (vec![], 0, 0, 0, 0);
            for q in open {
                for seed in 0..seeds {
                    let (ids, end) = reply(m, tok, s, &mut mk(seed), q, 300);
                    n += 1;
                    ended += end as usize;
                    let ls = loop_score(&ids);
                    looped += (ls > 10) as usize;
                    rep += (ls > 2) as usize;
                    if seed == 0 { eprintln!("--- [{}] {q} ({} tokens)\n{}\n", s.name, ids.len(), tok.decode(&ids)); }
                    lens.push(ids.len());
                }
            }
            lens.sort();
            format!("{:<24} arithmetic {:>3}%  facts {:>3}% | open: median {:>3} tok, p90 {:>3}, ended {:>3}%, loops {:>2}/{n}, any repeats {:>2}/{n}",
                s.name, a, f, lens[n / 2], lens[n * 9 / 10], 100 * ended / n, looped, rep)
        })).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for r in rows { println!("{r}"); }
}
