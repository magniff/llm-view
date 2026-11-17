//! Interactive text completion.
//!
//!   cargo run                    # SmolLM2-135M, the model the visualizer shows
//!   cargo run -- --size 360m     # the bigger one
//!
//! Models are downloaded into `models/` on first run. Each prompt is an
//! independent completion: type the start of a text, the model continues it.
//! End a line with `\` to keep typing on the next line.

use std::io::{BufRead, Write};

use engine::download;
use engine::generate::generate;
use engine::hooks::NoHooks;
use engine::model::Model;
use engine::sampler::Sampler;
use engine::tokenizer::Tokenizer;

const DIM: &str = "\x1b[2m";
const BOLD: &str = "\x1b[1m";
const RESET: &str = "\x1b[0m";

const HELP: &str = "commands: /raw (toggle showing the exact tokens)  /help  /quit   (Ctrl-D also quits)";

struct Args {
    model: Option<String>,
    size: String,
    temp: f32,
    top_k: usize,
    top_p: f32,
    repeat_penalty: f32,
    dry: f32,
    seed: u64,
    max_new: usize,
    ctx: usize,
}

fn parse_args() -> engine::Result<Args> {
    let seed = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos() as u64;
    let mut a = Args { model: None, size: "135m".into(), temp: 0.8, top_k: 0, top_p: 0.9, repeat_penalty: 1.0, dry: 0.8, seed, max_new: 200, ctx: 2048 };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--model" => a.model = Some(val()?),
            "--size" => a.size = val()?,
            "--temp" => a.temp = val()?.parse()?,
            "--top-k" => a.top_k = val()?.parse()?,
            "--top-p" => a.top_p = val()?.parse()?,
            "--repeat-penalty" => a.repeat_penalty = val()?.parse()?,
            "--dry" => a.dry = val()?.parse()?,
            "--seed" => a.seed = val()?.parse()?,
            "-n" | "--max-new" => a.max_new = val()?.parse()?,
            "--ctx" => a.ctx = val()?.parse()?,
            "-h" | "--help" => {
                println!(
                    "cargo run -- [--size 135m|360m] [--model DIR] [--temp T (0.8)] [--top-k K] [--top-p P (0.9)]\n     [--seed N] [--repeat-penalty R] [--dry STRENGTH (0.8; 0 = off)]\n     [-n MAX_NEW_TOKENS (200)] [--ctx CONTEXT_LEN]\n\n{HELP}"
                );
                std::process::exit(0);
            }
            _ => return Err(format!("unknown argument {flag} (try --help)").into()),
        }
    }
    Ok(a)
}

/// Read one prompt; a trailing `\` continues onto the next line. `None` on EOF.
fn read_prompt() -> Option<String> {
    let stdin = std::io::stdin();
    let mut msg = String::new();
    print!("{BOLD}prompt>{RESET} ");
    std::io::stdout().flush().ok();
    loop {
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).ok()? == 0 {
            return if msg.is_empty() { None } else { Some(msg) };
        }
        let line = line.trim_end_matches(['\n', '\r']);
        if let Some(l) = line.strip_suffix('\\') {
            msg.push_str(l);
            msg.push('\n');
            print!("{BOLD}..{RESET} ");
            std::io::stdout().flush().ok();
        } else {
            msg.push_str(line);
            return Some(msg);
        }
    }
}

/// Show tokens the way the model sees them: boundaries marked, whitespace visible.
fn show_tokens(tok: &Tokenizer, ids: &[u32]) -> String {
    let parts: Vec<String> = ids
        .iter()
        .map(|&id| tok.token_text(id).replace('\n', "\\n").replace('\t', "\\t"))
        .collect();
    format!("{} tokens: {}", ids.len(), parts.join("│"))
}

fn main() -> engine::Result<()> {
    let a = parse_args()?;
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../models");
    let spec = download::spec(&a.size).ok_or_else(|| format!("unknown --size {} (use 135m or 360m)", a.size))?;
    let dir = match &a.model {
        Some(d) => d.clone(),
        None => {
            let d = format!("{root}/{}", spec.dir);
            download::ensure_model(std::path::Path::new(&d), spec)?;
            d
        }
    };

    let t = std::time::Instant::now();
    let model = Model::load(&dir)?;
    let tok = Tokenizer::from_file(format!("{dir}/tokenizer.json"))?;
    // DRY against loops; penalties apply to words only, so a continuation can
    // repeat numbers (see `Sampler::words_only`).
    let specials: Vec<u32> = tok.specials.iter().map(|&(_, id)| id).collect();
    let mut sampler = Sampler::new(a.temp, a.top_k, a.top_p, a.seed)
        .with_dry(model.cfg.vocab, |id| specials.contains(&id) || Sampler::dry_breaker_bytes(&tok.decode_bytes(&[id])))
        .words_only(model.cfg.vocab, &specials, |id| tok.decode_bytes(&[id]));
    sampler.dry_multiplier = a.dry;
    sampler.repeat_penalty = a.repeat_penalty;
    let stop: Vec<u32> = [model.cfg.eos_token, 0].into_iter().collect();

    eprintln!(
        "{DIM}{} · {:.1}M params · loaded in {:.0?} · temp {} top-p {} top-k {} repeat-penalty {} dry {}\n{HELP}{RESET}",
        dir.rsplit('/').next().unwrap_or(&dir),
        model.n_params() as f64 / 1e6,
        t.elapsed(),
        a.temp,
        a.top_p,
        a.top_k,
        a.repeat_penalty,
        a.dry,
    );

    let mut cache = model.new_cache(a.ctx);
    let mut raw = false;
    while let Some(text) = read_prompt() {
        match text.trim() {
            "" => continue,
            "/quit" | "/exit" => break,
            "/help" => {
                println!("{DIM}{HELP}{RESET}");
                continue;
            }
            "/raw" => {
                raw = !raw;
                println!("{DIM}raw token view {}{RESET}", if raw { "on" } else { "off" });
                continue;
            }
            _ => {}
        }
        cache.clear(); // every prompt is a fresh completion
        let prompt = tok.encode(&text);
        if prompt.is_empty() {
            println!("{DIM}that text encodes to no tokens{RESET}");
            continue;
        }
        if prompt.len() + 1 >= cache.max_seq {
            println!("{DIM}the prompt does not fit in the context (raise --ctx){RESET}");
            continue;
        }
        if raw {
            println!("{DIM}fed {}{RESET}", show_tokens(&tok, &prompt));
        }
        print!("{DIM}{text}{RESET}");
        std::io::stdout().flush()?;
        let stats = generate(&model, &tok, &mut cache, &prompt, &mut sampler, a.max_new, &stop, &mut NoHooks, |s| {
            print!("{s}");
            std::io::stdout().flush().ok();
        });
        println!();
        if raw {
            println!("{DIM}generated {}{RESET}", show_tokens(&tok, &stats.generated));
        }
        let ended = match stats.stopped_by {
            Some(id) => format!("stopped at {}", tok.token_text(id)),
            None if cache.len >= cache.max_seq => "context window full (raise --ctx)".into(),
            None => format!("hit the -n limit of {} tokens", a.max_new),
        };
        println!("{DIM}[{} · {ended}]{RESET}", stats.summary());
    }
    Ok(())
}
