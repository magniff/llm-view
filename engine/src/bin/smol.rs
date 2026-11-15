//! Interactive REPL.
//!
//!   cargo run                    # chat with SmolLM2-360M-Instruct
//!   cargo run -- --size 135m     # the tiny one (fast, but can't do arithmetic)
//!   cargo run -- --complete      # plain text completion with the base model
//!
//! Models are downloaded into `models/` on first run.
//!
//! Chat mode keeps one KV cache for the whole conversation, so each turn only
//! prefills the new tokens. End a line with `\` to keep typing on the next line.

use std::io::{BufRead, Write};

use engine::chat::Conversation;
use engine::download;
use engine::generate::generate;
use engine::hooks::NoHooks;
use engine::model::Model;
use engine::sampler::Sampler;
use engine::tokenizer::Tokenizer;

const DIM: &str = "\x1b[2m";
const BOLD: &str = "\x1b[1m";
const RESET: &str = "\x1b[0m";

const HELP: &str = "commands: /reset  /stats  /raw (toggle showing the exact tokens fed in)  /help  /quit   (Ctrl-D also quits)";

struct Args {
    complete: bool,
    model: Option<String>,
    size: String,
    system: Option<String>,
    temp: Option<f32>,
    top_k: usize,
    top_p: f32,
    repeat_penalty: f32,
    seed: u64,
    max_new: usize,
    ctx: usize,
}

fn parse_args() -> engine::Result<Args> {
    let seed = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos() as u64;
    let mut a = Args { complete: false, model: None, size: "360m".into(), system: None, temp: None, top_k: 0, top_p: 0.9, repeat_penalty: 1.1, seed, max_new: 1024, ctx: 2048 };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--complete" => a.complete = true,
            "--model" => a.model = Some(val()?),
            "--size" => a.size = val()?,
            "--system" => a.system = Some(val()?),
            "--temp" => a.temp = Some(val()?.parse()?),
            "--top-k" => a.top_k = val()?.parse()?,
            "--top-p" => a.top_p = val()?.parse()?,
            "--repeat-penalty" => a.repeat_penalty = val()?.parse()?,
            "--seed" => a.seed = val()?.parse()?,
            "-n" | "--max-new" => a.max_new = val()?.parse()?,
            "--ctx" => a.ctx = val()?.parse()?,
            "-h" | "--help" => {
                println!(
                    "cargo run -- [--complete] [--size 135m|360m] [--model DIR] [--system TEXT] [--temp T] [--top-k K] [--top-p P]\n     [--seed N] [--repeat-penalty R]\n     [-n MAX_NEW_TOKENS (default 1024)] [--ctx CONTEXT_LEN]\n\n{HELP}"
                );
                std::process::exit(0);
            }
            _ => return Err(format!("unknown argument {flag} (try --help)").into()),
        }
    }
    Ok(a)
}

/// Read one message; a trailing `\` continues onto the next line. `None` on EOF.
fn read_message(prompt: &str) -> Option<String> {
    let stdin = std::io::stdin();
    let mut msg = String::new();
    print!("{BOLD}{prompt}{RESET}");
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
    let spec = download::spec(&a.size, !a.complete).ok_or_else(|| format!("unknown --size {} (use 135m or 360m)", a.size))?;
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
    // Chat models: 0.2 / top-p 0.9 is what SmolLM2's authors suggest. Completion: a bit livelier.
    let temp = a.temp.unwrap_or(if a.complete { 0.8 } else { 0.2 });
    let mut sampler = Sampler::new(temp, a.top_k, a.top_p, a.seed);
    sampler.repeat_penalty = a.repeat_penalty;
    let stop: Vec<u32> = [model.cfg.eos_token, 0].into_iter().collect(); // EOS and <|endoftext|>

    eprintln!(
        "{DIM}{} · {:.1}M params · loaded in {:.0?} · {} mode · temp {temp} top-p {} top-k {} repeat-penalty {}\n{HELP}{RESET}",
        dir.rsplit('/').next().unwrap_or(&dir),
        model.n_params() as f64 / 1e6,
        t.elapsed(),
        if a.complete { "completion" } else { "chat" },
        a.top_p,
        a.top_k,
        a.repeat_penalty,
    );

    let mut conv = Conversation::new(&model, &tok, a.system.as_deref(), a.ctx);
    let mut raw = false;

    while let Some(msg) = read_message(if a.complete { "prompt> " } else { "you> " }) {
        match msg.trim() {
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
            "/reset" => {
                conv.reset(&tok);
                println!("{DIM}conversation cleared{RESET}");
                continue;
            }
            "/stats" => {
                println!(
                    "{DIM}context: {} / {} positions · KV cache {:.2} MB{RESET}",
                    conv.cache.len,
                    conv.cache.max_seq,
                    conv.cache.bytes_used() as f64 / 1e6
                );
                continue;
            }
            _ => {}
        }

        let prompt = if a.complete {
            conv.cache.clear(); // completions are independent
            tok.encode(&msg)
        } else {
            conv.prompt_for(&tok, &msg)
        };
        if prompt.len() + 1 >= conv.room_left() {
            println!("{DIM}context full — /reset to start over (or raise --ctx){RESET}");
            if !a.complete {
                conv.reset(&tok);
            }
            continue;
        }
        if raw {
            println!("{DIM}fed {}{RESET}", show_tokens(&tok, &prompt));
        }

        if a.complete {
            print!("{DIM}{msg}{RESET}");
        } else {
            print!("{BOLD}smol>{RESET} ");
        }
        std::io::stdout().flush()?;
        let stats = generate(&model, &tok, &mut conv.cache, &prompt, &mut sampler, a.max_new, &stop, &mut NoHooks, |s| {
            print!("{s}");
            std::io::stdout().flush().ok();
        });
        println!();
        if raw {
            println!("{DIM}generated {}{RESET}", show_tokens(&tok, &stats.generated));
        }
        let ended = match stats.stopped_by {
            Some(id) => format!("stopped at {}", tok.token_text(id)),
            None if conv.cache.len >= conv.cache.max_seq => "context window full (/reset, or raise --ctx)".into(),
            None => format!("hit the -n limit of {} tokens", a.max_new),
        };
        println!("{DIM}[{} · {ended} · context {}/{}]{RESET}", stats.summary(), conv.cache.len, conv.cache.max_seq);

        if !a.complete {
            conv.record(&tok, &prompt, &stats.generated);
        }
    }
    Ok(())
}
