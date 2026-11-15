//! Fetch model files from the Hugging Face Hub on first use.
//!
//! Shells out to `curl` rather than pulling in an HTTP + TLS stack: the
//! files are plain HTTPS downloads from `huggingface.co/<repo>/resolve/main/<file>`.
//!
//! The weights file is checked against a pinned SHA-256. This is not paranoia:
//! during development a download arrived with the right size, an HTTP 200, and
//! corrupted bytes 21 MB in. The model loaded fine and produced fluent-looking
//! gibberish. Nothing short of a checksum catches that.

use std::path::Path;
use std::process::Command;

use crate::Result;

pub struct ModelSpec {
    pub repo: &'static str,
    /// Directory name under `models/`.
    pub dir: &'static str,
    /// SHA-256 of `model.safetensors` (Hugging Face shows it as the LFS "x-linked-etag").
    pub weights_sha256: &'static str,
}

pub const SMOLLM2_135M: ModelSpec = ModelSpec {
    repo: "HuggingFaceTB/SmolLM2-135M",
    dir: "smollm2-135m",
    weights_sha256: "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1",
};

pub const SMOLLM2_135M_INSTRUCT: ModelSpec = ModelSpec {
    repo: "HuggingFaceTB/SmolLM2-135M-Instruct",
    dir: "smollm2-135m-instruct",
    weights_sha256: "5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c",
};

pub const SMOLLM2_360M: ModelSpec = ModelSpec {
    repo: "HuggingFaceTB/SmolLM2-360M",
    dir: "smollm2-360m",
    weights_sha256: "7aaff6661428bed033abba9522bec81938678642cca3181fe752b6ca9e1e540f",
};

pub const SMOLLM2_360M_INSTRUCT: ModelSpec = ModelSpec {
    repo: "HuggingFaceTB/SmolLM2-360M-Instruct",
    dir: "smollm2-360m-instruct",
    weights_sha256: "e6bffe7435d7ddc10fd3b9a9efd429dafbacb1cb17015fb5562664e7532bf86e",
};

/// Look up a model by size and flavour.
pub fn spec(size: &str, instruct: bool) -> Option<&'static ModelSpec> {
    Some(match (size, instruct) {
        ("135m", false) => &SMOLLM2_135M,
        ("135m", true) => &SMOLLM2_135M_INSTRUCT,
        ("360m", false) => &SMOLLM2_360M,
        ("360m", true) => &SMOLLM2_360M_INSTRUCT,
        _ => return None,
    })
}

const FILES: [&str; 4] = ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"];
const ATTEMPTS: usize = 3;

/// Make sure `dir` contains the model files, downloading any that are missing.
pub fn ensure_model(dir: &Path, spec: &ModelSpec) -> Result<()> {
    let missing: Vec<&str> = FILES.iter().copied().filter(|f| !dir.join(f).exists()).collect();
    if missing.is_empty() {
        return Ok(());
    }
    std::fs::create_dir_all(dir)?;
    eprintln!("downloading {} into {}", spec.repo, dir.display());
    for f in missing {
        let url = format!("https://huggingface.co/{}/resolve/main/{f}", spec.repo);
        let part = dir.join(format!("{f}.part"));
        let expected = (f == "model.safetensors").then_some(spec.weights_sha256);
        let mut ok = false;
        for attempt in 1..=ATTEMPTS {
            eprintln!("  {f}{}", if attempt > 1 { format!(" (attempt {attempt}/{ATTEMPTS})") } else { String::new() });
            if let Err(e) = curl(&url, &part) {
                eprintln!("  {e}");
                continue;
            }
            match expected {
                None => ok = true,
                Some(want) => {
                    let got = sha256_file(&part)?;
                    if got == want {
                        eprintln!("  sha256 ok");
                        ok = true;
                    } else {
                        eprintln!("  sha256 mismatch: got {got}, want {want}; retrying");
                    }
                }
            }
            if ok {
                break;
            }
        }
        if !ok {
            let _ = std::fs::remove_file(&part);
            return Err(format!(
                "could not download a verified copy of {url} after {ATTEMPTS} attempts \
                 (if the repo was updated upstream, the pinned hash in download.rs needs updating)"
            )
            .into());
        }
        // Rename only once complete and verified, so a bad download is retried next run.
        std::fs::rename(&part, dir.join(f))?;
    }
    Ok(())
}

fn curl(url: &str, out: &Path) -> Result<()> {
    let status = Command::new("curl")
        .args(["-fL", "--retry", "3", "--progress-bar", "-o"])
        .arg(out)
        .arg(url)
        .status()
        .map_err(|e| format!("could not run curl ({e}); download {url} manually"))?;
    if status.success() { Ok(()) } else { Err(format!("curl failed for {url}").into()) }
}

/// SHA-256 via the system tool (`shasum` on macOS, `sha256sum` on Linux).
fn sha256_file(path: &Path) -> Result<String> {
    for (cmd, args) in [("shasum", &["-a", "256"][..]), ("sha256sum", &[][..])] {
        if let Ok(out) = Command::new(cmd).args(args).arg(path).output()
            && out.status.success()
        {
            let s = String::from_utf8_lossy(&out.stdout);
            return Ok(s.split_whitespace().next().unwrap_or("").to_string());
        }
    }
    Err("neither shasum nor sha256sum is available to verify the download".into())
}
