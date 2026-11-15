//! Byte-level BPE, the GPT-2 family tokenizer (SmolLM2 uses it with a 49,152 vocab).
//!
//! Encoding a string happens in stages:
//!
//! 1. **Special tokens** like `<|im_start|>` are split out first and map straight to an id.
//! 2. **Digit split**: every numeric character becomes its own piece, so numbers
//!    are always tokenized digit by digit ("2026" → 2,0,2,6).
//! 3. **Pre-tokenization** with the GPT-2 rule: split into words, runs of
//!    digits, runs of punctuation, and whitespace, where a word usually carries
//!    its leading space (" world"). BPE never merges across these boundaries.
//! 4. **Byte mapping**: each word's UTF-8 bytes are mapped to 256 printable
//!    stand-in characters (space → 'Ġ', newline → 'Ċ'), so every possible byte
//!    sequence is representable and nothing is ever "unknown".
//! 5. **BPE merges**: start from single bytes and repeatedly merge the adjacent
//!    pair that was learned *earliest* in training (lowest rank), until no
//!    learned pair remains.

use std::collections::HashMap;
use std::path::Path;

use crate::Result;

pub struct Tokenizer {
    /// id → token string (in byte-mapped form, e.g. "Ġworld").
    pub vocab: Vec<String>,
    pub token_to_id: HashMap<String, u32>,
    /// (left id, right id) → (rank, merged id). Lower rank = merge first.
    merges: HashMap<(u32, u32), (u32, u32)>,
    /// Special tokens, longest first.
    pub specials: Vec<(String, u32)>,
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
    /// Token id for each single byte. SmolLM2 omits 21 bytes: ones that can
    /// never appear in valid UTF-8 (0xC0, 0xC1, 0xF5..=0xFF) and a few rare
    /// control characters. Like HF (no `unk_token`), we silently drop those.
    byte_ids: [Option<u32>; 256],
}

/// GPT-2's reversible byte → printable-char table. Bytes that are already
/// printable (and not space) map to themselves; the rest are shifted to 256+.
fn bytes_to_unicode() -> [char; 256] {
    let mut table = ['\0'; 256];
    let printable = |b: u32| (0x21..=0x7E).contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b);
    let mut n = 0;
    for b in 0..256u32 {
        table[b as usize] = if printable(b) {
            char::from_u32(b).unwrap()
        } else {
            n += 1;
            char::from_u32(255 + n).unwrap()
        };
    }
    table
}

/// One step in the tokenization of a pre-token, for visualization.
#[derive(Debug, Clone)]
pub struct WordTrace {
    pub word: String,
    /// Token ids after each merge; `steps[0]` is the raw bytes.
    pub steps: Vec<Vec<u32>>,
}

impl Tokenizer {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_json_str(&std::fs::read_to_string(path)?)
    }

    pub fn from_json_str(json: &str) -> Result<Self> {
        let j: serde_json::Value = serde_json::from_str(json)?;
        let model = &j["model"];
        if model["type"] != "BPE" {
            return Err("only BPE tokenizers are supported".into());
        }
        let vmap = model["vocab"].as_object().ok_or("missing vocab")?;
        let mut vocab = vec![String::new(); vmap.len()];
        let mut token_to_id = HashMap::with_capacity(vmap.len());
        for (tok, id) in vmap {
            let id = id.as_u64().ok_or("bad vocab id")? as u32;
            vocab[id as usize] = tok.clone();
            token_to_id.insert(tok.clone(), id);
        }

        let mut merges = HashMap::new();
        for (rank, m) in model["merges"].as_array().ok_or("missing merges")?.iter().enumerate() {
            // Merges come either as "a b" strings or ["a", "b"] pairs.
            let (a, b) = match m {
                serde_json::Value::String(s) => s.split_once(' ').ok_or("bad merge")?,
                serde_json::Value::Array(p) => (p[0].as_str().ok_or("bad merge")?, p[1].as_str().ok_or("bad merge")?),
                _ => return Err("bad merge".into()),
            };
            let id = |s: &str| token_to_id.get(s).copied().ok_or_else(|| format!("merge token {s:?} not in vocab"));
            let merged = id(&format!("{a}{b}"))?;
            merges.insert((id(a)?, id(b)?), (rank as u32, merged));
        }

        let mut specials: Vec<(String, u32)> = j["added_tokens"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|t| Some((t["content"].as_str()?.to_string(), t["id"].as_u64()? as u32)))
                    .collect()
            })
            .unwrap_or_default();
        specials.sort_by_key(|(s, _)| std::cmp::Reverse(s.len()));

        let byte_to_char = bytes_to_unicode();
        let char_to_byte = byte_to_char.iter().enumerate().map(|(b, &c)| (c, b as u8)).collect();
        let mut byte_ids = [None; 256];
        for (b, c) in byte_to_char.iter().enumerate() {
            byte_ids[b] = token_to_id.get(&c.to_string()).copied();
        }
        Ok(Self { vocab, token_to_id, merges, specials, byte_to_char, char_to_byte, byte_ids })
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.encode_traced(text, None)
    }

    /// Like `encode`, but optionally records each pre-token's merge sequence.
    pub fn encode_traced(&self, text: &str, mut trace: Option<&mut Vec<WordTrace>>) -> Vec<u32> {
        let mut ids = Vec::new();
        for seg in self.split_specials(text) {
            match seg {
                Segment::Special(id) => ids.push(id),
                Segment::Text(s) => {
                    for piece in split_digits(s) {
                        for word in pretokenize(piece) {
                            let steps = self.bpe(word, trace.is_some());
                            ids.extend_from_slice(steps.last().unwrap());
                            if let Some(t) = trace.as_deref_mut() {
                                t.push(WordTrace { word: word.to_string(), steps });
                            }
                        }
                    }
                }
            }
        }
        ids
    }

    /// Returns the merge history if `keep_steps`, otherwise just the final ids.
    fn bpe(&self, word: &str, keep_steps: bool) -> Vec<Vec<u32>> {
        let mut ids: Vec<u32> = word.bytes().filter_map(|b| self.byte_ids[b as usize]).collect();
        let mut steps = vec![];
        loop {
            if keep_steps {
                steps.push(ids.clone());
            }
            // Find the adjacent pair with the lowest merge rank.
            let best = ids
                .windows(2)
                .filter_map(|p| self.merges.get(&(p[0], p[1])).map(|&(rank, new)| (rank, p[0], p[1], new)))
                .min_by_key(|m| m.0);
            let Some((_, a, b, new)) = best else { break };
            // Merge every occurrence of that pair, left to right.
            let mut out = Vec::with_capacity(ids.len());
            let mut i = 0;
            while i < ids.len() {
                if i + 1 < ids.len() && ids[i] == a && ids[i + 1] == b {
                    out.push(new);
                    i += 2;
                } else {
                    out.push(ids[i]);
                    i += 1;
                }
            }
            ids = out;
        }
        if !keep_steps {
            steps.push(ids);
        }
        steps
    }

    /// Raw bytes for a token sequence (may end mid-way through a UTF-8 character).
    pub fn decode_bytes(&self, ids: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        for &id in ids {
            let tok = &self.vocab[id as usize];
            if self.specials.iter().any(|(_, sid)| *sid == id) {
                out.extend_from_slice(tok.as_bytes());
            } else {
                out.extend(tok.chars().filter_map(|c| self.char_to_byte.get(&c)));
            }
        }
        out
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids)).into_owned()
    }

    /// Human-readable form of one token: its actual text, not the Ġ-mapped form.
    pub fn token_text(&self, id: u32) -> String {
        self.decode(&[id])
    }

    pub fn byte_char(&self, b: u8) -> char {
        self.byte_to_char[b as usize]
    }

    fn split_specials<'a>(&self, text: &'a str) -> Vec<Segment<'a>> {
        let mut out = vec![];
        let (mut start, mut i) = (0, 0);
        while i < text.len() {
            if let Some((s, id)) = self.specials.iter().find(|(s, _)| text[i..].starts_with(s.as_str())) {
                if start < i {
                    out.push(Segment::Text(&text[start..i]));
                }
                out.push(Segment::Special(*id));
                i += s.len();
                start = i;
            } else {
                i += text[i..].chars().next().unwrap().len_utf8();
            }
        }
        if start < text.len() {
            out.push(Segment::Text(&text[start..]));
        }
        out
    }
}

enum Segment<'a> {
    Text(&'a str),
    Special(u32),
}

/// Every numeric char becomes its own piece; everything between stays together.
fn split_digits(s: &str) -> Vec<&str> {
    let mut out = vec![];
    let mut start = 0;
    for (i, c) in s.char_indices() {
        if c.is_numeric() {
            if start < i {
                out.push(&s[start..i]);
            }
            out.push(&s[i..i + c.len_utf8()]);
            start = i + c.len_utf8();
        }
    }
    if start < s.len() {
        out.push(&s[start..]);
    }
    out
}

/// The GPT-2 pre-tokenization regex, written out by hand:
///
/// ```text
/// 's|'t|'re|'ve|'m|'ll|'d | ?\p{L}+ | ?\p{N}+ | ?[^\s\p{L}\p{N}]+ | \s+(?!\S) | \s+
/// ```
///
/// Alternatives are tried in order at each position, like a regex engine does.
/// The trickiest one is `\s+(?!\S)`: a run of whitespace *gives back its last
/// space* when a word follows, so that space can attach to the word
/// ("a   b" → "a", "  ", " b").
pub fn pretokenize(s: &str) -> Vec<&str> {
    let is_l = |c: char| c.is_alphabetic();
    let is_n = |c: char| c.is_numeric();
    let is_ws = |c: char| c.is_whitespace();
    let is_other = |c: char| !is_ws(c) && !is_l(c) && !is_n(c);

    let chars: Vec<(usize, char)> = s.char_indices().collect();
    let n = chars.len();
    let byte_at = |i: usize| if i < n { chars[i].0 } else { s.len() };
    let run = |mut i: usize, f: &dyn Fn(char) -> bool| {
        while i < n && f(chars[i].1) {
            i += 1;
        }
        i
    };

    let mut out = vec![];
    let mut i = 0;
    while i < n {
        let c = chars[i].1;
        let next = chars.get(i + 1).map(|x| x.1);
        let end = 'm: {
            // contractions
            if c == '\'' {
                let rest = &s[byte_at(i + 1)..];
                for suf in ["re", "ve", "ll", "s", "t", "m", "d"] {
                    if rest.starts_with(suf) {
                        break 'm i + 1 + suf.len();
                    }
                }
            }
            // optional leading space, then a run of one class
            for class in [&is_l as &dyn Fn(char) -> bool, &is_n, &is_other] {
                if c == ' ' && next.is_some_and(class) {
                    break 'm run(i + 1, class);
                }
                if class(c) {
                    break 'm run(i, class);
                }
            }
            // whitespace
            let j = run(i, &is_ws);
            if j == n || j - i == 1 { j } else { j - 1 }
        };
        out.push(&s[byte_at(i)..byte_at(end)]);
        i = end;
    }
    out
}
