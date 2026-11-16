//! Turning logits into a token.
//!
//! The model outputs one score per vocabulary entry. Sampling is pure
//! post-processing on that vector:
//!
//! 1. **temperature** — divide logits by T. T→0 is argmax (greedy); T>1 flattens.
//! 2. **top-k** — keep only the k highest-scoring tokens.
//! 3. **top-p** (nucleus) — keep the smallest set whose probability mass ≥ p.
//! 4. renormalize and draw one at random.
//!
//! Optionally, before all that, a **repetition penalty** (as in HF / CTRL):
//! every token that appeared in the recent window has its logit pushed toward
//! "less likely" (positive logits ÷ penalty, negative × penalty). Small models
//! easily fall into loops, repeating a sentence forever, and this breaks them.
//! But it is blunt: it also punishes the words an answer *must* repeat
//! ("What is 7 + 5?" → "7 + 5 = 12"), and the punctuation every sentence needs.
//!
//! The sharper tool is **DRY** ("don't repeat yourself", from the local-LLM
//! community: koboldcpp, llama.cpp). It leaves single tokens alone and only
//! penalizes a token that would *extend a sequence that already occurred*.
//! If the last L tokens also appeared earlier, followed by token c, then c's
//! logit drops by `multiplier · base^(L − allowed_len)`: negligible for short
//! coincidences, crushing for a loop about to repeat a whole sentence.

use crate::ops::softmax;

pub struct Sampler {
    pub temperature: f32,
    /// 0 = disabled
    pub top_k: usize,
    /// 1.0 = disabled
    pub top_p: f32,
    /// 1.0 = disabled; 1.1 is a common mild setting.
    pub repeat_penalty: f32,
    /// How many recent tokens the repetition penalty looks at.
    pub repeat_window: usize,
    /// Tokens the penalty never touches. Meant for the chat control tokens:
    /// every prompt contains `<|im_end|>`, and penalizing it makes the model
    /// reluctant to end its turn, so short answers turn into rambles.
    pub no_penalty: Vec<u32>,
    /// DRY strength; 0 = disabled. 0.8 is the usual setting.
    pub dry_multiplier: f32,
    /// How fast the DRY penalty grows with the length of the repeat.
    pub dry_base: f32,
    /// Repeats up to this many tokens are free.
    pub dry_allowed_len: usize,
    /// Tokens that end a DRY match, sorted (newlines, quotes, colons and the
    /// chat control tokens): structure is supposed to repeat; prose isn't.
    pub dry_breakers: Vec<u32>,
    /// Tokens DRY never penalizes, indexed by id (empty = none). For digits,
    /// operators and punctuation: "What is 100 + 200?" → "100 + 200 = 300"
    /// legitimately repeats "100 + 20", and penalizing the next "0" turns the
    /// answer into "100 + 201 = 301". Loops are made of words.
    pub dry_exempt: Vec<bool>,
    rng: u64,
}

/// Matches longer than this all get the maximum penalty (and the search stops).
const DRY_MAX_MATCH: usize = 48;

impl Sampler {
    pub fn new(temperature: f32, top_k: usize, top_p: f32, seed: u64) -> Self {
        Self {
            temperature,
            top_k,
            top_p,
            repeat_penalty: 1.0,
            repeat_window: 64,
            no_penalty: vec![],
            dry_multiplier: 0.0,
            dry_base: 1.75,
            dry_allowed_len: 2,
            dry_breakers: vec![],
            dry_exempt: vec![],
            rng: seed,
        }
    }

    pub fn greedy() -> Self {
        Self::new(0.0, 0, 1.0, 0)
    }

    /// splitmix64 → uniform f32 in [0, 1).
    fn next_f32(&mut self) -> f32 {
        self.rng = self.rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Turn DRY on with the usual settings. `breaker(id)` says whether a token
    /// ends a match (call it with the tokenizer: see [`Self::dry_breaker_bytes`]).
    pub fn with_dry(mut self, vocab: usize, breaker: impl Fn(u32) -> bool) -> Self {
        self.dry_multiplier = 0.8;
        self.dry_breakers = (0..vocab as u32).filter(|&id| breaker(id)).collect();
        self
    }

    /// The default DRY sequence breakers, as bytes a token may contain.
    pub fn dry_breaker_bytes(bytes: &[u8]) -> bool {
        bytes.iter().any(|b| matches!(b, b'\n' | b':' | b'"' | b'*'))
    }

    /// Penalize words only: neither penalty touches a token with no letter in
    /// it (digits, operators, punctuation, whitespace), nor the control tokens.
    /// Each digit is its own token in SmolLM2's vocabulary, so "100 + 200 = 300"
    /// must repeat the question's digits, and DRY would read the echoed
    /// "100 + 20" as a loop and push the next "0" down ("100 + 201 = 301").
    /// Loops are made of words. `decode` gives a token's bytes.
    pub fn words_only(mut self, vocab: usize, specials: &[u32], decode: impl Fn(u32) -> Vec<u8>) -> Self {
        let letterless: Vec<bool> = (0..vocab as u32)
            .map(|id| !specials.contains(&id) && decode(id).iter().all(|b| b.is_ascii() && !b.is_ascii_alphabetic()))
            .collect();
        self.no_penalty = specials.iter().copied().chain((0..vocab as u32).filter(|&id| letterless[id as usize])).collect();
        self.dry_exempt = letterless;
        self
    }

    /// The penalties [`Self::sample_with_history`] would apply, as
    /// `(token, logit delta)`: what the visualizer shows as "after penalties".
    pub fn penalties(&self, logits: &[f32], history: &[u32]) -> Vec<(u32, f32)> {
        let pen = self.penalized(logits, history);
        match pen {
            None => vec![],
            Some(l) => l.iter().zip(logits).enumerate().filter(|(_, (a, b))| a != b).map(|(i, (a, b))| (i as u32, a - b)).collect(),
        }
    }

    /// Sample, applying the repetition penalty and DRY against `history` (most recent last).
    pub fn sample_with_history(&mut self, logits: &[f32], history: &[u32]) -> u32 {
        match self.penalized(logits, history) {
            Some(l) => self.sample(&l),
            None => self.sample(logits),
        }
    }

    /// `logits` after the history-based penalties: what [`Self::sample`] sees
    /// inside [`Self::sample_with_history`].
    pub fn adjust(&self, logits: &[f32], history: &[u32]) -> Vec<f32> {
        self.penalized(logits, history).unwrap_or_else(|| logits.to_vec())
    }

    /// Logits after the history-based penalties, or None if none apply.
    fn penalized(&self, logits: &[f32], history: &[u32]) -> Option<Vec<f32>> {
        // A penalty must be a positive factor; 0 or NaN would turn logits into
        // NaN/inf. Anything else is treated as "off" rather than trusted.
        let p = self.repeat_penalty;
        let rep_on = p != 1.0 && p > 0.0 && p.is_finite();
        let dry_on = self.dry_multiplier > 0.0 && self.dry_multiplier.is_finite() && self.dry_base >= 1.0;
        if history.is_empty() || !(rep_on || dry_on) {
            return None;
        }
        let mut l = logits.to_vec();
        if rep_on {
            self.apply_repeat_penalty(&mut l, history);
        }
        if dry_on {
            self.apply_dry(&mut l, history);
        }
        Some(l)
    }

    fn apply_repeat_penalty(&self, l: &mut [f32], history: &[u32]) {
        let p = self.repeat_penalty;
        let mut seen = std::collections::HashSet::new();
        for &id in history.iter().rev().take(self.repeat_window) {
            if self.no_penalty.contains(&id) {
                continue;
            }
            if let (true, Some(x)) = (seen.insert(id), l.get_mut(id as usize)) {
                *x = if *x > 0.0 { *x / p } else { *x * p };
            }
        }
    }

    /// For every earlier position e, measure how long the text ending at e
    /// matches the text ending now (the current suffix). If that match is long
    /// enough, the token that came after e is what a loop would produce next.
    /// Returns `(token, longest match)` for each such token, longest first.
    pub fn dry_matches(&self, h: &[u32]) -> Vec<(u32, usize)> {
        let n = h.len();
        let brk = |id: u32| self.dry_breakers.binary_search(&id).is_ok();
        if n < 2 || brk(h[n - 1]) {
            return vec![];
        }
        let mut longest: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
        for e in (0..n - 1).rev() {
            if h[e] != h[n - 1] {
                continue;
            }
            // Overlap is allowed: in "a b a b a b" the match runs through it,
            // which is exactly how a short loop gets caught.
            let mut len = 0;
            while len < DRY_MAX_MATCH && len <= e && h[e - len] == h[n - 1 - len] && !brk(h[e - len]) {
                len += 1;
            }
            if len >= self.dry_allowed_len && self.dry_exempt.get(h[e + 1] as usize) != Some(&true) {
                let best = longest.entry(h[e + 1]).or_insert(0);
                *best = (*best).max(len);
            }
        }
        let mut v: Vec<(u32, usize)> = longest.into_iter().collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        v
    }

    /// The logit penalty DRY gives a token that would extend a repeat of `len` tokens.
    pub fn dry_penalty(&self, len: usize) -> f32 {
        self.dry_multiplier * self.dry_base.powi(len.saturating_sub(self.dry_allowed_len) as i32)
    }

    fn apply_dry(&self, l: &mut [f32], h: &[u32]) {
        for (id, len) in self.dry_matches(h) {
            if let Some(x) = l.get_mut(id as usize) {
                *x -= self.dry_penalty(len);
            }
        }
    }

    /// The distribution [`Self::sample`] draws from, after temperature, top-k
    /// and top-p, most likely first. Greedy settings give the argmax with
    /// probability 1.
    pub fn distribution(&self, logits: &[f32]) -> Vec<(u32, f32)> {
        self.nucleus(logits).unwrap_or_else(|| vec![(argmax(logits), 1.0)])
    }

    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        let Some(dist) = self.nucleus(logits) else {
            return argmax(logits);
        };
        let r = self.next_f32();
        let mut acc = 0.0;
        for &(id, p) in &dist {
            acc += p;
            if r < acc {
                return id;
            }
        }
        dist[dist.len() - 1].0
    }

    /// `(token, probability)` after temperature, top-k and top-p, most likely
    /// first, or None when sampling is greedy.
    fn nucleus(&self, logits: &[f32]) -> Option<Vec<(u32, f32)>> {
        // `!(t > 0)` rather than `t <= 0` so that NaN is greedy too.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if !(self.temperature > 0.0) {
            return None;
        }
        let mut cand: Vec<(u32, f32)> =
            logits.iter().enumerate().map(|(i, &l)| (i as u32, l / self.temperature)).collect();
        // A tiny temperature overflows l/T to ±inf, and softmax of inf is NaN.
        // In that limit sampling *is* argmax, so take it.
        if cand.iter().any(|c| !c.1.is_finite()) {
            return None;
        }
        let by_desc = |a: &(u32, f32), b: &(u32, f32)| b.1.total_cmp(&a.1);
        if self.top_k > 0 && self.top_k < cand.len() {
            cand.select_nth_unstable_by(self.top_k - 1, by_desc);
            cand.truncate(self.top_k);
        }
        cand.sort_unstable_by(by_desc);

        let mut p: Vec<f32> = cand.iter().map(|c| c.1).collect();
        softmax(&mut p);
        if self.top_p < 1.0 {
            let mut mass = 0.0;
            let keep = p.iter().position(|&pi| {
                mass += pi;
                mass >= self.top_p
            });
            let n = keep.map_or(p.len(), |i| i + 1);
            p.truncate(n);
            let s: f32 = p.iter().sum();
            p.iter_mut().for_each(|x| *x /= s);
        }
        Some(cand.iter().zip(p).map(|(c, p)| (c.0, p)).collect())
    }
}

pub fn argmax(v: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logits() -> Vec<f32> {
        let mut l: Vec<f32> = (0..1000).map(|i| ((i * 7919) % 1000) as f32 / 500.0 - 1.0).collect();
        l[123] = 5.0;
        l[7] = 4.9;
        l
    }

    #[test]
    fn tiny_or_odd_temperature_is_greedy() {
        let l = logits();
        for t in [0.0, -1.0, f32::NAN, 1e-38, 1e-30, f32::MIN_POSITIVE] {
            for seed in 0..20 {
                assert_eq!(Sampler::new(t, 0, 1.0, seed).sample(&l), 123, "T={t} seed={seed}");
            }
        }
    }

    #[test]
    fn top_k_1_and_top_p_0_are_greedy() {
        let l = logits();
        for seed in 0..50 {
            assert_eq!(Sampler::new(1.5, 1, 1.0, seed).sample(&l), 123);
            assert_eq!(Sampler::new(1.5, 0, 0.0, seed).sample(&l), 123);
        }
    }

    #[test]
    fn samples_follow_the_softmax() {
        // Two tokens with logits ln 3 and 0: p = 0.75 / 0.25.
        let l = [3f32.ln(), 0.0];
        let mut s = Sampler::new(1.0, 0, 1.0, 42);
        let n = 20_000;
        let zeros = (0..n).filter(|_| s.sample(&l) == 0).count();
        let f = zeros as f64 / n as f64;
        assert!((f - 0.75).abs() < 0.015, "{f}");
    }

    #[test]
    fn top_p_keeps_the_smallest_nucleus() {
        // p ≈ [0.5, 0.3, 0.2]: top_p 0.7 keeps the first two only.
        let l = [0.5f32.ln(), 0.3f32.ln(), 0.2f32.ln()];
        let mut s = Sampler::new(1.0, 0, 0.7, 1);
        let mut seen = [0; 3];
        for _ in 0..5000 {
            seen[s.sample(&l) as usize] += 1;
        }
        assert_eq!(seen[2], 0);
        assert!(seen[0] > seen[1] && seen[1] > 0, "{seen:?}");
    }

    #[test]
    fn dry_penalizes_continuing_a_repeat_and_nothing_else() {
        let mut s = Sampler::greedy();
        s.dry_multiplier = 1.0;
        s.dry_base = 2.0;
        s.dry_allowed_len = 2;
        let l = vec![0.0f32; 10];
        // "1 2 3 4 | 1 2 3": continuing with 4 would repeat "1 2 3 4".
        // Match length 3 → penalty 1 · 2^(3−2) = 2 on token 4 only.
        let p = s.penalties(&l, &[1, 2, 3, 4, 9, 1, 2, 3]);
        assert_eq!(p, vec![(4, -2.0)]);
        // A 2-token match is allowed: 1 · 2^0 = 1 (allowed_len means "free below").
        assert_eq!(s.penalties(&l, &[2, 3, 4, 9, 2, 3]), vec![(4, -1.0)]);
        // A single repeated token is not a sequence: nothing.
        assert!(s.penalties(&l, &[5, 6, 7, 5]).is_empty());
        // A breaker inside the earlier occurrence cuts the match short.
        s.dry_breakers = vec![2];
        assert!(s.penalties(&l, &[1, 2, 3, 4, 9, 1, 2, 3]).is_empty());
        s.dry_breakers.clear();
        // A token looping on itself: "7 7 7 7" → 7 is penalized hard.
        let p = s.penalties(&l, &[7, 7, 7, 7, 7, 7]);
        assert_eq!(p.len(), 1);
        assert!(p[0].0 == 7 && p[0].1 <= -8.0, "{p:?}");
        // The longer the loop, the stronger the push away from it.
        let short = s.penalties(&l, &[1, 2, 3, 4, 9, 1, 2, 3])[0].1;
        let long = s.penalties(&l, &[1, 2, 3, 4, 5, 6, 9, 1, 2, 3, 4, 5])[0].1;
        assert!(long < short);
    }

    #[test]
    fn repetition_penalty_matches_hf_and_rejects_nonsense() {
        // Token 0 would win, but it was just seen; with penalty 2 its logit
        // halves (2.0 → 1.0) and token 1 (1.5) wins under greedy.
        let l = [2.0, 1.5, -1.0];
        let mut s = Sampler::greedy();
        s.repeat_penalty = 2.0;
        assert_eq!(s.sample_with_history(&l, &[0]), 1);
        // Negative logits get *multiplied*: -1 × 2 = -2 falls below token 1 (-1.5).
        assert_eq!(s.sample_with_history(&[-1.0, -1.5], &[0]), 1);
        // Out-of-window history doesn't count.
        s.repeat_window = 1;
        assert_eq!(s.sample_with_history(&l, &[0, 2]), 0);
        // Exempt tokens keep their logit.
        s.repeat_window = 64;
        s.no_penalty = vec![0];
        assert_eq!(s.sample_with_history(&l, &[0]), 0);
        s.no_penalty.clear();
        // Invalid penalties and out-of-vocab ids don't panic or produce NaN.
        for p in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            s.repeat_penalty = p;
            assert_eq!(s.sample_with_history(&l, &[0, 99_999]), 0, "penalty {p}");
        }
    }
}
