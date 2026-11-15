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
    rng: u64,
}

impl Sampler {
    pub fn new(temperature: f32, top_k: usize, top_p: f32, seed: u64) -> Self {
        Self { temperature, top_k, top_p, repeat_penalty: 1.0, repeat_window: 64, rng: seed }
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

    /// Sample, applying the repetition penalty against `history` (most recent last).
    pub fn sample_with_history(&mut self, logits: &[f32], history: &[u32]) -> u32 {
        // A penalty must be a positive factor; 0 or NaN would turn logits into
        // NaN/inf. Anything else is treated as "off" rather than trusted.
        let p = self.repeat_penalty;
        if p == 1.0 || !(p > 0.0 && p.is_finite()) || history.is_empty() {
            return self.sample(logits);
        }
        let mut l = logits.to_vec();
        let mut seen = std::collections::HashSet::new();
        for &id in history.iter().rev().take(self.repeat_window) {
            if let (true, Some(x)) = (seen.insert(id), l.get_mut(id as usize)) {
                *x = if *x > 0.0 { *x / p } else { *x * p };
            }
        }
        self.sample(&l)
    }

    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        // `!(t > 0)` rather than `t <= 0` so that NaN is greedy too.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if !(self.temperature > 0.0) {
            return argmax(logits);
        }
        let mut cand: Vec<(u32, f32)> =
            logits.iter().enumerate().map(|(i, &l)| (i as u32, l / self.temperature)).collect();
        // A tiny temperature overflows l/T to ±inf, and softmax of inf is NaN.
        // In that limit sampling *is* argmax, so take it.
        if cand.iter().any(|c| !c.1.is_finite()) {
            return argmax(logits);
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

        let r = self.next_f32();
        let mut acc = 0.0;
        for (i, &pi) in p.iter().enumerate() {
            acc += pi;
            if r < acc {
                return cand[i].0;
            }
        }
        cand[p.len() - 1].0
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
        // Invalid penalties and out-of-vocab ids don't panic or produce NaN.
        for p in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            s.repeat_penalty = p;
            assert_eq!(s.sample_with_history(&l, &[0, 99_999]), 0, "penalty {p}");
        }
    }
}
