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
        if self.repeat_penalty == 1.0 || history.is_empty() {
            return self.sample(logits);
        }
        let mut l = logits.to_vec();
        let mut seen = std::collections::HashSet::new();
        for &id in history.iter().rev().take(self.repeat_window) {
            if seen.insert(id) {
                let x = &mut l[id as usize];
                *x = if *x > 0.0 { *x / self.repeat_penalty } else { *x * self.repeat_penalty };
            }
        }
        self.sample(&l)
    }

    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        if self.temperature <= 0.0 {
            return argmax(logits);
        }
        let mut cand: Vec<(u32, f32)> =
            logits.iter().enumerate().map(|(i, &l)| (i as u32, l / self.temperature)).collect();
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
