//! The numeric kernels. A Llama needs surprisingly few:
//! matrix multiply, RMSNorm, SiLU, softmax, and RoPE.

use crate::tensor::Tensor;

// ---------------------------------------------------------------- dot products
//
// Everything heavy in a transformer is a dot product: every matmul output is
// one. So these two functions are where nearly all the time goes.
//
// A dot product written as a plain loop runs at ~3 GMAC/s here: LLVM leaves it
// scalar, one float at a time, because it may not reorder float additions.
// The fix is explicit SIMD. On ARM, NEON registers hold 4 floats, and
// `vfmaq_f32(acc, a, b)` computes `acc + a*b` for all 4 lanes in one
// instruction. We keep 4 such accumulators (16 lanes) running in parallel so
// that each fused multiply-add doesn't have to wait for the previous one to
// finish (they have ~3-4 cycles of latency).
//
// Splitting the sum into lanes changes the rounding compared to PyTorch's
// summation order, which is why the golden tests compare with a tolerance.

/// `a · b`
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "aarch64")]
    // SAFETY: lengths checked equal; NEON is always present on aarch64.
    unsafe {
        neon::dot(a, b)
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    // SAFETY: lengths checked equal; built with simd128 enabled.
    unsafe {
        wasm_simd::dot(a, b)
    }
    #[cfg(not(any(target_arch = "aarch64", all(target_arch = "wasm32", target_feature = "simd128"))))]
    portable::dot(a, b)
}

/// Four dot products sharing one `w`: `[x0·w, x1·w, x2·w, x3·w]`.
///
/// "Register blocking": each chunk of `w` is loaded into registers once and
/// used four times, so the loop does four multiply-adds per weight load instead
/// of one. This matters when the same weights are applied to many tokens
/// (prefill, logit lens); decode has only one token, so it uses plain [`dot`].
#[inline]
fn dot4(w: &[f32], x: [&[f32]; 4]) -> [f32; 4] {
    assert!(x.iter().all(|r| r.len() == w.len()));
    #[cfg(target_arch = "aarch64")]
    // SAFETY: lengths checked equal; NEON is always present on aarch64.
    unsafe {
        neon::dot4(w, x)
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    // SAFETY: lengths checked equal; built with simd128 enabled.
    unsafe {
        wasm_simd::dot4(w, x)
    }
    #[cfg(not(any(target_arch = "aarch64", all(target_arch = "wasm32", target_feature = "simd128"))))]
    x.map(|r| portable::dot(r, w))
}

#[cfg(not(any(target_arch = "aarch64", all(target_arch = "wasm32", target_feature = "simd128"))))]
mod portable {
    pub fn dot(a: &[f32], b: &[f32]) -> f32 {
        let mut acc = [0.0f32; 16];
        let (ca, ra) = a.as_chunks::<16>();
        let (cb, rb) = b.as_chunks::<16>();
        for (x, y) in ca.iter().zip(cb) {
            for i in 0..16 {
                acc[i] += x[i] * y[i];
            }
        }
        acc.iter().sum::<f32>() + ra.iter().zip(rb).map(|(x, y)| x * y).sum::<f32>()
    }
}

/// The same kernels for WebAssembly's 128-bit SIMD (4 × f32 per register).
/// Baseline wasm SIMD has no fused multiply-add, so it's a multiply then an add.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm_simd {
    use std::arch::wasm32::*;

    #[inline]
    unsafe fn load(p: *const f32) -> v128 {
        // SAFETY: caller keeps p..p+4 in bounds; wasm loads may be unaligned.
        unsafe { v128_load(p as *const v128) }
    }

    #[inline]
    fn hsum(v: v128) -> f32 {
        f32x4_extract_lane::<0>(v) + f32x4_extract_lane::<1>(v) + f32x4_extract_lane::<2>(v) + f32x4_extract_lane::<3>(v)
    }

    /// Caller guarantees `a.len() == b.len()`.
    #[inline]
    pub unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut acc = [f32x4_splat(0.0); 4];
        let mut i = 0;
        while i + 16 <= n {
            for k in 0..4 {
                // SAFETY: i + 16 <= n.
                let (x, y) = unsafe { (load(pa.add(i + 4 * k)), load(pb.add(i + 4 * k))) };
                acc[k] = f32x4_add(acc[k], f32x4_mul(x, y));
            }
            i += 16;
        }
        let mut s = hsum(f32x4_add(f32x4_add(acc[0], acc[1]), f32x4_add(acc[2], acc[3])));
        while i < n {
            s += a[i] * b[i];
            i += 1;
        }
        s
    }

    /// Caller guarantees every `x[r].len() == w.len()`. Bit-identical to [`dot`] per row.
    #[inline]
    pub unsafe fn dot4(w: &[f32], x: [&[f32]; 4]) -> [f32; 4] {
        let n = w.len();
        let pw = w.as_ptr();
        let px = x.map(|r| r.as_ptr());
        let mut acc = [[f32x4_splat(0.0); 4]; 4];
        let mut i = 0;
        while i + 16 <= n {
            // SAFETY: i + 16 <= n for w and every x[r].
            unsafe {
                let wv = [load(pw.add(i)), load(pw.add(i + 4)), load(pw.add(i + 8)), load(pw.add(i + 12))];
                for r in 0..4 {
                    for k in 0..4 {
                        acc[r][k] = f32x4_add(acc[r][k], f32x4_mul(load(px[r].add(i + 4 * k)), wv[k]));
                    }
                }
            }
            i += 16;
        }
        std::array::from_fn(|r| {
            let a = &acc[r];
            let mut s = hsum(f32x4_add(f32x4_add(a[0], a[1]), f32x4_add(a[2], a[3])));
            for j in i..n {
                s += x[r][j] * w[j];
            }
            s
        })
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    /// Caller guarantees `a.len() == b.len()`.
    #[inline]
    pub unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        unsafe {
            let mut acc = [vdupq_n_f32(0.0); 4];
            let mut i = 0;
            while i + 16 <= n {
                for k in 0..4 {
                    acc[k] = vfmaq_f32(acc[k], vld1q_f32(pa.add(i + 4 * k)), vld1q_f32(pb.add(i + 4 * k)));
                }
                i += 16;
            }
            let mut s = vaddvq_f32(vaddq_f32(vaddq_f32(acc[0], acc[1]), vaddq_f32(acc[2], acc[3])));
            while i < n {
                s += a[i] * b[i];
                i += 1;
            }
            s
        }
    }

    /// Caller guarantees every `x[r].len() == w.len()`.
    ///
    /// Per row, exactly the lane layout and reduction order of [`dot`], so
    /// `dot4(w, x)[r] == dot(x[r], w)` bit for bit, and a token gets identical
    /// logits whether it was part of a prefill or fed alone during decode.
    #[inline]
    pub unsafe fn dot4(w: &[f32], x: [&[f32]; 4]) -> [f32; 4] {
        let n = w.len();
        let pw = w.as_ptr();
        let px = x.map(|r| r.as_ptr());
        unsafe {
            // 4 rows × 4 registers = 16 accumulators; plus 4 for w. NEON has 32.
            let mut acc = [[vdupq_n_f32(0.0); 4]; 4];
            let mut i = 0;
            while i + 16 <= n {
                let wv = [vld1q_f32(pw.add(i)), vld1q_f32(pw.add(i + 4)), vld1q_f32(pw.add(i + 8)), vld1q_f32(pw.add(i + 12))];
                for r in 0..4 {
                    for k in 0..4 {
                        acc[r][k] = vfmaq_f32(acc[r][k], vld1q_f32(px[r].add(i + 4 * k)), wv[k]);
                    }
                }
                i += 16;
            }
            std::array::from_fn(|r| {
                let a = &acc[r];
                let mut s = vaddvq_f32(vaddq_f32(vaddq_f32(a[0], a[1]), vaddq_f32(a[2], a[3])));
                for j in i..n {
                    s += x[r][j] * w[j];
                }
                s
            })
        }
    }
}

/// `y = x · Wᵀ` — exactly PyTorch's bias-free `nn.Linear`.
///
/// `x` is `[T, in]`, `w` is `[out, in]`, result is `[T, out]`. Both operands
/// are read along contiguous rows. The weight row is the outer loop so it
/// stays in cache while we reuse it for all T tokens — during decode T = 1,
/// so every weight is touched exactly once per token, which is why decoding
/// is memory-bandwidth bound.
///
/// Big products are split across threads by output neuron: each thread owns a
/// contiguous band of weight rows. Small ones stay on one thread, because
/// starting threads costs ~10–20 µs, more than a 576×576 decode matmul takes.
/// That's why decode barely benefits from more cores while prefill (and the
/// 49k-row vocabulary projection) does. Every output is the same `dot` call
/// either way, so threaded and serial results are bit-identical.
pub fn linear(x: &Tensor, w: &Tensor) -> Tensor {
    let (t, d_in) = (x.rows(), x.cols());
    let (d_out, w_in) = (w.shape[0], w.shape[1]);
    assert_eq!(d_in, w_in, "linear: x has {d_in} features, W expects {w_in}");

    const PAR_MIN_MACS: usize = 1 << 22; // ~4M multiply-adds
    let threads = if t * d_out * d_in < PAR_MIN_MACS { 1 } else { n_threads().min(d_out) };
    if threads <= 1 {
        return Tensor::new(vec![t, d_out], linear_band(&x.data, t, d_in, &w.data, 0, d_out));
    }

    let band = d_out.div_ceil(threads);
    let bands: Vec<(usize, Vec<f32>)> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|i| (i * band, ((i + 1) * band).min(d_out)))
            .filter(|(o0, o1)| o0 < o1)
            .map(|(o0, o1)| s.spawn(move || (o0, linear_band(&x.data, t, d_in, &w.data, o0, o1))))
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut y = vec![0.0f32; t * d_out];
    for (o0, part) in bands {
        let n = part.len() / t;
        for r in 0..t {
            y[r * d_out + o0..][..n].copy_from_slice(&part[r * n..][..n]);
        }
    }
    Tensor::new(vec![t, d_out], y)
}

/// Single-threaded `linear`, for callers that parallelize at a higher level.
pub fn linear_serial(x: &Tensor, w: &Tensor) -> Tensor {
    let (t, d_in, d_out) = (x.rows(), x.cols(), w.shape[0]);
    assert_eq!(d_in, w.shape[1]);
    Tensor::new(vec![t, d_out], linear_band(&x.data, t, d_in, &w.data, 0, d_out))
}

/// Output neurons `o0..o1` for all `t` rows, as a `[t, o1 - o0]` block.
fn linear_band(x: &[f32], t: usize, d_in: usize, w: &[f32], o0: usize, o1: usize) -> Vec<f32> {
    let n = o1 - o0;
    let mut y = vec![0.0f32; t * n];
    let xr = |r: usize| &x[r * d_in..(r + 1) * d_in];
    for o in o0..o1 {
        let wr = &w[o * d_in..(o + 1) * d_in];
        let mut r = 0;
        while r + 4 <= t {
            let d = dot4(wr, [xr(r), xr(r + 1), xr(r + 2), xr(r + 3)]);
            for (k, v) in d.into_iter().enumerate() {
                y[(r + k) * n + (o - o0)] = v;
            }
            r += 4;
        }
        for r in r..t {
            y[r * n + (o - o0)] = dot(xr(r), wr);
        }
    }
    y
}

pub fn n_threads() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
}

/// RMSNorm: `y = x / sqrt(mean(x²) + eps) * weight`, per row.
/// Unlike LayerNorm there is no mean subtraction and no bias.
pub fn rmsnorm(x: &Tensor, weight: &[f32], eps: f32) -> Tensor {
    let d = x.cols();
    assert_eq!(d, weight.len());
    let mut y = x.clone();
    for r in 0..x.rows() {
        let row = y.row_mut(r);
        let ms = row.iter().map(|v| v * v).sum::<f32>() / d as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for (v, w) in row.iter_mut().zip(weight) {
            *v = *v * inv * w;
        }
    }
    y
}

/// SiLU (a.k.a. swish): `x · sigmoid(x)`.
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Numerically stable softmax in place: subtract the max so `exp` never overflows.
pub fn softmax(v: &mut [f32]) {
    let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for x in v.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    for x in v.iter_mut() {
        *x /= sum;
    }
}

/// RoPE inverse frequencies: `θ^(-2i/d)` for `i in 0..d/2`.
/// Pair 0 rotates fastest (once per token); the last pair barely rotates,
/// so different dimension pairs encode position at different "wavelengths".
pub fn rope_inv_freq(head_dim: usize, theta: f32) -> Vec<f32> {
    (0..head_dim / 2)
        .map(|i| 1.0 / theta.powf((2 * i) as f32 / head_dim as f32))
        .collect()
}

/// Rotate one head vector in place for absolute position `pos`.
///
/// This is the "rotate_half" (non-interleaved) convention used by HF Llama:
/// dimension `i` is paired with `i + d/2` (not with `i + 1`), and each pair
/// `(a, b)` is rotated by angle `pos · inv_freq[i]`:
///
/// ```text
/// a' = a·cos − b·sin
/// b' = b·cos + a·sin
/// ```
///
/// Because q·k after rotating both depends only on the angle *difference*,
/// attention scores end up depending on relative position.
pub fn rope(v: &mut [f32], pos: usize, inv_freq: &[f32]) {
    let half = inv_freq.len();
    debug_assert_eq!(v.len(), 2 * half);
    for i in 0..half {
        let angle = pos as f32 * inv_freq[i];
        let (sin, cos) = angle.sin_cos();
        let (a, b) = (v[i], v[i + half]);
        v[i] = a * cos - b * sin;
        v[i + half] = b * cos + a * sin;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random values in [-1, 1).
    fn rnd(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 40) as f32 / (1u64 << 23) as f32 - 1.0
            })
            .collect()
    }

    fn dot_f64(a: &[f32], b: &[f32]) -> f64 {
        a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum()
    }

    #[test]
    fn dot_matches_f64_for_every_tail_length() {
        // 0..70 covers empty input and every remainder modulo 4, 8 and 16.
        for n in 0..70 {
            let (a, b) = (rnd(n, 1), rnd(n, 2));
            let got = dot(&a, &b) as f64;
            let want = dot_f64(&a, &b);
            assert!((got - want).abs() <= 1e-5 * (1.0 + want.abs()), "n={n}: {got} vs {want}");
        }
    }

    #[test]
    fn dot4_is_bit_identical_to_dot() {
        for n in [0, 1, 7, 15, 16, 17, 33, 64, 576, 577] {
            let w = rnd(n, 3);
            let xs: Vec<Vec<f32>> = (0..4).map(|r| rnd(n, 10 + r)).collect();
            let got = dot4(&w, [&xs[0], &xs[1], &xs[2], &xs[3]]);
            for r in 0..4 {
                assert_eq!(got[r].to_bits(), dot(&xs[r], &w).to_bits(), "n={n} row={r}");
            }
        }
    }

    /// Prefill (many rows, dot4 + threads) and decode (one row, dot) must give
    /// bit-identical results for the same row, for any shape.
    #[test]
    fn linear_is_independent_of_batching_and_threads() {
        for (t, d_in, d_out) in [(1, 5, 3), (3, 17, 9), (7, 64, 130), (9, 576, 1000), (5, 33, 3000)] {
            let x = Tensor::new(vec![t, d_in], rnd(t * d_in, 5));
            let w = Tensor::new(vec![d_out, d_in], rnd(d_out * d_in, 6));
            let all = linear(&x, &w);
            let serial = linear_serial(&x, &w);
            assert_eq!(all, serial, "threaded vs serial, shape {t}x{d_in}x{d_out}");
            for r in 0..t {
                let one = linear(&Tensor::new(vec![1, d_in], x.row(r).to_vec()), &w);
                assert_eq!(one.row(0), all.row(r), "row {r} alone vs in a batch, shape {t}x{d_in}x{d_out}");
                for o in [0, d_out / 2, d_out - 1] {
                    let want = dot_f64(x.row(r), w.row(o));
                    assert!((all.row(r)[o] as f64 - want).abs() <= 1e-4 * (1.0 + want.abs()));
                }
            }
        }
    }

    #[test]
    fn rope_is_identity_at_zero_and_preserves_pair_norms() {
        let inv = rope_inv_freq(64, 100_000.0);
        let v = rnd(64, 7);
        let mut z = v.clone();
        rope(&mut z, 0, &inv);
        assert_eq!(z, v);
        for pos in [1, 17, 4095, 8191] {
            let mut r = v.clone();
            rope(&mut r, pos, &inv);
            for i in 0..32 {
                let before = v[i].hypot(v[i + 32]);
                let after = r[i].hypot(r[i + 32]);
                assert!((before - after).abs() < 1e-5, "pos {pos} pair {i}");
            }
        }
    }

    /// The point of RoPE: q·k depends only on the distance between positions.
    #[test]
    fn rope_scores_depend_only_on_relative_position() {
        let inv = rope_inv_freq(64, 100_000.0);
        let (q, k) = (rnd(64, 8), rnd(64, 9));
        let score = |pq: usize, pk: usize| {
            let (mut a, mut b) = (q.clone(), k.clone());
            rope(&mut a, pq, &inv);
            rope(&mut b, pk, &inv);
            dot(&a, &b)
        };
        let base = score(10, 3);
        for shift in [1, 100, 1000] {
            assert!((score(10 + shift, 3 + shift) - base).abs() < 2e-3, "shift {shift}");
        }
    }

    #[test]
    fn softmax_is_stable_and_normalized() {
        let mut v = vec![1000.0, 1001.0, 999.0, -1e30];
        softmax(&mut v);
        assert!(v.iter().all(|p| p.is_finite() && *p >= 0.0));
        assert!((v.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(v[1] > v[0] && v[0] > v[2]);
        let mut one = vec![-5.0];
        softmax(&mut one);
        assert_eq!(one, vec![1.0]);
    }

    #[test]
    fn rmsnorm_gives_unit_rms() {
        let x = Tensor::new(vec![2, 8], rnd(16, 11).iter().map(|v| v * 30.0).collect());
        let y = rmsnorm(&x, &[1.0; 8], 1e-5);
        for r in 0..2 {
            let rms = (y.row(r).iter().map(|v| v * v).sum::<f32>() / 8.0).sqrt();
            assert!((rms - 1.0).abs() < 1e-4);
        }
    }
}
