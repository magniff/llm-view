//! Measure matmul throughput: `cargo run --release --example bench_linear`
use std::time::Instant;
use engine::ops::{linear, linear_serial};
use engine::tensor::Tensor;

fn main() {
    let rnd = |n: usize, seed: u32| -> Vec<f32> { (0..n).map(|i| ((i as u32).wrapping_mul(2654435761).wrapping_add(seed) >> 8) as f32 / 16777216.0 - 0.5).collect() };
    let w = Tensor::new(vec![49152, 576], rnd(49152 * 576, 1));
    for t in [1usize, 4, 32] {
        let x = Tensor::new(vec![t, 576], rnd(t * 576, 2));
        let macs = (t * 49152 * 576) as f64;
        let reps = (2e9 / macs).ceil() as usize;
        let s = Instant::now();
        for _ in 0..reps { std::hint::black_box(linear_serial(&x, &w)); }
        let st = macs * reps as f64 / s.elapsed().as_secs_f64() / 1e9;
        let s = Instant::now();
        for _ in 0..reps { std::hint::black_box(linear(&x, &w)); }
        let mt = macs * reps as f64 / s.elapsed().as_secs_f64() / 1e9;
        println!("T={t:>3}: 1 thread {st:6.1} GMAC/s | all threads {mt:6.1} GMAC/s");
    }
}
