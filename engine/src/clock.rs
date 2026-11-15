//! Wall-clock timing that also works in the browser.
//!
//! `std::time::Instant` panics on `wasm32-unknown-unknown` (there is no OS
//! clock), so in WebAssembly we import `performance.now()` from JavaScript
//! as `env.now_ms`.

use std::time::Duration;

#[cfg(not(target_arch = "wasm32"))]
pub fn now_ms() -> f64 {
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    START.get_or_init(std::time::Instant::now).elapsed().as_secs_f64() * 1000.0
}

#[cfg(target_arch = "wasm32")]
pub fn now_ms() -> f64 {
    #[link(wasm_import_module = "env")]
    unsafe extern "C" {
        #[link_name = "now_ms"]
        fn js_now_ms() -> f64;
    }
    // SAFETY: a plain function import with no arguments.
    unsafe { js_now_ms() }
}

#[derive(Clone, Copy)]
pub struct Stopwatch(f64);

impl Stopwatch {
    pub fn start() -> Self {
        Self(now_ms())
    }

    pub fn elapsed(&self) -> Duration {
        Duration::from_secs_f64(((now_ms() - self.0) / 1000.0).max(0.0))
    }

    pub fn ms(&self) -> f64 {
        now_ms() - self.0
    }
}
