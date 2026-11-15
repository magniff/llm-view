#!/bin/sh
# Build the in-browser engine and put it next to the page.
#   scripts/build-web.sh            → viz/web/engine.wasm
# The result (viz/web/) is a fully static site: serve it with any static host.
set -e
cd "$(dirname "$0")/.."
rustup target add wasm32-unknown-unknown >/dev/null 2>&1 || true
cargo build --release -p smol-wasm --target wasm32-unknown-unknown
cp target/wasm32-unknown-unknown/release/smol_wasm.wasm viz/web/engine.wasm
if command -v wasm-opt >/dev/null; then
  wasm-opt -O3 --enable-simd --enable-bulk-memory --enable-nontrapping-float-to-int viz/web/engine.wasm -o viz/web/engine.wasm
fi
ls -l viz/web/engine.wasm
