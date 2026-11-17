# llm-view

**Watch a real language model think, one tensor at a time.**

**▶ Try it live: [llm-view.magniff.dev](https://llm-view.magniff.dev)**. Everything runs in your browser. The page opens
right away and asks before it downloads anything: the first time you run a prompt, it offers to fetch the 269 MB model
from Hugging Face, and later visits load it from the browser cache without asking.

llm-view is a from-scratch inference engine for [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M), written in Rust,
plus a visualizer that runs entirely in your browser. Every intermediate tensor of the forward pass goes through a hook,
so you can follow your prompt through the network one operation at a time, with the real numbers, and watch the
next-token distribution being reshaped by the sampler.

The model is the base (not chat) one: it continues whatever text you give it.

It is built for learning: the engine is small enough to read in an afternoon, and it matches Hugging Face
`transformers` tensor by tensor.

## What's inside

- **A complete Llama-architecture engine, by hand.** Safetensors parsing, byte-level BPE tokenizer, RMSNorm, RoPE,
  grouped-query attention, SwiGLU MLP, KV cache, sampling. The only dependency is `serde_json`.
- **Verified, not approximately right.** All 513 intermediate tensors match `transformers` within 1e-4 relative error,
  greedy generations match token for token, and the tokenizer matches HF on 174 cases including a sweep across
  every Unicode script.
- **Hooks everywhere.** A `Hooks` callback sees (and may modify) every activation: q/k/v before and after RoPE,
  attention probabilities, each head's output, MLP gates, the residual stream after every sublayer.
- **Runs in the browser.** The engine compiles to WebAssembly (with SIMD). The page downloads the weights straight
  from the Hugging Face CDN once, caches them, and does all inference locally. There's no server to pay for.
- **Fast enough to play with.** Hand-written NEON and wasm-simd128 kernels plus multithreaded matmuls give about 70
  tokens/s in the browser on a laptop.

## The visualizer

Type the start of a text and run it. Two tabs:

| Tab | What you see |
|---|---|
| **Next token** | The model's distribution (blue) and what the sampler turns it into (orange). Change temperature, top-p, top-k and the penalties in the **Sampler** menu and watch the bars move. Click a token to force it. |
| **Step by step** | The forward pass one operation at a time, on the whole prompt as `[tokens, …]` matrices: tokenize, one-hot × embedding table, RMSNorm (measure, divide, × gain), q/k/v projections, split into heads, RoPE, scores, causal mask, softmax, weighted sum of values, output projection, residual add, then the MLP, and a peek (marked as not part of the model) at what the stream would predict if the model stopped after that layer (the logit lens). Next continues into the following layer; after layer 29 come the final norm, logits and softmax. Each step names the checkpoint tensors it uses (`model.layers.0.self_attn.q_proj.weight [576, 576]`), gives the formula with shapes, and shows the real numbers. Next / Previous, or ← / →. An **Explain** button opens a prose explanation of each step, with links to papers, Wikipedia and videos. |

Step one token at a time, generate 16 at once, or double-click a token to rewind the KV cache to that point. The
**About the model** dialog has a glossary of the terms the page uses.

## Quick start

You need [Rust](https://rustup.rs) (stable). Model weights are downloaded from Hugging Face on first run
(270 MB for 135M) and verified against a pinned SHA-256.

**Text completion in the terminal:**

```sh
cargo run                      # SmolLM2-135M: type the start of a text, it continues it
cargo run -- --size 360m       # the bigger one
cargo run -- --help            # sampling flags: --temp, --top-p, --top-k, --dry, --repeat-penalty, -n, ...
```

**The visualizer:**

```sh
scripts/build-web.sh                         # build the engine to viz/web/engine.wasm
cargo run --release -p viz -- --static       # serve it at http://127.0.0.1:7860
```

On localhost the page loads the weights from the repo's `models/` folder (fetched by the server on first run) instead
of Hugging Face; add `?source=hf` to the URL to load from the Hub anyway. Without `--static`, the server also loads the
model itself, and `?native` makes the page use it instead of WebAssembly, which is handy when debugging the engine.

**Deploying:** `viz/web/` is a fully static site. Build the wasm, then upload that folder to any static host
(GitHub Pages, Cloudflare Pages, S3). Visitors download the weights from Hugging Face directly.

## Sampling

A 135M model falls into loops easily ("a source of strength, a source of comfort, a source of joy…"). The page's
default preset, **Focused**, picks the model's favourite token except when DRY, the anti-loop sampler, sees a phrase
starting to repeat; then it pushes down the token that would continue the repeat, harder the longer the repeat is.

The penalties apply **only to tokens containing a letter**: numbers, operators and punctuation are exempt, because
loops are made of words, while "100 + 200 = 300" *has* to reuse the digits of the question. The Sampler menu has
other presets (greedy, the model card's, more random ones), and the Next token view marks the tokens DRY pushed
down (↺).

## How a token flows through the model

```text
"The capital of France is"
        │  tokenizer.rs   byte-level BPE (GPT-2 pre-tokenizer, digits split)
        ▼
 [504, 3575, 282, 4649, 314]
        │  model.rs       embed.out                                  [T, 576]
        ▼
 ┌─ × 30 layers ─────────────────────────────────────────────────────────────┐
 │  attn_norm.out ─ q, k, v ─ q_rope, k_rope ─▶ KV cache                      │
 │                 attn.probs = softmax(q·kᵀ/√64 + causal mask)   9 heads     │
 │                 attn.out ─ o_proj ─▶ resid.post_attn    (x += attn(norm x))│
 │  mlp_norm.out ─ gate, up ─ act = silu(gate) ─ hidden = act·up ─ down       │
 │                                   ─▶ resid.post_mlp     (x += mlp(norm x)) │
 └────────────────────────────────────────────────────────────────────────────┘
        │  final_norm.out ─ logits = x · embedᵀ (tied)                [T, 49152]
        ▼
 sampler.rs   penalties → temperature → top-k → top-p → draw          " Paris"
```

Every name in that diagram is a hook point, and the same names are used by the golden tests and by the reference
script that dumps them from `transformers`.

## Project layout

```text
engine/          the inference engine (library + `smol` CLI)
  src/
    safetensors.rs   weight file parsing, bf16 → f32
    tokenizer.rs     byte-level BPE, hand-written pre-tokenizer regex
    model.rs         config, weights, forward pass
    ops.rs           matmul (NEON / wasm-simd / portable, threaded), RMSNorm, RoPE, softmax
    kv_cache.rs      per-layer K/V storage
    hooks.rs         the observe / intervene interface
    sampler.rs       temperature, top-k, top-p, repetition penalty, DRY
    generate.rs      prefill + decode loop, streaming UTF-8
    trace.rs         records the residual stream of a session by position
    walk.rs          one position through one layer in full, for Step by step
    session.rs       the visualizer's API: method + JSON args → JSON
  tests/           golden tests against HF, session API tests
wasm/            the engine compiled for the browser (raw ABI, no wasm-bindgen)
viz/
  src/main.rs      dev server (axum): static files, /models, optional native API
  web/             the static site: app.js (page, Next token), walk.js and walk-info.js (Step by step and its explanations), glossary.js, worker.js (engine in a Web Worker)
reference/       Python scripts that produce the golden data from `transformers`
scripts/         build-web.sh, test-wasm.mjs
```

## Testing

```sh
cargo test                      # unit tests + golden tests + session API tests
scripts/build-web.sh && node scripts/test-wasm.mjs   # the same checks against the wasm build
```

The golden tests need reference data from `transformers`. Generate it once:

```sh
python -m venv .venv && .venv/bin/pip install torch transformers tokenizers safetensors
.venv/bin/python reference/dump_golden.py       # writes golden/
```

## Things you can learn from it

- Why decoding is memory-bound and prefill is compute-bound (watch the ms/token in the status bar).
- What the KV cache buys you, and what it costs in bytes per position (46 KB here).
- Why grouped-query attention exists: 9 query heads share 3 key/value heads, a 3× smaller cache.
- Attention sinks: many heads park their attention on the first token when they have nothing to do.
- What RoPE does to a query and a key, and why position 0 comes out unchanged.
- What temperature, top-p and penalties actually do to a distribution, and why a tiny model loops without them.

## Credits

SmolLM2 is by [Hugging Face](https://huggingface.co/HuggingFaceTB) (Apache 2.0). The DRY sampler follows the design
by p-e-w used in koboldcpp and llama.cpp. This project is MIT licensed: see [LICENSE](LICENSE).
