'use strict';
// Runs the engine (engine.wasm) inside a Web Worker, so the page stays
// responsive while the model computes. Model files come from Hugging Face's
// CDN (kept afterwards in the browser's Cache Storage, so later visits load
// offline), or, for local development, from the dev server's /models/.
//
// Messages in:  {id, method, args}
//   "__status" {model}: is the model already in the browser cache? (no network)
//   "__load" {model, source: 'hf'|'local'}: download (or read from cache) and load it
//   anything else goes to Session::call
// Messages out: {id, ok, result | error}  and  {progress: {...}} while loading.

const MODELS = {
  'smollm2-135m': { repo: 'HuggingFaceTB/SmolLM2-135M', sha256: '80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1', bytes: 269060552 },
};
const CACHE = 'smollm2-files-v1';
// Inspector positions. Its trace grows with N positions as ~0.33 MB·N (residual
// snapshots + MLP activations) plus 540 B·N² (every head's attention row), so
// about 170 + 140 MB at 512. Raise it with care: wasm32 tops out at 4 GB.
const MAX_SEQ = 512;

let ex = null; // wasm exports
let crashed = '';
let lastPanic = '';
const enc = new TextEncoder();
const dec = new TextDecoder();

const progress = (p) => postMessage({ progress: p });

// ---------------------------------------------------------------- wasm ABI

// Copy bytes into wasm memory; the export that receives (ptr, len) takes ownership.
function bytesIn(u8) {
  const p = ex.alloc(u8.length);
  new Uint8Array(ex.memory.buffer, p, u8.length).set(u8); // view made after alloc: memory may have grown
  return [p, u8.length];
}
const strIn = (s) => bytesIn(enc.encode(s));
const out = () => dec.decode(new Uint8Array(ex.memory.buffer, ex.out_ptr(), ex.out_len()));

function invoke(f) {
  if (crashed) throw new Error(crashed);
  let status;
  try {
    status = f();
  } catch (e) {
    // A Rust panic traps the instance; its state can't be trusted afterwards.
    crashed = (lastPanic || String(e)) + ' (reload the page to restart the engine)';
    throw new Error(crashed);
  }
  const o = out();
  if (status !== 0) throw new Error(o);
  return o;
}

// ---------------------------------------------------------------- downloads

async function sha256hex(u8) {
  const h = new Uint8Array(await crypto.subtle.digest('SHA-256', u8));
  return Array.from(h, (b) => b.toString(16).padStart(2, '0')).join('');
}

async function openCache() {
  try { return await caches.open(CACHE); } catch { return null; } // unavailable outside secure contexts
}

// Every file is kept in the browser's Cache API, whichever source it came from,
// so a reload reads it from disk instead of the network. A file with a pinned
// checksum is cached under that checksum: the same weights from localhost or
// from Hugging Face are one entry, and they're re-verified on every load.
const weightsKey = (sha) => new URL(`/__weights/${sha}`, self.location.href).href;

async function fetchBytes(url, label, sha) {
  const cache = await openCache();
  const key = sha ? weightsKey(sha) : url;
  let resp = cache ? await cache.match(key) : null;
  const cached = !!resp;
  if (!resp) {
    resp = await fetch(url);
    if (!resp.ok) throw new Error(`${label}: HTTP ${resp.status} from ${url}`);
  }
  const total = +resp.headers.get('content-length') || 0;
  const reader = resp.body.getReader();
  // Preallocate when the size is known (it is for the weights), else collect chunks.
  let buf = total ? new Uint8Array(total) : null;
  let chunks = buf ? null : [];
  let got = 0, lastReport = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (buf && got + value.length > buf.length) { chunks = [buf.subarray(0, got)]; buf = null; } // size header lied (compression)
    if (buf) buf.set(value, got); else chunks.push(value);
    got += value.length;
    if (got - lastReport > 2e6) { lastReport = got; progress({ stage: label, got, total, cached }); }
  }
  if (!buf || got !== buf.length) {
    const all = new Uint8Array(got);
    let o = 0;
    for (const c of chunks || [buf.subarray(0, got)]) { all.set(c, o); o += c.length; }
    buf = all;
  }
  progress({ stage: label, got, total: got, cached });

  if (sha) {
    if (crypto.subtle) {
      progress({ stage: 'verifying checksum', got: 0, total: 0 });
      const h = await sha256hex(buf);
      if (h !== sha) {
        if (cache) await cache.delete(key);
        throw new Error(`${label}: checksum mismatch (got ${h.slice(0, 12)}…, want ${sha.slice(0, 12)}…). The download was corrupted; reload to retry.`);
      }
    } else {
      console.warn('crypto.subtle unavailable (not a secure context): skipping checksum');
    }
  }
  if (!cached && cache) {
    try { await cache.put(key, new Response(buf, { headers: { 'content-length': String(got) } })); } catch (e) { console.warn('could not cache', url, e); }
  }
  return buf;
}

// ---------------------------------------------------------------- loading

async function load(name, source) {
  const m = MODELS[name];
  if (!m) throw new Error(`unknown model ${name}; choose one of ${Object.keys(MODELS).join(', ')}`);
  // 'local': the dev server (`cargo run -p viz`) serves the repo's models/ folder.
  const local = source === 'local';
  const base = local ? `/models/${name}/` : `https://huggingface.co/${m.repo}/resolve/main/`;
  const from = local ? ' (localhost)' : '';

  progress({ stage: 'engine', got: 0, total: 0 });
  const imports = {
    env: {
      now_ms: () => performance.now(),
      log_str: (p, n) => { lastPanic = dec.decode(new Uint8Array(ex.memory.buffer, p, n)); console.error(lastPanic); },
    },
  };
  const { instance } = await WebAssembly.instantiateStreaming(fetch('engine.wasm', { cache: 'no-cache' }), imports); // revalidate: a stale engine breaks the protocol
  ex = instance.exports;
  ex.install_panic_hook();

  const cfg = dec.decode(await fetchBytes(base + 'config.json', 'config' + from, null));
  const tok = dec.decode(await fetchBytes(base + 'tokenizer.json', 'tokenizer' + from, null));
  let w = await fetchBytes(base + 'model.safetensors', 'weights' + from, m.sha256);

  // safetensors: [u64 header length][JSON header][raw tensor bytes]
  const n = Number(new DataView(w.buffer, w.byteOffset, 8).getBigUint64(0, true));
  const data0 = 8 + n;
  const { tensors } = JSON.parse(invoke(() => ex.init(...strIn(cfg), ...strIn(tok), ...bytesIn(w.subarray(8, data0)), ...strIn(name))));
  for (let i = 0; i < tensors.length; i++) {
    const [tname, a, b] = tensors[i];
    invoke(() => ex.put_tensor(...strIn(tname), ...bytesIn(w.subarray(data0 + a, data0 + b))));
    if (i % 8 === 0) progress({ stage: 'converting bf16 → f32', got: i + 1, total: tensors.length });
  }
  w = null; // let the 269 MB download be garbage-collected
  progress({ stage: 'warming up', got: 0, total: 0 });
  invoke(() => ex.finish(MAX_SEQ));
  return { name, max_seq: MAX_SEQ };
}

// Whether loading would hit the network: the page asks before a download.
// Only the weights matter (config and tokenizer are a couple of MB).
async function status(name) {
  const m = MODELS[name];
  if (!m) throw new Error(`unknown model ${name}`);
  const cache = await openCache();
  const hit = cache ? await cache.match(weightsKey(m.sha256)) : null;
  return { cached: !!hit, cacheable: !!cache, bytes: m.bytes, repo: m.repo };
}

// ---------------------------------------------------------------- messages

onmessage = async (e) => {
  const { id, method, args } = e.data;
  try {
    const result = method === '__load' ? await load(args.model, args.source)
      : method === '__status' ? await status(args.model)
      : JSON.parse(invoke(() => ex.call(...strIn(method), ...strIn(JSON.stringify(args ?? null)))));
    postMessage({ id, ok: true, result });
  } catch (err) {
    postMessage({ id, ok: false, error: err.message || String(err) });
  }
};
