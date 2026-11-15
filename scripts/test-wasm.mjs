// Test the WebAssembly build (what visitors actually run) against the
// HuggingFace golden data, through the same ABI viz/web/worker.js uses.
//
//   scripts/build-web.sh && node scripts/test-wasm.mjs
//
// Needs models/smollm2-135m-instruct/ and golden/ (reference/dump_golden.py).

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import assert from 'node:assert/strict';

const root = fileURLToPath(new URL('..', import.meta.url));
const modelDir = root + 'models/smollm2-135m-instruct/';
const enc = new TextEncoder(), dec = new TextDecoder();

let ex, lastPanic = '';
const { instance } = await WebAssembly.instantiate(readFileSync(root + 'viz/web/engine.wasm'), {
  env: { now_ms: () => performance.now(), log_str: (p, n) => { lastPanic = dec.decode(new Uint8Array(ex.memory.buffer, p, n)); } },
});
ex = instance.exports;
ex.install_panic_hook();
const bytesIn = (u8) => { const p = ex.alloc(u8.length); new Uint8Array(ex.memory.buffer, p, u8.length).set(u8); return [p, u8.length]; };
const strIn = (s) => bytesIn(enc.encode(s));
const out = () => dec.decode(new Uint8Array(ex.memory.buffer, ex.out_ptr(), ex.out_len()));
function invoke(f) {
  let st;
  try { st = f(); } catch (e) { throw new Error(lastPanic || String(e)); }
  const o = out();
  if (st !== 0) throw new Error(o);
  return o;
}
const call = (m, a) => JSON.parse(invoke(() => ex.call(...strIn(m), ...strIn(JSON.stringify(a ?? null)))));

let failures = 0;
async function test(name, f) {
  const t = performance.now();
  try { await f(); console.log(`ok    ${name} (${(performance.now() - t).toFixed(0)} ms)`); }
  catch (e) { failures++; console.log(`FAIL  ${name}\n      ${e.message.split('\n').join('\n      ')}`); }
}

await test('load weights tensor by tensor', () => {
  const w = new Uint8Array(readFileSync(modelDir + 'model.safetensors'));
  const n = Number(new DataView(w.buffer, w.byteOffset, 8).getBigUint64(0, true));
  const { tensors } = JSON.parse(invoke(() => ex.init(
    ...strIn(readFileSync(modelDir + 'config.json', 'utf8')), ...strIn(readFileSync(modelDir + 'tokenizer.json', 'utf8')),
    ...bytesIn(w.subarray(8, 8 + n)), ...strIn('smollm2-135m-instruct'))));
  assert.equal(tensors.length, 272);
  for (const [name, a, b] of tensors) invoke(() => ex.put_tensor(...strIn(name), ...bytesIn(w.subarray(8 + n + a, 8 + n + b))));
  invoke(() => ex.finish(256, 1024));
  assert.ok(ex.memory.buffer.byteLength < 700e6, `wasm memory ${ex.memory.buffer.byteLength}`);
});

await test('tokenizer matches HF', () => {
  for (const c of JSON.parse(readFileSync(root + 'golden/tokenizer_cases.json', 'utf8'))) {
    if (!c.text) continue;
    assert.deepEqual(call('tokenize', { text: c.text }).ids, c.ids, JSON.stringify(c.text));
  }
});

await test('greedy chat replies match HF', () => {
  for (const c of JSON.parse(readFileSync(root + 'golden/chat.json', 'utf8'))) {
    call('chat_reset', {});
    call('chat_send', { text: c.user });
    const ids = [];
    for (;;) {
      const r = call('chat_step', { n: 8, temperature: 0, repeat_penalty: 1, dry: 0 });
      ids.push(...r.tokens.map((t) => t.id));
      if (r.done || ids.length >= c.reply_ids.length) break;
    }
    assert.deepEqual(ids.slice(0, c.reply_ids.length), c.reply_ids, c.user);
  }
});

await test('decode after prefill is bit-identical to a full prefill', () => {
  call('run', { text: 'The capital of France is' });
  const a = call('step', { n: 4, temperature: 0, repeat_penalty: 1, dry: 0 });
  const b = call('run', { ids: a.tokens.map((t) => t.id) });
  assert.deepEqual(b.next, a.next);
});

await test('errors come back as errors, not traps', () => {
  assert.throws(() => call('nope', {}), /unknown method/);
  assert.throws(() => call('attn', { layer: 999 }), /no such layer/);
  // usize is 32-bit here: 2^32 must be rejected, not wrapped to layer 0.
  assert.throws(() => call('attn', { layer: 2 ** 32 }), /non-negative integer/);
  assert.throws(() => call('run', { ids: [2 ** 32 + 5] }), /token ids/);
  assert.throws(() => call('chat_step', { repeat_penalty: 0 }), /repeat_penalty/);
  call('state'); // still alive
});

console.log(failures ? `\n${failures} failed` : '\nall passed');
process.exit(failures ? 1 : 0);
