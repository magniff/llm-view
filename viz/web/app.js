'use strict';
// Inside SmolLM2: a page over the engine's trace. All state lives on the server;
// this file fetches views of it and draws them.

const $ = (s) => document.querySelector(s);
const S = {
  st: null,        // /api/state
  sel: 0,          // selected position
  layer: 0,
  head: 0,
  tab: 'lens',
  mid: false,      // logit lens: include post-attention rows
  lensColor: 'answer',
  lensStride: 1,
  kvWhich: 'k',
  kvHead: 0,
  cache: {},       // per-state fetched views
  busy: false,
  renderGen: 0,
};

// ------------------------------------------------------------------ helpers

function el(tag, props = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (v == null || v === false) continue;
    if (k === 'class') e.className = v;
    else if (k === 'text') e.textContent = v;
    else if (k === 'style') e.style.cssText = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? '' : v);
  }
  for (const c of kids.flat(Infinity)) if (c != null && c !== false) e.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return e;
}

// Make whitespace visible: tokens often *are* whitespace.
const vis = (s) => (s === '' ? '∅' : s.replace(/ /g, '·').replace(/\n/g, '↵').replace(/\t/g, '⇥'));
const clip = (s, n) => (s.length > n ? s.slice(0, n - 1) + '…' : s);
const pct = (p) => (p >= 0.995 ? '100%' : p >= 0.1 ? (p * 100).toFixed(0) + '%' : p >= 0.001 ? (p * 100).toFixed(1) + '%' : p.toExponential(0));
const fmtBytes = (b) => (b >= 1e6 ? (b / 1e6).toFixed(2) + ' MB' : (b / 1e3).toFixed(1) + ' KB');

// ------------------------------------------------------------------ backend
//
// By default the engine runs in this browser tab: engine.wasm inside a Web
// Worker (worker.js). With ?native, calls go to the Rust server instead
// (`cargo run -p viz`). Either way it's the same Session::call(method, args).

const PARAMS = new URLSearchParams(location.search);
const NATIVE = PARAMS.has('native');
const MODEL = 'smollm2-135m-instruct';
// Where the weights come from: the local dev server when the page itself is on
// localhost (no 269 MB downloads while developing), Hugging Face otherwise.
const IS_LOCAL = ['localhost', '127.0.0.1', '[::1]'].includes(location.hostname);
const SOURCE = PARAMS.get('source') || (IS_LOCAL ? 'local' : 'hf');

let worker = null;
const pending = new Map();
let nextId = 1;

function startWorker(onProgress) {
  worker = new Worker('worker.js');
  worker.onmessage = (e) => {
    const m = e.data;
    if (m.progress) return onProgress(m.progress);
    const p = pending.get(m.id);
    pending.delete(m.id);
    m.ok ? p.resolve(m.result) : p.reject(new Error(m.error));
  };
  worker.onerror = (e) => flash('engine worker failed: ' + e.message);
}

async function call(method, args = null) {
  if (NATIVE) {
    const r = await fetch('/api/call/' + method, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(args) });
    const j = await r.json().catch(() => ({ error: r.statusText }));
    if (!r.ok || j.error) throw new Error(j.error || r.statusText);
    return j;
  }
  return new Promise((resolve, reject) => {
    const id = nextId++;
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, method, args });
  });
}

// Fetch a view once per state version.
async function cached(key, method, args) {
  if (!(key in S.cache)) S.cache[key] = await call(method, args);
  return S.cache[key];
}

function flash(msg) {
  const f = $('#flash');
  f.textContent = msg;
  f.style.display = 'block';
  clearTimeout(flash.t);
  flash.t = setTimeout(() => (f.style.display = 'none'), 5000);
}

async function act(fn) {
  if (S.busy) return;
  S.busy = true;
  document.body.classList.add('busy');
  try { await fn(); } catch (e) { flash(e.message); console.error(e); }
  finally { S.busy = false; document.body.classList.remove('busy'); }
}

const tip = $('#tip');
function showTip(e, ...content) {
  tip.replaceChildren(...content.flat(Infinity).filter((c) => c != null));
  tip.style.display = 'block';
  const w = tip.offsetWidth, h = tip.offsetHeight;
  tip.style.left = Math.max(4, Math.min(e.clientX + 14, innerWidth - w - 8)) + 'px';
  tip.style.top = (e.clientY + 16 + h > innerHeight ? e.clientY - h - 10 : e.clientY + 16) + 'px';
}
const hideTip = () => (tip.style.display = 'none');

// One hue, varying only in lightness: slate (unlikely) → ice blue (likely).
// "Brighter = more" reads without a legend, and nothing competes with the
// orange accents used for the final answer / lock-in markers.
const RAMP = [[22, 26, 34], [30, 40, 56], [46, 66, 96], [78, 108, 148], [130, 162, 200], [226, 236, 250]];
function heatRGB(v) {
  v = Math.max(0, Math.min(1, v || 0)) * (RAMP.length - 1);
  const i = Math.min(Math.floor(v), RAMP.length - 2), f = v - i;
  return RAMP[i].map((c, k) => Math.round(c + (RAMP[i + 1][k] - c) * f));
}
const heat = (v) => `rgb(${heatRGB(v).join(',')})`;
const heatText = (v) => (v > 0.6 ? '#0e1522' : '#c9d4e3');
// Diverging: blue (negative) → near-black (0) → orange (positive). v in [-1, 1].
function divRGB(v) {
  v = Math.max(-1, Math.min(1, v || 0));
  const z = [24, 27, 33], neg = [70, 150, 240], pos = [245, 150, 60];
  const t = Math.abs(v) ** 0.7, end = v < 0 ? neg : pos;
  return z.map((c, k) => Math.round(c + (end[k] - c) * t));
}

function legend(label, fn, lo, hi) {
  const c = el('canvas', { width: 160, height: 10 });
  const g = c.getContext('2d');
  for (let x = 0; x < 160; x++) { g.fillStyle = fn(x / 159); g.fillRect(x, 0, 1, 10); }
  return el('div', { class: 'legend' }, label ? el('span', { text: label }) : null, el('span', { text: lo }), c, el('span', { text: hi }));
}

function slider(label, value, max, oninput) {
  const out = el('b', { text: String(value) });
  const r = el('input', { type: 'range', min: 0, max, value });
  r.addEventListener('input', () => { out.textContent = r.value; oninput(+r.value); });
  return el('label', {}, label, r, out);
}

function hiDPI(canvas, w, h) {
  const d = devicePixelRatio || 1;
  canvas.width = w * d; canvas.height = h * d;
  canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
  const g = canvas.getContext('2d');
  g.scale(d, d);
  return g;
}

// ------------------------------------------------------------------ state

// A numeric input, clamped to [lo, hi]; empty or garbage falls back to `def`.
// (Not `+v || def`: that turns a deliberate 0, e.g. top-p 0 = greedy, into def.)
function numIn(sel, def, lo = -Infinity, hi = Infinity) {
  const v = parseFloat($(sel).value);
  return Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : def;
}

function samp() {
  return {
    temperature: numIn('#temp', 0, 0),
    top_p: numIn('#topp', 1, 0, 1),
    top_k: Math.floor(numIn('#topk', 0, 0, 1e6)),
    repeat_penalty: numIn('#rep', 1, 0.01, 100),
  };
}

// selMode: 'last' = select the newest token, 'keep' = keep selection if valid.
function setState(st, selMode = 'keep') {
  S.st = st;
  S.cache = {};
  if (selMode === 'last' || S.sel >= st.tokens.length) S.sel = st.tokens.length - 1;
  S.layer = Math.min(S.layer, st.model.n_layers - 1);
  renderHeader();
  renderTokens();
  renderStatus();
  renderTab();
}

function renderHeader() {
  const m = S.st.model;
  $('#model-info').textContent =
    `${m.name} · ${(m.params / 1e6).toFixed(1)}M params · ${m.n_layers} layers · d_model ${m.hidden} · ` +
    `${m.n_heads} query heads share ${m.n_kv_heads} KV heads × ${m.head_dim} dims · MLP ${m.intermediate} · vocab ${m.vocab}`;
}

// Optional per-token weights drawn behind the token strip (attention overlay).
function tokenOverlay() {
  if (S.tab !== 'attn') return null;
  const a = S.cache['attn' + S.layer];
  return a ? a.heads[S.head][S.sel] : null;
}

function renderTokens() {
  const box = $('#tokens');
  const w = tokenOverlay();
  const max = w ? Math.max(...w) : 1;
  box.replaceChildren(...S.st.tokens.map((t, i) => {
    const c = el('span', {
      class: 'tok' + (t.gen ? ' gen' : '') + (t.special ? ' special' : '') + (i === S.sel ? ' sel' : ''),
      text: vis(t.text),
    });
    if (w && i < w.length) c.style.setProperty('--w', (w[i] / max) ** 0.6);
    c.onclick = () => { S.sel = i; renderTokens(); renderTab(); };
    c.ondblclick = () => act(async () => setState(await call('rewind', { len: i + 1 }), 'last'));
    c.onmousemove = (e) => showTip(e,
      el('b', { text: `position ${i} · token id ${t.id}` }),
      el('div', { class: 'mono', text: JSON.stringify(t.text) }),
      el('div', { class: 'dim', text: (t.special ? 'special token · ' : '') + (t.gen ? 'generated' : 'prompt') }),
      w && i < w.length ? el('div', { text: `attention from selected token: ${pct(w[i])}` }) : null,
      el('div', { class: 'dim', text: 'click: select · double-click: rewind generation to here' }));
    c.onmouseleave = hideTip;
    return c;
  }));
}

function renderStatus() {
  const { last_op: op, cache: c } = S.st;
  let s = '';
  if (op.kind === 'prefill')
    s = `prefill: ${op.n} token${op.n === 1 ? '' : 's'} (positions 0–${op.n - 1}) in one forward pass · ${op.ms.toFixed(0)} ms` + (op.note ? ` · ${op.note}` : '');
  else if (op.kind === 'decode')
    s = `decode: ${op.n} step${op.n === 1 ? '' : 's'} from position ${op.pos0}, each feeding 1 token and reading all earlier K/V from the cache · ` +
      `${(op.ms / Math.max(op.n, 1)).toFixed(0)} ms/token · ${op.stop}`;
  else if (op.kind === 'rewind')
    s = `rewound: cache truncated to ${op.pos0} positions, then position ${op.pos0} re-fed to get its logits · ${op.ms.toFixed(0)} ms`;
  $('#status').replaceChildren(
    el('span', { text: s }),
    el('span', { class: 'cachebar', title: 'KV cache fill' }, el('i', { style: `width:${(100 * c.len) / c.max_seq}%` })),
    el('span', { text: `KV cache ${c.len}/${c.max_seq} positions · ${fmtBytes(c.bytes)}` }));
}

// ------------------------------------------------------------------ tabs

const TABS = [
  ['lens', 'Logit lens', viewLens],
  ['attn', 'Attention', viewAttn],
  ['resid', 'Residual stream', viewResid],
  ['kv', 'KV cache', viewKV],
  ['mlp', 'MLP neurons', viewMLP],
  ['next', 'Next token', viewNext],
  ['tok', 'Tokenizer', viewTok],
];

function renderTabs() {
  $('#tabs').replaceChildren(...TABS.map(([id, name]) =>
    el('button', { class: S.tab === id ? 'active' : '', text: name, onclick: () => { S.tab = id; history.replaceState(null, '', '#' + id); renderTabs(); renderTokens(); renderTab(); } })));
}

async function renderTab() {
  const gen = ++S.renderGen;
  const panel = $('#panel');
  const view = TABS.find((t) => t[0] === S.tab)[2];
  // Slow views (the logit lens on a long prompt) get a note instead of a frozen page.
  const slow = setTimeout(() => {
    if (gen === S.renderGen) panel.replaceChildren(el('p', { class: 'explain', text: S.tab === 'lens'
      ? `computing the logit lens: ${S.st.tokens.length} positions × ${S.st.model.n_layers + 1} layers, each projected onto all ${S.st.model.vocab.toLocaleString()} vocabulary tokens…`
      : 'computing…' }));
  }, 250);
  try {
    const nodes = await view();
    clearTimeout(slow);
    if (gen !== S.renderGen) return; // a newer render started meanwhile
    panel.replaceChildren(...[].concat(nodes).flat(Infinity).filter((c) => c != null && c !== false));
    if (S.tab === 'attn') renderTokens(); // overlay depends on fetched data
  } catch (e) {
    clearTimeout(slow);
    if (gen === S.renderGen) panel.replaceChildren(el('p', { class: 'explain', text: 'Error: ' + e.message }));
  }
}

const explain = (...parts) => el('p', { class: 'explain' }, ...parts);
const B = (t) => el('b', { text: t });

// ---------------------------------------------------------------- logit lens

const LENS_COLOR = {
  answer: ['final answer', 'probability each layer gives to the token the model finally outputs here'],
  next: ['actual next', 'probability of the token that actually came next (blank: not generated yet)'],
  conf: ['confidence', 'probability of the layer\'s own top guess'],
};

// For column p: the lowest stage from which every layer's top guess equals the final answer.
function lockIn(L, p) {
  const ans = L.answers[p][0];
  let i = L.stages.length - 1;
  while (i > 0 && L.cells[i - 1][p].top[0][0] === ans) i--;
  return i;
}

// Run-length encode the top guess up the layers: [{text, id, from, to}].
function story(L, p) {
  const out = [];
  L.cells.forEach((row, si) => {
    const [id, text] = row[p].top[0];
    const last = out[out.length - 1];
    if (last && last.id === id) last.to = si;
    else out.push({ id, text, from: si, to: si });
  });
  return out;
}

function lensValue(c) {
  if (S.lensColor === 'answer') return c.f ? c.f[0] : null;
  if (S.lensColor === 'next') return c.t ? c.t[0] : null;
  return c.top[0][2];
}

function lensTip(e, L, si, p) {
  const c = L.cells[si][p], toks = S.st.tokens, n = toks.length;
  const ans = L.answers[p];
  showTip(e,
    el('b', { text: `${L.stages[si]} · position ${p}, after ${JSON.stringify(toks[p].text)}` }),
    el('div', { class: 'bars', style: 'margin:6px 0' },
      c.top.map(([tid, tt, pr]) => [
        el('span', { class: 'mono', style: tid === ans[0] ? 'color:var(--accent)' : '', text: clip(vis(tt), 16) }),
        el('div', { class: 'bar', style: `width:${Math.max(1, pr * 160)}px` }),
        el('span', { text: pct(pr) })])),
    el('div', { text: `final answer ${JSON.stringify(ans[1])}: ${pct(c.f[0])}, rank ${c.f[1] + 1}` }),
    p + 1 < n
      ? el('div', { text: `actual next ${JSON.stringify(toks[p + 1].text)}: ${pct(c.t[0])}, rank ${c.t[1] + 1}` })
      : el('div', { class: 'dim', text: 'next token not generated yet' }),
    el('div', { class: 'dim', text: `entropy ${c.h.toFixed(2)} bits (0 = certain, 15.6 = uniform)` }));
}

function lensGrid(L) {
  const toks = S.st.tokens, n = toks.length, ns = L.stages.length;
  const locks = toks.map((_, p) => lockIn(L, p));
  const shown = [];
  for (let si = ns - 1; si >= 0; si--) if (si === 0 || si === ns - 1 || si % S.lensStride === 0) shown.push(si);

  const head = [
    el('tr', {}, el('th', { text: 'input →' }),
      toks.map((t, p) => el('th', { class: p === S.sel ? 'sel' : '', title: `position ${p}`, text: clip(vis(t.text), 9) }))),
    el('tr', { class: 'nextrow' }, el('th', { text: 'actual next →' }),
      toks.map((t, p) => el('th', { class: p === S.sel ? 'sel' : '', text: p + 1 < n ? clip(vis(toks[p + 1].text), 9) : '?' }))),
  ];

  const rows = shown.map((si) => {
    const final = si === ns - 1;
    return el('tr', { class: final ? 'final' : '' },
      el('th', { text: final ? L.stages[si] + ' (output)' : L.stages[si] }),
      L.cells[si].map((c, p) => {
        const [id, text, prob] = c.top[0];
        const agree = id === L.answers[p][0];
        const v = lensValue(c);
        const td = el('td', {
          class: [agree ? 'agree' : '', si === locks[p] && si > 0 ? 'lock' : '', p === S.sel ? 'selcol' : ''].join(' '),
          style: v == null
            ? 'background:#20242c;color:#8a93a3'
            : `background:${heat(v)};color:${heatText(v)}`,
          text: clip(vis(text), 8),
        });
        // Unsure guesses fade out, so the eye goes to what the layer actually believes.
        if (!agree) td.style.opacity = (0.35 + 0.65 * Math.sqrt(prob)).toFixed(2);
        td.onmousemove = (e) => lensTip(e, L, si, p);
        td.onmouseleave = hideTip;
        td.onclick = () => { S.sel = p; renderTokens(); renderTab(); };
        return td;
      }));
  });

  const wrap = el('div', { class: 'scroll', style: 'max-height:78vh' },
    el('table', { class: 'lens' }, el('thead', {}, head), el('tbody', {}, rows)));
  requestAnimationFrame(() => {
    const th = wrap.querySelector('thead th.sel');
    if (th) wrap.scrollLeft = Math.max(0, th.offsetLeft - wrap.clientWidth / 2);
  });
  return wrap;
}

// Probability of the final answer / actual next / top guess across all stages, log scale.
function lensChart(L, p) {
  const ns = L.stages.length, toks = S.st.tokens;
  const W = 940, H = 210, ml = 48, mr = 12, mt = 10, mb = 26;
  const cv = el('canvas');
  const g = hiDPI(cv, W, H);
  const x = (si) => ml + (si / Math.max(1, ns - 1)) * (W - ml - mr);
  const LO = -5; // 1e-5
  const y = (pr) => mt + (1 - (Math.max(LO, Math.log10(Math.max(pr, 1e-12))) - LO) / -LO) * (H - mt - mb);
  g.font = '10px ui-monospace, Menlo, monospace';
  g.textBaseline = 'middle';
  g.textAlign = 'right';
  for (let e = 0; e >= LO; e--) {
    g.strokeStyle = '#2a303b';
    g.beginPath(); g.moveTo(ml, y(10 ** e)); g.lineTo(W - mr, y(10 ** e)); g.stroke();
    g.fillStyle = '#8a93a3';
    g.fillText(['100%', '10%', '1%', '0.1%', '0.01%', '≤0.001%'][-e], ml - 6, y(10 ** e));
  }
  g.textAlign = 'center';
  L.stages.forEach((lab, si) => {
    if (si === 0 || si === ns - 1 || (ns > 40 ? si % 8 === 0 : si % 4 === 0)) g.fillText(lab, x(si), H - 10);
  });
  const lock = lockIn(L, p);
  if (lock > 0) {
    g.fillStyle = 'rgba(240,160,80,.10)';
    g.fillRect(x(lock), mt, W - mr - x(lock), H - mt - mb);
  }
  const series = [
    ['#8a93a3', 1, [3, 3], (c) => c.top[0][2]],
    ['#f0a050', 2.5, [], (c) => c.f[0]],
  ];
  if (p + 1 < toks.length && L.answers[p][0] !== toks[p + 1].id) series.push(['#5fb3f0', 2, [], (c) => c.t[0]]);
  for (const [color, w, dash, f] of series) {
    g.strokeStyle = color; g.lineWidth = w; g.setLineDash(dash);
    g.beginPath();
    L.cells.forEach((row, si) => (si ? g.lineTo(x(si), y(f(row[p]))) : g.moveTo(x(si), y(f(row[p])))));
    g.stroke();
  }
  g.setLineDash([]);
  cv.onmousemove = (e) => {
    const si = Math.round(((e.clientX - cv.getBoundingClientRect().left - ml) / (W - ml - mr)) * (ns - 1));
    if (si < 0 || si >= ns) return hideTip();
    lensTip(e, L, si, p);
  };
  cv.onmouseleave = hideTip;
  return cv;
}

function lensDetail(L) {
  const p = S.sel, toks = S.st.tokens, n = toks.length;
  const [ansId, ansText] = L.answers[p];
  const finalProb = L.cells[L.stages.length - 1][p].top[0][2];
  const lock = lockIn(L, p);
  const nextKnown = p + 1 < n;
  const correct = nextKnown && toks[p + 1].id === ansId;

  const segs = story(L, p);
  const storyEls = [];
  segs.forEach((sg, i) => {
    if (i) storyEls.push(el('span', { class: 'dim', text: ' → ' }));
    const range = sg.from === sg.to ? L.stages[sg.from] : `${L.stages[sg.from]}–${L.stages[sg.to]}`;
    storyEls.push(el('span', { class: 'seg' + (sg.id === ansId ? ' ans' : '') },
      el('span', { class: 'dim', text: range + ' ' }), el('b', { class: 'mono', text: vis(sg.text) })));
  });

  return el('div', { class: 'lensdetail' },
    el('h3', { text: `position ${p}: what comes after ${JSON.stringify(toks[p].text)}?` }),
    el('p', {},
      'The model\'s answer is ', el('b', { class: 'mono', style: 'color:var(--accent)', text: JSON.stringify(ansText) }), ` (${pct(finalProb)}). `,
      nextKnown
        ? [correct ? el('span', { style: 'color:var(--good)', text: 'That is what came next. ' })
            : ['The actual next token was ', el('b', { class: 'mono', style: 'color:var(--accent2)', text: JSON.stringify(toks[p + 1].text) }), '. ']]
        : 'Nothing has been generated after it yet. ',
      lock > 0
        ? ['It ', el('b', { text: `locks in at ${L.stages[lock]}` }), ': from there up, every layer\'s top guess is the final answer.']
        : 'Even the raw embedding already points at it.'),
    el('div', { class: 'story' }, el('span', { class: 'dim', text: 'top guess, bottom to top: ' }), storyEls),
    el('div', { class: 'legend', style: 'margin:10px 0 4px' },
      el('span', { style: 'color:#f0a050', text: '━ final answer' }),
      nextKnown && !correct ? el('span', { style: 'color:#5fb3f0', text: '━ actual next' }) : null,
      el('span', { text: '┅ that layer\'s top guess' }),
      el('span', { style: 'color:#f0a050', text: '▒ locked in' })),
    el('div', { class: 'scroll' }, lensChart(L, p)));
}

async function viewLens() {
  const L = await cached('lens' + S.mid, 'lens', { mid: S.mid });
  const pick = (label, cur, opts, set) => el('span', { class: 'pick' }, label + ' ', ...opts.map(([v, t, title]) =>
    el('button', { class: cur === v ? 'on' : '', text: t, title, onclick: () => { set(v); renderTab(); } })));
  const midBox = el('input', { type: 'checkbox' });
  midBox.checked = S.mid;
  midBox.onchange = () => { S.mid = midBox.checked; renderTab(); };

  return [
    explain('The ', B('residual stream'), ' is a running sum that every layer adds to. The logit lens reads it ', B('after every layer'),
      ' as if that layer were the last (final RMSNorm, then the unembedding) and shows the top guess for the ', B('next'),
      ' token. Each column is one position; read it bottom to top to watch the guess form. ',
      B('Bold'), ' = already agrees with the final answer; faded = an unsure guess; an ', el('span', { style: 'color:var(--accent)', text: 'orange edge' }),
      ' marks the layer where the answer locks in for good. Click a column for its story.'),
    el('div', { class: 'controls' },
      pick('colour by', S.lensColor, Object.entries(LENS_COLOR).map(([k, [t, title]]) => [k, t, title]), (v) => (S.lensColor = v)),
      pick('rows', S.lensStride, [[1, 'all'], [2, 'every 2nd'], [4, 'every 4th']], (v) => (S.lensStride = v)),
      el('label', { title: 'insert a row after each attention block, before the MLP' }, midBox, 'half-layers'),
      legend(LENS_COLOR[S.lensColor][0] + ' probability', heat, '0', '1')),
    lensGrid(L),
    lensDetail(L),
  ];
}

// ---------------------------------------------------------------- attention

function drawAttnThumb(canvas, rows) {
  const n = rows.length;
  canvas.width = n; canvas.height = n;
  const g = canvas.getContext('2d');
  const img = g.createImageData(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const o = (i * n + j) * 4;
      const [r, gg, b] = j <= i ? heatRGB(Math.sqrt(rows[i][j])) : [10, 12, 15];
      img.data[o] = r; img.data[o + 1] = gg; img.data[o + 2] = b; img.data[o + 3] = 255;
    }
  }
  g.putImageData(img, 0, 0);
}

function drawAttnBig(rows, toks) {
  const n = rows.length;
  const cell = Math.max(3, Math.min(22, Math.floor(620 / n)));
  const labels = cell >= 9;
  const m = labels ? 92 : 4;
  const W = m + n * cell + 4, H = m + n * cell + 4;
  const canvas = el('canvas');
  const g = hiDPI(canvas, W, H);
  g.fillStyle = '#0a0c0f';
  g.fillRect(m, m, n * cell, n * cell);
  for (let i = 0; i < n; i++)
    for (let j = 0; j <= i; j++) {
      g.fillStyle = heat(Math.sqrt(rows[i][j]));
      g.fillRect(m + j * cell, m + i * cell, cell - (cell > 5 ? 1 : 0), cell - (cell > 5 ? 1 : 0));
    }
  // selected query row
  g.strokeStyle = '#fff'; g.lineWidth = 1.5;
  g.strokeRect(m - 1, m + S.sel * cell - 1, n * cell + 2, cell + 2);
  if (labels) {
    g.font = `${Math.min(12, cell)}px ui-monospace, Menlo, monospace`;
    g.fillStyle = '#8a93a3'; g.textBaseline = 'middle';
    for (let i = 0; i < n; i++) {
      const t = clip(vis(toks[i].text), 11);
      g.textAlign = 'right';
      g.fillText(t, m - 4, m + i * cell + cell / 2);
      g.save();
      g.translate(m + i * cell + cell / 2, m - 4);
      g.rotate(-Math.PI / 2);
      g.textAlign = 'left';
      g.fillText(t, 0, 0);
      g.restore();
    }
  }
  canvas.onmousemove = (e) => {
    const r = canvas.getBoundingClientRect();
    const j = Math.floor((e.clientX - r.left - m) / cell), i = Math.floor((e.clientY - r.top - m) / cell);
    if (i < 0 || j < 0 || i >= n || j >= n) return hideTip();
    if (j > i) return showTip(e, el('div', { class: 'dim', text: `position ${i} cannot see position ${j}: it is in the future (causal mask)` }));
    showTip(e, el('b', { text: `query ${i} ${JSON.stringify(toks[i].text)} → key ${j} ${JSON.stringify(toks[j].text)}` }),
      el('div', { text: `attention weight ${pct(rows[i][j])}` }));
  };
  canvas.onmouseleave = hideTip;
  canvas.onclick = (e) => {
    const r = canvas.getBoundingClientRect();
    const i = Math.floor((e.clientY - r.top - m) / cell);
    if (i >= 0 && i < n) { S.sel = i; renderTokens(); renderTab(); }
  };
  return canvas;
}

async function viewAttn() {
  const m = S.st.model, toks = S.st.tokens;
  const A = await cached('attn' + S.layer, 'attn', { layer: S.layer });
  const group = m.n_heads / m.n_kv_heads;
  const ablated = new Set(S.st.ablate.map(([l, h]) => l + ':' + h));

  const setAblation = (h, on) => act(async () => {
    const set = new Set(ablated);
    on ? set.add(S.layer + ':' + h) : set.delete(S.layer + ':' + h);
    setState(await call('ablate', { heads: [...set].map((k) => k.split(':').map(Number)) }));
  });

  const groups = [];
  for (let g = 0; g < m.n_kv_heads; g++) {
    const heads = [];
    for (let h = g * group; h < (g + 1) * group; h++) {
      const c = el('canvas');
      drawAttnThumb(c, A.heads[h]);
      const box = el('input', { type: 'checkbox' });
      box.checked = ablated.has(S.layer + ':' + h);
      box.onclick = (e) => e.stopPropagation();
      box.onchange = () => setAblation(h, box.checked);
      heads.push(el('div', {
        class: 'head' + (h === S.head ? ' sel' : '') + (box.checked ? ' ablated' : ''),
        onclick: () => { S.head = h; renderTokens(); renderTab(); },
      }, c, el('span', { text: `head ${h}` }), el('label', { title: 'zero this head\'s output and re-run' }, box, 'ablate')));
    }
    groups.push(el('div', { class: 'group' }, el('div', { class: 'dim', text: `KV head ${g}: these ${group} query heads share one K and one V` }), el('div', { class: 'heads' }, heads)));
  }

  const hn = A.head_norm[S.sel] || [];
  const hmax = Math.max(...hn, 1e-6);
  const normBars = el('div', { class: 'bars' }, hn.map((v, h) => [
    el('span', { text: `head ${h}` }), el('div', { class: 'bar' + (h === S.head ? ' alt' : ''), style: `width:${(v / hmax) * 300}px` }), el('span', { text: v.toFixed(2) })]));

  return [
    explain('Each head turns every token into a ', B('query'), ' ("what am I looking for?") and a ', B('key'), ' ("what do I contain?"). ',
      'Row i of a map is how token i spreads its attention over tokens 0…i, from softmax(q·k / √d). Everything above the diagonal is the causal mask: no peeking ahead. ',
      'Grouped-query attention: query heads come in groups that share one key/value head, which shrinks the KV cache ', String(group), '×. ',
      'Many heads park most of their attention on position 0, an "attention sink" used as a no-op. ',
      'The token strip above is shaded by where the selected token looks in the selected head. Colour is √weight so that small weights stay visible.'),
    el('div', { class: 'controls' },
      slider('layer', S.layer, m.n_layers - 1, (v) => { S.layer = v; renderTab(); }),
      S.st.ablate.length ? el('span', { class: 'dim', text: `ablated: ${S.st.ablate.map(([l, h]) => `L${l}H${h}`).join(', ')}` }) : null,
      S.st.ablate.length ? el('button', { text: 'clear ablations', onclick: () => act(async () => setState(await call('ablate', { heads: [] }))) }) : null),
    el('div', { class: 'groups' }, groups),
    el('h3', { text: `layer ${S.layer} · head ${S.head} (KV head ${Math.floor(S.head / group)})` }),
    el('div', { class: 'scroll' }, drawAttnBig(A.heads[S.head], toks)),
    el('h3', { text: `how much each head writes at position ${S.sel} (‖head output‖, before o_proj)` }),
    normBars,
  ];
}

// ---------------------------------------------------------------- residual stream

async function viewResid() {
  const st = S.st, m = st.model, toks = st.tokens, n = toks.length;
  const R = st.resid_norm; // [stage][pos]
  // Heatmap of ‖x‖ at the output of each layer, log scale.
  const stages = [0, ...Array.from({ length: m.n_layers }, (_, l) => 2 * l + 2)];
  const vals = stages.flatMap((s) => R[s]);
  const lo = Math.log(Math.min(...vals) + 1e-6), hi = Math.log(Math.max(...vals) + 1e-6);
  const cw = Math.max(4, Math.min(24, Math.floor(900 / n))), ch = 12, ml = 56;
  const hm = el('canvas');
  const g = hiDPI(hm, ml + n * cw, stages.length * ch + 4);
  g.font = '10px ui-monospace, Menlo, monospace'; g.textBaseline = 'middle'; g.textAlign = 'right';
  stages.forEach((s, si) => {
    const y = (stages.length - 1 - si) * ch;
    g.fillStyle = '#8a93a3';
    if (si % 2 === 0 || si === stages.length - 1) g.fillText(si === 0 ? 'embed' : 'L' + (si - 1), ml - 6, y + ch / 2);
    for (let p = 0; p < n; p++) {
      g.fillStyle = heat((Math.log(R[s][p] + 1e-6) - lo) / (hi - lo || 1));
      g.fillRect(ml + p * cw, y, cw - 1, ch - 1);
    }
  });
  g.strokeStyle = '#fff'; g.strokeRect(ml + S.sel * cw - 0.5, 0, cw, stages.length * ch);
  hm.onmousemove = (e) => {
    const r = hm.getBoundingClientRect();
    const p = Math.floor((e.clientX - r.left - ml) / cw), si = stages.length - 1 - Math.floor((e.clientY - r.top) / ch);
    if (p < 0 || p >= n || si < 0 || si >= stages.length) return hideTip();
    showTip(e, el('b', { text: `${si === 0 ? 'embedding' : 'after layer ' + (si - 1)} · position ${p} ${JSON.stringify(toks[p].text)}` }),
      el('div', { text: `‖x‖ = ${R[stages[si]][p].toFixed(2)}` }));
  };
  hm.onmouseleave = hideTip;
  hm.onclick = (e) => {
    const p = Math.floor((e.clientX - hm.getBoundingClientRect().left - ml) / cw);
    if (p >= 0 && p < n) { S.sel = p; renderTokens(); renderTab(); }
  };

  // Per-layer writes for the selected position.
  const p = S.sel;
  const aw = st.attn_write.map((v) => v[p]), mw = st.mlp_write.map((v) => v[p]);
  const rn = stages.map((s) => R[s][p]);
  const W = 900, H = 230, pad = 34, bw = (W - pad * 2) / m.n_layers;
  const ch2 = el('canvas');
  const g2 = hiDPI(ch2, W, H);
  const ymax = Math.max(...aw, ...mw, 1e-6), rmax = Math.max(...rn, 1e-6);
  g2.fillStyle = '#8a93a3'; g2.font = '10px ui-monospace, Menlo, monospace'; g2.textAlign = 'center';
  for (let l = 0; l < m.n_layers; l++) {
    const x = pad + l * bw;
    const ha = (aw[l] / ymax) * (H - 50), hmv = (mw[l] / ymax) * (H - 50);
    g2.fillStyle = '#5fb3f0'; g2.fillRect(x + 2, H - 20 - ha, bw / 2 - 3, ha);
    g2.fillStyle = '#f0a050'; g2.fillRect(x + bw / 2, H - 20 - hmv, bw / 2 - 3, hmv);
    g2.fillStyle = '#8a93a3'; if (l % 2 === 0) g2.fillText(String(l), x + bw / 2, H - 6);
  }
  g2.strokeStyle = '#d8dee9'; g2.lineWidth = 1.5; g2.beginPath();
  rn.forEach((v, i) => {
    const x = pad + (i - 0.5) * bw + bw / 2, y = H - 20 - (v / rmax) * (H - 50);
    i ? g2.lineTo(Math.max(pad, x), y) : g2.moveTo(pad, y);
  });
  g2.stroke();
  ch2.onmousemove = (e) => {
    const l = Math.floor((e.clientX - ch2.getBoundingClientRect().left - pad) / bw);
    if (l < 0 || l >= m.n_layers) return hideTip();
    showTip(e, el('b', { text: `layer ${l} at position ${p}` }),
      el('div', { text: `attention adds ‖Δ‖ = ${aw[l].toFixed(2)}` }),
      el('div', { text: `MLP adds ‖Δ‖ = ${mw[l].toFixed(2)}` }),
      el('div', { text: `‖x‖ after the layer = ${rn[l + 1].toFixed(2)}` }));
  };
  ch2.onmouseleave = hideTip;

  return [
    explain('The residual stream is the model\'s working memory: one ', String(m.hidden), '-dim vector per token. No layer ever overwrites it; each one ',
      B('adds'), ' a correction (x ← x + attn(norm(x)), then x ← x + mlp(norm(x))). RMSNorm before each sublayer means the layers see the direction, ',
      'not the size. That lets the stream grow freely, and some positions, often the first token, develop huge norms: those are the "massive activations" that attention sinks rely on.'),
    el('h3', { text: '‖residual‖ at each layer × position (log colour scale) · click a column to select' }),
    el('div', { class: 'scroll' }, hm),
    el('h3', { text: `what each layer writes at position ${p} ${JSON.stringify(toks[p].text)}` }),
    el('div', { class: 'legend' }, el('span', { style: 'color:#5fb3f0', text: '■ attention ‖Δ‖' }), el('span', { style: 'color:#f0a050', text: '■ MLP ‖Δ‖' }), el('span', { text: '— ‖x‖ (own scale)' })),
    el('div', { class: 'scroll' }, ch2),
  ];
}

// ---------------------------------------------------------------- KV cache

async function viewKV() {
  const st = S.st, m = st.model, c = st.cache, toks = st.tokens, op = st.last_op;
  const KV = await cached('kv' + S.layer, 'kv', { layer: S.layer });
  const group = m.n_heads / m.n_kv_heads;
  const full = c.bytes_per_token * group;

  // Occupancy: layers × positions.
  const shown = Math.min(c.max_seq, Math.max(c.len + 24, 64));
  const cw = Math.max(2, Math.min(14, Math.floor(900 / shown))), chh = 6;
  const occ = el('canvas');
  const g = hiDPI(occ, shown * cw, m.n_layers * chh);
  const lo = op.pos0, hi = op.pos0 + op.n;
  for (let l = 0; l < m.n_layers; l++)
    for (let p = 0; p < shown; p++) {
      g.fillStyle = p >= c.len ? '#1d222b' : p >= lo && p < hi ? '#f0a050' : '#3d6690';
      g.fillRect(p * cw, l * chh, cw - 1, chh - 1);
    }

  // K or V of one KV head: positions × head_dim.
  const rows = KV[S.kvWhich];
  const off = S.kvHead * m.head_dim;
  let amax = 1e-6;
  for (const r of rows) for (let d = 0; d < m.head_dim; d++) amax = Math.max(amax, Math.abs(r[off + d]));
  const cell = Math.max(3, Math.min(12, Math.floor(560 / rows.length))), dc = 9, ml = 70;
  const hmc = el('canvas');
  const g2 = hiDPI(hmc, ml + m.head_dim * dc, rows.length * cell + 16);
  g2.font = '10px ui-monospace, Menlo, monospace'; g2.textBaseline = 'middle'; g2.textAlign = 'right';
  rows.forEach((r, p) => {
    if (cell >= 9) { g2.fillStyle = p >= lo && p < hi ? '#f0a050' : '#8a93a3'; g2.fillText(clip(vis(toks[p].text), 9), ml - 4, p * cell + cell / 2); }
    for (let d = 0; d < m.head_dim; d++) {
      g2.fillStyle = `rgb(${divRGB(r[off + d] / amax).join(',')})`;
      g2.fillRect(ml + d * dc, p * cell, dc - 1, cell - (cell > 4 ? 1 : 0));
    }
  });
  g2.fillStyle = '#8a93a3'; g2.textAlign = 'center';
  for (let d = 0; d < m.head_dim; d += 8) g2.fillText(String(d), ml + d * dc + dc / 2, rows.length * cell + 8);
  hmc.onmousemove = (e) => {
    const r = hmc.getBoundingClientRect();
    const d = Math.floor((e.clientX - r.left - ml) / dc), p = Math.floor((e.clientY - r.top) / cell);
    if (d < 0 || d >= m.head_dim || p < 0 || p >= rows.length) return hideTip();
    const half = m.head_dim / 2, i = d % half;
    showTip(e, el('b', { text: `${S.kvWhich.toUpperCase()}[pos ${p} ${JSON.stringify(toks[p].text)}][dim ${d}] = ${rows[p][off + d].toFixed(3)}` }),
      S.kvWhich === 'k' ? el('div', { class: 'dim', text: `RoPE pair (${i}, ${i + half}): rotates ${(1 / Math.pow(m.rope_theta, (2 * i) / m.head_dim)).toPrecision(3)} rad per position` }) : null);
  };
  hmc.onmouseleave = hideTip;

  const pick = (label, cur, opts, set) => el('span', {}, label + ' ', ...opts.map(([v, t]) =>
    el('button', { class: cur === v ? 'on' : '', text: t, onclick: () => { set(v); renderTab(); } })));

  return [
    explain('During decoding, each new token needs the keys and values of every earlier token, in every layer. Those never change, so they are computed once and ',
      B('cached'), '. A decode step then runs the model on just 1 token, appends 1 row of K and V per layer, and attends over the cache. ',
      'Prefill writes a whole block at once; each decode step writes one column (orange = written by the last operation). ',
      'Recomputing instead would make generating n tokens cost O(n²) forward work.'),
    el('div', { class: 'kpi' },
      el('div', {}, el('b', { text: fmtBytes(c.bytes) }), `in use (${c.len} positions)`),
      el('div', {}, el('b', { text: fmtBytes(c.bytes_per_token) }), `per token = ${m.n_layers} layers × 2 (K,V) × ${c.kv_dim} floats × 4 B`),
      el('div', {}, el('b', { text: fmtBytes(full) }), `per token without GQA (${m.n_heads} KV heads instead of ${m.n_kv_heads})`),
      el('div', {}, el('b', { text: fmtBytes(c.bytes_per_token * 8192) }), 'at the model\'s full 8192-token context')),
    el('h3', { text: `occupancy: ${m.n_layers} layers (rows) × positions (columns)` }),
    el('div', { class: 'scroll' }, occ),
    el('div', { class: 'controls', style: 'margin-top:14px' },
      slider('layer', S.layer, m.n_layers - 1, (v) => { S.layer = v; renderTab(); }),
      pick('show', S.kvWhich, [['k', 'K (keys, after RoPE)'], ['v', 'V (values)']], (v) => (S.kvWhich = v)),
      pick('KV head', S.kvHead, Array.from({ length: m.n_kv_heads }, (_, i) => [i, String(i)]), (v) => (S.kvHead = v))),
    explain(S.kvWhich === 'k'
      ? `Keys are stored after RoPE. Dimension i and i+${m.head_dim / 2} form a pair rotated by an angle proportional to the position; pair 0 turns fastest (1 rad per token), and the last pairs barely move. Hover a column to see its rotation speed. Fast pairs encode fine-grained relative position; slow pairs act almost like position-free channels.`
      : 'Values are what gets mixed together, weighted by attention. They carry no positional rotation.'),
    el('div', { class: 'scroll' }, hmc),
    legend('', (v) => `rgb(${divRGB(v * 2 - 1).join(',')})`, '−' + amax.toFixed(2), '+' + amax.toFixed(2)),
  ];
}

// ---------------------------------------------------------------- MLP

async function viewMLP() {
  const m = S.st.model, toks = S.st.tokens, p = S.sel;
  const M = await cached(`mlp${S.layer}:${p}`, 'mlp', { layer: S.layer, pos: p });
  const h = M.hidden;
  const amax = Math.max(...h.map(Math.abs), 1e-6);
  const order = h.map((v, i) => [i, v]).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  const small = h.filter((v) => Math.abs(v) < 0.05 * amax).length;

  const W = 1000, H = 160;
  const cv = el('canvas');
  const g = hiDPI(cv, W, H);
  const bw = W / h.length;
  g.fillStyle = '#2a303b'; g.fillRect(0, H / 2, W, 1);
  h.forEach((v, i) => {
    g.fillStyle = v >= 0 ? '#f0a050' : '#5fb3f0';
    const len = (Math.abs(v) / amax) * (H / 2 - 4);
    g.fillRect(i * bw, v >= 0 ? H / 2 - len : H / 2, Math.max(1, bw), len);
  });
  cv.onmousemove = (e) => {
    const i = Math.floor((e.clientX - cv.getBoundingClientRect().left) / bw);
    if (i < 0 || i >= h.length) return hideTip();
    showTip(e, el('b', { text: `neuron ${i}` }), el('div', { text: `silu(gate)·up = ${h[i].toFixed(3)}` }));
  };
  cv.onmouseleave = hideTip;

  return [
    explain('The MLP is where most parameters live (', (3 * m.hidden * m.intermediate / 1e6).toFixed(2), 'M per layer of ', String((S.st.model.params / m.n_layers / 1e6).toFixed(1)), 'M). ',
      'SwiGLU: two projections up to ', String(m.intermediate), ' "neurons", one passed through SiLU as a ', B('gate'), ', multiplied together, and projected back down. ',
      'Each neuron is a feature detector: it fires when the input matches a pattern, and its down-projection column says what to add to the stream. ',
      'Notice how few neurons carry most of the signal: activations are sparse-ish and heavy-tailed.'),
    el('div', { class: 'controls' },
      slider('layer', S.layer, m.n_layers - 1, (v) => { S.layer = v; renderTab(); }),
      el('span', { class: 'dim', text: `position ${p} ${JSON.stringify(toks[p].text)} (click a token to change)` })),
    el('div', { class: 'kpi' },
      el('div', {}, el('b', { text: amax.toFixed(2) }), 'max |activation|'),
      el('div', {}, el('b', { text: ((100 * small) / h.length).toFixed(1) + '%' }), 'of neurons under 5% of max'),
      el('div', {}, el('b', { text: String(order.findIndex((_, k) => order.slice(0, k + 1).reduce((s, [, v]) => s + v * v, 0) >= 0.5 * h.reduce((s, v) => s + v * v, 0)) + 1) }), 'neurons hold half the energy')),
    el('h3', { text: `all ${h.length} hidden activations, layer ${S.layer}` }),
    el('div', { class: 'scroll' }, cv),
    el('h3', { text: 'strongest neurons' }),
    el('div', { class: 'bars' }, order.slice(0, 15).map(([i, v]) => [
      el('span', { text: `#${i}` }), el('div', { class: 'bar' + (v < 0 ? '' : ' alt'), style: `width:${(Math.abs(v) / amax) * 400}px` }), el('span', { text: v.toFixed(3) })])),
  ];
}

// ---------------------------------------------------------------- next token

function samplingDist(top, lse) {
  const s = samp();
  const recent = new Set(S.st.tokens.slice(-64).map((t) => t.id));
  let cand = top.map(([id, text, l]) => {
    if (s.repeat_penalty !== 1 && recent.has(id)) l = l > 0 ? l / s.repeat_penalty : l * s.repeat_penalty;
    return { id, text, raw: Math.exp(top.find((x) => x[0] === id)[2] - lse), l };
  });
  cand.sort((a, b) => b.l - a.l);
  if (s.temperature <= 0) return cand.map((c, i) => ({ ...c, p: i === 0 ? 1 : 0 }));
  let z = cand.map((c) => c.l / s.temperature);
  const mx = Math.max(...z);
  let e = z.map((v) => Math.exp(v - mx));
  if (s.top_k > 0) e = e.map((v, i) => (i < s.top_k ? v : 0));
  let sum = e.reduce((a, b) => a + b, 0);
  let p = e.map((v) => v / sum);
  if (s.top_p < 1) {
    let acc = 0, cut = p.length;
    for (let i = 0; i < p.length; i++) { acc += p[i]; if (acc >= s.top_p) { cut = i + 1; break; } }
    p = p.map((v, i) => (i < cut ? v : 0));
    sum = p.reduce((a, b) => a + b, 0);
    p = p.map((v) => v / sum);
  }
  return cand.map((c, i) => ({ ...c, p: p[i] }));
}

async function viewNext() {
  const { top, lse } = S.st.next;
  const toks = S.st.tokens;
  const dist = samplingDist(top, lse).slice(0, 30);
  const covered = top.reduce((a, [, , l]) => a + Math.exp(l - lse), 0);
  const rows = dist.map((d) => {
    const row = [
      el('span', { class: 'mono clickrow', text: clip(vis(d.text), 18), title: `id ${d.id}: click to force this token` }),
      el('div', { class: 'barcell' },
        el('div', { class: 'bar', style: `width:${Math.max(1, d.raw * 420)}px`, title: 'model probability' }),
        el('div', { class: 'bar alt', style: `width:${Math.max(d.p > 0 ? 1 : 0, d.p * 420)}px`, title: 'after sampling settings' })),
      el('span', { text: `${pct(d.raw)} → ${d.p > 0 ? pct(d.p) : '0'}` }),
    ];
    row[0].onclick = () => act(async () => setState(await call('step', { n: 1, force: d.id, ...samp() }), 'last'));
    return row;
  });
  return [
    explain('The final layer produces one logit per vocabulary entry; softmax turns them into the ', B('model\'s distribution'), ' (blue). ',
      'Sampling settings then reshape it (orange): temperature divides the logits, top-k and top-p cut the tail, and the repetition penalty pushes down recently used tokens. ',
      'Change the settings above and watch the orange bars move. Click a token to force it as the next token: an intervention on the sampler, not the model.'),
    el('div', { class: 'kpi' },
      el('div', {}, el('b', { text: JSON.stringify(toks[toks.length - 1].text) }), `last token (position ${toks.length - 1})`),
      el('div', {}, el('b', { text: pct(covered) }), 'of probability mass in the top 50 shown'),
      el('div', {}, el('b', { text: lse.toFixed(2) }), 'log Σ exp(logits)')),
    el('div', { class: 'bars' }, rows),
    el('p', { class: 'dim', style: 'font-size:12px', text: 'Rows are sorted by the adjusted score, so tokens hit by the repetition penalty can sit below less likely ones. Orange is renormalized over the top 50 tokens, a close approximation unless the temperature is very high.' }),
  ];
}

// ---------------------------------------------------------------- tokenizer

async function viewTok() {
  const text = $('#prompt').value;
  const T = await call('tokenize', { text });
  const bytes = new TextEncoder().encode(text).length;
  const words = T.words.map((w) => {
    const steps = w.steps;
    const shown = steps.length > 7 ? [...steps.slice(0, 3), null, ...steps.slice(-3)] : steps;
    return el('div', { class: 'wordbox' },
      el('div', { class: 'dim mono', text: JSON.stringify(w.word) }),
      shown.map((s, si) => s === null
        ? el('div', { class: 'dim', text: `… ${steps.length - 6} more merges …` })
        : el('div', { class: 'chips' }, s.map(([id, t]) => el('span', {
          class: 'chip' + (si > 0 && s.length < (shown[si - 1] || s).length && !(shown[si - 1] || []).some((x) => x[0] === id) ? ' new' : ''),
          title: `id ${id}`, text: vis(t),
        })))));
  });
  return [
    explain('Byte-level BPE. The text is first split into "pre-tokens" (words with their leading space, single digits, punctuation runs, whitespace). ',
      'Each starts as raw UTF-8 bytes, then the tokenizer repeatedly merges the adjacent pair that was learned ', B('earliest'), ' in training, until no learned pair is left. ',
      'Each box below shows one pre-token being merged, top to bottom (orange = the newly merged token). Special tokens like <|im_start|> skip all this. This view tokenizes the prompt box above.'),
    el('div', { class: 'kpi' },
      el('div', {}, el('b', { text: String(bytes) }), 'UTF-8 bytes'),
      el('div', {}, el('b', { text: String(T.ids.length) }), 'tokens'),
      el('div', {}, el('b', { text: (bytes / Math.max(1, T.ids.length)).toFixed(2) }), 'bytes per token')),
    el('div', {}, words),
  ];
}

// ------------------------------------------------------------------ loading screen

function showLoading(on, error) {
  $('#loading').style.display = on ? 'flex' : 'none';
  if (error) {
    $('#load-stage').textContent = 'Could not load the model';
    $('#load-detail').textContent = error;
    $('#load-bar').style.width = '0';
  }
}

function showProgress(p) {
  const frac = p.total ? p.got / p.total : 0;
  $('#load-stage').textContent = p.stage + (p.cached ? ' (from browser cache)' : '');
  $('#load-bar').style.width = (100 * frac).toFixed(1) + '%';
  $('#load-detail').textContent = p.total > 1e5
    ? `${(p.got / 1e6).toFixed(0)} / ${(p.total / 1e6).toFixed(0)} MB`
    : p.total ? `${p.got} / ${p.total}` : '';
}

// ------------------------------------------------------------------ modes

let MODE = 'complete';
let READY = false; // the engine is loaded

function setMode(mode) {
  MODE = mode;
  $('#complete-view').hidden = mode !== 'complete';
  $('#chat-view').hidden = mode !== 'chat';
  document.querySelectorAll('#modes button').forEach((b) => b.classList.toggle('active', b.dataset.mode === mode));
  history.replaceState(null, '', '#' + (mode === 'chat' ? 'chat' : S.tab));
  hideTip();
  if (mode === 'chat') {
    if (READY && !C.st) refreshChat();
    $('#chat-input').focus();
  } else if (S.st) {
    renderTokens();
    renderTab();
  }
}

// ------------------------------------------------------------------ chat mode
//
// A normal chat app on top of the same engine, with a lighter look inside:
// each reply token is tinted by how unsure the model was (hover for the
// alternatives), and the side panel shows what each turn cost.

const C = {
  st: null,          // chat_state from the engine
  busy: false,
  stopRequested: false,
  raw: false,        // show the raw token stream instead of bubbles
  tint: true,        // tint uncertain tokens
};

function chatSamp() {
  return {
    temperature: numIn('#c-temp', 0, 0),
    top_p: numIn('#c-topp', 1, 0, 1),
    repeat_penalty: numIn('#c-rep', 1, 0.01, 100),
  };
}

function tintOf(p) {
  // Confident tokens stay plain; the less sure the model was, the stronger the tint.
  const a = Math.max(0, Math.min(0.6, (1 - p) * 0.75));
  return a < 0.04 ? '' : `rgba(127,163,207,${a.toFixed(2)})`;
}

function tokenTip(e, t) {
  showTip(e,
    el('b', { text: `chose ${JSON.stringify(t.text)} · ${pct(t.p)}` }),
    el('div', { class: 'bars', style: 'margin:6px 0' }, t.alts.map(([text, p]) => [
      el('span', { class: 'mono', style: text === t.text ? 'color:var(--accent)' : '', text: clip(vis(text), 16) }),
      el('div', { class: 'bar', style: `width:${Math.max(1, p * 160)}px` }),
      el('span', { text: pct(p) })])),
    el('div', { class: 'dim', text: t.p < 0.5 ? 'the model was unsure here: other tokens were nearly as likely' : 'the model\'s top 5 candidates at this point' }));
}

// Append reply tokens to an assistant bubble. Bytes go through a streaming UTF-8
// decoder, because one character (emoji, CJK) can be split over several tokens.
function appendTokens(bubble, tokens) {
  for (const t of tokens) {
    const piece = bubble.decoder.decode(new Uint8Array(t.bytes), { stream: true });
    const span = el('span', { class: 'ct', text: piece });
    span.dataset.p = t.p;
    if (C.tint) span.style.background = tintOf(t.p);
    span.onmousemove = (e) => tokenTip(e, t);
    span.onmouseleave = hideTip;
    bubble.body.append(span);
  }
  const box = $('#messages');
  box.scrollTop = box.scrollHeight;
}

// End of a reply: a character cut off mid-way (the reply stopped inside an
// emoji, say) is still in the decoder. Flush it as U+FFFD rather than drop it.
function finishBubble(bubble) {
  const tail = bubble.decoder.decode();
  if (tail) bubble.body.append(el('span', { class: 'ct', text: tail }));
}

function addBubble(role, text) {
  const body = el('div', { class: 'body' }, text || '');
  const meta = el('div', { class: 'meta dim' });
  const node = el('div', { class: 'msg ' + role }, body, meta);
  $('#messages').append(node);
  $('#messages').scrollTop = $('#messages').scrollHeight;
  return { node, body, meta, decoder: new TextDecoder() };
}

function renderChatRaw() {
  const box = $('#messages');
  const toks = C.st ? C.st.tokens : [];
  box.replaceChildren(
    el('p', { class: 'explain' }, 'This is exactly what the model sees: ', B(`${toks.length} tokens`),
      '. Chat is still text completion. The conversation is written in the ChatML format the model was fine-tuned on, ',
      'and the reply is the model completing it. Each ', el('span', { class: 'mono', style: 'color:var(--accent)', text: '<|im_start|>' }),
      ' / ', el('span', { class: 'mono', style: 'color:var(--accent)', text: '<|im_end|>' }),
      ' is a single special token. The model ends its turn by producing <|im_end|> itself.'),
    el('div', { class: 'rawstream' }, toks.map((t) => el('span', { class: 'tok' + (t.special ? ' special' : ''), text: vis(t.text), title: `id ${t.id}` }))));
}

let chatLog = []; // [{role, text} | {role:'assistant', tokens, stop}] for re-rendering

function renderChatBubbles() {
  $('#messages').replaceChildren();
  if (!chatLog.length) {
    $('#messages').append(el('div', { class: 'empty dim' },
      el('p', { text: 'Say something to SmolLM2-135M-Instruct.' }),
      el('p', { text: 'It is a tiny model (135M parameters): fluent and quick, but often wrong. Hover any word of its replies to see what else it considered.' })));
    return;
  }
  for (const m of chatLog) {
    if (m.role === 'user') { addBubble('user', m.text); continue; }
    const b = addBubble('assistant');
    appendTokens(b, m.tokens);
    finishBubble(b);
    if (m.stop) b.meta.textContent = m.stop;
  }
}

function renderMessages() {
  C.raw ? renderChatRaw() : renderChatBubbles();
}

function renderChatSide() {
  if (!C.st) return;
  const c = C.st.context, t = C.st.last_turn;
  $('#c-ctxbar').style.width = (100 * c.len / c.max_seq).toFixed(1) + '%';
  $('#c-ctx').textContent = `${c.len} / ${c.max_seq} tokens · KV cache ${fmtBytes(c.bytes)}`;
  const turn = $('#c-turn');
  if (!t) { turn.replaceChildren(el('span', { class: 'dim', text: 'nothing yet' })); return; }
  const rate = t.generated / Math.max(1e-6, t.decode_ms / 1000);
  turn.replaceChildren(
    el('div', {}, B(String(t.prefilled)), ' new tokens prefilled in one pass', el('span', { class: 'dim', text: ` (${t.prefill_ms.toFixed(0)} ms)` })),
    el('div', {}, B(String(t.reused)), ' earlier tokens reused from the KV cache'),
    el('div', {}, B(String(t.generated)), ' tokens generated', t.generated ? el('span', { class: 'dim', text: ` · ${rate.toFixed(0)} tok/s` }) : null),
    t.stop ? el('div', { class: 'dim', text: t.stop }) : null);
}

async function refreshChat() {
  C.st = await call('chat_state');
  const s = C.st.sampler;
  $('#c-temp').value = +s.temperature.toFixed(3);
  $('#c-topp').value = +s.top_p.toFixed(3);
  $('#c-rep').value = +s.repeat_penalty.toFixed(2);
  if (!$('#c-system').value) $('#c-system').value = C.st.system;
  renderChatSide();
  // Bubbles are built incrementally while chatting; only the raw view is redrawn from state.
  if (C.raw) renderChatRaw();
}

function setChatBusy(on) {
  C.busy = on;
  $('#send').hidden = on;
  $('#stop').hidden = !on;
}

async function sendChat() {
  const input = $('#chat-input');
  const text = input.value.trim();
  if (!text || C.busy || !READY) return;
  input.value = '';
  setChatBusy(true);
  C.stopRequested = false;
  if (C.raw) { C.raw = false; $('#c-raw').checked = false; renderMessages(); }
  if (!chatLog.length) $('#messages').replaceChildren();
  chatLog.push({ role: 'user', text });
  addBubble('user', text);
  const msg = { role: 'assistant', tokens: [], stop: '' };
  const bubble = addBubble('assistant');
  bubble.node.classList.add('streaming');
  try {
    C.st = await call('chat_send', { text });
    chatLog.push(msg);
    renderChatSide();
    for (;;) {
      if (C.stopRequested) { await call('chat_stop'); msg.stop = 'stopped'; break; }
      const r = await call('chat_step', { n: 2, ...chatSamp() });
      msg.tokens.push(...r.tokens);
      appendTokens(bubble, r.tokens);
      C.st.context = r.context;
      C.st.last_turn = r.last_turn;
      renderChatSide();
      if (r.done) { msg.stop = r.stop || ''; break; }
    }
  } catch (e) {
    msg.stop = 'error: ' + e.message;
    flash(e.message);
  } finally {
    finishBubble(bubble);
    bubble.node.classList.remove('streaming');
    bubble.meta.textContent = msg.stop;
    setChatBusy(false);
    try { C.st = await call('chat_state'); renderChatSide(); } catch {}
    input.focus();
  }
}

async function newChat() {
  if (C.busy) return;
  C.st = await call('chat_reset', { system: $('#c-system').value });
  chatLog = [];
  renderChatSide();
  renderMessages();
  $('#chat-input').focus();
}

// Hand the conversation to completion mode for the full set of views, as the
// exact token ids chat produced. The inspector keeps a full trace per position,
// so it holds fewer positions than chat; a long conversation is cut to its tail.
async function inspectChat() {
  if (C.busy || !C.st || !C.st.tokens.length) return;
  const limit = S.st.cache.max_seq - 8;
  let toks = C.st.tokens;
  if (toks.length > limit) {
    toks = toks.slice(-limit);
    flash(`The conversation is ${C.st.tokens.length} tokens; the inspector holds ${S.st.cache.max_seq} positions (it stores every activation at every position), so it shows the last ${limit}.`);
  }
  $('#prompt').value = toks.map((t) => t.text).join('');
  setMode('complete');
  await act(async () => setState(await call('run', { ids: toks.map((t) => t.id) }), 'last'));
}

function setupChat() {
  const num = (id, value, step, title) => el('input', { id, type: 'number', step, min: 0, value, title });
  const raw = el('input', { type: 'checkbox', id: 'c-raw' });
  raw.onchange = () => { C.raw = raw.checked; if (C.raw) refreshChat(); else renderMessages(); };
  const tint = el('input', { type: 'checkbox' });
  tint.checked = true;
  tint.onchange = () => {
    C.tint = tint.checked;
    document.querySelectorAll('.msg .ct').forEach((s) => (s.style.background = C.tint ? tintOf(+s.dataset.p) : ''));
  };
  $('#chat-side').append(
    el('h3', { text: 'context window' }),
    el('div', { class: 'ctxbar' }, el('i', { id: 'c-ctxbar' })),
    el('div', { id: 'c-ctx', class: 'dim small' }),
    el('h3', { text: 'last turn' }),
    el('div', { id: 'c-turn', class: 'small turn' }),
    el('p', { class: 'explain small' },
      'Only the new tokens of each turn go through the model in one pass (', B('prefill'), '); everything before is already in the ',
      B('KV cache'), '. The reply then comes one token at a time (', B('decode'), '), each step reading the whole cache.'),
    el('h3', { text: 'view' }),
    el('label', { class: 'block' }, tint, 'tint tokens the model was unsure about'),
    el('label', { class: 'block' }, raw, 'show the raw token stream'),
    el('button', { class: 'wide', text: 'Inspect this conversation →', title: 'open the transcript in completion mode with every view', onclick: () => inspectChat() }),
    el('h3', { text: 'sampling' }),
    el('div', { class: 'row tight' },
      el('label', {}, 'temp', num('c-temp', 0.2, 0.1)),
      el('label', {}, 'top-p', num('c-topp', 0.9, 0.05)),
      el('label', {}, 'repeat', num('c-rep', 1.1, 0.05))),
    el('h3', { text: 'system prompt' }),
    el('textarea', { id: 'c-system', rows: 3, spellcheck: 'false' }),
    el('button', { class: 'wide', text: 'New chat', title: 'clear the conversation (applies the system prompt)', onclick: () => newChat() }));

  $('#composer').onsubmit = (e) => { e.preventDefault(); sendChat(); };
  $('#chat-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); sendChat(); }
  });
  $('#stop').onclick = () => (C.stopRequested = true);
  document.querySelectorAll('#modes button').forEach((b) => (b.onclick = () => setMode(b.dataset.mode)));
  renderMessages();
}

// ------------------------------------------------------------------ wiring

$('#run').onclick = () => act(async () => setState(await call('run', { text: $('#prompt').value }), 'last'));
$('#step').onclick = () => act(async () => setState(await call('step', { n: 1, ...samp() }), 'last'));
$('#gen').onclick = () => act(async () => setState(await call('step', { n: 16, ...samp() }), 'last'));
$('#prompt').addEventListener('keydown', (e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); $('#run').click(); } });
for (const id of ['#temp', '#topp', '#topk', '#rep']) $(id).addEventListener('input', () => { if (S.tab === 'next') renderTab(); });
document.addEventListener('keydown', (e) => {
  if (MODE !== 'complete' || !S.st || e.target.matches('textarea, input')) return;
  if (e.key === 'ArrowLeft' && S.sel > 0) S.sel--;
  else if (e.key === 'ArrowRight' && S.sel < S.st.tokens.length - 1) S.sel++;
  else return;
  e.preventDefault();
  renderTokens();
  renderTab();
});

(async () => {
  const fromHash = location.hash.slice(1);
  if (TABS.some((t) => t[0] === fromHash)) S.tab = fromHash;
  renderTabs();
  setupChat();
  setMode(fromHash === 'chat' ? 'chat' : 'complete');
  try {
    $('#chat-input').disabled = true;
    if (!NATIVE) {
      showLoading(true);
      startWorker(showProgress);
      await call('__load', { model: MODEL, source: SOURCE });
      showLoading(false);
    }
    READY = true;
    $('#chat-input').disabled = false;
    const st = await call('state');
    const s = st.sampler;
    $('#temp').value = +s.temperature.toFixed(3); $('#topp').value = +s.top_p.toFixed(3); $('#topk').value = s.top_k; $('#rep').value = +s.repeat_penalty.toFixed(2);
    $('#prompt').value = st.tokens.map((t) => t.text).join('');
    setState(st, 'last');
    if (MODE === 'chat') await refreshChat();
  } catch (e) {
    showLoading(true, e.message);
    flash(e.message);
  }
})();
