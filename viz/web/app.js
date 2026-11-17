'use strict';
// Inside SmolLM2: a page over the engine's trace. All state lives on the server;
// this file fetches views of it and draws them.

const $ = (s) => document.querySelector(s);
const S = {
  st: null,        // /api/state
  sel: 0,          // selected position
  tab: 'next',
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
const MODEL = 'smollm2-135m';
// Where the weights come from: the local dev server when the page itself is on
// localhost (no 269 MB downloads while developing), Hugging Face otherwise.
const IS_LOCAL = ['localhost', '127.0.0.1', '[::1]'].includes(location.hostname);
const SOURCE = PARAMS.get('source') || (IS_LOCAL ? 'local' : 'hf');

let worker = null;
const pending = new Map();
let nextId = 1;

function startWorker(onProgress) {
  // Always the current worker (it is tiny): a cached old one next to a new
  // app.js would speak an outdated protocol. The weights have their own cache.
  worker = new Worker('worker.js?t=' + Date.now());
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
  f.classList.add('show');
  clearTimeout(flash.t);
  flash.t = setTimeout(() => f.classList.remove('show'), 5000);
}

async function act(fn) {
  if (S.busy) return;
  S.busy = true;
  document.body.classList.add('busy', 'working');
  try { await fn(); } catch (e) { flash(e.message); console.error(e); }
  finally { S.busy = false; document.body.classList.remove('busy', 'working'); }
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

// Range inputs show their fill as a gradient up to the thumb (--fill), and a
// knob's <output> shows the value, or a word like "off" at a neutral setting.
function syncRange(r) {
  const min = +r.min || 0, max = r.max === '' ? 100 : +r.max, v = +r.value;
  r.style.setProperty('--fill', (100 * (v - min) / Math.max(1e-9, max - min)).toFixed(2) + '%');
  const out = r.nextElementSibling;
  if (out && out.tagName === 'OUTPUT') {
    const d = out.dataset;
    const dec = (r.step.split('.')[1] || '').length;
    out.textContent = (v === 0 && d.zero) || (v === max && d.max) || (v === min && d.min) || v.toFixed(dec);
    out.classList.toggle('neutral', out.textContent !== v.toFixed(dec));
  }
}
document.addEventListener('input', (e) => { if (e.target.type === 'range') syncRange(e.target); });

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
    dry: numIn('#dry', 0, 0, 100),
  };
}

// ------------------------------------------------------------------ sampler presets
//
// One click sets every sampler knob. They run from "the model alone" to "the
// sampler steps in a lot": the Next token tab shows the model's distribution
// next to what the sampler draws from.
const PRESETS = [
  { id: 'greedy', name: 'Greedy', s: { temp: 0, topp: 1, topk: 0, dry: 0, rep: 1 },
    about: 'Always the model\'s favourite token, nothing changed: the model alone. Deterministic, and long continuations tend to loop.' },
  { id: 'focused', name: 'Focused', s: { temp: 0, topp: 0.9, topk: 0, dry: 0.8, rep: 1 },
    about: 'The model\'s favourite, unless anti-loop (DRY) sees a phrase starting to repeat. Deterministic: the same prompt always continues the same way.' },
  { id: 'card', name: 'Model card', s: { temp: 0.2, topp: 0.9, topk: 0, dry: 0, rep: 1 },
    about: 'Temperature 0.2, top-p 0.9, no penalties: close to greedy with a little variety. Still loops.' },
  { id: 'balanced', name: 'Balanced', s: { temp: 0.2, topp: 0.9, topk: 0, dry: 0.8, rep: 1.1 },
    about: 'Model card plus anti-loop and a repeat penalty on words. Varied and never loops, but the penalty often pushes common words like "is" and "the" down.' },
  { id: 'creative', name: 'Creative', s: { temp: 0.8, topp: 0.95, topk: 0, dry: 0.8, rep: 1.1 },
    about: 'Hotter and wider: more varied, more surprising, more often off the rails.' },
  { id: 'odds', name: 'Model\'s own odds', s: { temp: 1, topp: 1, topk: 0, dry: 0, rep: 1 },
    about: 'Draws straight from the model\'s distribution: every token gets exactly the chance the model gave it. Unfiltered, so it wanders.' },
];

// A preset <select> for the knobs with ids `${prefix}temp`, `${prefix}topp`, ...
// It shows "Custom" once a slider is moved off every preset, and tells
// `onMatch` the current preset's name.
function presetPicker(prefix, def, onMatch = () => {}) {
  const sel = el('select', { id: prefix + 'preset', 'aria-label': 'sampler preset' },
    PRESETS.map((p) => el('option', { value: p.id, text: p.name })),
    el('option', { value: 'custom', text: 'Custom', disabled: '' }));
  const about = el('div', { class: 'preset-about dim small' });
  const knobs = () => Object.keys(PRESETS[0].s).map((k) => [k, document.getElementById(prefix + k)]).filter(([, r]) => r);
  const match = () => {
    const p = PRESETS.find((p) => knobs().every(([k, r]) => +r.value === p.s[k]));
    sel.value = p ? p.id : 'custom';
    about.textContent = p ? p.about : 'Your own mix. Pick a preset to start over.';
    onMatch(p ? p.name : 'Custom');
  };
  const apply = (id) => {
    const p = PRESETS.find((p) => p.id === id);
    for (const [k, r] of knobs()) {
      r.value = p.s[k];
      syncRange(r);
      r.dispatchEvent(new Event('input', { bubbles: true }));
    }
    match();
  };
  sel.onchange = () => apply(sel.value);
  // Call once the knobs are in the page.
  const wire = () => {
    for (const [, r] of knobs()) r.addEventListener('input', match);
    apply(def);
  };
  return { sel, about, wire };
}

// selMode: 'last' = select the newest token, 'keep' = keep selection if valid.
function setState(st, selMode = 'keep') {
  S.st = st;
  S.cache = {};
  if (selMode === 'last' || S.sel >= st.tokens.length) S.sel = st.tokens.length - 1;
  renderHeader();
  renderTokens();
  renderStatus();
  renderTab();
}

function renderHeader() {
  const m = S.st.model;
  if (!$('#model-info').childElementCount) {
    const chip = (b, t, title) => el('span', { class: 'chip', title }, el('b', { text: b }), t);
    $('#model-info').replaceChildren(
      chip(`${(m.params / 1e6).toFixed(0)}M`, 'params', m.name),
      chip(String(m.n_layers), 'layers'),
      chip(String(m.hidden), 'd_model', 'width of the residual stream'),
      chip(`${m.n_heads}/${m.n_kv_heads}`, 'heads', `${m.n_heads} query heads share ${m.n_kv_heads} KV heads (grouped-query attention), ${m.head_dim} dims each`),
      chip(String(m.intermediate), 'MLP', 'hidden size of each SwiGLU MLP'),
      chip(`${(m.vocab / 1000).toFixed(0)}k`, 'vocab', `${m.vocab} tokens`));
  }
}

function renderTokens() {
  const box = $('#tokens');
  if (!S.st) { box.replaceChildren(el('span', { class: 'dim', text: 'tokens appear here once the prompt has run' })); return; }
  // Tokens that weren't in the strip last time pop in, one after another.
  const ids = S.st.tokens.map((t) => t.id), prev = S.shownIds || [];
  let same = 0;
  while (same < ids.length && same < prev.length && ids[same] === prev[same]) same++;
  S.shownIds = ids;
  box.replaceChildren(...S.st.tokens.map((t, i) => {
    const c = el('span', {
      class: 'tok' + (t.gen ? ' gen' : '') + (t.special ? ' special' : '') + (i === S.sel ? ' sel' : '') + (i >= same ? ' pop' : ''),
      text: vis(t.text),
    });
    if (i >= same) c.style.animationDelay = Math.min(i - same, 40) * 18 + 'ms';
    c.onclick = () => { S.sel = i; renderTokens(); renderTab(); };
    c.ondblclick = () => act(async () => setState(await call('rewind', { len: i + 1 }), 'last'));
    c.onmousemove = (e) => showTip(e,
      el('b', { text: `position ${i} · token id ${t.id}` }),
      el('div', { class: 'mono', text: JSON.stringify(t.text) }),
      el('div', { class: 'dim', text: (t.special ? 'special token · ' : '') + (t.gen ? 'generated' : 'prompt') }),
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
  const st = $('#status');
  if (!st.childElementCount) {
    st.append(el('span', { class: 'op' }), el('span', { class: 'cache' },
      el('span', { class: 'cachebar', title: 'KV cache fill' }, el('i')), el('span', { class: 'cachetext' })));
  }
  st.querySelector('.op').replaceChildren(el('span', { class: 'opkind ' + op.kind, text: op.kind }), s.replace(/^\w+: /, ''));
  st.querySelector('.cachebar i').style.width = `${(100 * c.len) / c.max_seq}%`;
  st.querySelector('.cachetext').textContent = `KV cache ${c.len}/${c.max_seq} · ${fmtBytes(c.bytes)}`;
}

// ------------------------------------------------------------------ tabs

const TABS = [
  ['next', 'Next token', viewNext],
  ['walk', 'Step by step', viewWalk],
];
// A tab id from elsewhere (glossary, an old cached glossary.js, a URL) may
// name a tab that no longer exists: those go to Step by step.
const tabOf = (id) => (TABS.some((t) => t[0] === id) ? id : 'walk');
const tabName = (id) => TABS.find((t) => t[0] === tabOf(id))[1];

const TAB_ICONS = {
  walk: 'M2 13.5h3v-3h3v-3h3v-3h3',
  next: 'M3 13V9M6.5 13V5M10 13V7M13.5 13V3',
};

// The tab bar is built once and only toggled afterwards, so the underline
// (.ink) can slide from the old tab to the new one.
function renderTabs() {
  const bar = $('#tabs');
  if (!bar.childElementCount) {
    bar.append(...TABS.map(([id, name]) => el('button', {
      'data-tab': id,
      onclick: () => {
        if (S.tab === id) return;
        S.tab = id; S.enter = true;
        history.replaceState(null, '', '#' + id);
        renderTabs(); renderTokens(); renderTab();
      },
    }, svgIcon(TAB_ICONS[id]), name)), el('span', { class: 'ink' }));
  }
  bar.querySelectorAll('button').forEach((b) => b.classList.toggle('active', b.dataset.tab === S.tab));
  moveInk(bar);
}

function svgIcon(d) {
  const ns = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('viewBox', '0 0 16 16');
  svg.setAttribute('aria-hidden', 'true');
  const path = document.createElementNS(ns, 'path');
  path.setAttribute('d', d);
  svg.append(path);
  return svg;
}

// Slide a container's .ink under its active button.
function moveInk(bar) {
  const ink = bar.querySelector('.ink'), b = bar.querySelector('button.active');
  if (!ink || !b || !b.offsetWidth) return;
  ink.style.width = b.offsetWidth + 'px';
  ink.style.transform = `translateX(${b.offsetLeft}px)`;
}
addEventListener('resize', () => moveInk($('#tabs')));

async function renderTab() {
  const gen = ++S.renderGen;
  const panel = $('#panel');
  if (!S.st) { panel.replaceChildren(viewNoModel()); return; }
  const view = TABS.find((t) => t[0] === S.tab)[2];
  // A slow view gets a note instead of a frozen page.
  const slow = setTimeout(() => {
    if (gen === S.renderGen) panel.replaceChildren(el('p', { class: 'explain', text: 'computing…' }));
  }, 250);
  try {
    const nodes = await view();
    clearTimeout(slow);
    if (gen !== S.renderGen) return; // a newer render started meanwhile
    panel.replaceChildren(...[].concat(nodes).flat(Infinity).filter((c) => c != null && c !== false));
    if (S.enter) {
      // Fresh tab: fade its sections up and grow its bars, once.
      S.enter = false;
      panel.classList.remove('enter');
      void panel.offsetWidth;
      panel.classList.add('enter');
      clearTimeout(S.enterT);
      S.enterT = setTimeout(() => panel.classList.remove('enter'), 900);
    }
  } catch (e) {
    clearTimeout(slow);
    if (gen === S.renderGen) panel.replaceChildren(el('p', { class: 'explain', text: 'Error: ' + e.message }));
  }
}

const explain = (...parts) => el('p', { class: 'explain' }, ...parts);
const B = (t) => el('b', { text: t });

// ---------------------------------------------------------------- next token

function samplingDist(top, lse) {
  const s = samp(), cfg = S.st.sampler;
  // The engine penalizes only tokens with a letter in them (or non-ASCII text):
  // digits, operators and punctuation have to repeat in answers like "100 + 200 = 300".
  const wordy = (t) => /[A-Za-z]|[^\x00-\x7F]/.test(t.text);
  const recent = new Set(S.st.tokens.slice(-cfg.repeat_window).filter((t) => !t.special && wordy(t)).map((t) => t.id));
  // DRY: the engine reports, per token, the length of the repeat it would extend;
  // the penalty for the slider's strength is computed here, so dragging is live.
  const dryLen = new Map(S.st.next.dry || []);
  let cand = top.map(([id, text, l]) => {
    if (s.repeat_penalty !== 1 && recent.has(id)) l = l > 0 ? l / s.repeat_penalty : l * s.repeat_penalty;
    const len = dryLen.get(id);
    const dry = len && s.dry > 0 ? s.dry * cfg.dry_base ** (len - cfg.dry_allowed_len) : 0;
    return { id, text, raw: Math.exp(top.find((x) => x[0] === id)[2] - lse), l: l - dry, dry, len };
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
      el('span', {}, `${pct(d.raw)} → ${d.p > 0 ? pct(d.p) : '0'}`,
        d.dry ? el('span', { class: 'drytag', title: `this token would repeat a ${d.len}-token sequence that already appeared; DRY lowers its logit by ${d.dry.toFixed(2)}`, text: `↺ ${d.len} −${d.dry.toFixed(1)}` }) : null),
    ];
    row[0].onclick = () => act(async () => setState(await call('step', { n: 1, force: d.id, ...samp() }), 'last'));
    return row;
  });
  return [
    explain('The final layer produces one logit per vocabulary entry; softmax turns them into the ', B('model\'s distribution'), ' (blue). ',
      'Sampling settings then reshape it (orange): temperature divides the logits, top-k and top-p cut the tail, the repetition penalty pushes down recently used tokens, and ',
      B('DRY'), ' (↺) pushes down a token that would continue a sequence the text already contains, harder the longer the repeat: the loop-breaker a model this small badly needs. ',
      'Change them in the Sampler menu above and watch the orange bars move. Click a token to force it as the next token: an intervention on the sampler, not the model.'),
    el('div', { class: 'kpi' },
      el('div', {}, el('b', { text: JSON.stringify(toks[toks.length - 1].text) }), `last token (position ${toks.length - 1})`),
      el('div', {}, el('b', { text: pct(covered) }), 'of probability mass in the top 50 shown'),
      el('div', {}, el('b', { text: lse.toFixed(2) }), 'log Σ exp(logits)')),
    el('div', { class: 'bars' }, rows),
    el('p', { class: 'dim', style: 'font-size:12px', text: 'Rows are sorted by the adjusted score, so tokens hit by a penalty can sit below less likely ones. ↺ n −x: would extend an n-token repeat, logit lowered by x. Orange is renormalized over the top 50 tokens, a close approximation unless the temperature is very high.' }),
  ];
}

// ------------------------------------------------------------------ model download
//
// The page opens straight into the UI without fetching the weights. The first
// action that needs the engine calls ensureModel(): if the weights are already
// in the browser cache it just loads them, otherwise it asks first.

let READY = false;   // the engine is loaded
let MODEL_INFO = null; // worker's __status: {cached, cacheable, bytes, repo}
let loading = null;  // the in-flight ensureModel() promise
const mb = (b) => `${Math.round(b / 1e6)} MB`;

function overlay(which) {
  $('#loading').style.display = which ? 'flex' : 'none';
  $('#consent').hidden = which !== 'consent';
  $('#progress').hidden = which !== 'progress';
}

// Resolves true if the visitor agrees to the download.
function askDownload() {
  const i = MODEL_INFO, size = mb(i.bytes);
  $('#consent-size').textContent = size;
  $('#consent-yes').lastChild.textContent = `Download ${size}`;
  $('#consent-src').textContent = SOURCE === 'local' ? 'this dev server (localhost)' : 'Hugging Face';
  $('#consent-cache').textContent = i.cacheable
    ? "It's kept in your browser's cache, so later visits start in about a second."
    : "This browser won't let the page cache it (private window, or not HTTPS), so it downloads again on every visit.";
  overlay('consent');
  $('#consent-yes').focus();
  return new Promise((resolve) => {
    const done = (ok) => {
      overlay(null);
      $('#consent-yes').onclick = $('#consent-no').onclick = null;
      removeEventListener('keydown', onKey, true);
      resolve(ok);
    };
    const onKey = (e) => { if (e.key === 'Escape') { e.preventDefault(); done(false); } };
    $('#consent-yes').onclick = () => done(true);
    $('#consent-no').onclick = () => done(false);
    addEventListener('keydown', onKey, true);
  });
}

function showProgress(p) {
  const frac = p.total ? p.got / p.total : 0;
  $('#load-stage').textContent = p.stage + (p.cached ? ' (from browser cache)' : '');
  $('#load-bar').style.width = (100 * frac).toFixed(1) + '%';
  $('#load-detail').textContent = p.total > 1e5
    ? `${(p.got / 1e6).toFixed(0)} / ${(p.total / 1e6).toFixed(0)} MB`
    : p.total ? `${p.got} / ${p.total}` : '';
}

// Shows the error and waits: true = retry, false = give up.
function showLoadError(msg) {
  $('#load-stage').textContent = 'Could not load the model';
  $('#load-detail').textContent = msg;
  $('#load-bar').style.width = '0';
  $('#load-fail').hidden = false;
  return new Promise((resolve) => {
    const done = (retry) => { $('#load-fail').hidden = true; resolve(retry); };
    $('#load-retry').onclick = () => done(true);
    $('#load-close').onclick = () => done(false);
  });
}

// Load the engine, asking before any download. Resolves true when it's ready,
// false if the visitor said no (the action that asked is then dropped).
function ensureModel({ ask = true } = {}) {
  if (READY) return Promise.resolve(true);
  return (loading ??= (async () => {
    try {
      if (!MODEL_INFO.cached && ask && !(await askDownload())) return false;
      for (;;) {
        overlay('progress');
        showProgress({ stage: MODEL_INFO.cached ? 'reading the model from the browser cache' : 'starting', got: 0, total: 0 });
        try {
          await call('__load', { model: MODEL, source: SOURCE });
          break;
        } catch (e) {
          // A failed load leaves a fresh worker behind, so a retry starts clean.
          worker.terminate(); startWorker(showProgress);
          if (!(await showLoadError(e.message))) { overlay(null); return false; }
        }
      }
      overlay(null);
      await onModelReady();
      return true;
    } finally {
      loading = null;
    }
  })());
}

// The engine starts on its own default prompt; bring it in line with
// whatever the visitor typed before it was loaded.
async function onModelReady() {
  READY = true;
  MODEL_INFO.cached = true;
  renderModelChip();
  let st = await call('state');
  const text = $('#prompt').value;
  if (text && text !== st.tokens.map((t) => t.text).join('')) st = await call('run', { text });
  setState(st, 'last');
}

// Before the model is loaded, the header offers the download instead of stats.
function renderModelChip() {
  const box = $('#model-info');
  if (READY) { if (box.querySelector('.getmodel')) box.replaceChildren(); return; }
  box.replaceChildren(el('button', {
    class: 'getmodel', title: MODEL_INFO.cached ? 'load SmolLM2-135M from the browser cache' : `download SmolLM2-135M (${mb(MODEL_INFO.bytes)}) and run it in this tab`,
    onclick: () => ensureModel(),
  }, svgIcon('M8 2.5v8M4.5 7 8 10.5 11.5 7M3 13.5h10'), MODEL_INFO.cached ? 'Load model' : `Get the model · ${mb(MODEL_INFO.bytes)}`));
}

// What the completion view shows before there is any engine state.
function viewNoModel() {
  const dl = !MODEL_INFO || !MODEL_INFO.cached;
  return el('div', { class: 'nomodel' },
    el('div', { class: 'empty-orb', 'aria-hidden': 'true' }),
    el('h2', { text: 'The model isn’t loaded yet' }),
    el('p', { class: 'dim', text: dl
      ? `Run the prompt above (or press Step) and the page will offer to download SmolLM2-135M: ${mb(MODEL_INFO ? MODEL_INFO.bytes : 269060552)}, once, then cached in your browser. Every view below is computed from the real model in this tab.`
      : 'It’s in your browser cache already; run the prompt to load it.' }),
    el('button', { class: 'primary', onclick: () => $('#run').click() }, svgIcon('M4 2.5v11l9-5.5z'), dl ? 'Download & run the prompt' : 'Load & run the prompt'));
}

// ------------------------------------------------------------------ about the model
//
// A static description of SmolLM2-135M, from its config.json and
// model card, so it can be read before the weights are downloaded.

const CFG = {
  vocab: 49152, hidden: 576, layers: 30, heads: 9, kv_heads: 3, head_dim: 64, inter: 1536,
  ctx: 8192, rope_theta: 100000, merges: 48900, specials: 17,
};
const P = (() => {
  const { vocab, hidden: d, inter, kv_heads, head_dim, layers } = CFG, kv = kv_heads * head_dim;
  const embed = vocab * d, attn = 2 * d * d + 2 * d * kv, mlp = 3 * d * inter, norms = 2 * d;
  return { embed, attn, mlp, norms, layer: attn + mlp + norms, total: embed + layers * (attn + mlp + norms) + d };
})();
const M = (n) => (n >= 1e6 ? (n / 1e6).toFixed(n >= 1e8 ? 1 : 2) + 'M' : n >= 1e3 ? (n / 1e3).toFixed(n >= 1e4 ? 0 : 1) + 'k' : String(n));

// Close the dialog and show one of the completion tabs.
function gotoTab(id) {
  id = tabOf(id);
  $('#info').close();
  const b = document.querySelector(`#tabs button[data-tab="${id}"]`);
  if (b && !b.classList.contains('active')) b.click();
}

function buildInfo() {
  const c = CFG, kvBytes = c.layers * 2 * c.kv_heads * c.head_dim * 4;
  const see = (tab) => {
    const name = tabName(tab);
    return el('button', { class: 'see', type: 'button', onclick: () => gotoTab(tab), title: `close this and open the ${name} tab` }, name + ' →');
  };
  const stat = (v, label, hint) => el('div', { class: 'stat', title: hint || '' }, el('b', { text: v }), el('span', { text: label }));
  const shape = (s) => el('code', { class: 'shape', text: s });
  const box = (cls, name, what, sh, link) => el('div', { class: 'blk ' + cls }, el('div', {}, el('b', { text: name }), el('span', { class: 'dim', text: what })), el('div', { class: 'blkr' }, sh ? shape(sh) : null, link));
  const arrow = () => el('div', { class: 'arr', 'aria-hidden': 'true' });
  const feat = (title, ...body) => el('li', {}, el('b', { text: title }), ' ', ...body);
  const share = [
    ['MLP (SwiGLU)', c.layers * P.mlp, 'mlp'],
    ['token embedding (shared with the output head)', P.embed, 'emb'],
    ['attention (q, k, v, o)', c.layers * P.attn, 'att'],
    ['RMSNorm scales', c.layers * P.norms + c.hidden, 'nrm'],
  ];

  return [
    el('div', { class: 'info-head' },
      el('div', {},
        el('h2', { id: 'info-title' }, 'SmolLM2-', el('span', { class: 'grad', text: '135M' })),
        el('div', { class: 'dim' }, 'by Hugging Face · Apache 2.0 · Llama architecture · ',
          el('a', { href: 'https://huggingface.co/HuggingFaceTB/SmolLM2-135M', target: '_blank', rel: 'noopener', text: 'model card' }), ' · ',
          el('a', { href: 'https://arxiv.org/abs/2502.02737', target: '_blank', rel: 'noopener', text: 'paper' }))),
      el('button', { class: 'icon close', type: 'button', autofocus: true, 'aria-label': 'Close', title: 'Close (Esc)', onclick: () => $('#info').close() },
        svgIcon('M4 4l8 8M12 4l-8 8'))),

    el('div', { class: 'stats' },
      stat(`${(P.total / 1e6).toFixed(1)}M`, 'parameters', `${P.total.toLocaleString()} exactly`),
      stat(String(c.layers), 'layers'),
      stat(String(c.hidden), 'model width', 'd_model: the size of the residual stream vector at every position'),
      stat(`${c.heads} / ${c.kv_heads}`, 'query / KV heads', `${c.head_dim} dimensions per head`),
      stat(c.inter.toLocaleString(), 'MLP hidden'),
      stat(c.vocab.toLocaleString(), 'vocabulary'),
      stat(c.ctx.toLocaleString(), 'context (trained)'),
      stat('2T', 'training tokens')),

    el('p', { class: 'dim small gloss-hint' }, 'New to some of these words? ', el('button', { class: 'see', type: 'button', onclick: () => showInfoView('glossary') }, 'Open the glossary →')),
    el('h3', { text: 'Structure' }),
    el('p', { class: 'dim', text: `A decoder-only transformer. Every position carries one ${c.hidden}-number vector (the residual stream); each layer reads it through a norm and adds its result back. T is the number of tokens.` }),
    el('div', { class: 'arch' },
      box('io', 'Tokenizer', 'text → token ids (byte-level BPE)', '[T]', see('next')),
      arrow(),
      box('emb', 'Token embedding', `look up row id of a ${c.vocab.toLocaleString()} × ${c.hidden} table`, `[T, ${c.hidden}]`, see('walk')),
      arrow(),
      el('div', { class: 'loop' },
        el('div', { class: 'loop-tag', text: `× ${c.layers} layers` }),
        box('nrm', 'RMSNorm', `scale x to unit RMS, then by a learned ${c.hidden}-vector`, null),
        arrow(),
        box('att', 'Grouped-query self-attention', `q: ${c.heads} heads, k & v: ${c.kv_heads} heads, ${c.head_dim} dims each · RoPE on q and k · causal mask · softmax(q·kᵀ/√${c.head_dim})·v · output projection`, `[T, ${c.hidden}]`, see('walk')),
        el('div', { class: 'add', text: '+ add back into the residual stream' }),
        box('nrm', 'RMSNorm', 'a second, separate norm', null),
        arrow(),
        box('mlp', 'SwiGLU MLP', `down( silu(gate·x) ⊙ up·x ): ${c.hidden} → ${c.inter.toLocaleString()} → ${c.hidden}, per position`, `[T, ${c.hidden}]`, null),
        el('div', { class: 'add', text: '+ add back into the residual stream' })),
      arrow(),
      box('nrm', 'Final RMSNorm', '', `[T, ${c.hidden}]`, see('walk')),
      arrow(),
      box('emb', 'Output head (tied)', 'dot product with every row of the embedding table', `[T, ${c.vocab.toLocaleString()}]`, null),
      arrow(),
      box('io', 'Softmax + sampler', 'logits → probabilities → one token', '1 token', see('next'))),

    el('h3', { text: 'Where the parameters are' }),
    el('div', { class: 'share' }, share.map(([name, n, cls]) => el('div', { class: 'srow' },
      el('span', { text: name }),
      el('div', { class: 'sbar' }, el('i', { class: cls, style: `width:${Math.max(0.6, (100 * n) / P.total).toFixed(1)}%` })),
      el('span', { class: 'mono', text: `${M(n)} · ${((100 * n) / P.total).toFixed(n / P.total < 0.01 ? 2 : 0)}%` })))),
    el('p', { class: 'dim small', text: `Each layer holds ${M(P.layer)} parameters: ${M(P.attn)} in attention, ${M(P.mlp)} in the MLP. At this size the embedding table is a fifth of the model, which is why it's shared with the output head instead of stored twice.` }),

    el('h3', { text: 'Design choices' }),
    el('ul', { class: 'feats' },
      feat('Pre-norm with RMSNorm.', `Each sublayer normalizes its input (ε = 10⁻⁵) and adds its output back, so the residual stream is a running sum. No biases anywhere in the model.`),
      feat('Rotary position embeddings (RoPE).', `Position isn't added to the input; q and k are rotated by an angle that grows with position (base θ = ${c.rope_theta.toLocaleString()}, rotate-half layout), so q·k depends on the distance between tokens.`),
      feat('Grouped-query attention.', `${c.heads} query heads share ${c.kv_heads} key/value heads, three per group. The KV cache is 3× smaller: ${(kvBytes / 1e3).toFixed(0)} KB per token here (f32), ${(kvBytes / 2e3).toFixed(0)} KB in the original bf16.`),
      feat('SwiGLU MLP.', `Two parallel projections up to ${c.inter.toLocaleString()}; one goes through SiLU and gates the other element-wise, then a third projects back down. It holds most of the parameters.`),
      feat('Tied embeddings.', `The output head is the embedding table itself, which saves ${M(P.embed)} parameters.`),
      feat('Deep and narrow.', `${c.layers} layers of width ${c.hidden}: a deep, narrow shape, common for models under a billion parameters.`)),

    el('h3', { text: 'Tokenizer' }),
    el('ul', { class: 'feats' },
      feat('Byte-level BPE,', `${c.vocab.toLocaleString()} tokens built from ${c.merges.toLocaleString()} merges. Any byte sequence can be encoded, so there's no "unknown" token. Digits are always split one per token.`),
      feat(`${c.specials} special tokens`, 'at ids 0–16, mostly for code (<file_sep>, <jupyter_code>…). <|endoftext|> (0) separates documents: when the model produces it, it considers the text finished.')),

    el('h3', { text: 'Training' }),
    el('ul', { class: 'feats' },
      feat('Pretraining:', '2 trillion tokens of FineWeb-Edu, DCLM and The Stack, plus new filtered datasets; bfloat16, on 64 H100s with nanotron.'),
      feat('A base model:', 'it was only trained to continue text, so it continues whatever you write. It does not follow instructions or answer questions the way a chat model does; that takes further fine-tuning (SmolLM2-135M-Instruct is that version).'),
      feat('Limits:', 'mostly English. At 135M parameters it’s fluent but often wrong, and it loops easily without an anti-repetition sampler.')),

    el('h3', { text: 'In this engine' }),
    el('ul', { class: 'feats' },
      feat('Weights', `are stored in bfloat16 (${Math.round((P.total * 2) / 1e6)} MB) and converted to f32 on load (${Math.round((P.total * 4) / 1e6)} MB in memory).`),
      feat('Context:', `the page holds 512 positions (it records every activation at every position); the model itself was trained on ${c.ctx.toLocaleString()}.`),
      feat('Sampling default:', 'the Focused preset: the model’s own top token, with DRY 0.8 against loops. The presets next to the prompt switch to others.')),
  ];
}

// The glossary (glossary.js): grouped entries with a filter box.
function buildGlossary() {
  const list = el('div', { class: 'gloss' });
  const empty = el('p', { class: 'dim', hidden: true, text: 'No term matches.' });
  const items = GLOSSARY.map((g) => {
    const node = el('div', { class: 'gitem', id: 'g-' + g.t.toLowerCase().replace(/[^a-z0-9]+/g, '-') },
      el('div', { class: 'gterm' }, el('b', { text: g.t }), g.tab ? el('button', { class: 'see', type: 'button', onclick: () => gotoTab(g.tab), title: 'close this and open that tab' },
        'see it: ' + tabName(g.tab) + ' →') : null),
      el('p', { text: g.d }),
      el('div', { class: 'glinks' }, g.l.map(([label, href]) => el('a', { href, target: '_blank', rel: 'noopener' }, label, ' ↗'))));
    return { g, node, hay: [g.t, g.aka || '', g.d].join(' ').toLowerCase() };
  });
  const sections = GLOSSARY_GROUPS.map((name) => {
    const mine = items.filter((i) => i.g.g === name);
    return { mine, node: el('section', {}, el('h3', { text: name }), mine.map((i) => i.node)) };
  });
  list.append(...sections.map((s) => s.node));
  const search = el('input', { type: 'search', class: 'gsearch', placeholder: `Filter ${GLOSSARY.length} terms… (e.g. cache, softmax, rope)`, 'aria-label': 'Filter the glossary' });
  search.oninput = () => {
    const words = search.value.toLowerCase().split(/\s+/).filter(Boolean);
    let shown = 0;
    for (const i of items) { const ok = words.every((w) => i.hay.includes(w)); i.node.hidden = !ok; shown += ok; }
    for (const s of sections) s.node.hidden = s.mine.every((i) => i.node.hidden);
    empty.hidden = shown > 0;
  };
  return [
    el('p', { class: 'dim', text: 'The words this page uses, explained for this model, with links for digging deeper. Terms with a "see it" button can be watched live in one of the tabs.' }),
    search, empty, list,
  ];
}

// view: 'model' or 'glossary'
function openInfo(view = 'model') {
  const d = $('#info');
  if (!d.childElementCount) {
    const [head, ...model] = buildInfo();
    const tabs = el('div', { class: 'info-tabs', role: 'tablist' },
      el('button', { type: 'button', 'data-view': 'model', role: 'tab', text: 'The model' }),
      el('button', { type: 'button', 'data-view': 'glossary', role: 'tab', text: `Glossary (${GLOSSARY.length})` }));
    tabs.onclick = (e) => { const v = e.target.closest('button')?.dataset.view; if (v) showInfoView(v); };
    d.append(head, tabs, el('div', { id: 'info-model' }, model.flat(Infinity)), el('div', { id: 'info-gloss', hidden: true }, buildGlossary()));
    d.addEventListener('click', (e) => { if (e.target === d) d.close(); }); // click on the backdrop
  }
  hideTip();
  showInfoView(view);
  d.showModal();
}

function showInfoView(view) {
  const d = $('#info');
  $('#info-model').hidden = view !== 'model';
  $('#info-gloss').hidden = view !== 'glossary';
  d.querySelectorAll('.info-tabs button').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.view === view)));
  d.scrollTop = 0;
}
$('#info-btn').onclick = () => openInfo();

// A tooltip shouldn't outlive the mouse leaving its element by scrolling.
addEventListener('scroll', () => hideTip(), { passive: true });

// ------------------------------------------------------------------ wiring

// Each needs the engine; ensureModel() runs the typed prompt as part of loading,
// so a fresh Run doesn't need to run it again.
$('#run').onclick = () => withModel((fresh) => fresh || act(async () => setState(await call('run', { text: $('#prompt').value }), 'last')));
$('#step').onclick = () => withModel(() => act(async () => setState(await call('step', { n: 1, ...samp() }), 'last')));
$('#gen').onclick = () => withModel(() => act(async () => setState(await call('step', { n: 16, ...samp() }), 'last')));
async function withModel(fn) {
  if (READY) return fn(false);
  if (S.busy) return;
  if (await ensureModel()) return fn(true);
}
$('#prompt').addEventListener('keydown', (e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); $('#run').click(); } });
for (const id of ['#temp', '#topp', '#topk', '#rep', '#dry']) $(id).addEventListener('input', () => { if (S.tab === 'next') renderTab(); });
// The sampler settings live in a popover (#sampler-menu) opened by the
// Sampler button, which names the active preset.
{
  const p = presetPicker('', 'focused', (name) => ($('#sampler-name').textContent = ' · ' + name));
  const menu = $('#sampler-menu');
  menu.querySelector('.knobs').prepend(el('div', { class: 'knob preset' }, el('span', { text: 'preset' }), p.sel));
  menu.querySelector('.knobs').after(p.about);
  p.wire();
  // Popovers open in the top layer, centred; place this one under its button.
  menu.addEventListener('beforetoggle', (e) => {
    if (e.newState !== 'open') return;
    const b = $('#sampler-btn').getBoundingClientRect();
    menu.style.top = `${b.bottom + 8}px`;
    menu.style.left = `${Math.max(8, Math.min(b.right - 320, innerWidth - 328))}px`;
  });
  addEventListener('resize', () => menu.hidePopover());
  addEventListener('scroll', () => { if (menu.matches(':popover-open')) menu.hidePopover(); }, { passive: true });
}
document.addEventListener('keydown', (e) => {
  // defaultPrevented: the Step-by-step tab already used the arrow key.
  if (!S.st || e.defaultPrevented || e.target.closest?.('textarea, input, select')) return;
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
  document.querySelectorAll('input[type=range]').forEach(syncRange);
  try {
    if (NATIVE) {
      MODEL_INFO = { cached: true };
      await onModelReady();
      return;
    }
    startWorker(showProgress);
    MODEL_INFO = await call('__status', { model: MODEL });
    renderModelChip();
    renderTokens();
    renderTab();
    // Already downloaded on an earlier visit: load it straight away, no questions.
    if (MODEL_INFO.cached) await ensureModel({ ask: false });
  } catch (e) {
    flash(e.message);
    console.error(e);
  }
})();
