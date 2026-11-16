'use strict';
// The Flow tab: one position's trip through the whole network.
//
// Left, a map of the network for the selected position: every layer's
// attention pulling information in from earlier tokens, how much each layer
// writes, and how much each layer pushes the final answer. Right, one stage
// in full: every vector the forward pass computes for that position, with the
// real numbers, recomputed by the engine from the trace and the KV cache.
//
// Loaded before app.js; uses its helpers (el, S, cached, explain, …) at call time.

const FLOW = {
  stage: -1,       // -1 = embedding, 0..L-1 = layer, L = output
  dim: null,       // hovered vector cell {group, i}: highlighted in every vector of that group
  grids: [],       // drawn vector grids, for that highlight
  play: null,      // interval while playing through the stages
  animPos: null,   // the map animates when the position changes
  mapGen: 0,
};

// ------------------------------------------------------------------ small parts

const fv = (x) => {
  const a = Math.abs(x);
  return a === 0 ? '0' : a >= 1000 ? x.toFixed(0) : a >= 10 ? x.toFixed(1) : a >= 0.01 ? x.toFixed(3) : x.toExponential(1);
};
const vnorm = (v) => Math.sqrt(v.reduce((s, x) => s + x * x, 0));
const sgn = (x, f = fv) => (x >= 0 ? '+' : '−') + f(Math.abs(x));
// A share as a percentage, without "−0%".
const share = (x, d = 0) => { const v = 100 * x; return (Math.abs(v) < 0.5 * 10 ** -d ? 0 : v).toFixed(d) + '%'; };
const argmax = (v) => v.reduce((b, x, i) => (x > v[b] ? i : b), 0);
const tokName = (i) => `${JSON.stringify(S.st.tokens[i].text)} (${i})`;

// 98th percentile of |v|: one huge coordinate (they exist: see "massive
// activations") would otherwise turn the rest of the grid black.
function robustMax(v) {
  const a = Array.from(v, Math.abs).sort((x, y) => x - y);
  return Math.max(a[Math.floor((a.length - 1) * 0.98)], a[a.length - 1] * 0.02, 1e-6);
}

function setDim(d) {
  const old = FLOW.dim;
  FLOW.dim = d;
  FLOW.grids = FLOW.grids.filter((g) => g.c.isConnected);
  for (const g of FLOW.grids) if ((old && g.group === old.group) || (d && g.group === d.group)) g.draw();
}

// A vector as a grid of cells: orange positive, blue negative, outlined when
// beyond the colour scale. Hovering a cell highlights the same coordinate in
// every grid of the same group (e.g. every 576-dim residual-stream vector).
function vecGrid(v, o = {}) {
  const n = v.length;
  const cols = o.cols || (n === 1536 ? 64 : n === 576 ? 32 : Math.min(n, 64));
  const rows = Math.ceil(n / cols);
  const cell = o.cell || (n > 1000 ? 5 : n === 576 ? 7 : 7);
  const s = o.scale || robustMax(v);
  const W = cols * cell, H = rows * cell;
  const c = el('canvas');
  const g = hiDPI(c, W, H);
  const draw = () => {
    g.clearRect(0, 0, W, H);
    for (let i = 0; i < n; i++) {
      const x = (i % cols) * cell, y = Math.floor(i / cols) * cell;
      g.fillStyle = `rgb(${divRGB(v[i] / s).join(',')})`;
      g.fillRect(x, y, cell - 1, cell - 1);
      if (Math.abs(v[i]) > s * 1.001 && cell >= 5) {
        g.strokeStyle = 'rgba(255,255,255,.85)'; g.lineWidth = 1;
        g.strokeRect(x + 0.5, y + 0.5, cell - 2, cell - 2);
      }
    }
    const d = FLOW.dim;
    if (o.group && d && d.group === o.group && d.i < n) {
      g.strokeStyle = '#ffe066'; g.lineWidth = 2;
      g.strokeRect((d.i % cols) * cell - 1, Math.floor(d.i / cols) * cell - 1, cell + 1, cell + 1);
    }
  };
  draw();
  if (o.group) FLOW.grids.push({ group: o.group, draw, c });
  c.onmousemove = (e) => {
    const r = c.getBoundingClientRect();
    const cx = Math.floor((e.clientX - r.left) / cell), cy = Math.floor((e.clientY - r.top) / cell);
    const i = cy * cols + cx;
    if (cx < 0 || cx >= cols || i < 0 || i >= n) return hideTip();
    if (o.group && (!FLOW.dim || FLOW.dim.i !== i || FLOW.dim.group !== o.group)) setDim({ group: o.group, i });
    showTip(e, el('b', { text: `${o.label ? o.label(i) : 'dim ' + i} = ${fv(v[i])}` }), o.tip ? o.tip(i) : null,
      o.group ? el('div', { class: 'dim', text: 'the same coordinate is outlined in every vector of this size' }) : null);
  };
  c.onmouseleave = () => { hideTip(); if (o.group) setDim(null); };
  return c;
}

// A captioned vector.
function vecBlock(title, v, o = {}) {
  const s = o.scale || robustMax(v);
  const off = v.reduce((k, x) => k + (Math.abs(x) > s * 1.001 ? 1 : 0), 0);
  const big = Math.max(...v.map(Math.abs));
  return el('figure', { class: 'vec' },
    el('figcaption', {}, el('b', { text: title }), el('span', { class: 'dim', text: o.shape || `[${v.length}]` }),
      o.stat === false ? null : el('span', { class: 'dim', text: `‖·‖ ${fv(vnorm(v))}` })),
    vecGrid(v, { ...o, scale: s }),
    el('div', { class: 'vfoot', text: `colour ±${fv(s)}` + (off ? ` · ${off} outlined beyond it, up to ${fv(big)}` : '') }));
}

const headLabel = (hd) => (i) => `head ${Math.floor(i / hd)}, dim ${i % hd}`;

function fcard(n, title, formula, ...body) {
  return el('section', { class: 'fcard' },
    el('header', {}, el('span', { class: 'fnum', text: String(n) }), el('h4', { text: title }),
      formula ? el('code', { class: 'fformula', text: formula }) : null),
    ...body);
}
const fnote = (...parts) => el('p', { class: 'fnote' }, ...parts);
const frow = (...kids) => el('div', { class: 'frow' }, ...kids);
const fop = (t) => el('span', { class: 'fop', text: t });

function chips(list, fmt) {
  return el('div', { class: 'tchips' }, list.map(([id, text, v]) =>
    el('span', { class: 'tchip', title: `token id ${id}` }, el('span', { class: 'mono', text: vis(text) }), el('i', { text: fmt(v) }))));
}

// Direct logit attribution of one write: the tokens it pushes up and down.
function pushPull(d) {
  const f = (x) => sgn(x, (a) => a.toFixed(2));
  return el('div', { class: 'fpush' },
    el('div', {}, el('span', { class: 'dim', text: 'pushes up' }), chips(d.up, f)),
    el('div', {}, el('span', { class: 'dim', text: 'pushes down' }), chips(d.down, f)));
}

// x + Δ = x′, with norms.
function sumRow(a, b, c, names, group) {
  const o = { group, cell: 5 };
  return frow(vecBlock(names[0], a, o), fop('+'), vecBlock(names[1], b, o), fop('='), vecBlock(names[2], c, o));
}

function topDims(v, k = 3) {
  return v.map((x, i) => [i, x]).sort((p, q) => Math.abs(q[1]) - Math.abs(p[1])).slice(0, k)
    .map(([i, x]) => `dim ${i} (${fv(x)})`).join(', ');
}

// ------------------------------------------------------------------ the map

function flowMap(M) {
  const st = S.st, m = st.model, L = m.n_layers, pos = M.pos, n = pos + 1, toks = st.tokens;
  const R = L + 2; // rows: embedding, layers 0..L-1, output
  const W = 430, labW = 32, barW = 50, gap = 10, top = 20, rh = 15;
  const areaW = W - labW - 2 * (barW + gap);
  const cw = areaW / n;
  const labels = cw >= 10;
  const H = top + R * rh + (labels ? 62 : 16);
  const yc = (r) => top + (R - 1 - r) * rh + rh / 2;
  const xc = (j) => labW + (j + 0.5) * cw;
  const bx1 = labW + areaW + gap, bx2 = bx1 + barW + gap;
  const aw = st.attn_write.map((v) => v[pos]), mw = st.mlp_write.map((v) => v[pos]);
  const wmax = Math.sqrt(Math.max(...aw, ...mw, 1e-6));
  const att = M.attrib.map((a) => a[0]);
  const amax = Math.max(...att.map(Math.abs), 1e-6);
  const rel = (l) => aw[l] / Math.max(st.resid_norm[2 * l][pos], 1e-6);
  const c = el('canvas', { class: 'fmap' });
  const g = hiDPI(c, W, H);
  const font = (px) => `${px}px ui-monospace, Menlo, monospace`;

  const draw = (upto) => {
    g.clearRect(0, 0, W, H);
    g.font = font(9.5); g.textBaseline = 'middle';
    g.fillStyle = '#8b95ad'; g.textAlign = 'left';
    g.fillText('attention: where it reads from', labW, 8);
    g.textAlign = 'center';
    g.fillText('writes', bx1 + barW / 2, 8);
    g.fillText('→ answer', bx2 + barW / 2, 8);
    // selected stage
    const sr = FLOW.stage + 1;
    g.fillStyle = 'rgba(139,125,255,.18)';
    g.fillRect(0, yc(sr) - rh / 2, W, rh);
    // residual streams: one vertical line per position
    for (let j = 0; j < n; j++) {
      g.strokeStyle = j === pos ? 'rgba(227,232,244,.6)' : 'rgba(148,163,214,.16)';
      g.lineWidth = j === pos ? 2 : 1;
      g.beginPath(); g.moveTo(xc(j), yc(0)); g.lineTo(xc(j), yc(j === pos ? Math.min(upto, R - 1) : Math.min(upto, L))); g.stroke();
    }
    for (let r = 0; r < Math.min(upto + 1, R); r++) {
      const y = yc(r);
      g.fillStyle = r === sr ? '#e3e8f4' : '#5b6580';
      g.textAlign = 'right'; g.font = font(9);
      g.fillText(r === 0 ? 'emb' : r === R - 1 ? 'out' : 'L' + (r - 1), labW - 6, y);
      if (r === 0) {
        for (let j = 0; j < n; j++) {
          g.fillStyle = j === pos ? '#e3e8f4' : '#5b6580';
          g.beginPath(); g.arc(xc(j), y, j === pos ? 3 : Math.min(2, cw / 3), 0, 7); g.fill();
        }
        drawSigned(att[0], y, '#8b95ad');
        continue;
      }
      if (r === R - 1) {
        g.fillStyle = '#f0a050';
        g.beginPath(); g.arc(xc(pos), y, 4, 0, 7); g.fill();
        g.font = font(10); g.textAlign = 'right';
        g.fillText(`${clip(vis(M.answer[1]), 14)} ${pct(M.answer[3])}`, Math.max(xc(pos) - 8, labW + 90), y);
        g.textAlign = 'center'; g.fillStyle = '#8b95ad'; g.font = font(9);
        g.fillText(M.answer[2].toFixed(1), bx2 + barW / 2, y);
        continue;
      }
      const l = r - 1, src = M.src[l], tot = Math.max(aw[l], 1e-9);
      const alpha = Math.min(1, 0.3 + 2 * rel(l));
      for (let j = 0; j < n; j++) {
        const sh = src[j] / tot;
        if (j === pos || Math.abs(sh) < 0.04) continue;
        g.strokeStyle = sh > 0 ? `rgba(95,179,240,${alpha})` : `rgba(255,107,125,${alpha * 0.8})`;
        g.lineWidth = 0.6 + 5 * Math.min(1, Math.abs(sh));
        // an arch inside this layer's row: from the source's stream up and over to this position
        const x0 = xc(j), x1 = xc(pos);
        g.beginPath(); g.moveTo(x0, y + rh * 0.4);
        g.bezierCurveTo(x0 + (x1 - x0) * 0.15, y - rh * 0.55, x1 - (x1 - x0) * 0.15, y - rh * 0.55, x1, y);
        g.stroke();
      }
      const self = src[pos] / tot;
      g.fillStyle = `rgba(95,179,240,${alpha})`;
      g.beginPath(); g.arc(xc(pos), y, 1.8 + 3 * Math.min(1, Math.max(0, self)), 0, 7); g.fill();
      // write sizes: attention (top half), MLP (bottom half), √ scale
      g.fillStyle = '#5fb3f0'; g.fillRect(bx1, y - 5, (Math.sqrt(aw[l]) / wmax) * barW, 4);
      g.fillStyle = '#f0a050'; g.fillRect(bx1, y + 1, (Math.sqrt(mw[l]) / wmax) * barW, 4);
      drawSigned(att[1 + 2 * l], y - 3, '#5fb3f0');
      drawSigned(att[2 + 2 * l], y + 3, '#f0a050');
    }
    g.strokeStyle = 'rgba(148,163,214,.3)'; g.lineWidth = 1;
    g.beginPath(); g.moveTo(bx2 + barW / 2, top); g.lineTo(bx2 + barW / 2, top + R * rh); g.stroke();
    // tokens under their columns
    g.font = font(9.5);
    if (labels) {
      for (let j = 0; j < n; j++) {
        g.save(); g.translate(xc(j), top + R * rh + 4); g.rotate(-Math.PI / 2.6);
        g.textAlign = 'right'; g.fillStyle = j === pos ? '#e3e8f4' : '#8b95ad';
        g.fillText(clip(vis(toks[j].text), 9), 0, 0); g.restore();
      }
    } else {
      g.textAlign = 'right'; g.fillStyle = '#e3e8f4';
      g.fillText(`${clip(vis(toks[pos].text), 12)} ↑`, xc(pos) + 4, top + R * rh + 8);
    }
  };
  function drawSigned(a, y, color) {
    const w = (a / amax) * (barW / 2);
    g.fillStyle = color;
    g.fillRect(bx2 + barW / 2 + Math.min(0, w), y - 2, Math.abs(w), 4);
  }

  if (FLOW.animPos !== pos && !matchMedia('(prefers-reduced-motion: reduce)').matches) {
    FLOW.animPos = pos;
    const gen = ++FLOW.mapGen, t0 = performance.now();
    const step = (t) => {
      if (gen !== FLOW.mapGen) return; // a newer map replaced this one
      const k = Math.min(R, ((t - t0) / 900) * R);
      draw(Math.floor(k));
      if (k < R) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
    draw(0);
  } else {
    FLOW.mapGen++;
    draw(R);
  }

  const rowAt = (e) => {
    const r = c.getBoundingClientRect();
    const y = e.clientY - r.top;
    const row = R - 1 - Math.floor((y - top) / rh);
    return y < top || row < 0 || row >= R ? null : row;
  };
  c.onmousemove = (e) => {
    const r = rowAt(e);
    if (r == null) { c.style.cursor = ''; return hideTip(); }
    c.style.cursor = 'pointer';
    const ans = JSON.stringify(M.answer[1]);
    if (r === 0) {
      return showTip(e, el('b', { text: 'embedding' }),
        el('div', { text: `each token starts as one row of the embedding table` }),
        el('div', { text: `pushes the answer ${ans}: ${sgn(att[0], (x) => x.toFixed(2))}` }),
        el('div', { class: 'dim', text: 'click to open this stage' }));
    }
    if (r === R - 1) {
      return showTip(e, el('b', { text: 'output' }),
        el('div', { text: `predicts ${ans} with ${pct(M.answer[3])} (logit ${M.answer[2].toFixed(2)})` }),
        M.next && !M.same ? el('div', { text: `the actual next token ${JSON.stringify(M.next[1])}: ${pct(M.next[3])}` }) : null,
        el('div', { class: 'dim', text: 'click to open this stage' }));
    }
    const l = r - 1, tot = Math.max(aw[l], 1e-9);
    const srcs = M.src[l].map((v, j) => [j, v / tot]).sort((a, b) => b[1] - a[1]).slice(0, 4).filter(([, s]) => s > 0.01);
    showTip(e, el('b', { text: `layer ${l} at position ${pos}` }),
      el('div', { text: `attention writes ‖Δ‖ ${fv(aw[l])} (${(100 * rel(l)).toFixed(0)}% of ‖x‖), taken from:` }),
      srcs.map(([j, s]) => el('div', { class: 'mono', text: `  ${share(s).padStart(4)}  ${j === pos ? 'itself' : tokName(j)}` })),
      el('div', { text: `MLP writes ‖Δ‖ ${fv(mw[l])}` }),
      el('div', { text: `pushes the answer ${ans}: attention ${sgn(att[1 + 2 * l], (x) => x.toFixed(2))}, MLP ${sgn(att[2 + 2 * l], (x) => x.toFixed(2))}` }),
      el('div', { class: 'dim', text: 'click to open this layer' }));
  };
  c.onmouseleave = hideTip;
  c.onclick = (e) => {
    const r = rowAt(e);
    if (r != null) flowGo(r - 1);
  };
  return c;
}

function mapLegend(M) {
  const sw = (color, text) => el('span', {}, el('i', { style: `background:${color}` }), text);
  return el('div', { class: 'fmaplegend' },
    sw('#5fb3f0', 'attention'), sw('#f0a050', 'MLP'), sw('#ff6b7d', 'pulls the other way'),
    el('p', {}, 'Curves: where each layer\'s attention write at this position came from; thicker = bigger share. ',
      '"writes": ‖Δ‖ of each sublayer (√ scale). "→ answer": how much each write raises or lowers the logit of the final answer ',
      el('b', { class: 'mono', text: JSON.stringify(M.answer[1]) }), '; they add up to its logit ', M.answer[2].toFixed(1), '.'));
}

// ------------------------------------------------------------------ stage: embedding

function embedCards(D) {
  const [id, text] = D.token;
  return [
    fcard(1, 'A token', `position ${D.pos}`,
      frow(el('span', { class: 'bigtok mono', text: vis(text) }), el('span', { class: 'dim', text: `token id ${id}` })),
      fnote('The tokenizer has already cut the text into tokens. From here on the model never sees letters, only this id.')),
    fcard(2, 'Look up its row', `x = E[${id}]`,
      vecBlock('embedding', D.vec, { group: 'd', shape: '[576]' }),
      fnote('The embedding table E has one row of ', B('576'), ' numbers for each of the 49,152 tokens: 28.3M parameters, 21% of the model. ',
        'This row is the token\'s starting vector, the same wherever it appears: no position is added here. ',
        'Llama-style models only bring in position later, inside attention (RoPE).'),
      el('div', { class: 'fsub' }, el('span', { class: 'dim', text: 'closest rows in the table (cosine similarity): tokens the model learned to treat alike' }),
        chips(D.nearest, (v) => v.toFixed(2)))),
    fcard(3, 'Into the residual stream', 'x₀ = E[id]',
      fnote('This vector is where the residual stream of position ', String(D.pos), ' begins. No layer ever replaces it: each of the 30 layers reads it and ',
        B('adds'), ' something. ‖x‖ is ', fv(D.norm), ' now; watch it grow on the map.'),
      el('button', { class: 'primary fnext', onclick: () => flowGo(0) }, 'Into layer 0 ›')),
  ];
}

// ------------------------------------------------------------------ stage: one layer

function ropeDials(pos) {
  const m = S.st.model, half = m.head_dim / 2, cell = 23, cols = 16, rad = 8.5;
  const ang = (i) => pos * Math.pow(m.rope_theta, (-2 * i) / m.head_dim);
  const c = el('canvas');
  const g = hiDPI(c, cols * cell, Math.ceil(half / cols) * cell);
  for (let i = 0; i < half; i++) {
    const cx = (i % cols) * cell + cell / 2, cy = Math.floor(i / cols) * cell + cell / 2, a = ang(i);
    g.strokeStyle = '#2b3550'; g.lineWidth = 1.2;
    g.beginPath(); g.arc(cx, cy, rad, 0, 7); g.stroke();
    g.strokeStyle = '#8b7dff'; g.lineWidth = 2;
    g.beginPath(); g.moveTo(cx, cy); g.lineTo(cx + rad * Math.cos(-a), cy + rad * Math.sin(-a)); g.stroke();
  }
  c.onmousemove = (e) => {
    const r = c.getBoundingClientRect();
    const i = Math.floor((e.clientY - r.top) / cell) * cols + Math.floor((e.clientX - r.left) / cell);
    if (i < 0 || i >= half) return hideTip();
    const a = ang(i);
    showTip(e, el('b', { text: `pair (${i}, ${i + half})` }),
      el('div', { text: `turns ${Math.pow(m.rope_theta, (-2 * i) / m.head_dim).toPrecision(3)} rad per position` }),
      el('div', { text: `at position ${pos}: ${a.toPrecision(4)} rad = ${(a / (2 * Math.PI)).toPrecision(3)} turns` }));
  };
  c.onmouseleave = hideTip;
  return c;
}

function headMini(D, h) {
  const hs = D.heads[h], n = hs.probs.length, W = 120, H = 30;
  const c = el('canvas');
  const g = hiDPI(c, W, H);
  const bw = W / n;
  g.fillStyle = '#0a0d15'; g.fillRect(0, 0, W, H);
  hs.probs.forEach((p, j) => {
    const bh = Math.max(p > 0.004 ? 1 : 0, p * (H - 2));
    g.fillStyle = j === D.pos ? '#e3e8f4' : '#5fb3f0';
    g.fillRect(j * bw + (bw > 3 ? 0.5 : 0), H - bh, Math.max(1, bw - (bw > 3 ? 1 : 0)), bh);
  });
  const best = argmax(hs.probs);
  return el('button', {
    class: 'fhead' + (h === S.head ? ' sel' : '') + (hs.ablated ? ' ablated' : ''), type: 'button',
    title: `head ${h} (KV head ${hs.kv}): its attention over positions 0–${D.pos}. Click for the numbers`,
    onclick: () => { S.head = h; renderTab(); },
  }, el('span', { class: 'fh-t' }, `head ${h}`, el('i', { text: hs.ablated ? 'ablated' : `writes ${fv(hs.write_norm)}` })), c,
  el('span', { class: 'fh-b mono', text: `→ ${clip(vis(S.st.tokens[best].text), 9)} ${pct(hs.probs[best])}` }));
}

function headTable(D, h) {
  const hs = D.heads[h], toks = S.st.tokens, tot = Math.max(vnorm(D.d_attn), 1e-9);
  const rows = hs.probs.map((p, j) => [j, p]).sort((a, b) => b[1] - a[1]).slice(0, 8);
  return el('table', { class: 'ftable' },
    el('tr', {}, el('th', { text: 'key position' }), el('th', { text: 'score q·k/8' }), el('th', { text: 'softmax' }), el('th', {}), el('th', { text: 'share of Δ', title: 'this (head, source) term, projected on the direction of the whole attention write' })),
    rows.map(([j, p]) => el('tr', {},
      el('td', { class: 'mono', text: `${clip(vis(toks[j].text), 14)}  ${j === D.pos ? 'self' : j}` }),
      el('td', { text: hs.scores[j].toFixed(2) }),
      el('td', { text: pct(p) }),
      el('td', {}, el('div', { class: 'fbar', style: `width:${Math.max(1, p * 120)}px` })),
      el('td', { text: hs.ablated ? '—' : share(hs.src[j] / tot, 1) }))));
}

function srcBars(vals, tot, k = 8) {
  const toks = S.st.tokens;
  const items = vals.map((v, j) => [j, v]).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, k);
  const mx = Math.max(...items.map((x) => Math.abs(x[1])), 1e-9);
  return el('div', { class: 'fbars' }, items.map(([j, v]) => [
    el('span', { class: 'mono', text: clip(vis(toks[j].text), 14) }),
    el('span', { class: 'dim', text: j === vals.length - 1 ? 'self' : `pos ${j}` }),
    el('div', { class: 'fbar' + (v < 0 ? ' neg' : ''), style: `width:${Math.max(1, (Math.abs(v) / mx) * 150)}px` }),
    el('span', { text: share(v / tot) })]));
}

function headBars(D) {
  const mx = Math.max(...D.heads.map((h) => h.write_norm), 1e-9), tot = Math.max(vnorm(D.d_attn), 1e-9);
  return el('div', { class: 'fbars' }, D.heads.map((h, i) => [
    el('span', { class: 'mono', text: `head ${i}` }),
    el('span', { class: 'dim', text: h.ablated ? 'ablated' : share(h.proj / tot) }),
    el('div', { class: 'fbar', style: `width:${Math.max(1, (h.write_norm / mx) * 150)}px` }),
    el('span', { class: 'dim mono', text: h.dla.up.slice(0, 2).map((t) => vis(t[1])).join(' ') })]));
}

function siluPlot(gate, hidden) {
  const W = 250, H = 150, pad = 22;
  const lo = Math.max(-8, Math.min(...gate)), hi = Math.min(8, Math.max(...gate, 1));
  const silu = (x) => x / (1 + Math.exp(-x));
  const ylo = -0.4, yhi = Math.max(silu(hi), 1);
  const X = (x) => pad + ((x - lo) / (hi - lo)) * (W - pad - 6), Y = (y) => H - pad - ((y - ylo) / (yhi - ylo)) * (H - pad - 8);
  const c = el('canvas');
  const g = hiDPI(c, W, H);
  g.strokeStyle = 'rgba(148,163,214,.25)'; g.lineWidth = 1;
  g.beginPath(); g.moveTo(pad, Y(0)); g.lineTo(W, Y(0)); g.moveTo(X(0), 4); g.lineTo(X(0), H - pad); g.stroke();
  g.strokeStyle = '#8b7dff'; g.lineWidth = 1.5; g.beginPath();
  for (let i = 0; i <= 100; i++) { const x = lo + ((hi - lo) * i) / 100; i ? g.lineTo(X(x), Y(silu(x))) : g.moveTo(X(x), Y(silu(x))); }
  g.stroke();
  const hm = Math.max(...hidden.map(Math.abs), 1e-9);
  gate.forEach((x, i) => {
    if (x < lo || x > hi) return;
    const k = Math.abs(hidden[i]) / hm;
    g.fillStyle = `rgba(240,160,80,${0.15 + 0.85 * Math.sqrt(k)})`;
    g.fillRect(X(x) - 1.2, Y(silu(x)) - 1.2, 2.4, 2.4);
  });
  g.fillStyle = '#8b95ad'; g.font = '9.5px ui-monospace, Menlo, monospace'; g.textAlign = 'center';
  g.fillText('gate', W / 2, H - 6);
  g.fillText(String(Math.round(lo)), X(lo) + 6, H - 6); g.fillText(String(Math.round(hi)), X(hi) - 6, H - 6);
  g.textAlign = 'left'; g.fillText('silu(gate)', pad + 2, 10);
  return c;
}

function neuronTable(D) {
  const tot = Math.max(vnorm(D.d_mlp), 1e-9);
  return el('table', { class: 'ftable' },
    el('tr', {}, ['neuron', 'gate', 'silu', '× up', '= h', 'write', 'share of Δ', 'pushes up'].map((t) => el('th', { text: t }))),
    D.neurons.map((n) => el('tr', {},
      el('td', { text: `#${n.i}` }), el('td', { text: fv(n.gate) }), el('td', { text: fv(n.act) }), el('td', { text: fv(n.up) }),
      el('td', { text: fv(n.hidden) }), el('td', { text: fv(n.size) }), el('td', { text: share(n.proj / tot) }),
      el('td', { class: 'fl' }, chips(n.dla.up.slice(0, 3), (v) => '+' + v.toFixed(1))))));
}

function layerCards(D) {
  const m = S.st.model, l = D.layer, pos = D.pos, hd = m.head_dim, H = m.n_heads, group = H / m.n_kv_heads;
  const h = Math.min(S.head, H - 1), hs = D.heads[h], kv = hs.kv;
  const slice = (v, i) => v.slice(i * hd, (i + 1) * hd);
  const dA = vnorm(D.d_attn), dM = vnorm(D.d_mlp), xin = vnorm(D.x_in), xmid = vnorm(D.x_mid), xout = vnorm(D.x_out);
  const big = Math.max(...D.x_in.map(Math.abs));
  const ang0 = pos, angLast = pos * Math.pow(m.rope_theta, -(hd - 2) / hd);
  const groups = [];
  for (let k = 0; k < m.n_kv_heads; k++) {
    groups.push(el('div', { class: 'fgroup' }, el('span', { class: 'dim', text: `share K/V head ${k}` }),
      el('div', { class: 'fheads' }, Array.from({ length: group }, (_, i) => headMini(D, k * group + i)))));
  }
  const best = argmax(hs.probs);
  const active = D.active;
  return [
    fcard(1, 'Residual stream in', `x · [${m.hidden}]`,
      vecBlock('x', D.x_in, { group: 'd' }),
      fnote(l === 0 ? 'Entering the first layer, x is still just the token\'s embedding. ' : `Position ${pos}'s embedding plus everything layers 0–${l - 1} added to it. `,
        '‖x‖ = ', B(fv(xin)), '. Largest coordinates: ', topDims(D.x_in), '.',
        big > 200 ? ' One coordinate dwarfs the rest: a "massive activation", which models use as a fixed bias (it is what lets a token act as an attention sink).' : '')),

    fcard(2, 'RMSNorm', 'x̂ = x / rms(x) ⊙ g',
      frow(vecBlock('gain g (learned)', D.attn_norm_w, { group: 'd', stat: false }), fop('→'), vecBlock('x̂', D.xn, { group: 'd' })),
      fnote('rms(x) = ', B(fv(D.attn_rms)), ' (that is ‖x‖ / √576). Dividing by it gives a vector of RMS 1, so attention sees the ', B('direction'), ' of x, not its size; ',
        'then every coordinate is scaled by its own learned gain. The stream itself is left untouched: x̂ is only the attention\'s input.')),

    fcard(3, 'Queries, keys and values', 'q = Wq x̂ · k = Wk x̂ · v = Wv x̂',
      frow(vecBlock('q', D.q_pre, { cols: 64, cell: 6, shape: `[${H} heads × ${hd}]`, label: headLabel(hd), group: 'q' }),
        el('div', { class: 'fcol' },
          vecBlock('k', D.k_pre, { cols: 64, cell: 6, shape: `[${m.n_kv_heads} × ${hd}]`, label: headLabel(hd), group: 'k' }),
          vecBlock('v', D.v, { cols: 64, cell: 6, shape: `[${m.n_kv_heads} × ${hd}]`, label: headLabel(hd), group: 'v' }))),
      fnote('Three matrix products of the same x̂. Each of the ', String(H), ' query heads gets 64 numbers ("what am I looking for?"); keys ("what do I contain?") and values ("what do I hand over?") get only ',
        String(m.n_kv_heads), ' heads, each shared by ', String(group), ' query heads: grouped-query attention. ',
        'This position\'s k and v are written to the KV cache, where every later position will read them; q is used once, right now.')),

    fcard(4, 'RoPE: rotate by position', `angle = ${pos} × θ^(−2i/64)`,
      ropeDials(pos),
      frow(vecBlock(`q, head ${h}, before`, slice(D.q_pre, h), { cols: 32, cell: 7, group: 'qh' }), fop('↻'),
        vecBlock('after', slice(D.q, h), { cols: 32, cell: 7, group: 'qh' })),
      fnote('A head\'s 64 numbers form 32 pairs (i, i+32), and each pair is rotated like a clock hand: pair 0 by ', B(`${fv(ang0)} rad`),
        ', the last by only ', B(`${fv(angLast)} rad`), ' at this position. Keys get the same treatment, so in q·k the two rotations cancel except for the ',
        B('difference'), ' of the positions: attention sees relative distance.', pos === 0 ? ' At position 0 every angle is 0, so nothing changes.' : '')),

    fcard(5, 'Scores and softmax', 'p = softmax(q·k / √64)',
      el('div', { class: 'fgroups' }, groups),
      el('h5', { text: `head ${h}: what it compares` }),
      headTable(D, h),
      fnote('Each head scores its query against the key of every position 0…', String(pos), ' (later positions are hidden by the causal mask); softmax turns the scores into weights that add up to 1. ',
        'A difference of 1 in score is a factor of e ≈ 2.7 in weight. Head ', String(h), ' gives ', B(pct(hs.probs[best])), ' to ', B(tokName(best)), '.',
        hs.ablated ? ' This head is ablated: its output is zeroed.' : '')),

    fcard(6, 'Mix the values', 'out_h = Σ p_j · v_j',
      vecBlock('head outputs', D.heads.flatMap((x) => x.out), { cols: 64, cell: 6, shape: `[${H} × ${hd}]`, label: headLabel(hd), group: 'o' }),
      fnote('Each head outputs the attention-weighted average of the value vectors it looked at: 64 numbers of information copied from those positions. ',
        'This is the only step in the whole network where one position reads from another.')),

    fcard(7, 'Output projection: the attention write', 'Δ = Wo · [out_0 … out_8]',
      frow(el('div', {}, el('h5', { text: 'where Δ came from' }), srcBars(D.src_total, dA)),
        el('div', {}, el('h5', { text: 'which head wrote it' }), headBars(D))),
      vecBlock('Δ attention', D.d_attn, { group: 'd' }),
      fnote('Wo maps the 9 concatenated head outputs back to 576 dims. Because everything here is linear, Δ splits exactly into one term per (head, source position); ',
        'the percentages are each term projected on Δ\'s direction, so they add up to 100%. ',
        'Read directly as tokens (', B('direct logit attribution'), ': the change this write alone makes to the final logits):'),
      pushPull(D.attn_dla)),

    fcard(8, 'Add it to the stream', 'x ← x + Δ',
      sumRow(D.x_in, D.d_attn, D.x_mid, ['x', 'Δ attention', 'x'], 'd'),
      fnote('‖x‖ goes from ', fv(xin), ' to ', B(fv(xmid)), '. The write is ', B(`${((100 * dA) / Math.max(xin, 1e-9)).toFixed(0)}%`), ' the size of the stream it is added to.')),

    fcard(9, 'RMSNorm again', 'x̂ = x / rms(x) ⊙ g₂',
      vecBlock('x̂', D.xn2, { group: 'd' }),
      fnote('rms(x) = ', B(fv(D.mlp_rms)), '. Same operation, its own learned gain: the MLP gets a normalized copy of the stream.')),

    fcard(10, 'Up to 1,536 dimensions, twice', 'gate = W_gate x̂ · up = W_up x̂',
      frow(vecBlock('gate', D.gate, { group: 'f', label: (i) => `neuron ${i}` }), vecBlock('up', D.up, { group: 'f', label: (i) => `neuron ${i}` })),
      fnote('Two separate projections into the MLP\'s 1,536-dim hidden space. Each of the 1,536 positions is a ', B('neuron'), '.')),

    fcard(11, 'SwiGLU: the gate decides', 'h = silu(gate) × up',
      frow(siluPlot(D.gate, D.hidden), vecBlock('h', D.hidden, { group: 'f', label: (i) => `neuron ${i}` })),
      fnote('silu(x) = x·sigmoid(x) is close to 0 for negative inputs and close to x for positive ones, so it acts like a soft switch on each neuron; ',
        'multiplying by "up" lets the switch scale a separate signal. Dots: every neuron on the silu curve, brighter = larger h. Only ', B(String(active)),
        ' of 1,536 neurons have |h| > 0.1 here.')),

    fcard(12, 'The neurons that write the most', 'Δ = Σ h_i · W_down[:, i]',
      neuronTable(D),
      fnote('Each neuron writes a fixed direction (its column of W_down), scaled by h. "write" is the size of that, "share of Δ" its part of the MLP write (they add up to 100% over all 1,536), ',
        'and "pushes up" reads its direction as tokens.')),

    fcard(13, 'Down projection: the MLP write', 'Δ = W_down h',
      vecBlock('Δ MLP', D.d_mlp, { group: 'd' }),
      fnote('Back to 576 dims. Read directly as tokens:'),
      pushPull(D.mlp_dla)),

    fcard(14, 'Add it to the stream', 'x ← x + Δ',
      sumRow(D.x_mid, D.d_mlp, D.x_out, ['x', 'Δ MLP', 'x out'], 'd'),
      fnote('‖x‖ goes from ', fv(xmid), ' to ', B(fv(xout)), '. If the model stopped after this layer (the ', B('logit lens'), '), it would predict:'),
      chips(D.lens, (p) => pct(p)),
      el('div', { class: 'frow', style: 'margin-top:12px' },
        el('button', { class: 'primary fnext', onclick: () => flowGo(l + 1) }, l + 1 < m.n_layers ? `Into layer ${l + 1} ›` : 'To the output ›'),
        el('span', { class: 'vfoot', text: Math.max(...D.check) === 0 ? 'recomputed from the KV cache: matches the real forward pass exactly' : `recomputed from the KV cache: matches the real forward pass to within ${fv(Math.max(...D.check))}` }))),
  ];
}

// ------------------------------------------------------------------ stage: output

function attribChart(M) {
  const L = S.st.model.n_layers, W = 640, H = 170, pad = 28, top = 12;
  const series = [M.attrib.map((a) => a[0])];
  if (M.next && !M.same) series.push(M.attrib.map((a) => a[1]));
  const n = series[0].length, bw = (W - pad - 8) / n;
  const cums = series.map((s) => s.reduce((acc, x) => (acc.push((acc.at(-1) || 0) + x), acc), []));
  const all = [...series[0], ...cums.flat(), 0];
  const lo = Math.min(...all), hi = Math.max(...all);
  const Y = (v) => top + ((hi - v) / (hi - lo || 1)) * (H - top - 22);
  const c = el('canvas');
  const g = hiDPI(c, W, H);
  g.strokeStyle = 'rgba(148,163,214,.3)'; g.beginPath(); g.moveTo(pad, Y(0)); g.lineTo(W, Y(0)); g.stroke();
  series[0].forEach((v, i) => {
    g.fillStyle = i === 0 ? '#8b95ad' : i % 2 ? '#5fb3f0' : '#f0a050';
    g.fillRect(pad + i * bw + 0.5, Math.min(Y(0), Y(v)), Math.max(1, bw - 1), Math.abs(Y(v) - Y(0)));
  });
  cums.forEach((cs, k) => {
    g.strokeStyle = k ? '#5fd39a' : '#e3e8f4'; g.lineWidth = 1.5; g.setLineDash(k ? [4, 3] : []);
    g.beginPath(); cs.forEach((v, i) => (i ? g.lineTo(pad + (i + 0.5) * bw, Y(v)) : g.moveTo(pad + 0.5 * bw, Y(v)))); g.stroke();
  });
  g.setLineDash([]);
  g.fillStyle = '#8b95ad'; g.font = '9.5px ui-monospace, Menlo, monospace'; g.textAlign = 'center';
  for (let l = 0; l < L; l += 5) g.fillText('L' + l, pad + (1 + 2 * l + 1) * bw, H - 6);
  g.textAlign = 'right'; g.fillText(hi.toFixed(0), pad - 4, Y(hi) + 3); g.fillText(lo.toFixed(0), pad - 4, Y(lo));
  c.onmousemove = (e) => {
    const i = Math.floor((e.clientX - c.getBoundingClientRect().left - pad) / bw);
    if (i < 0 || i >= n) return hideTip();
    const name = i === 0 ? 'embedding' : `layer ${Math.floor((i - 1) / 2)} ${i % 2 ? 'attention' : 'MLP'}`;
    showTip(e, el('b', { text: name }),
      el('div', { text: `${JSON.stringify(M.answer[1])}: ${sgn(series[0][i], (x) => x.toFixed(2))} (running total ${cums[0][i].toFixed(2)})` }),
      series[1] ? el('div', { text: `${JSON.stringify(M.next[1])}: ${sgn(series[1][i], (x) => x.toFixed(2))} (running total ${cums[1][i].toFixed(2)})` }) : null);
  };
  c.onmouseleave = hideTip;
  return c;
}

function outCards(D, M) {
  const m = S.st.model;
  const mx = D.top[0][2], mn = Math.min(...D.top.map((t) => t[2]));
  const next = D.next;
  return [
    fcard(1, 'Residual stream out of the last layer', `x · [${m.hidden}]`,
      vecBlock('x', D.x, { group: 'd' }),
      fnote('The embedding plus the writes of all ', String(m.n_layers), ' attention and ', String(m.n_layers), ' MLP sublayers. ‖x‖ = ', B(fv(vnorm(D.x))), '.')),
    fcard(2, 'Final RMSNorm', 'x̂ = x / rms(x) ⊙ g',
      frow(vecBlock('gain g', D.norm_w, { group: 'd', stat: false }), fop('→'), vecBlock('x̂', D.xn, { group: 'd' })),
      fnote('rms(x) = ', B(fv(D.rms)), '. One last normalization, so the scores below don\'t depend on how big the stream grew.')),
    fcard(3, 'Score every token', 'logit_t = x̂ · E[t]',
      el('div', { class: 'fbars' }, D.top.map(([id, text, l, p]) => [
        el('span', { class: 'mono', text: clip(vis(text), 14), title: `token id ${id}` }),
        el('span', { class: 'dim', text: l.toFixed(2) }),
        el('div', { class: 'fbar' + (id === M.answer[0] ? ' mlp' : ''), style: `width:${Math.max(2, 10 + ((l - mn) / Math.max(mx - mn, 1e-9)) * 140)}px` }),
        el('span', { text: pct(p) })])),
      fnote('The model\'s embeddings are tied: the same table E that turned token ids into vectors now scores the final vector against all ', B('49,152'),
        ' rows, one dot product each. Softmax turns logits into probabilities: p = exp(logit − ', D.lse.toFixed(2), '). ',
        'Entropy ', B(`${D.entropy.toFixed(2)} bits`), ' (0 = certain).',
        next ? [' The token that actually came next, ', B(JSON.stringify(next.text)), `, got ${pct(next.p)}, rank ${next.rank + 1}.`] : ' This is the last position: the sampler draws the next token from these probabilities.')),
    fcard(4, 'Where the answer\'s logit came from', 'logit = Σ every write · E[t] (scaled)',
      el('div', { class: 'scroll' }, attribChart(M)),
      el('div', { class: 'fmaplegend' },
        el('span', {}, el('i', { style: 'background:#5fb3f0' }), 'attention'), el('span', {}, el('i', { style: 'background:#f0a050' }), 'MLP'),
        el('span', {}, el('i', { style: 'background:#e3e8f4' }), `running total, ${JSON.stringify(M.answer[1])}`),
        M.next && !M.same ? el('span', {}, el('i', { style: 'background:#5fd39a' }), `running total, ${JSON.stringify(M.next[1])} (actual next)`) : null),
      fnote('The final vector is a sum of writes, and with the final RMSNorm\'s divisor fixed at its real value the logit is linear in it, so it splits exactly into one part per write. ',
        'The bars are those parts for the top answer ', B(JSON.stringify(M.answer[1])), ', in the order the layers run; the line adds them up to its logit of ', B(M.answer[2].toFixed(2)), '. ',
        'Much of any logit is an offset shared by every token, which softmax ignores, so what matters is the gap between lines',
        M.next && !M.same ? ': compare it with the token that actually came next.' : '.')),
  ];
}

// ------------------------------------------------------------------ the tab

function flowGo(stage) {
  const L = S.st.model.n_layers;
  FLOW.stage = Math.max(-1, Math.min(L, stage));
  renderTab();
}

const PLAY_ICON = 'M4 2.5v11l9-5.5z', PAUSE_ICON = 'M4 3h3v10H4zM9 3h3v10H9z';

function stopPlay() {
  clearInterval(FLOW.play);
  FLOW.play = null;
  const b = document.querySelector('.fplay');
  if (b) { b.classList.remove('on'); b.replaceChildren(svgIcon(PLAY_ICON), 'Play'); }
}

function togglePlay() {
  if (FLOW.play) return stopPlay();
  const L = S.st.model.n_layers;
  if (FLOW.stage >= L) FLOW.stage = -2;
  FLOW.play = setInterval(() => {
    if (S.tab !== 'flow' || !S.st || FLOW.stage >= L) return stopPlay();
    flowGo(FLOW.stage + 1);
  }, 1600);
  flowGo(FLOW.stage + 1);
}

function stageKey(stage, pos) {
  const L = S.st.model.n_layers;
  return stage < 0 ? ['femb' + pos, 'flow_embed', { pos }]
    : stage >= L ? ['fout' + pos, 'flow_out', { pos }]
      : [`fl${stage}:${pos}`, 'flow_layer', { layer: stage, pos }];
}

async function viewFlow() {
  const st = S.st, m = st.model, L = m.n_layers, pos = S.sel;
  FLOW.grids = [];
  FLOW.stage = Math.max(-1, Math.min(L, FLOW.stage));
  const s = FLOW.stage;
  const M = await cached('fmap' + pos, 'flow_map', { pos });
  const D = await cached(...stageKey(s, pos));
  // Fetch the next stage in the background, so stepping forward is instant.
  if (s < L) {
    const [k, method, args] = stageKey(s + 1, pos), cache = S.cache;
    if (!(k in cache)) setTimeout(() => { if (!(k in cache)) call(method, args).then((r) => (cache[k] = r)).catch(() => {}); }, 50);
  }
  const name = s < 0 ? 'Embedding' : s >= L ? 'Output' : `Layer ${s}`;
  const sub = s < 0 ? 'token id → vector' : s >= L ? 'vector → next-token probabilities' : `${s + 1} of ${L}: attention, then MLP`;
  const cards = s < 0 ? embedCards(D) : s >= L ? outCards(D, M) : layerCards(D);
  return [
    explain('Follow position ', B(tokName(pos)), ' through the whole network (click another token above to switch). ',
      'The map shows every layer at once: what its attention gathered from earlier tokens, how much it wrote, and how it moved the final prediction. ',
      'Click a row to open that stage on the right, with every vector it computes and the real numbers. Hover any cell to outline the same coordinate in every vector of that size.'),
    el('div', { class: 'flow' },
      el('aside', { class: 'fmapbox' },
        el('div', { class: 'scroll' }, flowMap(M)),
        mapLegend(M)),
      el('div', { class: 'fwalk' },
        el('div', { class: 'fstage' },
          el('button', { type: 'button', class: 'icon', title: 'previous stage (←)', disabled: s < 0, onclick: () => { stopPlay(); flowGo(s - 1); } }, '‹'),
          el('div', { class: 'fstage-t' }, el('b', { text: name }), el('span', { class: 'dim', text: sub })),
          el('button', { type: 'button', class: 'icon', title: 'next stage (→)', disabled: s >= L, onclick: () => { stopPlay(); flowGo(s + 1); } }, '›'),
          el('button', { type: 'button', class: 'fplay' + (FLOW.play ? ' on' : ''), onclick: togglePlay, title: 'step through every stage, embedding to output' },
            svgIcon(FLOW.play ? PAUSE_ICON : PLAY_ICON), FLOW.play ? 'Pause' : 'Play')),
        cards)),
  ];
}

// A tooltip shouldn't outlive the mouse leaving its element by scrolling.
addEventListener('scroll', () => hideTip(), { passive: true });

document.addEventListener('keydown', (e) => {
  if (S.tab !== 'flow' || !S.st || e.altKey || e.ctrlKey || e.metaKey) return;
  if (e.target.closest('input, textarea, select, [contenteditable]') || document.querySelector('dialog[open]')) return;
  if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
    e.preventDefault();
    stopPlay();
    flowGo(FLOW.stage + (e.key === 'ArrowRight' ? 1 : -1));
  }
});
