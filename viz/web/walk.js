'use strict';
// The Step-by-step tab: the forward pass one operation at a time, on the whole
// prompt at once, the way the engine computes it. Every step is a matrix
// operation on [tokens, …] matrices (rows = tokens), with the checkpoint's
// tensor names, the shapes, and the real numbers. Next/Back (or ←/→) move one
// operation; one layer is walked in full, the other 29 repeat it.
//
// Loaded after flow.js and before app.js; uses their helpers at call time.

const WALK = { step: 'tok', layer: 0, info: false }; // info: the prose explanation is open (stays open across steps)

// ------------------------------------------------------------------ pieces

const wtok = (i) => S.st.tokens[i];
const wlabel = (i) => vis(wtok(i).text);
const shape = (...d) => `[${d.join(', ')}]`;
const dot64 = (a, ao, b, bo, n) => { let s = 0; for (let i = 0; i < n; i++) s += a[ao + i] * b[bo + i]; return s; };

// TeX → typeset maths (KaTeX, from the CDN). Without it, the TeX source itself.
function tex(src, display = false) {
  const span = el('span', { class: 'tex' });
  if (window.katex) katex.render(src, span, { displayMode: display, throwOnError: false });
  else { span.textContent = src; span.classList.add('mono'); }
  return span;
}

// A step's formula, typeset, with the shapes involved underneath.
function wformula(src, dims) {
  return el('div', { class: 'wformula' }, tex(src, true), dims ? el('div', { class: 'wdims mono', text: dims }) : null);
}

// A tensor from the checkpoint, by its real name.
function wweight(name, dims, note, sym) {
  return el('div', { class: 'wweight' }, sym ? [tex(sym), el('span', { class: 'dim', text: '=' })] : null, el('span', { class: 'mono', text: name }), el('span', { class: 'dim mono', text: shape(...dims) }),
    note ? el('span', { class: 'dim', text: note }) : null);
}

// A [rows, n] matrix as a heatmap: one row per token, one pixel column per
// dimension (half a pixel past 768), so the widths compare like the shapes.
// Orange positive, blue negative; hover for the number.
function mat(title, rows, o = {}) {
  const n = rows[0].length, R = rows.length;
  const cw = o.cw || Math.min(1, 768 / n), W = Math.max(8, Math.round(n * cw)), rh = 18;
  const top = o.split ? 14 : 0;
  const s = o.scale || robustMax(rows.flat());
  const c = el('canvas');
  const g = hiDPI(c, W, top + R * rh);
  rows.forEach((r, y) => {
    for (let i = 0; i < n; i++) {
      g.fillStyle = `rgb(${divRGB(r[i] / s).join(',')})`;
      g.fillRect(i * cw, top + y * rh, Math.max(cw, 0.75), rh - 2);
    }
  });
  if (o.split) {
    g.fillStyle = '#8b95ad'; g.font = '9.5px ui-monospace, Menlo, monospace'; g.textAlign = 'center';
    for (let h = 0; h * o.split < n; h++) {
      if (h) { g.fillStyle = '#090c14'; g.fillRect(h * o.split * cw - 1, top, 2, R * rh); g.fillStyle = '#8b95ad'; }
      g.fillText((o.splitName || 'h') + h, (h + 0.5) * o.split * cw, 10);
    }
  }
  c.onmousemove = (e) => {
    const b = c.getBoundingClientRect();
    const i = Math.floor((e.clientX - b.left) / cw), y = Math.floor((e.clientY - b.top - top) / rh);
    if (i < 0 || i >= n || y < 0 || y >= R) return hideTip();
    const lab = o.split ? `${o.splitName === 'kv' ? 'kv head' : 'head'} ${Math.floor(i / o.split)}, dim ${i % o.split} (column ${i})` : `column ${i}`;
    showTip(e, el('b', { text: `${o.rowName ? o.rowName(y) : JSON.stringify(wtok(y).text)} · ${lab}` }), el('div', { class: 'mono', text: fv(rows[y][i]) }));
  };
  c.onmouseleave = hideTip;
  const labels = el('div', { class: 'wlabels', style: `padding-top:${top}px` },
    rows.map((_, y) => el('span', { class: 'mono', text: o.rowName ? o.rowName(y) : wlabel(y) })));
  return el('figure', { class: 'wmat' },
    el('figcaption', {}, el('b', { text: title }), el('span', { class: 'dim mono', text: shape(R, n) }),
      o.cw === undefined && n > 768 ? el('span', { class: 'dim', text: 'drawn at half width' }) : null),
    el('div', { class: 'wgrid' }, labels, c),
    el('div', { class: 'vfoot', text: `colour ±${fv(s)}` }));
}

// The first few numbers of each row, and its length: the numbers behind the colours.
function nums(rows, o = {}) {
  const k = o.k || 6, names = o.rowName || ((y) => JSON.stringify(wtok(y).text));
  return el('table', { class: 'wnums' },
    el('tr', {}, el('th', { text: '' }), el('th', { text: o.head || `first ${k} of ${rows[0].length}` }), el('th', { text: '‖row‖' }), o.extra ? el('th', { text: o.extra[0] }) : null),
    rows.map((r, y) => el('tr', {},
      el('td', { class: 'mono', text: names(y) }),
      el('td', { class: 'mono', text: r.slice(o.from || 0, (o.from || 0) + k).map(fv).join(', ') + (r.length > k ? ', …' : '') }),
      el('td', { class: 'mono', text: fv(vnorm(r)) }),
      o.extra ? el('td', { class: 'mono', text: o.extra[1](y) }) : null)));
}

const wop = (t) => el('span', { class: 'wop', text: t });
const wrow = (...kids) => el('div', { class: 'wrow' }, ...kids);
const wnote = (...parts) => el('p', { class: 'wnote' }, ...parts);

// Per-head [T, T] grids. kind: 'score' (diverging, optional causal mask) or 'prob' (heat).
function attnGrids(mats, kind, o = {}) {
  const T = mats[0].length, cell = Math.max(10, Math.min(44, Math.floor(220 / T)));
  const kv = S.st.model.n_heads / S.st.model.n_kv_heads;
  const s = kind === 'score' ? robustMax(mats.flat(2).filter(Number.isFinite)) : 1;
  return el('div', { class: 'wheads' }, mats.map((M, h) => {
    const c = el('canvas');
    const g = hiDPI(c, T * cell, T * cell);
    for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) {
      const masked = o.mask && j > i, v = M[i][j];
      if (masked) {
        g.fillStyle = '#0d1220'; g.fillRect(j * cell, i * cell, cell - 1, cell - 1);
        g.strokeStyle = 'rgba(255,107,125,.35)'; g.beginPath(); g.moveTo(j * cell + 2, i * cell + cell - 3); g.lineTo(j * cell + cell - 3, i * cell + 2); g.stroke();
        if (cell >= 30 && kind === 'score') {
          g.fillStyle = '#ff6b7d'; g.font = '10.5px ui-monospace, Menlo, monospace'; g.textAlign = 'center'; g.textBaseline = 'middle';
          g.fillText('−∞', j * cell + cell / 2, i * cell + cell / 2);
        }
      } else {
        g.fillStyle = kind === 'prob' ? heat(v) : `rgb(${divRGB(v / s).join(',')})`;
        g.fillRect(j * cell, i * cell, cell - 1, cell - 1);
        // Few tokens: the cells are big enough to carry their numbers.
        if (cell >= 30) {
          g.fillStyle = kind === 'prob' ? heatText(v) : '#e3e8f4';
          g.font = '10.5px ui-monospace, Menlo, monospace'; g.textAlign = 'center'; g.textBaseline = 'middle';
          g.fillText(kind === 'prob' ? pct(v) : v.toFixed(1), j * cell + cell / 2, i * cell + cell / 2);
        }
      }
    }
    c.onmousemove = (e) => {
      const b = c.getBoundingClientRect();
      const j = Math.floor((e.clientX - b.left) / cell), i = Math.floor((e.clientY - b.top) / cell);
      if (i < 0 || j < 0 || i >= T || j >= T) return hideTip();
      const masked = o.mask && j > i;
      showTip(e, el('b', { text: `head ${h}: ${JSON.stringify(wtok(i).text)} (row ${i}) looks at ${JSON.stringify(wtok(j).text)} (column ${j})` }),
        el('div', { class: 'mono', text: masked ? 'masked: −∞ (a later token)' : kind === 'prob' ? `${pct(M[i][j])}` : `score ${fv(M[i][j])}` }));
    };
    c.onmouseleave = hideTip;
    return el('figure', { class: 'whead' }, el('figcaption', { class: 'dim', text: `head ${h} · kv ${Math.floor(h / kv)}` }), c);
  }));
}

// q·k/√d for every (row, column) pair, future included: the mask comes after.
function allScores(D) {
  const m = S.st.model, hd = m.head_dim, kv = m.n_heads / m.n_kv_heads, T = D.n, sc = 1 / Math.sqrt(hd);
  return Array.from({ length: m.n_heads }, (_, h) => Array.from({ length: T }, (_, i) => Array.from({ length: T }, (_, j) =>
    dot64(D.q[i], h * hd, D.k[j], Math.floor(h / kv) * hd, hd) * sc)));
}

// The engine's softmax rows, padded with zeros for the masked future.
function allProbs(D) {
  const m = S.st.model, T = D.n;
  return Array.from({ length: m.n_heads }, (_, h) => Array.from({ length: T }, (_, i) => Array.from({ length: T }, (_, j) => (j <= i ? D.probs[i][h][j] : 0))));
}

// ------------------------------------------------------------------ the steps

// Each step: where it belongs (part), a title, the formula with shapes, the
// tensors it uses, and what to look at. `D` is one layer for every position,
// `E` the embedding rows and the output end.
function wsteps(L) {
  const m = S.st.model, d = m.hidden, H = m.n_heads, KV = m.n_kv_heads, hd = m.head_dim, F = m.intermediate, V = m.vocab, NL = m.n_layers;
  const p = `model.layers.${L}.`;
  const steps = [
    { id: 'tok', part: 'Input', title: 'Text → tokens',
      body: (c) => [
        el('div', { class: 'wtoks' }, c.idx.map((i) => el('span', { class: 'wtok' }, el('span', { class: 'mono', text: wlabel(i) }), el('i', { text: String(wtok(i).id) })))),
        wnote('The tokenizer cuts the text into pieces from a fixed vocabulary of ', B(V.toLocaleString()), ' and replaces each with its id. ',
          'A space belongs to the start of the next word (shown as ·), so "·capital" and "capital" are different tokens with unrelated ids: the model has to learn that they are the same word. ',
          'From here on there is no text, only these ', B(String(c.T)), ' numbers.'),
      ] },
    { id: 'onehot', part: 'Input', title: 'Ids → one-hot rows', tex: String.raw`\text{onehot}_{ij} = \begin{cases} 1 & j = \text{id}_i \\ 0 & \text{otherwise} \end{cases}`, dims: `${shape(c0().T)} → ${shape(c0().T, V)}`,
      body: (c) => [
        oneHot(c),
        wnote('Conceptually each id becomes a row of ', B(V.toLocaleString()), ' zeros with a single 1 in column = id. ',
          'Engines never build this matrix: the next step shows why it would be wasted work.'),
      ] },
    { id: 'embed', part: 'Input', title: 'One-hot × embedding table', tex: String.raw`x = \text{onehot} \cdot E \qquad\Longleftrightarrow\qquad x_i = E_{\,\text{id}_i}`, dims: `${shape(c0().T, V)} × ${shape(V, d)} → ${shape(c0().T, d)}`,
      weights: [['model.embed_tokens.weight', [V, d], 'one row per vocabulary token', 'E']],
      body: (c) => [
        mat('x: the embeddings', c.E.embed),
        nums(c.E.embed),
        wnote('Multiplying a one-hot row by the table picks out one row of the table: row ', B(String(wtok(0).id)), ' for ', B(JSON.stringify(wtok(0).text)), ', and so on. ',
          'So the engine just copies rows. The result is the ', B('residual stream'), ': one ', B(String(d)), '-number row per token, which every layer reads and adds to. ',
          'Two things are missing so far: ', B('position'), ' (the same token gets the same row anywhere in the text) and ', B('context'), ' (each row knows only its own token).'),
      ] },
    { id: 'stream', part: 'Attention', title: `The stream entering layer ${L}`, only: () => L > 0, tex: String.raw`x^{(${L})} = x^{(0)} + \sum_{\ell=0}^{${L - 1}} \left( \Delta^{(\ell)}_{\text{attn}} + \Delta^{(\ell)}_{\text{mlp}} \right)`, dims: `${shape(c0().T, d)}`,
      body: (c) => [
        wrow(mat('embeddings', c.E.embed), wop('→'), mat(`x entering layer ${L}`, c.D.x_in)),
        nums(c.D.x_in, { extra: ['‖embedding‖', (y) => fv(vnorm(c.E.embed[y]))] }),
        wnote('Layers 0 to ', String(L - 1), ' have each added an attention write and an MLP write to every row. This layer starts from that sum. Same shape, much larger rows.'),
      ] },
    { id: 'rms', part: 'Attention', title: 'RMSNorm 1/3: measure each row', tex: String.raw`\operatorname{rms}(x_i) = \sqrt{\frac{1}{${d}} \sum_{j=1}^{${d}} x_{ij}^2 + \varepsilon}`, dims: `${shape(c0().T, d)} → ${shape(c0().T, 1)}: one number per row`,
      body: (c) => [
        mat('x', c.D.x_in),
        nums(c.D.x_in, { extra: ['rms', (y) => fv(c.D.attn_rms[y])] }),
        wnote('Before attention, each row is normalised on its own (the tokens still do not interact). First its size: the root of the mean of its ', String(d),
          ' squared values. ε = ', Number(c.D.eps).toExponential(0).replace('e-', 'e−'), ' only guards against dividing by zero.'),
      ] },
    { id: 'div', part: 'Attention', title: 'RMSNorm 2/3: divide by it', tex: String.raw`\hat{x}_i = \frac{x_i}{\operatorname{rms}(x_i)}`, dims: `${shape(c0().T, d)} ÷ ${shape(c0().T, 1)} → ${shape(c0().T, d)}`,
      body: (c) => {
        const xh = c.D.x_in.map((r, y) => r.map((v) => v / c.D.attn_rms[y]));
        return [mat('x̂ = x / rms', xh), nums(xh),
          wnote('Every row now has rms 1, so length √', String(d), ' = ', B(fv(Math.sqrt(d))), '. Each token keeps its direction and loses its scale, so no row can dominate just by being large.')];
      } },
    { id: 'gain', part: 'Attention', title: 'RMSNorm 3/3: × learned gain', tex: String.raw`\mathrm{xn}_{ij} = \hat{x}_{ij} \, g_j \qquad (\mathrm{xn} = \hat{x} \odot g)`, dims: `${shape(c0().T, d)} ⊙ ${shape(d)} → ${shape(c0().T, d)}`,
      weights: [[p + 'input_layernorm.weight', [d], 'element by element', 'g']],
      body: (c) => {
        const g = c.D.attn_norm_w;
        const big = g.map((v, i) => [Math.abs(v), i]).sort((a, b) => b[0] - a[0]).slice(0, 3);
        return [mat('g', [g], { rowName: () => 'g' }), mat('xn = x̂ ⊙ g', c.D.xn), nums(c.D.xn),
          wnote('Not a matrix multiply: each column i is scaled by g[i], the same for every token. A learned volume knob per dimension. ',
            'The loudest here are columns ', big.map(([v, i]) => `${i} (${fv(g[i])})`).join(', '), '. Negative entries flip a dimension. ',
            'This normalised copy feeds attention; the stream x itself is left untouched.')];
      } },
    { id: 'q', part: 'Attention', title: 'Queries', tex: String.raw`q = \mathrm{xn} \, W_Q^{\top}`, dims: `${shape(c0().T, d)} × ${shape(d, H * hd)} → ${shape(c0().T, H * hd)}`,
      weights: [[p + 'self_attn.q_proj.weight', [H * hd, d], 'stored [out, in], so it is applied transposed', 'W_Q']],
      body: (c) => [mat('q', c.D.q_pre), nums(c.D.q_pre),
        wnote('The first real matrix multiply. A query is what a token is looking for in earlier tokens. PyTorch stores weights as [out, in], hence the ᵀ: ',
          'the shared ', String(d), ' disappears, [a, b] × [b, c] = [a, c].')] },
    { id: 'k', part: 'Attention', title: 'Keys', tex: String.raw`k = \mathrm{xn} \, W_K^{\top}`, dims: `${shape(c0().T, d)} × ${shape(d, KV * hd)} → ${shape(c0().T, KV * hd)}`,
      weights: [[p + 'self_attn.k_proj.weight', [KV * hd, d], null, 'W_K']],
      body: (c) => [mat('k', c.D.k_pre), nums(c.D.k_pre),
        wnote('A key is what a token offers to tokens looking for something. Only ', String(KV * hd), ' wide, a third of q: see the heads step for why.')] },
    { id: 'v', part: 'Attention', title: 'Values', tex: String.raw`v = \mathrm{xn} \, W_V^{\top}`, dims: `${shape(c0().T, d)} × ${shape(d, KV * hd)} → ${shape(c0().T, KV * hd)}`,
      weights: [[p + 'self_attn.v_proj.weight', [KV * hd, d], null, 'W_V']],
      body: (c) => [mat('v', c.D.v), nums(c.D.v),
        wnote('A value is what a token hands over when someone attends to it. q and k are only ever compared with each other; v is what eventually gets written back into the stream.')] },
    { id: 'heads', part: 'Attention', title: 'Split into heads', tex: String.raw`q = \big[\, q^{(0)} \,\big|\, q^{(1)} \,\big|\, \cdots \,\big|\, q^{(${H - 1})} \,\big], \qquad \text{query head } h \text{ reads kv head } \lfloor h / ${H / KV} \rfloor`, dims: `q ${shape(c0().T, H * hd)} → ${shape(c0().T, H, hd)}    k, v ${shape(c0().T, KV * hd)} → ${shape(c0().T, KV, hd)}`,
      body: (c) => [mat('q', c.D.q_pre, { split: hd }), mat('k', c.D.k_pre, { split: hd, splitName: 'kv' }), mat('v', c.D.v, { split: hd, splitName: 'kv' }),
        wnote('Only a reshape, no number changes: every ', String(hd), ' columns are one head, and each head attends on its own in its ', String(hd), '-dimensional slice. ',
          B(`${H} query heads share ${KV} key/value heads`), `, ${H / KV} each (grouped-query attention): q heads 0–${H / KV - 1} use kv head 0, and so on. `,
          'That makes k and v, the part the KV cache stores, ', String(H / KV), '× smaller.')] },
    { id: 'rope', part: 'Attention', title: 'RoPE: rotate q and k by position', tex: String.raw`\begin{pmatrix} q'_{i} \\ q'_{i+${hd / 2}} \end{pmatrix} = \begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix} \begin{pmatrix} q_{i} \\ q_{i+${hd / 2}} \end{pmatrix}, \qquad \theta_i = ${m.rope_theta}^{-2i/${hd}}, \quad m = \text{position}`, dims: `the same for k; shapes unchanged: q ${shape(c0().T, H * hd)}, k ${shape(c0().T, KV * hd)}`,
      body: (c) => {
        const t = c.sel, rows = c.idx.map((y) => {
          const a = c.D.q_pre[y], b = c.D.q[y];
          return el('tr', {}, el('td', { class: 'mono', text: `${JSON.stringify(wtok(y).text)} @ ${y}` }),
            el('td', { class: 'mono', text: `(${fv(a[0])}, ${fv(a[hd / 2])}) → (${fv(b[0])}, ${fv(b[hd / 2])})` }),
            el('td', { class: 'mono', text: `${fv(y)} rad` }),
            el('td', { class: 'mono', text: `${fv(vnorm(a.slice(0, hd)))} → ${fv(vnorm(b.slice(0, hd)))}` }));
        });
        return [
          wrow(mat('q before', c.D.q_pre, { split: hd }), wop('→'), mat('q after', c.D.q, { split: hd })),
          wrow(mat('k before', c.D.k_pre, { split: hd, splitName: 'kv' }), wop('→'), mat('k after', c.D.k, { split: hd, splitName: 'kv' })),
          el('table', { class: 'wnums' }, el('tr', {}, el('th', { text: 'token @ position' }), el('th', { text: 'q head 0, pair (0, 32)' }), el('th', { text: 'angle' }), el('th', { text: 'length of q head 0' })), rows),
          el('h5', { text: `the ${hd / 2} rotation angles for ${JSON.stringify(wtok(t).text)} at position ${t} (pick another token above)` }),
          ropeDials(t),
          wnote('Position enters only here. Row 0 is not changed at all (angle 0). Fast pairs turn 1 radian per position, slow ones barely move: a clock with ', String(hd / 2), ' hands. ',
            'Rotations keep lengths, and the dot product of a query rotated to position m with a key rotated to n depends only on ', B('m − n'), ': attention sees relative distance. ',
            'v is not rotated, and the rotated k is what the KV cache keeps.'),
        ];
      } },
    { id: 'scores', part: 'Attention', title: 'Scores: every query against every key', tex: String.raw`S^{(h)} = \frac{q^{(h)} \, \big(k^{(\lfloor h/${H / KV} \rfloor)}\big)^{\top}}{\sqrt{${hd}}}`, dims: `${shape(c0().T, hd)} × ${shape(hd, c0().T)} → ${shape(c0().T, c0().T)}, for each of the ${H} heads`,
      body: (c) => [attnGrids(allScores(c.D), 'score'),
        wnote('For each head, row i column j is how well token i\'s query matches token j\'s key: a dot product, divided by √', String(hd), ' = ', String(Math.sqrt(hd)),
          ' to keep the numbers in a range softmax handles well. The first time the tokens meet. ',
          'Mathematically every pair has a score, including the upper triangle, where a token would look at tokens that come ', B('after'), ' it.')] },
    { id: 'mask', part: 'Attention', title: 'Causal mask', tex: String.raw`S_{ij} = -\infty \quad \text{for } j > i`, dims: `the upper triangle of every ${shape(c0().T, c0().T)} grid`,
      body: (c) => [attnGrids(allScores(c.D), 'score', { mask: true }),
        wnote('A token may not see the future: when the model generates, the later tokens do not exist yet, so it must learn without them. ',
          'Setting those scores to −∞ makes their softmax weight exactly 0. Row 0 can now only look at itself.')] },
    { id: 'softmax', part: 'Attention', title: 'Softmax: scores → attention weights', tex: String.raw`P_{ij} = \frac{e^{S_{ij}}}{\sum_{j' \le i} e^{S_{ij'}}}`, dims: `each row of each ${shape(c0().T, c0().T)} grid sums to 1`,
      body: (c) => {
        const P = allProbs(c.D), t = c.sel;
        return [attnGrids(P, 'prob', { mask: true }),
          el('table', { class: 'wnums' }, el('tr', {}, el('th', { text: `row ${t} (${JSON.stringify(wtok(t).text)}): where each head looks` }), el('th', { text: 'top sources' })),
            P.map((M, h) => el('tr', {}, el('td', { class: 'mono', text: `head ${h}` }),
              el('td', { class: 'mono', text: M[t].map((v, j) => [v, j]).filter(([, j]) => j <= t).sort((a, b) => b[0] - a[0]).slice(0, 3).map(([v, j]) => `${JSON.stringify(wtok(j).text)} ${pct(v)}`).join(' · ') })))),
          wnote('Each row becomes a probability distribution over the tokens it may see. Brighter = more attention. Row 0 is always 100% on itself. ',
            'Many heads put most of their weight on token 0 even when it is not relevant: an "attention sink", a place to park attention when there is nothing useful to read.')];
      } },
    { id: 'mix', part: 'Attention', title: 'Weighted sum of values', tex: String.raw`o^{(h)} = P^{(h)} \, v^{(\lfloor h/${H / KV} \rfloor)}, \qquad o = \big[\, o^{(0)} \,\big|\, \cdots \,\big|\, o^{(${H - 1})} \,\big]`, dims: `${shape(c0().T, c0().T)} × ${shape(c0().T, hd)} → ${shape(c0().T, hd)} per head; side by side → ${shape(c0().T, H * hd)}`,
      body: (c) => [mat('heads\' outputs, concatenated', c.D.head_out, { split: hd }), nums(c.D.head_out),
        wnote('Each head\'s output for token i is the average of the value vectors it attended to, weighted by those probabilities. ',
          'This is where information actually moves between tokens. Row 0 can only contain its own value. The ', String(H), ' heads\' ', String(hd), '-wide outputs are laid side by side again.')] },
    { id: 'o', part: 'Attention', title: 'Output projection', tex: String.raw`\Delta_{\text{attn}} = o \, W_O^{\top}`, dims: `${shape(c0().T, H * hd)} × ${shape(H * hd, d)} → ${shape(c0().T, d)}`,
      weights: [[p + 'self_attn.o_proj.weight', [d, H * hd], null, 'W_O']],
      body: (c) => [mat('Δ attention', c.D.d_attn), nums(c.D.d_attn),
        wnote('One more matrix multiply mixes the heads together and maps the result back into the stream\'s ', String(d), ' dimensions: attention\'s write.')] },
    { id: 'add1', part: 'Attention', title: 'Residual add', tex: String.raw`x' = x + \Delta_{\text{attn}}`, dims: `${shape(c0().T, d)} + ${shape(c0().T, d)} → ${shape(c0().T, d)}`,
      body: (c) => [wrow(mat('x', c.D.x_in), wop('+'), mat('Δ attention', c.D.d_attn), wop('='), mat('x′', c.D.x_mid)),
        nums(c.D.x_mid, { extra: ['‖x‖ → ‖Δ‖', (y) => `${fv(vnorm(c.D.x_in[y]))} + ${fv(vnorm(c.D.d_attn[y]))}`] }),
        wnote('Attention\'s write is ', B('added'), ' to the stream; nothing is overwritten. That is why it is called the residual stream: every layer only adds a correction to what is already there.')] },
    { id: 'norm2', part: 'MLP', title: 'RMSNorm again, before the MLP', tex: String.raw`\mathrm{xn}_2 = \frac{x'}{\operatorname{rms}(x')} \odot g_2`, dims: `${shape(c0().T, d)}`,
      weights: [[p + 'post_attention_layernorm.weight', [d], null, 'g_2']],
      body: (c) => [mat('g₂', [c.D.mlp_norm_w], { rowName: () => 'g₂' }), mat('xn₂', c.D.xn2), nums(c.D.xn2, { extra: ['rms', (y) => fv(c.D.mlp_rms[y])] }),
        wnote('The same three steps as before attention (measure, divide, × gain), with this block\'s own gain. The MLP gets a normalised copy; x′ stays as it is.')] },
    { id: 'gateup', part: 'MLP', title: 'Gate and up projections', tex: String.raw`\text{gate} = \mathrm{xn}_2 \, W_{\text{gate}}^{\top}, \qquad \text{up} = \mathrm{xn}_2 \, W_{\text{up}}^{\top}`, dims: `${shape(c0().T, d)} × ${shape(d, F)} → ${shape(c0().T, F)}, twice`,
      weights: [[p + 'mlp.gate_proj.weight', [F, d], null, 'W_{\\text{gate}}'], [p + 'mlp.up_proj.weight', [F, d], null, 'W_{\\text{up}}']],
      body: (c) => [mat('gate', c.D.gate), mat('up', c.D.up), nums(c.D.gate, { head: 'gate, first 6' }),
        wnote('The MLP widens each row to ', String(F), ' "neurons", twice, with two different matrices. Still row by row: the MLP never mixes tokens, it works on what attention gathered into each row.')] },
    { id: 'silu', part: 'MLP', title: 'SiLU on the gate', tex: String.raw`\operatorname{silu}(z) = z \, \sigma(z) = \frac{z}{1 + e^{-z}}`, dims: `applied to each of the ${shape(c0().T, F)} numbers`,
      body: (c) => [wrow(siluPlot(c.D.gate[c.sel], c.D.hidden[c.sel]), mat('act', c.D.act)),
        wnote('The only nonlinearity in the layer. Positive gates pass almost unchanged, negative ones are squashed towards 0 (down to −0.28). ',
          'The dots are ', B(JSON.stringify(wtok(c.sel).text)), '\'s ', String(F), ' neurons; brighter ones end up writing more.')] },
    { id: 'mul', part: 'MLP', title: 'Gate × up', tex: String.raw`\text{hidden} = \operatorname{silu}(\text{gate}) \odot \text{up}`, dims: `${shape(c0().T, F)} ⊙ ${shape(c0().T, F)} → ${shape(c0().T, F)}`,
      body: (c) => [mat('hidden', c.D.hidden), nums(c.D.hidden, { extra: ['|h| > 0.1', (y) => String(c.D.hidden[y].filter((v) => Math.abs(v) > 0.1).length)] }),
        wnote('Element by element: the gate decides how much of each "up" value gets through (this pairing is SwiGLU). Most neurons end up near 0; a few carry the signal.')] },
    { id: 'down', part: 'MLP', title: 'Down projection', tex: String.raw`\Delta_{\text{mlp}} = \text{hidden} \, W_{\text{down}}^{\top}`, dims: `${shape(c0().T, F)} × ${shape(F, d)} → ${shape(c0().T, d)}`,
      weights: [[p + 'mlp.down_proj.weight', [d, F], null, 'W_{\\text{down}}']],
      body: (c) => [mat('Δ MLP', c.D.d_mlp), nums(c.D.d_mlp),
        wnote('Back to ', String(d), ' dimensions: each neuron writes its own column of down_proj, scaled by its activation, and they all add up.')] },
    { id: 'add2', part: 'MLP', title: 'Residual add', tex: String.raw`x'' = x' + \Delta_{\text{mlp}}`, dims: `${shape(c0().T, d)} + ${shape(c0().T, d)} → ${shape(c0().T, d)}`,
      body: (c) => [wrow(mat('x′', c.D.x_mid), wop('+'), mat('Δ MLP', c.D.d_mlp), wop('='), mat(`x″ (layer ${L + 1 < NL ? L + 1 : 'output'} input)`, c.D.x_out)),
        nums(c.D.x_out, { extra: ['‖x′‖ → ‖Δ‖', (y) => `${fv(vnorm(c.D.x_mid[y]))} + ${fv(vnorm(c.D.d_mlp[y]))}`] }),
        wnote(`Layer ${L} is done: x″ is what layer ${L + 1 < NL ? L + 1 : '(none: this was the last)'} receives. Same shape it started with.`)] },
    { id: 'repeat', part: 'Output', title: L + 1 < NL ? `Layers ${L + 1} … ${NL - 1}: the same again` : 'All layers done',
      body: () => [
        wnote(L + 1 < NL
          ? ['Every layer repeats exactly these steps with its own weights (', String(NL), ' layers, ', B('9 tensors each'), '). Each reads the stream and adds an attention write and an MLP write. ',
            'Walk the next layer in full, or jump to the end, where the last row of the stream becomes the prediction.']
          : ['That was the last of the ', String(NL), ' layers. Next: the stream becomes a prediction.']),
        el('div', { class: 'wjump' },
          L + 1 < NL ? el('button', { type: 'button', onclick: () => { WALK.layer = L + 1; WALK.step = 'stream'; renderTab(); } }, `Walk layer ${L + 1} →`) : null,
          L + 1 < NL ? el('button', { type: 'button', onclick: () => { WALK.layer = NL - 1; WALK.step = 'stream'; renderTab(); } }, `Walk the last layer (${NL - 1}) →`) : null,
          el('button', { type: 'button', class: 'primary', onclick: () => { WALK.step = 'fnorm'; renderTab(); } }, 'Skip to the output →')),
      ] },
    { id: 'fnorm', part: 'Output', title: 'Final RMSNorm', tex: String.raw`x_f = \frac{x}{\operatorname{rms}(x)} \odot g_f`, dims: `${shape(c0().T, d)}`,
      weights: [['model.norm.weight', [d], null, 'g_f']],
      body: (c) => [wrow(mat(`x after all ${NL} layers`, c.E.out.map((o) => o.x)), wop('→'), mat('xf', c.E.out.map((o) => o.xn))),
        nums(c.E.out.map((o) => o.x), { extra: ['rms', (y) => fv(c.E.out[y].rms)], head: 'x, first 6' }),
        wnote('One last normalisation of the stream after layer ', String(NL - 1), '. Compare the row lengths with the embeddings: the layers made them much larger.')] },
    { id: 'logits', part: 'Output', title: 'Logits: compare with every token', tex: String.raw`\text{logits} = x_f \, E^{\top}`, dims: `${shape(c0().T, d)} × ${shape(d, V)} → ${shape(c0().T, V)}`,
      weights: [['model.embed_tokens.weight', [V, d], 'the same table as the input ("tied" embeddings)', 'E']],
      body: (c) => [
        el('table', { class: 'wnums' }, el('tr', {}, el('th', { text: 'row' }), el('th', { text: `top 5 of ${V.toLocaleString()} logits` })),
          c.idx.map((y) => el('tr', {}, el('td', { class: 'mono', text: JSON.stringify(wtok(y).text) }),
            el('td', { class: 'mono', text: c.E.out[y].top.map(([, t, l]) => `${vis(t)} ${fv(l)}`).join(' · ') })))),
        wnote('Each row is compared, by dot product, with all ', V.toLocaleString(), ' rows of the embedding table: the very table that turned ids into vectors at the start. ',
          'A high logit means the row points the same way as that token\'s embedding.')] },
    { id: 'probs', part: 'Output', title: 'Softmax: logits → next-token probabilities', tex: String.raw`p_{iv} = \frac{e^{\,\text{logits}_{iv}}}{\sum_{u=1}^{${V}} e^{\,\text{logits}_{iu}}}`, dims: `${shape(c0().T, V)}, each row sums to 1`,
      body: (c) => [
        el('table', { class: 'wnums' }, el('tr', {}, el('th', { text: 'after' }), el('th', { text: 'predicts' }), el('th', { text: 'actually next' })),
          c.idx.map((y) => {
            const top = c.E.out[y].top, nxt = y + 1 < S.st.tokens.length ? wtok(y + 1) : null;
            return el('tr', { class: y === c.T - 1 ? 'wlast' : '' },
              el('td', { class: 'mono', text: JSON.stringify(wtok(y).text) }),
              el('td', { class: 'mono', text: top.slice(0, 3).map(([, t, , pr]) => `${vis(t)} ${pct(pr)}`).join(' · ') }),
              el('td', { class: 'mono', text: nxt ? `${vis(nxt.text)}${nxt.id === top[0][0] ? '  ✓ top guess' : ''}` : 'not written yet' }));
          })),
        wnote('Every row predicts the token after it, so one pass over a prompt checks ', String(c.T), ' predictions at once (that is how the model was trained). ',
          'Only the ', B('last row'), ' predicts something new: the sampler picks from it, the pick is appended, and the next pass needs only that one new row, because every earlier k and v is already in the KV cache.')] },
  ];
  return steps.filter((s) => !s.only || s.only());
}

// Shapes in formulas need T before the data is fetched.
function c0() { return { T: Math.min(16, S.st.tokens.length) }; }

function oneHot(c) {
  const V = S.st.model.vocab, W = 768, rh = 18;
  const cv = el('canvas');
  const g = hiDPI(cv, W, c.T * rh);
  c.idx.forEach((y) => {
    g.fillStyle = '#181b21'; g.fillRect(0, y * rh, W, rh - 2);
    g.fillStyle = '#f5963c'; g.fillRect(Math.floor((wtok(y).id / V) * W) - 1, y * rh, 3, rh - 2);
  });
  cv.onmousemove = (e) => {
    const b = cv.getBoundingClientRect(), y = Math.floor((e.clientY - b.top) / rh);
    if (y < 0 || y >= c.T) return hideTip();
    showTip(e, el('b', { text: `${JSON.stringify(wtok(y).text)}: column ${wtok(y).id} = 1` }), el('div', { class: 'dim', text: `the other ${(V - 1).toLocaleString()} columns are 0` }));
  };
  cv.onmouseleave = hideTip;
  return el('figure', { class: 'wmat' },
    el('figcaption', {}, el('b', { text: 'one-hot' }), el('span', { class: 'dim mono', text: shape(c.T, V) }), el('span', { class: 'dim', text: `drawn at 1/${Math.round(V / W)} width` })),
    el('div', { class: 'wgrid' }, el('div', { class: 'wlabels' }, c.idx.map((y) => el('span', { class: 'mono', text: wlabel(y) }))), cv));
}

// ------------------------------------------------------------------ the view

function walkGo(id) { WALK.step = id; renderTab(); }

const INFO_ICON = 'M8 1.5a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13zm-.8 5.6h1.6V12H7.2zM8 3.8a1 1 0 1 1 0 2 1 1 0 0 1 0-2z';

async function viewWalk() {
  const st = S.st, m = st.model, T = Math.min(16, st.tokens.length), L = Math.max(0, Math.min(m.n_layers - 1, WALK.layer));
  WALK.layer = L;
  const steps = wsteps(L);
  let k = steps.findIndex((s) => s.id === WALK.step);
  if (k < 0) { k = 0; WALK.step = steps[0].id; }
  const s = steps[k];
  const layerStep = s.part === 'Attention' || s.part === 'MLP';
  const E = await cached('walkends', 'walk_ends', null);
  const D = layerStep ? await cached('walk' + L, 'walk_layer', { layer: L }) : null;
  // Fetch the layer in the background while the input steps are on screen.
  if (!layerStep && !(('walk' + L) in S.cache)) { const cache = S.cache; call('walk_layer', { layer: L }).then((r) => (cache['walk' + L] = r)).catch(() => {}); }
  const c = { T, idx: [...Array(T).keys()], sel: Math.min(S.sel, T - 1), E, D, L };

  const go = (d) => { const n = steps[k + d]; if (n) walkGo(n.id); };
  const partName = (pt) => (pt === 'Attention' || pt === 'MLP' ? `Layer ${L} · ${pt.toLowerCase()}` : pt);
  const parts = [];
  for (const [i, x] of steps.entries()) {
    if (!parts.length || parts[parts.length - 1].name !== x.part) parts.push({ name: x.part, steps: [] });
    parts[parts.length - 1].steps.push([i, x]);
  }
  return [
    explain('The forward pass one operation at a time; one row per token, hover any cell for its value.',
      st.tokens.length > 16 ? ` First 16 of ${st.tokens.length} tokens shown.` : '',
      st.tokens.length > 6 ? [' ', el('button', { type: 'button', class: 'linkish', onclick: () => { $('#prompt').value = 'The capital'; $('#run').click(); } }, 'Try a short prompt'), '.'] : ''),
    el('div', { class: 'wbar' },
      el('button', { type: 'button', class: 'icon', title: 'previous step (←)', disabled: k === 0, onclick: () => go(-1) }, '‹'),
      el('div', { class: 'wtitle' },
        el('span', { class: 'dim', text: `${partName(s.part)} · step ${k + 1} of ${steps.length}` }),
        el('b', { text: s.title })),
      el('button', { type: 'button', class: 'winfo-btn' + (WALK.info ? ' on' : ''), title: 'explain this step in prose, with links to read or watch', 'aria-pressed': String(WALK.info),
        onclick: () => { WALK.info = !WALK.info; renderTab(); } }, svgIcon(INFO_ICON), WALK.info ? 'Hide explanation' : 'Explain'),
      el('label', { class: 'wlayer', title: 'which layer to walk through' }, 'layer',
        el('select', { onchange: (e) => { WALK.layer = +e.target.value; if (!layerStep && s.id !== 'repeat') return renderTab(); WALK.step = WALK.layer > 0 ? 'stream' : 'rms'; renderTab(); } },
          [...Array(m.n_layers).keys()].map((i) => el('option', { value: i, text: String(i), selected: i === L })))),
      el('button', { type: 'button', class: 'primary', title: 'next step (→)', disabled: k === steps.length - 1, onclick: () => go(1) }, 'Next ›')),
    el('div', { class: 'wprog' }, parts.map((pt) => el('div', { class: 'wpart' },
      el('span', { class: 'dim', text: partName(pt.name) }),
      el('div', { class: 'wdots' }, pt.steps.map(([i, x]) => el('button', { type: 'button', class: 'wdot' + (i === k ? ' on' : i < k ? ' done' : ''), title: x.title, 'aria-label': x.title, onclick: () => walkGo(x.id) })))))),
    el('section', { class: 'wcard' },
      WALK.info ? infoPanel(s.id) : null,
      s.tex ? wformula(s.tex, s.dims) : null,
      s.weights ? el('div', { class: 'wweights' }, s.weights.map(([n, dims, note, sym]) => wweight(n, dims, note, sym))) : null,
      s.body(c)),
  ];
}

document.addEventListener('keydown', (e) => {
  if (S.tab !== 'walk' || !S.st || e.altKey || e.ctrlKey || e.metaKey) return;
  if (e.target.closest('input, textarea, select, [contenteditable]') || document.querySelector('dialog[open]')) return;
  if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
  e.preventDefault();
  const steps = wsteps(WALK.layer), k = steps.findIndex((s) => s.id === WALK.step), n = steps[k + (e.key === 'ArrowRight' ? 1 : -1)];
  if (n) walkGo(n.id);
});
