/* PneumoShift browser demo.

   The same demo as app/app.py, without a server: both models run on the visitor's device with
   ONNX Runtime Web. This file repeats, step by step, what the Python code does:

     picture -> grey, centre crop, 256 px, centre crop 224 px     src/pneumoshift/preprocess.py
     one run of the exported network -> logit, embedding, heatmap   scripts/08_export_web.py
     score = sigmoid(logit / temperature), flagged if >= threshold  src/pneumoshift/inference.py
     distance of the embedding from the training data               src/pneumoshift/ood.py
     heatmap overlay                                                src/pneumoshift/cam.py
     the words on the page                                          app/app.py

   tests/web/parity.mjs checks that this page and the Python code agree on every example picture.

   Where the model files come from: the Hugging Face Hub (onnx/ folder of each model repo). If that
   fails, or with ?models=local in the address, the page looks for them in models/<name>/ next to itself. */
(() => {
'use strict';

const MODELS = [
  { key: 'kids_only', displayName: 'Children-only model', repo: 'RL8596/pneumoshift-kids-only' },
  { key: 'kids_plus_adults', displayName: 'Children + adults model', repo: 'RL8596/pneumoshift-kids-plus-adults' },
];
const ORT_DIST = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/';
const PARAMS = new URLSearchParams(location.search);
const STORE_SIZE = 256;            // pictures are reduced to this size ...
const IMG_SIZE = 224;              // ... and the model sees the centre of that
const MAX_UPLOAD = 30 * 1024 * 1024;
const MAX_SIDE = 4096;             // larger pictures are shrunk by the browser first (phones cannot hold them)
const AGE_TO_GROUP = { child_1_5: 'children', child_6_17: null, adult: 'adults' };   // nobody trained on ages 6 to 17
const AGE_WORDS = { child_1_5: 'children aged 1 to 5', child_6_17: 'children or teenagers aged 6 to 17', adult: 'adults' };
const GROUP_WORDS = { children: 'children aged 1 to 5', adults: 'adults' };
const EMPTY_STATUS = '<div class="ps-empty">Choose a chest X-ray or pick an example to see what this model says.</div>';

const $ = (id) => document.getElementById(id);
const state = { picture: null, age: 'unknown', seq: 0, examples: [], inferno: null };

class UserError extends Error {}   // a problem with the picture, shown to the user in plain words

// ---------------------------------------------------------------------------------------------
// Picture -> model input. Must give exactly the pixels that Pillow gives in preprocess.py.
// ---------------------------------------------------------------------------------------------
async function decode(blob) {
  /* file -> { data: RGBA bytes, width, height }. Transparent pixels become black, a rotated phone photo is
     turned upright, and the browser is told not to change the colours. */
  let source = null;
  if (typeof createImageBitmap === 'function') {
    try {
      source = await createImageBitmap(blob, { colorSpaceConversion: 'none', premultiplyAlpha: 'none', imageOrientation: 'from-image' });
    } catch {
      try { source = await createImageBitmap(blob); } catch { source = null; }
    }
  }
  if (!source) {
    source = await new Promise((resolve, reject) => {
      const img = new Image();
      const url = URL.createObjectURL(blob);
      img.onload = () => { URL.revokeObjectURL(url); resolve(img); };
      img.onerror = () => { URL.revokeObjectURL(url); reject(new UserError('The file could not be read as an image.')); };
      img.src = url;
    });
  }
  const w = source.width || source.naturalWidth, h = source.height || source.naturalHeight;
  if (!w || !h) throw new UserError('The image is empty.');
  const shrink = Math.min(1, MAX_SIDE / Math.max(w, h));
  const cw = Math.max(1, Math.round(w * shrink)), ch = Math.max(1, Math.round(h * shrink));
  const canvas = document.createElement('canvas');
  canvas.width = cw; canvas.height = ch;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, cw, ch);
  ctx.drawImage(source, 0, 0, cw, ch);
  if (source.close) source.close();
  let pixels;
  try { pixels = ctx.getImageData(0, 0, cw, ch); } catch { throw new UserError('The image is too large to open safely.'); }
  return { data: pixels.data, width: cw, height: ch };
}

function toSquareGray(rgba, width, height) {
  /* RGBA -> grey, centre crop to a square. The grey formula is Pillow's convert("L"). */
  const s = Math.min(width, height);
  const left = Math.floor((width - s) / 2), top = Math.floor((height - s) / 2);
  const out = new Uint8Array(s * s);
  for (let y = 0; y < s; y++) {
    let p = ((y + top) * width + left) * 4;
    const row = y * s;
    for (let x = 0; x < s; x++, p += 4) {
      out[row + x] = (rgba[p] * 19595 + rgba[p + 1] * 38470 + rgba[p + 2] * 7471 + 0x8000) >>> 16;
    }
  }
  return { gray: out, size: s };
}

function resizeCoefficients(inSize, outSize) {
  /* Pillow's precompute_coeffs + normalize_coeffs_8bpc for the BILINEAR filter (a triangle that gets
     wider when the picture is made smaller, so every source pixel counts). */
  const PRECISION = 22;
  const scale = inSize / outSize;
  const filterScale = Math.max(scale, 1);
  const support = filterScale;                       // filter support 1.0 * filterScale
  const ksize = Math.ceil(support) * 2 + 1;
  const bounds = new Int32Array(outSize * 2);
  const kk = new Int32Array(outSize * ksize);
  const k = new Float64Array(ksize);
  const ss = 1 / filterScale;
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    let xmin = Math.trunc(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    let xmax = Math.trunc(center + support + 0.5);
    if (xmax > inSize) xmax = inSize;
    xmax -= xmin;
    let ww = 0;
    for (let x = 0; x < xmax; x++) {
      let w = (x + xmin - center + 0.5) * ss;
      if (w < 0) w = -w;
      w = w < 1 ? 1 - w : 0;
      k[x] = w;
      ww += w;
    }
    for (let x = 0; x < xmax; x++) {
      const v = ww !== 0 ? k[x] / ww : k[x];
      kk[xx * ksize + x] = Math.trunc((v < 0 ? -0.5 : 0.5) + v * (1 << PRECISION));
    }
    bounds[xx * 2] = xmin;
    bounds[xx * 2 + 1] = xmax;
  }
  return { ksize, bounds, kk, half: 1 << (PRECISION - 1), shift: PRECISION };
}

function resizeGray(src, size, outSize) {
  /* Pillow's Image.resize((outSize, outSize), BILINEAR) for an 8-bit grey square: first along x, then along y,
     in whole numbers, exactly like the C code. */
  if (size === outSize) return src.slice();
  const c = resizeCoefficients(size, outSize);
  const { ksize, bounds, kk, half, shift } = c;
  const tmp = new Uint8Array(size * outSize);              // [size rows, outSize columns]
  for (let y = 0; y < size; y++) {
    const rowIn = y * size, rowOut = y * outSize;
    for (let xx = 0; xx < outSize; xx++) {
      const xmin = bounds[xx * 2], xmax = bounds[xx * 2 + 1], k0 = xx * ksize;
      let sum = half;
      for (let x = 0; x < xmax; x++) sum += src[rowIn + xmin + x] * kk[k0 + x];
      const v = sum >> shift;
      tmp[rowOut + xx] = v < 0 ? 0 : v > 255 ? 255 : v;
    }
  }
  const out = new Uint8Array(outSize * outSize);
  for (let yy = 0; yy < outSize; yy++) {
    const ymin = bounds[yy * 2], ymax = bounds[yy * 2 + 1], k0 = yy * ksize, rowOut = yy * outSize;
    for (let x = 0; x < outSize; x++) {
      let sum = half;
      for (let y = 0; y < ymax; y++) sum += tmp[(ymin + y) * outSize + x] * kk[k0 + y];
      const v = sum >> shift;
      out[rowOut + x] = v < 0 ? 0 : v > 255 ? 255 : v;
    }
  }
  return out;
}

function centerCrop(src, size, outSize) {
  const off = Math.floor((size - outSize) / 2);
  const out = new Uint8Array(outSize * outSize);
  for (let y = 0; y < outSize; y++) out.set(src.subarray((y + off) * size + off, (y + off) * size + off + outSize), y * outSize);
  return out;
}

function toTensor(gray, mean, std) {
  /* grey 0..255 -> three equal channels, scaled like ImageNet pictures (float32, as in preprocess.normalize). */
  const n = gray.length;
  const t = new Float32Array(3 * n);
  for (let ch = 0; ch < 3; ch++) {
    const m = Math.fround(mean[ch]), s = Math.fround(std[ch]);
    for (let i = 0; i < n; i++) t[ch * n + i] = Math.fround(Math.fround(gray[i] / 255) - m) / s;
  }
  return t;
}

async function preprocess(blob) {
  if (blob.size > MAX_UPLOAD) throw new UserError('The file is larger than 30 MB.');
  let decoded;
  try { decoded = await decode(blob); } catch (e) {
    throw e instanceof UserError ? e : new UserError('The file could not be read as an image.');
  }
  const square = toSquareGray(decoded.data, decoded.width, decoded.height);
  const stored = resizeGray(square.gray, square.size, STORE_SIZE);
  return { gray: centerCrop(stored, STORE_SIZE, IMG_SIZE) };
}

// ---------------------------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------------------------
function sources(model) {
  const local = `models/${model.key}/`;
  const hub = `https://huggingface.co/${model.repo}/resolve/main/onnx/`;
  return PARAMS.get('models') === 'local' ? [local] : [hub, local];
}

async function fetchOk(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return res;
}

async function sha256Hex(bytes) {
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('');
}

async function fetchBytes(url, info, onProgress) {
  /* Download a model file once and keep it on this device (Cache Storage). The download is checked
     against the size and SHA-256 in meta.json, so a broken or half-finished file is never used. */
  const key = `${url}?sha256=${info.sha256}`;        // only a name inside the cache, never requested
  let cache = null;
  try { cache = await caches.open('pneumoshift-models-1'); } catch { cache = null; }
  if (cache) {
    try {
      const hit = await cache.match(key);
      if (hit) {
        const buf = await hit.arrayBuffer();
        if (buf.byteLength === info.bytes) { if (onProgress) onProgress(info.bytes, info.bytes); return new Uint8Array(buf); }
      }
    } catch { /* fall through to the download */ }
  }
  const res = await fetchOk(url);
  let bytes;
  if (res.body && onProgress) {
    const reader = res.body.getReader();
    bytes = new Uint8Array(info.bytes);
    let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (got + value.length > info.bytes) throw new Error(`${url}: larger than expected`);
      bytes.set(value, got);
      got += value.length;
      onProgress(got, info.bytes);
    }
    if (got !== info.bytes) throw new Error(`${url}: download incomplete`);
  } else {
    bytes = new Uint8Array(await res.arrayBuffer());
    if (bytes.length !== info.bytes) throw new Error(`${url}: unexpected size`);
  }
  if (crypto && crypto.subtle && (await sha256Hex(bytes)) !== info.sha256) throw new Error(`${url}: checksum mismatch`);
  if (cache) cache.put(key, new Response(bytes.slice().buffer, { headers: { 'content-type': 'application/octet-stream' } })).catch(() => {});
  return bytes;
}

function readOod(bytes, layout) {
  const { n_classes: nClasses, dim, n_ref: nRef } = layout;
  const floats = new Float32Array(bytes.buffer, bytes.byteOffset, nClasses * dim + dim * dim + nRef);
  return {
    nClasses, dim, nRef, p95: layout.p95, p99: layout.p99,
    means: Float64Array.from(floats.subarray(0, nClasses * dim)),
    precision: Float64Array.from(floats.subarray(nClasses * dim, nClasses * dim + dim * dim)),
    refScores: floats.subarray(nClasses * dim + dim * dim),
  };
}

async function loadModel(i) {
  const model = MODELS[i];
  model.status = 'loading'; model.progress = [0, 0]; model.error = null;
  renderCard(i);
  let lastError = null;
  for (const base of sources(model)) {
    try {
      const meta = await (await fetchOk(base + 'meta.json', { cache: 'no-cache' })).json();
      model.meta = meta; model.config = meta.config || {};
      renderCardFrame(i);
      const [onnx, oodBytes] = await Promise.all([
        fetchBytes(base + 'model.onnx', meta.files['model.onnx'], (got, total) => { model.progress = [got, total]; renderProgress(i); }),
        meta.ood ? fetchBytes(base + 'ood.bin', meta.files['ood.bin']) : null,
      ]);
      model.session = await ort.InferenceSession.create(onnx, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
      model.ood = null;
      if (meta.ood && oodBytes) {
        try { model.ood = readOod(oodBytes, meta.ood); } catch (e) { console.warn('familiarity check is off:', e); }
      }
      model.temperature = Number(model.config.temperature ?? 1);
      model.threshold = Number(model.config.threshold ?? 0.5);
      if (!(model.temperature > 0)) throw new Error('temperature must be a positive number');
      model.status = 'ready';
      renderCard(i);
      const pic = state.picture;
      if (pic && pic.results[i] === undefined) analyse(i, pic);
      return;
    } catch (e) {
      lastError = e;
      console.warn(`could not load ${model.key} from ${base}:`, e);
    }
  }
  model.status = 'error'; model.error = lastError;
  renderCard(i);
}

function sigmoid(z, temperature) { return 1 / (1 + Math.exp(-z / temperature)); }

function oodScore(emb, ood) {
  /* Mahalanobis distance to the nearest class mean (ood.py: MahalanobisOOD.score). */
  const { dim, nClasses, means, precision } = ood;
  const d = new Float64Array(dim);
  let best = Infinity;
  for (let c = 0; c < nClasses; c++) {
    for (let j = 0; j < dim; j++) d[j] = emb[j] - means[c * dim + j];
    let m = 0;
    for (let a = 0; a < dim; a++) {
      let row = 0;
      const off = a * dim;
      for (let b = 0; b < dim; b++) row += precision[off + b] * d[b];
      m += d[a] * row;
    }
    if (m < best) best = m;
  }
  return Math.sqrt(Math.max(best, 0));
}

function oodPercentile(score, ood) {
  /* Share of validation X-rays that looked MORE familiar than this one (0..100). */
  const ref = ood.refScores;
  let lo = 0, hi = ref.length;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (ref[mid] < score) lo = mid + 1; else hi = mid; }
  return 100 * lo / ref.length;
}

function overlay(gray, cam) {
  /* grey picture + heatmap 0..1 -> RGBA overlay (cam.py: overlay). Weak areas stay transparent. */
  const n = gray.length, lut = state.inferno;
  const out = new Uint8ClampedArray(n * 4);
  for (let i = 0; i < n; i++) {
    let c = cam[i];
    c = c > 0 ? (c < 1 ? c : 1) : 0;                                 // also turns NaN into 0
    const colour = lut[Math.min(255, Math.floor(c * 256))];
    const a = 0.45 * Math.pow(c, 0.7), base = gray[i] / 255;
    for (let ch = 0; ch < 3; ch++) {
      const v = base * (1 - a) + colour[ch] * a;
      out[i * 4 + ch] = Math.floor((v < 0 ? 0 : v > 1 ? 1 : v) * 255);
    }
    out[i * 4 + 3] = 255;
  }
  return out;
}

async function runModel(model, pic) {
  const input = model.meta.input || {};
  const mean = input.mean || [0.485, 0.456, 0.406], std = input.std || [0.229, 0.224, 0.225];
  if (!pic.tensor) pic.tensor = toTensor(pic.gray, mean, std);
  const feeds = { [input.name || 'input']: new ort.Tensor('float32', pic.tensor, [1, 3, IMG_SIZE, IMG_SIZE]) };
  const out = await model.session.run(feeds);
  const logit = Number(out.logit.data[0]);
  if (!Number.isFinite(logit)) throw new Error('The model returned an invalid number for this image.');
  const cam = out.cam.data;
  const probability = sigmoid(logit, model.temperature);
  const r = {
    logit, probability, threshold: model.threshold, flagged: probability >= model.threshold,
    cam, heatmap: overlay(pic.gray, cam), oodScore: null, oodPercentile: null, oodLevel: null,
  };
  if (model.ood) {
    const score = oodScore(out.embedding.data, model.ood);
    r.oodScore = score;
    r.oodPercentile = oodPercentile(score, model.ood);
    r.oodLevel = score > model.ood.p99 ? 'very unusual' : score > model.ood.p95 ? 'unusual' : 'ok';
  }
  return r;
}

let chain = Promise.resolve();                 // one model run at a time
function analyse(i, pic) {
  pic.results[i] = 'running';
  renderCard(i);
  chain = chain.then(async () => {
    if (pic !== state.picture) return;
    await new Promise((resolve) => setTimeout(resolve, 0));       // let the page show "Analysing" first
    try { pic.results[i] = await runModel(MODELS[i], pic); } catch (e) { console.error(e); pic.results[i] = null; }
    if (pic === state.picture) renderCard(i);
  });
}

// ---------------------------------------------------------------------------------------------
// Words and HTML (a translation of app/app.py)
// ---------------------------------------------------------------------------------------------
function esc(x) {
  return String(x).replace(/[&<>"']/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#x27;' }[ch]));
}

function pct(p, decimals) {
  /* 0.985 -> '98.5%'. Whole numbers in the middle, one decimal near 0 and 100. */
  const v = 100 * p;
  if (decimals === undefined) {
    if (v < 0.1) return 'under 0.1%';
    if (v > 99.9) return 'over 99.9%';
    decimals = (v < 10 || v > 90) ? 1 : 0;
  }
  return `${v.toFixed(decimals)}%`;
}

function pctPair(p, threshold) {
  /* Score and threshold as text, with enough decimals to tell them apart. */
  let a = pct(p), b = pct(threshold);
  for (const decimals of [1, 2, 3, 4]) {
    if (a !== b || p === threshold) break;
    a = pct(p, decimals); b = pct(threshold, decimals);
  }
  return [a, b];
}

function trainGroups(cfg) {
  /* Patient groups a model was trained on: a Set of 'children' / 'adults', or null if not recorded. */
  if (Array.isArray(cfg.train_groups)) return new Set(cfg.train_groups.map((g) => String(g).trim().toLowerCase()));
  const described = (cfg.train_data || []).map(String).join(' ').toLowerCase();
  const found = [['children', 'child'], ['adults', 'adult']].filter(([, word]) => described.includes(word)).map(([g]) => g);
  return found.length ? new Set(found) : null;
}

function joinWords(items) {
  return items.length === 1 ? items[0] : `${items.slice(0, -1).join(', ')} and ${items[items.length - 1]}`;
}

function trainedOnText(cfg) {
  const data = (cfg.train_data || []).map(String);
  return data.length ? `Trained on ${data.join(' + ')}.` : 'Training data: not recorded.';
}

function familiarityHtml(r, model) {
  /* -> [HTML, true if the result should not be relied on] */
  const level = r.oodLevel;
  if (level === null) {
    return ['<div class="ps-check"><b>X-ray check:</b> not available for this model yet, so it cannot warn you about unusual X-rays.</div>', false];
  }
  if (level === 'ok') {
    return ['<div class="ps-check ps-check--ok"><b>X-ray check passed:</b> this X-ray looks like the X-rays this model learned from.</div>', false];
  }
  const p = r.oodPercentile, n = model.ood.nRef;
  let how;
  if (p >= 100) how = `None of the ${n} comparison X-rays looked this unfamiliar to the model.`;
  else if (p >= 99.5) how = 'More than 99 in 100 comparison X-rays looked more familiar to the model than this one.';
  else how = `About ${p.toFixed(0)} in 100 comparison X-rays looked more familiar to the model than this one.`;
  const blocking = level === 'very unusual';
  const head = blocking
    ? 'This X-ray looks unlike the X-rays this model learned from. Do not rely on this result.'
    : 'This X-ray looks somewhat unlike the X-rays this model learned from. Treat this result with caution.';
  return [`<div class="ps-alert" role="alert"><b>${head}</b><span>${how}</span></div>`, blocking];
}

function patientGroupHtml(cfg, age) {
  /* -> [HTML, true if the result should not be relied on] */
  if (!(age in AGE_TO_GROUP)) {
    return ['<div class="ps-check"><b>Patient check:</b> not done. Choose the patient age group to check whether this model was trained on such patients.</div>', false];
  }
  const groups = trainGroups(cfg);
  if (groups === null) {
    return ['<div class="ps-check"><b>Patient check:</b> not possible, the patient groups this model was trained on are not recorded.</div>', false];
  }
  if (groups.has(AGE_TO_GROUP[age])) {
    return [`<div class="ps-check ps-check--ok"><b>Patient check passed:</b> this model was trained on X-rays of ${AGE_WORDS[age]}.</div>`, false];
  }
  const ordered = [...groups].sort((a, b) => (a !== 'children') - (b !== 'children') || a.localeCompare(b));
  const known = joinWords(ordered.map((g) => GROUP_WORDS[g] || g));
  return [`<div class="ps-alert" role="alert"><b>This model was never trained on X-rays of ${AGE_WORDS[age]}. `
    + 'Its result for this patient may be wrong, even when it looks confident.</b>'
    + `<span>It only learned from X-rays of ${esc(known)}.</span></div>`, true];
}

function knownAnswerHtml(example, r) {
  /* For our examples: compare the model with the answer recorded in the dataset. */
  if (!example) return '';
  const truth = esc(example.true_label || '');
  const label = example.label;
  let right, text;
  if (label === 0 || label === 1) {
    right = r.flagged === Boolean(label);
    if (right) text = `<b>Correct.</b> The known answer for this example is: ${truth}.`;
    else if (label === 1) text = `<b>Missed.</b> The known answer for this example is: ${truth}.`;
    else text = `<b>False alarm.</b> The known answer for this example is: ${truth}.`;
  } else if (r.oodLevel === null) {       // a picture that is not an X-ray: the right reaction is a warning
    right = false; text = '<b>No warning.</b> This picture is not an X-ray, and this model has no X-ray check yet.';
  } else if (r.oodLevel === 'ok') {
    right = false; text = '<b>Missed.</b> This picture is not an X-ray, but the X-ray check did not notice.';
  } else {
    right = true; text = '<b>Correct warning.</b> This picture is not an X-ray.';
  }
  return `<div class="ps-truth ${right ? 'ps-truth--right' : 'ps-truth--wrong'}">${text}</div>`;
}

function statusHtml(model, r, age, example) {
  /* Everything one model says about one picture. */
  if (r === null) return '<div class="ps-empty">This model could not analyse the picture.</div>';
  const p = r.probability, thr = r.threshold;
  const [familiarity, blockA] = familiarityHtml(r, model);
  const [patient, blockB] = patientGroupHtml(model.config, age);
  const [pText, thrText] = pctPair(p, thr);
  let kind, title, detail;
  if (r.flagged) {
    kind = 'flag'; title = 'Flag for priority review'; detail = `this model flags at ${thrText} or above`;
  } else if (p < 0.5) {
    kind = 'clear'; title = 'No sign of pneumonia found'; detail = `this model flags at ${thrText} or above`;
  } else {      // only possible with a threshold above 50%: saying "no sign" would contradict the number
    kind = 'unsure'; title = 'Not flagged, but not clearly normal'; detail = `high, but below the ${thrText} at which this model flags`;
  }
  if (blockA || blockB) kind = 'muted';
  const tick = Math.min(Math.max(thr, 0), 1);
  const verdict = `<div class="ps-verdict ps-verdict--${kind}">`
    + `<div class="ps-verdict__title">${title}</div>`
    + `<div class="ps-verdict__sub">Pneumonia score: <b>${pText}</b> <span>(${detail})</span></div>`
    + `<div class="ps-bar" aria-hidden="true"><div class="ps-bar__fill" style="width:${(100 * p).toFixed(1)}%"></div>`
    + `<div class="ps-bar__tick" style="left:${(100 * tick).toFixed(1)}%"></div></div></div>`;
  const both = [familiarity, patient];
  const alerts = both.filter((h) => h.includes('ps-alert')).join('');
  const checks = both.filter((h) => !h.includes('ps-alert')).join('');
  return verdict + alerts + checks + knownAnswerHtml(example, r);
}

function metric(value, kind) {
  const v = Number(value);
  if (value === null || value === undefined || !Number.isFinite(v)) return 'n/a';
  if (kind === 'count') return Math.trunc(v).toLocaleString('en-US');
  return kind === 'share' ? `${(100 * v).toFixed(0)}%` : v.toFixed(3);
}

function interval(ci, kind) {
  if (!Array.isArray(ci) || ci.length < 2) return '';
  const lo = Number(ci[0]), hi = Number(ci[1]);
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return '';
  const text = kind === 'share' ? `${(100 * lo).toFixed(0)} to ${(100 * hi).toFixed(0)}%` : `${lo.toFixed(3)} to ${hi.toFixed(3)}`;
  return `<small title="95% confidence interval">${text}</small>`;
}

function testedHtml(cfg) {
  /* The "How well was this model tested?" panel. Shows only numbers stored with the model. */
  const head = '<div class="ps-tested__title">How well was this model tested?</div>';
  const rows = [];
  const metrics = cfg.metrics;
  if (metrics && typeof metrics === 'object') {
    for (const [group, m] of Object.entries(metrics)) {
      if (!m || typeof m !== 'object') continue;
      const name = String(group).replaceAll('_', ' ').trim();
      const cells = [`<td data-label="Number">${metric(m.n, 'count')}</td>`];
      for (const [key, kind, label] of [['sensitivity', 'share', 'Sensitivity'], ['specificity', 'share', 'Specificity'], ['auroc', 'score', 'AUROC']]) {
        cells.push(`<td data-label="${label}">${metric(m[key], kind)}${interval(m[`${key}_ci`], kind)}</td>`);
      }
      rows.push(`<tr><th scope="row">${esc(name.slice(0, 1).toUpperCase() + name.slice(1))}</th>${cells.join('')}</tr>`);
    }
  }
  if (!rows.length) return `<div class="ps-tested">${head}<div class="ps-tested__none">Not measured yet.</div></div>`;
  return `<div class="ps-tested">${head}<div class="ps-tablewrap"><table class="ps-table"><thead><tr>`
    + '<th scope="col">Test X-rays (never used for training)</th><th scope="col">Number</th>'
    + '<th scope="col">Sensitivity</th><th scope="col">Specificity</th><th scope="col">AUROC</th>'
    + `</tr></thead><tbody>${rows.join('')}</tbody></table></div>`
    + '<div class="ps-note">Sensitivity: the share of X-rays with pneumonia that the model flagged. '
    + 'Specificity: the share of X-rays without pneumonia that it did not flag. '
    + 'AUROC: how well it ranks sick above healthy (0.5 is guessing, 1 is perfect). '
    + 'Small print: the range we are 95% sure the true value lies in.</div></div>';
}

function noticeHtml(text, kind = 'info') {
  return text ? `<div class="ps-notice ps-notice--${kind}" role="status">${text}</div>` : '';
}

function exampleNotice(example) {
  const age = typeof example.age === 'number' ? `, age ${Math.trunc(example.age)}` : '';
  return noticeHtml(`<b>Example with a known answer: ${esc(example.true_label || '')}${age}.</b> <span>${esc(example.source || '')}</span>`);
}

// ---------------------------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------------------------
function buildCards() {
  $('cards').innerHTML = MODELS.map((m, i) => `
    <article class="ps-panel ps-card" id="card-${i}">
      <div class="ps-model"><h2 id="name-${i}">${esc(m.displayName)}</h2><p id="trained-${i}"></p></div>
      <div class="ps-status" id="status-${i}"></div>
      <div class="ps-pictures" id="pictures-${i}" hidden>
        <figure><canvas id="saw-${i}" width="${IMG_SIZE}" height="${IMG_SIZE}"></canvas><figcaption>What the model saw</figcaption></figure>
        <figure><canvas id="heat-${i}" width="${IMG_SIZE}" height="${IMG_SIZE}"></canvas><figcaption id="heatcap-${i}">Where the model looked</figcaption></figure>
      </div>
      <div id="tested-${i}"></div>
    </article>`).join('');
}

function renderCardFrame(i) {
  /* The parts of a card that depend on the model only: its name, training data and test results. */
  const m = MODELS[i], cfg = m.config || {};
  $(`name-${i}`).textContent = String(cfg.display_name || m.displayName);
  $(`trained-${i}`).textContent = m.meta ? trainedOnText(cfg) : '';
  $(`tested-${i}`).innerHTML = m.meta ? testedHtml(cfg) : '';
}

function megabytes(bytes) { return (bytes / 1e6).toFixed(0); }

function renderProgress(i) {
  const m = MODELS[i], [got, total] = m.progress;
  const text = $(`progress-text-${i}`), bar = $(`progress-bar-${i}`);
  if (!text || !bar || !total) return;
  text.textContent = got >= total ? 'Starting the model …' : `${megabytes(got)} of ${megabytes(total)} MB`;
  bar.style.width = `${(100 * got / total).toFixed(1)}%`;
}

function drawGray(canvas, gray) {
  const img = new ImageData(IMG_SIZE, IMG_SIZE);
  for (let i = 0; i < gray.length; i++) { img.data[i * 4] = img.data[i * 4 + 1] = img.data[i * 4 + 2] = gray[i]; img.data[i * 4 + 3] = 255; }
  canvas.getContext('2d').putImageData(img, 0, 0);
}

function renderCard(i) {
  const m = MODELS[i], pic = state.picture;
  const status = $(`status-${i}`), pictures = $(`pictures-${i}`);
  if (!status) return;
  renderCardFrame(i);
  pictures.hidden = true;
  if (m.status === 'error') {
    status.innerHTML = '<div class="ps-empty"><b>This model could not be loaded.</b> Check the internet connection, then '
      + `<button type="button" class="ps-button" id="retry-${i}">try again</button></div>`;
    $(`retry-${i}`).addEventListener('click', () => loadModel(i));
    return;
  }
  if (m.status !== 'ready') {
    status.innerHTML = '<div class="ps-empty">Loading this model … <span id="progress-text-' + i + '"></span>'
      + `<div class="ps-progress"><div id="progress-bar-${i}"></div></div>`
      + '<div class="ps-note">A one-time download of about 45 MB. After that it is kept on this device.'
      + (pic ? ' Your picture is analysed as soon as the model is ready.' : '') + '</div></div>';
    renderProgress(i);
    return;
  }
  if (!pic) { status.innerHTML = EMPTY_STATUS; return; }
  const r = pic.results[i];
  if (r === undefined || r === 'running') { status.innerHTML = '<div class="ps-empty">Analysing the picture …</div>'; return; }
  status.innerHTML = statusHtml(m, r, state.age, pic.example);
  if (r) {
    drawGray($(`saw-${i}`), pic.gray);
    $(`heat-${i}`).getContext('2d').putImageData(new ImageData(r.heatmap, IMG_SIZE, IMG_SIZE), 0, 0);
    // every area can speak against pneumonia; then the map is empty and there is nothing to colour
    $(`heatcap-${i}`).textContent = r.cam.some((v) => v > 0) ? 'Where the model looked'
      : 'Where the model looked: no area pushed it towards pneumonia';
    pictures.hidden = false;
  }
}

function renderAll() { MODELS.forEach((_, i) => renderCard(i)); }

function setAge(age) {
  state.age = age in AGE_TO_GROUP ? age : 'unknown';
  const input = document.querySelector(`input[name="age"][value="${state.age}"]`);
  if (input) input.checked = true;
}

function selectExample(index) {
  document.querySelectorAll('.ps-example').forEach((el, i) => {
    el.classList.toggle('is-selected', i === index);
    el.setAttribute('aria-pressed', i === index ? 'true' : 'false');
  });
}

function clearPicture(notice = '') {
  state.seq++;
  state.picture = null;
  $('preview').hidden = true; $('drop-text').hidden = false; $('clear').hidden = true;
  $('file').value = '';
  $('notice').innerHTML = notice;
  setAge('unknown');
  selectExample(-1);
  renderAll();
}

async function setPicture(blob, example, age) {
  /* A new picture is a new patient: the age group goes back to 'Not sure' unless an example sets it. */
  const seq = ++state.seq;
  let pre;
  try { pre = await preprocess(blob); } catch (e) {
    if (seq !== state.seq) return;
    const reason = e instanceof UserError ? e.message : 'The file could not be read as an image.';
    clearPicture(noticeHtml(`<b>This file cannot be used.</b> <span>${esc(reason)} Please choose a chest X-ray saved as a PNG or JPEG picture.</span>`, 'error'));
    return;
  }
  if (seq !== state.seq) return;
  const pic = { gray: pre.gray, tensor: null, example: example || null, results: MODELS.map(() => undefined), seq };
  state.picture = pic;
  setAge(age || 'unknown');
  drawGray($('preview'), pic.gray);
  $('preview').hidden = false; $('drop-text').hidden = true; $('clear').hidden = false;
  $('notice').innerHTML = example ? exampleNotice(example) : '';
  MODELS.forEach((m, i) => { if (m.status === 'ready') analyse(i, pic); else renderCard(i); });
  if (window.matchMedia('(max-width: 760px)').matches) $('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

async function loadExamples() {
  try {
    const manifest = await (await fetchOk('examples/examples.json')).json();
    state.examples = (manifest.examples || []).filter((e) => e && e.file);
  } catch (e) { console.warn('no examples:', e); state.examples = []; }
  if (!state.examples.length) return;
  $('examples').innerHTML = state.examples.map((e, i) =>
    `<button type="button" class="ps-example" data-index="${i}" aria-pressed="false">`
    + `<img src="examples/${encodeURIComponent(e.file)}" alt="" loading="lazy" width="96" height="96">`
    + `<span>${esc(e.caption || e.true_label || '')}</span></button>`).join('');
  $('examples-panel').hidden = false;
  $('examples').addEventListener('click', async (event) => {
    const button = event.target.closest('.ps-example');
    if (!button) return;
    const index = Number(button.dataset.index), example = state.examples[index];
    try {
      const blob = await (await fetchOk(`examples/${encodeURIComponent(example.file)}`)).blob();
      $('file').value = '';
      selectExample(index);
      await setPicture(blob, example, { children: 'child_1_5', adults: 'adult' }[example.group] || 'unknown');
      selectExample(index);
    } catch (e) {
      console.error(e);
      $('notice').innerHTML = noticeHtml('<b>The example could not be loaded.</b> <span>Check the internet connection and try again.</span>', 'error');
    }
  });
  const sources = [];
  for (const e of state.examples) {
    if (e.source && !sources.includes(e.source) && (e.label === 0 || e.label === 1)) sources.push(e.source);
  }
  if (sources.length) $('credits').textContent = `Example X-rays: ${sources.join(' ')}`;
}

function wireInputs() {
  const drop = $('drop'), file = $('file');
  const choose = (f) => { if (f) { selectExample(-1); setPicture(f, null, 'unknown'); } };
  file.addEventListener('change', () => choose(file.files && file.files[0]));
  for (const type of ['dragenter', 'dragover']) {
    drop.addEventListener(type, (e) => { e.preventDefault(); drop.classList.add('is-over'); });
  }
  for (const type of ['dragleave', 'drop']) drop.addEventListener(type, () => drop.classList.remove('is-over'));
  drop.addEventListener('drop', (e) => { e.preventDefault(); choose(e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]); });
  for (const type of ['dragover', 'drop']) window.addEventListener(type, (e) => e.preventDefault());   // a missed drop must not open the file
  $('clear').addEventListener('click', (e) => { e.preventDefault(); clearPicture(); });
  $('age').addEventListener('change', (e) => {
    if (e.target && e.target.name === 'age') { state.age = e.target.value; if (state.picture) renderAll(); }
  });
}

async function start() {
  buildCards();
  wireInputs();
  if (typeof WebAssembly !== 'object' || typeof ort === 'undefined') {
    $('notice').innerHTML = noticeHtml('<b>The demo cannot start in this browser.</b> <span>'
      + (typeof ort === 'undefined' ? 'A part of the page could not be downloaded. Check the internet connection and reload.'
        : 'It needs WebAssembly, which every current browser supports.') + '</span>', 'error');
    MODELS.forEach((m, i) => { m.status = 'error'; $(`status-${i}`).innerHTML = '<div class="ps-empty">Not available.</div>'; });
    loadExamples();
    return;
  }
  ort.env.wasm.numThreads = 1;                // more threads need special server settings that static hosts do not have
  ort.env.wasm.wasmPaths = ORT_DIST;
  ort.env.logLevel = 'error';
  try { state.inferno = await (await fetchOk('inferno.json')).json(); } catch (e) {
    console.warn('colour table missing, using grey:', e);
    state.inferno = Array.from({ length: 256 }, (_, i) => [i / 255, i / 255, i / 255]);
  }
  loadExamples();
  // one after the other: the first model is usable while the second is still downloading
  for (let i = 0; i < MODELS.length; i++) await loadModel(i);
}

// What the tests read (tests/web/parity.mjs). Not used by the page itself.
window.pneumoshift = {
  state, MODELS, setPicture,
  async summary() {
    const pic = state.picture;
    if (!pic) return null;
    return {
      inputSha256: await sha256Hex(pic.gray),
      results: pic.results.map((r) => {
        if (!r || r === 'running') return r === null ? null : undefined;
        let hottest = 0, camSum = 0;
        for (let i = 0; i < r.cam.length; i++) { camSum += r.cam[i]; if (r.cam[i] > r.cam[hottest]) hottest = i; }
        const rgb = [0, 0, 0];
        for (let i = 0; i < r.heatmap.length; i += 4) for (let ch = 0; ch < 3; ch++) rgb[ch] += r.heatmap[i + ch];
        return {
          logit: r.logit, probability: r.probability, flagged: r.flagged, oodScore: r.oodScore,
          oodPercentile: r.oodPercentile, oodLevel: r.oodLevel,
          camHottestYX: [Math.floor(hottest / IMG_SIZE), hottest % IMG_SIZE], camMean: camSum / r.cam.length,
          heatmapMeanRGB: rgb.map((v) => v / (r.heatmap.length / 4)),
        };
      }),
    };
  },
};

start();
})();
