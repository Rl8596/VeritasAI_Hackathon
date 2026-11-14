/* Does the browser demo (web/) give the same results as the Python code?

     python scripts/08_export_web.py        # writes tests/web/reference.json and web/models/
     node tests/web/parity.mjs              # needs Node and Playwright (npm install playwright)

   The test serves web/ on a local port, opens it in headless Chromium with ?models=local, clicks every
   example, chooses every test picture through the file dialog, and compares what the page computed with
   tests/web/reference.json (written by the Python code path of the Gradio app):

     - the 224 x 224 picture the model receives must be identical, byte for byte
     - logit, familiarity distance and heatmap must agree within the small limits below
     - the decision (flagged or not) and the warning level must be the same

   Optional: --screenshots <folder> saves pictures of the page on a desktop and a phone screen. */
import { createServer } from 'node:http';
import { readFile, mkdir } from 'node:fs/promises';
import { extname, join, normalize, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

const { chromium, devices } = createRequire(import.meta.url)('playwright');   // also finds a global install (NODE_PATH)

const HERE = dirname(fileURLToPath(import.meta.url));
const WEB = resolve(HERE, '..', '..', 'web');
const LIMIT = { logit: 1e-3, probability: 1e-4, oodScore: 1e-2, oodPercentile: 1e-9, camMean: 1e-4, heatmapMeanRGB: 0.05 };
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json',
  '.png': 'image/png', '.onnx': 'application/octet-stream', '.bin': 'application/octet-stream' };

const shotsAt = process.argv.indexOf('--screenshots');
const shots = shotsAt > 0 ? resolve(process.argv[shotsAt + 1]) : null;

const server = createServer(async (req, res) => {
  try {
    const path = normalize(decodeURIComponent(new URL(req.url, 'http://x').pathname)).replace(/^[/\\]+/, '') || 'index.html';
    const file = join(WEB, path);
    if (!file.startsWith(WEB)) throw new Error('outside');
    const body = await readFile(file);
    res.writeHead(200, { 'content-type': TYPES[extname(file)] || 'application/octet-stream', 'content-length': body.length });
    res.end(body);
  } catch { res.writeHead(404); res.end('not found'); }
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const url = `http://127.0.0.1:${server.address().port}/?models=local`;

const reference = JSON.parse(await readFile(join(HERE, 'reference.json'), 'utf8')).models;
const names = Object.keys(reference);
const proxy = process.env.HTTPS_PROXY ? { server: process.env.HTTPS_PROXY, bypass: '127.0.0.1,localhost' } : undefined;
const browser = await chromium.launch({ proxy });
let failures = 0, checks = 0;
const worst = Object.fromEntries(Object.keys(LIMIT).map((k) => [k, 0]));

function compare(file, modelName, want, got) {
  const fail = (what) => { failures++; console.log(`  FAIL ${file} / ${modelName}: ${what}`); };
  if (!got) return fail('no result');
  const near = (key, a, b) => {
    const d = Math.abs(a - b);
    worst[key] = Math.max(worst[key], d);
    if (!(d <= LIMIT[key])) fail(`${key} differs by ${d.toExponential(2)} (python ${a}, browser ${b})`);
  };
  near('logit', want.logit, got.logit);
  near('probability', want.probability, got.probability);
  near('oodScore', want.ood_score, got.oodScore);
  near('oodPercentile', want.ood_percentile, got.oodPercentile);
  near('camMean', want.cam_mean, got.camMean);
  want.heatmap_mean_rgb.forEach((v, ch) => near('heatmapMeanRGB', v, got.heatmapMeanRGB[ch]));
  if (want.flagged !== got.flagged) fail(`flagged: python ${want.flagged}, browser ${got.flagged}`);
  if (want.ood_level !== got.oodLevel) fail(`warning level: python ${want.ood_level}, browser ${got.oodLevel}`);
  if (want.cam_hottest_yx.join() !== got.camHottestYX.join()) fail(`hottest point: python ${want.cam_hottest_yx}, browser ${got.camHottestYX}`);
  checks++;
}

async function openPage(context) {
  const page = await context.newPage();
  page.on('pageerror', (e) => { failures++; console.log('  page error:', e.message); });
  page.on('console', (m) => { if (m.type() === 'error') console.log('  console error:', m.text()); });
  await page.goto(url);
  await page.waitForFunction(() => window.pneumoshift && window.pneumoshift.MODELS.every((m) => m.status === 'ready' || m.status === 'error'),
    null, { timeout: 180000 });
  const status = await page.evaluate(() => window.pneumoshift.MODELS.map((m) => `${m.key}: ${m.status}`));
  if (status.some((s) => !s.endsWith('ready'))) { failures++; console.log('  models not ready:', status.join(', ')); }
  return page;
}

async function waitForResults(page) {
  await page.waitForFunction(() => {
    const pic = window.pneumoshift.state.picture;
    return pic && pic.results.every((r) => r !== undefined && r !== 'running');
  }, null, { timeout: 60000 });
  return page.evaluate(() => window.pneumoshift.summary());
}

try {
  const context = await browser.newContext({ viewport: { width: 1366, height: 800 } });
  const page = await openPage(context);
  if (shots) { await mkdir(shots, { recursive: true }); await page.screenshot({ path: join(shots, 'desktop_empty.png'), fullPage: true }); }

  const files = Object.keys(reference[names[0]]);
  for (const file of files) {
    if (file.startsWith('examples/')) {
      const index = await page.evaluate((f) => window.pneumoshift.state.examples.findIndex((e) => `examples/${e.file}` === f), file);
      if (index < 0) { failures++; console.log(`  FAIL ${file}: not among the examples on the page`); continue; }
      await page.click(`.ps-example[data-index="${index}"]`);
    } else {
      await page.setInputFiles('#file', join(HERE, file));
    }
    await page.waitForFunction((seqBefore) => window.pneumoshift.state.picture && window.pneumoshift.state.picture.seq !== seqBefore,
      await page.evaluate(() => (window.pneumoshift.state.lastChecked || 0)), { timeout: 30000 });
    const got = await waitForResults(page);
    await page.evaluate(() => { window.pneumoshift.state.lastChecked = window.pneumoshift.state.picture.seq; });
    const wantSha = reference[names[0]][file].input_sha256;
    if (got.inputSha256 !== wantSha) { failures++; console.log(`  FAIL ${file}: the picture given to the model differs from Python`); }
    names.forEach((name, i) => compare(file, name, reference[name][file], got.results[i]));
    if (shots && file === 'examples/adult_lung_opacity_1.png') {
      await page.screenshot({ path: join(shots, 'desktop_adult.png'), fullPage: true });
      await page.screenshot({ path: join(shots, 'desktop_adult_fold.png') });
    }
    if (shots && file === 'examples/not_an_xray_document.png') await page.screenshot({ path: join(shots, 'desktop_not_xray.png'), fullPage: true });
  }

  // a file that is not a picture must give a readable message, not a crash
  await page.setInputFiles('#file', { name: 'notes.png', mimeType: 'image/png', buffer: Buffer.from('this is not a picture') });
  await page.waitForSelector('.ps-notice--error', { timeout: 10000 });
  checks++;
  await context.close();

  if (shots) {
    for (const [label, device] of [['phone', devices['iPhone 13']], ['tablet', { viewport: { width: 900, height: 1100 } }],
      ['dark', { viewport: { width: 1366, height: 800 }, colorScheme: 'dark' }]]) {
      const ctx = await browser.newContext(device);
      const p = await openPage(ctx);
      await p.screenshot({ path: join(shots, `${label}_empty.png`), fullPage: true });
      await p.click('.ps-example[data-index="8"]');
      await waitForResults(p);
      await p.waitForTimeout(600);
      await p.screenshot({ path: join(shots, `${label}_adult.png`), fullPage: true });
      await ctx.close();
    }
  }
} finally {
  await browser.close();
  server.close();
}

console.log(`largest differences, Python vs browser: ${Object.entries(worst).map(([k, v]) => `${k} ${v.toExponential(1)}`).join(', ')}`);
console.log(failures ? `FAILED: ${failures} problem(s) in ${checks} checks` : `OK: ${checks} checks, browser and Python agree`);
process.exit(failures ? 1 : 0);
