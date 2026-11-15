/* Records the short demo clip that the README shows, and one screenshot.

   It serves web/ on a local port (models from web/models/, see scripts/08_export_web.py), opens the page in
   headless Chromium, clicks the example "Adult (46), normal" and scrolls over the result. A drawn mouse pointer
   is added to the page for the recording only, because a headless browser shows none. Nothing else is changed.

     node tests/web/record_demo.mjs <output folder>      # needs Node, Playwright and ffmpeg

   Output: demo.mp4, demo.gif and example_adult_65.png. The README shows them from docs/media/.
*/
import { createServer } from 'node:http';
import { mkdir, readFile, rename, rm } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { dirname, extname, join, normalize, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

const { chromium } = createRequire(import.meta.url)('playwright');   // also finds a global install (NODE_PATH)
const WEB = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', 'web');
const OUT = resolve(process.argv[2] || 'demo_clip');
const SIZE = { width: 1200, height: 760 };
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json',
  '.png': 'image/png', '.woff2': 'font/woff2', '.wasm': 'application/wasm' };

const server = createServer(async (req, res) => {
  try {
    const path = normalize(decodeURIComponent(new URL(req.url, 'http://x').pathname)).replace(/^[/\\]+/, '') || 'index.html';
    const file = join(WEB, path);
    if (!file.startsWith(WEB)) throw new Error('outside');
    res.writeHead(200, { 'content-type': TYPES[extname(file)] || 'application/octet-stream' });
    res.end(await readFile(file));
  } catch { res.writeHead(404); res.end('not found'); }
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const url = `http://127.0.0.1:${server.address().port}/?models=local`;
const proxy = process.env.HTTPS_PROXY ? { server: process.env.HTTPS_PROXY, bypass: '127.0.0.1,localhost' } : undefined;
const browser = await chromium.launch({ proxy });
await mkdir(OUT, { recursive: true });

const ready = (page) => page.waitForFunction(() => window.pneumoshift
  && window.pneumoshift.MODELS.every((m) => m.status === 'ready'), null, { timeout: 180000 });
const results = (page) => page.waitForFunction(() => {
  const pic = window.pneumoshift.state.picture;
  return pic && pic.results.every((r) => r !== undefined && r !== 'running');
}, null, { timeout: 60000 });
const scrollTo = async (page, top, wait = 1000) => {
  await page.evaluate((y) => window.scrollTo({ top: y, behavior: 'smooth' }), top);
  await page.waitForTimeout(wait);
};

// ---- the clip
const context = await browser.newContext({ viewport: SIZE, recordVideo: { dir: OUT, size: SIZE } });
const page = await context.newPage();
const started = Date.now();
await page.goto(url);
await ready(page);
await page.evaluate(() => {          // a pointer and a click mark, for the recording only
  const dot = document.createElement('div');
  dot.style.cssText = 'position:fixed;left:0;top:0;width:22px;height:22px;z-index:99999;pointer-events:none;'
    + 'background:url("data:image/svg+xml;utf8,<svg xmlns=\'http://www.w3.org/2000/svg\' viewBox=\'0 0 22 22\'>'
    + '<path d=\'M3 2l14 8-6 1.6L8.4 18z\' fill=\'white\' stroke=\'black\' stroke-width=\'1.4\'/></svg>") no-repeat;';
  const ring = document.createElement('div');
  ring.style.cssText = 'position:fixed;width:34px;height:34px;margin:-17px 0 0 -17px;border:3px solid #f4c64f;'
    + 'border-radius:50%;z-index:99998;pointer-events:none;opacity:0;transition:opacity .5s, transform .5s;';
  document.body.append(ring, dot);
  addEventListener('mousemove', (e) => { dot.style.transform = `translate(${e.clientX - 3}px, ${e.clientY - 2}px)`; }, true);
  addEventListener('mousedown', (e) => {
    ring.style.transition = 'none'; ring.style.left = `${e.clientX}px`; ring.style.top = `${e.clientY}px`;
    ring.style.opacity = '1'; ring.style.transform = 'scale(.6)';
    requestAnimationFrame(() => { ring.style.transition = 'opacity .6s, transform .6s'; ring.style.opacity = '0'; ring.style.transform = 'scale(1.6)'; });
  }, true);
});
await page.mouse.move(760, 330);
const clipStart = (Date.now() - started) / 1000;
await page.waitForTimeout(1400);
await scrollTo(page, 600);
const box = await page.locator('.ps-example', { hasText: 'Adult (46), normal' }).boundingBox();
await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2, { steps: 30 });
await page.waitForTimeout(350);
await page.mouse.down(); await page.mouse.up();
await results(page);
await page.waitForTimeout(700);
await scrollTo(page, 150);
await page.mouse.move(980, 420, { steps: 25 });
await page.waitForTimeout(4200);
await scrollTo(page, 700);
await page.waitForTimeout(3400);
const clipEnd = (Date.now() - started) / 1000;
const video = page.video();
await context.close();
await rename(await video.path(), join(OUT, 'demo.webm'));

// ---- one screenshot: the example that both models get wrong
const still = await (await browser.newContext({ viewport: SIZE, deviceScaleFactor: 2 })).newPage();
await still.goto(url);
await ready(still);
await still.locator('.ps-example', { hasText: 'Adult (65), lung opacity' }).click();
await results(still);
await still.waitForTimeout(700);
await still.evaluate(() => window.scrollTo(0, 150));
await still.waitForTimeout(300);
await still.screenshot({ path: join(OUT, 'example_adult_65.png') });
console.log(JSON.stringify(await still.evaluate(() => window.pneumoshift.summary().results
  ? window.pneumoshift.summary().results : window.pneumoshift.summary()), null, 0).slice(0, 900));
await browser.close();
server.close();

// ---- cut the loading time away, then write an MP4 and a GIF
const cut = ['-ss', clipStart.toFixed(2), '-t', (clipEnd - clipStart).toFixed(2), '-i', join(OUT, 'demo.webm'), '-an'];
execFileSync('ffmpeg', ['-y', '-loglevel', 'error', ...cut, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '24',
  '-movflags', '+faststart', join(OUT, 'demo.mp4')]);
execFileSync('ffmpeg', ['-y', '-loglevel', 'error', ...cut, '-vf',
  'fps=6,scale=880:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff:max_colors=96[p];'
  + '[b][p]paletteuse=dither=none:diff_mode=rectangle', join(OUT, 'demo.gif')]);
await rm(join(OUT, 'demo.webm'));          // the raw recording, with the loading time still in it
console.log(`clip: ${(clipEnd - clipStart).toFixed(1)} s, written to ${OUT}`);
