// Headless/headed Chrome runner for the GPU pages (no bundler, no package.json deps).
//
//   node test/gpu/run.mjs [page] [--headless=new|shell|false] [--timeout=ms] [--query=a=1&b=2] [--shot=file.png]
//                         [--wait=ms --probe='js expr' --every=ms]   (pages without window.__done)
//                         [--in=file,file] (served at /__in/<basename>) [--out=dir] (page PUTs / POSTs (append) / GETs /__out/<sub/path>)
//
// page defaults to test/gpu/equivalence.html. Serves the repo root with a tiny static
// server, opens the page in puppeteer's Chrome with WebGPU flags, waits until the page
// sets window.__done = true (or document.title starts with PASS/FAIL), prints the page
// log and window.__result as JSON, exits 0 on PASS.
//
// Chrome: $CHROME_PATH, else the installed /Applications/Google Chrome.app. Always launched
// with a mock keychain, a throwaway profile and a throwaway HOME under $GPU_SCRATCH (or the OS
// temp dir), so macOS never shows a keychain prompt and the sandbox never needs ~/Library.
// puppeteer is resolved from: node_modules, $PUPPETEER_PATH, or the npx cache
// (~/.npm/_npx/*/node_modules/puppeteer; install with
// `npx --yes puppeteer@23.11.1 browsers install chrome`).
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const args = process.argv.slice(2);
const opt = (k, d) => { const a = args.find(s => s.startsWith(`--${k}=`)); return a ? a.slice(k.length + 3) : d; };
const page = args.find(a => !a.startsWith('--')) || 'test/gpu/equivalence.html';
const headlessArg = opt('headless', 'new');
const timeout = +opt('timeout', 600000);
const query = opt('query', '');
const INPUTS = new Map(opt('in', '').split(',').filter(Boolean).map((p) => [path.basename(p), path.resolve(p)]));
const OUTDIR = opt('out', '') ? path.resolve(opt('out', '')) : null;
if (OUTDIR) fs.mkdirSync(OUTDIR, { recursive: true });

async function loadPuppeteer() {
  const cands = [];
  if (process.env.PUPPETEER_PATH) cands.push(process.env.PUPPETEER_PATH);
  cands.push(path.join(ROOT, 'node_modules/puppeteer'));
  const npx = path.join(os.homedir(), '.npm/_npx');
  if (fs.existsSync(npx)) for (const d of fs.readdirSync(npx)) cands.push(path.join(npx, d, 'node_modules/puppeteer'));
  for (const c of cands) {
    if (!fs.existsSync(path.join(c, 'package.json'))) continue;
    const req = createRequire(path.join(c, 'package.json'));
    const entry = req.resolve('puppeteer');
    const mod = await import(pathToFileURL(entry).href);
    return mod.default || mod;
  }
  throw new Error('puppeteer not found: run `npx --yes puppeteer@23.11.1 browsers install chrome`');
}

const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json', '.wgsl': 'text/plain', '.bin': 'application/octet-stream', '.css': 'text/css', '.png': 'image/png' };
function serve() {
  return new Promise((res) => {
    const srv = http.createServer((req, resp) => {
      const u = decodeURIComponent(new URL(req.url, 'http://x').pathname);
      // --in=a,b: input files served at /__in/<basename>; --out=dir: PUT /__out/<name> writes <dir>/<name>
      if (u.startsWith('/__in/')) {
        const f = INPUTS.get(u.slice(6));
        if (!f) { resp.writeHead(404); resp.end('404'); return; }
        resp.writeHead(200, { 'Content-Type': 'application/octet-stream', 'Cache-Control': 'no-store' });
        fs.createReadStream(f).pipe(resp); return;
      }
      // /__out/<sub/path>: PUT writes <dir>/<sub/path> (subdirectories created), GET reads it back (404 if
      // missing; the RL loop polls for the learner's chunk-<k>/weights.json and <dir>/stop)
      if (u.startsWith('/__out/') && OUTDIR) {
        const rel = path.normalize(u.slice(7)).replace(/^([/\\.])+/, '');
        const fp = path.join(OUTDIR, rel);
        if (!fp.startsWith(path.resolve(OUTDIR) + path.sep)) { resp.writeHead(403); resp.end(); return; }
        if (req.method === 'DELETE') { try { fs.rmSync(fp, { force: true }); } catch (e) { /* ignore */ } resp.writeHead(200); resp.end('ok'); return; }
        // POST appends to <dir>/<sub/path> (streamed telemetry: the page never holds a whole events file, docs/scale-crashes.md)
        if (req.method === 'POST') {
          fs.mkdirSync(path.dirname(fp), { recursive: true });
          const ws = fs.createWriteStream(fp, { flags: 'a' });
          req.pipe(ws); ws.on('finish', () => { resp.writeHead(200); resp.end('ok'); }); ws.on('error', (e) => { resp.writeHead(500); resp.end(String(e)); }); return;
        }
        if (req.method === 'PUT') {
          fs.mkdirSync(path.dirname(fp), { recursive: true });
          const tmp = fp + '.part', ws = fs.createWriteStream(tmp);
          req.pipe(ws); ws.on('finish', () => { fs.renameSync(tmp, fp); resp.writeHead(200); resp.end('ok'); }); ws.on('error', (e) => { resp.writeHead(500); resp.end(String(e)); }); req.on('error', () => { try { ws.destroy(); } catch (e) { /* ignore */ } }); return;
        }
        if (!fs.existsSync(fp) || fs.statSync(fp).isDirectory()) { resp.writeHead(404, { 'Cache-Control': 'no-store' }); resp.end('404'); return; }
        resp.writeHead(200, { 'Content-Type': MIME[path.extname(fp)] || 'application/octet-stream', 'Cache-Control': 'no-store' });
        fs.createReadStream(fp).pipe(resp); return;
      }
      let f = path.join(ROOT, u);
      if (!f.startsWith(ROOT)) { resp.writeHead(403); resp.end(); return; }
      if (fs.existsSync(f) && fs.statSync(f).isDirectory()) f = path.join(f, 'index.html');
      fs.readFile(f, (err, data) => {
        if (err) { resp.writeHead(404); resp.end('404'); return; }
        resp.writeHead(200, { 'Content-Type': MIME[path.extname(f)] || 'application/octet-stream', 'Cache-Control': 'no-store' });
        resp.end(data);
      });
    });
    srv.listen(0, '127.0.0.1', () => res(srv));
  });
}

const puppeteer = await loadPuppeteer();
const srv = await serve();
const port = srv.address().port;
const headless = headlessArg === 'false' ? false : headlessArg === 'shell' ? 'shell' : true;
const SYS_CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const executablePath = process.env.CHROME_PATH || (fs.existsSync(SYS_CHROME) ? SYS_CHROME : undefined);
console.log('chrome:', executablePath);
const tmpRoot = fs.mkdtempSync(path.join(process.env.GPU_SCRATCH || os.tmpdir(), 'gpu-chrome-'));
const fakeHome = path.join(tmpRoot, 'home');
fs.mkdirSync(path.join(fakeHome, 'Library', 'Application Support'), { recursive: true });
if (!executablePath) throw new Error('no Chrome: set CHROME_PATH');
const browser = await puppeteer.launch({
  headless, executablePath,
  userDataDir: process.env.CHROME_PROFILE || path.join(tmpRoot, 'profile'),
  // the sandbox cannot write ~/Library: give Chrome a throwaway HOME (CFFIXED_USER_HOME on macOS)
  env: { ...process.env, HOME: fakeHome, CFFIXED_USER_HOME: fakeHome },
  protocolTimeout: timeout + 60000,
  dumpio: !!process.env.GPU_DUMPIO,   // Chrome stderr (crash / OOM diagnosis)
  args: ['--use-mock-keychain', '--password-store=basic', '--no-first-run', '--no-default-browser-check',
    '--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU', '--use-angle=metal', '--ignore-gpu-blocklist',
    '--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows',
    '--window-size=1280,800', '--enable-precise-memory-info', ...(process.env.GPU_JSFLAGS ? [`--js-flags=${process.env.GPU_JSFLAGS}`] : [])],
  defaultViewport: { width: 1280, height: 800 },
});
console.log('browser version:', await browser.version());
let code = 1;
try {
  const p = await browser.newPage();
  p.on('console', (m) => console.log('[page]', m.text()));
  p.on('pageerror', (e) => console.log('[pageerror]', e.message));
  p.on('error', (e) => console.log('[pagecrash]', e.message));
  p.on('close', () => console.log('[pageclose]'));
  const url = `http://127.0.0.1:${port}/${page}${query ? (page.includes('?') ? '&' : '?') + query : ''}`;
  console.log('open', url, 'headless=', headless);
  await p.goto(url, { waitUntil: 'load' });
  // --stackAfter=ms: pause the page's main thread (CDP Debugger.pause) after ms and print its JS stack (hang diagnosis)
  const stackAfter = +opt('stackAfter', 0);
  if (stackAfter) {
    const cdp = await p.createCDPSession(); await cdp.send('Debugger.enable');
    cdp.on('Debugger.paused', async (ev) => {
      console.log('STACK ' + ev.callFrames.slice(0, 15).map((f) => `${f.functionName || '(anon)'} @ ${f.url.split('/').slice(-2).join('/')}:${f.location.lineNumber + 1}`).join('\n      '));
      await cdp.send('Debugger.resume').catch(() => {});
    });
    setTimeout(() => cdp.send('Debugger.pause').catch((e) => console.log('pause failed', e.message)), stackAfter);
    setTimeout(() => cdp.send('Debugger.pause').catch(() => {}), stackAfter + 5000);
  }
  // --wait=ms: pages without a done flag (stream.html): wait, then evaluate --probe=<js expression> as the result
  const wait = +opt('wait', 0), probe = opt('probe', '');
  if (wait) {
    const every = +opt('every', 0);
    for (let t = 0; t < wait; t += every || wait) {
      await new Promise((r) => setTimeout(r, every || wait));
      if (every && probe) console.log('PROBE', t + every, JSON.stringify(await p.evaluate(probe)));
    }
  } else await p.waitForFunction(() => window.__done === true || /^(PASS|FAIL)/.test(document.title), { timeout, polling: 500 });
  const result = probe ? await p.evaluate(probe) : await p.evaluate(() => window.__result ?? document.getElementById('out')?.textContent ?? null);
  const shot = opt('shot', '');
  if (shot) { await p.screenshot({ path: shot }); console.log('screenshot', shot); }
  const title = await p.title();
  console.log('RESULT', JSON.stringify(result, null, 1));
  console.log('TITLE', title);
  code = /^PASS/.test(title) || (result && result.ok) ? 0 : 1;
} catch (e) { console.log('runner error:', e.message); }
await browser.close();
srv.close();
process.exit(code);
