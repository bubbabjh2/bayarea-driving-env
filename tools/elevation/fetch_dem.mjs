// Download the exact USGS 3DEP tiles pinned in tools/elevation/dem-sources.json into data/dem/raw/ (gitignored).
//   node tools/elevation/fetch_dem.mjs                 download missing / wrong tiles (resumes partial .part files)
//   node tools/elevation/fetch_dem.mjs --from <dir>    copy from a local cache laid out like data/dem/raw (13/, 1m/)
//   node tools/elevation/fetch_dem.mjs --check         only verify what is on disk
// Every file is verified against the pinned byte size and sha256; a mismatch is an error (USGS replaced the tile).
import { createReadStream, createWriteStream, existsSync, mkdirSync, statSync, renameSync, copyFileSync, readFileSync, unlinkSync } from 'node:fs';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const SRC = JSON.parse(readFileSync(path.join(ROOT, 'tools/elevation/dem-sources.json'), 'utf8'));
const RAW = path.resolve(ROOT, process.env.DEM_RAW || 'data/dem/raw');
const argv = process.argv.slice(2);
const from = argv.includes('--from') ? path.resolve(argv[argv.indexOf('--from') + 1]) : null;
const checkOnly = argv.includes('--check');

const sha256 = (f) => new Promise((res, rej) => { const h = createHash('sha256'); createReadStream(f).on('data', (d) => h.update(d)).on('end', () => res(h.digest('hex'))).on('error', rej); });
const okFile = async (f, t) => existsSync(f) && statSync(f).size === t.bytes && (await sha256(f)) === t.sha256;

async function download(t, dst) {
  const part = dst + '.part';
  let have = existsSync(part) ? statSync(part).size : 0;
  if (have > t.bytes) { unlinkSync(part); have = 0; }
  for (let attempt = 1; attempt <= 5 && have < t.bytes; attempt++) {
    const r = await fetch(t.url, { headers: have ? { Range: `bytes=${have}-` } : {} });
    if (!(r.status === 200 || r.status === 206)) throw new Error(`${t.url}: HTTP ${r.status}`);
    if (r.status === 200 && have) have = 0;          // server ignored the range: start over
    try { await pipeline(Readable.fromWeb(r.body), createWriteStream(part, { flags: have ? 'a' : 'w' })); }
    catch (e) { console.warn(`  interrupted (${e.message}), retry ${attempt}`); }
    have = statSync(part).size;
  }
  renameSync(part, dst);
}

let bad = 0;
for (const t of SRC.tiles) {
  const dst = path.join(RAW, t.path);
  mkdirSync(path.dirname(dst), { recursive: true });
  if (await okFile(dst, t)) { console.log('ok      ', t.path); continue; }
  if (checkOnly) { console.log('MISSING ', t.path); bad++; continue; }
  if (from) { const s = path.join(from, t.path); if (!existsSync(s)) throw new Error(`--from: ${s} not found`); console.log('copy    ', t.path); copyFileSync(s, dst); }
  else { console.log('download', t.path, `(${(t.bytes / 1e6).toFixed(0)} MB)`); await download(t, dst); }
  if (!(await okFile(dst, t))) { console.error('CHECKSUM MISMATCH', t.path, '(expected sha256', t.sha256 + ')'); bad++; }
  else console.log('verified', t.path);
}
if (bad) { console.error(`${bad} tile(s) missing or not matching dem-sources.json`); process.exit(1); }
console.log(`all ${SRC.tiles.length} tiles verified in ${path.relative(ROOT, RAW)}`);
