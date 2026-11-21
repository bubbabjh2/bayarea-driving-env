// Terrain heightfield for the landscape (render only).
//   node tools/landscape/build_terrain.mjs
// Source: the shared road-conformed ground grid data/dem/ground/ground10.f32 (+ .mask, + ground10.json), 10 m cells over
// the stream root square, built by the elevation build (tools/elevation/build_conform.mjs) from USGS 3DEP (1 m lidar in SF,
// 1/3" elsewhere): water = 0, land >= 0.4 m, at-grade road corridors = road surface - 0.15 m (feathered over 12 m),
// bridges / tunnels left natural. Roads in data/roads*.bin carry the same heights, so the ground meets them exactly.
// Out:
//   data/landscape/terrain.bin.gz   global 4096^2 (25 m, box-filtered): Int16 decimetres, delta-coded along each row
//   data/landscape/shore.bin.gz     Uint8 1024^2 distance offshore (16 m units), for the water colour gradient
//   data/landscape/h10/<tx>_<tz>.bin.gz  10 m detail tiles on the 32 x 32 land-cover grid (3.2 km): 324^2 samples
//        (320 + 2-sample gutter each side, cell centres), header 'H10\0', f32 base (m), u16 n, u16 gutter, f32 unit (m), then Int16 residuals of v = round((h - base) / unit) after the planar predictor left + up - upleft (first row: left, first column: up) (gzip)
//   data/landscape/terrain.json     meta
// Without ground10 (or with --compact) the committed 20 m compact grid is used instead.
import { readFileSync, writeFileSync, mkdirSync, openSync, readSync, closeSync, rmSync, existsSync } from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { ROOT_DIR, OUT_DIR, ROOT, log } from './common.mjs';

const GDIR = path.join(ROOT_DIR, 'data/dem/ground');
// Input: the full 10 m ground (npm run build:elevation) if present, else the committed compact 20 m grid
// (data/dem/compact, tools/elevation/compact.mjs) resampled bilinearly to 10 m.
const FULL = existsSync(path.join(GDIR, 'ground10.f32')) && !process.argv.includes('--compact');
let gmeta, GN, GC, G, M;
if (FULL) {
  gmeta = JSON.parse(readFileSync(path.join(GDIR, 'ground10.json'), 'utf8'));
  GN = gmeta.n; GC = gmeta.cell; G = new Float32Array(GN * GN);
  const fd = openSync(path.join(GDIR, 'ground10.f32'), 'r');
  const buf = Buffer.from(G.buffer);
  let off = 0; while (off < buf.length) { const n = readSync(fd, buf, off, Math.min(1 << 26, buf.length - off), off); if (n <= 0) throw new Error('short read'); off += n; }
  closeSync(fd);
  M = new Uint8Array(readFileSync(path.join(GDIR, 'ground10.mask')).buffer.slice(0));
} else {
  const { loadCompactGround } = await import('../elevation/compact.mjs');
  const c = loadCompactGround();
  gmeta = { root: c.meta.root, n: c.meta.n * 2, cell: c.meta.cell / 2, source: 'compact ' + c.meta.format + ' (20 m), resampled to 10 m' };
  GN = gmeta.n; GC = gmeta.cell; G = new Float32Array(GN * GN); M = new Uint8Array(GN * GN);
  const n = c.meta.n;
  for (let j = 0; j < GN; j++) for (let i = 0; i < GN; i++) {
    const x = gmeta.root.x0 + (i + 0.5) * GC, z = gmeta.root.z0 + (j + 0.5) * GC;
    let h = c.heightAt(x, z);
    const ci = i >> 1, cj = j >> 1;
    if (!Number.isFinite(h)) h = c.heights[cj * n + ci];
    M[j * GN + i] = c.mask[cj * n + ci];
    G[j * GN + i] = (M[j * GN + i] & 16) && h < 0.2 ? 0 : h;
  }
}
if (gmeta.root.x0 !== ROOT.x0 || gmeta.root.z0 !== ROOT.z0 || gmeta.root.size !== ROOT.size) throw new Error('ground root differs from the landscape root');
let bad = 0; for (let k = 0; k < G.length; k++) if (!Number.isFinite(G[k]) || G[k] < -100 || G[k] > 2000) { G[k] = 0; bad++; }
log(FULL ? 'ground10 loaded' : 'compact ground (20 m -> 10 m) loaded', GN, 'bad', bad);
mkdirSync(OUT_DIR, { recursive: true });

// ---- global 25 m (box filter over the 2.5 x 2.5 cell footprint)
const N = 4096, CELL = ROOT.size / N, r = CELL / GC;
const hq = new Int16Array(N * N);
let mn = Infinity, mx = -Infinity;
for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
  const a0 = i * r, a1 = (i + 1) * r, b0 = j * r, b1 = (j + 1) * r;
  let s = 0, w = 0;
  for (let b = Math.floor(b0); b < Math.ceil(b1); b++) {
    const wb = Math.min(b + 1, b1) - Math.max(b, b0);
    for (let a = Math.floor(a0); a < Math.ceil(a1); a++) { const wa = Math.min(a + 1, a1) - Math.max(a, a0); s += G[b * GN + a] * wa * wb; w += wa * wb; }
  }
  const v = s / w; hq[j * N + i] = Math.round(v * 10);
  if (v > 0.2) { if (v < mn) mn = v; if (v > mx) mx = v; }
}
const hd = new Int16Array(N * N);
for (let j = 0; j < N; j++) { let p = 0; for (let i = 0; i < N; i++) { const v = hq[j * N + i]; hd[j * N + i] = v - p; p = v; } }
writeFileSync(path.join(OUT_DIR, 'terrain.bin.gz'), zlib.gzipSync(Buffer.from(hd.buffer), { level: 9 }));

// ---- shore distance (water = mask bit4, or height 0)
{
  const W = new Uint8Array(N * N);
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) { const k = Math.floor((j + 0.5) * r) * GN + Math.floor((i + 0.5) * r); W[j * N + i] = (M[k] & 16) || G[k] === 0 ? 1 : 0; }
  // drop land specks (DEM noise, single rocks) so they do not make shallow halos offshore: land needs >= 5 land neighbours
  for (let pass = 0; pass < 2; pass++) for (let j = 1; j < N - 1; j++) for (let i = 1; i < N - 1; i++) {
    const k = j * N + i; if (W[k]) continue;
    let n = 0; for (let b = -1; b <= 1; b++) for (let a = -1; a <= 1; a++) if (!W[k + b * N + a]) n++;
    if (n < 6) W[k] = 1;
  }
  const D = new Float32Array(N * N); for (let k = 0; k < N * N; k++) D[k] = W[k] ? 1e9 : 0;
  const a = CELL, b = CELL * Math.SQRT2;
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) { const k = j * N + i; if (!D[k]) continue; let v = D[k];
    if (i > 0) v = Math.min(v, D[k - 1] + a); if (j > 0) { v = Math.min(v, D[k - N] + a); if (i > 0) v = Math.min(v, D[k - N - 1] + b); if (i < N - 1) v = Math.min(v, D[k - N + 1] + b); } D[k] = v; }
  for (let j = N - 1; j >= 0; j--) for (let i = N - 1; i >= 0; i--) { const k = j * N + i; if (!D[k]) continue; let v = D[k];
    if (i < N - 1) v = Math.min(v, D[k + 1] + a); if (j < N - 1) { v = Math.min(v, D[k + N] + a); if (i < N - 1) v = Math.min(v, D[k + N + 1] + b); if (i > 0) v = Math.min(v, D[k + N - 1] + b); } D[k] = v; }
  const DN = 1024, ds = N / DN, dq = new Uint8Array(DN * DN);
  for (let j = 0; j < DN; j++) for (let i = 0; i < DN; i++) { let s = 0; for (let y = 0; y < ds; y++) for (let x = 0; x < ds; x++) s += Math.min(D[(j * ds + y) * N + i * ds + x], 4080); dq[j * DN + i] = Math.round(s / (ds * ds) / 16); }
  writeFileSync(path.join(OUT_DIR, 'shore.bin.gz'), zlib.gzipSync(Buffer.from(dq.buffer), { level: 9 }));
}

// ---- 10 m detail tiles
const TD = path.join(OUT_DIR, 'h10');
if (existsSync(TD)) rmSync(TD, { recursive: true });
mkdirSync(TD, { recursive: true });
const T = 3200, NT = ROOT.size / T, PER = T / GC, GUT = 2, TN = PER + 2 * GUT;
const tiles = []; let bytes = 0;
for (let tz = 0; tz < NT; tz++) for (let tx = 0; tx < NT; tx++) {
  const s = new Float32Array(TN * TN); let lo = Infinity, hi = -Infinity, nonzero = false, roads = 0;
  for (let j = 0; j < TN; j++) for (let i = 0; i < TN; i++) {
    const gi = Math.min(GN - 1, Math.max(0, tx * PER - GUT + i)), gj = Math.min(GN - 1, Math.max(0, tz * PER - GUT + j));
    const v = G[gj * GN + gi]; s[j * TN + i] = v; if (v < lo) lo = v; if (v > hi) hi = v; if (v !== 0) nonzero = true; if (M[gj * GN + gi] & 1) roads++;
  }
  if (!nonzero || roads === 0) continue;   // no road corridor in the tile: the 25 m global field is enough there
  // 2 cm units around the tile's mid height (Int16: +-655 m); deltas along rows are stored in Int32 range-safe via two passes
  const UNIT = 0.05, base = (lo + hi) / 2;
  if ((hi - lo) / 2 / UNIT > 32000) throw new Error(`tile ${tx}_${tz} range ${hi - lo} m too large`);
  const q = new Int16Array(TN * TN);
  // planar predictor: residual = v - (left + up - upleft)   (Int16 wrap; the loader inverts it)
  const v = new Int32Array(TN * TN);
  for (let k = 0; k < TN * TN; k++) v[k] = Math.round((s[k] - base) / UNIT);
  for (let j = 0; j < TN; j++) for (let i = 0; i < TN; i++) {
    const L = i ? v[j * TN + i - 1] : 0, U = j ? v[(j - 1) * TN + i] : 0, UL = i && j ? v[(j - 1) * TN + i - 1] : 0;
    const pred = i && j ? L + U - UL : i ? L : U;
    q[j * TN + i] = v[j * TN + i] - pred;
  }
  const head = Buffer.alloc(16); head.write('H10\0', 0, 'latin1'); head.writeFloatLE(base, 4); head.writeUInt16LE(TN, 8); head.writeUInt16LE(GUT, 10); head.writeFloatLE(UNIT, 12);
  const gz = zlib.gzipSync(Buffer.concat([head, Buffer.from(q.buffer)]), { level: 9 });
  writeFileSync(path.join(TD, `${tx}_${tz}.bin.gz`), gz);
  tiles.push([tx, tz]); bytes += gz.length;
}
const meta = {
  format: 'bay-terrain/2', root: ROOT, n: N, cell: CELL, conformed: true,
  layout: 'terrain.bin.gz: Int16 dm, row deltas (N*N, row j = z0 + (j+0.5)*cell); shore.bin.gz: Uint8 1024^2 (16 m units); h10/: see header',
  shoreN: 1024, shoreUnit: 16,
  detail: { tile: T, grid: NT, cell: GC, n: TN, gutter: GUT, tiles, bytes },
  heightRange: [+mn.toFixed(1), +mx.toFixed(1)],
  source: (FULL ? 'data/dem/ground/ground10.f32 (' + (gmeta.source || '') + ')' : gmeta.source) + ', USGS 3DEP public domain; road-conformed',
  generated: new Date().toISOString(),
};
writeFileSync(path.join(OUT_DIR, 'terrain.json'), JSON.stringify(meta, null, 1));
log('terrain', meta.heightRange, 'detail tiles', tiles.length, (bytes / 1e6).toFixed(1), 'MB');
