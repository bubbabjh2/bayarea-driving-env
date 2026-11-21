// Mosaic the USGS 3DEP bare-earth DEMs into the project's ground caches (run once after downloading; ~2 min).
//   node tools/elevation/build_dem.mjs
// In:  data/dem/raw/13/USGS_13_n{38,39}w{122,123}.tif   1/3 arc-second (~10 m), NAD83 geographic, NAVD88 m
//      data/dem/raw/1m/USGS_1M_10_x5{4,5}y41{8,9}_CA_SanFrancisco_B23.tif   1 m lidar DEM, UTM 10N (NAD83), NAVD88 m
// Out: data/dem/ground/g13.f32   Float32, rows north->south, G13 grid (see ground.json), NaN = no data
//      data/dem/ground/g1m.f32   Float32 2 m cells (mean of 2x2 lidar pixels), UTM 10N, rows north->south;
//                                cross-faded into the 1/3" value over the last 150 m before its no-data edge
//                                (so the two sources join without a step); NaN = use g13
//      data/dem/ground/ground.json  grid definitions (read by tools/elevation/ground.mjs)
// Untrusted input: tools/landscape/tiff.mjs bounds-checks every offset; values outside [-100, 2000] m are no-data.
import { writeFileSync, existsSync, mkdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { readTiff } from '../landscape/tiff.mjs';
import { G13, G1M, utmFromLL } from './ground.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const RAW = path.join(ROOT, 'data/dem/raw'), OUT = path.join(ROOT, 'data/dem/ground');
mkdirSync(OUT, { recursive: true });
const log = (...a) => console.log(`[${new Date().toISOString().slice(11, 19)}]`, ...a);
const ok = (v) => Number.isFinite(v) && v > -100 && v < 2000;

// ---- 1/3 arc-second mosaic: target pixel centres coincide with the tiles' pixel centres (integer copy)
const g13 = new Float32Array(G13.rows * G13.cols).fill(NaN);
for (const [la, lo] of [[38, 123], [38, 122], [39, 123], [39, 122]]) {
  const f = path.join(RAW, '13', `USGS_13_n${la}w${lo}.tif`);
  if (!existsSync(f)) { log('missing', f); continue; }
  const t = readTiff(f);
  const sx = 1 / 10800, lon0 = t.tie[3], lat0 = t.tie[4];
  let n = 0;
  for (let j = 0; j < G13.rows; j++) {
    const lat = G13.lat0 - (j + 0.5) * sx;
    const r = Math.round((lat0 - lat) / sx - 0.5);
    if (r < 0 || r >= t.height) continue;
    for (let i = 0; i < G13.cols; i++) {
      const lon = G13.lon0 + (i + 0.5) * sx;
      const c = Math.round((lon - lon0) / sx - 0.5);
      if (c < 0 || c >= t.width) continue;
      const v = t.data[r * t.width + c];
      if (ok(v)) { g13[j * G13.cols + i] = v; n++; }
    }
  }
  log('13', path.basename(f), t.width, t.height, 'copied', n);
}
writeFileSync(path.join(OUT, 'g13.f32'), Buffer.from(g13.buffer));
log('g13', G13.cols, 'x', G13.rows);

// ---- 1 m lidar -> 2 m cells
const W = G1M.cols, H = G1M.rows, g1 = new Float32Array(W * H).fill(NaN);
for (const x of [54, 55]) for (const y of [418, 419]) {
  const f = path.join(RAW, '1m', `USGS_1M_10_x${x}y${y}_CA_SanFrancisco_B23.tif`);
  if (!existsSync(f)) { log('missing', f); continue; }
  const t = readTiff(f);
  const e0 = t.tie[3], n0 = t.tie[4]; // upper-left corner of pixel (0,0)
  let n = 0;
  for (let j = 0; j < H; j++) {
    const nTop = G1M.n0 - j * G1M.cell;                  // cell spans [nTop - 2, nTop]
    const r0 = Math.round(n0 - nTop); if (r0 < 0 || r0 + 1 >= t.height) continue;
    for (let i = 0; i < W; i++) {
      const eL = G1M.e0 + i * G1M.cell;
      const c0 = Math.round(eL - e0); if (c0 < 0 || c0 + 1 >= t.width) continue;
      let s = 0, k = 0;
      for (let dr = 0; dr < 2; dr++) for (let dc = 0; dc < 2; dc++) { const v = t.data[(r0 + dr) * t.width + c0 + dc]; if (ok(v)) { s += v; k++; } }
      if (k === 4) { g1[j * W + i] = s / 4; n++; }
    }
  }
  log('1m', path.basename(f), 'cells', n);
}
// distance (m) to the nearest no-data cell (chamfer 3-4), then cross-fade into the 1/3" value
const D = new Float32Array(W * H);
for (let k = 0; k < W * H; k++) D[k] = Number.isNaN(g1[k]) ? 0 : 1e9;
const a = G1M.cell, b = G1M.cell * Math.SQRT2;
for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) { const k = j * W + i; if (!D[k]) continue; let v = D[k];
  if (i > 0) v = Math.min(v, D[k - 1] + a); else v = Math.min(v, a);
  if (j > 0) { v = Math.min(v, D[k - W] + a); if (i > 0) v = Math.min(v, D[k - W - 1] + b); if (i < W - 1) v = Math.min(v, D[k - W + 1] + b); } else v = Math.min(v, a);
  D[k] = v; }
for (let j = H - 1; j >= 0; j--) for (let i = W - 1; i >= 0; i--) { const k = j * W + i; if (!D[k]) continue; let v = D[k];
  if (i < W - 1) v = Math.min(v, D[k + 1] + a); else v = Math.min(v, a);
  if (j < H - 1) { v = Math.min(v, D[k + W] + a); if (i < W - 1) v = Math.min(v, D[k + W + 1] + b); if (i > 0) v = Math.min(v, D[k + W - 1] + b); } else v = Math.min(v, a);
  D[k] = v; }
const { sample13 } = await import('./ground.mjs');
const FADE = 150;
let faded = 0, dsum = 0, dn = 0, dmax = 0;
for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
  const k = j * W + i; if (Number.isNaN(g1[k])) continue;
  const E = G1M.e0 + (i + 0.5) * G1M.cell, N = G1M.n0 - (j + 0.5) * G1M.cell;
  const h13 = sample13(g13, ...llFromUtm(E, N));
  if (!Number.isFinite(h13)) continue;
  if ((j & 15) === 0 && (i & 15) === 0 && D[k] > FADE && g1[k] > 1) { const d = g1[k] - h13; dsum += d; dn++; if (Math.abs(d) > dmax) dmax = Math.abs(d); }
  if (D[k] < FADE) { const w = D[k] / FADE; g1[k] = w * g1[k] + (1 - w) * h13; faded++; }
}
log('1m vs 1/3" mean diff', (dsum / dn).toFixed(2), 'm, max', dmax.toFixed(1), 'm; faded cells', faded);
writeFileSync(path.join(OUT, 'g1m.f32'), Buffer.from(g1.buffer));
writeFileSync(path.join(OUT, 'ground.json'), JSON.stringify({
  format: 'bay-ground/1', generated: new Date().toISOString(),
  g13: { file: 'g13.f32', ...G13, doc: 'Float32 row-major, row j centre lat = lat0 - (j+0.5)/10800, col i centre lon = lon0 + (i+0.5)/10800; NaN = no data' },
  g1m: { file: 'g1m.f32', ...G1M, crs: 'UTM zone 10N (NAD83)', doc: 'Float32 row-major, cell (i,j) spans E [e0+i*cell, +cell), N [n0-(j+1)*cell, n0-j*cell]; NaN = use g13' },
  vertical: 'NAVD88 metres (bare earth)', sampler: 'tools/elevation/ground.mjs (loadGround -> heightAt(x, z) in project metres)',
}, null, 1));
log('done');

function llFromUtm(E, N) {
  // inverse transverse Mercator (Snyder 8-18..8-25), GRS80, zone 10 (lon0 -123)
  const a = 6378137, f = 1 / 298.257222101, e2 = f * (2 - f), ep2 = e2 / (1 - e2), k0 = 0.9996;
  const x = E - 500000, y = N, M = y / k0;
  const mu = M / (a * (1 - e2 / 4 - 3 * e2 * e2 / 64 - 5 * e2 ** 3 / 256));
  const e1 = (1 - Math.sqrt(1 - e2)) / (1 + Math.sqrt(1 - e2));
  const p1 = mu + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * Math.sin(2 * mu) + (21 * e1 * e1 / 16 - 55 * e1 ** 4 / 32) * Math.sin(4 * mu)
    + (151 * e1 ** 3 / 96) * Math.sin(6 * mu) + (1097 * e1 ** 4 / 512) * Math.sin(8 * mu);
  const C1 = ep2 * Math.cos(p1) ** 2, T1 = Math.tan(p1) ** 2, N1 = a / Math.sqrt(1 - e2 * Math.sin(p1) ** 2);
  const R1 = a * (1 - e2) / (1 - e2 * Math.sin(p1) ** 2) ** 1.5, D = x / (N1 * k0);
  const lat = p1 - (N1 * Math.tan(p1) / R1) * (D * D / 2 - (5 + 3 * T1 + 10 * C1 - 4 * C1 * C1 - 9 * ep2) * D ** 4 / 24
    + (61 + 90 * T1 + 298 * C1 + 45 * T1 * T1 - 252 * ep2 - 3 * C1 * C1) * D ** 6 / 720);
  const lon = (D - (1 + 2 * T1 + C1) * D ** 3 / 6 + (5 - 2 * C1 + 28 * T1 - 3 * C1 * C1 + 8 * ep2 + 24 * T1 * T1) * D ** 5 / 120) / Math.cos(p1);
  return [lat * 180 / Math.PI, -123 + lon * 180 / Math.PI];
}
void utmFromLL;
