// Render ground conformed to the roads: the bare-earth DEM on a 10 m grid over the stream root square, with every
// at-grade road corridor set to the road surface minus 0.15 m (feathered back to the natural ground over 12 m), so the
// landscape terrain and the roads come from the same heights and the road always sits on (never under) the ground.
// Bridges and tunnels are NOT conformed (natural ground under decks / over tunnels). Bay and ocean are exactly 0.
//
//   node --max-old-space-size=12000 tools/elevation/build_conform.mjs [roadsDir=data] [outDir=data/dem/ground]
// Out: ground10.f32   Float32 10240 x 10240, row j = z0 + (j + 0.5) * 10, col i = x0 + (i + 0.5) * 10 (project metres,
//                     x0 -53000, z0 -57900: the data/tiles / data/landscape root square), metres NAVD88
//      ground10.mask   Uint8 same grid: bit0 conformed road corridor (weight 1), bit1 feather (0 < w < 1),
//                     bit2 bridge deck over the cell, bit3 tunnel under the cell, bit4 water (bay / ocean)
//      ground10.json   grid definition + stats
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadGround } from './ground.mjs';
import { scanFill } from '../landscape/raster.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const RD = path.resolve(ROOT, process.argv[2] || 'data'), OUT = path.resolve(ROOT, process.argv[3] || 'data/dem/ground');
mkdirSync(OUT, { recursive: true });
const log = (...a) => console.log(`[${new Date().toISOString().slice(11, 19)}]`, ...a);
const SQ = { x0: -53000, z0: -57900, size: 102400 }, CELL = 10, N = SQ.size / CELL;
const SINK = 0.15, FEATHER = 12, MARGIN = 1.0, LAND_MIN = 0.4, RAISE_MAX = 6;

// ---- natural ground
const G = loadGround();
const H = new Float32Array(N * N);
for (let j = 0; j < N; j++) { const z = SQ.z0 + (j + 0.5) * CELL; for (let i = 0; i < N; i++) { const v = G.heightAt(SQ.x0 + (i + 0.5) * CELL, z); H[j * N + i] = Number.isFinite(v) ? v : 0; } }
log('sampled DEM', N, 'x', N);
// ---- water (data/water.json, same rule as tools/landscape/build_terrain.mjs)
const mask = new Uint8Array(N * N), water = new Uint8Array(N * N);
{
  const wj = JSON.parse(readFileSync(path.join(ROOT, 'data/water.json'), 'utf8'));
  const px = (x) => (x - SQ.x0) / CELL, pz = (z) => (z - SQ.z0) / CELL;
  for (const poly of wj.polygons) {
    const rings = [poly.outer, ...poly.holes].map((r) => { const a = []; for (const [x, z] of r) a.push(px(x), pz(z)); return a; });
    scanFill(rings, N, N, (row, a, b) => { for (let i = a; i < b; i++) water[row * N + i] ^= 1; });
  }
  const bb = wj.meta.bbox;
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) { const x = SQ.x0 + (i + 0.5) * CELL, z = SQ.z0 + (j + 0.5) * CELL;
    if (x > bb.minX + 100 && x < bb.maxX - 100 && z > bb.minZ + 100 && z < bb.maxZ - 100) continue;
    water[j * N + i] = H[j * N + i] <= 0.05 ? 1 : 0; }
}
// ---- roads
const W = new Float32Array(N * N), R = new Float32Array(N * N), D = new Float32Array(N * N).fill(1e9);
let segs = 0;
for (const tier of ['roads', 'roads-local']) {
  const m = JSON.parse(readFileSync(path.join(RD, `${tier}.meta.json`), 'utf8')), b = readFileSync(path.join(RD, m.file));
  const ab = b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength), S = {};
  for (const [k, s] of Object.entries(m.sections)) S[k] = new (s.type === 'Float32' ? Float32Array : Uint32Array)(ab, s.byteOffset, s.length);
  const xyz = S.nodeXYZ, E = S.edges, LW = S.edgeLaneWidth;
  for (let e = 0; e < m.edgeCount; e++) {
    const tw = E[8 * e + 7]; if (tw !== 0xFFFFFFFF && tw < e) continue;
    const flags = E[8 * e + 6], f = E[8 * e], n = E[8 * e + 1];
    const struct = (flags & 2) ? 4 : (flags & 4) ? 8 : 0;
    const half = (E[8 * e + 2] + E[8 * e + 3]) * LW[e] / 2 + MARGIN, reach = struct ? half : half + FEATHER;
    for (let k = 1; k < n; k++) {
      const a = S.poly[f + k - 1], c = S.poly[f + k];
      const ax = xyz[3 * a], ay = xyz[3 * a + 1], az = xyz[3 * a + 2], cx = xyz[3 * c], cy = xyz[3 * c + 1], cz = xyz[3 * c + 2];
      const dx = cx - ax, dz = cz - az, L2 = dx * dx + dz * dz || 1e-9; segs++;
      const i0 = Math.max(0, Math.floor((Math.min(ax, cx) - reach - SQ.x0) / CELL)), i1 = Math.min(N - 1, Math.floor((Math.max(ax, cx) + reach - SQ.x0) / CELL));
      const j0 = Math.max(0, Math.floor((Math.min(az, cz) - reach - SQ.z0) / CELL)), j1 = Math.min(N - 1, Math.floor((Math.max(az, cz) + reach - SQ.z0) / CELL));
      for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) {
        const x = SQ.x0 + (i + 0.5) * CELL, z = SQ.z0 + (j + 0.5) * CELL;
        const t = Math.max(0, Math.min(1, ((x - ax) * dx + (z - az) * dz) / L2)), qx = ax + dx * t, qz = az + dz * t;
        const d = Math.hypot(x - qx, z - qz); if (d > reach) continue;
        const k2 = j * N + i;
        if (struct) { mask[k2] |= struct; continue; }
        // cells the road passes through are always conformed (a 10 m cell can be wider than the road)
        const w = d <= Math.max(half, CELL * 0.71) ? 1 : 1 - (d - Math.max(half, CELL * 0.71)) / FEATHER;
        if (w > W[k2] + 1e-6 || (w >= W[k2] - 1e-6 && d < D[k2])) { W[k2] = w; D[k2] = d; R[k2] = ay + (cy - ay) * t - SINK; }
      }
    }
  }
}
log('road segments', segs);
// ---- compose
let conf = 0, fea = 0, wat = 0, maxCut = 0, maxFill = 0;
const out = new Float32Array(N * N);
for (let k = 0; k < N * N; k++) {
  if (water[k] && W[k] < 1) { out[k] = 0; mask[k] |= 16; wat++; continue; } // bay / ocean (a road corridor over water is a causeway: keep it)
  let h = Math.max(LAND_MIN, H[k]);
  // a road more than 6 m above the natural ground is an embankment / ramp onto a structure the bare-earth DEM does not
  // hold (old water-deck approaches): leave the ground natural there and flag it like a deck
  if (W[k] > 0 && R[k] - h > RAISE_MAX) { mask[k] |= 4; W[k] = 0; }
  if (W[k] > 0) { const r = Math.max(LAND_MIN - SINK, R[k]); const d = r - h; if (W[k] >= 1) { conf++; mask[k] |= 1; if (d > maxFill) maxFill = d; if (-d > maxCut) maxCut = -d; } else { fea++; mask[k] |= 2; } h = h + (r - h) * W[k]; }
  out[k] = h;
}
writeFileSync(path.join(OUT, 'ground10.f32'), Buffer.from(out.buffer));
writeFileSync(path.join(OUT, 'ground10.mask'), Buffer.from(mask.buffer));
const meta = { format: 'bay-ground-conform/1', generated: new Date().toISOString(), roads: path.relative(ROOT, RD) || '.', root: SQ, cell: CELL, n: N,
  files: { height: 'ground10.f32 (Float32 LE, metres NAVD88)', mask: 'ground10.mask (Uint8)' },
  layout: 'row-major; row j centre z = z0 + (j + 0.5) * cell, col i centre x = x0 + (i + 0.5) * cell (project metres: equirectangular about 37.55 N, 122.15 W; x east, z south)',
  rules: { sink: SINK, featherM: FEATHER, corridor: '(lanesFwd + lanesBack) * laneWidth / 2 + 1 m, at least the cells the centreline crosses', bridgesTunnels: 'not conformed (natural ground)', raisedOver6m: 'not conformed, mask bit2', water: 'exactly 0 (data/water.json)', landMin: LAND_MIN },
  maskBits: { road: 1, feather: 2, bridgeDeck: 4, tunnel: 8, water: 16 },
  stats: { conformedCells: conf, featherCells: fea, waterCells: wat, maxRaiseM: +maxFill.toFixed(2), maxLowerM: +maxCut.toFixed(2) },
  source: 'tools/elevation/build_conform.mjs; ground = tools/elevation/ground.mjs (USGS 3DEP 1 m lidar in SF, 1/3 arc-second elsewhere)' };
writeFileSync(path.join(OUT, 'ground10.json'), JSON.stringify(meta, null, 1));
log('done', meta.stats);
