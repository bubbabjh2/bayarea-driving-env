// Outer ring of the landscape (render only): the 307.2 km square around the data root (what the 95 km zoom-out and
// whole-Bay map views see past the 102.4 km data square). 2048^2 (150 m/px) RGBA PNG: RGB = natural clay colour with a
// soft baked hillshade, A = water. From the USGS 3DEP 1" tiles in data/dem/raw (public domain); sea = DEM <= 0.05 m,
// no DEM coverage = open Pacific to the west, plain land elsewhere.
//   node tools/landscape/build_outer.mjs   -> data/landscape/outer.png (+ outer.json)
import { readdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { readTiff } from './tiff.mjs';
import { encodePNG } from './png.mjs';
import { ROOT_DIR, OUT_DIR, ROOT, toLL, log } from './common.mjs';

const SIZE = ROOT.size * 3, X0 = ROOT.x0 - ROOT.size, Z0 = ROOT.z0 - ROOT.size, N = 2048, C = SIZE / N;
const demDir = path.join(ROOT_DIR, 'data/dem/raw');
// 1" tiles (USGS_1_*.tif) if present in data/dem/raw, else the pinned 1/3" tiles of the elevation build (data/dem/raw/13)
const pick = (dir, re) => { try { return readdirSync(dir).filter((f) => re.test(f)).map((f) => path.join(dir, f)); } catch { return []; } };
let demFiles = pick(demDir, /^USGS_1_n\d+w\d+\.tif$/);
if (!demFiles.length) demFiles = pick(path.join(demDir, '13'), /^USGS_13_n\d+w\d+\.tif$/);
if (!demFiles.length) throw new Error('no DEM tiles in data/dem/raw (run npm run fetch:dem)');
const dems = demFiles.map((fp) => { const f = path.basename(fp); const t = readTiff(fp); log('dem', f); return { W: t.width, H: t.height, d: t.data, sx: t.scale[0], sy: t.scale[1], lon0: t.tie[3], lat0: t.tie[4] }; });
function demAt(lat, lon) {
  for (const t of dems) {
    const c = (lon - t.lon0) / t.sx - 0.5, r = (t.lat0 - lat) / t.sy - 0.5;
    if (c < 0 || r < 0 || c > t.W - 1.001 || r > t.H - 1.001) continue;
    const c0 = Math.floor(c), r0 = Math.floor(r), fc = c - c0, fr = r - r0, g = (rr, cc) => { const v = t.d[rr * t.W + cc]; return v < -1000 || !Number.isFinite(v) ? 0 : v; };
    return (g(r0, c0) * (1 - fc) + g(r0, c0 + 1) * fc) * (1 - fr) + (g(r0 + 1, c0) * (1 - fc) + g(r0 + 1, c0 + 1) * fc) * fr;
  }
  return lon < -122.45 ? 0 : 30;   // no coverage: Pacific to the west, plain land to the east
}
const H = new Float32Array(N * N);
for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) { const [lat, lon] = toLL(X0 + (i + 0.5) * C, Z0 + (j + 0.5) * C); H[j * N + i] = demAt(lat, lon); }
const hex = (h) => [(h >> 16) & 255, (h >> 8) & 255, h & 255];
const lerp = (a, b, t) => a.map((v, i) => v + (b[i] - v) * Math.max(0, Math.min(1, t)));
const img = new Uint8Array(N * N * 4);
for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
  const k = j * N + i, h = H[k], o = k * 4;
  if (h <= 0.05) { img.set([134, 183, 191, 255], o); continue; }
  const hx = (H[j * N + Math.min(N - 1, i + 1)] - H[j * N + Math.max(0, i - 1)]) / (2 * C), hz = (H[Math.min(N - 1, j + 1) * N + i] - H[Math.max(0, j - 1) * N + i]) / (2 * C);
  const north = Math.max(-1, Math.min(1, -hz * 6)), slope = Math.min(1, Math.hypot(hx, hz) * 6);
  let c = lerp(hex(0xb7bd85), hex(0xcdbf88), (h - 20) / 180);
  c = lerp(c, hex(0xb9b98a), (h - 350) / 400);
  c = lerp(c, hex(0x9fb177), north * 0.55 * slope + 0.15 * north);
  // soft baked hillshade (light from the east-southeast, morning)
  const n = [-hx * 2.5, 1, -hz * 2.5], L = Math.hypot(...n), sun = [0.55, 0.6, 0.35], sl = Math.hypot(...sun);
  const shade = Math.max(0.6, Math.min(1.3, ((n[0] * sun[0] + n[1] * sun[1] + n[2] * sun[2]) / (L * sl) + 0.45) / (sun[1] / sl + 0.45)));
  img.set([...c.map((v) => Math.max(0, Math.min(255, Math.round(v * shade)))), 0], o);
}
writeFileSync(path.join(OUT_DIR, 'outer.png'), encodePNG(img, N, N, 4, 9));
writeFileSync(path.join(OUT_DIR, 'outer.json'), JSON.stringify({ format: 'bay-outer/1', x0: X0, z0: Z0, size: SIZE, n: N, channels: 'RGB clay colour with baked hillshade (sRGB), A = water', source: 'USGS 3DEP 1 arc-second (public domain)', generated: new Date().toISOString() }));
log('outer done');
