// The project's ONE ground-height source (USGS 3DEP bare earth; built by tools/elevation/build_dem.mjs).
// Used by tools/elevation/build_heights.mjs (road node heights for sim + render), the terrain conform raster
// (tools/elevation/build_conform.mjs) and render-side tools that need the same ground (hero block, landscape).
//
//   import { loadGround } from './tools/elevation/ground.mjs';
//   const G = loadGround();            // ~1 GB of Float32 caches, read once
//   G.heightAt(x, z)                   // metres NAVD88 at project x/z (equirectangular about 37.55 N, 122.15 W,
//                                      // x east, z south = data/roads.schema.md); NaN off the data
//   G.heightLL(lat, lon); G.source(x, z) -> '1m' | '13' | null
// Sampling: bilinear on the 2 m lidar grid (San Francisco, USGS 1 m DEM CA_SanFrancisco_B23) where all four cells
// are valid, else bilinear on the 1/3 arc-second (~10 m) seamless grid. The lidar grid is already cross-faded into
// the 1/3" grid over 150 m at its edges, so there is no step between the sources.
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const G13 = { lat0: 38.10, lon0: -122.80, rows: 10800, cols: 13500, step: 1 / 10800 };
export const G1M = { e0: 540000, n0: 4190000, cell: 2, cols: 10000, rows: 10000, zone: 10 };
export const PROJ = { lat0: 37.55, lon0: -122.15, mLon: 88157.883, mLat: 111195.0802 };
export const toLL = (x, z) => [PROJ.lat0 - z / PROJ.mLat, PROJ.lon0 + x / PROJ.mLon];
export const toXZ = (lat, lon) => [(lon - PROJ.lon0) * PROJ.mLon, -(lat - PROJ.lat0) * PROJ.mLat];

/** UTM zone 10N easting/northing (GRS80; Snyder 8-9..8-10). */
export function utmFromLL(lat, lon) {
  const a = 6378137, f = 1 / 298.257222101, e2 = f * (2 - f), ep2 = e2 / (1 - e2), k0 = 0.9996;
  const p = lat * Math.PI / 180, l = (lon + 123) * Math.PI / 180;
  const N = a / Math.sqrt(1 - e2 * Math.sin(p) ** 2), T = Math.tan(p) ** 2, C = ep2 * Math.cos(p) ** 2, A = Math.cos(p) * l;
  const M = a * ((1 - e2 / 4 - 3 * e2 * e2 / 64 - 5 * e2 ** 3 / 256) * p - (3 * e2 / 8 + 3 * e2 * e2 / 32 + 45 * e2 ** 3 / 1024) * Math.sin(2 * p)
    + (15 * e2 * e2 / 256 + 45 * e2 ** 3 / 1024) * Math.sin(4 * p) - (35 * e2 ** 3 / 3072) * Math.sin(6 * p));
  const E = 500000 + k0 * N * (A + (1 - T + C) * A ** 3 / 6 + (5 - 18 * T + T * T + 72 * C - 58 * ep2) * A ** 5 / 120);
  const Nn = k0 * (M + N * Math.tan(p) * (A * A / 2 + (5 - T + 9 * C + 4 * C * C) * A ** 4 / 24 + (61 - 58 * T + T * T + 600 * C - 330 * ep2) * A ** 6 / 720));
  return [E, Nn];
}

/** Bilinear on the 1/3" grid (NaN if any of the four samples is missing). */
export function sample13(g, lat, lon) {
  const c = (lon - G13.lon0) / G13.step - 0.5, r = (G13.lat0 - lat) / G13.step - 0.5;
  if (!(c >= 0 && r >= 0 && c < G13.cols - 1 && r < G13.rows - 1)) return NaN;
  const c0 = Math.floor(c), r0 = Math.floor(r), fc = c - c0, fr = r - r0, k = r0 * G13.cols + c0;
  return (g[k] * (1 - fc) + g[k + 1] * fc) * (1 - fr) + (g[k + G13.cols] * (1 - fc) + g[k + G13.cols + 1] * fc) * fr;
}
export function sample1m(g, E, N) {
  const c = (E - G1M.e0) / G1M.cell - 0.5, r = (G1M.n0 - N) / G1M.cell - 0.5;
  if (!(c >= 0 && r >= 0 && c < G1M.cols - 1 && r < G1M.rows - 1)) return NaN;
  const c0 = Math.floor(c), r0 = Math.floor(r), fc = c - c0, fr = r - r0, k = r0 * G1M.cols + c0;
  return (g[k] * (1 - fc) + g[k + 1] * fc) * (1 - fr) + (g[k + G1M.cols] * (1 - fc) + g[k + G1M.cols + 1] * fc) * fr;
}

const DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../data/dem/ground');
const f32 = (f) => { const b = readFileSync(path.join(DIR, f)); return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4); };

export function loadGround({ lidar = true } = {}) {
  const g13 = f32('g13.f32'), g1 = lidar ? f32('g1m.f32') : null;
  const heightLL = (lat, lon) => {
    if (g1) { const [E, N] = utmFromLL(lat, lon); const v = sample1m(g1, E, N); if (Number.isFinite(v)) return v; }
    return sample13(g13, lat, lon);
  };
  const source = (x, z) => {
    const [lat, lon] = toLL(x, z);
    if (g1) { const [E, N] = utmFromLL(lat, lon); if (Number.isFinite(sample1m(g1, E, N))) return '1m'; }
    return Number.isFinite(sample13(g13, lat, lon)) ? '13' : null;
  };
  return { heightLL, heightAt: (x, z) => heightLL(...toLL(x, z)), source, g13, g1 };
}
