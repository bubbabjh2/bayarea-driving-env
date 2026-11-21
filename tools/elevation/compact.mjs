// Sampler for the committed compact ground (data/dem/compact/ground20.*, tools/elevation/build_compact.mjs).
//   import { loadCompactGround } from './tools/elevation/compact.mjs';
//   const g = loadCompactGround();  g.heightAt(x, z)  // metres NAVD88 at project (x east, z south) metres; NaN outside
//   g.maskAt(x, z)                                   // bits: road 1, feather 2, bridge deck 4, tunnel 8, water 16
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../data/dem/compact');

/** gzip'd row-delta Int16 decimetres -> Float32Array metres (n x n) */
export function decodeCompact(gz, n) {
  const raw = gunzipSync(gz), d = new Int16Array(raw.buffer, raw.byteOffset, raw.byteLength / 2), h = new Float32Array(n * n);
  for (let j = 0; j < n; j++) { let v = 0; for (let i = 0; i < n; i++) { v += d[j * n + i]; h[j * n + i] = v / 10; } }
  return h;
}

/** bilinear sample of a cell-centred grid {x0, z0, cell, n}; NaN outside the outer cell centres */
export function bilinear(H, g, x, z) {
  const c = (x - g.x0) / g.cell - 0.5, r = (z - g.z0) / g.cell - 0.5;
  if (!(c >= 0 && r >= 0 && c <= g.n - 1 && r <= g.n - 1)) return NaN;
  const c0 = Math.min(g.n - 2, Math.floor(c)), r0 = Math.min(g.n - 2, Math.floor(r)), fc = c - c0, fr = r - r0, k = r0 * g.n + c0;
  return (H[k] * (1 - fc) + H[k + 1] * fc) * (1 - fr) + (H[k + g.n] * (1 - fc) + H[k + g.n + 1] * fc) * fr;
}

export function loadCompactGround(dir = DIR) {
  const meta = JSON.parse(readFileSync(path.join(dir, 'ground20.json'), 'utf8'));
  const n = meta.n, H = decodeCompact(readFileSync(path.join(dir, 'ground20.dh16.gz')), n);
  const M = new Uint8Array(gunzipSync(readFileSync(path.join(dir, 'ground20.mask.gz'))));
  const g = { x0: meta.root.x0, z0: meta.root.z0, cell: meta.cell, n };
  return {
    meta, heights: H, mask: M,
    heightAt: (x, z) => bilinear(H, g, x, z),
    maskAt: (x, z) => { const i = Math.floor((x - g.x0) / g.cell), j = Math.floor((z - g.z0) / g.cell); return i >= 0 && j >= 0 && i < n && j < n ? M[j * n + i] : 0; },
  };
}
