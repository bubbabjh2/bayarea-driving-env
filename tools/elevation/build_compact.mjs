// Compact, committed copy of the road-conformed ground (data/dem/ground/ground10.*, 10 m, 400 MB) so a fresh clone has
// terrain without the 2 GB USGS download:
//   node tools/elevation/build_compact.mjs
// Out: data/dem/compact/ground20.dh16.gz  Int16 LE, 5120 x 5120 cells of 20 m (mean of each 2 x 2 block of ground10),
//        decimetres NAVD88, row-wise delta coded (cell i stores h[i] - h[i-1], h[-1] = 0), gzip -9
//      data/dem/compact/ground20.mask.gz Uint8 same grid: OR of the four ground10 mask bits (road 1, feather 2,
//        bridge deck 4, tunnel 8, water 16)
//      data/dem/compact/ground20.json    grid definition + measured error vs ground10
// Read it with tools/elevation/compact.mjs (loadCompactGround().heightAt(x, z)).
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { gzipSync } from 'node:zlib';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { decodeCompact, bilinear } from './compact.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const G = path.join(ROOT, 'data/dem/ground'), OUT = path.join(ROOT, 'data/dem/compact');
mkdirSync(OUT, { recursive: true });
const meta10 = JSON.parse(readFileSync(path.join(G, 'ground10.json'), 'utf8'));
const b = readFileSync(path.join(G, 'ground10.f32')), h10 = new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
const m10 = readFileSync(path.join(G, 'ground10.mask'));
const N10 = meta10.n, N = N10 / 2, CELL = meta10.cell * 2;
const dm = new Int16Array(N * N), mask = new Uint8Array(N * N);
for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
  const k = 2 * j * N10 + 2 * i;
  const v = (h10[k] + h10[k + 1] + h10[k + N10] + h10[k + N10 + 1]) / 4;
  dm[j * N + i] = Math.max(-32768, Math.min(32767, Math.round(v * 10)));
  mask[j * N + i] = m10[k] | m10[k + 1] | m10[k + N10] | m10[k + N10 + 1];
}
const delta = new Int16Array(N * N);
for (let j = 0; j < N; j++) { let prev = 0; for (let i = 0; i < N; i++) { const v = dm[j * N + i]; delta[j * N + i] = v - prev; prev = v; } }
const hz = gzipSync(Buffer.from(delta.buffer), { level: 9 }), mz = gzipSync(Buffer.from(mask.buffer), { level: 9 });
writeFileSync(path.join(OUT, 'ground20.dh16.gz'), hz); writeFileSync(path.join(OUT, 'ground20.mask.gz'), mz);
// error of the compact grid (bilinear at every ground10 cell centre) vs ground10
const H = decodeCompact(hz, N);
let se = 0, mx = 0, n = 0, seLand = 0, nLand = 0;
const half = { x0: meta10.root.x0, z0: meta10.root.z0, cell: CELL, n: N };
for (let j = 0; j < N10; j++) for (let i = 0; i < N10; i++) {
  const x = meta10.root.x0 + (i + 0.5) * meta10.cell, z = meta10.root.z0 + (j + 0.5) * meta10.cell;
  const d = bilinear(H, half, x, z) - h10[j * N10 + i]; if (!Number.isFinite(d)) continue; se += d * d; n++; if (Math.abs(d) > mx) mx = Math.abs(d);
  if (!(m10[j * N10 + i] & 16)) { seLand += d * d; nLand++; }
}
const meta = {
  format: 'bay-ground-compact/1', from: 'data/dem/ground/ground10.f32 (tools/elevation/build_conform.mjs)', root: meta10.root, cell: CELL, n: N,
  files: { height: 'ground20.dh16.gz (Int16 LE decimetres NAVD88, row-wise delta, gzip)', mask: 'ground20.mask.gz (Uint8, OR of the 2x2 ground10 mask bits)' },
  layout: 'row-major; row j centre z = z0 + (j + 0.5) * cell, col i centre x = x0 + (i + 0.5) * cell (project metres, x east, z south)',
  maskBits: meta10.maskBits, vertical: 'NAVD88 metres (bare earth, road corridors conformed to the road surface - 0.15 m)',
  errorVsGround10: { rmsM: +Math.sqrt(se / n).toFixed(3), rmsLandM: +Math.sqrt(seLand / Math.max(1, nLand)).toFixed(3), maxAbsM: +mx.toFixed(2) },
  sampler: 'tools/elevation/compact.mjs loadCompactGround().heightAt(x, z)',
  licence: 'Derived from USGS 3DEP (public domain) and OpenStreetMap road data (ODbL 1.0)',
};
writeFileSync(path.join(OUT, 'ground20.json'), JSON.stringify(meta, null, 1) + '\n');
console.log('ground20:', (hz.length / 1e6).toFixed(1), 'MB heights,', (mz.length / 1e6).toFixed(2), 'MB mask; error', JSON.stringify(meta.errorVsGround10));
