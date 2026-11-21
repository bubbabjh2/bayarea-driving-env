// Inspect road grades of a roads pack: worst segments per class (with names), or the profile of one edge.
//   node tools/elevation/inspect.mjs [dataDir] worst <class> [n] [minLen]
//   node tools/elevation/inspect.mjs [dataDir] edge <mergedEdgeId>
import { readFileSync } from 'node:fs';
import path from 'node:path';
const [dir = 'data', cmd = 'worst', a1 = 'motorway', a2 = '15', a3 = '10'] = process.argv.slice(2);
const T = ['roads', 'roads-local'].map((t) => {
  const m = JSON.parse(readFileSync(path.join(dir, t + '.meta.json'))), b = readFileSync(path.join(dir, m.file));
  const S = {}; for (const [k, s] of Object.entries(m.sections)) S[k] = new ({ Float32: Float32Array, Uint32: Uint32Array })[s.type](b.buffer, b.byteOffset + s.byteOffset, s.length);
  const j = JSON.parse(readFileSync(path.join(dir, m.json || t + '.json'))); return { m, S, j };
});
const Em = T[0].m.edgeCount, CL = T[0].m.classesAll;
const ll = (x, z) => `${(37.55 - z / 111195.0802).toFixed(5)},${(-122.15 + x / 88157.883).toFixed(5)}`;
function edge(id) { const t = id >= Em ? 1 : 0, e = t ? id - Em : id, { S, j } = T[t]; const f = S.edges[8 * e], n = S.edges[8 * e + 1];
  const pts = []; for (let k = 0; k < n; k++) { const v = S.poly[f + k]; pts.push([S.nodeXYZ[3 * v], S.nodeXYZ[3 * v + 1], S.nodeXYZ[3 * v + 2]]); }
  return { t, e, cls: CL[S.edges[8 * e + 4]], name: j.edges[e].name || j.edges[e].ref, bridge: j.edges[e].bridge, tunnel: j.edges[e].tunnel, layer: j.edges[e].layer, pts }; }
if (cmd === 'edge') { const E = edge(+a1); console.log(E.cls, E.name, { bridge: E.bridge, tunnel: E.tunnel, layer: E.layer });
  let s = 0; E.pts.forEach((p, k) => { if (k) s += Math.hypot(p[0] - E.pts[k - 1][0], p[2] - E.pts[k - 1][2]); console.log(s.toFixed(1), p[1].toFixed(2), k ? (100 * (p[1] - E.pts[k - 1][1]) / Math.max(1e-6, Math.hypot(p[0] - E.pts[k - 1][0], p[2] - E.pts[k - 1][2]))).toFixed(1) + '%' : '', ll(p[0], p[2])); }); }
else { const out = [];
  for (let id = 0; id < Em + T[1].m.edgeCount; id++) { const t = id >= Em ? 1 : 0, e = t ? id - Em : id; if (CL[T[t].S.edges[8 * e + 4]] !== a1) continue;
    const E = edge(id); let best = 0, at = null; for (let k = 1; k < E.pts.length; k++) { const p = E.pts[k], q = E.pts[k - 1], L = Math.hypot(p[0] - q[0], p[2] - q[2]); const g = Math.abs(p[1] - q[1]) / Math.max(+a3, L); if (g > best) { best = g; at = q; } }
    out.push([best, id, E.name, E.bridge ? 'B' : E.tunnel ? 'T' : '', at && ll(at[0], at[2]), at && at[1].toFixed(1)]); }
  out.sort((a, b) => b[0] - a[0]); for (const r of out.slice(0, +a2)) console.log((100 * r[0]).toFixed(1) + '%', ...r.slice(1)); }
if (cmd === 'named') { // node inspect.mjs dir named "Filbert Street" [latMin latMax lonMin lonMax]
  const [la0, la1, lo0, lo1] = process.argv.slice(5).map(Number);
  for (let id = 0; id < Em + T[1].m.edgeCount; id++) { const t = id >= Em ? 1 : 0, e = t ? id - Em : id; const j = T[t].j.edges[e]; if (j.name !== a1) continue;
    if (j.twin >= 0 && j.twin < e) continue;
    const E = edge(id), p0 = E.pts[0], p1 = E.pts[E.pts.length - 1]; const [la, lo] = ll(p0[0], p0[2]).split(',').map(Number);
    if (la0 && !(la >= la0 && la <= la1 && lo >= lo0 && lo <= lo1)) continue;
    let L = 0, mx = 0; for (let k = 1; k < E.pts.length; k++) { const q = E.pts[k - 1], p = E.pts[k], d = Math.hypot(p[0] - q[0], p[2] - q[2]); L += d; mx = Math.max(mx, Math.abs(p[1] - q[1]) / Math.max(d, 1)); }
    console.log(id, ll(p0[0], p0[2]), '->', ll(p1[0], p1[2]), 'L', L.toFixed(0), 'y', p0[1].toFixed(1), '->', p1[1].toFixed(1), 'avg', (100 * (p1[1] - p0[1]) / L).toFixed(1) + '%', 'max', (100 * mx).toFixed(1) + '%', E.pts.length + ' pts'); } }
