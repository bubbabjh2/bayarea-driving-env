// After build_heights: give traffic controls and gas stations the new node heights (their y is the y of their road
// node; x/z and everything else unchanged).  node tools/elevation/patch_points.mjs [dataDir]
import { readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
const dir = path.resolve(process.argv[2] || 'data');
const xyzOf = (t) => { const m = JSON.parse(readFileSync(path.join(dir, `${t}.meta.json`))), b = readFileSync(path.join(dir, m.file)), s = m.sections.nodeXYZ;
  return new Float32Array(b.buffer.slice(b.byteOffset + s.byteOffset, b.byteOffset + s.byteOffset + s.byteLength)); };
const XYZ = { main: xyzOf('roads'), local: xyzOf('roads-local') };
const packOf = (t) => { const m = JSON.parse(readFileSync(path.join(dir, `${t}.meta.json`))), b = readFileSync(path.join(dir, m.file)), ab = b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
  return { edges: new Uint32Array(ab, m.sections.edges.byteOffset, m.sections.edges.length), poly: new Uint32Array(ab, m.sections.poly.byteOffset, m.sections.poly.length) }; };
const PK = { main: packOf('roads'), local: packOf('roads-local') };
/** y on tier edge e at s metres (horizontal arc length) */
function edgeY(tier, e, s) { const { edges, poly } = PK[tier], A = XYZ[tier], f = edges[8 * e], n = edges[8 * e + 1]; let acc = 0;
  for (let k = 1; k < n; k++) { const a = poly[f + k - 1], b = poly[f + k], L = Math.hypot(A[3 * b] - A[3 * a], A[3 * b + 2] - A[3 * a + 2]);
    if (acc + L >= s || k === n - 1) { const t = L > 0 ? Math.min(1, Math.max(0, (s - acc) / L)) : 0; return A[3 * a + 1] + (A[3 * b + 1] - A[3 * a + 1]) * t; } acc += L; }
  return A[3 * poly[f] + 1]; }
for (const [name, listKey, secXYZ, info, stride, tierIdx, nodeIdx] of [
  ['controls', 'controls', 'controlXYZ', 'controlInfo', 6, 2, 3], ['stations', 'stations', 'stationXYZ', 'stationInfo', 4, 1, 0]]) {
  const doc = JSON.parse(readFileSync(path.join(dir, `${name}.json`))), meta = JSON.parse(readFileSync(path.join(dir, `${name}.meta.json`)));
  const bin = readFileSync(path.join(dir, meta.file)), ab = bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength);
  const xyz = new Float32Array(ab, meta.sections[secXYZ].byteOffset, meta.sections[secXYZ].length), inf = new Uint32Array(ab, meta.sections[info].byteOffset, meta.sections[info].length);
  let n = 0, dmax = 0;
  doc[listKey].forEach((c, i) => {
    if (inf[stride * i + nodeIdx] === 0xFFFFFFFF) { // mid-block control: the height of its first approach at the stop line
      const ap = c.approaches[0], y = edgeY(ap.tier, ap.tierEdge, ap.stopAt); dmax = Math.max(dmax, Math.abs(y - c.y)); c.y = +y.toFixed(2); xyz[3 * i + 1] = y; n++; return; }
    const tier = inf[stride * i + tierIdx] === 0 ? 'main' : 'local', node = inf[stride * i + nodeIdx], A = XYZ[tier];
    if (Math.hypot(A[3 * node] - c.x, A[3 * node + 2] - c.z) > 30) throw new Error(`${name} ${i}: node ${node} is not at the record`);
    const y = A[3 * node + 1]; dmax = Math.max(dmax, Math.abs(y - c.y)); c.y = +y.toFixed(2); xyz[3 * i + 1] = y; n++;
  });
  meta.heights = 'terrain/1 (tools/elevation/patch_points.mjs: y = road node y)';
  writeFileSync(path.join(dir, meta.file), Buffer.from(ab)); writeFileSync(path.join(dir, `${name}.json`), JSON.stringify(doc)); writeFileSync(path.join(dir, `${name}.meta.json`), JSON.stringify(meta, null, 1));
  console.log(name, n, 'max |dy|', dmax.toFixed(1));
}
