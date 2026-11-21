// JS oracle for bayenv_world::nav (Router, guidance) and bayenv_gpu::pack::pack_routes: routes random OD pairs with
// src/sim/route.js makeRouter (plain, avoid set, penalty map; weights 1.25 and 1) and packs routes with
// src/gpu/worldPack.js packRoutes (router results + scatter-style random walks + edge cases). Also times both.
//   node --max-old-space-size=8000 rust/tools/oracle/nav_dump.mjs <dataDir> <outDir> [--n=3000] [--seed=5]
// Out: <outDir>/nav.json
import { writeFileSync, mkdirSync } from 'node:fs';
import path from 'node:path';

const args = process.argv.slice(2);
const [dataDir, outDir] = args.filter((a) => !a.startsWith('--'));
const opt = (k, d) => { const a = args.find((s) => s.startsWith(`--${k}=`)); return a ? a.slice(k.length + 3) : d; };
mkdirSync(outDir, { recursive: true });
const { loadWorld } = await import('../../../src/sim/data.js');
const { makeRouter, largestSCC } = await import('../../../src/sim/route.js');
const { packRoutes } = await import('../../../src/gpu/worldPack.js');

const t0 = Date.now();
const { graph: g } = await loadWorld({ base: dataDir.endsWith('/') ? dataDir : dataDir + '/', controls: false, stations: false });
console.log(`world ${Date.now() - t0} ms`);
const N0 = +opt('n', 3000);
let rs = (+opt('seed', 5) * 2654435761) >>> 0 || 1;
const rnd = () => { rs ^= rs << 13; rs ^= rs >>> 17; rs ^= rs << 5; rs >>>= 0; return rs / 4294967296; };
const scc = largestSCC(g); const nodes = []; for (let i = 0; i < g.nodeCount; i++) if (scc[i]) nodes.push(i);
const pickNode = () => nodes[Math.floor(rnd() * nodes.length)];
const E = g.edges.length;

const r125 = makeRouter(g, { weight: 1.25, cacheMax: 0 }), r1 = makeRouter(g, { weight: 1, cacheMax: 0 });
const queries = [];
const od = []; for (let i = 0; i < N0; i++) od.push([pickNode(), pickNode()]);
let t = performance.now();
for (const [s, q] of od) { const r = r125(s, q); queries.push({ s, g: q, w: 1.25, r: r ? Array.from(r) : null }); }
const msPlain125 = (performance.now() - t) / od.length;
const od1 = od.slice(0, Math.floor(N0 / 4));
t = performance.now();
for (const [s, q] of od1) { const r = r1(s, q); queries.push({ s, g: q, w: 1, r: r ? Array.from(r) : null }); }
const msPlain1 = (performance.now() - t) / od1.length;
// avoid: every 3rd edge of the plain route + 20 random edges; penalty: route edges (k * 0.25 s, some 0) + random edges
const base = queries.slice(0, Math.floor(N0 / 4));
let tA = 0, tP = 0;
for (const q of base) {
  if (!q.r || !q.r.length) continue;
  const avoid = q.r.filter((_, k) => k % 3 === 1); for (let k = 0; k < 20; k++) avoid.push(Math.floor(rnd() * E));
  let t1 = performance.now(); const ra = r125(q.s, q.g, new Set(avoid)); tA += performance.now() - t1;
  queries.push({ s: q.s, g: q.g, w: 1.25, avoid, r: ra ? Array.from(ra) : null });
  const pen = q.r.map((e) => [e, Math.floor(rnd() * 240) * 0.25]); for (let k = 0; k < 50; k++) pen.push([Math.floor(rnd() * E), Math.floor(rnd() * 40) * 0.25]);
  t1 = performance.now(); const rp = r125(q.s, q.g, null, new Map(pen)); tP += performance.now() - t1;
  queries.push({ s: q.s, g: q.g, w: 1.25, pen, r: rp ? Array.from(rp) : null });
}
// pack: router results + random walks (scatter-like, some U-turns) + edge cases
const routes = queries.filter((q) => q.r && q.r.length).map((q) => q.r);
for (let r = 0; r < N0; r++) {
  let cur = Math.floor(rnd() * E); const route = [cur]; const L = 1 + Math.floor(rnd() * 200);
  for (let k = 1; k < L; k++) {
    const ec = g.edges[cur], a = g.outStart[ec.to], b = g.outStart[ec.to + 1]; if (a === b) break;
    let best = -1, bestScore = -Infinity;
    for (let q = a; q < b; q++) {
      const eo = g.outEdges[q]; if (eo === ec.reverse && rnd() < 0.9) continue;
      let dh = g.edges[eo].headingStart - ec.headingEnd; while (dh > Math.PI) dh -= 2 * Math.PI; while (dh < -Math.PI) dh += 2 * Math.PI;
      const score = -Math.abs(dh) * 1.2 + g.edges[eo].rank * 0.15 + rnd() * 1.5;
      if (score > bestScore) { bestScore = score; best = eo; }
    }
    if (best < 0) break; route.push(best); cur = best;
  }
  routes.push(route);
}
routes.push([0], [5, -1, 7], [E - 1, 0, 1], [3, 3]);
const offset = new Uint32Array(routes.length + 1); for (let r = 0; r < routes.length; r++) offset[r + 1] = offset[r] + routes[r].length;
const flat = new Int32Array(offset[routes.length]); routes.forEach((r, k) => flat.set(r, offset[k]));
t = performance.now(); const words = packRoutes(g, flat, offset); const msPackPool = performance.now() - t;
t = performance.now(); for (let r = 0; r < routes.length; r++) packRoutes(g, flat.subarray(offset[r], offset[r + 1]), Uint32Array.of(0, offset[r + 1] - offset[r])); const msPackEach = (performance.now() - t) / routes.length;
const timings = { msPlain125, msPlain1, msAvoid: tA / base.length, msPenalty: tP / base.length, msPackPool, msPackEach, packRoutes: routes.length, packEntries: flat.length };
console.log(JSON.stringify(timings));
writeFileSync(path.join(outDir, 'nav.json'), JSON.stringify({ queries, flat: Array.from(flat), offset: Array.from(offset), words: Array.from(words), timings }));
console.log(`queries ${queries.length}, routes ${routes.length}, ${Date.now() - t0} ms`);
