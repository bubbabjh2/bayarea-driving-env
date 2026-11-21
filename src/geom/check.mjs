// node src/geom/check.mjs [dataDir]  - continuity checks for the shared lane geometry.
// 1) every strict hop: each continuing lane's centre at the end of e vs the start of en
// 2) every vertex: lane centre just before / after each resampled vertex
// 3) sim lat continuity across hops (mapLatStrict) and the max lateral jump overall
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { RoadSet } from '../roadpack.js';
import { graphFromRoadSets, buildLaneGeometry, buildJunctions, hopMap, LANE_W } from './index.js';

const dir = process.argv[2] || new URL('../../data/', import.meta.url).pathname;
const TYPES = { Float32: Float32Array, Uint32: Uint32Array };
function load(metaName) {
  const meta = JSON.parse(readFileSync(path.join(dir, metaName), 'utf8'));
  const buf = readFileSync(path.join(dir, meta.file));
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  const arrays = {};
  for (const [name, s] of Object.entries(meta.sections)) arrays[name === 'nodeXYZ' ? 'xyz' : name] = new TYPES[s.type](ab, s.byteOffset, s.length);
  return new RoadSet(meta, arrays);
}
const t0 = Date.now();
const main = load('roads.meta.json'), local = load('roads-local.meta.json');
const G = graphFromRoadSets(main, local);
const LG = buildLaneGeometry(G);
console.log(`graph ${G.nodeCount} nodes ${G.edgeCount} edges; geometry ${LG.stats.vertices} vertices, ${LG.stats.strictHops} strict hops in ${LG.stats.groups} node groups, anchor conflicts ${LG.stats.anchorConflicts}; ${Date.now() - t0} ms`);

const a = {}, b = {};
let hopMax = 0, hopN = 0, hopBad = 0, worst = null;
for (let e = 0; e < G.edgeCount; e++) {
  for (const en of G.outEdges(G.to(e))) {
    if (en === G.twin[e]) continue;
    const h = hopMap(G, e, en);
    if (!h.strict) continue;
    for (let l = 0; l < G.lanes[e]; l++) {
      const l2 = l - h.shift;
      if (l2 < 0 || l2 >= G.lanes[en]) continue;
      LG.lanePoint(e, G.length[e], l, a);
      LG.lanePoint(en, 0, l2, b);
      const d = Math.hypot(a.x - b.x, a.z - b.z);
      hopN++;
      if (d > hopMax) { hopMax = d; worst = [e, en, l, l2, d]; }
      if (d >= 0.05) hopBad++;
    }
  }
}
console.log(`strict hops: ${hopN} lane continuations, max jump ${hopMax.toFixed(4)} m, >= 0.05 m: ${hopBad}`, worst ? `worst e${worst[0]}->e${worst[1]} lane ${worst[2]}->${worst[3]}` : '');

let vMax = 0, vN = 0, vBad = 0;
for (let e = 0; e < G.edgeCount; e++) {
  const v0 = LG.vFirst[e], v1 = LG.vFirst[e + 1] - 1;
  for (let i = v0 + 1; i < v1; i++) {
    const s = LG.vs[i];
    for (const l of [0, G.lanes[e] - 1]) {
      LG.lanePoint(e, s - 1e-4, l, a); LG.lanePoint(e, s + 1e-4, l, b);
      const d = Math.hypot(a.x - b.x, a.z - b.z) - 2e-4;
      vN++; if (d > vMax) vMax = d; if (d >= 0.05) vBad++;
    }
  }
}
console.log(`vertices: ${vN} lane samples, max jump ${vMax.toFixed(4)} m, >= 0.05 m: ${vBad}`);

// lane width sanity: neighbouring lane centres 3.7 m apart (on straights) and no inverted ribbons
let inv = 0, short = 0;
for (let e = 0; e < G.edgeCount; e++) {
  for (const s of LG.sampleS(e)) { const bd = LG.bounds(e, s); if (bd.right < bd.left - 1e-6) inv++; }
  if (G.length[e] < 60 && (LG.tapR[2 * e] || LG.tapR[2 * e + 1] || LG.tapL[2 * e] || LG.tapL[2 * e + 1])) short++;
}
console.log(`inverted pavement samples: ${inv}; tapers on edges shorter than 60 m: ${short}`);
// 4) junction connectors: position / heading continuity with the lanes they join, max turn per step
const J = buildJunctions(LG);
let jPos = 0, jHead = 0, jStep = 0, nm = 0, sharp = 0, headBad = 0;
const c = {};
for (let m = 0; m < J.moveCount; m++) {
  const d = J.describe(m);
  LG.lanePoint(d.inEdge, d.sIn, d.inLane, a); LG.lanePoint(d.outEdge, d.sOut, d.outLane, b);
  J.arcPoint(m, 0, c); jPos = Math.max(jPos, Math.hypot(c.x - a.x, c.z - a.z)); const h0 = Math.acos(Math.min(1, c.hx * a.hx + c.hz * a.hz)); jHead = Math.max(jHead, h0);
  if (h0 > Math.PI / 9) headBad++;
  J.arcPoint(m, d.length, c); jPos = Math.max(jPos, Math.hypot(c.x - b.x, c.z - b.z)); jHead = Math.max(jHead, Math.acos(Math.min(1, c.hx * b.hx + c.hz * b.hz)));
  const first = J.mInfo[7 * m + 4], np = J.mInfo[7 * m + 5], P = J.pts;
  for (let k = 1; k < np - 1; k++) {
    const ux = P[4 * (first + k)] - P[4 * (first + k - 1)], uz = P[4 * (first + k) + 2] - P[4 * (first + k - 1) + 2];
    const vx = P[4 * (first + k + 1)] - P[4 * (first + k)], vz = P[4 * (first + k + 1) + 2] - P[4 * (first + k) + 2];
    const lu = Math.hypot(ux, uz), lv = Math.hypot(vx, vz);
    if (lu > 0.5 && lv > 0.5) {
      const st = Math.acos(Math.max(-1, Math.min(1, (ux * vx + uz * vz) / (lu * lv))));
      jStep = Math.max(jStep, st);
      if (st > Math.PI / 6) { sharp++; break; }
    }
  }
  nm++;
}
// minimum turning radius along connectors (circumradius of consecutive samples >= 0.5 m apart)
let rSmall = 0, rMin = 1e9;
for (let m = 0; m < J.moveCount; m++) {
  const first = J.mInfo[7 * m + 4], np = J.mInfo[7 * m + 5], P = J.pts;
  let mr = 1e9;
  for (let k = 1; k < np - 1; k++) {
    const ax = P[4 * (first + k - 1)], az = P[4 * (first + k - 1) + 2], bx = P[4 * (first + k)], bz = P[4 * (first + k) + 2], cx = P[4 * (first + k + 1)], cz = P[4 * (first + k + 1) + 2];
    const ab = Math.hypot(bx - ax, bz - az), bc = Math.hypot(cx - bx, cz - bz), ca = Math.hypot(ax - cx, az - cz);
    if (ab < 0.5 || bc < 0.5) continue;
    const area2 = Math.abs((bx - ax) * (cz - az) - (bz - az) * (cx - ax));
    if (area2 < 1e-9) continue;
    mr = Math.min(mr, (ab * bc * ca) / (2 * area2));
  }
  if (mr < 6 - 0.05) rSmall++;
  rMin = Math.min(rMin, mr);
}
console.log(`connector turning radius: min ${rMin.toFixed(2)} m, connectors below 6 m: ${rSmall}`);
// (a) connector chains on short edges: the next connector must start after the previous one ends
const maxOut = new Float32Array(G.edgeCount).fill(-1), minIn = new Float32Array(G.edgeCount).fill(1e9);
for (let m = 0; m < J.moveCount; m++) {
  const d = J.describe(m);
  if (d.sOut > maxOut[d.outEdge]) maxOut[d.outEdge] = d.sOut;
  if (d.sIn < minIn[d.inEdge]) minIn[d.inEdge] = d.sIn;
}
let chainEdges = 0, chainWorst = 0;
for (let e = 0; e < G.edgeCount; e++) if (maxOut[e] >= 0 && minIn[e] < 1e9 && maxOut[e] > minIn[e] + 0.01) { chainEdges++; chainWorst = Math.max(chainWorst, maxOut[e] - minIn[e]); }
// (b) connector end tangents: first / last chord vs the lane tangent
let tanBad = 0, tanWorst = 0;
for (let m = 0; m < J.moveCount; m++) {
  const d = J.describe(m), first = J.mInfo[7 * m + 4], np = J.mInfo[7 * m + 5], P = J.pts;
  LG.lanePoint(d.inEdge, d.sIn, d.inLane, a); LG.lanePoint(d.outEdge, d.sOut, d.outLane, b);
  const f0x = P[4 * (first + 1)] - P[4 * first], f0z = P[4 * (first + 1) + 2] - P[4 * first + 2];
  const l0x = P[4 * (first + np - 1)] - P[4 * (first + np - 2)], l0z = P[4 * (first + np - 1) + 2] - P[4 * (first + np - 2) + 2];
  const e0 = Math.acos(Math.max(-1, Math.min(1, (f0x * a.hx + f0z * a.hz) / (Math.hypot(f0x, f0z) || 1))));
  const e1 = Math.acos(Math.max(-1, Math.min(1, (l0x * b.hx + l0z * b.hz) / (Math.hypot(l0x, l0z) || 1))));
  const err = Math.max(e0, e1);
  if (err > 0.02) tanBad++;
  tanWorst = Math.max(tanWorst, err);
}
// (c) freeway ramps handled as junction boxes (motorway / trunk / link -> motorway / trunk / link, non-strict)
// freeway ramp merges / diverges (a pure merge or diverge on the freeway system) must never be junction boxes;
// freeway-to-freeway connectors are only allowed at real at-grade intersections (several ins AND outs, turn >= 1.2 rad)
let fwBox = 0, fwIntersection = 0;
for (let m = 0; m < J.moveCount; m++) {
  const d = J.describe(m);
  if (!G.isFreeway(d.inEdge) || !G.isFreeway(d.outEdge)) continue;
  const n = G.to(d.inEdge);
  const outs = [...G.outEdges(n)].filter((o) => o !== G.twin[d.inEdge]).length, ins = [...G.inEdges(n)].filter((i) => i !== G.twin[d.outEdge]).length;
  if (outs <= 1 || ins <= 1 || Math.abs(d.turn) < 1.2) fwBox++; else fwIntersection++;
}
// (d) ramp lanes inside the mainline pavement before the gore: at every merge (ramp and mainline into
// one node) / diverge (out of one node), ramp lane centres from 30 m to 250 m before / after the node
const rampOverlap = (() => {
  let edges = 0, metres = 0;
  const P = {}, Q = {};
  const isLink = (e) => (G.cls[e] === 1 || G.cls[e] === 3) && G.opp[e] === 0, isFw = (e) => (G.cls[e] === 0 || G.cls[e] === 2) && G.opp[e] === 0;
  for (let n = 0; n < G.nodeCount; n++) {
    for (const [list, atEnd] of [[G.inEdges(n), true], [G.outEdges(n), false]]) {
      const ramps = [...list].filter(isLink), mains = [...list].filter(isFw);
      for (const r of ramps) for (const mm of mains) {
        let hit = 0;
        const Lr = G.length[r], Lm = G.length[mm];
        for (let d = 30; d <= Math.min(250, Lr); d += 5) {
          const sr = atEnd ? Lr - d : d;
          for (let l = 0; l < G.lanes[r]; l++) {
            LG.lanePoint(r, sr, l, P);
            // nearest mainline sample within 250 m of the node
            let best = 1e9, lat = 0, hw = 0, dy = 0;
            for (let dm = 0; dm <= Math.min(300, Lm); dm += 4) {
              const sm = atEnd ? Lm - dm : dm, bd = LG.bounds(mm, sm);
              LG.offsetPoint(mm, sm, (bd.right + bd.left) / 2, 0, Q);
              const dd = Math.hypot(P.x - Q.x, P.z - Q.z);
              if (dd < best) { best = dd; lat = Math.abs((P.x - Q.x) * -Q.hz + (P.z - Q.z) * Q.hx); hw = (bd.right - bd.left) / 2; dy = Math.abs(P.y - Q.y); }
            }
            if (best < 30 && lat < hw - 0.2 && dy < 4.5) { hit++; break; }
          }
        }
        if (hit) { edges++; metres += hit * 5; }
      }
    }
  }
  return { edges, metres };
})();
console.log(`ramp lanes inside the mainline pavement 30-250 m before a merge / after a diverge: ${rampOverlap.edges} ramp-mainline pairs, ${rampOverlap.metres} m`);
console.log(`connector chains overlapping on an edge (next sIn < previous sOut): ${chainEdges} edges, worst ${chainWorst.toFixed(1)} m; ` +
  `connector end tangent error > 0.02 rad: ${tanBad} (worst ${tanWorst.toFixed(3)} rad); freeway ramps built as junction boxes: ${fwBox} (freeway-system connectors at real at-grade intersections: ${fwIntersection})`);
console.log(`junctions: ${J.junctionCount} junctions, ${nm} lane connectors, ${J.conf.length / 3} conflicts; ` +
  `connector end gap max ${jPos.toFixed(3)} m, end heading error max ${(jHead * 180 / Math.PI).toFixed(1)} deg (Bezier ends use the lane tangent; the sampled first segment is a chord), max turn per sample step ${(jStep * 180 / Math.PI).toFixed(1)} deg, connectors with a step > 30 deg: ${sharp}, with an end heading error > 20 deg: ${headBad}`);
process.exit(hopBad > 20 || vBad || chainEdges || fwBox || rampOverlap.edges > 40 ? 1 : 0);
