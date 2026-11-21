// Pack a SimGraph (src/sim/roadGraph.js: buildGraph / fromRoadSet / loadWorld)
// into flat typed arrays for GPU storage buffers, plus the per-route-entry
// guidance codes the sensing kernel reads.
//
//   edges   Uint32Array(E*16)  EdgeInfo records (layout = struct Edge in kernels/common.wgsl)
//   pts     Float32Array(P*4)  polyline points (x, z, y, cum) in travel order
//   laneOff Float32Array(K)    lane centre offset (m, right-positive), index edge.laneBase + lane (lane 0 = rightmost)
//   adj     Uint32Array        [outStart N+1 | inStart N+1 | outEdges E | inEdges E]
//
// Route entries (u32) = edgeId (20 bits, < 1,048,576 edges) | code << 20, code = maneuverType | side << 4 for the
// transition at the END of that route edge (src/nav/guidance.js MANEUVER_TYPES order:
// 0 continue, 1 exit, 2 merge, 3 turn-left, 4 turn-right, 5 keep-left, 6 keep-right,
// 7 arrive, 8 refuel; side 0 none, 1 left, 2 right). Code 0 = nothing to announce.

import { hopMap as simHopMap } from '../sim/roadGraph.js';
import { Guidance, MANEUVER_TYPES } from '../nav/guidance.js';

export const EDGE_WORDS = 16;
export const GEO_WORDS = 12;   // + laneW, laneWEnd start, laneWEnd end (src/geom per-edge lane widths)
export const GEO_WORDS_OLD = 9;   // per edge: vFirst, Rd, RendStart, RendEnd, tauStart, tauEnd, tapers (tapR0|tapR1<<8|tapL0<<16|tapL1<<24), hopBase,
                              // run-outs (xR0|xR1<<8|xL0<<16|xL1<<24: extra paved lane widths at the start / end, LG.bounds)
export const MAN = { continue: 0, exit: 1, merge: 2, turnLeft: 3, turnRight: 4, keepLeft: 5, keepRight: 6, arrive: 7, refuel: 8 };
export const SIDE_NONE = 0, SIDE_LEFT = 1, SIDE_RIGHT = 2;
export const EDGE_ID_MASK = 0xFFFFF;
const SIM_FREEWAY = new Set(['motorway', 'motorway_link', 'trunk', 'trunk_link']); // roadGraph.js hopMap FREEWAY
const FREEWAY = new Set(['motorway', 'trunk']);

/** telFlags: optional Uint32Array per edge (e.g. per-edge counters; unused), stored in edge flag bits 8..13. */
// sim.js SHOULDER_ROADS: freeway-class roads get the 3 m outside shoulder (edge flag 64)
const SHOULDER_ROADS = new Set(['motorway', 'trunk', 'motorway_link', 'trunk_link']);
export function packWorld(graph, telFlags = null) {
  const E = graph.edgeCount ?? graph.edges.length, N = graph.nodeCount;
  if (E > EDGE_ID_MASK + 1) throw new Error(`packWorld: ${E} edges > 2^20 (route entries hold 20-bit edge ids)`);
  let P = 0, K = 0;
  for (const e of graph.edges) { P += e.px.length; K += e.lanes; }
  const edgesU = new Uint32Array(E * EDGE_WORDS), edgesF = new Float32Array(edgesU.buffer), edgesI = new Int32Array(edgesU.buffer);
  const pts = new Float32Array(P * 4);
  const laneOff = new Float32Array(Math.max(K, 1));
  const stopBack = computeStopBack(graph);
  let p = 0, k = 0;
  for (let i = 0; i < E; i++) {
    const e = graph.edges[i], n = e.px.length, o = i * EDGE_WORDS;
    edgesU[o] = p; edgesU[o + 1] = n; edgesU[o + 2] = e.lanes; edgesU[o + 3] = k;
    edgesF[o + 4] = e.length; edgesF[o + 5] = e.speedLimit; edgesU[o + 6] = e.from; edgesU[o + 7] = e.to;
    edgesI[o + 8] = e.reverse; edgesU[o + 9] = e.rank;
    edgesU[o + 10] = (e.isLink ? 1 : 0) | (e.bridge ? 2 : 0) | (e.tunnel ? 4 : 0) | (FREEWAY.has(e.highway) ? 16 : 0) | (SIM_FREEWAY.has(e.highway) ? 32 : 0) | (SHOULDER_ROADS.has(e.highway) ? 64 : 0) | (telFlags ? (telFlags[i] & 63) << 8 : 0)
      | ((e.surf | 0) & 15) << 16 | ((e.gndR | 0) & 15) << 20 | ((e.gndL | 0) & 15) << 24;   // road surface codes (src/sim/surface.js)
    edgesF[o + 11] = stopBack[i];
    edgesF[o + 12] = e.headingStart; edgesF[o + 13] = e.headingEnd; edgesF[o + 14] = e.y; edgesU[o + 15] = e.navRank ?? 0;
    for (let j = 0; j < n; j++) {
      pts[4 * (p + j)] = e.px[j]; pts[4 * (p + j) + 1] = e.pz[j];
      pts[4 * (p + j) + 2] = e.ys ? e.ys[j] : e.y; pts[4 * (p + j) + 3] = e.cum[j];
    }
    p += n;
    for (let l = 0; l < e.lanes; l++) laneOff[k + l] = e.laneOff[l];
    k += e.lanes;
  }
  const adj = new Uint32Array(2 * (N + 1) + 2 * E);
  adj.set(graph.outStart, 0); adj.set(graph.inStart, N + 1);
  adj.set(graph.outEdges, 2 * (N + 1)); adj.set(graph.inEdges, 2 * (N + 1) + E);
  // ---- lane geometry (src/geom LaneGeometry attached by data.js) + junction hop table
  const geo = new Uint32Array(E * GEO_WORDS), geoF = new Float32Array(geo.buffer);
  const LG = graph.laneGeom && graph.laneGeom.G && graph.laneGeom.G.edgeCount === E ? graph.laneGeom : null;
  let lgVerts = new Float32Array(6);
  if (LG) {
    const V = LG.vFirst[E];
    lgVerts = new Float32Array(V * 6);
    for (let v = 0; v < V; v++) { const o = 6 * v; lgVerts[o] = LG.vx[v]; lgVerts[o + 1] = LG.vz[v]; lgVerts[o + 2] = LG.vy[v]; lgVerts[o + 3] = LG.vs[v]; lgVerts[o + 4] = LG.mx[v]; lgVerts[o + 5] = LG.mz[v]; }
    for (let e = 0; e < E; e++) {
      const o = e * GEO_WORDS;
      geo[o] = LG.vFirst[e]; geoF[o + 1] = LG.Rd[e]; geoF[o + 2] = LG.Rend[2 * e]; geoF[o + 3] = LG.Rend[2 * e + 1];
      geoF[o + 4] = LG.tau[2 * e]; geoF[o + 5] = LG.tau[2 * e + 1];
      geo[o + 6] = LG.tapR[2 * e] | (LG.tapR[2 * e + 1] << 8) | (LG.tapL[2 * e] << 16) | (LG.tapL[2 * e + 1] << 24);
      if (LG.xR) geo[o + 8] = LG.xR[2 * e] | (LG.xR[2 * e + 1] << 8) | (LG.xL[2 * e] << 16) | (LG.xL[2 * e + 1] << 24);
      const w = LG.laneW ? LG.laneW[e] : 3.7;
      geoF[o + 9] = w; geoF[o + 10] = LG.laneWEnd ? LG.laneWEnd[2 * e] : w; geoF[o + 11] = LG.laneWEnd ? LG.laneWEnd[2 * e + 1] : w;
    }
  }
  // hop table: for edge e and the k-th out-edge en of e.to: strict (bit 0) | (shift + 64) << 1
  let hopN = 0;
  for (let e = 0; e < E; e++) { const t = graph.edges[e].to; geo[e * GEO_WORDS + 7] = hopN; hopN += graph.outStart[t + 1] - graph.outStart[t]; }
  const hops = new Uint32Array(Math.max(1, hopN));
  for (let e = 0; e < E; e++) {
    const ed = graph.edges[e], t = ed.to, b0 = geo[e * GEO_WORDS + 7];
    for (let q = graph.outStart[t]; q < graph.outStart[t + 1]; q++) {
      const h = simHopMap(graph, ed, graph.edges[graph.outEdges[q]]);
      hops[b0 + q - graph.outStart[t]] = (h.strict ? 1 : 0) | ((Math.max(-64, Math.min(63, h.shift)) + 64) << 1);
    }
  }
  return {
    E, N, P, K, edges: edgesU, pts, laneOff, adj, geo, lgVerts, hops, hasLG: !!LG,
    adjOff: { outStart: 0, inStart: N + 1, outEdges: 2 * (N + 1), inEdges: 2 * (N + 1) + E },
    laneBase: (e) => edgesU[e * EDGE_WORDS + 3],
    bytes: edgesU.byteLength + pts.byteLength + laneOff.byteLength + adj.byteLength + geo.byteLength + lgVerts.byteLength + hops.byteLength,
  };
}

/** Same rule as sim.js yieldRequired's lazily computed e.stopBack. */
export function computeStopBack(g) {
  const LW = g.laneWidth || 3.7, E = g.edges.length, out = new Float32Array(E);
  for (let i = 0; i < E; i++) {
    const e = g.edges[i];
    let inc = 0, hw = 0;
    for (let q = g.inStart[e.to]; q < g.inStart[e.to + 1]; q++) {
      const ei = g.inEdges[q]; if (ei === e.idx || ei === e.reverse) continue;
      inc++; const o = g.edges[ei]; hw = Math.max(hw, (o.reverse >= 0 ? 2 : 1) * o.lanes * LW * 0.5);
    }
    for (let q = g.outStart[e.to]; q < g.outStart[e.to + 1]; q++) {
      const ei = g.outEdges[q]; if (ei === e.reverse) continue;
      const o = g.edges[ei]; hw = Math.max(hw, (o.reverse >= 0 ? 2 : 1) * o.lanes * LW * 0.5);
    }
    out[i] = inc ? Math.min(12, Math.max(1.5, hw + 1)) : 1.5;
  }
  return out;
}

const wrap = (a) => { while (a > Math.PI) a -= 2 * Math.PI; while (a < -Math.PI) a += 2 * Math.PI; return a; };
const D30 = Math.PI / 6, D45 = Math.PI / 4;

/**
 * Maneuver code for the transition a -> b (edge objects), b = null for route end.
 * Approximates src/nav/guidance.js: freeway -> link = exit, link -> freeway = merge,
 * |heading change| > 30 deg (not freeway-freeway) = turn, another out edge of equal or
 * higher rank within 45 deg = fork (keep-left/right). Heading delta < 0 = left.
 */
export function maneuverCode(g, a, b) {
  if (!b) return MAN.arrive;
  const dh = wrap(b.headingStart - a.headingEnd);
  const side = dh < 0 ? SIDE_LEFT : SIDE_RIGHT;
  const fa = FREEWAY.has(a.highway), fb = FREEWAY.has(b.highway);
  if (fa && b.isLink) return MAN.exit | side << 4;
  if (a.isLink && fb) return MAN.merge;
  if (Math.abs(dh) > D30 && !(fa && fb)) return (dh < 0 ? MAN.turnLeft : MAN.turnRight) | side << 4;
  const n = a.to;
  for (let q = g.outStart[n]; q < g.outStart[n + 1]; q++) {
    const o = g.edges[g.outEdges[q]];
    if (o.idx === b.idx || o.idx === a.reverse || o.rank < b.rank) continue;
    const dho = wrap(o.headingStart - a.headingEnd);
    if (Math.abs(dho) > D45) continue;
    return dh < dho ? (MAN.keepLeft | SIDE_LEFT << 4) : (MAN.keepRight | SIDE_RIGHT << 4);
  }
  return 0;
}

/**
 * Pack a flat route pool (Int32Array of edge ids, routes delimited by `offset`)
 * into GPU route entries with guidance codes. Codes are memoised per (a,b) pair.
 */
export function packRoutes(g, flat, offset) {
  // Route words: edge (20 bits) | present << 20 | type << 21 (4 bits) | side << 25 (2 bits) | nextType << 27 (4 bits).
  // The maneuvers come from src/nav/guidance.js itself (what a CPU route-guidance consumer sees, incl. its
  // coalescing and "continue" rules), on the graph the driver factory uses (graph.nav || graph):
  // the maneuver at the node after route edge k is stored on entry k; nextType = type of the first
  // maneuver at entry >= k (for the input when the next maneuver is more than 2 km away).
  const out = new Uint32Array(Math.max(1, flat.length));
  const nav = g.nav || g, nR = offset.length - 1;
  const typeIdx = Object.fromEntries(MANEUVER_TYPES.map((t, i) => [t, i]));
  for (let r = 0; r < nR; r++) {
    const a0 = offset[r], b0 = offset[r + 1], n = b0 - a0;
    if (n <= 0) continue;
    const ids = Array.from(flat.subarray ? flat.subarray(a0, b0) : flat.slice(a0, b0));
    const code = new Uint16Array(n);
    let ms = null;
    try { ms = new Guidance(nav, ids).maneuvers; } catch (e) { ms = null; }
    if (ms) for (const m of ms) {
      const k = m.type === 'arrive' ? n - 1 : m.edgePos - 1;
      if (k < 0 || k >= n || code[k]) continue;
      const side = m.side === 'left' ? SIDE_LEFT : m.side === 'right' ? SIDE_RIGHT : SIDE_NONE;
      code[k] = 1 | ((typeIdx[m.type] ?? 0) << 1) | (side << 5);
    } else code[n - 1] = 1 | (MAN.arrive << 1);
    let next = MAN.arrive;
    for (let k = n - 1; k >= 0; k--) {
      if (code[k] & 1) next = (code[k] >> 1) & 15;
      out[a0 + k] = ((ids[k] & EDGE_ID_MASK) | (code[k] << 20) | (next << 27)) >>> 0;
    }
  }
  return out;
}

/** Pack one route (array of edge ids) on its own. */
export function packRoute(g, edgesList) {
  return packRoutes(g, Int32Array.from(edgesList), Uint32Array.of(0, edgesList.length));
}

/**
 * Traffic controls (src/sim/controls.js TrafficControls) for kernels/controls.wgsl:
 * edgeAp (approach + 1 of the first stop line along each edge, 0 none), approaches (4 words:
 * control, stopAt, group | protLeft << 8, edge), controls (20 words: type, cycle, offset,
 * nPhases, 4 x (group | left << 8, g, y, r)).
 */
export function packControls(C, E) {
  // every stop line per edge (sorted by position): CSR edgeApStart (E + 1) / edgeApList; older controls.js
  // (one approach per edge) is converted
  let start, list;
  if (C.edgeApStart) { start = Uint32Array.from(C.edgeApStart); list = Uint32Array.from(C.edgeApList); }
  else { start = new Uint32Array(E + 1); const l = []; for (let e = 0; e < E; e++) { const a = C.edgeApproach[e] ?? -1; if (a >= 0) l.push(a); start[e + 1] = l.length; } list = Uint32Array.from(l); }
  const nA = C.approachCount, nC = C.count;
  const ap = new Uint32Array(Math.max(1, nA) * 4), apF = new Float32Array(ap.buffer);
  for (let a = 0; a < nA; a++) { ap[4 * a] = C.apControl[a]; apF[4 * a + 1] = C.apStopAt[a]; ap[4 * a + 2] = (C.apGroup[a] & 255) | (C.apProtLeft[a] ? 256 : 0); ap[4 * a + 3] = Math.max(0, C.apEdge[a]); }
  const R = 24, ctl = new Uint32Array(Math.max(1, nC) * R), cF = new Float32Array(ctl.buffer);
  let tooMany = 0;
  for (let c = 0; c < nC; c++) {
    const o = R * c, ph = C.phases[c] || [];
    ctl[o] = C.type[c]; cF[o + 1] = C.cycle[c] || 60; cF[o + 2] = C.offset[c] || 0; ctl[o + 3] = Math.min(4, ph.length);
    if (ph.length > 4) tooMany++;
    for (let k = 0; k < Math.min(4, ph.length); k++) { const p = ph[k], q = o + 4 + 4 * k; ctl[q] = (p.group & 255) | (p.left ? 256 : 0); cF[q + 1] = p.g; cF[q + 2] = p.y; cF[q + 3] = p.r; }
    ctl[o + 20] = C.clusterOf ? C.clusterOf[c] : c;   // signals of one intersection (divided roads) share a cluster
  }
  if (tooMany) console.warn(`packControls: ${tooMany} signals with > 4 phases truncated`);
  const edgeAp = new Uint32Array(start.length + list.length); edgeAp.set(start); edgeAp.set(list, start.length);
  return { edgeAp, listOff: start.length, ap, ctl, nA, nC, rampStart: C.rampStart, rampEnd: C.rampEnd };
}

/**
 * Junction connectors (src/geom junctions.js J.arrays() / data/junctions.*) packed for the kernels
 * (only what they read; the hull is not packed):
 *   mouth   f32 x2 per edge [sExit, sStop], gore f32 per edge (-1 none)
 *   mov     u32 x9 per movement: inEdge | inLane << 20 | preferred << 28, outEdge | outLane << 20, firstPt, ptCount,
 *           sIn, sOut, length, turn (f32 bits)
 *   edgeMov CSR (E + 1 starts, then movement ids) of the movements leaving each in-edge, sorted by (inLane, outEdge)
 *   pts     f32 x4 per connector point [x, z, y, metres along]
 *   conf    CSR per movement (M + 1 starts, then u32 x4 entries: other, type, f32 s (mine), f32 sOther)
 */
/**
 * sim.js _conflicts(m): the conflict matrix plus every other movement of the same junction node (not from the same
 * entry lane) whose sampled centre path comes within 4.2 m of m's, as a crossing (type 0) at the closest approach.
 * Returns flat Float32Array [A, B, sA, sB] * n (A < B), computed once per world (bbox-pruned).
 */
export function extraConflicts(a, edgeTo) {
  const M = a.mInfo.length / 7, P = a.pts, R = 4.2;
  const byNode = new Map();
  for (let m = 0; m < M; m++) { const nd = edgeTo[a.mInfo[7 * m]]; let l = byNode.get(nd); if (!l) byNode.set(nd, (l = [])); l.push(m); }
  const known = new Set();
  for (let c = 0; c < a.conf.length / 3; c++) { const A = a.conf[3 * c], B = a.conf[3 * c + 1]; known.add(A < B ? A * M + B : B * M + A); }
  const bb = new Float32Array(4 * M);
  for (let m = 0; m < M; m++) {
    const f = a.mInfo[7 * m + 4], n = a.mInfo[7 * m + 5]; let x0 = 1e30, x1 = -1e30, z0 = 1e30, z1 = -1e30;
    for (let k = 0; k < n; k++) { const x = P[4 * (f + k)], z = P[4 * (f + k) + 2]; if (x < x0) x0 = x; if (x > x1) x1 = x; if (z < z0) z0 = z; if (z > z1) z1 = z; }
    bb[4 * m] = x0; bb[4 * m + 1] = x1; bb[4 * m + 2] = z0; bb[4 * m + 3] = z1;
  }
  const out = [];
  for (const list of byNode.values()) {
    for (let p = 0; p < list.length; p++) {
      const m = list[p], f1 = a.mInfo[7 * m + 4], n1 = a.mInfo[7 * m + 5];
      for (let q = p + 1; q < list.length; q++) {
        const o = list[q];
        if (a.mInfo[7 * o] === a.mInfo[7 * m] && a.mInfo[7 * o + 1] === a.mInfo[7 * m + 1]) continue;
        if (known.has(m < o ? m * M + o : o * M + m)) continue;
        if (bb[4 * o] > bb[4 * m + 1] + R || bb[4 * m] > bb[4 * o + 1] + R || bb[4 * o + 2] > bb[4 * m + 3] + R || bb[4 * m + 2] > bb[4 * o + 3] + R) continue;
        const f2 = a.mInfo[7 * o + 4], n2 = a.mInfo[7 * o + 5];
        let best = 1e18, sa = 0, sb = 0;
        for (let i = 0; i < n1; i++) {
          const xi = P[4 * (f1 + i)], zi = P[4 * (f1 + i) + 2];
          for (let k = 0; k < n2; k++) { const dx = xi - P[4 * (f2 + k)], dz = zi - P[4 * (f2 + k) + 2], d = dx * dx + dz * dz; if (d < best) { best = d; sa = P[4 * (f1 + i) + 3]; sb = P[4 * (f2 + k) + 3]; } }
        }
        if (best < R * R) out.push(m, o, sa, sb);
      }
    }
  }
  return Float32Array.from(out);
}

export function packJunctions(a, E, connectorSpeed = null, extra = null) {
  const M = a.mInfo.length / 7;
  const MW = 9, mov = new Uint32Array(M * MW), movF = new Float32Array(mov.buffer);
  for (let m = 0; m < M; m++) {
    const i = 7 * m, o = MW * m;
    mov[o] = (a.mInfo[i] & EDGE_ID_MASK) | ((a.mInfo[i + 1] & 255) << 20) | ((a.mInfo[i + 6] ? 1 : 0) << 28);
    mov[o + 1] = (a.mInfo[i + 2] & EDGE_ID_MASK) | ((a.mInfo[i + 3] & 255) << 20);
    mov[o + 2] = a.mInfo[i + 4]; mov[o + 3] = a.mInfo[i + 5];
    movF[o + 4] = a.mF[4 * m]; movF[o + 5] = a.mF[4 * m + 1]; movF[o + 6] = a.mF[4 * m + 2]; movF[o + 7] = a.mF[4 * m + 3];
    movF[o + 8] = connectorSpeed ? connectorSpeed(a, m) : 1e3;   // connectorSpeed below (comfortable speed through the bend)
  }
  const cnt = new Uint32Array(E + 1);
  for (let m = 0; m < M; m++) cnt[a.mInfo[7 * m] + 1]++;
  for (let e = 0; e < E; e++) cnt[e + 1] += cnt[e];
  const fill = cnt.slice(0, E), list = new Uint32Array(M);
  for (let m = 0; m < M; m++) list[fill[a.mInfo[7 * m]]++] = m;
  for (let e = 0; e < E; e++) { const sub = list.subarray(cnt[e], cnt[e + 1]); if (sub.length > 1) sub.sort((x, y) => a.mInfo[7 * x + 1] - a.mInfo[7 * y + 1] || a.mInfo[7 * x + 2] - a.mInfo[7 * y + 2]); }
  const edgeMov = new Uint32Array(E + 1 + M); edgeMov.set(cnt); edgeMov.set(list, E + 1);
  const P = a.pts.length / 4, pts = new Float32Array(P * 4);
  for (let p = 0; p < P; p++) { pts[4 * p] = a.pts[4 * p]; pts[4 * p + 1] = a.pts[4 * p + 2]; pts[4 * p + 2] = a.pts[4 * p + 1]; pts[4 * p + 3] = a.pts[4 * p + 3]; }
  const C0 = a.conf.length / 3, X = extra ? extra.length / 4 : 0, C = C0 + X, cc = new Uint32Array(M + 1);
  const cA = (c) => (c < C0 ? a.conf[3 * c] : extra[4 * (c - C0)]), cB = (c) => (c < C0 ? a.conf[3 * c + 1] : extra[4 * (c - C0) + 1]);
  for (let c = 0; c < C; c++) { cc[cA(c) + 1]++; cc[cB(c) + 1]++; }
  for (let m = 0; m < M; m++) cc[m + 1] += cc[m];
  const cfill = cc.slice(0, M), centry = new Uint32Array(2 * C * 4), cF = new Float32Array(centry.buffer);
  for (let c = 0; c < C; c++) {
    const A = cA(c), B = cB(c), t = c < C0 ? a.conf[3 * c + 2] : 0, sA = c < C0 ? a.confS[2 * c] : extra[4 * (c - C0) + 2], sB = c < C0 ? a.confS[2 * c + 1] : extra[4 * (c - C0) + 3];
    let k = cfill[A]++; centry[4 * k] = B; centry[4 * k + 1] = t; cF[4 * k + 2] = sA; cF[4 * k + 3] = sB;
    k = cfill[B]++; centry[4 * k] = A; centry[4 * k + 1] = t; cF[4 * k + 2] = sB; cF[4 * k + 3] = sA;
  }
  const conf = new Uint32Array(M + 1 + centry.length); conf.set(cc); conf.set(centry, M + 1);
  const mouth = Float32Array.from(a.mouthS), gore = Float32Array.from(a.goreS);
  const bytes = 4 * (mouth.length + gore.length + mov.length + edgeMov.length + pts.length + conf.length);
  return { mouth, gore, mov, edgeMov, pts, conf, M, P, C, extra: X, bytes };
}

/** Comfortable speed (m/s) through junction connector m: sqrt(3 m/s^2 / peak curvature), cached on J (world geometry;
 *  packed as movF[8], kept for policies that read the world pack). */
export function connectorSpeed(J, m) {
  const c = J._vTurn || (J._vTurn = new Float32Array(J.moveCount).fill(-1));
  if (c[m] >= 0) return c[m];
  const first = J.mInfo[7 * m + 4], n = J.mInfo[7 * m + 5], P = J.pts;
  const dir = (k) => { const a = 4 * (first + k); return Math.atan2(P[a + 6] - P[a + 2], P[a + 4] - P[a]); };
  let kap = 0;
  for (let k = 0; k < n - 1; k++) {
    let q = k + 1;
    while (q < n - 1 && P[4 * (first + q) + 3] - P[4 * (first + k) + 3] < 8) q++;
    const L = P[4 * (first + q) + 3] - P[4 * (first + k) + 3]; if (L < 1) continue;
    let d = dir(Math.min(q, n - 2)) - dir(k); d = Math.abs(Math.atan2(Math.sin(d), Math.cos(d)));
    kap = Math.max(kap, d / Math.max(L, 8));
    if (q >= n - 1) break;
  }
  if (n <= 2) kap = Math.abs(J.mF[4 * m + 3]) / Math.max(8, J.mLen[m]);
  return (c[m] = kap > 1e-4 ? Math.max(2.5, Math.sqrt(3.0 / kap)) : 1e3);
}
