// Lane lines: THE one definition of the painted markings, shared by the renderers (tiles,
// src/roads.js) and the drivers' perception (CPU sim now, GPU later). See src/geom/lines.md.
//
//   const LS = buildLineSet(LG, J, ctx)            // tools/lines/build_lines.mjs -> data/lines.bin
//   const LS = await loadLineSet('data/lines.meta.json', LG, J)   // browser / sim
//   LS.lineAt(e, k, s)            -> { type, color, pattern }  boundary k (0 = right edge .. lanes = left edge / centre)
//   LS.lineOffset(e, k, s)        -> m right of the reference polyline (= LG.dividerOffset, exact lane boundary)
//   LS.lineGeom(pos, opts, out)   -> the car's left / right lines, all lines across, heading, curvature,
//                                    and the lane ahead at 8 / 16 / 32 / 64 m along the planned route
//   emitPaint(LS, e, sink, opts)  -> paint triangles for edge e (renderers)
//
// Every lane boundary is derived from src/geom lanes.js (rightEdge - k * laneWidth), so a car on a
// lane centre (LG.laneCenterLat) is exactly half a lane width from each of its lines.
import { hopMap, turnAngle } from './hops.js';

export const LINE = { NONE: 0, SOLID: 1, DASHED: 2, DOUBLE: 3, DOTTED: 4 };
export const LINE_NAMES = ['none', 'solid', 'dashed', 'double', 'dotted'];
export const COLOR = { WHITE: 0, YELLOW: 1 };
export const COLOR_NAMES = ['white', 'yellow'];
/** Pseudo boundaries: outer pavement edge of a paved run-out wedge (lane drop / add taper). */
export const OUTER_R = 254, OUTER_L = 255;
/** Transverse marks: stop bar, yield line, crosswalk. */
export const MARK = { STOP: 1, YIELD: 2, CROSSWALK: 3 };
export const MARK_NAMES = ['', 'stop', 'yield', 'crosswalk'];
/** Mark span: 0 = inbound lanes of the edge (k = 0 .. lanes), 1 = the whole road pavement (both directions). */
export const SPAN = { INBOUND: 0, ROAD: 1 };
/** Marking level of an edge. */
export const LEVEL = { NONE: 0, CENTRE: 1, FULL: 2 };
export const LEVEL_NAMES = ['none', 'centre', 'full'];

/** Dash patterns [on, off] in metres (index stored per run). */
export const PATTERNS = [
  [0, 0],            // 0 solid / none
  [3.66, 10.97],     // 1 freeway lane line: 12 ft line, 36 ft gap (Caltrans Detail 9 spacing)
  [3.05, 9.14],      // 2 street lane line / broken yellow centre: 10 ft / 30 ft (MUTCD 3A.06)
  [0.91, 2.74],      // 3 wide dotted line: 3 ft / 9 ft (lane drop, acceleration lane, gore approach; MUTCD 3B.04)
];
export const PAINT = {
  lineW: 0.10,       // 4 in lane / edge / centre lines
  dottedW: 0.20,     // 8 in wide dotted lines
  doubleGap: 0.10,   // gap between the two lines of a double (centres at +-0.10 m)
  edgeInset: 0.045,  // edge-line paint centre this far inside the lane edge (the line sits on the pavement, <= 5 mm over a kerb-less edge)
  stopW: 0.45,       // stop bar depth (CA limit line 12-24 in)
  barW: 0.60, barGap: 0.60,   // continental crosswalk bars
  crosswalk: 3.0,    // crosswalk depth along the road (10 ft)
};

const FREEWAY_CLS = (c) => c <= 3;
const LINK_CLS = (c) => c === 1 || c === 3 || c === 5 || c === 7 || c === 9;
const RUN_K = 0xff, RUN_TYPE_SHIFT = 8, RUN_COLOR_SHIFT = 12, RUN_PAT_SHIFT = 16, RUN_OCCLUDED = 1 << 24;
/** runInfo flag: the line lies on another carriageway's lanes (overlapping drawn carriageways): not painted, type NONE. */
export const OCCLUDED_BIT = RUN_OCCLUDED;

// Zones (metres) used by the spec (docs/markings.md)
const Z = {
  goreSolid: 60,       // solid white channelizing line at a diverge / merge (gore approach / extension)
  goreDotted: 200,     // wide dotted exit-lane / acceleration-lane line beyond it
  dropFw: 300, dropSt: 50,   // wide dotted lane-drop line before a lane ends (freeway system / streets)
  addFw: 150, addSt: 50,     // wide dotted line after a lane is added
  signalSolid: 30,     // solid lane lines on the last 30 m of a signal approach (no lane changes at the stop bar)
  passCurveR: 300,     // no passing within passCurveSpan of a curve tighter than this radius
  passCurveSpan: 100,
  passJunction: 60,    // no passing within 60 m of a junction
  rampGore: 60,        // white left edge on the first / last 60 m of a link beside its mainline
};

function urbanGrid(G) {
  const cell = 500, dens = new Map();
  for (let e = 0; e < G.edgeCount; e++) {
    if (G.cls[e] < 8 || (G.twin[e] >= 0 && G.twin[e] < e)) continue;
    const a = G.from(e), b = G.to(e), k = Math.floor((G.x(a) + G.x(b)) / 2 / cell) * 100000 + Math.floor((G.z(a) + G.z(b)) / 2 / cell);
    dens.set(k, (dens.get(k) || 0) + G.length[e]);
  }
  return (x, z) => (dens.get(Math.floor(x / cell) * 100000 + Math.floor(z / cell)) || 0) > 2500;
}

/** Marking level of edge e (see docs/markings.md). */
export function markingLevel(G, e) {
  const c = G.cls[e], n = G.lanes[e], m = G.twin[e] >= 0 ? G.lanes[G.twin[e]] : 0;
  if (c <= 10) return LEVEL.FULL;                        // motorway .. unclassified
  if (c === 12) return LEVEL.NONE;                       // living_street: shared space, no paint
  // residential: unmarked unless OSM says otherwise
  if ((m > 0 && n + m >= 3) || (m === 0 && n >= 2)) return LEVEL.FULL;
  if (m > 0 && !(G.flags[e] & 8)) return LEVEL.CENTRE;   // lanes tagged in OSM (not LANES_FALLBACK): faint dashed centre
  return LEVEL.NONE;
}

/**
 * Build the line set from the lane geometry LG, the junction set J (mouths: lines stop at the
 * stop point and resume at the exit point of every junction box) and the context:
 *   ctx.controls   controls.json `controls` (stop bars, yield lines, signal approaches, crossing signals)
 *   ctx.legs       crosswalkLegs(LG, controls, points) -> {crosswalkStop, crosswalkExit} (junction crosswalks)
 *   ctx.points     crosswalks.json `points` [x, z, kind] (mid-block crosswalks)
 */
export function buildLineSet(LG, J, ctx = {}) {
  const G = LG.G, E = G.edgeCount;
  const urbanAt = urbanGrid(G);
  const level = new Uint8Array(E);
  for (let e = 0; e < E; e++) level[e] = markingLevel(G, e);
  const range = new Float64Array(2 * E);   // painted s range per edge: [sExit, sStop]
  for (let e = 0; e < E; e++) {
    range[2 * e] = J ? J.mouthS[2 * e] : 0; range[2 * e + 1] = J ? J.mouthS[2 * e + 1] : G.length[e];
  }
  // signal approaches (solid lane lines at the stop bar)
  const signalIn = new Set();
  for (const c of ctx.controls || []) if (c.type === 'signal' && c.node >= 0 && !c.midBlock) for (const a of c.approaches) signalIn.add(a.edge);
  // strict hops per edge end: outs[e] = [[b, shift]], ins[b] = [[a, shift]]
  const outs = new Map(), ins = new Map();
  for (let n = 0; n < G.nodeCount; n++) {
    const I = G.inEdges(n), O = G.outEdges(n);
    if (!I.length || !O.length) continue;
    for (const a of I) for (const b of O) {
      if (b === G.twin[a]) continue;
      const h = hopMap(G, a, b);
      if (!h.strict) continue;
      (outs.get(a) || outs.set(a, []).get(a)).push([b, h.shift]);
      (ins.get(b) || ins.set(b, []).get(b)).push([a, h.shift]);
    }
  }
  // junction-ish nodes (>= 3 edge ends of distinct roads) for no-passing zones
  const deg = new Uint8Array(G.nodeCount);
  for (let e = 0; e < E; e++) { if (G.twin[e] >= 0 && G.twin[e] < e) continue; deg[G.from(e)]++; deg[G.to(e)]++; }

  const runs = [];   // per edge: list of [k, s0, s1, type, color, pattern]
  const edgeRuns = new Array(E);
  // ---- no-passing intervals (1+1 two-way rural roads), in primary-direction s
  const curvy = (e) => {
    const v0 = LG.vFirst[e], v1 = LG.vFirst[e + 1] - 1, out = [];
    for (let i = v0 + 1; i < v1; i++) {
      const ax = LG.vx[i] - LG.vx[i - 1], az = LG.vz[i] - LG.vz[i - 1], bx = LG.vx[i + 1] - LG.vx[i], bz = LG.vz[i + 1] - LG.vz[i];
      const la = Math.hypot(ax, az), lb = Math.hypot(bx, bz);
      if (la < 1e-6 || lb < 1e-6) continue;
      const th = Math.acos(Math.max(-1, Math.min(1, (ax * bx + az * bz) / (la * lb))));
      if (th / ((la + lb) / 2) > 1 / Z.passCurveR) out.push([LG.vs[i] - Z.passCurveSpan, LG.vs[i] + Z.passCurveSpan]);
    }
    return out;
  };
  const centreRuns = (e) => {
    // centre line of two-way road e (primary direction), in e's s; returns [[s0, s1, type, pattern]]
    const tw = G.twin[e], L = G.length[e], n = G.lanes[e], m = G.lanes[tw];
    const a = Math.max(range[2 * e], L - range[2 * tw + 1]), b = Math.min(range[2 * e + 1], L - range[2 * tw]);
    if (b - a < 0.5) return [];
    const lv = level[e];
    if (lv === LEVEL.CENTRE) return [[a, b, LINE.DASHED, 2]];
    if (n + m >= 3) return [[a, b, LINE.DOUBLE, 0]];
    const mid = G.nodesOf(e)[G.count[e] >> 1];
    if (urbanAt(G.x(mid), G.z(mid)) || (G.flags[e] & 6)) return [[a, b, LINE.DOUBLE, 0]];   // urban, bridge, tunnel
    // rural two-lane road: broken yellow (passing permitted) except near curves and junctions
    const no = curvy(e);
    if (deg[G.from(e)] >= 3) no.push([-1, Z.passJunction]);
    if (deg[G.to(e)] >= 3) no.push([L - Z.passJunction, L + 1]);
    // passing zones shorter than 120 m are not worth a broken line
    const rs = splitByZones(a, b, no, [LINE.DOUBLE, 0], [LINE.DASHED, 2]);
    for (const r of rs) if (r[2] === LINE.DASHED && r[1] - r[0] < 120) { r[2] = LINE.DOUBLE; r[3] = 0; }
    const res = [];
    for (const r of rs) { const l = res[res.length - 1]; if (l && l[2] === r[2]) l[1] = r[1]; else res.push(r); }
    return res;
  };

  for (let e = 0; e < E; e++) {
    const lv = level[e];
    const R = [];
    edgeRuns[e] = R;
    if (lv === LEVEL.NONE) continue;
    const n = G.lanes[e], L = G.length[e], c = G.cls[e], tw = G.twin[e], twoWay = tw >= 0;
    const a = range[2 * e], b = range[2 * e + 1];
    if (b - a < 0.5) continue;
    // centre (two-way): computed on the primary direction, mirrored on the twin
    if (twoWay) {
      const prim = tw > e ? e : tw;
      const cr = centreRuns(prim);
      for (const [s0, s1, t, p] of cr) {
        const [x0, x1] = prim === e ? [s0, s1] : [L - s1, L - s0];
        R.push([n, x0, x1, t, COLOR.YELLOW, p]);
      }
    }
    if (lv === LEVEL.CENTRE) { R.sort(runOrder); continue; }
    // right edge (k = 0): solid white
    R.push([0, a, b, LINE.SOLID, COLOR.WHITE, 0]);
    // left edge of a one-way carriageway (k = n): solid yellow; white in the gore beside the mainline
    if (!twoWay) {
      const whiteAt = [];
      if (LINK_CLS(c)) {
        const T = Math.min(Z.rampGore, L / 2);
        // diverge: e leaves a carriageway that also continues elsewhere; merge: e joins beside another feeder
        const startMerge = (ins.get(e) || []).some(([p]) => (outs.get(p) || []).length > 1);
        const endMerge = (outs.get(e) || []).some(([q]) => (ins.get(q) || []).length > 1);
        if (startMerge) whiteAt.push([-1, T]);
        if (endMerge) whiteAt.push([L - T, L + 1]);
      }
      for (const [s0, s1, t] of splitByZones(a, b, whiteAt, [COLOR.WHITE], [COLOR.YELLOW])) R.push([n, s0, s1, LINE.SOLID, t, 0]);
    }
    // lane dividers 0 < k < n
    if (n > 1) {
      const pat = FREEWAY_CLS(c) ? 1 : 2;
      const zones = new Map();   // k -> list of [s0, s1, type, pattern, priority]
      const add = (k, s0, s1, t, p, pri) => { if (k <= 0 || k >= n) return; (zones.get(k) || zones.set(k, []).get(k)).push([Math.max(a, s0), Math.min(b, s1), t, p, pri]); };
      if (G.flags[e] & 4) for (let k = 1; k < n; k++) add(k, a, b, LINE.SOLID, 0, 9);   // tunnel: no lane changes
      // diverge at the end node: boundary between lanes bound for different carriageways
      const o = outs.get(e) || [];
      if (o.length > 1) for (const k of splitBoundaries(n, o, (l, sh, bl) => l - sh >= 0 && l - sh < bl, G)) {
        add(k, L - Z.goreSolid, L, LINE.SOLID, 0, 8);
        add(k, L - Z.goreSolid - Z.goreDotted, L - Z.goreSolid, LINE.DOTTED, 3, 7);
      }
      // merge at the start node: boundary between lanes coming from different carriageways
      const i = ins.get(e) || [];
      if (i.length > 1) for (const k of splitBoundaries(n, i, (l, sh, al) => l + sh >= 0 && l + sh < al, G)) {
        add(k, 0, Z.goreSolid, LINE.SOLID, 0, 8);
        add(k, Z.goreSolid, Z.goreSolid + Z.goreDotted, LINE.DOTTED, 3, 7);
      }
      // lanes that end at the end node (drop) / start at the start node (add): wide dotted line
      const dropM = FREEWAY_CLS(c) ? Z.dropFw : Z.dropSt, addM = FREEWAY_CLS(c) ? Z.addFw : Z.addSt;
      if (LG.tapR[2 * e + 1]) add(LG.tapR[2 * e + 1], L - dropM, L, LINE.DOTTED, 3, 6);
      if (LG.tapL[2 * e + 1]) add(n - LG.tapL[2 * e + 1], L - dropM, L, LINE.DOTTED, 3, 6);
      if (LG.tapR[2 * e]) add(LG.tapR[2 * e], 0, addM, LINE.DOTTED, 3, 6);
      if (LG.tapL[2 * e]) add(n - LG.tapL[2 * e], 0, addM, LINE.DOTTED, 3, 6);
      if (signalIn.has(e)) for (let k = 1; k < n; k++) add(k, b - Z.signalSolid, b, LINE.SOLID, 0, 5);
      for (let k = 1; k < n; k++) {
        for (const [s0, s1, t, p] of resolveZones(a, b, zones.get(k) || [], [LINE.DASHED, pat])) R.push([k, s0, s1, t, COLOR.WHITE, p]);
      }
    }
    // paved run-out wedges (src/geom bounds wider than the sim lanes): outer pavement edge line;
    // the lane edge (k = 0 / k = n) stays solid and the wedge gets chevrons (emitPaint)
    const T = Math.min(90, L);
    const wedge = (side, s0, s1) => { s0 = Math.max(a, s0); s1 = Math.min(b, s1); if (s1 - s0 > 1) R.push([side, s0, s1, LINE.SOLID, COLOR.WHITE, 0]); };
    if (LG.xR[2 * e]) wedge(OUTER_R, 0, T);
    if (LG.xR[2 * e + 1]) wedge(OUTER_R, L - T, L);
    if (!twoWay && LG.xL[2 * e]) wedge(OUTER_L, 0, T);
    if (!twoWay && LG.xL[2 * e + 1]) wedge(OUTER_L, L - T, L);
    R.sort(runOrder);
  }
  // ---- lines lying on another carriageway's lanes (overlapping drawn carriageways, e.g. a divided
  // road's two one-way carriageways converging into a two-way road) are not painted
  if (ctx.occlusion !== false) occlude(LG, edgeRuns);
  // ---- transverse marks
  const marks = buildMarks(LG, J, ctx, level);
  // ---- pack
  return LineSet.pack(LG, J, edgeRuns, marks, level);
}

const runOrder = (p, q) => (p[0] - q[0]) || (p[1] - q[1]);

/**
 * Occlusion: sample every painted run every 3 m; where the line lies more than 0.3 m inside the
 * lane area of another edge (not its own twin) within 2 m of height, the run is split and that
 * stretch becomes type NONE with the OCCLUDED flag (run[6] = 1).
 */
function occlude(LG, edgeRuns) {
  const G = LG.G, E = G.edgeCount, cell = 50, grid = new Map();
  const Q = [];   // per quad: e, y, 4 corners (x, z)
  const P = {};
  for (let e = 0; e < E; e++) {
    if (!edgeRuns[e]) continue;
    const L = G.length[e], n = G.lanes[e];
    const raw = LG.sampleS(e), ss = [raw[0]];
    for (let i = 1; i < raw.length; i++) { const d = raw[i] - raw[i - 1], k = Math.ceil(d / 15); for (let j = 1; j < k; j++) ss.push(raw[i - 1] + (d * j) / k); ss.push(raw[i]); }
    let prev = null;
    for (const s of ss) {
      const R = LG.rightEdge(e, s), w = LG.laneWidth(e, s), lo = R - n * w + 0.3, hi = R - 0.3;
      if (hi - lo < 0.2) { prev = null; continue; }
      LG.offsetPoint(e, s, hi, 0, P); const ax = P.x, az = P.z, y = P.y;
      LG.offsetPoint(e, s, lo, 0, P); const bx = P.x, bz = P.z;
      if (prev) {
        const q = Q.length / 10;
        Q.push(e, (y + prev[4]) / 2, prev[0], prev[1], ax, az, bx, bz, prev[2], prev[3]);
        const xs = [prev[0], ax, bx, prev[2]], zs = [prev[1], az, bz, prev[3]];
        const x0 = Math.floor(Math.min(...xs) / cell), x1 = Math.floor(Math.max(...xs) / cell), z0 = Math.floor(Math.min(...zs) / cell), z1 = Math.floor(Math.max(...zs) / cell);
        for (let gx = x0; gx <= x1; gx++) for (let gz = z0; gz <= z1; gz++) { const k = gx * 100000 + gz; let l = grid.get(k); if (!l) grid.set(k, (l = [])); l.push(q); }
      }
      prev = [ax, az, bx, bz, y];
    }
  }
  const inQuad = (q, x, z) => {
    const o = 10 * q; let sg = 0;
    for (let i = 0; i < 4; i++) {
      const ax = Q[o + 2 + 2 * i], az = Q[o + 3 + 2 * i], bx = Q[o + 2 + 2 * ((i + 1) % 4)], bz = Q[o + 3 + 2 * ((i + 1) % 4)];
      const c = (bx - ax) * (z - az) - (bz - az) * (x - ax);
      if (Math.abs(c) < 1e-9) continue;
      const t = c > 0 ? 1 : -1;
      if (!sg) sg = t; else if (t !== sg) return false;
    }
    return true;
  };
  let occluded = 0;
  for (let e = 0; e < E; e++) {
    const R = edgeRuns[e]; if (!R || !R.length) continue;
    const tw = G.twin[e], n = G.lanes[e], out = [];
    for (const run of R) {
      const [k, s0, s1, t] = run;
      if (!t) { out.push(run); continue; }
      const off = (s) => (k === OUTER_R ? LG.bounds(e, s).right : k === OUTER_L ? LG.bounds(e, s).left : LG.dividerOffset(e, s, k));
      const N = Math.max(1, Math.ceil((s1 - s0) / 3)), hid = new Uint8Array(N + 1);
      let any = false;
      for (let i = 0; i <= N; i++) {
        const s = s0 + ((s1 - s0) * i) / N;
        LG.offsetPoint(e, s, off(s), 0, P);
        const l = grid.get(Math.floor(P.x / cell) * 100000 + Math.floor(P.z / cell));
        if (!l) continue;
        for (const q of l) {
          const g = Q[10 * q];
          if (g === e || g === tw || Math.abs(Q[10 * q + 1] - P.y) > 2) continue;
          if (inQuad(q, P.x, P.z)) { hid[i] = 1; any = true; break; }
        }
      }
      if (!any) { out.push(run); continue; }
      // split into visible / occluded stretches (a sample hides +-half a step around it)
      let i = 0;
      while (i <= N) {
        let j = i; while (j + 1 <= N && hid[j + 1] === hid[i]) j++;
        const a = i === 0 ? s0 : s0 + ((s1 - s0) * (i - 0.5)) / N, b = j === N ? s1 : s0 + ((s1 - s0) * (j + 0.5)) / N;
        if (b - a > 1e-6) {
          if (hid[i]) { out.push([k, a, b, LINE.NONE, run[4], 0, 1]); occluded += b - a; }
          else out.push([k, a, b, t, run[4], run[5]]);
        }
        i = j + 1;
      }
    }
    edgeRuns[e] = out;
  }
  return occluded;
}

/** Boundaries k (0 < k < n) with every lane below k mapped only into one set of hop targets and every lane at or above k only into another. */
function splitBoundaries(n, hops, maps, G) {
  const owner = new Array(n).fill(null);
  for (let l = 0; l < n; l++) {
    const set = [];
    for (const [x, sh] of hops) if (maps(l, sh, G.lanes[x])) set.push(x);
    owner[l] = set.sort((p, q) => p - q).join(',');
  }
  const out = [];
  for (let k = 1; k < n; k++) if (owner[k - 1] && owner[k] && owner[k - 1] !== owner[k] && !owner[k - 1].split(',').some((x) => owner[k].split(',').includes(x))) out.push(k);
  return out;
}

/** Split [a, b] into runs: inside the zones -> inVal, elsewhere -> outVal (arrays appended to [s0, s1]). */
function splitByZones(a, b, zones, inVal, outVal) {
  const z = zones.map(([s0, s1]) => [Math.max(a, s0), Math.min(b, s1)]).filter(([s0, s1]) => s1 > s0).sort((p, q) => p[0] - q[0]);
  const merged = [];
  for (const iv of z) { if (merged.length && iv[0] <= merged[merged.length - 1][1]) merged[merged.length - 1][1] = Math.max(merged[merged.length - 1][1], iv[1]); else merged.push(iv.slice()); }
  const out = [];
  let s = a;
  for (const [s0, s1] of merged) { if (s0 - s > 0.5) out.push([s, s0, ...outVal]); out.push([Math.max(s, s0), s1, ...inVal]); s = s1; }
  if (b - s > 0.5) out.push([s, b, ...outVal]);
  else if (out.length) out[out.length - 1][1] = b;
  // absorb slivers into the neighbour
  return out.filter((r) => r[1] - r[0] > 1e-6);
}

/** Resolve prioritised zones [s0, s1, type, pattern, priority] over [a, b] with a default value. */
function resolveZones(a, b, zones, def) {
  const cuts = new Set([a, b]);
  for (const z of zones) { if (z[1] > z[0]) { cuts.add(Math.max(a, z[0])); cuts.add(Math.min(b, z[1])); } }
  const xs = [...cuts].filter((x) => x >= a && x <= b).sort((p, q) => p - q);
  const out = [];
  for (let i = 0; i < xs.length - 1; i++) {
    const s0 = xs[i], s1 = xs[i + 1], mid = (s0 + s1) / 2;
    if (s1 - s0 < 1e-6) continue;
    let best = null;
    for (const z of zones) if (z[0] <= mid && mid < z[1] && (!best || z[4] > best[4])) best = z;
    const t = best ? best[2] : def[0], p = best ? best[3] : def[1];
    const last = out[out.length - 1];
    if (last && last[2] === t && last[3] === p && Math.abs(last[1] - s0) < 1e-6) last[1] = s1;
    else out.push([s0, s1, t, p]);
  }
  // fold runs shorter than 2 m into the previous run (no paint slivers)
  const res = [];
  for (const r of out) { if (res.length && r[1] - r[0] < 2) res[res.length - 1][1] = r[1]; else res.push(r); }
  if (res.length > 1 && res[0][1] - res[0][0] < 2) { res[1][0] = res[0][0]; res.shift(); }
  return res;
}

/**
 * Transverse marks per edge: [kind, span, s0, s1].
 *  - stop bars at signal / stop / ramp-meter approaches, yield lines at give_way, at the sim's stop
 *    point (src/geom sStop at junction mouths, the control's own stop line mid-block);
 *  - crosswalks on every junction leg that src/geom gave a crosswalk setback (signals, all-way
 *    stops, painted / signalised OSM crossings within 25 m), between the stop point and the box;
 *  - mid-block crosswalks at signalised crossings and painted OSM crossing nodes.
 */
function buildMarks(LG, J, ctx, level) {
  const G = LG.G, E = G.edgeCount, out = new Array(E);
  for (let e = 0; e < E; e++) out[e] = [];
  const cw = PAINT.crosswalk;
  const roadOf = (e) => (G.twin[e] >= 0 && G.twin[e] < e ? G.twin[e] : e);
  const placed = new Map();   // road -> [s in primary frame] (dedupe crosswalks)
  const addCross = (e, s0, s1) => {
    const p = roadOf(e), L = G.length[e];
    const [a, b] = p === e ? [s0, s1] : [L - s1, L - s0];
    if (a < 0 || b > L || b - a < 1) return false;
    const l = placed.get(p) || placed.set(p, []).get(p);
    if (l.some((m) => Math.abs(m - (a + b) / 2) < cw + 1)) return false;
    l.push((a + b) / 2);
    out[p].push([MARK.CROSSWALK, SPAN.ROAD, a, b]);
    return true;
  };
  const mouthStop = (e) => (J ? J.mouthS[2 * e + 1] : G.length[e]);
  const mouthExit = (e) => (J ? J.mouthS[2 * e] : 0);
  // junction crosswalks (legs from crosswalkLegs: the same sets src/geom used for the setbacks)
  const legs = ctx.legs || { crosswalkStop: new Set(), crosswalkExit: new Set() };
  for (const e of legs.crosswalkStop) {
    const L = G.length[e], s = mouthStop(e);
    if (s > L - 0.01 || G.cls[e] === 0 || G.cls[e] === 2) continue;
    addCross(e, s + 0.3, Math.min(L, s + 0.3 + cw));
  }
  for (const e of legs.crosswalkExit) {
    if (G.twin[e] >= 0) continue;   // two-way: drawn from the twin's stop side
    const s = mouthExit(e);
    if (s < 0.01 || G.cls[e] === 0 || G.cls[e] === 2) continue;
    addCross(e, Math.max(0, s - 0.3 - cw), s - 0.3);
  }
  // controls: stop bars, yield lines, crossing signals
  // junction controls first: where a crossing signal shares a stop line with a signal / stop, the
  // junction control's bar (at the sim's stop point) is the one kept
  const rank = (c) => (c.type === 'crossing_signal' ? 2 : c.midBlock ? 1 : 0);
  const ctls = (ctx.controls || []).slice().sort((p, q) => rank(p) - rank(q));
  for (const c of ctls) {
    for (const ap of c.approaches) {
      const g = ap.edge;
      if (g >= E) continue;
      const L = G.length[g], sG = mouthStop(g);
      const s = Math.max(0.5, Math.min(L, !c.midBlock && sG < L - 0.01 && c.type !== 'ramp_meter' ? sG : ap.stopAt));
      if (c.type === 'signal' || c.type === 'stop' || c.type === 'ramp_meter') out[g].push([MARK.STOP, SPAN.INBOUND, s - PAINT.stopW, s]);
      else if (c.type === 'give_way') out[g].push([MARK.YIELD, SPAN.INBOUND, s - 0.3, s]);
      if (c.type === 'crossing_signal' || (c.midBlock && c.type === 'signal')) {
        // crosswalk just past the stop line (stop line ~1.2 m before the crosswalk)
        if (c.type === 'crossing_signal') out[g].push([MARK.STOP, SPAN.INBOUND, s - PAINT.stopW, s]);
        addCross(g, s + 1.2, s + 1.2 + cw);
      }
    }
  }
  // painted / signalised OSM crossing nodes away from junction mouths
  const pts = ctx.points || [];
  if (pts.length) {
    const cell = 50, grid = new Map();
    for (let e = 0; e < E; e++) {
      if (roadOf(e) !== e || G.cls[e] === 0 || G.cls[e] === 2) continue;
      const nd = G.nodesOf(e);
      for (let k = 0; k < nd.length - 1; k++) {
        const x0 = Math.min(G.x(nd[k]), G.x(nd[k + 1])), x1 = Math.max(G.x(nd[k]), G.x(nd[k + 1]));
        const z0 = Math.min(G.z(nd[k]), G.z(nd[k + 1])), z1 = Math.max(G.z(nd[k]), G.z(nd[k + 1]));
        for (let gx = Math.floor(x0 / cell); gx <= Math.floor(x1 / cell); gx++) for (let gz = Math.floor(z0 / cell); gz <= Math.floor(z1 / cell); gz++) {
          const key = gx * 100000 + gz; let l = grid.get(key); if (!l) grid.set(key, (l = [])); l.push(e, k);
        }
      }
    }
    for (const [px, pz] of pts) {
      const l = grid.get(Math.floor(px / cell) * 100000 + Math.floor(pz / cell));
      if (!l) continue;
      let best = 4, be = -1, bs = 0;
      for (let i = 0; i < l.length; i += 2) {
        const e = l[i], k = l[i + 1], nd = G.nodesOf(e);
        const ax = G.x(nd[k]), az = G.z(nd[k]), bx = G.x(nd[k + 1]), bz = G.z(nd[k + 1]);
        const dx = bx - ax, dz = bz - az, LL = dx * dx + dz * dz || 1;
        const t = Math.max(0, Math.min(1, ((px - ax) * dx + (pz - az) * dz) / LL));
        const d = Math.hypot(ax + dx * t - px, az + dz * t - pz);
        if (d < best) {
          let acc = 0; for (let j = 0; j < k; j++) acc += Math.hypot(G.x(nd[j + 1]) - G.x(nd[j]), G.z(nd[j + 1]) - G.z(nd[j]));
          best = d; be = e; bs = acc + Math.sqrt(LL) * t;
        }
      }
      if (be < 0) continue;
      // polyline s -> data s, and keep clear of the junction boxes (those legs got their crosswalk above)
      let geo = 0; const nd = G.nodesOf(be); for (let j = 0; j < nd.length - 1; j++) geo += Math.hypot(G.x(nd[j + 1]) - G.x(nd[j]), G.z(nd[j + 1]) - G.z(nd[j]));
      const L = G.length[be], s = geo > 0 ? (bs * L) / geo : bs;
      const tw = G.twin[be];
      const lo = Math.max(mouthExit(be), tw >= 0 ? L - mouthStop(tw) : 0), hi = Math.min(mouthStop(be), tw >= 0 ? L - mouthExit(tw) : L);
      if (s - cw / 2 < lo + 1 || s + cw / 2 > hi - 1) continue;
      // within 25 m of a junction mouth the point already gave that leg its crosswalk (crosswalkLegs)
      const jEnd = mouthStop(be) < L - 0.01 || (tw >= 0 && mouthExit(tw) > 0.01), jStart = mouthExit(be) > 0.01 || (tw >= 0 && mouthStop(tw) < L - 0.01);
      if ((jEnd && L - s <= 25) || (jStart && s <= 25)) continue;
      addCross(be, s - cw / 2, s + cw / 2);
    }
  }
  // one stop / yield bar per approach position (a control and a crossing signal can share a stop line)
  for (let e = 0; e < E; e++) {
    // a stop bar wins over a yield line at the same place (a signal and a give_way sharing a node)
    const keep = [];
    for (const kind of [MARK.STOP, MARK.YIELD]) for (const m of out[e]) if (m[0] === kind && !keep.some((p) => Math.abs(p[3] - m[3]) < 1)) keep.push(m);
    out[e] = keep.concat(out[e].filter((m) => m[0] === MARK.CROSSWALK)).sort((p, q) => p[2] - q[2]);
  }
  return out;
}

// ------------------------------------------------------------------ the line set
/** data/lines.bin: runs, marks and the per-edge offset parameters (small; renderers + sim). */
export const LINE_SECTIONS = ['level', 'runFirst', 'runInfo', 'runS', 'markFirst', 'markInfo', 'markS',
  'lanes', 'length', 'laneW', 'laneWEnd', 'Rd', 'Rend', 'tau', 'xR', 'xL', 'mW'];
/** data/lines-geom.bin: the reference-polyline mirror for the GPU (positions, normals, heading, curvature). */
export const LINE_GEOM_SECTIONS = ['vFirst', 'vx', 'vy', 'vz', 'mx', 'mz', 'vs', 'vHead', 'vCurv'];

export class LineSet {
  /** Pack per-edge run lists into flat typed arrays. */
  static pack(LG, J, edgeRuns, marks, level) {
    const G = LG.G, E = G.edgeCount;
    let R = 0, M = 0;
    for (let e = 0; e < E; e++) { R += edgeRuns[e].length; M += marks[e].length; }
    const runFirst = new Uint32Array(E + 1), runInfo = new Uint32Array(R), runS = new Float32Array(2 * R);
    const markFirst = new Uint32Array(E + 1), markInfo = new Uint32Array(M), markS = new Float32Array(2 * M);
    let r = 0, m = 0;
    for (let e = 0; e < E; e++) {
      runFirst[e] = r;
      for (const [k, s0, s1, t, c, p, occ] of edgeRuns[e]) { runInfo[r] = (k | (t << RUN_TYPE_SHIFT) | (c << RUN_COLOR_SHIFT) | (p << RUN_PAT_SHIFT) | (occ ? RUN_OCCLUDED : 0)) >>> 0; runS[2 * r] = s0; runS[2 * r + 1] = s1; r++; }
      markFirst[e] = m;
      for (const [kind, span, s0, s1] of marks[e]) { markInfo[m] = kind | (span << 8); markS[2 * m] = s0; markS[2 * m + 1] = s1; m++; }
    }
    runFirst[E] = r; markFirst[E] = m;
    // lane-geometry mirror (offsets as a function of s) + reference-line heading / curvature per vertex
    const V = LG.vFirst[E];
    const vHead = new Float32Array(V), vCurv = new Float32Array(V);
    for (let e = 0; e < E; e++) {
      const v0 = LG.vFirst[e], v1 = LG.vFirst[e + 1] - 1;
      for (let i = v0; i <= v1; i++) {
        const a = Math.max(v0, i - 1), b = Math.min(v1, i + 1);
        const pa = i > v0 ? Math.atan2(LG.vz[i] - LG.vz[a], LG.vx[i] - LG.vx[a]) : null;
        const pb = i < v1 ? Math.atan2(LG.vz[b] - LG.vz[i], LG.vx[b] - LG.vx[i]) : null;
        vHead[i] = pa === null ? pb : pb === null ? pa : pa + wrap(pb - pa) / 2;
        if (pa !== null && pb !== null) {
          const ds = (LG.vs[b] - LG.vs[a]) / 2 || 1;
          vCurv[i] = -wrap(pb - pa) / ds;   // + = curving left
        }
      }
    }
    let mW = new Float32Array(0);
    if (J) {
      mW = new Float32Array(2 * J.moveCount);
      for (let q = 0; q < J.moveCount; q++) {
        const a = J.mInfo[7 * q], b = J.mInfo[7 * q + 2];
        mW[2 * q] = LG.laneWidth(a, J.mF[4 * q]); mW[2 * q + 1] = LG.laneWidth(b, J.mF[4 * q + 1]);
      }
    }
    const f32 = (a) => Float32Array.from(a);
    const arrays = { level, runFirst, runInfo, runS, markFirst, markInfo, markS,
      lanes: Uint8Array.from(G.lanes), length: f32(G.length), laneW: f32(LG.laneW), laneWEnd: f32(LG.laneWEnd), Rd: f32(LG.Rd), Rend: f32(LG.Rend), tau: f32(LG.tau),
      xR: Uint8Array.from(LG.xR), xL: Uint8Array.from(LG.xL),
      vFirst: Uint32Array.from(LG.vFirst), vx: f32(LG.vx), vy: f32(LG.vy), vz: f32(LG.vz), mx: f32(LG.mx), mz: f32(LG.mz), vs: f32(LG.vs), vHead, vCurv, mW };
    return new LineSet(LG, J, arrays);
  }

  constructor(LG, J, a) {
    this.LG = LG; this.G = LG.G; this.J = J; Object.assign(this, a);
    this.runCount = this.runInfo.length; this.markCount = this.markInfo.length;
    this._tmp = {}; this._tmp2 = {};
  }
  arrays() { const o = {}; for (const k of [...LINE_SECTIONS, ...LINE_GEOM_SECTIONS]) if (this[k]) o[k] = this[k]; return o; }

  // ---------------------------------------------------------------- runs
  /** Runs of edge e: [{k, s0, s1, type, color, pattern}]. */
  runs(e) {
    const out = [];
    for (let r = this.runFirst[e]; r < this.runFirst[e + 1]; r++) out.push(this.run(r));
    return out;
  }
  run(r) {
    const i = this.runInfo[r];
    return { k: i & RUN_K, s0: this.runS[2 * r], s1: this.runS[2 * r + 1], type: (i >> RUN_TYPE_SHIFT) & 15, color: (i >> RUN_COLOR_SHIFT) & 15, pattern: (i >> RUN_PAT_SHIFT) & 255, occluded: !!(i & RUN_OCCLUDED) };
  }
  marks(e) {
    const out = [];
    for (let q = this.markFirst[e]; q < this.markFirst[e + 1]; q++) {
      const i = this.markInfo[q];
      out.push({ kind: i & 255, span: (i >> 8) & 255, s0: this.markS[2 * q], s1: this.markS[2 * q + 1] });
    }
    return out;
  }
  /** Line of boundary k at s: {type, color, pattern, occluded} (type NONE where nothing is painted; occluded: the
   *  line would lie on another carriageway's lanes, where drawn carriageways overlap). */
  lineAt(e, k, s, out = {}) {
    out.type = 0; out.color = 0; out.pattern = 0; out.occluded = false;
    for (let r = this.runFirst[e]; r < this.runFirst[e + 1]; r++) {
      const i = this.runInfo[r];
      if ((i & RUN_K) !== k) continue;
      if (s >= this.runS[2 * r] && s <= this.runS[2 * r + 1]) {
        out.type = (i >> RUN_TYPE_SHIFT) & 15; out.color = (i >> RUN_COLOR_SHIFT) & 15; out.pattern = (i >> RUN_PAT_SHIFT) & 255; out.occluded = !!(i & RUN_OCCLUDED);
        return out;
      }
    }
    return out;
  }
  /**
   * Lateral position of boundary k at s, in metres right of the reference polyline. k = 0 .. lanes
   * is the exact lane boundary (src/geom dividerOffset); OUTER_R / OUTER_L the run-out pavement edge.
   * In sim lat (m left of the right carriageway edge): lat = k * LG.laneWidth(e, s).
   */
  lineOffset(e, k, s) {
    if (k === OUTER_R) return this.LG.bounds(e, s).right;
    if (k === OUTER_L) return this.LG.bounds(e, s).left;
    return this.LG.dividerOffset(e, s, k);
  }
  /** Offset of the paint centre of boundary k (edge lines are inset so the whole line is on the pavement). */
  paintOffset(e, k, s) {
    const n = this.G.lanes[e];
    const o = this.lineOffset(e, k, s);
    if (k === 0 || k === OUTER_R) return o - PAINT.edgeInset;
    if (k === OUTER_L || (k === n && this.G.twin[e] < 0)) return o + PAINT.edgeInset;
    return o;
  }

  // ---------------------------------------------------------------- perception
  /**
   * What a driver sees of the lines: pos = {edge, s, lat} (sim coordinates: lat m left of the
   * right carriageway edge) or {conn, t, lat} on a junction connector (lat m left of the
   * connector's right lane edge, 0 .. width; the connector centre is lat = width / 2).
   * opts: { route: Int32Array|number[] of edges, routePos: index of pos.edge in it (or of the
   *   connector's in-edge), ahead: [8, 16, 32, 64] metres }.
   * Returns (into `out` if given):
   *   lane, lanes, laneWidth, kind ('edge' | 'conn'),
   *   right / left: { dist (m from the car to the line centre), type, color },
   *   lines: [{ k, lat, type, color }] every boundary across the carriageway (k = 0 .. lanes),
   *   hx, hz (unit heading of the car's lane), curvature (1/m, + = curving left),
   *   ahead: [{ d, kind, edge, s, conn, t, lane, laneWidth, x, y, z, hx, hz, curvature,
   *            fwd, side (m forward / m to the left, in the frame of the car's lane tangent), dHeading (rad, + = left),
   *            right: {type, color}, left: {type, color}, laneEnds (the lane does not continue) }]
   * On connectors the lines are virtual lane edges (type NONE): there is no paint in a junction box.
   */
  lineGeom(pos, opts = {}, out = {}) {
    const LG = this.LG, G = this.G, P = this._tmp;
    const aheadD = opts.ahead || [8, 16, 32, 64];
    out.right = out.right || {}; out.left = out.left || {};
    if (pos.conn !== undefined && pos.conn >= 0) {
      const m = pos.conn, J = this.J, len = J.mF[4 * m + 2], u = len > 0 ? Math.max(0, Math.min(1, pos.t / len)) : 0;
      const w = this.mW[2 * m] + (this.mW[2 * m + 1] - this.mW[2 * m]) * u;
      out.kind = 'conn'; out.lane = 0; out.lanes = 1; out.laneWidth = w;
      out.right.dist = pos.lat; out.right.type = 0; out.right.color = 0;
      out.left.dist = w - pos.lat; out.left.type = 0; out.left.color = 0;
      out.lines = [{ k: 0, lat: 0, type: 0, color: 0 }, { k: 1, lat: w, type: 0, color: 0 }];
      this._connPose(m, pos.t, P);
      out.hx = P.hx; out.hz = P.hz; out.curvature = P.k; out.x = P.x; out.y = P.y; out.z = P.z;
    } else {
      const e = pos.edge, s = Math.max(0, Math.min(G.length[e], pos.s)), n = G.lanes[e], w = LG.laneWidth(e, s);
      let lane = Math.floor(pos.lat / w); lane = lane < 0 ? 0 : lane >= n ? n - 1 : lane;
      out.kind = 'edge'; out.lane = lane; out.lanes = n; out.laneWidth = w;
      const lt = this.lineAt(e, lane, s, this._tmp2);
      out.right.dist = pos.lat - lane * w; out.right.type = lt.type; out.right.color = lt.color;
      this.lineAt(e, lane + 1, s, lt);
      out.left.dist = (lane + 1) * w - pos.lat; out.left.type = lt.type; out.left.color = lt.color;
      out.lines = [];
      for (let k = 0; k <= n; k++) { this.lineAt(e, k, s, lt); out.lines.push({ k, lat: k * w, type: lt.type, color: lt.color }); }
      this._edgePose(e, s, (lane + 0.5) * w, P);
      out.hx = P.hx; out.hz = P.hz; out.curvature = P.k; out.x = P.x; out.y = P.y; out.z = P.z;
    }
    // look-ahead along the car's lane (lane centre), following the route through connectors
    const hx = out.hx, hz = out.hz, x0 = out.x, z0 = out.z, h0 = Math.atan2(hz, hx);
    out.ahead = out.ahead || [];
    out.ahead.length = aheadD.length;
    const st = pos.conn !== undefined && pos.conn >= 0
      ? { kind: 1, e: -1, s: 0, m: pos.conn, t: pos.t, lane: 0, rp: opts.routePos ?? -1, ended: false }
      : { kind: 0, e: pos.edge, s: Math.max(0, Math.min(G.length[pos.edge], pos.s)), m: -1, t: 0, lane: out.lane, rp: opts.routePos ?? -1, ended: false };
    let done = 0;
    for (let i = 0; i < aheadD.length; i++) {
      this._advance(st, aheadD[i] - done, opts.route);
      done = aheadD[i];
      const A = out.ahead[i] || (out.ahead[i] = { right: {}, left: {} });
      A.d = aheadD[i]; A.laneEnds = st.ended;
      if (st.kind === 1) {
        const m = st.m, len = this.J.mF[4 * m + 2], u = len > 0 ? Math.max(0, Math.min(1, st.t / len)) : 0;
        this._connPose(m, st.t, P);
        A.kind = 'conn'; A.edge = -1; A.s = 0; A.conn = m; A.t = st.t; A.lane = 0;
        A.laneWidth = this.mW[2 * m] + (this.mW[2 * m + 1] - this.mW[2 * m]) * u;
        A.right.type = 0; A.right.color = 0; A.left.type = 0; A.left.color = 0;
      } else {
        const e = st.e, w = LG.laneWidth(e, st.s);
        this._edgePose(e, st.s, (st.lane + 0.5) * w, P);
        A.kind = 'edge'; A.edge = e; A.s = st.s; A.conn = -1; A.t = 0; A.lane = st.lane; A.laneWidth = w;
        const lt = this.lineAt(e, st.lane, st.s, this._tmp2); A.right.type = lt.type; A.right.color = lt.color;
        this.lineAt(e, st.lane + 1, st.s, lt); A.left.type = lt.type; A.left.color = lt.color;
      }
      A.x = P.x; A.y = P.y; A.z = P.z; A.hx = P.hx; A.hz = P.hz; A.curvature = P.k;
      const dx = P.x - x0, dz = P.z - z0;
      A.fwd = dx * hx + dz * hz; A.side = dx * hz - dz * hx;   // side: m to the left (left vector = (hz, -hx))
      A.dHeading = -wrap(Math.atan2(P.hz, P.hx) - h0);
    }
    return out;
  }
  /** Pose + curvature of the lane path at (e, s, lat): numeric over +-2 m (exact lane geometry, tapers included). */
  _edgePose(e, s, lat, P) {
    const LG = this.LG, L = this.G.length[e], d = 2;
    const sa = Math.max(0, s - d), sb = Math.min(L, s + d);
    LG.latPoint(e, s, lat, P);
    const A = LG.latPoint(e, sa, lat, this._pa || (this._pa = {})), B = LG.latPoint(e, sb, lat, this._pb || (this._pb = {}));
    const arc = Math.hypot(B.x - A.x, B.z - A.z);
    P.k = arc > 1e-3 ? -wrap(Math.atan2(B.hz, B.hx) - Math.atan2(A.hz, A.hx)) / arc : 0;
    return P;
  }
  _connPose(m, t, P) {
    const J = this.J, len = J.mF[4 * m + 2], d = 2;
    J.arcPoint(m, Math.max(0, Math.min(len, t)), P);
    const A = J.arcPoint(m, Math.max(0, t - d), this._pa || (this._pa = {})), B = J.arcPoint(m, Math.min(len, t + d), this._pb || (this._pb = {}));
    const arc = Math.hypot(B.x - A.x, B.z - A.z);
    P.k = arc > 1e-3 ? -wrap(Math.atan2(B.hz, B.hx) - Math.atan2(A.hz, A.hx)) / arc : 0;
    return P;
  }
  /** Next edge after e: the route's, else the straightest strict continuation, else the straightest movement. */
  _nextEdge(e, st, route) {
    const G = this.G;
    if (route && st.rp >= 0 && st.rp + 1 < route.length && route[st.rp] === e) return route[st.rp + 1];
    const outs = G.outEdges(G.to(e));
    let best = -1, bt = 9, bestStrict = false;
    for (const b of outs) {
      if (b === G.twin[e] && outs.length > 1) continue;
      const strict = hopMap(G, e, b).strict, t = Math.abs(turnAngle(G, e, b));
      if ((strict && !bestStrict) || (strict === bestStrict && t < bt)) { best = b; bt = t; bestStrict = strict; }
    }
    return best;
  }
  /** Move the look-ahead state d metres along the lane (lane centre), through hops and connectors. */
  _advance(st, d, route) {
    const G = this.G, J = this.J;
    let guard = 0;
    while (d > 1e-6 && guard++ < 64) {
      if (st.kind === 1) {
        const len = J.mF[4 * st.m + 2];
        if (st.t + d <= len) { st.t += d; return; }
        d -= len - st.t;
        st.kind = 0; st.e = J.mInfo[7 * st.m + 2]; st.lane = J.mInfo[7 * st.m + 3]; st.s = J.mF[4 * st.m + 1];
        if (route && st.rp >= 0) st.rp++;
        continue;
      }
      const e = st.e, L = G.length[e];
      const en = st.ended ? -1 : this._nextEdge(e, st, route);
      let strict = en >= 0 && hopMap(G, e, en).strict, m = -1, end = L;
      if (en >= 0 && !strict && J) {
        m = J.movement(e, st.lane, en);
        if (m < 0) for (let l = 0; l < G.lanes[e] && m < 0; l++) m = J.movement(e, l, en);   // a lane without that movement: nearest one
        if (m >= 0) end = J.mF[4 * m];
      }
      if (st.s + d <= end || en < 0 || (!strict && m < 0)) {
        const want = st.s + d;
        st.s = Math.min(want, Math.max(st.s, end));
        if (want > end) st.ended = true;
        return;
      }
      d -= Math.max(0, end - st.s);
      if (strict) {
        const h = hopMap(G, e, en), l = st.lane - h.shift;
        if (l < 0 || l >= G.lanes[en]) { st.s = L; st.ended = true; return; }   // this lane ends at the node
        st.e = en; st.s = 0; st.lane = l;
        if (route && st.rp >= 0) st.rp++;
      } else { st.kind = 1; st.m = m; st.t = 0; }
    }
  }
}

function wrap(a) { while (a > Math.PI) a -= 2 * Math.PI; while (a < -Math.PI) a += 2 * Math.PI; return a; }

const TYPES = { Float32: Float32Array, Uint32: Uint32Array, Uint8: Uint8Array };
/** Load data/lines.bin (tools/lines/build_lines.mjs) for the lane geometry LG (and junctions J, for connectors). */
export async function loadLineSet(metaUrl, LG, J = null, { geom = false } = {}) {
  const meta = await (await fetch(metaUrl)).json();
  const dir = metaUrl.replace(/[^/]*$/, '');
  const bufs = {};
  for (const f of new Set(Object.values(meta.sections).map((s) => s.file || meta.file))) {
    if (!geom && f !== meta.file) continue;
    bufs[f] = await (await fetch(dir + f)).arrayBuffer();
  }
  return lineSetFromBuffers(meta, bufs, LG, J);
}
/** bufs: { [file]: ArrayBuffer } (lines.bin required, lines-geom.bin optional). */
export function lineSetFromBuffers(meta, bufs, LG, J = null) {
  if (meta.edgeCount !== LG.G.edgeCount) throw new Error(`lines.bin has ${meta.edgeCount} edges, the road graph ${LG.G.edgeCount}: rebuild (node tools/lines/build_lines.mjs)`);
  const a = {};
  for (const [k, s] of Object.entries(meta.sections)) {
    const buf = bufs[s.file || meta.file];
    if (buf) a[k] = new TYPES[s.type](buf, s.byteOffset, s.length);
  }
  return new LineSet(LG, J, a);
}

// ------------------------------------------------------------------ paint (renderers)
/**
 * Paint triangles for edge e. sink(role, V, T, tag): V flat [x, y, z, ...] world positions (y = road
 * elevation + opts.lift), T triangle indices (front face up). Roles: 'white' (solid white,
 * chevrons), 'dash' (white dashed / dotted), 'yellow' (all yellow), 'stopbar', 'crosswalk'.
 * The centre line of a two-way road is drawn from its primary direction only. tag = {kind: 'line', e, k,
 * type, color, ss (s per vertex pair), d (double: +-gap)} | {kind: 'chevron' | 'mark', e, ...} (checks).
 * opts: { lift = 0.02, far = false (solid edge / centre lines only, widened by farScale), farScale = 3,
 *         marks = true, maxStep = Infinity (densify solid lines) }
 */
export function emitPaint(LS, e, sink, opts = {}) {
  const LG = LS.LG, G = LS.G, n = G.lanes[e], tw = G.twin[e], primary = tw < 0 || e < tw;
  const lift = opts.lift ?? 0.02, far = !!opts.far, fs = opts.farScale ?? 3, maxStep = opts.maxStep ?? Infinity;
  const A = {}, B = {};
  const verts = (s, o0, o1, V) => {
    LG.offsetPoint(e, s, o0, 0, A); LG.offsetPoint(e, s, o1, 0, B);
    V.push(A.x, A.y + lift, A.z, B.x, B.y + lift, B.z);
  };
  // strip between offset functions lo(s) / hi(s) over the samples
  const strip = (role, ss, lo, hi, tag) => {
    if (ss.length < 2) return;
    const V = [], T = [];
    for (const s of ss) verts(s, lo(s), hi(s), V);
    for (let i = 0; i < ss.length - 1; i++) { const a = 2 * i; tri(V, T, a, a + 2, a + 1); tri(V, T, a + 1, a + 2, a + 3); }
    if (tag) tag.ss = ss;
    sink(role, V, T, tag);
  };
  // lines are straight between the geometry's samples (vertices + transition steps): no densification needed
  const samples = (s0, s1) => {
    const out = [s0];
    let last = s0;
    for (const s of LG.sampleS(e)) if (s > s0 + 0.05 && s < s1 - 0.05 && (!far || s - last > 30)) { out.push(s); last = s; }
    out.push(s1);
    const res = [out[0]];
    for (let i = 1; i < out.length; i++) { const d = out[i] - out[i - 1], k = Math.ceil(d / maxStep); for (let j = 1; j < k; j++) res.push(out[i - 1] + (d * j) / k); res.push(out[i]); }
    return res;
  };
  for (let r = LS.runFirst[e]; r < LS.runFirst[e + 1]; r++) {
    const i = LS.runInfo[r], k = i & RUN_K, type = (i >> RUN_TYPE_SHIFT) & 15, color = (i >> RUN_COLOR_SHIFT) & 15, pat = (i >> RUN_PAT_SHIFT) & 255;
    const s0 = LS.runS[2 * r], s1 = LS.runS[2 * r + 1];
    if (type === LINE.NONE) continue;
    if (k === n && tw >= 0 && !primary) continue;   // centre drawn once
    const outer = k === 0 || k === n || k >= OUTER_R;
    if (far && (!outer || k >= OUTER_R || G.cls[e] > 7)) continue;
    const off = (s) => LS.paintOffset(e, k, s);
    const yellow = color === COLOR.YELLOW;
    const tag = () => ({ kind: 'line', e, k, type, color });
    if (far) { const hw = (type === LINE.DOUBLE ? PAINT.doubleGap + PAINT.lineW : PAINT.lineW) * fs / 2; strip(yellow ? 'yellow' : 'white', samples(s0, s1), (s) => off(s) - hw, (s) => off(s) + hw, { ...tag(), far: true }); continue; }
    if (type === LINE.SOLID || type === LINE.DOUBLE) {
      const hw = PAINT.lineW / 2, ss = samples(s0, s1);
      const role = yellow ? 'yellow' : 'white';
      if (type === LINE.SOLID) strip(role, ss, (s) => off(s) - hw, (s) => off(s) + hw, tag());
      else for (const d of [-PAINT.doubleGap, PAINT.doubleGap]) strip(role, ss, (s) => off(s) + d - hw, (s) => off(s) + d + hw, { ...tag(), d });
    } else {
      const [on, gap] = PATTERNS[pat] || PATTERNS[2], period = on + gap;
      const hw = (type === LINE.DOTTED ? PAINT.dottedW : PAINT.lineW) / 2, role = yellow ? 'yellow' : 'dash';
      // dash phase from s = 0 of the edge (continuous inside a run, gap centred on s = 0)
      for (let s = Math.floor((s0 - gap / 2) / period) * period + gap / 2; s < s1; s += period) {
        const a = Math.max(s0, s), b = Math.min(s1, s + on);
        if (b - a < Math.min(0.3, on * 0.5)) continue;
        strip(role, b - a > maxStep ? samples(a, b) : [a, b], (x) => off(x) - hw, (x) => off(x) + hw, tag());
      }
    }
    // chevrons across a run-out wedge (between the lane edge and the outer pavement edge)
    if (k >= OUTER_R) {
      const lane = (s) => (k === OUTER_R ? LS.lineOffset(e, 0, s) : LS.lineOffset(e, n, s));
      for (let sc = s0 + 2; sc + 4 < s1; sc += 8) {
        const o1 = LS.lineOffset(e, k, sc + 4), o0 = lane(sc);
        if (Math.abs(o1 - lane(sc + 4)) < 0.8) continue;
        const sg = k === OUTER_R ? 1 : -1;
        stripe(LG, e, sc, o0 + sg * 0.1, sc + 4, o1 - sg * 0.15, 0.15, lift, sink, 'white', { kind: 'chevron', e, k, s0: sc, s1: sc + 4 });
      }
    }
  }
  if (far || opts.marks === false) return;
  for (let q = LS.markFirst[e]; q < LS.markFirst[e + 1]; q++) {
    const info = LS.markInfo[q], kind = info & 255, span = (info >> 8) & 255, s0 = LS.markS[2 * q], s1 = LS.markS[2 * q + 1];
    const sm = (s0 + s1) / 2;
    let oR, oL;
    if (span === SPAN.ROAD) {
      oR = LG.bounds(e, sm).right;
      oL = tw >= 0 ? -LG.bounds(tw, G.length[e] - sm).right : LG.bounds(e, sm).left;
    } else {
      oR = LS.lineOffset(e, 0, sm); oL = LS.lineOffset(e, n, sm);
    }
    const lo = Math.min(oR, oL), hi = Math.max(oR, oL);
    const centreClear = tw >= 0 ? PAINT.doubleGap + PAINT.lineW : 0.1;
    const tg = { kind: 'mark', e, mark: kind, s0, s1 };
    if (kind === MARK.STOP) patch(LG, e, s0, s1, lo + centreClear, hi - 0.1, lift, sink, 'stopbar', tg);
    else if (kind === MARK.YIELD) for (let o = lo + centreClear; o < hi - 0.5; o += 1.0) patch(LG, e, s0, s1, o, o + 0.5, lift, sink, 'stopbar', tg);
    else if (kind === MARK.CROSSWALK) for (let o = lo + 0.3; o + PAINT.barW <= hi - 0.3; o += PAINT.barW + PAINT.barGap) patch(LG, e, s0, s1, o, o + PAINT.barW, lift, sink, 'crosswalk', tg);
  }
}

/** Front-face-up triangle (counter-clockwise seen from above: (b - a) x (c - a) has +y). */
function tri(V, T, a, b, c) {
  const ax = V[3 * a], az = V[3 * a + 2], bx = V[3 * b], bz = V[3 * b + 2], cx = V[3 * c], cz = V[3 * c + 2];
  const ny = (bz - az) * (cx - ax) - (bx - ax) * (cz - az);
  if (ny >= 0) T.push(a, b, c); else T.push(a, c, b);
}
function patch(LG, e, s0, s1, o0, o1, lift, sink, role, tag) {
  if (o1 - o0 < 0.05) return;
  const V = [], T = [], P = {};
  for (const s of [s0, s1]) for (const o of [o0, o1]) { LG.offsetPoint(e, s, o, 0, P); V.push(P.x, P.y + lift, P.z); }
  tri(V, T, 0, 1, 2); tri(V, T, 1, 3, 2);
  sink(role, V, T, tag);
}
function stripe(LG, e, sA, oA, sB, oB, hw, lift, sink, role, tag) {
  const P = {}, Q = {};
  LG.offsetPoint(e, sA, oA, 0, P); LG.offsetPoint(e, sB, oB, 0, Q);
  let dx = Q.x - P.x, dz = Q.z - P.z; const len = Math.hypot(dx, dz) || 1; dx /= len; dz /= len;
  const px = -dz * hw, pz = dx * hw;
  const V = [P.x - px, P.y + lift, P.z - pz, P.x + px, P.y + lift, P.z + pz, Q.x - px, Q.y + lift, Q.z - pz, Q.x + px, Q.y + lift, Q.z + pz], T = [];
  tri(V, T, 0, 1, 2); tri(V, T, 1, 3, 2);
  sink(role, V, T, tag);
}
