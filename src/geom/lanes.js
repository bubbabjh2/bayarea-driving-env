// Lane geometry: THE single definition of where every lane is (renderer ribbons + markings,
// the sim's edgePoint, and the GPU mirror). See src/geom/README.md.
//
//   const G = graphFromRoadSets(main, local);
//   const LG = buildLaneGeometry(G);
//   LG.latPoint(e, s, lat)    -> {x, y, z, hx, hz}   lat = m from the right carriageway edge (sim)
//   LG.lanePoint(e, s, lane)  -> same, lane 0 = rightmost lane centre
//   LG.offsetPoint(e, s, off) -> same, off = m to the right of the reference polyline
//   LG.rightEdge(e, s)        -> off of the right carriageway edge (sim lat 0)
//   LG.bounds(e, s)           -> {right, left} drawn pavement edges (offsets, incl. tapers)
// s is the DATA edge length coordinate (0 .. edgeLength[e]); heading (hx, hz) is a unit vector.
import { LANE_W, LANE_WIDTH_CLASS } from './graph.js';
import { hopMap, turnAngle } from './hops.js';

const FILLET_MIN_TURN = 0.2;     // rad (~11.5 deg): vertices turning more get a curve
const FILLET_STEP = 0.15;        // rad per resampled vertex (~8.6 deg)
const FILLET_MAX_CUT = 12;       // m cut back from the corner along each segment
const TAPER_M = 90;              // lane add/drop taper (shorter only if the edge is shorter)
const MIN_TAU = 60;              // min length of a lateral re-anchoring transition
const TAU_PER_M = 30;            // ... and 30 m per metre of lateral shift (about 30:1)
// opposite-carriageway clearance (docs/carriageway-fix.md); GEOM_NO_OPPOSING=1 (node) or opts.opposing = false turns it off
// taper slope limit of the opposing pass (junction-fix build, 2026-10-24 ~21:30): on with the data swap; until then
// GEOM_TAPER_LIMIT=1 (node) builds the new geometry
const TAPER_LIMIT = true;   // swapped in with the junction-crossing data (old: data/pre-junction-fix/)
const OPPOSING_DEFAULT = true;   // swapped in 2026-10-24 with data/junctions.bin, data/lines.bin, data/tiles (old: data/pre-carriageway-fix/)

function smooth(u) { u = u < 0 ? 0 : u > 1 ? 1 : u; return u * u * (3 - 2 * u); }

export function buildLaneGeometry(G, opts = {}) {
  const E = G.edgeCount;
  const LW = opts.laneWidth || LANE_W;
  // ---------------------------------------------------------------- 1. resampled polylines
  const vFirst = new Uint32Array(E + 1);
  const px = [], py = [], pz = [], ps = [];
  for (let e = 0; e < E; e++) {
    vFirst[e] = px.length;
    const f = G.first[e], c = G.count[e];
    const X = (k) => G.x(G.poly[f + k]), Y = (k) => G.y(G.poly[f + k]), Z = (k) => G.z(G.poly[f + k]);
    const segL = [];
    for (let k = 0; k < c - 1; k++) segL.push(Math.hypot(X(k + 1) - X(k), Z(k + 1) - Z(k)));
    const push = (x, y, z) => { px.push(x); py.push(y); pz.push(z); };
    push(X(0), Y(0), Z(0));
    for (let k = 1; k < c - 1; k++) {
      const ax = X(k) - X(k - 1), az = Z(k) - Z(k - 1), bx = X(k + 1) - X(k), bz = Z(k + 1) - Z(k);
      const la = segL[k - 1], lb = segL[k];
      if (la < 1e-6 || lb < 1e-6) continue;
      const th = Math.acos(Math.max(-1, Math.min(1, (ax * bx + az * bz) / (la * lb))));
      if (th < FILLET_MIN_TURN) { push(X(k), Y(k), Z(k)); continue; }
      const cut = Math.min(FILLET_MAX_CUT, 0.45 * la, 0.45 * lb);
      const A = [X(k) - (ax / la) * cut, Y(k) + (Y(k - 1) - Y(k)) * (cut / la), Z(k) - (az / la) * cut];
      const B = [X(k) + (bx / lb) * cut, Y(k) + (Y(k + 1) - Y(k)) * (cut / lb), Z(k) + (bz / lb) * cut];
      const V = [X(k), Y(k), Z(k)];
      const n = Math.max(2, Math.ceil(th / FILLET_STEP));
      for (let j = 0; j <= n; j++) {
        const t = j / n, a = (1 - t) * (1 - t), b = 2 * (1 - t) * t, d = t * t;
        push(a * A[0] + b * V[0] + d * B[0], a * A[1] + b * V[1] + d * B[1], a * A[2] + b * V[2] + d * B[2]);
      }
    }
    push(X(c - 1), Y(c - 1), Z(c - 1));
    // cumulative geometric length, rescaled to the data length so s is the sim's coordinate
    const v0 = vFirst[e], n = px.length - v0;
    let acc = 0;
    ps.push(0);
    for (let i = 1; i < n; i++) { acc += Math.hypot(px[v0 + i] - px[v0 + i - 1], pz[v0 + i] - pz[v0 + i - 1]); ps.push(acc); }
    const k = acc > 0 ? G.length[e] / acc : 1;
    for (let i = 0; i < n; i++) ps[v0 + i] *= k;
  }
  vFirst[E] = px.length;
  const V = px.length;
  const vx = Float64Array.from(px), vy = Float32Array.from(py), vz = Float64Array.from(pz), vs = Float64Array.from(ps);

  // ---------------------------------------------------------------- 2. strict groups at nodes
  // edge-end key: 2e (start) / 2e+1 (end). Union-find over strict hops.
  const parent = new Int32Array(2 * E).map((_, i) => i);
  const find = (a) => { while (parent[a] !== a) { parent[a] = parent[parent[a]]; a = parent[a]; } return a; };
  const hops = [];   // [a, b, shift]
  for (let n = 0; n < G.nodeCount; n++) {
    const ins = G.inEdges(n), outs = G.outEdges(n);
    if (!ins.length || !outs.length) continue;
    for (const a of ins) for (const b of outs) {
      if (b === G.twin[a]) continue;
      const h = hopMap(G, a, b);
      if (!h.strict) continue;
      hops.push(a, b, h.shift);
      const ra = find(2 * a + 1), rb = find(2 * b);
      if (ra !== rb) parent[rb] = ra;
    }
  }
  const H = hops.length / 3;
  // group frame = bisector of the straightest strict pair in the group
  const segDir = (e, end) => {
    const v0 = vFirst[e], v1 = vFirst[e + 1] - 1;
    const a = end ? v1 - 1 : v0, b = end ? v1 : v0 + 1;
    const dx = vx[b] - vx[a], dz = vz[b] - vz[a], L = Math.hypot(dx, dz) || 1;
    return [dx / L, dz / L];
  };
  const bestTurn = new Map(), frame = new Map();
  for (let k = 0; k < H; k++) {
    const a = hops[3 * k], b = hops[3 * k + 1], r = find(2 * a + 1);
    const t = Math.abs(turnAngle(G, a, b));
    if (!bestTurn.has(r) || t < bestTurn.get(r)) {
      bestTurn.set(r, t);
      const ua = segDir(a, true), ub = segDir(b, false);
      let fx = ua[0] + ub[0], fz = ua[1] + ub[1];
      const L = Math.hypot(fx, fz);
      if (L < 1e-6) { fx = ua[0]; fz = ua[1]; } else { fx /= L; fz /= L; }
      frame.set(r, [fx, fz]);
    }
  }

  // ---------------------------------------------------------------- 3. right-edge anchoring
  // per-edge lane width: OSM width / lanes where tagged, else the class default (data section
  // edgeLaneWidth); opts.laneWidth forces one width everywhere
  const wE = new Float32Array(E);
  for (let e = 0; e < E; e++) wE[e] = opts.laneWidth || (G.laneW ? G.laneW[e] : LANE_WIDTH_CLASS[G.cls[e]]) || LANE_W;
  const Rd = new Float32Array(E);
  const twoWay = (e) => G.opp[e] > 0;
  for (let e = 0; e < E; e++) Rd[e] = twoWay(e) ? G.lanes[e] * wE[e] : (G.lanes[e] * wE[e]) / 2;
  const medianShift = medianClearance(G, vFirst, vx, vy, vz, Rd, wE);
  const rampShift = rampClearance(G, vFirst, vx, vy, vz, Rd, wE);
  const Rend = new Float32Array(2 * E).fill(NaN);
  // adjacency of hop constraints per key
  const adj = new Map();
  const addAdj = (k, v) => { let l = adj.get(k); if (!l) adj.set(k, (l = [])); l.push(v); };
  // node width: all edge ends of a strict group share one lane width at the node (the anchor
  // edge's), each edge blends to its own width away from the node
  const groups = new Map();
  for (let k = 0; k < H; k++) {
    for (const key of [2 * hops[3 * k] + 1, 2 * hops[3 * k + 1]]) {
      const r = find(key); let l = groups.get(r); if (!l) groups.set(r, (l = []));
      if (!l.includes(key)) l.push(key);
    }
  }
  const anchorOf = new Map(), wEnd = new Float32Array(2 * E);
  for (let e = 0; e < E; e++) { wEnd[2 * e] = wE[e]; wEnd[2 * e + 1] = wE[e]; }
  for (const [r, keys] of groups) {
    // anchor: a two-way edge end (its divider is fixed at the polyline), else the widest edge
    let anchor = keys[0], best = -1;
    for (const k of keys) {
      const e = k >> 1, score = (twoWay(e) ? 100 : 0) + G.lanes[e];
      if (score > best) { best = score; anchor = k; }
    }
    anchorOf.set(r, anchor);
    for (const k of keys) wEnd[k] = wE[anchor >> 1];
  }
  for (let k = 0; k < H; k++) {
    const a = hops[3 * k], b = hops[3 * k + 1], sh = hops[3 * k + 2], wn = wEnd[2 * a + 1];
    addAdj(2 * a + 1, [2 * b, -sh * wn]);   // R(b start) = R(a end) - shift * node width
    addAdj(2 * b, [2 * a + 1, sh * wn]);
  }
  let conflicts = 0;
  // strict groups: the anchor end takes its Rd (+ the group's opposing-clearance raise, see
  // opposingClearance), every other end of the group follows by the hop shifts
  const groupRaise = new Map(), PC = {};
  const anchorGroups = (count) => {
    for (const [r, keys] of groups) {
      for (const k of keys) Rend[k] = NaN;
      const anchor = anchorOf.get(r);
      Rend[anchor] = Rd[anchor >> 1];
      const q = [anchor];
      while (q.length) {
        const k = q.pop();
        for (const [k2, d] of adj.get(k) || []) {
          let v = Rend[k] + d;
          if (twoWay(k2 >> 1)) v = G.lanes[k2 >> 1] * wEnd[k2];      // divider on the polyline
          if (Number.isNaN(Rend[k2])) { Rend[k2] = v; q.push(k2); }
          else if (count && Math.abs(Rend[k2] - (Rend[k] + d)) > 1e-3) conflicts++;
        }
      }
      // opposing-clearance raise of an all-one-way group: every end moves by the same amount (lanes
      // stay continuous), clamped so that no end exceeds its edge's cap
      let gr = groupRaise.get(r) || 0;
      if (gr !== 0 && PC.groupCap) {
        let hi = Infinity, lo = -Infinity;
        for (const k of keys) { hi = Math.min(hi, PC.groupCap(k) - Rend[k]); lo = Math.max(lo, PC.groupFloor(k) - Rend[k]); }
        gr = Math.min(Math.max(gr, Math.min(0, lo)), Math.max(0, hi)); groupRaise.set(r, gr);
        for (const k of keys) Rend[k] += gr;
      }
    }
  };
  anchorGroups(true);
  for (let k = 0; k < 2 * E; k++) if (Number.isNaN(Rend[k])) Rend[k] = Rd[k >> 1];
  // tau: length of the smoothstep from an end's Rend back to Rd
  const tauMin = new Float32Array(2 * E);       // opposing-clearance tapers (0 = none)
  const tauOf = (e, end) => {
    const L = G.length[e];
    const d = Math.max(Math.abs(Rend[2 * e + end] - Rd[e]), G.lanes[e] * Math.abs(wEnd[2 * e + end] - wE[e]));
    if (d < 1e-4) return 0;
    // ramps converge onto the mainline in a short gore (they stay beside it until then)
    const link = G.cls[e] === 1 || G.cls[e] === 3;
    const t = link ? Math.max(20, Math.min(30, 6 * d)) : Math.max(MIN_TAU, TAU_PER_M * d);
    return Math.min(L / 2, Math.max(t, tauMin[2 * e + end]));
  };
  // opposite-direction one-way carriageways (dual carriageways converging at a node, both
  // directions drawn on one corridor, ramps pushed together): docs/carriageway-fix.md
  const opposing = !(opts.opposing ?? OPPOSING_DEFAULT) || (typeof process !== 'undefined' && process.env && process.env.GEOM_NO_OPPOSING) ? { shifted: 0 } : opposingClearance(G, Object.assign(PC, { vFirst, vx, vy, vz, vs, Rd, Rend, wE, tauMin, tauOf, adj, find, anchorOf, groupRaise, anchorGroups, twoWay }), opts);
  const tau = new Float32Array(2 * E);
  for (let e = 0; e < E; e++) for (const end of [0, 1]) tau[2 * e + end] = tauOf(e, end);

  // ---------------------------------------------------------------- 4. normals (mitres)
  const mx = new Float64Array(V), mz = new Float64Array(V);
  for (let e = 0; e < E; e++) {
    const v0 = vFirst[e], v1 = vFirst[e + 1] - 1;
    for (let i = v0; i <= v1; i++) {
      let ux = 0, uz = 0;
      const seg = (a, b) => { const dx = vx[b] - vx[a], dz = vz[b] - vz[a], L = Math.hypot(dx, dz) || 1; return [dx / L, dz / L]; };
      if (i > v0) { const u = seg(i - 1, i); ux += u[0]; uz += u[1]; }
      if (i < v1) { const u = seg(i, i + 1); ux += u[0]; uz += u[1]; }
      let L = Math.hypot(ux, uz) || 1; ux /= L; uz /= L;
      let sc = 1;
      if (i > v0 && i < v1) { const u = seg(i - 1, i); sc = 1 / Math.max(0.5, u[0] * ux + u[1] * uz); }
      mx[i] = -uz * sc; mz[i] = ux * sc;
    }
    for (const end of [0, 1]) {
      const r = find(2 * e + end);
      if (r === 2 * e + end && !adj.has(2 * e + end)) continue;
      const f = frame.get(r);
      if (!f) continue;
      const i = end ? v1 : v0;
      mx[i] = -f[1]; mz[i] = f[0];
    }
  }

  // ---------------------------------------------------------------- 5. lane ends / adds (tapers)
  // per edge end: number of lanes ending on the right / left (end) or starting (start)
  const tapR = new Uint8Array(2 * E), tapL = new Uint8Array(2 * E);
  const outsOf = new Map(), insOf = new Map();
  for (let k = 0; k < H; k++) {
    const a = hops[3 * k], b = hops[3 * k + 1], sh = hops[3 * k + 2];
    let l = outsOf.get(a); if (!l) outsOf.set(a, (l = [])); l.push([b, sh]);
    let m = insOf.get(b); if (!m) insOf.set(b, (m = [])); m.push([a, sh]);
  }
  for (const [a, lst] of outsOf) {
    const n = G.lanes[a], cont = new Uint8Array(n);
    for (const [b, sh] of lst) for (let l = 0; l < n; l++) if (l - sh >= 0 && l - sh < G.lanes[b]) cont[l] = 1;
    let r = 0; while (r < n && !cont[r]) r++;
    let t = 0; while (t < n - r && !cont[n - 1 - t]) t++;
    if (r < n) { tapR[2 * a + 1] = r; tapL[2 * a + 1] = t; }
  }
  for (const [b, lst] of insOf) {
    const n = G.lanes[b], cont = new Uint8Array(n);
    for (const [a, sh] of lst) for (let l = 0; l < n; l++) if (l + sh >= 0 && l + sh < G.lanes[a]) cont[l] = 1;
    let r = 0; while (r < n && !cont[r]) r++;
    let t = 0; while (t < n - r && !cont[n - 1 - t]) t++;
    if (r < n) { tapR[2 * b] = r; tapL[2 * b] = t; }
  }

  // paved run-out: lanes that end at a node keep their pavement into the straightest strict
  // successor, narrowing to zero over the taper there (and lanes that start get a widening on the
  // straightest predecessor), so the pavement edge is continuous while every sim lane is paved
  // to its node (xR/xL: extra lane widths at an edge end, 2 per edge)
  const xR = new Uint8Array(2 * E), xL = new Uint8Array(2 * E);
  const straightest = (lst, fn) => { let best = -1, bt = 9; for (const [o] of lst) { const t = Math.abs(fn(o)); if (t < bt) { bt = t; best = o; } } return best; };
  for (const [a, lst] of outsOf) {
    if (!tapR[2 * a + 1] && !tapL[2 * a + 1]) continue;
    const b = straightest(lst, (o) => turnAngle(G, a, o));
    if (b < 0) continue;
    xR[2 * b] = Math.max(xR[2 * b], tapR[2 * a + 1]); xL[2 * b] = Math.max(xL[2 * b], tapL[2 * a + 1]);
  }
  for (const [b, lst] of insOf) {
    if (!tapR[2 * b] && !tapL[2 * b]) continue;
    const a = straightest(lst, (o) => turnAngle(G, o, b));
    if (a < 0) continue;
    xR[2 * a + 1] = Math.max(xR[2 * a + 1], tapR[2 * b]); xL[2 * a + 1] = Math.max(xL[2 * a + 1], tapL[2 * b]);
  }

  return new LaneGeometry(G, { vFirst, vx, vy, vz, vs, mx, mz, Rd, Rend, tau, tapR, tapL, xR, xL, LW, laneW: wE, laneWEnd: wEnd,
    stats: { strictHops: H, groups: groups.size, anchorConflicts: conflicts, vertices: V, medianShifted: medianShift, rampsShifted: rampShift, opposing } });
}

/**
 * Opposing one-way freeway carriageways must not overlap as drawn (audit r9): where the opposite
 * carriageway's polyline is within 50 m to the left, the carriageway is shifted right (Rd grows)
 * so its left (median) edge stays at least 0.3 m short of the midline between the two polylines.
 * Lanes keep their width; the sim drives the shifted lanes (same module).
 */
function medianClearance(G, vFirst, vx, vy, vz, Rd, wE) {
  const isFw = (e) => (G.cls[e] === 0 || G.cls[e] === 2) && G.opp[e] === 0;
  const cell = 60, grid = new Map(), samples = [];
  for (let e = 0; e < G.edgeCount; e++) {
    if (!isFw(e)) continue;
    const v0 = vFirst[e], v1 = vFirst[e + 1] - 1;
    for (let i = v0; i < v1; i++) {
      const dx = vx[i + 1] - vx[i], dz = vz[i + 1] - vz[i], L = Math.hypot(dx, dz);
      if (L < 1e-6) continue;
      const n = Math.max(1, Math.ceil(L / 15));
      for (let k = 0; k < n; k++) {
        const t = (k + 0.5) / n;
        const rec = { e, x: vx[i] + dx * t, z: vz[i] + dz * t, y: vy[i] + (vy[i + 1] - vy[i]) * t, hx: dx / L, hz: dz / L };
        samples.push(rec);
        const key = Math.floor(rec.x / cell) + ',' + Math.floor(rec.z / cell);
        let l = grid.get(key); if (!l) grid.set(key, (l = [])); l.push(rec);
      }
    }
  }
  const need = new Float32Array(G.edgeCount).fill(-1);
  for (const a of samples) {
    let bd = 50, best = null;
    const cx = Math.floor(a.x / cell), cz = Math.floor(a.z / cell);
    for (let i = -1; i <= 1; i++) for (let j = -1; j <= 1; j++) for (const b of grid.get((cx + i) + ',' + (cz + j)) || []) {
      if (b.e === a.e || a.hx * b.hx + a.hz * b.hz > -0.94 || Math.abs(a.y - b.y) > 3) continue;
      const d = Math.hypot(b.x - a.x, b.z - a.z);
      if (d < bd) { bd = d; best = b; }
    }
    if (!best) continue;
    const gap = (best.x - a.x) * a.hz + (best.z - a.z) * -a.hx;   // > 0: the other polyline lies to the left
    if (gap <= 0) continue;
    const W = G.lanes[a.e] * wE[a.e];
    const r = W - Math.max(0, gap / 2 - 0.3);                     // R such that left edge = -(gap/2 - 0.3)
    if (r > need[a.e]) need[a.e] = r;
  }
  let shifted = 0;
  for (let e = 0; e < G.edgeCount; e++) {
    if (need[e] > Rd[e] + 1e-3) { Rd[e] = Math.min(need[e], (G.lanes[e] + 2) * wE[e]); shifted++; }
  }
  return shifted;
}

/**
 * Ramps (one-way motorway_link / trunk_link) drawn alongside a same-direction freeway carriageway
 * they are not joined to (no shared node) must not overlap it: the ramp is shifted sideways (Rd)
 * away from the carriageway until its pavement clears it by 0.3 m (at most 2 lanes). Near the
 * merge / diverge node the strict-node anchoring still brings the ramp onto its lanes.
 */
function rampClearance(G, vFirst, vx, vy, vz, Rd, wE) {
  const isFw = (e) => (G.cls[e] === 0 || G.cls[e] === 2) && G.opp[e] === 0;
  const isLink = (e) => (G.cls[e] === 1 || G.cls[e] === 3) && G.opp[e] === 0;
  const cell = 40, grid = new Map();
  const samples = (e, fn) => {
    const v0 = vFirst[e], v1 = vFirst[e + 1] - 1;
    let acc = 0;
    for (let i = v0; i < v1; i++) {
      const dx = vx[i + 1] - vx[i], dz = vz[i + 1] - vz[i], L = Math.hypot(dx, dz);
      if (L < 1e-6) continue;
      const n = Math.max(1, Math.ceil(L / 8));
      for (let k = 0; k < n; k++) {
        const t = (k + 0.5) / n;
        fn(vx[i] + dx * t, vy[i] + (vy[i + 1] - vy[i]) * t, vz[i] + dz * t, dx / L, dz / L, acc + L * t);
      }
      acc += L;
    }
  };
  for (let e = 0; e < G.edgeCount; e++) {
    if (!isFw(e)) continue;
    const W = G.lanes[e] * wE[e], c = Rd[e] - W / 2;          // pavement centre offset (right-positive)
    samples(e, (x, y, z, hx, hz) => {
      const rec = { e, x: x - hz * c, y, z: z + hx * c, hx, hz, hw: W / 2 };
      const key = Math.floor(rec.x / cell) + ',' + Math.floor(rec.z / cell);
      let l = grid.get(key); if (!l) grid.set(key, (l = [])); l.push(rec);
    });
  }
  const GORE = 30;          // a ramp joined to the carriageway may converge onto it only this close to the shared node
  let shifted = 0;
  for (let e = 0; e < G.edgeCount; e++) {
    if (!isLink(e)) continue;
    const W = G.lanes[e] * wE[e], c = Rd[e] - W / 2, Le = G.length[e];
    let right = 0, left = 0, jRight = 0, jLeft = 0;              // required shift (any / joined carriageway)
    let gateDist = Infinity, gateSide = 0;                       // side of the joined carriageway nearest the gore
    samples(e, (x, y, z, hx, hz, d) => {
      const px = x - hz * c, pz = z + hx * c;
      const cx = Math.floor(px / cell), cz = Math.floor(pz / cell);
      for (let i = -1; i <= 1; i++) for (let j = -1; j <= 1; j++) for (const r of grid.get((cx + i) + ',' + (cz + j)) || []) {
        if (r.hx * hx + r.hz * hz < 0.8 || Math.abs(r.y - y) > 4.5) continue;
        // joined at a merge / diverge node: the ramp stays beside the carriageway until the gore
        const atEnd = G.to(e) === G.to(r.e) || G.to(e) === G.from(r.e), atStart = G.from(e) === G.from(r.e) || G.from(e) === G.to(r.e);
        if ((atEnd && Le - d < GORE) || (atStart && d < GORE)) continue;
        const lon = (px - r.x) * r.hx + (pz - r.z) * r.hz;
        if (Math.abs(lon) > 6) continue;
        const lat = (px - r.x) * -r.hz + (pz - r.z) * r.hx;      // > 0: ramp centre right of the carriageway centre
        const over = W / 2 + r.hw + 0.3 - Math.abs(lat);
        if (over <= 0) continue;
        if (lat >= 0) right = Math.max(right, over); else left = Math.max(left, over);
        if (atEnd || atStart) {
          if (lat >= 0) jRight = Math.max(jRight, over); else jLeft = Math.max(jLeft, over);
          const dn = atEnd ? Le - d : d;
          if (dn < gateDist) { gateDist = dn; gateSide = lat >= 0 ? 1 : -1; }
        }
      }
    });
    // a ramp always clears the carriageway it merges into / diverges from (it joins from that side);
    // between two unrelated carriageways it is left alone
    let sh = 0;
    if (jRight > 0 && !jLeft) sh = Math.max(jRight, right);
    else if (jLeft > 0 && !jRight) sh = -Math.max(jLeft, left);
    else if (right > 0 && !left) sh = right;
    else if (left > 0 && !right) sh = -left;
    sh = Math.sign(sh) * Math.min(Math.abs(sh), 4 * wE[e]);
    if (Math.abs(sh) > 1e-3) { Rd[e] += sh; shifted++; }
  }
  return shifted;
}

/**
 * Opposite-direction one-way carriageways (docs/carriageway-fix.md). OSM draws each carriageway of
 * a dual road on its own way; where the two ways converge (at the node where the dual road ends, at
 * a junction, or where both directions are drawn on one corridor) centred carriageways overlap.
 * Every one-way edge (any class) is sampled every 3 m; for each opposite-heading one-way sample
 * beside it (|dy| < 3 m) at polyline gap g (positive = to the left), the right edge must satisfy
 *   R >= W + (c - g) / 2        (W = lanes * lane width)
 * i.e. the inner (left) pavement edge stays c/2 short of the midline between the two polylines,
 * so the two inner pavement edges keep >= c (opts.medianMin, default 1.0 m). Within NOSE m of a
 * strict end (a continuation into a two-way road, where the median really ends) c tapers to 0.
 * The requirement is met with the existing right-edge parameters, so every consumer (sim, GPU
 * mirror, renderer, lines) gets it: a violation near a free (junction) end raises that end's Rend
 * with a smoothstep taper of >= 40 m (tauMin); near a strict end of an all-one-way group it raises
 * the group's anchor (all ends of the group move together, lanes stay continuous); elsewhere it
 * raises Rd. Strict ends pinned to a two-way road are not moved (the median nose taper).
 * Pairs drawn left-hand (the opposite carriageway clearly on the right) are left alone (counted).
 */
function opposingClearance(G, P, opts) {
  const { vFirst, vx, vy, vz, vs, Rd, Rend, wE, tauMin, tauOf, adj, find, anchorOf, groupRaise, anchorGroups, twoWay } = P;
  const E = G.edgeCount, CLR = opts.medianMin ?? 1.0, NOSE = 30, STEP = 3, CELL = 16, ZONE = 120, TMIN = 40;
  const SLOPE_MAX = 0.04;     // max lateral slope of a taper set here (about 1:25)
  const oneWay = (e) => G.opp[e] === 0 && G.twin[e] < 0;
  // ---- samples along the resampled polylines of one-way edges
  const se = [], ss = [], sx = [], sy = [], sz = [], shx = [], shz = [];
  const sFirst = new Int32Array(E + 1);
  for (let e = 0; e < E; e++) {
    sFirst[e] = se.length;
    if (!oneWay(e)) continue;
    const v0 = vFirst[e], v1 = vFirst[e + 1] - 1;
    for (let i = v0; i < v1; i++) {
      const dx = vx[i + 1] - vx[i], dz = vz[i + 1] - vz[i], L = Math.hypot(dx, dz);
      if (L < 1e-6) continue;
      const n = Math.max(1, Math.ceil(L / STEP)), last = i === v1 - 1;
      for (let k = 0; k <= (last ? n : n - 1); k++) {
        const t = k / n;
        se.push(e); ss.push(vs[i] + (vs[i + 1] - vs[i]) * t);
        sx.push(vx[i] + dx * t); sy.push(vy[i] + (vy[i + 1] - vy[i]) * t); sz.push(vz[i] + dz * t);
        shx.push(dx / L); shz.push(dz / L);
      }
    }
  }
  sFirst[E] = se.length;
  const NS = se.length, grid = new Map(), gkey = (x, z) => Math.floor(x / CELL) * 100000 + Math.floor(z / CELL);
  for (let i = 0; i < NS; i++) { const k = gkey(sx[i], sz[i]); let l = grid.get(k); if (!l) grid.set(k, (l = [])); l.push(i); }
  const dStrict = (i) => {
    const e = se[i], L = G.length[e];
    return Math.min(adj.has(2 * e) ? ss[i] : Infinity, adj.has(2 * e + 1) ? L - ss[i] : Infinity);
  };
  // ---- neighbours: (sample i of a, sample j of b) beside each other with opposite headings
  const recI = [], recJ = [], recG = [], pairG = new Map();
  for (let i = 0; i < NS; i++) {
    const a = se[i], hx = shx[i], hz = shz[i], cx = Math.floor(sx[i] / CELL), cz = Math.floor(sz[i] / CELL);
    for (let u = -1; u <= 1; u++) for (let w = -1; w <= 1; w++) for (const j of grid.get((cx + u) * 100000 + cz + w) || []) {
      const b = se[j];
      if (b === a || hx * shx[j] + hz * shz[j] > -0.8 || Math.abs(sy[i] - sy[j]) > 3) continue;
      const dx = sx[j] - sx[i], dz = sz[j] - sz[i];
      if (Math.abs(dx * hx + dz * hz) > STEP / 2 + 0.1) continue;
      const g = dx * hz - dz * hx;                         // > 0: b lies to the left of a
      if (Math.abs(g) > 30) continue;
      recI.push(i); recJ.push(j); recG.push(g);
      const pk = Math.min(a, b) * 4194304 + Math.max(a, b), p = pairG.get(pk);   // symmetric: both views
      if (p) { p[0] += g; p[1]++; } else pairG.set(pk, [g, 1]);
    }
  }
  // LB / UBo: active bounds (this pass moves the edge to meet them); UBs: same-direction limit (min-max only)
  const LB = new Float32Array(NS).fill(-Infinity), UBo = new Float32Array(NS).fill(Infinity), UBs = new Float32Array(NS).fill(Infinity);
  let leftHand = 0;
  const lhPairs = new Set();
  for (let r = 0; r < recI.length; r++) {
    const i = recI[r], j = recJ[r], a = se[i], b = se[j], p = pairG.get(Math.min(a, b) * 4194304 + Math.max(a, b));
    const c = CLR * Math.min(1, dStrict(i) / NOSE, dStrict(j) / NOSE);
    if (p[0] / p[1] < -1) {
      // drawn left-hand (b clearly on the right): both move to their own left, away from each
      // other: the right edges keep c/2 short of the midline, R <= (|g| - c) / 2
      lhPairs.add(Math.min(a, b) * 4194304 + Math.max(a, b));
      const ub = (-recG[r] - c) / 2;
      if (ub < UBo[i]) UBo[i] = ub;
      continue;
    }
    const need = G.lanes[a] * wE[a] + (c - recG[r]) / 2;
    if (need > LB[i]) LB[i] = need;
  }
  leftHand = lhPairs.size;
  // same-direction freeway carriageways / ramps on the right (not joined within the 30 m gore):
  // never push a carriageway into one (R <= its left pavement edge - 0.3 m), the ramp rule
  const isFwSys = (e) => G.cls[e] <= 3;
  const R0 = (e, s) => {
    const L = G.length[e], rd = Rd[e], ts = tauOf(e, 0), te = tauOf(e, 1);
    let r = rd;
    if (ts > 0) r += (Rend[2 * e] - rd) * smooth(1 - s / ts);
    if (te > 0) r += (Rend[2 * e + 1] - rd) * smooth(1 - (L - s) / te);
    return r;
  };
  let sameDir = 0;
  for (let i = 0; i < NS; i++) {
    const a = se[i];
    if (!(LB[i] > -Infinity) && !(UBo[i] < Infinity)) continue;   // only where this pass may move the edge
    const hx = shx[i], hz = shz[i], cx = Math.floor(sx[i] / CELL), cz = Math.floor(sz[i] / CELL), La = G.length[a];
    for (let u = -1; u <= 1; u++) for (let w = -1; w <= 1; w++) for (const j of grid.get((cx + u) * 100000 + cz + w) || []) {
      const b = se[j];
      if (b === a || !isFwSys(b) || hx * shx[j] + hz * shz[j] < 0.8 || Math.abs(sy[i] - sy[j]) > 4.5) continue;
      const dx = sx[j] - sx[i], dz = sz[j] - sz[i];
      if (Math.abs(dx * hx + dz * hz) > STEP / 2 + 0.1) continue;
      const gr = -(dx * hz - dz * hx);                     // > 0: b lies to the right of a
      if (gr <= 0 || gr > 30) continue;
      const atEnd = G.to(a) === G.to(b) || G.to(a) === G.from(b), atStart = G.from(a) === G.from(b) || G.from(a) === G.to(b);
      if ((atEnd && La - ss[i] < 30) || (atStart && ss[i] < 30)) continue;
      const ub = gr + R0(b, ss[j]) - G.lanes[b] * wE[b] - 0.3;
      if (ub < UBs[i]) { UBs[i] = ub; sameDir++; }
    }
  }
  // ---- fit the right-edge parameters of every edge with a violation
  const sm = (u) => { u = u < 0 ? 0 : u > 1 ? 1 : u; return u * u * (3 - 2 * u); };
  const set = new Uint8Array(2 * E);          // free end raised by this pass (else it tracks Rd)
  const kind = (key) => {                     // 0 free (junction end), 1 strict all-one-way group, 2 pinned to a two-way road
    if (!adj.has(key)) return 0;
    return twoWay(anchorOf.get(find(key)) >> 1) ? 2 : 1;
  };
  const tauP = (e, end) => Math.max(tauOf(e, end), Math.min(G.length[e] / 2, tauMin[2 * e + end]));
  const R = (e, s) => {
    const L = G.length[e], rd = Rd[e], ts = tauP(e, 0), te = tauP(e, 1);
    let r = rd;
    if (ts > 0) r += (Rend[2 * e] - rd) * sm(1 - s / ts);
    if (te > 0) r += (Rend[2 * e + 1] - rd) * sm(1 - (L - s) / te);
    return r;
  };
  const changed = new Set();
  let unresolved = 0;
  // a carriageway moves at most until its inner edge is c/2 + 1.5 m right of its polyline (or keeps an
  // existing larger median / ramp shift)
  const Rd0 = Float32Array.from(Rd), Rend0 = Float32Array.from(Rend);
  const capOf = (e) => Math.max(Rd0[e], Rend0[2 * e], Rend0[2 * e + 1], G.lanes[e] * wE[e] + CLR / 2 + 1.5);
  const floorOf = (e) => Math.min(Rd0[e], Rend0[2 * e], Rend0[2 * e + 1], -(CLR / 2 + 1.5));   // mirror (left-hand pairs)
  P.groupCap = (key) => capOf(key >> 1); P.groupFloor = (key) => floorOf(key >> 1);
  // Per edge, a small exact search: for each candidate Rd (0.1 m steps between floor and cap) the end
  // values nearest their current value that meet every sample (the start and end smoothsteps are each at
  // most L/2 long, so each end is solved on its own half), then the cheapest feasible candidate (cost =
  // moved area: |Rd change| x L + |end change| x taper / 2), else the one with the least violation.
  const SAME_W = 0.5;   // a head-on overlap weighs 4x a same-direction (ramp over mainline) one
  const violOf = (i, r) => Math.max(LB[i] - r, r - UBo[i], SAME_W * (r - UBs[i]));
  const fitEdge = (e) => {
    const i0 = sFirst[e], i1 = sFirst[e + 1], L = G.length[e];
    const ks = kind(2 * e), ke = kind(2 * e + 1);
    let extS = -1, extE = -1, cur = 0;
    for (let i = i0; i < i1; i++) {
      const r = R(e, ss[i]), v = violOf(i, r);
      cur = Math.max(cur, v);
      if (Math.max(LB[i] - r, r - UBo[i]) <= 0.02) continue;      // triggered by the opposing bounds only
      if (ss[i] <= L / 2) extS = Math.max(extS, ss[i]); else extE = Math.max(extE, L - ss[i]);
    }
    if (extS < 0 && extE < 0) return true;
    changed.add(e);
    const cap = capOf(e), flo = floorOf(e), rd0 = Rd[e];
    const plan = (end, ext, k) => k === 2 ? -1 : (ext >= 0 || set[2 * e + end] || k === 1) ? Math.min(L / 2, Math.max(TMIN, 2 * Math.max(0, ext) + 10, tauMin[2 * e + end])) : 0;
    const pS = plan(0, extS, ks), pE = plan(1, extE, ke);
    const cand = [rd0];
    for (let k = 1; k <= 80; k++) { if (rd0 + 0.1 * k <= cap) cand.push(rd0 + 0.1 * k); if (rd0 - 0.1 * k >= flo) cand.push(rd0 - 0.1 * k); }
    let best = null;
    for (const rd of cand) {
      Rd[e] = rd;
      const ts = pS < 0 ? tauOf(e, 0) : pS, te = pE < 0 ? tauOf(e, 1) : pE;
      const baseS = ks !== 0 ? Rend[2 * e] : set[2 * e] ? Rend[2 * e] : rd;
      const baseE = ke !== 0 ? Rend[2 * e + 1] : set[2 * e + 1] ? Rend[2 * e + 1] : rd;
      // feasible interval for each end value
      let sLo = -Infinity, sHi = Infinity, eLo = -Infinity, eHi = Infinity;
      for (let i = i0; i < i1; i++) {
        const s = ss[i], as = ts > 0 ? sm(1 - s / ts) : 0, ae = te > 0 ? sm(1 - (L - s) / te) : 0;
        const ub = UBo[i];
        if (as > 0.02 && ks !== 2) { if (LB[i] > -Infinity) sLo = Math.max(sLo, (LB[i] - rd * (1 - as)) / as); if (ub < Infinity) sHi = Math.min(sHi, (ub - rd * (1 - as)) / as); }
        else if (ae > 0.02 && ke !== 2) { if (LB[i] > -Infinity) eLo = Math.max(eLo, (LB[i] - rd * (1 - ae)) / ae); if (ub < Infinity) eHi = Math.min(eHi, (ub - rd * (1 - ae)) / ae); }
      }
      const pick = (base, lo, hi, k) => {
        if (k === 2) return base;
        const cp = Math.max(cap, base), fl = Math.min(flo, base);
        lo = Math.min(lo, cp); hi = Math.max(hi, fl);
        let v = lo > hi ? (lo + hi) / 2 : Math.min(Math.max(base, lo), hi);
        return Math.min(cp, Math.max(fl, v));
      };
      let rs = pick(baseS, sLo, sHi, ks), re = pick(baseE, eLo, eHi, ke);
      // lateral slope limit of an end taper this pass sets (a smoothstep's max slope is 1.5 d / tau): lanes
      // must stay near-parallel to the polyline at the junction mouths, or connectors see a false turn
      // (on a short edge the taper cannot fit: the whole edge moves with Rd instead)
      const taper = (k, r, base, plan) => {
        if (!TAPER_LIMIT || k === 2 || Math.abs(r - rd) < 1e-3 || Math.abs(r - base) < 1e-3) return [r, plan < 0 ? 0 : plan];
        let tau = Math.min(L / 2, Math.max(plan, TAU_PER_M * Math.abs(r - rd)));
        if (1.5 * Math.abs(r - rd) / tau > SLOPE_MAX) r = rd + Math.sign(r - rd) * SLOPE_MAX * tau / 1.5;
        return [r, tau];
      };
      let tsE = ts, teE = te;
      if (ks !== 2) [rs, tsE] = taper(ks, rs, baseS, ts);
      if (ke !== 2) [re, teE] = taper(ke, re, baseE, te);
      let viol = 0;
      for (let i = i0; i < i1; i++) {
        const s = ss[i], as = tsE > 0 ? sm(1 - s / tsE) : 0, ae = teE > 0 ? sm(1 - (L - s) / teE) : 0;
        viol = Math.max(viol, violOf(i, rd * (1 - as - ae) + rs * as + re * ae));
      }
      const cost = Math.abs(rd - rd0) * L + Math.abs(rs - baseS) * tsE / 2 + Math.abs(re - baseE) * teE / 2;
      const ok = viol <= 0.02;
      if (!best || (ok && (!best.ok || cost < best.cost)) || (!ok && !best.ok && viol < best.viol - 1e-3)) best = { ok, cost, viol, rd, rs, re, ts: tsE, te: teE };
    }
    if (!best.ok && best.viol >= cur - 1e-3) { Rd[e] = rd0; return false; }    // no improvement: leave it
    Rd[e] = best.rd;
    for (const [end, k, r, tp] of [[0, ks, best.rs, best.ts], [1, ke, best.re, best.te]]) {
      const key = 2 * e + end;
      if (k === 2) continue;
      if (k === 1) {
        const d = r - Rend[key];
        if (Math.abs(d) > 1e-3) { const g = find(key); groupRaise.set(g, (groupRaise.get(g) || 0) + d); Rend[key] = r; tauMin[key] = Math.max(tauMin[key], tp); }
      } else if (Math.abs(r - best.rd) > 1e-3) { Rend[key] = r; set[key] = 1; tauMin[key] = Math.max(tauMin[key], tp); }
      else { Rend[key] = best.rd; set[key] = 0; }
    }
    return best.ok;
  };
  for (let pass = 0; pass < 4; pass++) {
    let bad = 0;
    for (let e = 0; e < E; e++) if (sFirst[e + 1] > sFirst[e] && !fitEdge(e)) bad++;
    anchorGroups(false);
    for (let k = 0; k < 2 * E; k++) if (!adj.has(k) && !set[k]) Rend[k] = Rd[k >> 1];
    unresolved = bad;
  }
  // remaining violations (measured on the final parameters)
  let viol = 0, violM = 0;
  for (let i = 0; i < NS; i++) { const r = R(se[i], ss[i]); if (Math.max(LB[i] - r, r - UBo[i]) > 0.05) { viol++; violM += STEP; } }
  let moved = 0, maxShift = 0;
  for (const e of changed) { const W = G.lanes[e] * wE[e]; maxShift = Math.max(maxShift, Math.abs(Rd[e] - Rd0[e]), Math.abs(Rend[2 * e] - Rend0[2 * e]), Math.abs(Rend[2 * e + 1] - Rend0[2 * e + 1])); moved++; }
  return { shifted: moved, unresolvedEdges: unresolved, sameDirLimits: sameDir, samples: NS, pairsLeftHand: leftHand, groupsRaised: groupRaise.size, residualSamples: viol, residualM: violM, clearance: CLR, maxShift: +maxShift.toFixed(2) };
}

export class LaneGeometry {
  constructor(G, a) { this.G = G; Object.assign(this, a); }
  /** Right carriageway edge offset (m right of the polyline) at s. */
  rightEdge(e, s) {
    const L = this.G.length[e], rd = this.Rd[e];
    let r = rd;
    const ts = this.tau[2 * e], te = this.tau[2 * e + 1];
    if (ts > 0) r += (this.Rend[2 * e] - rd) * smooth(1 - s / ts);
    if (te > 0) r += (this.Rend[2 * e + 1] - rd) * smooth(1 - (L - s) / te);
    return r;
  }
  _rightEdgeSlope(e, s) { const d = 0.5; return (this.rightEdge(e, Math.min(this.G.length[e], s + d)) - this.rightEdge(e, Math.max(0, s - d))) / (2 * d); }
  /** Drawn pavement edges at s (offsets right of the polyline), with lane add/drop tapers. */
  /**
   * Drawn pavement edges at s (offsets right of the polyline). Every sim lane is paved to its
   * node (lane centres never leave the pavement). A lane that ends at a node continues as a
   * paved run-out on the straightest successor, narrowing to zero over min(90 m, length) with a
   * painted chevron gore (a starting lane gets the mirror widening on the predecessor), so the
   * pavement edge stays continuous. laneRight / laneLeft are the outer edges of the sim lanes.
   */
  bounds(e, s) {
    const n = this.G.lanes[e], w = this.laneWidth(e, s), L = this.G.length[e];
    const R = this.rightEdge(e, s);
    const T = Math.min(TAPER_M, L);
    const fs = T > 0 ? Math.max(0, 1 - s / T) : 0, fe = T > 0 ? Math.max(0, 1 - (L - s) / T) : 0;
    const exR = (this.xR[2 * e] * fs + this.xR[2 * e + 1] * fe) * w, exL = (this.xL[2 * e] * fs + this.xL[2 * e + 1] * fe) * w;
    return { right: R + exR, left: R - n * w - exL, R, laneRight: R, laneLeft: R - n * w, laneWidth: w };
  }
  /**
   * Lane width (m) of edge e at s: the edge's own width (OSM width / lanes, or the class default),
   * blending to the shared node width over the same transition as the right edge, so lanes stay
   * continuous across strict hops between roads of different widths.
   */
  laneWidth(e, s) {
    const L = this.G.length[e], w = this.laneW[e];
    const ts = this.tau[2 * e], te = this.tau[2 * e + 1];
    let r = w;
    if (ts > 0) r += (this.laneWEnd[2 * e] - w) * smooth(1 - s / ts);
    if (te > 0) r += (this.laneWEnd[2 * e + 1] - w) * smooth(1 - (L - s) / te);
    return r;
  }
  /** Sim lat (m from the right carriageway edge) of the centre of lane l at s. */
  laneCenterLat(e, s, l) { return (l + 0.5) * this.laneWidth(e, s); }
  /** Lane index containing lat at s (may be < 0 or >= lanes when off the carriageway). */
  latToLane(e, s, lat) { return Math.floor(lat / this.laneWidth(e, s)); }
  /** Offset (m right of the polyline) of the line between lane k-1 and lane k (k = 0: right edge, k = lanes: left edge). */
  dividerOffset(e, s, k) { return this.rightEdge(e, s) - k * this.laneWidth(e, s); }
  /**
   * Lanes that end at the end node / start at the start node of e (no continuation through any
   * strict hop), counted from the right (endRight / startRight) and from the left (endLeft /
   * startLeft), and the s range where the gore is painted. Sim hook: a car in lane l < endRight
   * (or l >= lanes - endLeft) must have left that lane by s = gore.endFrom; laneEndDist is
   * measured to that point.
   */
  laneEnds(e) {
    const L = this.G.length[e], T = Math.min(TAPER_M, L);
    return { endRight: this.tapR[2 * e + 1], endLeft: this.tapL[2 * e + 1], startRight: this.tapR[2 * e], startLeft: this.tapL[2 * e],
      gore: { endFrom: L - T, endTo: L, startFrom: 0, startTo: T } };
  }
  _locate(e, s) {
    const v0 = this.vFirst[e], v1 = this.vFirst[e + 1] - 1, vs = this.vs;
    if (s <= 0) return [v0, 0];
    if (s >= vs[v1]) return [v1 - 1, 1];
    let lo = v0, hi = v1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (vs[m] <= s) lo = m; else hi = m; }
    const d = vs[lo + 1] - vs[lo];
    return [lo, d > 0 ? (s - vs[lo]) / d : 0];
  }
  /** World point `off` m right of the reference polyline at s; heading = unit tangent of that offset curve. */
  offsetPoint(e, s, off, offSlope = 0, out = {}) {
    const [i, t] = this._locate(e, s);
    const j = i + 1, { vx, vy, vz, mx, mz } = this;
    const nx = mx[i] + (mx[j] - mx[i]) * t, nz = mz[i] + (mz[j] - mz[i]) * t;
    out.x = vx[i] + (vx[j] - vx[i]) * t + off * nx;
    out.y = vy[i] + (vy[j] - vy[i]) * t;
    out.z = vz[i] + (vz[j] - vz[i]) * t + off * nz;
    const ds = this.vs[j] - this.vs[i] || 1;
    let hx = (vx[j] - vx[i] + off * (mx[j] - mx[i])) / ds + offSlope * nx;
    let hz = (vz[j] - vz[i] + off * (mz[j] - mz[i])) / ds + offSlope * nz;
    const L = Math.hypot(hx, hz) || 1;
    out.hx = hx / L; out.hz = hz / L;
    return out;
  }
  /** Sim coordinate: lat = metres from the right carriageway edge (positive = left). */
  latPoint(e, s, lat, out) { return this.offsetPoint(e, s, this.rightEdge(e, s) - lat, this._rightEdgeSlope(e, s), out); }
  /** Centre of lane `lane` (0 = rightmost). */
  lanePoint(e, s, lane, out) { return this.latPoint(e, s, (lane + 0.5) * this.laneWidth(e, s), out); }
  /** Sample positions (data s) for drawing edge e: every vertex, plus <= 6 m steps inside transitions/tapers. */
  sampleS(e) {
    const v0 = this.vFirst[e], v1 = this.vFirst[e + 1] - 1, L = this.G.length[e];
    const out = [];
    for (let i = v0; i <= v1; i++) out.push(this.vs[i]);
    const zones = [];
    const ts = this.tau[2 * e], te = this.tau[2 * e + 1], T = Math.min(TAPER_M, L);
    if (ts > 0) zones.push([0, ts]);
    if (te > 0) zones.push([L - te, L]);
    if (this.tapR[2 * e] || this.tapL[2 * e]) zones.push([0, T]);
    if (this.tapR[2 * e + 1] || this.tapL[2 * e + 1]) zones.push([L - T, L]);
    for (const [a, b] of zones) { const n = Math.ceil((b - a) / 6); for (let k = 1; k < n; k++) out.push(a + ((b - a) * k) / n); }
    out.sort((a, b) => a - b);
    const res = [out[0]];
    for (let k = 1; k < out.length; k++) if (out[k] - res[res.length - 1] > 0.05) res.push(out[k]);
    if (L - res[res.length - 1] > 1e-6) res.push(L);
    else res[res.length - 1] = L;
    return res;
  }
}
