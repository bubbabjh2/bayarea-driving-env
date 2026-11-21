// Junction geometry (world-owned): connector paths through intersections, their conflict
// matrix, the stop / yield point at every junction mouth, and ramp-merge wait (gore) points.
// See src/geom/README.md "Junctions".
//
//   const J = buildJunctions(LG, { crosswalkStop, crosswalkExit });   // ~3 s for the whole corridor, or
//     (crosswalk sets: crosswalkLegs(LG, controls, crosswalkPoints) below; without them the
//      mouths have no crosswalk setback and will not match data/junctions.bin / controls)
//   const J = await loadJunctions('data/junctions.meta.json', LG);   // precomputed (tools/build_junctions.mjs)
//   J.mouth(e)        -> { sStop, sExit }  sStop: stop/yield point (m along e) at the mouth of its end
//                        junction; sExit: where e starts after the box at its start junction
//   J.movement(a, la, b) -> movement id (in-edge a, in-lane la, out-edge b) or -1
//   J.arcPoint(m, t, out) -> {x,y,z,hx,hz}  t = metres along the connector (0 .. J.mLen[m])
//   J.conflictsOf(m)  -> [{other, type, s, sOther}]  type 0 crossing / paths within 3 m, 1 merge (exits within 3 m)
//   J.gore(e)         -> wait position s on a ramp edge e whose lanes merge into the mainline without an
//                        added lane (-1 if none): there the waiting car is outside the mainline lane 0
import { hopMap, turnAngle } from './hops.js';

const CURB = 1.0;            // stop point this far back from the crossing pavement edge
const CROSSWALK = 3.5;
const R_MIN = 6.0;           // minimum connector radius (a sedan turns on ~5.5 m)       // + this where the leg has a crosswalk (signals, all-way stops, painted OSM crossings)
const CONFLICT_SEP = 3.0;    // two connector centre paths closer than this conflict (a truck is ~2.6 m wide)
const GORE_CLEAR = 1.3;      // ramp wait point: lane centre at least this far outside the mainline edge

/** Setback (m) from node n back along edge g (or forward along it, at its start) to the mouth. */
function setbackAt(LG, n, g) {
  const G = LG.G, gt = G.twin[g], roads = new Set();
  let hw = 0;
  const visit = (h) => {
    if (h === g || h === gt) return;
    const t = G.twin[h];
    roads.add(t >= 0 ? Math.min(h, t) : h);
    hw = Math.max(hw, (G.lanes[h] + G.opp[h]) * LG.laneW[h] / 2);
  };
  for (const h of G.outEdges(n)) visit(h);
  for (const h of G.inEdges(n)) visit(h);
  return roads.size >= 2 ? hw + CURB : 0;
}

/**
 * Junction legs with a crosswalk: approaches of signalised junctions and all-way stops (from
 * data/controls.json records) and legs with a painted / signalised OSM crossing node within
 * 25 m of the junction (data/crosswalks.json points). Returns { crosswalkStop, crosswalkExit }:
 * edges whose END mouth / START mouth is set back behind the crosswalk.
 */
export function crosswalkLegs(LG, controls = null, points = null) {
  const G = LG.G, stop = new Set(), exit = new Set();
  const mark = (g, atEnd) => {
    const t = G.twin[g];
    if (atEnd) { stop.add(g); if (t >= 0) exit.add(t); } else { exit.add(g); if (t >= 0) stop.add(t); }
  };
  if (controls) {
    for (const c of controls) {
      if (c.node < 0) continue;
      if (c.type === 'signal' || (c.type === 'stop' && c.allWay)) for (const a of c.approaches) mark(a.edge, true);
    }
  }
  if (points && points.length) {
    const cell = 50, grid = new Map();
    for (let e = 0; e < G.edgeCount; e++) {
      if (G.twin[e] >= 0 && G.twin[e] < e) continue;
      const nd = G.nodesOf(e);
      for (let k = 0; k < nd.length - 1; k++) {
        const x0 = Math.min(G.x(nd[k]), G.x(nd[k + 1])), x1 = Math.max(G.x(nd[k]), G.x(nd[k + 1]));
        const z0 = Math.min(G.z(nd[k]), G.z(nd[k + 1])), z1 = Math.max(G.z(nd[k]), G.z(nd[k + 1]));
        for (let gx = Math.floor(x0 / cell); gx <= Math.floor(x1 / cell); gx++) for (let gz = Math.floor(z0 / cell); gz <= Math.floor(z1 / cell); gz++) {
          const key = gx + ',' + gz; let l = grid.get(key); if (!l) grid.set(key, (l = [])); l.push(e, k);
        }
      }
    }
    for (const [px, pz] of points) {
      const l = grid.get(Math.floor(px / cell) + ',' + Math.floor(pz / cell));
      if (!l) continue;
      let best = 4, be = -1, bs = 0;
      for (let i = 0; i < l.length; i += 2) {
        const e = l[i], k = l[i + 1], nd = G.nodesOf(e);
        const ax = G.x(nd[k]), az = G.z(nd[k]), bx = G.x(nd[k + 1]), bz = G.z(nd[k + 1]);
        const dx = bx - ax, dz = bz - az, LL = dx * dx + dz * dz || 1;
        const t = Math.max(0, Math.min(1, ((px - ax) * dx + (pz - az) * dz) / LL));
        const d = Math.hypot(ax + dx * t - px, az + dz * t - pz);
        if (d < best) {
          let acc = 0; for (let q = 0; q < k; q++) acc += Math.hypot(G.x(nd[q + 1]) - G.x(nd[q]), G.z(nd[q + 1]) - G.z(nd[q]));
          best = d; be = e; bs = acc + Math.sqrt(LL) * t;
        }
      }
      if (be < 0) continue;
      const L = G.length[be];
      if (L - bs <= 25) mark(be, true);
      if (bs <= 25) mark(be, false);
    }
  }
  return { crosswalkStop: stop, crosswalkExit: exit };
}

/**
 * One lane connector from (a, la) to (b, lb) with a minimum turning radius rMin.
 * - A corner turn (the lane tangents meet ahead of the entry and behind the exit) is a fillet:
 *   straight, circular arc of radius R = min(tangent lengths) / tan(turn / 2) (capped at 40 m),
 *   straight. If R would be below rMin, the entry moves back along a and the exit forward along
 *   b (at most 30 m each, never past 10 % / 90 % of the edge) until the tangents fit rMin.
 * - Gentle S-offsets use a cubic Bezier (radius checked; the ends are stretched if needed).
 * - Hairpins / no corner: arc - straight - arc (Dubins) with radius rMin.
 * Returns { curve: [[x,y,z]...], sIn, sOut, radius (smallest along the curve) }.
 */
function connector(LG, a, la, b, lb, sIn0, sOut0, th, rMin, lim = {}) {
  const G = LG.G, A = {}, B = {};
  const La = G.length[a], Lb = G.length[b];
  const minIn = Math.max(Math.min(sIn0, 0.05 * La), sIn0 - 40, lim.floorIn ?? -Infinity);
  const maxOut = Math.min(Math.max(sOut0, 0.95 * Lb), sOut0 + 40, lim.capOut ?? Infinity);
  let sIn = sIn0, sOut = sOut0, curve = null;
  for (let iter = 0; iter < 10; iter++) {
    LG.lanePoint(a, sIn, la, A); LG.lanePoint(b, sOut, lb, B);
    const cr = A.hx * B.hz - A.hz * B.hx, dt = A.hx * B.hx + A.hz * B.hz;
    const turn = Math.atan2(cr, dt);                       // > 0: right turn (heading angle increases)
    let t0 = -1, t1 = -1;
    if (Math.abs(cr) > 1e-3) {
      t0 = ((B.x - A.x) * B.hz - (B.z - A.z) * B.hx) / cr;   // corner distance ahead of the entry
      t1 = -((B.x - A.x) * A.hz - (B.z - A.z) * A.hx) / cr;  // corner distance behind the exit
    }
    const corner = Math.abs(turn) > 0.15 && Math.abs(turn) < 2.6 && t0 > 0.05 && t1 > 0.05 && !(Math.abs(th) < 0.35);
    if (corner) {
      const tn = Math.tan(Math.abs(turn) / 2), need = rMin * tn;
      if ((t0 < need - 0.01 || t1 < need - 0.01) && iter < 9) {
        const nIn = Math.max(minIn, sIn - Math.max(0, need - t0) * 1.2 - 0.5), nOut = Math.min(maxOut, sOut + Math.max(0, need - t1) * 1.2 - 0 + 0.5);
        if (nIn < sIn - 0.05 || nOut > sOut + 0.05) { sIn = nIn; sOut = nOut; continue; }
      }
      const R = Math.min(40, Math.min(t0, t1) / tn), T = R * tn;
      curve = fillet(A, B, t0, t1, T, R, turn);
      break;
    }
    // a turn whose corner lies behind the entry (t0 <= 0) or past the exit (t1 <= 0), e.g. a lane offset to the
    // side of the crossing road: start the connector earlier / end it later before falling back to a Dubins loop
    if (!corner && Math.abs(turn) > 0.15 && Math.abs(turn) < 2.6 && !(Math.abs(th) < 0.35) && iter < 9) {
      const nIn = t0 <= 0.05 ? Math.max(minIn, sIn - 4) : sIn, nOut = t1 <= 0.05 ? Math.min(maxOut, sOut + 4) : sOut;
      if (nIn < sIn - 0.05 || nOut > sOut + 0.05) { sIn = nIn; sOut = nOut; continue; }
    }
    // a straight-through movement (graph turn < 0.35 rad) is never a Dubins loop: lane offsets / tapers on either
    // side can make the lane headings differ a little and the lanes jog sideways, which is an S-curve
    if (!(Math.abs(th) < 0.35) && (Math.abs(turn) >= 2.6 || (Math.abs(turn) > 0.15 && !corner))) { curve = dubins(A, B, turn >= 0 ? 1 : -1, rMin); break; }
    // near-straight / S-offset: Bezier with handles of a third of the chord
    const chord = Math.hypot(B.x - A.x, B.z - A.z), k = Math.max(0.5, chord * 0.4);
    const P0 = [A.x, A.y, A.z], P1 = [A.x + A.hx * k, A.y, A.z + A.hz * k], P2 = [B.x - B.hx * k, B.y, B.z - B.hz * k], P3 = [B.x, B.y, B.z];
    const n = Math.min(40, Math.max(4, Math.ceil(chord / 1.5)));
    curve = []; for (let i = 0; i <= n; i++) curve.push(bezier(P0, P1, P2, P3, i / n));
    if (minRadius(curve) >= rMin - 0.05 || iter >= 9) break;
    const nIn = Math.max(minIn, sIn - 4), nOut = Math.min(maxOut, sOut + 4);
    if (nIn >= sIn - 0.05 && nOut <= sOut + 0.05) break;
    sIn = nIn; sOut = nOut;
  }
  // the curve must be computed at the final sIn / sOut, and its ends tangent to the lanes
  LG.lanePoint(a, sIn, la, A); LG.lanePoint(b, sOut, lb, B);
  curve = tangentEnds(curve, A, B);
  return { curve, sIn, sOut, radius: minRadius(curve) };
}

/** Make the first / last chord follow the lane tangents exactly (a short straight lead-in / lead-out). */
function tangentEnds(c, A, B) {
  if (c.length < 2) return c;
  const ang = (dx, dz, hx, hz) => Math.acos(Math.max(-1, Math.min(1, (dx * hx + dz * hz) / (Math.hypot(dx, dz) || 1))));
  const out = c.slice();
  out[0] = [A.x, A.y, A.z]; out[out.length - 1] = [B.x, B.y, B.z];
  const d0 = Math.hypot(out[1][0] - out[0][0], out[1][2] - out[0][2]);
  // short lead-in / lead-out along the lane tangents
  if (ang(out[1][0] - out[0][0], out[1][2] - out[0][2], A.hx, A.hz) > 0.005 && d0 > 0.03) {
    const k = Math.min(0.5, d0 / 3);
    out.splice(1, 0, [A.x + A.hx * k, A.y, A.z + A.hz * k]);
  }
  const n = out.length, d1 = Math.hypot(out[n - 1][0] - out[n - 2][0], out[n - 1][2] - out[n - 2][2]);
  if (ang(out[n - 1][0] - out[n - 2][0], out[n - 1][2] - out[n - 2][2], B.hx, B.hz) > 0.005 && d1 > 0.03) {
    const k = Math.min(0.5, d1 / 3);
    out.splice(n - 1, 0, [B.x - B.hx * k, B.y, B.z - B.hz * k]);
  }
  return out;
}

/** Straight - circular arc (radius R, tangent length T from the corner) - straight. */
function fillet(A, B, t0, t1, T, R, turn) {
  const out = [];
  const lerpLine = (x0, y0, z0, x1, y1, z1, step) => {
    const L = Math.hypot(x1 - x0, z1 - z0), n = Math.max(1, Math.ceil(L / step));
    for (let i = 0; i < n; i++) out.push([x0 + (x1 - x0) * (i / n), y0 + (y1 - y0) * (i / n), z0 + (z1 - z0) * (i / n)]);
  };
  const Cx = A.x + A.hx * t0, Cz = A.z + A.hz * t0;
  const TAx = Cx - A.hx * T, TAz = Cz - A.hz * T, TBx = Cx + B.hx * T, TBz = Cz + B.hz * T;
  const total = (t0 - T) + Math.abs(turn) * R + (t1 - T) || 1;
  const yAt = (d) => A.y + (B.y - A.y) * Math.min(1, d / total);
  lerpLine(A.x, A.y, A.z, TAx, yAt(t0 - T), TAz, 6);
  const side = turn >= 0 ? 1 : -1;
  const nx = side > 0 ? -A.hz : A.hz, nz = side > 0 ? A.hx : -A.hx;   // towards the arc centre
  const Ox = TAx + nx * R, Oz = TAz + nz * R;
  const a0 = Math.atan2(TAz - Oz, TAx - Ox), sweep = Math.abs(turn);
  const n = Math.max(2, Math.ceil(sweep / 0.12));
  for (let i = 0; i < n; i++) {
    const ang = a0 + side * sweep * (i / n);
    out.push([Ox + R * Math.cos(ang), yAt(t0 - T + R * sweep * (i / n)), Oz + R * Math.sin(ang)]);
  }
  lerpLine(TBx, yAt(t0 - T + R * sweep), TBz, B.x, B.y, B.z, 6);
  out.push([B.x, B.y, B.z]);
  // drop near-duplicate points (a zero-length straight before / after the arc), keep the end point exact
  const res = [out[0]];
  for (let i = 1; i < out.length; i++) {
    const q = res[res.length - 1];
    if (Math.hypot(out[i][0] - q[0], out[i][2] - q[2]) > 0.05) res.push(out[i]);
    else if (i === out.length - 1) res[res.length - 1] = out[i];
  }
  return res;
}

/** Smallest circumradius over consecutive sample triples (points >= 0.5 m apart). */
function minRadius(c) {
  let mr = 1e9;
  for (let k = 1; k < c.length - 1; k++) {
    const [ax, , az] = c[k - 1], [bx, , bz] = c[k], [cx, , cz] = c[k + 1];
    const ab = Math.hypot(bx - ax, bz - az), bc = Math.hypot(cx - bx, cz - bz), ca = Math.hypot(ax - cx, az - cz);
    if (ab < 0.5 || bc < 0.5) continue;
    const area2 = Math.abs((bx - ax) * (cz - az) - (bz - az) * (cx - ax));
    if (area2 > 1e-9) mr = Math.min(mr, (ab * bc * ca) / (2 * area2));
  }
  return mr;
}

function bezier(P0, P1, P2, P3, t) {
  const a = (1 - t) ** 3, b = 3 * (1 - t) ** 2 * t, c = 3 * (1 - t) * t * t, d = t ** 3;
  return [a * P0[0] + b * P1[0] + c * P2[0] + d * P3[0], a * P0[1] + b * P1[1] + c * P2[1] + d * P3[1],
    a * P0[2] + b * P1[2] + c * P2[2] + d * P3[2]];
}

/** Closest distance between segments AB and CD (2D), with the parameters u on AB and v on CD. */
function segSegDist(ax, az, bx, bz, cx, cz, dx, dz) {
  const pt = (px, pz, x0, z0, x1, z1) => {
    const ex = x1 - x0, ez = z1 - z0, L = ex * ex + ez * ez || 1e-12;
    const t = Math.max(0, Math.min(1, ((px - x0) * ex + (pz - z0) * ez) / L));
    return [Math.hypot(x0 + ex * t - px, z0 + ez * t - pz), t];
  };
  let best = { d: 1e18, u: 0, v: 0 };
  let r = pt(ax, az, cx, cz, dx, dz); if (r[0] < best.d) best = { d: r[0], u: 0, v: r[1] };
  r = pt(bx, bz, cx, cz, dx, dz); if (r[0] < best.d) best = { d: r[0], u: 1, v: r[1] };
  r = pt(cx, cz, ax, az, bx, bz); if (r[0] < best.d) best = { d: r[0], u: r[1], v: 0 };
  r = pt(dx, dz, ax, az, bx, bz); if (r[0] < best.d) best = { d: r[0], u: r[1], v: 1 };
  return best;
}

function segCross(ax, az, bx, bz, cx, cz, dx, dz) {
  const d1x = bx - ax, d1z = bz - az, d2x = dx - cx, d2z = dz - cz;
  const den = d1x * d2z - d1z * d2x;
  if (Math.abs(den) < 1e-12) return -1;
  const u = ((cx - ax) * d2z - (cz - az) * d2x) / den, v = ((cx - ax) * d1z - (cz - az) * d1x) / den;
  return u >= 0 && u <= 1 && v >= 0 && v <= 1 ? u : -1;
}

function buildJunctionsOnce(LG, opts = {}) {
  const G = LG.G, E = G.edgeCount, N = G.nodeCount;
  // ---- mouths: per edge [sStart, sEnd] (sStart = exit point after the start box, sEnd = stop point)
  const mouthS = new Float32Array(2 * E);
  const isJ = new Uint8Array(N);
  for (let n = 0; n < N; n++) {
    const ins = G.inEdges(n), outs = G.outEdges(n);
    let junction = false;
    for (const a of ins) { for (const b of outs) if (b !== G.twin[a] && !hopMap(G, a, b).strict) { junction = true; break; } if (junction) break; }
    isJ[n] = junction ? 1 : 0;
  }
  const cwStop = opts.crosswalkStop || new Set(), cwExit = opts.crosswalkExit || new Set();
  for (let e = 0; e < E; e++) {
    const L = G.length[e];
    const s0 = isJ[G.from(e)] ? Math.min(setbackAt(LG, G.from(e), e) + (cwExit.has(e) ? CROSSWALK : 0), 0.45 * L) : 0;
    const s1 = isJ[G.to(e)] ? Math.min(setbackAt(LG, G.to(e), e) + (cwStop.has(e) ? CROSSWALK : 0), 0.45 * L) : 0;
    mouthS[2 * e] = s0; mouthS[2 * e + 1] = L - s1;
  }
  // mouths pushed out so that connectors crossing this edge's lanes do it inside a junction box (crossingFix)
  if (opts.mouthAdjust) for (const [e, [sx, st]] of opts.mouthAdjust) {
    const L = G.length[e];
    mouthS[2 * e] = Math.min(L, Math.max(mouthS[2 * e], sx));
    mouthS[2 * e + 1] = Math.min(L, Math.max(mouthS[2 * e], Math.min(mouthS[2 * e + 1], st)));
  }
  // ---- connectors: computed first for every movement, then connector chains on short edges are
  // separated (the next connector may not start before the previous one ends on the same edge)
  const specs = [];
  for (let n = 0; n < N; n++) {
    if (!isJ[n]) continue;
    const ins = G.inEdges(n), outs = G.outEdges(n);
    for (const a of ins) for (const b of outs) {
      if (b === G.twin[a] && ins.length + outs.length > 2) continue;
      if (hopMap(G, a, b).strict) continue;
      const na = G.lanes[a], nb = G.lanes[b], th = turnAngle(G, a, b);
      for (let la = 0; la < na; la++) {
        const lb = Math.min(nb - 1, Math.floor(((la + 0.5) * nb) / na));
        specs.push({ n, a, la, b, lb, th, cn: connector(LG, a, la, b, lb, mouthS[2 * a + 1], mouthS[2 * b], th, R_MIN) });
      }
    }
  }
  const capOut = new Float64Array(E).fill(Infinity), floorIn = new Float64Array(E).fill(-Infinity);
  const split = new Uint8Array(E);
  for (let pass = 0; pass < 6; pass++) {
    const maxOutE = new Float64Array(E).fill(-1), minInE = new Float64Array(E).fill(Infinity);
    for (const sp of specs) { maxOutE[sp.b] = Math.max(maxOutE[sp.b], sp.cn.sOut); minInE[sp.a] = Math.min(minInE[sp.a], sp.cn.sIn); }
    let changed = 0;
    for (let e = 0; e < E; e++) {
      if (maxOutE[e] >= 0 && minInE[e] < Infinity && maxOutE[e] > minInE[e] + 1e-6) {
        const mid = Math.max(mouthS[2 * e], Math.min(mouthS[2 * e + 1], (Math.min(maxOutE[e], capOut[e]) + Math.max(minInE[e], floorIn[e])) / 2));
        capOut[e] = Math.min(capOut[e], mid); floorIn[e] = Math.max(floorIn[e], Math.min(mid, capOut[e]));
        split[e] = 1; changed++;
      }
    }
    if (!changed) break;
    for (const sp of specs) {
      if (sp.cn.sOut > capOut[sp.b] + 1e-6 || sp.cn.sIn < floorIn[sp.a] - 1e-6) {
        sp.cn = connector(LG, sp.a, sp.la, sp.b, sp.lb, mouthS[2 * sp.a + 1], mouthS[2 * sp.b], sp.th, R_MIN,
          { floorIn: floorIn[sp.a], capOut: capOut[sp.b] });
      }
    }
  }
  let chainsSplit = 0; for (let e = 0; e < E; e++) chainsSplit += split[e];
  let specI = 0;

  // ---- movements (lane-level connectors) + conflicts per junction
  let tooTight = 0;
  const jNode = [], jFirstM = [], jCountM = [], jFirstC = [], jCountC = [], jFirstH = [], jCountH = [];
  const mInfo = [], mF = [], pts = [], conf = [], confS = [], hull = [];
  const A = {}, B = {};
  for (let n = 0; n < N; n++) {
    if (!isJ[n]) continue;
    const ins = G.inEdges(n), outs = G.outEdges(n);
    const m0 = mInfo.length / 7;
    const corners = [];
    for (const e of [...ins, ...outs]) {
      const s = ins.includes(e) ? mouthS[2 * e + 1] : mouthS[2 * e];
      const bd = LG.bounds(e, s);
      LG.offsetPoint(e, s, bd.right, 0, A); corners.push([A.x, A.z]);
      LG.offsetPoint(e, s, bd.left, 0, A); corners.push([A.x, A.z]);
    }
    while (specI < specs.length && specs[specI].n === n) {
      {
        {
          const { a, la, b, lb, th, cn } = specs[specI++];
          const na = G.lanes[a];
          const curve = cn.curve;
          if (cn.radius < R_MIN - 0.05) tooTight++;
          const first = pts.length / 4;
          let acc = 0, prev = null;
          for (const p of curve) {
            if (prev) acc += Math.hypot(p[0] - prev[0], p[2] - prev[2]);
            pts.push(p[0], p[1], p[2], acc);
            prev = p;
          }
          // preferred lane use: right turns from the rightmost lane, left turns from the leftmost
          const pref = th > 0.5 ? la === 0 : th < -0.5 ? la === na - 1 : true;
          mInfo.push(a, la, b, lb, first, curve.length, pref ? 1 : 0);
          mF.push(cn.sIn, cn.sOut, acc, th);
        }
      }
    }
    const m1 = mInfo.length / 7;
    // conflicts between movements of this junction
    const c0 = conf.length / 3;
    for (let i = m0; i < m1; i++) {
      for (let j = i + 1; j < m1; j++) {
        const fi = mInfo[7 * i + 4], ci = mInfo[7 * i + 5], fj = mInfo[7 * j + 4], cj = mInfo[7 * j + 5];
        const sameIn = mInfo[7 * i] === mInfo[7 * j] && mInfo[7 * i + 1] === mInfo[7 * j + 1];
        const sameOut = mInfo[7 * i + 2] === mInfo[7 * j + 2] && mInfo[7 * i + 3] === mInfo[7 * j + 3];
        if (sameIn) continue;      // same entry lane: an ordinary queue, not stored (derivable from mInfo)
        // exits within a car width: merge conflict (also the same exit lane)
        const exi = 4 * (fi + ci - 1), exj = 4 * (fj + cj - 1);
        if (sameOut || Math.hypot(pts[exi] - pts[exj], pts[exi + 2] - pts[exj + 2]) < CONFLICT_SEP) {
          conf.push(i, j, 1); confS.push(pts[exi + 3], pts[exj + 3]); continue;
        }
        // crossing of the centre paths, or swept bodies touching
        let hit = false;
        for (let p = 0; p < ci - 1 && !hit; p++) {
          const ax = pts[4 * (fi + p)], az = pts[4 * (fi + p) + 2], bx = pts[4 * (fi + p + 1)], bz = pts[4 * (fi + p + 1) + 2];
          for (let q = 0; q < cj - 1; q++) {
            const cx = pts[4 * (fj + q)], cz = pts[4 * (fj + q) + 2], dx = pts[4 * (fj + q + 1)], dz = pts[4 * (fj + q + 1) + 2];
            if (Math.max(ax, bx) + 2 < Math.min(cx, dx) || Math.max(cx, dx) + 2 < Math.min(ax, bx) ||
                Math.max(az, bz) + 2 < Math.min(cz, dz) || Math.max(cz, dz) + 2 < Math.min(az, bz)) continue;
            const u = segCross(ax, az, bx, bz, cx, cz, dx, dz);
            if (u >= 0) {
              const si = pts[4 * (fi + p) + 3] + u * (pts[4 * (fi + p + 1) + 3] - pts[4 * (fi + p) + 3]);
              conf.push(i, j, 0); confS.push(si, pts[4 * (fj + q) + 3]); hit = true; break;
            }
          }
        }
        if (hit) continue;
        // centre paths passing closer than CONFLICT_SEP (segment-to-segment distance)
        for (let p = 0; p < ci - 1 && !hit; p++) {
          const ax = pts[4 * (fi + p)], az = pts[4 * (fi + p) + 2], bx = pts[4 * (fi + p + 1)], bz = pts[4 * (fi + p + 1) + 2];
          for (let q = 0; q < cj - 1; q++) {
            const cx = pts[4 * (fj + q)], cz = pts[4 * (fj + q) + 2], dx = pts[4 * (fj + q + 1)], dz = pts[4 * (fj + q + 1) + 2];
            if (Math.max(ax, bx) + CONFLICT_SEP < Math.min(cx, dx) || Math.max(cx, dx) + CONFLICT_SEP < Math.min(ax, bx) ||
                Math.max(az, bz) + CONFLICT_SEP < Math.min(cz, dz) || Math.max(cz, dz) + CONFLICT_SEP < Math.min(az, bz)) continue;
            const r = segSegDist(ax, az, bx, bz, cx, cz, dx, dz);
            if (r.d < CONFLICT_SEP) {
              const si = pts[4 * (fi + p) + 3] + r.u * (pts[4 * (fi + p + 1) + 3] - pts[4 * (fi + p) + 3]);
              const sj = pts[4 * (fj + q) + 3] + r.v * (pts[4 * (fj + q + 1) + 3] - pts[4 * (fj + q) + 3]);
              conf.push(i, j, 0); confS.push(si, sj); hit = true; break;
            }
          }
        }
      }
    }
    // junction box: convex hull of the mouth corners
    const h0 = hull.length / 2;
    for (const [x, z] of convexHull(corners)) hull.push(x, z);
    jNode.push(n); jFirstM.push(m0); jCountM.push(m1 - m0); jFirstC.push(c0); jCountC.push(conf.length / 3 - c0);
    jFirstH.push(h0); jCountH.push(hull.length / 2 - h0);
  }
  // ---- ramp gores: merges without an added lane (ramp lanes continue on the mainline's lanes)
  const goreS = new Float32Array(E).fill(-1);
  const goreFallback = [];
  for (let n = 0; n < N; n++) {
    const ins = G.inEdges(n), outs = G.outEdges(n);
    if (ins.length < 2 || outs.length !== 1) continue;
    const b = outs[0];
    for (const a of ins) {
      if (!(G.cls[a] === 1 || G.cls[a] === 3)) continue;                 // motorway_link / trunk_link
      const h = hopMap(G, a, b);
      if (!h.strict) continue;
      // the mainline in-edge of this merge (the widest other in-edge)
      let m = -1;
      for (const o of ins) if (o !== a && (m < 0 || G.lanes[o] > G.lanes[m])) m = o;
      if (m < 0) continue;
      const hm = hopMap(G, m, b);
      // lanes of b used by the mainline: [-hm.shift .. nm-1-hm.shift]; ramp lanes: [-h.shift .. na-1-h.shift]
      const mLo = -hm.shift, rHi = G.lanes[a] - 1 - h.shift;
      if (rHi < mLo) continue;                                            // ramp joins its own (added) lane
      // walk back along the ramp until its lane 0 centre clears the mainline's right pavement edge
      const L = G.length[a], Lm = G.length[m];
      // which side does the ramp join from? (signed offset of the ramp's far end from the mainline line)
      const sideOf = (P, right) => {
        let best = 1e9, sd = 0;
        for (let sm = Lm; sm >= Math.max(0, Lm - 450); sm -= 3) {
          LG.offsetPoint(m, sm, right, 0, B);
          const d = Math.hypot(P.x - B.x, P.z - B.z);
          if (d < best) { best = d; sd = (P.x - B.x) * -B.hz + (P.z - B.z) * B.hx; }
        }
        return sd;
      };
      LG.lanePoint(a, Math.max(0, L - 400), 0, A);
      const fromRight = sideOf(A, 0) >= 0;
      // the mainline's outer lane edge on that side, last 450 m (+ right normal)
      const edgePts = [];
      for (let sm = Lm; sm >= Math.max(0, Lm - 450); sm -= 3) {
        const bd = LG.bounds(m, sm);
        LG.offsetPoint(m, sm, fromRight ? bd.laneRight : bd.laneLeft, 0, B); edgePts.push(B.x, B.z, -B.hz, B.hx);
      }
      let sg = -1, bestS = Math.max(0, L - 60), bestSide = -1e9;
      for (let s = L; s >= Math.max(0, L - 400); s -= 2) {
        // the ramp lane nearest the mainline: lane 0 for a right-side merge, the leftmost for a left-side one
        LG.lanePoint(a, s, fromRight ? 0 : G.lanes[a] - 1, A);
        let best = 1e9, side = 0;
        for (let q = 0; q < edgePts.length; q += 4) {
          const d = Math.hypot(A.x - edgePts[q], A.z - edgePts[q + 1]);
          if (d < best) { best = d; side = (A.x - edgePts[q]) * edgePts[q + 2] + (A.z - edgePts[q + 1]) * edgePts[q + 3]; }
        }
        if (!fromRight) side = -side;
        if (side > bestSide + 0.1) { bestSide = side; bestS = s; }
        // that lane's centre must lie outside the mainline's outer lane edge by half a lane + clearance
        if (side > 0 && side - LG.laneWidth(a, s) / 2 >= GORE_CLEAR) { sg = s; break; }
      }
      // no fully clear point (a ramp drawn hugging the mainline): wait where the clearance is largest
      goreS[a] = sg >= 0 ? sg : bestS;
      if (sg < 0) goreFallback.push(a);
    }
  }
  return new JunctionSet(G, LG, {
    mouthS, goreS, jNode: Uint32Array.from(jNode), jFirstM: Uint32Array.from(jFirstM), jCountM: Uint32Array.from(jCountM),
    jFirstC: Uint32Array.from(jFirstC), jCountC: Uint32Array.from(jCountC), jFirstH: Uint32Array.from(jFirstH), jCountH: Uint32Array.from(jCountH),
    mInfo: Uint32Array.from(mInfo), mF: Float32Array.from(mF), pts: Float32Array.from(pts),
    conf: Uint32Array.from(conf), confS: Float32Array.from(confS), hull: Float32Array.from(hull),
  }, { goreFallback, tooTight, chainsSplit });
}

/**
 * Junctions with connector / edge crossings resolved (default; opts.crossingFix = false: one plain build).
 * A connector whose path runs over another edge's lanes outside that edge's junction mouths (s in
 * [sExit, sStop], within 3 m of height) is invisible to cars on that edge: they are not on a connector
 * there, so the conflict matrix cannot pair them (scale agent 2026-10-24: m148717 over edge 3879).
 * Fix, up to 3 rounds:
 *  1. find the crossings (connectorEdgeCrossings);
 *  2. push the crossed edge's nearer mouth past the crossing (+3 m; at most 30 m; only toward an end
 *     that is a junction), so cars there are on that junction's connectors;
 *  3. rebuild the connectors with those mouths.
 * Then the crossing is between two connectors: in one junction the normal conflict matrix holds it;
 * between two junctions of a cluster, crossConflicts() appends the pair to conf (after the last junction's
 * range; JunctionSet.conflictsOf, the sim and the GPU pack read conf globally).
 */
export function buildJunctions(LG, opts = {}) {
  if (opts.crossingFix === false) return buildJunctionsOnce(LG, opts);
  const adjust = new Map();
  let J = buildJunctionsOnce(LG, opts), rounds = 0, first = null;
  for (; rounds < (opts.crossingRounds ?? 6); rounds++) {
    const cr = connectorEdgeCrossings(J);
    if (!first) first = cr.length;
    let changed = 0;
    const G = LG.G;
    for (const c of cr) {
      const { sStop, sExit } = J.mouth(c.e), PAD = 3, MAXEXT = 30;
      const canStart = J.junctionAt(G.from(c.e)) >= 0, canEnd = J.junctionAt(G.to(c.e)) >= 0;
      const extS = c.s2 + PAD - sExit, extE = sStop - (c.s1 - PAD);
      let sx = -Infinity, st = Infinity;
      if (canStart && (!canEnd || extS <= extE) && extS <= MAXEXT) sx = c.s2 + PAD;
      else if (canEnd && extE <= MAXEXT) st = c.s1 - PAD;
      else continue;
      const cur = adjust.get(c.e) || [-Infinity, Infinity];
      const nx = Math.max(cur[0], sx), nt = Math.min(cur[1], st);
      if (nx > cur[0] + 0.01 || nt < cur[1] - 0.01) { adjust.set(c.e, [nx, nt]); changed++; }
    }
    if (!changed) break;
    J = buildJunctionsOnce(LG, { ...opts, mouthAdjust: adjust });
  }
  const added = crossConflicts(J);
  const left = connectorEdgeCrossings(J);
  J.crossingFix = { crossingsBefore: first, crossingsAfter: left.length, edgesAdjusted: adjust.size, rounds, crossConflicts: added };
  return J;
}

/**
 * Connector paths over another edge's lanes outside its mouths: [{m, e, s1, s2, n}] (s range on e, n path
 * points). The connector centre must be at least 0.5 m inside e's lane span, within 0.9 m along e's
 * sample, within 3 m of height, over >= 2 points. e is neither the connector's in- nor out-edge.
 */
export function connectorEdgeCrossings(J) {
  const LG = J.LG, G = LG.G, P = J.pts, M = J.moveCount, NP = P.length / 4, E = G.edgeCount;
  const owner = new Int32Array(NP);
  for (let m = 0; m < M; m++) { const f = J.mInfo[7 * m + 4], n = J.mInfo[7 * m + 5]; for (let k = 0; k < n; k++) owner[f + k] = m; }
  let x0 = Infinity, z0 = Infinity, x1 = -Infinity, z1 = -Infinity;
  for (let p = 0; p < NP; p++) { const x = P[4 * p], z = P[4 * p + 2]; if (x < x0) x0 = x; if (x > x1) x1 = x; if (z < z0) z0 = z; if (z > z1) z1 = z; }
  if (!NP) return [];
  const C = 20, NX = Math.ceil((x1 - x0) / C) + 1, NZ = Math.ceil((z1 - z0) / C) + 1;
  const cellOf = (x, z) => Math.floor((x - x0) / C) * NZ + Math.floor((z - z0) / C);
  const start = new Uint32Array(NX * NZ + 1);
  for (let p = 0; p < NP; p++) start[cellOf(P[4 * p], P[4 * p + 2]) + 1]++;
  for (let c = 0; c < NX * NZ; c++) start[c + 1] += start[c];
  const fill = start.slice(0, NX * NZ), idx = new Uint32Array(NP);
  for (let p = 0; p < NP; p++) idx[fill[cellOf(P[4 * p], P[4 * p + 2])]++] = p;
  const Q = {}, hits = new Map();
  for (let e = 0; e < E; e++) {
    const sStop = J.mouthS[2 * e + 1], sExit = J.mouthS[2 * e];
    for (let s = sExit; s <= sStop; s += 1.5) {
      const b = LG.bounds(e, s), mid = (b.laneRight + b.laneLeft) / 2, hw = (b.laneRight - b.laneLeft) / 2;
      LG.offsetPoint(e, s, mid, 0, Q);
      if (Q.x < x0 - C || Q.x > x1 + C || Q.z < z0 - C || Q.z > z1 + C) continue;
      const cx = Math.floor((Q.x - x0) / C), cz = Math.floor((Q.z - z0) / C);
      for (let i = -1; i <= 1; i++) for (let j = -1; j <= 1; j++) {
        const X = cx + i, Z = cz + j; if (X < 0 || Z < 0 || X >= NX || Z >= NZ) continue;
        const c = X * NZ + Z;
        for (let q = start[c]; q < start[c + 1]; q++) {
          const p = idx[q], m = owner[p];
          if (J.mInfo[7 * m] === e || J.mInfo[7 * m + 2] === e || J.mInfo[7 * m] === J.mInfo[7 * m + 2]) continue;   // (in = out: U-turn on a loop road)
          const dx = P[4 * p] - Q.x, dz = P[4 * p + 2] - Q.z;
          if (Math.abs(dx * Q.hx + dz * Q.hz) > 0.9 || Math.abs(P[4 * p + 1] - Q.y) > 3) continue;
          if (Math.abs(dx * -Q.hz + dz * Q.hx) > hw - 0.5) continue;
          const k = m * E + e, h = hits.get(k);
          if (h) { h.n++; h.s1 = Math.min(h.s1, s); h.s2 = Math.max(h.s2, s); } else hits.set(k, { m, e, s1: s, s2: s, n: 1 });
        }
      }
    }
  }
  return [...hits.values()].filter((h) => h.n >= 2);
}

/** Conflict test of two connector paths (as within a junction): [type, si, sj] or null. */
function pairConflict(J, i, j) {
  const pts = J.pts, mInfo = J.mInfo;
  const fi = mInfo[7 * i + 4], ci = mInfo[7 * i + 5], fj = mInfo[7 * j + 4], cj = mInfo[7 * j + 5];
  if (mInfo[7 * i] === mInfo[7 * j] && mInfo[7 * i + 1] === mInfo[7 * j + 1]) return null;
  const exi = 4 * (fi + ci - 1), exj = 4 * (fj + cj - 1);
  const sameOut = mInfo[7 * i + 2] === mInfo[7 * j + 2] && mInfo[7 * i + 3] === mInfo[7 * j + 3];
  if (sameOut || (Math.hypot(pts[exi] - pts[exj], pts[exi + 2] - pts[exj + 2]) < CONFLICT_SEP && Math.abs(pts[exi + 1] - pts[exj + 1]) < 3)) return [1, pts[exi + 3], pts[exj + 3]];
  let best = null;
  for (let p = 0; p < ci - 1; p++) {
    const ax = pts[4 * (fi + p)], az = pts[4 * (fi + p) + 2], bx = pts[4 * (fi + p + 1)], bz = pts[4 * (fi + p + 1) + 2], ay = pts[4 * (fi + p) + 1];
    for (let q = 0; q < cj - 1; q++) {
      const cx = pts[4 * (fj + q)], cz = pts[4 * (fj + q) + 2], dx = pts[4 * (fj + q + 1)], dz = pts[4 * (fj + q + 1) + 2];
      if (Math.abs(ay - pts[4 * (fj + q) + 1]) > 3) continue;
      if (Math.max(ax, bx) + CONFLICT_SEP < Math.min(cx, dx) || Math.max(cx, dx) + CONFLICT_SEP < Math.min(ax, bx) ||
          Math.max(az, bz) + CONFLICT_SEP < Math.min(cz, dz) || Math.max(cz, dz) + CONFLICT_SEP < Math.min(az, bz)) continue;
      const u = segCross(ax, az, bx, bz, cx, cz, dx, dz);
      if (u >= 0) return [0, pts[4 * (fi + p) + 3] + u * (pts[4 * (fi + p + 1) + 3] - pts[4 * (fi + p) + 3]), pts[4 * (fj + q) + 3]];
      const r = segSegDist(ax, az, bx, bz, cx, cz, dx, dz);
      if (r.d < CONFLICT_SEP && (!best || r.d < best.d)) best = { d: r.d, si: pts[4 * (fi + p) + 3] + r.u * (pts[4 * (fi + p + 1) + 3] - pts[4 * (fi + p) + 3]), sj: pts[4 * (fj + q) + 3] + r.v * (pts[4 * (fj + q + 1) + 3] - pts[4 * (fj + q) + 3]) };
    }
  }
  return best ? [0, best.si, best.sj] : null;
}

/**
 * Conflicts between connectors of DIFFERENT junctions (clustered junctions: divided roads, offset
 * crossings), with the same test as inside a junction. Only junctions whose nodes are within 120 m.
 * Appended to conf / confS (after the last junction's range). Returns the number added.
 */
function crossConflicts(J) {
  const G = J.G, M = J.moveCount, P = J.pts;
  const jOf = new Int32Array(M);
  for (let j = 0; j < J.junctionCount; j++) for (let m = J.jFirstM[j]; m < J.jFirstM[j] + J.jCountM[j]; m++) jOf[m] = j;
  const bb = new Float32Array(4 * M);
  for (let m = 0; m < M; m++) {
    const f = J.mInfo[7 * m + 4], n = J.mInfo[7 * m + 5]; let a = Infinity, b = -Infinity, c = Infinity, d = -Infinity;
    for (let k = 0; k < n; k++) { const x = P[4 * (f + k)], z = P[4 * (f + k) + 2]; if (x < a) a = x; if (x > b) b = x; if (z < c) c = z; if (z > d) d = z; }
    bb[4 * m] = a; bb[4 * m + 1] = b; bb[4 * m + 2] = c; bb[4 * m + 3] = d;
  }
  // junctions by 120 m cells
  const cell = 120, grid = new Map(), key = (x, z) => Math.floor(x / cell) * 1e6 + Math.floor(z / cell);
  for (let j = 0; j < J.junctionCount; j++) { const n = J.jNode[j], k = key(G.x(n), G.z(n)); let l = grid.get(k); if (!l) grid.set(k, (l = [])); l.push(j); }
  const known = new Set();
  for (let c = 0; c < J.conf.length / 3; c++) { const a = J.conf[3 * c], b = J.conf[3 * c + 1]; known.add(a < b ? a * M + b : b * M + a); }
  const add = [], addS = [];
  for (let j = 0; j < J.junctionCount; j++) {
    const n = J.jNode[j], cx = Math.floor(G.x(n) / cell), cz = Math.floor(G.z(n) / cell);
    for (let u = -1; u <= 1; u++) for (let v = -1; v <= 1; v++) for (const k of grid.get((cx + u) * 1e6 + cz + v) || []) {
      if (k <= j) continue;
      const n2 = J.jNode[k];
      if (Math.hypot(G.x(n2) - G.x(n), G.z(n2) - G.z(n)) > 120) continue;
      for (let m = J.jFirstM[j]; m < J.jFirstM[j] + J.jCountM[j]; m++) for (let o = J.jFirstM[k]; o < J.jFirstM[k] + J.jCountM[k]; o++) {
        if (bb[4 * o] > bb[4 * m + 1] + CONFLICT_SEP || bb[4 * m] > bb[4 * o + 1] + CONFLICT_SEP || bb[4 * o + 2] > bb[4 * m + 3] + CONFLICT_SEP || bb[4 * m + 2] > bb[4 * o + 3] + CONFLICT_SEP) continue;
        // consecutive connectors of a chain (m ends on the edge o starts on, or vice versa) follow each other
        if (J.mInfo[7 * m + 2] === J.mInfo[7 * o] || J.mInfo[7 * o + 2] === J.mInfo[7 * m]) continue;
        if (known.has(m < o ? m * M + o : o * M + m)) continue;
        const r = pairConflict(J, m, o);
        if (r) { add.push(m, o, r[0]); addS.push(r[1], r[2]); }
      }
    }
  }
  if (add.length) {
    const conf = new Uint32Array(J.conf.length + add.length); conf.set(J.conf); conf.set(add, J.conf.length);
    const confS = new Float32Array(J.confS.length + addS.length); confS.set(J.confS); confS.set(addS, J.confS.length);
    J.conf = conf; J.confS = confS; J._conf = null;
  }
  return add.length / 3;
}

function maxStep(c) {
  let m = 0;
  for (let k = 1; k < c.length - 1; k++) {
    const ux = c[k][0] - c[k - 1][0], uz = c[k][2] - c[k - 1][2], vx = c[k + 1][0] - c[k][0], vz = c[k + 1][2] - c[k][2];
    const lu = Math.hypot(ux, uz), lv = Math.hypot(vx, vz);
    if (lu > 0.05 && lv > 0.05) m = Math.max(m, Math.acos(Math.max(-1, Math.min(1, (ux * vx + uz * vz) / (lu * lv)))));
  }
  return m;
}

/** True if a sampled path reverses direction (a cusp / fold) anywhere. */
function folds(c) {
  for (let k = 1; k < c.length - 1; k++) {
    const ux = c[k][0] - c[k - 1][0], uz = c[k][2] - c[k - 1][2], vx = c[k + 1][0] - c[k][0], vz = c[k + 1][2] - c[k][2];
    if (Math.hypot(ux, uz) > 1e-3 && Math.hypot(vx, vz) > 1e-3 && ux * vx + uz * vz < 0) return true;
  }
  return false;
}

/**
 * Dubins-style arc - straight - arc path (same turn direction, radius r) from A (heading hx,hz)
 * to B: no cusps for any configuration. side +1 = right turn (heading angle increasing).
 */
function dubins(A, B, side, r) {
  const nA = side > 0 ? [-A.hz, A.hx] : [A.hz, -A.hx], nB = side > 0 ? [-B.hz, B.hx] : [B.hz, -B.hx];
  const C1 = [A.x + r * nA[0], A.z + r * nA[1]], C2 = [B.x + r * nB[0], B.z + r * nB[1]];
  let ux = C2[0] - C1[0], uz = C2[1] - C1[1];
  const d = Math.hypot(ux, uz);
  if (d < 1e-6) { ux = A.hx; uz = A.hz; } else { ux /= d; uz /= d; }
  const nU = side > 0 ? [-uz, ux] : [uz, -ux];
  const T1 = [C1[0] - r * nU[0], C1[1] - r * nU[1]], T2 = [C2[0] - r * nU[0], C2[1] - r * nU[1]];
  const ang = (p, c) => Math.atan2(p[1] - c[1], p[0] - c[0]);
  const sweep = (from, to) => { let a = side > 0 ? to - from : from - to; a = ((a % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI); return a; };
  const out = [];
  const arc = (C, a0, phi, y) => {
    const n = Math.max(1, Math.ceil(phi / 0.25));
    for (let i = 0; i <= n; i++) { const a = a0 + side * phi * (i / n); out.push([C[0] + r * Math.cos(a), y, C[1] + r * Math.sin(a)]); }
  };
  const a1 = ang([A.x, A.z], C1), p1 = sweep(a1, ang(T1, C1));
  arc(C1, a1, p1, A.y);
  const a2 = ang(T2, C2), p2 = sweep(a2, ang([B.x, B.z], C2));
  const nLine = Math.max(1, Math.ceil(d / 3));
  for (let i = 1; i < nLine; i++) out.push([T1[0] + (T2[0] - T1[0]) * (i / nLine), A.y + (B.y - A.y) * (i / nLine), T1[1] + (T2[1] - T1[1]) * (i / nLine)]);
  const start = out.length;
  arc(C2, a2, p2, B.y);
  out[out.length - 1] = [B.x, B.y, B.z];
  out[0] = [A.x, A.y, A.z];
  // drop duplicate points
  return out.filter((p, i) => i === 0 || Math.hypot(p[0] - out[i - 1][0], p[2] - out[i - 1][2]) > 1e-3 || i === start);
}

function convexHull(p) {
  const pts = p.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (pts.length < 3) return pts;
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower = [], upper = [];
  for (const q of pts) { while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], q) <= 0) lower.pop(); lower.push(q); }
  for (let i = pts.length - 1; i >= 0; i--) { const q = pts[i]; while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], q) <= 0) upper.pop(); upper.push(q); }
  return lower.slice(0, -1).concat(upper.slice(0, -1));
}

export const JUNCTION_SECTIONS = ['mouthS', 'goreS', 'jNode', 'jFirstM', 'jCountM', 'jFirstC', 'jCountC', 'jFirstH', 'jCountH',
  'mInfo', 'mF', 'pts', 'conf', 'confS', 'hull'];

export class JunctionSet {
  constructor(G, LG, a, extra = {}) {
    this.G = G; this.LG = LG; Object.assign(this, a);
    this.goreFallback = extra.goreFallback || null;
    this.tooTight = extra.tooTight ?? null;
    this.chainsSplit = extra.chainsSplit ?? null;   // short edges where two connector chains were separated      // connectors that could not reach R_MIN (edges too short)
    this.moveCount = this.mInfo.length / 7;
    this.junctionCount = this.jNode.length;
    // index: node -> junction, (in-edge) -> first movement
    this.jOfNode = new Map();
    for (let j = 0; j < this.junctionCount; j++) this.jOfNode.set(this.jNode[j], j);
    this.mLen = new Float32Array(this.moveCount);
    for (let m = 0; m < this.moveCount; m++) this.mLen[m] = this.mF[4 * m + 2];
    this._byIn = null;
  }
  mouth(e) { return { sStop: this.mouthS[2 * e + 1], sExit: this.mouthS[2 * e] }; }
  gore(e) { return this.goreS[e]; }
  junctionAt(node) { const j = this.jOfNode.get(node); return j === undefined ? -1 : j; }
  movementsAt(node) {
    const j = this.junctionAt(node); if (j < 0) return [];
    const out = []; for (let m = this.jFirstM[j]; m < this.jFirstM[j] + this.jCountM[j]; m++) out.push(m); return out;
  }
  movement(a, la, b) {
    if (!this._byIn) {
      this._byIn = new Map();
      for (let m = 0; m < this.moveCount; m++) {
        const k = this.mInfo[7 * m] * 64 + this.mInfo[7 * m + 1];
        let l = this._byIn.get(k); if (!l) this._byIn.set(k, (l = [])); l.push(m);
      }
    }
    for (const m of this._byIn.get(a * 64 + la) || []) if (this.mInfo[7 * m + 2] === b) return m;
    return -1;
  }
  describe(m) {
    const i = this.mInfo, f = this.mF;
    return { inEdge: i[7 * m], inLane: i[7 * m + 1], outEdge: i[7 * m + 2], outLane: i[7 * m + 3], preferred: !!i[7 * m + 6],
      sIn: f[4 * m], sOut: f[4 * m + 1], length: f[4 * m + 2], turn: f[4 * m + 3] };
  }
  /** Position / heading t metres along connector m. */
  arcPoint(m, t, out = {}) {
    const first = this.mInfo[7 * m + 4], n = this.mInfo[7 * m + 5], P = this.pts;
    let k = 0;
    while (k < n - 2 && P[4 * (first + k + 1) + 3] < t) k++;
    const a = 4 * (first + k), b = 4 * (first + k + 1);
    const d = P[b + 3] - P[a + 3] || 1, u = Math.max(0, Math.min(1, (t - P[a + 3]) / d));
    out.x = P[a] + (P[b] - P[a]) * u; out.y = P[a + 1] + (P[b + 1] - P[a + 1]) * u; out.z = P[a + 2] + (P[b + 2] - P[a + 2]) * u;
    const hx = P[b] - P[a], hz = P[b + 2] - P[a + 2], L = Math.hypot(hx, hz) || 1;
    out.hx = hx / L; out.hz = hz / L;
    return out;
  }
  conflictsOf(m) {
    if (!this._conf) {
      this._conf = new Map();
      for (let c = 0; c < this.conf.length / 3; c++) {
        const a = this.conf[3 * c], b = this.conf[3 * c + 1], t = this.conf[3 * c + 2];
        const sa = this.confS[2 * c], sb = this.confS[2 * c + 1];
        (this._conf.get(a) || this._conf.set(a, []).get(a)).push({ other: b, type: t, s: sa, sOther: sb });
        (this._conf.get(b) || this._conf.set(b, []).get(b)).push({ other: a, type: t, s: sb, sOther: sa });
      }
    }
    return this._conf.get(m) || [];
  }
  /** Flat typed arrays for a GPU upload / data file. */
  arrays() { const o = {}; for (const k of JUNCTION_SECTIONS) o[k] = this[k]; return o; }
}

const TYPES = { Float32: Float32Array, Uint32: Uint32Array };
/** Load data/junctions.bin (written by tools/build_junctions.mjs) for the lane geometry LG. */
export async function loadJunctions(metaUrl, LG) {
  const meta = await (await fetch(metaUrl)).json();
  const buf = await (await fetch(metaUrl.replace(/[^/]*$/, '') + meta.file)).arrayBuffer();
  const a = {};
  for (const [k, s] of Object.entries(meta.sections)) a[k] = new TYPES[s.type](buf, s.byteOffset, s.length);
  return new JunctionSet(LG.G, LG, a);
}
