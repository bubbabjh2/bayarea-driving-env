// The shared lane-mapping rule across a node (one place; the sim, the renderer and the
// build's bannedHops all use it; tools/build_roads.py compute_hops() is its Python mirror).
//
// hopMap(G, e, en) -> { strict, shift }
//  STRICT hop (a continuous carriageway): a plain continuation (e has no other way on and en
//   no other feeder), or both edges on the freeway system with |turn| < 0.5 rad. Lane l of e
//   continues as lane l - shift of en (lane 0 = rightmost). Lanes that map outside en end at
//   the node (the geometry tapers them).
//   shift: simple continuation: e.lanes - en.lanes (the left / median lanes are continuous;
//   lanes are added or dropped on the right). Diverge: the out-edges are ordered right to
//   left by GEOMETRY (lateral offset of the point 50 m into each out-edge, in e's end frame;
//   ties under 1 m by turn angle) and take consecutive lane blocks from the right. Merge: the
//   in-edges are ordered the same way (point 50 m before the node, in en's start frame).
//  JUNCTION hop (intersection turn): no lane continuity (the sim re-projects the car).
import { LANE_W, FREEWAY } from './graph.js';

const TURN_STRICT = 1.2;

function heading(G, e, end) {
  const f = G.first[e], c = G.count[e];
  const a = end ? G.poly[f + c - 2] : G.poly[f], b = end ? G.poly[f + c - 1] : G.poly[f + 1];
  return Math.atan2(G.z(b) - G.z(a), G.x(b) - G.x(a));
}
export function turnAngle(G, e, en) {
  let d = heading(G, en, false) - heading(G, e, true);
  while (d > Math.PI) d -= 2 * Math.PI;
  while (d < -Math.PI) d += 2 * Math.PI;
  return d; // > 0 = right turn (x east, z south)
}
/** Point `dist` m into the edge from its start (fromEnd=false) or back from its end. */
export function pointInto(G, e, dist, fromEnd) {
  const f = G.first[e], c = G.count[e];
  let acc = 0;
  for (let k = 0; k < c - 1; k++) {
    const a = G.poly[fromEnd ? f + c - 1 - k : f + k], b = G.poly[fromEnd ? f + c - 2 - k : f + k + 1];
    const L = Math.hypot(G.x(b) - G.x(a), G.z(b) - G.z(a));
    if (acc + L >= dist) {
      const t = L > 0 ? (dist - acc) / L : 0;
      return [G.x(a) + (G.x(b) - G.x(a)) * t, G.z(a) + (G.z(b) - G.z(a)) * t];
    }
    acc += L;
  }
  const n = G.poly[fromEnd ? f : f + c - 1];
  return [G.x(n), G.z(n)];
}
function side(G, h, node, p) {
  const dx = Math.cos(h), dz = Math.sin(h);
  return (p[0] - G.x(node)) * -dz + (p[1] - G.z(node)) * dx; // right-positive
}

const cacheOf = new WeakMap();
export function hopMap(G, e, en) {
  let c = cacheOf.get(G);
  if (!c) { c = new Map(); cacheOf.set(G, c); }
  const key = e * G.edgeCount + en;
  let h = c.get(key);
  if (h) return h;
  const node = G.to(e);
  const outs = [], ins = [];
  for (const o of G.outEdges(node)) if (o !== G.twin[e]) outs.push(o);
  for (const i of G.inEdges(G.from(en))) if (i !== G.twin[en]) ins.push(i);
  const simple = outs.length <= 1 && ins.length <= 1;
  // freeway system (motorway / trunk and their links) to freeway system is always a continuous
  // carriageway (gore merge / diverge), never a junction box, whatever the drawn angle
  // (a pure merge or diverge at any drawn angle; at a node with several ins AND outs only hops
  // under ~69 deg, so at-grade expressway intersections with ramps stay junctions)
  const strict = simple || (FREEWAY.has(G.className(e)) && FREEWAY.has(G.className(en)) &&
    (outs.length <= 1 || ins.length <= 1 || Math.abs(turnAngle(G, e, en)) < TURN_STRICT));
  let shift = 0;
  if (strict && simple) shift = G.lanes[e] - G.lanes[en];
  else if (strict && G.opp[e] > 0 && G.opp[en] > 0) shift = G.lanes[e] - G.lanes[en];   // two-way: divider-aligned
  else if (strict) {
    let div = 0, mer = 0;
    if (outs.length > 1) {
      const he = heading(G, e, true);
      const sd = new Map(outs.map((o) => [o, side(G, he, node, pointInto(G, o, 50, false))]));
      outs.sort((a, b) => (Math.abs(sd.get(a) - sd.get(b)) >= 1 ? sd.get(b) - sd.get(a) : turnAngle(G, e, b) - turnAngle(G, e, a)));
      for (const o of outs) { if (o === en) break; div += G.lanes[o]; }
      div = Math.max(0, Math.min(div, G.lanes[e] - G.lanes[en]));
    }
    if (ins.length > 1) {
      const hn = heading(G, en, false);
      const sd = new Map(ins.map((i) => [i, side(G, hn, node, pointInto(G, i, 50, true))]));
      ins.sort((a, b) => (Math.abs(sd.get(a) - sd.get(b)) >= 1 ? sd.get(b) - sd.get(a) : turnAngle(G, b, en) - turnAngle(G, a, en)));
      for (const i of ins) { if (i === e) break; mer += G.lanes[i]; }
      mer = Math.max(0, Math.min(mer, G.lanes[en] - G.lanes[e]));
    }
    shift = div - mer;
  }
  h = { strict, shift, scale: G.lanes[en] / G.lanes[e] };
  c.set(key, h);
  return h;
}
/**
 * Weave node (several in- and several out-edges): one lane numbering for the whole node, so all
 * hops through it are mutually consistent. In-edges are ordered right to left (by the side of
 * the point 50 m before the node, in the frame of the widest out-edge), out-edges likewise (in
 * the frame of the widest in-edge); lanes are numbered from the left (median) edge across each
 * stack: shift = e.lanes - en.lanes + offLeftIn(e) - offLeftOut(en), unclamped.
 */
const weaveCache = new WeakMap();
function weaveShift(G, node, e, en) {
  let wc = weaveCache.get(G);
  if (!wc) { wc = new Map(); weaveCache.set(G, wc); }
  let w = wc.get(node);
  if (!w) {
    const ins = [...G.inEdges(node)], outs = [...G.outEdges(node)];
    const widest = (l) => l.reduce((a, b) => (G.lanes[b] > G.lanes[a] || (G.lanes[b] === G.lanes[a] && b < a) ? b : a));
    const hIn = heading(G, widest(ins), true), hOut = heading(G, widest(outs), false);
    const so = new Map(outs.map((o) => [o, side(G, hIn, node, pointInto(G, o, 50, false))]));
    const si = new Map(ins.map((i) => [i, side(G, hOut, node, pointInto(G, i, 50, true))]));
    outs.sort((a, b) => (Math.abs(so.get(a) - so.get(b)) >= 1 ? so.get(b) - so.get(a) : a - b));
    ins.sort((a, b) => (Math.abs(si.get(a) - si.get(b)) >= 1 ? si.get(b) - si.get(a) : a - b));
    // offsets counted from the LEFT (median) side: lanes stay continuous on the left, like a plain continuation
    const offOut = new Map(), offIn = new Map();
    let acc = 0; for (let k = outs.length - 1; k >= 0; k--) { offOut.set(outs[k], acc); acc += G.lanes[outs[k]]; }
    acc = 0; for (let k = ins.length - 1; k >= 0; k--) { offIn.set(ins[k], acc); acc += G.lanes[ins[k]]; }
    w = { offOut, offIn };
    wc.set(node, w);
  }
  return G.lanes[e] - G.lanes[en] + w.offIn.get(e) - w.offOut.get(en);
}

/** Lane of en that lane l of e continues as (may be out of range at a strict lane end). */
export function mapLane(G, e, en, l) {
  const h = hopMap(G, e, en);
  return h.strict ? l - h.shift : Math.min(G.lanes[en] - 1, Math.floor((l + 0.5) * h.scale));
}
/** lat (m from the right carriageway edge) on en for a strict hop; null for junction hops. */
export function mapLatStrict(G, e, en, lat) {
  const h = hopMap(G, e, en);
  return h.strict ? lat - h.shift * LANE_W : null;
}
