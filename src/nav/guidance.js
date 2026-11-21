// "Google Maps" guidance layer: turn a route (list of directed edge indices)
// into turn-by-turn maneuvers and answer live queries by distance along route.
//
// Maneuver types: continue | exit | merge | turn-left | turn-right |
//                 keep-left | keep-right | arrive | refuel
// Stops: `opts.stops = [{node, type: 'refuel', station}]` add a mandatory stop
// maneuver where the route passes `node` ("stop for gas at the station on the
// right"). at() reports it as the next maneuver (stop-at-node semantics like
// arrive) until completeStop(maneuver) is called, then guidance continues.
// Detection at each node between consecutive route edges e1 -> e2:
//   exit       : motorway/trunk -> *_link
//   merge      : *_link -> motorway/trunk (or any non-link major road)
//   turn-l/r   : |heading change| > 30 deg (sign per geom.js convention);
//                bends inside a single way with no alternative are ignored, and
//                freeway<->freeway/ramp geometry never counts as a turn (forks do)
//   keep-l/r   : fork: another outgoing edge of rank >= e2's rank also leaves
//                within +-45 deg of straight; side = which way e2 goes vs it;
//                same-side keeps within 400 m, and opposite-side keeps within
//                300 m toward the same road, are coalesced into one step
//   continue   : road ref/name changes without a turn (dropped if < 200 m
//                after another maneuver)
//   arrive     : end of route
// Lane hints: 'right'/'left' for upcoming exit/turn/keep maneuvers once within
// laneHintHorizon (default 2000 m); 'none' otherwise.
import { wrapAngle, turnSide, rad, MS_TO_MPH } from './geom.js';
import { CLASS_RANK } from './graph.js';
import { laneEndAhead, laneMapper } from './laneMap.js';
/** The sim keeps headingRel across hops (src/sim/vehicle.js, Oct 20): no junction heading kinks. */
export const KINKS_ENABLED = false;

export const MANEUVER_TYPES = ['continue', 'exit', 'merge', 'turn-left', 'turn-right', 'keep-left', 'keep-right', 'arrive', 'refuel'];
const TURN_THRESHOLD = rad(30);
const FORK_CONE = rad(45);
const MAJOR = new Set(['motorway', 'trunk']);
const rk = (e) => e.navRank ?? e.rank; // SimGraph edges carry the nav ordering as navRank
const isMajorOrLink = (e) => MAJOR.has(e.highway) || e.highway === 'motorway_link' || e.highway === 'trunk_link';

export function formatDistance(m) {
  const mi = m / 1609.344;
  if (mi >= 0.2) return `${mi.toFixed(1)} mi`;
  const ft = Math.max(50, Math.round(m * 3.28084 / 50) * 50);
  return `${ft} ft`;
}

function roadLabel(e) {
  if (e.ref && e.name) return `${e.ref} (${e.name})`;
  return e.ref || e.name || (e.isLink ? 'the ramp' : 'the road');
}

/** Follow the route from edge position i forward until a non-link edge; return it (or null). */
function destinationAfterLinks(edges, i) {
  for (let k = i; k < edges.length; k++) if (!edges[k].isLink) return edges[k];
  return null;
}

export class Guidance {
  /**
   * @param {import('./graph.js').Graph} graph
   * @param {number[]} routeEdgeIds   directed edge ids in travel order
   * @param {object} [opts] {laneHintHorizon, stops: [{node, type, station}]}
   */
  constructor(graph, routeEdgeIds, opts = {}) {
    this.graph = graph;
    this.edgeIds = routeEdgeIds.slice();
    this.edges = this.edgeIds.map((id) => graph.edges[id]);
    this.laneHintHorizon = opts.laneHintHorizon ?? 2000;
    const n = this.edges.length;
    this.startDist = new Float64Array(n + 1);
    this.startTime = new Float64Array(n + 1);
    for (let i = 0; i < n; i++) {
      const e = this.edges[i];
      this.startDist[i + 1] = this.startDist[i] + e.length;
      this.startTime[i + 1] = this.startTime[i] + e.length / e.speed;
    }
    this.totalLength = this.startDist[n];
    this.totalTime = this.startTime[n];
    this.stops = opts.stops || [];
    this.maneuvers = this._buildManeuvers();
    this.stopDwell = opts.stopDwell ?? 0;
  }

  /**
   * Where does lane `lane` of the route edge at distance d end? Follows it through the
   * junctions with the SIM's lane mapping (nav/laneMap.js = src/sim/roadGraph.js mapLane):
   * {distance (incl. the 40 m taper), side (+1 merge left, -1 merge right)} | null.
   * Defaults = the sim's own lookahead (3 hops, 250 m); an uncompleted refuel stop ends the scan.
   */
  laneEndAhead(d, lane, opts = {}) {
    const i = this.edgePosAt(d);
    let stopAt = -1;
    for (const m of this.maneuvers) if (m.stop && !m.done) { stopAt = m.edgePos - 1; break; }
    if (stopAt === i) return null;
    return laneEndAhead(this.graph, this.edgeIds, i, d - this.startDist[i], lane, { ...opts, stopAt });
  }

  /**
   * Next heading kink. DISABLED (KINKS_ENABLED = false): since Oct 20 ~02:00 the sim keeps headingRel across
   * every hop (the road frame bends with the road at nodes too), so there is no kink; inputs 63/64 stay
   * in the layout and always read 'none'. (Before that, the sim kept the WORLD heading across a
   * non-strict hop with |turn| < 0.5 rad and this returned that turn.)
   * Returns {distance (m to the node), angle d (rad, + = right turn)} for the first such hop within
   * maxDist, or null. Strict hops and sharper turns keep the lane-relative heading (no kink).
   */
  nextKink(d, maxDist = 80) {
    if (!KINKS_ENABLED) return null;
    if (!this._kinks) {
      const M = laneMapper(this.graph), n = this.edges.length;
      this._kinks = new Float32Array(n);
      for (let i = 0; i + 1 < n; i++) {
        const e = this.edges[i], en = this.edges[i + 1];
        if (M.hop(e, en).strict) continue;
        const a = wrapAngle(en.headingStart - e.headingEnd);
        if (Math.abs(a) < 0.5) this._kinks[i] = a;
      }
    }
    for (let i = this.edgePosAt(d); i + 1 < this.edges.length; i++) {
      const dist = this.startDist[i + 1] - d;
      if (dist > maxDist) return null;
      if (dist >= 0 && this._kinks[i] !== 0) return { distance: dist, angle: this._kinks[i] };
    }
    return null;
  }

  /** Mark a stop maneuver as completed so at() moves on to the next one. */
  completeStop(m) { m.done = true; }

  _buildManeuvers() {
    const edges = this.edges, g = this.graph;
    const list = [];
    for (let i = 0; i + 1 < edges.length; i++) {
      const e1 = edges[i], e2 = edges[i + 1];
      const node = e1.to;
      const delta = wrapAngle(e2.headingStart - e1.headingEnd);
      // alternatives leaving this node (excluding the chosen edge and the u-turn back along e1)
      const others = g.out[node].map((id) => g.edges[id]).filter((o) => o !== e2 && o.to !== e1.from);
      // a bend inside one way with no alternative is not a maneuver
      if (others.length === 0 && e1.srcId === e2.srcId) continue;
      let type = null, side = null, toEdge = null;
      if (!e1.isLink && e2.isLink && MAJOR.has(e1.highway)) {
        type = 'exit';
        const main = others.find((o) => !o.isLink) || null;
        side = main ? turnSide(wrapAngle(e2.headingStart - main.headingStart)) : turnSide(delta);
        if (side === 'straight') side = 'right';
        toEdge = destinationAfterLinks(edges, i + 1);
      } else if (e1.isLink && !e2.isLink && (MAJOR.has(e2.highway) || rk(e2) >= CLASS_RANK.primary)) {
        type = 'merge';
        const mainIn = g.in[node].map((id) => g.edges[id]).find((o) => o !== e1 && !o.isLink) || null;
        // ramp coming from the right means the ramp heading is rotated left of the mainline
        side = mainIn ? (turnSide(wrapAngle(e1.headingEnd - mainIn.headingEnd)) === 'left' ? 'right' : 'left') : 'right';
        toEdge = e2;
      } else if (Math.abs(delta) > TURN_THRESHOLD && !(isMajorOrLink(e1) && isMajorOrLink(e2))) {
        side = turnSide(delta);
        type = side === 'left' ? 'turn-left' : 'turn-right';
        toEdge = e2;
      } else {
        const fork = others.find((o) => rk(o) >= rk(e2) && Math.abs(wrapAngle(o.headingStart - e1.headingEnd)) < FORK_CONE);
        if (fork) {
          side = turnSide(wrapAngle(e2.headingStart - fork.headingStart));
          if (side === 'straight') side = turnSide(delta) === 'left' ? 'left' : 'right';
          type = side === 'left' ? 'keep-left' : 'keep-right';
          toEdge = e2.isLink ? destinationAfterLinks(edges, i + 1) || e2 : e2;
        } else if (e1.isLink === e2.isLink && (e2.ref || e2.name) && (e2.ref !== e1.ref || (!e2.ref && e2.name !== e1.name))) {
          type = 'continue';
          toEdge = e2;
        }
      }
      if (!type) continue;
      const atDistance = this.startDist[i + 1];
      const prev = list[list.length - 1];
      if (type === 'continue' && prev && atDistance - prev.atDistance < 200) continue;
      list.push(this._mk(type, atDistance, i + 1, node, side, e1, e2, toEdge));
    }
    // coalesce duplicate keep-left/right steps (ramp spaghetti, split carriageways):
    //  * same side within 400 m -> one step (toward the later destination)
    //  * opposite sides within 300 m toward the SAME road -> one step; the dropped
    //    fork is remembered as `a.then` ('left'|'right') for lane-level consumers
    const isKeep = (m) => m.type === 'keep-left' || m.type === 'keep-right';
    for (let i = 1; i < list.length; i++) {
      const a = list[i - 1], b = list[i];
      if (!isKeep(a) || !isKeep(b)) continue;
      const gap = b.atDistance - a.atDistance;
      if (b.type === a.type && gap < 400) {
        if (b.toRef || b.toName) { a.toRef = b.toRef; a.toName = b.toName; a.instruction = b.instruction; }
        list.splice(i, 1); i--;
      } else if (b.type !== a.type && gap < 300 && (b.toRef || b.toName) && b.toRef === a.toRef && b.toName === a.toName) {
        a.then = b.side;
        list.splice(i, 1); i--;
      }
    }
    // mandatory stops (refuel): inserted where the route first reaches the stop node
    for (const stop of this.stops) {
      let pos = -1;
      for (let i = 0; i < edges.length; i++) if (edges[i].to === stop.node) { pos = i; break; }
      if (pos < 0) continue;
      const m = this._mk(stop.type || 'refuel', this.startDist[pos + 1], pos + 1, stop.node, 'right', edges[pos], edges[pos + 1] || edges[pos], null);
      m.station = stop.station || null; m.stop = true; m.done = false;
      let k = 0; while (k < list.length && list[k].atDistance < m.atDistance) k++; // a stop goes BEFORE a turn at the same node
      list.splice(k, 0, m);
    }
    const last = edges[edges.length - 1];
    list.push(this._mk('arrive', this.totalLength, edges.length - 1, last ? last.to : -1, null, last, last, null));
    // fill in "In X mi," prefixes relative to the previous maneuver (step-list style)
    let prevD = 0;
    for (const m of list) { m.text = `In ${formatDistance(m.atDistance - prevD)}, ${m.instruction}`; prevD = m.atDistance; }
    return list;
  }

  _mk(type, atDistance, edgePos, nodeIndex, side, e1, e2, toEdge) {
    const toRef = toEdge ? toEdge.ref : '';
    const toName = toEdge ? toEdge.name : '';
    const toLabel = toEdge ? roadLabel(toEdge) : '';
    let instruction;
    switch (type) {
      case 'exit': instruction = `take the ${side} exit${toLabel ? ` toward ${toLabel}` : ''}`; break;
      case 'merge': instruction = `merge onto ${toLabel}${side ? ` (from the ${side})` : ''}`; break;
      case 'turn-left': instruction = `turn left onto ${toLabel}`; break;
      case 'turn-right': instruction = `turn right onto ${toLabel}`; break;
      case 'keep-left': instruction = `keep left${toLabel ? ` toward ${toLabel}` : ''}`; break;
      case 'keep-right': instruction = `keep right${toLabel ? ` toward ${toLabel}` : ''}`; break;
      case 'continue': instruction = `continue onto ${toLabel}`; break;
      case 'arrive': instruction = 'arrive at your destination'; break;
      case 'refuel': instruction = 'stop for gas at the station on the right'; break;
    }
    return {
      type, atDistance, edgePos, nodeIndex, side, instruction, text: '',
      roadRef: e1 ? e1.ref : '', roadName: e1 ? e1.name : '', toRef, toName,
    };
  }

  /** Position of a graph edge id in this route (first occurrence), or -1. */
  edgePosOf(edgeId) {
    if (!this._edgePos) { this._edgePos = new Map(); for (let i = this.edges.length - 1; i >= 0; i--) this._edgePos.set(this.edges[i].id, i); }
    const p = this._edgePos.get(edgeId);
    return p === undefined ? -1 : p;
  }

  /** Index of the route edge containing distance d. */
  edgePosAt(d) {
    const sd = this.startDist;
    let lo = 0, hi = this.edges.length - 1;
    if (d <= 0) return 0;
    if (d >= this.totalLength) return hi;
    while (lo < hi) { const m = (lo + hi + 1) >> 1; if (sd[m] <= d) lo = m; else hi = m - 1; }
    return lo;
  }


  /**
   * Track progress: project (x, z) onto the route near the previous distance
   * prevD (searching the current edge and `window` edges ahead). Returns
   * {d, lateral (abs m from the polyline), onRoute}. Cheap: a few segments.
   */
  advance(prevD, x, z, { window = 3, maxOff = 40 } = {}) {
    const i0 = this.edgePosAt(prevD);
    const i1 = Math.min(this.edges.length - 1, i0 + window);
    let bestD = prevD, bestLat = Infinity;
    for (let i = i0; i <= i1; i++) {
      const e = this.edges[i], p = e.points, cum = e.cum;
      for (let k = 0; k + 1 < cum.length; k++) {
        const x0 = p[2 * k], z0 = p[2 * k + 1], x1 = p[2 * k + 2], z1 = p[2 * k + 3];
        const sx = x1 - x0, sz = z1 - z0, l2 = sx * sx + sz * sz || 1e-9;
        let t = ((x - x0) * sx + (z - z0) * sz) / l2;
        if (t < 0) t = 0; else if (t > 1) t = 1;
        const px = x0 + sx * t, pz = z0 + sz * t;
        const lat = Math.hypot(x - px, z - pz);
        const d = this.startDist[i] + cum[k] + (cum[k + 1] - cum[k]) * t;
        // prefer the closest; on near-ties prefer not jumping backwards
        if (lat < bestLat - 0.05 || (Math.abs(lat - bestLat) <= 0.05 && Math.abs(d - prevD) < Math.abs(bestD - prevD))) { bestLat = lat; bestD = d; }
      }
    }
    if (bestD < prevD - 50) bestD = prevD; // never leap backwards (overlapping geometry)
    return { d: bestD, lateral: bestLat, onRoute: bestLat <= maxOff };
  }

  /** World pose [x, z, heading] at distance d along the route. */
  poseAt(d) {
    const i = this.edgePosAt(d);
    return this.graph.pointOnEdge(this.edges[i], d - this.startDist[i]);
  }

  /**
   * Live query. Returns a fresh object (cheap, ~10 fields) — call once per
   * decision tick, not per frame per sensor.
   */
  at(d) {
    const ms = this.maneuvers;
    // first maneuver strictly ahead (allow a 1 m grace so we don't skip at the node)
    let lo = 0, hi = ms.length - 1;
    while (lo < hi) { const m = (lo + hi) >> 1; if (ms[m].atDistance < d - 1) lo = m + 1; else hi = m; }
    // a completed stop (refuel done) is skipped; an uncompleted stop we have passed by a few metres still counts
    while (lo < ms.length - 1 && ms[lo].done) lo++;
    for (let k = lo - 1; k >= 0 && d - ms[k].atDistance < 15; k--) if (ms[k].stop && !ms[k].done) { lo = k; break; } // (a turn may share the stop's node)
    const next = ms[lo];
    let fi = lo + 1; while (fi < ms.length - 1 && ms[fi].done) fi++;
    const following = fi < ms.length ? ms[fi] : null;
    const i = this.edgePosAt(d);
    const e = this.edges[i];
    const distanceToNext = Math.max(0, next.atDistance - d);
    let laneHint = 'none';
    if (next.side && next.type !== 'merge' && distanceToNext <= this.laneHintHorizon) laneHint = next.side;
    const remainingDistance = Math.max(0, this.totalLength - d);
    const tAt = this.startTime[i] + Math.min(e.length, Math.max(0, d - this.startDist[i])) / e.speed;
    return {
      nextManeuver: next,
      following, // the maneuver after next (so drivers can brake for a stop just past a turn)
      distanceToNext,
      liveText: `In ${formatDistance(distanceToNext)}, ${next.instruction}`,
      currentRoad: { ref: e.ref, name: e.name, highway: e.highway, isLink: e.isLink, lanes: e.lanes, speed: e.speed, speedMph: e.speed * MS_TO_MPH, edgeId: e.id, edgePos: i, distanceOnEdge: d - this.startDist[i] },
      laneHint,
      remainingDistance,
      etaSeconds: Math.max(0, this.totalTime - tAt),
      arrived: d >= this.totalLength - 0.5,
    };
  }

  /** Human-readable step list. */
  describe() {
    return this.maneuvers.map((m) => `${(m.atDistance / 1609.344).toFixed(2).padStart(6)} mi  ${m.type.padEnd(10)} ${m.text}`).join('\n');
  }
}

export function buildGuidance(graph, routeEdgeIds, opts) { return new Guidance(graph, routeEdgeIds, opts); }
