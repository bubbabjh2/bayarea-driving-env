// Junction lane mapping: the SIM's rule (src/sim/roadGraph.js hopMap / mapLane / mapLat)
// is the single source of truth for the brain too (guidance lane ends, the driver's
// target-lane bookkeeping, routeSim's lateral frame at edge hops). Read-only import.
//
// The sim functions need a SimGraph (CSR adjacency, edge.idx / reverse). A src/nav Graph
// is adapted once through fromNavGraph (edge ids preserved) and the adapter is cached on it.
//
//   laneMapper(graph) -> { mapLane(e, en, l), mapLat(e, en, lat), hop(e, en) }   (e, en: edges of `graph`)
//   laneEndAhead(graph, edgeIds, pos, sOnEdge, lane, {maxDist, stopAt}) -> {distance, side, mergeCap} | null  (src/sim/laneEnds.js)
//     follows `lane` of route edge `pos` through up to maxHops hops while the distance to
//     the hop is < maxDist (the sim: 4 hops / laneEndLookahead 450 m); distance = to the end
//     of the lane: with src/geom lane geometry (edge.LG) the lane narrows over the last 90 m
//     BEFORE the node (distance = node - 90 m), else it tapers 40 m past it (node + 40 m);
//     side = +1 surviving lanes are to the LEFT (merge left), -1 right -- exactly sim.js _laneEnd.
import { hopMap, mapLane as simMapLane, mapLat as simMapLat, fromNavGraph, LANE_WIDTH } from '../sim/roadGraph.js';
import { TAPER } from '../sim/vehicle.js';
import { laneEndAhead as simLaneEndAhead } from '../sim/laneEnds.js';

export { TAPER, LANE_WIDTH };
export const KINKS = false; // see Guidance.nextKink
export const SIM_LANE_END_HOPS = 4;          // sim.js _laneEnd (layout v4)
export const SIM_LANE_END_LOOKAHEAD = 450;   // SIM_DEFAULTS.laneEndLookahead
export const LANE_END_TAPER = 0;             // sim.js LANE_END_TAPER (Oct 20: src/geom paves every lane to its node; was 90)

const cache = new WeakMap();

export function laneMapper(graph) {
  let m = cache.get(graph);
  if (m) return m;
  const sg = graph.outStart ? graph : (graph._simAdapter || (graph._simAdapter = fromNavGraph(graph)));
  const E = (e) => (sg === graph ? e : sg.edges[e.id]);
  m = {
    simGraph: sg,
    hop: (e, en) => hopMap(sg, E(e), E(en)),
    mapLane: (e, en, l) => simMapLane(sg, E(e), E(en), l),
    mapLat: (e, en, lat) => simMapLat(sg, E(e), E(en), lat),
  };
  cache.set(graph, m);
  return m;
}

const _out = { dist: 1e9, side: 0, mergeCap: 1e9 };
/**
 * The SIM's own lane-end look-ahead (src/sim/laneEnds.js, the single source of truth since Oct 20):
 * edgeIds = the route, pos = index of my edge in it, sOnEdge = m along it, lane = my lane.
 * Returns {distance (m to the node where my lane ends; lanes are paved to the node), side
 * (+1 merge left / -1 right), mergeCap (m/s, Infinity none)} or null.
 */
export function laneEndAhead(graph, edgeIds, pos, sOnEdge, lane, { maxDist = SIM_LANE_END_LOOKAHEAD, stopAt = -1 } = {}) {
  const sg = laneMapper(graph).simGraph;
  const e = sg.edges[edgeIds[pos]];
  if (lane < 0 || lane >= e.lanes) return null;
  const o = simLaneEndAhead(sg, edgeIds, pos, sOnEdge, lane, { lookahead: maxDist, stopIndex: stopAt }, _out);
  if (!(o.dist < 1e8)) return null;
  return { distance: o.dist, side: o.side, mergeCap: o.mergeCap < 1e8 ? o.mergeCap : Infinity };
}

/**
 * Next junction heading kink on a route (pure; mirrors src/sim/vehicle.js: across a NON-strict hop with
 * |turn| < 0.5 rad the sim keeps the world heading, so headingRel jumps by the turn angle).
 * edgeIds = route, pos = my edge index in it, sOnEdge = m along it. Returns {distance, angle (+ = right)} | null.
 * Same values as Guidance.nextKink (eyes inputs 63, 64); for brainAdapter / the GPU path.
 */
export function kinkAhead(graph, edgeIds, pos, sOnEdge, maxDist = 80) {
  if (!KINKS) return null; // disabled: the sim keeps headingRel across hops (Oct 20)
  const M = laneMapper(graph), sg = M.simGraph;
  let dist = sg.edges[edgeIds[pos]].length - sOnEdge;
  for (let i = pos; i + 1 < edgeIds.length && dist <= maxDist; i++) {
    const e = sg.edges[edgeIds[i]], en = sg.edges[edgeIds[i + 1]];
    if (!hopMap(sg, e, en).strict) {
      let a = en.headingStart - e.headingEnd; while (a > Math.PI) a -= 2 * Math.PI; while (a < -Math.PI) a += 2 * Math.PI;
      if (Math.abs(a) < 0.5 && a !== 0 && dist >= 0) return { distance: dist, angle: a };
    }
    dist += en.length;
  }
  return null;
}
