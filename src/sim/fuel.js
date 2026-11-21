// Fuel model, gas-station network, refuel detours and pump queues.
//
// Consumption comes from the physics module (src/sim/physics/dynamics.js): engine power
// (traction power / driveline efficiency + accessories) through a BSFC map, with an idle
// floor. A sedan cruising at 100 km/h burns ~0.09 L/km (~11 km/L); stop-and-go is worse
// because every acceleration pays m a v at poor part-load efficiency.
//
// Refuel planning (sim.js calls planRefuel when a vehicle spawns and every
// fuelCheckInterval s): if fuel is below `lowFraction` of the tank, or the range at
// the planning rate (L/km at 80 km/h x 1.3) does not cover the remaining route +
// reserveKm, the vehicle is re-routed through the min-detour station: the brain's
// src/nav/stations.js routeViaStation (candidates = stations nearest the remaining
// route, min total time) driven through a Router adapter over the sim's own A*.
// The stop is the end of the route edge whose `to` is the station node. There the
// car leaves the road into the station lot (not in lanes / collisions), waits for a
// free pump (FIFO, pumps from data/stations.bin), pumps U(3, 5) min, fills to 100 %
// and re-enters the next route edge when its lane has room.
// Running dry: throttle is cut, the car coasts/brakes to a stop in its lane and is
// STRANDED (an obstacle that blocks its lane, like a crash scene) until it is towed
// after ~30 min (log-normal sigma 0.3); the trip fails as 'stranded'.

import { VEHICLE_PARAMS, resistance, fuelFlow } from './physics/index.js';
import { routeViaStation as brainRouteViaStation } from '../nav/stations.js';

export const FUEL_DEFAULTS = Object.freeze({
  lowFraction: 0.12,        // always refuel below this fraction of the tank
  reserveKm: 15,            // keep this much range beyond the remaining route
  planSpeed: 22.2,          // m/s (80 km/h) for range estimates
  planFactor: 1.3,          // margin on the planning consumption (stop-and-go)
  pumpMin: 180, pumpMax: 300,
  towMedian: 30 * 60, towSigma: 0.3,
  checkInterval: 60,        // s between refuel checks per vehicle
  maxCandidates: 6,
  minFillToSkip: 0.5,       // never plan a refuel above this fraction
});

/** Litres per km at steady speed v on a level road (physics model: road load / driveline -> BSFC). */
export function litresPerKm(typeIdx, v) {
  const P = VEHICLE_PARAMS[typeIdx] || VEHICLE_PARAMS[0];
  const Pw = resistance(P, v, 0) * v / 0.88 + P.accessoryW;
  return fuelFlow(P, Pw) * 1000 / Math.max(v, 0.1);
}

/** Adapter exposing the sim's A* (route.js makeRouter) with the src/nav Router interface routeViaStation needs. */
export function makeRouterAdapter(graph, route) {
  const toResult = (edges) => {
    if (!edges) return null;
    const nodes = new Array(edges.length + 1);
    let length = 0, time = 0;
    if (edges.length === 0) return null;
    nodes[0] = graph.edges[edges[0]].from;
    for (let k = 0; k < edges.length; k++) { const e = graph.edges[edges[k]]; nodes[k + 1] = e.to; length += e.length; time += e.length / e.speedLimit; }
    return { edges: Array.from(edges), nodes, length, time };
  };
  return {
    nx: graph.nx, nz: graph.nz,
    route(a, b) { return toResult(route(a, b)); },
    routeVia(list) {
      const all = [];
      for (let k = 0; k + 1 < list.length; k++) { const r = route(list[k], list[k + 1]); if (!r) return null; for (const e of r) all.push(e); }
      return toResult(all);
    },
  };
}

/** Station runtime state (pumps, FIFO queue) + planning helpers. */
export class StationNetwork {
  constructor(graph, stations, route, cfg = {}) {
    this.graph = graph;
    this.cfg = { ...FUEL_DEFAULTS, ...cfg };
    this.list = stations.map((s, k) => ({ ...s, idx: k }));
    this.byNode = new Map(); for (const s of this.list) this.byNode.set(s.node, s);
    this.busy = new Int32Array(this.list.length);       // pumps in use
    this.queue = this.list.map(() => []);               // waiting slots (FIFO)
    this.served = new Int32Array(this.list.length);
    this.router = makeRouterAdapter(graph, route);
    this.route = route;
    this.stats = { planned: 0, planFailed: 0, refuelled: 0, maxQueue: 0, waitSum: 0, waitN: 0 };
  }

  /**
   * Min-detour station route from `fromNode` to `goalNode`. Returns
   * {edges (Int32Array), station, stopIndex (route index of the edge ending at the station), detourTime} or null.
   */
  plan(fromNode, goalNode) {
    if (!this.list.length) return null;
    let r = null;
    try { r = brainRouteViaStation(this.graph, this.router, fromNode, goalNode, this.list, { maxCandidates: this.cfg.maxCandidates }); } catch (e) { r = null; }
    if (!r || r.stopEdgePos < 0) { this.stats.planFailed++; return null; }
    this.stats.planned++;
    return { edges: Int32Array.from(r.route.edges), station: r.station, stopIndex: r.stopEdgePos, detourTime: r.detourTime };
  }

  /** Does this vehicle need fuel for `remainingM` metres? */
  needsFuel(typeIdx, fuel, tank, remainingM) {
    const f = fuel / tank;
    if (f >= this.cfg.minFillToSkip) return false;
    if (f < this.cfg.lowFraction) return true;
    const lpk = litresPerKm(typeIdx, this.cfg.planSpeed) * this.cfg.planFactor;
    return fuel / lpk < remainingM / 1000 + this.cfg.reserveKm;
  }
}
