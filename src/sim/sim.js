// Simulation core (CPU reference): clock, spawning, car following / driver
// dispatch, steering-only motion, junction yielding and zipper merging, fuel and
// gas stations, crashes and statistics. The state layout and the exact per-step
// order are documented in src/sim/LAYOUT.md (the GPU agent mirrors it).
//
// Step order (fixed dt, default 0.05 s; headless runs use 0.1 s):
//   1. clock: time += dt, stepCount++
//   2. spawn: trips whose departure time has passed (+ cars leaving gas stations)
//   3. rebuild spatial hash + per-(edge, lane) sorted lists from last step's poses
//   4. leaders: gap/leader per driving car over every lane its footprint occupies
//      (same edge, across edges with junction lane alignment, lane-end tapers,
//      zipper merges, crash-scene obstacles)
//   5. collisions: footprint overlap (longitudinal + lateral) with the cars ahead in
//      each occupied lane list, cross-edge negative gaps
//   6. drivers: the external policy (driverFactory / env.js 'external') -> {steer, gas, brake};
//      cars without a driver get opts.defaultAction (env-only build: no built-in driver)
//      written to the per-vehicle reaction-time ring
//   7. integrate: delayed controls -> steer angle, accel, speed, bicycle model (s, lat,
//      headingRel), edge hops, fuel burn; off-road crashes, arrivals, refuel stops,
//      stranding, stuck removal, periodic refuel planning
//   8. stochastic accidents
//   9. scenes: crash / stranded scene timers (active -> shoulder/squeeze -> cleared), pumps

import { VehicleStore, STATE_FREE, STATE_DRIVING, STATE_CRASHED, STATE_REMOVED, STATE_REFUELING,
  FLAG_LEARNER, FLAG_COLLIDED, FLAG_BRAIN, FLAG_SHOULDER, FLAG_SPEEDING, FLAG_STRANDED, FLAG_REFUEL, FLAG_DISABLED, CTRL_RING, TAPER, connectorPoint } from './vehicle.js';
import { hopMap, mapLane, edgePoint } from './roadGraph.js';
import { SpatialHash, LaneIndex } from './spatial.js';
import { WorldQuery } from './worldQuery.js';
import { severityOf, clearanceFor, incidentTimeline, DISABLE_KMH, RUBBERNECK } from './crashModel.js';
import { obbContact, sweptContact, resolveContact, separate, PHYS } from './physics/index.js';
import { generatePopulation, tripsOf, synthAxis } from './population.js';
import { makeRng } from './rng.js';
import { crashShare } from './reward.js';
import { LANE_WIDTH, laneW, buildGraph } from './roadGraph.js';
import { makeSynthGraph } from './synthGraph.js';
import { makeRouter } from './route.js';
import { RouteStore } from './routeStore.js';
import { StationNetwork, FUEL_DEFAULTS } from './fuel.js';
import { CTRL, LIGHT, MOVE, movementOf } from './controls.js';
import { laneEndAhead } from './laneEnds.js';
import { VEHICLE_TYPES } from './vehicleTypes.js';

export const SIM_DEFAULTS = Object.freeze({
  dt: 0.05,
  startTime: 5 * 3600,
  endTime: 24 * 3600,
  speed: 1,                     // clock multiplier for advance() (1x .. 600x)
  maxStepsPerFrame: 400,
  learnerFraction: 0.1,         // share of agents with an online learner (brain drivers only)
  lodRadius: Infinity,          // brain drivers only within this distance of sim.focus
  brainEvery: 1,                // brain drivers step every N sim steps (staggered)
  shoulder: 1.0,                // m of paved shoulder beyond the outer lane before the barrier (static collider), surface roads
  freewayShoulder: 3.0,         // m: freeways / links (10 ft outside shoulder)
  oppositeMargin: 0.5,          // two-way roads: the far curb is this far beyond the opposing lanes
  pairCooldown: 30,             // (unused since v8: a pair of cars counts one crash; a later disabling contact still counts)
  touchKmh: 1.0,                // contacts whose largest delta-v is below this are touches (physics still resolves them), not crash events
  layerGap: 2.5,                // m of elevation difference beyond which bodies cannot touch (bridges / overpasses)
  stuckTimeout: 30 * 60,        // s stopped before a car is LABELLED stuck (diagnostic only: waiting is normal traffic, never a failure)
  passWrecks: false,            // opposing-lane passing of wrecks on two-lane roads (off: not required; see _opposing)
  spawnBlockedTimeout: Infinity, // s a departure waits for room before it is forced in (Infinity: never - a car only enters where its body fits)
  eventLimit: 20000,
  cellSize: 25,
  tableBits: 18,
  leaderLookahead: 250,
  mergeLookahead: 80,           // m before a node where zipper-merge leaders are considered
  laneEndLookahead: 450,
  controlLookahead: 250,        // m: traffic-control observation range        // m: lane-end warnings look this far ahead across edges
  reroute: true,                // drivers who see a closed road ahead on their route re-plan around it
  rerouteSight: 2000,           // m of route ahead in which a closure is known (nav-app traffic layer)
  squeezeDelay: 15 * 60,        // s after which a full closure lets trapped traffic squeeze past
  squeezeSpeed: 3,              // m/s while squeezing past a scene
  densityInterval: 60,          // s between freeway density / LOS samples
  pathSight: 0,                 // body-on-my-path perception (docs/scale-crashes.md): 0 = off (before 2026-10-25); 1 = sample my route path to the stopping sight distance, any body on it is a leader
  deadlockBreak: 0,             // 1 = the lowest slot of a closed wait loop (> 20 s standing) creeps on (docs/scale-crashes.md); 0 = off
  spawnLaneRoute: 0,            // 1 = a departing car takes a lane that continues on its route when one has room (docs/scale-crashes.md); 0 = lowest lane with room
  junctionSight: 0,             // junction perception version (docs/scale-crashes.md): 0 = only the front car of each conflicting lane on its own approach edge (before 2026-10-25); 1 = the first 3 cars of the lane, plus cars upstream of a short approach (clustered junctions)
  actionMode: 'position',       // steering command: 'position' (layout v9: target wheel position in [-1,1]) | 'rate' (legacy steerRate checkpoints)
  trafficMode: 'full',          // 'full' | 'sparse' (full + static wrecks) | 'empty' (agents ignore each other)
  fuel: true,
  seed: 1,
  tripLog: true,
});

/** Road direction at s on edge e into out.hx/hz (unit). */
function edgePointDir(e, s, out, seg) { return e.LG ? e.LG.offsetPoint(e.idx, s, 0, 0, out) : edgePoint(e, s, 0, out, seg); }
/** Direction (hx, hz) of slot i's road frame at its position: its junction connector, else its edge. */
/** Room (m) a merging car needs behind the car it lets go first: standstill gap + 1 s headway. */
function cfgGap(v) { return 2 + 1.0 * v; }
function frameDir(g, vs, i, out) {
  const m = vs.conn[i];
  if (m >= 0 && g.J) return connectorPoint(g.J, m, vs.s[i] - g.J.mF[4 * m], out);
  return edgePointDir(g.edges[vs.edge[i]], vs.s[i], out, vs.seg[i]);
}
/** End of slot i's current frame (connector end in in-edge coordinates, else the edge length). */
function frameEnd(g, vs, i) {
  const m = vs.conn[i];
  return m >= 0 && g.J ? g.J.mF[4 * m] + g.J.mLen[m] : g.edges[vs.edge[i]].length;
}
/** Lane-end wall for a car that is already merging: the node itself (the taper end). */
function prevEndNode(dEnd, vs, i, sim) { return dEnd + (sim.graph.laneGeom ? LANE_END_TAPER : 0); }
function clockOf(t) { const h = Math.floor(t / 3600), m = Math.floor(t / 60) % 60; return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}`; }
const MERGE_ROOM = 15;
const SHOULDER_ROADS = new Set(['motorway', 'trunk', 'motorway_link', 'trunk_link']);
const PULLOVER_ROADS = new Set(['motorway', 'trunk']);   // mainline shoulders wide enough to pull a car fully off the lanes
const LANE_END_TAPER = 0;  // src/geom paves every lane to its node (the run-out is beyond it): a car must have left an ending lane by the node
// Junction-query config used by sim.ctx (yieldRequired etc.; values of the original built-in driver's config).
export const CTX_DEFAULTS = Object.freeze({ s0: 2.0, speedFactorMin: 0.85, speedFactorSpan: 0.30, laneCheckInterval: 1.0, yieldLook: 40, stopMargin: 1.5, prepBase: 250, prepPerLane: 250 });
// Command for a car that has no driver (env-only build: there is no built-in driver policy).
export const DEFAULT_ACTION = Object.freeze({ steer: 0, gas: 0, brake: 0.3 });
/**
 * Scripted obstacle prop (placeObstacles kinds 'slow' / 'braking' only; NOT a traffic policy): a proportional
 * lane-centre + speed hold so a moving obstacle body stays in its lane. Position-mode steer in [-1, 1].
 */
function propControl(sim, i, vT) {
  const vs = sim.vs, e = sim.graph.edges[vs.edge[i]], lw = laneW(e, vs.s[i]);
  const off = vs.lat[i] - (vs.lane[i] + 0.5) * lw;   // + = left of the lane centre
  const steer = Math.max(-1, Math.min(1, (-0.08 * off - 1.2 * vs.headingRel[i]) / Math.max(0.1, vs.maxSteer[i])));
  const dv = Math.min(vT, e.speedLimit) - vs.speed[i];
  return { steer, gas: dv > 0 ? Math.min(1, 0.3 * dv) : 0, brake: dv < -1 ? Math.min(1, -0.1 * dv) : 0 };
}
export const FAIL = Object.freeze({ PENDING: 0, ARRIVED: 1, CRASH: 2, STRANDED: 3, STUCK: 4, NOROUTE: 5 });
const OUTCOME_NAME = ['pending', 'arrived', 'crash', 'stranded', 'stuck', 'noRoute'];

export class Simulation {
  /**
   * @param graph SimGraph (roadGraph.js buildGraph / fromRoadSet / fromNavGraph)
   * @param opts  SIM_DEFAULTS plus
   *   pop: a generatePopulation() result (preferred: carries the packed routes), or
   *   commuters: commuter list (routes are packed here), or population: generatePopulation options
   *   stations: [{node, pumps, x, z, ...}] (data.js stationsOnGraph); fuelCfg: FUEL_DEFAULTS overrides
   *   driverFactory(graph, startNode, goalNode, pose, extra) -> {step(dt, worldQuery)} (env.js 'external')
   *   defaultAction: {steer, gas, brake} for cars without a driver (default: hold the brake)
   *   accident: ACCIDENT_DEFAULTS overrides; idm: junction-query config overrides (yieldLook, ...)
   */
  constructor(graph, opts = {}) {
    this.graph = graph;
    this.opts = { ...SIM_DEFAULTS, ...opts };
    this.idmCfg = { ...CTX_DEFAULTS, ...(opts.idm || {}) };
    this.defaultAction = { ...DEFAULT_ACTION, ...(opts.defaultAction || {}) };
    this.rng = makeRng(this.opts.seed);
    this.time = this.opts.startTime;
    this.speed = this.opts.speed;
    this.dt = this.opts.dt;
    this.stepCount = 0;
    this.accumulator = 0;
    this.lagging = false;
    this.focus = { x: 0, z: 0 };
    this.lodRadius = this.opts.lodRadius;
    this.driverFactory = opts.driverFactory || null;
    this.empty = this.opts.trafficMode === 'empty';

    // population + packed routes
    // streaming mode (stream.js): a CommuterTable + routeFor(id, leg, from, to) -> Int32Array | null | undefined (not ready)
    this.table = opts.table || null;
    this.routeFor = opts.routeFor || null;
    let pop = opts.pop || null;
    if (this.table) this.commuters = new Array(this.table.n);   // filled at departure, dropped at trip end
    else {
      if (!pop && !opts.commuters) pop = generatePopulation(graph, { seed: this.opts.seed, ...(opts.population || {}) });
      if (pop) { this.population = pop; this.commuters = pop.commuters; }
      else this.commuters = opts.commuters;
    }
    this.routeStore = new RouteStore(this.table ? null : pop ? pop.routes : packCommuterRoutes(this.commuters));
    this.router = opts.router || makeRouter(graph, { weight: 1.25 });
    const nC = this.commuters.length;
    this.learners = new Uint8Array(nC);
    for (let i = 0; i < nC; i++) this.learners[i] = this.rng.random() < this.opts.learnerFraction ? 1 : 0;
    this.outcome = new Uint8Array(nC);            // outbound (leg 0) outcome, FAIL.*
    this.outcomeBack = new Uint8Array(nC);        // return leg outcome
    this.commuterFuel = new Float32Array(nC);     // litres carried between legs
    for (let i = 0; i < nC; i++) {
      const T = this.table ? VEHICLE_TYPES[this.table.vehicleType[i]] : VEHICLE_TYPES[this.commuters[i].vehicleType] || VEHICLE_TYPES[0];
      this.commuterFuel[i] = (this.table ? this.table.fuelFill[i] : (this.commuters[i].fuelFill ?? 1)) * T.tankL;
    }
    this.trips = this.table ? [] : tripsOf(this.commuters);
    this.tripPtr = 0;
    this.tablePtr = this.table ? this.table.indexAt(this.opts.startTime - 3600) : 0;
    this.slotPool = opts.slotPool || 0;            // fixed vehicle-slot pool (GPU-style); 0 = grow on demand
    this.pendingSpawns = [];

    // stations
    this.stations = this.opts.fuel && opts.stations && opts.stations.length ? new StationNetwork(graph, opts.stations, this.router, opts.fuelCfg) : null;
    this.fuelCfg = { ...FUEL_DEFAULTS, ...(opts.fuelCfg || {}) };
    this.lot = [];
    this._pairEvent = new Map();
    this._stuckLabelled = new Map();
    this._connCars = new Map(); this._passers = new Map(); this._conflictWith = new Map(); this._offLane = new Map();
    this.controls = opts.controls || null;         // TrafficControls (controls.js); perceived, never enforced
    this._ctlOut = {};                                // slots in station lots (queued / pumping / waiting to re-enter)

    // state
    const cap = this.slotPool || Math.max(1024, Math.min(Math.ceil(nC * 0.6) + 64, 1 << 20));
    this.vs = new VehicleStore(cap);
    this.vs.actionMode = this.opts.actionMode === 'rate' ? 'rate' : 'position';
    this.vs._dt = this.dt;
    this.hash = new SpatialHash({ cellSize: this.opts.cellSize, tableBits: this.opts.tableBits, capacity: cap });
    this.lanes = new LaneIndex(graph.edgeCount, cap);
    this.drivers = new Array(cap).fill(null);
    this.wq = new WorldQuery(this);
    this.events = [];
    this.crashes = [];            // all incident records (crashes, stranded cars, wrecks)
    this.activeCrashes = [];      // records not yet cleared
    this.edgeCrashes = new Map(); // edge -> records (active or shoulder)
    this.edgeHasCrash = new Uint8Array(graph.edgeCount);
    this.tripLog = [];
    this._pose = { x: 0, z: 0, y: 0, hx: 1, hz: 0, seg: 0 };
    this.counters = {
      spawned: 0, arrived: 0, arrivedOutbound: 0, onTime: 0, crashes: 0, crashedVehicles: 0, crashesWhileSpeeding: 0,
      stuck: 0, noRoute: 0, cleared: 0, laneChanges: 0, secondaryCrashes: 0, minorContacts: 0, touches: 0, disablingCrashes: 0, repeatContacts: 0, bySeverity: {}, pullOvers: 0, towedToShoulder: 0, stuckBy: {},
      stranded: 0, towed: 0, refuelPlanned: 0, refuelled: 0, refuelPlanFailed: 0, forcedSpawns: 0, signalViolations: 0, stopViolations: 0,
    };
    this.accum = { vehicleSec: 0, vehicleKm: 0, speedingSec: 0, speedingKm: 0, lateSum: 0, lateN: 0, fuelL: 0, pumpWaitSum: 0, pumpWaitN: 0 };
    this.live = { driving: 0, crashed: 0, refueling: 0, sumSpeed: 0, brainSteps: 0, learnerSteps: 0 };
    this._spawnedKeys = new Set();
    this._lastStepMs = 0;
    this.onTransition = opts.onTransition || null;
    this.onEvent = opts.onEvent || null;
    this.onStep = opts.onStep || null;           // (sim) at the end of every step (telemetry)
    this.ctx = this._makeCtx();
    // freeway density sampling (HCM LOS by density: D-F above 26 pc/mi/ln = 16.2 veh/km/ln)
    this._nextDensity = this.time;
    this._edgeCount = new Uint16Array(graph.edgeCount);
    this._fwy = []; let fwyLaneKm = 0;
    for (const e of graph.edges) if ((e.highway === 'motorway' || e.highway === 'trunk') && e.length >= 100) { this._fwy.push(e.idx); fwyLaneKm += e.length * e.lanes / 1000; }
    this._fwyLaneKm = fwyLaneKm;
    this.density = { samples: 0, peakMeanDensity: 0, peakMeanAt: 0, peakLosDFShare: 0, peakLosDFAt: 0, peakSegmentDensity: 0, meanDensitySum: 0, losDFShareSum: 0, last: null };
    this._sig = { collision: false, stranded: false, arrived: false, lateBy: undefined, harshBraking: false, offRoute: false, laneViolation: false, missedExit: false, timeToWorkStart: 0, remainingRouteDistance: 0, fuelFraction: 1, refuelPlanned: false };
    if (this.opts.trafficMode === 'sparse' && (opts.wrecks ?? 0) > 0) this.placeWrecks(opts.wrecks);
    if (opts.obstacles) this.placeObstacles(opts.obstacles);
    if (opts.staticObstacles) this.placeStaticObstacles(opts.staticObstacles);
  }

  // ---------------------------------------------------------------- clock
  advance(realDt) {
    this.accumulator += realDt * this.speed;
    let steps = Math.floor(this.accumulator / this.dt);
    if (steps > this.opts.maxStepsPerFrame) { steps = this.opts.maxStepsPerFrame; this.lagging = true; this.accumulator = steps * this.dt; }
    else this.lagging = false;
    for (let k = 0; k < steps; k++) this.step();
    this.accumulator -= steps * this.dt;
    return steps;
  }

  get clock() { const t = this.time; const h = Math.floor(t / 3600) % 24, m = Math.floor(t / 60) % 60, s = Math.floor(t) % 60; return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`; }
  get isIdle() { return this.vs.count === 0 && this.pendingSpawns.length === 0 && this.tripPtr >= this.trips.length && (!this.table || this.tablePtr >= this.table.n); }

  // ---------------------------------------------------------------- step
  step() {
    const t0 = typeof performance !== 'undefined' ? performance.now() : 0;
    const dt = this.dt;
    this.time += dt;
    this.stepCount++;
    this.ctx.time = this.time;
    this._spawn();
    this.hash.rebuild(this.vs);
    this._contacts();
    this.lanes.rebuild(this.vs);
    this._findLeaders();
    if (this.controls) this._controlObs();
    if (this._yieldBy) this._yieldBy.fill(-1);   // deadlockBreak read last step's junction-yield blockers; this step's driver calls refill it
    this._drive(dt);
    this._integrate(dt);
    if (this.vs.missed && this.vs.missed.size) this._missedExits();
    this._scenes();
    if (this.time >= (this._nextTraffic || 0)) { this._trafficSample(); this._nextTraffic = this.time + 60; }
    if (this.opts.reroute && this.time >= (this._nextReroute || 0)) { this._rerouteBlocked(); this._nextReroute = this.time + 2; }
    if (this.time >= this._nextDensity) { this._sampleDensity(); this._nextDensity = this.time + this.opts.densityInterval; }
    if (this.onStep) this.onStep(this);
    this._lastStepMs = typeof performance !== 'undefined' ? performance.now() - t0 : 0;
  }

  // ---------------------------------------------------------------- spawning
  _spawn() {
    const trips = this.trips;
    this._spawnedKeys.clear();
    if (this.lot.length) this._leaveStations();
    if (this.pendingSpawns.length) {
      const keep = [];
      for (const trip of this.pendingSpawns) if (!this._trySpawn(trip)) keep.push(trip);
      this.pendingSpawns = keep;
    }
    while (this.tripPtr < trips.length && trips[this.tripPtr].depart <= this.time) {
      const trip = trips[this.tripPtr++];
      if (!this._trySpawn(trip)) this.pendingSpawns.push(trip);
    }
    if (this.table) {
      const T = this.table;
      while (this.tablePtr < T.n && T.sortedDepart[this.tablePtr] <= this.time) {
        const id = T.order[this.tablePtr++];
        const trip = { commuter: id, from: T.home[id], to: T.work[id], route: null, depart: T.depart[id], leg: 0 };
        if (!this._trySpawn(trip)) this.pendingSpawns.push(trip);
      }
    }
  }

  /** Lane of edge `ei` (from `lane0` upward) with room to enter at s = 0, or -1. */
  _entryLane(ei, lane0 = 0, vlen = 5, next = -1) {
    const e = this.graph.edges[ei], vs = this.vs;
    if (this.empty) return Math.min(lane0, e.lanes - 1);
    // spawnLaneRoute 1 (docs/scale-crashes.md): first the lanes that continue onto my next route edge (a car
    // spawned in a lane that ends right away on a short edge could never leave it), then any lane
    const g = this.graph, sr = this.opts.spawnLaneRoute >= 1 && next >= 0;
    const goes = (l) => { const en = g.edges[next]; if (g.J && e.sStop < e.length) return g.J.movement(ei, l, next) >= 0; const m = mapLane(g, e, en, l); return m >= 0 && m < en.lanes; };
    for (let k = 0; k < (sr ? 2 : 1) * e.lanes; k++) {
      const l = (lane0 + k) % e.lanes;
      if (sr && (k < e.lanes) !== goes(l)) continue;
      const key = ei * 8 + l;
      if (this._spawnedKeys.has(key)) continue;
      const f = this.lanes.first(ei, l);
      if (f >= 0 && vs.s[f] - 0.5 * vs.length[f] < (e.sExit > 0 ? e.sExit : 0) + vlen + 4) continue;   // cars spawn at the box exit sExit
      if (this._upstreamConflict(ei, l)) continue;
      return l;
    }
    return -1;
  }
  /** A car about to enter edge ei lane l from upstream within its stopping reach. */
  _upstreamConflict(ei, l) {
    const g = this.graph, vs = this.vs, node = g.edges[ei].from;
    for (let q = g.inStart[node]; q < g.inStart[node + 1]; q++) {
      const e2 = g.inEdges[q], E2 = g.edges[e2];
      for (let k = 0; k < E2.lanes; k++) {
        const j = this.lanes.last(e2, k); if (j < 0 || vs.state[j] !== STATE_DRIVING) continue;
        const rj = vs.routes[j], pj = vs.routePos[j];
        if (pj + 1 >= rj.length || rj[pj + 1] !== ei || mapLane(g, E2, g.edges[ei], vs.lane[j]) !== l) continue;
        const d = E2.length - vs.s[j], v = vs.speed[j];
        if (d < 8 + v * 2 + v * v / 6) return true;
      }
    }
    return false;
  }

  _trySpawn(trip) {
    if (this.slotPool && this.vs.count >= this.slotPool) return false;   // pool full: wait for a free slot
    let c = this.commuters[trip.commuter];
    let route;
    if (this.table) {
      if (!trip.route) {
        const r = this.routeFor ? this.routeFor(trip.commuter, trip.leg, trip.from, trip.to) : undefined;
        if (r === undefined) return false;                                // route not computed yet
        trip.route = r || new Int32Array(0);
      }
      if (!c) c = this.commuters[trip.commuter] = this.table.commuter(trip.commuter);
      route = trip.route;
    } else route = this.routeStore.popRoute(trip.commuter, trip.leg) || trip.route;
    if (!route || route.length === 0) {
      this.counters.noRoute++;
      this._event('noRoute', this.graph.nx[trip.from] || 0, this.graph.nz[trip.from] || 0, [], { commuter: c.id, leg: trip.leg, from: trip.from, to: trip.to });
      this._setOutcome(c.id, trip.leg, FAIL.NOROUTE);
      this._logTrip(c, trip, -1, 'noRoute', 0, 0);
      return true;
    }
    const vt = VEHICLE_TYPES[c.vehicleType] || VEHICLE_TYPES[0];
    let lane = this._entryLane(route[0], 0, vt.length, route.length > 1 ? route[1] : -1);
    if (lane < 0) {
      if (this.time - trip.depart < this.opts.spawnBlockedTimeout) return false;
      lane = 0; this.counters.forcedSpawns++;
    }
    this._spawnedKeys.add(route[0] * 8 + lane);
    if (this.table) route = this.routeStore.add(route, (cb) => { const vs = this.vs; for (let k = 0; k < vs.high; k++) if (vs.state[k] !== STATE_FREE && vs.routeOff[k] >= 0) cb(k, vs.routeOff[k], vs.routeLen[k]); }, (slot, no, v) => { this.vs.routeOff[slot] = no; this.vs.routes[slot] = v; }).view;
    const learner = this.learners[c.id] === 1;
    const traits = c.genome.traits;
    const vs = this.vs;
    const slot = vs.alloc({ typeIdx: c.vehicleType, traits, route, commuter: c.id, leg: trip.leg, time: this.time, learner, dt: this.dt, step: this.stepCount, graph: this.graph, lane });
    vs.setReactionTime(slot, traits.reactionTime ?? c.reactionTime ?? 0.9, this.dt);
    vs.routeOff[slot] = this.routeStore.offsetOf(route);
    vs.fuel[slot] = Math.min(vs.tank[slot], this.commuterFuel[c.id]);
    vs.updatePose(slot, this.graph, this._pose); vs.savePose(slot);
    this.counters.spawned++;
    if (this.stations) this._checkFuel(slot, true);
    if (this.driverFactory) {
      const nav = this.graph.nav || this.graph;
      const pose = { x: vs.x[slot], z: vs.z[slot], y: vs.y[slot], heading: vs.heading[slot], speed: 0, edge: vs.routes[slot][0], s: 0, lane };
      try {
        this.drivers[slot] = this.driverFactory(nav, trip.from, trip.to, pose, { genome: c.genome, route: vs.routes[slot], commuter: c, slot, sim: this, learner, stop: this._stopInfo(slot) });
        if (this.drivers[slot]) vs.flags[slot] |= FLAG_BRAIN;
      } catch (err) { this.drivers[slot] = null; if (!this._driverErr) { this._driverErr = err; console.warn('driverFactory failed, the car uses opts.defaultAction:', err); } }
    }
    this._event('spawn', vs.x[slot], vs.z[slot], [slot], { commuter: c.id, leg: trip.leg });
    return true;
  }

  // ---------------------------------------------------------------- routes
  _setRoute(i, edges, routePos = 0) {
    const vs = this.vs, rs = this.routeStore;
    const { off, view } = rs.add(edges, (cb) => { for (let k = 0; k < vs.high; k++) if (vs.state[k] !== STATE_FREE && vs.routeOff[k] >= 0) cb(k, vs.routeOff[k], vs.routeLen[k]); },
      (slot, no, v) => { vs.routeOff[slot] = no; vs.routes[slot] = v; });
    vs.routes[i] = view; vs.routeOff[i] = off; vs.routeLen[i] = edges.length; vs.routePos[i] = routePos;
    let L = 0; for (let k = 0; k < edges.length; k++) L += this.graph.edges[edges[k]].length;
    vs.routeTotal[i] = L;
  }
  _stopInfo(i) {
    const vs = this.vs;
    if (vs.refuelState[i] !== 1 || !this.stations) return null;
    const st = this.stations.list[vs.station[i]];
    return { routeIndex: vs.stopPos[i], node: st.node, station: st, type: 'refuel' };
  }

  // ---------------------------------------------------------------- fuel
  /** Plan a refuel detour if needed. atSpawn: route starts at the current edge start. */
  _checkFuel(i, atSpawn = false) {
    const vs = this.vs, st = this.stations;
    vs.lastFuelCheck[i] = this.time;
    if (!st || vs.refuelState[i] !== 0) return false;
    const remaining = vs.routeTotal[i] - vs.routeDist[i];
    if (!st.needsFuel(vs.typeIdx[i], vs.fuel[i], vs.tank[i], remaining)) return false;
    const route = vs.routes[i], rp = vs.routePos[i], g = this.graph;
    const goal = g.edges[route[route.length - 1]].to;
    let from, prefix;
    if (atSpawn) { from = g.edges[route[0]].from; prefix = []; }
    else {
      // keep the current edge; detour from its end node (must not be the goal)
      from = g.edges[route[rp]].to; prefix = [route[rp]];
      if (from === goal) return false;
    }
    const plan = st.plan(from, goal);
    if (!plan) { this.counters.refuelPlanFailed++; return false; }
    const edges = new Int32Array(prefix.length + plan.edges.length);
    edges.set(prefix); edges.set(plan.edges, prefix.length);
    const sOnEdge = atSpawn ? 0 : vs.s[i];
    this._setRoute(i, edges, 0);
    vs.routeDist[i] = sOnEdge;
    vs.refuelState[i] = 1; vs.station[i] = plan.station.idx; vs.stopPos[i] = plan.stopIndex + prefix.length;
    vs.flags[i] |= FLAG_REFUEL;
    this.counters.refuelPlanned++;
    const drv = this.drivers[i];
    if (drv && drv.setRoute) { try { drv.setRoute(vs.routes[i], this._stopInfo(i), sOnEdge); } catch (e) { /* keep going */ } }
    this._event('refuelPlanned', vs.x[i], vs.z[i], [i], { station: plan.station.idx, detourTime: plan.detourTime });
    return true;
  }

  /** Vehicle i reached its station stop: leave the road, join the pump queue. */
  _enterStation(i) {
    const vs = this.vs, st = this.stations, k = vs.station[i];
    vs.state[i] = STATE_REFUELING; vs.speed[i] = 0; vs.accel[i] = 0; vs.refuelState[i] = 2;
    vs.clearControls(i);
    vs.pumpUntil[i] = this.time; // queue entry time until a pump is free
    st.queue[k].push(i);
    if (st.queue[k].length > st.stats.maxQueue) st.stats.maxQueue = st.queue[k].length;
    this.lot.push(i);
    this._event('refuelArrive', vs.x[i], vs.z[i], [i], { station: k, queue: st.queue[k].length });
  }

  _pumps() {
    const st = this.stations; if (!st || !this.lot.length) return;
    const vs = this.vs, cfg = this.fuelCfg;
    for (let k = 0; k < st.list.length; k++) {
      const q = st.queue[k];
      while (q.length && st.busy[k] < st.list[k].pumps) {
        const i = q.shift(); st.busy[k]++;
        this.accum.pumpWaitSum += this.time - vs.pumpUntil[i]; this.accum.pumpWaitN++;
        vs.refuelState[i] = 3; vs.pumpUntil[i] = this.time + cfg.pumpMin + (cfg.pumpMax - cfg.pumpMin) * this.rng.random();
      }
    }
  }

  /** Cars done pumping re-enter the road on the next route edge when there is room. */
  _leaveStations() {
    const vs = this.vs, st = this.stations, keep = [];
    for (const i of this.lot) {
      if (vs.state[i] !== STATE_REFUELING) continue;
      if (vs.refuelState[i] === 3 && this.time >= vs.pumpUntil[i]) {
        const k = vs.station[i];
        this.accum.fuelL += vs.tank[i] - vs.fuel[i];
        vs.fuel[i] = vs.tank[i];
        st.busy[k]--; st.served[k]++; vs.refuelState[i] = 4; // 4 = filled, waiting to re-enter
        this.counters.refuelled++;
        this._event('refuelled', vs.x[i], vs.z[i], [i], { station: k });
      }
      if (vs.refuelState[i] === 4) {
        const next = vs.stopPos[i] + 1, route = vs.routes[i];
        if (next >= route.length) { this._arrive(i); continue; }
        const lane = this._entryLane(route[next], 0, vs.length[i]);
        if (lane < 0 && this.time - vs.pumpUntil[i] < this.opts.spawnBlockedTimeout) { keep.push(i); continue; }
        this._spawnedKeys.add(route[next] * 8 + Math.max(0, lane));
        vs.state[i] = STATE_DRIVING; vs.refuelState[i] = 0; vs.routePos[i] = next; vs.edge[i] = route[next]; vs.s[i] = 0; vs.seg[i] = 0;
        vs.lat[i] = (Math.max(0, lane) + 0.5) * laneW(this.graph.edges[route[next]], 0); vs.headingRel[i] = 0; vs.steerAngle[i] = 0; vs.steerSl[i] = 0; vs.targetLane[i] = -1; vs.clearControls(i);
        vs.updateLanes(i, this.graph.edges[route[next]]); vs.stopPos[i] = -1; vs.station[i] = -1;
        vs.flags[i] &= ~FLAG_REFUEL; vs.stoppedSince[i] = -1;
        vs.speed[i] = 0; vs.vLat[i] = 0; vs.yawRate[i] = 0;
        vs.updatePose(i, this.graph, this._pose); vs.savePose(i);
        const drv = this.drivers[i]; if (drv && drv.refuelDone) { try { drv.refuelDone(); } catch (e) { /* ignore */ } }
        continue;
      }
      keep.push(i);
    }
    this.lot = keep;
  }

  _strand(i) {
    const vs = this.vs, g = this.graph, e = g.edges[vs.edge[i]];
    vs.flags[i] |= FLAG_STRANDED;
    this.counters.stranded++;
    const tow = this.fuelCfg.towMedian * Math.exp(this.fuelCfg.towSigma * this.rng.gaussian());
    const rec = this._openScene([i], { cause: 'stranded', severity: 'stranded', impactSpeed: 0, impactMph: 0, lanesBlocked: 1, blockDuration: tow, clearDuration: tow }, e, false);
    this._failTrip(i, FAIL.STRANDED, 'stranded');
    return rec;
  }

  /** Static stopped wrecks for curriculum stage 1 ('sparse'): obstacles on random main-tier edges. */
  placeWrecks(n) {
    const g = this.graph, cand = [];
    for (const e of g.edges) if (e.rank >= 3 && e.length > 150) cand.push(e.idx);
    for (let k = 0; k < n && cand.length; k++) {
      const ei = cand[this.rng.int(cand.length)], e = g.edges[ei];
      const route = Int32Array.of(ei);
      const slot = this.vs.alloc({ typeIdx: 0, route, commuter: -1, time: this.time, dt: this.dt, step: this.stepCount, graph: g });
      const vs = this.vs; vs.s[slot] = e.length * (0.3 + 0.4 * this.rng.random()); vs.lat[slot] = (this.rng.int(e.lanes) + 0.5) * laneW(e, vs.s[slot]); vs.updateLanes(slot, e);
      vs.updatePose(slot, g, this._pose); vs.savePose(slot);
      this._openScene([slot], { cause: 'wreck', severity: 'wreck', impactSpeed: 0, impactMph: 0, lanesBlocked: 1, blockDuration: 1e9, clearDuration: 1e9 }, e, false);
    }
  }

  /**
   * Explicit static obstacles (scenario worlds / tests): [{edge, at (m along the edge), lane | lat, type}].
   * type: vehicle type name ('sedan' | 'suv' | 'truck' | 'bus') or index (default sedan). Each is a disabled
   * car (a real collidable body) in a permanent scene. Returns the placed slots.
   */
  placeStaticObstacles(list) {
    const g = this.graph, vs = this.vs, out = [];
    const TYPES = { sedan: 0, suv: 1, truck: 2, bus: 3 };
    for (const o of list) {
      const e = g.edges[o.edge]; if (!e) continue;
      const s = Math.max(0, Math.min(e.length, +o.at || 0));
      const lat = o.lat !== undefined ? +o.lat : ((o.lane ?? 0) + 0.5) * laneW(e, s);
      const typeIdx = typeof o.type === 'number' ? o.type : (TYPES[o.type] ?? 0);
      const slot = vs.alloc({ typeIdx, route: Int32Array.of(e.idx), commuter: -1, time: this.time, dt: this.dt, step: this.stepCount, graph: g });
      vs.s[slot] = s; vs.lat[slot] = lat; vs.updateLanes(slot, e); vs.updatePose(slot, g, this._pose); vs.savePose(slot);
      this._openScene([slot], { cause: 'obstacle', severity: 'wreck', impactSpeed: 0, impactMph: 0, lanesBlocked: 1, blockDuration: 1e9, clearDuration: 1e9 }, e, false);
      vs.flags[slot] |= FLAG_DISABLED;
      out.push(slot);
    }
    this.staticObstacles = out;
    return out;
  }

  /**
   * Obstacle placer for pre-training (createTrainingEnv `obstacles`). Every obstacle is a real
   * collidable body, perceived only through the normal observation (lane lists -> lead / lead2,
   * junction perception -> junction obs / leader -3, swept-path bodies).
   *   spec = { count = 40, kinds = [...all], seed = 1, slowKmh = [10, 20], brakeEvery = [15, 40] }
   * kinds:
   *   'midblock'   stopped (disabled) car in a lane in the middle of a >= 150 m edge
   *   'afterCurve' stopped car 5-30 m after a bend (a hop turning > 0.3 rad)
   *   'afterMerge' stopped car 30-80 m after a strict merge (ramp / lane join)
   *   'junction'   stopped car inside a junction box, on a connector
   *   'rampEnd'    stopped car at the end of an on-ramp (just before the gore / merge node)
   *   'partial'    stopped car straddling a lane line (one-lane roads: an edge line), half in each
   *   'slow'       a driving car limited to slowKmh (10-20 km/h) on a 30-edge route
   *   'braking'    a driving car that brakes hard (full brake for 1.5-3 s) every brakeEvery s
   * Static ones are permanent scenes (they never clear); unless spec.allowSingleLane, lane blockers avoid
   * single-lane edges (and 1-lane-to-1-lane connectors) so trips stay possible. Returns [{slot, kind, edge, s}].
   */
  placeObstacles(spec = {}) {
    const g = this.graph, vs = this.vs, J = g.J;
    const R = makeRng(spec.seed ?? 1), count = spec.count ?? 40;
    const kinds = spec.kinds || ['midblock', 'afterCurve', 'afterMerge', 'junction', 'rampEnd', 'partial', 'slow', 'braking'];
    const slowKmh = spec.slowKmh || [10, 20], brakeEvery = spec.brakeEvery || [15, 40], allow1 = !!spec.allowSingleLane;
    const pools = {};
    const pool = (k) => {
      if (pools[k]) return pools[k];
      const out = [];
      for (const e of g.edges) {
        const multi = allow1 || e.lanes > 1;
        if (k === 'slow' || k === 'braking') { if (e.rank >= 1 && e.length >= 150) out.push(e.idx); continue; }
        if (k === 'midblock' || k === 'partial') { if (e.rank >= 1 && e.length >= 150 && multi) out.push(e.idx); continue; }
        if (k === 'rampEnd') { if (e.gore !== undefined && e.gore >= 0 && e.length > 30) out.push(e.idx); continue; }
        if (k === 'afterCurve' || k === 'afterMerge') {
          if (e.length < 90) continue;
          const ins = []; for (let q = g.inStart[e.from]; q < g.inStart[e.from + 1]; q++) if (g.inEdges[q] !== e.reverse) ins.push(g.edges[g.inEdges[q]]);
          if (k === 'afterMerge' && ins.length >= 2 && ins.every(a => hopMap(g, a, e).strict)) out.push(e.idx);
          if (k === 'afterCurve' && ins.length === 1 && multi) { let d = e.headingStart - ins[0].headingEnd; d = Math.abs(Math.atan2(Math.sin(d), Math.cos(d))); if (d > 0.3 && hopMap(g, ins[0], e).strict) out.push(e.idx); }
        }
      }
      return (pools[k] = out);
    };
    const placed = [];
    const staticCar = (e, s, lat, conn = -1) => {
      const slot = vs.alloc({ typeIdx: 0, route: Int32Array.of(e.idx), commuter: -1, time: this.time, dt: this.dt, step: this.stepCount, graph: g });
      vs.s[slot] = s; vs.lat[slot] = lat; vs.conn[slot] = conn; vs.updateLanes(slot, e);
      vs.updatePose(slot, g, this._pose); vs.savePose(slot);
      this._openScene([slot], { cause: 'obstacle', severity: 'wreck', impactSpeed: 0, impactMph: 0, lanesBlocked: 1, blockDuration: 1e9, clearDuration: 1e9 }, e, false);
      vs.flags[slot] |= FLAG_DISABLED;
      return slot;
    };
    const walk = (e0) => { const r = [e0.idx]; let e = e0; for (let k = 0; k < 30; k++) { const outs = []; for (let q = g.outStart[e.to]; q < g.outStart[e.to + 1]; q++) if (g.outEdges[q] !== e.reverse) outs.push(g.outEdges[q]); if (!outs.length) break; e = g.edges[outs[R.int(outs.length)]]; r.push(e.idx); } return Int32Array.from(r); };
    for (let n = 0; n < count; n++) {
      const kind = kinds[n % kinds.length];
      let slot = -1, e = null, s = 0;
      if (kind === 'junction') {
        if (!J || !J.moveCount) continue;
        // a connector point within ~4 m of the junction's node
        let m = R.int(J.moveCount), d = J.describe(m), t = 0.5 * d.length;
        for (let tries = 0; tries < 20; tries++) {
          m = R.int(J.moveCount); d = J.describe(m); e = g.edges[d.inEdge];
          if (!allow1 && e.lanes === 1 && g.edges[d.outEdge].lanes === 1 && tries < 19) continue;   // keep single-lane trips passable
          let bt = -1, bd = 1e9; const q = {};
          for (let u = 0; u <= d.length; u += 0.5) { J.arcPoint(m, u, q); const dd = Math.hypot(q.x - g.nx[e.to], q.z - g.nz[e.to]); if (dd < bd) { bd = dd; bt = u; } }
          t = bt; if (bd < 4) break;
        }
        e = g.edges[d.inEdge]; s = d.sIn + t;
        slot = staticCar(e, s, (d.inLane + 0.5) * laneW(e, d.sIn), m);
      } else {
        const P = pool(kind); if (!P.length) continue;
        e = g.edges[P[R.int(P.length)]];
        const lane = R.int(e.lanes), W = laneW(e, e.length / 2);
        if (kind === 'midblock') { s = e.length * R.range(0.3, 0.7); slot = staticCar(e, s, (lane + 0.5) * W); }
        else if (kind === 'afterCurve') { s = Math.min(e.length - 10, R.range(5, 30)); slot = staticCar(e, s, (lane + 0.5) * laneW(e, s)); }
        else if (kind === 'afterMerge') { s = Math.min(e.length - 10, R.range(30, 80)); slot = staticCar(e, s, (lane + 0.5) * laneW(e, s)); }
        else if (kind === 'rampEnd') { s = Math.max(5, e.length - R.range(3, 12)); slot = staticCar(e, s, 0.5 * laneW(e, s)); }
        else if (kind === 'partial') { s = e.length * R.range(0.3, 0.7); const k = e.lanes > 1 ? 1 + R.int(e.lanes - 1) : R.int(2); slot = staticCar(e, s, k * laneW(e, s) + R.range(-0.4, 0.4)); }   // on a lane line (one-lane roads: an edge line)
        else {
          s = e.length * R.range(0.1, 0.5);
          slot = vs.alloc({ typeIdx: 0, route: walk(e), commuter: -1, time: this.time, dt: this.dt, step: this.stepCount, graph: g, lane });
          vs.s[slot] = s; vs.lat[slot] = (lane + 0.5) * laneW(e, s); vs.updateLanes(slot, e); vs.updatePose(slot, g, this._pose); vs.savePose(slot);
          if (kind === 'slow') { vs.maxSpeed[slot] = R.range(slowKmh[0], slowKmh[1]) / 3.6; vs.speed[slot] = vs.maxSpeed[slot]; const sim = this, sl = slot; this.drivers[slot] = { step() { return propControl(sim, sl, vs.maxSpeed[sl]); } }; }
          else {
            vs.speed[slot] = Math.min(e.speedLimit, 15);
            const sim = this, sl = slot; let next = this.time + R.range(brakeEvery[0], brakeEvery[1]), until = -1;
            this.drivers[slot] = { step() {
              const c = propControl(sim, sl, vs.maxSpeed[sl]);
              if (sim.time >= next) { until = sim.time + R.range(1.5, 3); next = until + R.range(brakeEvery[0], brakeEvery[1]); }
              if (sim.time < until) return { steer: c.steer, gas: 0, brake: 1 };
              return { steer: c.steer, gas: c.gas, brake: c.brake };
            } };
          }
        }
      }
      if (slot >= 0) placed.push({ slot, kind, edge: e.idx, s });
    }
    this.obstacles = placed;
    return placed;
  }

  // ---------------------------------------------------------------- leaders
  /**
   * First vehicle at or ahead of `a` in `lane` whose body physically overlaps the lane
   * footprint. Visibility is decided from the body position only (never from the shoulder
   * flag): a car entirely off the lanes (laneHi < laneLo) is skipped.
   */
  _blocking(a, lane) {
    const vs = this.vs;
    while (a >= 0 && vs.laneHi[a] < vs.laneLo[a]) a = this.lanes.ahead(a, lane);
    return a;
  }

  _findLeaders() {
    const vs = this.vs, g = this.graph, lanes = this.lanes, edges = g.edges;
    const high = vs.high, look = this.opts.leaderLookahead, zipLook = this.opts.mergeLookahead;
    if (this.empty) {
      // agents ignore each other; only static obstacles (wrecks / scenes) and lane ends are seen
      for (let i = 0; i < high; i++) if (vs.state[i] === STATE_DRIVING) { vs.leader[i] = -1; vs.gap[i] = 1e9; this._laneEnd(i); this._sceneLeader(i); }
      return;
    }
    if (g.J) this._indexConnectors();
    const off = this._offLane; for (const l of off.values()) l.length = 0; if (off.size > 20000) off.clear();
    for (let i = 0; i < high; i++) if ((vs.state[i] === STATE_DRIVING || vs.state[i] === STATE_CRASHED) && vs.laneHi[i] < vs.laneLo[i]) { let l = off.get(vs.edge[i]); if (!l) off.set(vs.edge[i], (l = [])); l.push(i); }
    const passers = this._passers || (this._passers = new Map());
    for (const l of passers.values()) l.length = 0;
    for (let i = 0; i < high; i++) if (vs.passWreck[i] >= 0 && vs.state[i] === STATE_DRIVING) { let l = passers.get(vs.edge[i]); if (!l) passers.set(vs.edge[i], (l = [])); l.push(i); }
    if (passers.size > 5000) passers.clear();
    for (let i = 0; i < high; i++) {
      if (vs.state[i] !== STATE_DRIVING) continue;
      let leader = -1, gap = 1e9;
      const e = edges[vs.edge[i]], route = vs.routes[i], rp = vs.routePos[i];
      const stopHere = vs.refuelState[i] === 1 && vs.stopPos[i] === rp;
      const lo = vs.laneLo[i], hi = vs.laneHi[i], si = vs.s[i], hl = 0.5 * vs.length[i];
      let seen = false;
      const pw = vs.passWreck[i];
      for (let l = lo; l <= hi; l++) {
        let a = this._blocking(lanes.ahead(i, l), l);
        // passing a wreck through the opposing lane: the wrecks up to it are beside me, not ahead
        while (pw >= 0 && a >= 0 && vs.state[a] === STATE_CRASHED && vs.s[a] <= vs.s[pw] + 0.1) a = this._blocking(lanes.ahead(a, l), l);
        if (a >= 0) { seen = true; const gg = vs.s[a] - si - hl - 0.5 * vs.length[a]; if (gg < gap) { gap = gg; leader = a; } }
      }
      if (!seen && !stopHere) {
        // across edge boundaries: my occupied lanes mapped through the hop lane mapping
        let dist = e.length - si, prev = e, plo = lo, phi = hi, s0 = 0;
        if (g.J && rp + 1 < route.length) {
          // through the junction connector I am on (or will take from my lane): the next edge starts at sOut
          let m = vs.conn[i]; if (m < 0 && e.sStop < e.length) m = g.J.movement(e.idx, vs.lane[i], route[rp + 1]);
          if (m >= 0) { dist = g.J.mF[4 * m] + g.J.mLen[m] - si; s0 = g.J.mF[4 * m + 1]; }
        }
        for (let k = 1; k <= 3 && rp + k < route.length && dist < look; k++) {
          const en = edges[route[rp + k]];
          const nlo = mapLane(g, prev, en, plo), nhi = mapLane(g, prev, en, phi);
          let found = false;
          for (let l2 = Math.max(0, nlo); l2 <= Math.min(en.lanes - 1, nhi); l2++) {
            const f = this._blocking(lanes.first(en.idx, l2), l2);
            if (f >= 0) { found = true; const gg = dist + vs.s[f] - (k === 1 ? s0 : 0) - hl - 0.5 * vs.length[f]; if (gg < gap) { gap = gg; leader = f; } }
          }
          if (found || nhi < 0 || nlo >= en.lanes) break;
          plo = Math.max(0, nlo); phi = Math.min(en.lanes - 1, nhi);
          dist += en.length - (k === 1 ? s0 : 0); prev = en;
        }
      }
      // zipper merge: a car on another incoming edge of my next node, closer to the node,
      // heading into the same lane of the same next edge, goes first
      const dNode = e.length - si;
      // merge order is settled early enough to adjust speed comfortably: ~7 s ahead at speed
      if (rp + 1 < route.length && dNode < Math.max(zipLook, 7 * vs.speed[i]) && !stopHere) {   // (a merging car can be closer than my current leader)
        const en = edges[route[rp + 1]], myLane = mapLane(g, e, en, vs.lane[i]), node = e.to;
        for (let q = g.inStart[node]; q < g.inStart[node + 1]; q++) {
          const e2 = g.inEdges[q]; if (e2 === e.idx) continue;
          const E2 = edges[e2];
          for (let l = 0; l < E2.lanes; l++) {
            // a wreck sitting at the merge point is an obstacle for everyone merging here
            const w = lanes.last(e2, l);
            if (w >= 0 && vs.state[w] === STATE_CRASHED && vs.laneHi[w] >= vs.laneLo[w] && E2.length - vs.s[w] < 15) {
              const gw = dNode - 0.5 * vs.length[i] - 3; if (gw < gap) { gap = gw; leader = w; }
            }
            // a ramp merging into the mainline (src/geom gore merge, right side) converges with the mainline's lane 0
            // physically before the node, whatever the lane bookkeeping says: zipper them
            const rampMerge = (e.gore >= 0 && !(E2.gore >= 0) && l === 0) || (E2.gore >= 0 && !(e.gore >= 0) && vs.lane[i] === 0);
            if (!rampMerge && mapLane(g, E2, en, l) !== myLane) continue;
            const j = lanes.last(e2, l); if (j < 0 || vs.state[j] !== STATE_DRIVING) continue;
            const rj = vs.routes[j], pj = vs.routePos[j];
            if (pj + 1 >= rj.length || rj[pj + 1] !== en.idx) continue;
            if (vs.refuelState[j] === 1 && vs.stopPos[j] === pj) continue;
            // it gives way here (sign on its approach, none on mine) or must stop (red / stop sign): it does not go first
            if ((this._signed(j) && !this._signed(i)) || this._mustStop(j)) continue;
            // a ramp car that has not passed its gore (wait) point is not in the mainline lane yet: the ramp yields
            if (E2.gore !== undefined && E2.gore >= 0 && e.gore !== undefined && e.gore < 0 && vs.s[j] + vs.hl[j] < E2.gore + 0.5) continue;
            const dj = E2.length - vs.s[j];
            // I give way here (sign on my approach, none on its): wait at my line while it arrives within 5 s
            if (this._signed(i) && !this._signed(j) && vs.ctlDist[i] > -0.5 && vs.ctlDist[i] < 40) {
              if (dj / Math.max(vs.speed[j], 2) < 5) { const gl = vs.ctlDist[i] - 0.5; if (gl < gap) { gap = gl; leader = -3; this._conflictWith.set(i, j); } }
              continue;
            }
            if (dj < dNode || (dj === dNode && j < i)) {
              const gz = dNode - dj - hl - 0.5 * vs.length[j];
              // a ramp merging into the mainline's lanes (no added lane, src/geom J.gore): without room to
              // tuck in behind that car, wait at the gore point, outside the mainline lane, not inside it
              const gore = e.gore;
              if (gore >= 0 && E2.lanes >= e.lanes && vs.s[i] + hl < gore + 0.5 && gz < cfgGap(vs.speed[i])) {
                const gg = gore - vs.s[i] - hl - 0.5; if (gg < gap) { gap = gg; leader = -3; }
              } else if (gz < gap) { gap = gz; leader = j; }
            }
          }
        }
      }
      vs.leader[i] = leader; vs.gap[i] = gap;
      // the car in front of my leader (brake lights two cars ahead / a wreck hidden by the leader)
      vs.leader2[i] = -1; vs.gap2[i] = 1e9;
      if (leader >= 0 && vs.edge[leader] === vs.edge[i] && vs.laneHi[leader] >= vs.laneLo[leader]) {
        const l2 = this._blocking(lanes.ahead(leader, vs.lane[leader]), vs.lane[leader]);
        if (l2 >= 0 && l2 !== i) { vs.leader2[i] = l2; vs.gap2[i] = vs.s[l2] - vs.s[i] - hl - 0.5 * vs.length[l2]; }
      }
      this._laneEnd(i);
      this._offLaneBodies(i);
      if (pw < 0) this._sceneLeader(i);
      if (g.J) { this._junction(i); if (vs.conn[i] < 0) this._nodeBodies(i, e); }
      if (vs.conn[i] < 0) this._convergeBodies(i, e);
      if (e.reverse >= 0) this._opposing(i, e);
      if (this.opts.pathSight >= 1) this._pathBodies(i);
      // a stopped obstacle (crash scene / stranded car / wreck) ahead: unless I am already
      // steering out, stop MERGE_ROOM short of it so there is room to creep around it
      const L = vs.leader[i];
      // the room kept depends on how far my body still overlaps the obstacle sideways: a car needs ~4 m of
      // travel per metre of sideways move to steer around it at walking pace, so it never noses up to it
      if (L >= 0 && vs.state[L] === STATE_CRASHED && vs.edge[L] === vs.edge[i]) {
        const hi_ = vs.halfExtent(i), hL = vs.halfExtent(L);
        const ov = Math.min(vs.lat[i] + hi_, vs.lat[L] + hL) - Math.max(vs.lat[i] - hi_, vs.lat[L] - hL);
        if (ov > 0) vs.gap[i] -= Math.min(MERGE_ROOM, 2 + 4 * (ov + 0.3));
      }
    }
    this._courtesy();
    if (this.opts.deadlockBreak >= 1) this._breakDeadlocks();
  }

  /**
   * deadlockBreak 1 (docs/scale-crashes.md): cars standing > 20 s that wait on each other in a closed loop
   * (leader / conflict point / junction yield, e.g. overlapping pavements where opposite carriageways meet,
   * three cars in a box) never move again. The lowest slot of each loop creeps on at walking pace (its blocker
   * becomes a point 3 m ahead), as real drivers inch forward and squeeze past; a touch at that speed is harmless.
   * Built-in (IDM) drivers only; loops through learned drivers are only detected through them.
   */
  _breakDeadlocks() {
    const vs = this.vs, cw = this._conflictWith, yb = this._yieldBy, now = this.time;
    if (!this._creepUntil || this._creepUntil.length < vs.capacity) {
      const u = new Float64Array(vs.capacity), g2 = new Int32Array(vs.capacity).fill(-1);
      if (this._creepUntil) { u.set(this._creepUntil); g2.set(this._creepIgnore); }
      this._creepUntil = u; this._creepIgnore = g2;
    }
    const CU = this._creepUntil, CI = this._creepIgnore;
    const waits = (k) => { const L = vs.leader[k]; if (L >= 0) return L; if (L === -3 && cw) { const j = cw.get(k); if (j !== undefined) return j; } return yb && yb[k] >= 0 ? yb[k] : -1; };
    const still = (k) => vs.state[k] === STATE_DRIVING && vs.speed[k] < 0.3 && vs.stoppedSince[k] >= 0 && now - vs.stoppedSince[k] > 20;
    const seen = this._dlSeen || (this._dlSeen = []);
    for (let i = 0; i < vs.high; i++) {
      // built-in drivers only: a learned driver does not creep toward a perceived 3 m point, it would only see a
      // phantom obstacle there and stand still (measured: SF 3000 ckpt-507, docs/scale-crashes.md)
      if (vs.state[i] !== STATE_DRIVING || (vs.flags[i] & FLAG_BRAIN)) continue;
      if (CU[i] > now) {                                    // creeping: the blocker it squeezes past stays ignored
        const w = waits(i);
        if (w === CI[i] && vs.state[w] !== STATE_CRASHED) { vs.leader[i] = -3; vs.gap[i] = 3; if (cw) cw.delete(i); }
        continue;
      }
      if (!still(i)) continue;
      seen.length = 0; let k = i, lo = i, ok = false;
      for (let n = 0; n < 10; n++) {
        const w = waits(k); if (w < 0 || !still(w)) break;
        if (w === i) { ok = true; break; }
        if (seen.includes(w)) break;
        seen.push(w); if (w < lo) lo = w; k = w;
      }
      if (ok && lo === i) {
        CU[i] = now + 10; CI[i] = waits(i);
        vs.leader[i] = -3; vs.gap[i] = 3; if (cw) cw.delete(i);
        this.counters.deadlockBreaks = (this.counters.deadlockBreaks || 0) + 1;
      }
    }
  }


  /**
   * Zipper courtesy in slow traffic: a slow car that has to leave its lane (its lane ends ahead, or a
   * stopped obstacle blocks it) and is ahead of the first car behind it in the target lane is treated by
   * that car as its leader, so the follower holds back and leaves the gap (each car lets one in).
   */
  _courtesy() {
    const vs = this.vs, lanes = this.lanes, pair = this._cpair || (this._cpair = [-1, -1]);
    for (let k = 0; k < vs.high; k++) {
      if (vs.state[k] !== STATE_DRIVING || vs.conn[k] >= 0 || vs.speed[k] > 6) continue;
      let side = 0;
      if (vs.laneEndSide[k] !== 0 && vs.laneEndDist[k] < 80) side = vs.laneEndSide[k];
      else if (vs.targetLane[k] >= 0 && vs.targetLane[k] !== vs.lane[k]) { const L = vs.leader[k]; if ((L >= 0 && vs.state[L] === STATE_CRASHED) || L === -2 || vs.gap[k] < 15) side = vs.targetLane[k] > vs.lane[k] ? 1 : -1; }
      if (!side) continue;
      const e = this.graph.edges[vs.edge[k]], tl = vs.lane[k] + side;
      if (tl < 0 || tl >= e.lanes) continue;
      lanes.neighborsAt(e.idx, tl, vs.s[k], pair, k);
      const b = pair[1]; if (b < 0 || vs.state[b] !== STATE_DRIVING || vs.conn[b] >= 0) continue;
      const gg = vs.s[k] - vs.s[b] - vs.hl[k] - vs.hl[b];
      if (gg < 0.5 || gg > 40) continue;
      if (gg < vs.gap[b]) { vs.gap[b] = gg; vs.leader[b] = k; }
    }
  }
  /**
   * Lane ends. A lane that does not continue (followed through roadGraph.mapLane) ends at
   * that node and tapers out over TAPER m of the next edge. A car whose centre lane ends
   * within the lookahead (or that is already in a taper, outside the carriageway) sees a
   * stopped obstacle (leader -2): at the taper end while it is merging (built-in driver:
   * targetLane != lane; any driver: heading toward the surviving lanes), else
   * MERGE_ROOM m earlier, so a car that had to stop still has room to creep in while steering.
   * Also sets vs.laneEndDist / laneEndSide for the driver observation.
   */
  _laneEnd(i) {
    const vs = this.vs, g = this.graph, route = vs.routes[i], rp = vs.routePos[i];
    vs.laneEndDist[i] = 1e9; vs.laneEndSide[i] = 0;
    const e = g.edges[vs.edge[i]], W = laneW(e, vs.s[i]), lat = vs.lat[i];
    let dEnd = -1, side = 0;
    const m = vs.conn[i];
    if (m >= 0 && g.J) {
      // on a junction connector: its exit lane exists by construction; look on from the exit point
      const J = g.J, rest = J.mF[4 * m] + J.mLen[m] - vs.s[i];
      const en = g.edges[J.mInfo[7 * m + 2]], ol = Math.min(en.lanes - 1, Math.max(0, mapLane(g, e, en, vs.lane[i])));   // = the connector's exit lane for my lane
      const o = laneEndAhead(g, route, rp + 1, J.mF[4 * m + 1], ol, { lookahead: this.opts.laneEndLookahead, stopIndex: vs.refuelState[i] === 1 ? vs.stopPos[i] : -1 }, this._le || (this._le = {}));
      vs.mergeCap[i] = o.mergeCap;
      if (o.side === 0) return;
      vs.laneEndDist[i] = rest + o.dist; vs.laneEndSide[i] = o.side;
      return;   // no lane-change room needed inside the box itself
    }
    if (vs.taper[i] > 0 && (lat < 0 || lat > e.lanes * W)) { dEnd = vs.taper[i] - vs.s[i]; side = lat < 0 ? 1 : -1; }
    else {
      const o = laneEndAhead(g, route, rp, vs.s[i], vs.lane[i], { lookahead: this.opts.laneEndLookahead, stopIndex: vs.refuelState[i] === 1 ? vs.stopPos[i] : -1 }, this._le || (this._le = {}));
      vs.mergeCap[i] = o.mergeCap;
      if (o.side !== 0) { side = o.side; dEnd = g.laneGeom ? o.dist : o.dist + TAPER; }
    }
    if (dEnd < 0) return;
    vs.laneEndDist[i] = dEnd; vs.laneEndSide[i] = side;
    const merging = (vs.targetLane[i] >= 0 && vs.targetLane[i] !== vs.lane[i] && !(vs.flags[i] & FLAG_BRAIN)) || vs.headingRel[i] * side > 0.04;
    const atNode = prevEndNode(dEnd, vs, i, this);
    // a merging car whose body already reaches into the surviving lane may carry on into the paved run-out
    // (src/geom draws the ended lane narrowing past the node) while it completes the move
    const into = side > 0 ? vs.laneHi[i] > vs.lane[i] : vs.laneLo[i] < vs.lane[i];
    const gg = (merging ? (into && g.laneGeom ? dEnd + 30 : atNode) : dEnd - MERGE_ROOM) - 0.5 * vs.length[i] - 1;
    if (gg < vs.gap[i]) { vs.gap[i] = gg; vs.leader[i] = -2; }
  }
  /**
   * Bodies entirely off the travel lanes (wrecks on the shoulder, pulled-over cars) are in no lane list;
   * a car whose own body reaches off the lanes (pulling over, drifting) sees them when they overlap it
   * laterally on its edge ahead.
   */
  _offLaneBodies(i) {
    const vs = this.vs, e = this.graph.edges[vs.edge[i]];
    const he = vs.halfExtent(i), lat = vs.lat[i], W = laneW(e, vs.s[i]);
    if (lat - he > 0.05 && lat + he < e.lanes * W - 0.05) return;    // my body is inside the lanes
    const list = this._offLane.get(e.idx); if (!list) return;
    for (const j of list) {
      if (j === i || vs.s[j] <= vs.s[i]) continue;
      if (Math.abs(vs.lat[j] - lat) > he + vs.halfExtent(j) + 0.3) continue;
      const gg = vs.s[j] - vs.s[i] - vs.hl[i] - vs.hl[j];
      if (gg < vs.gap[i]) { vs.gap[i] = gg; vs.leader[i] = j; }
    }
  }
  /** Active scene on my edge, in my lane, ahead of me: a stationary obstacle. */
  _sceneLeader(i) {
    const vs = this.vs;
    if (!this.edgeHasCrash[vs.edge[i]]) return;
    const recs = this.edgeCrashes.get(vs.edge[i]);
    if (recs) for (const r of recs) {
      if (r.responseState !== 'active' || r.squeeze || r.s <= vs.s[i]) continue;
      let hit = false; for (let l = vs.laneLo[i]; l <= vs.laneHi[i]; l++) if ((r.laneMask >> l) & 1) hit = true;
      if (!hit) continue;
      const gg = r.s - vs.s[i] - 0.5 * vs.length[i] - 3;
      if (gg < vs.gap[i]) { vs.gap[i] = gg; vs.leader[i] = r.vehicles[0]; }
    }
  }

  // ---------------------------------------------------------------- passing wrecks on two-lane roads
  /**
   * Opposing-lane interaction on a two-way edge e for slot i:
   *  - cars passing a wreck in the opposing lane (from the reverse edge) are stationary obstacles for
   *    me when my body is in my inner (leftmost) lane;
   *  - while I pass (passWreck), cars in the reverse edge's inner lane ahead are obstacles for me, and the
   *    pass ends once I am past the wreck and back in my lane (or the wreck is gone);
   *  - a car stopped > 5 s behind a wreck that closes its only lane (1 lane each way) starts a pass when
   *    the oncoming lane is clear for the whole manoeuvre (no body in it over the pass zone, and no car
   *    that could reach the zone within the pass time + 25 m), and the pass ends before the junction mouth.
   */
  _opposing(i, e) {
    const vs = this.vs, g = this.graph, lanes = this.lanes, er = g.edges[e.reverse], li = er.lanes - 1;
    const hl = vs.hl[i], si = vs.s[i];
    const opp = this._passers.get(e.reverse);
    if (opp && opp.length && vs.laneHi[i] === e.lanes - 1) for (const p of opp) {
      const sp = e.length - vs.s[p];
      if (sp > si) { const gg = sp - si - hl - vs.hl[p] - 2; if (gg < vs.gap[i]) { vs.gap[i] = gg; vs.leader[i] = -3; } }
    }
    const R = this._rng2 || (this._rng2 = [0, 0]);
    let pw = vs.passWreck[i];
    if (pw >= 0) {
      const W = laneW(e, si), back = Math.abs(vs.lat[i] - (vs.lane[i] + 0.5) * W) < 0.6 && vs.laneHi[i] >= vs.laneLo[i];
      if (vs.state[pw] !== STATE_CRASHED || vs.edge[pw] !== e.idx || (si - hl > vs.s[pw] + vs.hl[pw] + 2 && back)) { vs.passWreck[i] = -1; return; }
      lanes.range(er.idx, li, R);
      for (let k = R[0]; k < R[1]; k++) {
        const j = lanes.order[k], sj = e.length - vs.s[j];
        if (sj > si) { const gg = sj - si - hl - vs.hl[j] - 2; if (gg < vs.gap[i]) { vs.gap[i] = gg; vs.leader[i] = -3; } }
      }
      return;
    }
    const L = vs.leader[i];
    if (!this.opts.passWrecks) return;
    if (e.lanes !== 1 || er.lanes !== 1 || L < 0 || vs.state[L] !== STATE_CRASHED || vs.edge[L] !== e.idx || vs.conn[i] >= 0) return;
    if (vs.gap[i] > 15 || vs.speed[i] > 1 || vs.stoppedSince[i] < 0 || this.time - vs.stoppedSince[i] < 5) return;
    // the wrecks in a row ahead (within 15 m of each other) are passed together
    let k = L;
    for (let n = 0; n < 6; n++) { const q = lanes.ahead(k, vs.lane[k] >= 0 ? Math.min(vs.laneHi[k], Math.max(vs.laneLo[k], 0)) : 0); if (q < 0 || vs.state[q] !== STATE_CRASHED || vs.s[q] - vs.s[k] > 15) break; k = q; }
    const zoneEnd = vs.s[k] + vs.hl[k] + 8, zoneStart = si - hl - 5;
    const limit = e.sStop !== undefined && e.sStop < e.length ? e.sStop : e.length;
    if (zoneEnd + 10 > limit) return;
    const Tp = (zoneEnd - zoneStart) / 4 + 5;
    lanes.range(er.idx, li, R);
    for (let q = R[0]; q < R[1]; q++) {
      const j = lanes.order[q], sj = e.length - vs.s[j];
      if (sj > zoneStart && sj - zoneEnd < vs.speed[j] * Tp + 25) return;
    }
    // nobody else from my side already in the opposing lane over the zone
    const mine = this._passers.get(e.idx);
    if (mine) for (const p of mine) if (p !== i && vs.s[p] > zoneStart - 10) return;
    vs.passWreck[i] = k; this.counters.wreckPasses = (this.counters.wreckPasses || 0) + 1;
    (this._passers.get(e.idx) || this._passers.set(e.idx, []).get(e.idx)).push(i);
  }

  // ---------------------------------------------------------------- observation helpers (brain)
  /** Nearest car ahead in the opposing (reverse edge's inner) lane: {gap m (Infinity none), closing m/s}. */
  oncomingObs(i, out = {}) {
    const vs = this.vs, g = this.graph, e = g.edges[vs.edge[i]];
    out.gap = Infinity; out.closing = 0;
    if (e.reverse < 0) return out;
    const er = g.edges[e.reverse], R = this._rng3 || (this._rng3 = [0, 0]);
    this.lanes.range(er.idx, er.lanes - 1, R);
    const si = vs.s[i], hl = vs.hl[i];
    for (let k = R[0]; k < R[1]; k++) {
      const j = this.lanes.order[k], sj = e.length - vs.s[j];
      if (sj <= si) continue;
      const gg = Math.max(0, sj - si - hl - vs.hl[j]);
      if (gg < out.gap) { out.gap = gg; out.closing = vs.speed[i] + (vs.state[j] === STATE_DRIVING ? vs.speed[j] : 0); }
    }
    return out;
  }
  /**
   * Junction conflicts ahead for slot i (its connector, or the one it takes from its lane at the next
   * mouth): {mouthDistance (front bumper to the stop point; 0 inside the box, Infinity none), inBox,
   * distance (m from my front bumper to the nearest conflict point still occupied / about to be),
   * timeToConflict (s for that other body to reach it; 0 = on it), stopped (that body is stationary)}.
   */
  junctionObs(i, out = {}) {
    const vs = this.vs, g = this.graph, J = g.J, e = g.edges[vs.edge[i]];
    out.mouthDistance = Infinity; out.inBox = vs.conn[i] >= 0; out.distance = Infinity; out.timeToConflict = Infinity; out.stopped = false;
    if (!J) return out;
    let m = vs.conn[i];
    if (m < 0) {
      const route = vs.routes[i], rp = vs.routePos[i];
      if (!(e.sStop < e.length) || !route || rp + 1 >= route.length) return out;
      m = J.movement(e.idx, vs.lane[i], route[rp + 1]); if (m < 0) return out;
    }
    const ti = vs.s[i] - J.mF[4 * m], hl = vs.hl[i];
    out.mouthDistance = Math.max(0, -ti - hl);
    for (const c of this._conflicts(m)) {
      const list = this._connCars && this._connCars.get(c.other); if (!list) continue;
      const di = c.s - ti - hl; if (di < -2 * hl) continue;
      for (const j of list) {
        const dj = c.sOther - (vs.s[j] - J.mF[4 * c.other]);
        if (dj < -vs.hl[j] - vs.hw[i] - 0.5) continue;
        const onIt = Math.abs(dj) < vs.hl[j] + vs.hw[i] + 0.5;
        if (di < out.distance) { out.distance = Math.max(0, di); out.timeToConflict = onIt ? 0 : dj / Math.max(0.1, vs.speed[j]); out.stopped = vs.state[j] !== STATE_DRIVING || vs.speed[j] < 0.5; }
      }
    }
    return out;
  }

  // ---------------------------------------------------------------- missed exits
  /**
   * Cars whose lane led them onto another out-edge than planned at a strict diverge (vehicle.integrate
   * records them in vs.missed): emit 'missedExit', re-plan from the edge actually taken to the same goal
   * and apply it (brain drivers through wrapper.setRoute), emit 'reroute'.
   */
  _missedExits() {
    const vs = this.vs, g = this.graph, M = vs.missed;
    for (const [i, m] of M) {
      if (vs.state[i] !== STATE_DRIVING) { M.delete(i); continue; }
      if (vs.edge[i] !== m.took) { if (vs.edge[i] !== m.from) M.delete(i); continue; }   // still approaching / in the box: later
      M.delete(i);
      this.counters.missedExits = (this.counters.missedExits || 0) + 1;
      this._event('missedExit', vs.x[i], vs.z[i], [i], { from: m.from, planned: m.planned, took: m.took });
      const took = g.edges[m.took];
      const path = took.to === m.goal ? new Int32Array(0) : this.router(took.to, m.goal);
      if (!path) { this.counters.rerouteFailed = (this.counters.rerouteFailed || 0) + 1; continue; }
      const edges = new Int32Array(1 + path.length); edges[0] = m.took; edges.set(path, 1);
      const sOn = vs.s[i];
      this._setRoute(i, edges, 0); vs.routeDist[i] = sOn;
      if (vs.targetLane[i] >= 0) vs.targetLane[i] = vs.lane[i];
      this.counters.reroutes = (this.counters.reroutes || 0) + 1;
      const drv = this.drivers[i];
      if (drv && drv.setRoute) { try { drv.setRoute(vs.routes[i], this._stopInfo(i), sOn); } catch (err) { /* keep going */ } }
      this._event('reroute', vs.x[i], vs.z[i], [i], { reason: 'missedExit', from: m.took });
    }
  }

  // ---------------------------------------------------------------- live traffic (nav-app knowledge)
  /**
   * Every 60 s: mean speed of the driving cars on each edge, smoothed (alpha 0.5, ~2-minute moving
   * average). sim.trafficPenalty = Map(edge -> extra seconds over free flow) for edges observed below
   * 70 % of their limit: what a nav app's traffic layer knows. Used by re-routing (the sim's router
   * and the brain's Router.route penalty option).
   */
  _trafficSample() {
    const vs = this.vs, g = this.graph, sum = new Map(), cnt = new Map();
    for (let i = 0; i < vs.high; i++) {
      if (vs.state[i] !== STATE_DRIVING) continue;
      const e = vs.edge[i]; sum.set(e, (sum.get(e) || 0) + vs.speed[i]); cnt.set(e, (cnt.get(e) || 0) + 1);
    }
    const ema = this._edgeSpeed || (this._edgeSpeed = new Map());
    for (const [e, v] of ema) if (!cnt.has(e)) { const lim = g.edges[e].speedLimit, nv = 0.5 * v + 0.5 * lim; if (nv > 0.95 * lim) ema.delete(e); else ema.set(e, nv); }
    for (const [e, sv] of sum) { const m = sv / cnt.get(e), old = ema.get(e); ema.set(e, old === undefined ? m : 0.5 * old + 0.5 * m); }
    const pen = this.trafficPenalty || (this.trafficPenalty = new Map()); pen.clear();
    for (const [e, v] of ema) {
      const E = g.edges[e]; if (v >= 0.7 * E.speedLimit) continue;
      pen.set(e, E.length / Math.max(0.5, v) - E.length / E.speedLimit);
    }
  }

  // ---------------------------------------------------------------- re-routing around closures
  /**
   * Drivers who perceive a closed road on their route ahead (an active crash / stranded scene
   * blocking every lane of an edge within rerouteSight m) re-plan from the end of their current
   * edge, avoiding every currently closed edge. At most once per 120 s per car; not once committed
   * to the junction movement (past the mouth's stop point minus 5 m, or on a connector).
   */
  _rerouteBlocked() {
    const vs = this.vs, g = this.graph, closed = new Set();
    for (const r of this.activeCrashes) {
      if (r.responseState !== 'active' || r.squeeze) continue;
      const e = g.edges[r.edge]; let all = true;
      for (let l = 0; l < e.lanes; l++) if (!((r.laneMask >> l) & 1)) { all = false; break; }
      if (all) closed.add(r.edge);
    }
    const pen = this.trafficPenalty && this.trafficPenalty.size ? this.trafficPenalty : null;
    if (!closed.size && !pen) return;
    const last = this._lastReroute || (this._lastReroute = new Map()), sight = this.opts.rerouteSight;
    let budget = 200;
    for (let i = 0; i < vs.high && budget > 0; i++) {
      if (vs.state[i] !== STATE_DRIVING || vs.conn[i] >= 0 || vs.refuelState[i] !== 0) continue;
      const route = vs.routes[i], rp = vs.routePos[i]; if (!route || rp + 1 >= route.length) continue;
      const e = g.edges[route[rp]];
      const stopAt = e.sStop !== undefined && e.sStop < e.length ? e.sStop : e.length;
      if (vs.s[i] + vs.hl[i] > stopAt - 5) continue;
      let d = e.length - vs.s[i], hit = false;
      for (let k = rp + 1; k < route.length && d < sight; k++) { if (closed.has(route[k])) { hit = true; break; } d += g.edges[route[k]].length; }
      // stopped for 90 s in slow traffic: check the nav app for a faster way (at most every 5 min)
      const slow = !hit && pen && vs.stoppedSince[i] >= 0 && this.time - vs.stoppedSince[i] > 90;
      if (!hit && !slow) continue;
      const lt = last.get(i);
      if (lt !== undefined && lt.t > vs.spawnTime[i] && this.time - lt.t < (hit ? 120 : 300)) continue;
      last.set(i, { t: this.time });
      budget--;
      const goal = g.edges[route[route.length - 1]].to;
      if (e.to === goal) continue;
      const path = this.router(e.to, goal, closed.size ? closed : null, pen);
      if (slow && path) {
        // only switch when the new way is clearly faster with the current traffic (> 60 s)
        const cost = (p) => { let t = 0; for (const x of p) { const E = g.edges[x]; t += E.length / E.speedLimit + (pen.get(x) || 0); } return t; };
        if (cost(path) > cost(route.subarray(rp + 1)) - 60) continue;
      }
      if (!path || !path.length || path.some(x => closed.has(x))) { this.counters.rerouteFailed = (this.counters.rerouteFailed || 0) + 1; continue; }
      const edges = new Int32Array(1 + path.length); edges[0] = route[rp]; edges.set(path, 1);
      const sOn = vs.s[i];
      this._setRoute(i, edges, 0);
      vs.routeDist[i] = sOn;
      if (vs.targetLane[i] >= 0) vs.targetLane[i] = vs.lane[i];
      this.counters.reroutes = (this.counters.reroutes || 0) + 1;
      const drv = this.drivers[i];
      if (drv && drv.setRoute) { try { drv.setRoute(vs.routes[i], this._stopInfo(i), sOn); } catch (err) { /* keep going */ } }
      this._event('reroute', vs.x[i], vs.z[i], [i], { reason: hit ? 'closure' : 'traffic', avoid: [...closed].filter(x => route.includes(x)) });
    }
    if (last.size > 100000) last.clear();
  }

  // ---------------------------------------------------------------- junctions (src/geom junctions.js)
  /** Cars (driving or wrecked) on each junction connector this step. */
  _indexConnectors() {
    const vs = this.vs, M = this._connCars || (this._connCars = new Map());
    if (M.size > 50000) M.clear(); else for (const l of M.values()) l.length = 0;
    const NC = this._nodeConn || (this._nodeConn = new Map()), J = this.graph.J, E = this.graph.edges;
    if (NC.size > 50000) NC.clear(); else for (const l of NC.values()) l.length = 0;
    for (let i = 0; i < vs.high; i++) {
      const m = vs.conn[i];
      if (m < 0 || (vs.state[i] !== STATE_DRIVING && vs.state[i] !== STATE_CRASHED)) continue;
      let l = M.get(m); if (!l) M.set(m, (l = [])); l.push(i);
      const node = E[J.mInfo[7 * m]].to;
      let ln = NC.get(node); if (!ln) NC.set(node, (ln = [])); ln.push(i);
    }
  }

  /**
   * Junction perception for slot i (on a connector, or approaching the mouth of one), body-based:
   *  1. any stopped / crawling body (any edge, wrecks included) that my swept path through the box
   *     touches: an obstacle at that point of my path;
   *  2. cars on movements that conflict with mine (J.conflictsOf): a merge = follow whichever is
   *     nearer the merge point; a crossing = stop short of it while the other body is on it, or
   *     when the other car reaches it first (and I can still stop). Before the mouth I stop at the
   *     mouth, never inside the box. Crossing points are stationary obstacles (leader -3).
   */
  _junction(i) {
    const vs = this.vs, g = this.graph, J = g.J, e = g.edges[vs.edge[i]];
    const hl = vs.hl[i], v = vs.speed[i];
    let m = vs.conn[i];
    if (m < 0) {
      if (!(e.sStop < e.length)) return;
      const route = vs.routes[i], rp = vs.routePos[i];
      if (rp + 1 >= route.length || (vs.refuelState[i] === 1 && vs.stopPos[i] === rp)) return;
      if (e.sStop - vs.s[i] - hl > Math.max(30, v * v / 4 + 15)) return;
      m = J.movement(e.idx, vs.lane[i], route[rp + 1]); if (m < 0) return;
    }
    const sIn = J.mF[4 * m], len = J.mLen[m], ti = vs.s[i] - sIn;   // ti < 0: before the mouth
    const myNode = g.edges[J.mInfo[7 * m]].to;
    const stuckIn = (k) => vs.speed[k] < 0.3 && vs.stoppedSince[k] >= 0 && this.time - vs.stoppedSince[k] > 20;
    const toMouth = -ti - hl - 0.5;                                  // front bumper to the stop point
    let best = vs.gap[i], who = vs.leader[i], whoJ = -1;
    // 1. stationary / crawling bodies on my swept path
    const t0 = Math.max(0, ti + hl);
    if (t0 < len) {
      const P = this._jp || (this._jp = {}), cand = this._jc || (this._jc = []);
      cand.length = 0;
      J.arcPoint(m, 0.5 * (t0 + len), P);
      this.hash.forEachNear(P.x, P.z, 0.5 * (len - t0) + 8, (j) => {
        // bodies at walking pace or slower (stationary obstacles), and slow-moving bodies in the box too: paths
        // that pass within a car width of each other are not always in the conflict matrix (2 m centre rule)
        // (and slow movers on the connectors of a neighbouring junction box: those are not in my conflict matrix)
        const otherBox = vs.conn[j] >= 0 && g.edges[J.mInfo[7 * vs.conn[j]]].to !== myNode;
        if (j === i || !(vs.state[j] === STATE_CRASHED || vs.speed[j] < 2 || (otherBox && vs.speed[j] < 7))) return;
        if (vs.edge[j] === vs.edge[i] && vs.s[j] < vs.s[i] && vs.state[j] === STATE_DRIVING) return;   // behind me on my own approach (hairpin paths fold back past it)
        // a stopped car that is itself waiting for me: whoever is already in the box (else the lower slot) goes
        if (vs.state[j] === STATE_DRIVING && vs.leader[j] === i) { const iIn = vs.conn[i] >= 0, jIn = vs.conn[j] >= 0; if ((iIn && !jIn) || (iIn === jIn && i < j)) return; }
        // two cars both stuck > 20 s inside the box blocking each other's paths through longer chains (three-way
        // gridlock): the lower slot creeps on (at walking pace a touch is harmless, a gridlock is not)
        if (vs.state[j] === STATE_DRIVING && vs.conn[i] >= 0 && vs.conn[j] >= 0 && i < j && stuckIn(i) && stuckIn(j)) return;
        cand.push(j);
      });
      if (cand.length) {
        const clear = vs.hw[i] + 0.4, off = vs.lat[i] - (J.mInfo[7 * m + 1] + 0.5) * laneW(e, sIn);
        for (let t = t0; t <= len + 0.01; t += 1) {
          J.arcPoint(m, t, P);
          const px = P.x + off * P.hz, pz = P.z - off * P.hx;
          let hit = -1;
          for (let k = 0; k < cand.length; k++) {
            const j = cand[k], c = Math.cos(vs.heading[j]), sn = Math.sin(vs.heading[j]);
            const dx = px - vs.x[j], dz = pz - vs.z[j];
            const lx = Math.abs(dx * c + dz * sn) - vs.hl[j], lz = Math.abs(dz * c - dx * sn) - vs.hw[j];
            if (lx < clear && lz < clear && Math.hypot(Math.max(lx, 0), Math.max(lz, 0)) < clear) { hit = j; break; }
          }
          if (hit >= 0) { const gg = ti + hl < 0 ? Math.min(toMouth, t - ti - hl - 0.5) : t - ti - hl - 0.5; if (gg < best) { best = gg; who = hit; } break; }
        }
      }
    }
    // 2. conflicting movements
    const M = this._connCars;
    for (const c of this._conflicts(m)) {
      const list = M.get(c.other); if (!list || !list.length) continue;
      const di = c.s - ti;
      if (di < -hl - 1) continue;                                     // I am past it
      for (const j of list) {
        if (j === i) continue;
        // it is itself waiting for me (I am its leader / the conflict it yields to): I go
        if (vs.speed[j] < 0.5 && (vs.leader[j] === i || (vs.leader[j] === -3 && this._conflictWith && this._conflictWith.get(j) === i))) continue;
        if (vs.state[j] === STATE_DRIVING && vs.conn[i] >= 0 && i < j && stuckIn(i) && stuckIn(j)) continue;   // box gridlock: see the swept check
        const tj = vs.s[j] - J.mF[4 * c.other], dj = c.sOther - tj, hlj = vs.hl[j];
        if (c.type === 1) {                                           // merge: follow whoever is nearer the merge point
          if (dj < di || (dj === di && j < i)) { const gg = di - dj - hl - hlj - 1; if (gg < best) { best = gg; who = j; } }
          continue;
        }
        if (dj < -hlj - vs.hw[i] - 0.5) continue;                      // it has cleared the crossing
        // side by side from the same approach: the car turning from the wrong lane gives way to the one in its lane
        if (J.mInfo[7 * m] === J.mInfo[7 * c.other]) {
          const myPref = J.mInfo[7 * m + 6], hisPref = J.mInfo[7 * c.other + 6];
          if (myPref && !hisPref) continue;
          if (!myPref && hisPref) { const g2 = Math.min(di - hl - vs.hw[j] - 1.5, ti + hl < 0 ? toMouth : 1e9); if (g2 < best) { best = g2; who = -3; whoJ = j; } continue; }
        }
        let gStop = di - hl - vs.hw[j] - 1.5;
        if (ti + hl < 0) gStop = Math.min(gStop, toMouth);             // not entered yet: wait at the mouth
        if (dj < hlj + vs.hw[i] + 0.5) {                               // its body is on the crossing
          // both bodies already on it (entered together): the one with less left to clear goes on
          if (di < hl + vs.hw[j] + 0.5) { const clearI = di + hl + vs.hw[j], clearJ = dj + hlj + vs.hw[i]; if (clearI < clearJ || (clearI === clearJ && i < j)) continue; }
          if (gStop < best) { best = gStop; who = -3; whoJ = j; }
          continue;
        }
        const tI = di / Math.max(v, 1), tJ = dj / Math.max(vs.speed[j], 1);
        if ((tJ < tI || (tJ === tI && j < i)) && gStop > v * v / 8) { if (gStop < best) { best = gStop; who = -3; whoJ = j; } }
      }
    }
    if (best < vs.gap[i]) { vs.gap[i] = best; vs.leader[i] = who; if (who === -3) { const cw = this._conflictWith || (this._conflictWith = new Map()); cw.set(i, whoJ); } }
  }

  /**
   * Cars on a strict carriageway (no connector of their own) near a junction node: bodies on that node's
   * connectors (turning in / out, wrecks in the box) that lie ahead of me within my body width are
   * obstacles (moving the same way: followed; otherwise a stationary conflict point, leader -3).
   */
  _nodeBodies(i, e) {
    const vs = this.vs, g = this.graph, J = g.J, NC = this._nodeConn;
    const hl = vs.hl[i], hw = vs.hw[i];
    const check = (node) => {
      const list = NC.get(node); if (!list || !list.length) return;
      const yaw = vs.heading[i], fx = Math.cos(yaw), fz = Math.sin(yaw);
      for (const j of list) {
        if (j === i) continue;
        // it is itself stopped waiting for me (I am its leader / its conflict): I move on and clear its path
        if (vs.state[j] === STATE_DRIVING && vs.speed[j] < 0.5 && (vs.leader[j] === i || (vs.leader[j] === -3 && this._conflictWith.get(j) === i))) continue;
        const dx = vs.x[j] - vs.x[i], dz = vs.z[j] - vs.z[i];
        const along = dx * fx + dz * fz; if (along <= 0 || along > 60) continue;
        const lat = dx * fz - dz * fx;
        let dpsi = vs.heading[j] - yaw; dpsi = Math.atan2(Math.sin(dpsi), Math.cos(dpsi));
        const sa = Math.abs(Math.sin(dpsi)), ca = Math.abs(Math.cos(dpsi));
        if (Math.abs(lat) > hw + sa * vs.hl[j] + ca * vs.hw[j] + 0.3) continue;
        const gg = along - hl - (ca * vs.hl[j] + sa * vs.hw[j]) - 0.5;
        if (gg < vs.gap[i]) { vs.gap[i] = gg; vs.leader[i] = (Math.abs(dpsi) < 0.5 && vs.state[j] === STATE_DRIVING) ? j : -3; if (vs.leader[i] === -3) this._conflictWith.set(i, j); }
      }
    };
    if (vs.s[i] - hl < 40 && e.sExit > 0) check(e.from);
    if (e.length - vs.s[i] < 60 && e.sStop < e.length) {
      const route = vs.routes[i], rp = vs.routePos[i];
      if (!route || rp + 1 >= route.length || J.movement(e.idx, vs.lane[i], route[rp + 1]) < 0) check(e.to);
    }
  }

  /**
   * Converging carriageways (a ramp joining a mainline, two roads merging at a strict node): bodies on the
   * other in-edges of my end node that continue onto my next edge, ahead of me in my heading frame and
   * within my body width, are followed. Pavements converge before the node, so lane bookkeeping (which edge,
   * which lane) can miss them; the bodies cannot.
   */
  _convergeBodies(i, e) {
    const vs = this.vs, g = this.graph, lanes = this.lanes;
    const dNode = e.length - vs.s[i]; if (dNode > 150) return;
    const route = vs.routes[i], rp = vs.routePos[i]; if (!route || rp + 1 >= route.length) return;
    const next = route[rp + 1], node = e.to;
    if (g.inStart[node + 1] - g.inStart[node] < 2 || !hopMap(g, e, g.edges[next]).strict) return;   // carriageway merges (no junction box)
    const yaw = vs.heading[i], fx = Math.cos(yaw), fz = Math.sin(yaw), hl = vs.hl[i], hw = vs.hw[i];
    for (let q = g.inStart[node]; q < g.inStart[node + 1]; q++) {
      const E2 = g.edges[g.inEdges[q]]; if (E2.idx === e.idx || E2.idx === g.edges[next].reverse) continue;
      for (let l = 0; l < E2.lanes; l++) {
        let j = lanes.last(E2.idx, l), n = 0;
        while (j >= 0 && n++ < 4) {
          const rj = vs.routes[j], pj = vs.routePos[j];
          if (E2.length - vs.s[j] > dNode + 60) break;
          // a car waiting at its own yield / stop line (or red) has not entered the merged carriageway
          const waitingAtLine = vs.state[j] === STATE_DRIVING && vs.speed[j] < 1 && vs.ctlType[j] >= 0 && vs.ctlDist[j] > -0.5 && vs.ctlDist[j] < 15 && (vs.ctlState[j] === 2 || vs.ctlState[j] === 3);
          if (!waitingAtLine && (vs.state[j] === STATE_CRASHED || (rj && pj + 1 < rj.length && rj[pj + 1] === next && hopMap(g, E2, g.edges[next]).strict))) {
            const dx = vs.x[j] - vs.x[i], dz = vs.z[j] - vs.z[i];
            const along = dx * fx + dz * fz;
            if (along > 0 && along < 80) {
              const lat = dx * fz - dz * fx;
              let dpsi = vs.heading[j] - yaw; dpsi = Math.atan2(Math.sin(dpsi), Math.cos(dpsi));
              const sa = Math.abs(Math.sin(dpsi)), ca = Math.abs(Math.cos(dpsi));
              if (Math.abs(dpsi) < 0.5 && Math.abs(lat) < hw + sa * vs.hl[j] + ca * vs.hw[j] + 0.4) {   // converging, same direction
                const gg = along - hl - (ca * vs.hl[j] + sa * vs.hw[j]);
                if (gg < vs.gap[i]) { vs.gap[i] = gg; vs.leader[i] = j; }
              }
            }
          }
          j = lanes.behind(j, l);
        }
      }
    }
  }

  /**
   * Conflicts of movement m: the world's conflict matrix (paths within 3 m) plus any other movement of the
   * same junction whose centre path passes within 4.2 m of mine (two car bodies at an angle need about a
   * half-length plus a half-width; the 3 m rule misses e.g. a roundabout's exit passing an entry mouth).
   * The extra ones are crossings at the closest approach. Cached per movement.
   */
  _conflicts(m) {
    const cache = this._confCache || (this._confCache = new Map());
    let c = cache.get(m); if (c) return c;
    const J = this.graph.J, base = J.conflictsOf(m), known = new Set(base.map(x => x.other));
    c = base.slice();
    const node = this.graph.edges[J.mInfo[7 * m]].to, inE = J.mInfo[7 * m], inL = J.mInfo[7 * m + 1];
    const P = J.pts, f1 = J.mInfo[7 * m + 4], n1 = J.mInfo[7 * m + 5];
    for (const o of J.movementsAt(node)) {
      if (o === m || known.has(o) || (J.mInfo[7 * o] === inE && J.mInfo[7 * o + 1] === inL)) continue;
      const f2 = J.mInfo[7 * o + 4], n2 = J.mInfo[7 * o + 5];
      let best = 1e9, sa = 0, sb = 0;
      for (let a = 0; a < n1; a++) for (let b = 0; b < n2; b++) {
        const d = Math.hypot(P[4 * (f1 + a)] - P[4 * (f2 + b)], P[4 * (f1 + a) + 2] - P[4 * (f2 + b) + 2]);
        if (d < best) { best = d; sa = P[4 * (f1 + a) + 3]; sb = P[4 * (f2 + b) + 3]; }
      }
      if (best < 4.2) c.push({ other: o, type: 0, s: sa, sOther: sb });
    }
    if (cache.size > 200000) cache.clear();
    cache.set(m, c);
    return c;
  }

  /** A sign (stop / give way) governs slot j's approach to the junction ahead. */
  _signed(j) { const vs = this.vs; return (this.vs.ctlType[j] === 1 || vs.ctlType[j] === 2) && vs.ctlDist[j] < 60; }
  /** Slot j's own control makes it stop before the junction ahead (red, or a stop sign not yet stopped at). */
  _mustStop(j) {
    const vs = this.vs; if (vs.ctlType[j] < 0 || vs.ctlDist[j] > 60) return false;
    return vs.ctlState[j] === 2 || (vs.ctlType[j] === 1 && !vs.ctlStopped[j]);
  }

  /**
   * Junction entry decision for slot i approaching the mouth of e (src/geom junctions): must I
   * wait at the stop point? Yes if (a) a body on a conflicting movement is in the box and has
   * not cleared our conflict point, (b) a car approaching on a conflicting movement has priority
   * (sign vs no sign, road rank, a left turn gives way, else first to arrive) and reaches the
   * conflict soon, or (c) there is no room beyond the box for my whole length. No time-outs:
   * a car still in the box is always seen.
   */
  _yieldJ(i, e) {
    const vs = this.vs, g = this.graph, J = g.J;
    if (vs.conn[i] >= 0 || !(e.sStop < e.length)) return false;
    const route = vs.routes[i], rp = vs.routePos[i];
    if (rp + 1 >= route.length) return false;
    const m = J.movement(e.idx, vs.lane[i], route[rp + 1]); if (m < 0) return false;
    const hl = vs.hl[i], dS = Math.max(0, Math.min(e.sStop, J.mF[4 * m]) - vs.s[i] - hl), v = vs.speed[i];
    const L0 = vs.leader[i];
    let yb = this._yieldBy; if (!yb || yb.length < vs.capacity) { const n = new Int32Array(vs.capacity).fill(-1); if (yb) n.set(yb); yb = this._yieldBy = n; }
    yb[i] = -1;
    const W = (c, j) => { yb[i] = j ?? -1; if (this.opts.debugYield) (this._yieldWhy || (this._yieldWhy = new Map())).set(i, [c, j]); return true; };
    // don't block the box: a slow / stopped car beyond it must leave room for my whole length
    if (L0 >= 0 && vs.state[L0] === STATE_DRIVING && vs.speed[L0] < 1 && (vs.edge[L0] !== e.idx || vs.conn[L0] >= 0) && vs.gap[i] - dS - J.mLen[m] < vs.length[i] + 1) return W('no-room', L0);
    const left = J.mF[4 * m + 3] < -0.5, myStop = this._signed(i), M = this._connCars, lanes = this.lanes, sight = this.opts.junctionSight >= 1;
    for (const c of this._conflicts(m)) {
      const o = c.other;
      const list = M.get(o);
      if (list && c.type === 0) for (const j of list) if (vs.s[j] - J.mF[4 * o] - vs.hl[j] < c.sOther + vs.hw[i] + 1) return W('in-box', j);
      const oIn = J.mInfo[7 * o], oLane = J.mInfo[7 * o + 1], oOut = J.mInfo[7 * o + 2];
      const E2 = g.edges[oIn];
      // does approaching car j (d2 m from its mouth) have priority over me and reach our conflict soon?
      const first = (j, d2) => {
        const v2 = vs.speed[j];
        if (d2 > 80 || this._mustStop(j)) return false;
        // it is standing at its mouth held by an obstacle right in front of it (a queue / wreck it cannot pass,
        // or no room beyond the box): it cannot enter, so it is no conflict until it moves
        if (v2 < 0.3 && vs.leader[j] !== -1 && vs.gap[j] < d2 + 3) return false;
        const tI = (dS + c.s) / Math.max(v, 2), tJ = (d2 + c.sOther) / Math.max(v2, 2);
        let jFirst;
        const jStop = this._signed(j);
        if (myStop !== jStop) jFirst = myStop;                         // a sign gives way to the uncontrolled road
        else if (E2.rank !== e.rank) jFirst = E2.rank > e.rank;
        else if (c.type === 0 && left !== (J.mF[4 * o + 3] < -0.5)) jFirst = left;   // a left turn gives way
        else jFirst = tJ < tI || (tJ === tI && j < i);
        return jFirst && (tJ < 8 || d2 < 10);
      };
      if (!sight) {
        let j = lanes.last(oIn, oLane);
        while (j >= 0 && vs.conn[j] >= 0) j = lanes.behind(j, oLane);
        if (j < 0 || j === i || vs.state[j] !== STATE_DRIVING) continue;
        const rj = vs.routes[j], pj = vs.routePos[j];
        if (pj + 1 >= rj.length || rj[pj + 1] !== oOut || (vs.refuelState[j] === 1 && vs.stopPos[j] === pj)) continue;
        if (first(j, Math.max(0, E2.sStop - vs.s[j] - vs.hl[j]))) return W('approach', j);
        continue;
      }
      // junctionSight 1: the cars a driver can see coming, not only the front car of that lane: the first few
      // cars of the lane that take movement o (a car ahead of them turning elsewhere does not hide them), and,
      // when the approach is short (clustered junctions: divided roads, offset crossings), the cars still
      // upstream of it - on the connectors of the previous junction that lead onto it, and on their approaches
      if (this._seeApproach(i, oIn, oLane, oOut, first)) return W('approach+', this._seenJ);
    }
    // through traffic on a continuous carriageway (strict hop, no connector) crossing the box: it has no
    // connector in the conflict matrix, so judge it directly. In the box (past its mouth) = always wait;
    // approaching with priority (higher / equal rank road, no sign for it, not stopping) = wait if it is near.
    const node = e.to, myOut = route[rp + 1];
    for (let q = g.inStart[node]; q < g.inStart[node + 1]; q++) {
      const o = g.edges[g.inEdges[q]];
      if (o.idx === e.idx || o.idx === g.edges[myOut].reverse) continue;
      for (let l = 0; l < o.lanes; l++) {
        let j = lanes.last(o.idx, l), n = 0;
        while (j >= 0 && n++ < 3) {
          if (vs.state[j] === STATE_DRIVING && vs.conn[j] < 0) {
            const rj = vs.routes[j], pj = vs.routePos[j];
            if (rj && pj + 1 < rj.length && rj[pj + 1] !== myOut && J.movement(o.idx, l, rj[pj + 1]) < 0) {
              const dj = o.length - vs.s[j] - vs.hl[j];
              if (vs.s[j] + vs.hl[j] > o.sStop && dj > -vs.length[j]) return W('through-in-box', j);           // its body is in the box
              if (o.rank >= e.rank && !this._signed(j) && !this._mustStop(j) && !myStop) {
                const tJ = (o.sStop - vs.s[j] - vs.hl[j]) / Math.max(vs.speed[j], 2);
                if (tJ < 6) return W('through-approach', j);
              }
            }
            break;
          }
          j = lanes.behind(j, l);
        }
      }
    }
    return false;
  }

  /**
   * junctionSight 1 (sim._yieldJ): cars approaching the mouth of movement o (in-edge oIn, lane oLane, out-edge
   * oOut) within 80 m, front to back; first(j, d2) is called with each one's distance to the mouth and stops the
   * search when it returns true (then true is returned). Up to 3 cars per lane that take o; upstream of a short
   * approach, the cars on the previous junction's connectors onto oIn (and their approach lanes) that continue
   * onto oOut. Cars in the box itself are handled by the caller.
   */
  _seeApproach(i, oIn, oLane, oOut, first) {
    const vs = this.vs, g = this.graph, J = g.J, lanes = this.lanes, E2 = g.edges[oIn], R = 80;
    const ok = (j, rp1) => j !== i && vs.state[j] === STATE_DRIVING && !(vs.refuelState[j] === 1 && vs.stopPos[j] === vs.routePos[j]) && vs.routes[j] && vs.routes[j][vs.routePos[j] + rp1] === oOut;
    let j = lanes.last(oIn, oLane), n = 0;
    while (j >= 0 && n < 3) {
      if (vs.conn[j] < 0) {
        const d2 = Math.max(0, E2.sStop - vs.s[j] - vs.hl[j]);
        if (d2 > R) return false;
        n++;
        if (ok(j, 1) && first(j, d2)) { this._seenJ = j; return true; }
      }
      j = lanes.behind(j, oLane);
    }
    // upstream: only when the whole approach lane is shorter than the sight range
    const sE = E2.sStop;
    if (!(sE < R)) return false;
    const ups = this._movesInto(oIn);
    for (const m2 of ups) {
      const list = this._connCars.get(m2), len2 = J.mLen[m2], sIn2 = J.mF[4 * m2], sOut2 = J.mF[4 * m2 + 1];
      const base = Math.max(0, sE - sOut2);              // from the end of connector m2 to the mouth of oIn
      if (list) for (const k of list) {
        const d2 = base + Math.max(0, sIn2 + len2 - vs.s[k]) - vs.hl[k];
        if (d2 <= R && ok(k, 2) && first(k, Math.max(0, d2))) { this._seenJ = k; return true; }
      }
      // cars on the approach of m2, short of its mouth
      const inE = J.mInfo[7 * m2], inL = J.mInfo[7 * m2 + 1];
      let q = lanes.last(inE, inL), c = 0;
      while (q >= 0 && c < 2) {
        if (vs.conn[q] < 0) {
          const d2 = base + len2 + Math.max(0, sIn2 - vs.s[q]) - vs.hl[q];
          if (d2 > R) break;
          c++;
          if (vs.routes[q] && vs.routes[q][vs.routePos[q] + 1] === oIn && ok(q, 2) && !this._mustStop(q) && first(q, d2)) { this._seenJ = q; return true; }
        }
        q = lanes.behind(q, inL);
      }
    }
    return false;
  }
  /** Movements whose out-edge is e (cached). */
  _movesInto(e) {
    let M = this._mvInto;
    if (!M) {
      M = this._mvInto = new Map(); const J = this.graph.J;
      for (let m = 0; m < J.moveCount; m++) { const k = J.mInfo[7 * m + 2]; let l = M.get(k); if (!l) M.set(k, (l = [])); l.push(m); }
    }
    return M.get(e) || [];
  }

  /**
   * pathSight 1 (docs/scale-crashes.md): a driver sees any body physically on the path it is about to drive,
   * whatever edge, lane list or junction box that body is booked on. The path is sampled every 2 m from my
   * front bumper up to the stopping sight distance (current lateral offset on my edge, then the connector of
   * my movement and the lane centres of the next route edges). The first body my swept width touches is a
   * leader: followed when it moves my way, else a stationary conflict point (leader -3). Catches what the
   * lane / junction bookkeeping misses: overlapping pavements where carriageways converge, connector paths
   * that run over another approach, wrecks and stopped cars in a junction box beyond a short edge.
   */
  _pathBodies(i) {
    const vs = this.vs, g = this.graph, J = g.J, v = vs.speed[i];
    const D = Math.min(80, 10 + 1.5 * v + v * v / 8), hl = vs.hl[i], hw = vs.hw[i];
    const yaw = vs.heading[i], fx = Math.cos(yaw), fz = Math.sin(yaw);
    const cand = this._pbC || (this._pbC = []); cand.length = 0;
    const stuck = (k) => vs.speed[k] < 0.3 && vs.stoppedSince[k] >= 0 && this.time - vs.stoppedSince[k] > 20;
    const cw = this._conflictWith;
    this.hash.forEachNear(vs.x[i] + fx * 0.5 * D, vs.z[i] + fz * 0.5 * D, 0.5 * D + 9, (j) => {
      if (j === i) return;
      const st = vs.state[j]; if (st !== STATE_DRIVING && st !== STATE_CRASHED) return;
      const dx = vs.x[j] - vs.x[i], dz = vs.z[j] - vs.z[i];
      if (dx * fx + dz * fz < hl) return;                                   // beside or behind my front
      if (st === STATE_DRIVING && vs.speed[j] < 0.5 && (vs.leader[j] === i || (vs.leader[j] === -3 && cw && cw.get(j) === i))) return;   // waiting for me
      if (st === STATE_DRIVING && i < j && stuck(i) && stuck(j)) return;   // mutual standstill: the lower slot creeps (as _junction)
      cand.push(j);
    });
    if (!cand.length) return;
    // walk my path
    const route = vs.routes[i]; let rp = vs.routePos[i];
    let e = g.edges[vs.edge[i]], m = vs.conn[i], pos, end, off, lat = vs.lat[i], lane = vs.lane[i];
    if (m >= 0) { pos = vs.s[i] - J.mF[4 * m]; end = J.mLen[m]; off = lat - (J.mInfo[7 * m + 1] + 0.5) * laneW(e, J.mF[4 * m]); }
    else {
      pos = vs.s[i]; end = e.length;
      if (J && e.sStop < e.length && route && rp + 1 < route.length) { const mm = J.movement(e.idx, lane, route[rp + 1]); if (mm >= 0) end = J.mF[4 * mm]; }
    }
    const P = this._pbP || (this._pbP = {}), clear = hw + 0.3;
    let d = 0, t = hl + 0.5, frames = 0;
    // advance the frame until the sample at path distance t is inside it
    for (;;) {
      if (t > D) return;
      const rest = end - pos;
      if (d + rest < t) {                                                 // next frame
        d += Math.max(0, rest);
        if (++frames > 6 || !route) return;
        if (m >= 0) {                                                     // connector -> its out edge
          const oe = J.mInfo[7 * m + 2], ol = J.mInfo[7 * m + 3]; rp++; e = g.edges[oe]; pos = J.mF[4 * m + 1]; m = -1;
          lane = Math.min(e.lanes - 1, ol); lat = (lane + 0.5) * laneW(e, pos); end = e.length;
          if (J && e.sStop < e.length && rp + 1 < route.length) { const mm = J.movement(e.idx, lane, route[rp + 1]); if (mm >= 0) end = J.mF[4 * mm]; }
          continue;
        }
        if (rp + 1 >= route.length) return;
        const mm = J && e.sStop < e.length ? J.movement(e.idx, lane, route[rp + 1]) : -1;
        if (mm >= 0) { m = mm; pos = 0; end = J.mLen[m]; off = lat - (J.mInfo[7 * m + 1] + 0.5) * laneW(e, J.mF[4 * m]); continue; }
        const en = g.edges[route[rp + 1]]; rp++;
        const nl = mapLane(g, e, en, lane); lane = Math.min(en.lanes - 1, Math.max(0, nl)); e = en; pos = 0; end = e.length; lat = (lane + 0.5) * laneW(e, 0);
        if (J && e.sStop < e.length && rp + 1 < route.length) { const m2 = J.movement(e.idx, lane, route[rp + 1]); if (m2 >= 0) end = J.mF[4 * m2]; }
        continue;
      }
      const u = pos + (t - d);
      if (m >= 0) { connectorPoint(J, m, u, P); P.x += off * P.hz; P.z -= off * P.hx; }
      else if (e.LG) e.LG.latPoint(e.idx, Math.min(u, e.length), lat, P); else return;
      for (let k = 0; k < cand.length; k++) {
        const j = cand[k], c = Math.cos(vs.heading[j]), sn = Math.sin(vs.heading[j]);
        const dx = P.x - vs.x[j], dz = P.z - vs.z[j];
        const lx = Math.abs(dx * c + dz * sn) - vs.hl[j], lz = Math.abs(dz * c - dx * sn) - vs.hw[j];
        if (lx < clear && lz < clear && Math.hypot(Math.max(lx, 0), Math.max(lz, 0)) < clear) {
          const gg = t - hl - 0.5;
          if (gg < vs.gap[i]) {
            let dpsi = vs.heading[j] - Math.atan2(P.hz, P.hx); dpsi = Math.atan2(Math.sin(dpsi), Math.cos(dpsi));
            vs.gap[i] = gg;
            if (vs.state[j] === STATE_DRIVING && Math.abs(dpsi) < 0.6) vs.leader[i] = j;
            else { vs.leader[i] = -3; (this._conflictWith || (this._conflictWith = new Map())).set(i, j); }
          }
          return;
        }
      }
      t += 2;
    }
  }

  /** On a green: is a body on a crossing movement still in the box, short of our conflict point? */
  _boxBusyJ(i, e) {
    const vs = this.vs, J = this.graph.J;
    if (vs.conn[i] >= 0 || !(e.sStop < e.length)) return false;
    const route = vs.routes[i], rp = vs.routePos[i];
    if (rp + 1 >= route.length) return false;
    const m = J.movement(e.idx, vs.lane[i], route[rp + 1]); if (m < 0) return false;
    for (const c of this._conflicts(m)) {
      if (c.type !== 0) continue;
      const list = this._connCars.get(c.other); if (!list) continue;
      for (const j of list) if (vs.s[j] - J.mF[4 * c.other] - vs.hl[j] < c.sOther + vs.hw[i] + 1) return true;
    }
    return false;
  }

  // ---------------------------------------------------------------- traffic controls (observation)
  /**
   * Next control on each driving car's route within controlLookahead m: approach, type,
   * distance from the front bumper to the stop line, light state / time in state / time to
   * change for the car's movement there. Logs signalViolation / stopViolation events when a
   * car crosses a stop line on red (more than 0.5 s into the red) or without stopping at a sign.
   */
  _controlObs() {
    const vs = this.vs, g = this.graph, C = this.controls, out = this._ctlOut, look = this.opts.controlLookahead;
    for (let i = 0; i < vs.high; i++) {
      if (vs.state[i] !== STATE_DRIVING) continue;
      const route = vs.routes[i], rp = vs.routePos[i], hl = 0.5 * vs.length[i];
      let a = -1, dist = 0, k = rp, e = g.edges[vs.edge[i]];
      // current edge (stop line still ahead of my front bumper, or just crossed: 1 m grace)
      // stop lines of the signalised intersection I am already inside (a divided road's inner stop
      // lines, part of the same controller) belong to the movement I entered on: skip them
      const inside = vs.ctlInside[i];
      const skip = (x) => inside >= 0 && C.clusterOf[C.apControl[x]] === inside;
      let ap = -1;
      for (let q = C.edgeApStart[e.idx]; q < C.edgeApStart[e.idx + 1]; q++) { const x = C.edgeApList[q]; if (C.apStopAt[x] - (vs.s[i] + hl) > -1 && !skip(x)) { ap = x; break; } }
      if (ap >= 0) { a = ap; dist = C.apStopAt[ap] - (vs.s[i] + hl); }
      else {
        let d = e.length - vs.s[i] - hl, s0 = 0;
        if (g.J && rp + 1 < route.length) {
          // through the junction connector (mine, or the one my lane takes): the next edge starts at sOut
          let m = vs.conn[i]; if (m < 0 && e.sStop < e.length) m = g.J.movement(e.idx, vs.lane[i], route[rp + 1]);
          if (m >= 0) { d = g.J.mF[4 * m] + g.J.mLen[m] - vs.s[i] - hl; s0 = g.J.mF[4 * m + 1]; }
        }
        for (k = rp + 1; k < route.length && d < look && a < 0; k++) {
          const en = g.edges[route[k]], off = k === rp + 1 ? s0 : 0;
          for (let q = C.edgeApStart[en.idx]; q < C.edgeApStart[en.idx + 1]; q++) { const x = C.edgeApList[q]; if (!skip(x)) { a = x; dist = d + C.apStopAt[x] - off; break; } }
          if (a >= 0) break;
          d += en.length - off;
        }
      }
      const prevA = vs.ctlApproach[i], prevDist = vs.ctlDist[i];
      // crossed the previous stop line this step?
      // the front bumper crossed the stop line this step (once per approach: prevDist >= 0 -> < 0 or moved on)
      if (prevA >= 0 && prevDist >= 0 && prevDist < 30 && (a !== prevA || dist < 0) && !(vs.ctlInside[i] >= 0 && C.clusterOf[C.apControl[prevA]] === vs.ctlInside[i])) {
        const st = vs.ctlState[i], ty = vs.ctlType[i];
        if ((ty === CTRL.SIGNAL || ty === CTRL.RAMP_METER || ty === CTRL.CROSSING) && st === LIGHT.RED && vs.ctlTimeIn[i] > 0.5 && !(vs.ctlMove[i] === MOVE.RIGHT && vs.ctlStopped[i])) {
          this.counters.signalViolations++; this._event('signalViolation', vs.x[i], vs.z[i], [i], { approach: prevA, control: 'red', controlId: C.apControl[prevA], node: g.edges[C.apEdge[prevA]].to, speed: vs.speed[i], redFor: vs.ctlTimeIn[i] });
        } else if (ty === CTRL.STOP && !vs.ctlStopped[i]) {
          this.counters.stopViolations++; this._event('signalViolation', vs.x[i], vs.z[i], [i], { approach: prevA, control: 'stop', controlId: C.apControl[prevA], node: g.edges[C.apEdge[prevA]].to, speed: vs.speed[i] });
        }
      }
      if (a !== prevA) {
        vs.ctlStopped[i] = 0;
        // crossed a signal's stop line: I am inside its intersection until I am 60 m past it
        if (prevA >= 0 && prevDist >= -1 && prevDist < 30 && C.type[C.apControl[prevA]] === CTRL.SIGNAL) { vs.ctlInside[i] = C.clusterOf[C.apControl[prevA]]; vs.ctlInsideT[i] = this.time; }
      }
      if (vs.ctlInside[i] >= 0 && this.time - vs.ctlInsideT[i] > 15) vs.ctlInside[i] = -1;
      vs.ctlApproach[i] = a;
      if (a < 0) { vs.ctlType[i] = -1; vs.ctlDist[i] = 1e9; continue; }
      const appE = g.edges[C.apEdge[a]];
      // my movement at that control: approach edge -> the route edge after it
      let mv = MOVE.THROUGH;
      const pos = appE.idx === vs.edge[i] ? rp : k;
      if (pos + 1 < route.length) mv = movementOf(appE, g.edges[route[pos + 1]]);
      C.stateOf(a, mv, this.time, out);
      vs.ctlType[i] = out.type; vs.ctlState[i] = out.state; vs.ctlTimeIn[i] = out.timeIn; vs.ctlTimeLeft[i] = out.timeLeft;
      vs.ctlDist[i] = dist; vs.ctlMove[i] = mv;
      if (dist < 10 && vs.speed[i] < 0.3) vs.ctlStopped[i] = 1; // the stop-line controller holds a stopped car with the line < 9.5 m ahead
    }
  }

  // ---------------------------------------------------------------- drivers
  /** Per-step signals for the driver of slot i (src/brain/README.md obs.signals). Reused object. */
  signalsFor(i) {
    const vs = this.vs, s = this._sig, c = this.commuters[vs.commuter[i]];
    s.collision = (vs.flags[i] & FLAG_COLLIDED) !== 0;
    // rewardVersion 2 (reward.js crashPenalty): this car's crash event of this step {dvKmh, share, disabled}, read once
    s.crash = null; if (this._crashSig && this._crashSig.has(i)) { s.crash = this._crashSig.get(i); this._crashSig.delete(i); }
    s.stranded = (vs.flags[i] & FLAG_STRANDED) !== 0;
    s.arrived = false; s.lateBy = undefined;
    s.harshBraking = vs.accel[i] < -4;
    s.offRoute = false; s.laneViolation = false; s.missedExit = false;
    s.timeToWorkStart = c ? (vs.leg[i] === 0 ? c.workStart - this.time : 1e5) : 0;
    s.remainingRouteDistance = Math.max(0, vs.routeTotal[i] - vs.routeDist[i]);
    s.fuelFraction = vs.tank[i] > 0 ? vs.fuel[i] / vs.tank[i] : 1;
    s.refuelPlanned = vs.refuelState[i] === 1;
    return s;
  }

  _drive(dt) {
    const vs = this.vs, high = vs.high, ctx = this.ctx;
    const lod2 = this.lodRadius === Infinity ? Infinity : this.lodRadius * this.lodRadius;
    const fx = this.focus.x, fz = this.focus.z, every = this.opts.brainEvery;
    let brainSteps = 0, learnerSteps = 0;
    for (let i = 0; i < high; i++) {
      if (vs.state[i] !== STATE_DRIVING) continue;
      const drv = this.drivers[i];
      let useBrain = drv !== null && drv !== undefined;
      if (useBrain && lod2 !== Infinity) { const dx = vs.x[i] - fx, dz = vs.z[i] - fz; useBrain = dx * dx + dz * dz <= lod2; }
      if (useBrain) {
        if (every > 1 && (this.stepCount + i) % every !== 0) continue; // keep last command (held in the ring)
        brainSteps++;
        let cmd;
        try { cmd = drv.step(dt * every, this.wq.bind(i)); }
        catch (err) { if (!this._brainErr) { this._brainErr = err; console.warn('driver.step failed; the car uses opts.defaultAction:', err); } this.drivers[i] = null; cmd = null; }
        // the brain asked for a new route around a blockage (edge ids starting with my current edge):
        // the sim owns routes; applied once I am not committed to a junction connector
        const rr = drv && drv.brain && drv.brain.rerouteRequest;
        if (rr && vs.conn[i] < 0) {
          drv.brain.rerouteRequest = null;
          if (rr.length > 1 && rr[0] === vs.edge[i] && vs.refuelState[i] === 0) {
            const sOn = vs.s[i];
            this._setRoute(i, Int32Array.from(rr), 0); vs.routeDist[i] = sOn;
            this.counters.reroutes = (this.counters.reroutes || 0) + 1;
            try { drv.setRoute(vs.routes[i], this._stopInfo(i), sOn); } catch (err) { /* keep going */ }
            this._event('reroute', vs.x[i], vs.z[i], [i], { reason: 'brain', from: vs.edge[i] });
          }
        }
        if (cmd) {
          vs.flags[i] &= ~FLAG_COLLIDED;
          if (cmd.throttle !== undefined || cmd.rawThrottle !== undefined) legacyCommand(vs, i, cmd, this.stepCount); // {throttle, steer} drivers (pre steering-only brain)
          else vs.command(i, steerCommand(vs, i, cmd, dt * every), +cmd.gas || 0, +cmd.brake || 0, this.stepCount);
          if (vs.flags[i] & FLAG_LEARNER) learnerSteps++;
          continue;
        }
      }
      const c = this.defaultAction;
      vs.command(i, c.steer, c.gas, c.brake, this.stepCount);
      if (this.onTransition && (vs.flags[i] & FLAG_LEARNER)) this.onTransition(i, this.signalsFor(i));
      vs.flags[i] &= ~FLAG_COLLIDED;
    }
    this.live.brainSteps = brainSteps; this.live.learnerSteps = learnerSteps;
  }

  // ---------------------------------------------------------------- motion
  _integrate(dt) {
    const vs = this.vs, high = vs.high, g = this.graph, fuelOn = this.opts.fuel;
    let driving = 0, crashed = 0, refueling = 0, sumSpeed = 0, speedingSec = 0, speedingKm = 0, totalSec = 0, totalKm = 0;
    const checkEvery = this.fuelCfg.checkInterval;
    for (let i = 0; i < high; i++) {
      const st = vs.state[i];
      if (st === STATE_CRASHED) {
        crashed++;
        // disabled cars slide to rest under tyre friction (wheels locked), then stay put
        if (vs.speed[i] > 0.01 || Math.abs(vs.vLat[i]) > 0.01 || Math.abs(vs.yawRate[i]) > 0.01) {
          vs.savePose(i); vs.integrate(i, dt, this.time, g, this.stepCount, true); vs.updatePose(i, g, this._pose);
        } else vs.savePose(i);
        continue;
      }
      if (st === STATE_REFUELING) { refueling++; continue; }
      if (st !== STATE_DRIVING) continue;
      if (!fuelOn) vs.fuel[i] = vs.tank[i];
      vs.savePose(i);
      let done = vs.integrate(i, dt, this.time, g, this.stepCount);
      const lim = g.edges[vs.edge[i]].speedLimit;
      if (vs.speed[i] > 1.1 * lim) { speedingSec += dt; speedingKm += vs.speed[i] * dt / 1000; vs.flags[i] |= FLAG_SPEEDING; } else vs.flags[i] &= ~FLAG_SPEEDING;
      totalSec += dt; totalKm += vs.speed[i] * dt / 1000;
      if (done === 0) {
        const route = vs.routes[i], rp = vs.routePos[i], rem = g.edges[vs.edge[i]].length - vs.s[i];
        const stopHere = vs.refuelState[i] === 1 && vs.stopPos[i] === rp;
        if ((rp + 1 >= route.length || stopHere) && (rem < 2.5 || (rem < 8 && vs.speed[i] < 0.3))) done = stopHere ? 2 : 1;
      }
      if (done === 1) { this._arrive(i); continue; }
      if (done === 2) { vs.updatePose(i, g, this._pose); this._enterStation(i); refueling++; continue; }
      if (vs.fuel[i] <= 0 && vs.speed[i] < 0.05) { vs.updatePose(i, g, this._pose); this._strand(i); crashed++; continue; }
      // waiting (behind a wreck, in a queue) is normal traffic: a long stop is only labelled, once per stop
      if (vs.stoppedSince[i] >= 0 && this.time - vs.stoppedSince[i] > this.opts.stuckTimeout && this._stuckLabelled.get(i) !== vs.stoppedSince[i]) {
        this._stuckLabelled.set(i, vs.stoppedSince[i]);
        this.counters.stuck++; const k = this._stuckCause(i); this.counters.stuckBy[k] = (this.counters.stuckBy[k] || 0) + 1;
        this._event('stuck', vs.x[i], vs.z[i], [i], { cause: k, stoppedFor: this.time - vs.stoppedSince[i] });
      }
      if (this.stations && this.time - vs.lastFuelCheck[i] > checkEvery) this._checkFuel(i, false);
      vs.updatePose(i, g, this._pose);
      driving++; sumSpeed += vs.speed[i];
    }
    this.live.driving = driving; this.live.crashed = crashed; this.live.refueling = refueling; this.live.sumSpeed = sumSpeed;
    const a = this.accum; a.vehicleSec += totalSec; a.vehicleKm += totalKm; a.speedingSec += speedingSec; a.speedingKm += speedingKm;
  }

  _setOutcome(cid, leg, o) { if (leg === 0) { if (this.outcome[cid] === FAIL.PENDING) this.outcome[cid] = o; } else if (this.outcomeBack[cid] === FAIL.PENDING) this.outcomeBack[cid] = o; }

  _arrive(i) {
    const vs = this.vs, c = this.commuters[vs.commuter[i]];
    if (!c) { this._release(i); return; }
    const leg = vs.leg[i];
    const lateBy = leg === 0 ? this.time - c.workStart : 0;
    const drv = this.drivers[i];
    if (drv && drv.arrive) { try { drv.arrive(lateBy); } catch (e) { /* ignore */ } }
    this.counters.arrived++;
    this.commuterFuel[c.id] = vs.fuel[i];
    const dur = this.time - vs.spawnTime[i];
    this._setOutcome(c.id, leg, FAIL.ARRIVED);
    if (leg === 0) {
      this.counters.arrivedOutbound++;
      c.actualArrival = this.time;
      c.lateBy = Math.max(0, lateBy); c.earlyBy = Math.max(0, -lateBy);
      if (lateBy <= 0) this.counters.onTime++;
      this.accum.lateSum += c.lateBy; this.accum.lateN++;
    }
    this._logTrip(c, { leg }, i, 'arrived', dur, vs.odometer[i]);
    const g = c.genome;
    if (g) g.fitness = (g.fitness || 0) + 10 + (leg === 0 ? (lateBy <= 0 ? 2 : Math.max(-10, -0.5 * lateBy / 60)) : 0);
    this._event('arrive', vs.x[i], vs.z[i], [i], { commuter: c.id, leg, duration: dur, distance: vs.odometer[i], lateBy });
    this._release(i);
  }

  /** Record a failed trip. remove = take the vehicle off the road now (stuck); crash/stranded stay as scenes. */
  _failTrip(i, code, reason, remove = false) {
    const vs = this.vs, c = this.commuters[vs.commuter[i]];
    if (code === FAIL.STUCK) { this.counters.stuck++; const k = this._stuckCause(i); this.counters.stuckBy[k] = (this.counters.stuckBy[k] || 0) + 1; }
    if (c) {
      this.commuterFuel[c.id] = Math.max(vs.fuel[i], code === FAIL.STRANDED ? 0.2 * vs.tank[i] : 0); // towed cars get a can of fuel
      this._setOutcome(c.id, vs.leg[i], code);
      if (c.genome) c.genome.fitness = (c.genome.fitness || 0) - 20;
      this._logTrip(c, { leg: vs.leg[i] }, i, reason, this.time - vs.spawnTime[i], vs.odometer[i]);
    }
    const drv = this.drivers[i];
    if (drv && drv.fail) { try { drv.fail(reason); } catch (e) { /* ignore */ } }
    this._event(reason, vs.x[i], vs.z[i], [i], { commuter: c ? c.id : -1, leg: vs.leg[i] });
    if (remove) this._release(i);
  }

  /**
   * Crash type and maneuver (telemetry / diagnosis): type from the pre-impact headings and the
   * contact normal in the first car's frame: 'barrier' | 'rear-end' (normal along the car, headings
   * within 30 deg) | 'sideswipe' (normal across, headings within 30 deg) | 'head-on' (> 150 deg) |
   * 'angle' (T-bone / junction); maneuver of the striking car: 'lane-change' | 'merge' (near a node,
   * cars on different edges) | 'queue-approach' (leader slow / stopped) | 'junction' | 'wreck' | 'cruise'.
   */
  _classifyCrash(slots, c, pre) {
    const vs = this.vs, g = this.graph, a = slots[0];
    const ha = pre ? pre.heading[0] : vs.heading[a];
    let type = 'barrier';
    if (slots.length > 1) {
      const b = slots[1], hb = pre ? pre.heading[1] : vs.heading[b];
      let d = Math.abs(ha - hb) % (2 * Math.PI); if (d > Math.PI) d = 2 * Math.PI - d;
      const along = Math.abs(Math.cos(ha) * c.nx + Math.sin(ha) * c.nz);
      type = d > 2.6 ? 'head-on' : d > 0.52 ? 'angle' : along > 0.7 ? 'rear-end' : 'sideswipe';
      if (vs.state[b] === STATE_CRASHED || vs.state[a] === STATE_CRASHED) type += '-wreck';
    }
    // striking car = the faster one
    let st = a; if (slots.length > 1 && (pre ? pre.speed[1] > pre.speed[0] : vs.speed[slots[1]] > vs.speed[a])) st = slots[1];
    const e = g.edges[vs.edge[st]], dNode = Math.min(vs.s[st], e.length - vs.s[st]);
    let maneuver = 'cruise';
    // a lane change only while the body actually crosses a lane line (straddles two lanes) and is moving across
    const changing = vs.laneHi[st] > vs.laneLo[st] && vs.conn[st] < 0 && ((vs.targetLane[st] >= 0 && vs.targetLane[st] !== vs.lane[st]) || Math.abs(vs.headingRel[st]) > 0.035 || Math.abs(vs.vLat[st]) > 0.3);
    if (slots.length > 1 && vs.state[slots[0]] === STATE_CRASHED && vs.state[slots[1]] === STATE_CRASHED) maneuver = 'wreck';
    else if (changing) maneuver = 'lane-change';
    else if (slots.length > 1 && vs.edge[slots[0]] !== vs.edge[slots[1]] && dNode < 40) maneuver = (SHOULDER_ROADS.has(e.highway) ? 'merge' : 'junction');
    else if (vs.leader[st] >= 0 && (vs.state[vs.leader[st]] !== STATE_DRIVING || vs.speed[vs.leader[st]] < 2) && vs.speed[st] > 5) maneuver = 'queue-approach';
    else if (dNode < 30 && g.inStart[e.to + 1] - g.inStart[e.to] > 1) maneuver = 'junction';
    return { type, maneuver, roadClass: e.highway };
  }

  /** Why a car is stuck: follow its leader chain to the head of the queue. */
  _stuckCause(i) {
    const vs = this.vs, g = this.graph, seen = new Set();
    // walk the queue forward while the car ahead is itself (nearly) stopped and close: the last one is
    // the car actually holding the queue
    let k = i, n = 0;
    while (vs.state[k] === STATE_DRIVING && vs.leader[k] >= 0 && !seen.has(k) && n++ < 200) {
      const L = vs.leader[k];
      if (vs.state[L] === STATE_DRIVING && (vs.speed[L] > 1 || vs.gap[k] > 30)) break;
      seen.add(k); k = L;
    }
    if (vs.state[k] === STATE_DRIVING && vs.pullUntil[k] > this.time) return 'pulled-over-blocking';
    if (vs.leader[k] >= 0 && seen.has(k)) return 'deadlock-loop';
    if (vs.state[k] === STATE_CRASHED) return vs.flags[k] & FLAG_STRANDED ? 'behind-stranded' : 'behind-wreck';
    if (vs.state[k] !== STATE_DRIVING) return 'other';
    if (vs.leader[k] === -2) return 'lane-end';
    if (vs.leader[k] === -3) return 'junction-conflict';
    const e = g.edges[vs.edge[k]];
    if (vs.ctlType[k] >= 0 && vs.ctlDist[k] < 30) {
      const t = vs.ctlType[k], st = vs.ctlState[k];
      if (t === CTRL.STOP) return this.controls.flags[this.controls.apControl[vs.ctlApproach[k]]] & 1 ? 'all-way-stop' : 'stop-sign';
      if (t === CTRL.GIVE_WAY) return 'give-way';
      if (st === LIGHT.GREEN && vs.ctlMove[k] === MOVE.LEFT) return 'signal-permissive-left';
      return 'signal';
    }
    if (this.ctx.yieldRequired(k, e, e.length - vs.s[k])) return 'junction-yield';
    if (this.edgeHasCrash[vs.edge[k]]) return 'scene-on-edge';
    return 'head-stopped-other';
  }

  _release(i) {
    const vs = this.vs;
    if (this.table && vs.commuter[i] >= 0) this.commuters[vs.commuter[i]] = undefined;
    if (vs.state[i] === STATE_REFUELING && this.stations && vs.station[i] >= 0) {
      const k = vs.station[i];
      if (vs.refuelState[i] === 3) this.stations.busy[k]--;
      const q = this.stations.queue[k], p = q.indexOf(i); if (p >= 0) q.splice(p, 1);
    }
    this.drivers[i] = null;
    vs.release(i);
  }

  _logTrip(c, trip, slot, outcome, duration, distance) {
    if (!this.opts.tripLog) return;
    this.tripLog.push({ commuter: c.id, leg: trip.leg, outcome, duration, distance, avgSpeed: duration > 0 ? distance / duration : 0, time: this.time, expected: c.expectedTravel, routeLength: c.L });
  }

  // ---------------------------------------------------------------- crashes (physics only)
  /**
   * Rigid-body contacts (SPEC "Crashes: real physics only"). Every DRIVING / CRASHED car is
   * an oriented box in world space. Broadphase: the spatial hash (25 m cells); narrowphase:
   * separating-axis test swept from the start-of-step pose to the current pose in <= 0.5 m
   * relative sub-steps (no tunnelling). A contact gets an impulse (restitution 0.2, Coulomb
   * friction 0.5, angular terms) and positional correction; disabled cars at rest are static.
   * Road-edge barriers (shoulder width past the outer lane; the far curb past the opposing
   * lanes on two-way roads; one more lane inside a lane-end taper) are static colliders too.
   * A NEW contact (not touching in the previous step) is a crash event with the per-vehicle
   * delta-v of the impulse; cars with delta-v >= 10 km/h are disabled.
   */
  _contacts() {
    const vs = this.vs, high = vs.high, hash = this.hash, g = this.graph;
    const A = this._bA || (this._bA = {}), B = this._bB || (this._bB = {}), A0 = this._bA0 || (this._bA0 = {}), B0 = this._bB0 || (this._bB0 = {});
    const c = this._c || (this._c = {}), res = this._res || (this._res = {}), ta = {}, tb = {};
    const touching = this._touch || (this._touch = new Map()), now = new Map();
    const layer = this.opts.layerGap;
    for (let i = 0; i < high; i++) {
      const si = vs.state[i];
      if (si !== STATE_DRIVING && si !== STATE_CRASHED) continue;
      const movI = si === STATE_DRIVING || vs.speed[i] > 0.01 || Math.abs(vs.vLat[i]) > 0.01;
      // no edge barriers inside a junction box (on a connector, or on the box part of an edge: before its
      // start mouth sExit or past its end mouth sStop)
      if (si === STATE_DRIVING && !this.empty && vs.conn[i] < 0) { const eb = g.edges[vs.edge[i]]; if (!(g.J && (vs.s[i] < eb.sExit + 1 || vs.s[i] > eb.sStop))) this._barrier(i); }
      if (this.empty && si === STATE_DRIVING) continue; // stage 0: agents pass through each other
      const sweepI = Math.hypot(vs.x[i] - vs.px[i], vs.z[i] - vs.pz[i]);
      const hdI = Math.hypot(vs.hl[i], vs.hw[i]);
      const xi = vs.x[i], zi = vs.z[i], yi = vs.y[i];
      hash.forEachNear(xi, zi, hdI + 6.5 + sweepI + 5, (j) => {
        if (j <= i) return;
        const sj = vs.state[j];
        if (sj !== STATE_DRIVING && sj !== STATE_CRASHED) return;
        const movJ = sj === STATE_DRIVING || vs.speed[j] > 0.01 || Math.abs(vs.vLat[j]) > 0.01;
        if (!movI && !movJ) return;
        if (Math.abs(vs.y[j] - yi) > layer) return;
        // same edge but far apart along it: the world boxes only meet because lane offsets fold
        // inside a tight polyline bend (a geometry artefact, not a physical contact)
        if (vs.edge[j] === vs.edge[i] && Math.abs(vs.s[j] - vs.s[i]) > vs.hl[i] + vs.hl[j] + 1) return;
        // grade separation (data/crossings.json): see _adjacentOrFreewayFree
        if (vs.edge[j] !== vs.edge[i] && !this._adjacentOrFreewayFree(vs.edge[i], vs.edge[j])) return;
        const sweepJ = Math.hypot(vs.x[j] - vs.px[j], vs.z[j] - vs.pz[j]);
        const hdJ = Math.hypot(vs.hl[j], vs.hw[j]);
        if (Math.hypot(vs.x[j] - xi, vs.z[j] - zi) > hdI + hdJ + sweepI + sweepJ) return;
        this._body(i, A, A0, movI); this._body(j, B, B0, movJ);
        const t = sweptContact(A0, A, B0, B, c, 0.5, ta, tb);
        if (t < 0) return;
        if (t < 1) { A.x = ta.x; A.z = ta.z; B.x = tb.x; B.z = tb.z; }
        const key = i * 1048576 + j;
        now.set(key, 1);
        const pre = this._pre || (this._pre = { speed: [0, 0], heading: [0, 0], nx: 0, nz: 0 });
        pre.speed[0] = Math.hypot(A.vx, A.vz); pre.speed[1] = Math.hypot(B.vx, B.vz); pre.heading[0] = A.yaw; pre.heading[1] = B.yaw; pre.nx = c.nx; pre.nz = c.nz;
        resolveContact(A, B, c, PHYS.restitution, PHYS.friction, res);
        separate(A, B, c);
        this._writeBack(i, A); this._writeBack(j, B);
        if (!touching.has(key) && res.jn > 0 && (si === STATE_DRIVING || sj === STATE_DRIVING)) {
          // one crash per pair of cars (the same two bodies touching again, e.g. a car pressed against a
          // wreck, is the same crash) unless the new contact is itself disabling
          this._crashEvent([i, j], [res.dvA, res.dvB], c, res.closing, 'collision', pre);
        }
      });
    }
    this._touch = now;
  }

  /**
   * Can bodies on edges a and b touch? Same / adjacent edges (shared node) always; a freeway-system
   * edge and an edge it shares no node with only where data/crossings.json reports the pair at
   * (nearly) the same height or as a pavement overlap (otherwise one passes over the other).
   * Without crossings data: never across a freeway (grade-separated by definition).
   */
  _adjacentOrFreewayFree(a, b) {
    const g = this.graph, ea = g.edges[a], eb = g.edges[b];
    if (ea.from === eb.from || ea.from === eb.to || ea.to === eb.from || ea.to === eb.to) return true;
    if (!SHOULDER_ROADS.has(ea.highway) && !SHOULDER_ROADS.has(eb.highway)) return true;
    return g.contactPairs ? g.contactPairs.has(g.contactPairKey(a, b)) : false;
  }

  /** Fill body b (and its start-of-step pose b0) from slot i. Disabled cars at rest are static. */
  _body(i, b, b0, moving) {
    const vs = this.vs, yaw = vs.heading[i], fx = Math.cos(yaw), fz = Math.sin(yaw), u = vs.speed[i], v = vs.vLat[i];
    b.x = vs.x[i]; b.z = vs.z[i]; b.yaw = yaw; b.hl = vs.hl[i]; b.hw = vs.hw[i];
    b.vx = u * fx + v * fz; b.vz = u * fz - v * fx; b.w = -vs.yawRate[i];   // left of (fx, fz) is (fz, -fx)
    const fixed = !moving && vs.state[i] === STATE_CRASHED;
    b.invMass = fixed ? 0 : 1 / vs.mass[i]; b.invInertia = fixed ? 0 : 1 / vs.inertia[i];
    b0.x = vs.px[i]; b0.z = vs.pz[i]; b0.yaw = vs.pyaw[i]; b0.hl = b.hl; b0.hw = b.hw;
    b._x = b.x; b._z = b.z;
  }

  /** Write a body's velocity / position back into slot i (Frenet state). */
  _writeBack(i, b) {
    if (b.invMass === 0) return;
    const vs = this.vs, e = this.graph.edges[vs.edge[i]];
    const yaw = vs.heading[i], fx = Math.cos(yaw), fz = Math.sin(yaw);
    let u = b.vx * fx + b.vz * fz;
    const v = b.vx * fz - b.vz * fx;
    if (u < 0) u = 0; // no reverse gear: backward momentum is absorbed by the drivetrain / brakes
    vs.speed[i] = u; vs.vLat[i] = v; vs.yawRate[i] = -b.w;
    // positional change (sweep rewind + penetration correction) into s / lat along the road
    const dx = b.x - b._x, dz = b.z - b._z;
    if (dx !== 0 || dz !== 0) {
      const p = this._pose; frameDir(this.graph, vs, i, p);
      const ds = dx * p.hx + dz * p.hz, dl = dx * p.hz - dz * p.hx; // left of the road direction is (hz, -hx)
      vs.s[i] = Math.min(frameEnd(this.graph, vs, i), Math.max(0, vs.s[i] + ds)); vs.lat[i] += dl;
      vs.updateLanes(i, e); vs.updatePose(i, this.graph, this._pose);
    }
  }

  /** Road-edge barriers for slot i (static colliders along the carriageway edges). */
  _barrier(i) {
    const vs = this.vs, e = this.graph.edges[vs.edge[i]], W = laneW(e, vs.s[i]);
    let padR = 0, padL = e.lanes * W;
    if (e.LG) { const bd = e.LG.bounds(e.idx, vs.s[i]); padR = bd.R - bd.right; padL = bd.R - bd.left; } // pavement incl. lane add/drop tapers
    else { const taper = vs.taper[i] > vs.s[i] ? W : 0; if (vs.lat[i] < 0) padR -= taper; if (vs.lat[i] > padL) padL += taper; }
    const right = padR - (SHOULDER_ROADS.has(e.highway) ? this.opts.freewayShoulder : this.opts.shoulder);
    const left = padL + (e.reverse >= 0 ? (e.oppLanes || e.lanes) * W + this.opts.oppositeMargin : this.opts.shoulder);
    const hr = vs.headingRel[i], sh = Math.sin(hr), ch = Math.cos(hr), hl = vs.hl[i], hw = vs.hw[i], lat = vs.lat[i];
    const ext = Math.abs(sh) * hl + Math.abs(ch) * hw;
    let side = 0, depth = 0;
    if (lat - ext < right) { side = -1; depth = right - (lat - ext); }
    else if (lat + ext > left) { side = 1; depth = lat + ext - left; }
    const key = -1 - i, touching = this._touch;
    if (!side) return;
    // contact point: the corner furthest toward the barrier
    let best = 1e9, cx = 0, cz = 0;
    const yaw = vs.heading[i], fx = Math.cos(yaw), fz = Math.sin(yaw);
    for (let a = -1; a <= 1; a += 2) for (let b2 = -1; b2 <= 1; b2 += 2) {
      const l = -side * (a * hl * sh + b2 * hw * ch);
      if (l < best) { best = l; cx = vs.x[i] + a * hl * fx + b2 * hw * fz; cz = vs.z[i] + a * hl * fz - b2 * hw * fx; }
    }
    const p = this._pose; edgePointDir(e, vs.s[i], p, vs.seg[i]);
    const nx = side > 0 ? p.hz : -p.hz, nz = side > 0 ? -p.hx : p.hx; // from the car toward the barrier
    const A = this._bW || (this._bW = {}), Wb = this._bWall || (this._bWall = { x: 0, z: 0, yaw: 0, vx: 0, vz: 0, w: 0, hl: 0, hw: 0, invMass: 0, invInertia: 0 });
    this._body(i, A, this._bA0 || (this._bA0 = {}), true);
    Wb.x = cx; Wb.z = cz;
    const c = { depth, nx, nz, px: cx, pz: cz }, res = this._res || (this._res = {});
    const pre = { speed: [Math.hypot(A.vx, A.vz)], heading: [A.yaw], nx, nz };
    resolveContact(A, Wb, c, PHYS.wallRestitution, PHYS.wallFriction, res);
    this._writeBack(i, A);
    vs.lat[i] -= side * depth; vs.updateLanes(i, e); vs.updatePose(i, this.graph, this._pose);
    this._touchNext = this._touchNext || new Set();
    const was = this._barrierTouch && this._barrierTouch.has(i);
    (this._barrierNow || (this._barrierNow = new Set())).add(i);
    if (!was && res.jn > 0) this._crashEvent([i], [res.dvA], c, res.closing, 'barrier', pre);
  }

  /**
   * A new physical contact: log it, signal the drivers, disable cars with delta-v >= 10 km/h
   * (they become stationary obstacles where physics leaves them) and open a scene.
   */
  _crashEvent(slots, dvs, c, closing, cause, pre = null) {
    const vs = this.vs, kmh = dvs.map(d => d * 3.6), maxDv = Math.max(...kmh);
    if (maxDv < this.opts.touchKmh) { this.counters.touches++; return null; } // resting / grazing contact, resolved by physics
    // one crash per pair of cars (decided here, so wrappers of _crashEvent see the same rule)
    let repeat = false;
    if (slots.length === 2) {
      const a = Math.min(slots[0], slots[1]), b = Math.max(slots[0], slots[1]), key = a * 1048576 + b;
      const last = this._pairEvent.get(key);
      repeat = last !== undefined && last >= vs.spawnTime[a] && last >= vs.spawnTime[b];
      if (!repeat) this._pairEvent.set(key, this.time);
    }
    if (repeat) {
      // the same pair touching again: not a new crash, unless this contact disables a driving car
      let dis = false; for (let k = 0; k < slots.length; k++) if (vs.state[slots[k]] === STATE_DRIVING && kmh[k] >= DISABLE_KMH) dis = true;
      if (!dis) { this.counters.repeatContacts++; return null; }
    }
    const severity = severityOf(maxDv);
    const disabled = [];
    for (let k = 0; k < slots.length; k++) {
      const s = slots[k];
      vs.flags[s] |= FLAG_COLLIDED;
      if (vs.state[s] === STATE_DRIVING && kmh[k] >= DISABLE_KMH) disabled.push(s);
    }
    const speeding = slots.some(s => (vs.flags[s] & FLAG_SPEEDING) !== 0);
    if (speeding) this.counters.crashesWhileSpeeding++;
    this.counters.crashes++;
    this.counters.bySeverity[severity] = (this.counters.bySeverity[severity] || 0) + 1;
    const secondary = slots.some(s => vs.state[s] === STATE_CRASHED);
    if (secondary) this.counters.secondaryCrashes++;
    const e = this.graph.edges[vs.edge[slots[0]]];
    const tl = incidentTimeline(severity, this.rng), clear = tl.clear;
    const log = { id: this.crashes.length, time: this.time, cause, severity, vehicles: slots.slice(), types: slots.map(s => vs.typeIdx[s]),
      commuters: slots.map(s => vs.commuter[s]), x: c.px, z: c.pz, y: vs.y[slots[0]],
      contact: { x: c.px, z: c.pz }, closingSpeed: closing, deltaVKmh: kmh, disabled: disabled.slice(), clearEnd: this.time + clear,
      edge: e.idx, road: e.name || e.ref || e.highway, secondary, speeding, isCrash: true,
      kind: cause === 'barrier' ? 'barrier' : 'vehicle', ...this._classifyCrash(slots, c, pre),
      preSpeed: pre ? pre.speed.slice(0, slots.length) : slots.map(s => vs.speed[s]), heading: pre ? pre.heading.slice(0, slots.length) : slots.map(s => vs.heading[s]),
      normal: { nx: c.nx, nz: c.nz } };
    // rewardVersion 2: per-car severity of this event for the drivers' reward (signalsFor .crash; reward.js crashPenalty)
    if (!this._crashSig) this._crashSig = new Map();
    for (let k = 0; k < slots.length; k++) {
      const sh = slots.length === 2 && pre ? crashShare(pre.speed[k], pre.heading[k], pre.speed[1 - k], pre.heading[1 - k], pre.nx ?? c.nx, pre.nz ?? c.nz)
        : pre ? crashShare(pre.speed[0], pre.heading[0], 0, 0, pre.nx ?? c.nx, pre.nz ?? c.nz) : 0.5;
      if (vs.state[slots[k]] === STATE_DRIVING) this._crashSig.set(slots[k], { dvKmh: kmh[k], share: sh, disabled: kmh[k] >= DISABLE_KMH });
    }
    // drivers of every involved car see collision = true on one step (reward -20)
    for (const s of slots) { const drv = this.drivers[s]; if (drv && drv.step && vs.state[s] === STATE_DRIVING) { try { drv.step(this.dt, this.wq.bind(s)); } catch (err) { /* ignore */ } } }
    for (const s of slots) this._crashSig.delete(s);   // only the drivers' step above sees it
    if (disabled.length) {
      this.counters.disablingCrashes++;
      this.counters.crashedVehicles += disabled.length;
      for (const s of disabled) { vs.state[s] = STATE_CRASHED; vs.flags[s] |= FLAG_DISABLED; vs.clearControls(s); }
      const rec = this._openScene(disabled, { cause, severity, impactSpeed: closing, impactMph: closing / 0.44704, lanesBlocked: 0, blockDuration: tl.block, clearDuration: clear }, e, true);
      Object.assign(rec, { notifyAt: this.time + tl.notify, respondersAt: this.time + tl.arrive, deltaVKmh: kmh, contact: log.contact, closingSpeed: closing, secondary, speeding, involved: slots.slice(), kind: log.kind, preSpeed: log.preSpeed, heading: log.heading, normal: log.normal, type: log.type, maneuver: log.maneuver, roadClass: log.roadClass });
      for (const s of disabled) this._failTrip(s, FAIL.CRASH, 'crashed');
      this._event('crash', c.px, c.pz, slots, { crash: rec.id, severity, deltaVKmh: kmh, closingSpeed: closing, cause, kind: log.kind, preSpeed: log.preSpeed, heading: log.heading, nx: c.nx, nz: c.nz });
      return rec;
    }
    this.counters.minorContacts++;
    // drivable after a minor crash: on a freeway the drivers pull onto the shoulder to exchange
    // details (5-15 min, off the travel lanes), then carry on; elsewhere they drive on to a safe spot
    if (maxDv >= 3) for (const sl of slots) {
      const ee = this.graph.edges[vs.edge[sl]];
      if (vs.state[sl] === STATE_DRIVING && PULLOVER_ROADS.has(ee.highway)) { vs.pullUntil[sl] = this.time + 300 + 600 * this.rng.random(); this.counters.pullOvers++; }
    }
    this.crashes.push(log);
    this._event('crash', c.px, c.pz, slots, { crash: log.id, severity, deltaVKmh: kmh, closingSpeed: closing, cause, kind: log.kind, preSpeed: log.preSpeed, heading: log.heading, nx: c.nx, nz: c.nz });
    return log;
  }

  /** Open a lane-blocking scene (crash / stranded / wreck) for vehicles `fresh` on edge e. */
  _openScene(fresh, cls, e, isCrash, all = fresh) {
    const vs = this.vs;
    let laneMask = 0, sMax = 0;
    for (const s of all) if (vs.edge[s] === e.idx) { for (let l = vs.laneLo[s]; l <= vs.laneHi[s]; l++) laneMask |= 1 << l; if (vs.s[s] > sMax) sMax = vs.s[s]; }
    if (!laneMask) { laneMask = 1 << vs.lane[fresh[0]]; sMax = vs.s[fresh[0]]; }
    let blocked = 0; for (let l = 0; l < e.lanes; l++) if ((laneMask >> l) & 1) blocked++;
    const rec = {
      id: this.crashes.length, time: this.time, cause: cls.cause, severity: cls.severity, impactSpeed: cls.impactSpeed, impactMph: cls.impactMph,
      vehicles: fresh.slice(), commuters: fresh.map(s => vs.commuter[s]), types: fresh.map(s => vs.typeIdx[s]),
      lanesBlocked: blocked, laneMask, edge: e.idx, s: sMax, x: vs.x[fresh[0]], z: vs.z[fresh[0]], y: vs.y[fresh[0]],
      blockStart: this.time, blockEnd: this.time + cls.blockDuration, clearEnd: this.time + cls.clearDuration,
      squeezeAt: this.time + Math.min(cls.blockDuration, (blocked >= e.lanes || all.some(q => vs.edge[q] !== e.idx) ? 1 : 2) * this.opts.squeezeDelay),
      squeeze: false, responseState: 'active', road: e.name || e.ref || e.highway, secondary: false, speeding: false, isCrash,
      permanent: cls.blockDuration >= 1e8,   // training wrecks / obstacles: never squeezed or towed
    };
    for (const s of fresh) {
      vs.state[s] = STATE_CRASHED; vs.clearControls(s);
      vs.crashedUntil[s] = rec.clearEnd;
    }
    this.crashes.push(rec); this.activeCrashes.push(rec);
    let list = this.edgeCrashes.get(e.idx); if (!list) { list = []; this.edgeCrashes.set(e.idx, list); } list.push(rec);
    this.edgeHasCrash[e.idx] = 1;
    if (e.reverse >= 0) this.edgeHasCrash[e.reverse] = 1;
    return rec;
  }

  /**
   * Move a scene's wrecks to a real off-lane position: onto the paved right shoulder when it
   * is wide enough for the body (freeways / links), at least 15 m from either node so the wreck
   * never lands in an intersection corner. Where there is no room (surface roads with a 1 m
   * shoulder) the tow truck hauls the car away: the slot is released now and the scene keeps
   * only its record (rubbernecking zone) until clearEnd.
   */
  _moveOffLanes(r) {
    const vs = this.vs, g = this.graph;
    for (const s of r.vehicles) {
      if (vs.state[s] !== STATE_CRASHED || (r.gone && r.gone.includes(s))) continue;
      vs.flags[s] |= FLAG_SHOULDER;
      const ee = g.edges[vs.edge[s]];
      const sh = SHOULDER_ROADS.has(ee.highway) ? this.opts.freewayShoulder : this.opts.shoulder;
      if (sh < 2 * vs.hw[s] + 0.4 || ee.length < 30) {
        if (vs.flags[s] & FLAG_STRANDED) this.counters.towed++; else this.counters.cleared++;
        this.counters.hauledAway = (this.counters.hauledAway || 0) + 1;
        (r.gone || (r.gone = [])).push(s); this._release(s); continue;
      }
      vs.conn[s] = -1; // back in the in-edge frame
      vs.s[s] = Math.min(ee.length - 15, Math.max(15, vs.s[s]));
      vs.lat[s] = -(vs.hw[s] + 0.3); vs.headingRel[s] = 0; vs.speed[s] = 0; vs.vLat[s] = 0; vs.yawRate[s] = 0;
      vs.updateLanes(s, ee); vs.updatePose(s, g, this._pose); vs.savePose(s); this.counters.towedToShoulder++;
    }
  }

  _scenes() {
    if (this.stations && this.lot.length) this._pumps();
    this._barrierTouch = this._barrierNow || new Set(); this._barrierNow = new Set();
    if (this._pairEvent.size > 200000) for (const [k, t] of this._pairEvent) if (this.time - t > 7200) this._pairEvent.delete(k);
    if (this.activeCrashes.length === 0) return;
    const vs = this.vs, keep = [];
    for (const r of this.activeCrashes) {
      // disabled cars may still be sliding: the blocked lanes follow their hitboxes
      if (r.responseState === 'active' && r.isCrash && !r.squeeze) {
        let m = 0, sMax = 0; for (const q of r.vehicles) if (vs.edge[q] === r.edge && vs.state[q] === STATE_CRASHED) { for (let l = vs.laneLo[q]; l <= vs.laneHi[q]; l++) m |= 1 << l; if (vs.s[q] > sMax) sMax = vs.s[q]; }
        if (m) { r.laneMask = m; r.s = sMax; }
      }
      if (r.responseState === 'active' && !r.squeeze && !r.permanent && this.time >= r.squeezeAt && this.time < r.blockEnd) {
        r.squeeze = true; // full closure: responders push the wrecks aside, trapped traffic is waved past at walking pace
        this._moveOffLanes(r);
        r.laneMask = 0;
        this._event('squeeze', r.x, r.z, r.vehicles, { crash: r.id });
      }
      if (r.responseState === 'active' && this.time >= r.blockEnd && this.time < r.clearEnd) {
        r.responseState = 'shoulder';
        // responders / tow trucks move the wrecks off the travel lanes (or haul them away)
        this._moveOffLanes(r);
        this._event('shoulder', r.x, r.z, r.vehicles, { crash: r.id });
      }
      if (this.time >= r.clearEnd) {
        r.responseState = 'cleared';
        for (const s of r.vehicles) if (vs.state[s] === STATE_CRASHED && !(r.gone && r.gone.includes(s))) { if (vs.flags[s] & FLAG_STRANDED) this.counters.towed++; else this.counters.cleared++; this._release(s); }
        const list = this.edgeCrashes.get(r.edge);
        if (list) { const k = list.indexOf(r); if (k >= 0) list.splice(k, 1); if (list.length === 0) { this.edgeCrashes.delete(r.edge); this.edgeHasCrash[r.edge] = 0; const rv = this.graph.edges[r.edge].reverse; if (rv >= 0 && !this.edgeCrashes.has(rv)) this.edgeHasCrash[rv] = 0; } }
        this._event('cleared', r.x, r.z, r.vehicles, { crash: r.id, severity: r.severity });
      } else keep.push(r);
    }
    this.activeCrashes = keep;
  }

  // ---------------------------------------------------------------- driver context
  _makeCtx() {
    const sim = this, vs = this.vs, g = this.graph;
    const ctx = {
      graph: g, vs, lanes: this.lanes, rng: this.rng, cfg: this.idmCfg, time: this.time, scratchPair: [-1, -1], scratchPair2: [-1, -1],
      rubberneck(i, e) {
        if (!sim.edgeHasCrash[e.idx]) return 1;
        const s = vs.s[i], lane = vs.lane[i], D = RUBBERNECK.dist;
        const own = sim.edgeCrashes.get(e.idx);
        if (own) for (const r of own) if (r.isCrash && (!((r.laneMask >> lane) & 1) || r.responseState === 'shoulder')) { if (Math.abs(r.s - s) < D) return RUBBERNECK.factor; }
        if (e.reverse >= 0) {
          const opp = sim.edgeCrashes.get(e.reverse);
          if (opp) for (const r of opp) if (r.isCrash && Math.abs((e.length - r.s) - s) < D) return RUBBERNECK.factor;
        }
        return 1;
      },
      speedCap(i, e) {
        if (!sim.edgeHasCrash[e.idx]) return Infinity;
        const own = sim.edgeCrashes.get(e.idx), s = vs.s[i];
        if (own) for (const r of own) if (r.squeeze && r.responseState === 'active' && r.s - s < 60 && s - r.s < 20) return sim.opts.squeezeSpeed;
        return Infinity;
      },
      /** +1/-1: my lane is blocked by a scene / stopped obstacle ahead (preferred escape side), 0 otherwise. */
      blockedAhead(i, e) {
        const L = vs.leader[i];
        // blocked: a stopped obstacle is my leader, or a scene blocks my lane within 150 m (seen past the queue)
        const direct = L >= 0 && vs.state[L] === STATE_CRASHED && vs.laneHi[L] >= vs.laneLo[L] && vs.gap[i] < 150;
        if (!direct && !(sim.edgeHasCrash[e.idx] && !ctx.laneOpen(i, e, vs.lane[i], 150))) return 0;
        const lane = vs.lane[i];
        for (let d = 1; d < e.lanes; d++) {
          if (lane + d < e.lanes && ctx.laneOpen(i, e, lane + d)) return 1;
          if (lane - d >= 0 && ctx.laneOpen(i, e, lane - d)) return -1;
        }
        return 0; // every lane blocked: wait (squeeze after squeezeDelay)
      },
      laneOpen(i, e, lane, within = 1e9) {
        if (lane < 0 || lane >= e.lanes) return false;
        if (!sim.edgeHasCrash[e.idx]) return true;
        const recs = sim.edgeCrashes.get(e.idx);
        if (recs) for (const r of recs) if (r.responseState === 'active' && !r.squeeze && ((r.laneMask >> lane) & 1) && r.s > vs.s[i] && r.s - vs.s[i] < within) return false;
        return true;
      },
      yieldRequired(i, e, dist) {
        if (sim.empty) return false;
        if (g.J) {
          const r = sim._yieldJ(i, e);
          // deadlockBreak: a creeping car does not wait for the car it squeezes past
          if (r && sim._creepUntil && sim._creepUntil[i] > sim.time && sim._yieldBy[i] === sim._creepIgnore[i]) return false;
          return r;
        }
        if (dist < 1.5) return false;
        let conf = e._conf;
        if (!conf) {
          conf = { inc: [], out: [] };
          for (let k = g.inStart[e.to]; k < g.inStart[e.to + 1]; k++) { const ei = g.inEdges[k]; if (ei !== e.idx && ei !== e.reverse) conf.inc.push(ei); }
          for (let k = g.outStart[e.to]; k < g.outStart[e.to + 1]; k++) { const ei = g.outEdges[k]; if (ei !== e.reverse) conf.out.push(ei); }
          let hw = 0;
          for (const ei of conf.inc) { const o = g.edges[ei]; hw = Math.max(hw, (o.reverse >= 0 ? 2 : 1) * o.lanes * LANE_WIDTH * 0.5); }
          for (const ei of conf.out) { const o = g.edges[ei]; hw = Math.max(hw, (o.reverse >= 0 ? 2 : 1) * o.lanes * LANE_WIDTH * 0.5); }
          e.stopBack = conf.inc.length ? Math.min(12, Math.max(1.5, hw + 1)) : 1.5;
          e._conf = conf;
        }
        if (conf.inc.length === 0) return false;
        const L0 = vs.leader[i];
        // don't block the box: need room beyond the node for my whole length
        if (L0 >= 0 && vs.speed[L0] < 1 && vs.state[L0] === STATE_DRIVING && vs.gap[i] - dist < vs.length[i] + 3) return true;
        const route = vs.routes[i], rp = vs.routePos[i];
        const nextEdge = rp + 1 < route.length ? route[rp + 1] : -1;
        const nextRev = nextEdge >= 0 ? g.edges[nextEdge].reverse : -1;
        const waited = vs.stoppedSince[i] >= 0 ? sim.time - vs.stoppedSince[i] : 0;
        const patienceScale = Math.max(0.3, 1 - waited / 90);
        const v = vs.speed[i];
        const tMe = dist / Math.max(v, 2);
        // decisions land a reaction time later: two cars arriving within that window plus the time to
        // clear the box (~2 s) conflict
        const win = Math.max(2 * patienceScale, vs.reactionTime[i] + 1.5);
        // cars still inside the junction box on another out-edge
        for (let q = 0; q < conf.out.length; q++) {
          const eo = g.edges[conf.out[q]];
          for (let l = 0; l < eo.lanes; l++) {
            const j = sim.lanes.first(eo.idx, l);
            if (j < 0 || vs.state[j] !== STATE_DRIVING) continue;
            if (vs.speed[j] < 0.5 && waited > 20) continue;
            // its whole body may still be in the box: length + junction half-width + margin
            if (vs.s[j] < vs.length[j] + e.stopBack + 3 && !(eo.idx === nextEdge && l === mapLane(g, e, eo, vs.lane[i]))) return true;
          }
        }
        for (let q = 0; q < conf.inc.length; q++) {
          const e2 = g.edges[conf.inc[q]];
          if (e2.idx === nextRev) continue; // oncoming traffic from where I'm heading
          for (let l = 0; l < e2.lanes; l++) {
            const j = sim.lanes.last(e2.idx, l);
            if (j < 0 || vs.state[j] !== STATE_DRIVING) continue;
            const rj = vs.routes[j], pj = vs.routePos[j];
            if (pj + 1 >= rj.length) continue;                                 // its trip ends at this node
            if (vs.refuelState[j] === 1 && vs.stopPos[j] === pj) continue;   // it stops at a station here
            const d2 = e2.length - vs.s[j];
            const v2 = vs.speed[j];
            if (v2 < 0.3 && d2 > 3 && waited > 5) continue;                    // queued / yielding itself
            const t2 = d2 / Math.max(v2, 2);
            if (e2.rank > e.rank) { if (t2 < 4 * patienceScale || d2 < 6) return true; }
            else if (e2.rank === e.rank) {
              if (d2 < 60 || t2 < 6) { if (t2 < tMe - win || (Math.abs(t2 - tMe) <= win && (t2 < tMe || (t2 === tMe && j < i))) || d2 < 3) return true; }
            }
            else if (d2 < 5 && waited < 30) return true;
          }
        }
        return false;
      },
      /**
       * Is the junction box ahead physically occupied by a car whose path I would cross? (cars
       * still crossing on an out-edge other than my own next lane, or a car entering from
       * another in-edge right now). Real drivers do not drive into a car in the intersection,
       * whatever the light shows.
       */
      boxBusy(i, e) {
        if (g.J) return sim._boxBusyJ(i, e);
        const route = vs.routes[i], rp = vs.routePos[i];
        const nextEdge = rp + 1 < route.length ? route[rp + 1] : -1;
        for (let q = g.outStart[e.to]; q < g.outStart[e.to + 1]; q++) {
          const eo = g.edges[g.outEdges[q]]; if (eo.idx === e.reverse) continue;
          for (let l = 0; l < eo.lanes; l++) {
            const j = sim.lanes.first(eo.idx, l); if (j < 0 || vs.state[j] === STATE_FREE) continue;
            if (eo.idx === nextEdge) continue;                       // going my way: it is my leader, not a crossing
            if (vs.s[j] < vs.length[j] + 4) return true;
          }
        }
        for (let q = g.inStart[e.to]; q < g.inStart[e.to + 1]; q++) {
          const e2 = g.edges[g.inEdges[q]]; if (e2.idx === e.idx) continue;
          for (let l = 0; l < e2.lanes; l++) {
            const j = sim.lanes.last(e2.idx, l); if (j < 0 || vs.state[j] !== STATE_DRIVING) continue;
            if (e2.length - vs.s[j] < 3 && vs.speed[j] > 1) return true;
          }
        }
        return false;
      },
      /**
       * Is it safe to start moving into `target` lane now? Delay-aware gap acceptance
       * against the cars listed in that lane (footprints, so cars already straddling
       * into it count). relaxed (stuck behind a blockage, everyone crawling): 1 m each side.
       */
      laneGapOk(i, e, target, relaxed = false, transit = false) {
        if (target < 0 || target >= e.lanes) return false;
        if (!transit && sim.edgeHasCrash[e.idx]) {
          const recs = sim.edgeCrashes.get(e.idx);
          if (recs) for (const r of recs) if (r.responseState === 'active' && !r.squeeze && ((r.laneMask >> target) & 1) && r.s > vs.s[i] && r.s - vs.s[i] < 60) return false;
        }
        const nb = sim.lanes.neighborsAt(e.idx, target, vs.s[i], ctx.scratchPair, i);
        const v = vs.speed[i], half = 0.5 * vs.length[i];
        const ah = sim._blocking(nb[0], target), bh = nb[1];
        // a lane change takes ~3 s of lateral travel after my reaction time: judge the gaps over that window
        const tLC = 3.0;
        if (ah >= 0) {
          const ga = vs.s[ah] - vs.s[i] - half - 0.5 * vs.length[ah], dv = v - (vs.state[ah] === STATE_DRIVING ? vs.speed[ah] : 0);
          const need = relaxed && v < 2 ? 1 : Math.max(3, 2 + v * 0.5 + Math.max(0, dv) * (vs.reactionTime[i] + tLC));
          if (ga < need) return false;
        }
        if (bh >= 0 && vs.state[bh] === STATE_DRIVING) {
          const gb = vs.s[i] - vs.s[bh] - half - 0.5 * vs.length[bh], vb = vs.speed[bh];
          const need = relaxed && vb < 2 ? 1 : Math.max(3, 2 + vb * 0.5 + Math.max(0, vb - v) * (vs.reactionTime[bh] + tLC));
          if (gb < need) return false;
        }
        // mirror / blind spot on the far side: a car in the next lane over that is moving into the same
        // target lane (lateral velocity toward it) alongside me
        const far = target + (target - vs.lane[i]);
        if (far >= 0 && far < e.lanes && far !== vs.lane[i]) {
          const dir = target - far; // +1: far lane is to the right of the target
          const p2 = sim.lanes.neighborsAt(e.idx, far, vs.s[i], ctx.scratchPair2, i);
          for (let k = 0; k < 2; k++) {
            const j = p2[k]; if (j < 0 || vs.state[j] !== STATE_DRIVING) continue;
            const latVel = vs.speed[j] * Math.sin(vs.headingRel[j]) + vs.vLat[j] * Math.cos(vs.headingRel[j]);
            const heading = latVel * dir > 0.2 || (vs.targetLane[j] === target && !(vs.flags[j] & FLAG_BRAIN));
            if (!heading) continue;
            const gap = Math.abs(vs.s[j] - vs.s[i]) - half - 0.5 * vs.length[j];
            if (gap < 5 + Math.abs(vs.speed[j] - v) * 2) return false;
          }
        }
        return true;
      },
    };
    return ctx;
  }

  /**
   * Freeway density sample: driving + stopped cars per (motorway / trunk mainline edge >= 100 m),
   * density = cars / (length km x lanes). Records the network mean (veh/km/lane over all freeway
   * lane-km), the share of freeway lane-km at LOS D-F (> 16.2 veh/km/ln) and their peaks.
   */
  _sampleDensity() {
    const vs = this.vs, cnt = this._edgeCount, g = this.graph;
    if (!this._fwy.length) return;
    for (let i = 0; i < vs.high; i++) { const st = vs.state[i]; if (st === STATE_DRIVING || st === STATE_CRASHED) { const e = vs.edge[i]; if (cnt[e] < 65535) cnt[e]++; } }
    let cars = 0, dfLaneKm = 0, peakSeg = 0;
    for (const ei of this._fwy) {
      const e = g.edges[ei], lk = e.length * e.lanes / 1000, c = cnt[ei];
      cars += c; const d = c / lk;
      if (d > 16.2) dfLaneKm += lk;
      if (e.length >= 300 && d > peakSeg) peakSeg = d;
    }
    for (let i = 0; i < vs.high; i++) cnt[vs.edge[i]] = 0;
    const mean = cars / this._fwyLaneKm, share = dfLaneKm / this._fwyLaneKm, D = this.density;
    D.samples++; D.meanDensitySum += mean; D.losDFShareSum += share;
    if (mean > D.peakMeanDensity) { D.peakMeanDensity = mean; D.peakMeanAt = this.time; }
    if (share > D.peakLosDFShare) { D.peakLosDFShare = share; D.peakLosDFAt = this.time; }
    if (peakSeg > D.peakSegmentDensity) D.peakSegmentDensity = peakSeg;
    D.last = { time: this.time, meanDensity: mean, losDFShare: share, freewayCars: cars };
  }

  // ---------------------------------------------------------------- events & stats
  _event(type, x, z, vehicleIds, extra) {
    if (!this.onEvent && !this.opts.tripLog && type !== 'crash') return null;
    const ev = { time: this.time, type, x, z, vehicleIds, ...extra };
    this.events.push(ev);
    if (this.events.length > this.opts.eventLimit) this.events.splice(0, this.events.length >> 1);
    if (this.onEvent) this.onEvent(ev);
    return ev;
  }

  /** Crash event log rows {severity, impactSpeed, vehicles, lanesBlocked, blockStart, blockEnd, responseState, ...}. */
  crashLog() {
    return this.crashes.filter(r => r.isCrash).map(r => ({ id: r.id, time: r.time, severity: r.severity, cause: r.cause, vehicles: r.involved || r.vehicles, disabled: r.disabled || r.vehicles, contact: r.contact, closingSpeed: r.closingSpeed, deltaVKmh: r.deltaVKmh, lanesBlocked: r.lanesBlocked || 0, edge: r.edge, road: r.road, blockStart: r.blockStart ?? r.time, blockEnd: r.blockEnd ?? r.time, clearEnd: r.clearEnd, responseState: r.responseState || 'drivable', secondary: r.secondary, speeding: r.speeding }));
  }

  getStats() {
    const c = this.counters, n = this.commuters.length || 1;
    const fb = [0, 0, 0, 0, 0, 0];
    for (let i = 0; i < this.outcome.length; i++) fb[this.outcome[i]]++;
    const sev = {};
    Object.assign(sev, c.bySeverity);
    const st = this.stations;
    return {
      clock: this.clock, time: this.time, speed: this.speed, lagging: this.lagging, stepMs: this._lastStepMs,
      active: this.vs.count, driving: this.live.driving, crashedOnScene: this.live.crashed, refueling: this.live.refueling,
      spawned: c.spawned, arrived: c.arrived, arrivedOutbound: c.arrivedOutbound,
      crashes: c.crashes, crashedVehicles: c.crashedVehicles, secondaryCrashes: c.secondaryCrashes, activeCrashes: this.activeCrashes.length,
      cleared: c.cleared, stuck: c.stuck, stuckBy: c.stuckBy, pullOvers: c.pullOvers, towedToShoulder: c.towedToShoulder, noRoute: c.noRoute, laneChanges: c.laneChanges, minorContacts: c.minorContacts, touches: c.touches, disablingCrashes: c.disablingCrashes, repeatContacts: c.repeatContacts, missedExits: c.missedExits || 0, reroutes: c.reroutes || 0, wreckPasses: c.wreckPasses || 0,
      stranded: c.stranded, towed: c.towed, refuelPlanned: c.refuelPlanned, refuelled: c.refuelled, refuelPlanFailed: c.refuelPlanFailed,
      forcedSpawns: c.forcedSpawns,
      avgSpeed: this.live.driving ? this.live.sumSpeed / this.live.driving : 0,
      avgSpeedMph: this.live.driving ? (this.live.sumSpeed / this.live.driving) / 0.44704 : 0,
      pendingSpawns: this.pendingSpawns.length, tripsRemaining: this.trips.length - this.tripPtr,
      commuters: this.commuters.length,
      // outbound (home -> work) trips: arrivalRate is the SPEC's ">90% reach work"
      arrivalRate: fb[FAIL.ARRIVED] / n,
      failures: { crash: fb[FAIL.CRASH], stranded: fb[FAIL.STRANDED], stuck: fb[FAIL.STUCK], noRoute: fb[FAIL.NOROUTE], unfinished: fb[FAIL.PENDING] },
      completionRate: c.arrived / Math.max(1, c.spawned),
      brainSteps: this.live.brainSteps, learnerSteps: this.live.learnerSteps, learners: this.learners.reduce((a, b) => a + b, 0),
      onTimeRate: c.onTime / n, onTimeOfArrived: c.arrivedOutbound ? c.onTime / c.arrivedOutbound : 0,
      meanLatenessMin: this.accum.lateN ? this.accum.lateSum / this.accum.lateN / 60 : 0,
      speedingRate: this.accum.vehicleSec ? this.accum.speedingSec / this.accum.vehicleSec : 0,
      vehicleKm: this.accum.vehicleKm, speedingKm: this.accum.speedingKm,
      crashRatePerMillionKm: this.accum.vehicleKm > 0 ? 1e6 * c.crashes / this.accum.vehicleKm : 0,
      crashesWhileSpeeding: c.crashesWhileSpeeding,
      crashRatePer1000kmSpeeding: this.accum.speedingKm > 0 ? 1000 * c.crashesWhileSpeeding / this.accum.speedingKm : 0,
      crashRatePer1000kmNotSpeeding: (this.accum.vehicleKm - this.accum.speedingKm) > 0 ? 1000 * (c.crashes - c.crashesWhileSpeeding) / (this.accum.vehicleKm - this.accum.speedingKm) : 0,
      crashBySeverity: sev,
      fuelPumpedL: this.accum.fuelL, meanPumpWaitMin: this.accum.pumpWaitN ? this.accum.pumpWaitSum / this.accum.pumpWaitN / 60 : 0,
      maxPumpQueue: st ? st.stats.maxQueue : 0,
      density: {
        freewayLaneKm: this._fwyLaneKm,
        peakMeanVehPerKmLane: this.density.peakMeanDensity, peakMeanAt: clockOf(this.density.peakMeanAt),
        peakLosDFShare: this.density.peakLosDFShare, peakLosDFAt: clockOf(this.density.peakLosDFAt),
        peakSegmentVehPerKmLane: this.density.peakSegmentDensity,
      },
    };
  }

  tripStats() {
    const arrived = this.tripLog.filter(t => t.outcome === 'arrived');
    const n = arrived.length || 1;
    return {
      arrived: arrived.length,
      meanDurationMin: arrived.reduce((a, t) => a + t.duration, 0) / n / 60,
      meanDistanceKm: arrived.reduce((a, t) => a + t.distance, 0) / n / 1000,
      meanSpeedMph: arrived.reduce((a, t) => a + t.avgSpeed, 0) / n / 0.44704,
      outcomes: this.tripLog.reduce((m, t) => { m[t.outcome] = (m[t.outcome] || 0) + 1; return m; }, {}),
    };
  }

  // ---------------------------------------------------------------- genomes
  exportGenomes() { return this.commuters.map(c => ({ id: c.id, genome: { delta: Array.from(c.genome.delta || c.genome.weights || []), fitness: c.genome.fitness || 0 } })); }
  importGenomes(list) {
    let n = 0;
    for (const { id, genome } of list) {
      const c = this.commuters[id]; if (!c) continue;
      const d = c.genome.delta || c.genome.weights; const src = genome.delta || genome.weights;
      if (d && src && src.length === d.length) { d.set(src); n++; }
      c.genome.fitness = genome.fitness || 0;
    }
    return n;
  }
}

/**
 * Map a pre-steering-only control record {throttle in [-1,1], steer in [-1,1] (+ = left yaw command)}
 * onto the SPEC action space: gas/brake from throttle, steerRate drives the wheel angle toward the
 * angle that produces the requested yaw rate (routeSim.js mapping: yaw = steer * min(0.6, max(0.05, 0.08 v))).
 */
export function legacyCommand(vs, i, cmd, step) {
  const th = +(cmd.rawThrottle ?? cmd.throttle) || 0, st = +(cmd.rawSteer ?? cmd.steer) || 0, v = vs.speed[i];
  const yaw = st * Math.min(0.6, Math.max(0.05, 0.08 * v));
  const dDes = Math.atan(vs.wheelbase[i] * yaw / Math.max(v, 1));
  const sr = vs.actionMode === 'position' ? dDes / vs.maxSteer[i] : (dDes - vs.steerAngle[i]) / (vs.maxSteerRate[i] * 0.3);
  vs.command(i, sr, th > 0 ? th : 0, th < 0 ? -th : 0, step);
}

/**
 * Steering value for the ring from a driver command, in the store's action mode.
 *  position mode: cmd.steer (target wheel position in [-1,1]); a legacy {steerRate} driver is integrated
 *    from its last commanded target (target += steerRate * maxSteerRate * dt / maxSteer), so rate-trained
 *    checkpoints keep working during the transition.
 *  rate mode: cmd.steerRate; a {steer} position command is turned into a rate toward that position.
 */
export function steerCommand(vs, i, cmd, dt) {
  if (vs.actionMode === 'position') {
    if (cmd.steer !== undefined && cmd.steerRate === undefined) return +cmd.steer || 0;
    const R = vs.ctrlRing, o = (i * CTRL_RING + ((vs.cmdStep[i] % CTRL_RING) + CTRL_RING) % CTRL_RING) * 3;
    const t = R[o] + (+cmd.steerRate || 0) * vs.maxSteerRate[i] * dt / vs.maxSteer[i];
    return t > 1 ? 1 : t < -1 ? -1 : t;
  }
  if (cmd.steerRate !== undefined) return +cmd.steerRate || 0;
  return ((+cmd.steer || 0) * vs.maxSteer[i] - vs.steerAngle[i]) / (vs.maxSteerRate[i] * 0.4);
}

/** Pack commuter.routeOut / routeBack into the RouteStore layout (for callers that pass bare commuters). */
export function packCommuterRoutes(commuters) {
  let total = 0; for (const c of commuters) total += (c.routeOut?.length || 0) + (c.routeBack?.length || 0);
  const flat = new Int32Array(total), offset = new Uint32Array(2 * commuters.length + 1);
  let p = 0;
  for (let i = 0; i < commuters.length; i++) {
    const c = commuters[i];
    offset[2 * i] = p; if (c.routeOut) { flat.set(c.routeOut, p); p += c.routeOut.length; }
    offset[2 * i + 1] = p; if (c.routeBack) { flat.set(c.routeBack, p); p += c.routeBack.length; }
  }
  offset[2 * commuters.length] = p;
  return { flat, offset };
}

/**
 * Headless run (node). runHeadless(simHours, n, opts) on a synthetic graph unless
 * opts.graph / opts.sim is given (real data: see headless.js / createTrainingEnv).
 * Prints arrivalRate, failure breakdown (crash, stranded, stuck, unfinished), on-time
 * rate, mean lateness, speeding share and crash rate.
 */
export function runHeadless(simHours = 4, n = 2000, opts = {}) {
  const log = opts.log === undefined ? console.log : opts.log;
  let graph = opts.graph;
  let popOpts = opts.population || {};
  if (!graph && !opts.sim) {
    graph = buildGraph(makeSynthGraph(opts.synth || { blocks: 30, cross: 6, bridges: 3 }));
    popOpts = { axis: synthAxis(graph), long: { minM: 5000, maxM: 20000 }, departures: 'spec', ...popOpts };
  }
  const sim = opts.sim || new Simulation(graph, { ...opts, population: { n, ...popOpts } });
  return runSim(sim, simHours, { log, stopWhenIdle: opts.stopWhenIdle });
}

/** Drive an existing Simulation for simHours, logging every 30 sim-minutes; returns {sim, stats}. */
export function runSim(sim, simHours, { log = console.log, stopWhenIdle = false, every = 1800 } = {}) {
  const end = sim.time + simHours * 3600;
  let nextLog = sim.time + every, steps = 0;
  const t0 = Date.now();
  while (sim.time < end) {
    sim.step(); steps++;
    if (stopWhenIdle && sim.isIdle) break;
    if (log && sim.time >= nextLog) {
      const s = sim.getStats();
      log(`[${s.clock}] active=${s.active} arrived(out)=${s.arrivedOutbound} crashes=${s.crashes} stranded=${s.stranded} stuck=${s.stuck} refuel=${s.refuelled}/${s.refuelPlanned} avg=${s.avgSpeedMph.toFixed(1)}mph arrivalRate=${(s.arrivalRate * 100).toFixed(1)}% wall=${((Date.now() - t0) / 1000).toFixed(0)}s`);
      nextLog += every;
    }
  }
  const stats = sim.getStats();
  stats.wallMs = Date.now() - t0; stats.steps = steps; stats.msPerStep = stats.wallMs / Math.max(1, steps);
  stats.simHours = simHours;
  stats.trips = sim.tripStats();
  if (log) log(formatReport(stats));
  return { sim, stats };
}

export function formatReport(s) {
  const pct = (x) => (x * 100).toFixed(1) + '%';
  return [
    `arrivalRate ${pct(s.arrivalRate)} (${s.arrivedOutbound}/${s.commuters} reached work)`,
    `failures: crash ${s.failures.crash}, stranded ${s.failures.stranded}, stuck ${s.failures.stuck}, unfinished ${s.failures.unfinished}, noRoute ${s.failures.noRoute}`,
    `on-time ${pct(s.onTimeRate)} of all (${pct(s.onTimeOfArrived)} of arrived), mean lateness ${s.meanLatenessMin.toFixed(2)} min`,
    `speeding share ${pct(s.speedingRate)} of driving time; crashes ${s.crashes} (${s.crashRatePerMillionKm.toFixed(1)} per million vehicle-km, ${s.secondaryCrashes} secondary) ${JSON.stringify(s.crashBySeverity)}`,
    `freeway density: peak network mean ${s.density.peakMeanVehPerKmLane.toFixed(1)} veh/km/lane at ${s.density.peakMeanAt}, peak share of freeway lane-km at LOS D-F ${pct(s.density.peakLosDFShare)} at ${s.density.peakLosDFAt}, densest 300 m+ segment ${s.density.peakSegmentVehPerKmLane.toFixed(0)} veh/km/lane`,
    `fuel: ${s.refuelled} refuels (${s.refuelPlanned} planned), mean pump wait ${s.meanPumpWaitMin.toFixed(1)} min, ${s.stranded} stranded`,
    `${s.steps} steps, ${s.msPerStep.toFixed(2)} ms/step, wall ${(s.wallMs / 1000).toFixed(0)} s`,
  ].join('\n');
}

Simulation.runHeadless = runHeadless;
export { STATE_DRIVING, STATE_CRASHED, STATE_FREE, STATE_REMOVED, STATE_REFUELING, LANE_WIDTH, FLAG_LEARNER, FLAG_BRAIN, FLAG_SHOULDER, FLAG_COLLIDED, FLAG_STRANDED, OUTCOME_NAME };
