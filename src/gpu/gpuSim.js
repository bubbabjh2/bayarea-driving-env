// GPU traffic simulation: every vehicle's state lives in WebGPU storage buffers and every
// per-vehicle stage is a compute kernel (src/gpu/kernels/*.wgsl), mirroring the CPU
// reference sim (src/sim/sim.js; state contract src/sim/LAYOUT.md, layout v3) step for step:
//
//   tick -> [ops] -> station re-entry + spawn -> footprint lane lists (count, scan, scatter, sort)
//   -> leaders (+ lane ends / tapers) -> collide (every collisionInterval steps)
//   -> [sense (+ perceive): observation vectors, opts.obs] -> act (external policy actions, setActions)
//   -> integrate (reaction-time ring, pedals, kinematic bicycle, edge hops, arrival, station
//      stop, off-road, stranding, stuck, fuel checks, pose, stats) -> accidents
//
// Action space: {steer (target wheel position, -1..1), gas, brake} per vehicle, written by the caller into the
// `actions` buffer (setActions) and applied through the 40-step reaction-time ring. There is no built-in driver.
// The CPU keeps what needs A* or global bookkeeping: trip schedule and slot allocation,
// incident scenes (severity, lanes blocked, squeeze / shoulder / clearance), refuel plans
// (src/sim/fuel.js StationNetwork) and the pump queues. GPU -> CPU traffic is one small
// async readback per frame (stats + event queue); CPU -> GPU is writeBuffer of spawn
// entries, ops, routes and scene blocks.
//
//   const sim = await GPUSim.create(graph, { capacity: 131072 });
//   sim.setPopulation(pop, { stations });   // population.js SoA output or popPack.js
//   each frame: sim.advance(realDt); sim.getStats();
//   sim.scatter(100000)                      // benchmark: n vehicles on random routes at once
//   sim.importSnapshot(exportState(cpuSim), cpuSim.commuters); await sim.readState()   // equivalence

import { initGPU, loadKernelSource, Profiler } from './device.js';
import { packWorld, packRoutes, packControls, packJunctions, extraConflicts, EDGE_ID_MASK, connectorSpeed } from './worldPack.js';
import * as L from './layout.js';
import { VEHICLE_TYPES, VEHICLE_TYPE_WEIGHTS } from '../sim/vehicleTypes.js';
import * as simGenome from '../sim/genome.js';
import * as simCollisions from '../sim/collisions.js';
import * as simCrash from '../sim/crashModel.js';
import * as simFuel from '../sim/fuel.js';
import { VEHICLE_PARAMS, FUEL_DENSITY } from '../sim/physics/params.js';
import { SURF_PK, SURF_SK, SURF_CRR, PAVED_SHOULDER_FREEWAY } from '../sim/surface.js';
import { graphSig, sigMismatch } from '../sim/worldSig.js';
import { makeRouter, routingPack } from '../sim/route.js';
import { makeRng } from '../sim/rng.js';
import { hopMap as simHopMap } from '../sim/roadGraph.js';
import * as guidanceMod from '../nav/guidance.js';
import * as perc from '../sim/obsLayout.js';

// Observation vector v2 layout (70 floats; the original project's eyes.js offsets, see src/gpu/README.md)
export const OBS_V2 = Object.freeze({ INPUT_SIZE: 70, NEIGHBOR_OFFSET: 8, MANEUVER_OFFSET: 20, HINT_OFFSET: 30, TRAIT_OFFSET: 33, LATE_OFFSET: 37, FUEL_OFFSET: 39,
  WHEEL_OFFSET: 41, LANE_END_OFFSET: 44, CONTROL_OFFSET: 46, KINK_OFFSET: 63, JUNCTION_OFFSET: 65, RANGE: 120 });
const eyes = OBS_V2;

const ACC = simCollisions.ACCIDENT_DEFAULTS || { ratePerKm: 5e-5, speedRef: 25, closeGap: 30, mergeFactor: 3, mergeDist: 60 };
const IDM = {};   // (built-in driver parameters; unused without a built-in driver)
const SEC = simCrash.SECONDARY || { queueRiskWindow: 300, queueRiskDist: 500, queueRiskFactor: 6, rubberneckDist: 120, rubberneckFactor: 0.8 };
const FUEL = simFuel.FUEL_DEFAULTS || { lowFraction: 0.12, reserveKm: 15, planSpeed: 22.2, planFactor: 1.3, pumpMin: 180, pumpMax: 300, towMedian: 1800, towSigma: 0.3, checkInterval: 60, minFillToSkip: 0.5 };
const DEF_TRAITS = simGenome.DEFAULT_TRAITS || { aggression: 0.5, patience: 0.5, reactionTime: 0.9, preferredHeadway: 1.5, maxAccel: 2.5, maxDecel: 5.0 };
const LANE_W = 3.7;
const CTL_WORDS = 36;   // 32 junction-yield blocker slot, 33 creep-until time, 34 creep-ignored slot, 35 deadlock creep decision (docs/scale-crashes.md); 28 conflictWith, 29 courtesy gap (atomic min), 30 next off-lane body on my edge   // 24..27: braking-lead obstacle schedule (next, until, every lo, every hi)   // controls.wgsl side record per slot (CTL_W)
// src/sim/crashModel.js CRASH_TABLE as of layout v3 (used if the module stops exporting it)
const CRASH_TABLE_FALLBACK = { minor: { blockMedian: 300, clearMedian: 1200, sigma: 0.4, lanes: 1 }, moderate: { blockMedian: 2700, clearMedian: 2700, sigma: 0.4, lanes: 2 }, severe: { blockMedian: 6600, clearMedian: 6600, sigma: 0.4, lanes: 3 }, fatal: { blockMedian: 12000, clearMedian: 12000, sigma: 0.35, lanes: 99 } };

export const GPU_DEFAULTS = Object.freeze({
  dt: 0.05, startTime: 5 * 3600, endTime: 24 * 3600, speed: 1, maxStepsPerFrame: 400,
  capacity: 131072, seed: 1,
  leaderLookahead: 250, mergeLookahead: 80, collisionInterval: 1,
  collisionTolerance: 0.3, contactSpeed: 2.0, lateralTolerance: 0.15, offroadMargin: 1.0,
  squeezeDelay: 15 * 60, squeezeSpeed: 3,
  s0: IDM.s0 ?? 2, speedFactorMin: IDM.speedFactorMin ?? 0.85, speedFactorSpan: IDM.speedFactorSpan ?? 0.3,
  laneCheckInterval: IDM.laneCheckInterval ?? 1, yieldLook: IDM.yieldLook ?? 40, stopMargin: IDM.stopMargin ?? 1.5,
  anticipation: IDM.anticipation ?? 0.6, prepBase: IDM.prepBase ?? 250, prepPerLane: IDM.prepPerLane ?? 250,
  stuckTimeout: 30 * 60, spawnBlockedTimeout: 1e30,   // sim.js: Infinity (never forced into occupied space)
  decideEvery: 1, obs: 'v2', actionInput: 'position',
  accident: ACC, fuelCfg: FUEL,
  features: { laneChanges: true, accidents: false, fuel: true, collisions: true, yield: true, policy: true, zipFix: true, empty: false },
  tableBits: 18, cellSize: 25, layerGap: 2.5, pairCooldown: 7200, touchKmh: 1.0, shoulder: 1.0, freewayShoulder: 3.0, oppositeMargin: 0.5, laneEndLookahead: 450, contactCap: 131072, physDt: 0.01,
  evCap: 16384, spawnQueue: 16384, opCap: 16384, maxBlocks: 4096, routeSlack: 1 << 22,
  refuelBudgetMs: 4, profile: false, returnTrips: true, fuelScale: 1,
});

const FEAT_BITS = { laneChanges: L.FEAT.LANECHANGE, accidents: L.FEAT.ACCIDENTS, fuel: L.FEAT.FUEL, collisions: L.FEAT.COLLISIONS, yield: L.FEAT.YIELD, policy: L.FEAT.POLICY, zipFix: L.FEAT.ZIPFIX, empty: L.FEAT.EMPTY };
/**
 * Driver accel / brake limits exactly as src/sim/vehicle.js alloc, for both vehicle-type
 * schemas: physics params (m, F0: src/sim/physics/params.js) or the older maxAccel field.
 */
export function driverLimits(t, traits = {}) {
  const ta = traits.maxAccel ?? DEF_TRAITS.maxAccel, td = traits.maxDecel ?? DEF_TRAITS.maxDecel;
  const emer = t.emergencyDecel ?? 8;
  if (t.m && t.F0) return { maxAccel: Math.min(ta * Math.min(1, 1500 / t.m * 1.6), 0.9 * t.F0 / t.m), maxDecel: Math.min(td * (emer / 8.0) * 1.6, emer) };
  return { maxAccel: (t.maxAccel ?? 2.5) * (ta / (DEF_TRAITS.maxAccel || 2.5)), maxDecel: Math.min(emer, td * (emer / 8.0)) };
}
// fuel-model constants per type (src/sim/fuel.js); fallbacks for the physics-params schema
const FUEL_FALLBACK = [{ fuelBase: 0.050, fuelAero: 5.3e-5 }, { fuelBase: 0.065, fuelAero: 7.0e-5 }, { fuelBase: 0.200, fuelAero: 1.6e-4 }, { fuelBase: 0.220, fuelAero: 1.7e-4 }];
const fuelConst = (k, f) => {
  const t = VEHICLE_TYPES[k] || VEHICLE_TYPES[0];
  if (f === 'idleLps') return t.idleLps ?? (t.idleLph != null ? t.idleLph / 3600 : 0.00025);
  if (f === 'mass') return t.mass ?? t.m ?? 1500;
  return t[f] ?? FUEL_FALLBACK[k][f];
};

/** WGSL const arrays of the per-type physics parameters (layout v4). */
export function physicsDefines() {
  const sf = (a) => `array<f32, 16>(${Array.from(a, (x) => Number(x).toExponential(7)).join(', ')})`;
  const P = VEHICLE_PARAMS, f = (fn) => `array<f32, 4>(${[0, 1, 2, 3].map((k) => Number(fn(P[k] || P[0])).toExponential(7)).join(', ')})`;
  return {
    PT_M: f((p) => p.m), PT_IZ: f((p) => p.Iz), PT_A: f((p) => p.a), PT_B: f((p) => p.b), PT_L: f((p) => p.L), PT_HCG: f((p) => p.hcg),
    PT_MU: f((p) => p.mu), PT_CF: f((p) => p.Cf), PT_CR: f((p) => p.Cr), PT_F0: f((p) => p.F0), PT_PMAX: f((p) => p.Pmax), PT_VMAX: f((p) => p.vmax),
    PT_BRAKE: f((p) => p.brakeDecel), PT_BFRONT: f((p) => p.brakeFront), PT_CDA: f((p) => p.CdA), PT_CRR: f((p) => p.Crr),
    PT_MAXSTEER: f((p) => p.maxSteer), PT_MAXSTEERRATE: f((p) => p.maxSteerRate), PT_STEERLAG: f((p) => p.steerLag || 0.1), PT_STEERSLEW: f((p) => p.steerSlew || p.maxSteerRate), PT_BSFC: f((p) => p.bsfcMin), PT_IDLE: f((p) => p.idleLph / 3600),
    PT_ACCW: f((p) => p.accessoryW), PT_RHOF: f((p) => FUEL_DENSITY[p.fuel]),
    // src/sim/surface.js per-code friction / rolling factors (integrate.wgsl surfFactors)
    SF_PK: sf(SURF_PK), SF_SK: sf(SURF_SK), SF_CRR: sf(SURF_CRR), PAVED_SH_FW: Number(PAVED_SHOULDER_FREEWAY).toExponential(7),
    PT_DRIVE: `array<u32, 4>(${[0, 1, 2, 3].map((k) => ({ front: 0, rear: 1, all: 2 }[(P[k] || P[0]).drive] ?? 0) + 'u').join(', ')})`,
  };
}

const LH_W = 48;   // perceive.wgsl leader-history record (tag, n, head, lastLeader, sinceChange, g[20], r[20], pad)
const FUEL_MAXQ = 3;   // in-flight requests per fuel / route worker (GPUSim._fuelWorker)
const f32buf = new Float32Array(1), u32buf = new Uint32Array(f32buf.buffer);
const f32bits = (x) => { f32buf[0] = x; return u32buf[0]; };
export const TANK_L = VEHICLE_TYPES.map((t) => t.tankL ?? 50);

export class GPUSim {
  static async create(graph, opts = {}) {
    const gpu = opts.gpu || await initGPU();
    const sim = new GPUSim(gpu, graph, opts);
    await sim._init();
    return sim;
  }

  constructor(gpu, graph, opts) {
    this.gpu = gpu; this.device = gpu.device; this.graph = graph;
    this.opts = { ...GPU_DEFAULTS, ...opts, features: { ...GPU_DEFAULTS.features, ...(opts.features || {}) }, accident: { ...GPU_DEFAULTS.accident, ...(opts.accident || {}) }, fuelCfg: { ...FUEL, ...(opts.fuelCfg || {}) } };
    this.cap = this.opts.capacity;
    this.dt = this.opts.dt; this.speed = this.opts.speed;
    this.time = this.opts.startTime; this.step = 0; this.accumulator = 0; this.lagging = false;
    this.rngState = (this.opts.seed * 2654435761) >>> 0 || 1;
    this.events = []; this.onEvent = null;
    this.counters = { minorContacts: 0, crashEvents: 0, bySeverity: {}, crashes: 0, crashedVehicles: 0, secondaryCrashes: 0, offroad: 0, stranded: 0, towed: 0, cleared: 0, spawned: 0, refuelPlanned: 0, refuelPlanFailed: 0, refuelled: 0, noRoute: 0, eventsDropped: 0, reentered: 0, signalViolations: 0, stopViolations: 0 };
    this.gpuStats = new Uint32Array(64);
    this.kernelMs = null;
    this.crashes = []; this.activeCrashes = [];
    this.lot = 0;
    this._inCrash = new Set();
    this._tripX = new Map(); this._crashX = new Map();   // detail records (EV_TRIPX / EV_CRASHX_*) by slot until their main event
  }

  // ------------------------------------------------------------------ setup
  async _init() {
    const { gpu, graph, opts } = this, cap = this.cap;
    const w = this.world = packWorld(graph, null);
    const ptsOff = w.edges.length, laneOffOff = ptsOff + w.pts.length, adjOff = laneOffOff + w.laneOff.length;
    const ebOff = adjOff + w.adj.length, blkOff = ebOff + w.E;
    const geoOff = blkOff + opts.maxBlocks * L.BLOCK_WORDS, lgvOff = geoOff + w.geo.length, hopOff = lgvOff + w.lgVerts.length;
    // traffic controls (opts.controls: src/sim/controls.js TrafficControls, e.g. loadWorld().controls)
    const cp = this.ctlPack = opts.controls ? packControls(opts.controls, w.E) : null;
    const ctlEdgeOff = hopOff + w.hops.length, ctlApOff = ctlEdgeOff + (cp ? cp.edgeAp.length : 0), ctlCOff = ctlApOff + (cp ? cp.ap.length : 0);
    // grade separations (data/crossings.json via loadWorld: graph.contactPairs) as sorted (lo, hi) edge pairs
    const pairs = [];
    if (graph.contactPairs) for (const k of graph.contactPairs) { const a = Math.floor(k / 4194304), b = k - a * 4194304; pairs.push([Math.min(a, b), Math.max(a, b)]); }
    pairs.sort((p, q) => p[0] - q[0] || p[1] - q[1]);
    const pairOff = ctlCOff + (cp ? cp.ctl.length : 0);
    // physical junctions (layout v8: graph.J from loadWorld, src/geom junctions.js)
    const jp = this.jPack = graph.J && graph.J.mouthS && graph.J.mouthS.length === 2 * w.E && opts.junctions !== false ? packJunctions(graph.J, w.E, connectorSpeed, extraConflicts(graph.J, Int32Array.from(graph.edges, (e) => e.to))) : null;   // + sim.js _conflicts (paths within 4.2 m)
    const jMouthOff = pairOff + 2 * pairs.length, jMovOff = jMouthOff + (jp ? 3 * w.E : 0), jEdgeMovOff = jMovOff + (jp ? jp.mov.length : 0);
    const jPtsOff = jEdgeMovOff + (jp ? jp.edgeMov.length : 0), jConfOff = jPtsOff + (jp ? jp.pts.length : 0);
    // movements into each edge (CSR, sim.js _movesInto): E + 1 starts, then movement ids
    let into = null;
    if (jp) { const M = jp.M, cnt = new Uint32Array(w.E + 1); for (let m = 0; m < M; m++) cnt[graph.J.mInfo[7 * m + 2] + 1]++; for (let e = 0; e < w.E; e++) cnt[e + 1] += cnt[e]; const fill = cnt.slice(0, w.E), list = new Uint32Array(M); for (let m = 0; m < M; m++) list[fill[graph.J.mInfo[7 * m + 2]]++] = m; into = new Uint32Array(w.E + 1 + M); into.set(cnt); into.set(list, w.E + 1); }
    const jIntoOff = jConfOff + (jp ? jp.conf.length : 0);
    const blob = new Uint32Array(jIntoOff + (into ? into.length : 0));
    if (into) blob.set(into, jIntoOff);
    if (jp) {
      const mg = new Float32Array(3 * w.E); for (let e = 0; e < w.E; e++) { mg[3 * e] = jp.mouth[2 * e]; mg[3 * e + 1] = jp.mouth[2 * e + 1]; mg[3 * e + 2] = jp.gore[e]; }
      blob.set(new Uint32Array(mg.buffer), jMouthOff); blob.set(jp.mov, jMovOff); blob.set(jp.edgeMov, jEdgeMovOff); blob.set(new Uint32Array(jp.pts.buffer), jPtsOff); blob.set(jp.conf, jConfOff);
    }
    if (cp) { blob.set(cp.edgeAp, ctlEdgeOff); blob.set(cp.ap, ctlApOff); blob.set(cp.ctl, ctlCOff); }
    pairs.forEach((p, k) => { blob[pairOff + 2 * k] = p[0]; blob[pairOff + 2 * k + 1] = p[1]; });
    this.pairN = pairs.length;
    blob.set(w.edges, 0); blob.set(new Uint32Array(w.pts.buffer), ptsOff); blob.set(new Uint32Array(w.laneOff.buffer), laneOffOff); blob.set(w.adj, adjOff);
    blob.set(w.geo, geoOff); blob.set(new Uint32Array(w.lgVerts.buffer), lgvOff); blob.set(w.hops, hopOff);
    this.layoutOff = { ptsOff, laneOffOff, adjOff, ebOff, blkOff, geoOff, lgvOff, hopOff, ctlEdgeOff, ctlApOff, ctlCOff, pairOff, jMouthOff, jMovOff, jEdgeMovOff, jPtsOff, jConfOff, jIntoOff };
    gpu.storage('world', blob.byteLength, blob);
    this.edgeHead = new Uint32Array(w.E);
    this.blockFree = []; for (let b = opts.maxBlocks - 1; b >= 0; b--) this.blockFree.push(b);
    this.blockNext = new Uint32Array(opts.maxBlocks);

    gpu.storage('veh', cap * L.VEH_WORDS * 4);
    gpu.storage('par', cap * L.PAR_WORDS * 4);
    gpu.storage('ring', cap * L.CTRL_RING * L.CTRL_STRIDE * 4);
    gpu.storage('pose', cap * 16, null, GPUBufferUsage.VERTEX);
    this.nCounts = w.K + 1;
    gpu.storage('counts', this.nCounts * 4);
    gpu.storage('items', L.MAX_OCC * cap * 4);
    gpu.storage('order', L.MAX_OCC * cap * 4);
    gpu.storage('vidx', cap * 16);
    gpu.storage('claim', cap * 4);
    this.T = 1 << opts.tableBits; this.wBits = Math.ceil(opts.tableBits / 2);
    gpu.storage('hcounts', (this.T + 1) * 4);
    gpu.storage('hitems', cap * 4);
    gpu.storage('hidx', cap * 8);
    gpu.storage('contacts', (4 + opts.contactCap * 8 + 262144 * 4 + cap) * 4);   // list | pair table | per-car contact counts
    gpu.storage('laneClaim', Math.max(1, w.K) * 4);
    this.ioWords = L.IO.EVDATA + 4 * opts.evCap;
    gpu.storage('io', this.ioWords * 4);
    gpu.storage('queue', opts.spawnQueue * L.SPAWN_WORDS * 4);
    gpu.storage('ops', opts.opCap * 16);
    this.paramsBuf = gpu.uniform('params', L.PARAM_WORDS * 4);
    this.paramData = new ArrayBuffer(L.PARAM_WORDS * 4);
    this.paramView = L.recordView(this.paramData, L.PARAM_FIELDS);
    this.routeWords = 0; this.routeCap = 0;
    this._ensureRoutes(1 << 16);
    this._scanLevels = [];
    let n = this.nCounts, src = gpu.buffers.get('counts'), lvl = 0;
    for (;;) {
      const blocks = Math.ceil(n / 512);
      const sums = gpu.storage(`scanSums${lvl}`, Math.max(blocks, 1) * 4);
      const ub = gpu.uniform(`scanP${lvl}`, 16);
      gpu.write(ub, new Uint32Array([n, 0, 0, 0]));
      this._scanLevels.push({ n, blocks, data: src, sums, ub });
      if (blocks <= 1) break;
      n = blocks; src = sums; lvl++;
    }
    this._hscan = [];
    { let n2 = this.T + 1, src2 = gpu.buffers.get('hcounts'), l2 = 0;
      for (;;) {
        const blocks = Math.ceil(n2 / 512);
        const sums = gpu.storage(`hscanSums${l2}`, Math.max(blocks, 1) * 4), ub = gpu.uniform(`hscanP${l2}`, 16);
        gpu.write(ub, new Uint32Array([n2, 0, 0, 0]));
        this._hscan.push({ n: n2, blocks, data: src2, sums, ub });
        if (blocks <= 1) break;
        n2 = blocks; src2 = sums; l2++;
      } }
    // observations (opts.obs): 'v2' = 70 floats per slot (sense.wgsl), 'v3' = 358 (sense + perceive.wgsl, the leader-history
    // records follow the inputs in the same buffer), 'none' = no observation kernels
    this.obsMode = opts.obs === 'v3' ? 'v3' : opts.obs === 'none' ? 'none' : 'v2';
    this.v3 = this.obsMode === 'v3';
    this.IN = this.v3 ? perc.OBS3 : eyes.INPUT_SIZE;
    gpu.storage('inputs', cap * this.IN * 4 + (this.v3 ? cap * LH_W * 4 : 0));
    gpu.storage('actions', cap * 3 * 4);   // external policy: [steer, gas, brake] per slot (setActions)
    this.nAgents = opts.agents ?? cap;
    gpu.storage('agents', Math.max(1, this.nAgents) * 16);
    gpu.storage('edgeCount', Math.max(1, w.E) * 4);
    gpu.storage('edgeSpeed', Math.max(1, w.E) * 4);
    gpu.storage('telemetry', 256 * 4);   // per-step counters written by integrate.wgsl (no reader in this repo)
    gpu.storage('ctl', (cap * CTL_WORDS + (this.jPack ? this.jPack.M + w.E + Math.ceil(w.E / 32) + w.E : 0) + 1) * 4);   // + off-lane body heads per edge   // + junction movement heads, wreck-passer heads, closed-edge bits   // per slot CTL_W words, then per junction movement the head of its car list   // controls.wgsl CTL_W per slot   // per-step counters
    this._staging = [0, 1, 2].map((k) => ({ buf: gpu.readback(`staging${k}`, this.ioWords * 4), busy: false }));
    this.profiler = new Profiler(gpu, 64);
    await this._compile();
    this._makeBindGroups();
    this.free = []; for (let i = cap - 1; i >= 0; i--) this.free.push(i);
    this.slotTrip = new Int32Array(cap).fill(-1);
    this.slotRoute = new Uint32Array(cap * 2);
    this.slotStation = new Int32Array(cap).fill(-1);
    this.queueFree = []; for (let k = opts.spawnQueue - 1; k >= 0; k--) this.queueFree.push(k);
    this.queueHigh = 0;
    this.opsPending = [];
    this.wrecks = new Map();
    this.refuelQueue = [];
    this.pumping = [];
    this.trips = null; this.tripPtr = 0;
    this._writeParams(0);
  }

  _defines() {
    const arr = (f) => `array<f32, 4>(${[0, 1, 2, 3].map((k) => Number(fuelConst(k, f)).toExponential(6)).join(', ')})`;
    return {
      VT_IDLE: arr('idleLps'), VT_FBASE: arr('fuelBase'), VT_FAERO: arr('fuelAero'), VT_MASS: arr('mass'),
      INS: `${this.IN}u`, OFF_NB: `${eyes.NEIGHBOR_OFFSET}u`, OFF_MAN: `${eyes.MANEUVER_OFFSET}u`, OFF_HINT: `${eyes.HINT_OFFSET}u`,
      OFF_TRAIT: `${eyes.TRAIT_OFFSET}u`, OFF_LATE: `${eyes.LATE_OFFSET ?? 37}u`, OFF_FUEL: `${eyes.FUEL_OFFSET ?? 39}u`, OFF_WHEEL: `${eyes.WHEEL_OFFSET ?? 41}u`, OFF_LANEEND: `${eyes.LANE_END_OFFSET ?? 44}u`, OFF_CTL: `${eyes.CONTROL_OFFSET ?? 46}u`, OFF_KINK: `${eyes.KINK_OFFSET ?? 63}u`, CTL_W: `${CTL_WORDS}u`, KINKS_ON: `${guidanceMod.KINKS_ENABLED ? 1 : 0}u`,
      RANGE: `${(eyes.RANGE ?? 120).toFixed(1)}`,
      // src/sim/physics/params.js VEHICLE_PARAMS (physics.wgsl)
      ...physicsDefines(),
      V3: `${this.v3 ? 1 : 0}u`, LH_W: `${LH_W}u`, LH_OFF: `${this.v3 ? this.cap * this.IN : 0}u`,
      ...(this.v3 ? { NB_OFF3: `${perc.NB_OFF}u` } : {}),
    };
  }

  async _compile() {
    const gpu = this.gpu, K = {}, defines = this._defines();
    const src = async (n) => loadKernelSource(n);
    const kern = (name, code, entryPoint = 'main') => gpu.kernel(name, code, { entryPoint, defines, label: name });
    const build = await src('build');
    for (const e of ['tick', 'clear', 'count', 'scatter', 'sortLanes']) K[e] = await kern(`build.${e}`, build, e);
    const scan = await src('scan');
    K.scanBlocks = await gpu.kernel('scanBlocks', scan, { entryPoint: 'scanBlocks' });
    K.addSums = await gpu.kernel('addSums', scan, { entryPoint: 'addSums' });
    for (const n of ['leaders', 'integrate', 'ops', 'act', 'sense']) K[n] = await kern(n, await src(n));
    const hs = await src('hash');
    for (const e of ['hclear', 'hcount', 'hscatter']) K[e] = await kern(`hash.${e}`, hs, e);
    const ct = await src('contacts');
    for (const e of ['barrier', 'detect', 'resolve', 'ownClear', 'ownMin', 'resolveRound']) K[e] = await kern(`contacts.${e}`, ct, e);
    K.newPose = await kern('integrate.newPose', await src('integrate'), 'newPose');
    const sp = await src('spawn');
    K.spawn = await kern('spawn', sp, 'spawn');
    K.leave = await kern('leave', sp, 'leave');
    const dens = await src('density');
    K.densClear = await kern('density.clear', dens, 'clearCounts');
    K.densCount = await kern('density.count', dens, 'countCars');
    K.trafClear = await kern('density.trafClear', dens, 'clearTraffic'); K.trafCount = await kern('density.trafCount', dens, 'countTraffic');
    if (this.jPack) {   // layout v8 junctions
      const jn = await src('junction'), jy = await src('jyield');
      K.jclear = await kern('junction.clear', jn, 'jclear'); K.jclearOff = await kern('junction.clearOff', jn, 'jclearOff');
      const ct = await src('courtesy'); K.cpropose = await kern('courtesy.propose', ct, 'propose'); K.capply = await kern('courtesy.apply', ct, 'apply'); K.jlink = await kern('junction.link', jn, 'jlink');
      K.jperceive = await kern('junction.perceive', jn, 'jperceive'); K.jdlFind = await kern('junction.dlFind', jn, 'jdlFind'); K.jdlApply = await kern('junction.dlApply', jn, 'jdlApply'); K.jyield = await kern('junction.yield', jy, 'jyield');
      K.rerouteScan = await kern('junction.reroute', jy, 'rerouteScan');
    }
    if (this.v3) K.perceive = await kern('perceive', await src('perceive'));
    this.K = K;
  }

  _reg() {
    const b = this.gpu.buffers;
    return { P: this.paramsBuf, io: b.get('io'), world: b.get('world'), routes: b.get('routes'), veh: b.get('veh'), par: b.get('par'), ring: b.get('ring'), pose: b.get('pose'),
      counts: b.get('counts'), starts: b.get('counts'), items: b.get('items'), order: b.get('order'), vidx: b.get('vidx'), claim: b.get('claim'), laneClaim: b.get('laneClaim'),
      queue: b.get('queue'), ops: b.get('ops'), inputs: b.get('inputs'), actions: b.get('actions'),
      agents: b.get('agents'), edgeCount: b.get('edgeCount'), edgeSpeed: b.get('edgeSpeed'), telemetry: b.get('telemetry'), ctl: b.get('ctl'),
      hcounts: b.get('hcounts'), hstarts: b.get('hcounts'), hitems: b.get('hitems'), hidx: b.get('hidx'), contacts: b.get('contacts'),
    };
  }

  _makeBindGroups() {
    const reg = this._reg(), gpu = this.gpu, BG = {};
    for (const [name, k] of Object.entries(this.K)) {
      if (name === 'scanBlocks' || name === 'addSums') continue;
      BG[name] = k.layouts.map((_, g) => gpu.bindGroup(k, reg, g));
    }
    BG.scan = this._scanLevels.map((lv) => [gpu.bindGroup(this.K.scanBlocks, { SP: lv.ub, data: lv.data, sums: lv.sums }, 0)]);
    BG.add = this._scanLevels.map((lv) => [gpu.bindGroup(this.K.addSums, { SP: lv.ub, data: lv.data, sums: lv.sums }, 0)]);
    BG.hscanB = this._hscan.map((lv) => [gpu.bindGroup(this.K.scanBlocks, { SP: lv.ub, data: lv.data, sums: lv.sums }, 0)]);
    BG.hscanA = this._hscan.map((lv) => [gpu.bindGroup(this.K.addSums, { SP: lv.ub, data: lv.data, sums: lv.sums }, 0)]);
    this.BG = BG;
  }

  _ensureRoutes(words) {
    if (words <= this.routeCap) return;
    const cap = Math.max(words, Math.ceil(this.routeCap * 1.5));
    const old = this.gpu.buffers.get('routes');
    const nb = this.device.createBuffer({ label: 'routes', size: cap * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    if (old && this.routeWords > 0) {
      const enc = this.device.createCommandEncoder();
      enc.copyBufferToBuffer(old, 0, nb, 0, this.routeWords * 4);
      this.device.queue.submit([enc.finish()]);
    }
    if (old) old.destroy();
    this.gpu.buffers.set('routes', nb);
    this.routeCap = cap;
    if (this.K) this._makeBindGroups();
  }

  /** Append packed route words to the GPU route pool (and the CPU mirror); returns the offset. */
  _appendRoute(words) {
    this._ensureRoutes(this.routeWords + words.length);
    const off = this.routeWords;
    this.gpu.write(this.gpu.buffers.get('routes'), words, off * 4);
    this.routeWords += words.length;
    if (!this.routeMirror || this.routeMirror.length < this.routeWords) { const m = new Uint32Array(Math.max(this.routeWords + (1 << 16), (this.routeMirror?.length || 0) * 1.5 | 0)); if (this.routeMirror) m.set(this.routeMirror); this.routeMirror = m; }
    this.routeMirror.set(words, off);
    return off;
  }

  _features() { let f = 0; for (const [k, bit] of Object.entries(FEAT_BITS)) if (this.opts.features[k]) f |= bit; return f; }

  _writeParams(opCount) {
    const o = this.opts, w = this.world, lo = this.layoutOff, acc = o.accident, fc = o.fuelCfg;
    const set = (k, x) => this.paramView.set(0, k, x);
    set('cap', this.cap); set('E', w.E); set('N', w.N); set('K', w.K);
    set('T', this.T); set('wBits', this.wBits); set('wMask', (1 << this.wBits) - 1); set('hMask', (1 << (o.tableBits - this.wBits)) - 1);
    set('ptsOff', lo.ptsOff); set('laneOffOff', lo.laneOffOff); set('adjOff', lo.adjOff); set('ebOff', lo.ebOff);
    set('dt', this.dt); set('time0', this.time); set('step0', this.step); set('features', this._features());
    set('cellSize', o.cellSize); set('leaderLookahead', o.leaderLookahead); set('junctionMargin', 12); set('collisionInterval', o.collisionInterval);
    set('s0', o.s0); set('collideShrink', 0.9); set('speedFactorMin', o.speedFactorMin); set('speedFactorSpan', o.speedFactorSpan);
    set('laneCheckInterval', Math.min(o.laneCheckInterval, 3e38)); set('yieldLook', o.yieldLook); set('stopMargin', o.stopMargin); set('stuckTimeout', o.stuckTimeout);
    set('spawnBlockedTimeout', o.spawnBlockedTimeout); set('spawnCount', this.queueHigh); set('opCount', opCount); set('decideEvery', o.decideEvery);
    set('accRate', acc.ratePerKm); set('accSpeedRef', acc.speedRef); set('accCloseGap', acc.closeGap); set('accMergeFactor', acc.mergeFactor);
    set('accMergeDist', acc.mergeDist); set('rubberDist', SEC.rubberneckDist); set('rubberFactor', SEC.rubberneckFactor); set('towTime', fc.towMedian);
    set('lowFuelFrac', fc.lowFraction); set('seed', o.seed); set('evCap', o.evCap); set('inSize', this.IN);
    set('qrWindow', SEC.queueRiskWindow); set('qrDist', SEC.queueRiskDist); set('qrFactor', SEC.queueRiskFactor); set('nBlocks', this.activeCrashes.length ? o.maxBlocks : 0);
    set('blkOff', lo.blkOff); set('maxCellReach', 6); set('fuelScale', o.fuelScale); set('anticipation', o.anticipation);
    set('mergeLookahead', o.mergeLookahead); set('collisionTol', o.collisionTolerance); set('contactSpeed', o.contactSpeed); set('lateralTol', o.lateralTolerance);
    set('offroadMargin', o.offroadMargin); set('squeezeSpeed', o.squeezeSpeed); set('prepBase', o.prepBase); set('prepPerLane', o.prepPerLane);
    set('fuelCheckInterval', fc.checkInterval); set('fuelMinFillToSkip', fc.minFillToSkip); set('fuelLowFraction', fc.lowFraction); set('fuelPlanFactor', fc.planFactor);
    set('fuelReserveKm', fc.reserveKm); set('fuelPlanSpeed', fc.planSpeed); set('progressW', o.progressWeight ?? 1.0); set('jerkW', o.jerkWeight ?? 0.5); set('nAgents', this.nAgents);
    { const cs = o.crashSev; set('crashOn', cs ? 1 : 0); set('crashBase', cs ? cs.base : 0); set('crashK', cs ? cs.k : 0); set('crashCap', cs ? cs.cap : 0); set('crashFault', cs ? cs.fault : 0); set('crashFail', cs ? cs.fail : 0); }   // rewardVersion 2 (sim/reward.js CRASH_SEV; null = flat -20)
    set('geoOff', lo.geoOff); set('lgvOff', lo.lgvOff); set('hopOff', lo.hopOff); set('hasLG', w.hasLG ? 1 : 0);
    set('ctlEdgeOff', lo.ctlEdgeOff); set('ctlListOff', this.ctlPack ? lo.ctlEdgeOff + this.ctlPack.listOff : 0); set('ctlApOff', lo.ctlApOff); set('ctlCOff', lo.ctlCOff); set('hasCtl', this.ctlPack && o.features.controls !== false ? 1 : 0);
    set('ctlLook', o.controlLookahead ?? 250); set('rampStart', this.ctlPack ? this.ctlPack.rampStart : 6 * 3600); set('rampEnd', this.ctlPack ? this.ctlPack.rampEnd : 10 * 3600); set('freewayShoulder', o.freewayShoulder); set('pairOff', lo.pairOff); set('pairN', this.pairN || 0); set('labelMode', 0);
    set('jMouthOff', lo.jMouthOff); set('jMovOff', lo.jMovOff); set('jEdgeMovOff', lo.jEdgeMovOff); set('jPtsOff', lo.jPtsOff); set('jConfOff', lo.jConfOff); set('jIntoOff', lo.jIntoOff ?? 0);
    set('jSight', o.junctionSight ?? 0); set('pathSight', o.pathSight ?? 0); set('dlBreak', o.deadlockBreak ?? 0); set('spawnLane', o.spawnLaneRoute ?? 0);   // docs/scale-crashes.md (sim.js SIM_DEFAULTS)
    set('rerouteSight', o.rerouteSight ?? 2000); set('passWrecks', o.passWrecks ? 1 : 0); set('trafficOn', this.trafficPenalty && this.trafficPenalty.size && o.reroute !== false ? 1 : 0); set('steerMode', 1 | (o.actionInput === 'rate' ? 2 : 0) | (o.gradeObs === false ? 0 : 8));   // bit 0 position mode, 1 rate actions, 3 grade obs (inputs 41..43)
    { set('rlOn', o.jerkReward ? 1 : 0); set('rlRec', 0); set('rlSeed', 1); set('rlChunk', 0); set('rlStd0', 1); set('rlStd1', 1); set('rlStd2', 1); set('rlRowCap', 0); }   // rlOn = steering-smoothness reward term (integrate.wgsl)
    set('hasJ', this.jPack ? 1 : 0); set('jM', this.jPack ? this.jPack.M : 0); set('jHeadOff', this.cap * CTL_WORDS);
    set('layerGap', o.layerGap); set('pairCooldown', o.pairCooldown); set('touchKmh', o.touchKmh); set('shoulder', o.shoulder);
    set('oppositeMargin', o.oppositeMargin); set('laneEndLookahead', o.laneEndLookahead); set('contactCap', o.contactCap); set('physDt', o.physDt);
    this.gpu.write(this.paramsBuf, this.paramData);
  }

  /** Per-agent reward sums and outbound outcome codes (agents buffer x / y & 255). */
  async readRewardSums() {
    const u = new Uint32Array(await this.gpu.read(this.gpu.buffers.get('agents'), this.nAgents * 16)), f = new Float32Array(u.buffer);
    const R = new Float32Array(this.nAgents), out = new Uint8Array(this.nAgents);
    for (let a = 0; a < this.nAgents; a++) { R[a] = f[4 * a]; out[a] = u[4 * a + 1] & 255; }
    return { R, outcome: out };
  }

  /**
   * External policy actions for the next decision(s): Float32Array(3 * n) in slot order [steer, gas, brake] per slot,
   * starting at slot `offsetSlots`. steer = target wheel position / maxSteer in [-1, 1] (opts.actionInput 'rate':
   * a steer rate in [-1, 1] of maxSteerRate, integrated). Every DRIVING slot reads its triple at each decision
   * (opts.decideEvery steps); the values persist in the buffer until overwritten.
   */
  setActions(actions, offsetSlots = 0) {
    if (!(actions instanceof Float32Array)) actions = Float32Array.from(actions);
    if (offsetSlots + actions.length / 3 > this.cap) throw new Error('GPUSim.setActions: more actions than slots');
    this.gpu.write(this.gpu.buffers.get('actions'), actions, offsetSlots * 12);
  }
  /** Observation vectors of every slot (Float32Array(cap * obsSize)); rows of non-DRIVING slots are stale. */
  async readObs() {
    if (this.obsMode === 'none') throw new Error("GPUSim.readObs: created with obs 'none'");
    return new Float32Array(await this.gpu.read(this.gpu.buffers.get('inputs'), this.cap * this.IN * 4));
  }
  get obsSize() { return this.obsMode === 'none' ? 0 : this.IN; }

  // ------------------------------------------------------------------ stepping
  _dispatch(pass, k, bgs, x) {
    pass.setPipeline(k.pipeline);
    for (let g = 0; g < bgs.length; g++) pass.setBindGroup(g, bgs[g]);
    pass.dispatchWorkgroups(Math.max(1, x));
  }

  _stages(stepIndex, first, opCount) {
    const K = this.K, BG = this.BG, cap = this.cap, o = this.opts, f = o.features;
    const g = (n, wg) => Math.ceil(n / wg);
    const st = [];
    st.push(['tick', (p) => this._dispatch(p, K.tick, BG.tick, 1)]);
    if (first && opCount) st.push(['ops', (p) => this._dispatch(p, K.ops, BG.ops, g(opCount, 64))]);
    const fresh = first && this._newPose;   // scatter / import: GPU-side pose (LaneGeometry) for F_NEW cars
    if (this.queueHigh || this.lot || fresh) st.push(['spawn', (p) => {
      if (fresh) this._newPose = false;
      if (this.lot) this._dispatch(p, K.leave, BG.leave, g(cap, 128));
      if (this.queueHigh) this._dispatch(p, K.spawn, BG.spawn, g(this.queueHigh, 64));
      this._dispatch(p, K.newPose, BG.newPose, g(cap, 256));
    }]);
    if (f.collisions) st.push(['contacts', (p) => {
      this._dispatch(p, K.hclear, BG.hclear, g(this.T + 1, 256));
      this._dispatch(p, K.hcount, BG.hcount, g(cap, 256));
      const hl = this._hscan;
      for (let l = 0; l < hl.length; l++) this._dispatch(p, K.scanBlocks, BG.hscanB[l], hl[l].blocks);
      for (let l = hl.length - 2; l >= 0; l--) this._dispatch(p, K.addSums, BG.hscanA[l], hl[l].blocks);
      this._dispatch(p, K.hscatter, BG.hscatter, g(cap, 256));
    }]);
    if (f.collisions) st.push(['barrier', (p) => this._dispatch(p, K.barrier, BG.barrier, g(cap, 128))]);
    if (f.collisions) st.push(['detect', (p) => this._dispatch(p, K.detect, BG.detect, g(cap, 128))]);
    if (f.collisions) st.push(['resolve', (p) => {
      const wg = Math.ceil(o.contactCap / 128);
      for (let r = 0; r < (o.resolveRounds ?? 12); r++) {
        this._dispatch(p, K.ownClear, BG.ownClear, wg); this._dispatch(p, K.ownMin, BG.ownMin, wg); this._dispatch(p, K.resolveRound, BG.resolveRound, wg);
      }
      this._dispatch(p, K.resolve, BG.resolve, 1);
    }]);
    st.push(['lists', (p) => {
      this._dispatch(p, K.clear, BG.clear, g(this.nCounts, 256));
      this._dispatch(p, K.count, BG.count, g(cap, 256));
      const lv = this._scanLevels;
      for (let l = 0; l < lv.length; l++) this._dispatch(p, K.scanBlocks, BG.scan[l], lv[l].blocks);
      for (let l = lv.length - 2; l >= 0; l--) this._dispatch(p, K.addSums, BG.add[l], lv[l].blocks);
      this._dispatch(p, K.scatter, BG.scatter, g(cap, 256));
      this._dispatch(p, K.sortLanes, BG.sortLanes, g(cap, 256));
    }]);
    if (this.jPack && !f.empty) st.push(['jlists', (p) => { this._dispatch(p, K.jclear, BG.jclear, g(this.jPack.M + this.world.E, 256)); this._dispatch(p, K.jclearOff, BG.jclearOff, g(this.world.E, 256)); this._dispatch(p, K.jlink, BG.jlink, g(cap, 256)); }]);
    st.push(['leaders', (p) => this._dispatch(p, K.leaders, BG.leaders, g(cap, 128))]);
    if (this.jPack && !f.empty) st.push(['junctions', (p) => {
            if (f.collisions) this._dispatch(p, K.jperceive, BG.jperceive, g(cap, 128));
      this._dispatch(p, K.cpropose, BG.cpropose, g(cap, 128)); this._dispatch(p, K.capply, BG.capply, g(cap, 128));
      if ((o.deadlockBreak ?? 0) !== 0) { this._dispatch(p, K.jdlFind, BG.jdlFind, g(cap, 128)); this._dispatch(p, K.jdlApply, BG.jdlApply, g(cap, 128)); }   // sim.js _breakDeadlocks (after _courtesy)
      this._dispatch(p, K.jyield, BG.jyield, g(cap, 128));
      if ((this._closedN || (this.trafficPenalty && this.trafficPenalty.size)) && o.reroute !== false && stepIndex % Math.max(1, Math.round(2 / this.dt)) === 0) this._dispatch(p, K.rerouteScan, BG.rerouteScan, g(cap, 128));
    }]);
    if (f.policy) {
      if (this.obsMode !== 'none') st.push(['sense', (p) => this._dispatch(p, K.sense, BG.sense, g(cap, 128))]);
      if (this.v3) st.push(['perceive', (p) => this._dispatch(p, K.perceive, BG.perceive, g(cap, 64))]);
      st.push(['act', (p) => this._dispatch(p, K.act, BG.act, g(cap, 128))]);
    }
    st.push(['integrate', (p) => this._dispatch(p, K.integrate, BG.integrate, g(cap, 256))]);
    return st;
  }

  _encodeSteps(enc, n, opCount, profile) {
    for (let k = 0; k < n; k++) {
      const stepIndex = this.step + k + 1;
      let stages = this._stages(stepIndex, k === 0, opCount);
      if (this.opts.skipStages) stages = stages.filter(([nm]) => !this.opts.skipStages.includes(nm));   // diagnosis only
      if (profile && k === 0) {
        for (const [label, fn] of stages) { const p = enc.beginComputePass(this.profiler.pass(label)); fn(p); p.end(); }
      } else {
        const p = enc.beginComputePass();
        for (const [, fn] of stages) fn(p);
        p.end();
      }
    }
  }

  /** Run n sim steps in one submit; resolves after this frame's stats / events were processed. */
  runSteps(n, { profile = this.opts.profile, readback = true } = {}) {
    if (n <= 0) return Promise.resolve();
    const tEnd = this.time + n * this.dt, dbg = this.opts.debugPhases ? (p) => console.log(`[phase] ${this.clock} ${p}`) : null;
    if (dbg) dbg('spawns'); this._enqueueSpawns(tEnd);
    if (dbg) dbg('scenes'); this._scenes();
    if (dbg) dbg('pumps'); this._pumps();
    if (dbg) dbg('refuels'); this._processRefuels();
    if (dbg) dbg('reroutes');
    if (this.jPack) { this._updateClosed(); this._processReroutes(); if (this.time >= (this._nextTraffic ?? -1e9) && !this._trafficBusy) { this._nextTraffic = this.time + 60; this._trafficSample(); } }
    if (dbg) dbg('ops'); const opCount = this._flushOps();
    this._writeParams(opCount);
    if (dbg) dbg('encode');
    const enc = this.device.createCommandEncoder();
    this._encodeSteps(enc, n, opCount, profile);
    const profiled = profile ? this.profiler.resolve(enc) : false;
    let stage = null;
    if (readback) {
      stage = this._staging.find((s) => !s.busy);
      if (stage) {
        enc.copyBufferToBuffer(this.gpu.buffers.get('io'), 0, stage.buf, 0, this.ioWords * 4);
        enc.clearBuffer(this.gpu.buffers.get('io'), L.IO.EVCOUNT * 4, 4);
        stage.busy = true;
      }
    }
    this.device.queue.submit([enc.finish()]);
    this.time = tEnd; this.step += n;
    const work = [];
    if (profiled) work.push(this.profiler.collect().then((ms) => { if (ms) this.kernelMs = ms; }));
    if (stage) work.push(stage.buf.mapAsync(GPUMapMode.READ).then(() => {
      const data = new Uint32Array(stage.buf.getMappedRange().slice(0));
      stage.buf.unmap(); stage.busy = false;
      if (this.opts.debugPhases) console.log(`[phase] ${this.clock} ingest`);
      this._ingest(data);
      if (this.opts.debugPhases) console.log(`[phase] ${this.clock} ingested`);
    }));
    return Promise.all(work);
  }

  advance(realDt) {
    this.accumulator += realDt * this.speed;
    let steps = Math.floor(this.accumulator / this.dt);
    if (steps > this.opts.maxStepsPerFrame) { steps = this.opts.maxStepsPerFrame; this.lagging = true; this.accumulator = steps * this.dt; }
    else this.lagging = false;
    this.accumulator -= steps * this.dt;
    this._lastPromise = this.runSteps(steps);
    return steps;
  }

  async sync() { await this.device.queue.onSubmittedWorkDone(); if (this._lastPromise) await this._lastPromise; }

  // ------------------------------------------------------------------ readback ingest
  _ingest(io) {
    this.gpuStats.set(io.subarray(L.IO.STATS, L.IO.STATS + 64));
    const n = io[L.IO.EVCOUNT], cap = this.opts.evCap;
    if (n > cap) this.counters.eventsDropped += n - cap;
    const m = Math.min(n, cap), base = L.IO.EVDATA;
    const f32 = new Float32Array(io.buffer);
    for (let k = 0; k < m; k++) {
      const t = io[base + 4 * k];
      if ((t & 255) === L.EV.WRECK) this.wrecks.set(io[base + 4 * k + 1], { lo: (t >> 8) & 15, hi: (t >> 12) & 15, type: (t >> 16) & 255, edge: io[base + 4 * k + 2], s: f32[base + 4 * k + 3] });
    }
    for (let k = 0; k < m; k++) {
      const o = base + 4 * k, t = io[o] & 255, slot = io[o + 1], a = io[o + 2], b = io[o + 3];
      switch (t) {
        case L.EV.ARRIVE: if (this.lot && this.slotStation[slot] >= 0 && this._inLot?.has(slot)) { this._inLot.delete(slot); this.lot--; } this._release(slot, 'arrived', f32[o + 3], this._tripX.get(slot)); this._tripX.delete(slot); break;
        case L.EV.CTLVIOL: { const kind = (b & 255) === 1 ? 'red' : 'stop', C = this.opts.controls; if (kind === 'red') this.counters.signalViolations++; else this.counters.stopViolations++;
          const ae = C ? C.apEdge[a] : -1; this._event('signalViolation', slot, { approach: a, control: kind, controlId: C ? C.apControl[a] : null, node: ae >= 0 ? this.graph.edges[ae].to : null, speed: (b >>> 8) / 10 }); break; }
        case L.EV.TRIPX: this._tripX.set(slot, { distance: f32[o + 2], duration: f32[o + 3] }); break;
        case L.EV.CRASHX_V: this._crashX.set(slot, { slot, other: a, preSpeedA: (b & 0xFFFF) / 10, preSpeedB: (b >>> 16) / 10 }); break;
        case L.EV.MISSED: {   // layout v8: a strict diverge / a turn from the wrong lane took the car onto another out-edge: re-plan from it
          const via = (a >>> 31) === 1, took = a & 0x7fffffff;
          this.counters.missedExits = (this.counters.missedExits || 0) + 1; this._event('missedExit', slot, { took, viaJunction: via });
          (this.rerouteQueue ||= []).push({ slot, edge: took, trip: this.slotTrip[slot], missed: true, prefix: via ? b : -1, t: this.time }); break;
        }
        case L.EV.REROUTE: (this.rerouteQueue ||= []).push({ slot, edge: a, trip: this.slotTrip[slot], slow: (b >>> 31) === 1, rp: b & 0x7fffffff, t: this.time }); break;
        case L.EV.CRASHX_M: { const c = this._crashX.get(slot); if (c) { c.maneuver = ['cruise', 'lane-change', 'merge', 'junction', 'queue-approach', 'wreck'][a & 255] || 'cruise'; c.anyCrashed = (a & 768) !== 0; c.strikeEdge = b; } break; }
        case L.EV.CRASHX_N: { const c = this._crashX.get(slot); if (c) { c.nx = f32[o + 2]; c.nz = f32[o + 3]; } break; }
        case L.EV.CRASHX_P: { const c = this._crashX.get(slot); if (c) { c.x = f32[o + 2]; c.z = f32[o + 3]; } break; }
        case L.EV.CRASHX_H: if (this._crashX.has(slot)) { const hh = (u) => u / 65536 * 2 * Math.PI - Math.PI; Object.assign(this._crashX.get(slot), { headingA: hh(a & 0xFFFF), headingB: hh(a >>> 16), closing: (b & 0xFFFF) / 10, laneA: (b >>> 16) & 255, laneB: b >>> 24 }); } break;
        case L.EV.STUCK: this.counters.stuckLabels = (this.counters.stuckLabels || 0) + 1; this._event('stuck', slot, { trip: this.slotTrip[slot], stoppedFor: f32[o + 3] }); break;   // layout v8: a label, not a failure
        case L.EV.SPAWN: this.queueFree.push(a); this.counters.spawned++; this._event('spawn', slot, { trip: this.slotTrip[slot], lane: b }); break;
        case L.EV.LOWFUEL: this.refuelQueue.push({ slot, edge: a, atSpawn: b === 1 }); break;
        case L.EV.STATION: this._enterStation(slot); break;
        case L.EV.REENTER: if (this._inLot?.delete(slot)) this.lot--; this.slotStation[slot] = -1; this.counters.reentered++; this._event('refuelLeave', slot, {}); break;
        case L.EV.CRASH: { const cx = this._crashX.get(slot) || null; this._crashX.delete(slot); this._crashV4(slot, a, (b & 0xFFFF) / 10, (b >>> 16) / 10, cx); break; }
        case L.EV.CONTACT: { const dv = Math.max((b & 0xFFFF) / 10, (b >>> 16) / 10);
          // sim.js: minor crash (delta-v >= 3 km/h) on a freeway-class road: drivers pull onto the shoulder 5-15 min
          if (dv >= 3) { if (io[o] & 256) this._pull(slot); if ((io[o] & 512) && a !== 0xFFFFFFFF) this._pull(a); } this.counters.minorContacts++; this.counters.crashEvents++; this._bySev(dv); if (this.opts.contactEvents) this._event('contact', slot, { other: a === 0xFFFFFFFF ? -1 : a, kind: a === 0xFFFFFFFF ? 'barrier' : 'vehicle', deltaVKmh: [(b & 0xFFFF) / 10, (b >>> 16) / 10] }); break; }
        case L.EV.STRANDED: this._strand(slot); break;
        case L.EV.CRASHPEN: (this.crashPenLog ||= []).push({ slot, t: +this.time.toFixed(1), penalty: +f32[o + 2].toFixed(3), dvKmh: (b & 0xFFFF) / 10, share: +(((b >>> 16) & 255) / 255).toFixed(3), disabled: (b >>> 24) & 1, barrier: (b >>> 25) & 1 }); if (this.crashPenLog.length > 200000) this.crashPenLog.shift(); break;   // rewardVersion 2 log (contacts.wgsl crashCharge)
        default: break;
      }
    }
  }

  _event(type, slot, extra) {
    const ev = { time: this.time, type, slot, ...extra };
    this.events.push(ev); if (this.events.length > 20000) this.events.splice(0, 10000);
    if (this.onEvent) this.onEvent(ev);
  }

  _release(slot, outcome, fuel = 0, trip = null) {
    if (this.slotTrip[slot] === -2) return;
    const tr = this.slotTrip[slot];
    if (this.pop && tr >= 0) {
      const id = tr >> 1;
      if (outcome === 'arrived') this.fuelLeft[id] = fuel;
      else if (outcome === 'towed') this.fuelLeft[id] = 0.2 * TANK_L[this.pop.arrays.vehicleType[id]];
      if ((tr & 1) === 0 && !this.outcome[id]) this.outcome[id] = { arrived: 1, crash: 2, stranded: 3, stuck: 4 }[outcome] || 0;
    }
    this.slotTrip[slot] = -2;
    if (this._arena) this._arenaFree(slot);
    this.free.push(slot);
    this._event(outcome, slot, trip ? { trip: tr, duration: trip.duration, distance: trip.distance } : { trip: tr });
  }
  _markOutcome(slot, code) {
    const tr = this.slotTrip[slot];
    if (this.pop && tr >= 0 && (tr & 1) === 0 && !this.outcome[tr >> 1]) this.outcome[tr >> 1] = code;
  }

  // ------------------------------------------------------------------ ops
  op(code, slot, a = 0, b = 0) { this.opsPending.push(code, slot, a >>> 0, b >>> 0); }
  _flushOps() {
    const n = Math.min(this.opsPending.length / 4, this.opts.opCap);
    if (!n) return 0;
    this.gpu.write(this.gpu.buffers.get('ops'), new Uint32Array(this.opsPending.slice(0, n * 4)));
    this.opsPending = this.opsPending.slice(n * 4);
    return n;
  }

  // ------------------------------------------------------------------ population & spawning
  /**
   * Use a population (src/sim/population.js SoA: routes {flat, offset}, arrays, deltas, or
   * src/gpu/popPack.js). Trip k = 2*id + leg. stations: data.js stationsOnGraph output.
   */
  setPopulation(pop, { stations = null } = {}) {
    const E = this.graph.edges.length;
    if (pop.meta && pop.meta.edgeCount != null && pop.meta.edgeCount !== E) throw new Error(`population pack was built for ${pop.meta.edgeCount} edges but the world has ${E} (world rebuilt?): regenerate it, e.g. node src/gpu/tools/packPopulation.mjs ${pop.meta.n || pop.arrays.depart.length} ${pop.meta.seed ?? 42}`);
    if (sigMismatch(pop.meta, this.graph)) throw new Error(`population pack was built for graph ${pop.meta.graphSig}, the world is ${graphSig(this.graph)} (roads rebuilt: node ids or lengths changed): regenerate it, e.g. node src/gpu/tools/packPopulation.mjs ${pop.meta.n || pop.arrays.depart.length} ${pop.meta.seed ?? 42}`);
    if (pop.meta && pop.meta.edgeCount != null && !pop.meta.graphSig) console.warn('population pack has no graph signature (built before 2026-10-24 ~20:00?): node ids may be stale, regenerate it');
    this.pop = pop;
    const { flat, offset } = pop.routes;
    const packed = packRoutes(this.graph, flat, offset);
    this.routeWords = 0; this.routeMirror = null;
    this._ensureRoutes(packed.length + this.opts.routeSlack);
    this.tripRouteBase = this._appendRoute(packed);
    this.tripRouteEnd = this.routeWords;
    const A = pop.arrays, n = A.depart.length;
    if (n > this.nAgents) { this.nAgents = n; this.gpu.storage('agents', n * 16); this._makeBindGroups(); }
    const trips = [];
    for (let i = 0; i < n; i++) {
      if (offset[2 * i + 1] > offset[2 * i]) trips.push(2 * i); else this.counters.noRoute++;
      if (this.opts.returnTrips && Number.isFinite(A.returnDepart[i]) && offset[2 * i + 2] > offset[2 * i + 1]) trips.push(2 * i + 1);
    }
    const dep = (k) => (k & 1 ? A.returnDepart[k >> 1] : A.depart[k >> 1]);
    trips.sort((a, b) => dep(a) - dep(b) || a - b);
    this.trips = Int32Array.from(trips); this.tripDepart = Float32Array.from(trips, dep); this.tripPtr = 0;
    this.fuelLeft = new Float32Array(n).fill(-1);
    this.outcome = new Uint8Array(n);
    this.setStations(stations);
    return this;
  }

  /** Gas stations (src/sim/data.js stationsOnGraph) -> src/sim/fuel.js StationNetwork for refuel plans + pumps. */
  setStations(stations) {
    if (!stations || !stations.length || !this.opts.features.fuel) return;
    this.router ||= makeRouter(this.graph, { weight: 1.25 });
    this.stations = new simFuel.StationNetwork(this.graph, stations, this.router, this.opts.fuelCfg);
    // plans run in Web Workers when available (each plan is 100-800 ms of A*); the main thread only applies results
    if (typeof Worker !== 'undefined' && this.opts.fuelWorkers !== 0 && !this._fuelPool) {
      const n = this.opts.fuelWorkers ?? 2, dataBase = String(this.opts.dataBase || new URL('../../data/', import.meta.url));
      // bounded in-flight work per worker (out[k] <= FUEL_MAXQ): refuel plans go first, reroutes wait in rerouteQueue
      // (stale ones dropped) - an unbounded backlog of reroute A* calls used to starve the refuel plans at full scale
      const pool = this._fuelPool = { ws: [], out: [], ready: 0, rr: 0, seq: 0, pending: new Map(), done: [], failed: false };
      for (let k = 0; k < n; k++) {
        const w = new Worker(new URL('./fuelWorker.js', import.meta.url), { type: 'module' });
        w.onmessage = (m) => {
          const d = m.data;
          if (d.route !== undefined) { const req = pool.pending.get(d.id); pool.pending.delete(d.id); if (req) { pool.out[req.w]--; (pool.rdone ||= []).push({ ...req, route: d.route }); this._pumpSoon(); } return; }
          if (d.ready) { if (d.stations === this.stations.list.length) pool.ready++; else { pool.failed = true; console.warn('fuel worker: station list differs, planning on the main thread'); } return; }
          if (d.error) { pool.failed = true; console.warn('fuel worker failed, planning on the main thread:', d.error); return; }
          const req = pool.pending.get(d.id); pool.pending.delete(d.id);
          if (req) { pool.out[req.w]--; pool.done.push({ ...req, plan: d.plan }); this._pumpSoon(); }
        };
        w.onerror = (e) => { pool.failed = true; console.warn('fuel worker error:', e.message); };
        // the worker gets a compact routing graph + the station list (no world load per worker, docs/scale-crashes.md)
        this._routingPack ||= routingPack(this.graph);
        w.postMessage({ init: true, dataBase, cfg: this.opts.fuelCfg, pack: this._routingPack, stations: this.stations.list.map(({ idx, ...s }) => s) });
        pool.ws.push(w); pool.out.push(0);
      }
    }
  }

  _rand() { let x = this.rngState; x ^= x << 13; x ^= x >>> 17; x ^= x << 5; this.rngState = x >>> 0; return this.rngState / 4294967296; }
  _gauss() { let u = 0; while (u === 0) u = this._rand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * this._rand()); }

  /** VPar words (src/sim/vehicle.js alloc semantics) into `out` at word offset o. */
  _par(out, o, type, traits, routeOff, routeLen, workStart, routeLength, trip) {
    const t = VEHICLE_TYPES[type] || VEHICLE_TYPES[0];
    const f = new Float32Array(out.buffer, out.byteOffset);
    const rt = traits.reactionTime ?? DEF_TRAITS.reactionTime;
    out[o] = routeOff; out[o + 1] = routeLen; out[o + 2] = type; out[o + 3] = Math.min(L.CTRL_RING - 1, Math.max(0, Math.round(rt / this.dt)));
    f[o + 4] = t.length; f[o + 5] = t.width; f[o + 6] = t.maxSpeed;
    const lim = driverLimits(t, traits);
    f[o + 7] = lim.maxAccel; f[o + 8] = lim.maxDecel;
    f[o + 9] = traits.preferredHeadway ?? DEF_TRAITS.preferredHeadway;
    f[o + 10] = traits.aggression ?? DEF_TRAITS.aggression; f[o + 11] = traits.patience ?? DEF_TRAITS.patience;
    f[o + 12] = rt; f[o + 13] = workStart; f[o + 14] = routeLength; out[o + 15] = trip;
    f[o + 16] = t.wheelbase ?? 2.8; f[o + 17] = t.maxSteer ?? 0.6; f[o + 18] = t.maxSteerRate ?? 0.7; f[o + 19] = t.maxSpeed;
  }

  /**
   * Curriculum stage 1 'sparse' (sim.js placeWrecks): n static stopped wrecks on random main-tier edges
   * (rank >= 3, longer than 150 m), each blocking its lane for the whole day (scene 'wreck').
   */
  /**
   * Training obstacles (sim.js placeObstacles, layout v8; same spec, RNG and placement as the CPU):
   *   { count = 40, kinds: ['midblock', 'afterCurve', 'afterMerge', 'junction', 'rampEnd', 'partial', 'slow', 'braking'],
   *     seed = 1, slowKmh: [10, 20], brakeEvery: [15, 40] }
   * Static kinds are permanent scenes (blockDuration 1e9); 'slow' cars drive at slowKmh, 'braking' leads brake
   * hard (1.5-3 s) every brakeEvery s (drive.wgsl, schedule in ctl words 24..27). Listed in this.obstacles.
   * Or an explicit list [{ edge, at, lane | lat, type }] (sim staticObstacles / src/scenarios placeScenarioObstacles):
   * stopped, disabled cars at s = clamp(at, 1, length - 1), lat = lat ?? (lane + 0.5) x lane width, vehicle type
   * `type` (default 0), each a permanent scene.
   */
  placeObstacles(spec = {}) {
    const g = this.graph, J = g.J, LG = g.laneGeom;
    const laneW = (e, s) => (LG && LG.laneWidth ? LG.laneWidth(e.idx, Math.min(e.length, Math.max(0, s))) : LANE_W);
    const R = makeRng(spec.seed ?? 1), count = spec.count ?? 40;
    const kinds = spec.kinds || ['midblock', 'afterCurve', 'afterMerge', 'junction', 'rampEnd', 'partial', 'slow', 'braking'];
    const slowKmh = spec.slowKmh || [10, 20], brakeEvery = spec.brakeEvery || [15, 40];
    const pools = {};
    const pool = (k) => {
      if (pools[k]) return pools[k];
      const out = [];
      for (const e of g.edges) {
        if (k === 'midblock' || k === 'partial' || k === 'slow' || k === 'braking') { if (e.rank >= 1 && e.length >= 150) out.push(e.idx); continue; }
        if (k === 'rampEnd') { if (e.gore !== undefined && e.gore >= 0 && e.length > 30) out.push(e.idx); continue; }
        if (k === 'afterCurve' || k === 'afterMerge') {
          if (e.length < 90) continue;
          const ins = []; for (let q = g.inStart[e.from]; q < g.inStart[e.from + 1]; q++) if (g.inEdges[q] !== e.reverse) ins.push(g.edges[g.inEdges[q]]);
          if (k === 'afterMerge' && ins.length >= 2 && ins.every((a) => simHopMap(g, a, e).strict)) out.push(e.idx);
          if (k === 'afterCurve' && ins.length === 1) { let d = e.headingStart - ins[0].headingEnd; d = Math.abs(Math.atan2(Math.sin(d), Math.cos(d))); if (d > 0.3 && simHopMap(g, ins[0], e).strict) out.push(e.idx); }
        }
      }
      return (pools[k] = out);
    };
    const VW = L.VEH_WORDS, PW = L.PAR_WORDS, gb = this.gpu.buffers;
    const writeCar = (e, route, s, lat, { conn = -1, state = L.STATE.CRASHED, speed = 0, maxSpeed = null, type = 0 } = {}) => {
      if (!this.free.length) return -1;
      const slot = this.free.pop();
      const routeOff = this._appendRoute(packRoutes(g, route, Uint32Array.of(0, route.length)));
      const veh = new Uint32Array(VW), vf = new Float32Array(veh.buffer), vi = new Int32Array(veh.buffer), par = new Uint32Array(PW), pf = new Float32Array(par.buffer);
      let total = 0; for (const x of route) total += g.edges[x].length;
      this._par(par, 0, type, DEF_TRAITS, routeOff, route.length, 0, total, 0xFFFFFFFE);
      if (maxSpeed != null) { pf[6] = maxSpeed; pf[19] = maxSpeed; }
      const ln = Math.max(0, Math.min(e.lanes - 1, Math.floor(lat / laneW(e, s))));
      veh[0] = e.idx; vf[1] = s; vf[2] = speed; vf[4] = lat; veh[7] = state; veh[10] = L.FLAG.NEW | (state === L.STATE.CRASHED ? L.FLAG.DISABLED : 0);
      vi[11] = this.step; vf[12] = -1; veh[14] = L.packLanes(ln, ln, ln, -1); vf[15] = -1e9; vf[16] = 10; vf[17] = 50; vf[18] = this.time;
      vi[20] = -1; vf[21] = 1e9; vi[23] = -1; vf[30] = conn >= 0 ? conn + 1 : 0; vf[28] = total; vf[29] = this.time; vf[31] = 1e9;
      this.gpu.write(gb.get('veh'), veh, slot * VW * 4); this.gpu.write(gb.get('par'), par, slot * PW * 4);
      this.slotTrip[slot] = -3; this._newPose = true;
      return slot;
    };
    const staticCar = (e, s, lat, conn = -1, type = 0) => {
      const slot = writeCar(e, Int32Array.of(e.idx), s, lat, { conn, type });
      if (slot < 0) return -1;
      const ln = Math.max(0, Math.min(e.lanes - 1, Math.floor(lat / laneW(e, s))));
      this.wrecks.set(slot, { lo: ln, hi: ln, type: 0, edge: e.idx, s });
      this._inCrash.add(slot);
      this._openScene([slot], { cause: 'obstacle', severity: 'wreck', impactSpeed: 0, impactMph: 0, lanesBlocked: 1, blockDuration: 1e9, clearDuration: 1e9 }, false);
      return slot;
    };
    const walk = (e0) => { const r = [e0.idx]; let e = e0; for (let k = 0; k < 30; k++) { const outs = []; for (let q = g.outStart[e.to]; q < g.outStart[e.to + 1]; q++) if (g.outEdges[q] !== e.reverse) outs.push(g.outEdges[q]); if (!outs.length) break; e = g.edges[outs[R.int(outs.length)]]; r.push(e.idx); } return Int32Array.from(r); };
    const placed = [], brakes = [];
    if (Array.isArray(spec)) {
      for (const o of spec) {
        const e = g.edges[o.edge]; if (!e) continue;
        const s = Math.min(e.length - 1, Math.max(1, o.at));
        const slot = staticCar(e, s, o.lat ?? ((o.lane ?? 0) + 0.5) * laneW(e, s), -1, o.type ?? 0);
        if (slot >= 0) placed.push({ slot, kind: 'static', edge: e.idx, s });
      }
      this.obstacles = placed;
      return placed;
    }
    for (let n = 0; n < count; n++) {
      const kind = kinds[n % kinds.length];
      let slot = -1, e = null, s = 0;
      if (kind === 'junction') {
        if (!J || !J.moveCount) continue;
        let m = R.int(J.moveCount), d = J.describe(m), t = 0.5 * d.length;
        for (let tries = 0; tries < 20; tries++) {
          m = R.int(J.moveCount); d = J.describe(m); e = g.edges[d.inEdge];
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
        else if (kind === 'partial') { s = e.length * R.range(0.3, 0.7); const k = e.lanes > 1 ? 1 + R.int(e.lanes - 1) : R.int(2); slot = staticCar(e, s, k * laneW(e, s) + R.range(-0.4, 0.4)); }
        else {
          s = e.length * R.range(0.1, 0.5);
          const route = walk(e);
          if (kind === 'slow') { const vmax = R.range(slowKmh[0], slowKmh[1]) / 3.6; slot = writeCar(e, route, s, (lane + 0.5) * laneW(e, s), { state: L.STATE.DRIVING, speed: vmax, maxSpeed: vmax }); }
          else {
            slot = writeCar(e, route, s, (lane + 0.5) * laneW(e, s), { state: L.STATE.DRIVING, speed: Math.min(e.speedLimit, 15) });
            if (slot >= 0) brakes.push({ slot, next: this.time + R.range(brakeEvery[0], brakeEvery[1]) });
          }
        }
      }
      if (slot >= 0) placed.push({ slot, kind, edge: e.idx, s });
    }
    // braking leads: schedule words in ctl (drive.wgsl draws the following periods with its own RNG)
    for (const b of brakes) {
      // owner tag (word 8 = spawn time) so leaders.wgsl does not reset the record for this new occupant
      const rec = new Float32Array(CTL_WORDS), ru = new Uint32Array(rec.buffer);
      ru[0] = 0xFFFFFFFF; ru[1] = 0xFFFFFFFF; rec[5] = 1e9; rec[7] = 1e9; ru[9] = 0xFFFFFFFF; ru[16] = 0xFFFFFFFF; ru[20] = 0xFFFFFFFF;
      rec[8] = this.time; rec[24] = b.next; rec[25] = -1; rec[26] = brakeEvery[0]; rec[27] = brakeEvery[1];
      this.gpu.write(gb.get('ctl'), rec, b.slot * CTL_WORDS * 4);
    }
    this.obstacles = placed;
    return placed;
  }

  placeWrecks(n) {
    const g = this.graph, cand = [];
    for (const e of g.edges) if (e.rank >= 3 && e.length > 150) cand.push(e.idx);
    const VW = L.VEH_WORDS, PW = L.PAR_WORDS, gb = this.gpu.buffers, placed = [];
    for (let k = 0; k < n && cand.length && this.free.length; k++) {
      const ei = cand[Math.floor(this._rand() * cand.length)], e = g.edges[ei];
      const slot = this.free.pop();
      const routeOff = this._appendRoute(packRoutes(g, Int32Array.of(ei), Uint32Array.of(0, 1)));
      const veh = new Uint32Array(VW), vf = new Float32Array(veh.buffer), vi = new Int32Array(veh.buffer), par = new Uint32Array(PW);
      this._par(par, 0, 0, DEF_TRAITS, routeOff, 1, 0, e.length, 0xFFFFFFFE);
      const lane = Math.floor(this._rand() * e.lanes), sPos = e.length * (0.3 + 0.4 * this._rand());
      veh[0] = ei; vf[1] = sPos; vf[4] = (lane + 0.5) * (g.laneGeom && g.laneGeom.laneWidth ? g.laneGeom.laneWidth(ei, sPos) : LANE_W); veh[7] = L.STATE.CRASHED; veh[10] = L.FLAG.NEW | L.FLAG.DISABLED;
      vi[11] = this.step; vf[12] = -1; veh[14] = L.packLanes(lane, lane, lane, -1); vf[15] = -1e9; vf[16] = 10; vf[17] = 50; vf[18] = this.time;
      vi[20] = -1; vf[21] = 1e9; vi[23] = -1; vf[28] = e.length; vf[29] = this.time; vf[31] = 1e9;
      this.gpu.write(gb.get('veh'), veh, slot * VW * 4); this.gpu.write(gb.get('par'), par, slot * PW * 4);
      this.slotTrip[slot] = -3;
      this.wrecks.set(slot, { lo: lane, hi: lane, type: 0, edge: ei, s: sPos });
      this._inCrash.add(slot);
      this._openScene([slot], { cause: 'wreck', severity: 'wreck', impactSpeed: 0, impactMph: 0, lanesBlocked: 1, blockDuration: 1e9, clearDuration: 1e9 }, false);
      placed.push(slot);
    }
    this._newPose = true;
    return placed.length;
  }

  _routeLength(flat, off, len) { let s = 0; for (let k = off; k < off + len; k++) s += this.graph.edges[flat[k]].length; return s; }

  /**
   * Streaming population (the CPU reference is src/sim/stream.js + Simulation({table, routeFor, slotPool})):
   * the GPU slot pool (capacity) holds only the cars on the road; commuters come from a
   * CommuterTable (SoA sorted by departure, outbound leg) and get their route at departure:
   *   routeFor(id, leg, from, to) -> Int32Array (edges) | null (no route) | undefined (not ready: retried every batch).
   * Default routeFor: a synchronous src/sim/stream.js RouteProvider (A* + OD LRU cache). Routes live in a
   * size-class arena in the GPU route buffer and are freed when the trip ends; the 64-float
   * delta comes from table.deltaFor(id) at departure. Pool full = departures wait (as slotPool).
   */
  async setStreaming(table, { routeFor = null, routes = null, routeAhead = 300, stations = null, deltas = null } = {}) {
    const E = this.graph.edges.length;
    if (table.meta && table.meta.edgeCount != null && table.meta.edgeCount !== E) throw new Error(`commuter table was built for ${table.meta.edgeCount} edges but the world has ${E} (world rebuilt?): regenerate it with node src/gpu/tools/packTable.mjs ${table.n} ${table.meta.seed ?? 42}`);
    if (sigMismatch(table.meta, this.graph)) throw new Error(`commuter table was built for graph ${table.meta.graphSig}, the world is ${graphSig(this.graph)}: regenerate it with node src/gpu/tools/packTable.mjs ${table.n} ${table.meta.seed ?? 42}`);
    if (table.meta && !table.meta.graphSig) console.warn('commuter table has no graph signature: node ids may be stale, regenerate it');
    this.streamDeltas = deltas;   // optional Float32Array(n*64) per commuter (evolution); default table.deltaFor(id)
    // routes: an async pool (src/gpu/streamPack.js RouteWorkers: routeFor / prefetch / flush); departures
    // up to routeAhead sim s ahead are queued so their routes are ready when they leave
    if (routes) { routeFor = routes.routeFor; this.routePool = routes; this.routeAhead = routeAhead; this.prefetchPtr = 0; }
    if (!routeFor) { const { RouteProvider } = await import('../sim/stream.js'); const rp = new RouteProvider(this.graph); routeFor = (id, leg, from, to) => rp.get(from, to); this.routeProvider = rp; }
    this.table = table; this.routeFor = routeFor;
    this.pop = { arrays: table, routes: null, deltas: null, table };   // table has the population.js array names
    const n = table.n;
    if (n > this.nAgents) { this.nAgents = n; this.gpu.storage('agents', n * 16); this._makeBindGroups(); }
    this.trips = null; this.tablePtr = table.indexAt(this.time - 3600); this.pendingStream = []; this.prefetchPtr = this.tablePtr;
    this.fuelLeft = new Float32Array(n).fill(-1);
    this.outcome = new Uint8Array(n);
    this.routeWords = Math.max(this.routeWords || 0, 1); this.routeMirror = null;
    this._arena = { free: new Map(), slot: new Int32Array(this.cap).fill(-1), cls: new Uint8Array(this.cap) };
    this.counters.forcedSpawns = 0; this.streamStats = { routed: 0, waitingRoute: 0, poolFull: 0 };
    this.setStations(stations);
    return this;
  }
  _arenaAlloc(words) {
    const A = this._arena; let c = 6; while ((1 << c) < words.length) c++;
    const fl = A.free.get(c); let off;
    if (fl && fl.length) off = fl.pop();
    else { off = this.routeWords; this._ensureRoutes(off + (1 << c)); this.routeWords = off + (1 << c); }
    this.gpu.write(this.gpu.buffers.get('routes'), words, off * 4);
    if (!this.routeMirror || this.routeMirror.length < this.routeWords) { const m = new Uint32Array(Math.max(this.routeWords + (1 << 16), (this.routeMirror?.length || 0) * 1.5 | 0)); if (this.routeMirror) m.set(this.routeMirror); this.routeMirror = m; }
    this.routeMirror.set(words, off);
    return { off, c };
  }
  _arenaFree(slot) {
    const A = this._arena; if (!A || A.slot[slot] < 0) return;
    const c = A.cls[slot]; let fl = A.free.get(c); if (!fl) A.free.set(c, fl = []);
    fl.push(A.slot[slot]); A.slot[slot] = -1;
  }
  _enqueueStream(tEnd) {
    const T = this.table, st = this.streamStats;
    const entry = new Uint32Array(L.SPAWN_WORDS), ef = new Float32Array(entry.buffer);
    const qbuf = this.gpu.buffers.get('queue'), pol = this.opts.features.policy, g = this.graph;
    const delta = new Float32Array(64);
    const take = () => {
      if (this.pendingStream.length) { const keep = []; for (const id of this.pendingStream) if (!place(id)) keep.push(id); this.pendingStream = keep; }
      while (this.tablePtr < T.n && T.sortedDepart[this.tablePtr] <= tEnd + 2) {
        const id = T.order[this.tablePtr++];
        if (!place(id)) this.pendingStream.push(id);
      }
    };
    const place = (id) => {
      if (!this.free.length || !this.queueFree.length) { st.poolFull++; return false; }
      const r = this.routeFor(id, 0, T.home[id], T.work[id]);
      if (r === undefined) { st.waitingRoute++; return false; }
      if (!r || !r.length) { this.counters.noRoute++; this.outcome[id] = 5; this._event('noRoute', -1, { commuter: id, leg: 0 }); return true; }
      st.routed++;
      const words = packRoutes(g, r, Int32Array.of(0, r.length));
      const slot = this.free.pop(), q = this.queueFree.pop();
      const { off, c } = this._arenaAlloc(words);
      this._arena.slot[slot] = off; this._arena.cls[slot] = c;
      if (this.streamDeltas) delta.set(this.streamDeltas.subarray(id * 64, id * 64 + 64)); else T.deltaFor(id, delta);
      const traits = { aggression: delta[0], patience: delta[1], reactionTime: delta[2], preferredHeadway: delta[3], maxAccel: delta[4], maxDecel: delta[5] };
      const type = T.vehicleType[id];
      let len = 0; for (let k = 0; k < r.length; k++) len += g.edges[r[k]].length;
      this._par(entry, 0, type, traits, off, r.length, T.workStart[id], len, 2 * id);
      const tank = TANK_L[type] ?? 50;
      ef[20] = T.depart[id]; entry[21] = 1; ef[22] = (T.fuelFill ? T.fuelFill[id] : 1) * tank; ef[23] = tank;
      entry[24] = slot; entry[25] = pol ? L.FLAG.POLICY : 0; entry[26] = 0; entry[27] = 0;
      this.gpu.write(qbuf, entry, q * L.SPAWN_WORDS * 4);
      this.slotTrip[slot] = 2 * id; this.slotRoute[2 * slot] = off; this.slotRoute[2 * slot + 1] = r.length;
      if (q + 1 > this.queueHigh) this.queueHigh = q + 1;
      return true;
    };
    take();
    if (this.routePool) {
      if (this.prefetchPtr < this.tablePtr) this.prefetchPtr = this.tablePtr;
      while (this.prefetchPtr < T.n && T.sortedDepart[this.prefetchPtr] <= tEnd + this.routeAhead) { const id = T.order[this.prefetchPtr++]; this.routePool.prefetch(id, T.home[id], T.work[id]); }
      this.routePool.flush();
    }
  }

  _enqueueSpawns(tEnd) {
    if (this.table) return this._enqueueStream(tEnd);
    if (!this.trips) return;
    const pop = this.pop, A = pop.arrays, { flat, offset } = pop.routes, D = pop.deltas;
    const entry = new Uint32Array(L.SPAWN_WORDS), ef = new Float32Array(entry.buffer);
    const qbuf = this.gpu.buffers.get('queue');
    const pol = this.opts.features.policy;
    while (this.tripPtr < this.trips.length && this.tripDepart[this.tripPtr] <= tEnd + 2) {
      if (!this.free.length || !this.queueFree.length) break;
      const k = this.trips[this.tripPtr++], id = k >> 1;
      const slot = this.free.pop(), q = this.queueFree.pop();
      const roff = offset[k], rlen = offset[k + 1] - offset[k];
      const traits = D ? { aggression: D[id * 64], patience: D[id * 64 + 1], reactionTime: D[id * 64 + 2], preferredHeadway: D[id * 64 + 3], maxAccel: D[id * 64 + 4], maxDecel: D[id * 64 + 5] } : DEF_TRAITS;
      const type = A.vehicleType[id];
      this._par(entry, 0, type, traits, this.tripRouteBase + roff, rlen, A.workStart[id], this._routeLength(flat, roff, rlen), k);
      const tank = TANK_L[type] ?? 50;
      const fuel = this.fuelLeft[id] >= 0 ? this.fuelLeft[id] : (A.fuelFill ? A.fuelFill[id] : 1) * tank;
      ef[20] = this.tripDepart[this.tripPtr - 1]; entry[21] = 1; ef[22] = fuel; ef[23] = tank;
      entry[24] = slot; entry[25] = pol ? L.FLAG.POLICY : 0; entry[26] = 0; entry[27] = 0;
      this.gpu.write(qbuf, entry, q * L.SPAWN_WORDS * 4);
      this.slotTrip[slot] = k; this.slotRoute[2 * slot] = this.tripRouteBase + roff; this.slotRoute[2 * slot + 1] = rlen;
      if (q + 1 > this.queueHigh) this.queueHigh = q + 1;
    }
  }

  // ------------------------------------------------------------------ benchmark population
  /**
   * n vehicles all on the road at once (benchmarks / "max it out"). A pool of `pool` random-walk
   * routes (no U-turns, prefer straight-ish continuation, up to `routeEdges` edges) is shared:
   * each vehicle takes a random pool route and starts at rest at a random position along it,
   * centred in a lane, >= `spacing` m from any other car in that lane and >= spacing/2 from
   * edge ends. Road classes / spacing adapt to n (rank >= 3 roads first, all roads when
   * crowded). State is built and uploaded in chunks so millions of vehicles fit.
   */
  scatter(n, { seed = 7, routeEdges = 160, pool = Math.min(n, 100000), minRank = null, spacing = null, policy = this.opts.features.policy, chunk = 1 << 18 } = {}) {
    const g = this.graph, E = g.edges.length, cap = this.cap;
    n = Math.min(n, cap);
    let rs = (seed * 2654435761) >>> 0 || 1;
    const rnd = () => { rs ^= rs << 13; rs ^= rs >>> 17; rs ^= rs << 5; rs >>>= 0; return rs / 4294967296; };
    const gauss = () => { let u = 0; while (u === 0) u = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rnd()); };
    const laneM = (r) => { let m = 0; for (const e of g.edges) if (e.rank >= r) m += e.length * e.lanes; return m; };
    if (minRank == null) minRank = n * 28 < 0.35 * laneM(3) ? 3 : 0;
    if (spacing == null) spacing = Math.max(9, Math.min(28, 0.35 * laneM(minRank) / n));
    const cand = [];
    for (let e = 0; e < E; e++) if (g.edges[e].rank >= minRank && g.outStart[g.edges[e].to + 1] > g.outStart[g.edges[e].to]) cand.push(e);
    const cum = new Float64Array(cand.length); let acc = 0;
    for (let k = 0; k < cand.length; k++) { const e = g.edges[cand[k]]; acc += e.length * e.lanes; cum[k] = acc; }
    const pick = () => { const x = rnd() * acc; let lo = 0, hi = cand.length - 1; while (lo < hi) { const m = (lo + hi) >> 1; if (cum[m] < x) lo = m + 1; else hi = m; } return cand[lo]; };
    // route pool
    const poolOff = new Uint32Array(pool + 1), poolEdges = [], poolWords = [];
    let rOff = 0;
    for (let r = 0; r < pool; r++) {
      const route = [pick()]; let cur = route[0];
      for (let k = 1; k < routeEdges; k++) {
        const ec = g.edges[cur], a = g.outStart[ec.to], b = g.outStart[ec.to + 1];
        let best = -1, bestScore = -Infinity;
        for (let q = a; q < b; q++) {
          const eo = g.outEdges[q]; if (eo === ec.reverse) continue;
          let dh = g.edges[eo].headingStart - ec.headingEnd; while (dh > Math.PI) dh -= 2 * Math.PI; while (dh < -Math.PI) dh += 2 * Math.PI;
          const score = -Math.abs(dh) * 1.2 + g.edges[eo].rank * 0.15 + rnd() * 1.5;
          if (score > bestScore) { bestScore = score; best = eo; }
        }
        if (best < 0) break;
        route.push(best); cur = best;
      }
      const words = packRoutes(g, Int32Array.from(route), Uint32Array.of(0, route.length));
      poolEdges.push(Int32Array.from(route)); poolWords.push(words);
      poolOff[r] = rOff; rOff += words.length;
    }
    poolOff[pool] = rOff;
    const all = new Uint32Array(rOff); for (let r = 0; r < pool; r++) all.set(poolWords[r], poolOff[r]);
    this.routeWords = 0; this.routeMirror = null;
    this._ensureRoutes(all.length + this.opts.routeSlack);
    this._appendRoute(all);
    for (let r = 0; r < pool; r++) { const t = poolEdges[r]; let L = 0; for (const e of t) L += g.edges[e].length; poolEdges[r].total = L; }
    // placement + chunked upload
    const VW = L.VEH_WORDS, PW = L.PAR_WORDS;
    const typeCum = []; { let c = 0; for (const w of VEHICLE_TYPE_WEIGHTS) { c += w; typeCum.push(c); } }
    const used = new Map();
    const gb = this.gpu.buffers;
    let placed = 0, tries = 0, chunkStart = 0;
    let veh = new Uint32Array(chunk * VW), vf = new Float32Array(veh.buffer), vi = new Int32Array(veh.buffer);
    let par = new Uint32Array(chunk * PW), pose = new Float32Array(chunk * 4);
    const flush = (count) => {
      if (!count) return;
      this.gpu.write(gb.get('veh'), veh.subarray(0, count * VW), chunkStart * VW * 4);
      this.gpu.write(gb.get('par'), par.subarray(0, count * PW), chunkStart * PW * 4);
      this.gpu.write(gb.get('pose'), pose.subarray(0, count * 4), chunkStart * 16);
      chunkStart += count; veh.fill(0); par.fill(0);
    };
    while (placed < n && tries < n * 30) {
      tries++;
      const r = Math.floor(rnd() * pool), route = poolEdges[r];
      if (route.length < 2) continue;
      const k = Math.floor(rnd() * (route.length - 1));
      const e0 = route[k], ed = g.edges[e0];
      if (ed.length < spacing) continue;
      const lane = Math.floor(rnd() * ed.lanes);
      const s = spacing / 2 + rnd() * (ed.length - spacing);
      const key = e0 * 8 + lane;
      let list = used.get(key); if (!list) { list = []; used.set(key, list); }
      let clash = false; for (const x of list) if (Math.abs(x - s) < spacing) { clash = true; break; }
      if (clash) continue;
      list.push(s);
      const i = placed++, o = (i - chunkStart) * VW;
      const rr = rnd(); const type = typeCum.findIndex((c) => rr < c);
      const traits = { aggression: rnd(), patience: rnd(), reactionTime: Math.min(1.8, Math.max(0.6, 0.9 * Math.exp(0.27 * gauss()))), preferredHeadway: 1.0 + rnd() * 1.2, maxAccel: DEF_TRAITS.maxAccel, maxDecel: DEF_TRAITS.maxDecel };
      let before = 0; for (let q = 0; q < k; q++) before += g.edges[route[q]].length;
      this._par(par, (i - chunkStart) * PW, type, traits, poolOff[r], route.length, 0, route.total, 2 * i + 1);
      const lat = (lane + 0.5) * LANE_W, tank = TANK_L[type];
      veh[o] = e0; vf[o + 1] = s; vf[o + 4] = lat; veh[o + 7] = L.STATE.DRIVING;
      veh[o + 8] = k; veh[o + 10] = (policy ? L.FLAG.POLICY : 0) | L.FLAG.NEW; vi[o + 11] = this.step;
      vf[o + 12] = -1; veh[o + 14] = L.packLanes(lane, lane, lane, -1); vf[o + 15] = -1e9;
      vf[o + 16] = tank * (0.1 + 0.9 * rnd()); vf[o + 17] = tank; vf[o + 18] = this.time; vf[o + 19] = before + s;
      vi[o + 20] = -1; vf[o + 21] = 1e9; vi[o + 23] = -1; vf[o + 28] = route.total; vf[o + 29] = this.time; vf[o + 31] = 1e9;
      const ps = poseAt(g, e0, s, lat); pose.set(ps, (i - chunkStart) * 4);
      vf[o + 36] = ps[0]; vf[o + 37] = ps[2]; vf[o + 38] = ps[3];   // start pose = pose (no sweep on step 1)
      this.slotTrip[i] = 2 * i + 1;
      if (placed - chunkStart === chunk) flush(chunk);
    }
    flush(placed - chunkStart);
    // free slots: hidden pose, zero state; ring cleared on the GPU
    const enc = this.device.createCommandEncoder();
    enc.clearBuffer(gb.get('ring'));
    if (placed < cap) {
      enc.clearBuffer(gb.get('veh'), placed * VW * 4, (cap - placed) * VW * 4);
      const hidden = new Float32Array((cap - placed) * 4); for (let j = 1; j < hidden.length; j += 4) hidden[j] = -1000;
      this.gpu.write(gb.get('pose'), hidden, placed * 16);
    }
    this.device.queue.submit([enc.finish()]);
    const io = new Uint32Array(L.IO.EVDATA); io[0] = this.step;
    this.gpu.write(gb.get('io'), io);
    this.free = []; for (let i = cap - 1; i >= placed; i--) this.free.push(i);
    this._newPose = true;
    return { placed, tries, routeWords: all.length, minRank, spacing: +spacing.toFixed(1), pool };
  }

  // ------------------------------------------------------------------ CPU snapshot import (equivalence)
  /**
   * Load a CPU snapshot (src/sim/state.js exportState, layout v3) slot for slot, incl. the
   * reaction-time rings, and take over its clock. commuters: sim.commuters (workStart).
   * Station lots and scenes are not copied (equivalence runs disable crashes / fuel stops).
   */
  importSnapshot(snap, commuters = null) {
    if (snap.version !== 3) console.warn(`GPUSim.importSnapshot: layout ${snap.version}, mirror written for 3`);
    const A = snap.arrays, g = this.graph, cap = this.cap, high = snap.meta.high;
    if (high > cap) throw new Error(`capacity ${cap} < cpu high water ${high}`);
    const VW = L.VEH_WORDS, PW = L.PAR_WORDS;
    const veh = new Uint32Array(cap * VW), vf = new Float32Array(veh.buffer), vi = new Int32Array(veh.buffer);
    const par = new Uint32Array(cap * PW), pf = new Float32Array(par.buffer);
    const routeList = []; let rOff = 0;
    for (let i = 0; i < high; i++) {
      if (A.state[i] === 0) continue;
      const ro = A.routeOff[i], rl = A.routeLen[i];
      const route = ro >= 0 ? snap.routes.flat.subarray(ro, ro + rl) : Int32Array.of(A.edge[i]);
      const words = packRoutes(g, Int32Array.from(route), Uint32Array.of(0, route.length));
      routeList.push(words);
      const c = commuters && A.commuter[i] >= 0 ? commuters[A.commuter[i]] : null;
      const o = i * VW, p = i * PW;
      veh[o] = A.edge[i]; vf[o + 1] = A.s[i]; vf[o + 2] = A.speed[i]; vf[o + 3] = A.accel[i];
      vf[o + 4] = A.lat[i]; vf[o + 5] = A.headingRel[i]; vf[o + 6] = A.steerAngle[i]; veh[o + 7] = A.state[i];
      veh[o + 8] = A.routePos[i]; veh[o + 9] = A.seg[i]; veh[o + 10] = A.flags[i] & ~L.FLAG.BRAIN & ~32; vi[o + 11] = A.cmdStep[i];
      vf[o + 12] = A.stoppedSince[i]; vf[o + 13] = A.odometer[i]; veh[o + 14] = A.laneHi[i] < A.laneLo[i] ? L.packLanes(A.lane[i], A.lane[i] + 1, A.lane[i], A.targetLane[i]) : L.packLanes(A.lane[i], A.laneLo[i], A.laneHi[i], A.targetLane[i]);   // off every lane list (layout v6) vf[o + 15] = A.lastDecision[i];
      vf[o + 16] = A.fuel[i]; vf[o + 17] = A.tank[i]; vf[o + 18] = A.spawnTime[i]; vf[o + 19] = A.routeDist[i];
      vi[o + 20] = A.leader[i]; vf[o + 21] = A.gap[i]; vf[o + 22] = A.pumpUntil[i]; vi[o + 23] = A.stopPos[i];
      veh[o + 24] = A.refuelState[i]; vf[o + 25] = A.steerSl ? A.steerSl[i] : A.steerAngle[i]; vf[o + 26] = A.gas[i]; vf[o + 27] = A.brake[i];
      vf[o + 28] = A.routeTotal[i]; vf[o + 29] = A.lastFuelCheck[i]; vf[o + 30] = A.taper[i]; vf[o + 31] = A.laneEndDist[i]; vi[o + 32] = A.laneEndSide[i];
      if (A.vLat) { vf[o + 33] = A.vLat[i]; vf[o + 34] = A.yawRate[i]; vf[o + 35] = A.axPrev[i]; vf[o + 36] = A.px[i]; vf[o + 37] = A.pz[i]; vf[o + 38] = A.pyaw[i]; vf[o + 39] = A.power[i]; }
      if (A.flags[i] & 32) veh[o + 10] |= L.FLAG.DISABLED;   // LAYOUT v4 flag bit5 DISABLED -> GPU flag
      par[p] = rOff; par[p + 1] = route.length; par[p + 2] = A.typeIdx[i]; par[p + 3] = A.delaySteps[i];
      pf[p + 4] = A.length[i]; pf[p + 5] = A.width[i]; pf[p + 6] = A.maxSpeed[i]; pf[p + 7] = A.maxAccel[i];
      pf[p + 8] = A.maxDecel[i]; pf[p + 9] = A.headwayPref[i]; pf[p + 10] = A.aggression[i]; pf[p + 11] = A.patience[i];
      pf[p + 12] = A.reactionTime[i]; pf[p + 13] = c ? c.workStart : 0; pf[p + 14] = A.routeTotal[i]; par[p + 15] = (Math.max(0, A.commuter[i]) * 2 + A.leg[i]) >>> 0;
      pf[p + 16] = A.wheelbase[i]; pf[p + 17] = A.maxSteer[i]; pf[p + 18] = A.maxSteerRate[i]; pf[p + 19] = (VEHICLE_TYPES[A.typeIdx[i]] || VEHICLE_TYPES[0]).maxSpeed;
      rOff += words.length;
      this.slotTrip[i] = A.commuter[i] * 2 + A.leg[i];
    }
    const all = new Uint32Array(Math.max(1, rOff)); { let q = 0; for (const w of routeList) { all.set(w, q); q += w.length; } }
    this.routeWords = 0; this.routeMirror = null; this._ensureRoutes(all.length + 1024); this._appendRoute(all);
    const ring = new Float32Array(cap * L.CTRL_RING * L.CTRL_STRIDE);
    ring.set(A.ctrlRing.subarray(0, Math.min(A.ctrlRing.length, ring.length)));
    this.time = snap.meta.time; this.step = snap.meta.stepCount; this.dt = snap.meta.dt;
    this.gpu.write(this.gpu.buffers.get('veh'), veh);
    this.gpu.write(this.gpu.buffers.get('par'), par);
    this.gpu.write(this.gpu.buffers.get('ring'), ring);
    const pose = new Float32Array(cap * 4);
    for (let i = 0; i < cap; i++) {
      if (i >= high || A.state[i] === 0) { pose[i * 4 + 1] = -1000; continue; }
      pose[i * 4] = A.x[i]; pose[i * 4 + 1] = A.y[i]; pose[i * 4 + 2] = A.z[i]; pose[i * 4 + 3] = A.heading[i];
    }
    this.gpu.write(this.gpu.buffers.get('pose'), pose);
    const io = new Uint32Array(L.IO.EVDATA); io[0] = this.step;
    this.gpu.write(this.gpu.buffers.get('io'), io);
    this.free = []; for (let i = cap - 1; i >= 0; i--) if (i >= high || A.state[i] === 0) this.free.push(i);
  }

  async readState() {
    await this.device.queue.onSubmittedWorkDone();
    const [v, p, ps] = await Promise.all([this.gpu.read(this.gpu.buffers.get('veh')), this.gpu.read(this.gpu.buffers.get('par')), this.gpu.read(this.gpu.buffers.get('pose'))]);
    return { veh: L.recordView(v, L.VEH_FIELDS), par: L.recordView(p, L.PAR_FIELDS), pose: new Float32Array(ps) };
  }

  /**
   * Float32Array(cap*4) x, y, z, heading per slot (y = -1000: free slot) for a Three.js instanced mesh.
   * heading: yaw in the x-z plane, forward = (cos h, sin h). Safe to pipeline (call again while one is
   * pending): uses a small pool of reusable staging buffers and output arrays (a returned array is
   * reused only after `keep` = 3 later reads, so hold at most two). No per-frame GPU allocation.
   */
  async readPoses() {
    const src = this.gpu.buffers.get('pose'), size = src.size;
    const pool = this._posePool || (this._posePool = { free: [], outs: [], k: 0 });
    let rb = pool.free.pop();
    if (!rb || rb.size !== size) rb = this.device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(src, 0, rb, 0, size);
    this.device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const k = pool.k++ % 3;
    if (!pool.outs[k] || pool.outs[k].length !== size / 4) pool.outs[k] = new Float32Array(size / 4);
    pool.outs[k].set(new Float32Array(rb.getMappedRange()));
    rb.unmap();
    if (pool.free.length < 3) pool.free.push(rb); else rb.destroy();
    return pool.outs[k];
  }

  // ------------------------------------------------------------------ scenes (crash / stranded)
  _classify(infos, impact, lanes) {
    const mph = impact / 0.44704;
    const MASS = simCrash.MASS_BY_TYPE || [1500, 2100, 15000, 13000];
    let heavy = false, mass = 0;
    for (const inf of infos) { const m = MASS[inf.type] || 1500; mass += m; if (m > 5000) heavy = true; }
    const nV = Math.max(1, infos.length);
    const eff = mph * Math.sqrt(mass / (1500 * nV));
    const TH = simCrash.SEVERITY_THRESHOLDS || { minorMph: 15, moderateMph: 40, pFatalSevere: 0.08, pFatalModerate: 0.01 };
    let severity = eff > TH.moderateMph || heavy || nV >= 3 ? 'severe' : eff > TH.minorMph ? 'moderate' : 'minor';
    if (severity === 'severe' && this._rand() < TH.pFatalSevere) severity = 'fatal';
    else if (severity === 'moderate' && this._rand() < TH.pFatalModerate) severity = 'fatal';
    const row = (simCrash.CRASH_TABLE || CRASH_TABLE_FALLBACK)[severity] || CRASH_TABLE_FALLBACK[severity];
    let lanesBlocked = Math.min(lanes, Math.max(row.lanes === 99 ? lanes : row.lanes, nV));
    if (severity === 'minor') lanesBlocked = Math.min(lanes, nV);
    const blockDuration = row.blockMedian * Math.exp(row.sigma * this._gauss());
    const clearDuration = Math.max(blockDuration, row.clearMedian * Math.exp(row.sigma * this._gauss()));
    return { severity, impactSpeed: impact, impactMph: mph, lanesBlocked, blockDuration, clearDuration };
  }

  _openScene(fresh, cls, isCrash, all = fresh) {
    const g = this.graph, first = this.wrecks.get(fresh[0]);
    if (!first) return null;
    const e = g.edges[first.edge];
    let laneMask = 0, sMax = 0;
    for (const s of all) { const inf = this.wrecks.get(s); if (inf && inf.edge === first.edge) { for (let l = inf.lo; l <= inf.hi; l++) laneMask |= 1 << l; if (inf.s > sMax) sMax = inf.s; } }
    let blocked = 0; for (let l = 0; l < e.lanes; l++) if ((laneMask >> l) & 1) blocked++;
    for (let l = 0; l < e.lanes && blocked < cls.lanesBlocked; l++) if (!((laneMask >> l) & 1)) { laneMask |= 1 << l; blocked++; }
    const p = poseAt(g, first.edge, first.s, (first.lo + 0.5) * LANE_W);
    const rec = { id: this.crashes.length, time: this.time, cause: cls.cause, severity: cls.severity, impactSpeed: cls.impactSpeed, impactMph: cls.impactMph,
      vehicles: fresh.slice(), lanesBlocked: blocked, laneMask, edge: first.edge, s: sMax, x: p[0], y: p[1], z: p[2],
      blockStart: this.time, blockEnd: this.time + cls.blockDuration, clearEnd: this.time + cls.clearDuration,
      squeezeAt: this.time + Math.min(cls.blockDuration, (blocked >= e.lanes || all.some((q) => this.wrecks.get(q) && this.wrecks.get(q).edge !== first.edge) ? 1 : 2) * this.opts.squeezeDelay),
      squeeze: false, responseState: 'active', isCrash, block: -1, secondary: false, stranded: !isCrash };
    this.crashes.push(rec); this.activeCrashes.push(rec);
    this._blockInsert(rec);
    return rec;
  }

  _crash(slotA, slotB, impact, cause) {
    const slots = [slotA]; if (slotB !== 0xFFFFFFFF && slotB !== undefined) slots.push(slotB);
    const fresh = slots.filter((s) => this.wrecks.has(s) && !this._inCrash.has(s));
    if (!fresh.length) return;
    for (const s of fresh) this._inCrash.add(s);
    const infos = slots.map((s) => this.wrecks.get(s)).filter(Boolean);
    const cls = this._classify(infos, impact, this.graph.edges[this.wrecks.get(fresh[0]).edge].lanes);
    cls.cause = cause;
    const rec = this._openScene(fresh, cls, true, slots);
    if (!rec) return;
    rec.secondary = slots.length > fresh.length;
    this.counters.crashes++; this.counters.crashedVehicles += fresh.length; if (rec.secondary) this.counters.secondaryCrashes++;
    for (const s of fresh) this._markOutcome(s, 2);
    this._event('crash', fresh[0], { crash: rec.id, severity: rec.severity, impactMph: rec.impactMph, lanesBlocked: rec.lanesBlocked, cause });
  }

  _bySev(kmh) { const sv = (simCrash.severityOf || ((d) => (d < 10 ? 'minor' : d < 25 ? 'moderate' : d < 50 ? 'severe' : 'very severe')))(kmh); this.counters.bySeverity[sv] = (this.counters.bySeverity[sv] || 0) + 1; return sv; }
  /**
   * Layout v4 crash event (sim.js _crashEvent): severity from the larger per-vehicle delta-v;
   * the cars the GPU disabled (delta-v >= 10 km/h) get a scene with clearanceFor(severity).
   */
  _pull(slot) { this.counters.pullOvers = (this.counters.pullOvers || 0) + 1; const until = this.time + 300 + 600 * this._rand(); this.op(L.OP.SET_PULL, slot, new Uint32Array(Float32Array.of(until).buffer)[0]); }
  _crashV4(slotA, slotB, dvA, dvB, cx = null) {
    const slots = [slotA]; if (slotB !== 0xFFFFFFFF && slotB !== undefined) slots.push(slotB);
    const severity = this._bySev(Math.max(dvA, dvB));
    this.counters.crashEvents++;
    const fresh = slots.filter((s) => this.wrecks.has(s) && !this._inCrash.has(s));
    if (!fresh.length) return;
    for (const s of fresh) this._inCrash.add(s);
    const rng = { gaussian: () => this._gauss() };
    // layout v6 incident timeline (crashModel.js incidentTimeline): notify, responders, lanes blocked until
    // the wrecks are moved to the shoulder (block), scene cleared (clear)
    const tl = simCrash.incidentTimeline ? simCrash.incidentTimeline(severity, rng) : null;
    const clear = tl ? tl.clear : simCrash.clearanceFor ? simCrash.clearanceFor(severity, rng) : 1800, block = tl ? tl.block : clear;
    const rec = this._openScene(fresh, { cause: 'collision', severity, impactSpeed: Math.max(dvA, dvB) / 3.6, impactMph: Math.max(dvA, dvB) / 1.609, lanesBlocked: 0, blockDuration: block, clearDuration: clear }, true, slots);
    if (!rec) return;
    if (tl) { rec.notifyAt = this.time + tl.notify; rec.respondersAt = this.time + tl.arrive; }
    rec.deltaVKmh = [dvA, dvB].slice(0, slots.length); rec.secondary = slots.some((s) => this._inCrash.has(s) && !fresh.includes(s));
    rec.wall = slots.length === 1; rec.kind = rec.wall ? 'barrier' : 'vehicle';
    if (cx) {   // GPU crash detail (EV_CRASHX_*): contact point, pre-impact speeds (m/s), headings (rad), closing speed, lanes
      rec.x = cx.x; rec.z = cx.z; rec.closingKmh = cx.closing * 3.6; rec.impactMph = cx.closing / 0.44704;
      // sim.js crash log fields: kind, preSpeed (m/s), heading (rad), normal, contact, closingSpeed
      rec.kind = rec.wall ? 'barrier' : 'vehicle'; rec.preSpeed = [cx.preSpeedA, cx.preSpeedB].slice(0, slots.length); rec.heading = [cx.headingA, cx.headingB].slice(0, slots.length);
      rec.normal = { nx: cx.nx, nz: cx.nz }; rec.contact = { x: cx.x, z: cx.z }; rec.closingSpeed = cx.closing; rec.involved = slots.slice();
      // sim.js _classifyCrash: type from headings + normal in the first car's frame, maneuver of the striking car, road class
      let type = 'barrier';
      if (slots.length > 1) {
        let d = Math.abs(cx.headingA - cx.headingB) % (2 * Math.PI); if (d > Math.PI) d = 2 * Math.PI - d;
        const along = Math.abs(Math.cos(cx.headingA) * (cx.nx ?? 0) + Math.sin(cx.headingA) * (cx.nz ?? 0));
        type = d > 2.6 ? 'head-on' : d > 0.52 ? 'angle' : along > 0.7 ? 'rear-end' : 'sideswipe';
        if (cx.anyCrashed) type += '-wreck';
      }
      rec.type = type; rec.maneuver = cx.maneuver || 'cruise'; rec.roadClass = cx.strikeEdge != null ? this.graph.edges[cx.strikeEdge]?.highway : this.graph.edges[rec.edge]?.highway;
      rec.vehicleDetail = slots.map((s, k) => ({ slot: s, edge: this.wrecks.get(s)?.edge ?? null, lane: k ? cx.laneB : cx.laneA, heading: k ? cx.headingB : cx.headingA,
        preSpeedKmh: (k ? cx.preSpeedB : cx.preSpeedA) * 3.6, deltaVKmh: k ? dvB : dvA, disabled: fresh.includes(s), type: this.wrecks.get(s)?.type ?? null }));
    }
    this.counters.crashes++; this.counters.crashedVehicles += fresh.length; if (rec.secondary) this.counters.secondaryCrashes++;
    for (const s of fresh) this._markOutcome(s, 2);
    this._event('crash', fresh[0], { crash: rec.id, severity, cause: rec.wall ? 'barrier' : 'collision', kind: rec.kind, type: rec.type, maneuver: rec.maneuver, roadClass: rec.roadClass, preSpeed: rec.preSpeed, heading: rec.heading, normal: rec.normal, contact: rec.contact, closingSpeed: rec.closingSpeed, deltaVKmh: rec.deltaVKmh, closingKmh: rec.closingKmh, x: rec.x, z: rec.z, edge: rec.edge, vehicles: rec.vehicleDetail });
  }

  _strand(slot) {
    this._inCrash.add(slot);
    this.counters.stranded++;
    const fc = this.opts.fuelCfg, tow = fc.towMedian * Math.exp(fc.towSigma * this._gauss());
    this._openScene([slot], { cause: 'stranded', severity: 'stranded', impactSpeed: 0, impactMph: 0, lanesBlocked: 1, blockDuration: tow, clearDuration: tow }, false);
    this._markOutcome(slot, 3);
    this._event('stranded', slot, {});
  }

  _blockWrite(b, rec, state) {
    const w = new Uint32Array(L.BLOCK_WORDS), f = new Float32Array(w.buffer);
    w[0] = rec.edge; w[1] = rec.laneMask; f[2] = rec.s; w[3] = state; f[4] = rec.time; w[5] = this.blockNext[b]; w[6] = rec.vehicles[0];
    w[7] = (rec.squeeze ? 1 : 0) | (rec.isCrash ? 2 : 0);
    this.gpu.write(this.gpu.buffers.get('world'), w, (this.layoutOff.blkOff + b * L.BLOCK_WORDS) * 4);
  }
  _headWrite(edge) { this.gpu.write(this.gpu.buffers.get('world'), Uint32Array.of(this.edgeHead[edge]), (this.layoutOff.ebOff + edge) * 4); }
  _blockInsert(rec) {
    if (!this.blockFree.length) return;
    const b = this.blockFree.pop();
    rec.block = b;
    this.blockNext[b] = this.edgeHead[rec.edge];
    this.edgeHead[rec.edge] = b + 1;
    this._blockWrite(b, rec, 1);
    this._headWrite(rec.edge);
  }
  _blockRemove(rec) {
    const b = rec.block; if (b < 0) return;
    let prev = -1, cur = this.edgeHead[rec.edge];
    while (cur && cur - 1 !== b) { prev = cur - 1; cur = this.blockNext[cur - 1]; }
    if (cur) {
      if (prev < 0) { this.edgeHead[rec.edge] = this.blockNext[b]; this._headWrite(rec.edge); }
      else { this.blockNext[prev] = this.blockNext[b]; const pr = this.activeCrashes.find((r) => r.block === prev); if (pr) this._blockWrite(prev, pr, pr.responseState === 'shoulder' ? 2 : 1); }
    }
    this._blockWrite(b, rec, 0);
    this.blockFree.push(b); rec.block = -1;
  }
  /** Scene timeline (sim.js _scenes): squeeze past full closures, shoulder, cleared / towed. */
  /**
   * sim.js _moveOffLanes (layout v8): each wreck goes onto the paved shoulder (off every lane list, >= 15 m
   * from either node); where the shoulder is narrower than the car (+0.4 m) or the edge is short, the tow
   * truck hauls it away (slot released now, the scene keeps its record until clearEnd).
   */
  _moveOffLanes(r) {
    const g = this.graph, SHOULDER = new Set(['motorway', 'trunk', 'motorway_link', 'trunk_link']);
    for (const s of r.vehicles) {
      if ((r.gone && r.gone.includes(s)) || !this.wrecks.has(s)) continue;
      const w = this.wrecks.get(s), ee = g.edges[w.edge], t = VEHICLE_TYPES[w.type] || VEHICLE_TYPES[0];
      const sh = SHOULDER.has(ee.highway) ? this.opts.freewayShoulder : this.opts.shoulder;
      if (sh < t.width + 0.4 || ee.length < 30) {
        (r.gone || (r.gone = [])).push(s);
        this.counters.hauledAway = (this.counters.hauledAway || 0) + 1;
        this.op(L.OP.FREE, s); this._inCrash.delete(s); this.wrecks.delete(s);
        if (r.stranded) { this.counters.towed++; this._release(s, 'towed'); } else { this.counters.cleared++; this._release(s, 'cleared'); }
        continue;
      }
      this.op(L.OP.TOW_SHOULDER, s); this._newPose = true;
      this.counters.towedToShoulder = (this.counters.towedToShoulder || 0) + 1;
    }
  }
  _scenes() {
    if (!this.activeCrashes.length) return;
    const keep = [];
    for (const r of this.activeCrashes) {
      if (r.clearEnd - r.time >= 1e8 && r.blockEnd - r.time >= 1e8) { keep.push(r); continue; }   // layout v8: permanent scenes (training obstacles) are never squeezed or towed
      if (r.responseState === 'active' && !r.squeeze && this.time >= r.squeezeAt && this.time < r.blockEnd) {
        r.squeeze = true;   // full closure: responders push the wrecks aside, trapped traffic is waved past
        this._moveOffLanes(r);
        r.laneMask = 0;
        if (r.block >= 0) this._blockWrite(r.block, r, 1);
        this._event('squeeze', r.vehicles[0], { crash: r.id });
      }
      if (r.responseState === 'active' && this.time >= r.blockEnd && this.time < r.clearEnd) {
        r.responseState = 'shoulder';
        this._moveOffLanes(r);   // responders / tow trucks move the wrecks off the travel lanes (or haul them away)
        if (r.block >= 0) this._blockWrite(r.block, r, 2);
        this._event('shoulder', r.vehicles[0], { crash: r.id });
      }
      if (this.time >= r.clearEnd) {
        r.responseState = 'cleared';
        for (const s of r.vehicles) {
          if (r.gone && r.gone.includes(s)) continue;
          this.op(L.OP.FREE, s); this._inCrash.delete(s); this.wrecks.delete(s);
          if (r.stranded) { this.counters.towed++; this._release(s, 'towed'); } else { this.counters.cleared++; this._release(s, 'cleared'); }
        }
        this._blockRemove(r);
        this._event('cleared', r.vehicles[0], { crash: r.id, severity: r.severity });
      } else keep.push(r);
    }
    this.activeCrashes = keep;
  }

  // ------------------------------------------------------------------ fuel: plans, lots, pumps
  _routeOf(slot) {
    const off = this.slotRoute[2 * slot], len = this.slotRoute[2 * slot + 1];
    return this.routeMirror ? this.routeMirror.subarray(off, off + len) : null;
  }
  /** Low-fuel requests -> StationNetwork.plan (src/sim/fuel.js) within refuelBudgetMs per frame. */
  /** A worker finished a request: apply results and refill the workers (refuel plans and reroutes take turns). */
  _pumpSoon() {
    const pool = this._fuelPool; if (!pool || pool.pumpQ) return;
    pool.pumpQ = true;
    queueMicrotask(() => { pool.pumpQ = false; pool.flip = !pool.flip; if (pool.flip) { this._processRefuels(); if (this.jPack) this._processReroutes(); } else { if (this.jPack) this._processReroutes(); this._processRefuels(); } });
  }
  /** Fuel / route worker with the fewest in-flight requests if below maxq, else -1. */
  _fuelWorker(maxq) {
    const pool = this._fuelPool; let best = -1;
    for (let k = 0; k < pool.ws.length; k++) if (pool.out[k] < maxq && (best < 0 || pool.out[k] < pool.out[best])) best = k;
    return best;
  }
  _processRefuels() {
    const pool = this._fuelPool;
    if (pool && !pool.failed && pool.done.length) { const done = pool.done; pool.done = []; for (const r of done) this._applyRefuel(r.slot, r.trip, r.edge, r.plan); }
    if (!this.refuelQueue.length) return;
    if (!this.stations) { for (const r of this.refuelQueue) this.op(L.OP.FUEL_PLAN_FAILED, r.slot); this.refuelQueue.length = 0; return; }
    const g = this.graph, t0 = performance.now();
    while (this.refuelQueue.length && performance.now() - t0 < this.opts.refuelBudgetMs) {
      const { slot, edge } = this.refuelQueue.shift();
      if (this.slotTrip[slot] < 0) continue;
      const route = this._routeOf(slot);
      if (!route || !route.length) { this.op(L.OP.FUEL_PLAN_FAILED, slot); continue; }
      const goal = g.edges[route[route.length - 1] & EDGE_ID_MASK].to;
      const from = g.edges[edge].to;
      if (from === goal) { this.op(L.OP.FUEL_PLAN_FAILED, slot); continue; }
      if (pool && !pool.failed) {
        const w = pool.ready < pool.ws.length ? -1 : this._fuelWorker(FUEL_MAXQ);
        if (w < 0) { this.refuelQueue.unshift({ slot, edge }); break; }   // workers loading / busy: next frame
        const id = ++pool.seq; pool.pending.set(id, { slot, edge, trip: this.slotTrip[slot], w }); pool.out[w]++;
        pool.ws[w].postMessage({ id, from, goal });
        continue;
      }
      this._applyRefuel(slot, this.slotTrip[slot], edge, this.stations.plan(from, goal), true);
    }
  }
  /** Apply a refuel plan (worker or main thread) to slot if it still carries the same trip. */
  _applyRefuel(slot, trip, edge, plan, local = false) {
    const g = this.graph;
    if (this.slotTrip[slot] !== trip) return;
    {
      if (plan && !local) plan = { ...plan, station: this.stations.list[plan.station], edges: Int32Array.from(plan.edges) };
      if (!plan) { this.counters.refuelPlanFailed++; this.op(L.OP.FUEL_PLAN_FAILED, slot); return; }
      const edges = new Int32Array(plan.edges.length + 1); edges[0] = edge; edges.set(plan.edges, 1);
      let total = 0; for (const e of edges) total += g.edges[e].length;
      const words = packRoutes(g, edges, Uint32Array.of(0, edges.length));
      const block = new Uint32Array(words.length + 2); block[0] = plan.stopIndex + 1; block[1] = f32bits(total); block.set(words, 2);
      const off = this._appendRoute(block) + 2;
      this.op(L.OP.SET_ROUTE_REFUEL, slot, off, edges.length);
      this.slotRoute[2 * slot] = off; this.slotRoute[2 * slot + 1] = edges.length;
      this.slotStation[slot] = plan.station.idx;
      this.counters.refuelPlanned++;
      this._event('refuelPlanned', slot, { station: plan.station.idx, detourTime: plan.detourTime });
    }
  }
  /** sim.js _trafficSample (every 60 s): per-edge mean speed -> EMA -> trafficPenalty (extra seconds over free flow). */
  async _trafficSample() {
    this._trafficBusy = true;
    try {
      const E = this.world.E, enc = this.device.createCommandEncoder(), p = enc.beginComputePass();
      this._dispatch(p, this.K.trafClear, this.BG.trafClear, Math.ceil(E / 256)); this._dispatch(p, this.K.trafCount, this.BG.trafCount, Math.ceil(this.cap / 256));
      p.end(); this.device.queue.submit([enc.finish()]);
      const [cb, sb] = await Promise.all([this.gpu.read(this.gpu.buffers.get('edgeCount'), E * 4), this.gpu.read(this.gpu.buffers.get('edgeSpeed'), E * 4)]);
      const cnt = new Uint32Array(cb), sum = new Uint32Array(sb), g = this.graph;
      const ema = this._edgeSpeed || (this._edgeSpeed = new Map());
      for (const [e, v] of ema) if (!cnt[e]) { const lim = g.edges[e].speedLimit, nv = 0.5 * v + 0.5 * lim; if (nv > 0.95 * lim) ema.delete(e); else ema.set(e, nv); }
      for (let e = 0; e < E; e++) if (cnt[e]) { const m = sum[e] / 100 / cnt[e], old = ema.get(e); ema.set(e, old === undefined ? m : 0.5 * old + 0.5 * m); }
      const pen = this.trafficPenalty || (this.trafficPenalty = new Map()); pen.clear();
      for (const [e, v] of ema) { const Ed = g.edges[e]; if (v >= 0.7 * Ed.speedLimit) continue; pen.set(e, Ed.length / Math.max(0.5, v) - Ed.length / Ed.speedLimit); }
      const pool = this._fuelPool;
      if (pool && !pool.failed) { const keys = Int32Array.from(pen.keys()), vals = Float32Array.from(pen.values()); for (const w of pool.ws) w.postMessage({ penalty: keys, values: vals }); }
    } finally { this._trafficBusy = false; }
  }
  /** sim.js _rerouteBlocked: edges every lane of which an active (not squeezed) scene blocks -> GPU bitmask. */
  _updateClosed() {
    const g = this.graph, closed = [];
    for (const r of this.activeCrashes) {
      if (r.responseState !== 'active' || r.squeeze) continue;
      const e = g.edges[r.edge]; let all = true;
      for (let l = 0; l < e.lanes; l++) if (!((r.laneMask >> l) & 1)) { all = false; break; }
      if (all) closed.push(r.edge);
    }
    closed.sort((a, b) => a - b);
    const key = closed.join(',');
    if (key === this._closedKey) return;
    this._closedKey = key; this._closedN = closed.length; this.closedEdges = closed;
    const E = this.world.E, bits = new Uint32Array(Math.ceil(E / 32));
    for (const e of closed) bits[e >> 5] |= 1 << (e & 31);
    this.gpu.write(this.gpu.buffers.get('ctl'), bits, (this.cap * CTL_WORDS + this.jPack.M + E) * 4);
  }
  /** Re-plan cars that see a closure on their route (once per 120 s per trip; Web-Worker A* when available). */
  _processReroutes() {
    const g = this.graph, pool = this._fuelPool;
    if (pool && pool.rdone && pool.rdone.length) { const done = pool.rdone; pool.rdone = []; for (const r of done) this._applyReroute(r.slot, r.trip, r.edge, r.route, r.missed, r.slow, r.rp, r.prefix ?? -1); }
    let q = this.rerouteQueue; if (!q || !q.length) return;
    if (q.length > 4000) q = this.rerouteQueue = q.filter((r) => this.slotTrip[r.slot] === r.trip && (r.missed || this.time - r.t <= 60));
    const last = this._lastReroute || (this._lastReroute = new Map());
    let budget = 200;
    if (this.opts.debugReroute) { const c = { stale: 0, trip: 0, recent: 0 }; for (const r of q) { if (this.slotTrip[r.slot] !== r.trip) c.trip++; else if (!r.missed && this.time - (r.t ?? this.time) > 60) c.stale++; } console.log(`[reroute] ${this.clock} q ${q.length} ${JSON.stringify(c)} ready ${pool && pool.ready}/${pool && pool.ws.length} out ${pool && pool.out} failed ${pool && pool.failed}`); }
    while (q.length && budget > 0) {
      const { slot, edge, trip, missed, slow, rp, prefix = -1, t = this.time } = q.shift();
      if (this.slotTrip[slot] !== trip || trip < 0) continue;
      if (!missed && this.time - t > 60) continue;   // stale (the scan asks again while the closure / jam is still there)
      const lt = last.get(slot);
      if (!missed && lt && lt.trip === trip && this.time - lt.t < (slow ? 300 : 120)) continue;
      if (!missed) last.set(slot, { t: this.time, trip });
      budget--;
      const route = this._routeOf(slot); if (!route || !route.length) continue;
      const goal = g.edges[route[route.length - 1] & EDGE_ID_MASK].to, from = g.edges[edge].to;
      if (from === goal) { if (missed) this._applyReroute(slot, trip, edge, new Int32Array(0), true, false, 0, prefix); continue; }
      if (pool && !pool.failed && pool.ready === pool.ws.length) {
        const w = this._fuelWorker(FUEL_MAXQ);
        if (w < 0) { q.unshift({ slot, edge, trip, missed, slow, rp, prefix, t: this.time }); if (!missed) last.delete(slot); break; }
        const id = ++pool.seq; pool.pending.set(id, { slot, edge, trip, missed, slow, rp, prefix, w }); pool.out[w]++;
        pool.ws[w].postMessage({ id, from, goal, avoid: missed ? [] : this.closedEdges || [], usePenalty: !missed });
      } else {
        this.router ||= makeRouter(g, { weight: 1.25 });
        let r = null; try { r = missed ? this.router(from, goal) : this.router(from, goal, this.closedEdges && this.closedEdges.length ? new Set(this.closedEdges) : null, this.trafficPenalty && this.trafficPenalty.size ? this.trafficPenalty : null); } catch (e) { r = null; }
        this._applyReroute(slot, trip, edge, r, missed, slow, rp, prefix);
      }
    }
    if (last.size > 300000) last.clear();
  }
  _applyReroute(slot, trip, edge, path, missed = false, slow = false, rp = 0, prefix = -1) {
    if (this.slotTrip[slot] !== trip) return;
    if (slow && path) {   // only switch when the new way is clearly faster with the current traffic (> 60 s)
      const g0 = this.graph, pen = this.trafficPenalty || new Map(), cost = (p) => { let t = 0; for (const x of p) { const Ed = g0.edges[x & EDGE_ID_MASK]; t += Ed.length / Ed.speedLimit + (pen.get(x & EDGE_ID_MASK) || 0); } return t; };
      const cur = this._routeOf(slot); if (!cur || cost(path) > cost(Array.from(cur.subarray(rp + 1))) - 60) return;
    }
    const closed = new Set(missed ? [] : this.closedEdges || []);
    if (!path || (!path.length && !missed) || Array.from(path).some((x) => closed.has(x))) { this.counters.rerouteFailed = (this.counters.rerouteFailed || 0) + 1; return; }
    const g = this.graph, pre = prefix >= 0 ? 1 : 0, edges = new Int32Array(pre + 1 + path.length);
    if (pre) edges[0] = prefix;   // still on the junction connector: the route starts with the in-edge
    edges[pre] = edge; edges.set(path, pre + 1);
    let total = 0; for (const e of edges) total += g.edges[e].length;
    const words = packRoutes(g, edges, Uint32Array.of(0, edges.length));
    const block = new Uint32Array(words.length + 1); block[0] = f32bits(total); block.set(words, 1);
    const off = this._appendRoute(block) + 1;
    this.op(L.OP.SET_ROUTE, slot, off, edges.length);
    this.slotRoute[2 * slot] = off; this.slotRoute[2 * slot + 1] = edges.length;
    this.counters.reroutes = (this.counters.reroutes || 0) + 1;
    this._event('reroute', slot, { avoid: this.closedEdges });
  }
  /** Car reached its station stop: join the station's FIFO pump queue (sim.js _enterStation). */
  _enterStation(slot) {
    this.lot++;
    (this._inLot ||= new Set()).add(slot);
    const k = this.slotStation[slot];
    if (!this.stations || k < 0) { this.op(L.OP.PUMPED, slot, f32bits(this.time)); return; }
    const st = this.stations;
    st.queue[k].push({ slot, since: this.time });
    if (st.queue[k].length > st.stats.maxQueue) st.stats.maxQueue = st.queue[k].length;
    this._event('refuelArrive', slot, { station: k, queue: st.queue[k].length });
  }
  _pumps() {
    const st = this.stations; if (!st || !this.lot) return;
    const fc = this.opts.fuelCfg, keep = [];
    for (const p of this.pumping) {
      if (this.time >= p.until) { st.busy[p.station]--; st.served[p.station]++; this.counters.refuelled++; this.op(L.OP.PUMPED, p.slot, f32bits(this.time)); this._event('refuelled', p.slot, { station: p.station }); }
      else keep.push(p);
    }
    this.pumping = keep;
    for (let k = 0; k < st.list.length; k++) {
      const q = st.queue[k];
      while (q.length && st.busy[k] < st.list[k].pumps) {
        const { slot, since } = q.shift(); st.busy[k]++;
        st.stats.waitSum += this.time - since; st.stats.waitN++;
        this.pumping.push({ slot, station: k, until: this.time + fc.pumpMin + (fc.pumpMax - fc.pumpMin) * this._rand() });
      }
    }
  }

  // ------------------------------------------------------------------ training interface helpers
  /** Cars per freeway edge (motorway / trunk) right now: Uint32Array(E). */
  async freewayCounts() {
    const enc = this.device.createCommandEncoder(), p = enc.beginComputePass();
    this._dispatch(p, this.K.densClear, this.BG.densClear, Math.ceil(this.world.E / 256));
    this._dispatch(p, this.K.densCount, this.BG.densCount, Math.ceil(this.cap / 256));
    p.end(); this.device.queue.submit([enc.finish()]);
    return new Uint32Array(await this.gpu.read(this.gpu.buffers.get('edgeCount'), this.world.E * 4));
  }
  /** Per-agent day results: {rewardSum Float32, outcome Uint8 (leg 0), outcomeBack Uint8, tripTime Float32, crashes Uint32}. */
  async readAgents() {
    await this.device.queue.onSubmittedWorkDone();
    const u = new Uint32Array(await this.gpu.read(this.gpu.buffers.get('agents'), this.nAgents * 16)), f = new Float32Array(u.buffer);
    const n = this.nAgents, out = { rewardSum: new Float32Array(n), outcome: new Uint8Array(n), outcomeBack: new Uint8Array(n), tripTime: new Float32Array(n), crashes: new Uint32Array(n), progress: new Float32Array(n) };
    for (let a = 0; a < n; a++) { out.rewardSum[a] = f[4 * a]; out.outcome[a] = u[4 * a + 1] & 255; out.outcomeBack[a] = (u[4 * a + 1] >> 8) & 255; out.progress[a] = (u[4 * a + 1] >>> 16) / 65535; out.tripTime[a] = f[4 * a + 2]; out.crashes[a] = u[4 * a + 3]; }
    return out;
  }
  /** Observation vectors (obsSize floats each) of the given slots, concatenated. */
  async readInputs(slots) {
    const IN = this.IN, staging = this.device.createBuffer({ size: Math.max(16, slots.length * IN * 4), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder(), src = this.gpu.buffers.get('inputs');
    slots.forEach((s, k) => enc.copyBufferToBuffer(src, s * IN * 4, staging, k * IN * 4, IN * 4));
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(staging.getMappedRange().slice(0)); staging.unmap(); staging.destroy();
    return out;
  }
  /**
   * Start a new simulated day with the same world / population / capacity: every slot freed,
   * scenes, queues, counters and the per-agent results cleared, clock set to startTime.
   */
  resetDay(startTime = this.opts.startTime) {
    const gb = this.gpu.buffers, enc = this.device.createCommandEncoder();
    for (const k of ['veh', 'ring', 'agents', 'queue', 'laneClaim', 'claim', 'telemetry', 'ctl']) enc.clearBuffer(gb.get(k));
    if (this.v3) enc.clearBuffer(gb.get('inputs'), this.cap * this.IN * 4, this.cap * LH_W * 4);   // v3 leader history
    enc.clearBuffer(gb.get('world'), this.layoutOff.ebOff * 4, this.world.E * 4);
    this.device.queue.submit([enc.finish()]);
    const hidden = new Float32Array(this.cap * 4); for (let j = 1; j < hidden.length; j += 4) hidden[j] = -1000;
    this.gpu.write(gb.get('pose'), hidden);
    this.time = startTime; this.step = 0; this.accumulator = 0;
    const io = new Uint32Array(L.IO.EVDATA); this.gpu.write(gb.get('io'), io);
    this.gpuStats.fill(0);
    this.free = []; for (let i = this.cap - 1; i >= 0; i--) this.free.push(i);
    this.slotTrip.fill(-1); this.slotStation.fill(-1);
    this.queueFree = []; for (let k = this.opts.spawnQueue - 1; k >= 0; k--) this.queueFree.push(k);
    this.queueHigh = 0; this.opsPending = []; this.wrecks.clear(); this._inCrash.clear(); this.refuelQueue = []; this.pumping = []; this.lot = 0; this._inLot?.clear();
    this.crashes = []; this.activeCrashes = []; this.edgeHead.fill(0); this.blockNext.fill(0);
    this.blockFree = []; for (let b = this.opts.maxBlocks - 1; b >= 0; b--) this.blockFree.push(b);
    for (const k of Object.keys(this.counters)) this.counters[k] = typeof this.counters[k] === 'object' && this.counters[k] ? {} : 0;   // bySeverity / stuckBy stay maps
    // host-side queues of the previous day (reroutes, live traffic, closures)
    this.rerouteQueue = []; this._lastReroute?.clear(); this._closedKey = undefined; this._closedN = 0; this.closedEdges = [];
    this._edgeSpeed?.clear(); this.trafficPenalty?.clear(); this._nextTraffic = -1e9;
    if (this._fuelPool) { this._fuelPool.done = []; this._fuelPool.rdone = []; }
    if (this.table) {
      this.tablePtr = this.table.indexAt(startTime - 3600); this.pendingStream = []; this.prefetchPtr = this.tablePtr; this.fuelLeft.fill(-1); this.outcome.fill(0);
      this._arena = { free: new Map(), slot: new Int32Array(this.cap).fill(-1), cls: new Uint8Array(this.cap) }; this.routeWords = 1;
    } else if (this.pop) {
      this.tripPtr = 0; this.fuelLeft.fill(-1); this.outcome.fill(0);
      // routes added during the day (refuel detours) are dropped: keep only the population pack
      this.routeWords = this.tripRouteEnd ?? this.routeWords;
      if (this.stations) { const st = this.stations; st.busy.fill(0); for (const q of st.queue) q.length = 0; }
    }
    this.events = [];
  }

  // ------------------------------------------------------------------ stats
  _u64(k) { return this.gpuStats[k] + this.gpuStats[k + 1] * 4294967296; }
  get clock() { const t = this.time; const h = Math.floor(t / 3600) % 24, m = Math.floor(t / 60) % 60, s = Math.floor(t) % 60; return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`; }

  /** Same shape as Simulation.getStats() (+ gpu: {memoryMB, kernelMs}). */
  getStats() {
    const s = this.gpuStats, S = L.S, c = this.counters;
    const n = this.pop ? this.pop.arrays.depart.length : (this.cap || 1);
    const driving = s[S.LIVE_DRIVING];
    const arrived = s[S.ARRIVED], arrivedOut = s[S.ARRIVED_OUT], onTime = s[S.ONTIME];
    const vehKm = this._u64(S.VEHM_LO) / 100000, speedKm = this._u64(S.SPEEDM_LO) / 100000;
    const vehSteps = this._u64(S.VEHSTEPS_LO), speedSteps = this._u64(S.SPEEDSTEPS_LO);
    const fb = [0, 0, 0, 0, 0, 0]; if (this.outcome) for (let i = 0; i < this.outcome.length; i++) fb[this.outcome[i]]++;
    const avg = driving ? s[S.LIVE_SUMSPEED] / 100 / driving : 0;
    const st = this.stations;
    return {
      clock: this.clock, time: this.time, speed: this.speed, lagging: this.lagging, stepMs: null,
      active: this.cap - this.free.length, driving, crashedOnScene: s[S.LIVE_CRASHED], refueling: s[S.LIVE_DWELL],
      spawned: s[S.SPAWNED], arrived, arrivedOutbound: arrivedOut,
      // crashes = every contact >= touchKmh (CPU sim.js counters.crashes); crashScenes = those that disabled a car
      crashes: c.crashEvents, crashScenes: c.crashes, crashedVehicles: c.crashedVehicles, secondaryCrashes: c.secondaryCrashes, activeCrashes: this.activeCrashes.length,
      cleared: c.cleared, stuck: s[S.STUCK], noRoute: c.noRoute, laneChanges: s[S.LANECHANGES], bumps: s[S.BUMPS],
      stranded: c.stranded, towed: c.towed, refuelPlanned: c.refuelPlanned, refuelled: c.refuelled, refuelPlanFailed: c.refuelPlanFailed,
      avgSpeed: avg, avgSpeedMph: avg / 0.44704,
      pendingSpawns: this.opts.spawnQueue - this.queueFree.length, tripsRemaining: this.trips ? this.trips.length - this.tripPtr : 0,
      commuters: n, arrivalRate: arrivedOut / n,
      failures: { crash: fb[2], stranded: fb[3], stuck: fb[4], noRoute: c.noRoute, unfinished: this.outcome ? fb[0] - c.noRoute : 0 },
      completionRate: s[S.SPAWNED] ? arrived / s[S.SPAWNED] : 0,
      policySteps: s[S.LIVE_POLICY],
      onTimeRate: onTime / n, onTimeOfArrived: arrivedOut ? onTime / arrivedOut : 0,
      meanLatenessMin: arrivedOut ? this._u64(S.LATESUM_LO) / arrivedOut / 60 : 0,
      speedingRate: vehSteps ? speedSteps / vehSteps : 0, vehicleKm: vehKm, speedingKm: speedKm,
      crashRatePerMillionKm: vehKm > 0 ? 1e6 * c.crashEvents / vehKm : 0,
      crashBySeverity: { ...c.bySeverity }, crashEvents: c.crashEvents, contactsLastStep: this.gpuStats[46], contactsSerial: this.gpuStats[47], signalViolations: c.signalViolations, stopViolations: c.stopViolations, pullOvers: c.pullOvers || 0, towedToShoulder: c.towedToShoulder || 0, repeatContacts: this.gpuStats[24], wreckPasses: this.gpuStats[25], reroutes: c.reroutes || 0, missedExits: c.missedExits || 0, disablingCrashes: c.crashes, hauledAway: c.hauledAway || 0, minorContacts: c.minorContacts,
      meanPumpWaitMin: st && st.stats.waitN ? st.stats.waitSum / st.stats.waitN / 60 : 0, maxPumpQueue: st ? st.stats.maxQueue : 0,
      accidents: s[S.ACCIDENTS], offroad: c.offroad, eventsDropped: c.eventsDropped,
      gpu: { memoryMB: this.gpu.memoryBytes() / 1048576, kernelMs: this.kernelMs },
    };
  }
}

/** [x, y, z, heading] at distance s along SimGraph edge ei, lat m from its right boundary (headingRel 0). */
export function poseAt(g, ei, s, lat) {
  const e = g.edges[ei];
  const off = (e.rightEdge ?? (e.laneOff[0] + LANE_W / 2)) - lat;
  const { px, pz, cum } = e, n = px.length;
  let i = 0; while (i < n - 2 && cum[i + 1] < s) i++;
  const segLen = cum[i + 1] - cum[i];
  const t = segLen > 0 ? Math.min(1, Math.max(0, (s - cum[i]) / segLen)) : 0;
  let dx = px[i + 1] - px[i], dz = pz[i + 1] - pz[i]; const Ln = Math.hypot(dx, dz) || 1; dx /= Ln; dz /= Ln;
  const y = e.ys ? e.ys[i] + (e.ys[i + 1] - e.ys[i]) * t : e.y;
  return [px[i] + dx * t * segLen - dz * off, y, pz[i] + dz * t * segLen + dx * off, Math.atan2(dz, dx)];
}

export { EDGE_ID_MASK };
