// Multi-agent driving environment (curriculum: trafficMode empty -> sparse -> full).
//
//   const env = await createTrainingEnv({ graph, stations, density, trafficMode, seed, ...opts });
//
//   graph       SimGraph (roadGraph.js) or a src/nav Graph (adapted with fromNavGraph); omitted ->
//               the real two-tier road packs from data/ (data.js loadWorld; stations too)
//   controls    TrafficControls (controls.js); omitted with the real graph -> data/controls.*
//   stations    [{node, pumps, x, z}] (data.js stationsOnGraph) or a src/nav/stations.js list
//               ({node, pumps, x, z} on the same graph); omitted with the real graph -> data/stations.bin
//   density     agents: an integer count (>= 1) or a fraction of 100,000 (0 < density <= 1)
//   trafficMode 'empty'  stage 0: agents do not see / hit each other (no accidents), static scenes only
//               'sparse' stage 1: full interaction + `wrecks` stopped wrecks (default 40) on main roads
//   staticObstacles: [{edge, at, lane | lat, type}] -> Simulation.placeStaticObstacles on every reset
//               (scenario worlds: exact disabled cars, permanent scenes)
//   obstacles: { count, kinds, seed, slowKmh, brakeEvery } -> Simulation.placeObstacles (any traffic mode):
//               real collidable bodies seen only through the normal obs. kinds: midblock, afterCurve,
//               afterMerge, junction (on a connector), rampEnd, partial (half-lane), slow (10-20 km/h
//               movers), braking (hard-braking leads). sim.obstacles lists what was placed.
//               'full'   stage 2: everything (crashes, accidents, rubbernecking, pump queues)
//   seed        population + sim seed (deterministic)
//   The env-only build has NO built-in driver: every car is driven by the caller (multi-agent, all cars are agents):
//               env.observe() -> per-agent obs, env.act(slot, controls), env.step(). A car with no command this step
//               holds {steer 0, gas 0, brake 0.3} (opts.defaultAction overrides). opts.driverFactory may replace
//               the external driver entirely (sim.js driverFactory contract).
//   opts.population  generatePopulation overrides (returnTrips defaults to false for training)
//   opts.startTime / dt / learnerFraction / wrecks / fuel / accident / any SIM_DEFAULTS key
//
// The env object:
//   env.sim, env.graph, env.population, env.deltas (Float32Array n*64: every agent's 64-float driver profile (traits.js),
//   GPU-uploadable; the agents' Genomes are views into it), env.n
//   env.step(k = 1)               advance k sim steps
//   env.run(simHours, {log})      run and return the stats (arrivalRate, failures, ...)
//   env.runUntilDone(maxHours)    run until every trip has finished (or maxHours)
//   env.stats()                   sim.getStats()
//   env.fitness()                 Float32Array(n): per-agent trip fitness (reward-shaped, see sim._arrive / _failTrip)
//   env.outcomes()                Uint8Array(n): FAIL.* per agent (outbound trip)
//   env.setDeltas(buf)            copy a new population of deltas in (evolution); env.reset() restarts the day
//   env.observe({v3})              [{slot, agent, obs, signals}] for every driving car this step
//   env.act(slot, {steer, gas, brake})        command for this step (delayed by reaction time);
//               steer = target wheel position in [-1,1] (opts.actionMode 'position', default) or pass
//               {steerRate} (legacy, opts.actionMode 'rate'; also accepted in position mode, integrated)
//   env.reset({seed})             a fresh Simulation over the same population (deltas kept)

import { Simulation, FAIL } from './sim.js';
import { generatePopulation } from './population.js';
import { fromNavGraph } from './roadGraph.js';
import { fillObs, fillObs3 } from './obs.js';
import { STATE_DRIVING } from './vehicle.js';

const DEFAULT_N = 100000;

export async function createTrainingEnv({ graph = null, stations = null, controls = null, density = 2000, trafficMode = 'full', seed = 1, ...opts } = {}) {
  let g = graph, st = stations, ctl = controls;
  if (!g) {
    const { loadWorld } = await import('./data.js');
    const w = await loadWorld(opts.dataBase ? { base: opts.dataBase } : {});
    g = w.graph; if (!st) st = w.stations; if (!ctl) ctl = w.controls;
  } else if (!g.outStart) g = fromNavGraph(g);   // a src/nav Graph
  const n = density > 1 ? Math.round(density) : Math.max(1, Math.round(density * DEFAULT_N));
  const popOpts = { n, seed, returnTrips: false, ...(opts.population || {}) };
  let pop;
  if (opts.pop) pop = opts.pop;
  else if (n >= 4000 && typeof process !== 'undefined' && process.versions?.node && !opts.population?.axis && !graph) {
    const { generatePopulationParallel } = await import('./popParallel.js');
    pop = await generatePopulationParallel(g, popOpts, { dataBase: opts.dataBase });
  } else pop = generatePopulation(g, popOpts);
  return new TrainingEnv(g, st || [], pop, { trafficMode, seed, ...opts, controls: ctl });
}

export class TrainingEnv {
  constructor(graph, stations, pop, opts) {
    this.graph = graph; this.stations = stations; this.population = pop; this.opts = opts;
    this.n = pop.commuters.length;
    this.deltas = pop.deltas;
    this._external = new Map();
    this.reset();
  }

  reset({ seed } = {}) {
    const o = this.opts;
    for (const c of this.population.commuters) { c.genome.fitness = 0; c.actualArrival = -1; c.lateBy = 0; c.earlyBy = 0; }
    const env = this;
    const driverFactory = o.driverFactory || ((graph, from, to, pose, extra) => ({
      step() { return env._external.get(extra.slot) || env._idle; },
    }));
    this._idle = { ...ZERO, ...(o.defaultAction || {}) };
    const simOpts = {
      ...o, pop: this.population, stations: this.stations, driverFactory, seed: seed ?? o.seed,
      startTime: o.startTime ?? defaultStart(this.population), dt: o.dt ?? 0.1,
      trafficMode: o.trafficMode, wrecks: o.trafficMode === 'sparse' ? (o.wrecks ?? 40) : 0,
      tripLog: o.tripLog ?? false,
    };
    if (o.trafficMode === 'empty') simOpts.accident = { ratePerKm: 0, ...(o.accident || {}) };
    delete simOpts.graph;
    this.sim = new Simulation(this.graph, simOpts);
    this._external.clear();
    return this;
  }

  step(k = 1) { for (let q = 0; q < k; q++) this.sim.step(); this._external.clear(); return this; }

  run(simHours, { log = null } = {}) {
    const sim = this.sim, end = sim.time + simHours * 3600;
    let next = sim.time + 1800;
    while (sim.time < end) {
      sim.step();
      if (log && sim.time >= next) { const s = sim.getStats(); log(`[${s.clock}] active=${s.active} arrived=${s.arrivedOutbound}/${this.n} crashes=${s.crashes} stuck=${s.stuck}`); next += 1800; }
    }
    return sim.getStats();
  }

  /** Run until all trips are done (no pending departures, nobody on the road) or maxHours elapse. */
  runUntilDone(maxHours = 8, { log = null } = {}) {
    const sim = this.sim, end = sim.time + maxHours * 3600;
    while (sim.time < end) {
      for (let k = 0; k < 600; k++) sim.step();
      if (sim.isIdle) break;
      if (log) { const s = sim.getStats(); log(`[${s.clock}] active=${s.active} arrived=${s.arrivedOutbound}/${this.n}`); }
    }
    return sim.getStats();
  }

  stats() { return this.sim.getStats(); }
  outcomes() { return this.sim.outcome; }
  fitness() { const f = new Float32Array(this.n); this.population.commuters.forEach((c, i) => { f[i] = c.genome.fitness || 0; }); return f; }
  setDeltas(buf) { if (buf.length !== this.deltas.length) throw new Error('delta buffer size mismatch'); this.deltas.set(buf); }

  /** Observations of every driving agent (fresh objects; neighbours from the footprint lane lists).
   *  opts {v3: true} adds obs.v3 (Float32Array(358), obsLayout.js) per agent. */
  observe({ v3 = false } = {}) {
    const sim = this.sim, vs = sim.vs, out = [];
    for (let i = 0; i < vs.high; i++) {
      if (vs.state[i] !== STATE_DRIVING) continue;
      const obs = { neighbors: new Float32Array(12) };
      fillObs(sim, i, obs); if (v3) fillObs3(sim, i, obs);
      obs.signals = { ...sim.signalsFor(i) };
      out.push({ slot: i, agent: vs.commuter[i], obs, delta: this.deltas.subarray(vs.commuter[i] * 64, vs.commuter[i] * 64 + 64) });
    }
    return out;
  }
  act(slot, controls) { const c = { gas: +controls.gas || 0, brake: +controls.brake || 0 }; if (controls.steer !== undefined) c.steer = +controls.steer || 0; if (controls.steerRate !== undefined) c.steerRate = +controls.steerRate || 0; if (c.steer === undefined && c.steerRate === undefined) c.steer = 0; this._external.set(slot, c); }
}

const ZERO = Object.freeze({ steer: 0, gas: 0, brake: 0.3 });
function defaultStart(pop) {
  let t = Infinity; for (const c of pop.commuters) if (c.depart < t) t = c.depart;
  return Math.floor((Number.isFinite(t) ? t : 5 * 3600) / 60) * 60 - 60;
}

export { FAIL };
