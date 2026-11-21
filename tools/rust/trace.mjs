// Differential-test driver for the Rust port (rust/bayenv-sim): builds a JS env, exports its world and
// population (tools/rust/export.mjs), runs it under a deterministic action policy and records the full
// per-car state (every VehicleStore typed array, slots [0, high)) after every step.
//
//   node tools/rust/trace.mjs --world scenario:junction-s1 --policy cruise --steps 600 --seed 1 --out /tmp/t
//     --world  synthetic | real | scenario:<dir name in data/scenarios>
//     --n      agents (synthetic / real)        --mode full|sparse|empty     --opts '{"pathSight":1}' (sim opts JSON)
//     --policy idle | const | random | cruise | cruisenoise
//     --every  record every k-th step (default 1)  --fields all | pose (a compact subset)
//     --obs 1  also record env.observe({v3: true}) on every recorded step (after the step, before the next actions):
//              sections obs.n (agents per recorded step), obs.slot, obs.scalars (f64 rows of OBS_FIELDS, the
//              rust/crates/bayenv-sim/src/obs.rs flat layout), obs.v3 (f32 rows of 358)
// Writes <out>/world.pack, <out>/pop.pack, <out>/run.json, <out>/trace.pack.
import fs from 'node:fs';
import { createTrainingEnv } from '../../src/sim/env.js';
import { buildGraph, laneW } from '../../src/sim/roadGraph.js';
import { makeSynthGraph } from '../../src/sim/synthGraph.js';
import { synthAxis } from '../../src/sim/population.js';
import { makeRng } from '../../src/sim/rng.js';
import { STATE_DRIVING } from '../../src/sim/vehicle.js';
import { VEHICLE_PARAMS } from '../../src/sim/physics/params.js';
import { PackWriter } from './pack.mjs';
import { exportWorld, exportPopulation } from './export.mjs';

const argv = process.argv.slice(2);
const arg = (k, d) => { const i = argv.indexOf('--' + k); return i >= 0 ? argv[i + 1] : d; };

export async function buildEnv({ world = 'synthetic', n = 300, seed = 1, mode = 'full', opts = {} }) {
  if (world === 'synthetic') {
    const graph = buildGraph(makeSynthGraph({ blocks: 12, cross: 6, bridges: 2 }));
    const env = await createTrainingEnv({ graph, density: n, trafficMode: mode, seed, ...opts, population: { axis: synthAxis(graph), long: { minM: 5000, maxM: 16000 }, departures: 'spec' } });
    return { env, stations: [], controls: null, obstacles: [] };
  }
  if (world === 'real') {
    const { loadWorld } = await import('../../src/sim/data.js');
    const w = await loadWorld({});
    const env = await createTrainingEnv({ graph: w.graph, stations: w.stations, controls: w.controls, density: n, trafficMode: mode, seed, ...opts });
    return { env, stations: w.stations, controls: w.controls, obstacles: [] };
  }
  if (world.startsWith('scenario:')) {
    const { loadScenario, attachScenario } = await import('../../src/scenarios/load.js');
    const sw = await loadScenario(`data/scenarios/${world.slice(9)}/`);
    const env = await createTrainingEnv({ ...sw.envOpts({ seed }), trafficMode: mode, ...opts });
    attachScenario(env, sw);
    return { env, stations: sw.stations, controls: sw.controls, obstacles: sw.scenario.obstacles || [] };
  }
  throw new Error(`unknown world ${world}`);
}

/** Deterministic policies (mirrored by rust/bayenv-sim/src/harness.rs). */
export function makePolicy(name, seed) {
  const R = makeRng((seed * 7919 + 13) >>> 0);
  const clamp = (x) => (x > 1 ? 1 : x < -1 ? -1 : x);
  return (env) => {
    const sim = env.sim, vs = sim.vs, g = sim.graph;
    if (name === 'idle') return;
    for (let i = 0; i < vs.high; i++) {
      if (vs.state[i] !== STATE_DRIVING) continue;
      if (name === 'const') { env.act(i, { steer: 0, gas: 0.3, brake: 0 }); continue; }
      if (name === 'random') { const a = R.random(), b = R.random(), c = R.random(); env.act(i, { steer: a * 0.2 - 0.1, gas: b, brake: c * 0.2 }); continue; }
      const e = g.edges[vs.edge[i]], lw = laneW(e, vs.s[i]);
      const off = vs.lat[i] - (vs.lane[i] + 0.5) * lw;
      let steer = clamp((-0.08 * off - 1.2 * vs.headingRel[i]) / vs.maxSteer[i]);
      let gas = vs.speed[i] < 0.8 * e.speedLimit ? 0.4 : 0, brake = vs.gap[i] < 15 ? 0.6 : 0;
      if (name === 'cruisenoise') { const a = R.random(), b = R.random(); steer = clamp(steer + (a - 0.5) * 0.1); if (b < 0.05) { gas = 1; brake = 0; } else if (b < 0.08) brake = 0; }
      env.act(i, { steer, gas, brake });
    }
  };
}

/** obs scalar row: same order as OBS_FIELDS in rust/crates/bayenv-sim/src/obs.rs */
export const OBS_FIELDS = ['slot', 'agent', 'x', 'z', 'heading', 'speed', 'laneIndex', 'laneCount', 'laneWidth', 'laneOffset', 'onConnector', 'laneHeading',
  'speedLimit', 'roadId', 'routeDistance', 'steerAngle', 'steerTarget', 'gas', 'brake', 'actionMode', 'maxSteerAngle', 'headingRel', 'lat',
  'laneEndDistance', 'laneEndSide', 'maxSteer', 'maxSteerRate', 'wheelbase', 'width', 'length', 'mass', 'vehicleType', 'vLat', 'yawRate',
  'control.type', 'control.state', 'control.distance', 'control.timeInState', 'control.timeToChange', 'control.movement', 'control.stopped', 'control.yieldRequired',
  ...Array.from({ length: 12 }, (_, k) => `neighbors.${k}`),
  'lead2.gap', 'lead2.rel', 'lead2.stopped', 'pullOver', 'oncoming.gap', 'oncoming.closing',
  'junction.mouthDistance', 'junction.inBox', 'junction.distance', 'junction.timeToConflict', 'junction.stopped', 'passingWreck',
  'grade.0', 'grade.1', 'grade.2', 'leaderId',
  'signals.collision', 'signals.stranded', 'signals.harshBraking', 'signals.timeToWorkStart', 'signals.remainingRouteDistance', 'signals.fuelFraction', 'signals.refuelPlanned'];
const getPath = (o, p) => { for (const k of p.split('.')) o = o == null ? undefined : o[k]; return o; };
let obsUndefined = 0;
export function obsRow(it) {
  const o = it.obs;
  return OBS_FIELDS.map((f) => {
    let v;
    if (f === 'slot') v = it.slot; else if (f === 'agent') v = it.agent;
    else if (f === 'actionMode') v = o.actionMode === 'position' ? 1 : 0;
    else if (f === 'vehicleType') v = VEHICLE_PARAMS.findIndex((p) => p.name === o.vehicleType);
    else v = getPath(o, f);
    if (v === undefined) { obsUndefined++; return NaN; }
    return typeof v === 'boolean' ? (v ? 1 : 0) : +v;
  });
}

const POSE = ['state', 'flags', 'edge', 's', 'lat', 'headingRel', 'speed', 'vLat', 'yawRate', 'x', 'y', 'z', 'heading', 'leader', 'gap', 'conn'];

async function main() {
  const world = arg('world', 'synthetic'), n = +arg('n', 300), seed = +arg('seed', 1), mode = arg('mode', 'full');
  const steps = +arg('steps', 600), every = +arg('every', 1), policy = arg('policy', 'cruise'), out = arg('out', 'trace-out');
  const opts = JSON.parse(arg('opts', '{}')), fieldSet = arg('fields', 'all'), withObs = +arg('obs', 0) === 1;
  fs.mkdirSync(out, { recursive: true });
  const t0 = Date.now();
  const { env, stations, controls, obstacles } = await buildEnv({ world, n, seed, mode, opts });
  const sim = env.sim;
  exportWorld(sim.graph, { stations, controls, path: `${out}/world.pack`, extraMeta: { world } });
  exportPopulation(env.population, `${out}/pop.pack`);
  const simOpts = {};
  for (const [k, v] of Object.entries(sim.opts)) if (typeof v === 'number' || typeof v === 'string' || typeof v === 'boolean') simOpts[k] = Number.isFinite(v) || typeof v !== 'number' ? v : (v > 0 ? 'Infinity' : '-Infinity');
  const run = { world, n, seed, mode, steps, every, policy, hook: arg('hook', '0') === '1', opts, obs: withObs, simOpts, obstacles, startTime: sim.opts.startTime, wrecks: sim.opts.wrecks ?? 0,
    defaultAction: sim.defaultAction, obstaclesSpec: sim.opts.obstacles || null, staticObstacles: sim.opts.staticObstacles || null };
  fs.writeFileSync(`${out}/run.json`, JSON.stringify(run, null, 1));
  const tLoad = Date.now() - t0;
  const act0 = makePolicy(policy, seed), hook = arg('hook', '0') === '1';
  // --hook 1: the policy runs inside sim._drive (batchDrive / Rust Sim::batch_policy semantics) instead of before env.step
  if (hook) { const orig = sim._drive.bind(sim); sim._drive = (dt) => { act0(env); orig(dt); }; }
  const act = hook ? () => {} : act0;
  const vs0 = sim.vs;
  const fields = Object.keys(vs0).filter((k) => ArrayBuffer.isView(vs0[k]) && vs0[k].length % vs0.capacity === 0 && (fieldSet === 'all' || POSE.includes(k)));
  const chunks = Object.fromEntries(fields.map((k) => [k, []]));
  const highs = [], scal = [];
  const obsN = [], obsSlot = [], obsScal = [], obsV3 = []; let obsNs = 0, obsAgents = 0;
  const t1 = Date.now();
  for (let k = 1; k <= steps; k++) {
    act(env);
    env.step();
    if (k % every) continue;
    const vs = env.sim.vs, h = vs.high;
    highs.push(h);
    for (const f of fields) { const a = vs[f], st = a.length / vs.capacity; chunks[f].push(a.slice(0, h * st)); }
    const c = env.sim.counters;
    scal.push(env.sim.time, env.sim.stepCount, vs.count, c.spawned, c.arrived, c.crashes, c.disablingCrashes, c.minorContacts, c.touches, c.repeatContacts,
      env.sim.crashes.length, env.sim.activeCrashes.length, c.stuck, c.stranded, c.refuelPlanned, c.reroutes || 0, c.missedExits || 0, env.sim.pendingSpawns.length);
    if (withObs) {
      const tq = process.hrtime.bigint();
      const items = env.observe({ v3: true });
      obsNs += Number(process.hrtime.bigint() - tq); obsAgents += items.length;
      obsN.push(items.length);
      for (const it of items) { obsSlot.push(it.slot); for (const v of obsRow(it)) obsScal.push(v); obsV3.push(it.obs.v3); }
    }
  }
  const wall = (Date.now() - t1) / 1000;
  const w = new PackWriter();
  w.add('high', Uint32Array.from(highs)).add('scalars', Float64Array.from(scal));
  if (withObs) {
    const v3 = new Float32Array(obsV3.length * 358); obsV3.forEach((a, k) => v3.set(a, k * 358));
    w.add('obs.n', Uint32Array.from(obsN)).add('obs.slot', Int32Array.from(obsSlot)).add('obs.scalars', Float64Array.from(obsScal)).add('obs.v3', v3);
  }
  for (const f of fields) {
    const parts = chunks[f], C = parts.length ? parts[0].constructor : Float32Array;
    let L = 0; for (const p of parts) L += p.length;
    const all = new C(L); let o = 0; for (const p of parts) { all.set(p, o); o += p.length; }
    w.add('vs.' + f, all);
  }
  w.meta = { kind: 'trace', fields, strides: Object.fromEntries(fields.map((f) => [f, vs0[f].length / vs0.capacity])), scalarNames: ['time', 'stepCount', 'count', 'spawned', 'arrived', 'crashes', 'disablingCrashes', 'minorContacts', 'touches', 'repeatContacts', 'crashRecords', 'activeCrashes', 'stuck', 'stranded', 'refuelPlanned', 'reroutes', 'missedExits', 'pending'], every, steps,
    ...(withObs ? { obsFields: OBS_FIELDS, obsUndefined } : {}) };
  w.write(`${out}/trace.pack`);
  const st = env.stats();
  console.log(JSON.stringify({ world, policy, steps, loadMs: tLoad, stepsPerSec: +(steps / wall).toFixed(1), clock: st.clock, active: st.active, arrived: st.arrivedOutbound, crashes: st.crashes, out,
    ...(withObs ? { observeNsPerAgent: +(obsNs / Math.max(1, obsAgents)).toFixed(0), obsAgents, obsUndefined } : {}) }));
}

if (import.meta.url === `file://${process.argv[1]}`) await main();
