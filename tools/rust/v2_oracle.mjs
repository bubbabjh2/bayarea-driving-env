// Oracle for the Rust v2 brain-input encoder (rust/crates/bayenv-sim/src/obs_v2.rs): runs a JS env exactly like
// tools/rust/trace.mjs (same buildEnv / makePolicy, same action stream) and computes, for every driving car on every
// recorded step, the 70-float input-version-2 vector of the training repo's CPU driver
// (bayarea-driving-sim src/brain/driver.js Driver.sense = eyes.js Eyes.look + adjust.js adjustInputs + the
// position-mode grade inputs 41..43), fed the way bayarea-driving-sim src/sim/brainAdapter.js feeds it:
//   * one Driver per trip, created at spawn from the sim's route (no refuel stop: the adapter ignores extra.stop),
//     delta = the agent's 64-float profile (env.deltas view), vehicle = VEHICLE_PARAMS[typeIdx], inputVersion 2;
//   * sim setRoute calls (refuel detour / reroute) rebuild its Guidance (with the refuel stop when one is planned),
//     as brainAdapter wrapper.setRoute;
//   * driver.sense(dt, null, obs) every step for every driving car, obs = env.observe() (src/sim/obs.js fillObs + signals);
//   * the pending-steer ring (Driver.act predictDelay bookkeeping) gets the steer command the env applied this step
//     (env.act's steer, or the idle default), as Driver.act would push its own command.
// The brain code is imported from the private repo by absolute path (read-only).
//
//   node tools/rust/v2_oracle.mjs --world scenario:junction-s1 --policy cruisenoise --steps 600 --seed 1 --out /tmp/v2
//     --world / --n / --mode / --opts / --policy / --steps / --seed / --every : as trace.mjs
//     --brain  path of the private repo (default /Users/bradenhuffman/Documents/youtube/bayarea-driving-sim)
// Writes <out>/world.pack, <out>/pop.pack, <out>/run.json (trace.mjs format) and <out>/v2.pack:
//   v2.n (u32 per recorded step: driving cars), v2.slot (i32), v2.x (f32 rows of 70),
//   edge.ref / edge.name (i32 per edge: interned road ref / name, 0 = '', for guidance 'continue' maneuvers),
//   v2.near (f64 rows of NEAR_FIELDS.length: the driver's eyes.near after sense, booleans 0/1, Infinity kept),
//   meta {kind: 'v2', every, steps, inputSize, nearFields, setRouteCalls, stopRoutes, stopsCompleted}
// (v2.near / meta.nearFields are additions; readers of older packs without them skip the near check)
import fs from 'node:fs';
import { buildEnv, makePolicy } from './trace.mjs';
import { STATE_DRIVING } from '../../src/sim/vehicle.js';
import { VEHICLE_PARAMS } from '../../src/sim/physics/params.js';
import { PackWriter } from './pack.mjs';
import { exportWorld, exportPopulation } from './export.mjs';

const argv = process.argv.slice(2);
const arg = (k, d) => { const i = argv.indexOf('--' + k); return i >= 0 ? argv[i + 1] : d; };
const BRAIN = arg('brain', '/Users/bradenhuffman/Documents/youtube/bayarea-driving-sim');
const { Driver } = await import(`${BRAIN}/src/brain/driver.js`);
const { Guidance } = await import(`${BRAIN}/src/nav/guidance.js`);
const { INPUT_SIZE } = await import(`${BRAIN}/src/brain/eyes.js`);

const world = arg('world', 'synthetic'), n = +arg('n', 300), seed = +arg('seed', 1), mode = arg('mode', 'full');
const steps = +arg('steps', 600), every = +arg('every', 1), policy = arg('policy', 'cruisenoise'), out = arg('out', 'v2-out');
const opts = JSON.parse(arg('opts', '{}'));
fs.mkdirSync(out, { recursive: true });
const { env, stations, controls, obstacles } = await buildEnv({ world, n, seed, mode, opts });
const sim = env.sim;
exportWorld(sim.graph, { stations, controls, path: `${out}/world.pack`, extraMeta: { world } });
exportPopulation(env.population, `${out}/pop.pack`);
const simOpts = {};
for (const [k, v] of Object.entries(sim.opts)) if (typeof v === 'number' || typeof v === 'string' || typeof v === 'boolean') simOpts[k] = Number.isFinite(v) || typeof v !== 'number' ? v : (v > 0 ? 'Infinity' : '-Infinity');
fs.writeFileSync(`${out}/run.json`, JSON.stringify({ world, n, seed, mode, steps, every, policy, opts, obs: false, simOpts, obstacles, startTime: sim.opts.startTime, wrecks: sim.opts.wrecks ?? 0,
  defaultAction: sim.defaultAction, obstaclesSpec: sim.opts.obstacles || null, staticObstacles: sim.opts.staticObstacles || null }, null, 1));

// ---- brainAdapter-style drivers, one per trip (keyed by the sim's per-spawn driver wrapper)
const graph = sim.graph.nav || sim.graph;
const recs = new Map();   // wrapper -> {drv}
let setRouteCalls = 0, stopRoutes = 0;
const origFactory = sim.driverFactory;
sim.driverFactory = (g, from, to, pose, extra) => {
  const w = origFactory(g, from, to, pose, extra);
  const P = VEHICLE_PARAMS[sim.vs.typeIdx[extra.slot]];
  const delta = env.deltas.subarray(extra.commuter.id * 64, extra.commuter.id * 64 + 64);
  const drv = new Driver({ graph: g, router: null, route: Array.from(extra.route), delta, mode: 'heuristic', id: extra.slot, speedCap: sim.vs.maxSpeed[extra.slot],
    inputVersion: 2, actionMode: 'position',
    vehicle: { name: P.name, wheelbase: P.L, maxSteer: P.maxSteer, maxSteerRate: P.maxSteerRate, maxSpeed: P.vmax, width: P.width, length: P.length, params: P } });
  recs.set(w, { drv });
  // brainAdapter wrapper.setRoute (called by sim._checkFuel / reroutes)
  w.setRoute = (edges, stop, routeDistance) => {
    setRouteCalls++;
    if (stop) {
      stopRoutes++;
      drv.guidance = new Guidance(g, Array.from(edges), { laneHintHorizon: undefined, stops: [{ node: stop.node, type: 'refuel', station: stop.station }] });
      drv.routeDistance = routeDistance || 0; drv._prevRouteDistance = null;
    } else drv.setRoute(Array.from(edges));
  };
  return w;
};
if (sim.vs.count > 0) throw new Error('cars on the road before the first step: drivers would be missing');

// pending-steer ring update = Driver.act predictDelay bookkeeping with the executed command
function pushPending(drv, cmd) {
  const keep = Math.min(39, Math.max(0, Math.round(drv.traits.reactionTime / sim.dt)));
  const P = drv._pend;
  if (drv._pendN >= keep) { if (keep > 0) { P.copyWithin(0, drv._pendN - keep + 1, drv._pendN); drv._pendN = keep - 1; } else drv._pendN = 0; }
  if (keep > 0) P[drv._pendN++] = cmd;
}

const act = makePolicy(policy, seed);
// driver.js eyes.near keys after Driver.sense (eyes.js look + Driver._adjust), order = obs_v2.rs NEAR_FIELDS
const NEAR_FIELDS = ['leaderGap', 'leaderSpeed', 'leaderRel', 'behindGap', 'behindSpeed',
  'leftAhead', 'leftBehind', 'leftAheadSpeed', 'leftBehindSpeed', 'rightAhead', 'rightBehind', 'rightAheadSpeed', 'rightBehindSpeed',
  'leftExists', 'rightExists', 'headingError', 'laneEndGap', 'leftEndGap', 'rightEndGap', 'endDist', 'endSide', 'leftEndDist', 'rightEndDist'];
const nearRows = [];
const vN = [], vSlot = [], rows = [];
let senseNs = 0n, senses = 0, stops0 = 0;
const t0 = Date.now();
for (let k = 1; k <= steps; k++) {
  act(env);
  {
    const vs = sim.vs;
    for (let i = 0; i < vs.high; i++) {
      if (vs.state[i] !== STATE_DRIVING) continue;
      const r = recs.get(sim.drivers[i]); if (!r) throw new Error(`no driver for slot ${i}`);
      const c = env._external.get(i) || env._idle;
      pushPending(r.drv, c.steer !== undefined ? c.steer : (c.steerRate ?? 0));
    }
  }
  env.step();
  const items = env.observe();
  const rec = k % every === 0;
  if (rec) vN.push(items.length);
  for (const it of items) {
    const r = recs.get(sim.drivers[it.slot]);
    const tq = process.hrtime.bigint();
    const input = r.drv.sense(sim.dt, null, it.obs);
    senseNs += process.hrtime.bigint() - tq; senses++;
    if (rec) {
      vSlot.push(it.slot); rows.push(Float32Array.from(input.subarray(0, INPUT_SIZE)));
      const nr = r.drv.eyes.near;
      nearRows.push(Float64Array.from(NEAR_FIELDS, (k) => { const v = nr[k]; if (v === undefined) throw new Error(`eyes.near.${k} undefined`); return typeof v === 'boolean' ? +v : v; }));
    }
  }
}
for (const { drv } of recs.values()) stops0 += drv.stopsCompleted || 0;
// GC the drivers of finished trips is not needed for these run lengths
const X = new Float32Array(rows.length * INPUT_SIZE); rows.forEach((a, q) => X.set(a, q * INPUT_SIZE));
const NX = new Float64Array(nearRows.length * NEAR_FIELDS.length); nearRows.forEach((a, q) => NX.set(a, q * NEAR_FIELDS.length));
const E = sim.graph.edges, refs = [''], names = [''];
const intern = (tab, s) => { s = s || ''; let q = tab.indexOf(s); if (q < 0) { tab.push(s); q = tab.length - 1; } return q; };
const eRef = Int32Array.from(E, (e) => intern(refs, e.ref)), eName = Int32Array.from(E, (e) => intern(names, e.name));
const w = new PackWriter();
w.add('v2.n', Uint32Array.from(vN)).add('v2.slot', Int32Array.from(vSlot)).add('v2.x', X).add('edge.ref', eRef).add('edge.name', eName).add('v2.near', NX);
w.meta = { kind: 'v2', every, steps, inputSize: INPUT_SIZE, nearFields: NEAR_FIELDS, setRouteCalls, stopRoutes, stopsCompleted: stops0, refs: refs.length, names: names.length };
w.write(`${out}/v2.pack`);
console.log(JSON.stringify({ world, policy, steps, rows: rows.length, trips: recs.size, setRouteCalls, stopRoutes, stopsCompleted: stops0, refuelPlanned: sim.counters.refuelPlanned, refuelled: sim.counters.refuelled, reroutes: sim.counters.reroutes || 0, missedExits: sim.counters.missedExits || 0, senseNsPerAgent: Number(senseNs / BigInt(Math.max(1, senses))), wallS: (Date.now() - t0) / 1000, out }));
