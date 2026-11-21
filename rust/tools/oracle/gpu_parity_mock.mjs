// JS oracle for the Rust GPU host's population / refuel / obstacle / snapshot paths (no GPU needed): runs
// src/gpu/gpuSim.js against the recording WebGPU mock (mock_webgpu.mjs) and dumps the host-side uploads and state
// that rust/crates/bayenv-gpu/examples/parity_check.rs compares bit for bit.
//
//   node --max-old-space-size=14000 rust/tools/oracle/gpu_parity_mock.mjs <dataDir> <outDir> --scenario=pop
//        [--n=20000] [--seed=42] [--start=25200] [--cap=32768] [--refuels=40]
//   ... --scenario=obstacles [--cap=4096] [--count=40] [--oseed=3] [--wrecks=12]
//   ... --scenario=snapshot  [--density=600] [--steps=2400]
// pop: <outDir>/pop.{json,bin} (popPack format, generated once with popParallel.js, reused by the Chrome page and
// the Rust side), mock.json + mock.bin (setPopulation routes, spawn queue, params, slot tables, refuel plans,
// pump / station / arrival event script results). obstacles: placeObstacles(spec) + placeWrecks(n) +
// placeObstacles(list) -> veh / par / ctl / world-block words. snapshot: CPU sim (createTrainingEnv) exportState ->
// snap.{json,bin} and the importSnapshot uploads.
import { writeFileSync, existsSync, readFileSync, mkdirSync } from 'node:fs';
import path from 'node:path';
import { gpu } from './mock_webgpu.mjs';

const args = process.argv.slice(2);
const [dataDir, outDir] = args.filter((a) => !a.startsWith('--'));
const opt = (k, d) => { const a = args.find((s) => s.startsWith(`--${k}=`)); return a ? a.slice(k.length + 3) : d; };
mkdirSync(outDir, { recursive: true });
const scenario = opt('scenario', 'pop');
const { GPUSim } = await import('../../../src/gpu/gpuSim.js');
const L = await import('../../../src/gpu/layout.js');
const { loadWorld } = await import('../../../src/sim/data.js');
const base = dataDir.endsWith('/') ? dataDir : dataDir + '/';
const t0 = Date.now();
const world = await loadWorld({ base });
console.log(`world ${Date.now() - t0} ms`);

// binary dump: named sections (typed arrays) -> bin + meta.sections
const parts = []; let off = 0; const sections = {};
const add = (name, arr) => { const pad = (8 - (off % 8)) % 8; if (pad) { parts.push(new Uint8Array(pad)); off += pad; } sections[name] = { type: arr.constructor.name, byteOffset: off, length: arr.length }; parts.push(new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength)); off += arr.byteLength; };
const writeOut = (name, meta) => { writeFileSync(path.join(outDir, name + '.bin'), Buffer.concat(parts.map((p) => Buffer.from(p.buffer, p.byteOffset, p.byteLength)))); writeFileSync(path.join(outDir, name + '.json'), JSON.stringify({ ...meta, sections })); parts.length = 0; off = 0; for (const k of Object.keys(sections)) delete sections[k]; };
const buf = (name) => new Uint32Array(gpu.buffers.get(name).data.buffer);
const io = (events) => { const a = new Uint32Array(L.IO.EVDATA + 4 * events.length); a[L.IO.EVCOUNT] = events.length; events.forEach((e, k) => a.set(e, L.IO.EVDATA + 4 * k)); return a; };
const f32u = (x) => new Uint32Array(Float32Array.of(x).buffer)[0];
const takeOps = (sim) => { const o = sim.opsPending.slice(); sim.opsPending = []; return o; };

const mkSim = (cap, extra = {}) => GPUSim.create(world.graph, { gpu, capacity: cap, obs: 'v2', features: { accidents: true, fuel: true, collisions: true, laneChanges: true }, controls: world.controls, fuelWorkers: 0, refuelBudgetMs: Infinity, ...extra });

if (scenario === 'pop') {
  const n = +opt('n', 20000), seed = +opt('seed', 42), start = +opt('start', 25200), cap = +opt('cap', 32768), R = +opt('refuels', 40);
  // population (popPack format), generated once
  const popBase = path.join(outDir, 'pop');
  if (!existsSync(popBase + '.bin')) {
    const { generatePopulationParallel } = await import('../../../src/sim/popParallel.js');
    const { ARRAY_FIELDS } = await import('../../../src/gpu/popPack.js');
    const t = Date.now();
    const pop = await generatePopulationParallel(world.graph, { n, seed }, {});
    console.log(`population ${n} seed ${seed}: ${Date.now() - t} ms`);
    add('routesFlat', pop.routes.flat); add('routesOffset', pop.routes.offset); add('deltas', pop.deltas);
    for (const k of ARRAY_FIELDS) if (pop.arrays[k]) add(k, pop.arrays[k]);
    writeOut('pop', { format: 'gpu-pop/1', n, seed, edgeCount: world.graph.edgeCount, ms: Date.now() - t });
  }
  const meta = JSON.parse(readFileSync(popBase + '.json', 'utf8')), pbin = readFileSync(popBase + '.bin');
  const TY = { Int32Array, Uint32Array, Float32Array, Uint8Array };
  const sec = (k) => { const s = meta.sections[k]; return s ? new TY[s.type](pbin.buffer, pbin.byteOffset + s.byteOffset, s.length) : null; };
  const pop = { meta: {}, routes: { flat: sec('routesFlat'), offset: sec('routesOffset') }, deltas: sec('deltas'), arrays: {} };
  for (const k of ['home', 'work', 'kind', 'vehicleType', 'workStart', 'depart', 'returnDepart', 'expectedTravel', 'routeLength', 'intended', 'slack', 'reactionTime', 'fuelFill', 'crossesBay']) if (sec(k)) pop.arrays[k] = sec(k);
  const sim = await mkSim(cap, { startTime: start });
  { const A = new Float32Array(sim.cap * 3); for (let i = 0; i < sim.cap; i++) A[3 * i + 1] = 0.3; sim.setActions(A); }
  const tp = Date.now(); sim.setPopulation(pop, { stations: world.stations }); const setPopMs = Date.now() - tp;
  const out = { n, seed, start, cap, setPopMs, routeWordsPop: sim.routeWords, tripRouteBase: sim.tripRouteBase, nAgents: sim.nAgents, noRoute: sim.counters.noRoute, trips: sim.trips.length };
  add('routesPop', sim.routeMirror.slice(0, sim.routeWords)); add('trips', sim.trips); add('tripDepart', sim.tripDepart);
  // first frame: runSteps(10) (spawns enqueued for depart <= t + 0.5 + 2)
  const te = Date.now(); sim._enqueueSpawns(sim.time + 10 * sim.dt); out.enqueueMs = Date.now() - te; out.queueHigh0 = sim.queueHigh;
  await sim.runSteps(10);
  out.queueHigh = sim.queueHigh; out.freeLen = sim.free.length; out.time1 = sim.time;
  add('queue1', buf('queue').slice(0, sim.queueHigh * L.SPAWN_WORDS)); add('params1', buf('params').slice()); add('slotTrip1', sim.slotTrip.slice()); add('slotRoute1', sim.slotRoute.slice());
  // refuel plans for the first R spawned slots (from their first route edge), main thread
  const req = []; for (let s = 0; s < sim.cap && req.length < R; s++) if (sim.slotTrip[s] >= 0) { const r = sim._routeOf(s); req.push([s, r[0] & 0xFFFFF]); }
  for (const [slot, edge] of req) sim.refuelQueue.push({ slot, edge, atSpawn: false });
  const rw0 = sim.routeWords, tr = Date.now(); sim._processRefuels(); out.refuelMs = Date.now() - tr; out.refuelRequests = req.length;
  out.refuelPlanned = sim.counters.refuelPlanned; out.refuelPlanFailed = sim.counters.refuelPlanFailed; out.routeWords2 = sim.routeWords;
  add('refuelReq', Uint32Array.from(req.flat())); add('refuelOps', Uint32Array.from(takeOps(sim))); add('routesRefuel', sim.routeMirror.slice(rw0, sim.routeWords)); add('slotStation2', sim.slotStation.slice());
  // stations: every requested slot reaches its stop (EV_STATION); pumps at t, t+60, ..., t+600; then REENTER
  const evs = req.map(([s]) => [L.EV.STATION, s, 0, 0]);
  sim._ingest(io(evs)); const opsSt = takeOps(sim); out.lot3 = sim.lot;
  const pumpOps = [];
  for (let k = 0; k <= 10; k++) { sim._pumps(); pumpOps.push(...takeOps(sim), 0xFFFFFFFF); sim.time += 60; }
  out.refuelled = sim.counters.refuelled; out.maxQueue = sim.stations.stats.maxQueue; out.waitSum = sim.stations.stats.waitSum; out.waitN = sim.stations.stats.waitN;
  sim._ingest(io(req.slice(0, R / 2).map(([s]) => [L.EV.REENTER, s, 0, 0]))); out.lot4 = sim.lot; out.reentered = sim.counters.reentered;
  add('stationOps', Uint32Array.from(opsSt)); add('pumpOps', Uint32Array.from(pumpOps));
  // spawned + arrivals (fuel carried to the return leg), then a long enqueue (return legs included)
  // the queued entries spawned: their queue slots return to the free list (queue order)
  const qh = sim.queueHigh; sim._ingest(io(Array.from({ length: qh }, (_, q) => [L.EV.SPAWN, 0, q, 0]))); out.spawned = sim.counters.spawned;
  const arr = []; for (let s = 0; s < sim.cap && arr.length < 300; s++) if (sim.slotTrip[s] >= 0 && (sim.slotTrip[s] & 1) === 0) arr.push([L.EV.ARRIVE, s, 0, f32u(3 + (s % 17))]);
  sim._ingest(io(arr)); out.arrived = arr.length;
  sim.time = 18 * 3600; sim._enqueueSpawns(sim.time + 0.5);
  out.queueHigh5 = sim.queueHigh; out.tripPtr5 = sim.tripPtr; out.freeLen5 = sim.free.length;
  add('queue5', buf('queue').slice(0, sim.queueHigh * L.SPAWN_WORDS)); add('fuelLeft5', sim.fuelLeft.slice()); add('outcome5', sim.outcome.slice()); add('slotTrip5', sim.slotTrip.slice());
  add('arriveEv', Uint32Array.from(arr.flat()));
  writeOut('mock', out);
  console.log(JSON.stringify(out));
} else if (scenario === 'obstacles') {
  const cap = +opt('cap', 4096), count = +opt('count', 40), oseed = +opt('oseed', 3), nw = +opt('wrecks', 12);
  const sim = await mkSim(cap, { startTime: 8 * 3600, seed: 5 });
  const placed = sim.placeObstacles({ count, seed: oseed });
  const w = sim.placeWrecks(nw);
  const E = world.graph.edges;
  const list = [{ edge: placed[0].edge, at: 20, lane: 0 }, { edge: placed[1].edge, at: -5, lat: 2.5, type: 2 }, { edge: 1234, at: 1e6, lane: 1 }, { edge: E.length + 5, at: 3 }];
  const placed2 = sim.placeObstacles(list);
  const lo = sim.layoutOff;
  add('veh', buf('veh').slice()); add('par', buf('par').slice()); add('ctl', buf('ctl').slice(0, cap * 36));
  add('blocks', buf('world').slice(lo.blkOff, lo.blkOff + 256 * L.BLOCK_WORDS)); add('edgeHead', sim.edgeHead.slice()); add('routes', sim.routeMirror.slice(0, sim.routeWords)); add('slotTrip', sim.slotTrip.slice());
  // doubles as binary sections (JSON number parsing is not always exact on the Rust side)
  add('placedS', Float64Array.from(placed, (p) => p.s)); add('placed2S', Float64Array.from(placed2, (p) => p.s)); add('sceneF', Float64Array.from(sim.activeCrashes.flatMap((r) => [r.s, r.squeezeAt])));
  const out = { cap, count, oseed, nw, placed: placed.map((p) => [p.slot, p.kind, p.edge, p.s]), wrecks: w, placed2: placed2.map((p) => [p.slot, p.kind, p.edge, p.s]), list, routeWords: sim.routeWords, scenes: sim.activeCrashes.length,
    scene0: sim.activeCrashes.slice(0, 50).map((r) => [r.edge, r.laneMask, r.s, r.block, r.squeezeAt, r.cause]), free: sim.free.length };
  writeOut('mock', out);
  console.log(JSON.stringify({ placed: placed.length, wrecks: w, placed2: placed2.length, scenes: out.scenes }));
} else if (scenario === 'snapshot') {
  const density = +opt('density', 4000), steps = +opt('steps', 6000), cap = +opt('cap', 2048);
  const { createTrainingEnv } = await import('../../../src/sim/env.js');
  const { exportState, STATE_FIELDS } = await import('../../../src/sim/state.js');
  const env = await createTrainingEnv({ graph: world.graph, stations: world.stations, controls: world.controls, density, seed: 3 });
  const ts = Date.now(); for (let k = 0; k < steps; k++) env.sim.step(); console.log(`cpu ${steps} steps ${Date.now() - ts} ms, high ${env.sim.vs.high}`);
  const snap = exportState(env.sim);
  for (const [name] of STATE_FIELDS) add(name, snap.arrays[name]);
  add('routesFlat', snap.routes.flat); add('commuterWorkStart', Float64Array.from(env.sim.commuters, (c) => c.workStart));
  writeOut('snap', { version: snap.version, ...snap.meta });
  const sim = await mkSim(cap, { startTime: 5 * 3600 });
  sim.importSnapshot(snap, env.sim.commuters);
  add('veh', buf('veh').slice()); add('par', buf('par').slice()); add('ring', buf('ring').slice()); add('pose', buf('pose').slice()); add('routes', sim.routeMirror.slice(0, sim.routeWords)); add('slotTrip', sim.slotTrip.slice());
  writeOut('mock', { cap, high: snap.meta.high, time: sim.time, step: sim.step, dt: sim.dt, free: sim.free.length, routeWords: sim.routeWords });
  console.log(JSON.stringify({ high: snap.meta.high, count: snap.meta.count, time: sim.time }));
}
process.exit(0);
