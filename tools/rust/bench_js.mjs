// JS baseline for rust/bayenv-sim/BENCHMARKS.md: same env + policy as tools/rust/trace.mjs, no recording.
//   node tools/rust/bench_js.mjs --world real --n 10000 --policy cruise --steps 2000 --warmup 600 --opts '{"startTime":27000}' [--out dir]
// --out also writes world.pack / pop.pack / run.json there (inputs for rust bench: cargo run --release -p bayenv-sim --bin bench -- dir).
import fs from 'node:fs';
import { buildEnv, makePolicy } from './trace.mjs';
import { exportWorld, exportPopulation } from './export.mjs';
import { STATE_DRIVING } from '../../src/sim/vehicle.js';

const argv = process.argv.slice(2);
const arg = (k, d) => { const i = argv.indexOf('--' + k); return i >= 0 ? argv[i + 1] : d; };
const world = arg('world', 'synthetic'), n = +arg('n', 300), seed = +arg('seed', 1), mode = arg('mode', 'full');
const steps = +arg('steps', 2000), warm = +arg('warmup', 0), policy = arg('policy', 'cruise'), out = arg('out', null);
const opts = JSON.parse(arg('opts', '{}'));
const t0 = Date.now();
const { env, stations, controls, obstacles } = await buildEnv({ world, n, seed, mode, opts });
const tLoad = Date.now() - t0;
const sim = env.sim;
if (out) {
  fs.mkdirSync(out, { recursive: true });
  exportWorld(sim.graph, { stations, controls, path: `${out}/world.pack`, extraMeta: { world } });
  exportPopulation(env.population, `${out}/pop.pack`);
  const simOpts = {};
  for (const [k, v] of Object.entries(sim.opts)) if (typeof v === 'number' || typeof v === 'string' || typeof v === 'boolean') simOpts[k] = Number.isFinite(v) || typeof v !== 'number' ? v : (v > 0 ? 'Infinity' : '-Infinity');
  fs.writeFileSync(`${out}/run.json`, JSON.stringify({ world, n, seed, mode, steps, every: 1, policy, opts, simOpts, obstacles, startTime: sim.opts.startTime, wrecks: sim.opts.wrecks ?? 0, defaultAction: sim.defaultAction, staticObstacles: sim.opts.staticObstacles || null }, null, 1));
}
const act = makePolicy(policy, seed);
for (let k = 0; k < warm; k++) { act(env); env.step(); }
let agentSteps = 0;
const t1 = process.hrtime.bigint();
for (let k = 0; k < steps; k++) { const vs = env.sim.vs; for (let i = 0; i < vs.high; i++) if (vs.state[i] === STATE_DRIVING) agentSteps++; act(env); env.step(); }
const s = Number(process.hrtime.bigint() - t1) / 1e9;
console.log(JSON.stringify({ world, n, policy, steps, warm, loadMs: tLoad, stepsPerSec: +(steps / s).toFixed(1), agentStepsPerSec: Math.round(agentSteps / s), meanAgents: +(agentSteps / steps).toFixed(1), clock: env.stats().clock, crashes: env.stats().crashes }));
