// Smoke test / API example: drive every car with a trivial policy (no model).
//   node examples/random_policy.mjs [--world synthetic|real] [--n 300] [--steps 600] [--policy random|cruise] [--seed 1]
// Prints steps/s, active cars, arrivals, crashes and the per-agent reward sum (src/sim/reward.js).
import { createTrainingEnv } from '../src/sim/env.js';
import { buildGraph } from '../src/sim/roadGraph.js';
import { makeSynthGraph } from '../src/sim/synthGraph.js';
import { synthAxis } from '../src/sim/population.js';
import { reward, REWARD_WEIGHTS } from '../src/sim/reward.js';

const argv = process.argv.slice(2);
const arg = (k, d) => { const i = argv.indexOf('--' + k); return i >= 0 ? argv[i + 1] : d; };
const world = arg('world', 'synthetic'), n = +arg('n', 300), steps = +arg('steps', 600), policy = arg('policy', 'cruise'), seed = +arg('seed', 1);

let opts = { density: n, trafficMode: 'full', seed };
if (world === 'synthetic') {
  const graph = buildGraph(makeSynthGraph({ blocks: 12, cross: 6, bridges: 2 }));
  opts = { ...opts, graph, population: { axis: synthAxis(graph), long: { minM: 5000, maxM: 16000 }, departures: 'spec' } };
}
const t0 = Date.now();
const env = await createTrainingEnv(opts);
const tLoad = Date.now() - t0;

let rnd = seed >>> 0; const rand = () => ((rnd = (Math.imul(rnd ^ (rnd >>> 15), 1 | rnd) + 0x6D2B79F5) >>> 0) / 4294967296);
const prevDist = new Map(); let rewardSum = 0, obsCount = 0;
const warm = +arg('warmup', 20000);   // run the policy until ~5 % of the agents are on the road (not timed)
let t1 = Date.now(), k = -1, timed = false;
for (let w = 0; ; w++) {
  if (!timed && (env.stats().active >= 0.05 * n || w >= warm)) { timed = true; k = 0; t1 = Date.now(); obsCount = 0; rewardSum = 0; }
  if (timed && k++ >= steps) break;
  const agents = env.observe();
  for (const { slot, obs } of agents) {
    obsCount++;
    // 'cruise': hold the lane (P on lane offset / heading) at ~the limit; 'random': uniform random pedals and wheel
    const a = policy === 'random'
      ? { steer: rand() * 0.2 - 0.1, gas: rand(), brake: rand() * 0.2 }
      : { steer: Math.max(-1, Math.min(1, (-0.08 * obs.laneOffset - 1.2 * obs.headingRel) / obs.maxSteer)),
          gas: obs.speed < 0.8 * obs.speedLimit ? 0.4 : 0, brake: obs.neighbors[0] < 15 ? 0.6 : 0 };
    env.act(slot, a);
    const d = obs.routeDistance, p = prevDist.has(slot) ? (d - prevDist.get(slot)) / Math.max(1, obs.speedLimit * env.sim.dt) : 0;
    prevDist.set(slot, d);
    const sig = env.sim.signalsFor(slot);
    rewardSum += reward({ progress: p, collision: sig.collision, stranded: sig.stranded, dt: env.sim.dt }, REWARD_WEIGHTS);
  }
  env.step();
}
const wall = (Date.now() - t1) / 1000, st = env.stats();
console.log(JSON.stringify({ world, policy, n, steps, loadMs: tLoad, stepsPerSec: +(steps / wall).toFixed(1), agentStepsPerSec: Math.round(obsCount / wall),
  clock: st.clock, active: st.active, arrived: st.arrivedOutbound, crashes: st.crashes, rewardSum: +rewardSum.toFixed(2) }));
