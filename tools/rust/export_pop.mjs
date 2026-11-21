// Population oracle for the Rust generator (rust/crates/bayenv-sim/src/population_gen.rs): builds a world like
// tools/rust/trace.mjs buildEnv, generates its population exactly as createTrainingEnv would and writes
// <out>/world.pack (unless --noworld and it exists) + <out>/pop.pack (meta: world, n, seed, popOpts, projection, ms).
//
//   node --max-old-space-size=14000 tools/rust/export_pop.mjs --world real --n 10000 --seed 2 --out /tmp/p [--parallel] [--opts '{"returnTrips":true}']
//     --world     synthetic | real | scenario:<dir name in data/scenarios>
//     --parallel  real world only: popParallel.js generatePopulationParallel (what env.js uses for n >= 4000)
//     --opts      population overrides (generatePopulation options; scenario: scenarioPopulation options), e.g. a region
//                 restriction the way tools/train/region-pack.mjs builds it:
//                 --parallel --nostations --opts '{"returnTrips":true,"region":{"preset":"sf"}}'
//     --workers   popParallel worker count (default cpus - 2; output does not depend on it)
//     --nostations  real world: loadWorld({ stations: false }) (region-pack.mjs; the graph is the same)
import fs from 'node:fs';
import { buildGraph } from '../../src/sim/roadGraph.js';
import { makeSynthGraph } from '../../src/sim/synthGraph.js';
import { generatePopulation, synthAxis } from '../../src/sim/population.js';
import { exportWorld, exportPopulation } from './export.mjs';

const argv = process.argv.slice(2);
const arg = (k, d) => { const i = argv.indexOf('--' + k); return i >= 0 ? argv[i + 1] : d; };
const flag = (k) => argv.includes('--' + k);

const world = arg('world', 'synthetic'), n = +arg('n', 300), seed = +arg('seed', 1), out = arg('out', 'pop-out');
const extra = JSON.parse(arg('opts', '{}'));
fs.mkdirSync(out, { recursive: true });
const needWorld = !(flag('noworld') && fs.existsSync(`${out}/world.pack`));
let graph, stations = [], controls = null, pop, ms, popOpts;
if (world === 'synthetic') {
  graph = buildGraph(makeSynthGraph({ blocks: 12, cross: 6, bridges: 2 }));
  popOpts = { n, seed, returnTrips: false, axis: synthAxis(graph), long: { minM: 5000, maxM: 16000 }, departures: 'spec', ...extra };
  const t = Date.now(); pop = generatePopulation(graph, popOpts); ms = Date.now() - t;
} else if (world === 'real') {
  const { loadWorld } = await import('../../src/sim/data.js');
  const w = await loadWorld(flag('nostations') ? { stations: false } : {}); graph = w.graph; stations = w.stations ?? []; controls = w.controls;
  popOpts = { n, seed, returnTrips: false, ...extra };
  const t = Date.now();
  if (flag('parallel')) { const { generatePopulationParallel } = await import('../../src/sim/popParallel.js'); pop = await generatePopulationParallel(graph, popOpts, arg('workers') ? { workers: +arg('workers') } : {}); }
  else pop = generatePopulation(graph, popOpts);
  ms = Date.now() - t;
} else if (world.startsWith('scenario:')) {
  const { loadScenario } = await import('../../src/scenarios/load.js');
  const sw = await loadScenario(`data/scenarios/${world.slice(9)}/`);
  graph = sw.graph; stations = sw.stations; controls = sw.controls;
  popOpts = { seed, ...extra };
  const t = Date.now(); pop = sw.population(popOpts); ms = Date.now() - t;
} else throw new Error(`unknown world ${world}`);
if (needWorld) exportWorld(graph, { stations, controls, path: `${out}/world.pack`, extraMeta: { world } });
const { axis, ...plain } = popOpts;
exportPopulation(pop, `${out}/pop.pack`, { world, seed, popOpts: plain, projection: graph.meta?.projection ?? null, ms, parallel: flag('parallel') });
console.log(JSON.stringify({ world, n: pop.commuters.length, seed, ms, projection: graph.meta?.projection ?? null, stats: { lengthRejectKept: pop.stats.lengthRejectKept, meanAttempts: pop.stats.meanAttempts }, out }));
