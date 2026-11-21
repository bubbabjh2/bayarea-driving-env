// Synthetic training worlds: public API (see README.md).
//
// Curriculum hook for the brain (beats.js): a level whose world is a scenario pack instead of the Bay Area.
//   const lvl = await scenarioLevel('junction+merge/120', { seed: round + 1 });   // or 'all/200', 'junction:control=signal/40'
//   const env = await createTrainingEnv({ ...lvl.envOpts, trafficMode: 'full', driver: 'brain', brain, learnerFraction: 0.1 });
//   lvl.attach(env);                       // static obstacles (and again after every env.reset())
//   ... run the day ...;  const rep = lvl.report(env);   // per family / per tile arrival + contacts
// Node only for generation (writes data/scenarios/cache/<key>/ once, then loads it through the sim's own loader).
export { World, packSections, CLASSES_ALL, FLAG } from './builder.js';
export { FAMILIES, makeScenario } from './families.js';
export { packWorld, mixedPack, parseSpec } from './pack.js';
export { loadScenario, scenarioPopulation, placeScenarioObstacles, attachScenario, scenarioReport } from './load.js';
import { FAMILIES, makeScenario } from './families.js';
import { packWorld, parseSpec } from './pack.js';
import { loadScenario, attachScenario, scenarioReport } from './load.js';
import { makeRng, hashSeed } from '../sim/rng.js';

/**
 * Parse a scenario level spec  '<families>/<tiles>[/<rate>]':
 *   families: 'all' | 'freeway' | 'surface' | a family | several joined with '+'; a family may carry overrides
 *             'junction:control=signal,movement=left' (FAMILIES[f].params keys)
 *   tiles:    number of scenarios in the pack (default 60); rate: demand multiplier (default 1)
 * -> { families: ['junction:control=signal', ...], tiles, rate }
 */
export function parseScenarioLevel(str) {
  const [fams = 'all', tiles = '60', rate = '1'] = String(str).split('/');
  const groups = { all: Object.keys(FAMILIES), freeway: ['merge', 'diverge', 'weave', 'rampmeter', 'interchange', 'lanes', 'straight'],
    surface: ['junction', 'grid', 'urban', 'roundabout', 'curve', 'sbend', 'hill', 'obstacle', 'overpass'] };
  const families = fams.split('+').flatMap((f) => groups[f] || [f]);
  for (const f of families) if (!FAMILIES[parseSpec(f)[0]]) throw new Error(`scenario level: unknown family '${f}'`);
  return { families, tiles: +tiles, rate: +rate };
}

/**
 * Build (cached) and load a scenario-pack level. Returns
 * { envOpts (graph, stations, controls, pop, density, startTime, seed), attach(env), report(env), sw, agents, key }.
 * The same (spec, seed) always gives the same pack; use a new seed per round for fresh scenarios.
 */
export async function scenarioLevel(spec, { seed = 1, cacheDir = null, popSeed = seed, deltaBase = null } = {}) {
  const [{ default: fs }, { default: path }, { fileURLToPath }, { writeScenario }] = await Promise.all([import('node:fs'), import('node:path'), import('node:url'), import('./write.js')]);
  const L = parseScenarioLevel(spec);
  const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
  const key = `lvl-${String(spec).replace(/[^a-zA-Z0-9]+/g, '_')}-s${seed}`;
  const dir = path.join(cacheDir || path.join(root, 'data/scenarios/cache'), key);
  if (!fs.existsSync(path.join(dir, 'junctions.meta.json'))) {
    const R = makeRng(hashSeed(`level:${spec}:${seed}`)), worlds = [];
    for (let k = 0; k < L.tiles; k++) { const [f, ov] = parseSpec(L.families[k % L.families.length]); worlds.push(makeScenario(f, 1 + R.int(1e9), ov, { rateScale: L.rate })); }
    writeScenario(packWorld(worlds, { name: key, seed }).finish(), dir, { json: false });
  }
  const sw = await loadScenario(dir);
  const envOpts = sw.envOpts({ seed: popSeed, deltaBase });
  return { envOpts, sw, key, dir, agents: envOpts.pop.commuters.length,
    attach: (env) => attachScenario(env, sw), report: (env) => scenarioReport(env.sim, sw, env.population) };
}
