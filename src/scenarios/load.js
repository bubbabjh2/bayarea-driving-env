// Load a scenario (or world pack) directory written by write.js and turn it into the inputs of the
// sim's createTrainingEnv: the graph / stations / controls come from the sim's OWN loader
// (src/sim/data.js loadWorld({ base })), so nothing in the sim, GPU or brain changes; the
// population comes from the scenario's trip list instead of the SPEC commute generator.
//
//   const sw = await loadScenario('data/scenarios/left-signal-s1/');      // node path or URL
//   const env = await createTrainingEnv({ ...sw.envOpts(), trafficMode: 'full', driver: 'brain', brain });
//   placeScenarioObstacles(env.sim, sw.scenario);                        // or: attachScenario(env, sw)
//   const report = scenarioReport(env.sim, sw);                          // per-tile / per-family outcome
import { loadWorld } from '../sim/data.js';
import { makeRouter } from '../sim/route.js';
import { assemble, mergeCfg, synthAxis, deltaFor } from '../sim/population.js';
import { makeGenome } from '../sim/traits.js';
import { makeRng, hashSeed, prefixSums } from '../sim/rng.js';
import { VEHICLE_TYPE_WEIGHTS } from '../sim/vehicleTypes.js';

const isNode = typeof process !== 'undefined' && !!process.versions?.node && typeof window === 'undefined';
async function readJSON(base, f) {
  const url = base instanceof URL ? new URL(f, base) : (String(base).endsWith('/') ? base + f : base + '/' + f);
  if (isNode) { const fs = await import('node:fs/promises'); return JSON.parse(await fs.readFile(url instanceof URL ? url.pathname : url, 'utf8')); }
  const r = await fetch(url); if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`); return r.json();
}

/** Load the world (sim loader) + scenario doc; returns { graph, stations, controls, main, local, scenario, base, population(opts), envOpts(opts) }. */
export async function loadScenario(base, { stations = true, controls = true } = {}) {
  const b = isNode && !(base instanceof URL) && !String(base).endsWith('/') ? base + '/' : base;
  const scenario = await readJSON(b, 'scenario.json');
  const world = await loadWorld({ base: b, stations, controls });
  const sw = { ...world, scenario, base: b };
  sw.population = (opts = {}) => scenarioPopulation(world.graph, scenario, opts);
  /** createTrainingEnv options: { graph, stations, controls, pop, density, startTime } (+ any overrides). */
  sw.envOpts = (opts = {}) => {
    const pop = sw.population(opts);
    return { graph: world.graph, stations: world.stations, controls: world.controls, pop, density: pop.commuters.length,
      startTime: scenario.startTime ?? defaultStart(pop), seed: opts.seed ?? scenario.seed ?? 1 };
  };
  return sw;
}

function defaultStart(pop) { let t = Infinity; for (const c of pop.commuters) t = Math.min(t, c.depart); return Math.floor((Number.isFinite(t) ? t : 7 * 3600) / 60) * 60 - 60; }

/**
 * Population for a scenario's trips. Each trip spec {from, to, via, count, depart: [t0, t1], types?, tile?}
 * becomes `count x scale` commuters (rounded, >= 1 if count > 0) with routes from the sim's own A*
 * (src/sim/route.js) through the via nodes, departures spread evenly over the window with jitter,
 * reaction time / vehicle type / delta drawn exactly like src/sim/population.js, a full tank
 * (fuelFill, no refuel detours) and time pressure from the SPEC slack rule.
 * pop.tripTile[i] / pop.tripSpec[i] map commuters back to the scenario tile and trip.
 */
export function scenarioPopulation(graph, scenario, { seed = scenario.seed ?? 1, scale = 1, fuelFill = 1, maxAgents = Infinity, deltaBase = null, timeShift = 0 } = {}) {
  const cfg = mergeCfg({ seed, n: 0, returnTrips: false });
  const route = makeRouter(graph, { weight: 1, cacheMax: 0 });
  const base = deltaBase ? makeGenome(deltaBase) : makeGenome();
  const typeCum = prefixSums(VEHICLE_TYPE_WEIGHTS);
  const recs = [], tripTile = [], tripSpec = [];
  const rc = cfg.reactionTime, sig = Math.log(rc.p95 / rc.median) / 1.6449;
  const edges = graph.edges;
  scenario.trips.forEach((t, ti) => {
    const n = t.count > 0 ? Math.max(1, Math.round(t.count * scale)) : 0;
    let r = null;
    const legs = [t.from, ...(t.via || []), t.to];
    const parts = [];
    for (let k = 0; k + 1 < legs.length; k++) { const p = route(legs[k], legs[k + 1]); if (!p || !p.length) { parts.length = 0; break; } parts.push(p); }
    if (parts.length) { let L = 0; for (const p of parts) L += p.length; r = new Int32Array(L); let o = 0; for (const p of parts) { r.set(p, o); o += p.length; } }
    if (!r) throw new Error(`scenario ${scenario.name}: trip ${ti} (${t.from} -> ${t.to}) has no route`);
    if (t.route) r = Int32Array.from(t.route);
    let L = 0, T = 0; for (const e of r) { L += edges[e].length; T += edges[e].length / (edges[e].speedLimit * cfg.travelSpeedFactor); }
    for (let j = 0; j < n && recs.length < maxAgents; j++) {
      const id = recs.length, rng = makeRng(hashSeed(`${seed}:scn:${ti}:${j}`));
      const [d0, d1] = t.depart;
      const depart = timeShift + d0 + (n > 1 ? (d1 - d0) * (j + rng.random() * 0.8) / n : rng.random() * (d1 - d0));
      const slack = rng.range(cfg.slack.min, cfg.slack.max);
      const reactionTime = Math.min(rc.max, Math.max(rc.min, rc.median * Math.exp(sig * rng.gaussian())));
      const vehicleType = t.types ? t.types[rng.int(t.types.length)] : rng.weightedIndex(typeCum);
      recs.push({ id, home: t.from, work: t.to, kind: 'local', d: L, L, lengthOk: true, attempts: 1, routeOut: r, routeBack: new Int32Array(0),
        expectedTravel: T, workStart: depart + slack * T, slack, depart, returnDepart: Infinity, vehicleType, fuelFill, reactionTime,
        delta: deltaFor(cfg, base, id, reactionTime), side: [0, 0], crossesBay: false });
      tripTile.push(t.tile ?? 0); tripSpec.push(ti);
    }
  });
  const pop = assemble(recs, cfg, synthAxis(graph));
  pop.tripTile = Int32Array.from(tripTile); pop.tripSpec = Int32Array.from(tripSpec);
  pop.scenario = scenario.name;
  return pop;
}

/**
 * Place the scenario's static obstacles (stopped, disabled cars: real collidable bodies, permanent scenes,
 * perceived only through the normal observation) into a Simulation. Same body / scene setup as
 * Simulation.placeObstacles' static kinds. Returns the slots.
 */
export function placeScenarioObstacles(sim, scenario) {
  const g = sim.graph, vs = sim.vs, out = [];
  for (const o of scenario.obstacles || []) {
    const e = g.edges[o.edge];
    const s = Math.min(e.length - 1, Math.max(1, o.at));
    const lw = g.laneGeom ? g.laneGeom.laneWidth(o.edge, s) : 3.7;
    const lat = o.lat ?? (o.lane + 0.5) * lw;
    const slot = vs.alloc({ typeIdx: o.type ?? 0, route: Int32Array.of(o.edge), commuter: -1, time: sim.time, dt: sim.dt, step: sim.stepCount, graph: g });
    vs.s[slot] = s; vs.lat[slot] = lat; if (vs.conn) vs.conn[slot] = -1; vs.updateLanes(slot, e);
    vs.updatePose(slot, g, sim._pose); vs.savePose(slot);
    sim._openScene([slot], { cause: 'obstacle', severity: 'wreck', impactSpeed: 0, impactMph: 0, lanesBlocked: 1, blockDuration: 1e9, clearDuration: 1e9 }, e, false);
    vs.flags[slot] |= 32;   // FLAG_DISABLED
    out.push(slot);
  }
  return out;
}

/** Keep obstacles across env.reset() (TrainingEnv makes a fresh Simulation per day). */
export function attachScenario(env, sw) {
  const reset = env.reset.bind(env);
  env.reset = (o) => { reset(o); placeScenarioObstacles(env.sim, sw.scenario); return env; };
  placeScenarioObstacles(env.sim, sw.scenario);
  return env;
}

/**
 * Outcome per tile (scenario) and per family, from the sim's per-commuter outcome array
 * (FAIL: 0 pending, 1 arrived, 2 crash, 3 stranded, 4 stuck, 5 noRoute).
 */
export function scenarioReport(sim, sw, pop = sim.population) {
  const tiles = sw.scenario.tiles || [{ name: sw.scenario.name, family: sw.scenario.family }];
  const per = tiles.map((t) => ({ name: t.name, family: t.family, n: 0, arrived: 0, crash: 0, stranded: 0, stuck: 0, pending: 0, noRoute: 0 }));
  const names = ['pending', 'arrived', 'crash', 'stranded', 'stuck', 'noRoute'];
  for (let i = 0; i < sim.outcome.length; i++) { const t = per[pop.tripTile[i]] || per[0]; t.n++; t[names[sim.outcome[i]]]++; }
  // physical contacts (sim.crashes, isCrash) located by tile box; disabling = a car's own delta-v >= 10 km/h
  for (const t of per) { t.contacts = 0; t.disabling = 0; t.barrier = 0; t.types = {}; }
  for (const c of sim.crashes || []) {
    const x = c.contact ? c.contact.x : c.x, z = c.contact ? c.contact.z : c.z;
    if (!c.isCrash || x == null) continue;
    const k = tiles.findIndex((t) => x >= t.bbox.minX - 30 && x <= t.bbox.maxX + 30 && z >= t.bbox.minZ - 30 && z <= t.bbox.maxZ + 30);
    const t = per[k >= 0 ? k : 0]; t.contacts++; if ((c.disabled || c.vehicles || []).length && c.responseState) t.disabling++; if (c.kind === 'barrier') t.barrier++;
    const key = `${c.type || c.kind || '?'}/${c.maneuver || '?'}`; t.types[key] = (t.types[key] || 0) + 1;
  }
  const fam = {};
  for (const t of per) { const f = fam[t.family] || (fam[t.family] = { family: t.family, tiles: 0, n: 0, arrived: 0, crash: 0, stranded: 0, stuck: 0, pending: 0, noRoute: 0, contacts: 0, disabling: 0, barrier: 0 }); f.tiles++; for (const k of ['n', ...names, 'contacts', 'disabling', 'barrier']) f[k] += t[k]; f.types = f.types || {}; for (const [k, v] of Object.entries(t.types)) f.types[k] = (f.types[k] || 0) + v; }
  for (const f of Object.values(fam)) f.arrival = f.n ? +(f.arrived / f.n).toFixed(4) : null;
  for (const t of per) t.arrival = t.n ? +(t.arrived / t.n).toFixed(4) : null;
  return { tiles: per, families: Object.values(fam) };
}
