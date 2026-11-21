// Commuter population generator, exactly per SPEC.md "Population and jobs".
//
// Per commuter i (own RNG seeded from (seed, i), so any chunking / worker split
// gives identical results):
//  1. kind: local with p = 0.8, else long.
//  2. home: uniform over residential / tertiary (+ living_street) nodes of the
//     graph's largest strongly connected component (so outbound and return trips
//     are always routable).
//  3. intended length d:
//       local ~ LogNormal(median 5 km, sigma = ln(12/5)/1.2816 -> p90 = 12 km), clipped [0.5, 15] km,
//             direction uniform;
//       long  ~ U(10, 75) km along the SJ->SF axis u toward whichever end has room
//             (room toward SF = D - t, toward SJ = t, t = home's axis coordinate;
//             if both fit, pick proportional to room; if neither, the larger),
//             + N(0, 4 km) lateral jitter along v, then mirrored across the bay
//             centreline (line through the San Mateo Bridge midpoint, direction u)
//             with p = 0.4 to force bridge crossings.
//  4. workplace: snap the target point to the nearest SCC node; resample (new d /
//     direction) if the snap distance > max(1 km, d/4) (target in the bay / off the
//     map), the node is unreachable, or the routed length < 0.7 d. Up to
//     maxAttempts tries, then the best routed attempt is kept (counted in stats).
//  5. workStart ~ N(8:30, 40 min) clipped [6:30, 10:30].
//     T = sum(edge length / (speed limit * 0.8)) over the route (free flow x 0.8).
//     depart = workStart - f T, f ~ U(0.83, 1.33). Return trip departs
//     workStart + N(8.5 h, 20 min) (never before arrival is plausible) on the work->home route.
//  6. vehicle type by VEHICLE_TYPE_WEIGHTS; fuel fill U(0.1, 1.0);
//     driver = base agent + mutated 64-float delta (src/brain/evolve.js mutate),
//     reaction time LogNormal(median 0.9 s, p95 1.4 s) clipped [0.6, 1.8] in delta[2].
//
// Output is struct-of-arrays (GPU-uploadable) plus convenience commuter objects:
//   pop.routes = { flat: Int32Array, offset: Uint32Array(2n+1) }  route of trip k = 2*id+leg
//     is flat[offset[k] .. offset[k+1])  (directed SimGraph edge ids, in travel order)
//   pop.deltas = Float32Array(n*64)   per-agent brain delta (views in commuter.genome.delta)
//   pop.arrays = { home, work, kind, vehicleType, workStart, depart, returnDepart, expectedTravel,
//                  routeLength, intended, slack, reactionTime, fuelFill, crossesBay }

import { makeRng, hashSeed, prefixSums } from './rng.js';
import { HIGHWAY_RANK } from './roadGraph.js';
import { makeRouter, largestSCC } from './route.js';
import { VEHICLE_TYPES, VEHICLE_TYPE_WEIGHTS } from './vehicleTypes.js';
import { makeGenome, mutate, Genome, DELTA_SIZE } from './traits.js';

const H = 3600, MIN = 60;

export const POPULATION_DEFAULTS = Object.freeze({
  n: 2000,
  seed: 42,
  localShare: 0.8,
  local: { medianM: 5000, sigma: Math.log(12 / 5) / 1.2816, minM: 500, maxM: 15000 },
  long: { minM: 10000, maxM: 75000, lateralSigmaM: 4000, pCross: 0.4 },
  snapTolerance: (d) => Math.max(1000, d / 4),
  maxAttempts: 12,
  minRouteRatio: 0.7,
  workStart: { mean: 8.5 * H, sd: 40 * MIN, min: 6.5 * H, max: 10.5 * H },
  travelSpeedFactor: 0.8,
  slack: { min: 0.83, max: 1.33 },
  reactionTime: { median: 0.9, p95: 1.4, min: 0.6, max: 1.8 },
  workDuration: { mean: 8.5 * H, sd: 20 * MIN },
  fuelFill: { min: 0.1, max: 1.0 },
  genome: { sigma: 0.03, traitSigma: 0.12 },
  returnTrips: true,
  projection: { lat0: 37.55, lon0: -122.15 },
  sanJose: { lat: 37.3382, lon: -121.8863 },
  sanFrancisco: { lat: 37.7749, lon: -122.4194 },
  sanMateoBridgeMid: { lat: 37.595, lon: -122.192 },
  homeClasses: ['residential', 'tertiary', 'living_street'],
  astarWeight: 1.25,
  // departures: 'curve' = time leaving home drawn from the ACS 2024 15-min departure curve
  // (docs/realism.md, video/graphics/data/realism.json) and workStart = depart + f T, so the
  // SPEC slack rule depart = workStart - f T still holds; 'spec' = workStart ~ N(8:30, 40 min)
  departures: 'curve',
  departureCurve: {
    start: 6 * H, bin: 15 * MIN,
    share: [0.037, 0.042, 0.037, 0.041, 0.079, 0.08, 0.076, 0.079, 0.106, 0.095, 0.073, 0.067, 0.057, 0.049, 0.044, 0.038],
  },
  // trip mix preset: 'spec' (SPEC: 80/20 local/long, 40% of long mirrored across the bay) |
  // 'realism' (comparison only: LODES straight-line length bins, ~4.5% bay crossings, observed vehicle mix)
  mix: 'spec',
});

/** Trip-mix preset 'realism' (docs/realism.md; NOT the default). */
export const REALISM_MIX = Object.freeze({
  // LEHD LODES8 2022 straight-line commute length bins (km) and shares
  lengthBins: [[0.5, 5, 0.183], [5, 15, 0.282], [15, 30, 0.241], [30, 50, 0.125], [50, 75, 0.065], [75, 75, 0.104]],
  pCross: 0.05,           // of trips >= 15 km mirrored across the bay (bay-crossing AM vehicle trips ~4.5%)
  vehicleWeights: [0.452, 0.51, 0.035, 0.003], // sedan (+motorcycle), SUV (+pickup, van), truck, bus
});

const R_EARTH = 6371008.8;
export function projectLatLon({ lat, lon }, { lat0, lon0 }) {
  const kx = Math.cos(lat0 * Math.PI / 180) * R_EARTH * Math.PI / 180;
  const kz = R_EARTH * Math.PI / 180;
  return { x: (lon - lon0) * kx, z: -(lat - lat0) * kz };
}
export function makeAxis(cfg, graphMeta) {
  const proj = graphMeta?.projection ? { lat0: graphMeta.projection.lat0, lon0: graphMeta.projection.lon0 } : cfg.projection;
  const ax = finishAxis({ sj: projectLatLon(cfg.sanJose, proj), sf: projectLatLon(cfg.sanFrancisco, proj), mid: projectLatLon(cfg.sanMateoBridgeMid, proj) });
  ax.proj = proj;
  return ax;
}
export function finishAxis({ sj, sf, mid }) {
  const dx = sf.x - sj.x, dz = sf.z - sj.z;
  const D = Math.hypot(dx, dz) || 1;
  const u = { x: dx / D, z: dz / D };
  const v = { x: -u.z, z: u.x };
  const sideOf = (x, z) => ((x - mid.x) * v.x + (z - mid.z) * v.z > 0 ? 1 : 0);
  return { sj, sf, mid, D, u, v, sideOf };
}
/** Axis for a synthetic graph: long axis = -z (north), bay centreline x = mid. */
export function synthAxis(graph) {
  const b = graph.bounds;
  return finishAxis({ sj: { x: 0, z: b.maxZ }, sf: { x: 0, z: b.minZ }, mid: { x: 0.5 * (b.minX + b.maxX), z: 0.5 * (b.minZ + b.maxZ) } });
}

const clip = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

export function mergeCfg(options) {
  const pre = options?.region?.preset && REGION_PRESETS[options.region.preset]?.pop;   // preset population overrides (explicit options win)
  if (pre) { const o = { ...pre, ...options }; for (const k of Object.keys(pre)) if (typeof pre[k] === 'object') o[k] = { ...pre[k], ...(options[k] || {}) }; options = o; }
  const cfg = { ...POPULATION_DEFAULTS, ...options };
  for (const k of ['local', 'long', 'workStart', 'slack', 'workDuration', 'reactionTime', 'fuelFill', 'genome', 'departureCurve']) cfg[k] = { ...POPULATION_DEFAULTS[k], ...(options[k] || {}) };
  return cfg;
}

/**
 * Region restriction for homes AND workplaces (CPU runs that need real freeway density):
 *   region: { bbox: [minX, minZ, maxX, maxZ] } | { side: 0 | 1 } (bay side, 0 = Peninsula / west)
 *         | { preset: 'peninsula-101' } (west of the bay within 8 km of the bay centreline: US-101 corridor)
 *         | { preset: 'eastbay-880' } | (x, z) => boolean
 */
/**
 * Named Bay Area region presets (docs/regions.md). Each is a union of lat/lon boxes [latMin, latMax, lonMin, lonMax],
 * converted into the graph's own projection (axis.proj = graph.meta.projection, the frame bug of the first 'sf' box);
 * homes AND workplaces are restricted to it (routes may leave it). 'sf' / 'wsj' keep their original x/z definitions
 * (see regionFilter) and the listed boxes are only descriptive for them. 'bay' = no restriction: whole graph.
 * `pop` = optional population overrides (mergeCfg) for that preset. `centre` = the point that the sample check
 * (tools/train/regions.mjs check) expects the sampled homes to sit near. Add a region: add an entry here (unique key,
 * also used in the GPU cache key train-pop-<n>-s<seed>-<key>), then run `npm run train -- regions check`.
 */
export const REGION_PRESETS = Object.freeze({
  sf:      { name: 'San Francisco', terrain: 'steep hills, dense signal grid, 101/280/80 + Bay Bridge west approach', centre: { lat: 37.7749, lon: -122.4194 }, boxes: [[37.712, 37.838, -122.499, -122.340]] },
  pen:     { name: 'Peninsula (San Bruno to Palo Alto)', terrain: 'El Camino arterial, US-101, I-280 foothills, 92/84 approaches', centre: { lat: 37.545, lon: -122.30 }, boxes: [[37.43, 37.66, -122.46, -122.13]] },
  sj:      { name: 'Silicon Valley / San Jose', terrain: 'suburban grids, wide expressways, 101/87/880/280/85 interchanges', centre: { lat: 37.34, lon: -121.95 }, boxes: [[37.25, 37.43, -122.10, -121.80]] },
  eb:      { name: 'East Bay north (Oakland / Berkeley / Richmond edge)', terrain: 'Oakland-Berkeley hills, 580/880/24/80, MacArthur maze, Bay Bridge east approach', centre: { lat: 37.81, lon: -122.23 }, boxes: [[37.72, 37.90, -122.32, -122.14]] },
  ebs:     { name: 'East Bay south (Hayward / Fremont)', terrain: 'flat suburban grids, 880/238/92/84, San Mateo + Dumbarton east ends', centre: { lat: 37.59, lon: -122.02 }, boxes: [[37.48, 37.70, -122.14, -121.90]] },
  marin:   { name: 'Southern Marin + northern SF (Golden Gate)', terrain: 'US-101 over the Golden Gate Bridge, Waldo grade, Sausalito / Mill Valley hills, Presidio', centre: { lat: 37.86, lon: -122.49 }, boxes: [[37.83, 37.97, -122.56, -122.44], [37.77, 37.81, -122.52, -122.40]] },
  bridges: { name: 'Bridge approaches (Bay Bridge, San Mateo, Dumbarton)', terrain: 'long bridge decks, toll plazas, flat shorelines; longer local trips so many cross the water', centre: { lat: 37.62, lon: -122.20 },
             boxes: [[37.77, 37.84, -122.42, -122.26], [37.53, 37.65, -122.33, -122.06], [37.44, 37.54, -122.20, -121.99]], pop: { localShare: 0.9, local: { medianM: 9000, minM: 3000, maxM: 20000 } } },
  hills:   { name: 'Rural / hilly edges', terrain: 'Skyline/Woodside/280 Crystal Springs, Santa Cruz Mtns edge (Saratoga / Los Gatos), Oakland-Berkeley hills (24 / Skyline), east San Jose foothills', centre: { lat: 37.47, lon: -122.17 },
             boxes: [[37.38, 37.53, -122.40, -122.22], [37.25, 37.30, -122.08, -121.93], [37.80, 37.90, -122.25, -122.16], [37.30, 37.42, -121.84, -121.80]] },
  wsj:     { name: 'West San Jose / Cupertino (old pre-2026-10-24 sf box)', terrain: 'suburban, 280/85', centre: { lat: 37.29, lon: -121.97 }, boxes: [[37.223, 37.353, -122.044, -121.884]] },
  bay:     { name: 'Whole Bay (every region, commuters cross between them)', terrain: 'all of the above', centre: null, boxes: null },
});
/** Region keys used by the multi-region eval breakdown (first matching box wins; 'other' = none). */
export const BREAKDOWN_REGIONS = ['sf', 'marin', 'eb', 'bridges', 'pen', 'ebs', 'hills', 'sj'];
/** (x, z) -> preset key of the first BREAKDOWN_REGIONS box containing it, or 'other'. */
export function regionClassifier(proj = POPULATION_DEFAULTS.projection, keys = BREAKDOWN_REGIONS) {
  const fs = keys.map((k) => [k, latLonBoxesFilter(REGION_PRESETS[k].boxes, proj)]);
  return (x, z) => { for (const [k, f] of fs) if (f(x, z)) return k; return 'other'; };
}
function latLonBoxesFilter(boxes, proj) {
  const bb = boxes.map(([la0, la1, lo0, lo1]) => { const a = projectLatLon({ lat: la1, lon: lo0 }, proj), b = projectLatLon({ lat: la0, lon: lo1 }, proj); return [a.x, a.z, b.x, b.z]; });
  return (x, z) => { for (const b of bb) if (x >= b[0] && x <= b[2] && z >= b[1] && z <= b[3]) return true; return false; };
}

export function regionFilter(region, axis) {
  if (!region) return null;
  if (region.preset === 'bay') return null;   // whole Bay: no restriction
  if (typeof region === 'function') return region;
  if (region.bbox) { const [a, b, c, d] = region.bbox; return (x, z) => x >= a && x <= c && z >= b && z <= d; }
  const off = (x, z) => (x - axis.mid.x) * axis.v.x + (z - axis.mid.z) * axis.v.z; // signed distance from the bay centreline
  if (region.side != null) return (x, z) => axis.sideOf(x, z) === region.side;
  if (region.preset === 'peninsula-101') return (x, z) => axis.sideOf(x, z) === 0 && Math.abs(off(x, z)) < 8000;
  if (region.preset === 'sf') return (x, z) => Math.abs(x - axis.sf.x) < 7000 && Math.abs(z - axis.sf.z) < 7000; // San Francisco city: 14 km box around cfg.sanFrancisco in the graph's own projection (the 2026-10-21 bbox used the wrong frame and was west San Jose/Cupertino)
  if (region.preset === 'wsj') return (x, z) => x >= 9382 && x <= 23487 && z >= 21865 && z <= 36321; // west San Jose / Cupertino: the old (pre-2026-10-24) 'sf' box, kept as extra training variety
  if (region.preset === 'eastbay-880') return (x, z) => axis.sideOf(x, z) === 1 && Math.abs(off(x, z)) < 8000;
  const P = REGION_PRESETS[region.preset];
  if (P && P.boxes) return latLonBoxesFilter(P.boxes, axis.proj || POPULATION_DEFAULTS.projection);
  throw new Error('unknown region ' + JSON.stringify(region));
}

/** Shared per-graph context (pools, router, axis). */
export function _context(graph, cfg) {
  const scc = largestSCC(graph);
  const homeRanks = new Set(cfg.homeClasses.map(c => HIGHWAY_RANK[c]));
  const axis = cfg.axis || makeAxis(cfg, graph.meta);
  const inRegion = regionFilter(cfg.region, axis);
  const homes = [], all = [];
  for (let i = 0; i < graph.nodeCount; i++) {
    if (!scc[i]) continue;
    if (inRegion && !inRegion(graph.nx[i], graph.nz[i])) continue;
    all.push(i);
    if (homeRanks.has(graph.nodeLowRank[i])) homes.push(i);
  }
  if (all.length < 2) throw new Error('graph has too few routable nodes');
  return {
    scc, all, homes: homes.length ? homes : all,
    axis,
    route: cfg.router || makeRouter(graph, { weight: cfg.astarWeight, cacheMax: 0 }),
    typeCum: prefixSums(VEHICLE_TYPE_WEIGHTS), typeCumRealism: prefixSums(REALISM_MIX.vehicleWeights),
    base: cfg.baseGenome ? makeGenome(cfg.baseGenome.delta || cfg.baseGenome) : makeGenome(),
    inScc: inRegion ? (k) => scc[k] === 1 && inRegion(graph.nx[k], graph.nz[k]) : (k) => scc[k] === 1,
  };
}

/**
 * Generate one commuter (pure function of (cfg.seed, id)). Returns a plain record
 * with routeOut/routeBack as Int32Arrays and the delta as Float32Array(64).
 */
export function generateCommuter(graph, cfg, ctx, id) {
  const rng = makeRng(hashSeed(`${cfg.seed}:${id}`));
  const R = () => rng.random(); R.gauss = () => rng.gaussian();
  const { nx, nz } = graph, { sj, D, u, v, mid, sideOf } = ctx.axis;
  const routeLength = (r) => { let L = 0; for (let i = 0; i < r.length; i++) L += graph.edges[r[i]].length; return L; };
  const realism = cfg.mix === 'realism';
  let isLocal = rng.random() < cfg.localShare, rBin = null;
  if (realism) { // draw the straight-line length bin; short trips go in a random direction
    let q = rng.random(); for (const b of REALISM_MIX.lengthBins) { if (q < b[2]) { rBin = b; break; } q -= b[2]; }
    rBin = rBin || REALISM_MIX.lengthBins[0]; isLocal = rBin[1] <= 15;
  }
  let home = ctx.homes[rng.int(ctx.homes.length)];
  let best = null; // {work, route, L, d, ok}
  let attempts = 0, homeTries = 0;
  while (true) {
    for (let a = 0; a < cfg.maxAttempts; a++) {
      attempts++;
      let d, cx, cz;
      const hx = nx[home], hz = nz[home];
      if (isLocal) {
        d = realism ? rng.range(rBin[0], rBin[1]) * 1000 : clip(Math.exp(Math.log(cfg.local.medianM) + cfg.local.sigma * rng.gaussian()), cfg.local.minM, cfg.local.maxM);
        const th = rng.random() * 2 * Math.PI;
        cx = hx + d * Math.cos(th); cz = hz + d * Math.sin(th);
      } else {
        d = realism ? rng.range(rBin[0], rBin[1]) * 1000 : rng.range(cfg.long.minM, cfg.long.maxM);
        const t = (hx - sj.x) * u.x + (hz - sj.z) * u.z;
        const roomSF = D - t, roomSJ = t;
        let s;
        if (d <= roomSF && d <= roomSJ) s = rng.random() * (roomSF + roomSJ) < roomSF ? 1 : -1;
        else if (d <= roomSF) s = 1; else if (d <= roomSJ) s = -1;
        else s = roomSF >= roomSJ ? 1 : -1;
        const jit = rng.gaussian() * cfg.long.lateralSigmaM;
        cx = hx + s * d * u.x + jit * v.x; cz = hz + s * d * u.z + jit * v.z;
        if (rng.random() < (realism ? REALISM_MIX.pCross : cfg.long.pCross)) { const k = 2 * ((cx - mid.x) * v.x + (cz - mid.z) * v.z); cx -= k * v.x; cz -= k * v.z; }
      }
      const w = graph.nearestNode(cx, cz, ctx.inScc);
      if (w < 0 || w === home) continue;
      if (Math.hypot(nx[w] - cx, nz[w] - cz) > cfg.snapTolerance(d)) continue;   // in the bay / off the map
      const r = ctx.route(home, w);
      if (!r || r.length === 0) continue;                                          // unreachable
      const L = routeLength(r);
      const ok = L >= cfg.minRouteRatio * d;
      if (!best || (ok && !best.ok) || (ok === best.ok && L / d > best.L / best.d)) best = { work: w, route: r, L, d, ok };
      if (ok) break;
    }
    if (best) break;
    if (++homeTries > 3) {
      // the intended trip does not fit this graph (e.g. a small test graph): any reachable node
      for (let k = 0; k < 50 && !best; k++) {
        const w = ctx.all[rng.int(ctx.all.length)]; if (w === home) continue;
        const r = ctx.route(home, w); if (r && r.length) best = { work: w, route: r, L: routeLength(r), d: routeLength(r), ok: false };
      }
      if (best) break;
      if (homeTries > 20) throw new Error(`commuter ${id}: no workplace found`);
    }
    home = ctx.homes[rng.int(ctx.homes.length)];
  }
  const { work, route: routeOut, L, d, ok } = best;
  const routeBack = cfg.returnTrips ? ctx.route(work, home) : new Int32Array(0);
  // timing
  let T = 0; for (let i = 0; i < routeOut.length; i++) { const e = graph.edges[routeOut[i]]; T += e.length / (e.speedLimit * cfg.travelSpeedFactor); }
  const slack = rng.range(cfg.slack.min, cfg.slack.max);
  let workStart, depart;
  if (cfg.departures === 'curve') {
    const dc = cfg.departureCurve; let q = rng.random() * dc.share.reduce((a, b) => a + b, 0), k = 0;
    while (k < dc.share.length - 1 && q >= dc.share[k]) { q -= dc.share[k]; k++; }
    depart = dc.start + (k + rng.random()) * dc.bin;
    workStart = depart + slack * T;
  } else {
    workStart = clip(cfg.workStart.mean + rng.gaussian() * cfg.workStart.sd, cfg.workStart.min, cfg.workStart.max);
    depart = workStart - slack * T;
  }
  const returnDepart = workStart + Math.max(4 * H, cfg.workDuration.mean + rng.gaussian() * cfg.workDuration.sd);
  // vehicle, fuel, driver
  const vehicleType = rng.weightedIndex(realism ? ctx.typeCumRealism : ctx.typeCum);
  const fuelFill = rng.range(cfg.fuelFill.min, cfg.fuelFill.max);
  const rc = cfg.reactionTime, sig = Math.log(rc.p95 / rc.median) / 1.6449;
  const reactionTime = clip(rc.median * Math.exp(sig * rng.gaussian()), rc.min, rc.max);
  const g = { delta: deltaFor(cfg, ctx.base, id, reactionTime) };
  const sh = sideOf(nx[home], nz[home]), sw = sideOf(nx[work], nz[work]);
  return {
    id, home, work, kind: isLocal ? 'local' : 'long', d, L, lengthOk: ok, attempts, routeOut, routeBack: routeBack || new Int32Array(0),
    expectedTravel: T, workStart, slack, depart, returnDepart, vehicleType, fuelFill, reactionTime, delta: g.delta,
    side: [sh, sw], crossesBay: sh !== sw,
  };
}

/**
 * The agent's brain delta (base agent + mutation, src/brain/evolve.js), a pure function of
 * (seed, id) with its own RNG stream, so streaming / lazy consumers can rebuild it at departure.
 */
export function deltaFor(cfg, base, id, reactionTime, out = null) {
  const rng = makeRng(hashSeed(`${cfg.seed}:delta:${id}`));
  const R = () => rng.random(); R.gauss = () => rng.gaussian();
  const g = mutate(base, cfg.genome.sigma, R, { weightSigma: cfg.genome.sigma, traitSigma: cfg.genome.traitSigma });
  g.delta[2] = reactionTime;
  if (out) { out.set(g.delta); return out; }
  return g.delta;
}

/**
 * Generate commuters [range[0], range[1]) (default all n). Synchronous.
 * Returns { commuters, routes, deltas, arrays, axis, sideOf, seed, config, stats }.
 */
export function generatePopulation(graph, options = {}) {
  const cfg = mergeCfg(options);
  const ctx = _context(graph, cfg);
  const [a, b] = options.range || [0, cfg.n];
  const recs = [];
  for (let id = a; id < b; id++) recs.push(generateCommuter(graph, cfg, ctx, id));
  return assemble(recs, cfg, ctx.axis);
}

/** Pack commuter records into the struct-of-arrays population. */
export function assemble(recs, cfg, axis) {
  const n = recs.length;
  let total = 0; for (const r of recs) total += r.routeOut.length + r.routeBack.length;
  const flat = new Int32Array(total), offset = new Uint32Array(2 * n + 1);
  const deltas = new Float32Array(n * DELTA_SIZE);
  const A = {
    home: new Int32Array(n), work: new Int32Array(n), kind: new Uint8Array(n), vehicleType: new Uint8Array(n),
    workStart: new Float32Array(n), depart: new Float32Array(n), returnDepart: new Float32Array(n), expectedTravel: new Float32Array(n),
    routeLength: new Float32Array(n), intended: new Float32Array(n), slack: new Float32Array(n), reactionTime: new Float32Array(n),
    fuelFill: new Float32Array(n), crossesBay: new Uint8Array(n),
  };
  let p = 0;
  const commuters = new Array(n);
  for (let i = 0; i < n; i++) {
    const r = recs[i];
    offset[2 * i] = p; flat.set(r.routeOut, p); p += r.routeOut.length;
    offset[2 * i + 1] = p; flat.set(r.routeBack, p); p += r.routeBack.length;
    deltas.set(r.delta, i * DELTA_SIZE);
    A.home[i] = r.home; A.work[i] = r.work; A.kind[i] = r.kind === 'local' ? 0 : 1; A.vehicleType[i] = r.vehicleType;
    A.workStart[i] = r.workStart; A.depart[i] = r.depart; A.returnDepart[i] = r.returnDepart; A.expectedTravel[i] = r.expectedTravel;
    A.routeLength[i] = r.L; A.intended[i] = r.d; A.slack[i] = r.slack; A.reactionTime[i] = r.reactionTime; A.fuelFill[i] = r.fuelFill; A.crossesBay[i] = r.crossesBay ? 1 : 0;
  }
  offset[2 * n] = p;
  for (let i = 0; i < n; i++) {
    const r = recs[i];
    const routeOut = flat.subarray(offset[2 * i], offset[2 * i + 1]), routeBack = flat.subarray(offset[2 * i + 1], offset[2 * i + 2]);
    const genome = new Genome(deltas.subarray(i * DELTA_SIZE, (i + 1) * DELTA_SIZE));
    commuters[i] = {
      id: i, home: r.home, work: r.work, kind: r.kind, d: r.d, L: r.L, lengthOk: r.lengthOk, side: r.side, crossesBay: r.crossesBay,
      routeOut, routeBack, expectedTravel: r.expectedTravel, workStart: r.workStart, slack: r.slack, depart: r.depart,
      returnDepart: cfg.returnTrips ? r.returnDepart : Infinity, vehicleType: r.vehicleType, fuelFill: r.fuelFill, reactionTime: r.reactionTime,
      genome, actualArrival: -1, lateBy: 0, earlyBy: 0,
    };
  }
  const stats = populationStats(commuters);
  stats.lengthRejectKept = recs.filter(r => !r.lengthOk).length;
  stats.meanAttempts = recs.reduce((s, r) => s + r.attempts, 0) / Math.max(1, n);
  return { commuters, routes: { flat, offset }, deltas, arrays: A, axis, sideOf: axis.sideOf, seed: cfg.seed, config: cfg, stats };
}

/** Set a trait on a genome (brain Genome: traits live in delta[0..5]). */
export function setGenomeTrait(genome, name, value) {
  const ORDER = ['aggression', 'patience', 'reactionTime', 'preferredHeadway', 'maxAccel', 'maxDecel'];
  const k = ORDER.indexOf(name);
  if (genome.delta && k >= 0) genome.delta[k] = value;
  else if (genome.traits) genome.traits[name] = value;
}

export function populationStats(commuters, binKm = 5) {
  const n = commuters.length || 1;
  let cross = 0, local = 0, longN = 0, longCross = 0, maxL = 0, sumL = 0, sumRt = 0;
  for (const c of commuters) {
    if (c.crossesBay) cross++;
    if (c.kind === 'local') local++; else { longN++; if (c.crossesBay) longCross++; }
    if (c.L > maxL) maxL = c.L; sumL += c.L; sumRt += c.reactionTime || 0;
  }
  const bins = new Array(Math.floor(maxL / (binKm * 1000)) + 1).fill(0);
  for (const c of commuters) bins[Math.floor(c.L / (binKm * 1000))]++;
  const departures = new Array(96).fill(0);
  for (const c of commuters) departures[clip(Math.floor(c.depart / 900), 0, 95)]++;
  const slackBins = new Array(6).fill(0);
  for (const c of commuters) slackBins[clip(Math.floor((c.slack - 0.8) / 0.1), 0, 5)]++;
  const localL = commuters.filter(c => c.kind === 'local').map(c => c.L).sort((a, b) => a - b);
  const q = (arr, p) => (arr.length ? arr[Math.min(arr.length - 1, Math.floor(p * arr.length))] : 0);
  return {
    n: commuters.length, crossBayShare: cross / n, longCrossBayShare: longN ? longCross / longN : 0,
    localShare: local / n, longShare: longN / n, meanKm: sumL / n / 1000,
    localMedianKm: q(localL, 0.5) / 1000, localP90Km: q(localL, 0.9) / 1000,
    slackHistogram: slackBins.map((count, i) => ({ from: +(0.8 + i * 0.1).toFixed(2), to: +(0.9 + i * 0.1).toFixed(2), count })),
    meanReactionTime: sumRt / n,
    histogram: bins.map((count, i) => ({ fromKm: i * binKm, toKm: (i + 1) * binKm, count })),
    departuresPer15min: departures.map((count, i) => ({ hour: (i * 15) / 60, count })).filter(b => b.count > 0),
  };
}

/** Flatten commuters into a trip list sorted by departure: {commuter, from, to, route, depart, leg}. */
export function tripsOf(commuters) {
  const trips = [];
  for (const c of commuters) {
    trips.push({ commuter: c.id, from: c.home, to: c.work, route: c.routeOut, depart: c.depart, leg: 0 });
    if (Number.isFinite(c.returnDepart) && c.routeBack && c.routeBack.length) trips.push({ commuter: c.id, from: c.work, to: c.home, route: c.routeBack, depart: c.returnDepart, leg: 1 });
  }
  trips.sort((a, b) => a.depart - b.depart || a.commuter - b.commuter || a.leg - b.leg);
  return trips;
}

export { VEHICLE_TYPES };

/**
 * Vehicles on the road over time for a population, from each outbound trip's departure and
 * an assumed in-vehicle duration = expectedTravel x durationFactor (expectedTravel is free flow
 * x 0.8; congestion makes real trips longer, so durationFactor ~1.2-1.5 at full scale).
 * Returns {peak, peakAt (s), at9 (vehicles at 9:00), series: [{t, n}] every `step` s, perCommuter: peak / n}.
 */
export function concurrency(commuters, { durationFactor = 1.3, step = 300, t0 = 4 * H, t1 = 12 * H } = {}) {
  const nb = Math.ceil((t1 - t0) / step) + 1, delta = new Float64Array(nb + 1);
  for (const c of commuters) {
    const a = c.depart, b = c.depart + c.expectedTravel * durationFactor;
    const ia = Math.max(0, Math.ceil((a - t0) / step)), ib = Math.min(nb, Math.ceil((b - t0) / step));
    if (ib > ia) { delta[ia]++; delta[ib]--; }
  }
  const series = []; let n = 0, peak = 0, peakAt = t0, at9 = 0;
  for (let k = 0; k < nb; k++) {
    n += delta[k]; const t = t0 + k * step; series.push({ t, n });
    if (n > peak) { peak = n; peakAt = t; }
    if (Math.abs(t - 9 * H) < step / 2) at9 = n;
  }
  return { peak, peakAt, at9, series, perCommuter: peak / Math.max(1, commuters.length) };
}

/**
 * Population size whose concurrency peaks at `targetPeak` (default realism.json: 230,000 at
 * ~8:00, 160,000 at 9:00), estimated from a sample population on the same graph.
 */
export function sizeForPeak(graph, targetPeak = 230000, options = {}, sampleN = 3000, durationFactor = 1.3) {
  const sample = generatePopulation(graph, { ...options, n: sampleN });
  const c = concurrency(sample.commuters, { durationFactor });
  return { n: Math.round(targetPeak / c.perCommuter), perCommuter: c.perCommuter, at9PerCommuter: c.at9 / sampleN, sample: c };
}
