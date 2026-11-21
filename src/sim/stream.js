// Full-scale streaming population (GPU plan: a ~250k slot pool, ~1.4M commuters streamed in
// at departure, routes computed on demand, deltas gathered into the slot at departure).
//
//   const table = await buildCommuterTable(graph, { n: 1380000, seed: 42 }, { workers: 12 });
//     -> CommuterTable: struct-of-arrays, NO routes / deltas kept (~60 B per commuter)
//   for (const id of table.window(t0, t1)) ...        ids departing in [t0, t1), by departure
//   table.deltaFor(id, out)                           Float32Array(64) brain delta, rebuilt from (seed, id)
//   table.commuter(id)                                plain commuter object (for Simulation)
//   const routes = new RouteProvider(graph)           sync A* with an LRU cache (OD pairs)
//   const pool = new RouteWorkerPool({ workers: 10 }) async: await pool.routes([[from, to], ...])
//   new Simulation(graph, { table, routeFor: (id, leg) => ..., slotPool: 250000, ... })
//
// Generation still routes every commuter once (workplace validation needs the routed length);
// the route is dropped and recomputed at departure. bannedHops are honoured by every router.

import { POPULATION_DEFAULTS, mergeCfg, _context, generateCommuter, deltaFor } from './population.js';
import { makeRouter } from './route.js';
import { makeGenome, Genome } from './traits.js';

const FIELDS = [
  ['home', Int32Array], ['work', Int32Array], ['depart', Float32Array], ['workStart', Float32Array], ['returnDepart', Float32Array],
  ['expectedTravel', Float32Array], ['routeLength', Float32Array], ['intended', Float32Array], ['slack', Float32Array],
  ['reactionTime', Float32Array], ['fuelFill', Float32Array], ['kind', Uint8Array], ['vehicleType', Uint8Array], ['crossesBay', Uint8Array],
];

export class CommuterTable {
  constructor(n, cfg) {
    this.n = n; this.cfg = cfg;
    for (const [k, T] of FIELDS) this[k] = new T(n);
    this.order = null; this._base = null;
  }
  set(id, r) {
    this.home[id] = r.home; this.work[id] = r.work; this.depart[id] = r.depart; this.workStart[id] = r.workStart; this.returnDepart[id] = r.returnDepart;
    this.expectedTravel[id] = r.expectedTravel; this.routeLength[id] = r.L; this.intended[id] = r.d; this.slack[id] = r.slack;
    this.reactionTime[id] = r.reactionTime; this.fuelFill[id] = r.fuelFill; this.kind[id] = r.kind === 'local' ? 0 : 1;
    this.vehicleType[id] = r.vehicleType; this.crossesBay[id] = r.crossesBay ? 1 : 0;
  }
  /** Sort ids by departure (call once after filling). */
  finish() {
    const ids = new Uint32Array(this.n); for (let i = 0; i < this.n; i++) ids[i] = i;
    const d = this.depart; ids.sort((a, b) => d[a] - d[b] || a - b);
    this.order = ids; this.sortedDepart = new Float32Array(this.n); for (let k = 0; k < this.n; k++) this.sortedDepart[k] = d[ids[k]];
    return this;
  }
  /** Index of the first departure >= t in `order`. */
  indexAt(t) { const a = this.sortedDepart; let lo = 0, hi = a.length; while (lo < hi) { const m = (lo + hi) >> 1; if (a[m] < t) lo = m + 1; else hi = m; } return lo; }
  /** Ids departing in [t0, t1), in departure order. */
  *window(t0, t1) { for (let k = this.indexAt(t0); k < this.n && this.sortedDepart[k] < t1; k++) yield this.order[k]; }
  deltaFor(id, out = null) { if (!this._base) this._base = makeGenome(); return deltaFor(this.cfg, this._base, id, this.reactionTime[id], out); }
  /** Commuter object as Simulation expects (no routes; genome rebuilt from (seed, id)). */
  commuter(id) {
    return { id, home: this.home[id], work: this.work[id], kind: this.kind[id] ? 'long' : 'local', L: this.routeLength[id], d: this.intended[id],
      expectedTravel: this.expectedTravel[id], workStart: this.workStart[id], slack: this.slack[id], depart: this.depart[id],
      returnDepart: this.returnDepart[id], vehicleType: this.vehicleType[id], fuelFill: this.fuelFill[id], reactionTime: this.reactionTime[id],
      crossesBay: !!this.crossesBay[id], genome: new Genome(this.deltaFor(id)), actualArrival: -1, lateBy: 0, earlyBy: 0 };
  }
  bytes() { let b = 0; for (const [k] of FIELDS) b += this[k].byteLength; return b + (this.order ? this.order.byteLength * 2 : 0); }
}

/** Build the table with worker threads (node) or in-process (browser / small n). */
export async function buildCommuterTable(graph, options = {}, { workers, dataBase = null, log = null } = {}) {
  const cfg = mergeCfg({ ...options, returnTrips: false });
  const n = cfg.n, table = new CommuterTable(n, cfg);
  const isNode = typeof process !== 'undefined' && !!process.versions?.node;
  if (!isNode || n < 4000 || options.axis || options.router) {
    const ctx = _context(graph, cfg);
    for (let id = 0; id < n; id++) table.set(id, generateCommuter(graph, cfg, ctx, id));
    return table.finish();
  }
  const os = await import('node:os');
  const { Worker } = await import('node:worker_threads');
  const W = Math.max(1, Math.min(workers ?? Math.max(1, os.cpus().length - 2), Math.ceil(n / 2000)));
  const plain = { ...options, returnTrips: false }; delete plain.snapTolerance;
  let done = 0; const t0 = Date.now();
  await Promise.all(Array.from({ length: W }, (_, w) => new Promise((resolve, reject) => {
    const a = Math.floor((w * n) / W), b = Math.floor(((w + 1) * n) / W);
    const worker = new Worker(new URL('./streamWorker.js', import.meta.url), { workerData: { task: 'commuters', options: plain, range: [a, b], dataBase: dataBase ? String(dataBase) : null } });
    worker.on('message', (m) => {
      if (m.progress) { done += m.progress; if (log) log(`commuter table: ${done}/${n} (${((Date.now() - t0) / 1000).toFixed(0)} s)`); return; }
      for (let k = 0; k < m.ids.length; k++) {
        const id = m.ids[k], f = m.f, o = k * 14;
        table.home[id] = f[o]; table.work[id] = f[o + 1]; table.depart[id] = f[o + 2]; table.workStart[id] = f[o + 3]; table.returnDepart[id] = f[o + 4];
        table.expectedTravel[id] = f[o + 5]; table.routeLength[id] = f[o + 6]; table.intended[id] = f[o + 7]; table.slack[id] = f[o + 8];
        table.reactionTime[id] = f[o + 9]; table.fuelFill[id] = f[o + 10]; table.kind[id] = f[o + 11]; table.vehicleType[id] = f[o + 12]; table.crossesBay[id] = f[o + 13];
      }
      if (m.last) resolve();
    });
    worker.on('error', reject);
    worker.on('exit', (code) => { if (code !== 0) reject(new Error(`commuter worker exited ${code}`)); });
  })));
  return table.finish();
}

/** Synchronous on-demand router with an LRU cache keyed by the OD pair. */
export class RouteProvider {
  constructor(graph, { cacheSize = 200000, weight = 1.25 } = {}) {
    this.route = makeRouter(graph, { weight, cacheMax: 0 });
    this.cache = new Map(); this.cacheSize = cacheSize; this.hits = 0; this.misses = 0;
  }
  get(from, to) {
    const key = from * 4294967296 + to;
    let r = this.cache.get(key);
    if (r !== undefined) { this.hits++; this.cache.delete(key); this.cache.set(key, r); return r; }
    this.misses++;
    r = this.route(from, to) || null;
    if (r) r = Int32Array.from(r);
    if (this.cache.size >= this.cacheSize) this.cache.delete(this.cache.keys().next().value);
    this.cache.set(key, r);
    return r;
  }
}

/**
 * Worker-thread A* pool (node). Each worker loads the road packs once.
 *   const pool = new RouteWorkerPool({ workers: 8 }); await pool.ready();
 *   const routes = await pool.routes([[from, to], ...]);   // Int32Array | null each
 *   pool.close();
 */
export class RouteWorkerPool {
  constructor({ workers = 8, dataBase = null } = {}) { this.n = workers; this.dataBase = dataBase; this.ws = []; this.next = 0; this.pending = new Map(); this.seq = 0; }
  async ready() {
    const { Worker } = await import('node:worker_threads');
    await Promise.all(Array.from({ length: this.n }, () => new Promise((resolve, reject) => {
      const w = new Worker(new URL('./streamWorker.js', import.meta.url), { workerData: { task: 'routes', dataBase: this.dataBase ? String(this.dataBase) : null } });
      w.on('message', (m) => { if (m.ready) return resolve(); const p = this.pending.get(m.seq); if (p) { this.pending.delete(m.seq); p(m.routes); } });
      w.on('error', reject);
      this.ws.push(w);
    })));
    return this;
  }
  /** Route a batch of [from, to] pairs, split over the workers. */
  async routes(pairs) {
    const per = Math.ceil(pairs.length / this.ws.length), parts = [];
    for (let k = 0; k < this.ws.length; k++) {
      const slice = pairs.slice(k * per, (k + 1) * per); if (!slice.length) continue;
      const seq = ++this.seq, w = this.ws[k];
      parts.push(new Promise((res) => { this.pending.set(seq, res); w.postMessage({ seq, pairs: slice }); }));
    }
    return (await Promise.all(parts)).flat();
  }
  close() { for (const w of this.ws) w.terminate(); this.ws = []; }
}

export { POPULATION_DEFAULTS };
