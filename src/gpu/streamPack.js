// Streaming population for the browser GPU runs:
//  - loadCommuterTable(baseUrl): src/gpu/tools/packTable.mjs output -> src/sim/stream.js CommuterTable
//  - RouteWorkers: Web-Worker A* pool (src/gpu/routeWorker.js) giving GPUSim.setStreaming a
//    non-blocking routeFor(id, leg, from, to) (undefined until the route is ready) with prefetch.
import { CommuterTable } from '../sim/stream.js';
import { mergeCfg } from '../sim/population.js';
import { routingPack } from '../sim/route.js';

export const TABLE_FIELDS = ['home', 'work', 'depart', 'workStart', 'returnDepart', 'expectedTravel', 'routeLength', 'intended', 'slack',
  'reactionTime', 'fuelFill', 'kind', 'vehicleType', 'crossesBay'];

export async function loadCommuterTable(baseUrl) {
  const meta = await (await fetch(`${baseUrl}.json`)).json();
  const buf = await (await fetch(`${baseUrl}.bin`)).arrayBuffer();
  const T = new CommuterTable(meta.n, mergeCfg({ ...meta.options, returnTrips: false }));
  const view = (name) => { const s = meta.sections[name]; return new globalThis[s.type](buf, s.byteOffset, s.length); };
  for (const k of TABLE_FIELDS) T[k] = view(k);
  T.order = view('order'); T.sortedDepart = view('sortedDepart');
  T.meta = meta;
  return T;
}

export class RouteWorkers {
  /** workers: count; dataBase: URL of data/; graph: the page's world graph (preferred: the workers get a compact
   *  routing pack instead of each loading the world, ~16 MB vs ~650 MB of JS heap per worker, docs/scale-crashes.md) */
  constructor({ workers = Math.max(2, Math.min(12, (navigator.hardwareConcurrency || 8) - 2)), dataBase, cacheSize = 400000, graph = null } = {}) {
    this.n = workers; this.dataBase = String(dataBase); this.ws = []; this.rr = 0; this.pack = graph ? routingPack(graph) : null;
    this.done = new Map(); this.inflight = new Set(); this.cacheSize = cacheSize;
    this.stats = { requested: 0, computed: 0, odHits: 0, msTotal: 0 };
    this.od = new Map();   // OD cache: (from, to) -> route (shared by commuters with the same OD)
  }
  async ready() {
    await Promise.all(Array.from({ length: this.n }, () => new Promise((resolve, reject) => {
      const w = new Worker(new URL('./routeWorker.js', import.meta.url), { type: 'module' });
      w.onmessage = (m) => {
        if (m.data.ready) return resolve();
        if (m.data.error) return reject(new Error(`route worker: ${m.data.error}`));
        const { ids, keys, routes, ms } = m.data;
        this.stats.msTotal += ms;
        for (let k = 0; k < ids.length; k++) {
          const r = routes[k] || null; this.inflight.delete(ids[k]); this.done.set(ids[k], r); this.stats.computed++;
          if (this.od.size >= this.cacheSize) this.od.delete(this.od.keys().next().value);
          this.od.set(keys[k], r);
        }
      };
      w.onerror = (e) => reject(new Error(`route worker: ${e.message || e}`));
      w.onmessageerror = (e) => reject(new Error('route worker message error'));
      w.postMessage({ init: true, dataBase: this.dataBase, pack: this.pack });   // structured clone: each worker gets its own copy (outside the JS heap)
      this.ws.push(w);
    })));
    this.batch = []; return this;
  }
  /** GPUSim.setStreaming routeFor: route | null | undefined (queued). */
  routeFor = (id, leg, from, to) => {
    if (this.done.has(id)) { const r = this.done.get(id); this.done.delete(id); return r; }
    const key = from * 4294967296 + to;
    if (this.od.has(key)) { this.stats.odHits++; return this.od.get(key); }
    if (!this.inflight.has(id)) { this.inflight.add(id); this.stats.requested++; this.batch.push([id, from, to, key]); if (this.batch.length >= 64) this.flush(); }
    return undefined;
  };
  /** Queue a route ahead of its departure (does not consume a finished one). */
  prefetch(id, from, to) {
    if (this.done.has(id) || this.inflight.has(id) || this.od.has(from * 4294967296 + to)) return;
    this.inflight.add(id); this.stats.requested++; this.batch.push([id, from, to, from * 4294967296 + to]); if (this.batch.length >= 64) this.flush();
  }
  flush() {
    if (!this.batch.length) return;
    const b = this.batch; this.batch = [];
    this.ws[this.rr++ % this.ws.length].postMessage({ ids: b.map((x) => x[0]), pairs: b.map((x) => [x[1], x[2]]), keys: b.map((x) => x[3]) });
  }
  close() { for (const w of this.ws) w.terminate(); this.ws = []; }
}
