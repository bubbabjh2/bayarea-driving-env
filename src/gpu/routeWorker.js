// Web Worker: A* routes for the streaming GPU population (src/gpu/streamPack.js RouteWorkers).
import { loadWorld } from '../sim/data.js';
import { makeRouter } from '../sim/route.js';
// init with { pack } (route.js routingPack, typed arrays, ~16 MB): no world load in the worker; a worker that loads the
// whole world graph holds ~650 MB of JS heap, and six of them overflowed the renderer's 4 GB V8 heap cage (docs/scale-crashes.md)

let route = null;
self.onmessage = async (m) => {
  const d = m.data;
  if (d.init) {
    try {
    const graph = d.pack || (await loadWorld({ base: new URL(d.dataBase), stations: false, controls: false, geom: false })).graph;
    route = makeRouter(graph, { weight: 1.25, cacheMax: 0 });
    self.postMessage({ ready: true });
    } catch (e) { self.postMessage({ error: String(e && e.stack || e) }); }
    return;
  }
  const t0 = performance.now();
  const routes = d.pairs.map(([f, t]) => { const r = route(f, t); return r ? Int32Array.from(r) : null; });
  self.postMessage({ ids: d.ids, keys: d.keys, routes, ms: performance.now() - t0 });
};
