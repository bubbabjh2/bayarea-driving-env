// Web Worker: refuel route plans (src/sim/fuel.js StationNetwork.plan, 100-800 ms each) off the main
// thread for GPUSim. Loads the road packs + stations itself (same station indices as the page).
import { loadWorld } from '../sim/data.js';
import { makeRouter, liteGraph } from '../sim/route.js';
import { StationNetwork } from '../sim/fuel.js';

let net = null, router = null, penalty = null;
self.onmessage = async (m) => {
  const d = m.data;
  if (d.init) {
    try {
      // init { pack, stations } (route.js routingPack + the page's station list): no world load here (a worker holding
      // the whole world graph is ~650 MB of JS heap; docs/scale-crashes.md); else the old path
      const { graph, stations } = d.pack && d.stations ? { graph: liteGraph(d.pack), stations: d.stations } : await loadWorld({ base: new URL(d.dataBase), controls: false, geom: false, junctions: false });
      router = makeRouter(graph, { weight: 1.25, cacheMax: 4000 });   // bounded: a worker isolate OOMs (~600 MB) on the full-scale morning otherwise
      net = new StationNetwork(graph, stations, router, d.cfg || {});
      self.postMessage({ ready: true, stations: stations.length });
    } catch (e) { self.postMessage({ error: String(e && e.stack || e) }); }
    return;
  }
  const t0 = performance.now();
  if (d.penalty) { penalty = new Map(); for (let k = 0; k < d.penalty.length; k++) penalty.set(d.penalty[k], d.values[k]); if (!penalty.size) penalty = null; return; }
  if (d.avoid) {   // re-route around closed edges / live traffic (sim.js _rerouteBlocked)
    let r = null; try { r = router(d.from, d.goal, d.avoid.length ? new Set(d.avoid) : null, d.usePenalty ? penalty : null); } catch (e) { r = null; }
    self.postMessage({ id: d.id, ms: performance.now() - t0, route: r ? Int32Array.from(r) : null });
    return;
  }
  let p = null;
  try { p = net.plan(d.from, d.goal); } catch (e) { p = null; }
  self.postMessage({ id: d.id, ms: performance.now() - t0, plan: p ? { edges: p.edges, station: p.station.idx, stopIndex: p.stopIndex, detourTime: p.detourTime } : null });
};
