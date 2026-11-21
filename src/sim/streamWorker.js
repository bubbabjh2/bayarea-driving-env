// Worker body for src/sim/stream.js: 'commuters' (generate a compact commuter range) or
// 'routes' (A* service). Loads the road packs itself (no lane geometry needed).
import { parentPort, workerData } from 'node:worker_threads';
import { loadWorld } from './data.js';
import { mergeCfg, _context, generateCommuter } from './population.js';
import { makeRouter } from './route.js';

const { graph } = await loadWorld(workerData.dataBase ? { base: workerData.dataBase, stations: false, controls: false, geom: false } : { stations: false, controls: false, geom: false });
if (workerData.task === 'commuters') {
  const cfg = mergeCfg(workerData.options), ctx = _context(graph, cfg);
  const [a, b] = workerData.range, CH = 2000;
  for (let s = a; s < b; s += CH) {
    const e = Math.min(b, s + CH), ids = new Uint32Array(e - s), f = new Float64Array((e - s) * 14);
    for (let id = s; id < e; id++) {
      const r = generateCommuter(graph, cfg, ctx, id), k = id - s, o = k * 14;
      ids[k] = id;
      f[o] = r.home; f[o + 1] = r.work; f[o + 2] = r.depart; f[o + 3] = r.workStart; f[o + 4] = r.returnDepart; f[o + 5] = r.expectedTravel;
      f[o + 6] = r.L; f[o + 7] = r.d; f[o + 8] = r.slack; f[o + 9] = r.reactionTime; f[o + 10] = r.fuelFill; f[o + 11] = r.kind === 'local' ? 0 : 1;
      f[o + 12] = r.vehicleType; f[o + 13] = r.crossesBay ? 1 : 0;
    }
    parentPort.postMessage({ progress: e - s });
    parentPort.postMessage({ ids, f, last: e >= b }, [ids.buffer, f.buffer]);
  }
  if (a >= b) parentPort.postMessage({ ids: new Uint32Array(0), f: new Float64Array(0), last: true });
} else {
  const route = makeRouter(graph, { weight: 1.25, cacheMax: 0 });
  parentPort.on('message', ({ seq, pairs }) => {
    const routes = pairs.map(([f, t]) => { const r = route(f, t); return r ? Int32Array.from(r) : null; });
    parentPort.postMessage({ seq, routes });
  });
  parentPort.postMessage({ ready: true });
}
