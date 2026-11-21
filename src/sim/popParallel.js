// Node-only: generate a large population on the real two-tier graph with
// worker_threads (each worker loads the road packs itself and generates a
// contiguous id range; per-commuter seeding makes the result identical to the
// serial generatePopulation). Optional on-disk cache (src/sim/cache/).
//
//   const pop = await generatePopulationParallel(graph, { n: 100000, seed: 42 }, { workers: 12 });

import { generatePopulation, generateCommuter, assemble, POPULATION_DEFAULTS } from './population.js';

const isMain = typeof process !== 'undefined';

export async function generatePopulationParallel(graph, options = {}, { workers, dataBase, log = null } = {}) {
  const os = await import('node:os');
  const { Worker } = await import('node:worker_threads');
  const n = options.n ?? POPULATION_DEFAULTS.n;
  const W = Math.max(1, Math.min(workers ?? Math.max(1, os.cpus().length - 2), Math.ceil(n / 500)));
  if (W === 1 || options.axis || options.router) return generatePopulation(graph, options);
  const plain = { ...options }; delete plain.snapTolerance;
  const chunks = [];
  const t0 = Date.now();
  let done = 0;
  await Promise.all(Array.from({ length: W }, (_, w) => new Promise((resolve, reject) => {
    const a = Math.floor((w * n) / W), b = Math.floor(((w + 1) * n) / W);
    const worker = new Worker(new URL(import.meta.url), { workerData: { options: plain, range: [a, b], dataBase: dataBase ? String(dataBase) : null } });
    worker.on('message', (m) => {
      if (m.progress) { done += m.progress; if (log) log(`population: ${done}/${n} (${((Date.now() - t0) / 1000).toFixed(0)} s)`); return; }
      chunks[w] = m.recs; resolve();
    });
    worker.on('error', reject);
    worker.on('exit', (code) => { if (code !== 0 && !chunks[w]) reject(new Error(`population worker exited ${code}`)); });
  })));
  const recs = chunks.flat();
  const { makeAxis, mergeCfg } = await import('./population.js');
  const cfg = mergeCfg(options);
  return assemble(recs, cfg, options.axis || makeAxis(cfg, graph.meta));
}

// ---------------------------------------------------------------- worker body
if (isMain) {
  const wt = await import('node:worker_threads');
  if (!wt.isMainThread && wt.workerData && wt.workerData.range) {
    const { loadWorld } = await import('./data.js');
    const { options, range, dataBase } = wt.workerData;
    const { graph } = await loadWorld(dataBase ? { base: dataBase, stations: false } : { stations: false });
    const { mergeCfg, _context } = await import('./population.js');
    const cfg = mergeCfg(options);
    const ctx = _context(graph, cfg);
    const recs = [];
    let last = 0;
    for (let id = range[0]; id < range[1]; id++) {
      recs.push(generateCommuter(graph, cfg, ctx, id));
      if (recs.length - last >= 1000) { wt.parentPort.postMessage({ progress: recs.length - last }); last = recs.length; }
    }
    wt.parentPort.postMessage({ recs });
  }
}
