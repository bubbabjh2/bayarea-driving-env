// Node: generate a commuter population on the real two-tier graph (src/sim/popParallel.js,
// worker threads) and save it as one binary the browser GPU demo can load without A*.
//
//   node src/gpu/tools/packPopulation.mjs [n=100000] [seed=42]
//   -> src/gpu/cache/pop-<n>-s<seed>.bin + .json  (loadPopulationPack() in src/gpu/popPack.js)
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadWorld } from '../../sim/data.js';
import { generatePopulationParallel } from '../../sim/popParallel.js';
import { ARRAY_FIELDS } from '../popPack.js';
import { graphSig } from '../../sim/worldSig.js';

const n = +(process.argv[2] || 100000), seed = +(process.argv[3] || 42);
const dir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../cache');
fs.mkdirSync(dir, { recursive: true });
const t0 = Date.now();
const { graph } = await loadWorld({ stations: false });
console.log(`graph ${graph.edgeCount} edges, ${graph.nodeCount} nodes (${Date.now() - t0} ms)`);
const pop = await generatePopulationParallel(graph, { n, seed }, { log: (s) => process.stdout.write(`\r${s}   `) });
console.log(`\npopulation ${n} in ${((Date.now() - t0) / 1000).toFixed(1)} s`, JSON.stringify(pop.stats).slice(0, 300));
const parts = [], meta = { format: 'gpu-pop/1', n, seed, edgeCount: graph.edgeCount, nodeCount: graph.nodeCount, graphSig: graphSig(graph), sections: {} };
let off = 0;
const add = (name, arr) => { const pad = (4 - (off % 4)) % 4; if (pad) { parts.push(new Uint8Array(pad)); off += pad; } meta.sections[name] = { type: arr.constructor.name, byteOffset: off, length: arr.length }; parts.push(new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength)); off += arr.byteLength; };
add('routesFlat', pop.routes.flat); add('routesOffset', pop.routes.offset); add('deltas', pop.deltas);
for (const k of ARRAY_FIELDS) if (pop.arrays[k]) add(k, pop.arrays[k]);
const base = path.join(dir, `pop-${n}-s${seed}`);
fs.writeFileSync(base + '.bin', Buffer.concat(parts.map((p) => Buffer.from(p.buffer, p.byteOffset, p.byteLength))));
meta.stats = pop.stats;
fs.writeFileSync(base + '.json', JSON.stringify(meta));
console.log(`wrote ${base}.bin (${(off / 1048576).toFixed(1)} MB)`);
process.exit(0);
