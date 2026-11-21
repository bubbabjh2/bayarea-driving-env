// Node: build a full-scale streaming CommuterTable (src/sim/stream.js buildCommuterTable, worker
// threads) and save it for the browser GPU runs (loadCommuterTable() in src/gpu/streamPack.js).
//
//   node src/gpu/tools/packTable.mjs [n=1380000] [seed=42] [workers]
//   -> src/gpu/cache/table-<n>-s<seed>.bin + .json   (~60 B per commuter, no routes)
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadWorld } from '../../sim/data.js';
import { buildCommuterTable } from '../../sim/stream.js';
import { TABLE_FIELDS } from '../streamPack.js';
import { graphSig } from '../../sim/worldSig.js';

const n = +(process.argv[2] || 1380000), seed = +(process.argv[3] || 42), workers = process.argv[4] ? +process.argv[4] : undefined;
const dir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../cache');
fs.mkdirSync(dir, { recursive: true });
const t0 = Date.now();
const { graph } = await loadWorld({ stations: false, controls: false, geom: false });
console.log(`graph ${graph.edgeCount} edges (${Date.now() - t0} ms)`);
const table = await buildCommuterTable(graph, { n, seed }, { workers, log: (s) => process.stdout.write(`\r${s}   `) });
console.log(`\ntable ${n} in ${((Date.now() - t0) / 1000).toFixed(1)} s, ${(table.bytes() / 1048576).toFixed(1)} MB`);
const meta = { format: 'gpu-table/1', n, seed, options: { n, seed }, edgeCount: graph.edgeCount, nodeCount: graph.nodeCount, graphSig: graphSig(graph), sections: {} };
const parts = []; let off = 0;
const add = (name, arr) => { const pad = (4 - (off % 4)) % 4; if (pad) { parts.push(new Uint8Array(pad)); off += pad; } meta.sections[name] = { type: arr.constructor.name, byteOffset: off, length: arr.length }; parts.push(new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength)); off += arr.byteLength; };
for (const k of TABLE_FIELDS) add(k, table[k]);
add('order', table.order); add('sortedDepart', table.sortedDepart);
const base = path.join(dir, `table-${n}-s${seed}`);
fs.writeFileSync(base + '.bin', Buffer.concat(parts.map((p) => Buffer.from(p.buffer, p.byteOffset, p.byteLength))));
fs.writeFileSync(base + '.json', JSON.stringify(meta));
console.log(`wrote ${base}.bin (${(off / 1048576).toFixed(1)} MB)`);
process.exit(0);
