#!/usr/bin/env node
// Decompress the committed data/*.gz files next to themselves (data/roads.json.gz -> data/roads.json, ...).
//   node tools/osm/unpack.mjs [--force]
// Existing outputs are kept unless --force. The JSON copies are needed by the build tools
// (build_controls / build_crossings / build_junctions / build_lines read controls.json, roads*.json).
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { fileURLToPath } from 'node:url';
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const force = process.argv.includes('--force');
for (const dir of ['data', 'data/src/flat']) {
  const d = path.join(ROOT, dir);
  if (!fs.existsSync(d)) continue;
  for (const f of fs.readdirSync(d)) {
    if (!f.endsWith('.gz')) continue;
    const out = path.join(d, f.slice(0, -3));
    if (fs.existsSync(out) && !force) { console.log(`keep ${path.relative(ROOT, out)}`); continue; }
    fs.writeFileSync(out + '.tmp', zlib.gunzipSync(fs.readFileSync(path.join(d, f))));
    fs.renameSync(out + '.tmp', out);
    console.log(`unpacked ${path.relative(ROOT, out)}`);
  }
}
