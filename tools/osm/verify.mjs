#!/usr/bin/env node
// Check world data against the original project's hashes (tools/osm/world-manifest.json).
//   node tools/osm/verify.mjs [--data data] [--flat data/osm/flat] [--strict]
// data/<file> keys are checked in --data, flat/<file> keys (the flat-ground OSM build that the elevation
// step takes as input) in --flat. Missing files are reported and skipped unless --strict.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const argv = process.argv.slice(2);
const opt = (k, d) => { const i = argv.indexOf(k); return i >= 0 ? argv[i + 1] : d; };
const dataDir = path.resolve(ROOT, opt('--data', 'data')), flatDir = path.resolve(ROOT, opt('--flat', 'data/osm/flat'));
const M = JSON.parse(fs.readFileSync(path.join(ROOT, 'tools/osm/world-manifest.json'), 'utf8')).files;
let ok = 0, bad = 0, missing = 0;
for (const [key, want] of Object.entries(M)) {
  const file = key.startsWith('flat/') ? path.join(flatDir, key.slice(5)) : path.join(dataDir, key);
  if (!fs.existsSync(file)) { console.log(`missing   ${path.relative(ROOT, file)}`); missing++; continue; }
  const h = crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
  if (h === want.sha256) { console.log(`OK        ${path.relative(ROOT, file)}`); ok++; }
  else { console.log(`MISMATCH  ${path.relative(ROOT, file)} (${fs.statSync(file).size} bytes, want ${want.bytes})`); bad++; }
}
console.log(`${ok} OK, ${bad} mismatched, ${missing} missing`);
process.exit(bad || (missing && argv.includes('--strict')) ? 1 : 0);
