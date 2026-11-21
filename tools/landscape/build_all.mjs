// Landscape render assets (data/landscape/, gitignored; render only, the sim never reads them).
//   node tools/landscape/build_all.mjs [--fetch] [--compact]
// 1. raw downloads in data/landscape/raw/ (fetched when missing, or always with --fetch): OSM land cover (Overpass, attic
//    date pinned), DataSF building footprints and street trees; checked against tools/landscape/raw-lock.json (mismatch = warning)
// 2. terrain (ground10 from `npm run build:elevation`, else the committed 20 m compact grid; --compact forces it),
//    outer ring, land cover + trees, buildings, bridge structures.
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url)), ROOT = path.resolve(HERE, '../..'), RAW = path.join(ROOT, 'data/landscape/raw');
const argv = process.argv.slice(2), force = argv.includes('--fetch');
const run = (script, args = [], mem = 8000) => {
  console.log(`\n== ${script} ${args.join(' ')}`);
  const r = spawnSync(process.execPath, [`--max-old-space-size=${mem}`, path.join(HERE, script), ...args], { stdio: 'inherit', cwd: ROOT });
  if (r.status !== 0) throw new Error(`${script} failed (${r.status})`);
};
const has = (d) => existsSync(path.join(RAW, d)) && readdirSync(path.join(RAW, d)).some((f) => f.endsWith('.json'));
if (force || !has('osm-landcover')) run('fetch_osm.mjs', ['landcover', ...(force ? ['--force'] : [])]);
if (force || !has('sf-buildings')) run('fetch_sf_buildings.mjs', force ? ['--force'] : []);
if (force || !has('sf-trees')) run('fetch_sf_trees.mjs', force ? ['--force'] : []);

const lock = JSON.parse(readFileSync(path.join(HERE, 'raw-lock.json'), 'utf8'));
let ok = 0, bad = [];
for (const [rel, sha] of Object.entries(lock.files)) {
  const f = path.join(RAW, rel);
  if (!existsSync(f)) { bad.push(rel + ' (missing)'); continue; }
  if (createHash('sha256').update(readFileSync(f)).digest('hex') === sha) ok++; else bad.push(rel);
}
console.log(`raw lock: ${ok}/${Object.keys(lock.files).length} files match`);
if (bad.length) console.warn('WARNING: raw downloads differ from the pinned snapshot, the landscape will not be byte-identical:\n  ' + bad.join('\n  '));

run('build_terrain.mjs', argv.includes('--compact') ? ['--compact'] : []);
run('build_outer.mjs', [], 12000);
run('build_landcover.mjs', [], 16000);
run('build_buildings.mjs');
run('build_structures.mjs');
console.log('\nlandscape assets written to data/landscape/');
