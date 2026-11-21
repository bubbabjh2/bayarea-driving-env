#!/usr/bin/env node
// Rebuild data/controls.* (+ the junction and lane-line packs that depend on it) from the OSM controls of the
// flat-ground build (data/src/flat/controls.*) and the pinned authoritative snapshots (data/controls-src/).
//   node tools/controls/build.mjs [--osm-only] [--skip-lines] [--no-verify]      (npm run build:controls)
// Steps: unpack (if needed) -> tools/controls/merge.py -> tools/elevation/patch_points.mjs data
//        -> tools/build_junctions.mjs data/ -> tools/lines/build_lines.mjs data -> data/controls.json.gz -> verify:world
// --osm-only skips the merge (copies the OSM controls) and checks world-manifest.json "baselines.osm-only-rebuild"
// (the original OSM-only world, "baselines.osm-only", except 17 stop lines; see the manifest).
import { spawnSync } from 'node:child_process';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { gzipSync } from 'node:zlib';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const argv = process.argv.slice(2), has = (k) => argv.includes(k);
const HEAP = '--max-old-space-size=' + (process.env.HEAP_MB || 14000);
const run = (cmd, args) => {
  console.log(`\n== ${cmd} ${args.join(' ')}`);
  const r = spawnSync(cmd, args, { cwd: ROOT, stdio: 'inherit' });
  if (r.status !== 0) { console.error(`failed (${r.status}): ${cmd} ${args.join(' ')}`); process.exit(r.status || 1); }
};
const node = (script, args = [], heap = false) => run(process.execPath, [...(heap ? [HEAP] : []), script, ...args]);

const need = ['data/roads.json', 'data/roads-local.json', 'data/src/flat/controls.json'];
if (need.some((f) => !existsSync(path.join(ROOT, f)))) node('tools/osm/unpack.mjs');
run('python3', ['-I', 'tools/controls/merge.py', ...(has('--osm-only') ? ['--osm-only'] : [])]);
node('tools/elevation/patch_points.mjs', ['data']);
node('tools/build_junctions.mjs', ['data/'], true);
if (!has('--skip-lines')) node('tools/lines/build_lines.mjs', ['data'], true);
writeFileSync(path.join(ROOT, 'data/controls.json.gz'), gzipSync(readFileSync(path.join(ROOT, 'data/controls.json')), { level: 9 }));
console.log('wrote data/controls.json.gz');
if (!has('--no-verify')) node('tools/osm/verify.mjs', has('--osm-only') ? ['--baseline', 'osm-only-rebuild'] : []);
