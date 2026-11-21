// Full elevation rebuild from a fresh clone (npm run build:elevation):
//   node tools/elevation/build_all.mjs [--from <dir with 13/ and 1m/ tiles>] [--install] [--skip-fetch] [--skip-conform]
//
//  1. fetch_dem.mjs     USGS 3DEP tiles pinned in dem-sources.json -> data/dem/raw/ (sha256-verified)
//  2. build_dem.mjs     -> data/dem/ground/{g13.f32, g1m.f32, ground.json}
//  3. unpack the committed flat-ground road snapshot data/src/flat (model structures/2) -> data/build/elev/flat/
//  4. build_heights.mjs -> data/build/elev/out/roads{,-local}.{bin,json,meta.json}   (model terrain/1)
//  5. patch_points.mjs  -> data/build/elev/out/{controls,stations}.*  (y = road node heights)
//  6. build_conform.mjs -> data/dem/ground/ground10.{f32,mask,json}   (road-conformed ground, 10 m)
//  7. build_compact.mjs -> data/dem/compact/ground20.*                 (the small committed copy)
//  8. verify.mjs        sha256 of every output vs tools/elevation/expected.sha256
// --install copies the step-4/5 outputs into data/ (roads*.bin + meta, roads*.json.gz, stations.*, controls.*).
// Note: data/controls.* in the repo is the later output of tools/build_junctions.mjs (stop lines from the junction
// build), which takes these patched controls as its input; run it after --install to get the committed controls back.
import { spawnSync } from 'node:child_process';
import { mkdirSync, copyFileSync, readFileSync, writeFileSync, readdirSync, existsSync } from 'node:fs';
import { gunzipSync, gzipSync } from 'node:zlib';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const argv = process.argv.slice(2), has = (k) => argv.includes(k);
const HEAP = '--max-old-space-size=' + (process.env.ELEV_HEAP_MB || 12000);
const run = (script, args = [], heap = true) => {
  console.log(`\n== node ${script} ${args.join(' ')}`);
  const r = spawnSync(process.execPath, [...(heap ? [HEAP] : []), path.join(ROOT, script), ...args], { cwd: ROOT, stdio: 'inherit' });
  if (r.status !== 0) { console.error(`${script} failed (${r.status})`); process.exit(r.status || 1); }
};
const B = path.join(ROOT, 'data/build/elev'), FLAT = path.join(B, 'flat'), OUT = path.join(B, 'out');

if (!has('--skip-fetch')) run('tools/elevation/fetch_dem.mjs', has('--from') ? ['--from', argv[argv.indexOf('--from') + 1]] : [], false);
run('tools/elevation/build_dem.mjs');

mkdirSync(FLAT, { recursive: true }); mkdirSync(OUT, { recursive: true });
const SNAP = path.join(ROOT, 'data/src/flat');
for (const f of readdirSync(SNAP)) {
  if (f.endsWith('.gz')) writeFileSync(path.join(FLAT, f.slice(0, -3)), gunzipSync(readFileSync(path.join(SNAP, f))));
  else copyFileSync(path.join(SNAP, f), path.join(FLAT, f));
}
console.log('unpacked data/src/flat ->', path.relative(ROOT, FLAT));
run('tools/elevation/build_heights.mjs', ['--in', FLAT, '--out', OUT]);
for (const f of ['controls.bin', 'controls.json', 'controls.meta.json', 'stations.bin', 'stations.json', 'stations.meta.json']) copyFileSync(path.join(FLAT, f), path.join(OUT, f));
run('tools/elevation/patch_points.mjs', [OUT], false);
if (!has('--skip-conform')) run('tools/elevation/build_conform.mjs', [OUT, 'data/dem/ground']);
if (existsSync(path.join(ROOT, 'data/dem/ground/ground10.f32'))) run('tools/elevation/build_compact.mjs', [], false);

if (has('--install')) {
  const D = path.join(ROOT, 'data');
  for (const t of ['roads', 'roads-local']) {
    copyFileSync(path.join(OUT, t + '.bin'), path.join(D, t + '.bin'));
    copyFileSync(path.join(OUT, t + '.meta.json'), path.join(D, t + '.meta.json'));
    writeFileSync(path.join(D, t + '.json.gz'), gzipSync(readFileSync(path.join(OUT, t + '.json')), { level: 9 }));
  }
  for (const f of ['stations.bin', 'stations.json', 'stations.meta.json', 'controls.bin', 'controls.meta.json']) copyFileSync(path.join(OUT, f), path.join(D, f));
  writeFileSync(path.join(D, 'controls.json.gz'), gzipSync(readFileSync(path.join(OUT, 'controls.json')), { level: 9 }));
  console.log('installed roads / stations / controls into data/ (now run tools/build_junctions.mjs for the final controls)');
}
run('tools/elevation/verify.mjs', [], false);
