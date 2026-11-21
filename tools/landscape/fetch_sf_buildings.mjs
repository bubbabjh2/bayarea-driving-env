// San Francisco building footprints with lidar heights (DataSF "Building Footprints", ynuv-fyni, PDDL 1.0 public domain)
// for the landscape's whole-city buildings (render only). Paged Socrata API -> data/landscape/raw/sf-buildings/page_<n>.json
//   node tools/landscape/fetch_sf_buildings.mjs [--force]
import { existsSync, mkdirSync, writeFileSync, renameSync } from 'node:fs';
import path from 'node:path';
import { OUT_DIR } from './common.mjs';
const DIR = path.join(OUT_DIR, 'raw/sf-buildings'); mkdirSync(DIR, { recursive: true });
const force = process.argv.includes('--force');
const FIELDS = 'sf16_bldgid,gnd_mediancm,gnd_min_m,hgt_mediancm,hgt_maxcm,hgt_majoritycm,median_1st_m,peak_1st_m,shape';
const PAGE = 40000;
for (let n = 0; n < 20; n++) {
  const out = path.join(DIR, `page_${n}.json`);
  if (existsSync(out) && !force) continue;
  const url = `https://data.sf.gov/resource/ynuv-fyni.json?$select=${FIELDS}&$order=sf16_bldgid&$limit=${PAGE}&$offset=${n * PAGE}`;
  let txt = null;
  for (let a = 0; a < 6 && txt == null; a++) {
    try {
      const r = await fetch(url, { headers: { 'user-agent': 'bayarea-driving-env/0.1 (render assets)' }, signal: AbortSignal.timeout(600000) });
      const t = await r.text(); if (!r.ok) throw new Error(r.status + ' ' + t.slice(0, 200));
      const j = JSON.parse(t); if (!Array.isArray(j)) throw new Error('not an array');   // validate (untrusted JSON, parsed only)
      txt = t; console.log(`page ${n}: ${j.length} rows, ${(t.length / 1e6).toFixed(1)} MB`);
      if (!j.length) { process.exit(0); }
    } catch (e) { console.log(`page ${n} attempt ${a + 1}: ${e.message}`); await new Promise((r) => setTimeout(r, 10000 * (a + 1))); }
  }
  if (txt == null) throw new Error('failed page ' + n);
  writeFileSync(out + '.tmp', txt); renameSync(out + '.tmp', out);
  if (JSON.parse(txt).length < PAGE) break;
}
writeFileSync(path.join(DIR, 'SOURCE.txt'), 'DataSF "Building Footprints" (dataset ynuv-fyni, https://data.sfgov.org), lidar-derived heights (2010 SF lidar). Licence: ODC PDDL 1.0 (public domain).\nFetched by tools/landscape/fetch_sf_buildings.mjs on ' + new Date().toISOString() + '\n');
console.log('done');
