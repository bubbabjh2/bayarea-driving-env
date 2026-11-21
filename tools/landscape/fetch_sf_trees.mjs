// San Francisco street trees (DataSF "Street Tree Inventory", tkzw-k3nq, ODC PDDL 1.0) for the landscape trees (render only).
//   node tools/landscape/fetch_sf_trees.mjs  -> data/landscape/raw/sf-trees/trees.json  ([[lat, lon, dbh_in, palm 0/1], ...])
import { mkdirSync, writeFileSync, existsSync } from 'node:fs';
import path from 'node:path';
import { OUT_DIR } from './common.mjs';
const DIR = path.join(OUT_DIR, 'raw/sf-trees'); mkdirSync(DIR, { recursive: true });
const out = path.join(DIR, 'trees.json');
if (existsSync(out) && !process.argv.includes('--force')) process.exit(0);
const rows = [];
for (let off = 0; off < 400000; off += 50000) {
  const url = `https://data.sf.gov/resource/tkzw-k3nq.json?$select=latitude,longitude,dbhrange,species,legalstatus&$where=latitude IS NOT NULL&$order=treeid&$limit=50000&$offset=${off}`;
  const r = await fetch(url, { headers: { 'user-agent': 'bayarea-driving-env/0.1 (render assets)' } });
  if (!r.ok) throw new Error(r.status + ' ' + (await r.text()).slice(0, 200));
  const j = await r.json(); if (!Array.isArray(j)) throw new Error('not an array');
  for (const t of j) {
    const lat = +t.latitude, lon = +t.longitude;
    if (!(lat > 37.6 && lat < 37.85 && lon > -122.6 && lon < -122.3)) continue;
    const dbh = parseFloat(String(t.dbhrange || '').replace(/[^0-9.].*$/, '')) || 6;
    rows.push([+lat.toFixed(6), +lon.toFixed(6), dbh, /palm|phoenix|washingtonia|syagrus/i.test(t.species || '') ? 1 : 0]);
  }
  console.log('offset', off, j.length);
  if (j.length < 50000) break;
}
writeFileSync(out, JSON.stringify(rows));
writeFileSync(path.join(DIR, 'SOURCE.txt'), 'DataSF "Street Tree Inventory" (tkzw-k3nq, https://data.sfgov.org). Licence: ODC PDDL 1.0 (public domain). Fetched ' + new Date().toISOString() + '\n');
console.log('trees', rows.length);
