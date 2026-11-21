// Download the authoritative traffic-control datasets listed in tools/controls/sources.json.
//   node tools/controls/fetch.mjs            fetch every source -> data/controls-src/<id>.geojson.gz (role 'merge', committed)
//                                            or data/controls-src/verify/<id>.geojson.gz (role 'verify', gitignored); update sources.lock.json
//   node tools/controls/fetch.mjs --check    fetch and compare with sources.lock.json (exit 1 on drift); writes nothing
//   node tools/controls/fetch.mjs --only sf-signals,sj-signals
// The committed role-'merge' snapshots are what tools/controls/merge.py writes into data/, so a rebuild is reproducible
// without the network; the lock file pins download date, feature count and sha256 of the normalised GeoJSON
// (features sorted by their id field, so the hash does not depend on the server's page order).
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { gzipSync } from 'node:zlib';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const SRC = JSON.parse(readFileSync(path.join(ROOT, 'tools/controls/sources.json'), 'utf8'));
const LOCK = path.join(ROOT, 'tools/controls/sources.lock.json'), OUT = path.join(ROOT, 'data/controls-src');
const argv = process.argv.slice(2), check = argv.includes('--check');
const only = argv.includes('--only') ? argv[argv.indexOf('--only') + 1].split(',') : null;
let lock = {}; try { lock = JSON.parse(readFileSync(LOCK, 'utf8')); } catch { /* first run */ }
const UA = { 'User-Agent': 'bayarea-driving-env-data-fetch/1.0' };

async function get(url, tries = 5) {
  for (let k = 0; ; k++) {
    try { const r = await fetch(url, { headers: UA, signal: AbortSignal.timeout(300000) }); if (!r.ok) throw new Error(`HTTP ${r.status}`); return await r.json(); }
    catch (e) { if (k + 1 >= tries) throw new Error(`${url}: ${e.message}`); await new Promise((s) => setTimeout(s, 5000 * 2 ** k)); }
  }
}
async function socrata(s) {
  const feats = [];
  for (let off = 0; ; off += 50000) {
    const j = await get(`${s.url}?$limit=50000&$offset=${off}&$order=objectid`);
    feats.push(...j.features); if (j.features.length < 50000) break;
  }
  return feats;
}
async function arcgis(s) {
  const meta = await get(`${s.url}?f=json`);
  const idf = (meta.fields || []).find((f) => f.type === 'esriFieldTypeOID')?.name || 'OBJECTID';
  const ids = (await get(`${s.url}/query?where=1%3D1&returnIdsOnly=true&f=json`)).objectIds.sort((a, b) => a - b);
  const feats = [];
  for (let i = 0; i < ids.length; i += 250) {
    const sub = ids.slice(i, i + 250).join(',');
    const j = await get(`${s.url}/query?objectIds=${sub}&outFields=*&outSR=4326&f=geojson`);
    feats.push(...j.features);
  }
  return { feats, idf };
}
mkdirSync(OUT, { recursive: true });
let drift = 0;
for (const s of SRC.sources) {
  if (only && !only.includes(s.id)) continue;
  let feats, idf = 'objectid';
  if (s.type === 'socrata') feats = await socrata(s); else ({ feats, idf } = await arcgis(s));
  const key = (f) => +(f.properties?.[idf] ?? f.properties?.objectid ?? f.id ?? 0);
  feats.sort((a, b) => key(a) - key(b));
  for (const f of feats) { delete f.properties?.data_loaded_at; }   // Socrata reload stamp: changes without the data changing
  const doc = { type: 'FeatureCollection', source: s.id, features: feats };
  const body = JSON.stringify(doc), sha = createHash('sha256').update(body).digest('hex');
  const rec = { downloaded: new Date().toISOString().slice(0, 10), count: feats.length, sha256: sha, url: s.url, licence: s.licence };
  const old = lock[s.id];
  if (check) { const same = old && old.sha256 === sha; if (!same) drift++; console.log(`${s.id}: ${feats.length} features ${same ? 'match' : `DIFFER from the lock (${old ? old.count + ' features, ' + old.downloaded : 'not locked'})`}`); continue; }
  const dir = s.role === 'verify' ? path.join(OUT, 'verify') : OUT;
  mkdirSync(dir, { recursive: true });
  writeFileSync(path.join(dir, `${s.id}.geojson.gz`), gzipSync(body, { level: 9 }));
  lock[s.id] = rec; console.log(`${s.id}: ${feats.length} features, sha256 ${sha.slice(0, 12)}`);
}
if (!check) writeFileSync(LOCK, JSON.stringify(lock, null, 1) + '\n');
process.exit(check && drift ? 1 : 0);
