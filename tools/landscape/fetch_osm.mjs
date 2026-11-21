// Fetch OSM land cover + building footprints for the landscape render assets (Overpass API, ODbL).
// Render-only data: nothing here feeds the sim. Raw responses are cached (untrusted, parsed as JSON only) in
//   data/landscape/raw/osm-landcover/<i>_<j>.json   and   data/landscape/raw/osm-buildings/<i>_<j>.json
//   node tools/landscape/fetch_osm.mjs [landcover] [--force]
//   node tools/landscape/fetch_osm.mjs buildings --bbox S,W,N,E [--name area]   (on demand, per area: OSM buildings are
//        only fetched for areas being rendered; San Francisco uses DataSF footprints, tools/landscape/fetch_sf_buildings.mjs)
import { existsSync, mkdirSync, writeFileSync, readFileSync, renameSync, readdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const RAW = path.join(ROOT, 'data/landscape/raw');
const BBOX = [37.20, -122.62, 38.02, -121.72];   // S W N E (same as the coastline query)
const MIRRORS = ['https://overpass-api.de/api/interpreter'];   // (other public mirrors were down / timing out on 2026-10-24)
const what = process.argv[2] || 'all';
const force = process.argv.includes('--force');

// pinned to the snapshot of the published build (tools/landscape/raw-lock.json) via an Overpass attic date
const ATTIC = '[date:"2026-10-25T01:32:18Z"]';
const LANDCOVER = (b) => `[out:json][timeout:600][maxsize:1073741824]${ATTIC};
(
  nwr["landuse"](${b});
  nwr["natural"~"^(wood|scrub|heath|grassland|wetland|sand|beach|bare_rock|water|scree|shingle|mud|fell|tree_row)$"](${b});
  nwr["leisure"~"^(park|golf_course|pitch|nature_reserve|garden|playground|stadium|track|recreation_ground|dog_park|common|sports_centre)$"](${b});
  nwr["aeroway"~"^(aerodrome|runway|taxiway|apron)$"](${b});
  nwr["amenity"~"^(parking|school|university|college|hospital|grave_yard)$"](${b});
  nwr["boundary"~"^(protected_area|national_park)$"](${b});
  nwr["waterway"="riverbank"](${b});
  nwr["place"="island"](${b});
  // (OSM natural=tree nodes: fetched for the first chunks only; SF street trees come from DataSF, fetch_sf_trees.mjs)
);
out geom qt;`;
const BUILDINGS = (b) => `[out:json][timeout:600][maxsize:1073741824];
(way["building"](${b}); relation["building"](${b}););
convert B ::id=id(),::geom=geom(),h=t["height"],l=t["building:levels"],b=t["building"],mh=t["min_height"],ml=t["building:min_level"];
out geom;`;

async function query(q, out, maxAttempts = 8) {
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    const url = MIRRORS[attempt % MIRRORS.length];
    try {
      const t0 = Date.now();
      const r = await fetch(url, { method: 'POST', headers: { 'user-agent': 'bayarea-driving-env/0.1 (render assets)', 'content-type': 'application/x-www-form-urlencoded' }, body: 'data=' + encodeURIComponent(q), signal: AbortSignal.timeout(900000) });
      const txt = await r.text();
      if (!r.ok || !txt.trimStart().startsWith('{')) throw new Error(`HTTP ${r.status} ${txt.slice(0, 200).replace(/\s+/g, ' ')}`);
      const j = JSON.parse(txt);                          // validate: must be JSON with an elements array
      if (!Array.isArray(j.elements)) throw new Error('no elements');
      if (j.remark && /error|timed out/i.test(j.remark)) throw new Error('remark: ' + j.remark);
      writeFileSync(out + '.tmp', txt); renameSync(out + '.tmp', out);
      console.log(`  ${path.basename(out)}: ${j.elements.length} elements, ${(txt.length / 1e6).toFixed(1)} MB, ${((Date.now() - t0) / 1000).toFixed(0)} s (${url})`);
      return true;
    } catch (e) {
      console.log(`  ${path.basename(out)} attempt ${attempt + 1} failed: ${e.message}`);
      await new Promise((r) => setTimeout(r, Math.min(240000, 30000 * (attempt + 1))));
    }
  }
  return false;
}

// a chunk that keeps timing out is split into quadrants (recursively, up to depth 3): <name>_<q>.json
async function fetchBox(make, box, out, depth = 0) {
  const b = box.map((v) => v.toFixed(5)).join(',');
  console.log(`${path.basename(out)} (${b})`);
  const ok = await query(make(b), out, depth >= 2 ? 10 : 3);
  if (ok !== false) return;
  if (depth >= 2) throw new Error('giving up on ' + out);
  const [S, W, N, E] = box, mLat = (S + N) / 2, mLon = (W + E) / 2;
  const quads = [[S, W, mLat, mLon], [S, mLon, mLat, E], [mLat, W, N, mLon], [mLat, mLon, N, E]];
  for (let k = 0; k < 4; k++) {
    const o = out.replace(/\.json$/, `_${k}.json`);
    if (existsSync(o) && !force) continue;
    await fetchBox(make, quads[k], o, depth + 1);
  }
}

async function grid(kind, nLat, nLon, make) {
  const dir = path.join(RAW, 'osm-' + kind); mkdirSync(dir, { recursive: true });
  const [S, W, N, E] = BBOX, dLat = (N - S) / nLat, dLon = (E - W) / nLon;
  for (let i = 0; i < nLat; i++) for (let j = 0; j < nLon; j++) {
    const out = path.join(dir, `${i}_${j}.json`);
    if (existsSync(out) && !force) continue;
    if (readdirSync(dir).some((f) => f.startsWith(`${i}_${j}_`)) && !force) { await fetchBox(make, [S + i * dLat, W + j * dLon, S + (i + 1) * dLat, W + (j + 1) * dLon], out, 0); continue; }
    await fetchBox(make, [S + i * dLat, W + j * dLon, S + (i + 1) * dLat, W + (j + 1) * dLon], out, 0);
  }
  writeFileSync(path.join(dir, 'SOURCE.txt'), `OpenStreetMap via Overpass API (${MIRRORS.join(', ')}), ODbL 1.0, (c) OpenStreetMap contributors.\nFetched ${new Date().toISOString()} by tools/landscape/fetch_osm.mjs, bbox ${BBOX.join(',')} in a ${nLat}x${nLon} grid.\n`);
}

const argBbox = process.argv.includes('--bbox') ? process.argv[process.argv.indexOf('--bbox') + 1].split(',').map(Number) : null;
const argName = process.argv.includes('--name') ? process.argv[process.argv.indexOf('--name') + 1].replace(/[^a-z0-9_-]/gi, '') : null;
if (what === 'landcover' || what === 'all') await grid('landcover', 3, 3, LANDCOVER);
if (what === 'buildings') {
  if (!argBbox || argBbox.length !== 4 || argBbox.some((v) => !Number.isFinite(v))) { console.error('buildings need --bbox S,W,N,E'); process.exit(2); }
  const dir = path.join(RAW, 'osm-buildings'); mkdirSync(dir, { recursive: true });
  const out = path.join(dir, `area_${argName || argBbox.map((v) => v.toFixed(3)).join('_')}.json`);
  if (!existsSync(out) || force) await fetchBox(BUILDINGS, argBbox, out, 0);
}
console.log('done');
