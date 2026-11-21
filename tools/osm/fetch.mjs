#!/usr/bin/env node
// Fetch the pinned OSM inputs (Overpass attic snapshots) into data/osm/raw/<name>.json.
//   node tools/osm/fetch.mjs [name ...] [--force] [--update-lock] [--sleep 30]
// Env: OVERPASS_URL overrides the endpoint in tools/osm/sources.json.
// Every response is checked against tools/osm/osm-lock.json: sha256 of the canonical element list
// (JSON.stringify(doc.elements); the "osm3s"/"generator" header varies by server and is ignored).
// The first successful fetch of a source records its hash (or use --update-lock). A mismatch is an error.
// Sequential, one request at a time, with a pause between requests (Overpass usage policy).
// Map data (c) OpenStreetMap contributors, ODbL 1.0.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const SRC = JSON.parse(fs.readFileSync(path.join(ROOT, 'tools/osm/sources.json'), 'utf8'));
const LOCK = path.join(ROOT, 'tools/osm/osm-lock.json');
const OUT = path.join(ROOT, 'data/osm/raw');
const argv = process.argv.slice(2);
const flag = (k) => argv.includes(k);
const sleepS = +(argv[argv.indexOf('--sleep') + 1] || 30);
const names = argv.filter((a, i) => !a.startsWith('--') && argv[i - 1] !== '--sleep');
const endpoint = process.env.OVERPASS_URL || SRC.endpoint;
const lock = fs.existsSync(LOCK) ? JSON.parse(fs.readFileSync(LOCK, 'utf8')) : { doc: 'sha256 of JSON.stringify(elements) per source (tools/osm/fetch.mjs)', sources: {} };
fs.mkdirSync(OUT, { recursive: true });

export function pinnedQuery(name, date) {
  const ql = fs.readFileSync(path.join(ROOT, 'tools/queries', `${name}.ql`), 'utf8');
  // insert the attic date into the first settings statement ([out:json][timeout:..];)
  return ql.replace(/^(\s*(?:\[[^\]]*\])+)\s*;/, `$1[date:"${date}"];`);
}
export function canonicalHash(doc) {
  return crypto.createHash('sha256').update(JSON.stringify(doc.elements || [])).digest('hex');
}

let failed = 0, first = true;
for (const s of SRC.sources) {
  if (names.length && !names.includes(s.name)) continue;
  const file = path.join(OUT, `${s.name}.json`);
  let doc;
  if (fs.existsSync(file) && !flag('--force')) {
    doc = JSON.parse(fs.readFileSync(file, 'utf8'));
    console.log(`${s.name}: cached ${file}`);
  } else {
    if (!first) await new Promise((r) => setTimeout(r, sleepS * 1000));
    first = false;
    const q = pinnedQuery(s.name, s.date);
    console.log(`${s.name}: POST ${endpoint} (date ${s.date})`);
    const t0 = Date.now();
    // retry with exponential backoff when the server is busy (429 / 502 / 503 / 504)
    let res, text;
    for (let attempt = 0, wait = 60; ; attempt++, wait *= 2) {
      res = await fetch(endpoint, { method: 'POST', headers: { 'User-Agent': 'bayarea-driving-env-build/1.0', 'Content-Type': 'application/x-www-form-urlencoded' }, body: 'data=' + encodeURIComponent(q) });
      text = await res.text();
      if (![429, 502, 503, 504].includes(res.status) || attempt >= 5) break;
      console.log(`${s.name}: HTTP ${res.status}, retry in ${wait} s`);
      await new Promise((r) => setTimeout(r, wait * 1000));
    }
    if (!res.ok) { console.error(`${s.name}: HTTP ${res.status} ${text.slice(0, 300)}`); failed++; continue; }
    try { doc = JSON.parse(text); } catch (e) { console.error(`${s.name}: not JSON: ${text.slice(0, 300)}`); failed++; continue; }
    if (doc.remark && /error|timed out|runtime/i.test(doc.remark)) { console.error(`${s.name}: Overpass remark: ${doc.remark}`); failed++; continue; }
    fs.writeFileSync(file + '.tmp', text); fs.renameSync(file + '.tmp', file);
    console.log(`${s.name}: ${doc.elements.length} elements, ${(text.length / 1e6).toFixed(1)} MB in ${((Date.now() - t0) / 1000).toFixed(0)} s`);
  }
  const h = canonicalHash(doc), want = lock.sources[s.name];
  if (!want || flag('--update-lock')) {
    lock.sources[s.name] = { sha256: h, elements: doc.elements.length, date: s.date, endpoint, recorded: new Date().toISOString() };
    console.log(`${s.name}: recorded sha256 ${h}`);
  } else if (want.sha256 !== h) {
    console.error(`${s.name}: sha256 MISMATCH ${h} (lock ${want.sha256}); the server returned a different snapshot`); failed++;
  } else console.log(`${s.name}: sha256 OK`);
}
fs.writeFileSync(LOCK, JSON.stringify(lock, null, 1) + '\n');
process.exit(failed ? 1 : 0);
