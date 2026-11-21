#!/usr/bin/env node
// Per-edge road surface + off-road ground class (docs/SURFACES.md).
//   node tools/surfaces/build.mjs [--data data] [--raw data/osm/raw] [--if-raw (skip quietly without the raw snapshots)]
// Inputs (read-only): data/roads.json + data/roads-local.json (readable copies, `npm run unpack`), the pinned raw
// Overpass snapshots in data/osm/raw/*.json (`node tools/osm/fetch.mjs`, sha256-locked in tools/osm/osm-lock.json;
// they carry each way's full tags incl. surface=*, bridge=movable) and data/water.json (shoreline).
// Outputs: data/surfaces.bin (2 bytes per edge, main tier edges then local tier edges = SimGraph edge order) and
// data/surfaces.meta.json (code table, layout, coverage). Deterministic: same inputs -> same bytes.
// Map data (c) OpenStreetMap contributors, ODbL 1.0.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { SURFACE_CODES, SURFACE_NAMES, osmSurfaceCode } from '../../src/sim/surface.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const argv = process.argv.slice(2);
const opt = (k, d) => { const i = argv.indexOf(k); return i >= 0 ? argv[i + 1] : d; };
const DATA = path.resolve(ROOT, opt('--data', 'data')), RAW = path.resolve(ROOT, opt('--raw', 'data/osm/raw'));
const RAW_FILES = ['roads.json', 'roads-north.json', 'local-tile1.json', 'local-tile2a.json', 'local-tile2b.json', 'local-tile3a.json',
  'local-tile3b.json', 'local-tile4a.json', 'local-tile4b.json', 'local-tile5.json', 'local-tile6.json', 'local-tile7.json'];
const C = SURFACE_CODES;
const SHORE_M = 25;            // an at-grade edge within this distance of the water polygons gets a sand / rip-rap ground class
const FREEWAY = new Set(['motorway', 'motorway_link', 'trunk', 'trunk_link']);

// ---- way tags from the raw snapshots
const wayTags = new Map();
for (const f of RAW_FILES) {
  const p = path.join(RAW, f);
  if (!fs.existsSync(p)) {
    console.error(`missing ${path.relative(ROOT, p)}: run node tools/osm/fetch.mjs (or npm run build:osm)`);
    if (argv.includes('--if-raw')) { console.error('surfaces: raw snapshots not present, keeping the committed data/surfaces.bin'); process.exit(0); }
    process.exit(2);
  }
  for (const el of JSON.parse(fs.readFileSync(p, 'utf8')).elements || []) {
    if (el.type !== 'way' || !el.tags || !el.tags.highway) continue;
    if (!wayTags.has(el.id)) wayTags.set(el.id, el.tags);
  }
}

// ---- shoreline segments in a 200 m grid
const water = JSON.parse(fs.readFileSync(path.join(DATA, 'water.json'), 'utf8'));
const CELL = 200, grid = new Map();
const key = (cx, cz) => cx * 100000 + cz;
const addRing = (ring) => {
  for (let k = 0; k < ring.length; k++) {
    const a = ring[k], b = ring[(k + 1) % ring.length];
    const x0 = Math.floor((Math.min(a[0], b[0]) - SHORE_M) / CELL), x1 = Math.floor((Math.max(a[0], b[0]) + SHORE_M) / CELL);
    const z0 = Math.floor((Math.min(a[1], b[1]) - SHORE_M) / CELL), z1 = Math.floor((Math.max(a[1], b[1]) + SHORE_M) / CELL);
    for (let cx = x0; cx <= x1; cx++) for (let cz = z0; cz <= z1; cz++) {
      const kk = key(cx, cz); let l = grid.get(kk); if (!l) grid.set(kk, (l = [])); l.push(a[0], a[1], b[0], b[1]);
    }
  }
};
for (const poly of water.polygons) { addRing(poly.outer); for (const h of poly.holes || poly.inner || []) addRing(h); }
function nearShore(x, z) {
  const l = grid.get(key(Math.floor(x / CELL), Math.floor(z / CELL)));
  if (!l) return false;
  for (let k = 0; k < l.length; k += 4) {
    const ax = l[k], az = l[k + 1], dx = l[k + 2] - ax, dz = l[k + 3] - az, L2 = dx * dx + dz * dz;
    let t = L2 > 0 ? ((x - ax) * dx + (z - az) * dz) / L2 : 0; t = t < 0 ? 0 : t > 1 ? 1 : t;
    const ex = ax + t * dx - x, ez = az + t * dz - z;
    if (ex * ex + ez * ez < SHORE_M * SHORE_M) return true;
  }
  return false;
}

// ---- per tier
const UNPAVED = new Set([C.gravel, C.compacted, C.dirt, C.grass, C.sand]);
function sideGround(e, side, surf, shore) {
  // side: 'right' | 'left' of the directed edge
  if (e.bridge || e.tunnel || surf === C.metal_grate || surf === C.wood) return surf;     // deck / tube: the structure's own surface
  if (UNPAVED.has(surf)) return surf === C.grass || surf === C.sand ? surf : C.dirt;
  if (shore) return C.sand;
  if (FREEWAY.has(e.class)) return side === 'right' ? C.grass : C.gravel;                // roadside verge / median
  const sw = e.sidewalk;
  if (sw === 'both' || sw === side) return C.curb;
  if (sw === 'no' || (sw && sw !== side)) return e.class === 'unclassified' ? C.dirt : C.grass;
  return e.class === 'unclassified' ? C.dirt : C.curb;                                   // untagged: urban curb, rural lanes dirt
}
const out = [], tiers = {};
for (const [tier, file] of [['main', 'roads.json'], ['local', 'roads-local.json']]) {
  const p = path.join(DATA, file);
  if (!fs.existsSync(p)) { console.error(`missing ${path.relative(ROOT, p)}: run npm run unpack`); process.exit(2); }
  const J = JSON.parse(fs.readFileSync(p, 'utf8'));
  const nodes = J.nodes, E = J.edges.length, buf = new Uint8Array(2 * E);
  const st = { edges: E, osmTagged: 0, defaulted: 0, wayNotInSnapshot: 0, unknownTag: {}, bySurface: {}, byGroundRight: {}, osmTagValues: {} };
  const shoreOf = (e) => {
    if (e.bridge || e.tunnel) return false;
    const ids = e.nodes; const step = Math.max(1, Math.floor(ids.length / 6));
    for (let k = 0; k < ids.length; k += step) { const n = nodes[ids[k]]; if (nearShore(n.x, n.z)) return true; }
    const n = nodes[ids[ids.length - 1]]; return nearShore(n.x, n.z);
  };
  for (let i = 0; i < E; i++) {
    const e = J.edges[i];
    const tags = wayTags.get(e.way);
    let surf = -1, src = 0;
    if (!tags) st.wayNotInSnapshot++;
    else if (tags.surface) {
      st.osmTagValues[tags.surface] = (st.osmTagValues[tags.surface] || 0) + 1;
      surf = osmSurfaceCode(tags.surface);
      if (surf >= 0) src = 1; else st.unknownTag[tags.surface] = (st.unknownTag[tags.surface] || 0) + 1;
    }
    if (surf < 0) {   // default by structure / class
      surf = tags && tags.bridge === 'movable' ? C.metal_grate : e.bridge || e.tunnel ? C.concrete : C.asphalt;
    }
    if (src === 1) st.osmTagged++; else st.defaulted++;
    const shore = shoreOf(e);
    const gR = sideGround(e, 'right', surf, shore), gL = sideGround(e, 'left', surf, shore);
    buf[2 * i] = surf | (src << 4) | ((tags ? 0 : 1) << 5);
    buf[2 * i + 1] = gR | (gL << 4);
    st.bySurface[SURFACE_NAMES[surf]] = (st.bySurface[SURFACE_NAMES[surf]] || 0) + 1;
    st.byGroundRight[SURFACE_NAMES[gR]] = (st.byGroundRight[SURFACE_NAMES[gR]] || 0) + 1;
  }
  st.osmTaggedPct = +(100 * st.osmTagged / E).toFixed(2);
  st.defaultedPct = +(100 * st.defaulted / E).toFixed(2);
  tiers[tier] = st; out.push(buf);
  console.log(`${tier}: ${E} edges, OSM surface tag ${st.osmTagged} (${st.osmTaggedPct} %), defaulted ${st.defaulted} (${st.defaultedPct} %), way not in snapshot ${st.wayNotInSnapshot}`);
}
const all = new Uint8Array(out[0].length + out[1].length); all.set(out[0], 0); all.set(out[1], out[0].length);
fs.writeFileSync(path.join(DATA, 'surfaces.bin'), all);
const meta = {
  format: 'bay-surfaces/1',
  file: 'surfaces.bin',
  edgeCount: { main: tiers.main.edges, local: tiers.local.edges, total: tiers.main.edges + tiers.local.edges },
  order: 'main-tier edges 0..main-1, then local-tier edges (= SimGraph edge index; local edge k is record main + k)',
  layout: '2 bytes per edge. byte 0: bits 0-3 surface code of the pavement, bit 4 = 1 if from the OSM surface tag (0 = default), bit 5 = 1 if the edge\'s way is not in the raw snapshot. byte 1: bits 0-3 ground code right of the edge (driver\'s right), bits 4-7 ground code left of the edge (beyond the opposing lanes on two-way roads).',
  codes: SURFACE_NAMES,
  defaults: 'untagged: asphalt; bridge / tunnel: concrete; bridge=movable: metal_grate. ground: bridge / tunnel / metal / wood = the deck surface; unpaved roads = dirt (grass, sand as tagged); within 25 m of the water polygons = sand; motorway / trunk (+ links): right grass, left gravel; sidewalk tag on that side = curb; sidewalk=no = grass (unclassified: dirt); untagged: curb (unclassified: dirt)',
  sources: { osm: 'OpenStreetMap way tags surface=*, bridge=movable (data/osm/raw, tools/osm/osm-lock.json), ODbL 1.0', water: 'data/water.json', roads: 'data/roads.json, data/roads-local.json' },
  coverage: tiers,
};
fs.writeFileSync(path.join(DATA, 'surfaces.meta.json'), JSON.stringify(meta, null, 1) + '\n');
console.log(`wrote data/surfaces.bin (${all.length} bytes), data/surfaces.meta.json`);
