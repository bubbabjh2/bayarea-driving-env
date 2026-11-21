// Land-cover albedo for the landscape (render only): OSM land use / natural / leisure / aeroway polygons,
// building footprints (as a roof tint) and the coastline water mask -> clay-palette colour rasters.
//   node tools/landscape/build_landcover.mjs [--only tx,tz] [--no-global]
// In:  data/landscape/raw/osm-landcover/*.json, data/landscape/raw/osm-buildings/*.json (tools/landscape/fetch_osm.mjs),
//      data/water.json, data/landscape/terrain.bin.gz (build_terrain.mjs), data/roads-local.json (urban density only)
// Out: data/landscape/lc/<tx>_<tz>.png   detail tiles: 3.2 km of the 32x32 grid over the stream root, 512 px
//                                          (8 px gutter each side, 6.45 m/px), RGB = albedo (sRGB), A = water coverage
//      data/landscape/lc-global.png       4096^2 (25 m/px) over the whole root, same channels
//      data/landscape/landcover.json      tile list + palette + stats
// Shapes are drawn in priority order with the even-odd rule, 2x2 supersampled (anti-aliased edges).
import { readFileSync, writeFileSync, readdirSync, mkdirSync, existsSync } from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { scanFill, lineToRings } from './raster.mjs';
import { encodePNG } from './png.mjs';
import { ROOT_DIR, OUT_DIR, ROOT, toXZ, log } from './common.mjs';
import { loadRoadClearance } from './roads.mjs';
import { forEachFootprint } from './footprints.mjs';

const argv = process.argv.slice(2);
const only = argv.includes('--only') ? argv[argv.indexOf('--only') + 1].split(',').map(Number) : null;
const TILE = 3200, NT = ROOT.size / TILE, R = 512, G = 8, PX = TILE / (R - 2 * G), SS = 2, R2 = R * SS, PX2 = PX / SS;
const LC_DIR = path.join(OUT_DIR, 'lc');
mkdirSync(LC_DIR, { recursive: true });
const TR_DIR = path.join(OUT_DIR, 'trees');
mkdirSync(TR_DIR, { recursive: true });

// ------------------------------------------------------------------ palette (sRGB), 4 variants per class
const hex = (h) => [(h >> 16) & 255, (h >> 8) & 255, h & 255];
const C = {
  NATURAL: 0, WATER: 1, GRASS: 2, PARK: 3, FOREST: 4, SCRUB: 5, GOLF: 6, PITCH: 7, CEMETERY: 8, FARM: 9, ORCHARD: 10,
  RESIDENTIAL: 11, COMMERCIAL: 12, INDUSTRIAL: 13, INSTITUTION: 14, PARKING: 15, SAND: 16, ROCK: 17, WETLAND: 18, MUD: 19,
  SALTPOND: 20, AERODROME: 21, APRON: 22, TAXIWAY: 23, RUNWAY: 24, PROTECTED: 25, URBAN: 26, BUILDING: 27, TRACK: 28, CONSTRUCTION: 29,
};
const PALETTE = {
  [C.NATURAL]: [0xbdb784], [C.WATER]: [0x7fb2bd], [C.GRASS]: [0xaabd7a, 0xa5ba74, 0xb0c180, 0xa2b872], [C.PARK]: [0x93bd6c, 0x8db866, 0x98c172, 0x8fba6a],
  [C.FOREST]: [0x6f9b5f, 0x6a965a, 0x749f63, 0x679358], [C.SCRUB]: [0x9cab6c, 0x98a868, 0xa1ae70, 0x96a667], [C.GOLF]: [0x8fc56e, 0x8ac169, 0x93c873, 0x8cc36c],
  [C.PITCH]: [0x86bd69], [C.CEMETERY]: [0xa3bc86], [C.FARM]: [0xcfc78f, 0xc9c486, 0xd3c993, 0xc6c389], [C.ORCHARD]: [0xabb873, 0xa6b56f, 0xb0bb78, 0xa3b26d],
  [C.RESIDENTIAL]: [0xd3cbbc, 0xd1c9ba, 0xd5cdbf, 0xcfc8b9], [C.COMMERCIAL]: [0xd8c7bc, 0xd5c4b9, 0xdacabf, 0xd3c3b8], [C.INDUSTRIAL]: [0xc8c6cc, 0xc4c2c8, 0xcbc9ce, 0xc2c1c7],
  [C.INSTITUTION]: [0xe2d6bc], [C.PARKING]: [0xc8c5c1], [C.SAND]: [0xefe0b5], [C.ROCK]: [0xcdbfab], [C.WETLAND]: [0x9fb38f, 0x9ab08a, 0xa4b694, 0x98ae88],
  [C.MUD]: [0xc2b598], [C.SALTPOND]: [0xd8aa9e, 0xbba6bb, 0xa9c8bc, 0xcfba8c], [C.AERODROME]: [0xd6d6c6], [C.APRON]: [0xbcbcc1], [C.TAXIWAY]: [0xadaeb3],
  [C.RUNWAY]: [0x9a9ba2], [C.PROTECTED]: [0xb3b67f], [C.URBAN]: [0xd1c9ba], [C.BUILDING]: [0xc4beb5], [C.TRACK]: [0xc99a78], [C.CONSTRUCTION]: [0xd4c9b5],
};
// building roofs tint the class under them (urban fabric texture from altitude); everything else uses ROOF_DEFAULT
const ROOF_MUL = 0.86;
const PRIO = [C.NATURAL, C.PROTECTED, C.FARM, C.ORCHARD, C.RESIDENTIAL, C.COMMERCIAL, C.INDUSTRIAL, C.CONSTRUCTION, C.INSTITUTION, C.AERODROME,
  C.GRASS, C.PARK, C.CEMETERY, C.SCRUB, C.FOREST, C.WETLAND, C.MUD, C.SAND, C.ROCK, C.GOLF, C.PITCH, C.TRACK, C.PARKING,
  C.APRON, C.TAXIWAY, C.RUNWAY, C.WATER, C.SALTPOND];
const prioOf = new Map(PRIO.map((c, i) => [c, i]));

function classify(t) {
  const lu = t.landuse, na = t.natural, le = t.leisure, ae = t.aeroway, am = t.amenity;
  if (t.waterway === 'riverbank' || na === 'water' || lu === 'reservoir' || lu === 'basin') return C.WATER;
  if (lu === 'salt_pond') return C.SALTPOND;
  if (ae === 'runway') return C.RUNWAY; if (ae === 'taxiway') return C.TAXIWAY; if (ae === 'apron') return C.APRON; if (ae === 'aerodrome') return C.AERODROME;
  if (le === 'golf_course') return C.GOLF;
  if (le === 'pitch' || le === 'stadium') return C.PITCH;
  if (le === 'track') return C.TRACK;
  if (le === 'park' || le === 'garden' || le === 'playground' || le === 'dog_park' || le === 'common' || le === 'recreation_ground' || lu === 'recreation_ground' || lu === 'village_green' || lu === 'flowerbed') return C.PARK;
  if (lu === 'cemetery' || am === 'grave_yard') return C.CEMETERY;
  if (na === 'wood' || lu === 'forest') return C.FOREST;
  if (na === 'scrub' || na === 'heath' || na === 'fell') return C.SCRUB;
  if (na === 'grassland' || lu === 'grass' || lu === 'meadow') return C.GRASS;
  if (na === 'wetland') return C.WETLAND;
  if (na === 'mud') return C.MUD;
  if (na === 'sand' || na === 'beach' || na === 'shingle') return C.SAND;
  if (na === 'bare_rock' || na === 'scree' || lu === 'quarry' || lu === 'landfill') return C.ROCK;
  if (lu === 'farmland' || lu === 'farmyard' || lu === 'allotments' || lu === 'greenhouse_horticulture' || lu === 'plant_nursery') return C.FARM;
  if (lu === 'orchard' || lu === 'vineyard') return C.ORCHARD;
  if (lu === 'residential' || lu === 'garages') return C.RESIDENTIAL;
  if (lu === 'commercial' || lu === 'retail') return C.COMMERCIAL;
  if (lu === 'industrial' || lu === 'railway' || lu === 'port' || lu === 'military' || lu === 'depot' || lu === 'brownfield') return C.INDUSTRIAL;
  if (lu === 'construction' || lu === 'greenfield') return C.CONSTRUCTION;
  if (lu === 'education' || lu === 'religious' || lu === 'institutional' || am === 'school' || am === 'university' || am === 'college' || am === 'hospital') return C.INSTITUTION;
  if (am === 'parking') return C.PARKING;
  if (t.boundary === 'protected_area' || t.boundary === 'national_park' || le === 'nature_reserve') return C.PROTECTED;
  return -1;
}

// ------------------------------------------------------------------ geometry helpers
const projRing = (pts) => { const a = new Float64Array(pts.length * 2); for (let i = 0; i < pts.length; i++) { const [x, z] = toXZ(pts[i].lat, pts[i].lon); a[i * 2] = x; a[i * 2 + 1] = z; } return a; };
function chainRings(ways) {
  // ways: arrays of {lat,lon}; join open ways end to end into closed rings
  const key = (p) => p.lat.toFixed(7) + ',' + p.lon.toFixed(7);
  const rings = [], open = [];
  for (const w of ways) { if (!w || w.length < 2) continue; if (key(w[0]) === key(w[w.length - 1])) rings.push(w); else open.push(w.slice()); }
  while (open.length) {
    let cur = open.pop(), guard = 0;
    while (key(cur[0]) !== key(cur[cur.length - 1]) && guard++ < 10000) {
      const end = key(cur[cur.length - 1]);
      let k = open.findIndex((w) => key(w[0]) === end), rev = false;
      if (k < 0) { k = open.findIndex((w) => key(w[w.length - 1]) === end); rev = true; }
      if (k < 0) break;   // unclosed: close it straight
      const w = open.splice(k, 1)[0]; if (rev) w.reverse();
      cur = cur.concat(w.slice(1));
    }
    rings.push(cur);
  }
  return rings.filter((r) => r.length >= 3);
}
function bboxOf(rings) {
  let x0 = Infinity, z0 = Infinity, x1 = -Infinity, z1 = -Infinity;
  for (const r of rings) for (let i = 0; i < r.length; i += 2) { const x = r[i], z = r[i + 1]; if (x < x0) x0 = x; if (x > x1) x1 = x; if (z < z0) z0 = z; if (z > z1) z1 = z; }
  return [x0, z0, x1, z1];
}
function hashId(id) { let h = Math.imul((id ^ 0x9e3779b9) >>> 0, 0x85ebca6b); h ^= h >>> 13; h = Math.imul(h, 0xc2b2ae35); h ^= h >>> 16; return h >>> 0; }

// ------------------------------------------------------------------ load shapes
const shapes = [];   // {c, v, rings: Float64Array[], bb}
const tileShapes = Array.from({ length: NT * NT }, () => []);
function addShape(c, v, rings, area) {
  const bb = bboxOf(rings);
  const s = { c, v, rings, bb, p: prioOf.get(c) ?? 0, area: area ?? (bb[2] - bb[0]) * (bb[3] - bb[1]) };
  shapes.push(s);
  const g = PX * G + 1;
  const t0 = Math.max(0, Math.floor((bb[0] - g - ROOT.x0) / TILE)), t1 = Math.min(NT - 1, Math.floor((bb[2] + g - ROOT.x0) / TILE));
  const u0 = Math.max(0, Math.floor((bb[1] - g - ROOT.z0) / TILE)), u1 = Math.min(NT - 1, Math.floor((bb[3] + g - ROOT.z0) / TILE));
  for (let u = u0; u <= u1; u++) for (let t = t0; t <= t1; t++) tileShapes[u * NT + t].push(s);
}
const lcDir = path.join(OUT_DIR, 'raw/osm-landcover');
const seen = new Set();
let nTreeNodes = 0;
const osmTrees = Array.from({ length: NT * NT }, () => []);
// San Francisco street trees (DataSF inventory): [lat, lon, dbh inches, palm]
const sfTrees = Array.from({ length: NT * NT }, () => []);
const sfTreesFile = path.join(OUT_DIR, 'raw/sf-trees/trees.json');
const sfTreesAvail = existsSync(sfTreesFile);
if (sfTreesAvail) for (const [lat, lon, dbh] of JSON.parse(readFileSync(sfTreesFile, 'utf8'))) {
  const [x, z] = toXZ(lat, lon), ti = Math.floor((x - ROOT.x0) / TILE), tj = Math.floor((z - ROOT.z0) / TILE);
  if (ti >= 0 && tj >= 0 && ti < NT && tj < NT) sfTrees[tj * NT + ti].push(x, z, Math.min(8, Math.max(1.4, 1.2 + dbh * 0.12)));
}
const lcStats = {};
for (const f of readdirSync(lcDir).filter((f) => f.endsWith('.json')).sort()) {
  const j = JSON.parse(readFileSync(path.join(lcDir, f), 'utf8'));
  for (const e of j.elements) {
    const id = e.type[0] + e.id; if (seen.has(id)) continue; seen.add(id);
    const t = e.tags || {};
    if (e.type === 'node') {
      if (t.natural === 'tree' && e.lat != null) {
        nTreeNodes++;
        const [x, z] = toXZ(e.lat, e.lon), ti = Math.floor((x - ROOT.x0) / TILE), tj = Math.floor((z - ROOT.z0) / TILE);
        if (ti >= 0 && tj >= 0 && ti < NT && tj < NT && !(e.lat > 37.705 && e.lon < -122.355 && sfTreesAvail)) osmTrees[tj * NT + ti].push(x, z);
      }
      continue;
    }
    if (e.type === 'way' && t.natural === 'tree_row') continue;
    let c = classify(t);
    if (c < 0) continue;
    const v = hashId(e.id) & 3;
    let rings = [];
    if (e.type === 'way' && e.geometry) {
      if ((c === C.RUNWAY || c === C.TAXIWAY) && !(e.geometry.length > 3 && e.geometry[0].lat === e.geometry[e.geometry.length - 1].lat && e.geometry[0].lon === e.geometry[e.geometry.length - 1].lon)) {
        const w = parseFloat(t.width) || (c === C.RUNWAY ? 45 : 23);
        rings = lineToRings(projRing(e.geometry), Math.min(90, Math.max(10, w))).map((r) => Float64Array.from(r));
        for (const r of rings) addShape(c, v, [r], 1);   // one quad per segment (even-odd would cancel overlaps)
        lcStats[c] = (lcStats[c] || 0) + 1;
        continue;
      }
      if (e.geometry.length < 4) continue;
      rings = [projRing(e.geometry)];
    } else if (e.type === 'relation' && e.members) {
      const ways = e.members.filter((m) => m.type === 'way' && m.geometry && (m.role === 'outer' || m.role === 'inner' || m.role === '')).map((m) => m.geometry.filter((p) => p && p.lat != null));
      rings = chainRings(ways).map(projRing);
    }
    if (!rings.length) continue;
    // big protected areas only tint open land; skip enormous unrelated boundaries (> 4000 km^2)
    const bb = bboxOf(rings); const area = (bb[2] - bb[0]) * (bb[3] - bb[1]);
    if (c === C.PROTECTED && area > 4e9) continue;
    addShape(c, v, rings, area);
    lcStats[c] = (lcStats[c] || 0) + 1;
  }
  log('landcover', f, shapes.length, 'shapes');
}

// ------------------------------------------------------------------ building footprints (roof tint) + urban density grid
const GN = 4096, GC = ROOT.size / GN;
const built = new Float32Array(GN * GN);   // m^2 of roof per 25 m cell
const bShapes = Array.from({ length: NT * NT }, () => []);
let nB = 0;
forEachFootprint((e) => {
  const rings = [Float64Array.from(e.rings[0].flat())];
  const bb = bboxOf(rings);
  const r0 = rings[0]; let area = 0; for (let i = 0; i + 3 < r0.length; i += 2) area += r0[i] * r0[i + 3] - r0[i + 2] * r0[i + 1];
  area = Math.abs(area) / 2;
  const cx = (bb[0] + bb[2]) / 2, cz = (bb[1] + bb[3]) / 2;
  const gi = Math.floor((cx - ROOT.x0) / GC), gj = Math.floor((cz - ROOT.z0) / GC);
  if (gi >= 0 && gj >= 0 && gi < GN && gj < GN) built[gj * GN + gi] += area;
  const g2 = PX * G + 1;
  const t0 = Math.max(0, Math.floor((bb[0] - g2 - ROOT.x0) / TILE)), t1 = Math.min(NT - 1, Math.floor((bb[2] + g2 - ROOT.x0) / TILE));
  const u0 = Math.max(0, Math.floor((bb[1] - g2 - ROOT.z0) / TILE)), u1 = Math.min(NT - 1, Math.floor((bb[3] + g2 - ROOT.z0) / TILE));
  for (let u = u0; u <= u1; u++) for (let t = t0; t <= t1; t++) bShapes[u * NT + t].push(rings);
  nB++;
});
log('building footprints', nB);
// local-street length per 25 m cell (urban fabric where OSM has streets but no land-use / buildings)
const streets = new Float32Array(GN * GN);
{
  const lj = JSON.parse(readFileSync(path.join(ROOT_DIR, 'data/roads-local.json'), 'utf8'));
  const nodes = new Map(); for (const n of lj.nodes) nodes.set(n.id, n);
  const byIdx = lj.nodes;
  for (const e of lj.edges) {
    if (!['residential', 'living_street', 'unclassified', 'tertiary'].includes(e.class)) continue;
    const ns = e.nodes.map((k) => byIdx[k] || nodes.get(k)).filter(Boolean);
    for (let i = 0; i + 1 < ns.length; i++) {
      const a = ns[i], b = ns[i + 1], L = Math.hypot(b.x - a.x, b.z - a.z), steps = Math.max(1, Math.ceil(L / 10));
      for (let s = 0; s < steps; s++) {
        const x = a.x + (b.x - a.x) * (s + 0.5) / steps, z = a.z + (b.z - a.z) * (s + 0.5) / steps;
        const gi = Math.floor((x - ROOT.x0) / GC), gj = Math.floor((z - ROOT.z0) / GC);
        if (gi >= 0 && gj >= 0 && gi < GN && gj < GN) streets[gj * GN + gi] += L / steps;
      }
    }
  }
  log('local streets rasterised');
}
// box blur (radius r cells, separable, clamped)
function blur(src, r) {
  const tmp = new Float32Array(GN * GN), out = new Float32Array(GN * GN), w = 2 * r + 1;
  for (let j = 0; j < GN; j++) { let s = 0; for (let i = -r; i <= r; i++) s += src[j * GN + Math.min(GN - 1, Math.max(0, i))]; for (let i = 0; i < GN; i++) { tmp[j * GN + i] = s / w; s += src[j * GN + Math.min(GN - 1, i + r + 1)] - src[j * GN + Math.max(0, i - r)]; } }
  for (let i = 0; i < GN; i++) { let s = 0; for (let j = -r; j <= r; j++) s += tmp[Math.min(GN - 1, Math.max(0, j)) * GN + i]; for (let j = 0; j < GN; j++) { out[j * GN + i] = s / w; s += tmp[Math.min(GN - 1, j + r + 1) * GN + i] - tmp[Math.max(0, j - r) * GN + i]; } }
  return out;
}
const builtB = blur(built, 3), streetsB = blur(streets, 4);
// urban weight 0..1 per 25 m cell: >= ~12 % roof cover, or >= ~ 1 km of residential street per 10 ha
const urbanW = new Float32Array(GN * GN);
for (let k = 0; k < GN * GN; k++) {
  const roof = builtB[k] / (GC * GC), st = streetsB[k] / (GC * GC) * 10000;   // fraction, m per ha
  urbanW[k] = Math.max(Math.min(1, Math.max(0, (roof - 0.07) / 0.04)), Math.min(1, Math.max(0, (st - 80) / 25)));
}

// ------------------------------------------------------------------ terrain (height for the natural tint; water where height == 0 outside data/water.json)
const TN = 4096;
const terrain = (() => {
  const raw = zlib.gunzipSync(readFileSync(path.join(OUT_DIR, 'terrain.bin.gz')));
  const d = new Int16Array(raw.buffer, raw.byteOffset, TN * TN), h = new Int16Array(TN * TN);
  for (let j = 0; j < TN; j++) { let p = 0; for (let i = 0; i < TN; i++) { p += d[j * TN + i]; h[j * TN + i] = p; } }
  return h;
})();
const tAt = (x, z) => {
  const fi = (x - ROOT.x0) / GC - 0.5, fj = (z - ROOT.z0) / GC - 0.5;
  const i0 = Math.max(0, Math.min(TN - 2, Math.floor(fi))), j0 = Math.max(0, Math.min(TN - 2, Math.floor(fj)));
  const a = Math.max(0, Math.min(1, fi - i0)), b = Math.max(0, Math.min(1, fj - j0));
  const g = (i, j) => terrain[j * TN + i] / 10;
  return (g(i0, j0) * (1 - a) + g(i0 + 1, j0) * a) * (1 - b) + (g(i0, j0 + 1) * (1 - a) + g(i0 + 1, j0 + 1) * a) * b;
};
const gAt = (arr, x, z) => {
  const fi = (x - ROOT.x0) / GC - 0.5, fj = (z - ROOT.z0) / GC - 0.5;
  const i0 = Math.max(0, Math.min(GN - 2, Math.floor(fi))), j0 = Math.max(0, Math.min(GN - 2, Math.floor(fj)));
  const a = Math.max(0, Math.min(1, fi - i0)), b = Math.max(0, Math.min(1, fj - j0));
  return (arr[j0 * GN + i0] * (1 - a) + arr[j0 * GN + i0 + 1] * a) * (1 - b) + (arr[(j0 + 1) * GN + i0] * (1 - a) + arr[(j0 + 1) * GN + i0 + 1] * a) * b;
};
// open land colour by elevation + aspect: green lowland/creeks -> golden dry hills -> sage ridges (soft hypsometric tint)
function naturalColour(x, z) {
  const h = tAt(x, z);
  const dx = tAt(x + 30, z) - tAt(x - 30, z), dz = tAt(x, z + 30) - tAt(x, z - 30);
  const north = Math.max(-1, Math.min(1, -dz / 30));                // + on north-facing slopes (z = south)
  const slope = Math.min(1, Math.hypot(dx, dz) / 40);
  const lerp = (a, b, t) => a.map((v, i) => v + (b[i] - v) * Math.max(0, Math.min(1, t)));
  let c = lerp(hex(0xb7bd85), hex(0xcdbf88), (h - 20) / 180);        // lowland green-straw -> golden
  c = lerp(c, hex(0xb9b98a), (h - 350) / 400);                       // higher: sage
  c = lerp(c, hex(0x9fb177), north * 0.55 * slope + 0.15 * north);   // north slopes greener (oak / chaparral)
  return c;
}

// ------------------------------------------------------------------ water.json mask polygons (bay / ocean)
const wj = JSON.parse(readFileSync(path.join(ROOT_DIR, 'data/water.json'), 'utf8'));
const WB = wj.meta.bbox;
const waterPolys = wj.polygons.map((p) => { const rings = [p.outer, ...p.holes].map((r) => Float64Array.from(r.flat())); return { rings, bb: bboxOf(rings) }; });

// ------------------------------------------------------------------ tiles
const tileList = [];
const global = new Uint8Array(4096 * 4096 * 4);
const GPT = 4096 / NT;   // global px per tile (128)
const cls = new Uint8Array(R2 * R2), wat = new Uint8Array(R2 * R2), roof = new Uint8Array(R2 * R2);
const img = new Uint8Array(R * R * 4);
const t0 = Date.now();
let bytes = 0, treeBytes = 0, treeCount = 0; const treeTiles = [];
const ROAD = loadRoadClearance({ margin: 0.5 });
const hash01 = (a, b) => { let h = Math.imul((Math.floor(a) ^ 0x9e3779b9) >>> 0, 0x85ebca6b) ^ Math.imul((Math.floor(b) + 0x632be5ab) >>> 0, 0xc2b2ae35); h ^= h >>> 15; h = Math.imul(h, 0x2c1b3c6d); h ^= h >>> 12; return (h >>> 0) / 4294967296; };
for (let tz = 0; tz < NT; tz++) for (let tx = 0; tx < NT; tx++) {
  if (only && (only[0] !== tx || only[1] !== tz)) continue;
  const ox = ROOT.x0 + tx * TILE - G * PX, oz = ROOT.z0 + tz * TILE - G * PX;   // pixel (0,0) corner, 2x grid spacing PX2
  const toPx = (r) => { const a = new Float64Array(r.length); for (let i = 0; i < r.length; i += 2) { a[i] = (r[i] - ox) / PX2; a[i + 1] = (r[i + 1] - oz) / PX2; } return a; };
  cls.fill(C.NATURAL); wat.fill(0); roof.fill(0);
  const vbuf = new Uint8Array(R2 * R2);
  // water: polygons inside data/water.json's bbox, terrain height == 0 outside it
  const inWB = (x, z) => x > WB.minX + 100 && x < WB.maxX - 100 && z > WB.minZ + 100 && z < WB.maxZ - 100;
  for (const p of waterPolys) {
    if (p.bb[2] < ox || p.bb[0] > ox + R2 * PX2 || p.bb[3] < oz || p.bb[1] > oz + R2 * PX2) continue;
    scanFill(p.rings.map(toPx), R2, R2, (row, a, b) => { for (let i = a; i < b; i++) wat[row * R2 + i] ^= 1; });
  }
  for (let j = 0; j < R2; j++) for (let i = 0; i < R2; i++) {
    const x = ox + (i + 0.5) * PX2, z = oz + (j + 0.5) * PX2;
    if (!inWB(x, z)) wat[j * R2 + i] = tAt(x, z) < 0.2 ? 1 : 0;
  }
  // land cover by priority (bigger first within a class)
  const list = tileShapes[tz * NT + tx].slice().sort((a, b) => a.p - b.p || b.area - a.area);
  for (const s of list) {
    const c = s.c, v = s.v;
    scanFill(s.rings.map(toPx), R2, R2, (row, a, b) => {
      for (let i = a; i < b; i++) {
        const k = row * R2 + i;
        if (c === C.WATER) wat[k] = 1;
        if (c === C.PROTECTED && cls[k] !== C.NATURAL) continue;   // protected land only tints open land
        cls[k] = c; vbuf[k] = v;
      }
    });
  }
  // building roofs
  for (const rings of bShapes[tz * NT + tx]) scanFill(rings.map(toPx), R2, R2, (row, a, b) => { for (let i = a; i < b; i++) roof[row * R2 + i] = 1; });
  // compose 2x2 -> 1 (anti-aliased)
  let anyLand = false;
  const sub = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
  for (let j = 0; j < R; j++) for (let i = 0; i < R; i++) {
    const x = ox + (i + 0.5) * PX, z = oz + (j + 0.5) * PX;
    const nat = naturalColour(x, z), uw = gAt(urbanW, x, z);
    for (let b = 0; b < 2; b++) for (let a = 0; a < 2; a++) {
      const k = (j * 2 + b) * R2 + i * 2 + a;
      let c = cls[k], col, wa = 0;
      if (wat[k] && c !== C.SALTPOND && c !== C.RUNWAY && c !== C.APRON && c !== C.TAXIWAY) { col = c === C.WATER ? hex(PALETTE[C.WATER][0]) : hex(0x86b7bf); wa = 255; }
      else {
        if ((c === C.NATURAL || c === C.PROTECTED || c === C.GRASS) && uw >= 0.5) {
          col = hex(PALETTE[C.URBAN][0]);   // built-up without a land-use polygon: crisp urban fabric (no blotchy blends)
        } else if (c === C.NATURAL) col = nat;
        else { const P = PALETTE[c]; col = hex(P[vbuf[k] % P.length]); }
        if (c === C.PROTECTED) { const p = hex(PALETTE[C.PROTECTED][0]); col = col.map((v, q) => (v + p[q]) / 2); }
        if (roof[k] && c !== C.WATER) col = col.map((v, q) => v + ([203, 196, 186][q] - v) * 0.75);   // roofs: light clay tint
        anyLand = true;
      }
      const sv = sub[b * 2 + a]; sv[0] = Math.round(col[0]); sv[1] = Math.round(col[1]); sv[2] = Math.round(col[2]); sv[3] = wa;
    }
    // average of the 4 sub-samples: edge texels keep their coverage as a colour mix (the shader rebuilds crisp,
    // straight edges from it when magnified; mips / the global texture average naturally)
    const o = (j * R + i) * 4;
    for (let q = 0; q < 4; q++) img[o + q] = Math.round((sub[0][q] + sub[1][q] + sub[2][q] + sub[3][q]) / 4);
  }
  // ---- trees: OSM tree nodes + a sparse jittered scatter by land cover (never on roads / roofs / water)
  {
    const out = [];
    const cx0 = ROOT.x0 + tx * TILE, cz0 = ROOT.z0 + tz * TILE, ccx = cx0 + TILE / 2, ccz = cz0 + TILE / 2;
    const at = (x, z) => { const i = Math.floor((x - ox) / PX2), j = Math.floor((z - oz) / PX2); return (i < 0 || j < 0 || i >= R2 || j >= R2) ? -1 : j * R2 + i; };
    const ok = (k, x, z) => k >= 0 && !wat[k] && !roof[k] && !ROAD.onRoad(x, z, 1.2);
    const push = (x, z, kind, size) => { out.push(x, z, kind, size); };
    const o2 = osmTrees[tz * NT + tx];
    for (let i = 0; i < o2.length; i += 2) {
      const x = o2[i], z = o2[i + 1], k = at(x, z);
      if (ok(k, x, z)) push(x, z, 0, 2.4 + hash01(x * 7.1, z * 3.3) * 1.6);
    }
    // DataSF street trees: on the sidewalk, never on the asphalt; canopy kept off the carriageway
    const sf = sfTrees[tz * NT + tx];
    for (let i = 0; i < sf.length; i += 3) {
      const x = sf[i], z = sf[i + 1], k = at(x, z);
      if (k < 0 || wat[k] || roof[k]) continue;
      const gap = ROAD.asphaltGap(x, z);
      if (gap < 0.9) continue;
      push(x, z, 0, Math.max(1.2, Math.min(sf[i + 2], gap + 0.3)));
    }
    const CELL = 6, n = TILE / CELL, urbanDensity = o2.length > 200 ? 1 / 9000 : 1 / 3000;
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const h1 = hash01(tx * 9973 + i, tz * 7919 + j);
      const x = cx0 + (i + hash01(i * 3 + 1, j * 5 + tx)) * CELL, z = cz0 + (j + hash01(j * 7 + 2, i * 11 + tz)) * CELL;
      const k = at(x, z); if (k < 0) continue;
      const c = cls[k];
      let dens = 0, kind = 0, size = 3;
      const h2 = hash01(i * 13 + tz, j * 17 + tx);
      if (c === C.FOREST) { const elev = tAt(x, z); dens = 1 / 650; kind = h2 < 0.15 + 0.65 * Math.min(1, Math.max(0, (elev - 80) / 350)) ? 1 : 0; size = 3.0 + h2 * 2.4; }
      else if (c === C.PARK || c === C.CEMETERY) { dens = 1 / 900; kind = h2 < 0.2 ? 1 : 0; size = 2.6 + h2 * 2.0; }
      else if (c === C.GOLF) { dens = 1 / 1400; size = 3 + h2 * 1.6; }
      else if (c === C.INSTITUTION) { dens = 1 / 1600; size = 2.6 + h2 * 1.4; }
      else if (c === C.SCRUB) { dens = 1 / 750; kind = 2; size = 1.0 + h2 * 0.9; }
      else if (c === C.ORCHARD) { dens = 1 / 220; size = 1.8 + h2 * 0.6; }
      else if (c === C.RESIDENTIAL || c === C.URBAN || ((c === C.NATURAL || c === C.GRASS) && gAt(urbanW, x, z) > 0.5)) { dens = urbanDensity; size = 2.2 + h2 * 1.6; }
      else if (c === C.NATURAL || c === C.PROTECTED || c === C.GRASS) {
        // oak woodland on north-facing slopes and in creek bottoms of the open hills
        const dz = tAt(x, z + 30) - tAt(x, z - 30), north = Math.max(0, -dz / 30);
        dens = Math.min(1 / 500, north * 1 / 1600 + 1 / 15000); size = 3.0 + h2 * 2.0;
      }
      if (!dens || h1 > dens * CELL * CELL) continue;
      if (!ok(k, x, z)) continue;
      push(x, z, kind, size);
    }
    if (out.length) {
      const cnt = out.length / 4, buf = Buffer.alloc(4 + cnt * 6);
      buf.writeUInt32LE(cnt, 0);
      for (let q = 0; q < cnt; q++) {
        const o = 4 + q * 6;
        buf.writeInt16LE(Math.round((out[q * 4] - ccx) * 16), o); buf.writeInt16LE(Math.round((out[q * 4 + 1] - ccz) * 16), o + 2);
        buf[o + 4] = out[q * 4 + 2]; buf[o + 5] = Math.min(255, Math.round(out[q * 4 + 3] / 0.05));
      }
      const gz = zlib.gzipSync(buf, { level: 9 });
      writeFileSync(path.join(TR_DIR, `${tx}_${tz}.bin.gz`), gz);
      treeBytes += gz.length; treeCount += cnt; treeTiles.push([tx, tz, cnt]);
    }
  }
  // global: box-filter the tile interior (496 px -> 128 px)
  const f = (R - 2 * G) / GPT;
  for (let gj = 0; gj < GPT; gj++) for (let gi = 0; gi < GPT; gi++) {
    const acc = [0, 0, 0, 0]; let n = 0;
    for (let b = Math.floor(G + gj * f); b < Math.ceil(G + (gj + 1) * f); b++) for (let a = Math.floor(G + gi * f); a < Math.ceil(G + (gi + 1) * f); a++) {
      const o = (b * R + a) * 4; for (let q = 0; q < 4; q++) acc[q] += img[o + q]; n++;
    }
    const o = ((tz * GPT + gj) * 4096 + tx * GPT + gi) * 4;
    for (let q = 0; q < 4; q++) global[o + q] = Math.round(acc[q] / n);
  }
  const allWater = !anyLand;
  if (!allWater) {
    const png = encodePNG(img, R, R, 4, 9);
    writeFileSync(path.join(LC_DIR, `${tx}_${tz}.png`), png);
    bytes += png.length;
    tileList.push([tx, tz]);
  }
  if ((tz * NT + tx) % 64 === 63) log(`tiles ${tz * NT + tx + 1}/${NT * NT}  ${tileList.length} written  ${(bytes / 1e6).toFixed(1)} MB  ${((Date.now() - t0) / 1000).toFixed(0)} s`);
}
if (!only && !argv.includes('--no-global')) {
  const png = encodePNG(global, 4096, 4096, 4, 9);
  writeFileSync(path.join(OUT_DIR, 'lc-global.png'), png);
  bytes += png.length;
  const meta = {
    format: 'bay-landcover/1', root: ROOT, tile: TILE, grid: NT, tileSize: R, gutter: G, metresPerPx: PX, globalSize: 4096,
    channels: 'RGB albedo (sRGB, clay palette), A = water coverage (bay, ocean, lakes)',
    tiles: tileList, bytes, buildings: nB, treeNodes: nTreeNodes, trees: { tiles: treeTiles, count: treeCount, bytes: treeBytes, format: 'u32 count, then per tree i16 dx, i16 dz (1/16 m from the tile centre), u8 kind (0 broadleaf, 1 conifer, 2 bush), u8 canopy radius (0.05 m)' },
    shapesByClass: Object.fromEntries(Object.entries(lcStats).map(([k, v]) => [Object.keys(C).find((n) => C[n] === +k), v])),
    palette: Object.fromEntries(Object.entries(C).map(([n, c]) => [n, (PALETTE[c] || []).map((h) => '#' + h.toString(16).padStart(6, '0'))])),
    source: 'OpenStreetMap via Overpass (ODbL 1.0, (c) OpenStreetMap contributors); heights USGS 3DEP (public domain)',
    generated: new Date().toISOString(),
  };
  writeFileSync(path.join(OUT_DIR, 'landcover.json'), JSON.stringify(meta, null, 1));
}
log(`done: ${tileList.length} tiles, ${(bytes / 1e6).toFixed(1)} MB; trees ${treeCount} in ${treeTiles.length} tiles, ${(treeBytes / 1e6).toFixed(1)} MB; ${((Date.now() - t0) / 1000).toFixed(0)} s`);
