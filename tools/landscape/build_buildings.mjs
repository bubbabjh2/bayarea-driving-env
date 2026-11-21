// Building footprints for the landscape (render only): OSM buildings -> tiled, road-cleared footprints + heights.
//   node tools/landscape/build_buildings.mjs
// In:  data/landscape/raw/osm-buildings/*.json (fetch_osm.mjs, Overpass "convert" format), data/roads.json,
//      data/roads-local.json (clearance only: buildings never overlap road ribbons, parking, curbs or sidewalks)
// Out: data/landscape/bld/<tx>_<tz>.bin.gz   1.6 km tiles (64x64 grid over the stream root)
//        u32 count, then per building: u16 n, u16 height (dm), u8 style, u8 flags, then n x (i16 dx, i16 dz) in 1/32 m,
//        first vertex relative to the tile centre, the rest as deltas from the previous vertex (CCW seen from above)
//      data/landscape/buildings.json   tile list + stats
// Heights: OSM height / building:levels; else by type, footprint and place (SF 3 storeys, suburbs 1-2,
// downtown SF / Oakland / San Jose cores taller).
import { readFileSync, writeFileSync, readdirSync, mkdirSync, existsSync } from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { ROOT_DIR, OUT_DIR, ROOT, toXZ, log } from './common.mjs';
import { loadRoadClearance } from './roads.mjs';
import { forEachFootprint } from './footprints.mjs';

const TILE = 1600, NT = ROOT.size / TILE, Q = 32;
const DIR = path.join(OUT_DIR, 'bld'); mkdirSync(DIR, { recursive: true });
const hash = (n) => { let h = Math.imul((n ^ 0x9e3779b9) >>> 0, 0x85ebca6b); h ^= h >>> 13; h = Math.imul(h, 0xc2b2ae35); h ^= h >>> 16; return (h >>> 0) / 4294967296; };

// ------------------------------------------------------------------ road clearance (shared with the tree scatter)
const { clear } = loadRoadClearance();

// ------------------------------------------------------------------ height rules
const CORES = [[37.7915, -122.3990, 1100, 6, 14], [37.8045, -122.2711, 650, 4, 10], [37.3337, -121.8907, 750, 4, 9]].map(([la, lo, r, a, b]) => { const [x, z] = toXZ(la, lo); return { x, z, r, a, b }; });
const SF = (x, z) => { const [x0, z0] = toXZ(37.81, -122.515), [x1, z1] = toXZ(37.705, -122.355); return x > x0 && x < x1 && z > z0 && z < z1; };
function parseLen(s) {
  if (!s) return NaN;
  const m = String(s).trim().match(/^([0-9.]+)\s*(m|ft|')?/i); if (!m) return NaN;
  const v = parseFloat(m[1]); return /ft|'/i.test(m[2] || '') ? v * 0.3048 : v;
}
const SMALL = new Set(['garage', 'garages', 'shed', 'carport', 'hut', 'roof', 'kiosk', 'cabin', 'greenhouse']);
const BIG = new Set(['commercial', 'retail', 'industrial', 'warehouse', 'office', 'supermarket', 'school', 'hospital', 'university', 'hotel', 'civic', 'public', 'government', 'train_station']);
function heightOf(tags, area, cx, cz, id) {
  const r = hash(id);
  let h = parseLen(tags.h);
  if (!(h > 0)) { const lv = parseFloat(tags.l); if (lv > 0) h = lv * 3.2 + 1.0; }
  if (h > 0) return Math.min(330, Math.max(2.5, h));
  const b = tags.b || 'yes';
  if (SMALL.has(b) || area < 40) return 2.8 + r * 0.8;
  for (const c of CORES) {
    const d = Math.hypot(cx - c.x, cz - c.z);
    if (d < c.r && area > 250) { const k = 1 - d / c.r; return (c.a + Math.floor(r * (c.b - c.a) * (0.4 + 0.6 * k))) * 3.4 + 2; }
  }
  if (BIG.has(b) || area > 1800) return 7 + r * 5;
  if (b === 'apartments') return (3 + Math.floor(r * 3)) * 3.1 + 1;
  if (SF(cx, cz)) return 8.5 + r * 4.5;           // SF row houses: 2-3 storeys over garage
  return r < 0.65 ? 4.6 + r * 1.5 : 6.6 + r * 1.6;   // suburbs: mostly one storey, some two
}

// ------------------------------------------------------------------ footprints
const tiles = new Map();   // key -> array of building records
const st = { in: 0, dup: 0, tiny: 0, dropped: 0, pushed: 0, kept: 0, tagged: 0 };
function simplifyRing(pts, tol = 0.35) {
  // drop near-duplicate and collinear vertices (open ring, no closing point)
  let out = pts.slice();
  for (let pass = 0; pass < 2; pass++) {
    const r = [];
    for (let i = 0; i < out.length; i++) {
      const p = out[(i - 1 + out.length) % out.length], c = out[i], n = out[(i + 1) % out.length];
      const dx = n[0] - p[0], dz = n[1] - p[1], L = Math.hypot(dx, dz);
      const dist = L > 1e-6 ? Math.abs((c[0] - p[0]) * dz - (c[1] - p[1]) * dx) / L : Math.hypot(c[0] - p[0], c[1] - p[1]);
      if (dist > tol && Math.hypot(c[0] - p[0], c[1] - p[1]) > 0.2) r.push(c);
    }
    out = r; if (out.length < 3) return null;
  }
  return out;
}
const areaOf = (r) => { let a = 0; for (let i = 0; i < r.length; i++) { const p = r[i], q = r[(i + 1) % r.length]; a += p[0] * q[1] - q[0] * p[1]; } return a / 2; };
function addFootprint(e) {
  st.in++;
  let pts = simplifyRing(e.rings[0]); if (!pts) { st.tiny++; return; }
  let A = areaOf(pts);
  if (Math.abs(A) < 12) { st.tiny++; return; }
  if (A < 0) pts.reverse();   // CCW seen from above (x east, z south): positive signed area in x-z
  A = Math.abs(A);
  // road clearance: push offending vertices out; drop if the footprint collapses or the centre is on a road
  let cx = 0, cz = 0; for (const p of pts) { cx += p[0]; cz += p[1]; } cx /= pts.length; cz /= pts.length;
  const [, , cm] = clear(cx, cz);
  if (cm > 0) { st.dropped++; return; }
  let maxMove = 0; const np = pts.map(([x, z]) => { const [nx, nz, m] = clear(x, z); maxMove = Math.max(maxMove, m); return [nx, nz]; });
  const A2 = areaOf(np);
  if (maxMove > 0) { if (A2 < A * 0.55 || maxMove > 9) { st.dropped++; return; } st.pushed++; }
  pts = np;
  let h;
  if (e.src === 'sf' && e.h > 1.5) { h = Math.min(330, e.h); st.lidar++; }
  else { const tags = e.tags; if (parseLen(tags.h) > 0 || parseFloat(tags.l) > 0) st.tagged++; h = heightOf(tags, A2, cx, cz, e.id); }
  const tx = Math.floor((cx - ROOT.x0) / TILE), tz = Math.floor((cz - ROOT.z0) / TILE);
  if (tx < 0 || tz < 0 || tx >= NT || tz >= NT) return;
  const key = tx + '_' + tz; let l = tiles.get(key); if (!l) tiles.set(key, (l = []));
  const style = Math.floor(hash(e.id * 7 + 3) * 256);
  l.push({ pts, h, style, flags: (e.tags.b === 'house' || e.tags.b === 'detached' ? 1 : 0) | (h > 30 ? 2 : 0) });
  st.kept++;
}
st.lidar = 0;
const src = forEachFootprint(addFootprint);
log('sources', JSON.stringify(src));

// ------------------------------------------------------------------ write tiles
let bytes = 0; const list = [];
for (const [key, l] of tiles) {
  const [tx, tz] = key.split('_').map(Number);
  const ox = ROOT.x0 + (tx + 0.5) * TILE, oz = ROOT.z0 + (tz + 0.5) * TILE;
  let n = 4; for (const b of l) n += 6 + b.pts.length * 4;
  const buf = Buffer.alloc(n); let o = 0;
  buf.writeUInt32LE(l.length, o); o += 4;
  for (const b of l) {
    buf.writeUInt16LE(b.pts.length, o); buf.writeUInt16LE(Math.min(65535, Math.round(b.h * 10)), o + 2); buf[o + 4] = b.style; buf[o + 5] = b.flags; o += 6;
    let px = 0, pz = 0;
    for (const [x, z] of b.pts) {
      const qx = Math.round((x - ox) * Q), qz = Math.round((z - oz) * Q);
      buf.writeInt16LE(Math.max(-32768, Math.min(32767, qx - px)), o); buf.writeInt16LE(Math.max(-32768, Math.min(32767, qz - pz)), o + 2); o += 4;
      px = qx; pz = qz;
    }
  }
  const gz = zlib.gzipSync(buf, { level: 9 });
  writeFileSync(path.join(DIR, key + '.bin.gz'), gz);
  bytes += gz.length; list.push([tx, tz, l.length]);
}
const meta = { format: 'bay-buildings/1', root: ROOT, tile: TILE, grid: NT, quant: Q, tiles: list, bytes, stats: st,
  source: 'OpenStreetMap buildings via Overpass (ODbL 1.0, (c) OpenStreetMap contributors)', generated: new Date().toISOString() };
writeFileSync(path.join(OUT_DIR, 'buildings.json'), JSON.stringify(meta));
log('buildings', JSON.stringify(st), list.length, 'tiles', (bytes / 1e6).toFixed(1), 'MB');
