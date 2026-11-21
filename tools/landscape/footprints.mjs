// Building footprints for the landscape builders (render only):
//   San Francisco (whole city): DataSF "Building Footprints" with lidar heights (raw/sf-buildings/page_*.json)
//   elsewhere: OSM buildings fetched on demand per area (raw/osm-buildings/*.json, fetch_osm.mjs buildings --bbox ...)
// OSM buildings inside the DataSF coverage are skipped (one source per place).
import { readFileSync, readdirSync, existsSync } from 'node:fs';
import path from 'node:path';
import { OUT_DIR, toXZ } from './common.mjs';

/** calls fn({id, rings: [[ [x,z], ...] (open, first = outer)], h (m or NaN), tags, src}) for every footprint */
export function forEachFootprint(fn) {
  const sfDir = path.join(OUT_DIR, 'raw/sf-buildings');
  let sfBox = null;
  const stats = { sf: 0, osm: 0, osmSkipped: 0 };
  if (existsSync(sfDir)) {
    let x0 = Infinity, z0 = Infinity, x1 = -Infinity, z1 = -Infinity;
    for (const f of readdirSync(sfDir).filter((f) => /^page_\d+\.json$/.test(f)).sort()) {
      const rows = JSON.parse(readFileSync(path.join(sfDir, f), 'utf8'));
      if (!Array.isArray(rows)) continue;
      for (const r of rows) {
        const g = r.shape; if (!g || !g.coordinates) continue;
        const polys = g.type === 'MultiPolygon' ? g.coordinates : g.type === 'Polygon' ? [g.coordinates] : [];
        const hm = parseFloat(r.hgt_mediancm) / 100, hx = parseFloat(r.hgt_maxcm) / 100;
        for (const p of polys) {
          const rings = p.filter((ring) => ring.length >= 4).map((ring) => ring.slice(0, -1).map(([lon, lat]) => toXZ(lat, lon)));
          if (!rings.length) continue;
          for (const [x, z] of rings[0]) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (z < z0) z0 = z; if (z > z1) z1 = z; }
          fn({ id: Math.round(parseFloat(r.sf16_bldgid) * 1e7) % 2147483647, rings, h: hm, hMax: hx, tags: {}, src: 'sf' });
          stats.sf++;
        }
      }
    }
    if (stats.sf) sfBox = { x0, z0, x1, z1 };
  }
  // DataSF covers the city and county of San Francisco: skip OSM there (bbox test + SF's southern line ~ 37.708 N)
  const [, zSouth] = toXZ(37.7082, -122.4);
  const inSF = (x, z) => sfBox && x > sfBox.x0 && x < sfBox.x1 && z > sfBox.z0 && z < Math.min(sfBox.z1, zSouth);
  const bDir = path.join(OUT_DIR, 'raw/osm-buildings');
  const seen = new Set();
  if (existsSync(bDir)) for (const f of readdirSync(bDir).filter((f) => f.endsWith('.json')).sort()) {
    const j = JSON.parse(readFileSync(path.join(bDir, f), 'utf8'));
    for (const e of j.elements || []) {
      if (seen.has(e.id)) continue; seen.add(e.id);
      const g = e.geometry; if (!g) continue;
      let polys = [];
      if (g.type === 'LineString') polys = [[g.coordinates]];
      else if (g.type === 'Polygon') polys = [g.coordinates];
      else if (g.type === 'MultiPolygon') polys = g.coordinates;
      for (const p of polys) {
        const rings = p.filter((ring) => ring.length >= 4).map((ring) => ring.slice(0, -1).map(([lon, lat]) => toXZ(lat, lon)));
        if (!rings.length) continue;
        const [cx, cz] = rings[0][0];
        if (inSF(cx, cz)) { stats.osmSkipped++; continue; }
        fn({ id: e.id, rings, h: NaN, tags: e.tags || {}, src: 'osm' });
        stats.osm++;
      }
    }
  }
  return stats;
}
