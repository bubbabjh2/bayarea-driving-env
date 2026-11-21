// Road clearance bands for the landscape builders (render only; reads data/roads*.json read-only).
// Every road gets a band of half-width D = carriageway (lanes x 3.7 m, as src/geom) + parking / curb / verge /
// sidewalk where the streamed street model draws them (tools/tiles/build_tiles.mjs) + 0.5 m margin.
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { ROOT_DIR, log } from './common.mjs';

export function loadRoadClearance({ margin = 0.5 } = {}) {
  const segs = [];   // [ax, az, bx, bz, D]
  const SG = 40, segGrid = new Map();

  const local = JSON.parse(readFileSync(path.join(ROOT_DIR, 'data/roads-local.json'), 'utf8'));
  const main = JSON.parse(readFileSync(path.join(ROOT_DIR, 'data/roads.json'), 'utf8'));
  // urban density (local street metres per 500 m cell), as tools/tiles/build_tiles.mjs
  const dens = new Map(), ck = (x, z) => Math.floor(x / 500) * 100000 + Math.floor(z / 500);
  for (const e of local.edges) {
    if (e.twin >= 0 && e.twin < e.id) continue;
    const a = local.nodes[e.from], b = local.nodes[e.to];
    const k = ck((a.x + b.x) / 2, (a.z + b.z) / 2); dens.set(k, (dens.get(k) || 0) + (e.length || 0));
  }
  const urbanAt = (x, z) => (dens.get(ck(x, z)) || 0) > 2500;
  const LANE = 3.7;
  for (const [tier, R] of [['main', main], ['local', local]]) {
    for (const e of R.edges) {
      if (e.twin >= 0 && e.twin < e.id) continue;
      const ns = e.nodes.map((k) => R.nodes[k]);
      const mid = ns[ns.length >> 1];
      const urban = urbanAt(mid.x, mid.z);
      const half = ((e.lanes || 1) + (e.oppLanes || 0)) * LANE / 2;
      const c = e.class;
      let extra = 1.0;
      if (urban && (c === 'residential' || c === 'living_street')) extra = 2.0 + 0.3 + 1.5 + 1.8;
      else if (urban && (c === 'tertiary' || c === 'primary' || c === 'secondary' || c === 'unclassified')) extra = 0.3 + 1.8 + (e.oppLanes > 0 ? 2.0 : 0);
      else if (c === 'motorway' || c === 'trunk') extra = 3.0;
      if (e.bridge || mid.y > 2) extra += 3.0;
      const D = half + extra + margin;
      for (let i = 0; i + 1 < ns.length; i++) {
        const s = [ns[i].x, ns[i].z, ns[i + 1].x, ns[i + 1].z, D, half + (urban && (c === 'residential' || c === 'living_street') ? 2.0 : 0)];
        const id = segs.length; segs.push(s);
        const x0 = Math.floor((Math.min(s[0], s[2]) - D) / SG), x1 = Math.floor((Math.max(s[0], s[2]) + D) / SG);
        const z0 = Math.floor((Math.min(s[1], s[3]) - D) / SG), z1 = Math.floor((Math.max(s[1], s[3]) + D) / SG);
        for (let gx = x0; gx <= x1; gx++) for (let gz = z0; gz <= z1; gz++) {
          const k = gx * 1000003 + gz; let l = segGrid.get(k); if (!l) segGrid.set(k, (l = [])); l.push(id);
        }
      }
    }
  }
  log('road segments', segs.length);
  /** Push a point out of every road's clearance band; returns [x, z, moved]. */
  function clear(x, z) {
    let moved = 0;
    for (let it = 0; it < 3; it++) {
      const l = segGrid.get(Math.floor(x / SG) * 1000003 + Math.floor(z / SG)); if (!l) break;
      let worst = 0, px = 0, pz = 0;
      for (const id of l) {
        const [ax, az, bx, bz, D] = segs[id];
        const dx = bx - ax, dz = bz - az, L2 = dx * dx + dz * dz;
        let t = L2 > 0 ? ((x - ax) * dx + (z - az) * dz) / L2 : 0; t = Math.max(0, Math.min(1, t));
        const qx = ax + dx * t, qz = az + dz * t, ex = x - qx, ez = z - qz, d = Math.hypot(ex, ez);
        if (d < D && D - d > worst) {
          worst = D - d;
          if (d > 1e-6) { px = ex / d * (D - d + 0.05); pz = ez / d * (D - d + 0.05); }
          else { const n = Math.hypot(dx, dz) || 1; px = -dz / n * (D + 0.05); pz = dx / n * (D + 0.05); }
        }
      }
      if (!worst) break;
      x += px; z += pz; moved = Math.max(moved, Math.hypot(px, pz));
    }
    return [x, z, moved];
  }
  /** true if (x, z) is inside any road band widened by extra metres */
  function onRoad(x, z, extra = 0) {
    const l = segGrid.get(Math.floor(x / SG) * 1000003 + Math.floor(z / SG)); if (!l) return false;
    for (const id of l) {
      const [ax, az, bx, bz, D] = segs[id];
      const dx = bx - ax, dz = bz - az, L2 = dx * dx + dz * dz;
      let t = L2 > 0 ? ((x - ax) * dx + (z - az) * dz) / L2 : 0; t = Math.max(0, Math.min(1, t));
      if (Math.hypot(x - ax - dx * t, z - az - dz * t) < D + extra) return true;
    }
    return false;
  }
  /** metres from (x, z) to the nearest asphalt edge (carriageway + parking); negative = on the asphalt; 99 if none near */
  function asphaltGap(x, z) {
    const l = segGrid.get(Math.floor(x / SG) * 1000003 + Math.floor(z / SG)); if (!l) return 99;
    let g = 99;
    for (const id of l) {
      const [ax, az, bx, bz, , Da] = segs[id];
      const dx = bx - ax, dz = bz - az, L2 = dx * dx + dz * dz;
      let t = L2 > 0 ? ((x - ax) * dx + (z - az) * dz) / L2 : 0; t = Math.max(0, Math.min(1, t));
      g = Math.min(g, Math.hypot(x - ax - dx * t, z - az - dz * t) - Da);
    }
    return g;
  }
  return { segs, clear, onRoad, asphaltGap };
}
