// Bridge structures for the landscape (render only): deck girders + piers under every elevated road (main + local),
// and the Golden Gate Bridge layout (deck line, towers, anchorages) aligned to the road data.
//   node tools/landscape/build_structures.mjs        (re-run after any road elevation rebuild: deck heights come from data/)
// In:  data/roads.json, data/roads-local.json (read only)
// Out: data/landscape/structures.json
//   girders: [[ax, az, ay, bx, bz, by, width], ...]   one box per polyline segment > 1.5 m above the ground, top 0.3 m under the deck
//   piers:   [[x, z, yTop, width, heading], ...]       column bent every ~38 m; skipped where a road passes underneath
//   goldengate: { a: [x, z], b: [x, z] (deck line, south -> north), towers: [s, s] (m along), anchors: [s, s], deck: [[s, y], ...] }
import { readFileSync, writeFileSync, existsSync, openSync, readSync, closeSync } from 'node:fs';
import path from 'node:path';
import { ROOT_DIR, OUT_DIR, toXZ, log } from './common.mjs';
import { loadCompactGround } from '../elevation/compact.mjs';

const main = JSON.parse(readFileSync(path.join(ROOT_DIR, 'data/roads.json'), 'utf8'));
const local = JSON.parse(readFileSync(path.join(ROOT_DIR, 'data/roads-local.json'), 'utf8'));
const LANE = 3.7, ELEV = 3.0;
// Height above the ground decides what is a structure (roads carry real terrain heights): the road-conformed ground
// sits 0.15 m under at-grade roads and stays natural under bridges. ground10 (npm run build:elevation) if present,
// else the committed 20 m compact grid.
const groundAt = (() => {
  const dir = path.join(ROOT_DIR, 'data/dem/ground');
  if (existsSync(path.join(dir, 'ground10.f32'))) {
    const g = JSON.parse(readFileSync(path.join(dir, 'ground10.json'), 'utf8')), n = g.n, c = g.cell, H = new Float32Array(n * n);
    const fd = openSync(path.join(dir, 'ground10.f32'), 'r'), buf = Buffer.from(H.buffer);
    for (let off = 0; off < buf.length;) { const k = readSync(fd, buf, off, Math.min(1 << 26, buf.length - off), off); if (k <= 0) throw new Error('short read'); off += k; }
    closeSync(fd);
    log('ground: ground10');
    return (x, z) => {
      const fc = (x - g.root.x0) / c - 0.5, fr = (z - g.root.z0) / c - 0.5;
      if (!(fc >= 0 && fr >= 0 && fc <= n - 1 && fr <= n - 1)) return 0;
      const c0 = Math.min(n - 2, Math.floor(fc)), r0 = Math.min(n - 2, Math.floor(fr)), a = fc - c0, b = fr - r0, k = r0 * n + c0;
      return (H[k] * (1 - a) + H[k + 1] * a) * (1 - b) + (H[k + n] * (1 - a) + H[k + n + 1] * a) * b;
    };
  }
  const cg = loadCompactGround(); log('ground: compact 20 m');
  return (x, z) => { const h = cg.heightAt(x, z); return Number.isFinite(h) ? h : 0; };
})();
const above = (p) => p.y - groundAt(p.x, p.z);

// ground-level road segments (for "is there a road under this pier?")
const SG = 30, grid = new Map(), low = [];
for (const R of [main, local]) for (const e of R.edges) {
  if (e.twin >= 0 && e.twin < e.id) continue;
  const ns = e.nodes.map((k) => R.nodes[k]);
  const half = ((e.lanes || 1) + (e.oppLanes || 0)) * LANE / 2 + 1.5;
  for (let i = 0; i + 1 < ns.length; i++) {
    const a = ns[i], b = ns[i + 1];
    if (Math.max(above(a), above(b)) > ELEV) continue;
    const id = low.length; low.push([a.x, a.z, b.x, b.z, half]);
    for (let gx = Math.floor((Math.min(a.x, b.x) - half) / SG); gx <= Math.floor((Math.max(a.x, b.x) + half) / SG); gx++)
      for (let gz = Math.floor((Math.min(a.z, b.z) - half) / SG); gz <= Math.floor((Math.max(a.z, b.z) + half) / SG); gz++) {
        const k = gx * 1000003 + gz; let l = grid.get(k); if (!l) grid.set(k, (l = [])); l.push(id);
      }
  }
}
function roadBelow(x, z, r) {
  for (let gx = Math.floor((x - r) / SG); gx <= Math.floor((x + r) / SG); gx++) for (let gz = Math.floor((z - r) / SG); gz <= Math.floor((z + r) / SG); gz++) {
    for (const id of grid.get(gx * 1000003 + gz) || []) {
      const [ax, az, bx, bz, h] = low[id];
      const dx = bx - ax, dz = bz - az, L2 = dx * dx + dz * dz;
      let t = L2 > 0 ? ((x - ax) * dx + (z - az) * dz) / L2 : 0; t = Math.max(0, Math.min(1, t));
      if (Math.hypot(x - ax - dx * t, z - az - dz * t) < h + r) return true;
    }
  }
  return false;
}

const girders = [], piers = [];
let skipped = 0;
for (const R of [main, local]) for (const e of R.edges) {
  if (e.twin >= 0 && e.twin < e.id) continue;
  const ns = e.nodes.map((k) => R.nodes[k]);
  const width = Math.max(3, ((e.lanes || 1) + (e.twin >= 0 ? (e.oppLanes || 0) : 0)) * LANE - 0.8);
  // walk the polyline: girders under elevated stretches, piers every ~38 m where the deck is high enough
  let carry = 19;
  for (let i = 0; i + 1 < ns.length; i++) {
    const a = ns[i], b = ns[i + 1];
    const L = Math.hypot(b.x - a.x, b.z - a.z); if (L < 0.5) continue;
    const ha = above(a), hb = above(b);
    if (Math.min(ha, hb) > 1.5) girders.push([a.x, a.z, a.y, b.x, b.z, b.y, width].map((v) => +v.toFixed(2)));
    const hd = Math.atan2(b.x - a.x, -(b.z - a.z));
    for (let s = carry; s < L; s += 38) {
      const t = s / L, x = a.x + (b.x - a.x) * t, z = a.z + (b.z - a.z) * t, y = a.y + (b.y - a.y) * t;
      if (y - groundAt(x, z) < ELEV + 2) continue;
      if (roadBelow(x, z, width > 12 ? width / 2 : 2)) { skipped++; continue; }
      piers.push([+x.toFixed(2), +z.toFixed(2), +(y - 1.6).toFixed(2), +width.toFixed(1), +hd.toFixed(4)]);
    }
    carry = ((carry - L) % 38 + 38) % 38;
  }
}

// ---- Golden Gate Bridge: deck line from the two carriageway edges; towers / anchorages from their real positions
let goldengate = null;
{
  const gg = main.edges.filter((e) => /^Golden Gate Bridge$/i.test(e.name || '') && e.bridge && e.length > 1500);
  if (gg.length) {
    // the deck centreline: average of the carriageway end points (south -> north)
    const ends = gg.map((e) => { const ns = e.nodes.map((k) => main.nodes[k]); const p = ns[0], q = ns[ns.length - 1]; return p.z > q.z ? [p, q] : [q, p]; });
    const avg = (k) => ends.reduce((s, en) => [s[0] + en[k].x / ends.length, s[1] + en[k].z / ends.length], [0, 0]);
    const A = avg(0), B = avg(1), dx = B[0] - A[0], dz = B[1] - A[1], L = Math.hypot(dx, dz), ux = dx / L, uz = dz / L;
    const along = (lat, lon) => { const [x, z] = toXZ(lat, lon); return (x - A[0]) * ux + (z - A[1]) * uz; };
    // deck height profile from the data along the line (samples every 25 m, both carriageways averaged)
    const deck = [];
    for (let s = -150; s <= L + 150; s += 25) {
      let ys = [];
      for (const e of main.edges.filter((e) => /Golden Gate Bridge/i.test(e.name || ''))) {
        const ns = e.nodes.map((k) => main.nodes[k]);
        for (let i = 0; i + 1 < ns.length; i++) {
          const a = ns[i], b = ns[i + 1];
          const sa = (a.x - A[0]) * ux + (a.z - A[1]) * uz, sb = (b.x - A[0]) * ux + (b.z - A[1]) * uz;
          if ((s - sa) * (s - sb) <= 0 && sa !== sb) ys.push(a.y + (b.y - a.y) * (s - sa) / (sb - sa));
        }
      }
      if (ys.length) deck.push([s, +(ys.reduce((p, c) => p + c, 0) / ys.length).toFixed(2)]);
    }
    goldengate = {
      a: A.map((v) => +v.toFixed(2)), b: B.map((v) => +v.toFixed(2)), length: +L.toFixed(1),
      towers: [along(37.8107, -122.4772), along(37.8222, -122.4790)].map((v) => +v.toFixed(1)),
      anchors: [along(37.8077, -122.4766), along(37.8253, -122.4794)].map((v) => +v.toFixed(1)),
      deck, deckRealM: 67, towerRealM: 227, cableLowRealM: 76, saddleRealM: 218, width: 27.4, cableGap: 27.4,
      note: 'heights are drawn relative to the data deck (real deck 67 m above water); tower / anchor positions from their real lat/lon projected on the deck line',
    };
    // the main-span piers are the towers: no generic piers in the water between the anchorages
    const inGG = (x, z) => { const s = (x - A[0]) * ux + (z - A[1]) * uz, t = Math.abs((x - A[0]) * -uz + (z - A[1]) * ux); return t < 60 && s > goldengate.anchors[0] - 20 && s < goldengate.anchors[1] + 20; };
    for (let i = piers.length - 1; i >= 0; i--) if (inGG(piers[i][0], piers[i][1])) piers.splice(i, 1);
    for (let i = girders.length - 1; i >= 0; i--) if (inGG(girders[i][0], girders[i][1]) && inGG(girders[i][3], girders[i][4])) girders.splice(i, 1);
  }
}

const out = { format: 'bay-structures/1', girders, piers, goldengate, generated: new Date().toISOString(),
  source: 'derived from data/roads*.json (OSM, ODbL); Golden Gate tower positions from public coordinates' };
writeFileSync(path.join(OUT_DIR, 'structures.json'), JSON.stringify(out));
log(`girders ${girders.length}, piers ${piers.length} (skipped ${skipped} over roads), golden gate ${goldengate ? 'yes' : 'no'}`);
if (goldengate) log('GG', JSON.stringify({ ...goldengate, deck: goldengate.deck.length + ' samples' }));
