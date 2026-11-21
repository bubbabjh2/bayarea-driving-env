// Collision detection (oriented rectangles, SAT) over the spatial hash, plus the
// stochastic accident model. Crash state handling itself lives in sim.js.

import { STATE_FREE, STATE_REMOVED } from './vehicle.js';

/** Separating-axis test for two oriented rectangles (centres, headings, half extents). */
export function obbOverlap(ax, az, ah, al, aw, bx, bz, bh, bl, bw) {
  const ac = Math.cos(ah), as = Math.sin(ah), bc = Math.cos(bh), bs = Math.sin(bh);
  const dx = bx - ax, dz = bz - az;
  // axes: a forward (ac,as), a right (-as,ac), b forward, b right
  const axes = [ac, as, -as, ac, bc, bs, -bs, bc];
  for (let k = 0; k < 4; k++) {
    const ux = axes[2 * k], uz = axes[2 * k + 1];
    const ra = Math.abs(ac * ux + as * uz) * al + Math.abs(-as * ux + ac * uz) * aw;
    const rb = Math.abs(bc * ux + bs * uz) * bl + Math.abs(-bs * ux + bc * uz) * bw;
    if (Math.abs(dx * ux + dz * uz) > ra + rb) return false;
  }
  return true;
}

/**
 * Find overlapping vehicle pairs. onPair(i, j) for each pair (i < j). Vehicles with
 * state FREE/REMOVED are ignored. `shrink` (0..1) scales the boxes to tolerate the
 * lane-change blend and float jitter.
 */
export function detectCollisions(vs, hash, onPair, shrink = 0.9, maxLen = 12, junctionMargin = 0, edges = null) {
  const high = vs.high;
  const { x, z, heading, length, width, state } = vs;
  let pairs = 0;
  for (let i = 0; i < high; i++) {
    const si = state[i];
    if (si === STATE_FREE || si === STATE_REMOVED) continue;
    const xi = x[i], zi = z[i], li = length[i] * 0.5 * shrink, wi = width[i] * 0.5 * shrink, hi = heading[i];
    const reach = li + maxLen * 0.5;
    hash.forEachNear(xi, zi, reach, (j, d2) => {
      if (j <= i) return;
      const sj = state[j];
      if (sj === STATE_FREE || sj === STATE_REMOVED) return;
      const lj = length[j] * 0.5 * shrink, wj = width[j] * 0.5 * shrink;
      const rr = li + lj;
      if (d2 > rr * rr) return;
      if (junctionMargin > 0 && vs.edge[i] !== vs.edge[j]) {
        // different edges: ignore pairs inside the junction box (lane offsets jump between edges there)
        const ei = edges[vs.edge[i]], ej = edges[vs.edge[j]];
        if (vs.s[i] < junctionMargin || ei.length - vs.s[i] < junctionMargin || vs.s[j] < junctionMargin || ej.length - vs.s[j] < junctionMargin) return;
      }
      const minr = Math.min(wi, wj);
      if (d2 > 0 && d2 < minr * minr || obbOverlap(xi, zi, hi, li, wi, x[j], z[j], heading[j], lj, wj)) { pairs++; onPair(i, j); }
    });
  }
  return pairs;
}

export const ACCIDENT_DEFAULTS = Object.freeze({
  ratePerKm: 5e-5,      // base probability per vehicle-km (US average is ~2e-6; raised for drama)
  speedRef: 25,         // m/s; risk grows as (1 + (v/speedRef)^2)
  closeGap: 30,         // m; following closer than this doubles the risk
  mergeFactor: 3,       // on links / within mergeDist of a junction
  mergeDist: 60,
  clearMin: 10 * 60,    // crash clearing time range (s)
  clearMax: 40 * 60,
  rubberneckDist: 80,   // m, adjacent-lane slowdown zone around a crash
  rubberneckFactor: 0.4,// desired speed multiplier in that zone
});

/**
 * Stochastic accident roll for one vehicle. Returns true if an accident occurs.
 * dsKm = distance driven this step in km.
 */
export function accidentRoll(rng, cfg, dsKm, speed, gap, nearMerge) {
  if (dsKm <= 0) return false;
  let p = cfg.ratePerKm * dsKm;
  const r = speed / cfg.speedRef;
  p *= 1 + r * r;
  if (gap < cfg.closeGap) p *= 2;
  if (nearMerge) p *= cfg.mergeFactor;
  return rng.random() < p;
}
