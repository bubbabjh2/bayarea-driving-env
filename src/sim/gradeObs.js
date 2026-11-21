// Road-grade observation for the brain (input version 2, docs/grade-input.md). Mirrored 1:1 by
// src/gpu/kernels/sense.wgsl (gradeInputs). + = uphill. Raw grades (dy/ds); eyes slots 41..43 = clamp(g / 0.35, -1, 1).
//   g0 = physics grade under the car (vehicle.js edgeGrade, 12 m window; 0 on a junction connector)
//   g1 = (y(40) - y(0)) / 40,  g2 = (y(100) - y(40)) / 60, y(d) = height d m ahead along the route (edge frame)
import { edgeGrade } from './vehicle.js';

export const GRADE_AHEAD1 = 40, GRADE_AHEAD2 = 100, GRADE_MAX_HOPS = 12;

/** height of edge e at s (polyline ys, linear; e.y without ys) */
export function edgeY(e, s) {
  const ys = e.ys; if (!ys) return e.y || 0;
  const cum = e.cum, n = cum.length;
  let k = 0; while (k < n - 2 && cum[k + 1] < s) k++;
  const d = cum[k + 1] - cum[k]; let t = d > 0 ? (s - cum[k]) / d : 0; t = t < 0 ? 0 : t > 1 ? 1 : t;
  return ys[k] + (ys[k + 1] - ys[k]) * t;
}

/** height d m ahead of (route[pos], s) along the route */
export function routeY(edges, route, pos, routeLen, s, d) {
  let q = pos, rem = (s > 0 ? s : 0) + d, e = edges[route[q]], hops = 0;
  while (rem > e.length && q + 1 < routeLen && hops < GRADE_MAX_HOPS) { rem -= e.length; q++; hops++; e = edges[route[q]]; }
  return edgeY(e, rem < e.length ? rem : e.length);
}

/** [g0, g1, g2] raw grades into `out` for slot i of the CPU sim. */
export function gradeObs(sim, i, out) {
  const vs = sim.vs, edges = sim.graph.edges, ei = vs.edge[i], e = edges[ei], s = vs.s[i];
  out[0] = vs.conn[i] >= 0 ? 0 : edgeGrade(e, s, vs.seg[i]);
  const route = vs.routes[i], len = vs.routeLen[i];
  let pos = vs.routePos[i];
  if (!route || !(len > 0) || pos < 0 || pos >= len || route[pos] !== ei) {   // no route (or off it): this edge only
    const y0 = edgeY(e, s > 0 ? s : 0), y1 = edgeY(e, Math.min(e.length, s + GRADE_AHEAD1)), y2 = edgeY(e, Math.min(e.length, s + GRADE_AHEAD2));
    out[1] = (y1 - y0) / GRADE_AHEAD1; out[2] = (y2 - y1) / (GRADE_AHEAD2 - GRADE_AHEAD1); return out;
  }
  const y0 = routeY(edges, route, pos, len, s, 0), y1 = routeY(edges, route, pos, len, s, GRADE_AHEAD1), y2 = routeY(edges, route, pos, len, s, GRADE_AHEAD2);
  out[1] = (y1 - y0) / GRADE_AHEAD1; out[2] = (y2 - y1) / (GRADE_AHEAD2 - GRADE_AHEAD1);
  return out;
}
