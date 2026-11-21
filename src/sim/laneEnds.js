// Lane-end look-ahead as a pure function (shared by the sim and the brain's driver/eyes, so
// both see exactly the same lane ends). Lanes are followed through the hop lane mapping
// (roadGraph.mapLane = src/geom/hops.js for the real data). My lane "ends" ahead if, at some
// node within `lookahead` m, it maps outside the next carriageway while another lane of MY
// current edge still continues (then I must move toward it here). If no lane of my edge
// survives, the change has to happen on a later edge: no end is reported, but `mergeCap`
// says how slow to arrive so the lane changes on that (short) edge are still possible.
//
//   laneEndAhead(graph, route, routePos, s, lane, { lookahead = 450, stopIndex = -1 }, out)
//     -> out = { dist, side, mergeCap }   dist: m from s to the node where my lane ends
//        (lanes are paved to the node; 1e9 = none), side: +1 surviving lanes on my left,
//        -1 right, 0 none; mergeCap: m/s (1e9 = none)

import { mapLane, laneW } from './roadGraph.js';

const alive = new Int16Array(16), cur = new Int16Array(16);

export function laneEndAhead(g, route, rp, s, lane, { lookahead = 450, stopIndex = -1 } = {}, out = { dist: 1e9, side: 0, mergeCap: 1e9 }) {
  out.dist = 1e9; out.side = 0; out.mergeCap = 1e9;
  if (stopIndex === rp) return out;
  const e = g.edges[route[rp]];
  let dist = e.length - s, prev = e, endK = -1, myL = lane;
  const n = e.lanes;
  for (let l = 0; l < n; l++) { alive[l] = 1; cur[l] = l; }
  for (let k = 1; k <= 4 && rp + k < route.length && dist < lookahead; k++) {
    const en = g.edges[route[rp + k]];
    let any = false;
    for (let l = 0; l < n; l++) if (alive[l]) { const l2 = mapLane(g, prev, en, cur[l]); if (l2 < 0 || l2 >= en.lanes) alive[l] = 0; else { cur[l] = l2; any = true; } }
    if (!alive[lane] && !any && k > 1) {
      let need = 1e9; for (let l = 0; l < prev.lanes; l++) { const l2 = mapLane(g, prev, en, l); if (l2 >= 0 && l2 < en.lanes) need = Math.min(need, Math.abs(l - myL)); }
      if (need > 0 && need < 1e9) out.mergeCap = Math.max(5, Math.max(10, prev.length - 10) / (need * laneW(prev, prev.length) / 1.3));
      return out;
    }
    if (!alive[lane]) { if (any) endK = k; break; }
    myL = mapLane(g, prev, en, myL);
    if (stopIndex === rp + k) return out;
    dist += en.length; prev = en;
  }
  if (endK > 0) {
    let best = -1; for (let l = 0; l < n; l++) if (alive[l] && (best < 0 || Math.abs(l - lane) < Math.abs(best - lane))) best = l;
    out.side = best > lane ? 1 : -1;
    out.dist = dist;
  }
  return out;
}
