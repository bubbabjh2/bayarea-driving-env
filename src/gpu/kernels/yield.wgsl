// sim.js ctx.yieldRequired (junction gap judgement), shared by drive.wgsl and leaders.wgsl
// (obs input 62). Needs veh / par bindings, lists.wgsl and stopAt() from the including kernel.
// ---------------------------------------------------------------- yielding
fn yieldRequired(i: u32, e: Edge, eIdx: u32, dist: f32) -> bool {
  if (dist < 1.5 || hasFeat(FEAT_EMPTY)) { return false; }
  let v = veh[i];
  let pi = par[i];
  var nInc = 0u;
  for (var q = inStart(e.to); q < inStart(e.to + 1u); q = q + 1u) {
    let ei = inEdge(q);
    if (ei != eIdx && i32(ei) != e.reverse) { nInc = nInc + 1u; }
  }
  if (nInc == 0u) { return false; }
  let L0 = v.leader;
  if (L0 >= 0 && veh[u32(L0)].speed < 1.0 && veh[u32(L0)].state == DRIVING && v.gap - dist < pi.length + 3.0) { return true; }
  var nextEdge = NONE;
  var nextRev = -1;
  if (v.routePos + 1u < pi.routeLen) { nextEdge = routeEdge(pi, v.routePos + 1u); nextRev = eRev(nextEdge); }
  let now = simTime();
  var waited = 0.0;
  if (v.stoppedSince >= 0.0) { waited = now - v.stoppedSince; }
  let patienceScale = max(0.3, 1.0 - waited / 90.0);
  let tMe = dist / max(v.speed, 2.0);
  for (var q = outStart(e.to); q < outStart(e.to + 1u); q = q + 1u) {
    let eo = outEdge(q);
    if (i32(eo) == e.reverse) { continue; }
    let lanesO = eLanes(eo);
    for (var l = 0u; l < lanesO; l = l + 1u) {
      let jj = laneFirst(eo, l);
      if (jj < 0) { continue; }
      let j = u32(jj);
      if (veh[j].state != DRIVING) { continue; }
      if (veh[j].speed < 0.5 && waited > 20.0) { continue; }
      if (veh[j].s < 0.5 * par[j].length + e.stopBack + 2.0 && !(eo == nextEdge && i32(l) == mapLane(eIdx, eo, i32(vLane(v))))) { return true; }
    }
  }
  for (var q = inStart(e.to); q < inStart(e.to + 1u); q = q + 1u) {
    let e2 = inEdge(q);
    if (e2 == eIdx || i32(e2) == e.reverse || i32(e2) == nextRev) { continue; }
    let ed2 = edgeAt(e2);
    for (var l = 0u; l < ed2.lanes; l = l + 1u) {
      let jj = laneLast(e2, l);
      if (jj < 0) { continue; }
      let j = u32(jj);
      let vj = veh[j];
      if (vj.state != DRIVING) { continue; }
      let pj = par[j];
      if (vj.routePos + 1u >= pj.routeLen) { continue; }
      if (stopAt(vj, vj.routePos)) { continue; }
      let d2 = ed2.length - vj.s;
      let v2 = vj.speed;
      if (v2 < 0.3 && d2 > 3.0 && waited > 5.0) { continue; }
      let t2 = d2 / max(v2, 2.0);
      if (ed2.rank > e.rank) { if (t2 < 4.0 * patienceScale || d2 < 6.0) { return true; } }
      else if (ed2.rank == e.rank) {
        if (d2 < 60.0 || t2 < 6.0) {
          if (t2 < tMe - 2.0 * patienceScale || (abs(t2 - tMe) <= 2.0 * patienceScale && j < i) || d2 < 3.0) { return true; }
        }
      } else if (d2 < 5.0 && waited < 30.0) { return true; }
    }
  }
  return false;
}
