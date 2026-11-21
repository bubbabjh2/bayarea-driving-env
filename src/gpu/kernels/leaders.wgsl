// Leaders (sim.js _findLeaders + _laneEnd + _sceneLeader): gap / leader per driving car
// over every lane its footprint occupies; else across up to 3 route edges with the
// junction lane mapping; zipper merge (a car on another incoming edge of my next node,
// closer to it, heading into the same lane of the same next edge, goes first); lane-end
// taper wall (leader -2); active, non-squeeze scene blocks.
// GPU option FEAT_ZIPFIX: zipper leaders are flagged (F_ZIPLEAD) so collide.wgsl does not
// treat their virtual gap as a body overlap (the CPU sim currently does).
// #include common.wgsl
// #include lists.wgsl
// #include controls.wgsl
// #include yield.wgsl

@group(1) @binding(0) var<storage, read_write> veh: array<Veh>;
@group(1) @binding(1) var<storage, read> par: array<VPar>;
@group(1) @binding(2) var<storage, read_write> ctl: array<u32>;   // controls.wgsl CTL_W words per slot
@group(1) @binding(3) var<storage, read_write> inputs: array<f32>;   // obs inputs 44..62 for policy cars

fn stopAt(v: Veh, rp: u32) -> bool { return v.refuelState == 1u && v.stopPos == i32(rp); }

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= P.cap) { return; }
  let v = veh[i];
  if (v.state != DRIVING) { return; }
  let pi = par[i];
  let e = v.edge;
  { let o = i * CTL_W;
    if (ctl[o + 8u] != bitcast<u32>(v.spawnTime)) {   // new occupant of this slot: reset its control / leader2 record
      ctl[o] = bitcast<u32>(-1); ctl[o + 1u] = bitcast<u32>(-1); ctl[o + 5u] = bitcast<u32>(1e9); ctl[o + 6u] = 0u;
      ctl[o + 7u] = bitcast<u32>(1e9); ctl[o + 9u] = bitcast<u32>(-1); ctl[o + 11u] = 0u; ctl[o + 16u] = bitcast<u32>(-1); ctl[o + 20u] = bitcast<u32>(-1); ctl[o + 24u] = 0u; ctl[o + 31u] = bitcast<u32>(-1e9); ctl[o + 8u] = bitcast<u32>(v.spawnTime);
    } }
  let elen = eLength(e);
  let rp = v.routePos;
  let stopHere = stopAt(v, rp);
  let lo = vLo(v); let hi = vHi(v);
  let si = v.s; let hl = 0.5 * pi.length;
  var leader = -1;
  var gap = 1e9;
  var zip = false;
  var seen = false;
  let pw = bitcast<i32>(ctl[i * CTL_W + 20u]);   // passing a wreck through the opposing lane (jyield.wgsl)
  for (var l = lo; l <= hi; l = l + 1u) {
    var a = blocking(aheadIn(i, l), l);
    // passing: the wrecks up to the one I pass are beside me, not ahead
    for (var n = 0u; n < 8u; n = n + 1u) { if (pw < 0 || a < 0 || veh[u32(a)].state != CRASHED || veh[u32(a)].s > veh[u32(pw)].s + 0.1) { break; } a = blocking(aheadIn(u32(a), l), l); }
    if (a >= 0) { seen = true; let gg = veh[u32(a)].s - si - hl - 0.5 * par[u32(a)].length; if (gg < gap) { gap = gg; leader = a; } }
  }
  if (!seen && !stopHere) {
    var dist = elen - si;
    var prev = e;
    var plo = i32(lo); var phi = i32(hi);
    var s0 = 0.0;
    if (P.hasJ != 0u && rp + 1u < pi.routeLen) {
      // through the junction connector I am on (or will take from my lane): the next edge starts at sOut
      var m = vConn(v);
      if (m < 0 && eSStop(e) < elen) { m = jMovement(e, vLane(v), routeEdge(pi, rp + 1u)); }
      if (m >= 0) { dist = movSIn(u32(m)) + movLen(u32(m)) - si; s0 = movSOut(u32(m)); }
    }
    for (var k = 1u; k <= 3u; k = k + 1u) {
      if (rp + k >= pi.routeLen || dist >= P.leaderLookahead) { break; }
      let en = routeEdge(pi, rp + k);
      let nl = i32(eLanes(en));
      let nlo = mapLane(prev, en, plo); let nhi = mapLane(prev, en, phi);
      var found = false;
      for (var l2 = max(0, nlo); l2 <= min(nl - 1, nhi); l2 = l2 + 1) {
        let f = blocking(laneFirst(en, u32(l2)), u32(l2));
        if (f >= 0) { found = true; let gg = dist + veh[u32(f)].s - select(0.0, s0, k == 1u) - hl - 0.5 * par[u32(f)].length; if (gg < gap) { gap = gg; leader = f; } }
      }
      if (found || nhi < 0 || nlo >= nl) { break; }
      plo = max(0, nlo); phi = min(nl - 1, nhi);
      dist = dist + eLength(en) - select(0.0, s0, k == 1u); prev = en;
    }
  }
  // zipper merge (sim.js): settled ~7 s ahead at speed (a merging car can be closer than my current leader)
  let dNode = elen - si;
  if (rp + 1u < pi.routeLen && dNode < max(P.mergeLookahead, 7.0 * v.speed) && !stopHere) {
    let en = routeEdge(pi, rp + 1u);
    let myLane = mapLane(e, en, i32(vLane(v)));
    let node = eTo(e);
    for (var q = inStart(node); q < inStart(node + 1u); q = q + 1u) {
      let e2 = inEdge(q);
      if (e2 == e) { continue; }
      let l2n = eLanes(e2); let len2 = eLength(e2);
      for (var l = 0u; l < l2n; l = l + 1u) {
        // a wreck sitting at the merge point is an obstacle for everyone merging here
        let w = laneLast(e2, l);
        if (w >= 0 && veh[u32(w)].state == CRASHED && vHi(veh[u32(w)]) >= vLo(veh[u32(w)]) && len2 - veh[u32(w)].s < 15.0) {
          let gw = dNode - hl - 3.0; if (gw < gap) { gap = gw; leader = w; zip = false; }
        }
        // a ramp merging into the mainline (src/geom gore merge, right side) converges with the mainline's lane 0
        // physically before the node, whatever the lane bookkeeping says: zipper them
        let gE = eGore(e); let gE2 = eGore(e2);
        let rampMerge = (gE >= 0.0 && !(gE2 >= 0.0) && l == 0u) || (gE2 >= 0.0 && !(gE >= 0.0) && vLane(v) == 0u);
        if (!rampMerge && mapLane(e2, en, i32(l)) != myLane) { continue; }
        let jj = laneLast(e2, l);
        if (jj < 0) { continue; }
        let j = u32(jj);
        let vj = veh[j];
        if (vj.state != DRIVING) { continue; }
        let pj = par[j];
        if (vj.routePos + 1u >= pj.routeLen || routeEdge(pj, vj.routePos + 1u) != en) { continue; }
        if (stopAt(vj, vj.routePos)) { continue; }
        // it gives way here (sign on its approach, none on mine) or must stop (red / stop sign): it does not go first
        if ((signedL(j) && !signedL(i)) || mustStopL(j)) { continue; }
        // a ramp car that has not passed its gore (wait) point is not in the mainline lane yet: the ramp yields
        if (gE2 >= 0.0 && gE < 0.0 && vj.s + 0.5 * pj.length < gE2 + 0.5) { continue; }
        let dj = len2 - vj.s;
        // I give way here (sign on my approach, none on its): wait at my line while it arrives within 5 s
        if (signedL(i) && !signedL(j)) {
          let cd = bitcast<f32>(ctl[i * CTL_W + 5u]);
          if (cd > -0.5 && cd < 40.0) {
            if (dj / max(vj.speed, 2.0) < 5.0) { let gl = cd - 0.5; if (gl < gap) { gap = gl; leader = -3; zip = false; ctl[i * CTL_W + 28u] = j; } }
            continue;
          }
        }
        if (dj < dNode || (dj == dNode && j < i)) {
          let gz = dNode - dj - hl - 0.5 * pj.length;
          // a ramp merging into the mainline's lanes (no added lane, src/geom J.gore): without room to tuck in
          // behind that car, wait at the gore point, outside the mainline lane (sim.js, layout v8)
          let gore = eGore(e);
          if (gore >= 0.0 && l2n >= eLanes(e) && si + hl < gore + 0.5 && gz < 2.0 + v.speed) {
            let gg = gore - si - hl - 0.5; if (gg < gap) { gap = gg; leader = -3; zip = false; }
          } else if (gz < gap) { gap = gz; leader = i32(j); zip = true; }
        }
      }
    }
  }
  // the car in front of my leader (sim.js leader2 / gap2: brake lights two cars ahead, a hidden wreck)
  var leader2 = -1; var gap2 = 1e9;
  if (leader >= 0) {
    let vl = veh[u32(leader)];
    if (vl.edge == e && vHi(vl) >= vLo(vl)) {
      let l2 = blocking(aheadIn(u32(leader), vLane(vl)), vLane(vl));
      if (l2 >= 0 && u32(l2) != i) { leader2 = l2; gap2 = veh[u32(l2)].s - si - hl - 0.5 * par[u32(l2)].length; }
    }
  }
  ctl[i * CTL_W + 9u] = bitcast<u32>(leader2); ctl[i * CTL_W + 10u] = bitcast<u32>(gap2);
  // lane ends (sim.js _laneEnd, layout v4): with src/geom the ended lane narrows over the last
  // 90 m BEFORE the node; obstacle -2 at the taper start - MERGE_ROOM, or at the node once merging
  var dEnd = -1.0; var side = 0;
  let W = laneWAt(e, si);
  let nl0 = f32(eLanes(e));
  if (vConn(v) >= 0) { }   // on a junction connector: its exit lane exists by construction (sim.js _laneEnd)
  else if (P.hasLG == 0u && v.taper > 0.0 && (v.lat < 0.0 || v.lat > nl0 * W)) { dEnd = v.taper - si; side = select(-1, 1, v.lat < 0.0); }
  else if (!stopHere) {
    // follow every lane of my edge through the hops ahead; my lane "ends" only if another lane
    // of THIS edge survives (otherwise the change has to happen on a later edge: mergeCap)
    let lanesHere = min(eLanes(e), 16u);
    var alive = (1u << lanesHere) - 1u;
    var cur: array<i32, 16>;
    for (var l = 0u; l < lanesHere; l = l + 1u) { cur[l] = i32(l); }
    let myLane = min(vLane(v), 15u);
    var mergeCap = 1e9;
    var prev2 = e; var dist2 = elen - si; var endK = 0u; var myL = i32(myLane);
    for (var k = 1u; k <= 4u; k = k + 1u) {
      if (rp + k >= pi.routeLen || dist2 >= P.laneEndLookahead) { break; }
      let en = routeEdge(pi, rp + k);
      let nen = i32(eLanes(en));
      var any = false;
      for (var l = 0u; l < lanesHere; l = l + 1u) {
        if (((alive >> l) & 1u) == 0u) { continue; }
        let l2 = mapLane(prev2, en, cur[l]);
        if (l2 < 0 || l2 >= nen) { alive = alive & ~(1u << l); } else { cur[l] = l2; any = true; }
      }
      let myAlive = ((alive >> myLane) & 1u) != 0u;
      if (!myAlive && !any && k > 1u) {
        // no lane of my edge survives: cross the lanes on `prev2` at ~1.3 m/s lateral within its length
        var need = 1e9;
        for (var l = 0u; l < eLanes(prev2); l = l + 1u) { let l2 = mapLane(prev2, en, i32(l)); if (l2 >= 0 && l2 < nen) { need = min(need, abs(f32(i32(l) - myL))); } }
        if (need > 0.0 && need < 1e9) { let room = max(10.0, eLength(prev2) - 10.0); mergeCap = max(5.0, room / (need * laneWAt(prev2, eLength(prev2)) / 1.3)); }
        break;
      }
      if (!myAlive) { if (any) { endK = k; } break; }
      myL = mapLane(prev2, en, myL);
      if (stopAt(v, rp + k)) { break; }
      dist2 = dist2 + eLength(en); prev2 = en;
    }
    ctl[i * CTL_W + 7u] = bitcast<u32>(mergeCap);
    if (endK > 0u) {
      var best = -1;
      for (var l = 0u; l < lanesHere; l = l + 1u) { if (((alive >> l) & 1u) != 0u && (best < 0 || abs(i32(l) - i32(myLane)) < abs(best - i32(myLane)))) { best = i32(l); } }
      side = select(-1, 1, best > i32(myLane));
      dEnd = dist2;   // src/sim/laneEnds.js: m to the node where my lane ends (lanes are paved to the node)
    }
  }
  var endDist = 1e9;
  if (dEnd >= 0.0) {
    endDist = dEnd;
    let tl = vTarget(v);
    let merging = (tl >= 0 && tl != i32(vLane(v)) && (v.flags & F_POLICY) == 0u) || v.headingRel * f32(side) > 0.04;
    let atNode = dEnd + select(0.0, LANE_END_TAPER, P.hasLG != 0u);
    let gg = select(dEnd - MERGE_ROOM, atNode, merging) - hl - 1.0;
    if (gg < gap) { gap = gg; leader = -2; zip = false; }
  } else { side = 0; }
  // sim.js _offLaneBodies: if my body reaches off the lanes, same-edge bodies entirely off the lanes ahead of me
  // that overlap laterally are obstacles (pulled-over cars / towed wrecks on the shoulder)
  if (P.hasJ != 0u) {
    let heI = halfExtent(v, pi); let Wl = laneWAt(e, si);
    if (!(v.lat - heI > 0.05 && v.lat + heI < f32(eLanes(e)) * Wl - 0.05)) {
      var j = ctl[P.jHeadOff + P.jM + P.E + (P.E + 31u) / 32u + e];
      for (var gq = 0u; gq < 256u; gq = gq + 1u) {
        if (j == NONE) { break; }
        if (j != i) {
          let vj = veh[j];
          if (vj.s > si && abs(vj.lat - v.lat) <= heI + halfExtent(vj, par[j]) + 0.3) {
            let gg = vj.s - si - hl - 0.5 * par[j].length;
            if (gg < gap) { gap = gg; leader = i32(j); zip = false; }
          }
        }
        j = ctl[j * CTL_W + 30u];
      }
    }
  }
  veh[i].laneEndDist = endDist;
  veh[i].laneEndSide = side;
  // scene obstacles on my edge in one of my lanes (not while passing a wreck)
  var b = select(blockHead(e), 0u, pw >= 0);
  loop {
    if (b == 0u) { break; }
    let r = blockAt(b - 1u);
    if (r.state == 1u && (r.flags & 1u) == 0u && r.s > si) {
      var hit = false;
      for (var l3 = lo; l3 <= hi; l3 = l3 + 1u) { if (((r.laneMask >> l3) & 1u) != 0u) { hit = true; } }
      if (hit) { let gg = r.s - si - hl - 3.0; if (gg < gap) { gap = gg; leader = i32(r.lead); zip = false; } }
    }
    b = r.next;
  }
  // converging carriageways (sim.js _convergeBodies): bodies on the other in-edges of my end node that continue
  // onto my next edge, ahead of me in my heading frame and within my body width, are followed
  if (vConn(v) < 0 && !stopHere) { let cb = convergeBodies(i, v, pi, gap); if (cb.j >= 0 && cb.gap < gap) { gap = cb.gap; leader = cb.j; zip = false; } }
  // standoff from a stopped obstacle (sim.js, after all perception; with junctions in junction.wgsl jperceive)
  if (P.hasJ == 0u && leader >= 0 && veh[u32(leader)].state == CRASHED && veh[u32(leader)].edge == e) {
    let L = u32(leader); let hI = halfExtent(v, pi); let hL = halfExtent(veh[L], par[L]);
    let ov = min(v.lat + hI, veh[L].lat + hL) - max(v.lat - hI, veh[L].lat - hL);
    if (ov > 0.0) { gap = gap - min(MERGE_ROOM, 2.0 + 4.0 * (ov + 0.3)); }
  }
  veh[i].leader = leader;
  veh[i].gap = gap;
  if (P.hasCtl != 0u) { controlObs(i, v, pi); }
  if (hasFeat(FEAT_POLICY) && (v.flags & F_POLICY) != 0u) { policyInputs(i, endDist, side); }
  veh[i].flags = select(v.flags & ~F_ZIPLEAD, v.flags | F_ZIPLEAD, zip && hasFeat(FEAT_ZIPFIX));
}

// ---------------------------------------------------------------- traffic controls (sim.js _controlObs)
struct ConvRes { j: i32, gap: f32 };
/** sim.js _convergeBodies -> (leader slot, gap); gap0 if none closer */
fn convergeBodies(i: u32, v: Veh, pi: VPar, gap0: f32) -> ConvRes {
  var res = ConvRes(-1, gap0);
  let e = v.edge; let dNode = eLength(e) - v.s; if (dNode > 150.0) { return res; }
  let rp = v.routePos; if (rp + 1u >= pi.routeLen) { return res; }
  let nx = routeEdge(pi, rp + 1u); let node = eTo(e);
  if (inStart(node + 1u) - inStart(node) < 2u || hopMap(e, nx).x != 1) { return res; }
  let me = poseOf(v); let fx = cos(me.w); let fz = sin(me.w); let hl = 0.5 * pi.length; let hw = 0.5 * pi.width;
  for (var q = inStart(node); q < inStart(node + 1u); q = q + 1u) {
    let e2 = inEdge(q); if (e2 == e || i32(e2) == eRev(nx)) { continue; }
    let len2 = eLength(e2); let strict2 = hopMap(e2, nx).x == 1;
    for (var l = 0u; l < eLanes(e2); l = l + 1u) {
      var jj = laneLast(e2, l); var n = 0u;
      loop {
        if (jj < 0 || n >= 4u) { break; }
        n = n + 1u;
        let j = u32(jj); let vj = veh[j];
        if (len2 - vj.s > dNode + 60.0) { break; }
        let own = ctlOwnL(j); let ct = select(-1, bitcast<i32>(ctl[j * CTL_W + 1u]), own); let cd = bitcast<f32>(ctl[j * CTL_W + 5u]); let cs = ctl[j * CTL_W + 2u];
        let waitingAtLine = vj.state == DRIVING && vj.speed < 1.0 && ct >= 0 && cd > -0.5 && cd < 15.0 && (cs == 2u || cs == 3u);
        var cont = false;
        if (vj.state == DRIVING) { let pj = par[j]; cont = vj.routePos + 1u < pj.routeLen && routeEdge(pj, vj.routePos + 1u) == nx && strict2; }
        if (!waitingAtLine && (vj.state == CRASHED || cont)) {
          let pjw = poseOf(vj); let dx = pjw.x - me.x; let dz = pjw.z - me.z;
          let along = dx * fx + dz * fz;
          if (along > 0.0 && along < 80.0) {
            let lat = dx * fz - dz * fx;
            var dpsi = pjw.w - me.w; dpsi = atan2(sin(dpsi), cos(dpsi));
            let sa = abs(sin(dpsi)); let ca = abs(cos(dpsi)); let hlj = 0.5 * par[j].length; let hwj = 0.5 * par[j].width;
            if (abs(dpsi) < 0.5 && abs(lat) < hw + sa * hlj + ca * hwj + 0.4) {
              let gg = along - hl - (ca * hlj + sa * hwj);
              if (gg < res.gap) { res = ConvRes(i32(j), gg); }
            }
          }
        }
        jj = behindIn(j, l);
      }
    }
  }
  return res;
}
/** sim.js _signed / _mustStop from the control record of slot j (previous step's controlObs, as the CPU order) */
fn ctlOwnL(j: u32) -> bool { return ctl[j * CTL_W + 8u] == bitcast<u32>(veh[j].spawnTime); }
fn signedL(j: u32) -> bool { if (P.hasCtl == 0u || !ctlOwnL(j)) { return false; } let t = bitcast<i32>(ctl[j * CTL_W + 1u]); return (t == 1 || t == 2) && bitcast<f32>(ctl[j * CTL_W + 5u]) < 60.0; }
fn mustStopL(j: u32) -> bool {
  if (P.hasCtl == 0u || !ctlOwnL(j)) { return false; }
  let t = bitcast<i32>(ctl[j * CTL_W + 1u]);
  if (t < 0 || bitcast<f32>(ctl[j * CTL_W + 5u]) > 60.0) { return false; }
  return ctl[j * CTL_W + 2u] == 2u || (t == 1 && ((ctl[j * CTL_W + 6u] >> 8u) & 1u) == 0u);
}
fn controlObs(i: u32, v: Veh, pi: VPar) {
  let o = i * CTL_W;
  let rp = v.routePos; let hl = 0.5 * pi.length;
  let now = simTime();
  // stop lines of the signalised intersection I am already inside (a divided road's inner stop lines,
  // same controller cluster) belong to the movement I entered on: skip them
  let inside = bitcast<i32>(ctl[o + 16u]);
  var a = -1; var dist = 0.0; var k = rp;
  for (var q = edgeApBegin(v.edge); q < edgeApBegin(v.edge + 1u); q = q + 1u) {
    let x = edgeApAt(q);
    if (apStopAt(x) - (v.s + hl) > -1.0 && !(inside >= 0 && clusterOf(apControl(x)) == u32(inside))) { a = i32(x); dist = apStopAt(x) - (v.s + hl); break; }
  }
  if (a < 0) {
    var d = eLength(v.edge) - v.s - hl; var s0 = 0.0;
    if (P.hasJ != 0u && rp + 1u < pi.routeLen) {
      // through the junction connector (mine, or the one my lane takes): the next edge starts at sOut (sim.js)
      var m = vConn(v); if (m < 0 && eSStop(v.edge) < eLength(v.edge)) { m = jMovement(v.edge, vLane(v), routeEdge(pi, rp + 1u)); }
      if (m >= 0) { d = movSIn(u32(m)) + movLen(u32(m)) - v.s - hl; s0 = movSOut(u32(m)); }
    }
    k = rp + 1u;
    loop {
      if (k >= pi.routeLen || d >= P.ctlLook) { break; }
      let en = routeEdge(pi, k); let off = select(0.0, s0, k == rp + 1u);
      for (var q = edgeApBegin(en); q < edgeApBegin(en + 1u); q = q + 1u) {
        let x = edgeApAt(q);
        if (!(inside >= 0 && clusterOf(apControl(x)) == u32(inside))) { a = i32(x); dist = d + apStopAt(x) - off; break; }
      }
      if (a >= 0) { break; }
      d = d + eLength(en) - off; k = k + 1u;
    }
  }
  let prevA = bitcast<i32>(ctl[o]); let prevDist = bitcast<f32>(ctl[o + 5u]);
  let mvw = ctl[o + 6u]; var stopped = (mvw >> 8u) & 1u;
  // the front bumper crossed the stop line this step (once per approach), not inside its own cluster
  if (prevA >= 0 && prevDist >= 0.0 && prevDist < 30.0 && (a != prevA || dist < 0.0) && !(inside >= 0 && clusterOf(apControl(u32(prevA))) == u32(inside))) {
    let st = ctl[o + 2u]; let ty = bitcast<i32>(ctl[o + 1u]);
    let kmh10 = min(u32(v.speed * 10.0), 65535u) << 8u;
    if ((ty == CTL_SIGNAL || ty == CTL_RAMP || ty == CTL_CROSSING) && st == LIGHT_RED && bitcast<f32>(ctl[o + 3u]) > 0.5 && !((mvw & 255u) == MOVE_RIGHT && stopped != 0u)) {
      emit(EV_CTLVIOL, i, u32(prevA), 1u | kmh10);
    } else if (ty == CTL_STOP && stopped == 0u) {
      emit(EV_CTLVIOL, i, u32(prevA), 2u | kmh10);
    }
  }
  var ins = inside;
  if (a != prevA) {
    stopped = 0u;
    // crossed a signal's stop line: inside its intersection (cluster) for 15 s
    if (prevA >= 0 && prevDist >= -1.0 && prevDist < 30.0 && i32(world[P.ctlCOff + apControl(u32(prevA)) * CTL_REC]) == CTL_SIGNAL) { ins = i32(clusterOf(apControl(u32(prevA)))); ctl[o + 17u] = bitcast<u32>(now); }
  }
  if (ins >= 0 && now - bitcast<f32>(ctl[o + 17u]) > 15.0) { ins = -1; }
  ctl[o + 16u] = bitcast<u32>(ins);
  ctl[o] = bitcast<u32>(a);
  if (a < 0) { ctl[o + 1u] = bitcast<u32>(-1); ctl[o + 5u] = bitcast<u32>(1e9); ctl[o + 6u] = stopped << 8u; return; }
  let appE = apEdge(u32(a));
  var mv = MOVE_THROUGH;
  let pos = select(k, rp, appE == v.edge);
  if (pos + 1u < pi.routeLen) { mv = movementOf(appE, routeEdge(pi, pos + 1u)); }
  let so = ctlStateOf(u32(a), mv, now);
  ctl[o + 1u] = bitcast<u32>(so.ty); ctl[o + 2u] = so.state; ctl[o + 3u] = bitcast<u32>(so.timeIn); ctl[o + 4u] = bitcast<u32>(so.timeLeft);
  ctl[o + 5u] = bitcast<u32>(dist);
  if (dist < 10.0 && v.speed < 0.3) { stopped = 1u; }   // sim.js: the stop-line controller holds a stopped car with the line < 9.5 m ahead
  ctl[o + 6u] = mv | (stopped << 8u);
}

// ---------------------------------------------------------------- obs inputs from this stage (44..62)
// 44 own lane end min(d / 300, 1), 45 side (adjust.js with the sim's laneEndDist / Side);
// 46..62 next traffic control (src/sim/obs.js obs.control -> CONTROL_OFFSET block)
fn policyInputs(i: u32, endDist: f32, side: i32) {
  let b = i * INS;
  inputs[b + OFF_LANEEND] = select(1.0, clamp(endDist / 300.0, 0.0, 1.0), endDist < 1e8);
  inputs[b + OFF_LANEEND + 1u] = select(0.0, f32(side), endDist < 1e8);
  let c = b + OFF_CTL;
  for (var k = 0u; k < 17u; k = k + 1u) { inputs[c + k] = 0.0; }
  let o = i * CTL_W;
  let v = veh[i];
  let ct = select(-1, bitcast<i32>(ctl[o + 1u]), P.hasCtl != 0u && ctl[o + 8u] == bitcast<u32>(v.spawnTime));
  let cd = bitcast<f32>(ctl[o + 5u]);
  if (ct >= 0 && ct <= 4 && cd < 250.0) {
    inputs[c + u32(ct)] = 1.0;
    let st = ctl[o + 2u]; if (st <= 4u) { inputs[c + 5u + st] = 1.0; }
    inputs[c + 10u] = clamp(cd / 250.0, 0.0, 1.0);
    let tl = bitcast<f32>(ctl[o + 4u]);
    inputs[c + 11u] = select(0.0, clamp(tl / 30.0, 0.0, 1.0), tl > 0.0);   // > 0 && < Infinity (signs: 1e9 -> 1)
    let mv = ctl[o + 6u] & 255u; if (mv <= 2u) { inputs[c + 12u + mv] = 1.0; }
    inputs[c + 15u] = select(0.0, 1.0, ((ctl[o + 6u] >> 8u) & 1u) != 0u);
    let e = edgeAt(v.edge);
    inputs[c + 16u] = select(0.0, 1.0, yieldRequired(i, e, v.edge, e.length - v.s));
  } else { inputs[c + 10u] = 1.0; }
}
