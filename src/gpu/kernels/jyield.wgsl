// Junction entry decisions (layout v8, sim.js _yieldJ / _boxBusyJ), after junction.wgsl jperceive:
// ctl word 19 bit 0 = must wait at the mouth (yieldRequired), bit 1 = a crossing body is still in the box
// (boxBusy, for a green); policy cars also get obs input 62 (CONTROL_OFFSET + 16) from bit 0.
// #include common.wgsl
// #include lists.wgsl

@group(1) @binding(0) var<storage, read_write> veh: array<Veh>;
@group(1) @binding(1) var<storage, read> par: array<VPar>;
@group(1) @binding(2) var<storage, read_write> ctl: array<u32>;
@group(1) @binding(3) var<storage, read_write> inputs: array<f32>;

const CW: u32 = CTL_W;
fn headOf(m: u32) -> u32 { return ctl[P.jHeadOff + m]; }
fn nextOf(j: u32) -> u32 { return ctl[j * CW + 18u]; }
fn hlOf(j: u32) -> f32 { return 0.5 * par[j].length; }
fn hwOf(j: u32) -> f32 { return 0.5 * par[j].width; }
fn ctlOwn(j: u32) -> bool { return ctl[j * CW + 8u] == bitcast<u32>(veh[j].spawnTime); }
/** sim.js _signed: a sign (stop / give way) governs slot j's approach to the junction ahead */
fn signedJ(j: u32) -> bool { if (P.hasCtl == 0u || !ctlOwn(j)) { return false; } let t = bitcast<i32>(ctl[j * CW + 1u]); return (t == 1 || t == 2) && bitcast<f32>(ctl[j * CW + 5u]) < 60.0; }
/** sim.js _mustStop: slot j's own control makes it stop before the junction (red, or a stop sign not yet stopped at) */
fn mustStopJ(j: u32) -> bool {
  if (P.hasCtl == 0u || !ctlOwn(j)) { return false; }
  let t = bitcast<i32>(ctl[j * CW + 1u]);
  if (t < 0 || bitcast<f32>(ctl[j * CW + 5u]) > 60.0) { return false; }
  return ctl[j * CW + 2u] == 2u || (t == 1 && ((ctl[j * CW + 6u] >> 8u) & 1u) == 0u);
}
fn stopAtJ(v: Veh, rp: u32) -> bool { return v.refuelState == 1u && v.stopPos == i32(rp); }

var<private> ybJ: i32;   // the car the last yieldJ waits for (-1: none), ctl word 32 (sim.js _yieldBy)
/** sim.js _seeApproach ok(): car j can come through movement (.., oOut), rp1 route steps ahead */
fn okJ(i: u32, j: u32, rp1: u32, oOut: u32) -> bool {
  if (j == i) { return false; }
  let vj = veh[j]; let pj = par[j];
  if (vj.state != DRIVING || stopAtJ(vj, vj.routePos)) { return false; }
  return vj.routePos + rp1 < pj.routeLen && routeEdge(pj, vj.routePos + rp1) == oOut;
}
/** sim.js _yieldJ first(): does approaching car j (d2 m from its mouth) have priority over me and reach our conflict soon? */
fn prioJ(i: u32, e: u32, sp: f32, dS: f32, left: bool, myStop: bool, cs: f32, csO: f32, ctype: u32, o: u32, j: u32, d2: f32) -> bool {
  let vj = veh[j]; let v2 = vj.speed; let oIn = movInEdge(o);
  if (d2 > 80.0 || mustStopJ(j)) { return false; }
  // standing at its mouth held by an obstacle right in front of it: it cannot enter, no conflict until it moves
  if (v2 < 0.3 && vj.leader != -1 && vj.gap < d2 + 3.0) { return false; }
  let tI = (dS + cs) / max(sp, 2.0); let tJ = (d2 + csO) / max(v2, 2.0);
  var jFirst = false;
  let jStop = signedJ(j);
  let r2 = world[oIn * 16u + 9u]; let r1 = world[e * 16u + 9u];
  if (myStop != jStop) { jFirst = myStop; }                               // a sign gives way to the uncontrolled road
  else if (r2 != r1) { jFirst = r2 > r1; }
  else if (ctype == 0u && left != (movTurn(o) < -0.5)) { jFirst = left; }   // a left turn gives way
  else { jFirst = tJ < tI || (tJ == tI && j < i); }
  return jFirst && (tJ < 8.0 || d2 < 10.0);
}

fn yieldJ(i: u32, v: Veh, pi: VPar, mu: u32) -> bool {
  let e = v.edge;
  let hl = hlOf(i); let dS = max(0.0, min(eSStop(e), movSIn(mu)) - v.s - hl); let sp = v.speed;
  let L0 = v.leader;
  if (L0 >= 0) {
    let vl = veh[u32(L0)];
    if (vl.state == DRIVING && vl.speed < 1.0 && (vl.edge != e || vConn(vl) >= 0) && v.gap - dS - movLen(mu) < pi.length + 1.0) { ybJ = L0; return true; }
  }
  let left = movTurn(mu) < -0.5; let myStop = signedJ(i);
  for (var k = jConfBegin(mu); k < jConfBegin(mu + 1u); k = k + 1u) {
    let c = jConfAt(k); let o = c.x; let ctype = c.y; let cs = bitcast<f32>(c.z); let csO = bitcast<f32>(c.w);
    if (ctype == 0u) {
      var j = headOf(o);
      for (var gq = 0u; gq < 1024u; gq = gq + 1u) { if (j == NONE) { break; } if (veh[j].s - movSIn(o) - hlOf(j) < csO + hwOf(i) + 1.0) { ybJ = i32(j); return true; } j = nextOf(j); }
    }
    let oIn = movInEdge(o); let oLane = movInLane(o); let oOut = movOutEdge(o);
    if (P.jSight == 0u) {
      var jj = laneLast(oIn, oLane);
      for (var g = 0u; g < 64u; g = g + 1u) { if (jj < 0 || vConn(veh[u32(jj)]) < 0) { break; } jj = behindIn(u32(jj), oLane); }
      if (jj < 0 || u32(jj) == i) { continue; }
      let j = u32(jj);
      if (!okJ(i, j, 1u, oOut)) { continue; }
      if (prioJ(i, e, sp, dS, left, myStop, cs, csO, ctype, o, j, max(0.0, eSStop(oIn) - veh[j].s - hlOf(j)))) { ybJ = i32(j); return true; }
      continue;
    }
    // junctionSight 1 (sim.js _seeApproach): the first 3 cars of the lane that take o, then upstream of a short approach
    var jj = laneLast(oIn, oLane); var n = 0u; var far = false;
    for (var g = 0u; g < 64u; g = g + 1u) {
      if (jj < 0 || n >= 3u) { break; }
      let j = u32(jj);
      if (vConn(veh[j]) < 0) {
        let d2 = max(0.0, eSStop(oIn) - veh[j].s - hlOf(j));
        if (d2 > 80.0) { far = true; break; }
        n = n + 1u;
        if (okJ(i, j, 1u, oOut) && prioJ(i, e, sp, dS, left, myStop, cs, csO, ctype, o, j, d2)) { ybJ = i32(j); return true; }
      }
      jj = behindIn(j, oLane);
    }
    let sE = eSStop(oIn);
    if (far || !(sE < 80.0)) { continue; }
    for (var q = jIntoBegin(oIn); q < jIntoBegin(oIn + 1u); q = q + 1u) {
      let m2 = jIntoAt(q); let len2 = movLen(m2); let sIn2 = movSIn(m2); let base = max(0.0, sE - movSOut(m2));
      var k = headOf(m2);
      for (var gq = 0u; gq < 1024u; gq = gq + 1u) {
        if (k == NONE) { break; }
        let d2 = base + max(0.0, sIn2 + len2 - veh[k].s) - hlOf(k);
        if (d2 <= 80.0 && okJ(i, k, 2u, oOut) && prioJ(i, e, sp, dS, left, myStop, cs, csO, ctype, o, k, max(0.0, d2))) { ybJ = i32(k); return true; }
        k = nextOf(k);
      }
      let inE = movInEdge(m2); let inL = movInLane(m2);
      var qq = laneLast(inE, inL); var c = 0u;
      for (var g2 = 0u; g2 < 64u; g2 = g2 + 1u) {
        if (qq < 0 || c >= 2u) { break; }
        let qj = u32(qq); let vq = veh[qj];
        if (vConn(vq) < 0) {
          let d2 = base + len2 + max(0.0, sIn2 - vq.s) - hlOf(qj);
          if (d2 > 80.0) { break; }
          c = c + 1u;
          let pq = par[qj];
          if (vq.routePos + 1u < pq.routeLen && routeEdge(pq, vq.routePos + 1u) == oIn && okJ(i, qj, 2u, oOut) && !mustStopJ(qj) && prioJ(i, e, sp, dS, left, myStop, cs, csO, ctype, o, qj, d2)) { ybJ = i32(qj); return true; }
        }
        qq = behindIn(qj, inL);
      }
    }
  }
  // through traffic on a continuous carriageway (strict hop, no connector) crossing the box: judged directly.
  // Body in the box = always wait; approaching with priority (rank >= mine, no sign for it, not stopping) and
  // under 6 s from its mouth = wait
  let node = eTo(e); let myOut = routeEdge(pi, v.routePos + 1u); let myRev = eRev(myOut); let rI = world[e * 16u + 9u];
  for (var q = inStart(node); q < inStart(node + 1u); q = q + 1u) {
    let o = inEdge(q);
    if (o == e || i32(o) == myRev) { continue; }
    let oLen = eLength(o); let oStop = eSStop(o); let rO = world[o * 16u + 9u];
    for (var l = 0u; l < eLanes(o); l = l + 1u) {
      var jj = laneLast(o, l); var n = 0u;
      loop {
        if (jj < 0 || n >= 3u) { break; }
        n = n + 1u;
        let j = u32(jj); let vj = veh[j];
        if (vj.state == DRIVING && vConn(vj) < 0) {
          let pj = par[j];
          if (vj.routePos + 1u < pj.routeLen) {
            let nj = routeEdge(pj, vj.routePos + 1u);
            if (nj != myOut && jMovement(o, l, nj) < 0) {
              let hlj = hlOf(j); let dj = oLen - vj.s - hlj;
              if (vj.s + hlj > oStop && dj > -pj.length) { ybJ = i32(j); return true; }
              if (rO >= rI && !signedJ(j) && !mustStopJ(j) && !myStop) {
                if ((oStop - vj.s - hlj) / max(vj.speed, 2.0) < 6.0) { ybJ = i32(j); return true; }
              }
            }
          }
          break;
        }
        jj = behindIn(j, l);
      }
    }
  }
  return false;
}
fn boxBusyJ(i: u32, mu: u32) -> bool {
  for (var k = jConfBegin(mu); k < jConfBegin(mu + 1u); k = k + 1u) {
    let c = jConfAt(k); if (c.y != 0u) { continue; }
    let o = c.x; let csO = bitcast<f32>(c.w);
    var j = headOf(o);
    for (var gq = 0u; gq < 1024u; gq = gq + 1u) { if (j == NONE) { break; } if (veh[j].s - movSIn(o) - hlOf(j) < csO + hwOf(i) + 1.0) { return true; } j = nextOf(j); }
  }
  return false;
}

fn passWreckOf(i: u32) -> i32 { if (ctl[i * CW + 8u] != bitcast<u32>(veh[i].spawnTime)) { return -1; } return bitcast<i32>(ctl[i * CW + 20u]); }
fn passHead(e: u32) -> u32 { return ctl[P.jHeadOff + P.jM + e]; }
fn passNext(j: u32) -> u32 { return ctl[j * CW + 21u]; }
/** sim.js _opposing: opposing-lane interaction on a two-way edge (passing wrecks on two-lane roads) */
fn opposing(i: u32) {
  let v = veh[i]; let e = v.edge; let er = u32(eRev(e)); let li = eLanes(er) - 1u;
  let hl = hlOf(i); let si = v.s; let len = eLength(e);
  var gap = v.gap; var leader = v.leader;
  // cars passing a wreck in the opposing lane are obstacles when my body is in my inner lane
  if (vHi(v) == eLanes(e) - 1u && vHi(v) >= vLo(v)) {
    var p = passHead(er);
    for (var gq = 0u; gq < 1024u; gq = gq + 1u) { if (p == NONE) { break; } let sp = len - veh[p].s; if (sp > si) { let gg = sp - si - hl - hlOf(p) - 2.0; if (gg < gap) { gap = gg; leader = -3; } } p = passNext(p); }
  }
  let pw = passWreckOf(i);
  let k0 = starts[lkey(er, li)]; let k1 = starts[lkey(er, li) + 1u];
  if (pw >= 0) {
    let w = veh[u32(pw)];
    let back = abs(v.lat - (f32(vLane(v)) + 0.5) * laneWAt(e, si)) < 0.6 && vHi(v) >= vLo(v);
    if (w.state != CRASHED || w.edge != e || (si - hl > w.s + hlOf(u32(pw)) + 2.0 && back)) { ctl[i * CW + 20u] = bitcast<u32>(-1); veh[i].gap = gap; veh[i].leader = leader; return; }
    for (var k = k0; k < k1; k = k + 1u) { let j = order[k]; let sj = len - veh[j].s; if (sj > si) { let gg = sj - si - hl - hlOf(j) - 2.0; if (gg < gap) { gap = gg; leader = -3; } } }
    veh[i].gap = gap; veh[i].leader = leader; return;
  }
  veh[i].gap = gap; veh[i].leader = leader;
  let L = leader;
  if (P.passWrecks == 0u) { return; }   // sim.js opts.passWrecks (off by default)
  if (eLanes(e) != 1u || eLanes(er) != 1u || L < 0 || veh[u32(L)].state != CRASHED || veh[u32(L)].edge != e || vConn(v) >= 0) { return; }
  if (gap > 15.0 || v.speed > 1.0 || v.stoppedSince < 0.0 || simTime() - v.stoppedSince < 5.0) { return; }
  // the wrecks in a row ahead (within 15 m of each other) are passed together
  var k = u32(L);
  for (var n = 0u; n < 6u; n = n + 1u) {
    let vk = veh[k];
    if (vHi(vk) < vLo(vk)) { break; }
    let q = aheadIn(k, clamp(vLane(vk), vLo(vk), vHi(vk)));
    if (q < 0 || veh[u32(q)].state != CRASHED || veh[u32(q)].s - vk.s > 15.0) { break; }
    k = u32(q);
  }
  let zoneEnd = veh[k].s + hlOf(k) + 8.0; let zoneStart = si - hl - 5.0;
  var limit = len; if (eSStop(e) < len) { limit = eSStop(e); }
  if (zoneEnd + 10.0 > limit) { return; }
  let Tp = (zoneEnd - zoneStart) / 4.0 + 5.0;
  for (var q = k0; q < k1; q = q + 1u) { let j = order[q]; let sj = len - veh[j].s; if (sj > zoneStart && sj - zoneEnd < veh[j].speed * Tp + 25.0) { return; } }
  var p = passHead(e);
  for (var gq = 0u; gq < 1024u; gq = gq + 1u) { if (p == NONE) { break; } if (p != i && veh[p].s > zoneStart - 10.0) { return; } p = passNext(p); }
  ctl[i * CW + 20u] = k;
  statAdd(S_WRECK_PASSES, 1u);
}

// obs inputs 65..69 (JUNCTION_OFFSET, sim.js junctionObs): the conflicting movement body nearest along my
// path through the box (distance to the conflict point, its time to it, stopped), inBox, distance to the mouth.
// (Before 2026-10-23 nothing on the GPU wrote these: every GPU-driven net saw 0 = "conflict right here" all day.)
fn junctionInputs(i: u32, v: Veh, pi: VPar) {
  var mouth = 1e9; var dist = 1e9; var ttc = 1e9; var stopped = false;
  var m = vConn(v);
  if (m < 0 && eSStop(v.edge) < eLength(v.edge) && v.routePos + 1u < pi.routeLen) { m = jMovement(v.edge, vLane(v), routeEdge(pi, v.routePos + 1u)); }
  if (m >= 0) {
    let mu = u32(m); let ti = v.s - movSIn(mu); let hl = hlOf(i); let hw = hwOf(i);
    mouth = max(0.0, -ti - hl);
    for (var k = jConfBegin(mu); k < jConfBegin(mu + 1u); k = k + 1u) {
      let c = jConfAt(k); let cs = bitcast<f32>(c.z); let csO = bitcast<f32>(c.w);
      let di = cs - ti - hl; if (di < -2.0 * hl) { continue; }
      var j = headOf(c.x);
      for (var it = 0u; it < 256u; it = it + 1u) {
        if (j == NONE) { break; }
        let vj = veh[j];
        let dj = csO - (vj.s - movSIn(c.x));
        if (dj >= -hlOf(j) - hw - 0.5) {
          let onIt = abs(dj) < hlOf(j) + hw + 0.5;
          if (di < dist) { dist = max(0.0, di); ttc = select(dj / max(0.1, vj.speed), 0.0, onIt); stopped = vj.state != DRIVING || vj.speed < 0.5; }
        }
        j = nextOf(j);
      }
    }
  }
  let b = i * INS + OFF_KINK + 2u;
  inputs[b] = select(1.0, clamp(dist / 60.0, 0.0, 1.0), dist < 1e8);
  inputs[b + 1u] = select(1.0, clamp(ttc / 10.0, 0.0, 1.0), dist < 1e8 && ttc < 1e8);
  inputs[b + 2u] = select(0.0, 1.0, dist < 1e8 && stopped);
  inputs[b + 3u] = select(0.0, 1.0, vConn(v) >= 0);
  inputs[b + 4u] = select(1.0, clamp(mouth / 60.0, 0.0, 1.0), mouth < 1e8);
}

@compute @workgroup_size(128)
fn jyield(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= P.cap) { return; }
  let v = veh[i];
  if (v.state != DRIVING) { return; }
  if (eRev(v.edge) >= 0) { opposing(i); }
  let v2 = veh[i];
  var bits = 0u;
  let pi = par[i]; let e = v.edge;
  ybJ = -1; var yRaw = false;
  if (vConn(v) < 0 && eSStop(e) < eLength(e) && v.routePos + 1u < pi.routeLen) {
    let m = jMovement(e, vLane(v), routeEdge(pi, v.routePos + 1u));
    if (m >= 0) {
      var y = yieldJ(i, v2, pi, u32(m)); yRaw = y;
      // deadlockBreak: a creeping car does not wait for the car it squeezes past (sim.js ctx.yieldRequired)
      if (y && P.dlBreak != 0u && bitcast<f32>(ctl[i * CW + 33u]) > simTime() && ybJ == bitcast<i32>(ctl[i * CW + 34u])) { y = false; }
      if (y) { bits = bits | 1u; }
      if (boxBusyJ(i, u32(m))) { bits = bits | 2u; }
    }
  }
  ctl[i * CW + 19u] = bits;
  ctl[i * CW + 32u] = bitcast<u32>(select(-1, ybJ, yRaw));
  if (hasFeat(FEAT_POLICY) && (v.flags & F_POLICY) != 0u) {
    let c = i * INS + OFF_CTL;
    // input 62 (yieldRequired) is only set while a control is perceived (leaders.wgsl policyInputs)
    junctionInputs(i, v, pi);
    if (inputs[c + 10u] < 1.0 || inputs[c] + inputs[c + 1u] + inputs[c + 2u] + inputs[c + 3u] + inputs[c + 4u] > 0.0) { inputs[c + 16u] = select(0.0, 1.0, (bits & 1u) != 0u); }
  }
}

// sim.js _rerouteBlocked (scan): a closed edge (every lane blocked by an active scene; bitmask after the
// junction / pass heads in ctl) on my route within P.rerouteSight m, not committed to the junction ahead
@compute @workgroup_size(128)
fn rerouteScan(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= P.cap) { return; }
  let v = veh[i];
  if (v.state != DRIVING || vConn(v) >= 0 || v.refuelState != 0u) { return; }
  let pi = par[i]; let rp = v.routePos;
  if (rp + 1u >= pi.routeLen) { return; }
  let e = v.edge; var stopAt = eLength(e); if (eSStop(e) < stopAt) { stopAt = eSStop(e); }
  if (v.s + hlOf(i) > stopAt - 5.0) { return; }
  let base = P.jHeadOff + P.jM + P.E;
  let lastAsk = bitcast<f32>(ctl[i * CW + 23u]);
  var d = eLength(e) - v.s;
  for (var k = rp + 1u; k < pi.routeLen && d < P.rerouteSight; k = k + 1u) {
    let en = routeEdge(pi, k);
    if (((ctl[base + (en >> 5u)] >> (en & 31u)) & 1u) != 0u) {
      // at most once per 120 s per car (the CPU applies the same limit; avoids flooding the event queue)
      if (simTime() - lastAsk < 120.0 && lastAsk >= v.spawnTime) { return; }
      ctl[i * CW + 23u] = bitcast<u32>(simTime()); emit(EV_REROUTE, i, e, rp); return;
    }
    d = d + eLength(en);
  }
  // stopped for 90 s in slow traffic: ask the nav app for a faster way (CPU checks the gain)
  if (P.trafficOn != 0u && v.stoppedSince >= 0.0 && simTime() - v.stoppedSince > 90.0 && !(simTime() - lastAsk < 300.0 && lastAsk >= v.spawnTime)) {
    ctl[i * CW + 23u] = bitcast<u32>(simTime()); emit(EV_REROUTE, i, e, rp | 0x80000000u);
  }
}
