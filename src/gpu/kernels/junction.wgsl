// Physical junctions (layout v8, sim.js _indexConnectors / _junction / _yieldJ / _boxBusyJ), src/geom junctions.js.
// After leaders, before drive:
//   jclear   per movement: empty car list (ctl tail at P.jHeadOff)
//   jlink    per car on a connector (driving or wrecked): push onto its movement's list (next in ctl word 18)
//   jperceive per driving car on / approaching a connector: bodies on the swept path, conflicting movements
//            -> gap / leader (-3 = a crossing point)
//   jyield   per driving car approaching a mouth: entry yield (_yieldJ) and box busy (_boxBusyJ) -> ctl word 19
//            bits 0 / 1, and obs input 62 for policy cars
// #include common.wgsl


@group(1) @binding(0) var<storage, read_write> veh: array<Veh>;
@group(1) @binding(1) var<storage, read> par: array<VPar>;
@group(1) @binding(2) var<storage, read_write> ctl: array<atomic<u32>>;
@group(1) @binding(3) var<storage, read> pose: array<vec4f>;
@group(1) @binding(4) var<storage, read> hstarts: array<u32>;
@group(1) @binding(5) var<storage, read> hitems: array<u32>;

const CW: u32 = CTL_W;
fn headOf(m: u32) -> u32 { return atomicLoad(&ctl[P.jHeadOff + m]); }
fn nextOf(j: u32) -> u32 { return atomicLoad(&ctl[j * CW + 18u]); }
fn hlOf(j: u32) -> f32 { return 0.5 * par[j].length; }
fn hwOf(j: u32) -> f32 { return 0.5 * par[j].width; }

@compute @workgroup_size(256)
fn jclear(@builtin(global_invocation_id) g: vec3u) { if (g.x < P.jM + P.E) { atomicStore(&ctl[P.jHeadOff + g.x], NONE); } }   // movement heads, then per-edge heads of the wreck passers

fn offHeadIdx(e: u32) -> u32 { return P.jHeadOff + P.jM + P.E + (P.E + 31u) / 32u + e; }
@compute @workgroup_size(256)
fn jclearOff(@builtin(global_invocation_id) g: vec3u) { if (g.x < P.E) { atomicStore(&ctl[offHeadIdx(g.x)], NONE); } }

@compute @workgroup_size(256)
fn jlink(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= P.cap) { return; }
  let v = veh[i];
  if (v.state != DRIVING && v.state != CRASHED) { return; }
  atomicStore(&ctl[i * CW + 29u], 0x7f800000u);   // courtesy.wgsl: no proposal yet (+inf)
  // bodies entirely off the lanes (laneHi < laneLo), per edge (sim.js _offLane)
  if (vHi(v) < vLo(v)) { let pv = atomicExchange(&ctl[offHeadIdx(v.edge)], i); atomicStore(&ctl[i * CW + 30u], pv); }
  // cars passing a wreck through the opposing lane, per edge (sim.js _passers)
  if (v.state == DRIVING && atomicLoad(&ctl[i * CW + 8u]) == bitcast<u32>(v.spawnTime) && bitcast<i32>(atomicLoad(&ctl[i * CW + 20u])) >= 0) {
    let prevP = atomicExchange(&ctl[P.jHeadOff + P.jM + v.edge], i);
    atomicStore(&ctl[i * CW + 21u], prevP);
  }
  let m = vConn(v); if (m < 0) { return; }
  let prev = atomicExchange(&ctl[P.jHeadOff + u32(m)], i);
  atomicStore(&ctl[i * CW + 18u], prev);
}

fn cellOfC(cx: i32, cz: i32) -> u32 { return (bitcast<u32>(cx) & P.wMask) | ((bitcast<u32>(cz) & P.hMask) << P.wBits); }

// sim.js _nodeBodies (layout v9): a car not on a connector sees the bodies on the connectors of the node behind it
// (just after the box) and of the node ahead when its own hop there has no connector, in its heading frame
fn nodeCheck(i: u32, node: u32) {
  let pi = pose[i]; let fx = cos(pi.w); let fz = sin(pi.w); let hl = hlOf(i); let hw = hwOf(i);
  for (var q = inStart(node); q < inStart(node + 1u); q = q + 1u) {
    let o = inEdge(q);
    let s0 = world[P.jEdgeMovOff + o]; let s1 = world[P.jEdgeMovOff + o + 1u];
    for (var k = s0; k < s1; k = k + 1u) {
      var j = headOf(world[P.jEdgeMovOff + P.E + 1u + k]);
      for (var gq = 0u; gq < 256u; gq = gq + 1u) {
        if (j == NONE) { break; }
        let vjn = veh[j];
        let waitsMe = vjn.state == DRIVING && vjn.speed < 0.5 && (vjn.leader == i32(i) || (vjn.leader == -3 && atomicLoad(&ctl[j * CW + 28u]) == i));
        if (j != i && !waitsMe) {
          let pj = pose[j]; let dx = pj.x - pi.x; let dz = pj.z - pi.z;
          let along = dx * fx + dz * fz;
          if (along > 0.0 && along <= 60.0) {
            let lat = dx * fz - dz * fx;
            var dpsi = pj.w - pi.w; dpsi = atan2(sin(dpsi), cos(dpsi));
            let sa = abs(sin(dpsi)); let ca = abs(cos(dpsi));
            if (abs(lat) <= hw + sa * hlOf(j) + ca * hwOf(j) + 0.3) {
              let gg = along - hl - (ca * hlOf(j) + sa * hwOf(j)) - 0.5;
              if (gg < veh[i].gap) {
                veh[i].gap = gg;
                if (abs(dpsi) < 0.5 && veh[j].state == DRIVING) { veh[i].leader = i32(j); } else { veh[i].leader = -3; atomicStore(&ctl[i * CW + 28u], j); }
              }
            }
          }
        }
        j = nextOf(j);
      }
    }
  }
}
fn nodeBodies(i: u32) {
  let v = veh[i]; let e = v.edge; let pi = par[i]; let hl = hlOf(i);
  if (v.s - hl < 40.0 && eSExit(e) > 0.0) { nodeCheck(i, world[e * 16u + 6u]); }
  if (eLength(e) - v.s < 60.0 && eSStop(e) < eLength(e)) {
    let rp = v.routePos;
    if (rp + 1u >= pi.routeLen || jMovement(e, vLane(v), routeEdge(pi, rp + 1u)) < 0) { nodeCheck(i, eTo(e)); }
  }
}

@compute @workgroup_size(128)
fn jperceive(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= P.cap) { return; }
  let v = veh[i];
  if (v.state != DRIVING) { return; }
  jperceiveCar(i, v);
  if (vConn(veh[i]) < 0) { nodeBodies(i); }
  if (P.pathSight != 0u) { pathBodies(i); }
  // standoff from a stopped obstacle on my edge (sim.js, after all perception): keep room to steer around it,
  // ~4 m of travel per metre of sideways overlap
  let L = veh[i].leader;
  if (L >= 0 && veh[u32(L)].state == CRASHED && veh[u32(L)].edge == v.edge) {
    let pi = par[i]; let pl = par[u32(L)]; let vl = veh[u32(L)];
    let hI = 0.5 * (pi.width * abs(cos(v.headingRel)) + pi.length * abs(sin(v.headingRel)));
    let hL = 0.5 * (pl.width * abs(cos(vl.headingRel)) + pl.length * abs(sin(vl.headingRel)));
    let ov = min(v.lat + hI, vl.lat + hL) - max(v.lat - hI, vl.lat - hL);
    if (ov > 0.0) { veh[i].gap = veh[i].gap - min(15.0, 2.0 + 4.0 * (ov + 0.3)); }
  }
}
fn jperceiveCar(i: u32, v: Veh) {
  let pi = par[i]; let e = v.edge; let elen = eLength(e);
  let hl = hlOf(i); let sp = v.speed;
  var m = vConn(v);
  if (m < 0) {
    let sStop = eSStop(e);
    if (!(sStop < elen)) { return; }
    let rp = v.routePos;
    if (rp + 1u >= pi.routeLen || (v.refuelState == 1u && v.stopPos == i32(rp))) { return; }
    if (sStop - v.s - hl > max(30.0, sp * sp / 4.0 + 15.0)) { return; }
    m = jMovement(e, vLane(v), routeEdge(pi, rp + 1u)); if (m < 0) { return; }
  }
  let mu = u32(m);
  let sIn = movSIn(mu); let len = movLen(mu); let ti = v.s - sIn;     // ti < 0: before the mouth
  let myNode = eTo(movInEdge(mu));
  let toMouth = -ti - hl - 0.5;                                        // front bumper to the stop point
  var best = v.gap; var who = v.leader; var whoJ = NONE;
  // 1. stationary / crawling bodies on my swept path
  let t0 = max(0.0, ti + hl);
  if (t0 < len) {
    var cand: array<u32, 32>; var nc = 0u;
    let P0 = arcPointJ(mu, 0.5 * (t0 + len));
    let r = 0.5 * (len - t0) + 8.0;
    let cx0 = i32(floor((P0.x - r) / P.cellSize)); let cx1 = i32(floor((P0.x + r) / P.cellSize));
    let cz0 = i32(floor((P0.z - r) / P.cellSize)); let cz1 = i32(floor((P0.z + r) / P.cellSize));
    for (var cz = cz0; cz <= cz1; cz = cz + 1) {
      for (var cx = cx0; cx <= cx1; cx = cx + 1) {
        let c = cellOfC(cx, cz);
        for (var k = hstarts[c]; k < hstarts[c + 1u]; k = k + 1u) {
          let j = hitems[k];
          if (j == i || nc >= 32u) { continue; }
          let vj = veh[j];
          // slow movers on the connectors of a neighbouring junction box are not in my conflict matrix (sim.js)
          let otherBox = vConn(vj) >= 0 && eTo(movInEdge(u32(vConn(vj)))) != myNode;
          if (!(vj.state == CRASHED || vj.speed < 2.0 || (otherBox && vj.speed < 7.0))) { continue; }
          // a stopped car that is itself waiting for me: whoever is already in the box (else the lower slot) goes
          if (vj.state == DRIVING && vj.leader == i32(i)) { let iIn = vConn(v) >= 0; let jIn = vConn(vj) >= 0; if ((iIn && !jIn) || (iIn == jIn && i < j)) { continue; } }
          let pj = pose[j]; let dx = pj.x - P0.x; let dz = pj.z - P0.z;
          if (dx * dx + dz * dz > r * r) { continue; }
          cand[nc] = j; nc = nc + 1u;
        }
      }
    }
    if (nc > 0u) {
      let clear = hwOf(i) + 0.4; let off = v.lat - (f32(movInLane(mu)) + 0.5) * laneWAt(e, sIn);
      var t = t0;
      loop {
        if (t > len + 0.01) { break; }
        let Pt = arcPointJ(mu, t);
        let px = Pt.x + off * Pt.hz; let pz = Pt.z - off * Pt.hx;
        var hit = -1;
        for (var k = 0u; k < nc; k = k + 1u) {
          let j = cand[k]; let pj = pose[j]; let c = cos(pj.w); let sn = sin(pj.w);
          let dx = px - pj.x; let dz = pz - pj.z;
          let lx = abs(dx * c + dz * sn) - hlOf(j); let lz = abs(dz * c - dx * sn) - hwOf(j);
          if (lx < clear && lz < clear && length(vec2f(max(lx, 0.0), max(lz, 0.0))) < clear) { hit = i32(j); break; }
        }
        if (hit >= 0) {
          var gg = t - ti - hl - 0.5;
          if (ti + hl < 0.0) { gg = min(toMouth, gg); }
          if (gg < best) { best = gg; who = hit; }
          break;
        }
        t = t + 1.0;
      }
    }
  }
  // 2. conflicting movements
  for (var k = jConfBegin(mu); k < jConfBegin(mu + 1u); k = k + 1u) {
    let c = jConfAt(k); let other = c.x; let ctype = c.y; let cs = bitcast<f32>(c.z); let csO = bitcast<f32>(c.w);
    var j = headOf(other);
    if (j == NONE) { continue; }
    let di = cs - ti;
    if (di < -hl - 1.0) { continue; }                                     // I am past it
    for (var gq = 0u; gq < 1024u; gq = gq + 1u) {
      if (j == NONE) { break; }
      let vjw = veh[j];
      // it is itself waiting for me (I am its leader / the conflict it yields to): I go
      let waitsMe = j != i && vjw.speed < 0.5 && (vjw.leader == i32(i) || (vjw.leader == -3 && atomicLoad(&ctl[j * CW + 28u]) == i));
      if (j != i && !waitsMe) {
        let vj = veh[j]; let tj = vj.s - movSIn(other); let dj = csO - tj; let hlj = hlOf(j);
        if (ctype == 1u) {                                                 // merge: follow whoever is nearer the merge point
          if (dj < di || (dj == di && j < i)) { let gg = di - dj - hl - hlj - 1.0; if (gg < best) { best = gg; who = i32(j); } }
        } else if (!(dj < -hlj - hwOf(i) - 0.5)) {                         // not cleared the crossing yet
          // side by side from the same approach: the car turning from the wrong lane gives way to the one in its lane
          var sameApproach = 0;
          if (movInEdge(mu) == movInEdge(other)) {
            let myPref = (movW(mu, 0u) >> 28u) & 1u; let hisPref = (movW(other, 0u) >> 28u) & 1u;
            if (myPref == 1u && hisPref == 0u) { sameApproach = 1; }
            else if (myPref == 0u && hisPref == 1u) { sameApproach = 2; }
          }
          if (sameApproach == 2) { var g2 = di - hl - hwOf(j) - 1.5; if (ti + hl < 0.0) { g2 = min(g2, toMouth); } if (g2 < best) { best = g2; who = -3; whoJ = j; } }
          if (sameApproach == 0) {
          var gStop = di - hl - hwOf(j) - 1.5;
          if (ti + hl < 0.0) { gStop = min(gStop, toMouth); }              // not entered yet: wait at the mouth
          if (dj < hlj + hwOf(i) + 0.5) {                                 // its body is on the crossing
            // both bodies already on it (entered together): the one with less left to clear goes on
            var iGoes = false;
            if (di < hl + hwOf(j) + 0.5) { let clearI = di + hl + hwOf(j); let clearJ = dj + hlj + hwOf(i); iGoes = clearI < clearJ || (clearI == clearJ && i < j); }
            if (!iGoes && gStop < best) { best = gStop; who = -3; whoJ = j; }
          } else {
            let tI = di / max(sp, 1.0); let tJ = dj / max(vj.speed, 1.0);
            if ((tJ < tI || (tJ == tI && j < i)) && gStop > sp * sp / 8.0) { if (gStop < best) { best = gStop; who = -3; whoJ = j; } }
          }
          }
        }
      }
      j = nextOf(j);
    }
  }
  if (best < v.gap) { veh[i].gap = best; veh[i].leader = who; if (who == -3) { atomicStore(&ctl[i * CW + 28u], whoJ); } }
}

fn stuckJ(k: u32) -> bool { let vk = veh[k]; return vk.state == DRIVING && vk.speed < 0.3 && vk.stoppedSince >= 0.0 && simTime() - vk.stoppedSince > 20.0; }

/**
 * sim.js _pathBodies (pathSight 1, docs/scale-crashes.md): any body physically on the path I am about to drive is
 * a leader. The path is sampled every 2 m from my front to the stopping sight distance (my lateral offset on my
 * edge, my movement's connector, then the lane centres of the next route edges).
 */
fn pathBodies(i: u32) {
  let v = veh[i]; let pi = par[i]; let sp = v.speed;
  let D = min(80.0, 10.0 + 1.5 * sp + sp * sp / 8.0); let hl = hlOf(i); let hw = hwOf(i);
  let me = pose[i]; let fx = cos(me.w); let fz = sin(me.w);
  var cand: array<u32, 48>; var nc = 0u;
  let cxp = me.x + fx * 0.5 * D; let czp = me.z + fz * 0.5 * D; let r = 0.5 * D + 9.0;
  let cx0 = i32(floor((cxp - r) / P.cellSize)); let cx1 = i32(floor((cxp + r) / P.cellSize));
  let cz0 = i32(floor((czp - r) / P.cellSize)); let cz1 = i32(floor((czp + r) / P.cellSize));
  let stI = stuckJ(i);
  for (var cz = cz0; cz <= cz1; cz = cz + 1) {
    for (var cx = cx0; cx <= cx1; cx = cx + 1) {
      let c = cellOfC(cx, cz);
      for (var k = hstarts[c]; k < hstarts[c + 1u]; k = k + 1u) {
        let j = hitems[k];
        if (j == i || nc >= 48u) { continue; }
        let vj = veh[j];
        if (vj.state != DRIVING && vj.state != CRASHED) { continue; }
        let pj = pose[j]; let ex = pj.x - cxp; let ez = pj.z - czp;
        if (ex * ex + ez * ez > r * r) { continue; }
        let dx = pj.x - me.x; let dz = pj.z - me.z;
        if (dx * fx + dz * fz < hl) { continue; }                                    // beside or behind my front
        if (vj.state == DRIVING && vj.speed < 0.5 && (vj.leader == i32(i) || (vj.leader == -3 && atomicLoad(&ctl[j * CW + 28u]) == i))) { continue; }   // waiting for me
        if (vj.state == DRIVING && i < j && stI && stuckJ(j)) { continue; }        // mutual standstill: the lower slot creeps
        cand[nc] = j; nc = nc + 1u;
      }
    }
  }
  if (nc == 0u) { return; }
  var rp = v.routePos; var e = v.edge; var m = vConn(v); var pos = v.s; var end = eLength(e); var off = 0.0; var lat = v.lat; var lane = vLane(v);
  if (m >= 0) { let mu = u32(m); pos = v.s - movSIn(mu); end = movLen(mu); off = lat - (f32(movInLane(mu)) + 0.5) * laneWAt(e, movSIn(mu)); }
  else if (eSStop(e) < eLength(e) && rp + 1u < pi.routeLen) { let mm = jMovement(e, lane, routeEdge(pi, rp + 1u)); if (mm >= 0) { end = movSIn(u32(mm)); } }
  let clear = hw + 0.3;
  var d = 0.0; var t = hl + 0.5; var frames = 0u;
  for (var it = 0u; it < 96u; it = it + 1u) {
    if (t > D) { return; }
    let rest = end - pos;
    if (d + rest < t) {                                                              // next frame
      d = d + max(0.0, rest);
      frames = frames + 1u; if (frames > 6u) { return; }
      if (m >= 0) {                                                                  // connector -> its out edge
        let mu = u32(m); let oe = movOutEdge(mu); let ol = movOutLane(mu); rp = rp + 1u; e = oe; pos = movSOut(mu); m = -1;
        lane = min(eLanes(e) - 1u, ol); lat = (f32(lane) + 0.5) * laneWAt(e, pos); end = eLength(e);
        if (eSStop(e) < eLength(e) && rp + 1u < pi.routeLen) { let mm = jMovement(e, lane, routeEdge(pi, rp + 1u)); if (mm >= 0) { end = movSIn(u32(mm)); } }
        continue;
      }
      if (rp + 1u >= pi.routeLen) { return; }
      var mm = -1; if (eSStop(e) < eLength(e)) { mm = jMovement(e, lane, routeEdge(pi, rp + 1u)); }
      if (mm >= 0) { m = mm; pos = 0.0; end = movLen(u32(mm)); off = lat - (f32(movInLane(u32(mm))) + 0.5) * laneWAt(e, movSIn(u32(mm))); continue; }
      let en = routeEdge(pi, rp + 1u); rp = rp + 1u;
      let nl = mapLane(e, en, i32(lane)); lane = u32(clamp(nl, 0, i32(eLanes(en)) - 1)); e = en; pos = 0.0; end = eLength(e); lat = (f32(lane) + 0.5) * laneWAt(e, 0.0);
      if (eSStop(e) < eLength(e) && rp + 1u < pi.routeLen) { let m2 = jMovement(e, lane, routeEdge(pi, rp + 1u)); if (m2 >= 0) { end = movSIn(u32(m2)); } }
      continue;
    }
    let u = pos + (t - d);
    var px = 0.0; var pz = 0.0; var phx = 1.0; var phz = 0.0;
    if (m >= 0) { let Pt = connPointJ(u32(m), u); px = Pt.x + off * Pt.hz; pz = Pt.z - off * Pt.hx; phx = Pt.hx; phz = Pt.hz; }
    else { let Pt = latPointAt(e, min(u, eLength(e)), lat); px = Pt.x; pz = Pt.z; phx = Pt.hx; phz = Pt.hz; }
    for (var k = 0u; k < nc; k = k + 1u) {
      let j = cand[k]; let pj = pose[j]; let c = cos(pj.w); let sn = sin(pj.w);
      let dx = px - pj.x; let dz = pz - pj.z;
      let lx = abs(dx * c + dz * sn) - hlOf(j); let lz = abs(dz * c - dx * sn) - hwOf(j);
      if (lx < clear && lz < clear && length(vec2f(max(lx, 0.0), max(lz, 0.0))) < clear) {
        let gg = t - hl - 0.5;
        if (gg < veh[i].gap) {
          let dpsi = wrapAngle(pj.w - atan2(phz, phx));
          veh[i].gap = gg;
          if (veh[j].state == DRIVING && abs(dpsi) < 0.6) { veh[i].leader = i32(j); }
          else { veh[i].leader = -3; atomicStore(&ctl[i * CW + 28u], j); }
        }
        return;
      }
    }
    t = t + 2.0;
  }
}

/** deadlockBreak (sim.js _breakDeadlocks): who slot k waits for (leader / conflict point / junction-yield blocker) */
fn waitsJ(k: u32) -> i32 {
  let L = veh[k].leader; if (L >= 0) { return L; }
  if (L == -3) { let c = atomicLoad(&ctl[k * CW + 28u]); if (c != NONE) { return i32(c); } }
  return bitcast<i32>(atomicLoad(&ctl[k * CW + 32u]));
}
/** pass 1 (reads only): the lowest slot of a closed wait loop of cars standing > 20 s -> ctl word 35 = blocker + 1 */
@compute @workgroup_size(128)
fn jdlFind(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= P.cap) { return; }
  atomicStore(&ctl[i * CW + 35u], 0u);
  if (P.dlBreak == 0u || veh[i].state != DRIVING || (veh[i].flags & F_POLICY) != 0u) { return; }   // built-in drivers only (sim.js)
  if (bitcast<f32>(atomicLoad(&ctl[i * CW + 33u])) > simTime()) { return; }
  if (!stuckJ(i)) { return; }
  var seen: array<u32, 10>; var ns = 0u; var k = i; var lo = i; var ok = false;
  for (var n = 0u; n < 10u; n = n + 1u) {
    let w = waitsJ(k); if (w < 0 || u32(w) >= P.cap || !stuckJ(u32(w))) { break; }
    let wu = u32(w);
    if (wu == i) { ok = true; break; }
    var dup = false; for (var q = 0u; q < ns; q = q + 1u) { if (seen[q] == wu) { dup = true; } }
    if (dup) { break; }
    seen[ns] = wu; ns = ns + 1u; if (wu < lo) { lo = wu; } k = wu;
  }
  if (ok && lo == i) { atomicStore(&ctl[i * CW + 35u], u32(waitsJ(i) + 1)); }
}
/** pass 2: the chosen slots start creeping for 10 s (their blocker becomes a point 3 m ahead); creeping slots keep ignoring it */
@compute @workgroup_size(128)
fn jdlApply(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= P.cap) { return; }
  if (P.dlBreak == 0u || veh[i].state != DRIVING || (veh[i].flags & F_POLICY) != 0u) { return; }
  let now = simTime();
  let dec = atomicLoad(&ctl[i * CW + 35u]);
  if (dec != 0u) {
    atomicStore(&ctl[i * CW + 33u], bitcast<u32>(now + 10.0)); atomicStore(&ctl[i * CW + 34u], dec - 1u);
    veh[i].leader = -3; veh[i].gap = 3.0; atomicStore(&ctl[i * CW + 28u], NONE);
    return;
  }
  if (bitcast<f32>(atomicLoad(&ctl[i * CW + 33u])) > now) {
    let w = waitsJ(i);
    if (w >= 0 && u32(w) == atomicLoad(&ctl[i * CW + 34u]) && veh[u32(w)].state != CRASHED) { veh[i].leader = -3; veh[i].gap = 3.0; atomicStore(&ctl[i * CW + 28u], NONE); }
  }
}
