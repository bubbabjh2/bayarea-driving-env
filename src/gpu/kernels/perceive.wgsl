// Input version 3 perception (src/sim/perceive.js + src/sim/obsLayout.js, 1:1; docs/perception-design.md).
// After sense (inputs 0..69 final), for every policy car due a decision: bodies (NB block, 8 ego sectors x 2 nearest
// within 100 m on the same level, from the contacts hash hstarts / hitems, own-lane leader guaranteed), mirror inputs
// 70..78, controls extras 79..86 (own commands 87..93 and the leader history 94..100: sense.wgsl, buffer limit).
// Injected: INS (= OBS3), NB_OFF3
// #include common.wgsl
// #include controls.wgsl

@group(1) @binding(0) var<storage, read> veh: array<Veh>;
@group(1) @binding(1) var<storage, read> par: array<VPar>;
@group(1) @binding(2) var<storage, read> pose: array<vec4f>;
@group(1) @binding(3) var<storage, read> hstarts: array<u32>;
@group(1) @binding(4) var<storage, read> hitems: array<u32>;
@group(1) @binding(5) var<storage, read_write> inputs: array<f32>;
@group(1) @binding(6) var<storage, read> ctl: array<u32>;

const RNB: f32 = 100.0;
const LAYER: f32 = 2.5;
const PI: f32 = 3.14159265;

fn policyDue3(i: u32) -> bool {
  let v = veh[i];
  if (v.state != DRIVING || (v.flags & F_POLICY) == 0u) { return false; }
  return P.decideEvery <= 1u || (simStep() + i) % P.decideEvery == 0u;
}
fn senseLaneW3(v: Veh) -> f32 { let m = vConn(v); if (m >= 0) { return laneWAt(v.edge, movSIn(u32(m))); } return laneWAt(v.edge, v.s); }
fn senseLaneOff3(v: Veh, lw: f32) -> f32 { let m = vConn(v); if (m >= 0) { return v.lat - (f32(movInLane(u32(m))) + 0.5) * lw; } return v.lat - (f32(vLane(v)) + 0.5) * lw; }
fn sectorOf3(fwd: f32, left: f32) -> u32 { let a = atan2(left, fwd); return u32(clamp(floor((a + PI) / (PI / 4.0)), 0.0, 7.0)); }
/** perceive.js movementAhead */
fn movAhead(j: u32, within: f32) -> i32 {
  if (P.hasJ == 0u) { return -1; }
  let v = veh[j]; let m = vConn(v); if (m >= 0) { return m; }
  let pj = par[j];
  if (pj.routeLen == 0u || v.routePos + 1u >= pj.routeLen) { return -1; }
  if (eLength(v.edge) - v.s - 0.5 * pj.length > within) { return -1; }
  return jMovement(v.edge, vLane(v), routeEdge(pj, v.routePos + 1u));
}
/** perceive.js indicatorOf: x = left, y = right */
fn indicator3(j: u32) -> vec2f {
  let v = veh[j];
  if (v.state != DRIVING) { return vec2f(0.0); }
  let m = movAhead(j, max(60.0, 4.0 * v.speed));
  if (m >= 0) { let th = movTurn(u32(m)); return vec2f(select(0.0, 1.0, th < -0.5), select(0.0, 1.0, th > 0.5)); }
  if (v.laneEndDist < 100.0 && v.laneEndSide != 0) { return vec2f(select(0.0, 1.0, v.laneEndSide > 0), select(0.0, 1.0, v.laneEndSide < 0)); }
  return vec2f(0.0);
}
fn endSignType(e: u32) -> i32 {
  let a0 = edgeApBegin(e); let a1 = edgeApBegin(e + 1u);
  if (a1 <= a0) { return -1; }
  return i32(world[P.ctlCOff + apControl(edgeApAt(a1 - 1u)) * CTL_REC]);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= P.cap || !hasFeat(FEAT_POLICY) || !policyDue3(i)) { return; }
  let v = veh[i]; let pi = par[i]; let pp = pose[i];
  let b = i * INS;
  let x = pp.x; let y = pp.y; let z = pp.z; let h = pp.w;
  let c = cos(h); let s = sin(h);
  let hlI = 0.5 * pi.length; let hwI = 0.5 * pi.width; let sp = v.speed;
  let lw = senseLaneW3(v); let off = senseLaneOff3(v, lw); let hw = 0.5 * lw;
  var sd2: array<f32, 16>; var sj: array<i32, 16>;
  for (var k = 0u; k < 16u; k = k + 1u) { sd2[k] = 3.0e38; sj[k] = -1; }
  var altL = 0.0; var altR = 0.0; var ttcRL = 1e30; var ttcRR = 1e30; var gapRL = 1e30; var gapRR = 1e30; var ttcOwn = 1e30; var clrL = 1e30; var clrR = 1e30;
  let inv = 1.0 / P.cellSize;
  let cx0 = i32(floor((x - RNB) * inv)); let cx1 = i32(floor((x + RNB) * inv));
  let cz0 = i32(floor((z - RNB) * inv)); let cz1 = i32(floor((z + RNB) * inv));
  for (var cz = cz0; cz <= cz1; cz = cz + 1) {
    for (var cx = cx0; cx <= cx1; cx = cx + 1) {
      let cell = (bitcast<u32>(cx) & P.wMask) | ((bitcast<u32>(cz) & P.hMask) << P.wBits);
      for (var t = hstarts[cell]; t < hstarts[cell + 1u]; t = t + 1u) {
        let j = hitems[t]; if (j == i) { continue; }
        let st = veh[j].state; if (st != DRIVING && st != CRASHED) { continue; }
        let q = pose[j];
        if (abs(q.y - y) >= LAYER) { continue; }
        let dx = q.x - x; let dz = q.z - z; let d2 = dx * dx + dz * dz;
        if (d2 > RNB * RNB) { continue; }
        let fwd = dx * c + dz * s; let left = dx * s - dz * c;
        // perception.js offerBody: 2 nearest per sector (ties: lower slot)
        let b0 = sectorOf3(fwd, left) * 2u; let ji = i32(j);
        if (d2 < sd2[b0 + 1u] || (d2 == sd2[b0 + 1u] && ji < sj[b0 + 1u])) {
          if (d2 < sd2[b0] || (d2 == sd2[b0] && ji < sj[b0])) { sd2[b0 + 1u] = sd2[b0]; sj[b0 + 1u] = sj[b0]; sd2[b0] = d2; sj[b0] = ji; }
          else { sd2[b0 + 1u] = d2; sj[b0 + 1u] = ji; }
        }
        // mirror inputs
        let hlJ = 0.5 * par[j].length; let hwJ = 0.5 * par[j].width; let lat = left + off; let lo = lat - hwJ; let hi = lat + hwJ;
        var vj = 0.0; if (st == DRIVING) { vj = veh[j].speed; }
        let vf = vj * cos(q.w - h) - sp;
        let inL = hi > hw && lo < 3.0 * hw; let inR = lo < -hw && hi > -3.0 * hw;
        let inOwn = hi > min(-hw, off - hwI) && lo < max(hw, off + hwI);
        let along = abs(fwd) < hlI + hlJ + 1.0;
        if (along && inL) { altL = 1.0; }
        if (along && inR) { altR = 1.0; }
        if (fwd < -(hlI + hlJ)) {
          let gg = -fwd - hlI - hlJ;
          if (gg <= 60.0) {
            var tt = 1e30; if (vf > 0.1) { tt = gg / vf; }
            if (inL) { gapRL = min(gapRL, gg); ttcRL = min(ttcRL, tt); }
            if (inR) { gapRR = min(gapRR, gg); ttcRR = min(ttcRR, tt); }
            if (inOwn) { ttcOwn = min(ttcOwn, tt); }
          }
        }
        if (abs(fwd) < hlI + hlJ) {
          if (left > 0.0) { clrL = min(clrL, max(0.0, left - hwI - hwJ)); } else { clrR = min(clrR, max(0.0, -left - hwI - hwJ)); }
        }
      }
    }
  }
  // perception.js forceBody: my own-lane leader
  let L = v.leader;
  if (L >= 0) {
    let j = u32(L); let st = veh[j].state; let q = pose[j];
    if ((st == DRIVING || st == CRASHED) && abs(q.y - y) < LAYER) {
      let dx = q.x - x; let dz = q.z - z; let d2 = dx * dx + dz * dz;
      var present = false; for (var k = 0u; k < 16u; k = k + 1u) { if (sj[k] == L) { present = true; } }
      if (d2 <= RNB * RNB && !present) {
        let b0 = sectorOf3(dx * c + dz * s, dx * s - dz * c) * 2u;
        sd2[b0 + 1u] = d2; sj[b0 + 1u] = L;
        if (sj[b0] < 0 || d2 < sd2[b0] || (d2 == sd2[b0] && L < sj[b0])) { let t0 = sd2[b0]; let t1 = sj[b0]; sd2[b0] = sd2[b0 + 1u]; sj[b0] = sj[b0 + 1u]; sd2[b0 + 1u] = t0; sj[b0 + 1u] = t1; }
      }
    }
  }
  inputs[b + 70u] = altL; inputs[b + 71u] = altR;
  inputs[b + 72u] = select(min(1.0, ttcRL / 6.0), 1.0, ttcRL >= 1e29); inputs[b + 73u] = select(min(1.0, ttcRR / 6.0), 1.0, ttcRR >= 1e29);
  inputs[b + 74u] = select(min(1.0, gapRL / 60.0), 1.0, gapRL >= 1e29); inputs[b + 75u] = select(min(1.0, gapRR / 60.0), 1.0, gapRR >= 1e29);
  inputs[b + 76u] = select(min(1.0, ttcOwn / 6.0), 1.0, ttcOwn >= 1e29);
  inputs[b + 77u] = select(min(1.0, clrL / 3.7), 1.0, clrL >= 1e29); inputs[b + 78u] = select(min(1.0, clrR / 3.7), 1.0, clrR >= 1e29);
  // bodies: perception.js bodyFeatures
  let mMe = movAhead(i, 150.0);
  for (var k = 0u; k < 16u; k = k + 1u) {
    let at = b + NB_OFF3 + k * 16u;
    if (sj[k] < 0) { for (var d = 0u; d < 16u; d = d + 1u) { inputs[at + d] = 0.0; } continue; }
    let j = u32(sj[k]); let q = pose[j]; let vjv = veh[j]; let pj = par[j];
    let crashed = vjv.state != DRIVING;
    var vo = vjv.speed; if (crashed) { vo = 0.0; }
    let dx = q.x - x; let dz = q.z - z;
    let fwd = dx * c + dz * s; let left = dx * s - dz * c;
    let co = cos(q.w); let so = sin(q.w);
    let rvx = vo * co - sp * c; let rvz = vo * so - sp * s;
    let vf = rvx * c + rvz * s; let vl = rvx * s - rvz * c;
    let dd = sqrt(dx * dx + dz * dz);
    var ux = 1.0; var uz = 0.0; if (dd > 1e-6) { ux = dx / dd; uz = dz / dd; }
    let eMe = hlI * abs(ux * c + uz * s) + hwI * abs(ux * s - uz * c);
    let eO = 0.5 * pj.length * abs(ux * co + uz * so) + 0.5 * pj.width * abs(ux * so - uz * co);
    let gap = max(0.0, dd - eMe - eO);
    let dh0 = q.w - h; let dh = atan2(sin(dh0), cos(dh0));
    let vv = vf * vf + vl * vl; var ca = 1.0;
    if (vv > 1e-4) {
      let tt = -(fwd * vf + left * vl) / vv;
      if (tt > 0.0) { let mx = fwd + vf * tt; let ml = left + vl * tt; if (mx * mx + ml * ml <= 9.0) { ca = min(1.0, tt / 10.0); } }
    }
    let ind = indicator3(j);
    var conflict = 0.0;
    if (mMe >= 0) {
      let mj = movAhead(j, 150.0);
      if (mj >= 0) { for (var r = jConfBegin(u32(mMe)); r < jConfBegin(u32(mMe) + 1u); r = r + 1u) { if (jConfAt(r).x == u32(mj)) { conflict = 1.0; break; } } }
    }
    inputs[at] = fwd / RNB; inputs[at + 1u] = left / RNB; inputs[at + 2u] = min(gap / RNB, 1.0);
    inputs[at + 3u] = clamp(vf / 20.0, -1.0, 1.0); inputs[at + 4u] = clamp(vl / 20.0, -1.0, 1.0);
    inputs[at + 5u] = cos(dh); inputs[at + 6u] = sin(dh);
    inputs[at + 7u] = pj.length / 20.0; inputs[at + 8u] = pj.width / 3.0;
    inputs[at + 9u] = select(select(0.0, 1.0, vjv.brake > 0.05), 0.0, crashed);
    inputs[at + 10u] = ind.x; inputs[at + 11u] = ind.y;
    inputs[at + 12u] = select(0.0, 1.0, crashed); inputs[at + 13u] = conflict;
    inputs[at + 14u] = ca; inputs[at + 15u] = 1.0;
  }
  // controls extras 79..86 (perceive.js senseControls3)
  for (var k = 79u; k < 87u; k = k + 1u) { inputs[b + k] = 0.0; }
  inputs[b + 82u] = 1.0;
  if (P.hasCtl != 0u) {
    let o = i * CTL_W;
    let a = bitcast<i32>(ctl[o]); let ty = bitcast<i32>(ctl[o + 1u]); let lst = ctl[o + 2u]; let mv = ctl[o + 6u] & 255u;
    if (a >= 0 && ty == CTL_SIGNAL && lst == LIGHT_GREEN && mv == MOVE_LEFT && ((world[P.ctlApOff + u32(a) * AP_REC + 2u] >> 8u) & 1u) == 0u) { inputs[b + 79u] = 1.0; }
    if (mMe >= 0) {
      let inE = movInEdge(u32(mMe)); let my = endSignType(inE);
      var n = 0u; var signed = 0u; var stops = 0u;
      for (var r = jConfBegin(u32(mMe)); r < jConfBegin(u32(mMe) + 1u); r = r + 1u) {
        let e2 = movInEdge(jConfAt(r).x); if (e2 == inE) { continue; }
        let t2 = endSignType(e2); n = n + 1u;
        if (t2 == CTL_STOP || t2 == CTL_GIVE_WAY) { signed = signed + 1u; }
        if (t2 == CTL_STOP) { stops = stops + 1u; }
      }
      if (n > 0u && my != CTL_STOP && my != CTL_GIVE_WAY && signed == n) { inputs[b + 80u] = 1.0; }
      if (n > 0u && my == CTL_STOP && stops == n) { inputs[b + 81u] = 1.0; }
    }
    if (a >= 0 && ty >= 0) {
      let au = u32(a); let eA = apEdge(au);
      var pos = -1;
      for (var k = v.routePos; k < pi.routeLen && k < v.routePos + 12u; k = k + 1u) { if (routeEdge(pi, k) == eA) { pos = i32(k); break; } }
      if (pos >= 0) {
        var a2 = -1; var d = bitcast<f32>(ctl[o + 5u]); var k2 = u32(pos);
        for (var q = edgeApBegin(eA); q < edgeApBegin(eA + 1u); q = q + 1u) { let xa = edgeApAt(q); if (apStopAt(xa) > apStopAt(au)) { a2 = i32(xa); d = d + apStopAt(xa) - apStopAt(au); break; } }
        if (a2 < 0) {
          d = d + eLength(eA) - apStopAt(au);
          for (var k = u32(pos) + 1u; k < pi.routeLen && k < u32(pos) + 9u && d < 250.0; k = k + 1u) {
            let en = routeEdge(pi, k);
            if (edgeApBegin(en + 1u) > edgeApBegin(en)) { a2 = i32(edgeApAt(edgeApBegin(en))); d = d + apStopAt(u32(a2)); k2 = k; break; }
            d = d + eLength(en);
          }
        }
        if (a2 >= 0 && d < 250.0) {
          let e2 = apEdge(u32(a2));
          var p2 = k2; if (e2 == eA) { p2 = u32(pos); }
          var mv2 = MOVE_THROUGH; if (p2 + 1u < pi.routeLen) { mv2 = movementOf(e2, routeEdge(pi, p2 + 1u)); }
          let so2 = ctlStateOf(u32(a2), mv2, simTime());
          inputs[b + 82u] = min(1.0, max(0.0, d) / 250.0);
          if (so2.ty == CTL_SIGNAL) { inputs[b + 83u] = 1.0; } else if (so2.ty == CTL_STOP || so2.ty == CTL_GIVE_WAY) { inputs[b + 84u] = 1.0; } else { inputs[b + 85u] = 1.0; }
          inputs[b + 86u] = select(0.0, 1.0, so2.state == LIGHT_RED || so2.state == LIGHT_YELLOW);
        }
      }
    }
  }
  inputs[b + 101u] = 0.0;
}
