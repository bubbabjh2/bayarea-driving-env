// Stage (b): the policy observation vector (the obs v2 layout (src/gpu/README.md), INPUT_SIZE from the JS module) for
// every policy vehicle due a decision this step, from the same state the CPU adapter uses
// (src/sim/obs.js fillObs / fillNeighbors + sim.js signalsFor):
//   neighbours from the footprint lane lists: own ahead = leader (incl. the lane-end wall,
//   leader -2, as a stopped obstacle), own behind = previous car in my centre lane's list,
//   left / right = binary search in lane +-1 (self skipped); stopped (non-driving) cars rel = -v
//   guidance from the maneuver codes packed into the route entries (src/gpu/worldPack.js)
// Layout offsets are injected as consts by gpuSim.js from the obs v2 layout (src/gpu/README.md):
//   INS, OFF_NB, OFF_MAN, OFF_HINT, OFF_TRAIT, OFF_LATE, OFF_FUEL, OFF_WHEEL, RANGE
// #include common.wgsl
// #include lists.wgsl

@group(1) @binding(0) var<storage, read> veh: array<Veh>;
@group(1) @binding(1) var<storage, read> par: array<VPar>;
@group(1) @binding(2) var<storage, read_write> inputs: array<f32>;
@group(1) @binding(3) var<storage, read> ring: array<f32>;

fn policyDue(i: u32) -> bool {
  let v = veh[i];
  if (v.state != DRIVING || (v.flags & F_POLICY) == 0u) { return false; }
  return P.decideEvery <= 1u || (simStep() + i) % P.decideEvery == 0u;
}
fn putSlot(o: u32, gap: f32, rel: f32, exists: bool) {
  if (!exists) { inputs[o] = 0.0; inputs[o + 1u] = 0.0; return; }
  if (gap >= RANGE) { inputs[o] = 1.0; inputs[o + 1u] = 0.0; return; }
  inputs[o] = max(0.0, gap) / RANGE;
  inputs[o + 1u] = clamp(rel / 20.0, -1.0, 1.0);
}
fn otherSpeed(j: u32) -> f32 { return select(0.0, veh[j].speed, veh[j].state == DRIVING); }
/** src/sim/obs.js fillObs lane width / lane-centre offset: the connector path inside a junction box, else my lane */
fn senseLaneW(v: Veh) -> f32 { let m = vConn(v); if (m >= 0) { return laneWAt(v.edge, movSIn(u32(m))); } return laneWAt(v.edge, v.s); }
fn senseLaneOff(v: Veh, lw: f32) -> f32 { let m = vConn(v); if (m >= 0) { return v.lat - (f32(movInLane(u32(m))) + 0.5) * lw; } return v.lat - (f32(vLane(v)) + 0.5) * lw; }

// src/sim/gradeObs.js (input version 2): height of edge e at s, height d m ahead along the route, the three grades
fn gEdgeY(e: u32, s: f32) -> f32 { return yAlongEdge(world[e * 16u], world[e * 16u + 1u], s); }
fn gRouteY(pi: VPar, pos: u32, s: f32, d: f32) -> f32 {
  var q = pos; var rem = max(s, 0.0) + d; var e = routeEdge(pi, q); var hops = 0u;
  loop {
    if (!(rem > eLength(e) && q + 1u < pi.routeLen && hops < 12u)) { break; }
    rem = rem - eLength(e); q = q + 1u; hops = hops + 1u; e = routeEdge(pi, q);
  }
  return gEdgeY(e, min(rem, eLength(e)));
}
fn gradeInputs(v: Veh, pi: VPar, onC: bool) -> vec3f {
  var g0 = 0.0; if (!onC) { g0 = edgeGrade(v.edge, v.s); }
  let s = v.s; let L = eLength(v.edge);
  var y0 = 0.0; var y1 = 0.0; var y2 = 0.0;
  if (pi.routeLen == 0u || v.routePos >= pi.routeLen || routeEdge(pi, v.routePos) != v.edge) {
    y0 = gEdgeY(v.edge, max(s, 0.0)); y1 = gEdgeY(v.edge, min(L, s + 40.0)); y2 = gEdgeY(v.edge, min(L, s + 100.0));
  } else {
    y0 = gRouteY(pi, v.routePos, s, 0.0); y1 = gRouteY(pi, v.routePos, s, 40.0); y2 = gRouteY(pi, v.routePos, s, 100.0);
  }
  return vec3f(g0, (y1 - y0) / 40.0, (y2 - y1) / 60.0);
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= P.cap) { return; }
  if (!hasFeat(FEAT_POLICY) || !policyDue(i)) { return; }
  let v = veh[i];
  let pi = par[i];
  let e = edgeAt(v.edge);
  let b = i * INS;
  let half = 0.5 * pi.length;
  // src/sim/obs.js fillObs (2026-10-23): on a junction connector the lane frame is the movement's path (one lane,
  // index 0, offset from the path centre = lat - (inLane + 0.5) * W(in-edge, sIn)), no side lanes
  let cm = vConn(v);
  let onC = cm >= 0;
  var n = max(1u, e.lanes);
  var lane = min(vLane(v), n - 1u);
  if (onC) { n = 1u; lane = 0u; }
  let sp = v.speed;
  inputs[b + 0u] = sp / 40.0;
  inputs[b + 1u] = e.speedLimit / 40.0;
  inputs[b + 2u] = f32(lane) / f32(max(1u, n - 1u));
  inputs[b + 3u] = f32(n) / 6.0;
  let lw = senseLaneW(v);
  inputs[b + 4u] = clamp(senseLaneOff(v, lw) / lw, -1.0, 1.0);
  inputs[b + 5u] = clamp(wrapAngle(-v.headingRel) / 0.78539816, -1.0, 1.0);
  let leftEx = lane + 1u < n;
  let rightEx = lane > 0u;
  inputs[b + 6u] = select(0.0, 1.0, leftEx);
  inputs[b + 7u] = select(0.0, 1.0, rightEx);
  var gap = 1e30; var rel = 0.0;
  let L = v.leader;
  if (L >= 0 || L <= -2) { gap = max(0.0, v.gap); var ls = 0.0; if (L >= 0) { ls = otherSpeed(u32(L)); } rel = ls - sp; }
  putSlot(b + OFF_NB, gap, rel, true);
  gap = 1e30; rel = 0.0;
  let B = behindIn(i, vLane(v));
  if (B >= 0) { let j = u32(B); gap = max(0.0, v.s - veh[j].s - half - 0.5 * par[j].length); rel = veh[j].speed - sp; }
  putSlot(b + OFF_NB + 2u, gap, rel, true);
  for (var side = 0u; side < 2u; side = side + 1u) {
    let ex = select(rightEx, leftEx, side == 0u);
    var ga = 1e30; var ra = 0.0; var gb = 1e30; var rb = 0.0;
    let lc = i32(vLane(v)) + select(-1, 1, side == 0u);
    if (lc >= 0 && lc < i32(e.lanes)) {
      let nb = neighborsAt(v.edge, u32(lc), v.s, i);
      if (nb.x >= 0) { let j = u32(nb.x); ga = max(0.0, veh[j].s - v.s - half - 0.5 * par[j].length); ra = otherSpeed(j) - sp; }
      if (nb.y >= 0) { let j = u32(nb.y); gb = max(0.0, v.s - veh[j].s - half - 0.5 * par[j].length); rb = otherSpeed(j) - sp; }
    }
    putSlot(b + OFF_NB + 4u + 4u * side, ga, ra, ex);
    putSlot(b + OFF_NB + 6u + 4u * side, gb, rb, ex);
  }
  // guidance (src/nav/guidance.js at(), codes packed by worldPack.js packRoutes): next maneuver
  // within 2 km by scanning the route; beyond that its type from the entry's nextType field
  for (var k = OFF_MAN; k < OFF_TRAIT; k = k + 1u) { inputs[b + k] = 0.0; }
  var d = e.length - v.s;
  var k = v.routePos;
  var typ = 0u; var sd = 0u; var found = false;
  for (var it = 0u; it < 256u; it = it + 1u) {
    if (v.refuelState == 1u && v.stopPos == i32(k)) { typ = 8u; sd = 2u; found = true; break; }
    let code = routeCode(pi, k);
    if ((code & 1u) != 0u) { typ = (code >> 1u) & 15u; sd = (code >> 5u) & 3u; found = true; break; }
    if (k + 1u >= pi.routeLen || d > 2000.0) { break; }
    k = k + 1u;
    d = d + eLength(routeEdge(pi, k));
  }
  if (!found) { typ = (routeCode(pi, v.routePos) >> 7u) & 15u; sd = 0u; d = 1e9; }
  inputs[b + OFF_MAN] = min(d / 2000.0, 1.0);
  inputs[b + OFF_MAN + 1u + min(typ, 8u)] = 1.0;
  var hint = 1u;
  if (d <= 2000.0 && typ != 2u && sd == 1u) { hint = 0u; }
  if (d <= 2000.0 && typ != 2u && sd == 2u) { hint = 2u; }
  inputs[b + OFF_HINT + hint] = 1.0;
  inputs[b + OFF_TRAIT] = pi.aggression;
  inputs[b + OFF_TRAIT + 1u] = pi.patience;
  inputs[b + OFF_TRAIT + 2u] = pi.headway / 3.0;
  inputs[b + OFF_TRAIT + 3u] = pi.reactionTime;
  // time pressure (sim.js signalsFor + lateness): return leg = 1e5 s to spare
  var ttw = 1e5;
  if ((pi.trip & 1u) == 0u) { ttw = pi.workStart - simTime(); }
  let rem = max(0.0, v.routeTotal - v.routeDist);
  var lateF = 0.0; var ttwF = 0.0;
  if (e.speedLimit > 0.0) {
    lateF = clamp((rem / max(ttw, 60.0) - e.speedLimit) / e.speedLimit, -1.0, 1.0);
    ttwF = clamp(ttw / 3600.0, -1.0, 1.0);
  }
  inputs[b + OFF_LATE] = lateF;
  inputs[b + OFF_LATE + 1u] = ttwF;
  var ff = 1.0;
  if (v.tank > 0.0) { ff = clamp(v.fuel / v.tank, 0.0, 1.0); }
  inputs[b + OFF_FUEL] = ff;
  inputs[b + OFF_FUEL + 1u] = select(0.0, 1.0, v.refuelState == 1u);
  inputs[b + OFF_WHEEL] = clamp(v.steerAngle / pi.maxSteer, -1.0, 1.0);
  inputs[b + OFF_WHEEL + 1u] = clamp(v.gas, 0.0, 1.0);
  inputs[b + OFF_WHEEL + 2u] = clamp(v.brake, 0.0, 1.0);
  adjustInputs(i, b, v, pi, leftEx, rightEx);
  // position-mode actions (v9): the obs does not include its own wheel / pedals (41..43 = 0); it would learn to copy
  // its previous command. Legacy rate nets (P.steerMode bit 1) keep them
  if ((P.steerMode & 2u) == 0u) { inputs[b + OFF_WHEEL] = 0.0; inputs[b + OFF_WHEEL + 1u] = 0.0; inputs[b + OFF_WHEEL + 2u] = 0.0; }
  // input version 2 (docs/grade-input.md, P.steerMode bit 3): road grade in 41..43 (src/sim/gradeObs.js, 1:1)
  if ((P.steerMode & 8u) != 0u) {
    let gr = gradeInputs(v, pi, onC);
    inputs[b + OFF_WHEEL] = clamp(gr.x / 0.35, -1.0, 1.0);
    inputs[b + OFF_WHEEL + 1u] = clamp(gr.y / 0.35, -1.0, 1.0);
    inputs[b + OFF_WHEEL + 2u] = clamp(gr.z / 0.35, -1.0, 1.0);
  }
  // 63 / 64: next junction heading kink within 80 m (nav/guidance.js nextKink): a non-strict hop
  // turning |a| < 0.5 rad keeps the world heading, so the car has to steer through it
  var ka = 0.0; var kd = 1.0;
  var dk = e.length - v.s;
  if (KINKS_ON == 0u) { dk = 1e9; }   // nav/guidance.js KINKS_ENABLED (off: the sim keeps headingRel across hops)
  for (var q = v.routePos; q + 1u < pi.routeLen; q = q + 1u) {
    if (dk > 80.0) { break; }
    let ea = routeEdge(pi, q); let eb = routeEdge(pi, q + 1u);
    if (hopMap(ea, eb).x == 0) {
      let a = wrapAngle(bitcast<f32>(world[eb * 16u + 12u]) - bitcast<f32>(world[ea * 16u + 13u]));
      if (abs(a) < 0.5 && a != 0.0) { ka = clamp(a / 0.5, -1.0, 1.0); kd = clamp(dk / 80.0, 0.0, 1.0); break; }
    }
    dk = dk + eLength(eb);
  }
  inputs[b + OFF_KINK] = ka; inputs[b + OFF_KINK + 1u] = kd;
  // 65..69 junction inputs: written by jyield.wgsl when the world has junctions; without them they read 'none'
  if (P.hasJ == 0u) { inputs[b + OFF_KINK + 2u] = 1.0; inputs[b + OFF_KINK + 3u] = 1.0; inputs[b + OFF_KINK + 4u] = 0.0; inputs[b + OFF_KINK + 5u] = 0.0; inputs[b + OFF_KINK + 6u] = 1.0; }
  if (V3 != 0u) { senseV3(i, b, v, pi); }
}

// input version 3 (src/sim/perceive.js senseCommands + src/sim/obsLayout.js LeaderHistory, 1:1): own commands
// 87..93 from the reaction ring, leader history 94..100 (record per slot at inputs[LH_OFF + i * LH_W ..], after the
// input rows: the storage-buffer limit is 10 per stage)
fn senseV3(i: u32, b: u32, v: Veh, pi: VPar) {
  let L = veh[i].leader;
  // own commands 87..93 (perceive.js senseCommands)
  let S = simStep(); let dl = min(pi.delaySteps, CTRL_RING - 1u);
  var sn = S - 1u; if (dl >= 1u) { sn = S - dl; }
  let on = (i * CTRL_RING + sn % CTRL_RING) * CTRL_STRIDE; let ol = (i * CTRL_RING + (S - 1u) % CTRL_RING) * CTRL_STRIDE;
  inputs[b + 87u] = ring[on]; inputs[b + 88u] = ring[on + 1u]; inputs[b + 89u] = ring[on + 2u];
  inputs[b + 90u] = ring[ol]; inputs[b + 91u] = ring[ol + 1u]; inputs[b + 92u] = ring[ol + 2u];
  inputs[b + 93u] = 1.0;
  // leader history 94..100 (perception.js LeaderHistory.step); record (f32 values): spawnTime tag, n, head, lastLeader, sinceChange, g[20], r[20]
  let hb = LH_OFF + i * LH_W;
  if (inputs[hb] != v.spawnTime) { inputs[hb] = v.spawnTime; inputs[hb + 1u] = 0.0; inputs[hb + 2u] = 0.0; inputs[hb + 3u] = -1.0e9; inputs[hb + 4u] = 1.0e6; }
  let n = u32(inputs[hb + 1u]); let head = u32(inputs[hb + 2u]);
  let ago = vec3u(5u, 10u, 20u);
  for (var qq = 0u; qq < 3u; qq = qq + 1u) {
    let oo = b + 94u + 2u * qq;
    if (n == 0u) { inputs[oo] = inputs[b + 8u]; inputs[oo + 1u] = inputs[b + 9u]; continue; }
    let kk = min(ago[qq], n); let idx = (head + 20u - kk) % 20u;
    inputs[oo] = inputs[hb + 5u + idx]; inputs[oo + 1u] = inputs[hb + 25u + idx];
  }
  var since = inputs[hb + 4u];
  if (n > 0u && f32(L) != inputs[hb + 3u]) { since = 0.0; } else { since = min(since + 1.0, 1.0e6); }
  inputs[b + 100u] = select(0.0, 1.0, since < 20.0);
  inputs[hb + 3u] = f32(L); inputs[hb + 4u] = since;
  inputs[hb + 5u + head] = inputs[b + 8u]; inputs[hb + 25u + head] = inputs[b + 9u];
  inputs[hb + 2u] = f32((head + 1u) % 20u); inputs[hb + 1u] = f32(min(n + 1u, 20u));
  inputs[b + 101u] = 0.0;
}

// the original adjust.js (obs post-processing) adjustInputs, 1:1:
//  a) delay compensation: replay the steerRate commands issued but not applied yet (ring steps
//     t - delaySteps .. t - 1, oldest first) through dynamics.js stepLateral at the current speed;
//     overwrite inputs 4 (lane offset), 5 (heading error), 41 (steer angle)
//  b) lane ends of the left / right lanes (4 hops / 400 m, refuel stop ends the lookahead) as
//     stopped obstacles in their ahead slots; the own lane end is already the sim's leader (-2);
//     inputs 44 / 45 (own lane end) and 46..62 (next control) are written by leaders.wgsl
fn laneEndAhead(i: u32, v: Veh, pi: VPar, lane: i32) -> f32 {
  // src/sim/laneEnds.js (via nav/laneMap.js, lookahead 400 m): every lane of my edge is followed; `lane`
  // ends only if another lane of this edge survives; 4 hops; a refuel stop ends the lookahead
  let n = min(eLanes(v.edge), 16u);
  if (lane < 0 || lane >= i32(n)) { return 1e9; }
  if (v.refuelState == 1u && v.stopPos == i32(v.routePos)) { return 1e9; }
  var alive = (1u << n) - 1u;
  var cur: array<i32, 16>;
  for (var l = 0u; l < n; l = l + 1u) { cur[l] = i32(l); }
  var prev = v.edge; var dist = eLength(v.edge) - v.s; var endK = 0u;
  for (var k = 1u; k <= 4u; k = k + 1u) {
    if (v.routePos + k >= pi.routeLen || dist >= 400.0) { break; }
    let en = routeEdge(pi, v.routePos + k);
    var any = false;
    for (var l = 0u; l < n; l = l + 1u) {
      if (((alive >> l) & 1u) == 0u) { continue; }
      let l2 = mapLane(prev, en, cur[l]);
      if (l2 < 0 || l2 >= i32(eLanes(en))) { alive = alive & ~(1u << l); } else { cur[l] = l2; any = true; }
    }
    let mine = ((alive >> u32(lane)) & 1u) != 0u;
    if (!mine && !any && k > 1u) { break; }
    if (!mine) { if (any) { endK = k; } break; }
    if (v.refuelState == 1u && v.stopPos == i32(v.routePos + k)) { break; }
    dist = dist + eLength(en); prev = en;
  }
  if (endK == 0u) { return 1e9; }
  return dist;
}
fn putEnd(o: u32, gap: f32, rel: f32) {
  if (gap < RANGE && gap / RANGE < inputs[o]) { inputs[o] = gap / RANGE; inputs[o + 1u] = rel; }
}
fn adjustInputs(i: u32, b: u32, v: Veh, pi: VPar, leftEx: bool, rightEx: bool) {
  let sp = v.speed;
  let d = min(pi.delaySteps, CTRL_RING - 1u);
  let step = simStep();
  var steer = v.steerAngle;
  var he = wrapAngle(-v.headingRel);
  let lw = senseLaneW(v);
  var lat = senseLaneOff(v, lw);
  // src/sim/physics stepLateral; position mode (v9): arms lag steerLag (sl starts at the wheel angle, adjust.js),
  // the wheel moves at most steerSlew rad/s
  let posM = (P.steerMode & 1u) != 0u; let netRate = (P.steerMode & 2u) != 0u; var sl = steer;
  let lagK = min(1.0, P.dt / PT_STEERLAG[pi.typeIdx]); let slew = PT_STEERSLEW[pi.typeIdx] * P.dt;
  for (var k = d; k >= 1u; k = k - 1u) {
    let sr = clamp(ring[(i * CTRL_RING + (step + CTRL_RING - k) % CTRL_RING) * CTRL_STRIDE], -1.0, 1.0);
    if (netRate) {
      // rate actions (actionInput 'rate'): in a position-mode sim the ring holds the
      // integrated targets, the rate is their difference
      var r = sr;
      if (posM) { let sp0 = clamp(ring[(i * CTRL_RING + (step + 2u * CTRL_RING - k - 1u) % CTRL_RING) * CTRL_STRIDE], -1.0, 1.0); r = clamp((sr - sp0) * pi.maxSteer / (pi.maxSteerRate * P.dt), -1.0, 1.0); }
      steer = clamp(steer + r * pi.maxSteerRate * P.dt, -pi.maxSteer, pi.maxSteer);
    } else if (posM) { sl = sl + (sr * pi.maxSteer - sl) * lagK; steer = clamp(steer + clamp(sl - steer, -slew, slew), -pi.maxSteer, pi.maxSteer); }
    else { steer = clamp(steer + sr * pi.maxSteerRate * P.dt, -pi.maxSteer, pi.maxSteer); }
    // src/sim/physics stepLateral: yaw rate u * steer / (L + Kus u^2), Kus = m / L (b / Cf - a / Cr)
    let t = pi.typeIdx;
    let kus = PT_M[t] / PT_L[t] * (PT_B[t] / PT_CF[t] - PT_A[t] / PT_CR[t]);
    he = he - (sp * steer) / (pi.wheelbase + kus * sp * sp) * P.dt;
    lat = lat - sp * sin(he) * P.dt;
  }
  inputs[b + 4u] = clamp(lat / lw, -1.0, 1.0);
  inputs[b + 5u] = clamp(he / 0.78539816, -1.0, 1.0);
  inputs[b + OFF_WHEEL] = clamp(steer / pi.maxSteer, -1.0, 1.0);
  let rel = clamp(-sp / 20.0, -1.0, 1.0);
  let half = 0.5 * pi.length + 1.0;
  let lane = i32(vLane(v));
  if (leftEx) { let e = laneEndAhead(i, v, pi, lane + 1); if (e < 1e8) { putEnd(b + OFF_NB + 4u, max(0.0, e - half), rel); } }
  if (rightEx) { let e = laneEndAhead(i, v, pi, lane - 1); if (e < 1e8) { putEnd(b + OFF_NB + 8u, max(0.0, e - half), rel); } }
}
