// Stage (f), layout v4: rigid-body contacts (sim.js _contacts / _barrier / _crashEvent,
// src/sim/physics/collide.js). A crash happens if and only if hitboxes overlap (swept SAT, no
// tunnelling) or a hitbox reaches a road-edge barrier; the response is an impulse with
// restitution, Coulomb friction and angular terms plus positional correction; severity is the
// per-vehicle delta-v. GPU structure:
//   barrier  (parallel, DRIVING cars): static wall contact against the pavement bounds
//   detect   (parallel): hash neighbours j > i, filters as the CPU, swept SAT -> contact list
//   resolve  (one thread, list order): impulse + separation + write-back into the Frenet state;
//            a NEW contact (pair not touching last step, 30 s pair cooldown) with max delta-v >=
//            touchKmh is a crash event; cars with own delta-v >= 10 km/h become CRASHED + DISABLED.
// #include common.wgsl
// #include collidefns.wgsl


@group(1) @binding(0) var<storage, read_write> veh: array<Veh>;
@group(1) @binding(1) var<storage, read> par: array<VPar>;
@group(1) @binding(2) var<storage, read_write> pose: array<vec4f>;
@group(1) @binding(3) var<storage, read_write> ring: array<f32>;
@group(1) @binding(4) var<storage, read> hstarts: array<u32>;
@group(1) @binding(5) var<storage, read> hitems: array<u32>;
// contact list: [0] count (atomic), then 8 words per contact: i, j, t, depth, nx, nz, px, pz
@group(1) @binding(6) var<storage, read_write> contacts: array<atomic<u32>>;
// pair table (resolve only) in the same buffer after the list: 4 words per entry at
// pairBase + 4 h: i + 1 (0 = empty), last touch step, last event time (f32 bits), j; then per-car contact counts
fn pairBase() -> u32 { return 4u + P.contactCap * 8u; }
fn pget(o: u32) -> u32 { return atomicLoad(&contacts[pairBase() + o]); }
fn pset(o: u32, v: u32) { atomicStore(&contacts[pairBase() + o], v); }

const PAIR_N: u32 = 262144u;   // gpuSim.js PAIR_N
const DISABLE_KMH: f32 = 10.0;
const E_CAR: f32 = 0.2;  const MU_CAR: f32 = 0.5;
const E_WALL: f32 = 0.15; const MU_WALL: f32 = 0.4;
const SLOP: f32 = 0.01; const CORR: f32 = 0.8;

fn hlOf(i: u32) -> f32 { return 0.5 * par[i].length; }
fn hwOf(i: u32) -> f32 { return 0.5 * par[i].width; }
fn isMoving(v: Veh) -> bool { return v.state == DRIVING || v.speed > 0.01 || abs(v.vLat) > 0.01; }
fn bodyOf(i: u32, moving: bool) -> Body {
  let v = veh[i]; let p = pose[i];
  var b: Body;
  b.x = p.x; b.z = p.z; b.yaw = p.w; b.hl = hlOf(i); b.hw = hwOf(i);
  let fx = cos(p.w); let fz = sin(p.w);
  b.vx = v.speed * fx + v.vLat * fz; b.vz = v.speed * fz - v.vLat * fx; b.w = -v.yawRate;
  let fixed = !moving && v.state == CRASHED;
  let t = par[i].typeIdx;
  b.invMass = select(1.0 / PT_M[t], 0.0, fixed);
  b.invInertia = select(1.0 / PT_IZ[t], 0.0, fixed);   // vs.inertia = VEHICLE_PARAMS[type].Iz
  return b;
}
// write a body's velocity / position change back into slot i (sim.js _writeBack); x0,z0 = pose before
fn writeBack(i: u32, b: Body, x0: f32, z0: f32) {
  if (b.invMass == 0.0) { return; }
  let p = pose[i]; let fx = cos(p.w); let fz = sin(p.w);
  veh[i].speed = max(0.0, b.vx * fx + b.vz * fz);
  veh[i].vLat = b.vx * fz - b.vz * fx;
  veh[i].yawRate = -b.w;
  let dx = b.x - x0; let dz = b.z - z0;
  if (dx != 0.0 || dz != 0.0) {
    let s = veh[i].s;
    let rd = frameDir(veh[i]);
    veh[i].s = clamp(s + dx * rd.x + dz * rd.y, 0.0, frameEnd(veh[i]));
    veh[i].lat = veh[i].lat + (dx * rd.y - dz * rd.x);
    refreshPose(i);
  }
}
fn refreshPose(i: u32) {
  let v = veh[i];
  veh[i].lanes = derivedLanes(v.edge, v.s, v.lat, halfExtent(v, par[i]), vTarget(v));
  pose[i] = poseOf(v);
}
fn disable(j: u32) {
  veh[j].state = CRASHED; veh[j].flags = veh[j].flags | F_DISABLED;
  veh[j].steerSl = veh[j].steerAngle; veh[j].gas = 0.0; veh[j].brake = 0.0;
  for (var k = 0u; k < CTRL_RING * CTRL_STRIDE; k = k + 1u) { ring[j * CTRL_RING * CTRL_STRIDE + k] = 0.0; }
  emitWreck(j, veh[j], par[j].typeIdx);
}
// reward.js crashPenalty: min(cap, base + fail * disabled + k * dv * (1 + fault * (2 share - 1)))
fn crashCharge(k: u32, kmh: f32, share: f32, disabled: bool, wall: bool) {
  let sev = max(kmh, 0.0) * (1.0 + P.crashFault * (2.0 * share - 1.0));
  let pen = min(P.crashCap, P.crashBase + select(0.0, P.crashFail, disabled) + P.crashK * sev);
  let prev = veh[k].eventTime;
  veh[k].eventTime = select(0.0, prev, prev < 0.0) - pen;
  emit(EV_CRASHPEN, k, bitcast<u32>(pen), min(u32(kmh * 10.0), 65535u) | (u32(clamp(share, 0.0, 1.0) * 255.0 + 0.5) << 16u) | (select(0u, 1u, disabled) << 24u) | (select(0u, 1u, wall) << 25u));
}
// sim.js _crashEvent: below touchKmh a touch; otherwise flag the cars, disable >= 10 km/h, report
fn crashEvent(i: u32, j: u32, dvi: f32, dvj: f32, closing: f32, cpx: f32, cpz: f32, si: f32, sj_: f32, nx: f32, nz: f32, ps: u32, lastEv: f32) {
  let ki = dvi * 3.6; let kj = dvj * 3.6;
  if (max(ki, kj) < P.touchKmh) { return; }
  // one crash per pair of cars, decided after the touch threshold (sim.js _crashEvent)
  var repeat = false;
  if (j != NONE) {
    repeat = lastEv >= veh[i].spawnTime && lastEv >= veh[j].spawnTime;
    if (!repeat && ps != NONE) { pset(4u * ps + 2u, bitcast<u32>(simTime())); }
  }
  // sim.js (layout v8): the same pair touching again is not a new crash unless this contact disables a driving car
  if (repeat && !((veh[i].state == DRIVING && ki >= DISABLE_KMH) || (j != NONE && veh[j].state == DRIVING && kj >= DISABLE_KMH))) { statAdd(S_REPEAT_CONTACTS, 1u); return; }
  // detail: si / sj are the speeds before this contact's impulse (the caller's copies)
  let ci = veh[i].state == CRASHED; let cj = j != NONE && veh[j].state == CRASHED;   // before this event disables anyone (sim.js order)
  let hi = pose[i].w; let li = vLo(veh[i]);
  var sj = 0.0; var hj = 0.0; var lj = 255u;
  if (j != NONE) { sj = sj_; hj = pose[j].w; lj = vLo(veh[j]); }
  var n = 0u;
  veh[i].flags = veh[i].flags | F_COLLIDED;
  if (veh[i].state == DRIVING && ki >= DISABLE_KMH) { disable(i); n = n + 1u; }
  if (j != NONE) {
    veh[j].flags = veh[j].flags | F_COLLIDED;
    if (veh[j].state == DRIVING && kj >= DISABLE_KMH) { disable(j); n = n + 1u; }
  }
  // rewardVersion 2 (P.crashOn; src/sim/reward.js crashPenalty / crashShare, docs/crash-penalty.md): a severity-scaled
  // penalty per car that was not already a wreck, stashed (negative) in veh.eventTime (unused while DRIVING / CRASHED)
  // and charged by integrate.wgsl; one EV_CRASHPEN record per charged car for the host log
  if (P.crashOn != 0u) {
    var cA = si * (cos(hi) * nx + sin(hi) * nz); var cB = 0.0;
    if (j != NONE) { cB = -sj * (cos(hj) * nx + sin(hj) * nz); }
    if (cA + cB < 0.0) { cA = -cA; cB = -cB; }
    cA = max(cA, 0.0); cB = max(cB, 0.0);
    let ss = cA + cB; let shI = select(cA / max(ss, 1e-6), 0.5, ss < 1e-3); let shJ = select(1.0 - shI, 0.5, ss < 1e-3);
    if (!ci) { crashCharge(i, ki, shI, veh[i].state == CRASHED, j == NONE); }
    if (j != NONE && !cj) { crashCharge(j, kj, shJ, veh[j].state == CRASHED, false); }
  }
  // CRASH: (slot, other, packed delta-v km/h x10 i | j << 16), MINOR contact (no car disabled) as CONTACT
  let pk = min(u32(ki * 10.0), 65535u) | (min(u32(kj * 10.0), 65535u) << 16u);
  if (n > 0u) {
    emit(EV_CRASHX_V, i, j, dms(si) | (dms(sj) << 16u));
    emit(EV_CRASHX_P, i, bitcast<u32>(cpx), bitcast<u32>(cpz));
    emit(EV_CRASHX_N, i, bitcast<u32>(nx), bitcast<u32>(nz));
    // maneuver of the striking (faster) car, sim.js _classifyCrash
    var st = i; if (j != NONE && sj > si) { st = j; }
    let vs = veh[st]; let es = vs.edge; let Ls = eLength(es);
    let dNode = min(vs.s, Ls - vs.s);
    let tl = vTarget(vs);
    let changing = vHi(vs) > vLo(vs) && vConn(vs) < 0 && ((tl >= 0 && tl != i32(vLane(vs))) || abs(vs.headingRel) > 0.035 || abs(vs.vLat) > 0.3);   // body straddles a lane line
    var man = 0u;
    if (j != NONE && ci && cj) { man = 5u; }
    else if (changing) { man = 1u; }
    else if (j != NONE && veh[i].edge != veh[j].edge && dNode < 40.0) { man = select(3u, 2u, (eFlags(es) & 64u) != 0u); }
    else if (vs.leader >= 0 && (veh[u32(vs.leader)].state != DRIVING || veh[u32(vs.leader)].speed < 2.0) && vs.speed > 5.0) { man = 4u; }
    else if (dNode < 30.0 && inStart(eTo(es) + 1u) - inStart(eTo(es)) > 1u) { man = 3u; }
    emit(EV_CRASHX_M, i, man | (select(0u, 256u, ci) | select(0u, 512u, cj)), es);
    emit(EV_CRASHX_H, i, u16Heading(hi) | (u16Heading(hj) << 16u), dms(closing) | (min(li, 255u) << 16u) | (min(lj, 255u) << 24u));
  }
  // minor contact: bits 8/9 = car i / j DRIVING on a freeway-class road (sim.js pull-over to the shoulder)
  var pullBits = 0u;
  // sim.js PULLOVER_ROADS: motorway / trunk mainline (edge flag 16 = not links)
  if (veh[i].state == DRIVING && (eFlags(veh[i].edge) & 16u) != 0u) { pullBits = pullBits | 256u; }
  if (j != NONE && veh[j].state == DRIVING && (eFlags(veh[j].edge) & 16u) != 0u) { pullBits = pullBits | 512u; }
  emit(select(13u | pullBits, EV_CRASH, n > 0u), i, j, pk);
  statAdd(S_CRASH_EVENTS, 1u);
  statAdd(S_CRASHED_VEH, n);
}

// ---------------------------------------------------------------- barriers (parallel)
@compute @workgroup_size(128)
fn barrier(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= P.cap) { return; }
  let v0 = veh[i];
  if (v0.state != DRIVING) { return; }
  // rotate the touch flags (this step's touch becomes "previous")
  let prev = (v0.flags & F_WALL) != 0u;
  veh[i].flags = (v0.flags & ~(F_WALL | F_WALL_PREV)) | select(0u, F_WALL_PREV, prev);
  if (hasFeat(FEAT_EMPTY)) { return; }
  if (vConn(v0) >= 0) { return; }   // no edge barriers inside a junction box (sim.js)
  if (P.hasJ != 0u && (v0.s < eSExit(v0.edge) + 1.0 || v0.s > eSStop(v0.edge))) { return; }   // v9: nor at the box ends of an edge
  let e = v0.edge; let W = laneWAt(e, v0.s);
  var padR = 0.0; var padL = f32(eLanes(e)) * W;
  if (P.hasLG != 0u) { let bd = boundsAt(e, v0.s); padR = bd.z - bd.x; padL = bd.z - bd.y; }
  else { let tp = select(0.0, W, v0.taper > v0.s); if (v0.lat < 0.0) { padR = padR - tp; } if (v0.lat > padL) { padL = padL + tp; } }
  let right = padR - select(P.shoulder, P.freewayShoulder, (eFlags(e) & 64u) != 0u);   // sim.js: 3 m outside shoulder on freeway-class roads
  var left = padL + P.shoulder;
  if (eRev(e) >= 0) { left = padL + f32(eLanes(u32(eRev(e)))) * W + P.oppositeMargin; }
  let hr = v0.headingRel; let sh = sin(hr); let ch = cos(hr); let hl = hlOf(i); let hw = hwOf(i); let lat = v0.lat;
  let ext = abs(sh) * hl + abs(ch) * hw;
  var side = 0.0; var depth = 0.0;
  if (lat - ext < right) { side = -1.0; depth = right - (lat - ext); }
  else if (lat + ext > left) { side = 1.0; depth = lat + ext - left; }
  if (side == 0.0) { return; }
  let p = pose[i]; let fx = cos(p.w); let fz = sin(p.w);
  var best = 1e9; var cx = 0.0; var cz = 0.0;
  for (var a = -1.0; a <= 1.0; a = a + 2.0) {
    for (var b2 = -1.0; b2 <= 1.0; b2 = b2 + 2.0) {
      let l = -side * (a * hl * sh + b2 * hw * ch);
      if (l < best) { best = l; cx = p.x + a * hl * fx + b2 * hw * fz; cz = p.z + a * hl * fz - b2 * hw * fx; }
    }
  }
  let rd = roadDirAt(e, v0.s);
  var c: Contact; c.ok = true; c.depth = depth;
  c.nx = select(-rd.y, rd.y, side > 0.0); c.nz = select(rd.x, -rd.x, side > 0.0); c.px = cx; c.pz = cz;
  var A = bodyOf(i, true);
  var Wb: Body; Wb.x = cx; Wb.z = cz; Wb.yaw = 0.0; Wb.vx = 0.0; Wb.vz = 0.0; Wb.w = 0.0; Wb.hl = 0.0; Wb.hw = 0.0; Wb.invMass = 0.0; Wb.invInertia = 0.0;
  let res = resolveContact(&A, &Wb, c, E_WALL, MU_WALL);
  writeBack(i, A, A.x, A.z);
  veh[i].lat = veh[i].lat - side * depth;
  refreshPose(i);
  veh[i].flags = veh[i].flags | F_WALL;
  if (!prev && res.jn > 0.0) { crashEvent(i, NONE, res.dvA, 0.0, res.closing, cx, cz, v0.speed, 0.0, c.nx, c.nz, NONE, -1e9); }
}

// ---------------------------------------------------------------- detection (parallel)
fn cellOf2(cx: i32, cz: i32) -> u32 { return (bitcast<u32>(cx) & P.wMask) | ((bitcast<u32>(cz) & P.hMask) << P.wBits); }
// sim.js _adjacentOrFreewayFree: same / adjacent edges always; a freeway-class edge (SHOULDER_ROADS,
// flag 64) and an edge it shares no node with only for pairs in data/crossings.json (dy < 2.5 m or
// pavement overlaps: sorted (lo, hi) pairs at P.pairOff); without that data never
fn adjacentOrFreewayFree(a: u32, b: u32) -> bool {
  if (eFrom(a) == eFrom(b) || eFrom(a) == eTo(b) || eTo(a) == eFrom(b) || eTo(a) == eTo(b)) { return true; }
  if ((eFlags(a) & 64u) == 0u && (eFlags(b) & 64u) == 0u) { return true; }
  let lo = min(a, b); let hi = max(a, b);
  var l = 0u; var r = P.pairN;
  loop {
    if (l >= r) { break; }
    let m = (l + r) / 2u; let ma = world[P.pairOff + 2u * m]; let mb = world[P.pairOff + 2u * m + 1u];
    if (ma < lo || (ma == lo && mb < hi)) { l = m + 1u; } else { r = m; }
  }
  return l < P.pairN && world[P.pairOff + 2u * l] == lo && world[P.pairOff + 2u * l + 1u] == hi;
}
// previous-step pose; a jump > MAX_SWEEP (teleport: scatter / import / junction re-projection
// without a saved pose) counts as no sweep, or the detect cell range would explode
const MAX_SWEEP: f32 = 30.0;
fn prevOk(i: u32) -> bool { let v = veh[i]; let p = pose[i]; return length(vec2f(p.x - v.px, p.z - v.pz)) <= MAX_SWEEP; }
fn startBody(i: u32, b: Body) -> Body { var b0 = b; if (!prevOk(i)) { return b0; } let v = veh[i]; b0.x = v.px; b0.z = v.pz; b0.yaw = v.pyaw; return b0; }

@compute @workgroup_size(128)
fn detect(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= P.cap) { return; }
  let vi = veh[i];
  if (vi.state != DRIVING && vi.state != CRASHED) { return; }
  if (hasFeat(FEAT_EMPTY) && vi.state == DRIVING) { return; }
  let movI = isMoving(vi);
  let pi_ = pose[i];
  var sweepI = length(vec2f(pi_.x - vi.px, pi_.z - vi.pz)); if (sweepI > MAX_SWEEP) { sweepI = 0.0; }
  let hdI = length(vec2f(hlOf(i), hwOf(i)));
  let R = hdI + 6.5 + sweepI + 5.0;
  let cx0 = i32(floor((pi_.x - R) / P.cellSize)); let cx1 = i32(floor((pi_.x + R) / P.cellSize));
  let cz0 = i32(floor((pi_.z - R) / P.cellSize)); let cz1 = i32(floor((pi_.z + R) / P.cellSize));
  for (var cz = cz0; cz <= cz1; cz = cz + 1) {
    for (var cx = cx0; cx <= cx1; cx = cx + 1) {
      let c = cellOf2(cx, cz);
      for (var k = hstarts[c]; k < hstarts[c + 1u]; k = k + 1u) {
        let j = hitems[k];
        if (j <= i) { continue; }
        let vj = veh[j];
        if (vj.state != DRIVING && vj.state != CRASHED) { continue; }
        let movJ = isMoving(vj);
        if (!movI && !movJ) { continue; }
        let pj = pose[j];
        if (abs(pj.y - pi_.y) > P.layerGap) { continue; }
        if (vj.edge == vi.edge && abs(vj.s - vi.s) > hlOf(i) + hlOf(j) + 1.0) { continue; }
        if (vj.edge != vi.edge && !adjacentOrFreewayFree(vi.edge, vj.edge)) { continue; }
        var sweepJ = length(vec2f(pj.x - vj.px, pj.z - vj.pz)); if (sweepJ > MAX_SWEEP) { sweepJ = 0.0; }
        let hdJ = length(vec2f(hlOf(j), hwOf(j)));
        if (length(vec2f(pj.x - pi_.x, pj.z - pi_.z)) > hdI + hdJ + sweepI + sweepJ) { continue; }
        let A = bodyOf(i, movI); let B = bodyOf(j, movJ);
        let A0 = startBody(i, A); let B0 = startBody(j, B);
        // swept test: relative sub-steps <= 0.5 m
        let rel = length(vec2f((A.x - A0.x) - (B.x - B0.x), (A.z - A0.z) - (B.z - B0.z)));
        let n = max(1u, min(64u, u32(ceil(rel / 0.5))));
        for (var q = 1u; q <= n; q = q + 1u) {
          let t = f32(q) / f32(n);
          var ta = A; ta.x = A0.x + (A.x - A0.x) * t; ta.z = A0.z + (A.z - A0.z) * t; ta.yaw = lerpAngle(A0.yaw, A.yaw, t);
          var tb = B; tb.x = B0.x + (B.x - B0.x) * t; tb.z = B0.z + (B.z - B0.z) * t; tb.yaw = lerpAngle(B0.yaw, B.yaw, t);
          let cc = obbContact(ta, tb);
          if (cc.ok) {
            let slot = atomicAdd(&contacts[0], 1u);
            if (slot < P.contactCap) {
              let o = 4u + slot * 8u;
              atomicStore(&contacts[o], i); atomicStore(&contacts[o + 1u], j); atomicStore(&contacts[o + 2u], bitcast<u32>(t));
              atomicStore(&contacts[o + 3u], bitcast<u32>(cc.depth)); atomicStore(&contacts[o + 4u], bitcast<u32>(cc.nx)); atomicStore(&contacts[o + 5u], bitcast<u32>(cc.nz));
              atomicStore(&contacts[o + 6u], bitcast<u32>(cc.px)); atomicStore(&contacts[o + 7u], bitcast<u32>(cc.pz));
            }
            break;
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------- resolve
// Contacts that share no car commute, so most are resolved in parallel rounds (ownClear / ownMin /
// resolveRound); the rest on one thread in the CPU order (pairs by slot i, then j).
// Per-car owner words live after the pair table: ccBase() + slot.
fn ccBase() -> u32 { return pairBase() + PAIR_N * 4u; }

// pair table entry (4 words): i + 1 (0 = empty), last touch step, last event time (f32 bits), j.
// Claimed with compare-exchange so parallel resolvers never share an entry.
fn pairSlot(i: u32, j: u32) -> u32 {
  var h = pcg(i ^ pcg(j + 0x9e3779b9u)) % PAIR_N;
  let now = simTime(); let step = simStep();
  var free = NONE; var freeKey = 0u;
  for (var k = 0u; k < 32u; k = k + 1u) {
    let o = 4u * h;
    let kk = pget(o);
    if (kk == i + 1u && pget(o + 3u) == j) { return h; }
    let stale = kk == 0u || (pget(o + 1u) + 1u < step && now - bitcast<f32>(pget(o + 2u)) > P.pairCooldown);
    if (stale && free == NONE) { free = h; freeKey = kk; }
    if (kk == 0u) { break; }
    h = (h + 1u) % PAIR_N;
  }
  if (free == NONE) { return NONE; }
  let o = 4u * free;
  let r = atomicCompareExchangeWeak(&contacts[pairBase() + o], freeKey, i + 1u);
  if (!r.exchanged) { return NONE; }   // lost a race (parallel pass): no cooldown bookkeeping this step
  pset(o + 3u, j); pset(o + 1u, 0u); pset(o + 2u, bitcast<u32>(-1e9));
  return free;
}

fn resolveOne(k: u32) {
  let step = simStep(); let now = simTime();
  let o = 4u + k * 8u;
  let i = atomicLoad(&contacts[o]); let j = atomicLoad(&contacts[o + 1u]); let t = bitcast<f32>(atomicLoad(&contacts[o + 2u]));
  var c: Contact; c.ok = true;
  c.depth = bitcast<f32>(atomicLoad(&contacts[o + 3u])); c.nx = bitcast<f32>(atomicLoad(&contacts[o + 4u])); c.nz = bitcast<f32>(atomicLoad(&contacts[o + 5u]));
  c.px = bitcast<f32>(atomicLoad(&contacts[o + 6u])); c.pz = bitcast<f32>(atomicLoad(&contacts[o + 7u]));
  let vi = veh[i]; let vj = veh[j];
  let movI = isMoving(vi); let movJ = isMoving(vj);
  var A = bodyOf(i, movI); var B = bodyOf(j, movJ);
  let ax0 = A.x; let az0 = A.z; let bx0 = B.x; let bz0 = B.z;
  if (t < 1.0 && prevOk(i) && prevOk(j)) {
    A.x = vi.px + (A.x - vi.px) * t; A.z = vi.pz + (A.z - vi.pz) * t;
    B.x = vj.px + (B.x - vj.px) * t; B.z = vj.pz + (B.z - vj.pz) * t;
  }
  let res = resolveContact(&A, &B, c, E_CAR, MU_CAR);
  // separate (collide.js): split the correction by inverse mass
  let kM = A.invMass + B.invMass;
  if (kM > 0.0) {
    let corr = max(0.0, c.depth - SLOP) * CORR / kM;
    A.x = A.x - c.nx * corr * A.invMass; A.z = A.z - c.nz * corr * A.invMass;
    B.x = B.x + c.nx * corr * B.invMass; B.z = B.z + c.nz * corr * B.invMass;
  }
  writeBack(i, A, ax0, az0); writeBack(j, B, bx0, bz0);
  // new contact -> crash event (pair not touching last step, cooldown since its last event)
  let ps = pairSlot(i, j);
  var isNew = true; var lastEv = -1e9;
  if (ps != NONE) {
    let po = 4u * ps;
    isNew = pget(po + 1u) + 1u < step;
    lastEv = bitcast<f32>(pget(po + 2u));
    pset(po + 1u, step);
  }
  if (isNew && res.jn > 0.0 && (vi.state == DRIVING || vj.state == DRIVING)) {
    crashEvent(i, j, res.dvA, res.dvB, res.closing, c.px, c.pz, vi.speed, vj.speed, c.nx, c.nz, ps, lastEv);
  }
}

// Parallel rounds (contact graph matching): every unresolved contact claims both its cars with
// atomicMin of a random priority; a contact that owns both cars resolves this round. Contacts left after the rounds go to the serial pass. Done = t word 2.0.
// random priority (unique: low 18 bits = list index, contactCap <= 262144): long contact chains in
// queues resolve in O(log n) rounds instead of O(n)
fn prio(k: u32) -> u32 { return (pcg(k * 2654435761u + 1u) & 0xfffc0000u) | (k & 0x3ffffu); }
fn isDone(o: u32) -> bool { return bitcast<f32>(atomicLoad(&contacts[o + 2u])) > 1.5; }
@compute @workgroup_size(128)
fn ownClear(@builtin(global_invocation_id) g: vec3u) {
  let n = min(atomicLoad(&contacts[0]), P.contactCap);
  if (g.x >= n) { return; }
  let o = 4u + g.x * 8u;
  if (isDone(o)) { return; }
  atomicStore(&contacts[ccBase() + atomicLoad(&contacts[o])], NONE);
  atomicStore(&contacts[ccBase() + atomicLoad(&contacts[o + 1u])], NONE);
}
@compute @workgroup_size(128)
fn ownMin(@builtin(global_invocation_id) g: vec3u) {
  let n = min(atomicLoad(&contacts[0]), P.contactCap);
  if (g.x >= n) { return; }
  let o = 4u + g.x * 8u;
  if (isDone(o)) { return; }
  let key = prio(g.x);
  atomicMin(&contacts[ccBase() + atomicLoad(&contacts[o])], key);
  atomicMin(&contacts[ccBase() + atomicLoad(&contacts[o + 1u])], key);
}
@compute @workgroup_size(128)
fn resolveRound(@builtin(global_invocation_id) g: vec3u) {
  let n = min(atomicLoad(&contacts[0]), P.contactCap);
  if (g.x >= n) { return; }
  let o = 4u + g.x * 8u;
  if (isDone(o)) { return; }
  let key = prio(g.x);
  if (atomicLoad(&contacts[ccBase() + atomicLoad(&contacts[o])]) != key || atomicLoad(&contacts[ccBase() + atomicLoad(&contacts[o + 1u])]) != key) { return; }
  resolveOne(g.x);
  atomicStore(&contacts[o + 2u], bitcast<u32>(2.0));
}

fn lessPair(i: u32, j: u32, i2: u32, j2: u32) -> bool { return i < i2 || (i == i2 && j < j2); }

@compute @workgroup_size(1)
fn resolve() {
  let n = min(atomicLoad(&contacts[0]), P.contactCap);
  // compact the shared-car contacts (resolveFree marked the rest NONE), keeping list order
  var m = 0u;
  for (var k = 0u; k < n; k = k + 1u) {
    let o = 4u + k * 8u;
    if (isDone(o)) { continue; }
    if (m != k) { for (var w = 0u; w < 8u; w = w + 1u) { atomicStore(&contacts[4u + m * 8u + w], atomicLoad(&contacts[o + w])); } }
    m = m + 1u;
  }
  // CPU order: by slot i, then j (insertion sort; only for short lists, a scattered start can
  // produce thousands of overlaps)
  let nSort = select(0u, m, m <= 512u);
  for (var a = 1u; a < nSort; a = a + 1u) {
    var rec: array<u32, 8>;
    for (var w = 0u; w < 8u; w = w + 1u) { rec[w] = atomicLoad(&contacts[4u + a * 8u + w]); }
    var b = a;
    loop {
      if (b == 0u) { break; }
      let o = 4u + (b - 1u) * 8u;
      if (!lessPair(rec[0], rec[1], atomicLoad(&contacts[o]), atomicLoad(&contacts[o + 1u]))) { break; }
      for (var w = 0u; w < 8u; w = w + 1u) { atomicStore(&contacts[o + 8u + w], atomicLoad(&contacts[o + w])); }
      b = b - 1u;
    }
    for (var w = 0u; w < 8u; w = w + 1u) { atomicStore(&contacts[4u + b * 8u + w], rec[w]); }
  }
  for (var k = 0u; k < m; k = k + 1u) { resolveOne(k); }
  atomicStore(&io[IO_STATS + 46u], atomicLoad(&contacts[0])); atomicStore(&io[IO_STATS + 47u], m);   // diagnostics: contacts found / left to the serial pass
  atomicStore(&contacts[0], 0u);
}
