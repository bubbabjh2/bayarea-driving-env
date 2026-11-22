// bayenv-gpu model patches of the contacts stage (src/gpu/kernels/contacts.wgsl, also included by kernels/resolve_fused.wgsl),
// applied by kernels.rs `model_patches` when any of the GpuOptions flags below is on (each //#orig snippet must occur exactly
// once in the resolved source). The host prepends `const MF_*: bool` for every flag; a flag that is off takes the JS code
// path, so the patched source with all MF_* false computes the JS kernel. Mirrors bayenv-sim SimOpts (contacts.rs);
// see rust/docs/physics-audit.md.
//   MF_WRECK_REAL_MASS  wreck_real_mass: a crashed car at rest keeps its mass / inertia (JS: immovable). Walls stay immovable.
//   MF_IMPACT_REVERSE   impact_reverse: |speed| > 0.01 in the "still moving" test; writeBack keeps the backward speed of a
//                       CRASHED car; a DRIVING car pushed backwards keeps 0 unless this contact disables it (stash below).
//   MF_SOFT_PUSH        barrier_soft_push: barrier position correction min(depth, 0.25 m) per step (JS: the full depth).
//   MF_EPISODES         contact_episodes: impact steps (first touch, or own delta-v x 3.6 >= touchKmh; pairs: the larger of the
//                       two) add to an episode total and report an event with the total as its delta-v; the pair repeat
//                       filter is bypassed for later steps of an episode. Accumulators after the per-car owner words:
//                       epBase() + slot (barrier, f32), epBase() + cap + 2 pairSlot (pair, f32 i / j); the host sizes the
//                       contacts buffer for them when the flag is on.
//   MF_CONN_BARRIER     connector_barrier: cars on a junction connector get the in-edge's kerbs as lateral walls.
//   MF_BOX_BARRIER      box_barrier (bayenv-sim contacts.rs, DECISIONS.md 39): the junction-box parts of an edge (before
//                       sExit + 1, past sStop) get the edge barrier too, its lines widened by BOX_MARGIN (JS: no walls there).
//   MF_MOUNTABLE_KERBS  mountable_kerbs (bayenv-sim contacts.rs, DECISIONS.md 38): a barrier contact on a non-rigid edge (not
//                       the freeway system, edge flag 64, nor a bridge, edge flag 2) closing at under 2 m/s is a kerb scrape:
//                       restitution 0, friction 0.6, correction min(max(0, depth - 0.3), 0.25), the episode total is not
//                       increased, one event on the first touch with delta-v capped below the disable threshold. Closing
//                       speed at the contact point; on a kerb a later non-scrape step adds to the episode only when its
//                       own delta-v >= 2 m/s (a grind never sums to a crash).
//#orig barrier-resolve
  let res = resolveContact(&A, &Wb, c, E_WALL, MU_WALL);
//#end
//#patch barrier-resolve
  let scrape = kerbScrape(e, A, c);
  let res = resolveContact(&A, &Wb, c, select(E_WALL, 0.0, scrape), select(MU_WALL, KERB_FRICTION, scrape));
//#end
//#orig moving
fn isMoving(v: Veh) -> bool { return v.state == DRIVING || v.speed > 0.01 || abs(v.vLat) > 0.01; }
//#end
//#patch moving
fn isMoving(v: Veh) -> bool { return v.state == DRIVING || select(v.speed, abs(v.speed), MF_IMPACT_REVERSE) > 0.01 || abs(v.vLat) > 0.01; }
const BARRIER_PUSH_MAX: f32 = 0.25;
const BOX_MARGIN: f32 = 3.5;
fn barrierCorr(d: f32) -> f32 { return select(d, min(d, BARRIER_PUSH_MAX), MF_SOFT_PUSH); }
const KERB_MOUNT_SPEED: f32 = 2.0; const KERB_MOUNT_DEPTH: f32 = 0.3; const KERB_FRICTION: f32 = 0.6;
const KERB_SCRAPE_MAX_DV: f32 = (DISABLE_KMH - 0.5) / 3.6;
// mountable_kerbs: a low-speed contact with a kerb that is not a rigid barrier (freeway system / bridge) is a scrape
// closing speed at the contact point (body velocity + yaw rate x lever arm), as resolveContact sees it
fn kerbScrape(e: u32, A: Body, c: Contact) -> bool {
  let into = (A.vx - A.w * (c.pz - A.z)) * c.nx + (A.vz + A.w * (c.px - A.x)) * c.nz;
  return MF_MOUNTABLE_KERBS && !rigidEdge(e) && into < KERB_MOUNT_SPEED;
}
fn rigidEdge(e: u32) -> bool { return (eFlags(e) & 66u) != 0u; }   // freeway system (64) or bridge (2)
fn barrierCorrK(d: f32, scrape: bool) -> f32 { if (scrape) { return min(max(0.0, d - KERB_MOUNT_DEPTH), BARRIER_PUSH_MAX); } return barrierCorr(d); }
// impact_reverse: backward speeds of DRIVING cars from the current contact (writeBack), applied if it disables them
var<private> revN: u32 = 0u;
var<private> revSlot: array<u32, 2>;
var<private> revU: array<f32, 2>;
fn epBase() -> u32 { return ccBase() + P.cap; }
// barrier event (JS: first touch only; MF_EPISODES: every impact step, delta-v = the episode total)
fn barrierEvent(i: u32, prev: bool, res: Res, cx: f32, cz: f32, sp: f32, nx: f32, nz: f32, scrape: bool, e: u32) {
  if (scrape) {
    // a kerb scrape: reported once (first touch, delta-v below the disable threshold), the episode total carried
    if (MF_EPISODES) { let o = epBase() + i; atomicStore(&contacts[o], bitcast<u32>(select(0.0, bitcast<f32>(atomicLoad(&contacts[o])), prev))); }
    if (!prev && res.jn > 0.0) { crashEvent(i, NONE, min(res.dvA, KERB_SCRAPE_MAX_DV), 0.0, res.closing, cx, cz, sp, 0.0, nx, nz, NONE, -1e9); }
    return;
  }
  if (!MF_EPISODES) {
    if (!prev && res.jn > 0.0) { crashEvent(i, NONE, res.dvA, 0.0, res.closing, cx, cz, sp, 0.0, nx, nz, NONE, -1e9); }
    return;
  }
  let impact = !prev || res.dvA * 3.6 >= P.touchKmh;
  // on a kerb (not rigid), only real strikes (own delta-v >= KERB_MOUNT_SPEED) add to the episode
  let counts = impact && (!MF_MOUNTABLE_KERBS || rigidEdge(e) || res.dvA >= KERB_MOUNT_SPEED);
  let o = epBase() + i;
  var tot = select(0.0, bitcast<f32>(atomicLoad(&contacts[o])), prev);
  if (counts) { tot = tot + res.dvA; }
  atomicStore(&contacts[o], bitcast<u32>(tot));
  if (res.jn > 0.0 && (counts || !prev)) { crashEvent(i, NONE, tot, 0.0, res.closing, cx, cz, sp, 0.0, nx, nz, NONE, -1e9); }
}
// connector_barrier (bayenv-sim contacts.rs connector_limits / connector_barrier): the in-edge's kerbs at the movement's
// sIn, in the in-edge lateral coordinate the connector keeps; the wall normal from the connector tangent at the car
fn connBarrier(i: u32, v0: Veh, prev: bool) {
  let m = u32(vConn(v0)); let e = v0.edge; let sIn = movSIn(m);
  let W = laneWAt(e, sIn);
  var padR = 0.0; var padL = f32(eLanes(e)) * W;
  if (P.hasLG != 0u) { let bd = boundsAt(e, min(sIn, eLength(e))); padR = bd.z - bd.x; padL = bd.z - bd.y; }
  let right = padR - select(P.shoulder, P.freewayShoulder, (eFlags(e) & 64u) != 0u);
  var left = padL + P.shoulder;
  if (eRev(e) >= 0) { left = padL + f32(eLanes(u32(eRev(e)))) * W + P.oppositeMargin; }
  let hr = v0.headingRel; let hl = hlOf(i); let hw = hwOf(i); let lat = v0.lat;
  let ext = abs(sin(hr)) * hl + abs(cos(hr)) * hw;
  var side = 0.0; var depth = 0.0;
  if (lat - ext < right) { side = -1.0; depth = right - (lat - ext); }
  else if (lat + ext > left) { side = 1.0; depth = lat + ext - left; }
  if (side == 0.0) { return; }
  let cp = connPointJ(m, v0.s - sIn);
  let nx = select(-cp.hz, cp.hz, side > 0.0); let nz = select(cp.hx, -cp.hx, side > 0.0);
  // the footprint corner deepest toward the wall (largest p . n)
  let p = pose[i]; let fx = cos(p.w); let fz = sin(p.w);
  var best = -1e30; var cx = p.x; var cz = p.z;
  for (var a = -1.0; a <= 1.0; a = a + 2.0) {
    for (var b2 = -1.0; b2 <= 1.0; b2 = b2 + 2.0) {
      let qx = p.x + a * hl * fx + b2 * hw * fz; let qz = p.z + a * hl * fz - b2 * hw * fx;
      let d = qx * nx + qz * nz;
      if (d > best) { best = d; cx = qx; cz = qz; }
    }
  }
  var c: Contact; c.ok = true; c.depth = depth; c.nx = nx; c.nz = nz; c.px = cx; c.pz = cz;
  var A = bodyOf(i, true);
  var Wb: Body; Wb.x = cx; Wb.z = cz; Wb.yaw = 0.0; Wb.vx = 0.0; Wb.vz = 0.0; Wb.w = 0.0; Wb.hl = 0.0; Wb.hw = 0.0; Wb.invMass = 0.0; Wb.invInertia = 0.0;
  let scrape = kerbScrape(e, A, c);
  let res = resolveContact(&A, &Wb, c, select(E_WALL, 0.0, scrape), select(MU_WALL, KERB_FRICTION, scrape));
  writeBack(i, A, A.x, A.z);
  veh[i].lat = veh[i].lat - side * barrierCorrK(depth, scrape);
  refreshPose(i);
  veh[i].flags = veh[i].flags | F_WALL;
  barrierEvent(i, prev, res, cx, cz, v0.speed, nx, nz, scrape, e);
}
//#end
//#orig fixed
  let fixed = !moving && v.state == CRASHED;
//#end
//#patch fixed
  let fixed = !MF_WRECK_REAL_MASS && !moving && v.state == CRASHED;
//#end
//#orig writeback-speed
  veh[i].speed = max(0.0, b.vx * fx + b.vz * fz);
//#end
//#patch writeback-speed
  let uB = b.vx * fx + b.vz * fz;
  let keepRev = MF_IMPACT_REVERSE && uB < 0.0 && veh[i].state == CRASHED;
  veh[i].speed = select(max(0.0, uB), uB, keepRev);
  if (MF_IMPACT_REVERSE && uB < 0.0 && !keepRev && revN < 2u) { revSlot[revN] = i; revU[revN] = uB; revN = revN + 1u; }
//#end
//#orig disable-rev
  veh[j].state = CRASHED; veh[j].flags = veh[j].flags | F_DISABLED;
//#end
//#patch disable-rev
  veh[j].state = CRASHED; veh[j].flags = veh[j].flags | F_DISABLED;
  for (var q = 0u; q < revN; q = q + 1u) { if (revSlot[q] == j) { veh[j].speed = revU[q]; } }
//#end
//#orig conn-skip
  if (vConn(v0) >= 0) { return; }   // no edge barriers inside a junction box (sim.js)
//#end
//#patch conn-skip
  revN = 0u;
  if (vConn(v0) >= 0) { if (MF_CONN_BARRIER && P.hasJ != 0u) { connBarrier(i, v0, prev); } return; }   // JS: no barriers inside a junction box
//#end
//#orig box-skip
  if (P.hasJ != 0u && (v0.s < eSExit(v0.edge) + 1.0 || v0.s > eSStop(v0.edge))) { return; }   // v9: nor at the box ends of an edge
//#end
//#patch box-skip
  let inBox = P.hasJ != 0u && (v0.s < eSExit(v0.edge) + 1.0 || v0.s > eSStop(v0.edge));
  if (inBox && !MF_BOX_BARRIER) { return; }   // v9: nor at the box ends of an edge (box_barrier: widened walls there)
//#end
//#orig box-limits
  let right = padR - select(P.shoulder, P.freewayShoulder, (eFlags(e) & 64u) != 0u);   // sim.js: 3 m outside shoulder on freeway-class roads
  var left = padL + P.shoulder;
  if (eRev(e) >= 0) { left = padL + f32(eLanes(u32(eRev(e)))) * W + P.oppositeMargin; }
//#end
//#patch box-limits
  var right = padR - select(P.shoulder, P.freewayShoulder, (eFlags(e) & 64u) != 0u);   // sim.js: 3 m outside shoulder on freeway-class roads
  var left = padL + P.shoulder;
  if (eRev(e) >= 0) { left = padL + f32(eLanes(u32(eRev(e)))) * W + P.oppositeMargin; }
  if (inBox) { right = right - BOX_MARGIN; left = left + BOX_MARGIN; }
//#end
//#orig barrier-push
  veh[i].lat = veh[i].lat - side * depth;
//#end
//#patch barrier-push
  veh[i].lat = veh[i].lat - side * barrierCorrK(depth, scrape);
//#end
//#orig barrier-event
  if (!prev && res.jn > 0.0) { crashEvent(i, NONE, res.dvA, 0.0, res.closing, cx, cz, v0.speed, 0.0, c.nx, c.nz, NONE, -1e9); }
//#end
//#patch barrier-event
  barrierEvent(i, prev, res, cx, cz, v0.speed, c.nx, c.nz, scrape, e);
//#end
//#orig resolve-start
fn resolveOne(k: u32) {
  let step = simStep(); let now = simTime();
//#end
//#patch resolve-start
fn resolveOne(k: u32) {
  let step = simStep(); let now = simTime();
  revN = 0u;
//#end
//#orig pair-event
  if (isNew && res.jn > 0.0 && (vi.state == DRIVING || vj.state == DRIVING)) {
    crashEvent(i, j, res.dvA, res.dvB, res.closing, c.px, c.pz, vi.speed, vj.speed, c.nx, c.nz, ps, lastEv);
  }
//#end
//#patch pair-event
  if (MF_EPISODES) {
    let ongoing = !isNew;
    let impact = !ongoing || max(res.dvA, res.dvB) * 3.6 >= P.touchKmh;
    var ta = 0.0; var tb = 0.0;
    if (ps != NONE) {
      let eo = epBase() + P.cap + 2u * ps;
      if (ongoing) { ta = bitcast<f32>(atomicLoad(&contacts[eo])); tb = bitcast<f32>(atomicLoad(&contacts[eo + 1u])); }
      if (impact) { ta = ta + res.dvA; tb = tb + res.dvB; }
      atomicStore(&contacts[eo], bitcast<u32>(ta)); atomicStore(&contacts[eo + 1u], bitcast<u32>(tb));
    } else { ta = res.dvA; tb = res.dvB; }
    // a later step of the episode: no pair repeat filter, no pair event-time update (bayenv-sim crash_event_ep ongoing)
    if (res.jn > 0.0 && (vi.state == DRIVING || vj.state == DRIVING) && impact) {
      crashEvent(i, j, ta, tb, res.closing, c.px, c.pz, vi.speed, vj.speed, c.nx, c.nz, select(ps, NONE, ongoing), select(lastEv, -1e9, ongoing));
    }
  } else if (isNew && res.jn > 0.0 && (vi.state == DRIVING || vj.state == DRIVING)) {
    crashEvent(i, j, res.dvA, res.dvB, res.closing, c.px, c.pz, vi.speed, vj.speed, c.nx, c.nz, ps, lastEv);
  }
//#end
