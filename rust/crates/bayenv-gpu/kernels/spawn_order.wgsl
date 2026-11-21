// bayenv-gpu patch (GpuOptions.deterministic_spawn, default on; off = the JS kernel): text replacements in the
// resolved source of src/gpu/kernels/spawn.wgsl (kernels.rs `patch_text`). See rust/docs/determinism.md.
// The JS kernel claims an entry lane with atomicExchange: when two spawns (or two station re-entries) of one step want
// the same lane, the thread that gets there first takes it, and that differs from run to run. The patch adds claim /
// commit entry points that the host dispatches in rounds (spawnClaim, spawnCommit, ...; DET_ROUNDS each). In a claim
// pass each pending entry picks its lane exactly as entryLane does (claimed lanes of earlier rounds are skipped) and
// bids for it with atomicMax(0xffffffff - id) (id = queue entry k / slot i: the smallest id wins). In the commit pass
// the winner of each lane takes it (laneClaim = step tag, bid reset to 0) and spawns; losers try again in the next
// round with the lanes that are left, after the last round in the next step (the JS kernel: another lane in the same
// pass). laneClaim is allocated with 2 K + cap + spawnQueue words: [0, K) the JS claim
// tags, [K, 2 K) the bids, [2 K, 2 K + cap) the lane a leaving car chose, then the lane a queue entry chose
// (0 none, l + 1 a bid, 0x80000000 forced lane 0 without a bid, as the JS kernel does after spawnBlockedTimeout).
// The original `spawn` / `leave` entry points are kept (phase 2 = the JS code path).
//#orig entryLane
    if (atomicExchange(&laneClaim[lkey(ei, l)], tag) == tag) { continue; }
    return i32(l);
  }
  return -1;
}
//#end
//#patch entryLane
    if (find) { return i32(l); }
    if (atomicExchange(&laneClaim[lkey(ei, l)], tag) == tag) { continue; }
    return i32(l);
  }
  return -1;
}
fn entryLane(ei: u32, vlen: f32, next: i32) -> i32 { return entryLaneM(ei, vlen, next, false); }
fn entryLaneFind(ei: u32, vlen: f32, next: i32) -> i32 { return entryLaneM(ei, vlen, next, true); }
fn detBid(id: u32) -> u32 { return 0xffffffffu - id; }
fn bidIdx(e: u32, l: u32) -> u32 { return P.K + lkey(e, l); }
fn leaveChoice(i: u32) -> u32 { return 2u * P.K + i; }
fn spawnChoice(k: u32) -> u32 { return 2u * P.K + P.cap + k; }
const DET_FORCED: u32 = 0x80000000u;
// claim pass: record my lane (or none) and bid for it
fn detClaim(choice: u32, e: u32, l: i32, forced: bool, id: u32) {
  if (forced) { atomicStore(&laneClaim[choice], DET_FORCED); return; }
  if (l < 0) { atomicStore(&laneClaim[choice], 0u); return; }
  atomicStore(&laneClaim[choice], u32(l) + 1u);
  atomicMax(&laneClaim[bidIdx(e, u32(l))], detBid(id));
}
// commit pass: the lane I won (>= 0), -2 forced lane 0 without a bid, or -1 (no lane / lost)
fn detCommit(choice: u32, e: u32, id: u32) -> i32 {
  let c = atomicLoad(&laneClaim[choice]);
  atomicStore(&laneClaim[choice], 0u);
  if (c == 0u) { return -1; }
  if (c == DET_FORCED) { return -2; }
  let l = c - 1u;
  if (atomicLoad(&laneClaim[bidIdx(e, l)]) != detBid(id)) { return -1; }
  atomicStore(&laneClaim[bidIdx(e, l)], 0u);
  atomicStore(&laneClaim[lkey(e, l)], simStep() + 1u);
  return i32(l);
}
//#end
//#orig entryLaneHead
fn entryLane(ei: u32, vlen: f32, next: i32) -> i32 {
  let n = eLanes(ei);
//#end
//#patch entryLaneHead
fn entryLaneM(ei: u32, vlen: f32, next: i32, find: bool) -> i32 {
  let n = eLanes(ei);
//#end
//#orig spawnHead
@compute @workgroup_size(64)
fn spawn(@builtin(global_invocation_id) g: vec3u) {
  let k = g.x;
//#end
//#patch spawnHead
@compute @workgroup_size(64)
fn spawn(@builtin(global_invocation_id) g: vec3u) { spawnWith(g.x, 2u); }
@compute @workgroup_size(64)
fn spawnClaim(@builtin(global_invocation_id) g: vec3u) { spawnWith(g.x, 0u); }
@compute @workgroup_size(64)
fn spawnCommit(@builtin(global_invocation_id) g: vec3u) { spawnWith(g.x, 1u); }
fn spawnWith(k: u32, phase: u32) {
//#end
//#orig spawnLane
  var lane = entryLane(e, q.par.length, nx);
  if (lane < 0) {
    if (now - q.depart < P.spawnBlockedTimeout) { return; }
    lane = 0;
    atomicStore(&laneClaim[lkey(e, 0u)], simStep() + 1u);
  }
//#end
//#patch spawnLane
  var lane = -1;
  if (phase == 0u) {
    let l = entryLaneFind(e, q.par.length, nx);
    if (l < 0 && now - q.depart < P.spawnBlockedTimeout) { atomicStore(&laneClaim[spawnChoice(k)], 0u); return; }
    detClaim(spawnChoice(k), e, select(l, 0, l < 0), false, k);   // after spawnBlockedTimeout: lane 0 (bid like any other)
    return;
  } else if (phase == 1u) {
    lane = detCommit(spawnChoice(k), e, k);
    if (lane < 0) { return; }
  } else {
    lane = entryLane(e, q.par.length, nx);
    if (lane < 0) {
      if (now - q.depart < P.spawnBlockedTimeout) { return; }
      lane = 0;
      atomicStore(&laneClaim[lkey(e, 0u)], simStep() + 1u);
    }
  }
//#end
//#orig leaveHead
@compute @workgroup_size(128)
fn leave(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
//#end
//#patch leaveHead
@compute @workgroup_size(128)
fn leave(@builtin(global_invocation_id) g: vec3u) { leaveWith(g.x, 2u); }
@compute @workgroup_size(128)
fn leaveClaim(@builtin(global_invocation_id) g: vec3u) { leaveWith(g.x, 0u); }
@compute @workgroup_size(128)
fn leaveCommit(@builtin(global_invocation_id) g: vec3u) { leaveWith(g.x, 1u); }
fn leaveWith(i: u32, phase: u32) {
//#end
//#orig leaveLane
  var lane = entryLane(e, pi.length, -1);
  if (lane < 0) {
    if (now - v.eventTime < P.spawnBlockedTimeout) { return; }
    lane = 0;
  }
//#end
//#patch leaveLane
  var lane = -1;
  if (phase == 0u) {
    let l = entryLaneFind(e, pi.length, -1);
    if (l >= 0) { detClaim(leaveChoice(i), e, l, false, i); }
    else { detClaim(leaveChoice(i), e, -1, now - v.eventTime >= P.spawnBlockedTimeout, i); }
    return;
  } else if (phase == 1u) {
    lane = detCommit(leaveChoice(i), e, i);
    if (lane == -2) { lane = 0; }
    if (lane < 0) { return; }
  } else {
    lane = entryLane(e, pi.length, -1);
    if (lane < 0) {
      if (now - v.eventTime < P.spawnBlockedTimeout) { return; }
      lane = 0;
    }
  }
//#end
