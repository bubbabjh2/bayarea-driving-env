// Spawning (sim.js _trySpawn / _entryLane / _upstreamConflict) and station re-entry
// (_leaveStations). The CPU pre-assigns the slot (from its free list), writes the slot's
// delta and fills the queue entry; the GPU finds an entry lane against this step's
// pre-rebuild lane lists (like the CPU, spawning runs before the rebuild) and claims it so
// two spawns never share a lane in one step; after spawnBlockedTimeout it is forced into lane 0.
// #include common.wgsl

struct SpawnEntry {
  par: VPar,
  depart: f32, status: u32, fuel: f32, tank: f32,
  slot: u32, flags: u32, p0: u32, p1: u32,
};

@group(1) @binding(0) var<storage, read_write> veh: array<Veh>;
@group(1) @binding(1) var<storage, read_write> par: array<VPar>;
@group(1) @binding(2) var<storage, read_write> ring: array<f32>;
@group(1) @binding(3) var<storage, read_write> queue: array<SpawnEntry>;
@group(1) @binding(4) var<storage, read_write> laneClaim: array<atomic<u32>>;
@group(2) @binding(0) var<storage, read> starts: array<u32>;
@group(2) @binding(1) var<storage, read> order: array<u32>;

fn lkey(e: u32, l: u32) -> u32 { return eLaneBase(e) + l; }
fn laneFirst(e: u32, l: u32) -> i32 {
  let k = lkey(e, l);
  if (starts[k + 1u] > starts[k]) { return i32(order[starts[k]]); }
  return -1;
}
fn laneLast(e: u32, l: u32) -> i32 {
  let k = lkey(e, l);
  if (starts[k + 1u] > starts[k]) { return i32(order[starts[k + 1u] - 1u]); }
  return -1;
}
// a car about to enter edge ei lane l from upstream within its stopping reach
fn upstreamConflict(ei: u32, l: i32) -> bool {
  let node = eFrom(ei);
  for (var q = inStart(node); q < inStart(node + 1u); q = q + 1u) {
    let e2 = inEdge(q);
    let n2 = eLanes(e2);
    for (var k = 0u; k < n2; k = k + 1u) {
      let jj = laneLast(e2, k);
      if (jj < 0) { continue; }
      let j = u32(jj);
      let vj = veh[j];
      if (vj.state != DRIVING) { continue; }
      let pj = par[j];
      if (vj.routePos + 1u >= pj.routeLen || routeEdge(pj, vj.routePos + 1u) != ei || mapLane(e2, ei, i32(vLane(vj))) != l) { continue; }
      let d = eLength(e2) - vj.s; let sp = vj.speed;
      if (d < 8.0 + sp * 2.0 + sp * sp / 6.0) { return true; }
    }
  }
  return false;
}
// lane of edge ei with room to enter at s = 0 (claimed for this step), or -1
// spawnLaneRoute (sim.js _entryLane): with next >= 0, first the lanes that continue onto route edge `next`, then any
fn goesOn(ei: u32, l: u32, next: u32) -> bool {
  if (P.hasJ != 0u && eSStop(ei) < eLength(ei)) { return jMovement(ei, l, next) >= 0; }
  let m = mapLane(ei, next, i32(l)); return m >= 0 && m < i32(eLanes(next));
}
fn entryLane(ei: u32, vlen: f32, next: i32) -> i32 {
  let n = eLanes(ei);
  let tag = simStep() + 1u;
  let sr = P.spawnLane != 0u && next >= 0;
  var kN = n; if (sr) { kN = 2u * n; }
  for (var k = 0u; k < kN; k = k + 1u) {
    let l = k % n;
    if (sr && (k < n) != goesOn(ei, l, u32(next))) { continue; }
    if (atomicLoad(&laneClaim[lkey(ei, l)]) == tag) { continue; }
    let f = laneFirst(ei, l);
    if (f >= 0 && veh[u32(f)].s - 0.5 * par[u32(f)].length < max(eSExit(ei), 0.0) + vlen + 4.0) { continue; }   // cars spawn at the box exit sExit
    if (hasFeat(FEAT_EMPTY)) { return i32(l); }
    if (upstreamConflict(ei, i32(l))) { continue; }
    if (atomicExchange(&laneClaim[lkey(ei, l)], tag) == tag) { continue; }
    return i32(l);
  }
  return -1;
}
fn clearRing(i: u32) { for (var r = 0u; r < CTRL_RING * CTRL_STRIDE; r = r + 1u) { ring[i * CTRL_RING * CTRL_STRIDE + r] = 0.0; } }

@compute @workgroup_size(64)
fn spawn(@builtin(global_invocation_id) g: vec3u) {
  let k = g.x;
  if (k >= P.spawnCount) { return; }
  let q = queue[k];
  if (q.status != 1u) { return; }
  let now = simTime();
  if (q.depart > now) { return; }
  let e = routes[q.par.routeOff] & ROUTE_EDGE_MASK;
  var nx = -1; if (q.par.routeLen > 1u) { nx = i32(routes[q.par.routeOff + 1u] & ROUTE_EDGE_MASK); }
  var lane = entryLane(e, q.par.length, nx);
  if (lane < 0) {
    if (now - q.depart < P.spawnBlockedTimeout) { return; }
    lane = 0;
    atomicStore(&laneClaim[lkey(e, 0u)], simStep() + 1u);
  }
  let i = q.slot;
  par[i] = q.par;
  var v: Veh;
  v.edge = e; v.s = max(0.0, eSExit(e)); v.speed = 0.0; v.accel = 0.0;   // layout v8: cars spawn at the junction exit point
  v.lat = (f32(lane) + 0.5) * laneWAt(e, v.s); v.headingRel = 0.0; v.steerAngle = 0.0; v.state = DRIVING;
  v.routePos = 0u; v.seg = 0u; v.flags = q.flags | F_NEW; v.cmdStep = i32(simStep()) - 1;
  v.stoppedSince = -1.0; v.odometer = 0.0; v.lastDecision = -1e9;
  v.fuel = min(q.tank, q.fuel); v.tank = q.tank; v.spawnTime = now; v.routeDist = 0.0;
  v.leader = -1; v.gap = 1e9; v.eventTime = 0.0; v.stopPos = -1;
  v.refuelState = 0u; v.steerSl = 0.0; v.gas = 0.0; v.brake = 0.0;
  v.routeTotal = q.par.routeLength; v.lastFuelCheck = now;
  v.taper = 0.0; v.laneEndDist = 1e9; v.laneEndSide = 0;
  v.vLat = 0.0; v.yawRate = 0.0; v.axPrev = 0.0; v.power = 0.0; v.px = 0.0; v.pz = 0.0; v.pyaw = 0.0;
  v.lanes = derivedLanes(e, v.s, v.lat, 0.5 * q.par.width, -1);
  if (hasFeat(FEAT_FUEL) && needsFuel(q.par.typeIdx, v.fuel, v.tank, v.routeTotal)) {
    v.flags = v.flags | F_FUELREQ;
    emit(EV_LOWFUEL, i, e, 1u);
  }
  veh[i] = v;
  clearRing(i);
  queue[k].status = 2u;
  emit(EV_SPAWN, i, k, u32(lane));
  statAdd(S_SPAWNED, 1u);
}

// cars done pumping (refuelState 4, set by the CPU's pump queue) re-enter on the next route edge
@compute @workgroup_size(128)
fn leave(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= P.cap) { return; }
  let v = veh[i];
  if (v.state != REFUELING || v.refuelState != 4u) { return; }
  let pi = par[i];
  let nxt = u32(v.stopPos + 1);
  let now = simTime();
  if (nxt >= pi.routeLen) {
    emit(EV_TRIPX, i, bitcast<u32>(v.odometer), bitcast<u32>(now - v.spawnTime));
    emit(EV_ARRIVE, i, pi.trip, bitcast<u32>(v.fuel));
    statAdd(S_ARRIVED, 1u);
    veh[i].state = FREE;
    return;
  }
  let e = routeEdge(pi, nxt);
  var lane = entryLane(e, pi.length, -1);
  if (lane < 0) {
    if (now - v.eventTime < P.spawnBlockedTimeout) { return; }
    lane = 0;
  }
  veh[i].state = DRIVING; veh[i].refuelState = 0u; veh[i].routePos = nxt; veh[i].edge = e; veh[i].s = 0.0; veh[i].seg = 0u;
  veh[i].lat = (f32(lane) + 0.5) * laneWAt(e, 0.0); veh[i].headingRel = 0.0; veh[i].steerAngle = 0.0;
  veh[i].steerSl = 0.0; veh[i].gas = 0.0; veh[i].brake = 0.0;
  clearRing(i);
  veh[i].lanes = derivedLanes(e, 0.0, veh[i].lat, 0.5 * pi.width, -1);
  veh[i].stopPos = -1; veh[i].flags = (v.flags & ~(F_REFUEL | F_FUELREQ)) | F_NEW; veh[i].stoppedSince = -1.0; veh[i].taper = 0.0;
  veh[i].vLat = 0.0; veh[i].yawRate = 0.0; veh[i].axPrev = 0.0; veh[i].speed = 0.0;
  emit(EV_REENTER, i, pi.trip, u32(lane));
}
