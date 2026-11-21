// Shared declarations for every GPU sim kernel (included with `// #include common.wgsl`).
// Group 0 = params + io (clock/stats/events) + read-only world + routes; kernels put their
// own buffers in group 1 and lists.wgsl puts the lane lists in group 2. Every kernel stays
// within 10 storage buffers per stage (the common Chrome limit). Struct layouts mirror
// src/gpu/layout.js (keep in sync).

const FREE: u32 = 0u;
const DRIVING: u32 = 1u;
const CRASHED: u32 = 2u;      // every lane-blocking stopped vehicle: crash scenes, stranded (F_STRANDED), wrecks
const REFUELING: u32 = 5u;    // in a station lot (queued / pumping / waiting to re-enter): not in the lane lists

const F_LEARNER: u32 = 1u;
const F_COLLIDED: u32 = 2u;
const F_POLICY: u32 = 4u;
const F_SHOULDER: u32 = 8u;
const F_SPEEDING: u32 = 16u;
const F_FUELREQ: u32 = 32u;     // GPU only: a refuel plan was requested from the CPU, waiting for the route
const F_STRANDED: u32 = 64u;
const F_REFUEL: u32 = 128u;     // refuel stop planned (refuelState == 1)
const F_ZIPLEAD: u32 = 256u;
const F_SCORED: u32 = 512u;
const F_DISABLED: u32 = 1024u;   // crash delta-v >= 10 km/h (LAYOUT v4 flag bit5 DISABLED)
const F_WALL: u32 = 2048u;       // GPU only: touching a road-edge barrier this step (new contact detection)
const F_WALL_PREV: u32 = 4096u;
const F_NEW: u32 = 8192u;        // GPU only: spawned / re-entered this step, pose + start pose not set yet     // GPU only: crash / stranding already charged to the agent's reward    // GPU only: current leader is a zipper-merge car (virtual gap)

// feature bits (Params.features)
const FEAT_LANECHANGE: u32 = 1u;
const FEAT_ACCIDENTS: u32 = 2u;
const FEAT_FUEL: u32 = 4u;
const FEAT_COLLISIONS: u32 = 8u;
const FEAT_YIELD: u32 = 16u;
const FEAT_POLICY: u32 = 32u;
const FEAT_ZIPFIX: u32 = 64u;    // GPU: zipper virtual gaps never count as collisions
const FEAT_EMPTY: u32 = 128u;    // trafficMode 'empty': cars ignore each other (only scenes and lane ends)

// event types (EvQ records: type, slot, a, b)
const EV_ARRIVE: u32 = 1u;      // a = trip, b = bitcast<u32>(fuel litres left)
const EV_CRASH: u32 = 2u;       // slot, a = other slot or 0xffffffff, b = bitcast(impact m/s)
const EV_STUCK: u32 = 3u;
const EV_STRANDED: u32 = 4u;     // slot, a = trip, b = edge
const EV_TOWED: u32 = 5u;
const EV_LOWFUEL: u32 = 6u;     // refuel plan request: a = current edge (plan from its end node), b = 1 at spawn
const EV_SPAWN: u32 = 7u;       // slot, a = spawn queue entry, b = lane
const EV_STATION: u32 = 8u;     // reached the station stop (entered the lot), a = trip
const EV_ACCIDENT: u32 = 9u;    // stochastic single/rear-end, same payload as EV_CRASH
const EV_OFFROAD: u32 = 11u;    // left the carriageway (single-vehicle crash)
const EV_REENTER: u32 = 12u;    // left the station lot, back on the road
// detail records emitted immediately before the record they describe (telemetry, HOOKS.md)
const EV_CRASHX_V: u32 = 14u;   // slot i, a = other (or NONE), b = pre-impact speed i | j << 16 (dm/s)
const EV_CRASHX_P: u32 = 15u;   // slot i, a = bitcast(contact x), b = bitcast(contact z)
const EV_CRASHX_H: u32 = 16u;   // slot i, a = heading i | heading j << 16 (u16 of [-pi, pi)), b = closing speed dm/s | lane i << 16 | lane j << 24
const EV_CTLVIOL: u32 = 18u;    // slot, a = approach, b = kind (1 red light, 2 stop sign) | speed dm/s << 8
const EV_TRIPX: u32 = 17u;
const EV_CRASHX_N: u32 = 19u;
const EV_REROUTE: u32 = 21u;    // slot, a = current edge, b = routePos: a closed road is on my route within rerouteSight
const EV_CRASHX_M: u32 = 20u;
const EV_CRASHPEN: u32 = 23u;   // slot, a = bitcast(penalty), b = delta-v km/h x10 (u16) | fault x255 << 16 | disabled << 24 | barrier << 25 (only with P.crashOn)
const EV_MISSED: u32 = 22u;     // slot, a = out-edge actually taken (strict diverge, my lane led there), b = trip   // slot i, a = maneuver of the striking car (sim.js _classifyCrash: 0 cruise, 1 lane-change, 2 merge, 3 junction, 4 queue-approach, 5 wreck) | crashed flags << 8, b = striking car's edge   // slot i, a = bitcast(contact normal x), b = bitcast(normal z) (from i toward j / the wall)      // slot, a = bitcast(odometer m), b = bitcast(duration s since spawn); precedes EV_ARRIVE
fn u16Heading(h: f32) -> u32 { return u32(clamp((h + 3.14159265) / 6.2831853, 0.0, 0.99999) * 65536.0); }
fn dms(v: f32) -> u32 { return min(u32(max(v, 0.0) * 10.0 + 0.5), 65535u); }

const NONE: u32 = 0xffffffffu;
const ROUTE_EDGE_MASK: u32 = 0xfffffu;   // 20-bit edge ids; bits 20..25 maneuver code (type | side << 4)
const CTRL_RING: u32 = 40u;
const CTRL_STRIDE: u32 = 3u;     // ring record: steerRate, gas, brake
const MAX_OCC: u32 = 4u;
const LANE_W: f32 = 3.7;
const TAPER: f32 = 40.0;         // m an ended lane narrows into the next edge (vehicle.js TAPER)
const MERGE_ROOM: f32 = 15.0;    // sim.js MERGE_ROOM
const LANE_END_TAPER: f32 = 0.0;  // sim.js LANE_END_TAPER: src/geom paves every lane to its node (the run-out is beyond it)
const ROLLING: f32 = 0.15;       // the pedal / steering model (src/sim/physics)
const AERO: f32 = 0.0004;
const MAX_LANES: u32 = 8u;

struct Params {
  cap: u32, E: u32, N: u32, K: u32,
  T: u32, wBits: u32, wMask: u32, hMask: u32,
  ptsOff: u32, laneOffOff: u32, adjOff: u32, ebOff: u32,
  dt: f32, time0: f32, step0: u32, features: u32,
  cellSize: f32, leaderLookahead: f32, junctionMargin: f32, collisionInterval: u32,
  s0: f32, collideShrink: f32, speedFactorMin: f32, speedFactorSpan: f32,
  laneCheckInterval: f32, yieldLook: f32, stopMargin: f32, stuckTimeout: f32,
  spawnBlockedTimeout: f32, spawnCount: u32, opCount: u32, decideEvery: u32,
  accRate: f32, accSpeedRef: f32, accCloseGap: f32, accMergeFactor: f32,
  accMergeDist: f32, rubberDist: f32, rubberFactor: f32, towTime: f32,
  lowFuelFrac: f32, seed: u32, evCap: u32, inSize: u32,
  qrWindow: f32, qrDist: f32, qrFactor: f32, nBlocks: u32,
  blkOff: u32, maxCellReach: f32, fuelScale: f32, anticipation: f32,
  mergeLookahead: f32, collisionTol: f32, contactSpeed: f32, lateralTol: f32,
  offroadMargin: f32, squeezeSpeed: f32, prepBase: f32, prepPerLane: f32,
  fuelCheckInterval: f32, fuelMinFillToSkip: f32, fuelLowFraction: f32, fuelPlanFactor: f32,
  fuelReserveKm: f32, fuelPlanSpeed: f32, progressW: f32, nAgents: u32,
  geoOff: u32, lgvOff: u32, hopOff: u32, hasLG: u32,
  layerGap: f32, pairCooldown: f32, touchKmh: f32, shoulder: f32,
  oppositeMargin: f32, laneEndLookahead: f32, contactCap: u32, physDt: f32,
  ctlEdgeOff: u32, ctlApOff: u32, ctlCOff: u32, hasCtl: u32,
  ctlLook: f32, rampStart: f32, rampEnd: f32, freewayShoulder: f32,
  pairOff: u32, pairN: u32, labelMode: u32, ctlListOff: u32,
  jMovOff: u32, jEdgeMovOff: u32, jPtsOff: u32, jConfOff: u32,
  jMouthOff: u32, hasJ: u32, jM: u32, jHeadOff: u32,
  rerouteSight: f32, passWrecks: u32, trafficOn: u32, steerMode: u32,
  rlOn: u32, rlRec: f32, rlSeed: u32, rlChunk: u32,   // reward-only RL rollouts (rl.wgsl, netFast sampling)
  rlStd0: f32, rlStd1: f32, rlStd2: f32, rlRowCap: u32,
  jerkW: f32, crashBase: f32, crashK: f32, crashCap: f32,   // steering-smoothness reward (sim/reward.js JERK; integrate.wgsl); severity crash penalty
  crashFault: f32, crashOn: u32, crashFail: f32, padC3: u32,   // (sim/reward.js crashPenalty, contacts.wgsl crashEvent; crashOn 0 = legacy flat -20)
  jIntoOff: u32, jSight: u32, pathSight: u32, dlBreak: u32,
  spawnLane: u32, padS1: u32, padS2: u32, padS3: u32,   // docs/scale-crashes.md (sim.js junctionSight / pathSight / deadlockBreak)
};
/** RL: is agent a recorded in the current chunk (fixed random subset, share P.rlRec)? */
fn rlRecorded(a: u32) -> bool { return f32(pcg(pcg(a ^ (P.rlSeed * 0x9e3779b9u)) ^ (P.rlChunk * 0x85ebca6bu + 77u)) >> 8u) / 16777216.0 < P.rlRec; }

struct Edge {
  firstPt: u32, nPts: u32, lanes: u32, laneBase: u32,
  length: f32, speedLimit: f32, efrom: u32, to: u32,
  reverse: i32, rank: u32, flags: u32, stopBack: f32,
  hStart: f32, hEnd: f32, y: f32, navRank: u32,
};

// 40 words per vehicle slot (layout v4: + vLat, yawRate, axPrev, start pose px pz pyaw, power) (src/gpu/layout.js VEH_FIELDS); mirrors src/sim/vehicle.js
struct Veh {
  edge: u32, s: f32, speed: f32, accel: f32,
  lat: f32, headingRel: f32, steerAngle: f32, state: u32,
  routePos: u32, seg: u32, flags: u32, cmdStep: i32,
  stoppedSince: f32, odometer: f32, lanes: u32, lastDecision: f32,
  fuel: f32, tank: f32, spawnTime: f32, routeDist: f32,
  leader: i32, gap: f32, eventTime: f32, stopPos: i32,
  refuelState: u32, steerSl: f32, gas: f32, brake: f32,   // steerSl: lagged steering target (rad, v9)
  routeTotal: f32, lastFuelCheck: f32, taper: f32, laneEndDist: f32,
  laneEndSide: i32, vLat: f32, yawRate: f32, axPrev: f32,
  px: f32, pz: f32, pyaw: f32, power: f32,
};
// lanes word: lane (centre) | laneLo << 8 | laneHi << 16 | (targetLane + 1) << 24
fn vLane(v: Veh) -> u32 { return v.lanes & 255u; }
fn vLo(v: Veh) -> u32 { return (v.lanes >> 8u) & 255u; }
fn vHi(v: Veh) -> u32 { return (v.lanes >> 16u) & 255u; }
fn vTarget(v: Veh) -> i32 { return i32(v.lanes >> 24u) - 1; }
fn packLanes(lane: u32, lo: u32, hi: u32, tgtLane: i32) -> u32 { return lane | (lo << 8u) | (hi << 16u) | (u32(tgtLane + 1) << 24u); }

// 20 words per slot, written at spawn (src/gpu/layout.js PAR_FIELDS)
struct VPar {
  routeOff: u32, routeLen: u32, typeIdx: u32, delaySteps: u32,
  length: f32, width: f32, maxSpeed: f32, maxAccel: f32,
  maxDecel: f32, headway: f32, aggression: f32, patience: f32,
  reactionTime: f32, workStart: f32, routeLength: f32, trip: u32,
  wheelbase: f32, maxSteer: f32, maxSteerRate: f32, typeMaxSpeed: f32,
};

// crash / stranded / wreck scene: state 1 active, 2 shoulder; flags bit0 squeeze, bit1 isCrash
struct Block { edge: u32, laneMask: u32, s: f32, state: u32, time: f32, next: u32, lead: u32, flags: u32 };

// io: [0] clock step | [8..72) stats counters | [72] event count | [76..) events (4 words each)
const IO_STEP: u32 = 0u;
const IO_STATS: u32 = 8u;
const IO_EVCOUNT: u32 = 72u;
const IO_EVDATA: u32 = 76u;
// stats counters (index relative to IO_STATS); 0..31 cumulative, 32..63 reset every step
const S_ARRIVED: u32 = 0u;
const S_ARRIVED_OUT: u32 = 1u;
const S_ONTIME: u32 = 2u;
const S_CRASHED_VEH: u32 = 3u;
const S_CRASH_EVENTS: u32 = 4u;
const S_STUCK: u32 = 5u;
const S_STRANDED: u32 = 6u;
const S_TOWED: u32 = 7u;
const S_SPAWNED: u32 = 8u;
const S_LANECHANGES: u32 = 9u;
const S_REFUELED: u32 = 10u;
const S_LOWFUEL: u32 = 11u;
const S_VEHM_LO: u32 = 12u;      // vehicle-metres (64-bit lo/hi)
const S_SPEEDM_LO: u32 = 14u;    // speeding vehicle-metres
const S_VEHSTEPS_LO: u32 = 16u;
const S_SPEEDSTEPS_LO: u32 = 18u;
const S_LATESUM_LO: u32 = 20u;   // seconds late summed over outbound arrivals
const S_ACCIDENTS: u32 = 22u;
const S_LIVE_DRIVING: u32 = 32u;
const S_LIVE_CRASHED: u32 = 33u;
const S_LIVE_STRANDED: u32 = 34u;
const S_LIVE_SUMSPEED: u32 = 35u; // cm/s
const S_LIVE_DWELL: u32 = 36u;
const S_LIVE_POLICY: u32 = 37u;
const S_BUMPS: u32 = 23u;
const S_REPEAT_CONTACTS: u32 = 24u;
const S_WRECK_PASSES: u32 = 25u;

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> io: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> world: array<u32>;
@group(0) @binding(3) var<storage, read_write> routes: array<u32>;   // read_write: integrate patches a missed exit in place

fn edgeAt(e: u32) -> Edge {
  let b = e * 16u;
  var o: Edge;
  o.firstPt = world[b]; o.nPts = world[b + 1u]; o.lanes = world[b + 2u]; o.laneBase = world[b + 3u];
  o.length = bitcast<f32>(world[b + 4u]); o.speedLimit = bitcast<f32>(world[b + 5u]);
  o.efrom = world[b + 6u]; o.to = world[b + 7u];
  o.reverse = bitcast<i32>(world[b + 8u]); o.rank = world[b + 9u]; o.flags = world[b + 10u];
  o.stopBack = bitcast<f32>(world[b + 11u]);
  o.hStart = bitcast<f32>(world[b + 12u]); o.hEnd = bitcast<f32>(world[b + 13u]);
  o.y = bitcast<f32>(world[b + 14u]); o.navRank = world[b + 15u];
  return o;
}
fn eLength(e: u32) -> f32 { return bitcast<f32>(world[e * 16u + 4u]); }
fn eLanes(e: u32) -> u32 { return world[e * 16u + 2u]; }
fn eLaneBase(e: u32) -> u32 { return world[e * 16u + 3u]; }
fn eSpeedLimit(e: u32) -> f32 { return bitcast<f32>(world[e * 16u + 5u]); }
fn pt(k: u32) -> vec4f {
  let b = P.ptsOff + 4u * k;
  return vec4f(bitcast<f32>(world[b]), bitcast<f32>(world[b + 1u]), bitcast<f32>(world[b + 2u]), bitcast<f32>(world[b + 3u]));
}
fn adjAt(k: u32) -> u32 { return world[P.adjOff + k]; }
fn outStart(n: u32) -> u32 { return world[P.adjOff + n]; }
fn inStart(n: u32) -> u32 { return world[P.adjOff + P.N + 1u + n]; }
fn outEdge(k: u32) -> u32 { return world[P.adjOff + 2u * (P.N + 1u) + k]; }
fn inEdge(k: u32) -> u32 { return world[P.adjOff + 2u * (P.N + 1u) + P.E + k]; }
fn blockAt(k: u32) -> Block {
  let b = P.blkOff + 8u * k;
  var o: Block;
  o.edge = world[b]; o.laneMask = world[b + 1u]; o.s = bitcast<f32>(world[b + 2u]); o.state = world[b + 3u];
  o.time = bitcast<f32>(world[b + 4u]); o.next = world[b + 5u]; o.lead = world[b + 6u]; o.flags = world[b + 7u];
  return o;
}

fn simStep() -> u32 { return atomicLoad(&io[IO_STEP]); }
fn simTime() -> f32 { return P.time0 + f32(simStep() - P.step0) * P.dt; }
fn hasFeat(f: u32) -> bool { return (P.features & f) != 0u; }

fn routeEdge(par: VPar, k: u32) -> u32 { return routes[par.routeOff + k] & ROUTE_EDGE_MASK; }
// route word bits 20..30: present | type << 1 | side << 5 | nextType << 7 (worldPack.js packRoutes)
fn routeCode(par: VPar, k: u32) -> u32 { return (routes[par.routeOff + k] >> 20u) & 2047u; }

fn laneOffset(e: u32, lane: u32) -> f32 {
  return bitcast<f32>(world[P.laneOffOff + eLaneBase(e) + min(lane, eLanes(e) - 1u)]);
}

struct PosOut { x: f32, z: f32, y: f32, hx: f32, hz: f32, seg: u32 };

// Point on edge e at distance s, offset right by `off`; mirrors roadGraph.js edgePoint.
fn edgePoint(e: u32, s: f32, off: f32, segHint: u32) -> PosOut {
  let n = world[e * 16u + 1u];
  let b = world[e * 16u];
  var i = min(segHint, n - 2u);
  loop { if (i == 0u || pt(b + i).w <= s) { break; } i = i - 1u; }
  loop { if (i >= n - 2u || pt(b + i + 1u).w >= s) { break; } i = i + 1u; }
  let p0 = pt(b + i);
  let p1 = pt(b + i + 1u);
  let segLen = p1.w - p0.w;
  var t = 0.0;
  if (segLen > 0.0) { t = clamp((s - p0.w) / segLen, 0.0, 1.0); }
  var dx = p1.x - p0.x;
  var dz = p1.y - p0.y;
  var L = sqrt(dx * dx + dz * dz);
  if (L == 0.0) { L = 1.0; }
  dx = dx / L; dz = dz / L;
  var o: PosOut;
  o.x = p0.x + dx * t * segLen - dz * off;
  o.z = p0.y + dz * t * segLen + dx * off;
  o.y = p0.z + (p1.z - p0.z) * t;
  o.hx = dx; o.hz = dz; o.seg = i;
  return o;
}

// ---- RNG: stateless PCG hash of (slot, step, salt)
fn pcg(v: u32) -> u32 {
  let st = v * 747796405u + 2891336453u;
  let w = ((st >> ((st >> 28u) + 4u)) ^ st) * 277803737u;
  return (w >> 22u) ^ w;
}
fn rand01(slot: u32, salt: u32) -> f32 {
  let h = pcg(pcg(slot ^ (P.seed * 0x9e3779b9u)) ^ pcg(simStep() * 0x85ebca6bu + salt));
  return f32(h >> 8u) / 16777216.0;
}

fn wrapAngle(a: f32) -> f32 {
  var x = a;
  if (x > 3.14159265) { x = x - 6.2831853; }
  if (x < -3.14159265) { x = x + 6.2831853; }
  return x;
}

// ---- lanes / footprint (vehicle.js updateLanes): lat = metres from the RIGHT boundary,
// lane k spans [k, k+1) * laneWidth(e, s)
fn halfExtent(v: Veh, p: VPar) -> f32 {
  let h = v.headingRel;
  return 0.5 * (p.width * abs(cos(h)) + p.length * abs(sin(h)));
}
fn derivedLanes(e: u32, s: f32, lat: f32, he: f32, tgtLane: i32) -> u32 {
  let LW = laneWAt(e, s);   // src/geom per-edge lane width at s
  let n = i32(eLanes(e));
  var lo = i32(floor((lat - he) / LW));
  var hi = i32(floor((lat + he - 1e-4) / LW));
  var c = i32(floor(lat / LW));
  c = clamp(c, 0, n - 1);
  // body entirely off the travel lanes (shoulder / run-out): in no lane list (vehicle.js laneHi = laneLo - 1),
  // encoded as lo = c + 1, hi = c so every lo..hi loop is empty
  if (hi < 0 || lo > n - 1) { return packLanes(u32(c), u32(c + 1), u32(c), tgtLane); }
  lo = max(lo, 0); hi = min(hi, n - 1);
  if (lo > hi) { lo = c; hi = c; }
  if (hi - lo >= i32(MAX_OCC)) { hi = lo + i32(MAX_OCC) - 1; }
  return packLanes(u32(c), u32(lo), u32(hi), tgtLane);
}
fn rightEdge(e: u32) -> f32 { return laneOffset(e, 0u) + 0.5 * LANE_W; }

// ---- pedals (the pedal / steering model (src/sim/physics) pedalAccel / pedalsFor)
fn pedalAccel(gas: f32, brake: f32, v: f32, maxAccel: f32, maxDecel: f32, vmaxType: f32) -> f32 {
  let g = clamp(gas, 0.0, 1.0); let b = clamp(brake, 0.0, 1.0);
  var drag = 0.0;
  if (v > 0.01) { drag = ROLLING + AERO * v * v; }
  return g * maxAccel * max(0.2, 1.0 - 0.6 * v / vmaxType) - b * maxDecel - drag;
}
fn pedalsFor(acc: f32, v: f32, maxAccel: f32, maxDecel: f32, vmaxType: f32) -> vec2f {
  var drag = 0.0;
  if (v > 0.01) { drag = ROLLING + AERO * v * v; }
  let need = acc + drag;
  var o = vec2f(0.0, 0.0);
  if (need >= 0.0) { o.x = min(1.0, need / (maxAccel * max(0.2, 1.0 - 0.6 * v / vmaxType))); }
  else { o.y = min(1.0, -need / maxDecel); }
  if (v < 0.05 && acc <= 0.0) { o.x = 0.0; o.y = max(o.y, 0.3); }
  return o;
}

// ---- junction lane mapping (roadGraph.js hopMap / mapLane / mapLat, src/geom/hops.js):
// precomputed on the CPU with the sim's own hopMap into a table (worldPack.js)
const EF_FREEWAY_SIM: u32 = 32u;  // motorway, motorway_link, trunk, trunk_link (roadGraph FREEWAY)
fn turnAngle(a: u32, b: u32) -> f32 {   // > 0 = right turn
  return wrapAngle(bitcast<f32>(world[b * 16u + 12u]) - bitcast<f32>(world[a * 16u + 13u]));
}
fn eRev(e: u32) -> i32 { return bitcast<i32>(world[e * 16u + 8u]); }
fn eTo(e: u32) -> u32 { return world[e * 16u + 7u]; }
fn eFrom(e: u32) -> u32 { return world[e * 16u + 6u]; }
fn eFlags(e: u32) -> u32 { return world[e * 16u + 10u]; }
// telemetry edge flags (optional, unused here) in bits 8..13 of the edge flags:
// bits 0-1 road group (0 freeway, 1 arterial, 2 local), 2-4 bridge id (0 none, 1..5), 5 density edge
fn eTel(e: u32) -> u32 { return (eFlags(e) >> 8u) & 63u; }
fn eBridge(e: u32) -> u32 { return (eTel(e) >> 2u) & 7u; }
const TW_BRIDGE_ENTRIES: u32 = 24u;   // gpuLayout.js W.BRIDGE_ENTRIES
// x = strict (1/0), y = shift; (0, 0) when en does not leave e's end node
fn hopMap(e: u32, en: u32) -> vec2i {
  let t = eTo(e); let b0 = world[P.geoOff + e * GEO_W + 7u];
  for (var q = outStart(t); q < outStart(t + 1u); q = q + 1u) {
    if (outEdge(q) == en) { let h = world[P.hopOff + b0 + q - outStart(t)]; return vec2i(i32(h & 1u), i32(h >> 1u) - 64); }
  }
  return vec2i(0, 0);
}
fn mapLane(e: u32, en: u32, l: i32) -> i32 {
  let h = hopMap(e, en);
  if (h.x == 1) { return l - h.y; }
  return min(i32(eLanes(en)) - 1, i32(floor((f32(l) + 0.5) * f32(eLanes(en)) / f32(eLanes(e)))));
}
// lat on en of a car at lat on e: strict = same lane; junction = world re-projection kept on the pavement
fn mapLat(e: u32, en: u32, lat: f32, he: f32) -> f32 {
  let h = hopMap(e, en);
  if (h.x == 1) { return lat - f32(h.y) * laneWAt(e, eLength(e)); }
  let W = f32(eLanes(en)) * laneWAt(en, 0.0);
  let p = latPointAt(e, eLength(e), lat);
  let q = offsetPointAt(en, 0.0, 0.0, 0.0);
  var l2 = rightEdgeAt(en, 0.0) - ((p.x - q.x) * -q.hz + (p.z - q.z) * q.hx);
  if (l2 < he) { l2 = min(he, W / 2.0); } else if (l2 > W - he) { l2 = max(W - he, W / 2.0); }
  return l2;
}

// ---- fuel (VT_* vehicle-type consts are injected by gpuSim.js from src/sim/vehicleTypes.js)
// src/sim/fuel.js fuelRate (L/s) and litresPerKm
fn fuelRate(t: u32, v: f32, a: f32) -> f32 {
  let k = min(t, 3u);
  var r = VT_IDLE[k] + (VT_FBASE[k] + VT_FAERO[k] * v * v) * v * 1e-3;
  if (a > 0.0) { r = r + a * VT_MASS[k] * v * 1.25e-7; }
  return r;
}
fn litresPerKm(t: u32, v: f32) -> f32 {
  let k = min(t, 3u);
  var r = VT_FBASE[k] + VT_FAERO[k] * v * v;
  if (v > 0.1) { r = r + VT_IDLE[k] * 1000.0 / v; }
  return r;
}
fn needsFuel(t: u32, fuel: f32, tank: f32, remainingM: f32) -> bool {
  let f = fuel / tank;
  if (f >= P.fuelMinFillToSkip) { return false; }
  if (f < P.fuelLowFraction) { return true; }
  let lpk = litresPerKm(t, P.fuelPlanSpeed) * P.fuelPlanFactor;
  return fuel / lpk < remainingM / 1000.0 + P.fuelReserveKm;
}

// ---------------------------------------------------------------- lane geometry (src/geom/lanes.js)
// per-edge record at P.geoOff + GEO_W e: vFirst, Rd, RendStart, RendEnd, tauStart, tauEnd, tapers, hopBase, run-outs
// per-vertex record at P.lgvOff + 6 v: x, z, y, s (data-length coordinate), mitre normal mx, mz
const TAPER_M: f32 = 90.0;
const GEO_W: u32 = 12u;   // worldPack.js GEO_WORDS
fn geoU(e: u32, k: u32) -> u32 { return world[P.geoOff + e * GEO_W + k]; }
fn geoF(e: u32, k: u32) -> f32 { return bitcast<f32>(world[P.geoOff + e * GEO_W + k]); }
fn lgv(v: u32, k: u32) -> f32 { return bitcast<f32>(world[P.lgvOff + v * 6u + k]); }
fn smooth01(x: f32) -> f32 { let u = clamp(x, 0.0, 1.0); return u * u * (3.0 - 2.0 * u); }
/** right carriageway edge offset (m right of the reference polyline) at s */
fn rightEdgeAt(e: u32, s: f32) -> f32 {
  if (P.hasLG == 0u) { return laneOffset(e, 0u) + 0.5 * LANE_W; }
  let Lg = eLength(e); let rd = geoF(e, 1u);
  var r = rd;
  let ts = geoF(e, 4u); let te = geoF(e, 5u);
  if (ts > 0.0) { r = r + (geoF(e, 2u) - rd) * smooth01(1.0 - s / ts); }
  if (te > 0.0) { r = r + (geoF(e, 3u) - rd) * smooth01(1.0 - (Lg - s) / te); }
  return r;
}
/**
 * LaneGeometry.bounds: drawn pavement edges at s (offsets right of the polyline): x right, y left, z R.
 * Every sim lane is paved to its node; an ending / starting lane continues as a paved run-out
 * (xR / xL extra lane widths) narrowing to zero over min(90 m, length) on the neighbour edge.
 */
fn boundsAt(e: u32, s: f32) -> vec3f {
  let Lg = eLength(e); let n = f32(eLanes(e));
  let R = rightEdgeAt(e, s);
  if (P.hasLG == 0u) { return vec3f(R, R - n * LANE_W, R); }
  let T = min(TAPER_M, Lg);
  var fs = 0.0; var fe = 0.0;
  if (T > 0.0) { fs = max(0.0, 1.0 - s / T); fe = max(0.0, 1.0 - (Lg - s) / T); }
  let x = geoU(e, 8u);
  let w = laneWAt(e, s);
  let exR = (f32(x & 255u) * fs + f32((x >> 8u) & 255u) * fe) * w;
  let exL = (f32((x >> 16u) & 255u) * fs + f32((x >> 24u) & 255u) * fe) * w;
  return vec3f(R + exR, R - n * w - exL, R);
}
struct GeoPt { x: f32, y: f32, z: f32, hx: f32, hz: f32 };
/** world point `off` m right of the reference polyline at s (LaneGeometry.offsetPoint); heading of that offset curve */
fn offsetPointAt(e: u32, s: f32, off: f32, offSlope: f32) -> GeoPt {
  var o: GeoPt;
  if (P.hasLG == 0u) {
    let q = edgePoint(e, s, off, 0u);
    o.x = q.x; o.y = q.y; o.z = q.z; o.hx = q.hx; o.hz = q.hz; return o;
  }
  let v0 = geoU(e, 0u); let v1 = geoU(e + 1u, 0u) - 1u;
  var i = v0; var t = 0.0;
  if (s <= 0.0) { i = v0; t = 0.0; }
  else if (s >= lgv(v1, 3u)) { i = v1 - 1u; t = 1.0; }
  else {
    var lo = v0; var hi = v1;
    loop { if (hi - lo <= 1u) { break; } let m = (lo + hi) >> 1u; if (lgv(m, 3u) <= s) { lo = m; } else { hi = m; } }
    i = lo; let d = lgv(lo + 1u, 3u) - lgv(lo, 3u); if (d > 0.0) { t = (s - lgv(lo, 3u)) / d; }
  }
  let j = i + 1u;
  let nx = lgv(i, 4u) + (lgv(j, 4u) - lgv(i, 4u)) * t; let nz = lgv(i, 5u) + (lgv(j, 5u) - lgv(i, 5u)) * t;
  o.x = lgv(i, 0u) + (lgv(j, 0u) - lgv(i, 0u)) * t + off * nx;
  o.y = lgv(i, 2u) + (lgv(j, 2u) - lgv(i, 2u)) * t;
  o.z = lgv(i, 1u) + (lgv(j, 1u) - lgv(i, 1u)) * t + off * nz;
  var ds = lgv(j, 3u) - lgv(i, 3u); if (ds == 0.0) { ds = 1.0; }
  var hx = (lgv(j, 0u) - lgv(i, 0u) + off * (lgv(j, 4u) - lgv(i, 4u))) / ds + offSlope * nx;
  var hz = (lgv(j, 1u) - lgv(i, 1u) + off * (lgv(j, 5u) - lgv(i, 5u))) / ds + offSlope * nz;
  var Ln = sqrt(hx * hx + hz * hz); if (Ln == 0.0) { Ln = 1.0; }
  o.hx = hx / Ln; o.hz = hz / Ln;
  return o;
}
/** sim coordinate: lat = m from the right carriageway edge (LaneGeometry.latPoint) */
fn latPointAt(e: u32, s: f32, lat: f32) -> GeoPt {
  let Lg = eLength(e);
  let slope = (rightEdgeAt(e, min(Lg, s + 0.5)) - rightEdgeAt(e, max(0.0, s - 0.5))) / 1.0;
  return offsetPointAt(e, s, rightEdgeAt(e, s) - lat, select(0.0, slope, P.hasLG != 0u));
}
/** road direction at s (sim.js edgePointDir) */
fn roadDirAt(e: u32, s: f32) -> vec2f { let p = offsetPointAt(e, s, 0.0, 0.0); return vec2f(p.hx, p.hz); }
/** road slope dy/ds of the SimGraph polyline at s (vehicle.js edgeGrade): rise over a 12 m window centred on s,
 *  clipped to the edge, clamped to +-35 % (real terrain, docs/elevation.md) */
const GRADE_HALF: f32 = 6.0;
const MAX_GRADE: f32 = 0.35;
fn yAlongEdge(b: u32, n: u32, s: f32) -> f32 {
  var k = 0u;
  loop { if (k >= n - 2u || pt(b + k + 1u).w >= s) { break; } k = k + 1u; }
  let p0 = pt(b + k); let p1 = pt(b + k + 1u);
  let d = p1.w - p0.w;
  var t = 0.0; if (d > 0.0) { t = clamp((s - p0.w) / d, 0.0, 1.0); }
  return p0.z + (p1.z - p0.z) * t;
}
fn edgeGrade(e: u32, s: f32) -> f32 {
  let n = world[e * 16u + 1u]; let b = world[e * 16u];
  let L = pt(b + n - 1u).w;
  let s0 = max(0.0, s - GRADE_HALF); let s1 = min(L, s + GRADE_HALF);
  if (!(s1 - s0 > 0.5)) { return 0.0; }
  return clamp((yAlongEdge(b, n, s1) - yAlongEdge(b, n, s0)) / (s1 - s0), -MAX_GRADE, MAX_GRADE);
}

// first crash block record for edge e (0 = none, else index + 1)
fn blockHead(e: u32) -> u32 { if (P.nBlocks == 0u) { return 0u; } return world[P.ebOff + e]; }

// wreck info for the CPU scene record: type | laneLo << 8 | laneHi << 12 | vehicleType << 16, slot, edge, s
fn emitWreck(j: u32, v: Veh, typeIdx: u32) {
  emit(10u | (vLo(v) << 8u) | (vHi(v) << 12u) | (typeIdx << 16u), j, v.edge, bitcast<u32>(v.s));
}
fn emit(t: u32, slot: u32, a: u32, b: u32) {
  let k = atomicAdd(&io[IO_EVCOUNT], 1u);
  if (k >= P.evCap) { return; }
  let o = IO_EVDATA + 4u * k;
  atomicStore(&io[o], t); atomicStore(&io[o + 1u], slot); atomicStore(&io[o + 2u], a); atomicStore(&io[o + 3u], b);
}
fn statAdd(k: u32, v: u32) { if (v != 0u) { atomicAdd(&io[IO_STATS + k], v); } }
// 64-bit counter (lo at k, hi at k+1)
fn statAdd64(k: u32, v: u32) {
  if (v == 0u) { return; }
  let old = atomicAdd(&io[IO_STATS + k], v);
  if (old > 0xffffffffu - v) { atomicAdd(&io[IO_STATS + k + 1u], 1u); }
}

// ---------------------------------------------------------------- junctions (layout v8, src/geom junctions.js; worldPack.js packJunctions)
// A car on a junction connector keeps its in-edge frame (edge, s along the in-edge: t = s - sIn along the
// connector, lat = offset from the in-lane centre + in-lane centre). Its connector id is stored in
// Veh.taper as f32(m + 1) when P.hasJ (taper is only used without src/geom lane geometry).
fn vConn(v: Veh) -> i32 { if (P.hasJ == 0u) { return -1; } return i32(v.taper) - 1; }
fn connWord(m: i32) -> f32 { return f32(m + 1); }
fn eSExit(e: u32) -> f32 { if (P.hasJ == 0u) { return 0.0; } return bitcast<f32>(world[P.jMouthOff + 3u * e]); }
fn eSStop(e: u32) -> f32 { if (P.hasJ == 0u) { return eLength(e); } return bitcast<f32>(world[P.jMouthOff + 3u * e + 1u]); }
fn eGore(e: u32) -> f32 { if (P.hasJ == 0u) { return -1.0; } return bitcast<f32>(world[P.jMouthOff + 3u * e + 2u]); }
fn movW(m: u32, k: u32) -> u32 { return world[P.jMovOff + 9u * m + k]; }
/** src/geom: a car that must stop for the junction stops at min(sStop, sIn of its movement) (sIn <= sStop since the
 *  junctions rebuild; the connector starts there) */
fn jStopS(e: u32, m: i32) -> f32 { var s = eSStop(e); if (m >= 0) { s = min(s, bitcast<f32>(world[P.jMovOff + 9u * u32(m) + 4u])); } return s; }
fn movVTurn(m: u32) -> f32 { return bitcast<f32>(movW(m, 8u)); }
fn movInEdge(m: u32) -> u32 { return movW(m, 0u) & 0xfffffu; }
fn movInLane(m: u32) -> u32 { return (movW(m, 0u) >> 20u) & 255u; }
fn movOutEdge(m: u32) -> u32 { return movW(m, 1u) & 0xfffffu; }
fn movOutLane(m: u32) -> u32 { return (movW(m, 1u) >> 20u) & 255u; }
fn movSIn(m: u32) -> f32 { return bitcast<f32>(movW(m, 4u)); }
fn movSOut(m: u32) -> f32 { return bitcast<f32>(movW(m, 5u)); }
fn movLen(m: u32) -> f32 { return bitcast<f32>(movW(m, 6u)); }
fn movTurn(m: u32) -> f32 { return bitcast<f32>(movW(m, 7u)); }
/** J.movement(a, la, b): connector for (in-edge a, in-lane la) -> out-edge b, or -1 */
fn jMovement(a: u32, la: u32, b: u32) -> i32 {
  if (P.hasJ == 0u) { return -1; }
  let s0 = world[P.jEdgeMovOff + a]; let s1 = world[P.jEdgeMovOff + a + 1u];
  for (var k = s0; k < s1; k = k + 1u) {
    let m = world[P.jEdgeMovOff + P.E + 1u + k];
    if (movInLane(m) == la && movOutEdge(m) == b) { return i32(m); }
  }
  return -1;
}
/** J.arcPoint(m, t): point + segment heading at t metres along the connector */
fn arcPointJ(m: u32, t: f32) -> GeoPt {
  let first = movW(m, 2u); let n = movW(m, 3u);
  var k = 0u;
  loop { if (k >= n - 2u || bitcast<f32>(world[P.jPtsOff + 4u * (first + k + 1u) + 3u]) >= t) { break; } k = k + 1u; }
  let a = P.jPtsOff + 4u * (first + k); let b = a + 4u;
  let ax = bitcast<f32>(world[a]); let az = bitcast<f32>(world[a + 1u]); let ay = bitcast<f32>(world[a + 2u]); let ac = bitcast<f32>(world[a + 3u]);
  let bx = bitcast<f32>(world[b]); let bz = bitcast<f32>(world[b + 1u]); let by = bitcast<f32>(world[b + 2u]); let bc = bitcast<f32>(world[b + 3u]);
  var d = bc - ac; if (d == 0.0) { d = 1.0; }
  let u = clamp((t - ac) / d, 0.0, 1.0);
  let hx = bx - ax; let hz = bz - az; var L = sqrt(hx * hx + hz * hz); if (L == 0.0) { L = 1.0; }
  return GeoPt(ax + (bx - ax) * u, ay + (by - ay) * u, az + (bz - az) * u, hx / L, hz / L);
}
/** movements whose out-edge is e: world[P.jIntoOff + E + 1 + k], k in [jIntoBegin(e), jIntoBegin(e + 1)) (sim.js _movesInto) */
fn jIntoBegin(e: u32) -> u32 { return world[P.jIntoOff + e]; }
fn jIntoAt(k: u32) -> u32 { return world[P.jIntoOff + P.E + 1u + k]; }
/** conflicts of movement m: entries k in [jConfBegin(m), jConfBegin(m + 1)): other, type, s (mine), sOther */
fn jConfBegin(m: u32) -> u32 { return world[P.jConfOff + m]; }
fn jConfAt(k: u32) -> vec4u { let o = P.jConfOff + P.jM + 1u + 4u * k; return vec4u(world[o], world[o + 1u], world[o + 2u], world[o + 3u]); }
/** sim.js frameDir / frameEnd: road frame of a car (its junction connector, else its edge) */
fn frameDir(v: Veh) -> vec2f { let m = vConn(v); if (m >= 0) { let o = connPointJ(u32(m), v.s - movSIn(u32(m))); return vec2f(o.hx, o.hz); } return roadDirAt(v.edge, v.s); }
fn frameEnd(v: Veh) -> f32 { let m = vConn(v); if (m >= 0) { return movSIn(u32(m)) + movLen(u32(m)); } return eLength(v.edge); }
/** world pose of a car (vehicle.js updatePose): connector centre line + lateral offset, else the lane geometry */
fn poseOf(v: Veh) -> vec4f {
  let m = vConn(v);
  if (m >= 0) {
    let o = connPointJ(u32(m), v.s - movSIn(u32(m)));
    let off = v.lat - (f32(movInLane(u32(m))) + 0.5) * laneWAt(v.edge, movSIn(u32(m)));
    return vec4f(o.x + off * o.hz, o.y, o.z - off * o.hx, atan2(o.hz, o.hx) - v.headingRel);
  }
  let o = latPointAt(v.edge, v.s, v.lat);
  return vec4f(o.x, o.y, o.z, atan2(o.hz, o.hx) - v.headingRel);
}
/** src/geom LaneGeometry.laneWidth(e, s): the edge's lane width, blending to the shared node width near strict nodes */
fn laneWAt(e: u32, s0: f32) -> f32 {
  if (P.hasLG == 0u) { return LANE_W; }
  let Lg = eLength(e); let s = clamp(s0, 0.0, Lg);
  let w = geoF(e, 9u);
  var r = w;
  let ts = geoF(e, 4u); let te = geoF(e, 5u);
  if (ts > 0.0) { r = r + (geoF(e, 10u) - w) * smooth01(1.0 - s / ts); }
  if (te > 0.0) { r = r + (geoF(e, 11u) - w) * smooth01(1.0 - (Lg - s) / te); }
  return r;
}
/** vehicle.js connectorPoint: J.arcPoint position with a smooth heading (vertex tangents blended along each chord) */
fn connPointJ(m: u32, t: f32) -> GeoPt {
  var o = arcPointJ(m, t);
  let first = movW(m, 2u); let n = movW(m, 3u);
  if (n < 3u) { return o; }
  var k = 0u;
  loop { if (k >= n - 2u || bitcast<f32>(world[P.jPtsOff + 4u * (first + k + 1u) + 3u]) >= t) { break; } k = k + 1u; }
  let A = P.jPtsOff + 4u * (first + k); let B = A + 4u;
  let pa = vec2f(bitcast<f32>(world[A]), bitcast<f32>(world[A + 1u])); let pb = vec2f(bitcast<f32>(world[B]), bitcast<f32>(world[B + 1u]));
  var d = bitcast<f32>(world[B + 3u]) - bitcast<f32>(world[A + 3u]); if (d == 0.0) { d = 1.0; }
  let u = clamp((t - bitcast<f32>(world[A + 3u])) / d, 0.0, 1.0);
  var c = pb - pa; var cl = length(c); if (cl == 0.0) { cl = 1.0; }
  var a = c / cl; var b = a;
  if (k > 0u) { let pp = vec2f(bitcast<f32>(world[A - 4u]), bitcast<f32>(world[A - 3u])); var q = pa - pp; var ql = length(q); if (ql == 0.0) { ql = 1.0; } a = a + q / ql; }
  if (k < n - 2u) { let pn = vec2f(bitcast<f32>(world[B + 4u]), bitcast<f32>(world[B + 5u])); var q = pn - pb; var ql = length(q); if (ql == 0.0) { ql = 1.0; } b = b + q / ql; }
  var al = length(a); if (al == 0.0) { al = 1.0; } var bl = length(b); if (bl == 0.0) { bl = 1.0; }
  var h = (1.0 - u) * a / al + u * b / bl; var hl = length(h); if (hl == 0.0) { hl = 1.0; }
  o.hx = h.x / hl; o.hz = h.y / hl;
  return o;
}
