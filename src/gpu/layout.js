// Host-side mirror of the WGSL struct layouts in src/gpu/kernels/common.wgsl.
// Keep the field order identical to the WGSL structs.

export const VEH_FIELDS = [
  ['edge', 'u32'], ['s', 'f32'], ['speed', 'f32'], ['accel', 'f32'],
  ['lat', 'f32'], ['headingRel', 'f32'], ['steerAngle', 'f32'], ['state', 'u32'],
  ['routePos', 'u32'], ['seg', 'u32'], ['flags', 'u32'], ['cmdStep', 'i32'],
  ['stoppedSince', 'f32'], ['odometer', 'f32'], ['lanes', 'u32'], ['lastDecision', 'f32'],
  ['fuel', 'f32'], ['tank', 'f32'], ['spawnTime', 'f32'], ['routeDist', 'f32'],
  ['leader', 'i32'], ['gap', 'f32'], ['eventTime', 'f32'], ['stopPos', 'i32'],
  ['refuelState', 'u32'], ['steerSl', 'f32'], ['gas', 'f32'], ['brake', 'f32'],
  ['routeTotal', 'f32'], ['lastFuelCheck', 'f32'], ['taper', 'f32'], ['laneEndDist', 'f32'],
  ['laneEndSide', 'i32'], ['vLat', 'f32'], ['yawRate', 'f32'], ['axPrev', 'f32'],
  ['px', 'f32'], ['pz', 'f32'], ['pyaw', 'f32'], ['power', 'f32'],
];
export const PAR_FIELDS = [
  ['routeOff', 'u32'], ['routeLen', 'u32'], ['typeIdx', 'u32'], ['delaySteps', 'u32'],
  ['length', 'f32'], ['width', 'f32'], ['maxSpeed', 'f32'], ['maxAccel', 'f32'],
  ['maxDecel', 'f32'], ['headway', 'f32'], ['aggression', 'f32'], ['patience', 'f32'],
  ['reactionTime', 'f32'], ['workStart', 'f32'], ['routeLength', 'f32'], ['trip', 'u32'],
  ['wheelbase', 'f32'], ['maxSteer', 'f32'], ['maxSteerRate', 'f32'], ['typeMaxSpeed', 'f32'],
];
/** lanes word: lane | laneLo << 8 | laneHi << 16 | (targetLane + 1) << 24 */
export const packLanes = (lane, lo, hi, target = -1) => ((lane & 255) | ((lo & 255) << 8) | ((hi & 255) << 16) | (((target + 1) & 255) << 24)) >>> 0;
export const unpackLanes = (w) => ({ lane: w & 255, lo: (w >>> 8) & 255, hi: (w >>> 16) & 255, target: ((w >>> 24) & 255) - 1 });
export const PARAM_FIELDS = [
  ['cap', 'u32'], ['E', 'u32'], ['N', 'u32'], ['K', 'u32'],
  ['T', 'u32'], ['wBits', 'u32'], ['wMask', 'u32'], ['hMask', 'u32'],
  ['ptsOff', 'u32'], ['laneOffOff', 'u32'], ['adjOff', 'u32'], ['ebOff', 'u32'],
  ['dt', 'f32'], ['time0', 'f32'], ['step0', 'u32'], ['features', 'u32'],
  ['cellSize', 'f32'], ['leaderLookahead', 'f32'], ['junctionMargin', 'f32'], ['collisionInterval', 'u32'],
  ['s0', 'f32'], ['collideShrink', 'f32'], ['speedFactorMin', 'f32'], ['speedFactorSpan', 'f32'],
  ['laneCheckInterval', 'f32'], ['yieldLook', 'f32'], ['stopMargin', 'f32'], ['stuckTimeout', 'f32'],
  ['spawnBlockedTimeout', 'f32'], ['spawnCount', 'u32'], ['opCount', 'u32'], ['decideEvery', 'u32'],
  ['accRate', 'f32'], ['accSpeedRef', 'f32'], ['accCloseGap', 'f32'], ['accMergeFactor', 'f32'],
  ['accMergeDist', 'f32'], ['rubberDist', 'f32'], ['rubberFactor', 'f32'], ['towTime', 'f32'],
  ['lowFuelFrac', 'f32'], ['seed', 'u32'], ['evCap', 'u32'], ['inSize', 'u32'],
  ['qrWindow', 'f32'], ['qrDist', 'f32'], ['qrFactor', 'f32'], ['nBlocks', 'u32'],
  ['blkOff', 'u32'], ['maxCellReach', 'f32'], ['fuelScale', 'f32'], ['anticipation', 'f32'],
  ['mergeLookahead', 'f32'], ['collisionTol', 'f32'], ['contactSpeed', 'f32'], ['lateralTol', 'f32'],
  ['offroadMargin', 'f32'], ['squeezeSpeed', 'f32'], ['prepBase', 'f32'], ['prepPerLane', 'f32'],
  ['fuelCheckInterval', 'f32'], ['fuelMinFillToSkip', 'f32'], ['fuelLowFraction', 'f32'], ['fuelPlanFactor', 'f32'],
  ['fuelReserveKm', 'f32'], ['fuelPlanSpeed', 'f32'], ['progressW', 'f32'], ['nAgents', 'u32'],
  ['geoOff', 'u32'], ['lgvOff', 'u32'], ['hopOff', 'u32'], ['hasLG', 'u32'],
  ['layerGap', 'f32'], ['pairCooldown', 'f32'], ['touchKmh', 'f32'], ['shoulder', 'f32'],
  ['oppositeMargin', 'f32'], ['laneEndLookahead', 'f32'], ['contactCap', 'u32'], ['physDt', 'f32'],
  ['ctlEdgeOff', 'u32'], ['ctlApOff', 'u32'], ['ctlCOff', 'u32'], ['hasCtl', 'u32'],
  ['ctlLook', 'f32'], ['rampStart', 'f32'], ['rampEnd', 'f32'], ['freewayShoulder', 'f32'],
  ['pairOff', 'u32'], ['pairN', 'u32'], ['labelMode', 'u32'], ['ctlListOff', 'u32'],
  ['jMovOff', 'u32'], ['jEdgeMovOff', 'u32'], ['jPtsOff', 'u32'], ['jConfOff', 'u32'],
  ['jMouthOff', 'u32'], ['hasJ', 'u32'], ['jM', 'u32'], ['jHeadOff', 'u32'],
  ['rerouteSight', 'f32'], ['passWrecks', 'u32'], ['trafficOn', 'u32'], ['steerMode', 'u32'],
  ['rlOn', 'u32'], ['rlRec', 'f32'], ['rlSeed', 'u32'], ['rlChunk', 'u32'],
  ['rlStd0', 'f32'], ['rlStd1', 'f32'], ['rlStd2', 'f32'], ['rlRowCap', 'u32'],
  ['jerkW', 'f32'], ['crashBase', 'f32'], ['crashK', 'f32'], ['crashCap', 'f32'],   // steering-smoothness reward weight (src/sim/reward.js JERK); severity crash penalty (reward.js CRASH_SEV, docs/crash-penalty.md)
  ['crashFault', 'f32'], ['crashOn', 'u32'], ['crashFail', 'f32'], ['padC3', 'u32'],   // crashOn 0 = legacy flat -20 (rewardVersion 1)
  ['jIntoOff', 'u32'], ['jSight', 'u32'], ['pathSight', 'u32'], ['dlBreak', 'u32'],
  ['spawnLane', 'u32'], ['padS1', 'u32'], ['padS2', 'u32'], ['padS3', 'u32'],   // docs/scale-crashes.md: movements into each edge (CSR); sim.js junctionSight / pathSight / deadlockBreak
];
export const VEH_WORDS = VEH_FIELDS.length, PAR_WORDS = PAR_FIELDS.length, PARAM_WORDS = PARAM_FIELDS.length;
export const SPAWN_WORDS = 28; // VPar (20) + depart, status, fuel, tank, slot, flags, pad, pad
export const BLOCK_WORDS = 8;
export const CTRL_RING = 40;
export const CTRL_STRIDE = 3; // ring record: steer (layout v9 position mode: target wheel position in [-1,1]; rate mode: steerRate), gas, brake
export const MAX_OCC = 4;

export const STATE = { FREE: 0, DRIVING: 1, CRASHED: 2, REFUELING: 5 };   // stranded = CRASHED + FLAG.STRANDED (as src/sim/vehicle.js)
export const FLAG = { LEARNER: 1, COLLIDED: 2, POLICY: 4, SHOULDER: 8, SPEEDING: 16, FUELREQ: 32, STRANDED: 64, REFUEL: 128, ZIPLEAD: 256, SCORED: 512, DISABLED: 1024, WALL: 2048, WALL_PREV: 4096, NEW: 8192 };
export const FEAT = { LANECHANGE: 1, ACCIDENTS: 2, FUEL: 4, COLLISIONS: 8, YIELD: 16, POLICY: 32, ZIPFIX: 64, EMPTY: 128 };
export const EV = { ARRIVE: 1, CRASH: 2, STUCK: 3, STRANDED: 4, TOWED: 5, LOWFUEL: 6, SPAWN: 7, STATION: 8, ACCIDENT: 9, WRECK: 10, OFFROAD: 11, REENTER: 12, CONTACT: 13, CRASHX_V: 14, CRASHX_P: 15, CRASHX_H: 16, TRIPX: 17, CTLVIOL: 18, CRASHX_N: 19, CRASHX_M: 20, REROUTE: 21, MISSED: 22, CRASHPEN: 23 };
export const OP = { FREE: 1, SET_STATE: 2, SET_FLAGS: 3, CLEAR_FLAGS: 4, SET_FUEL: 7, SET_ROUTE_REFUEL: 9, PUMPED: 10, FUEL_PLAN_FAILED: 11, TOW_SHOULDER: 12, SET_PULL: 13, SET_ROUTE: 14 };

export const IO = { STEP: 0, STATS: 8, EVCOUNT: 72, EVDATA: 76 };
export const S = {
  ARRIVED: 0, ARRIVED_OUT: 1, ONTIME: 2, CRASHED_VEH: 3, CRASH_EVENTS: 4, STUCK: 5, STRANDED: 6, TOWED: 7,
  SPAWNED: 8, LANECHANGES: 9, REFUELED: 10, LOWFUEL: 11, VEHM_LO: 12, SPEEDM_LO: 14, VEHSTEPS_LO: 16,
  SPEEDSTEPS_LO: 18, LATESUM_LO: 20, ACCIDENTS: 22, BUMPS: 23,
  LIVE_DRIVING: 32, LIVE_CRASHED: 33, LIVE_STRANDED: 34, LIVE_SUMSPEED: 35, LIVE_DWELL: 36, LIVE_POLICY: 37,
};

/** Typed views over an ArrayBuffer of `n` records of `fields`. get(i, name) / set(i, name, v). */
export function recordView(buffer, fields) {
  const W = fields.length, u = new Uint32Array(buffer), f = new Float32Array(buffer), s = new Int32Array(buffer);
  const idx = Object.fromEntries(fields.map(([k, t], i) => [k, [i, t === 'f32' ? f : t === 'i32' ? s : u]]));
  return {
    W, u, f, s, idx,
    get(i, k) { const [o, a] = idx[k]; return a[i * W + o]; },
    set(i, k, v) { const [o, a] = idx[k]; a[i * W + o] = v; },
    /** column of field k for records [0, n) as a fresh typed array */
    column(k, n) { const [o, a] = idx[k]; const out = new a.constructor(n); for (let i = 0; i < n; i++) out[i] = a[i * W + o]; return out; },
  };
}
