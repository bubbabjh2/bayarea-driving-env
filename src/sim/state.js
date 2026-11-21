// State export / import for cross-checking the CPU reference against the GPU path
// (src/gpu, owned by the GPU agent). The layout is documented field by field in
// src/sim/LAYOUT.md; LAYOUT_VERSION changes with every layout change (changelog there).
//
//   const snap = exportState(sim)          // {version, meta, arrays: {name: TypedArray copy}, routes, scenes}
//   importState(sim, snap)                 // overwrite the sim's per-vehicle state from a snapshot
//   STATE_FIELDS                           // [name, ctor, perVehicle] for every per-vehicle array

import { CTRL_RING, CTRL_STRIDE, MAX_OCC } from './vehicle.js';

export const LAYOUT_VERSION = 9;

/** Per-vehicle arrays in VehicleStore: [name, TypedArray constructor, elements per vehicle]. */
export const STATE_FIELDS = [
  ['state', Uint8Array, 1], ['typeIdx', Uint8Array, 1], ['flags', Uint8Array, 1],
  ['edge', Int32Array, 1], ['s', Float32Array, 1], ['lat', Float32Array, 1], ['headingRel', Float32Array, 1], ['steerAngle', Float32Array, 1],
  ['lane', Int8Array, 1], ['laneLo', Int8Array, 1], ['laneHi', Int8Array, 1], ['seg', Int32Array, 1],
  ['speed', Float32Array, 1], ['vLat', Float32Array, 1], ['yawRate', Float32Array, 1], ['axPrev', Float32Array, 1], ['accel', Float32Array, 1],
  ['mass', Float32Array, 1], ['inertia', Float32Array, 1], ['hl', Float32Array, 1], ['hw', Float32Array, 1],
  ['px', Float32Array, 1], ['pz', Float32Array, 1], ['pyaw', Float32Array, 1], ['power', Float32Array, 1],
  ['x', Float32Array, 1], ['y', Float32Array, 1], ['z', Float32Array, 1], ['heading', Float32Array, 1],
  ['length', Float32Array, 1], ['width', Float32Array, 1], ['wheelbase', Float32Array, 1], ['maxSteer', Float32Array, 1], ['maxSteerRate', Float32Array, 1],
  ['maxSpeed', Float32Array, 1], ['maxAccel', Float32Array, 1], ['maxDecel', Float32Array, 1],
  ['reactionTime', Float32Array, 1], ['headwayPref', Float32Array, 1], ['aggression', Float32Array, 1], ['patience', Float32Array, 1],
  ['steerRate', Float32Array, 1], ['gas', Float32Array, 1], ['brake', Float32Array, 1],
  ['ctrlRing', Float32Array, CTRL_RING * CTRL_STRIDE], ['delaySteps', Uint32Array, 1], ['cmdStep', Int32Array, 1],
  ['lastDecision', Float32Array, 1], ['targetLane', Int8Array, 1],
  ['laneEndDist', Float32Array, 1], ['laneEndSide', Int8Array, 1], ['taper', Float32Array, 1], ['conn', Int32Array, 1], ['steerSl', Float32Array, 1], ['steerTarget', Float32Array, 1], ['passWreck', Int32Array, 1], ['mergeCap', Float32Array, 1],
  ['ctlApproach', Int32Array, 1], ['ctlType', Int8Array, 1], ['ctlState', Int8Array, 1], ['ctlDist', Float32Array, 1],
  ['ctlTimeIn', Float32Array, 1], ['ctlTimeLeft', Float32Array, 1], ['ctlMove', Int8Array, 1], ['ctlStopped', Uint8Array, 1], ['ctlInside', Int32Array, 1], ['ctlInsideT', Float32Array, 1],
  ['routePos', Int32Array, 1], ['commuter', Int32Array, 1], ['leg', Uint8Array, 1], ['spawnTime', Float32Array, 1], ['odometer', Float32Array, 1],
  ['stoppedSince', Float32Array, 1], ['crashedUntil', Float32Array, 1], ['leader', Int32Array, 1], ['gap', Float32Array, 1], ['leader2', Int32Array, 1], ['gap2', Float32Array, 1], ['pullUntil', Float32Array, 1],
  ['routeOff', Int32Array, 1], ['routeLen', Int32Array, 1], ['routeDist', Float32Array, 1], ['routeTotal', Float32Array, 1],
  ['fuel', Float32Array, 1], ['tank', Float32Array, 1], ['refuelState', Uint8Array, 1], ['station', Int32Array, 1], ['stopPos', Int32Array, 1],
  ['pumpUntil', Float32Array, 1], ['lastFuelCheck', Float32Array, 1],
];

/** Snapshot of everything a GPU mirror needs (copies; safe to keep). */
export function exportState(sim) {
  const vs = sim.vs, n = vs.high, arrays = {};
  for (const [name, , per] of STATE_FIELDS) arrays[name] = vs[name].slice(0, n * per);
  const L = sim.lanes;
  return {
    version: LAYOUT_VERSION,
    meta: { time: sim.time, stepCount: sim.stepCount, dt: sim.dt, high: n, count: vs.count, capacity: vs.capacity, ctrlRing: CTRL_RING, ctrlStride: CTRL_STRIDE, maxOcc: MAX_OCC, edgeCount: sim.graph.edgeCount },
    arrays,
    routes: { flat: sim.routeStore.flat.slice(0, sim.routeStore.used), baseLen: sim.routeStore.baseLen, popOffset: sim.routeStore.popOffset },
    laneIndex: { order: L.order.slice(0, L.nEntries), entPos: L.entPos.slice(0, n * MAX_OCC), touched: L.touched.slice(0, L.nTouched) },
    scenes: sim.activeCrashes.map(r => ({ id: r.id, edge: r.edge, s: r.s, laneMask: r.laneMask, responseState: r.responseState, squeeze: r.squeeze, blockEnd: r.blockEnd, clearEnd: r.clearEnd, vehicles: r.vehicles.slice(), severity: r.severity, isCrash: r.isCrash })),
  };
}

/**
 * Overwrite the sim's per-vehicle state from a snapshot (same graph, same population).
 * Route views are rebuilt from routeOff/routeLen into the snapshot's route buffer; lane
 * lists and the hash are rebuilt by the next step. Scenes are not re-created.
 */
export function importState(sim, snap) {
  if (snap.version !== LAYOUT_VERSION) throw new Error(`state layout ${snap.version} != ${LAYOUT_VERSION}`);
  const vs = sim.vs, n = snap.meta.high;
  while (vs.capacity < n) vs.grow();
  for (const [name, , per] of STATE_FIELDS) if (snap.arrays[name]) vs[name].set(snap.arrays[name].subarray(0, n * per));
  const rs = sim.routeStore;
  if (snap.routes && snap.routes.flat.length > rs.flat.length) { const f = new Int32Array(snap.routes.flat.length + (1 << 18)); rs.flat = f; }
  if (snap.routes) { rs.flat.set(snap.routes.flat); rs.used = snap.routes.flat.length; }
  vs.high = n; vs.count = 0; vs.free = [];
  for (let i = 0; i < n; i++) {
    if (vs.state[i] === 0) { vs.free.push(i); vs.routes[i] = null; continue; }
    vs.count++;
    vs.routes[i] = vs.routeOff[i] >= 0 ? rs.flat.subarray(vs.routeOff[i], vs.routeOff[i] + vs.routeLen[i]) : vs.routes[i];
  }
  vs.free.reverse();
  sim.time = snap.meta.time; sim.stepCount = snap.meta.stepCount;
  return sim;
}
