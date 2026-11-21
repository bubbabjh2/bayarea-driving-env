// Struct-of-arrays vehicle store. No per-vehicle objects in the hot loop; every
// field is a typed array indexed by slot. Slots are recycled through a free list.
// The full field list with units is in src/sim/LAYOUT.md (GPU contract).
//
// Motion model: src/sim/physics (SPEC "Vehicles and dynamics: full physics"): the
// driver's controls {steerRate, gas, brake} go through the per-vehicle reaction-time ring;
// the applied controls drive a dynamic bicycle model (power-limited traction, brakes,
// aero / rolling / grade, weight transfer, saturating tyres with a friction circle,
// 10 ms sub-steps) in the road (Frenet) frame: s along the edge, lat from the carriageway's
// right boundary, headingRel relative to the lane direction (+ = left), body velocities
// speed (= u, forward) and vLat (+ left), yaw rate (+ left). World pose (x, z, heading)
// is derived each step for collisions (physics/collide.js), sensing and rendering.
// Lane membership is derived from the body footprint (laneLo..laneHi). Edge hops keep the
// lateral position (roadGraph.mapLat; junction hops re-project the world position).

import { edgePoint, LANE_WIDTH, laneW, mapLat, mapLane, hopMap } from './roadGraph.js';
const FREEWAY_CLS = new Set(['motorway', 'trunk', 'motorway_link', 'trunk_link']);
import { VEHICLE_TYPES } from './vehicleTypes.js';
import { VEHICLE_PARAMS, stepDynamics, createState, boxInertia } from './physics/index.js';
import { carSurface } from './surface.js';
import { DEFAULT_TRAITS } from './genome.js';

export const STATE_FREE = 0, STATE_DRIVING = 1, STATE_CRASHED = 2, STATE_ARRIVED = 3, STATE_REMOVED = 4, STATE_REFUELING = 5;
// STATE_CRASHED covers every lane-blocking stopped vehicle: crash scenes, stranded cars (FLAG_STRANDED), wrecks.
// STATE_REFUELING: in a station lot (queued / pumping); not in the lane lists or the spatial hash.
export const FLAG_LEARNER = 1, FLAG_COLLIDED = 2, FLAG_BRAIN = 4, FLAG_SHOULDER = 8, FLAG_SPEEDING = 16, FLAG_DISABLED = 32, FLAG_STRANDED = 64, FLAG_REFUEL = 128;
export const MAX_LANES = 8;
export const MAX_OCC = 4;    // max lanes one footprint can occupy (list entries per vehicle)
export const CTRL_RING = 40; // control history per vehicle (steps); max delay = CTRL_RING-1 steps (3.9 s at 0.1 s)
export const CTRL_STRIDE = 3; // ring record: [steer (position mode: target in [-1,1]; rate mode: steerRate), gas, brake]
export const TAPER = 40;      // m an ended lane continues (narrowing) past the node where it ends

const _st = createState();
const _sf = { pk: 1, sk: 0.8, crr: 1, codeR: 0, codeL: 0 };
const _pp = {}, _cp = {};

/**
 * Point t metres along junction connector m (src/geom J.arcPoint position) with a smooth
 * heading: the vertex tangents (mean of the adjacent sample chords) blended along each chord,
 * so the frame turns continuously instead of in <= 12-34 degree steps at the samples.
 */
export function connectorPoint(J, m, t, out) {
  J.arcPoint(m, t, out);
  const first = J.mInfo[7 * m + 4], n = J.mInfo[7 * m + 5], P = J.pts;
  if (n < 3) return out;
  let k = 0;
  while (k < n - 2 && P[4 * (first + k + 1) + 3] < t) k++;
  const a = 4 * (first + k), b = a + 4;
  const d = P[b + 3] - P[a + 3] || 1, u = Math.max(0, Math.min(1, (t - P[a + 3]) / d));
  const cx = P[b] - P[a], cz = P[b + 2] - P[a + 2], cl = Math.hypot(cx, cz) || 1;
  let ax = cx / cl, az = cz / cl, bx = ax, bz = az;
  if (k > 0) { const px = P[a] - P[a - 4], pz = P[a + 2] - P[a - 2], pl = Math.hypot(px, pz) || 1; ax += px / pl; az += pz / pl; }
  if (k < n - 2) { const nx = P[b + 4] - P[b], nz = P[b + 6] - P[b + 2], nl = Math.hypot(nx, nz) || 1; bx += nx / nl; bz += nz / nl; }
  const al = Math.hypot(ax, az) || 1, bl = Math.hypot(bx, bz) || 1;
  let hx = (1 - u) * ax / al + u * bx / bl, hz = (1 - u) * az / al + u * bz / bl;
  const hl = Math.hypot(hx, hz) || 1; out.hx = hx / hl; out.hz = hz / hl;
  return out;
}

/** Road slope dy/ds of edge e at s (from the per-vertex elevations; 0 when flat). */
export function edgeGrade(e, s, seg) {
  const ys = e.ys; if (!ys) return 0;
  // real terrain (data elevation model terrain/1, docs/elevation.md): the rise over a GRADE_WINDOW (12 m) window
  // centred on s, clipped to the edge, so the grade is continuous along the road (no step at every vertex) and a
  // short vertex spacing cannot produce a spike; clamped to +-35 % (steepest real SF streets ~31-41 %)
  const cum = e.cum, n = cum.length, L = cum[n - 1];
  const s0 = s - GRADE_HALF > 0 ? s - GRADE_HALF : 0, s1 = s + GRADE_HALF < L ? s + GRADE_HALF : L;
  if (!(s1 - s0 > 0.5)) return 0;
  const gr = (yAlong(ys, cum, s1, seg) - yAlong(ys, cum, s0, seg)) / (s1 - s0);
  return gr > MAX_GRADE ? MAX_GRADE : gr < -MAX_GRADE ? -MAX_GRADE : gr;
}
function yAlong(ys, cum, s, seg) {
  const n = cum.length; let k = seg < 0 ? 0 : seg > n - 2 ? n - 2 : seg;
  while (k > 0 && cum[k] > s) k--; while (k < n - 2 && cum[k + 1] < s) k++;
  const d = cum[k + 1] - cum[k], t = d > 0 ? (s - cum[k]) / d : 0;
  return ys[k] + (ys[k + 1] - ys[k]) * (t < 0 ? 0 : t > 1 ? 1 : t);
}
export const MAX_GRADE = 0.35, GRADE_HALF = 6;

export class VehicleStore {
  constructor(capacity = 4096) {
    this.actionMode = 'position';         // steering command: 'position' (v9 default, target wheel position) | 'rate' (legacy steerRate)
    this.capacity = capacity;
    this.count = 0;            // slots in use
    this.high = 0;             // highest slot ever used + 1 (iteration bound)
    this.free = [];            // free slot stack
    const C = capacity;
    this.state = new Uint8Array(C);
    this.typeIdx = new Uint8Array(C);
    this.flags = new Uint8Array(C);
    // pose in the edge frame
    this.edge = new Int32Array(C);
    this.s = new Float32Array(C);            // m along the edge centreline
    this.lat = new Float32Array(C);          // m from the carriageway's right boundary (lane k centre = (k + 0.5) * laneW(e, s), src/geom per-edge width)
    this.headingRel = new Float32Array(C);   // rad, car heading minus lane direction, + = toward the left
    this.steerAngle = new Float32Array(C);   // rad at the front wheels, + = left (actual)
    this.steerSl = new Float32Array(C);      // rad: lagged steering target (hands / arms, position mode)
    this.steerTarget = new Float32Array(C);  // applied (delayed) command: target wheel position in [-1,1] (position mode) / steerRate (rate mode)
    this.lane = new Int8Array(C);            // DERIVED: lane under the car's centre (clamped)
    this.laneLo = new Int8Array(C);          // DERIVED: lowest lane the footprint overlaps
    this.laneHi = new Int8Array(C);          // DERIVED: highest lane the footprint overlaps
    this.seg = new Int32Array(C);            // polyline segment hint
    this.speed = new Float32Array(C);          // u: forward body velocity (m/s, >= 0)
    this.vLat = new Float32Array(C);           // v: lateral body velocity (m/s, + left)
    this.yawRate = new Float32Array(C);        // r: yaw rate (rad/s, + left); world d(heading)/dt = -r
    this.axPrev = new Float32Array(C);         // longitudinal accel of the last physics sub-step (weight transfer)
    this.accel = new Float32Array(C);          // m/s^2 over the last control step
    this.mass = new Float32Array(C);           // kg
    this.inertia = new Float32Array(C);        // yaw inertia kg m^2
    this.hl = new Float32Array(C); this.hw = new Float32Array(C); // hitbox half length / half width (m)
    this.px = new Float32Array(C); this.pz = new Float32Array(C); this.pyaw = new Float32Array(C); // pose at the start of the step (swept collisions)
    this.power = new Float32Array(C);          // engine output (W)
    this.surfR = new Uint8Array(C); this.surfL = new Uint8Array(C); // surface code under the right / left wheels (src/sim/surface.js; 0 asphalt)
    // world pose (derived each step, for rendering / spatial hash / brain eyes)
    this.x = new Float32Array(C); this.y = new Float32Array(C); this.z = new Float32Array(C);
    this.heading = new Float32Array(C);      // world heading atan2(dz, dx) = road heading - headingRel
    // vehicle + driver constants
    this.length = new Float32Array(C); this.width = new Float32Array(C);
    this.wheelbase = new Float32Array(C); this.maxSteer = new Float32Array(C); this.maxSteerRate = new Float32Array(C);
    this.maxSpeed = new Float32Array(C);
    this.maxAccel = new Float32Array(C); this.maxDecel = new Float32Array(C);
    this.reactionTime = new Float32Array(C); this.headwayPref = new Float32Array(C);
    this.aggression = new Float32Array(C); this.patience = new Float32Array(C);
    // controls: the driver's command is written into a ring and applied delaySteps later
    this.steerRate = new Float32Array(C);    // applied (delayed) controls
    this.gas = new Float32Array(C);
    this.brake = new Float32Array(C);
    this.ctrlRing = new Float32Array(C * CTRL_RING * CTRL_STRIDE);
    this.delaySteps = new Uint32Array(C);    // reactionTime quantised to sim steps
    this.cmdStep = new Int32Array(C);        // step index of the last command written
    this.lastDecision = new Float32Array(C); // time of the last lane-target decision (built-in driver)
    this.targetLane = new Int8Array(C);      // built-in driver's lane target (-1 none)
    this.laneEndDist = new Float32Array(C);  // m to the end of my lane (incl. taper), 1e9 = none (observation)
    this.laneEndSide = new Int8Array(C);     // +1: surviving lanes are to my left, -1: right, 0: none
    this.mergeCap = new Float32Array(C);
    // next traffic control on the route (observation; controls.js)
    this.ctlApproach = new Int32Array(C);    // approach index (-1 none within 250 m)
    this.ctlType = new Int8Array(C);         // -1 none, 0 signal, 1 stop, 2 give_way, 3 ramp_meter, 4 crossing
    this.ctlState = new Int8Array(C);        // LIGHT: 0 green, 1 yellow, 2 red, 3 sign, 4 dark
    this.ctlDist = new Float32Array(C);      // m from my front bumper to the stop line
    this.ctlTimeIn = new Float32Array(C);    // s in the current light state
    this.ctlTimeLeft = new Float32Array(C);  // s until it changes
    this.ctlMove = new Int8Array(C);         // my movement there: 0 through, 1 left, 2 right
    this.ctlStopped = new Uint8Array(C);     // 1 once I made a full stop within 6 m of this approach's stop line
    this.ctlInside = new Int32Array(C);      // signal cluster whose first stop line I crossed (inside that intersection; -1 none)
    this.ctlInsideT = new Float32Array(C);   // time I entered it (forgotten 15 s later)
    this.conn = new Int32Array(C).fill(-1);
    this.passWreck = new Int32Array(C).fill(-1); // wreck I am passing through the opposing lane (two-lane road), -1 none  // junction connector (src/geom junctions movement id) I am driving through, -1 none
    this.taper = new Float32Array(C);        // s (m) on the current edge up to which an ENDED lane still tapers (0 = none)
    // trip
    this.routePos = new Int32Array(C);
    this.routes = new Array(C).fill(null);   // Int32Array view per slot into RouteStore.flat
    this.commuter = new Int32Array(C);
    this.leg = new Uint8Array(C);
    this.spawnTime = new Float32Array(C);
    this.odometer = new Float32Array(C);
    this.stoppedSince = new Float32Array(C); // time speed dropped under 0.2 m/s, -1 if moving
    this.crashedUntil = new Float32Array(C);
    this.leader = new Int32Array(C);         // slot of the vehicle ahead (-1 none, -2 lane-end wall, -3 junction conflict point), refreshed each step
    this.gap = new Float32Array(C);          // bumper gap to the leader (m), 1e9 if none
    this.leader2 = new Int32Array(C);        // the car in front of my leader (same edge, -1 none)
    this.gap2 = new Float32Array(C);         // bumper gap to leader2 (m), 1e9 if none
    this.pullUntil = new Float32Array(C);    // sim time until which a driver involved in a minor crash waits on the shoulder
    // packed route (RouteStore.flat) + progress
    this.routeOff = new Int32Array(C);
    this.routeLen = new Int32Array(C);
    this.routeDist = new Float32Array(C);    // m driven along the current route
    this.routeTotal = new Float32Array(C);   // total length of the current route (m)
    // fuel
    this.fuel = new Float32Array(C);         // litres
    this.tank = new Float32Array(C);         // litres
    this.refuelState = new Uint8Array(C);    // 0 none, 1 driving to station, 2 queued, 3 pumping, 4 filled (re-entering)
    this.station = new Int32Array(C);
    this.stopPos = new Int32Array(C);        // route index whose edge ends at the station (-1)
    this.pumpUntil = new Float32Array(C);
    this.lastFuelCheck = new Float32Array(C);
  }

  grow() {
    const old = this.capacity, C = old * 2;
    for (const k of Object.keys(this)) {
      const v = this[k];
      if (ArrayBuffer.isView(v) && v.length % old === 0 && v.length >= old) { const n = new v.constructor((v.length / old) * C); n.set(v); this[k] = n; }
      else if (Array.isArray(v) && v.length === old && k === 'routes') { v.length = C; v.fill(null, old); }
    }
    this.capacity = C;
  }

  /** Allocate a slot and initialise type/driver-derived fields. Returns slot index. */
  alloc({ typeIdx = 0, traits = DEFAULT_TRAITS, route, commuter = -1, leg = 0, time = 0, learner = false, dt = 0.05, step = 0, graph = null, lane = 0 }) {
    if (this.free.length === 0 && this.high >= this.capacity) this.grow();
    const i = this.free.length ? this.free.pop() : this.high++;
    this.count++;
    const t = VEHICLE_TYPES[typeIdx] || VEHICLE_TYPES[0];
    this.state[i] = STATE_DRIVING; this.typeIdx[i] = typeIdx;
    this.length[i] = t.length; this.width[i] = t.width;
    this.wheelbase[i] = t.wheelbase; this.maxSteer[i] = t.maxSteer; this.maxSteerRate[i] = t.maxSteerRate;
    this.maxSpeed[i] = t.maxSpeed;
    // driver limits: comfort acceleration / hardest braking the driver uses, capped by the vehicle
    this.maxAccel[i] = Math.min((traits.maxAccel ?? DEFAULT_TRAITS.maxAccel) * Math.min(1, 1500 / t.m * 1.6), 0.9 * t.F0 / t.m);
    this.maxDecel[i] = Math.min((traits.maxDecel ?? DEFAULT_TRAITS.maxDecel) * (t.emergencyDecel / 8.0) * 1.6, t.emergencyDecel);
    this.mass[i] = t.m; this.inertia[i] = t.Iz; this.hl[i] = 0.5 * t.length; this.hw[i] = 0.5 * t.width;
    this.reactionTime[i] = traits.reactionTime ?? DEFAULT_TRAITS.reactionTime;
    this.headwayPref[i] = traits.preferredHeadway ?? DEFAULT_TRAITS.preferredHeadway;
    this.aggression[i] = traits.aggression ?? DEFAULT_TRAITS.aggression;
    this.patience[i] = traits.patience ?? DEFAULT_TRAITS.patience;
    this.routes[i] = route; this.routePos[i] = 0;
    this.edge[i] = route.length ? route[0] : -1;
    this.s[i] = 0; this.lat[i] = (lane + 0.5) * (graph && this.edge[i] >= 0 ? laneW(graph.edges[this.edge[i]], 0) : LANE_WIDTH); this.headingRel[i] = 0; this.steerAngle[i] = 0; this.steerSl[i] = 0; this.steerTarget[i] = 0; this.seg[i] = 0;
    this.lane[i] = lane; this.laneLo[i] = lane; this.laneHi[i] = lane; this.targetLane[i] = -1; this.taper[i] = 0; this.conn[i] = -1; this.passWreck[i] = -1;
    if (graph && this.edge[i] >= 0) { const e0 = graph.edges[this.edge[i]]; if (e0.sExit > 0) { this.s[i] = e0.sExit; this.lat[i] = (lane + 0.5) * laneW(e0, e0.sExit); } }
    this.speed[i] = 0; this.vLat[i] = 0; this.yawRate[i] = 0; this.axPrev[i] = 0; this.power[i] = 0; this.accel[i] = 0; this.steerRate[i] = 0; this.gas[i] = 0; this.brake[i] = 0;
    this.ctrlRing.fill(0, i * CTRL_RING * CTRL_STRIDE, (i + 1) * CTRL_RING * CTRL_STRIDE);
    this.delaySteps[i] = Math.min(CTRL_RING - 1, Math.max(0, Math.round(this.reactionTime[i] / dt)));
    this.cmdStep[i] = step - 1;
    this.lastDecision[i] = -1e9;
    this.commuter[i] = commuter; this.leg[i] = leg; this.spawnTime[i] = time; this.odometer[i] = 0;
    this.stoppedSince[i] = -1; this.crashedUntil[i] = 0; this.flags[i] = learner ? FLAG_LEARNER : 0;
    this.leader[i] = -1; this.gap[i] = 1e9; this.leader2[i] = -1; this.gap2[i] = 1e9; this.pullUntil[i] = 0;
    this.ctlApproach[i] = -1; this.ctlType[i] = -1; this.ctlState[i] = 0; this.ctlDist[i] = 1e9; this.ctlStopped[i] = 0; this.ctlInside[i] = -1; this.mergeCap[i] = 1e9;
    this.routeOff[i] = -1; this.routeLen[i] = route.length; this.routeDist[i] = 0; this.routeTotal[i] = 0;
    for (let k = 0; k < route.length && graph; k++) this.routeTotal[i] += graph.edges[route[k]].length;
    this.tank[i] = t.tankL; this.fuel[i] = t.tankL; this.refuelState[i] = 0; this.station[i] = -1; this.stopPos[i] = -1; this.pumpUntil[i] = 0; this.lastFuelCheck[i] = time;
    if (graph) this.updateLanes(i, graph.edges[this.edge[i]]);
    return i;
  }

  release(i) {
    if (this.state[i] === STATE_FREE) return;
    this.state[i] = STATE_FREE; this.routes[i] = null; this.count--; this.conn[i] = -1;
    this.free.push(i);
  }

  /**
   * Record the controls decided at `step`; applied delaySteps later.
   * steerRate in [-1, 1] (+ = steer left), gas in [0, 1], brake in [0, 1].
   */
  command(i, steerRate, gas, brake, step) {
    const o = (i * CTRL_RING + (step % CTRL_RING)) * CTRL_STRIDE, r = this.ctrlRing;
    r[o] = steerRate > 1 ? 1 : steerRate < -1 ? -1 : steerRate || 0;
    r[o + 1] = gas > 1 ? 1 : gas > 0 ? gas : 0;
    r[o + 2] = brake > 1 ? 1 : brake > 0 ? brake : 0;
    this.cmdStep[i] = step;
  }

  setReactionTime(i, seconds, dt) {
    this.reactionTime[i] = seconds;
    this.delaySteps[i] = Math.min(CTRL_RING - 1, Math.max(0, Math.round(seconds / dt)));
  }

  /** Clear the control ring (crash / refuel stop). */
  clearControls(i) {
    this.ctrlRing.fill(0, i * CTRL_RING * CTRL_STRIDE, (i + 1) * CTRL_RING * CTRL_STRIDE);
    this.steerRate[i] = 0; this.gas[i] = 0; this.brake[i] = 0;
  }

  /** Footprint lateral half-extent (m) for the current headingRel. */
  halfExtent(i) {
    const h = this.headingRel[i];
    return 0.5 * (this.width[i] * Math.abs(Math.cos(h)) + this.length[i] * Math.abs(Math.sin(h)));
  }

  /** Derive lane / laneLo / laneHi from lat and the footprint on edge e. */
  updateLanes(i, e) {
    const he = this.halfExtent(i), lat = this.lat[i], n = e.lanes, W = laneW(e, this.s[i]);
    let lo = Math.floor((lat - he) / W), hi = Math.floor((lat + he - 1e-4) / W), c = Math.floor(lat / W);
    if (c < 0) c = 0; if (c > n - 1) c = n - 1;
    // body entirely off the travel lanes (on the shoulder / run-out): in no lane list
    if (hi < 0 || lo > n - 1) { this.lane[i] = c; this.laneLo[i] = c; this.laneHi[i] = c - 1; return; }
    if (lo < 0) lo = 0; if (hi > n - 1) hi = n - 1;
    if (lo > hi) { lo = c; hi = c; }
    if (hi - lo >= MAX_OCC) hi = lo + MAX_OCC - 1;
    this.lane[i] = c; this.laneLo[i] = lo; this.laneHi[i] = hi;
  }

  /**
   * Integrate one vehicle for dt at step index `step` through the physics module.
   * locked = disabled car (no drive, wheels locked, sliding to rest).
   * Returns 0 (driving), 1 (reached the end of its route) or 2 (reached its refuel stop).
   */
  integrate(i, dt, now, graph, step = 0, locked = false) {
    const R = this.ctrlRing, base = i * CTRL_RING;
    let sr = 0, gas = 0, brake = 1;
    if (!locked) {
      const cur = (base + (step % CTRL_RING)) * CTRL_STRIDE;
      if (this.cmdStep[i] !== step) { // no new decision this step: hold the previous one
        const prev = (base + (((step - 1) % CTRL_RING) + CTRL_RING) % CTRL_RING) * CTRL_STRIDE;
        R[cur] = R[prev]; R[cur + 1] = R[prev + 1]; R[cur + 2] = R[prev + 2];
      }
      const o = (base + (((step - this.delaySteps[i]) % CTRL_RING) + CTRL_RING) % CTRL_RING) * CTRL_STRIDE;
      sr = R[o]; gas = R[o + 1]; brake = R[o + 2];
      if (this.fuel[i] <= 0) { gas = 0; if (brake < 0.3) brake = 0.3; } // out of fuel: engine dead, brake to a stop
    }
    this.steerRate[i] = sr; this.gas[i] = gas; this.brake[i] = brake;
    let e = graph.edges[this.edge[i]];
    const P = VEHICLE_PARAMS[this.typeIdx[i]];
    const st = _st;
    st.u = this.speed[i]; st.v = this.vLat[i]; st.r = this.yawRate[i]; st.steer = this.steerAngle[i]; st.sl = this.steerSl[i]; st.ax = this.axPrev[i];
    st.hr = this.headingRel[i]; st.lat = this.lat[i]; st.s = this.s[i]; st.fuelL = 0;
    const u0 = st.u, s0 = st.s;
    // road surface under the wheels (src/sim/surface.js; graphs without data/surfaces.bin are all dry asphalt)
    const sf = graph.surfaces ? carSurface(graph, e, this.s[i], this.lat[i], this.hw[i], this.conn[i], this.taper[i], _sf) : null;
    if (sf) { this.surfR[i] = sf.codeR; this.surfL[i] = sf.codeL; }
    stepDynamics(st, sr, gas, brake, P, dt, this.conn[i] >= 0 ? 0 : edgeGrade(e, this.s[i], this.seg[i]), locked, undefined, this.actionMode, sf);
    this.steerTarget[i] = sr; this.steerSl[i] = st.sl;
    this.speed[i] = st.u; this.vLat[i] = st.v; this.yawRate[i] = st.r; this.steerAngle[i] = st.steer; this.axPrev[i] = st.ax;
    this.headingRel[i] = st.hr; this.lat[i] = st.lat; this.power[i] = st.power;
    this.accel[i] = (st.u - u0) / dt;
    if (this.fuel[i] > 0) { this.fuel[i] -= st.fuelL; if (this.fuel[i] < 0) this.fuel[i] = 0; }
    let s = st.s;
    const ds = s - s0;
    this.odometer[i] += Math.abs(ds);
    this.routeDist[i] += ds;
    if (st.u < 0.2 && Math.abs(st.v) < 0.2) { if (this.stoppedSince[i] < 0) this.stoppedSince[i] = now; } else this.stoppedSince[i] = -1;
    if (s < 0) s = 0;
    let ret = 0;
    const J = graph.J;
    let sArrive = s0;   // where I was on the current edge at the start of the step (or where I arrived on it)
    for (;;) {
      // ---- junction connectors (src/geom junctions): past the mouth's stop point a car follows the
      // connector of its movement (in-edge, lane, out-edge) through the box; s keeps counting along the
      // in-edge frame (t = s - sIn along the connector), lat keeps its offset from the in-lane centre
      if (J) {
        const m = this.conn[i];
        if (m >= 0) {
          const sIn = J.mF[4 * m], end = sIn + J.mLen[m];
          if (s < end) break;
          if (locked) { s = end; break; }
          const en = graph.edges[J.mInfo[7 * m + 2]];
          const off = this.lat[i] - (J.mInfo[7 * m + 1] + 0.5) * laneW(e, sIn), lo = J.mInfo[7 * m + 3];
          this.routeDist[i] += e.length - end + J.mF[4 * m + 1]; // route distance counts edge lengths
          s = J.mF[4 * m + 1] + (s - end);
          this.lat[i] = (lo + 0.5) * laneW(en, J.mF[4 * m + 1]) + off;
          if (this.targetLane[i] >= 0) this.targetLane[i] = lo;
          this.routePos[i]++; this.edge[i] = en.idx; this.seg[i] = 0; this.taper[i] = 0; this.conn[i] = -1;
          e = en; sArrive = J.mF[4 * m + 1];
          continue;
        }
        if (e.sStop < e.length && !locked && !(this.refuelState[i] === 1 && this.routePos[i] === this.stopPos[i])) {
          const route = this.routes[i], next = this.routePos[i] + 1;
          if (next < route.length) {
            const la = Math.min(e.lanes - 1, Math.max(0, Math.floor(this.lat[i] / laneW(e, s))));
            let mv = J.movement(e.idx, la, route[next]);
            if (mv >= 0 && s < J.mF[4 * mv]) mv = -2;   // the movement leaves e at its own sIn (may differ from sStop)
            // a turn is made from its lane: a car still in a lane that is not the turn's lane goes the way its
            // lane leads (the straightest movement preferred from that lane) - a missed turn the sim re-plans
            // (only where there was room to get into lane: an edge >= 80 m, and not again within 5 min - on short
            // edges drivers make the turn from where they are)
            const mt = this.missedT || (this.missedT = new Map()), lastM = mt.get(i);
            if (mv >= 0 && !J.mInfo[7 * mv + 6] && e.lanes > 1 && e.length >= 80 && !(lastM !== undefined && lastM.spawn === this.spawnTime[i] && now - lastM.t < 300)) {
              let best = -1, bt = 1e9;
              for (let q = graph.outStart[e.to]; q < graph.outStart[e.to + 1]; q++) {
                const o = graph.outEdges[q]; if (o === e.reverse) continue;
                const m2 = J.movement(e.idx, la, o);
                if (m2 >= 0 && J.mInfo[7 * m2 + 6] && Math.abs(J.mF[4 * m2 + 3]) < bt) { bt = Math.abs(J.mF[4 * m2 + 3]); best = m2; }
              }
              if (best >= 0) {
                const took = J.mInfo[7 * best + 2];
                (this.missed || (this.missed = new Map())).set(i, { from: e.idx, planned: route[next], took, goal: graph.edges[route[route.length - 1]].to });
                const r2 = new Int32Array(route.length); r2.set(route); r2[next] = took; this.routes[i] = r2; this.routeOff[i] = -1;
                mv = best;
                mt.set(i, { t: now, spawn: this.spawnTime[i] }); if (mt.size > 100000) mt.clear();
              }
            }
            if (mv >= 0) {
              if (s < J.mF[4 * mv]) { /* not there yet */ } else {
              const sIn = J.mF[4 * mv];
              if (sArrive > sIn + 0.3 && e.LG) {
                // already past the movement's entry point (back-to-back junction boxes on a short edge): join the
                // connector at the point nearest my body, keeping world position and heading (no jump)
                const p = e.LG.latPoint(e.idx, Math.min(s, e.length), this.lat[i], _pp);
                const fOld = Math.atan2(p.hz, p.hx), len = J.mLen[mv];
                let bt = 0, bd = 1e18;
                // nearest in position and heading (a heading error costs like 6 m of distance per rad)
                const hOld = fOld - this.headingRel[i];
                for (let t = 0; t <= Math.min(len, s - sIn + 2); t += 0.5) {
                  connectorPoint(J, mv, t, _cp);
                  let dpsi = Math.atan2(_cp.hz, _cp.hx) - hOld; dpsi = Math.atan2(Math.sin(dpsi), Math.cos(dpsi));
                  const d = (_cp.x - p.x) ** 2 + (_cp.z - p.z) ** 2 + 36 * dpsi * dpsi; if (d < bd) { bd = d; bt = t; }
                }
                connectorPoint(J, mv, bt, _cp);
                const off = (p.x - _cp.x) * _cp.hz - (p.z - _cp.z) * _cp.hx;
                s = sIn + bt; this.lat[i] = (J.mInfo[7 * mv + 1] + 0.5) * laneW(e, sIn) + off;
                let dh = Math.atan2(_cp.hz, _cp.hx) - fOld; dh = Math.atan2(Math.sin(dh), Math.cos(dh));
                this.headingRel[i] += dh;
              }
              this.conn[i] = mv;
              if (this.targetLane[i] >= 0) this.targetLane[i] = la;
              continue;
              }
            }
          }
        }
      }
      if (s < e.length) break;
      if (locked) { s = e.length; break; }
      if (this.refuelState[i] === 1 && this.routePos[i] === this.stopPos[i]) { this.routeDist[i] -= s - e.length; s = e.length; ret = 2; break; }
      const route = this.routes[i];
      const next = this.routePos[i] + 1;
      if (next >= route.length) { this.routeDist[i] -= s - e.length; s = e.length; ret = 1; break; }
      s -= e.length;
      let en = graph.edges[route[next]];
      // a strict diverge (freeway exit / lane split): the car goes where its lane physically leads. If my
      // lane does not continue onto the planned edge but onto another out-edge, that is where I go (a
      // missed exit); the sim re-plans from there (this.missed)
      if (graph.outStart && hopMap(graph, e, en).strict) {
        const lane = Math.min(e.lanes - 1, Math.max(0, Math.floor(this.lat[i] / laneW(e, e.length))));
        const ml = mapLane(graph, e, en, lane);
        if (ml < 0 || ml >= en.lanes) {
          for (let q = graph.outStart[e.to]; q < graph.outStart[e.to + 1]; q++) {
            const o = graph.edges[graph.outEdges[q]];
            if (o.idx === en.idx || o.idx === e.reverse || !hopMap(graph, e, o).strict) continue;
            const l2 = mapLane(graph, e, o, lane);
            if (l2 >= 0 && l2 < o.lanes) {
              (this.missed || (this.missed = new Map())).set(i, { from: e.idx, planned: en.idx, took: o.idx, goal: graph.edges[route[route.length - 1]].to });
              const r2 = new Int32Array(route.length); r2.set(route); r2[next] = o.idx; this.routes[i] = r2; this.routeOff[i] = -1;
              en = o; break;
            }
          }
        }
      }
      const he = this.halfExtent(i), hop = hopMap(graph, e, en);
      let l2 = mapLat(graph, e, en, this.lat[i], he);   // same physical position in the next edge's frame
      // joining a carriageway (ramp merge / turn-in): if the body would sit on the edge line, start it on
      // the nearest lane centre (an ending lane, centre outside the carriageway, is left to the run-out)
      const W = laneW(en, s), n = en.lanes;
      if (l2 >= 0 && l2 <= n * W && (l2 - he < 0.3 || l2 + he > n * W - 0.3) && (!hop.strict || e.lanes !== n || FREEWAY_CLS.has(en.highway))) {
        const ln = Math.min(n - 1, Math.max(0, Math.floor(l2 / W))); l2 = (ln + 0.5) * W;
      }
      // headingRel is kept: the road frame bends with the road everywhere (inside edges and at nodes alike),
      // so the car's heading relative to its lane is continuous across the hop
      this.lat[i] = l2;
      if (this.targetLane[i] >= 0) { const t = mapLane(graph, e, en, this.targetLane[i]); this.targetLane[i] = t < 0 ? 0 : t >= en.lanes ? en.lanes - 1 : t; }
      // the lane I am in ended at this hop: it tapers out over TAPER m of the new edge
      this.taper[i] = !en.LG && (this.lat[i] < 0 || this.lat[i] > en.lanes * W) ? TAPER : 0; // with src/geom the taper is drawn before the node
      this.routePos[i] = next;
      this.edge[i] = en.idx; this.seg[i] = 0;
      e = en; sArrive = 0;
    }
    this.s[i] = s;
    this.updateLanes(i, e);
    return ret;
  }

  /** Save the current world pose as the start-of-step pose (swept collision tests). */
  savePose(i) { this.px[i] = this.x[i]; this.pz[i] = this.z[i]; this.pyaw[i] = this.heading[i]; }

  /** World x,y,z,heading from edge / s / lat / headingRel (src/geom lane geometry when attached). */
  updatePose(i, graph, out) {
    const e = graph.edges[this.edge[i]], m = this.conn[i];
    if (m >= 0 && graph.J) {
      // on a junction connector: its centre line, offset left (hz, -hx) by my offset from the in-lane centre
      const J = graph.J; connectorPoint(J, m, this.s[i] - J.mF[4 * m], out);
      const off = this.lat[i] - (J.mInfo[7 * m + 1] + 0.5) * laneW(e, J.mF[4 * m]);
      out.x += off * out.hz; out.z -= off * out.hx;
    } else if (e.LG) e.LG.latPoint(e.idx, this.s[i], this.lat[i], out);
    else { edgePoint(e, this.s[i], e.rightEdge - this.lat[i], out, this.seg[i]); this.seg[i] = out.seg; }
    this.x[i] = out.x; this.z[i] = out.z; this.y[i] = out.y;
    this.heading[i] = Math.atan2(out.hz, out.hx) - this.headingRel[i];
  }
}

export { LANE_WIDTH };
