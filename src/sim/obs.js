// Structured per-agent observation built from the sim's SoA state (env.js observe()).
//   obs = {x, z, heading, speed, laneIndex, laneCount, laneWidth, laneOffset (+ left), onConnector,
//          laneHeading, speedLimit, roadId, routeDistance, neighbors(12), steerAngle, steerTarget, gas, brake,
//          headingRel, lat, laneEndDistance, laneEndSide, vehicle geometry (maxSteer, wheelbase, width, length,
//          mass, vehicleType), vLat, yawRate, control {type, state, distance, timeInState, timeToChange,
//          movement, stopped, yieldRequired}, lead2 {gap, rel, stopped}, pullOver,
//          oncoming {gap, closing}, junction {mouthDistance, inBox, distance, timeToConflict, stopped},
//          passingWreck, grade(3)}
//   neighbors = [ownAhead, ownBehind, leftAhead, leftBehind, rightAhead, rightBehind] x {gap m, rel m/s}
//   fillObs3 adds obs.v3 = Float32Array(358): the "input version 3" body/mirror/controls/command-history
//   block (layout in obsLayout.js; slots 0..69 are left 0 here: they are a policy-side encoding).
// The sim applies the reaction-time delay itself (per-vehicle ring), so a policy must not delay again.

import { STATE_DRIVING } from './vehicle.js';
import { LANE_WIDTH, laneW } from './roadGraph.js';
import { VEHICLE_PARAMS } from './physics/params.js';
import { gradeObs } from './gradeObs.js';
import { senseBodies, senseControls3, senseCommands } from './perceive.js';
import { OBS3 } from './obsLayout.js';

export { LANE_WIDTH };
/** Observation for slot i from the SoA state. */
export function fillObs(sim, i, obs) {
  const vs = sim.vs, e = sim.graph.edges[vs.edge[i]];
  obs.x = vs.x[i]; obs.z = vs.z[i]; obs.heading = vs.heading[i]; obs.speed = vs.speed[i];
  const m = vs.conn[i], J = sim.graph.J;
  if (m >= 0 && J) {
    // on a junction connector the lane frame is the movement's own path (a single lane through the box):
    // offset from the path centre (vehicle.js updatePose: lat - (inLane + 0.5) * W(in-edge, sIn)), no
    // neighbour lanes. (Before 2026-10-23 this reported the in-edge lane under the car, floor(lat / W):
    // a car a little off the path saw its lane index and offset flip by a lane width mid-turn, and the
    // teacher, whose route lane target had already moved to the out-edge, steered across the box and
    // left the connector up to 5 m off its out-lane, into the barrier.)
    const lw = laneW(e, J.mF[4 * m]);
    obs.laneIndex = 0; obs.laneCount = 1; obs.laneWidth = lw;
    obs.laneOffset = vs.lat[i] - (J.mInfo[7 * m + 1] + 0.5) * lw;
  } else {
    const lw = laneW(e, vs.s[i]);   // per-edge lane width (src/geom), 3.7 on synthetic graphs
    obs.laneIndex = vs.lane[i]; obs.laneCount = e.lanes; obs.laneWidth = lw;
    obs.laneOffset = vs.lat[i] - (vs.lane[i] + 0.5) * lw;   // + = left of the lane centre
  }
  obs.onConnector = m >= 0;
  obs.laneHeading = vs.heading[i] + vs.headingRel[i];              // road direction
  obs.speedLimit = e.speedLimit; obs.roadId = e.idx;
  obs.routeDistance = vs.routeDist[i];
  obs.steerAngle = vs.steerAngle[i]; obs.gas = vs.gas[i]; obs.brake = vs.brake[i];
  // v9: actual wheel angle (rad) above; the commanded target (applied after the reaction delay) and the mode
  obs.steerTarget = vs.steerTarget[i]; obs.actionMode = vs.actionMode; obs.maxSteerAngle = vs.maxSteer[i];
  obs.headingRel = vs.headingRel[i]; obs.lat = vs.lat[i];
  obs.laneEndDistance = vs.laneEndDist[i] < 1e8 ? vs.laneEndDist[i] : Infinity; obs.laneEndSide = vs.laneEndSide[i];
  // inside a junction box the path is the lane: the in-edge's lane-end report is stale there (the connector already
  // leads into the out-lane), and with no side lanes the teacher would steer one lane width off the path (edgeMerge)
  if (obs.onConnector) { obs.laneEndDistance = Infinity; obs.laneEndSide = 0; }
  // vehicle geometry / physics for the driver's predictor (src/sim/physics/params.js)
  const P = VEHICLE_PARAMS[vs.typeIdx[i]];
  obs.maxSteer = P.maxSteer; obs.maxSteerRate = P.maxSteerRate; obs.wheelbase = P.L; obs.width = P.width; obs.length = P.length;
  obs.mass = P.m; obs.vehicleType = P.name; obs.vehicleParams = P;
  obs.vLat = vs.vLat[i]; obs.yawRate = vs.yawRate[i];
  // next traffic control (controls.js): type -1 none, 0 signal, 1 stop, 2 give_way, 3 ramp_meter, 4 crossing;
  // state 0 green, 1 yellow, 2 red, 3 sign, 4 dark; distance from the front bumper to the stop line
  const ctl = obs.control || (obs.control = {});
  ctl.type = vs.ctlType[i]; ctl.state = vs.ctlState[i]; ctl.distance = vs.ctlType[i] >= 0 ? vs.ctlDist[i] : Infinity;
  ctl.timeInState = vs.ctlTimeIn[i]; ctl.timeToChange = vs.ctlTimeLeft[i]; ctl.movement = vs.ctlMove[i]; ctl.stopped = vs.ctlStopped[i] === 1;
  // conflicting traffic at the next junction (the built-in driver's gap judgement: permissive lefts, right on red, signs)
  ctl.yieldRequired = !!sim.ctx.yieldRequired(i, e, e.length - vs.s[i]);
  fillNeighbors(sim, i, e, obs.neighbors);
  // two cars ahead in my lane (wrecks hidden behind the leader included; stopped / disabled = speed 0)
  const l2 = vs.leader2[i], ld2 = obs.lead2 || (obs.lead2 = {});
  ld2.gap = l2 >= 0 ? Math.max(0, vs.gap2[i]) : Infinity; ld2.rel = l2 >= 0 ? (vs.state[l2] === 1 ? vs.speed[l2] : 0) - vs.speed[i] : 0; ld2.stopped = l2 >= 0 && vs.state[l2] !== 1;
  obs.pullOver = vs.pullUntil[i] > sim.time;
  // layout v8: oncoming traffic in the opposing lane (passing wrecks on two-lane roads) and junction conflicts
  obs.oncoming = sim.oncomingObs(i, obs.oncoming || {});
  obs.junction = sim.junctionObs(i, obs.junction || {});
  obs.passingWreck = vs.passWreck[i] >= 0;   // minor crash on a freeway: wait on the shoulder until false
  gradeObs(sim, i, obs.grade || (obs.grade = new Float64Array(3)));   // road grade (docs/grade-input.md): input version 2 + teacher
  return obs;
}

/** Input version 3 (src/sim/perceive.js): bodies + mirror, controls extras, own commands into obs.v3; leader id. */
export function fillObs3(sim, i, obs) {
  const v3 = obs.v3 || (obs.v3 = new Float32Array(OBS3));
  senseBodies(sim, i, v3, obs.laneWidth, obs.laneOffset);
  senseControls3(sim, i, v3);
  senseCommands(sim, i, v3);
  obs.leaderId = sim.vs.leader[i];
  return obs;
}

/** Neighbour summary for the brain's eyes from the footprint lane lists. */
export function fillNeighbors(sim, i, e, out) {
  const vs = sim.vs, lanes = sim.lanes, lane = vs.lane[i], s = vs.s[i], v = vs.speed[i], half = 0.5 * vs.length[i];
  out.fill(Infinity);
  for (let k = 1; k < 12; k += 2) out[k] = 0;
  const L = vs.leader[i];
  if (L >= 0 || L <= -2) { out[0] = Math.max(0, vs.gap[i]); out[1] = (L >= 0 && vs.state[L] === STATE_DRIVING ? vs.speed[L] : 0) - v; }
  const B = lanes.behind(i, lane);
  if (B >= 0) { out[2] = Math.max(0, s - vs.s[B] - half - 0.5 * vs.length[B]); out[3] = vs.speed[B] - v; }
  const pair = sim.ctx.scratchPair2;
  const side = (l, o) => {
    if (l < 0 || l >= e.lanes) return;
    lanes.neighborsAt(e.idx, l, s, pair, i);
    if (pair[0] >= 0) { out[o] = Math.max(0, vs.s[pair[0]] - s - half - 0.5 * vs.length[pair[0]]); out[o + 1] = (vs.state[pair[0]] === STATE_DRIVING ? vs.speed[pair[0]] : 0) - v; }
    if (pair[1] >= 0) { out[o + 2] = Math.max(0, s - vs.s[pair[1]] - half - 0.5 * vs.length[pair[1]]); out[o + 3] = (vs.state[pair[1]] === STATE_DRIVING ? vs.speed[pair[1]] : 0) - v; }
  };
  side(lane + 1, 4);  // left
  side(lane - 1, 8);  // right
  return out;
}
