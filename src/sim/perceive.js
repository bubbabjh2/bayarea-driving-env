// CPU reference of the input-version-3 perception (src/sim/obsLayout.js layout, docs/perception-design.md);
// mirrored 1:1 by src/gpu/kernels/perceive.wgsl (bodies, mirror inputs, controls extras, command history).
// Reads the sim state at the decision (positions = end of the last integration, the same state the spatial hash
// holds; the GPU hash is built from the same pose buffer).
import { STATE_DRIVING, STATE_CRASHED, CTRL_RING, CTRL_STRIDE } from './vehicle.js';
import { CTRL, LIGHT, MOVE, movementOf } from './controls.js';
import {
  NB_K, NB_D, NB_OFF, NB_RANGE, LAYER_GAP, MIRROR_OFF, CTL3_OFF, HIST_OFF, HIST_VISIBLE,
  sectorOf, resetSelection, offerBody, forceBody, bodyFeatures, INDICATOR_DIST, CONFLICT_DIST,
} from './obsLayout.js';

/** movement (junction connector id) car j is on, or will take at the end of its edge if within `within` m; -1 none */
export function movementAhead(sim, j, within) {
  const vs = sim.vs, J = sim.graph.J; if (!J) return -1;
  const m = vs.conn[j]; if (m >= 0) return m;
  const e = sim.graph.edges[vs.edge[j]], route = vs.routes[j], rp = vs.routePos[j];
  if (!route || rp + 1 >= route.length) return -1;
  if (e.length - vs.s[j] - vs.hl[j] > within) return -1;
  return J.movement(e.idx, vs.lane[j], route[rp + 1]);
}
/** turn indicator of car j: {left, right} from its route intent (turn at the junction ahead, else a lane end) */
export function indicatorOf(sim, j, out) {
  const vs = sim.vs, J = sim.graph.J;
  out.left = false; out.right = false;
  if (vs.state[j] !== STATE_DRIVING) return out;
  const m = movementAhead(sim, j, Math.max(INDICATOR_DIST, 4 * vs.speed[j]));
  if (m >= 0) { const th = J.mF[4 * m + 3]; out.left = th < -0.5; out.right = th > 0.5; return out; }
  if (vs.laneEndDist[j] < 100 && vs.laneEndSide[j] !== 0) { out.left = vs.laneEndSide[j] > 0; out.right = vs.laneEndSide[j] < 0; }
  return out;
}
const ME = { x: 0, z: 0, h: 0, v: 0, hl: 0, hw: 0, length: 0, width: 0 };
const OT = { x: 0, z: 0, h: 0, v: 0, hl: 0, hw: 0, length: 0, width: 0, brake: 0, indL: false, indR: false, crashed: false, conflict: false };
const IND = { left: false, right: false };
const SEL = { d2: new Float64Array(NB_K), j: new Int32Array(NB_K) };

/**
 * Bodies (NB block) + mirror inputs 70..78 for slot i into out (an OBS3 vector).
 * laneW / laneOff: my lane width and offset from my lane centre (+ left), as eyes.js inputs 4 (before adjust.js).
 */
export function senseBodies(sim, i, out, laneW, laneOff) {
  const vs = sim.vs, hash = sim.hash, R = NB_RANGE;
  const x = vs.x[i], z = vs.z[i], y = vs.y[i], h = vs.heading[i];
  const c = Math.cos(h), s = Math.sin(h);
  const hlI = vs.hl[i], hwI = vs.hw[i], v = vs.speed[i];
  resetSelection(SEL);
  // mirror accumulators
  let altL = 0, altR = 0, ttcRL = Infinity, ttcRR = Infinity, gapRL = Infinity, gapRR = Infinity, ttcOwn = Infinity, clrL = Infinity, clrR = Infinity;
  const hw = 0.5 * laneW;
  hash.forEachNear(x, z, R, (j, d2) => {
    if (j === i) return;
    const st = vs.state[j]; if (st !== STATE_DRIVING && st !== STATE_CRASHED) return;
    if (Math.abs(vs.y[j] - y) >= LAYER_GAP) return;
    const dx = vs.x[j] - x, dz = vs.z[j] - z;
    const fwd = dx * c + dz * s, left = dx * s - dz * c;
    offerBody(SEL, sectorOf(fwd, left), d2, j);
    // mirror inputs (my lane frame approximated by my heading)
    const hlJ = vs.hl[j], hwJ = vs.hw[j], lat = left + laneOff, lo = lat - hwJ, hi = lat + hwJ;
    const vj = st === STATE_DRIVING ? vs.speed[j] : 0;
    const vf = vj * Math.cos(vs.heading[j] - h) - v;   // closing from behind when > 0
    const inL = hi > hw && lo < 3 * hw, inR = lo < -hw && hi > -3 * hw;
    const inOwn = hi > Math.min(-hw, laneOff - hwI) && lo < Math.max(hw, laneOff + hwI);
    const along = Math.abs(fwd) < hlI + hlJ + 1;
    if (along && inL) altL = 1;
    if (along && inR) altR = 1;
    if (fwd < -(hlI + hlJ)) {
      const g = -fwd - hlI - hlJ;
      if (g <= 60) {
        const t = vf > 0.1 ? g / vf : Infinity;
        if (inL) { if (g < gapRL) gapRL = g; if (t < ttcRL) ttcRL = t; }
        if (inR) { if (g < gapRR) gapRR = g; if (t < ttcRR) ttcRR = t; }
        if (inOwn && t < ttcOwn) ttcOwn = t;
      }
    }
    if (Math.abs(fwd) < hlI + hlJ) {
      if (left > 0) { const cl = Math.max(0, left - hwI - hwJ); if (cl < clrL) clrL = cl; }
      else { const cl = Math.max(0, -left - hwI - hwJ); if (cl < clrR) clrR = cl; }
    }
  });
  // guaranteed member: my own-lane leader
  const L = vs.leader[i];
  if (L >= 0 && (vs.state[L] === STATE_DRIVING || vs.state[L] === STATE_CRASHED) && Math.abs(vs.y[L] - y) < LAYER_GAP) {
    const dx = vs.x[L] - x, dz = vs.z[L] - z, d2 = dx * dx + dz * dz;
    if (d2 <= R * R) forceBody(SEL, sectorOf(dx * c + dz * s, dx * s - dz * c), d2, L);
  }
  const t1 = (t) => (t === Infinity ? 1 : Math.min(1, t / 6));
  const g1 = (g, n) => (g === Infinity ? 1 : Math.min(1, g / n));
  const M = MIRROR_OFF;
  out[M] = altL; out[M + 1] = altR; out[M + 2] = t1(ttcRL); out[M + 3] = t1(ttcRR);
  out[M + 4] = g1(gapRL, 60); out[M + 5] = g1(gapRR, 60); out[M + 6] = t1(ttcOwn);
  out[M + 7] = g1(clrL, 3.7); out[M + 8] = g1(clrR, 3.7);
  // body features
  ME.x = x; ME.z = z; ME.h = h; ME.v = v; ME.hl = hlI; ME.hw = hwI;
  const mMe = movementAhead(sim, i, CONFLICT_DIST);
  const conf = mMe >= 0 ? sim._conflicts(mMe) : null;
  for (let k = 0; k < NB_K; k++) {
    const at = NB_OFF + k * NB_D, j = SEL.j[k];
    if (j < 0) { out.fill(0, at, at + NB_D); continue; }
    const crashed = vs.state[j] !== STATE_DRIVING;
    OT.x = vs.x[j]; OT.z = vs.z[j]; OT.h = vs.heading[j]; OT.v = crashed ? 0 : vs.speed[j];
    OT.hl = vs.hl[j]; OT.hw = vs.hw[j]; OT.length = vs.length[j]; OT.width = vs.width[j];
    OT.brake = vs.brake[j]; OT.crashed = crashed;
    indicatorOf(sim, j, IND); OT.indL = IND.left; OT.indR = IND.right;
    OT.conflict = false;
    if (conf && conf.length) { const mj = movementAhead(sim, j, CONFLICT_DIST); if (mj >= 0) for (const q of conf) if (q.other === mj) { OT.conflict = true; break; } }
    bodyFeatures(ME, OT, out, at);
  }
  return out;
}

/** type of the stop line at the end of edge e (its last approach), or -1 */
function endSignType(C, e) { const a = C.edgeApStart[e + 1] - 1; return a >= C.edgeApStart[e] ? C.type[C.apControl[C.edgeApList[a]]] : -1; }
const ST = {};
/** controls extras 79..86 for slot i (after _controlObs this step) */
export function senseControls3(sim, i, out) {
  const vs = sim.vs, C = sim.controls, g = sim.graph, J = g.J, o = CTL3_OFF;
  for (let k = 0; k < 8; k++) out[o + k] = 0;
  out[o + 3] = 1;
  if (!C) return out;
  const a = vs.ctlApproach[i];
  if (a >= 0 && vs.ctlType[i] === CTRL.SIGNAL && vs.ctlState[i] === LIGHT.GREEN && vs.ctlMove[i] === MOVE.LEFT && C.apProtLeft[a] !== 1) out[o] = 1;
  const m = movementAhead(sim, i, CONFLICT_DIST);
  if (m >= 0) {
    const inE = J.mInfo[7 * m], my = endSignType(C, inE);
    let n = 0, signed = 0, stops = 0;
    for (const q of sim._conflicts(m)) {
      const e2 = J.mInfo[7 * q.other]; if (e2 === inE) continue;
      const t = endSignType(C, e2); n++;
      if (t === CTRL.STOP || t === CTRL.GIVE_WAY) signed++;
      if (t === CTRL.STOP) stops++;
    }
    if (n > 0 && my !== CTRL.STOP && my !== CTRL.GIVE_WAY && signed === n) out[o + 1] = 1;
    if (n > 0 && my === CTRL.STOP && stops === n) out[o + 2] = 1;
  }
  // second control on my route after the first one (vs.ctlApproach), within 250 m
  if (a >= 0 && vs.ctlType[i] >= 0) {
    const route = vs.routes[i], rp = vs.routePos[i], eA = C.apEdge[a];
    let pos = -1;
    for (let k = rp; k < route.length && k < rp + 12; k++) if (route[k] === eA) { pos = k; break; }
    if (pos >= 0) {
      let a2 = -1, d = vs.ctlDist[i], k = pos;
      for (let q = C.edgeApStart[eA]; q < C.edgeApStart[eA + 1]; q++) { const x = C.edgeApList[q]; if (C.apStopAt[x] > C.apStopAt[a]) { a2 = x; d += C.apStopAt[x] - C.apStopAt[a]; break; } }
      if (a2 < 0) {
        d += g.edges[eA].length - C.apStopAt[a];
        for (k = pos + 1; k < route.length && k < pos + 9 && d < 250 && a2 < 0; k++) {
          const en = route[k];
          if (C.edgeApStart[en + 1] > C.edgeApStart[en]) { a2 = C.edgeApList[C.edgeApStart[en]]; d += C.apStopAt[a2]; break; }
          d += g.edges[en].length;
        }
      } else k = pos;
      if (a2 >= 0 && d < 250) {
        const e2 = C.apEdge[a2], p2 = e2 === eA ? pos : k;
        let mv = MOVE.THROUGH; if (p2 + 1 < route.length) mv = movementOf(g.edges[e2], g.edges[route[p2 + 1]]);
        C.stateOf(a2, mv, sim.time, ST);
        out[o + 3] = Math.min(1, Math.max(0, d) / 250);
        const ty = ST.type;
        if (ty === CTRL.SIGNAL) out[o + 4] = 1; else if (ty === CTRL.STOP || ty === CTRL.GIVE_WAY) out[o + 5] = 1; else out[o + 6] = 1;
        out[o + 7] = ST.state === LIGHT.RED || ST.state === LIGHT.YELLOW ? 1 : 0;
      }
    }
  }
  return out;
}

/** own command history 87..93 from the reaction-delay ring (decision at sim.stepCount, before it is written) */
export function senseCommands(sim, i, out) {
  const vs = sim.vs, R = vs.ctrlRing, S = sim.stepCount, d = vs.delaySteps[i];
  const slot = (st) => (i * CTRL_RING + (((st % CTRL_RING) + CTRL_RING) % CTRL_RING)) * CTRL_STRIDE;
  const now = slot(d >= 1 ? S - d : S - 1), last = slot(S - 1);
  out[HIST_OFF] = R[now]; out[HIST_OFF + 1] = R[now + 1]; out[HIST_OFF + 2] = R[now + 2];
  out[HIST_OFF + 3] = R[last]; out[HIST_OFF + 4] = R[last + 1]; out[HIST_OFF + 5] = R[last + 2];
  out[HIST_VISIBLE] = 1;
  return out;
}
