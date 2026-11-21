// TEST FIXTURE ONLY: the delay-compensated lane-change steering controller of the original project's built-in
// driver, used by physics.test.js to check that the delayed steering loop of the physics is stable. Not part of the env.
import { VEHICLE_PARAMS, stepDynamics, createState } from '../../src/sim/physics/index.js';
import { CTRL_RING, CTRL_STRIDE } from '../../src/sim/vehicle.js';
const _pst = createState();

/**
 * Steering controller toward lateral position latT (m from the right boundary).
 * Predicts lat / headingRel / steerAngle one reaction time ahead using the commands
 * already in the delay ring (Smith-predictor style) so the delayed loop stays stable.
 */
export function steerToward(vs, i, latT) {
  const P = VEHICLE_PARAMS[vs.typeIdx[i]];
  // steady-state yaw gain of the dynamic model: r = u d / (L + Kus u^2), Kus = m/L (b/Cf - a/Cr)
  const Kus = P.m / P.L * (P.b / P.Cf - P.a / P.Cr);
  const u0 = vs.speed[i], L = vs.wheelbase[i] + Math.max(0, Kus) * u0 * u0, msr = vs.maxSteerRate[i], ms = vs.maxSteer[i];
  const d = vs.delaySteps[i], dtc = vs._dt || 0.05;
  // forward-simulate the commands already queued in the delay ring through the physics model
  // (exact Smith predictor of the delayed loop, 50 ms prediction sub-steps): state after the delay
  const st = _pst;
  st.u = vs.speed[i]; st.v = vs.vLat[i]; st.r = vs.yawRate[i]; st.steer = vs.steerAngle[i]; st.sl = vs.steerSl[i]; st.ax = vs.axPrev[i]; st.hr = vs.headingRel[i]; st.lat = vs.lat[i]; st.s = 0; st.fuelL = 0;
  const R = vs.ctrlRing, base = i * CTRL_RING, step = vs.cmdStep[i];
  for (let k = d - 1; k >= 0; k--) {
    const o = (base + (((step - k) % CTRL_RING) + CTRL_RING) % CTRL_RING) * CTRL_STRIDE;
    stepDynamics(st, R[o], R[o + 1], R[o + 2], P, dtc, 0, false, 0.05, vs.actionMode);
  }
  let delta = st.steer, v = st.u, hr = st.hr, lat = st.lat;
  // pure pursuit from the predicted state: aim at the target line Ld ahead
  const Ld = Math.max(12, v * (3.0 + 0.6 * d * dtc));
  // approach angle limited to a lateral speed of ~1.3 m/s (a normal lane change)
  const hrMax = v > 1.7 ? Math.asin(1.0 / v) : 0.6;
  let app = Math.atan2(latT - lat, Ld); if (app > hrMax) app = hrMax; else if (app < -hrMax) app = -hrMax;
  const alpha = app - hr;
  let dDes = Math.atan(2 * L * Math.sin(alpha) / Ld);
  // yaw-rate limit (0.4 rad/s) at the predicted speed keeps low-speed manoeuvres gentle
  const dYaw = Math.atan(L * 0.4 / Math.max(v, 1));
  const lim = Math.min(ms, dYaw);
  if (dDes > lim) dDes = lim; else if (dDes < -lim) dDes = -lim;
  if (vs.actionMode === 'position') return dDes / ms;   // target wheel position (the actuator lag / slew is in the physics)
  const sr = (dDes - delta) / (msr * 0.4);
  return sr > 1 ? 1 : sr < -1 ? -1 : sr;
}

