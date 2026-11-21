// Reward for the per-agent online learner and for evolutionary fitness.
//
// There is NO penalty for any driving behaviour (speeding, headway, harsh
// braking, lane use, off-route): those cost time, and time costs lateness.
// The only penalties are crashing and being late, plus ONE small exception (SPEC 2026-10-24):
// a steering-smoothness term on sharp changes of the steering command (see JERK below). Off-route / missed exits
// are handled by rerouting (which costs time).
//
//   reward({ progress, collision, arrived, lateBy, dt }, weights)
//     progress   route distance gained this step / (speedLimit * dt): ~1 at the
//                limit, 0 when stopped, negative when moving backwards / off route
//                (the driver computes it from its own route tracking)
//     collision  bool (this step)          arrived  bool (once, on the arrival step)
//     stranded   bool (ran out of fuel: the trip failed, same penalty as a crash)
//     lateBy     seconds late at arrival (negative = early), only with arrived
//   r = progressWeight * progress * dt            dense coaching term (see schedule)
//       - 20 * (collision || stranded)             crash / out of fuel (the trip has failed)
//                                                  (rewardVersion 2: crash = -crashPenalty(severity), see CRASH_SEV)
//       + 10 * arrived + 2 * (arrived && lateBy <= 0)
//       + max(-10, -0.5 * lateBy / 60) * arrived   -0.5 per minute late, capped
//       - steerJerk * dSteer^2 * jerkScale          steering smoothness (small; see JERK)
//
// STEERING SMOOTHNESS (user, 2026-10-24: "real drivers get honked at for jerking the wheel"). dSteer = change
// of the steering command (target wheel position, fraction of lock) between two consecutive decisions.
// jerkScale(speed, ttc) = min(4, (speed / 20)^2) * ease(ttc): a jerk matters with v^2 (a quarter-lock yank at
// 20 m/s spins the car, the same yank at 3 m/s is parking), and the term fades out when a collision is
// imminent (time to collision with the own-lane leader / obstacle below 3 s -> weight falls linearly to 0 at 1 s)
// so an evasive swerve is never discouraged. Weight steerJerk (REWARD_WEIGHTS, flag --jerk-rl, GPU
// Params.jerkW): default 0.5 -> a smooth teacher (|dSteer| ~ 2e-4 per step) pays nothing measurable; a
// chattering net (|dSteer| ~ 0.02 per step at 15-25 m/s) pays ~0.002 / s, a 0.3-lock yank at 20 m/s 0.045,
// both small next to a crash (-20). The same scale enters the BC / DAgger loss (train.js trainBC jerkWeight).
//
// PROGRESS SCHEDULE: `progress` (progressWeight) starts at 1.0 and is annealed
// toward a floor of 0.1 as the agents' arrival rate improves:
//   progressWeight = floor + (start - floor) * (1 - arrivalRate)   (annealProgress)
// so early training is coached by "keep moving along the route" and the final
// behaviour is driven by crash + lateness. The sim/trainer sets
// REWARD_WEIGHTS.progress (shared by all drivers) from its measured arrival rate.
export const REWARD_WEIGHTS = {
  progress: 1.0, progressFloor: 0.1, progressStart: 1.0,
  collision: -20, arrived: 10, onTime: 2, latePerMinute: -0.5, lateCap: -10,
  steerJerk: 0.5,
  crashSev: null,   // null = rewardVersion 1 (flat collision -20); an object like CRASH_SEV = rewardVersion 2 (see crashPenalty)
};

/**
 * SEVERITY-SCALED CRASH PENALTY (rewardVersion 2, user 2026-10-24; docs/crash-penalty.md). Off unless
 * REWARD_WEIGHTS.crashSev (CPU) / Params.crashOn (GPU, integrate.wgsl + contacts.wgsl) is set.
 *   penalty = min(cap, base + fail * disabled + k * sev)
 *   sev     = dvKmh * (1 + fault * (2 * share - 1))
 * dvKmh = the car's own delta-v of this contact (km/h; the sim's crash severity), share = crashShare(): the car's
 * fraction of the closing speed along the contact normal (1 = it drove into a stopped car / a wall, 0 = it was hit
 * while stopped, 0.5 = head-on with equal speeds), fault = weight of that share (0 = ignore who hit whom).
 * disabled = the contact disabled the car (own delta-v >= 10 km/h: the trip fails). Every crash event (largest
 * delta-v >= touchKmh 1 km/h) is charged, once per car, also the minor contacts after which the car drives on.
 */
export const CRASH_SEV = { base: 8, fail: 12, k: 0.4, fault: 0.5, cap: 80 };

/** Share of the closing speed along the contact normal that car A brought (pre-impact speed, yaw). B = null: a wall. */
export function crashShare(speedA, yawA, speedB, yawB, nx, nz) {
  let cA = speedA * (Math.cos(yawA) * nx + Math.sin(yawA) * nz);
  let cB = -(speedB || 0) * (Math.cos(yawB || 0) * nx + Math.sin(yawB || 0) * nz);
  if (cA + cB < 0) { cA = -cA; cB = -cB; }   // normal convention: make the closing speed positive
  cA = Math.max(0, cA); cB = Math.max(0, cB);
  const s = cA + cB;
  return s < 1e-3 ? 0.5 : cA / s;
}

/** Penalty (positive number; the reward is minus this) of one crash event for one car. */
export function crashPenalty({ dvKmh = 0, share = 0.5, disabled = false }, c = CRASH_SEV) {
  const sev = Math.max(0, dvKmh) * (1 + c.fault * (2 * share - 1));
  return Math.min(c.cap, c.base + (disabled ? c.fail : 0) + c.k * sev);
}

/** Steering-smoothness constants shared by the CPU driver, BC loss and the GPU reward (integrate.wgsl). */
export const JERK = { vRef: 20, speedCap: 4, ttcFree: 3, ttcOff: 1 };

/** Time to collision with the own-lane leader / obstacle (s; Infinity when not closing). */
export function leaderTTC(gap, leaderSpeed, speed) {
  const closing = speed - (leaderSpeed || 0);
  return gap < Infinity && closing > 0.1 ? Math.max(0, gap) / closing : Infinity;
}

/** Speed^2 scale of the jerk term, eased to 0 near an imminent collision (see header). */
export function jerkScale(speed, ttc, J = JERK) {
  const sv = Math.min(J.speedCap, (speed / J.vRef) * (speed / J.vRef));
  const ease = ttc >= J.ttcFree ? 1 : ttc <= J.ttcOff ? 0 : (ttc - J.ttcOff) / (J.ttcFree - J.ttcOff);
  return sv * ease;
}

export function annealProgress(arrivalRate, w = REWARD_WEIGHTS) {
  const r = arrivalRate < 0 ? 0 : arrivalRate > 1 ? 1 : arrivalRate;
  w.progress = w.progressFloor + (w.progressStart - w.progressFloor) * (1 - r);
  return w.progress;
}

export function reward(t, w = REWARD_WEIGHTS) {
  const dt = t.dt ?? 0.1;
  let r = w.progress * (t.progress || 0) * dt;
  if (w.crashSev) {   // rewardVersion 2: severity-scaled, charged once per crash event (t.crash from the sim)
    if (t.crash) r -= t.crash.penalty ?? crashPenalty(t.crash, w.crashSev);
    if (t.stranded) r += w.collision;
  } else if (t.collision || t.stranded) r += w.collision;
  if (w.steerJerk && t.steerJerk) r -= w.steerJerk * t.steerJerk * t.steerJerk * (t.jerkScale ?? 1);
  if (t.arrived) {
    r += w.arrived;
    const late = t.lateBy ?? 0;
    if (late <= 0) r += w.onTime;
    else r += Math.max(w.lateCap, (late / 60) * w.latePerMinute);
  }
  return r;
}

/** Trip-level fitness for evolve.js: the trip's reward sum (arrival, on-time, lateness,
 *  crash/stranded and the annealed progress term) -- nothing else, per SPEC. */
export function tripFitness({ rewardSum = 0 }) {
  return rewardSum;
}
