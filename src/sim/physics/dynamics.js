// Vehicle dynamics: the ONE motion model of the project (SPEC "Vehicles and dynamics:
// full physics"). Used by the sim (vehicle.js integrate), the brain's training sim and
// mirrored by the GPU kernels. API and conventions: src/sim/physics/README.md.
//
// State (plain object, `createState`):
//   u        longitudinal velocity of the CG in the body frame (m/s, >= 0: no reverse gear)
//   v        lateral velocity of the CG in the body frame (m/s, + = LEFT)
//   r        yaw rate (rad/s, + = turning LEFT; the world heading atan2(dz,dx) changes by -r)
//   steer    front-wheel angle (rad, + = LEFT), clamped to maxSteer. mode 'rate' (legacy): integrated
//            from steerRate. mode 'position' (v9 default): the command is a target position in [-1,1];
//            sl (lagged target, rad) follows target*maxSteer with time constant steerLag, and the wheel
//            moves toward sl at most steerSlew rad/s: sl += (target - sl) * min(1, h / steerLag);
//            steer += clamp(sl - steer, -steerSlew h, +steerSlew h)
//   ax       longitudinal acceleration of the last sub-step (m/s^2, for weight transfer)
//   hr, lat, s   pose in the road (Frenet) frame: heading relative to the lane direction
//            (+ = left), metres from the carriageway's right boundary, metres along the edge.
//            The frame bends with the road polyline (road curvature is absorbed by the frame).
//   fuelL    litres burnt (accumulates), power  engine output (W) of the last sub-step
// Controls: {steer (position mode: target in [-1,1] x maxSteer) | steerRate (rate mode) in [-1,1], gas in [0,1], brake in [0,1]}  (already reaction-delayed)
// Road: {grade}  (dy/ds of the current segment; + = uphill)
//
// Per sub-step h (semi-implicit Euler, h = 10 ms by default):
//   steer += steerRate * maxSteerRate * h  (clamped)
//   normal loads with longitudinal weight transfer: Fzf = m g b/L - m ax hcg/L, Fzr = m g a/L + m ax hcg/L
//   traction (power-limited CVT model): Fd = gas * min(F0, Pmax / u) split by drive layout
//   brakes: Fb = brake * m * brakeDecel, front share brakeFront; each axle's longitudinal force
//           limited to 0.98 mu Fz (ABS); the lateral capacity is what the friction circle leaves:
//           Fy_max = sqrt((mu Fz)^2 - Fx^2)
//   slip angles: af = steer - atan((v + a r)/u), ar = -atan((v - b r)/u)   (u floored at 1 m/s)
//   tyre lateral force (saturating, brush-like): Fy = Fy_max * tanh(C alpha / Fy_max)
//   resistance: aero 0.5 rho CdA u|u| + rolling Crr m g + grade m g sin(theta)
//   du = (Fxf cos d - Fyf sin d + Fxr - resist)/m + v r
//   dv = (Fxf sin d + Fyf cos d + Fyr)/m - u r
//   dr = (a (Fxf sin d + Fyf cos d) - b Fyr) / Iz
//   below 5 km/h the lateral state blends toward the kinematic bicycle (r = u tan d / L, v = b r)
//   pose: hr += r h; lat += (u sin hr + v cos hr) h; s += (u cos hr - v sin hr) h
// Fuel: engine power = traction power / 0.88 + accessories; flow = max(idle, P * BSFC(load)),
// BSFC(load) = bsfcMin * (1 + 0.9 (1 - load)^3) (g/kWh), load = P / Pmax.

import { VEHICLE_PARAMS, RHO_AIR, G, FUEL_DENSITY } from './params.js';

export const PHYS_DT = 0.01;           // s, physics sub-step
export const KIN_BLEND_LO = 0.7, KIN_BLEND_HI = 1.4; // m/s (2.5 .. 5 km/h)
// brake system (2026-10-23): ABS longitudinal limit while braking (fraction of mu Fz; the rest of the friction
// circle stays available for cornering) and the EBD front margin over the load-proportional split. Before this the
// rear axle took a fixed 32 % of a pedal that asks for 11 m/s^2 > mu g: under hard braking the rear tyres used
// 98 % of their grip longitudinally and kept ~2 % for cornering, so braking hard in a gentle 400 m curve spun a sedan.
export const ABS_LONG = 0.95, EBD_MARGIN = 0.05;

export function createState(o = {}) {
  return { u: 0, v: 0, r: 0, steer: 0, sl: 0, ax: 0, hr: 0, lat: 0, s: 0, fuelL: 0, power: 0, ...o };
}

/** Max longitudinal acceleration available from the powertrain at speed u on a level road, before tyre limits. */
export function maxDriveAccel(P, u) {
  const Fd = Math.min(P.F0, P.Pmax / Math.max(u, 0.1));
  return (Fd - resistance(P, u, 0)) / P.m;
}
/** Road-load force (N) at speed u on grade (dy/ds). */
export function resistance(P, u, grade, crrMul = 1) {
  const roll = u > 0.05 ? P.Crr * crrMul * P.m * G : 0;
  return 0.5 * RHO_AIR * P.CdA * u * Math.abs(u) + roll + P.m * G * grade / Math.sqrt(1 + grade * grade);
}

/**
 * Pedals that realise a desired longitudinal acceleration (inverse of the longitudinal
 * model, ignoring tyre limits). Writes {gas, brake} into out.
 */
export function pedalsFor(P, accel, u, grade = 0, out = {}) {
  const need = P.m * accel + resistance(P, u, grade);
  if (need >= 0) { out.gas = Math.min(1, need / Math.min(P.F0, P.Pmax / Math.max(u, 0.1))); out.brake = 0; }
  else { out.gas = 0; out.brake = Math.min(1, -need / (P.m * P.brakeDecel)); }
  if (u < 0.05 && accel <= 0) { out.gas = 0; out.brake = Math.max(out.brake, 0.3); }
  return out;
}

/** Brake-specific fuel consumption (g/kWh) at load fraction in (0, 1]. */
export function bsfc(P, load) { const q = 1 - Math.min(1, Math.max(0, load)); return P.bsfcMin * (1 + 0.9 * q * q * q); }

/** Fuel flow (L/s) at engine output power Pw (W). */
export function fuelFlow(P, Pw) {
  const idle = P.idleLph / 3600;
  if (Pw <= 0) return idle;
  const gps = (Pw / 1000) * bsfc(P, Pw / P.Pmax) / 3600;
  return Math.max(idle, gps / FUEL_DENSITY[P.fuel]);
}

/**
 * Advance state `st` by `dt` seconds of physics under controls (steerRate, gas, brake),
 * in sub-steps of at most PHYS_DT. `P` = VEHICLE_PARAMS entry (or typeIdx). `grade` = road slope.
 * `locked`: wheels locked / car disabled (no traction, full brake without ABS, no steering).
 * `surf`: road-surface factors under the wheels {pk, sk, crr} (src/sim/surface.js carSurface): peak friction
 * mu * pk, locked-wheel sliding friction mu * sk, rolling resistance Crr * crr. Default = dry asphalt {1, 0.8, 1}.
 */
export function stepDynamics(st, steerRate, gas, brake, P, dt, grade = 0, locked = false, h0 = PHYS_DT, mode = 'rate', surf = null) {
  if (typeof P === 'number') P = VEHICLE_PARAMS[P];
  const n = Math.max(1, Math.ceil(dt / h0 - 1e-9)), h = dt / n;
  const { m, Iz, a, b, L, hcg, Cf, Cr } = P;
  const mu = surf ? P.mu * surf.pk : P.mu, kSlide = surf ? surf.sk / surf.pk : 0.8, crrMul = surf ? surf.crr : 1;
  const sr = steerRate < -1 ? -1 : steerRate > 1 ? 1 : steerRate || 0;
  const g_ = gas < 0 ? 0 : gas > 1 ? 1 : gas || 0, b_ = brake < 0 ? 0 : brake > 1 ? 1 : brake || 0;
  const cosG = 1 / Math.sqrt(1 + grade * grade), sinG = grade * cosG;
  const Wg = P.m * G * sinG;           // gravity along the road (+ = pulls back, uphill)
  let { u, v, r, steer, ax, hr, lat, s } = st;
  const pos = mode === 'position', tgt = sr * P.maxSteer, kLag = Math.min(1, (dt / n) / (P.steerLag || 0.1)), slew = (P.steerSlew || P.maxSteerRate) * (dt / n);
  let sl = st.sl === undefined ? steer : st.sl;
  let fuel = 0, power = 0;
  for (let k = 0; k < n; k++) {
    if (!locked) {
      if (pos) { sl += (tgt - sl) * kLag; const d = sl - steer; steer += d > slew ? slew : d < -slew ? -slew : d; }
      else { steer += sr * P.maxSteerRate * h; sl = steer; }
      if (steer > P.maxSteer) steer = P.maxSteer; else if (steer < -P.maxSteer) steer = -P.maxSteer;
    }
    // normal loads (longitudinal weight transfer), floored at 10 % of static
    // (grade: the slope shifts load to the downhill axle, m g sin(theta) hcg / L, 2026-10-24 real terrain)
    let Fzf = m * G * cosG * b / L - (m * ax + Wg) * hcg / L, Fzr = m * G * cosG * a / L + (m * ax + Wg) * hcg / L;
    const fl = 0.1 * m * G;
    if (Fzf < fl) Fzf = fl; if (Fzr < fl) Fzr = fl;
    // demanded longitudinal tyre forces
    let Fd = locked || u >= P.vmax ? 0 : g_ * Math.min(P.F0, P.Pmax / Math.max(u, 0.1));
    const moving = u > 0.02;
    const Fb = (locked ? 1 : b_) * m * P.brakeDecel;
    let Fxf, Fxr;
    if (P.drive === 'front') { Fxf = Fd; Fxr = 0; } else if (P.drive === 'rear') { Fxf = 0; Fxr = Fd; } else { Fxf = Fd * Fzf / (Fzf + Fzr); Fxr = Fd - Fxf; }
    // brake proportioning (EBD, every car since the 2000s): the front share follows the axle loads (dynamic weight
    // transfer) plus a small front margin, never less than the fixed hydraulic split, so the rear axle is never
    // braked harder than the front relative to its grip (front-first saturation = stable understeer)
    let shareF = P.brakeFront;
    if (!locked) { const ideal = Fzf / (Fzf + Fzr) + EBD_MARGIN; if (ideal > shareF) shareF = ideal > 0.95 ? 0.95 : ideal; }
    const bf = Fb * shareF, br = Fb - bf;
    // brakes oppose motion; at standstill they only hold (cannot push backwards)
    if (moving) { Fxf -= bf; Fxr -= br; }
    // at standstill the brakes hold against whatever pushes the car forward: the drive force and, on a downhill,
    // gravity (before 2026-10-24 only the drive force, so a car braking at a downhill stop line crept forward)
    else { const hold = Math.min(Fb, Math.max(0, Fd - Wg)), hf = Math.min(bf, hold * shareF); Fxf -= hf; Fxr -= Math.min(br, hold - hf); }
    // friction circle per axle (ABS: 0.98 mu Fz; locked wheels: sliding, all capacity used longitudinally)
    const Ff = mu * Fzf, Fr = mu * Fzr;
    // driving: traction control at 0.98 mu Fz; braking: ABS holds the tyre near its peak slip, where it still has
    // about a third of its side grip (combined slip), so a car braking hard in a curve can still steer
    const xf = locked ? 1.0 : 0.98, xr = xf, xb = locked ? 1.0 : ABS_LONG;
    if (Fxf > xf * Ff) Fxf = xf * Ff; else if (Fxf < -xb * Ff) Fxf = -xb * Ff;
    if (Fxr > xr * Fr) Fxr = xr * Fr; else if (Fxr < -xb * Fr) Fxr = -xb * Fr;
    // stability control on the rear axle (EBD cornering brake control / ESC): while braking, the rear brakes only
    // take what the friction circle leaves after the side force the rear tyres need at their current slip angle,
    // so over-the-limit braking in a curve ends in front-axle understeer, not a spin
    if (!locked && Fxr < 0) {
      const uf0 = u > 1 ? u : 1, ar0 = -Math.atan((v - b * r) / uf0), lamR = u >= KIN_BLEND_HI ? 1 : u <= KIN_BLEND_LO ? 0 : (u - KIN_BLEND_LO) / (KIN_BLEND_HI - KIN_BLEND_LO);
      const yr0 = lamR * Fr * Math.tanh(Cr * ar0 / Fr), lim = Math.sqrt(Math.max(0, xb * xb * Fr * Fr - yr0 * yr0));
      if (Fxr < -lim) Fxr = -lim;
    }
    let Fyf = 0, Fyr = 0;
    const uf = u > 1 ? u : 1;
    const yfMax = Math.sqrt(Math.max(Ff * Ff - Fxf * Fxf, 0)) + (locked ? 0 : 1e-3), yrMax = Math.sqrt(Math.max(Fr * Fr - Fxr * Fxr, 0)) + (locked ? 0 : 1e-3);
    if (locked) {
      // sliding tyres: friction opposes the sliding velocity of each axle (kinetic mu = 0.8 x peak on asphalt; surf.sk / surf.pk on other surfaces)
      const vfx = u, vfy = v + a * r, vrx = u, vry = v - b * r;
      const sf = Math.hypot(vfx, vfy), sr2 = Math.hypot(vrx, vry), k = kSlide;
      // at rest the locked tyres hold by static friction against gravity on a downhill
      const pushF = Math.max(0, -Wg) * Fzf / (Fzf + Fzr), pushR = Math.max(0, -Wg) - pushF;
      if (sf > 0.01) { Fxf = -k * Ff * vfx / sf; Fyf = -k * Ff * vfy / sf; } else { Fxf = -Math.min(Ff, pushF); Fyf = 0; }
      if (sr2 > 0.01) { Fxr = -k * Fr * vrx / sr2; Fyr = -k * Fr * vry / sr2; } else { Fxr = -Math.min(Fr, pushR); Fyr = 0; }
    } else {
      const af = steer - Math.atan((v + a * r) / uf), ar = -Math.atan((v - b * r) / uf);
      // below the blend speed slip angles are meaningless (the kinematic model takes over the
      // lateral state); fade the tyre side forces out so they cannot hold a car at standstill
      const lamF = u >= KIN_BLEND_HI ? 1 : u <= KIN_BLEND_LO ? 0 : (u - KIN_BLEND_LO) / (KIN_BLEND_HI - KIN_BLEND_LO);
      Fyf = lamF * yfMax * Math.tanh(Cf * af / yfMax);
      Fyr = lamF * yrMax * Math.tanh(Cr * ar / yrMax);
    }
    const res = resistance(P, u, grade, crrMul);
    const cd = Math.cos(steer), sd = Math.sin(steer);
    const FxBody = Fxf * cd - Fyf * sd + Fxr - res;
    const FyBody = Fxf * sd + Fyf * cd + Fyr;
    let du = (FxBody / m + v * r) * h;
    let un = u + du;
    if (un < 0) { un = 0; } // no reverse: brakes / resistance stop the car
    let vn = v + (FyBody / m - u * r) * h;
    let rn = r + ((a * (Fxf * sd + Fyf * cd) - b * Fyr) / Iz) * h;
    // low-speed blend toward the kinematic bicycle (numerically stable at walking pace)
    if (un < KIN_BLEND_HI && !locked) {
      const lam = un <= KIN_BLEND_LO ? 0 : (un - KIN_BLEND_LO) / (KIN_BLEND_HI - KIN_BLEND_LO);
      const rk = un * Math.tan(steer) / L, vk = rk * b;
      rn = lam * rn + (1 - lam) * rk; vn = lam * vn + (1 - lam) * vk;
    }
    if (locked && un < 0.05 && Math.abs(vn) < 0.05) { un = 0; vn = 0; rn *= 0.5; if (Math.abs(rn) < 0.01) rn = 0; }
    ax = (un - u) / h;
    u = un; v = vn; r = rn;
    hr += r * h;
    if (hr > Math.PI) hr -= 2 * Math.PI; else if (hr < -Math.PI) hr += 2 * Math.PI;
    const ch = Math.cos(hr), sh = Math.sin(hr);
    lat += (u * sh + v * ch) * h;
    s += (u * ch - v * sh) * h * cosG;   // s is horizontal arc length (the road data's metres); u runs along the slope
    // engine power and fuel
    const tract = Math.max(0, (P.drive === 'front' ? Math.max(0, Fxf) : P.drive === 'rear' ? Math.max(0, Fxr) : Math.max(0, Fxf) + Math.max(0, Fxr))) * u;
    power = locked ? 0 : (g_ > 0 ? tract / 0.88 : 0) + P.accessoryW;
    fuel += (locked ? 0 : fuelFlow(P, power)) * h;
  }
  st.u = u; st.v = v; st.r = r; st.steer = steer; st.sl = sl; st.ax = ax; st.hr = hr; st.lat = lat; st.s = s;
  st.fuelL += fuel; st.power = power;
  return st;
}

export { VEHICLE_PARAMS };
