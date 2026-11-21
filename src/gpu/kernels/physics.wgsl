// src/sim/physics/dynamics.js stepDynamics, 1:1 (layout v4). Included by integrate.wgsl.
// Per-type constants (PT_*) are injected by gpuSim.js from src/sim/physics/params.js
// VEHICLE_PARAMS: m, Iz, a, b, L, hcg, mu, Cf, Cr, F0, Pmax, vmax, brakeDecel, brakeFront, CdA, Crr,
// maxSteer, maxSteerRate, drive (0 front, 1 rear, 2 all), bsfcMin, idle (L/s), accessoryW, fuel density (g/L).

const G_ACC: f32 = 9.81;
const RHO_AIR: f32 = 1.225;
const KIN_BLEND_LO: f32 = 0.7;
const KIN_BLEND_HI: f32 = 1.4;
const ABS_LONG: f32 = 0.95;    // dynamics.js ABS_LONG / EBD_MARGIN (brake system, 2026-10-23)
const EBD_MARGIN: f32 = 0.05;

// sl: lagged steering target (rad, layout v9 position mode); P.steerMode bit 0 = position mode (command = target
// wheel position in [-1,1]: sl += (target x maxSteer - sl) min(1, h / steerLag), the wheel moves <= steerSlew rad/s)
struct DynState { u: f32, v: f32, r: f32, steer: f32, ax: f32, hr: f32, lat: f32, s: f32, fuelL: f32, power: f32, sl: f32 };

fn resistanceF(t: u32, u: f32, grade: f32) -> f32 {
  var roll = 0.0; if (u > 0.05) { roll = PT_CRR[t] * PT_M[t] * G_ACC; }
  return 0.5 * RHO_AIR * PT_CDA[t] * u * abs(u) + roll + PT_M[t] * G_ACC * grade / sqrt(1.0 + grade * grade);
}
fn bsfcF(t: u32, load: f32) -> f32 { let q = 1.0 - clamp(load, 0.0, 1.0); return PT_BSFC[t] * (1.0 + 0.9 * q * q * q); }
fn fuelFlowF(t: u32, Pw: f32) -> f32 {
  let idle = PT_IDLE[t];
  if (Pw <= 0.0) { return idle; }
  let gps = (Pw / 1000.0) * bsfcF(t, Pw / PT_PMAX[t]) / 3600.0;
  return max(idle, gps / PT_RHOF[t]);
}
/** desired acceleration -> {gas, brake} (dynamics.js pedalsFor) */
fn pedalsForP(t: u32, accel: f32, u: f32, grade: f32) -> vec2f {
  let need = PT_M[t] * accel + resistanceF(t, u, grade);
  var o = vec2f(0.0, 0.0);
  if (need >= 0.0) { o.x = min(1.0, need / min(PT_F0[t], PT_PMAX[t] / max(u, 0.1))); }
  else { o.y = min(1.0, -need / (PT_M[t] * PT_BRAKE[t])); }
  if (u < 0.05 && accel <= 0.0) { o.x = 0.0; o.y = max(o.y, 0.3); }
  return o;
}

fn stepDynamics(st0: DynState, steerRate: f32, gas: f32, brake: f32, t: u32, dt: f32, grade: f32, locked: bool) -> DynState {
  return stepDynamicsH(st0, steerRate, gas, brake, t, dt, grade, locked, P.physDt);
}
// h0: max sub-step (dynamics.js stepDynamics h0; the drivers' predictor uses 50 ms)
fn stepDynamicsH(st0: DynState, steerRate: f32, gas: f32, brake: f32, t: u32, dt: f32, grade: f32, locked: bool, h0: f32) -> DynState {
  var st = st0;
  let n = max(1u, u32(ceil(dt / h0 - 1e-6)));
  let h = dt / f32(n);
  let m = PT_M[t]; let Iz = PT_IZ[t]; let a = PT_A[t]; let b = PT_B[t]; let L = PT_L[t]; let hcg = PT_HCG[t];
  let mu = PT_MU[t]; let Cf = PT_CF[t]; let Cr = PT_CR[t]; let drive = PT_DRIVE[t];
  let sr = clamp(steerRate, -1.0, 1.0);
  let g_ = clamp(gas, 0.0, 1.0); let b_ = clamp(brake, 0.0, 1.0);
  let cosG = 1.0 / sqrt(1.0 + grade * grade);
  let Wg = PT_M[t] * G_ACC * grade * cosG;   // dynamics.js: gravity along the road (+ = uphill)
  var u = st.u; var v = st.v; var r = st.r; var steer = st.steer; var ax = st.ax; var hr = st.hr; var lat = st.lat; var s = st.s;
  var fuel = 0.0; var power = 0.0;
  let posM = (P.steerMode & 1u) != 0u; let tgt = sr * PT_MAXSTEER[t]; let kLag = min(1.0, h / PT_STEERLAG[t]); let slew = PT_STEERSLEW[t] * h;
  var sl = st.sl;
  for (var k = 0u; k < n; k = k + 1u) {
    if (!locked) {
      if (posM) { sl = sl + (tgt - sl) * kLag; steer = steer + clamp(sl - steer, -slew, slew); }
      else { steer = steer + sr * PT_MAXSTEERRATE[t] * h; sl = steer; }
      steer = clamp(steer, -PT_MAXSTEER[t], PT_MAXSTEER[t]);
    }
    var Fzf = m * G_ACC * cosG * b / L - (m * ax + Wg) * hcg / L;
    var Fzr = m * G_ACC * cosG * a / L + (m * ax + Wg) * hcg / L;
    let fl = 0.1 * m * G_ACC;
    Fzf = max(Fzf, fl); Fzr = max(Fzr, fl);
    var Fd = g_ * min(PT_F0[t], PT_PMAX[t] / max(u, 0.1));
    if (locked || u >= PT_VMAX[t]) { Fd = 0.0; }
    let moving = u > 0.02;
    var bk = b_; if (locked) { bk = 1.0; }
    let Fb = bk * m * PT_BRAKE[t];
    var Fxf = 0.0; var Fxr = 0.0;
    if (drive == 0u) { Fxf = Fd; } else if (drive == 1u) { Fxr = Fd; } else { Fxf = Fd * Fzf / (Fzf + Fzr); Fxr = Fd - Fxf; }
    // dynamics.js (2026-10-23): EBD - front share = max(fixed split, load-proportional + EBD_MARGIN), <= 0.95
    var shareF = PT_BFRONT[t];
    if (!locked) { let ideal = Fzf / (Fzf + Fzr) + EBD_MARGIN; if (ideal > shareF) { shareF = min(ideal, 0.95); } }
    let bf = Fb * shareF; let br = Fb - bf;
    if (moving) { Fxf = Fxf - bf; Fxr = Fxr - br; }
    else { let hold = min(Fb, max(0.0, Fd - Wg)); let hf = min(bf, hold * shareF); Fxf = Fxf - hf; Fxr = Fxr - min(br, hold - hf); }
    let Ff = mu * Fzf; let Fr = mu * Fzr;
    var xf = 0.98; var xb = ABS_LONG; if (locked) { xf = 1.0; xb = 1.0; }
    Fxf = clamp(Fxf, -xb * Ff, xf * Ff); Fxr = clamp(Fxr, -xb * Fr, xf * Fr);
    // dynamics.js: rear-axle stability control while braking (rear brakes keep the side force the rear tyres need)
    if (!locked && Fxr < 0.0) {
      let ar0 = -atan((v - b * r) / max(u, 1.0));
      var lamR = 1.0; if (u < KIN_BLEND_HI) { lamR = select(0.0, (u - KIN_BLEND_LO) / (KIN_BLEND_HI - KIN_BLEND_LO), u > KIN_BLEND_LO); }
      let yr0 = lamR * Fr * tanh(clamp(Cr * ar0 / Fr, -15.0, 15.0));
      let lim = sqrt(max(0.0, xb * xb * Fr * Fr - yr0 * yr0));
      Fxr = max(Fxr, -lim);
    }
    var Fyf = 0.0; var Fyr = 0.0;
    let uf = max(u, 1.0);
    var eps = 1e-3; if (locked) { eps = 0.0; }
    let yfMax = sqrt(max(Ff * Ff - Fxf * Fxf, 0.0)) + eps; let yrMax = sqrt(max(Fr * Fr - Fxr * Fxr, 0.0)) + eps;
    if (locked) {
      let vfy = v + a * r; let vry = v - b * r;
      let sf = sqrt(u * u + vfy * vfy); let s2 = sqrt(u * u + vry * vry);
      let pushF = max(0.0, -Wg) * Fzf / (Fzf + Fzr); let pushR = max(0.0, -Wg) - pushF;
      if (sf > 0.01) { Fxf = -0.8 * Ff * u / sf; Fyf = -0.8 * Ff * vfy / sf; } else { Fxf = -min(Ff, pushF); Fyf = 0.0; }
      if (s2 > 0.01) { Fxr = -0.8 * Fr * u / s2; Fyr = -0.8 * Fr * vry / s2; } else { Fxr = -min(Fr, pushR); Fyr = 0.0; }
    } else {
      let af = steer - atan((v + a * r) / uf); let ar = -atan((v - b * r) / uf);
      // below the blend speed the tyre side forces fade out (dynamics.js lamF)
      var lamF = 1.0; if (u < KIN_BLEND_HI) { lamF = select(0.0, (u - KIN_BLEND_LO) / (KIN_BLEND_HI - KIN_BLEND_LO), u > KIN_BLEND_LO); }
      Fyf = lamF * yfMax * tanh(clamp(Cf * af / yfMax, -15.0, 15.0));
      Fyr = lamF * yrMax * tanh(clamp(Cr * ar / yrMax, -15.0, 15.0));
    }
    let res = resistanceF(t, u, grade);
    let cd = cos(steer); let sd = sin(steer);
    let FxBody = Fxf * cd - Fyf * sd + Fxr - res;
    let FyBody = Fxf * sd + Fyf * cd + Fyr;
    var un = u + (FxBody / m + v * r) * h;
    if (un < 0.0) { un = 0.0; }
    var vn = v + (FyBody / m - u * r) * h;
    var rn = r + ((a * (Fxf * sd + Fyf * cd) - b * Fyr) / Iz) * h;
    if (un < KIN_BLEND_HI && !locked) {
      var lam = 0.0; if (un > KIN_BLEND_LO) { lam = (un - KIN_BLEND_LO) / (KIN_BLEND_HI - KIN_BLEND_LO); }
      let rk = un * tan(steer) / L; let vk = rk * b;
      rn = lam * rn + (1.0 - lam) * rk; vn = lam * vn + (1.0 - lam) * vk;
    }
    if (locked && un < 0.05 && abs(vn) < 0.05) { un = 0.0; vn = 0.0; rn = rn * 0.5; if (abs(rn) < 0.01) { rn = 0.0; } }
    ax = (un - u) / h;
    u = un; v = vn; r = rn;
    hr = hr + r * h;
    if (hr > 3.14159265) { hr = hr - 6.2831853; } else if (hr < -3.14159265) { hr = hr + 6.2831853; }
    let ch = cos(hr); let sh = sin(hr);
    lat = lat + (u * sh + v * ch) * h;
    s = s + (u * ch - v * sh) * h * cosG;
    var tract = 0.0;
    if (drive == 0u) { tract = max(0.0, Fxf); } else if (drive == 1u) { tract = max(0.0, Fxr); } else { tract = max(0.0, Fxf) + max(0.0, Fxr); }
    tract = tract * u;
    power = 0.0;
    if (!locked) { power = select(0.0, tract / 0.88, g_ > 0.0) + PT_ACCW[t]; fuel = fuel + fuelFlowF(t, power) * h; }
  }
  st.sl = sl; st.u = u; st.v = v; st.r = r; st.steer = steer; st.ax = ax; st.hr = hr; st.lat = lat; st.s = s;
  st.fuelL = fuel; st.power = power;
  return st;
}
