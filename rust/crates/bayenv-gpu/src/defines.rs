//! Kernel `const` prefixes (`GPUSim._defines()` + `physicsDefines()`), generated from the vehicle / surface tables
//! (`src/sim/physics/params.js`, `src/sim/vehicleTypes.js`, `src/sim/surface.js`). Checked against the JS strings.

/// Per type (sedan, suv, truck, bus), src/sim/physics/params.js VEHICLE_PARAMS.
pub struct VParams { pub length: f64, pub width: f64, pub l: f64, pub a: f64, pub b: f64, pub hcg: f64, pub m: f64, pub iz: f64, pub cda: f64, pub crr: f64,
    pub pmax: f64, pub f0: f64, pub drive: u32, pub vmax: f64, pub brake_decel: f64, pub brake_front: f64, pub cf: f64, pub cr: f64, pub mu: f64,
    pub max_steer: f64, pub max_steer_rate: f64, pub steer_slew: f64, pub steer_lag: f64, pub diesel: bool, pub bsfc_min: f64, pub idle_lph: f64, pub accessory_w: f64, pub tank_l: f64,
    // legacy fields (vehicleTypes.js LEGACY)
    pub max_accel_legacy: f64, pub max_decel_legacy: f64, pub fuel_base: f64, pub fuel_aero: f64, pub idle_lps: f64 }

pub const VEHICLE_PARAMS: [VParams; 4] = [
    VParams { length: 4.5, width: 1.85, l: 2.75, a: 1.10, b: 1.65, hcg: 0.55, m: 1500.0, iz: 2400.0, cda: 0.66, crr: 0.011, pmax: 100e3, f0: 5600.0, drive: 0, vmax: 55.0,
        brake_decel: 11.0, brake_front: 0.68, cf: 90e3, cr: 100e3, mu: 1.0, max_steer: 0.60, max_steer_rate: 0.70, steer_slew: 0.80, steer_lag: 0.10, diesel: false, bsfc_min: 265.0, idle_lph: 0.8, accessory_w: 1500.0, tank_l: 50.0,
        max_accel_legacy: 2.8, max_decel_legacy: 3.5, fuel_base: 0.050, fuel_aero: 5.3e-5, idle_lps: 0.00025 },
    VParams { length: 4.9, width: 1.95, l: 2.90, a: 1.30, b: 1.60, hcg: 0.70, m: 2100.0, iz: 4100.0, cda: 0.95, crr: 0.012, pmax: 150e3, f0: 8500.0, drive: 2, vmax: 50.0,
        brake_decel: 10.5, brake_front: 0.65, cf: 115e3, cr: 125e3, mu: 1.0, max_steer: 0.58, max_steer_rate: 0.65, steer_slew: 0.73, steer_lag: 0.10, diesel: false, bsfc_min: 250.0, idle_lph: 1.1, accessory_w: 1100.0, tank_l: 70.0,
        max_accel_legacy: 2.4, max_decel_legacy: 3.3, fuel_base: 0.065, fuel_aero: 7.0e-5, idle_lps: 0.00030 },
    VParams { length: 9.0, width: 2.5, l: 5.5, a: 2.4, b: 3.1, hcg: 1.25, m: 15000.0, iz: 1.1e5, cda: 6.0, crr: 0.0065, pmax: 260e3, f0: 55e3, drive: 1, vmax: 31.0,
        brake_decel: 6.5, brake_front: 0.45, cf: 2.6e5, cr: 5.2e5, mu: 0.8, max_steer: 0.50, max_steer_rate: 0.45, steer_slew: 0.53, steer_lag: 0.12, diesel: true, bsfc_min: 200.0, idle_lph: 2.5, accessory_w: 3000.0, tank_l: 300.0,
        max_accel_legacy: 1.0, max_decel_legacy: 2.0, fuel_base: 0.200, fuel_aero: 1.6e-4, idle_lps: 0.00080 },
    VParams { length: 12.0, width: 2.55, l: 6.0, a: 3.6, b: 2.4, hcg: 1.15, m: 13000.0, iz: 1.6e5, cda: 7.0, crr: 0.008, pmax: 200e3, f0: 48e3, drive: 1, vmax: 30.0,
        brake_decel: 6.5, brake_front: 0.45, cf: 2.4e5, cr: 4.4e5, mu: 0.8, max_steer: 0.50, max_steer_rate: 0.45, steer_slew: 0.53, steer_lag: 0.12, diesel: true, bsfc_min: 205.0, idle_lph: 2.2, accessory_w: 3000.0, tank_l: 250.0,
        max_accel_legacy: 1.2, max_decel_legacy: 2.2, fuel_base: 0.220, fuel_aero: 1.7e-4, idle_lps: 0.00080 },
];
pub const VEHICLE_TYPE_WEIGHTS: [f64; 4] = [0.62, 0.30, 0.06, 0.02];
impl VParams {
    /// vehicleTypes.js: emergencyDecel = min(brakeDecel, mu * 9.81)
    pub fn emergency_decel(&self) -> f64 { bayenv_world::jsmath::min(self.brake_decel, self.mu * 9.81) }
}
/// genome.js DEFAULT_TRAITS
pub mod default_traits { pub const AGGRESSION: f64 = 0.5; pub const PATIENCE: f64 = 0.5; pub const REACTION_TIME: f64 = 1.0; pub const HEADWAY: f64 = 1.5; pub const MAX_ACCEL: f64 = 2.5; pub const MAX_DECEL: f64 = 5.0; }

/// gpuSim.js driverLimits (physics-params schema): (maxAccel, maxDecel)
pub fn driver_limits(t: &VParams, max_accel: f64, max_decel: f64) -> (f64, f64) {
    use bayenv_world::jsmath::min;
    let emer = t.emergency_decel();
    (min(max_accel * min(1.0, 1500.0 / t.m * 1.6), 0.9 * t.f0 / t.m), min(max_decel * (emer / 8.0) * 1.6, emer))
}

// surface.js SURFACES: muPeakDry, muSlideDry, crr
const SURFACES: [(f64, f64, f64); 12] = [(0.85, 0.68, 0.013), (0.85, 0.68, 0.013), (0.75, 0.60, 0.015), (0.65, 0.52, 0.020), (0.60, 0.55, 0.025), (0.65, 0.58, 0.020),
    (0.68, 0.65, 0.050), (0.45, 0.40, 0.100), (0.40, 0.35, 0.300), (0.55, 0.45, 0.013), (0.60, 0.50, 0.015), (0.85, 0.68, 0.013)];
pub fn surf_tables() -> ([f64; 16], [f64; 16], [f64; 16]) {
    let (mut pk, mut sk, mut crr) = ([0.0; 16], [0.0; 16], [0.0; 16]);
    let r = SURFACES[0];
    let round2 = |x: f64| bayenv_world::jsmath::round(x * 100.0) / 100.0;
    for c in 0..16 { let s = if c < 12 { SURFACES[c] } else { r }; let p = s.0 / r.0; pk[c] = p; sk[c] = p * round2(s.1 / s.0); crr[c] = s.2 / r.2; }
    (pk, sk, crr)
}
pub const PAVED_SHOULDER_FREEWAY: f64 = 3.0;
pub const FUEL_DENSITY_GASOLINE: f64 = 745.0;
pub const FUEL_DENSITY_DIESEL: f64 = 832.0;

/// JS `Number(x).toExponential(p)`.
pub fn to_exponential(x: f64, p: usize) -> String {
    let s = format!("{:.*e}", p, x);
    match s.find('e') { Some(i) if !s[i + 1..].starts_with('-') => format!("{}e+{}", &s[..i], &s[i + 1..]), _ => s }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObsMode { V2, V3, None }
impl ObsMode {
    pub fn size(self) -> usize { match self { ObsMode::V3 => OBS3, _ => 70 } }
}
pub const OBS3: usize = 358;
pub const NB_OFF3: usize = 102;

/// The define list in JS insertion order.
pub fn defines(obs: ObsMode, cap: usize) -> Vec<(&'static str, String)> {
    let p = &VEHICLE_PARAMS;
    let arr6 = |f: &dyn Fn(&VParams) -> f64| format!("array<f32, 4>({})", (0..4).map(|k| to_exponential(f(&p[k]), 6)).collect::<Vec<_>>().join(", "));
    let arr7 = |f: &dyn Fn(&VParams) -> f64| format!("array<f32, 4>({})", (0..4).map(|k| to_exponential(f(&p[k]), 7)).collect::<Vec<_>>().join(", "));
    let sf = |a: &[f64; 16]| format!("array<f32, 16>({})", a.iter().map(|&x| to_exponential(x, 7)).collect::<Vec<_>>().join(", "));
    let (pk, sk, crr) = surf_tables();
    let insz = obs.size();
    let mut d: Vec<(&'static str, String)> = vec![
        ("VT_IDLE", arr6(&|t| t.idle_lps)), ("VT_FBASE", arr6(&|t| t.fuel_base)), ("VT_FAERO", arr6(&|t| t.fuel_aero)), ("VT_MASS", arr6(&|t| t.m)),
        ("INS", format!("{insz}u")), ("OFF_NB", "8u".into()), ("OFF_MAN", "20u".into()), ("OFF_HINT", "30u".into()), ("OFF_TRAIT", "33u".into()),
        ("OFF_LATE", "37u".into()), ("OFF_FUEL", "39u".into()), ("OFF_WHEEL", "41u".into()), ("OFF_LANEEND", "44u".into()), ("OFF_CTL", "46u".into()),
        ("OFF_KINK", "63u".into()), ("CTL_W", format!("{}u", crate::layout::CTL_WORDS)), ("KINKS_ON", "0u".into()), ("RANGE", "120.0".into()),
        ("PT_M", arr7(&|t| t.m)), ("PT_IZ", arr7(&|t| t.iz)), ("PT_A", arr7(&|t| t.a)), ("PT_B", arr7(&|t| t.b)), ("PT_L", arr7(&|t| t.l)), ("PT_HCG", arr7(&|t| t.hcg)),
        ("PT_MU", arr7(&|t| t.mu)), ("PT_CF", arr7(&|t| t.cf)), ("PT_CR", arr7(&|t| t.cr)), ("PT_F0", arr7(&|t| t.f0)), ("PT_PMAX", arr7(&|t| t.pmax)), ("PT_VMAX", arr7(&|t| t.vmax)),
        ("PT_BRAKE", arr7(&|t| t.brake_decel)), ("PT_BFRONT", arr7(&|t| t.brake_front)), ("PT_CDA", arr7(&|t| t.cda)), ("PT_CRR", arr7(&|t| t.crr)),
        ("PT_MAXSTEER", arr7(&|t| t.max_steer)), ("PT_MAXSTEERRATE", arr7(&|t| t.max_steer_rate)), ("PT_STEERLAG", arr7(&|t| t.steer_lag)), ("PT_STEERSLEW", arr7(&|t| t.steer_slew)),
        ("PT_BSFC", arr7(&|t| t.bsfc_min)), ("PT_IDLE", arr7(&|t| t.idle_lph / 3600.0)), ("PT_ACCW", arr7(&|t| t.accessory_w)),
        ("PT_RHOF", arr7(&|t| if t.diesel { FUEL_DENSITY_DIESEL } else { FUEL_DENSITY_GASOLINE })),
        ("SF_PK", sf(&pk)), ("SF_SK", sf(&sk)), ("SF_CRR", sf(&crr)), ("PAVED_SH_FW", to_exponential(PAVED_SHOULDER_FREEWAY, 7)),
        ("PT_DRIVE", format!("array<u32, 4>({})", (0..4).map(|k| format!("{}u", p[k].drive)).collect::<Vec<_>>().join(", "))),
        ("V3", format!("{}u", if obs == ObsMode::V3 { 1 } else { 0 })), ("LH_W", format!("{}u", crate::layout::LH_W)),
        ("LH_OFF", format!("{}u", if obs == ObsMode::V3 { cap * insz } else { 0 })),
    ];
    if obs == ObsMode::V3 { d.push(("NB_OFF3", format!("{NB_OFF3}u"))); }
    d
}
pub fn prefix(obs: ObsMode, cap: usize) -> String {
    defines(obs, cap).iter().map(|(k, v)| format!("const {k} = {v};")).collect::<Vec<_>>().join("\n") + "\n"
}
