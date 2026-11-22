//! Host mirror of the WGSL struct layouts (`src/gpu/layout.js`, `kernels/common.wgsl`).
pub const VEH_FIELDS: [&str; 40] = ["edge", "s", "speed", "accel", "lat", "headingRel", "steerAngle", "state", "routePos", "seg", "flags", "cmdStep",
    "stoppedSince", "odometer", "lanes", "lastDecision", "fuel", "tank", "spawnTime", "routeDist", "leader", "gap", "eventTime", "stopPos",
    "refuelState", "steerSl", "gas", "brake", "routeTotal", "lastFuelCheck", "taper", "laneEndDist", "laneEndSide", "vLat", "yawRate", "axPrev",
    "px", "pz", "pyaw", "power"];
/// 'u' u32, 'f' f32, 'i' i32 per VEH field
pub const VEH_TYPES: &[u8; 40] = b"uffffffuuuuiffufffffiffiufffffffifffffff";
pub const PAR_FIELDS: [&str; 20] = ["routeOff", "routeLen", "typeIdx", "delaySteps", "length", "width", "maxSpeed", "maxAccel", "maxDecel", "headway",
    "aggression", "patience", "reactionTime", "workStart", "routeLength", "trip", "wheelbase", "maxSteer", "maxSteerRate", "typeMaxSpeed"];
pub const PAR_TYPES: &[u8; 20] = b"uuuufffffffffffuffff";
pub const VEH_WORDS: usize = 40;
pub const PAR_WORDS: usize = 20;
pub const SPAWN_WORDS: usize = 28;
pub const BLOCK_WORDS: usize = 8;
pub const CTRL_RING: usize = 40;
pub const CTRL_STRIDE: usize = 3;
pub const MAX_OCC: usize = 4;
pub const CTL_WORDS: usize = 36;
pub const LH_W: usize = 48;

pub mod veh {
    pub const EDGE: usize = 0; pub const S: usize = 1; pub const SPEED: usize = 2; pub const ACCEL: usize = 3; pub const LAT: usize = 4;
    pub const HEADING_REL: usize = 5; pub const STEER_ANGLE: usize = 6; pub const STATE: usize = 7; pub const ROUTE_POS: usize = 8; pub const SEG: usize = 9;
    pub const FLAGS: usize = 10; pub const CMD_STEP: usize = 11; pub const STOPPED_SINCE: usize = 12; pub const ODOMETER: usize = 13; pub const LANES: usize = 14;
    pub const LAST_DECISION: usize = 15; pub const FUEL: usize = 16; pub const TANK: usize = 17; pub const SPAWN_TIME: usize = 18; pub const ROUTE_DIST: usize = 19;
    pub const LEADER: usize = 20; pub const GAP: usize = 21; pub const EVENT_TIME: usize = 22; pub const STOP_POS: usize = 23; pub const REFUEL_STATE: usize = 24;
    pub const STEER_SL: usize = 25; pub const GAS: usize = 26; pub const BRAKE: usize = 27; pub const ROUTE_TOTAL: usize = 28; pub const LAST_FUEL_CHECK: usize = 29;
    pub const TAPER: usize = 30; pub const LANE_END_DIST: usize = 31; pub const LANE_END_SIDE: usize = 32; pub const V_LAT: usize = 33; pub const YAW_RATE: usize = 34;
    pub const AX_PREV: usize = 35; pub const PX: usize = 36; pub const PZ: usize = 37; pub const PYAW: usize = 38; pub const POWER: usize = 39;
}

pub fn pack_lanes(lane: u32, lo: u32, hi: u32, target: i32) -> u32 { (lane & 255) | ((lo & 255) << 8) | ((hi & 255) << 16) | ((((target + 1) as u32) & 255) << 24) }

pub mod state { pub const FREE: u32 = 0; pub const DRIVING: u32 = 1; pub const CRASHED: u32 = 2; pub const REFUELING: u32 = 5; }
pub mod flag {
    pub const LEARNER: u32 = 1; pub const COLLIDED: u32 = 2; pub const POLICY: u32 = 4; pub const SHOULDER: u32 = 8; pub const SPEEDING: u32 = 16;
    pub const FUELREQ: u32 = 32; pub const STRANDED: u32 = 64; pub const REFUEL: u32 = 128; pub const ZIPLEAD: u32 = 256; pub const SCORED: u32 = 512;
    pub const DISABLED: u32 = 1024; pub const WALL: u32 = 2048; pub const WALL_PREV: u32 = 4096; pub const NEW: u32 = 8192;
    /// GPU only (GpuOptions::guide_stop_spawn): the refuel route came from a plan requested while driving, so the v2 guidance shows its stop
    pub const GUIDE_STOP: u32 = 16384;
}
pub mod feat { pub const LANECHANGE: u32 = 1; pub const ACCIDENTS: u32 = 2; pub const FUEL: u32 = 4; pub const COLLISIONS: u32 = 8; pub const YIELD: u32 = 16; pub const POLICY: u32 = 32; pub const ZIPFIX: u32 = 64; pub const EMPTY: u32 = 128; }
pub mod ev {
    pub const ARRIVE: u32 = 1; pub const CRASH: u32 = 2; pub const STUCK: u32 = 3; pub const STRANDED: u32 = 4; pub const TOWED: u32 = 5; pub const LOWFUEL: u32 = 6;
    pub const SPAWN: u32 = 7; pub const STATION: u32 = 8; pub const ACCIDENT: u32 = 9; pub const WRECK: u32 = 10; pub const OFFROAD: u32 = 11; pub const REENTER: u32 = 12;
    pub const CONTACT: u32 = 13; pub const CRASHX_V: u32 = 14; pub const CRASHX_P: u32 = 15; pub const CRASHX_H: u32 = 16; pub const TRIPX: u32 = 17; pub const CTLVIOL: u32 = 18;
    pub const CRASHX_N: u32 = 19; pub const CRASHX_M: u32 = 20; pub const REROUTE: u32 = 21; pub const MISSED: u32 = 22; pub const CRASHPEN: u32 = 23;
}
pub mod op {
    pub const FREE: u32 = 1; pub const SET_STATE: u32 = 2; pub const SET_FLAGS: u32 = 3; pub const CLEAR_FLAGS: u32 = 4; pub const SET_FUEL: u32 = 7; pub const SET_ROUTE_REFUEL: u32 = 9;
    pub const PUMPED: u32 = 10; pub const FUEL_PLAN_FAILED: u32 = 11; pub const TOW_SHOULDER: u32 = 12; pub const SET_PULL: u32 = 13; pub const SET_ROUTE: u32 = 14;
}
pub mod io { pub const STEP: usize = 0; pub const STATS: usize = 8; pub const EVCOUNT: usize = 72; pub const EVDATA: usize = 76; }
pub mod stat {
    pub const ARRIVED: usize = 0; pub const ARRIVED_OUT: usize = 1; pub const ONTIME: usize = 2; pub const CRASHED_VEH: usize = 3; pub const CRASH_EVENTS: usize = 4;
    pub const STUCK: usize = 5; pub const STRANDED: usize = 6; pub const TOWED: usize = 7; pub const SPAWNED: usize = 8; pub const LANECHANGES: usize = 9;
    pub const REFUELED: usize = 10; pub const LOWFUEL: usize = 11; pub const VEHM_LO: usize = 12; pub const SPEEDM_LO: usize = 14; pub const VEHSTEPS_LO: usize = 16;
    pub const SPEEDSTEPS_LO: usize = 18; pub const LATESUM_LO: usize = 20; pub const ACCIDENTS: usize = 22; pub const BUMPS: usize = 23;
    pub const LIVE_DRIVING: usize = 32; pub const LIVE_CRASHED: usize = 33; pub const LIVE_STRANDED: usize = 34; pub const LIVE_SUMSPEED: usize = 35; pub const LIVE_DWELL: usize = 36; pub const LIVE_POLICY: usize = 37;
}

/// `PARAM_FIELDS` (uniform `Params`, 128 words): name, 'u' / 'f'.
pub const PARAM_FIELDS: [(&str, u8); 128] = [
    ("cap", b'u'), ("E", b'u'), ("N", b'u'), ("K", b'u'), ("T", b'u'), ("wBits", b'u'), ("wMask", b'u'), ("hMask", b'u'),
    ("ptsOff", b'u'), ("laneOffOff", b'u'), ("adjOff", b'u'), ("ebOff", b'u'), ("dt", b'f'), ("time0", b'f'), ("step0", b'u'), ("features", b'u'),
    ("cellSize", b'f'), ("leaderLookahead", b'f'), ("junctionMargin", b'f'), ("collisionInterval", b'u'), ("s0", b'f'), ("collideShrink", b'f'), ("speedFactorMin", b'f'), ("speedFactorSpan", b'f'),
    ("laneCheckInterval", b'f'), ("yieldLook", b'f'), ("stopMargin", b'f'), ("stuckTimeout", b'f'), ("spawnBlockedTimeout", b'f'), ("spawnCount", b'u'), ("opCount", b'u'), ("decideEvery", b'u'),
    ("accRate", b'f'), ("accSpeedRef", b'f'), ("accCloseGap", b'f'), ("accMergeFactor", b'f'), ("accMergeDist", b'f'), ("rubberDist", b'f'), ("rubberFactor", b'f'), ("towTime", b'f'),
    ("lowFuelFrac", b'f'), ("seed", b'u'), ("evCap", b'u'), ("inSize", b'u'), ("qrWindow", b'f'), ("qrDist", b'f'), ("qrFactor", b'f'), ("nBlocks", b'u'),
    ("blkOff", b'u'), ("maxCellReach", b'f'), ("fuelScale", b'f'), ("anticipation", b'f'), ("mergeLookahead", b'f'), ("collisionTol", b'f'), ("contactSpeed", b'f'), ("lateralTol", b'f'),
    ("offroadMargin", b'f'), ("squeezeSpeed", b'f'), ("prepBase", b'f'), ("prepPerLane", b'f'), ("fuelCheckInterval", b'f'), ("fuelMinFillToSkip", b'f'), ("fuelLowFraction", b'f'), ("fuelPlanFactor", b'f'),
    ("fuelReserveKm", b'f'), ("fuelPlanSpeed", b'f'), ("progressW", b'f'), ("nAgents", b'u'), ("geoOff", b'u'), ("lgvOff", b'u'), ("hopOff", b'u'), ("hasLG", b'u'),
    ("layerGap", b'f'), ("pairCooldown", b'f'), ("touchKmh", b'f'), ("shoulder", b'f'), ("oppositeMargin", b'f'), ("laneEndLookahead", b'f'), ("contactCap", b'u'), ("physDt", b'f'),
    ("ctlEdgeOff", b'u'), ("ctlApOff", b'u'), ("ctlCOff", b'u'), ("hasCtl", b'u'), ("ctlLook", b'f'), ("rampStart", b'f'), ("rampEnd", b'f'), ("freewayShoulder", b'f'),
    ("pairOff", b'u'), ("pairN", b'u'), ("labelMode", b'u'), ("ctlListOff", b'u'), ("jMovOff", b'u'), ("jEdgeMovOff", b'u'), ("jPtsOff", b'u'), ("jConfOff", b'u'),
    ("jMouthOff", b'u'), ("hasJ", b'u'), ("jM", b'u'), ("jHeadOff", b'u'), ("rerouteSight", b'f'), ("passWrecks", b'u'), ("trafficOn", b'u'), ("steerMode", b'u'),
    ("rlOn", b'u'), ("rlRec", b'f'), ("rlSeed", b'u'), ("rlChunk", b'u'), ("rlStd0", b'f'), ("rlStd1", b'f'), ("rlStd2", b'f'), ("rlRowCap", b'u'),
    ("jerkW", b'f'), ("crashBase", b'f'), ("crashK", b'f'), ("crashCap", b'f'), ("crashFault", b'f'), ("crashOn", b'u'), ("crashFail", b'f'), ("padC3", b'u'),
    ("jIntoOff", b'u'), ("jSight", b'u'), ("pathSight", b'u'), ("dlBreak", b'u'), ("spawnLane", b'u'), ("padS1", b'u'), ("padS2", b'u'), ("padS3", b'u'),
];

/// Params uniform builder (by field name).
#[derive(Clone)]
pub struct Params { pub words: [u32; 128] }
impl Default for Params { fn default() -> Self { Params { words: [0; 128] } } }
impl Params {
    fn idx(k: &str) -> (usize, u8) { let i = PARAM_FIELDS.iter().position(|(n, _)| *n == k).unwrap_or_else(|| panic!("param {k}")); (i, PARAM_FIELDS[i].1) }
    /// Set a field from a JS number (stored as f32 or u32 like a typed-array write).
    pub fn set(&mut self, k: &str, v: f64) { let (i, t) = Self::idx(k); self.words[i] = if t == b'f' { (v as f32).to_bits() } else { bayenv_world::jsmath::to_u32(v) }; }
    pub fn get_f(&self, k: &str) -> f32 { f32::from_bits(self.words[Self::idx(k).0]) }
    pub fn get_u(&self, k: &str) -> u32 { self.words[Self::idx(k).0] }
}
