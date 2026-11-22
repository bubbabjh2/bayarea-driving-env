//! src/sim/sim.js Simulation: clock, spawning, perception (leaders, lane ends, junctions, controls),
//! drivers -> reaction ring, integration, rigid-body contacts, crash scenes, fuel, re-routing.
//! Split over sim.rs (state, step, spawn, integrate, trips), leaders.rs, junction.rs, contacts.rs, extra.rs.
use crate::jm;
use crate::physics::VEHICLE_PARAMS;
use crate::population::Population;
use crate::rng::Rng;
use crate::spatial::{LaneIndex, SpatialHash};
use crate::vehicle::*;
use crate::world::World;
use crate::fx::{HashMap, HashSet};
use std::sync::Arc;

pub const FAIL_PENDING: u8 = 0;
pub const FAIL_ARRIVED: u8 = 1;
pub const FAIL_CRASH: u8 = 2;
pub const FAIL_STRANDED: u8 = 3;
pub const FAIL_STUCK: u8 = 4;
pub const FAIL_NOROUTE: u8 = 5;
pub const MERGE_ROOM: f64 = 15.0;
/// seconds of departures whose refuel plans are prefetched in one batch
pub const PLAN_LOOKAHEAD: f64 = 60.0;

/// SimOpts::steer_delay default (s)
pub const STEER_DELAY: f64 = 0.15;

/// SIM_DEFAULTS (the step-relevant keys).
#[derive(Clone, Debug)]
pub struct SimOpts {
    pub dt: f64, pub start_time: f64, pub learner_fraction: f64, pub shoulder: f64, pub freeway_shoulder: f64, pub opposite_margin: f64,
    pub touch_kmh: f64, pub layer_gap: f64, pub stuck_timeout: f64, pub pass_wrecks: bool, pub spawn_blocked_timeout: f64,
    pub cell_size: f64, pub table_bits: u32, pub leader_lookahead: f64, pub merge_lookahead: f64, pub lane_end_lookahead: f64,
    pub control_lookahead: f64, pub reroute: bool, pub reroute_sight: f64, pub squeeze_delay: f64, pub squeeze_speed: f64,
    pub density_interval: f64, pub path_sight: f64, pub deadlock_break: f64, pub spawn_lane_route: f64, pub junction_sight: f64,
    pub position_mode: bool, pub traffic_mode: TrafficMode, pub fuel: bool, pub seed: f64, pub wrecks: usize,
    pub default_action: Action,
    pub fuel_cfg: crate::fuel::FuelCfg,
    pub slot_pool: usize,
    /// run the per-car physics of a step on the rayon pool (results identical; worth it above ~1k moving cars)
    pub parallel: bool,
    /// Model fix (default ON; false = the JS, required for JS-oracle checks, see [`SimOpts::js`]): re-plan a missed turn at the missed-exit event (connector entry) with the
    /// in-edge as route prefix, as the GPU does (gpuSim.js:567), instead of once the car is on the taken edge. The JS
    /// re-plans one step late, so at the connector exit the next-box lookup sees the stale route and the late box
    /// entry is re-projected (headingRel += dh); rust/docs/crash-divergence.md "Cause A".
    pub replan_at_missed_event: bool,
    /// Model fix (default ON; false = the JS): lane-index ahead / behind use the slot's edge and lanes at the index
    /// rebuild (spatial.rs LaneIndex::snapshot). The JS reads the slot's current edge / lane_lo; after integrate (in the
    /// observation) a car that crossed onto another edge can make `behind` return the car itself and the junction-yield
    /// scan loop for ever (ENV-REQUESTS #15). Walks also stop at Sim::walk_limit, so neither mode can hang.
    pub lane_index_snapshot: bool,
    /// Model fix (default ON; false = the JS): resting wrecks and parked obstacles keep their real mass in car-car
    /// contacts (collide.js treats a crashed car at rest as immovable: a car hitting it bounces as off a wall, with
    /// twice the delta-v of a real two-body collision, and pileups never move wrecks). Barriers stay immovable.
    pub wreck_real_mass: bool,
    /// Model fix (default OFF until the lane polylines are smoothed, DECISIONS.md 23; false = the JS): the road frame's
    /// rotation no longer turns the car. The JS integrates
    /// heading_rel (heading relative to the lane) with the yaw rate only, so a car with the wheel straight followed
    /// every curve and junction turn at any speed (69 deg turns at 108 km/h: 14.6 g). With the fix the world heading
    /// changes by the yaw rate only; cornering needs steering and is limited by the tyres.
    pub curvature_dynamics: bool,
    /// The world holds the smoothed frame geometry (default OFF; DECISIONS.md 33): lane reference polylines and
    /// junction connectors filleted to realistic minimum radii by road class and speed limit (bayenv_world::smooth),
    /// built at load time with `LoadOptions::smooth_geometry` / `World::smooth_geometry`. Positions and frames both
    /// come from it. Meant for `curvature_dynamics` (required); `Sim::new` checks that it matches the world.
    pub smooth_geometry: bool,
    /// The world's turning junction connectors are re-fitted to vehicle-feasible radii (default OFF; DECISIONS.md 34):
    /// one arc of 8 m (local) / 12 m (arterial) / 15 m (freeway system), s_in / s_out moved to fit, never below 6 m
    /// where it can be helped (LoadOptions::junction_radii). Needs smooth_geometry; must match the world.
    pub junction_radii: bool,
    /// Spawn cars at least `spawn_clearance_m` before the junction entry their route takes next (default OFF;
    /// DECISIONS.md 34): when the first route edge leaves less, the car starts on the next route edge (after the
    /// junction) instead, as far as the route allows. The JS spawns at the first edge's s_exit.
    pub spawn_clearance: bool,
    pub spawn_clearance_m: f64,
    /// Model fix (default ON; false = the JS): a sliding wreck clamped at the end of its edge / connector loses its
    /// along-track speed (the JS kept the speed at a fixed position, e.g. 19.5 m/s decaying to 9 m/s standing still).
    pub wreck_end_stop: bool,
    /// Model fix (default ON; false = the JS): every contact step with delta-v >= touch_kmh reports an event, with the
    /// episode's accumulated delta-v as its severity (disables at the crash threshold). The JS reported only the first
    /// step of a touch, so a car riding a barrier could take a 12 km/h hit with no event.
    pub contact_episodes: bool,
    /// Model fix (default ON; false = the JS): junction connectors have a lateral limit (the in-edge's kerbs), with
    /// the same barrier response as edges. The JS had none: cars drifted up to 15 m off a connector and were pushed
    /// back in one step at the exit.
    pub connector_barrier: bool,
    /// Model fix (default ON; false = the JS): the junction-box parts of an edge (before its start mouth sExit + 1, past
    /// its end mouth sStop) have walls too: the edge's own barrier lines widened by `contacts::BOX_MARGIN` for the
    /// junction's corners. The JS had none there, so a car crossing the box at a large heading error (or pushed off its
    /// line in a collision) could drive tens to thousands of metres sideways in the edge frame; the first edge-barrier
    /// check past the box then met it at that depth: a "phantom wall" strike with delta-v near the full speed, at the
    /// kerb line, after a 1-2 s excursion off the road (DECISIONS.md 39).
    pub box_barrier: bool,
    /// Model fix (default ON; false = the JS): at a strict hop (freeway diverge) from a lane that continues onto no
    /// out-edge (a lane dropped at the node, e.g. 6 lanes -> 4 + a 1-lane ramp), the car takes the strict out-edge its
    /// lane overshoots least (a missed exit if that is not the planned edge). The JS kept the planned edge with the
    /// strict lat, so a car in the dropped far-left lane routed onto a right-hand ramp landed 16.6 m past the ramp's
    /// barrier line: a deep barrier contact (a teleport in the JS) on hr = 0 (DECISIONS.md 39).
    pub lane_drop_hop: bool,
    /// Model fix (default ON; false = the JS): a braked car below 0.02 m/s is held by static friction (the JS let
    /// stopped cars creep at up to 0.02 m/s under a held brake).
    pub brake_hold: bool,
    /// Model fix (default ON; false = the JS), replacing `brake_hold` (now default off: it counted rolling resistance
    /// as holding force, so a car pulling away with gas under ~3 % stayed parked for ever, ENV-REQUESTS #17): below
    /// 0.05 m/s the brake holds the car at 0 when the brake force alone beats the drive force minus the grade force;
    /// with no drive and no brake, rolling resistance holds it unless the grade force exceeds it.
    pub static_hold: bool,
    /// Model fix (default ON; false = the JS): kerbs are mountable. A barrier contact on a road that is not a rigid
    /// barrier (World::rigid_edge: the freeway system and bridges) with a closing speed under 2 m/s is a kerb scrape:
    /// no rebound, kerb friction 0.6, up to 0.3 m of mounting, one minor contact event that never disables. Faster
    /// strikes, and every strike on a rigid barrier, behave as before (audit finding 10).
    pub mountable_kerbs: bool,
    /// Model fix (default ON; false = the JS): automatic idle creep. The drive force never drops below an idle force
    /// of 4 x the rolling resistance at rest, fading to 0 at 2 m/s, so a car with no pedals on the flat creeps at about
    /// 1.5 m/s and a small brake (about 7 %) holds it.
    pub idle_creep: bool,
    /// Model fix (default ON; false = the JS): hill rollback. When drive + brake + rolling resistance can't hold the
    /// grade, a driving car rolls backwards under gravity (brakes and rolling resistance oppose it; static_hold with
    /// brake can hold it); the JS clamped every driving car's speed at >= 0, so an under-powered hill start just
    /// stopped dead. Rolling back into the car behind is a contact like any other.
    pub hill_rollback: bool,
    /// Model fix (default ON; false = the JS GPU's input version 3): obs v3 guidance inside a junction box skips the
    /// maneuver being crossed (car > 1 m past the in-edge end) and clamps the distance at 0, as obs v2 and
    /// bayenv-gpu `guide_in_box` (DECISIONS.md 43 / 46). CPU mirror of GpuOptions::guide_in_box for ObsMode::V3.
    pub v3_guide_in_box: bool,
    /// v3 bodies (obs.rs sense_bodies): a lateral offset under 1 cm counts as 0 when filing a body by sector (straight
    /// ahead / behind sit on a sector boundary; f32 vs f64 noise filed them differently on the GPU; bayenv-gpu
    /// GpuOptions::sector_snap, DECISIONS.md 49). Default on; js() off.
    pub v3_sector_snap: bool,
    /// Model fix (default ON; false = the JS GPU's input version 3): obs v3 own-ahead slot (inputs 8 / 9) gets the
    /// junction-conflict obstacle as obs v2 has it (adjust.js junctionObs; DECISIONS.md 45 / 46). CPU mirror of
    /// GpuOptions::junction_obstacle for ObsMode::V3.
    pub v3_junction_obstacle: bool,
    /// Model fix (default 0.15 s; negative = the JS): steering is applied after this delay (neuromuscular lag)
    /// instead of the full reaction time (0.6 - 1.8 s through the control ring) that still delays the pedals
    /// (ENV-REQUESTS #16). Capped at the reaction delay.
    pub steer_delay: f64,
    /// Model fix (default ON; false = the JS): trucks and buses brake at 4.7 m/s^2 (100-0 km/h in about 80 m, as
    /// loaded heavy vehicles; the JS's 6.5 gave 58 m)
    pub heavy_brakes: bool,
    /// Model fix (default ON; false = the JS): barrier penetration is corrected by at most BARRIER_PUSH_MAX per step
    /// (the JS removed the full depth at once: lateral jumps of up to 1.7 m)
    pub barrier_soft_push: bool,
    /// Model fix (default ON; false = the JS): a car knocked backwards by an impact keeps its backward speed while it
    /// is a wreck (sliding on locked wheels); the JS clamped it to 0. Driving cars still have no reverse.
    pub impact_reverse: bool,
    /// single-threaded integrate: two cars at a time with interleaved vehicle dynamics (identical results; A/B switch)
    pub batch_dynamics: bool,
    /// Bridge-rail breach (default OFF, CPU only, no JS counterpart; DECISIONS.md 46, breach.rs): a driving car that
    /// strikes the outer (right-hand) barrier of a bridge edge with an impact severity 1/2 m v_normal^2 above
    /// `breach_is_kj` goes through the rail: it is disabled, its trip fails as a crash, it leaves the vehicle store
    /// and falls (`Sim::falls`) to the ground or water below (`Sim::ground`).
    pub barrier_breach: bool,
    /// rail capacity for barrier_breach (kJ of impact severity); default MASH TL-3, test 3-11 (2270 kg, 100 km/h, 25 deg)
    pub breach_is_kj: f64,
}

/// one car's phase-A result (integrate_all)
pub(crate) struct PhaseA { pub slot: u32, pub locked: bool, pub c: Car, pub out: CarOut, pub ret: i32, pub pose: Pose }
/// pose fields of update_pose
#[derive(Clone, Copy, Default)]
pub(crate) struct Pose { pub x: f32, pub y: f32, pub z: f32, pub heading: f32, pub seg: i32 }
impl Pose { fn of(c: &Car, g: &World) -> Pose { let mut pc = *c; pc.update_pose(g); Pose { x: pc.x, y: pc.y, z: pc.z, heading: pc.heading, seg: pc.seg } } }

impl SimOpts {
    /// the integrate-side model fixes
    pub fn model_flags(&self) -> crate::vehicle::ModelFlags { crate::vehicle::ModelFlags { fixes: crate::physics::Fixes { hold: self.brake_hold, heavy_brakes: self.heavy_brakes, reverse: self.impact_reverse, static_hold: self.static_hold, idle_creep: self.idle_creep, rollback: self.hill_rollback }, curvature: self.curvature_dynamics, wreck_end_stop: self.wreck_end_stop, lane_drop_hop: self.lane_drop_hop,
        steer_delay_steps: if self.steer_delay < 0.0 { -1 } else { jm::round(self.steer_delay / self.dt) as i32 } } }
    /// The JS sim's model (oracle checks, trace_check / v2_check, JS-port tests): Default with the model fixes off
    /// (replan_at_missed_event = false). Checkpoints trained on the JS CPU sim saw this model.
    pub fn js() -> SimOpts { SimOpts { replan_at_missed_event: false, lane_index_snapshot: false, wreck_real_mass: false, curvature_dynamics: false, smooth_geometry: false, junction_radii: false, spawn_clearance: false, wreck_end_stop: false, contact_episodes: false, connector_barrier: false, box_barrier: false, lane_drop_hop: false, brake_hold: false, static_hold: false, mountable_kerbs: false, idle_creep: false, hill_rollback: false, v3_guide_in_box: false, v3_junction_obstacle: false, v3_sector_snap: false, steer_delay: -1.0, heavy_brakes: false, barrier_soft_push: false, impact_reverse: false, ..SimOpts::default() } }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrafficMode { Full, Sparse, Empty }

impl Default for SimOpts {
    fn default() -> Self {
        SimOpts { dt: 0.05, start_time: 5.0 * 3600.0, learner_fraction: 0.1, shoulder: 1.0, freeway_shoulder: 3.0, opposite_margin: 0.5,
            touch_kmh: 1.0, layer_gap: 2.5, stuck_timeout: 1800.0, pass_wrecks: false, spawn_blocked_timeout: f64::INFINITY,
            cell_size: 25.0, table_bits: 18, leader_lookahead: 250.0, merge_lookahead: 80.0, lane_end_lookahead: 450.0, control_lookahead: 250.0,
            reroute: true, reroute_sight: 2000.0, squeeze_delay: 900.0, squeeze_speed: 3.0, density_interval: 60.0, path_sight: 0.0,
            deadlock_break: 0.0, spawn_lane_route: 0.0, junction_sight: 0.0, position_mode: true, traffic_mode: TrafficMode::Full, fuel: true,
            seed: 1.0, wrecks: 0, default_action: Action { steer: Some(0.0), steer_rate: None, gas: 0.0, brake: 0.3 }, fuel_cfg: Default::default(), slot_pool: 0, parallel: false, replan_at_missed_event: true, batch_dynamics: true, lane_index_snapshot: true, wreck_real_mass: true, curvature_dynamics: false, smooth_geometry: false, junction_radii: false, spawn_clearance: false, spawn_clearance_m: 18.0, wreck_end_stop: true, contact_episodes: true, connector_barrier: true, box_barrier: true, lane_drop_hop: true, brake_hold: false, static_hold: true, mountable_kerbs: true, idle_creep: true, hill_rollback: true, v3_guide_in_box: true, v3_junction_obstacle: true, v3_sector_snap: true, steer_delay: STEER_DELAY, heavy_brakes: true, barrier_soft_push: true, impact_reverse: true, barrier_breach: false, breach_is_kj: crate::breach::MASH_TL3_IS_KJ }
    }
}

/// A driver command {steer | steerRate, gas, brake} (env.act).
#[derive(Clone, Copy, Debug, Default)]
pub struct Action { pub steer: Option<f64>, pub steer_rate: Option<f64>, pub gas: f64, pub brake: f64 }

/// Who drives a slot.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Driver {
    None,
    /// the env's external driver (env.act; opts.defaultAction when no command this step)
    External,
    /// placeObstacles 'slow' prop
    PropSlow,
    /// placeObstacles 'braking' prop: brake hard every brakeEvery s
    PropBraking { next: f64, until: f64 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Response { Active, Shoulder, Cleared }

/// Incident / crash record (sim.crashes entries; scenes are also in activeCrashes / edgeCrashes).
#[derive(Clone, Debug)]
pub struct Scene {
    pub id: usize, pub time: f64, pub cause: &'static str, pub severity: &'static str, pub is_crash: bool, pub scene: bool,
    pub vehicles: Vec<usize>, pub commuters: Vec<i32>, pub edge: i32, pub s: f64, pub lane_mask: i64, pub lanes_blocked: i32,
    pub x: f64, pub z: f64, pub y: f64, pub block_start: f64, pub block_end: f64, pub clear_end: f64, pub squeeze_at: f64,
    pub squeeze: bool, pub response: Response, pub permanent: bool, pub gone: Vec<usize>,
    pub delta_v_kmh: Vec<f64>, pub closing: f64, pub disabled: Vec<usize>, pub crash_type: String, pub maneuver: &'static str,
}

#[derive(Clone, Copy, Debug)]
pub struct Trip { pub commuter: usize, pub from: i32, pub to: i32, pub depart: f64, pub leg: u8 }

#[derive(Default, Clone, Debug)]
pub struct Counters {
    pub spawned: u64, pub arrived: u64, pub arrived_outbound: u64, pub on_time: u64, pub crashes: u64, pub crashed_vehicles: u64, pub crashes_while_speeding: u64,
    pub stuck: u64, pub spawn_moved: u64, pub no_route: u64, pub cleared: u64, pub secondary_crashes: u64, pub minor_contacts: u64, pub touches: u64, pub disabling_crashes: u64,
    pub repeat_contacts: u64, pub pull_overs: u64, pub towed_to_shoulder: u64, pub stranded: u64, pub towed: u64, pub refuel_planned: u64, pub refuelled: u64,
    pub refuel_plan_failed: u64, pub forced_spawns: u64, pub signal_violations: u64, pub stop_violations: u64, pub missed_exits: u64, pub reroutes: u64,
    pub reroute_failed: u64, pub wreck_passes: u64, pub deadlock_breaks: u64, pub hauled_away: u64,
    pub by_severity: HashMap<&'static str, u64>, pub stuck_by: HashMap<&'static str, u64>,
}

#[derive(Default, Clone, Debug)]
pub struct Accum { pub vehicle_sec: f64, pub vehicle_km: f64, pub speeding_sec: f64, pub speeding_km: f64, pub late_sum: f64, pub late_n: f64, pub fuel_l: f64, pub pump_wait_sum: f64, pub pump_wait_n: f64 }

#[derive(Default, Clone, Debug)]
pub struct Live { pub driving: u64, pub crashed: u64, pub refueling: u64, pub sum_speed: f64 }

#[derive(Default, Clone, Debug)]
pub struct Density { pub samples: u64, pub peak_mean_density: f64, pub peak_mean_at: f64, pub peak_los_df_share: f64, pub peak_los_df_at: f64, pub peak_segment_density: f64, pub mean_density_sum: f64, pub los_df_share_sum: f64 }

/// Routing service (src/sim/route.js makeRouter): route(start, goal, avoid, penalty) -> edge list or None.
/// Implemented by bayenv_world::nav (lead); `crate::route_stub` holds a test implementation until then.
pub trait Router: Send + Sync {
    fn route(&mut self, start: i32, goal: i32, avoid: Option<&std::collections::HashSet<i32>>, penalty: Option<&std::collections::HashMap<i32, f64>>) -> Option<Vec<i32>>;
    /// Several independent queries with the same constraints; results in query order. Implementations may run them
    /// in parallel; each result must equal `route` of the same query.
    fn route_batch(&mut self, queries: &[(i32, i32)], avoid: Option<&std::collections::HashSet<i32>>, penalty: Option<&std::collections::HashMap<i32, f64>>) -> Vec<Option<Vec<i32>>> {
        queries.iter().map(|&(s, g)| self.route(s, g, avoid, penalty)).collect()
    }
}

/// Per-car events since the last `take_events` (Env::step clears them): what reward.js needs.
#[derive(Clone, Copy, Debug)]
pub struct ArrivedEv { pub slot: usize, pub agent: i32, pub leg: u8, pub late_by: f64 }
/// A crash event for a car that was driving (sim._crashEvent / signalsFor .crash): own delta-v (km/h), its share of
/// the closing speed (reward.js crashShare), disabled = own delta-v >= 10 km/h (the trip fails).
#[derive(Clone, Copy, Debug)]
pub struct CrashedEv { pub slot: usize, pub agent: i32, pub dv_kmh: f64, pub share: f64, pub disabled: bool }
#[derive(Clone, Debug, Default)]
pub struct StepEvents { pub arrived: Vec<ArrivedEv>, pub crashed: Vec<CrashedEv>, pub stranded: Vec<(usize, i32)> }

/// Per-slot event (shared shape with the GPU backend; Env::events drains them as columns).
/// kind: 1 ARRIVED, 2 CRASHED (disabling: own delta-v >= 10 km/h), 3 CONTACT (crash event >= touchKmh, not disabling),
/// 4 STRANDED, 5 CLEARED (wreck towed / hauled away: slot freed), 6 SPAWNED, 7 MISSED_EXIT, 8 SIGNAL_VIOLATION.
/// Within a step events are sorted by (slot, kind), stable.
#[derive(Clone, Copy, Debug)]
pub struct Event { pub step: u64, pub time: f64, pub kind: u8, pub slot: i32, pub agent: i64, pub other: i32, pub dv_kmh: f32, pub other_dv_kmh: f32, pub share: f32,
    pub late_by: f32, pub progress_m: f32, pub trip_m: f32, pub trip_s: f32 }
pub const EV_ARRIVED: u8 = 1; pub const EV_CRASHED: u8 = 2; pub const EV_CONTACT: u8 = 3; pub const EV_STRANDED: u8 = 4;
pub const EV_CLEARED: u8 = 5; pub const EV_SPAWNED: u8 = 6; pub const EV_MISSED_EXIT: u8 = 7; pub const EV_SIGNAL_VIOLATION: u8 = 8;

/// Batched external policy hook (JS batchDrive / opts.batchPolicy): called once per step in `drive`, before any car
/// is commanded, with the slots of every driving car that has the external driver; it writes their commands into
/// `actions[slot]` (None = the default action). Runs at the point where JS calls driver.step, so it sees exactly the
/// state the per-car drivers see.
pub type BatchPolicy = Box<dyn FnMut(&Sim, &[usize], &mut [Option<Action>]) + Send + Sync>;

/// Per-step signals for a driver (sim.signalsFor).
#[derive(Clone, Copy, Debug, Default)]
pub struct Signals { pub collision: bool, pub stranded: bool, pub harsh_braking: bool, pub time_to_work_start: f64, pub remaining_route_distance: f64, pub fuel_fraction: f64, pub refuel_planned: bool }

pub struct Sim {
    pub g: Arc<World>,
    pub pop: Arc<Population>,
    pub opts: SimOpts,
    pub rng: Rng,
    pub time: f64,
    pub dt: f64,
    pub step_count: i64,
    pub empty: bool,
    pub vs: VehicleStore,
    pub hash: SpatialHash,
    pub lanes: LaneIndex,
    pub drivers: Vec<Driver>,
    /// external commands of this step (env.act), by slot
    pub actions: Vec<Option<Action>>,
    pub router: Box<dyn Router>,
    pub routes: crate::fuel::RouteStore,
    pub stations: Option<crate::fuel::StationNetwork>,
    pub learners: Vec<u8>,
    pub outcome: Vec<u8>, pub outcome_back: Vec<u8>,
    pub commuter_fuel: Vec<f32>,
    pub fitness: Vec<f64>, pub actual_arrival: Vec<f64>, pub late_by: Vec<f64>, pub early_by: Vec<f64>,
    pub trips: Vec<Trip>, pub trip_ptr: usize, pub pending: Vec<Trip>,
    pub lot: Vec<usize>,
    pub crashes: Vec<Scene>, pub active: Vec<usize>, pub edge_crashes: HashMap<i32, Vec<usize>>, pub edge_has_crash: Vec<u8>,
    pub counters: Counters, pub accum: Accum, pub live: Live, pub density: Density,
    pub obstacle_rng: Option<Rng>,
    pub brake_every: (f64, f64),
    // perception scratch / per-step maps (JS Maps whose iteration order matters are Vecs)
    pub(crate) spawned_keys: HashSet<i64>,
    /// first edges whose _entryLane inputs changed during this step's spawn pass
    pub(crate) spawn_dirty: HashSet<i32>,
    /// refuel plans prefetched for this spawn pass, by (from, goal)
    /// refuel plans by (from, goal), prefetched in router batches ahead of the departures (a plan is a pure function of
    /// the OD, so any cached plan equals a fresh one); value = (plan, sim time it was computed)
    pub(crate) plan_prefetch: HashMap<(i32, i32), (Option<crate::fuel::Plan>, f64)>,
    /// departures up to this time have had their plans prefetched
    pub(crate) plan_horizon: f64,
    /// integrate_all phase-A results (reused buffer)
    pub(crate) phase_a: Vec<PhaseA>,
    /// edge of each slot listed in the lane index at its last rebuild (-1 not listed)
    pub(crate) indexed: Vec<i32>,
    pub(crate) pair_event: HashMap<i64, f64>,
    pub(crate) stuck_labelled: HashMap<usize, f32>,
    pub(crate) conn_cars: crate::fx::ListMap<usize>,
    pub(crate) node_conn: crate::fx::ListMap<usize>,
    pub(crate) passers: crate::fx::ListMap<i32>,
    pub(crate) conflict_with: HashMap<usize, usize>,
    pub(crate) off_lane: crate::fx::ListMap<i32>,
    pub(crate) touch: HashSet<i64>,
    pub(crate) barrier_touch: HashSet<usize>, pub(crate) barrier_now: HashSet<usize>,
    /// diagnostics: Some = every barrier() contact is recorded here (None by default; no effect on the sim)
    pub barrier_dbg: Option<Vec<crate::contacts::BarrierDbg>>,
    /// contact_episodes: accumulated delta-v of the barrier / pair contact episodes (last step, this step)
    pub(crate) barrier_ep: HashMap<usize, f64>, pub(crate) barrier_ep_now: HashMap<usize, f64>,
    pub(crate) touch_ep: HashMap<i64, (f64, f64)>, pub(crate) touch_ep_now: HashMap<i64, (f64, f64)>,
    /// impact_reverse: backward speed of driving cars from this step's contacts (applied if the car is disabled)
    pub(crate) reverse_stash: HashMap<usize, f64>,
    pub(crate) yield_by: Option<Vec<i32>>,
    pub(crate) creep_until: Vec<f64>, pub(crate) creep_ignore: Vec<i32>,
    pub(crate) edge_speed: HashMap<i32, f64>,
    pub traffic_penalty: Option<HashMap<i32, f64>>,
    pub(crate) last_reroute: HashMap<usize, f64>,
    pub(crate) next_traffic: f64, pub(crate) next_reroute: f64, pub(crate) next_density: f64,
    pub(crate) edge_count_scratch: Vec<u16>, pub(crate) fwy: Vec<usize>, pub(crate) fwy_lane_km: f64,
    pub(crate) near_buf: Vec<usize>,
    pub static_obstacles: Vec<usize>,
    pub obstacles: Vec<(usize, &'static str, i32, f64)>,
    /// per-phase wall time (ns) when `profile` is set: spawn, hash, contacts, lanes, leaders, controls, drive, integrate, scenes, traffic+reroute+density
    pub profile: bool,
    pub events: StepEvents,
    pub event_log: Vec<Event>,
    /// speculative phase-2 perception re-runs (diagnostic)
    pub p2_reruns: u64,
    /// slots in a crash event (contact >= touchKmh) during the last step
    pub collided_last: Vec<usize>,
    pub batch_policy: Option<BatchPolicy>,
    pub phase_ns: [u64; 10],
    /// sub-phase timers (profile): 0 leaders phase 1, 1 integrate phase A, 2 reroute, 3 traffic sample, 4 spawn pre-eval
    pub sub_ns: [u64; 12],
    /// SimOpts::barrier_breach: every car that went through a bridge rail, falling or at rest below (breach.rs)
    pub falls: Vec<crate::breach::Fall>,
    /// the surface the falls land on (None: sea level 0); e.g. `breach::DemGround::load(data_dir)`
    pub ground: Option<Arc<dyn crate::breach::Ground>>,
}

#[inline(always)] pub(crate) fn f(x: f32) -> f64 { x as f64 }
/// A raw pointer that may cross threads (disjoint-index writes only; see the use sites).
#[derive(Clone, Copy)]
pub(crate) struct SyncPtr<T>(pub *mut T);
unsafe impl<T> Send for SyncPtr<T> {}
unsafe impl<T> Sync for SyncPtr<T> {}
impl<T> SyncPtr<T> { #[inline(always)] pub(crate) fn get(&self) -> *mut T { self.0 } }
#[inline(always)] pub(crate) fn r32(x: f64) -> f32 { x as f32 }

impl Sim {
    /// The world without an Arc clone (an atomic refcount bump on a cache line shared by every env of a VecEnv).
    /// Sound because `g` is set at construction and never replaced, and the reference never outlives the call.
    #[inline(always)]
    pub fn gw(&self) -> &'static World { unsafe { &*Arc::as_ptr(&self.g) } }
    #[inline(always)]
    pub fn popw(&self) -> &'static Population { unsafe { &*Arc::as_ptr(&self.pop) } }
    pub fn new(g: Arc<World>, pop: Arc<Population>, opts: SimOpts, router: Box<dyn Router>) -> Sim {
        let n_c = pop.n;
        let mut rng = Rng::new(opts.seed);
        let mut learners = vec![0u8; n_c];
        for l in learners.iter_mut() { *l = if rng.random() < opts.learner_fraction { 1 } else { 0 }; }
        let mut commuter_fuel = vec![0f32; n_c];
        for i in 0..n_c {
            let vt = pop.vehicle_type[i] as usize;
            let t = &VEHICLE_PARAMS[if vt < 4 { vt } else { 0 }];
            commuter_fuel[i] = (pop.fuel_fill[i] * t.tank_l) as f32;
        }
        let trips = pop.trips();
        let cap = if opts.slot_pool > 0 { opts.slot_pool } else { 1024usize.max(((n_c as f64 * 0.6).ceil() as usize + 64).min(1 << 20)) };
        assert!(opts.dt > 0.0 && opts.dt.is_finite(), "SimOpts.dt must be a positive finite number of seconds (got {})", opts.dt);
        assert!(opts.smooth_geometry == g.geometry_smoothed, "SimOpts.smooth_geometry ({}) must match the world (smoothed: {}): load it with LoadOptions::smooth_geometry / Env::load_world_smoothed",
            opts.smooth_geometry, g.geometry_smoothed);
        assert!(!opts.smooth_geometry || opts.curvature_dynamics, "SimOpts.smooth_geometry is the frame geometry of curvature_dynamics: turn curvature_dynamics on too");
        assert!(opts.junction_radii == g.junction_radii, "SimOpts.junction_radii ({}) must match the world (re-fitted: {}): load it with LoadOptions::junction_radii", opts.junction_radii, g.junction_radii);
        assert!(!opts.junction_radii || opts.smooth_geometry, "SimOpts.junction_radii needs smooth_geometry");
        let mut vs = VehicleStore::new(cap);
        vs.model = opts.model_flags();
        vs.position_mode = opts.position_mode;
        let ne = g.edge_count;
        let mut fwy = Vec::new();
        let mut fwy_lane_km = 0.0;
        for e in 0..ne { if g.e_hw[e] & crate::world::HW_MAINLINE != 0 && g.e_length[e] >= 100.0 { fwy.push(e); fwy_lane_km += g.e_length[e] * g.e_lanes[e] as f64 / 1000.0; } }
        let stations = if opts.fuel && !g.stations.is_empty() { Some(crate::fuel::StationNetwork::new(&g, opts.fuel_cfg.clone())) } else { None };
        let start = opts.start_time;
        let mut sim = Sim {
            hash: SpatialHash::new(opts.cell_size, opts.table_bits, cap), lanes: { let mut l = LaneIndex::new(ne, cap); l.snapshot = opts.lane_index_snapshot; l }, drivers: vec![Driver::None; cap], actions: vec![None; cap],
            routes: crate::fuel::RouteStore::new(&pop), stations, learners, outcome: vec![0; n_c], outcome_back: vec![0; n_c], commuter_fuel,
            fitness: vec![0.0; n_c], actual_arrival: vec![-1.0; n_c], late_by: vec![0.0; n_c], early_by: vec![0.0; n_c],
            trips, trip_ptr: 0, pending: Vec::new(), lot: Vec::new(), crashes: Vec::new(), active: Vec::new(), edge_crashes: HashMap::default(), edge_has_crash: vec![0; ne],
            counters: Counters::default(), accum: Accum::default(), live: Live::default(), density: Density::default(), obstacle_rng: None, brake_every: (15.0, 40.0),
            spawned_keys: HashSet::default(), spawn_dirty: HashSet::default(), plan_prefetch: HashMap::default(), plan_horizon: f64::NEG_INFINITY, phase_a: Vec::new(), indexed: Vec::new(), pair_event: HashMap::default(), stuck_labelled: HashMap::default(), conn_cars: Default::default(), node_conn: Default::default(),
            passers: Default::default(), conflict_with: HashMap::default(), off_lane: Default::default(), touch: HashSet::default(), barrier_touch: HashSet::default(), barrier_now: HashSet::default(), barrier_dbg: None, barrier_ep: HashMap::default(), barrier_ep_now: HashMap::default(), touch_ep: HashMap::default(), touch_ep_now: HashMap::default(), reverse_stash: HashMap::default(),
            yield_by: None, creep_until: Vec::new(), creep_ignore: Vec::new(), edge_speed: HashMap::default(), traffic_penalty: None,
            last_reroute: HashMap::default(), next_traffic: 0.0, next_reroute: 0.0, next_density: start, edge_count_scratch: vec![0; ne], fwy, fwy_lane_km,
            near_buf: Vec::new(), static_obstacles: Vec::new(), obstacles: Vec::new(), profile: std::env::var("BAYENV_PROFILE").is_ok(), phase_ns: [0; 10], sub_ns: [0; 12], events: StepEvents::default(), event_log: Vec::new(), p2_reruns: 0, collided_last: Vec::new(), batch_policy: None, falls: Vec::new(), ground: None,
            empty: opts.traffic_mode == TrafficMode::Empty, time: start, dt: opts.dt, step_count: 0, vs, rng, router, g, pop, opts,
        };
        if sim.opts.traffic_mode == TrafficMode::Sparse && sim.opts.wrecks > 0 { sim.place_wrecks(sim.opts.wrecks); }
        sim
    }

    /// Append an event for slot i (agent = 2 * commuter + leg, -1 without a commuter; progress = routeDist).
    pub(crate) fn emit(&mut self, kind: u8, i: usize, other: i32, dv: f32, odv: f32, share: f32, late: f32) {
        let vs = &self.vs;
        let c = vs.commuter[i];
        let (trip_m, trip_s) = if kind == EV_ARRIVED { (vs.odometer[i], (self.time - f(vs.spawn_time[i])) as f32) } else { (f32::NAN, f32::NAN) };
        self.event_log.push(Event { step: self.step_count as u64, time: self.time, kind, slot: i as i32, agent: if c >= 0 { 2 * c as i64 + vs.leg[i] as i64 } else { -1 }, other,
            dv_kmh: dv, other_dv_kmh: odv, share, late_by: late, progress_m: vs.route_dist[i], trip_m, trip_s });
    }

    pub fn is_idle(&self) -> bool { self.vs.count == 0 && self.pending.is_empty() && self.trip_ptr >= self.trips.len() }

    // ---------------------------------------------------------------- step
    pub fn step(&mut self) {
        let ev0 = self.event_log.len();
        self.collided_last.clear();
        self.step_inner();
        self.event_log[ev0..].sort_by_key(|e| (e.slot, e.kind));
    }

    fn step_inner(&mut self) {
        let dt = self.dt;
        self.time += dt;
        self.step_count += 1;
        let prof = self.profile;
        let mut t = if prof { Some(std::time::Instant::now()) } else { None };
        macro_rules! lap { ($k:expr) => { if let Some(t0) = t { let n = std::time::Instant::now(); self.phase_ns[$k] += (n - t0).as_nanos() as u64; t = Some(n); } }; }
        self.spawn(); lap!(0);
        self.hash.rebuild(&self.vs); lap!(1);
        self.contacts(); lap!(2);
        self.lanes.rebuild(&self.vs);
        { let vs = &self.vs; self.indexed.clear(); self.indexed.extend((0..vs.high).map(|i| if vs.state[i] == STATE_DRIVING || vs.state[i] == STATE_CRASHED { vs.edge[i] } else { -1 })); }
        lap!(3);
        self.find_leaders(); lap!(4);
        if self.g.controls.is_some() { self.control_obs(); } lap!(5);
        if let Some(yb) = &mut self.yield_by { yb.iter_mut().for_each(|x| *x = -1); }
        self.drive(dt); lap!(6);
        self.integrate_all(dt); lap!(7);
        if !self.falls.is_empty() { self.step_falls(dt); }
        if !self.vs.missed.is_empty() { self.missed_exits(); }
        self.scenes(); lap!(8);
        let tq = std::time::Instant::now();
        if self.time >= self.next_traffic { self.traffic_sample(); self.next_traffic = self.time + 60.0; }
        let tr = std::time::Instant::now(); if prof { self.sub_ns[3] += (tr - tq).as_nanos() as u64; }
        if self.opts.reroute && self.time >= self.next_reroute { self.reroute_blocked(); self.next_reroute = self.time + 2.0; }
        if prof { self.sub_ns[2] += tr.elapsed().as_nanos() as u64; }
        if self.time >= self.next_density { self.sample_density(); self.next_density = self.time + self.opts.density_interval; }
        lap!(9);
    }

    // ---------------------------------------------------------------- spawning
    fn spawn(&mut self) {
        self.spawned_keys.clear();
        let tl = std::time::Instant::now();
        if !self.lot.is_empty() { self.leave_stations(); }
        // the pending trips, then the trips due now, in JS order (failures stay pending, in that order)
        let mut batch = std::mem::take(&mut self.pending);
        let n_pending = batch.len();
        while self.trip_ptr < self.trips.len() && self.trips[self.trip_ptr].depart <= self.time { batch.push(self.trips[self.trip_ptr]); self.trip_ptr += 1; }
        if batch.is_empty() { return; }
        // entry lanes of every trip against the state at the start of the pass (parallel when enabled). A result stays
        // valid until a spawn earlier in the pass changes what _entryLane reads for that first edge: a spawn on the same
        // edge (spawnedKeys), or a recycled slot that is still listed in the lane index (its old edge, and the edges
        // that edge feeds, see _upstreamConflict). Those trips are re-evaluated in order, so the outcome is exact.
        let tp = std::time::Instant::now();
        let pre: Vec<i32> = if self.opts.parallel && batch.len() >= 256 {
            use rayon::prelude::*;
            let me = &*self;
            batch.par_iter().with_min_len(64).map(|t| me.trip_entry_lane(t)).collect()
        } else { batch.iter().map(|t| self.trip_entry_lane(t)).collect() };
        if self.profile { self.sub_ns[4] += tp.elapsed().as_nanos() as u64; self.sub_ns[5] += n_pending as u64; }
        let tpf = std::time::Instant::now();
        if self.stations.is_some() && self.opts.parallel {
            if self.time >= self.plan_horizon - 0.5 * PLAN_LOOKAHEAD { self.prefetch_plans_ahead(); }
            self.prefetch_spawn_plans(&batch, &pre);
        }
        if self.profile { self.sub_ns[10] += tpf.elapsed().as_nanos() as u64; }
        self.spawn_dirty.clear();
        let ts = std::time::Instant::now();
        let mut keep = Vec::with_capacity(n_pending);
        for (k, t) in batch.into_iter().enumerate() {
            if !self.try_spawn(t, pre[k]) { keep.push(t); }
        }
        if self.profile { self.sub_ns[11] += ts.elapsed().as_nanos() as u64; }
        self.pending = keep;
    }

    /// (from, goal) of a trip's refuel plan if it would check its fuel at spawn with its current commuter fuel
    fn trip_fuel_od(&self, t: &Trip) -> Option<(i32, i32)> {
        let g = &*self.g;
        let pop = &*self.pop;
        let st = self.stations.as_ref()?;
        let c = t.commuter;
        let r = pop.route(c, t.leg as usize);
        if r.is_empty() { return None; }
        let vt = pop.vehicle_type[c] as usize;
        let tank = VEHICLE_PARAMS[if vt < 4 { vt } else { 0 }].tank_l as f32;
        let fuel = jm::min(tank as f64, f(self.commuter_fuel[c])) as f32;
        let mut total = 0f32; for &e in r { total = (total as f64 + g.e_length[e as usize]) as f32; }
        if !st.needs_fuel(vt, fuel as f64, tank as f64, total as f64) { return None; }
        Some((g.e_from[r[0] as usize], g.e_to[r[r.len() - 1] as usize]))
    }

    fn prefetch_ods(&mut self, ods: Vec<(i32, i32)>) {
        if ods.is_empty() { return; }
        let g = self.gw();
        if self.profile { self.sub_ns[5] += ods.len() as u64 * 1000000; }
        let plans = self.stations.as_ref().unwrap().plan_many(&*g, &mut *self.router, &ods);
        let now = self.time;
        for (od, p) in ods.into_iter().zip(plans) { self.plan_prefetch.insert(od, (p, now)); }
    }

    /// Refuel plans of every trip departing within PLAN_LOOKAHEAD that would check its fuel at spawn (with the
    /// commuter fuel known now), as one large router batch (parallel), refreshed every half window. Trips whose need
    /// changes later are planned at spawn as before; results never depend on the cache.
    fn prefetch_plans_ahead(&mut self) {
        let until = self.time + PLAN_LOOKAHEAD;
        let mut ods: Vec<(i32, i32)> = Vec::new();
        let mut seen: HashSet<(i32, i32)> = HashSet::default();
        let mut add = |me: &Self, t: &Trip, ods: &mut Vec<(i32, i32)>| {
            if let Some(od) = me.trip_fuel_od(t) { if !me.plan_prefetch.contains_key(&od) && seen.insert(od) { ods.push(od); } }
        };
        for t in &self.pending { add(self, t, &mut ods); }
        let mut k = self.trip_ptr;
        while k < self.trips.len() && self.trips[k].depart <= until { let t = self.trips[k]; add(self, &t, &mut ods); k += 1; }
        let keep_after = self.time - 2.0 * PLAN_LOOKAHEAD;
        self.plan_prefetch.retain(|_, v| v.1 >= keep_after);
        self.prefetch_ods(ods);
        self.plan_horizon = until;
    }

    /// Plans for this step's departures not prefetched yet (StationNetwork.plan is a pure function of (from, goal));
    /// check_fuel takes them from the cache.
    fn prefetch_spawn_plans(&mut self, batch: &[Trip], pre: &[i32]) {
        let mut ods: Vec<(i32, i32)> = Vec::new();
        for (k, t) in batch.iter().enumerate() {
            if pre[k] < 0 { continue; }
            if let Some(od) = self.trip_fuel_od(t) { if !self.plan_prefetch.contains_key(&od) && !ods.contains(&od) { ods.push(od); } }
        }
        self.prefetch_ods(ods);
    }

    /// _entryLane for a trip's first edge (i32::MIN = no route / not evaluated).
    fn trip_entry_lane(&self, t: &Trip) -> i32 {
        let k = 2 * t.commuter + t.leg as usize;
        let (o0, o1) = (self.pop.route_offset[k] as usize, self.pop.route_offset[k + 1] as usize);
        if o1 == o0 { return i32::MIN; }
        let route = &self.pop.route_flat[o0..o1];
        let vt = self.pop.vehicle_type[t.commuter] as usize;
        self.entry_lane(route[0] as usize, 0, VEHICLE_PARAMS[if vt < 4 { vt } else { 0 }].length, if route.len() > 1 { route[1] } else { -1 })
    }

    /// _entryLane(ei, lane0, vlen, next)
    pub(crate) fn entry_lane(&self, ei: usize, lane0: i32, vlen: f64, next: i32) -> i32 {
        let g = &*self.g;
        let vs = &self.vs;
        let lanes = g.e_lanes[ei];
        if self.empty { return lane0.min(lanes - 1); }
        let sr = self.opts.spawn_lane_route >= 1.0 && next >= 0;
        let goes = |l: i32| -> bool {
            if let Some(j) = &g.j { if g.e_s_stop[ei] < g.e_length[ei] { return j.movement(ei, l, next) >= 0; } }
            let m = g.map_lane(ei, next as usize, l); m >= 0 && m < g.e_lanes[next as usize]
        };
        let kmax = (if sr { 2 } else { 1 }) * lanes;
        for k in 0..kmax {
            let l = (lane0 + k) % lanes;
            if sr && ((k < lanes) != goes(l)) { continue; }
            let key = ei as i64 * 8 + l as i64;
            if self.spawned_keys.contains(&key) { continue; }
            let fr = self.lanes.first(ei as i32, l);
            let sx = g.e_s_exit[ei];
            if fr >= 0 && f(vs.s[fr as usize]) - 0.5 * f(vs.length[fr as usize]) < (if sx > 0.0 { sx } else { 0.0 }) + vlen + 4.0 { continue; }
            if self.upstream_conflict(ei, l) { continue; }
            return l;
        }
        -1
    }

    fn upstream_conflict(&self, ei: usize, l: i32) -> bool {
        let g = &*self.g;
        let vs = &self.vs;
        let node = g.e_from[ei] as usize;
        for q in g.in_start[node]..g.in_start[node + 1] {
            let e2 = g.in_edges[q as usize];
            for k in 0..g.e_lanes[e2 as usize] {
                let j = self.lanes.last(e2, k);
                if j < 0 || vs.state[j as usize] != STATE_DRIVING { continue; }
                let j = j as usize;
                let rj = &vs.routes[j];
                let pj = vs.route_pos[j] as usize;
                if pj + 1 >= rj.len() || rj[pj + 1] != ei as i32 || g.map_lane(e2 as usize, ei, vs.lane[j] as i32) != l { continue; }
                let d = g.e_length[e2 as usize] - f(vs.s[j]);
                let v = f(vs.speed[j]);
                if d < 8.0 + v * 2.0 + v * v / 6.0 { return true; }
            }
        }
        false
    }

    pub(crate) fn traits_of(&self, c: usize) -> Traits {
        let d = &self.pop.deltas[c * 64..c * 64 + 6];
        Traits { aggression: d[0] as f64, patience: d[1] as f64, reaction_time: d[2] as f64, preferred_headway: d[3] as f64, max_accel: d[4] as f64, max_decel: d[5] as f64 }
    }

    fn try_spawn(&mut self, trip: Trip, pre: i32) -> bool {
        if self.opts.slot_pool > 0 && self.vs.count >= self.opts.slot_pool { return false; }
        let c = trip.commuter;
        let k = 2 * c + trip.leg as usize;
        let (o0, o1) = (self.pop.route_offset[k] as usize, self.pop.route_offset[k + 1] as usize);
        if o1 == o0 {
            self.counters.no_route += 1;
            self.set_outcome(c, trip.leg, FAIL_NOROUTE);
            return true;
        }
        let pop = self.popw();
        let route = &pop.route_flat[o0..o1];
        let vt = pop.vehicle_type[c] as usize;
        let vlen = VEHICLE_PARAMS[if vt < 4 { vt } else { 0 }].length;
        // spawn_clearance: start on the first route edge that leaves spawn_clearance_m before its junction entry
        let k0 = if self.opts.spawn_clearance { self.spawn_edge_index(route) } else { 0 };
        let e0 = route[k0];
        let pre = if k0 > 0 { i32::MIN } else { pre };
        let mut lane = if pre != i32::MIN && !self.spawn_dirty.contains(&e0) { pre } else { self.entry_lane(e0 as usize, 0, vlen, if route.len() > k0 + 1 { route[k0 + 1] } else { -1 }) };
        if lane < 0 {
            if self.time - trip.depart < self.opts.spawn_blocked_timeout { return false; }
            lane = 0; self.counters.forced_spawns += 1;
        }
        self.spawned_keys.insert(e0 as i64 * 8 + lane as i64);
        let learner = self.learners[c] == 1;
        let traits = self.traits_of(c);
        let g = self.gw();
        let slot = self.vs.alloc(&g, AllocArgs { type_idx: pop.vehicle_type[c], traits, route, commuter: c as i32, leg: trip.leg, time: self.time, learner, dt: self.dt, step: self.step_count, lane });
        self.spawn_dirty.insert(e0);
        if slot < self.indexed.len() && self.indexed[slot] >= 0 {
            let g = self.gw();
            let eo = self.indexed[slot] as usize;
            self.spawn_dirty.insert(eo as i32);
            let to = g.e_to[eo] as usize;
            for q in g.out_start[to]..g.out_start[to + 1] { self.spawn_dirty.insert(g.out_edges[q as usize]); }
        }
        self.vs.set_reaction_time(slot, traits.reaction_time, self.dt);
        self.vs.route_off[slot] = self.routes.pop_offset(k);
        self.vs.fuel[slot] = jm::min(f(self.vs.tank[slot]), f(self.commuter_fuel[c])) as f32;
        self.vs.update_pose(slot, &g); self.vs.save_pose(slot);
        self.counters.spawned += 1;
        self.emit(EV_SPAWNED, slot, -1, f32::NAN, f32::NAN, f32::NAN, f32::NAN);
        let tf = std::time::Instant::now();
        if self.stations.is_some() { self.check_fuel(slot, true); }
        if self.profile { self.sub_ns[6] += tf.elapsed().as_nanos() as u64; }
        // spawn_clearance: move the car to route[k0] at its s_exit, as if it had driven the skipped edges (after the
        // refuel check, which may re-plan the route from its start: then only when the plan still has e0 at k0 and its
        // station stop lies at or after it)
        if k0 > 0 && self.vs.routes[slot].get(k0) == Some(&e0) && (self.vs.refuel_state[slot] == 0 || self.vs.stop_pos[slot] >= k0 as i32) {
            let g = self.gw();
            let e = e0 as usize; let sx = if g.e_s_exit[e] > 0.0 { g.e_s_exit[e] } else { 0.0 };
            let vs = &mut self.vs;
            vs.route_pos[slot] = k0 as i32; vs.edge[slot] = e0; vs.s[slot] = sx as f32;
            vs.lat[slot] = ((lane as f64 + 0.5) * g.lane_w(e, sx)) as f32;
            vs.route_dist[slot] = route[..k0].iter().map(|&q| g.e_length[q as usize]).sum::<f64>() as f32;
            vs.update_lanes(slot, &g, e);
            vs.update_pose(slot, &g); vs.save_pose(slot);
            self.counters.spawn_moved += 1;
        }
        self.ensure_slot(slot);
        self.drivers[slot] = Driver::External;
        self.vs.flags[slot] |= FLAG_BRAIN;
        true
    }

    /// spawn_clearance: the first route index whose edge leaves at least spawn_clearance_m between its spawn point
    /// (s_exit) and the junction entry the route takes next (the earliest s_in over the lanes); the last edge when
    /// none does.
    pub fn spawn_edge_index(&self, route: &[i32]) -> usize {
        let g = &*self.g;
        let Some(j) = &g.j else { return 0 };
        let mut k = 0;
        while k + 1 < route.len() {
            let e = route[k] as usize;
            if !(g.e_s_stop[e] < g.e_length[e]) { return k; }
            let s0 = if g.e_s_exit[e] > 0.0 { g.e_s_exit[e] } else { 0.0 };
            let mut s_in = f64::INFINITY;
            for l in 0..g.e_lanes[e] { let m = j.movement(e, l, route[k + 1]); if m >= 0 { s_in = s_in.min(j.s_in(m as usize)); } }
            if !s_in.is_finite() || s_in - s0 >= self.opts.spawn_clearance_m { return k; }
            k += 1;
        }
        k
    }

    pub(crate) fn ensure_slot(&mut self, slot: usize) {
        if self.drivers.len() <= slot { self.drivers.resize(slot + 1, Driver::None); }
        if self.actions.len() <= slot { self.actions.resize(slot + 1, None); }
    }

    // ---------------------------------------------------------------- routes
    pub(crate) fn set_route(&mut self, i: usize, edges: &[i32], route_pos: i32) {
        let off = self.routes.add(edges, &mut self.vs);
        let r = &mut self.vs.routes[i]; r.clear(); r.extend_from_slice(edges);
        self.vs.route_off[i] = off; self.vs.route_len[i] = edges.len() as i32; self.vs.route_pos[i] = route_pos;
        let mut l = 0.0; for &e in edges { l += self.g.e_length[e as usize]; }
        self.vs.route_total[i] = l as f32;
    }

    // ---------------------------------------------------------------- drivers
    /// steerCommand(vs, i, cmd, dt)
    fn steer_command(&self, i: usize, cmd: &Action, dt: f64) -> f64 {
        let vs = &self.vs;
        if vs.position_mode {
            if let (Some(s), None) = (cmd.steer, cmd.steer_rate) { return jm::or(s, 0.0); }
            let o = (i * CTRL_RING + (((vs.cmd_step[i] as i64 % CTRL_RING as i64) + CTRL_RING as i64) % CTRL_RING as i64) as usize) * 3;
            let t = f(vs.ctrl_ring[o]) + jm::or(cmd.steer_rate.unwrap_or(0.0), 0.0) * f(vs.max_steer_rate[i]) * dt / f(vs.max_steer[i]);
            return if t > 1.0 { 1.0 } else if t < -1.0 { -1.0 } else { t };
        }
        if let Some(r) = cmd.steer_rate { return jm::or(r, 0.0); }
        (jm::or(cmd.steer.unwrap_or(0.0), 0.0) * f(vs.max_steer[i]) - f(vs.steer_angle[i])) / (f(vs.max_steer_rate[i]) * 0.4)
    }

    /// Prop driver step (placeObstacles slow / braking): may advance the obstacle rng.
    fn prop_step(&mut self, i: usize) -> Action {
        let vt = f(self.vs.max_speed[i]);
        let c = self.prop_control(i, vt);
        match self.drivers[i] {
            Driver::PropBraking { mut next, mut until } => {
                if self.time >= next {
                    let r = self.obstacle_rng.as_mut().unwrap();
                    until = self.time + r.range(1.5, 3.0);
                    next = until + r.range(self.brake_every.0, self.brake_every.1);
                }
                self.drivers[i] = Driver::PropBraking { next, until };
                if self.time < until { return Action { steer: c.steer, steer_rate: None, gas: 0.0, brake: 1.0 }; }
                c
            }
            _ => c,
        }
    }

    fn prop_control(&self, i: usize, v_t: f64) -> Action {
        let vs = &self.vs;
        let e = vs.edge[i] as usize;
        let lw = self.g.lane_w(e, f(vs.s[i]));
        let off = f(vs.lat[i]) - (vs.lane[i] as f64 + 0.5) * lw;
        let steer = jm::max(-1.0, jm::min(1.0, (-0.08 * off - 1.2 * f(vs.heading_rel[i])) / jm::max(0.1, f(vs.max_steer[i]))));
        let dv = jm::min(v_t, self.g.e_speed[e]) - f(vs.speed[i]);
        Action { steer: Some(steer), steer_rate: None, gas: if dv > 0.0 { jm::min(1.0, 0.3 * dv) } else { 0.0 }, brake: if dv < -1.0 { jm::min(1.0, -0.1 * dv) } else { 0.0 } }
    }

    /// One driver.step() for slot i (env external driver / prop): the command, or None (no driver).
    pub(crate) fn driver_cmd(&mut self, i: usize) -> Option<Action> {
        match self.drivers[i] {
            Driver::None => None,
            Driver::External => Some(self.actions[i].unwrap_or(self.opts.default_action)),
            Driver::PropSlow | Driver::PropBraking { .. } => Some(self.prop_step(i)),
        }
    }

    fn drive(&mut self, dt: f64) {
        let high = self.vs.high;
        if let Some(mut hook) = self.batch_policy.take() {
            let slots: Vec<usize> = (0..high).filter(|&i| self.vs.state[i] == STATE_DRIVING && self.drivers[i] == Driver::External).collect();
            let mut acts = std::mem::take(&mut self.actions);
            hook(self, &slots, &mut acts);
            self.actions = acts;
            self.batch_policy = Some(hook);
        }
        for i in 0..high {
            if self.vs.state[i] != STATE_DRIVING { continue; }
            if let Some(cmd) = self.driver_cmd(i) {
                self.vs.flags[i] &= !FLAG_COLLIDED;
                let s = self.steer_command(i, &cmd, dt);
                let step = self.step_count;
                self.vs.command(i, s, jm::or(cmd.gas, 0.0), jm::or(cmd.brake, 0.0), step);
                continue;
            }
            let c = self.opts.default_action;
            let step = self.step_count;
            self.vs.command(i, c.steer.unwrap_or(0.0), c.gas, c.brake, step);
            self.vs.flags[i] &= !FLAG_COLLIDED;
        }
    }

    pub fn signals_for(&self, i: usize) -> Signals {
        let vs = &self.vs;
        let c = vs.commuter[i];
        Signals {
            collision: vs.flags[i] & FLAG_COLLIDED != 0, stranded: vs.flags[i] & FLAG_STRANDED != 0, harsh_braking: vs.accel[i] < -4.0,
            time_to_work_start: if c >= 0 { if vs.leg[i] == 0 { self.pop.work_start[c as usize] - self.time } else { 1e5 } } else { 0.0 },
            remaining_route_distance: jm::max(0.0, f(vs.route_total[i]) - f(vs.route_dist[i])),
            fuel_fraction: if vs.tank[i] > 0.0 { f(vs.fuel[i]) / f(vs.tank[i]) } else { 1.0 },
            refuel_planned: vs.refuel_state[i] == 1,
        }
    }

    // ---------------------------------------------------------------- motion
    fn integrate_all(&mut self, dt: f64) {
        let g = self.gw();
        let high = self.vs.high;
        let fuel_on = self.opts.fuel;
        // phase A (pure per-car physics + pose), in parallel when enabled; single-threaded with 4 cars' vehicle dynamics
        // interleaved (integrate_car4). Identical results: each car reads only its own slot; written back in slot order.
        // Results are kept dense, in slot order (self.phase_a, reused across steps).
        let ta = std::time::Instant::now();
        let par = self.opts.parallel && high >= 512 && self.vs.missed_t.len() + high <= 100000;
        let batch = !par && self.opts.batch_dynamics && high >= 2 && self.vs.missed_t.len() + high <= 100000;
        let mut buf = std::mem::take(&mut self.phase_a);
        buf.clear();
        {
            let vs = &self.vs;
            let (time, step, pm) = (self.time, self.step_count, vs.position_mode);
            let eligible = |i: usize| -> Option<bool> {
                let st = vs.state[i];
                if st == STATE_DRIVING { Some(false) } else if st == STATE_CRASHED && (vs.speed[i].abs() > 0.01 || jm::abs(f(vs.v_lat[i])) > 0.01 || jm::abs(f(vs.yaw_rate[i])) > 0.01) { Some(true) } else { None }
            };
            let ring = |i: usize| &vs.ctrl_ring[i * 120..(i + 1) * 120];
            let start = |i: usize, locked: bool| { let mut c = vs.gather(i); if !locked && !fuel_on { c.fuel = vs.tank[i]; } PhaseA { slot: i as u32, locked, c, out: CarOut::default(), ret: 0, pose: Pose::default() } };
            let one = |e: &mut PhaseA| { let i = e.slot as usize; e.ret = VehicleStore::integrate_car(&mut e.c, &vs.routes[i], ring(i), vs.missed_t.get(&i).copied(), pm, &mut e.out, dt, time, &g, step, e.locked, vs.model); };
            if par {
                use rayon::prelude::*;
                // eligible slots first (cheap), then an indexed parallel map (collects in place, no concatenation)
                let elig: Vec<(usize, bool)> = (0..high).filter_map(|i| eligible(i).map(|l| (i, l))).collect();
                elig.into_par_iter().with_min_len(64).map(|(i, l)| { let mut e = start(i, l); one(&mut e); e.pose = Pose::of(&e.c, &g); e }).collect_into_vec(&mut buf);
            } else if batch {
                for i in 0..high { if let Some(l) = eligible(i) { buf.push(start(i, l)); } }
                let width = if std::env::var("BAYENV_BATCH2").is_ok() { 2 } else { 4 };
                for ch in buf.chunks_mut(width) {
                    match ch {
                        [a, b, c, d] => {
                            let (ia, ib, ic, id) = (a.slot as usize, b.slot as usize, c.slot as usize, d.slot as usize);
                            let r = VehicleStore::integrate_car4([
                                (&mut a.c, &vs.routes[ia], ring(ia), vs.missed_t.get(&ia).copied(), &mut a.out, a.locked), (&mut b.c, &vs.routes[ib], ring(ib), vs.missed_t.get(&ib).copied(), &mut b.out, b.locked),
                                (&mut c.c, &vs.routes[ic], ring(ic), vs.missed_t.get(&ic).copied(), &mut c.out, c.locked), (&mut d.c, &vs.routes[id], ring(id), vs.missed_t.get(&id).copied(), &mut d.out, d.locked)], pm, dt, time, &g, step, vs.model);
                            a.ret = r[0]; b.ret = r[1]; c.ret = r[2]; d.ret = r[3];
                        }
                        [a, b, rest @ ..] => {
                            let (ia, ib) = (a.slot as usize, b.slot as usize);
                            let (x, y) = VehicleStore::integrate_car2((&mut a.c, &vs.routes[ia], ring(ia), vs.missed_t.get(&ia).copied(), &mut a.out, a.locked), (&mut b.c, &vs.routes[ib], ring(ib), vs.missed_t.get(&ib).copied(), &mut b.out, b.locked), pm, dt, time, &g, step, vs.model);
                            a.ret = x; b.ret = y;
                            for e in rest { one(e); }
                        }
                        rest => { for e in rest { one(e); } }
                    }
                }
                for e in buf.iter_mut() { e.pose = Pose::of(&e.c, &g); }
            }
        }
        // stuck_cause (labels a car stuck past stuck_timeout) reads the cars ahead, which phase A has already
        // integrated: in a step where a label may be computed, use the in-order path instead (exact; rare)
        let mut have_pre = par || batch;
        if have_pre && buf.iter().any(|e| e.ret == 0 && !e.locked && e.c.stopped_since >= 0.0 && self.time - f(e.c.stopped_since) > self.opts.stuck_timeout
                && self.stuck_labelled.get(&(e.slot as usize)) != Some(&e.c.stopped_since)) {
            buf.clear(); have_pre = false;
        }
        if have_pre {
            // write the per-car results back (distinct slots; save_pose first: px = x before the pose update)
            let ptrs = crate::vehicle::CarPtrs::new(&mut self.vs);
            let (ppx, ppz, ppy) = (SyncPtr(self.vs.px.as_mut_ptr()), SyncPtr(self.vs.pz.as_mut_ptr()), SyncPtr(self.vs.pyaw.as_mut_ptr()));
            // SAFETY: one write per slot < high; the arrays were not reallocated since the pointers were taken
            let wr = |e: &PhaseA| unsafe { let i = e.slot as usize; *ppx.get().add(i) = e.c.x; *ppz.get().add(i) = e.c.z; *ppy.get().add(i) = e.c.heading; ptrs.write(i, &e.c); };
            if par { use rayon::prelude::*; buf.par_iter().with_min_len(64).for_each(wr); } else { buf.iter().for_each(wr); }
        }
        if self.profile { self.sub_ns[1] += ta.elapsed().as_nanos() as u64; }
        let (mut driving, mut crashed, mut refueling, mut sum_speed) = (0u64, 0u64, 0u64, 0.0f64);
        let (mut speeding_sec, mut speeding_km, mut total_sec, mut total_km) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let check_every = self.opts.fuel_cfg.check_interval;
        let mut cur = 0usize;
        macro_rules! pose { ($e:expr, $i:expr) => { match $e { Some(e) => { let pc = &e.pose; let v = &mut self.vs; v.x[$i] = pc.x; v.y[$i] = pc.y; v.z[$i] = pc.z; v.heading[$i] = pc.heading; v.seg[$i] = pc.seg; } None => self.vs.update_pose($i, &g) } }; }
        for i in 0..high {
            // this slot's phase-A result (buf is in slot order)
            while cur < buf.len() && (buf[cur].slot as usize) < i { cur += 1; }
            let res: Option<&PhaseA> = if have_pre { buf.get(cur).filter(|e| e.slot as usize == i) } else { None };
            let st = self.vs.state[i];
            if st == STATE_CRASHED {
                crashed += 1;
                if have_pre {
                    match res { Some(e) => { self.vs.scatter_side(i, &e.out); pose!(res, i); } None => self.vs.save_pose(i) }
                    continue;
                }
                let vs = &mut self.vs;
                if vs.speed[i].abs() > 0.01 || jm::abs(f(vs.v_lat[i])) > 0.01 || jm::abs(f(vs.yaw_rate[i])) > 0.01 {
                    vs.save_pose(i); vs.integrate(i, dt, self.time, &g, self.step_count, true); vs.update_pose(i, &g);
                } else { vs.save_pose(i); }
                continue;
            }
            if st == STATE_REFUELING { refueling += 1; continue; }
            if st != STATE_DRIVING { continue; }
            let mut done;
            if have_pre {
                let e = res.unwrap();
                self.vs.scatter_side(i, &e.out);
                done = e.ret;
            } else {
                if !fuel_on { self.vs.fuel[i] = self.vs.tank[i]; }
                self.vs.save_pose(i);
                done = self.vs.integrate(i, dt, self.time, &g, self.step_count, false);
            }
            let e = self.vs.edge[i] as usize;
            let lim = g.e_speed[e];
            let sp = f(self.vs.speed[i]);
            if sp > 1.1 * lim { speeding_sec += dt; speeding_km += sp * dt / 1000.0; self.vs.flags[i] |= FLAG_SPEEDING; } else { self.vs.flags[i] &= !FLAG_SPEEDING; }
            total_sec += dt; total_km += sp * dt / 1000.0;
            if done == 0 {
                let rp = self.vs.route_pos[i] as usize;
                let rem = g.e_length[e] - f(self.vs.s[i]);
                let stop_here = self.vs.refuel_state[i] == 1 && self.vs.stop_pos[i] == rp as i32;
                if (rp + 1 >= self.vs.routes[i].len() || stop_here) && (rem < 2.5 || (rem < 8.0 && sp < 0.3)) { done = if stop_here { 2 } else { 1 }; }
            }
            if done == 1 { self.arrive(i); continue; }
            if done == 2 { pose!(res, i); self.enter_station(i); refueling += 1; continue; }
            if self.vs.fuel[i] <= 0.0 && sp < 0.05 { pose!(res, i); self.strand(i); crashed += 1; continue; }
            let ss = self.vs.stopped_since[i];
            if ss >= 0.0 && self.time - f(ss) > self.opts.stuck_timeout && self.stuck_labelled.get(&i) != Some(&ss) {
                self.stuck_labelled.insert(i, ss);
                self.counters.stuck += 1;
                let k = self.stuck_cause(i);
                *self.counters.stuck_by.entry(k).or_insert(0) += 1;
            }
            if self.stations.is_some() && self.time - f(self.vs.last_fuel_check[i]) > check_every { let tf = std::time::Instant::now(); self.check_fuel(i, false); if self.profile { self.sub_ns[7] += tf.elapsed().as_nanos() as u64; } }
            pose!(res, i);
            driving += 1; sum_speed += f(self.vs.speed[i]);
        }
        self.live = Live { driving, crashed, refueling, sum_speed };
        let a = &mut self.accum;
        a.vehicle_sec += total_sec; a.vehicle_km += total_km; a.speeding_sec += speeding_sec; a.speeding_km += speeding_km;
        self.phase_a = buf;
    }

    pub(crate) fn set_outcome(&mut self, cid: usize, leg: u8, o: u8) {
        if leg == 0 { if self.outcome[cid] == FAIL_PENDING { self.outcome[cid] = o; } } else if self.outcome_back[cid] == FAIL_PENDING { self.outcome_back[cid] = o; }
    }

    pub(crate) fn arrive(&mut self, i: usize) {
        let c = self.vs.commuter[i];
        if c < 0 { self.release(i); return; }
        let c = c as usize;
        let leg = self.vs.leg[i];
        let late = if leg == 0 { self.time - self.pop.work_start[c] } else { 0.0 };
        self.events.arrived.push(ArrivedEv { slot: i, agent: c as i32, leg, late_by: late });
        self.emit(EV_ARRIVED, i, -1, f32::NAN, f32::NAN, f32::NAN, late as f32);
        self.counters.arrived += 1;
        self.commuter_fuel[c] = self.vs.fuel[i];
        self.set_outcome(c, leg, FAIL_ARRIVED);
        if leg == 0 {
            self.counters.arrived_outbound += 1;
            self.actual_arrival[c] = self.time;
            self.late_by[c] = jm::max(0.0, late); self.early_by[c] = jm::max(0.0, -late);
            if late <= 0.0 { self.counters.on_time += 1; }
            self.accum.late_sum += self.late_by[c]; self.accum.late_n += 1.0;
        }
        self.fitness[c] = self.fitness[c] + 10.0 + if leg == 0 { if late <= 0.0 { 2.0 } else { jm::max(-10.0, -0.5 * late / 60.0) } } else { 0.0 };
        self.release(i);
    }

    pub(crate) fn fail_trip(&mut self, i: usize, code: u8, remove: bool) {
        if code == FAIL_STUCK { self.counters.stuck += 1; let k = self.stuck_cause(i); *self.counters.stuck_by.entry(k).or_insert(0) += 1; }
        let c = self.vs.commuter[i];
        if c >= 0 {
            let c = c as usize;
            self.commuter_fuel[c] = jm::max(f(self.vs.fuel[i]), if code == FAIL_STRANDED { 0.2 * f(self.vs.tank[i]) } else { 0.0 }) as f32;
            self.set_outcome(c, self.vs.leg[i], code);
            self.fitness[c] -= 20.0;
        }
        if remove { self.release(i); }
    }

    pub(crate) fn release(&mut self, i: usize) {
        if self.vs.state[i] == STATE_REFUELING && self.stations.is_some() && self.vs.station[i] >= 0 {
            let k = self.vs.station[i] as usize;
            let st = self.stations.as_mut().unwrap();
            if self.vs.refuel_state[i] == 3 { st.busy[k] -= 1; }
            if let Some(p) = st.queue[k].iter().position(|&q| q == i) { st.queue[k].remove(p); }
        }
        self.drivers[i] = Driver::None;
        self.vs.release(i);
    }

    // ---------------------------------------------------------------- stats
    pub fn arrival_rate(&self) -> f64 { let n = self.pop.n.max(1); self.outcome.iter().filter(|&&o| o == FAIL_ARRIVED).count() as f64 / n as f64 }
}
