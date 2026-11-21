//! src/sim/sim.js Simulation: clock, spawning, perception (leaders, lane ends, junctions, controls),
//! drivers -> reaction ring, integration, rigid-body contacts, crash scenes, fuel, re-routing.
//! Split over sim.rs (state, step, spawn, integrate, trips), leaders.rs, junction.rs, contacts.rs, extra.rs.
use crate::jm;
use crate::physics::VEHICLE_PARAMS;
use crate::population::Population;
use crate::rng::Rng;
use crate::spatial::{LaneIndex, SpatialHash};
use crate::vehicle::*;
use crate::world::{Conflict, World};
use crate::fx::{HashMap, HashSet};
use std::sync::Arc;

pub const FAIL_PENDING: u8 = 0;
pub const FAIL_ARRIVED: u8 = 1;
pub const FAIL_CRASH: u8 = 2;
pub const FAIL_STRANDED: u8 = 3;
pub const FAIL_STUCK: u8 = 4;
pub const FAIL_NOROUTE: u8 = 5;
pub const MERGE_ROOM: f64 = 15.0;

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
            seed: 1.0, wrecks: 0, default_action: Action { steer: Some(0.0), steer_rate: None, gas: 0.0, brake: 0.3 }, fuel_cfg: Default::default(), slot_pool: 0, parallel: false }
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
    pub stuck: u64, pub no_route: u64, pub cleared: u64, pub secondary_crashes: u64, pub minor_contacts: u64, pub touches: u64, pub disabling_crashes: u64,
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
pub trait Router: Send {
    fn route(&mut self, start: i32, goal: i32, avoid: Option<&HashSet<i32>>, penalty: Option<&HashMap<i32, f64>>) -> Option<Vec<i32>>;
}

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
    pub(crate) pair_event: HashMap<i64, f64>,
    pub(crate) stuck_labelled: HashMap<usize, f32>,
    pub(crate) conn_cars: crate::fx::ListMap<usize>,
    pub(crate) node_conn: crate::fx::ListMap<usize>,
    pub(crate) passers: crate::fx::ListMap<i32>,
    pub(crate) conflict_with: HashMap<usize, usize>,
    pub(crate) off_lane: crate::fx::ListMap<i32>,
    pub(crate) touch: HashSet<i64>,
    pub(crate) barrier_touch: HashSet<usize>, pub(crate) barrier_now: HashSet<usize>,
    pub(crate) conf_cache: HashMap<usize, Arc<Vec<Conflict>>>,
    pub(crate) mv_into: Option<HashMap<u32, Vec<usize>>>,
    pub(crate) yield_by: Option<Vec<i32>>,
    pub(crate) creep_until: Vec<f64>, pub(crate) creep_ignore: Vec<i32>,
    pub(crate) edge_speed: HashMap<i32, f64>,
    pub traffic_penalty: Option<HashMap<i32, f64>>,
    pub(crate) last_reroute: HashMap<usize, f64>,
    pub(crate) next_traffic: f64, pub(crate) next_reroute: f64, pub(crate) next_density: f64,
    pub(crate) edge_count_scratch: Vec<u16>, pub(crate) fwy: Vec<usize>, pub(crate) fwy_lane_km: f64,
    pub(crate) seen_j: i32,
    pub(crate) near_buf: Vec<usize>,
    pub static_obstacles: Vec<usize>,
    pub obstacles: Vec<(usize, &'static str, i32, f64)>,
    /// per-phase wall time (ns) when `profile` is set: spawn, hash, contacts, lanes, leaders, controls, drive, integrate, scenes, traffic+reroute+density
    pub profile: bool,
    pub phase_ns: [u64; 10],
}

#[inline(always)] pub(crate) fn f(x: f32) -> f64 { x as f64 }
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
        let mut vs = VehicleStore::new(cap);
        vs.position_mode = opts.position_mode;
        let ne = g.edge_count;
        let mut fwy = Vec::new();
        let mut fwy_lane_km = 0.0;
        for e in 0..ne { if g.e_hw[e] & crate::world::HW_MAINLINE != 0 && g.e_length[e] >= 100.0 { fwy.push(e); fwy_lane_km += g.e_length[e] * g.e_lanes[e] as f64 / 1000.0; } }
        let stations = if opts.fuel && !g.stations.is_empty() { Some(crate::fuel::StationNetwork::new(&g, opts.fuel_cfg.clone())) } else { None };
        let start = opts.start_time;
        let mut sim = Sim {
            hash: SpatialHash::new(opts.cell_size, opts.table_bits, cap), lanes: LaneIndex::new(ne, cap), drivers: vec![Driver::None; cap], actions: vec![None; cap],
            routes: crate::fuel::RouteStore::new(&pop), stations, learners, outcome: vec![0; n_c], outcome_back: vec![0; n_c], commuter_fuel,
            fitness: vec![0.0; n_c], actual_arrival: vec![-1.0; n_c], late_by: vec![0.0; n_c], early_by: vec![0.0; n_c],
            trips, trip_ptr: 0, pending: Vec::new(), lot: Vec::new(), crashes: Vec::new(), active: Vec::new(), edge_crashes: HashMap::default(), edge_has_crash: vec![0; ne],
            counters: Counters::default(), accum: Accum::default(), live: Live::default(), density: Density::default(), obstacle_rng: None, brake_every: (15.0, 40.0),
            spawned_keys: HashSet::default(), pair_event: HashMap::default(), stuck_labelled: HashMap::default(), conn_cars: Default::default(), node_conn: Default::default(),
            passers: Default::default(), conflict_with: HashMap::default(), off_lane: Default::default(), touch: HashSet::default(), barrier_touch: HashSet::default(), barrier_now: HashSet::default(),
            conf_cache: HashMap::default(), mv_into: None, yield_by: None, creep_until: Vec::new(), creep_ignore: Vec::new(), edge_speed: HashMap::default(), traffic_penalty: None,
            last_reroute: HashMap::default(), next_traffic: 0.0, next_reroute: 0.0, next_density: start, edge_count_scratch: vec![0; ne], fwy, fwy_lane_km, seen_j: -1,
            near_buf: Vec::new(), static_obstacles: Vec::new(), obstacles: Vec::new(), profile: std::env::var("BAYENV_PROFILE").is_ok(), phase_ns: [0; 10],
            empty: opts.traffic_mode == TrafficMode::Empty, time: start, dt: opts.dt, step_count: 0, vs, rng, router, g, pop, opts,
        };
        if sim.opts.traffic_mode == TrafficMode::Sparse && sim.opts.wrecks > 0 { sim.place_wrecks(sim.opts.wrecks); }
        sim
    }

    pub fn is_idle(&self) -> bool { self.vs.count == 0 && self.pending.is_empty() && self.trip_ptr >= self.trips.len() }

    // ---------------------------------------------------------------- step
    pub fn step(&mut self) {
        let dt = self.dt;
        self.time += dt;
        self.step_count += 1;
        let prof = self.profile;
        let mut t = if prof { Some(std::time::Instant::now()) } else { None };
        macro_rules! lap { ($k:expr) => { if let Some(t0) = t { let n = std::time::Instant::now(); self.phase_ns[$k] += (n - t0).as_nanos() as u64; t = Some(n); } }; }
        self.spawn(); lap!(0);
        self.hash.rebuild(&self.vs); lap!(1);
        self.contacts(); lap!(2);
        self.lanes.rebuild(&self.vs); lap!(3);
        self.find_leaders(); lap!(4);
        if self.g.controls.is_some() { self.control_obs(); } lap!(5);
        if let Some(yb) = &mut self.yield_by { yb.iter_mut().for_each(|x| *x = -1); }
        self.drive(dt); lap!(6);
        self.integrate_all(dt); lap!(7);
        if !self.vs.missed.is_empty() { self.missed_exits(); }
        self.scenes(); lap!(8);
        if self.time >= self.next_traffic { self.traffic_sample(); self.next_traffic = self.time + 60.0; }
        if self.opts.reroute && self.time >= self.next_reroute { self.reroute_blocked(); self.next_reroute = self.time + 2.0; }
        if self.time >= self.next_density { self.sample_density(); self.next_density = self.time + self.opts.density_interval; }
        lap!(9);
    }

    // ---------------------------------------------------------------- spawning
    fn spawn(&mut self) {
        self.spawned_keys.clear();
        if !self.lot.is_empty() { self.leave_stations(); }
        if !self.pending.is_empty() {
            let list = std::mem::take(&mut self.pending);
            let mut keep = Vec::new();
            for t in list { if !self.try_spawn(t) { keep.push(t); } }
            self.pending = keep;
        }
        while self.trip_ptr < self.trips.len() && self.trips[self.trip_ptr].depart <= self.time {
            let t = self.trips[self.trip_ptr];
            self.trip_ptr += 1;
            if !self.try_spawn(t) { self.pending.push(t); }
        }
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

    fn try_spawn(&mut self, trip: Trip) -> bool {
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
        let mut lane = self.entry_lane(route[0] as usize, 0, vlen, if route.len() > 1 { route[1] } else { -1 });
        if lane < 0 {
            if self.time - trip.depart < self.opts.spawn_blocked_timeout { return false; }
            lane = 0; self.counters.forced_spawns += 1;
        }
        self.spawned_keys.insert(route[0] as i64 * 8 + lane as i64);
        let learner = self.learners[c] == 1;
        let traits = self.traits_of(c);
        let g = self.gw();
        let slot = self.vs.alloc(&g, AllocArgs { type_idx: pop.vehicle_type[c], traits, route, commuter: c as i32, leg: trip.leg, time: self.time, learner, dt: self.dt, step: self.step_count, lane });
        self.vs.set_reaction_time(slot, traits.reaction_time, self.dt);
        self.vs.route_off[slot] = self.routes.pop_offset(k);
        self.vs.fuel[slot] = jm::min(f(self.vs.tank[slot]), f(self.commuter_fuel[c])) as f32;
        self.vs.update_pose(slot, &g); self.vs.save_pose(slot);
        self.counters.spawned += 1;
        if self.stations.is_some() { self.check_fuel(slot, true); }
        self.ensure_slot(slot);
        self.drivers[slot] = Driver::External;
        self.vs.flags[slot] |= FLAG_BRAIN;
        true
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
        // phase A (pure per-car physics + pose), in parallel when enabled: identical results, written back in slot order
        let mut pre: Vec<Option<(Car, CarOut, i32, Car)>> = Vec::new();
        let par = self.opts.parallel && high >= 512 && self.vs.missed_t.len() + high <= 100000;
        if par {
            use rayon::prelude::*;
            let vs = &self.vs;
            let (time, step, pm) = (self.time, self.step_count, vs.position_mode);
            pre = (0..high).into_par_iter().with_min_len(64).map(|i| {
                let st = vs.state[i];
                let locked = if st == STATE_DRIVING { false } else if st == STATE_CRASHED && (vs.speed[i] > 0.01 || jm::abs(f(vs.v_lat[i])) > 0.01 || jm::abs(f(vs.yaw_rate[i])) > 0.01) { true } else { return None };
                let mut c = vs.gather(i);
                if !locked && !fuel_on { c.fuel = vs.tank[i]; }
                let mut out = CarOut::default();
                let ret = VehicleStore::integrate_car(&mut c, &vs.routes[i], &vs.ctrl_ring[i * 120..(i + 1) * 120], vs.missed_t.get(&i).copied(), pm, &mut out, dt, time, &g, step, locked);
                let mut pc = c; pc.update_pose(&g);
                Some((c, out, ret, pc))
            }).collect();
        }
        let (mut driving, mut crashed, mut refueling, mut sum_speed) = (0u64, 0u64, 0u64, 0.0f64);
        let (mut speeding_sec, mut speeding_km, mut total_sec, mut total_km) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let check_every = self.opts.fuel_cfg.check_interval;
        macro_rules! pose { ($i:expr) => { match pre.get($i).and_then(|p| p.as_ref()) { Some((_, _, _, pc)) => { let v = &mut self.vs; v.x[$i] = pc.x; v.y[$i] = pc.y; v.z[$i] = pc.z; v.heading[$i] = pc.heading; v.seg[$i] = pc.seg; } None => self.vs.update_pose($i, &g) } }; }
        for i in 0..high {
            let st = self.vs.state[i];
            if st == STATE_CRASHED {
                crashed += 1;
                if par {
                    self.vs.save_pose(i);
                    if let Some((c, out, _, _)) = &pre[i] { self.vs.scatter(i, c, out); pose!(i); }
                    continue;
                }
                let vs = &mut self.vs;
                if vs.speed[i] > 0.01 || jm::abs(f(vs.v_lat[i])) > 0.01 || jm::abs(f(vs.yaw_rate[i])) > 0.01 {
                    vs.save_pose(i); vs.integrate(i, dt, self.time, &g, self.step_count, true); vs.update_pose(i, &g);
                } else { vs.save_pose(i); }
                continue;
            }
            if st == STATE_REFUELING { refueling += 1; continue; }
            if st != STATE_DRIVING { continue; }
            let mut done;
            if par {
                self.vs.save_pose(i);
                let (c, out, ret, _) = pre[i].as_ref().unwrap();
                self.vs.scatter(i, c, out);
                done = *ret;
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
            if done == 2 { pose!(i); self.enter_station(i); refueling += 1; continue; }
            if self.vs.fuel[i] <= 0.0 && sp < 0.05 { pose!(i); self.strand(i); crashed += 1; continue; }
            let ss = self.vs.stopped_since[i];
            if ss >= 0.0 && self.time - f(ss) > self.opts.stuck_timeout && self.stuck_labelled.get(&i) != Some(&ss) {
                self.stuck_labelled.insert(i, ss);
                self.counters.stuck += 1;
                let k = self.stuck_cause(i);
                *self.counters.stuck_by.entry(k).or_insert(0) += 1;
            }
            if self.stations.is_some() && self.time - f(self.vs.last_fuel_check[i]) > check_every { self.check_fuel(i, false); }
            pose!(i);
            driving += 1; sum_speed += f(self.vs.speed[i]);
        }
        self.live = Live { driving, crashed, refueling, sum_speed };
        let a = &mut self.accum;
        a.vehicle_sec += total_sec; a.vehicle_km += total_km; a.speeding_sec += speeding_sec; a.speeding_km += speeding_km;
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
