//! src/sim/env.js TrainingEnv: multi-agent API over Sim (every car is an agent).
//!   let mut env = Env::new(world, pop, EnvConfig { .. }, router_factory);
//!   loop { for slot in env.driving() { env.act(slot, Action { .. }) }  env.step(1); }
use crate::population::Population;
use crate::sim::{Action, Router, Sim, SimOpts, TrafficMode};
use crate::vehicle::STATE_DRIVING;
use crate::world::World;
use std::sync::Arc;

pub type RouterFactory = Box<dyn Fn(Arc<World>) -> Box<dyn Router> + Send + Sync>;

/// Static obstacle (scenario.json obstacles / staticObstacles): edge, at, lane, lat, type.
#[derive(Clone, Copy, Debug)]
pub struct StaticObstacle { pub edge: i32, pub at: f64, pub lane: Option<i32>, pub lat: Option<f64>, pub type_idx: u8 }

#[derive(Clone)]
pub struct EnvConfig {
    pub opts: SimOpts,
    /// None: env.js defaultStart(pop) = floor(min depart / 60) * 60 - 60
    pub start_time: Option<f64>,
    pub static_obstacles: Vec<StaticObstacle>,
    /// scenarios/load.js placeScenarioObstacles (re-placed on every reset)
    pub scenario_obstacles: Vec<StaticObstacle>,
}

pub struct Env { pub world: Arc<World>, pub pop: Arc<Population>, pub cfg: EnvConfig, pub sim: Sim, router: RouterFactory }

pub fn default_start(pop: &Population) -> f64 {
    let mut t = f64::INFINITY;
    for &d in &pop.depart { if d < t { t = d; } }
    (if t.is_finite() { t } else { 5.0 * 3600.0 } / 60.0).floor() * 60.0 - 60.0
}

impl Env {
    pub fn new(world: Arc<World>, pop: Arc<Population>, cfg: EnvConfig, router: RouterFactory) -> Env {
        let sim = Self::make_sim(&world, &pop, &cfg, &router, None);
        Env { world, pop, cfg, sim, router }
    }
    fn make_sim(world: &Arc<World>, pop: &Arc<Population>, cfg: &EnvConfig, router: &RouterFactory, seed: Option<f64>) -> Sim {
        let mut o = cfg.opts.clone();
        if let Some(s) = seed { o.seed = s; }
        o.start_time = cfg.start_time.unwrap_or_else(|| default_start(pop));
        if o.traffic_mode != TrafficMode::Sparse { o.wrecks = 0; }
        let mut sim = Sim::new(world.clone(), pop.clone(), o, router(world.clone()));
        let so: Vec<_> = cfg.static_obstacles.iter().map(|o| (o.edge, o.at, o.lane, o.lat, o.type_idx)).collect();
        if !so.is_empty() { sim.place_static_obstacles(&so, false); }
        let sc: Vec<_> = cfg.scenario_obstacles.iter().map(|o| (o.edge, o.at, o.lane, o.lat, o.type_idx)).collect();
        if !sc.is_empty() { sim.place_static_obstacles(&sc, true); }
        sim
    }
    pub fn reset(&mut self, seed: Option<f64>) { self.sim = Self::make_sim(&self.world, &self.pop, &self.cfg, &self.router, seed); }
    /// Slots of every driving car (the agents that act this step).
    pub fn driving(&self) -> impl Iterator<Item = usize> + '_ { (0..self.sim.vs.high).filter(|&i| self.sim.vs.state[i] == STATE_DRIVING) }
    /// env.act(slot, {steer | steerRate, gas, brake}): command for the coming step(s), delayed by reaction time.
    pub fn act(&mut self, slot: usize, a: Action) {
        let mut c = Action { gas: crate::jm::or(a.gas, 0.0), brake: crate::jm::or(a.brake, 0.0), steer: a.steer.map(|s| crate::jm::or(s, 0.0)), steer_rate: a.steer_rate.map(|s| crate::jm::or(s, 0.0)) };
        if c.steer.is_none() && c.steer_rate.is_none() { c.steer = Some(0.0); }
        self.sim.ensure_slot(slot);
        self.sim.actions[slot] = Some(c);
    }
    /// env.step(k): k sim steps, then the commands are cleared.
    pub fn step(&mut self, k: usize) { for _ in 0..k { self.sim.step(); } for a in self.sim.actions.iter_mut() { *a = None; } }
    pub fn outcomes(&self) -> &[u8] { &self.sim.outcome }
}

/// Python-facing (bayenv-py `Env`) helpers: the same shapes as the GPU backend. Rows = slots 0..n_slots().
impl Env {
    pub fn n_slots(&self) -> usize { self.sim.vs.high }
    pub fn time(&self) -> f64 { self.sim.time }
    pub fn steps(&self) -> u64 { self.sim.step_count as u64 }
    /// act float32 (N, 3) [steer, gas, brake] for slots offset..offset+N (position-mode steer in [-1, 1]).
    pub fn act_rows(&mut self, rows: &[f32], offset: usize) {
        for (r, a) in rows.chunks_exact(3).enumerate() {
            let slot = offset + r;
            if slot >= self.sim.vs.high || self.sim.vs.state[slot] != STATE_DRIVING { continue; }
            self.act(slot, Action { steer: Some(a[0] as f64), steer_rate: None, gas: a[1] as f64, brake: a[2] as f64 });
        }
    }
    /// observe(): float32 (n_slots, obs::flat_stride(v3)); rows of non-driving slots are 0. Mutates perception caches
    /// exactly like the JS env.observe (call it between step() and act()).
    pub fn observe_into(&mut self, v3: bool, out: &mut Vec<f32>) -> usize {
        let stride = crate::obs::flat_stride(v3);
        let n = self.n_slots();
        out.clear(); out.resize(n * stride, 0.0);
        let slots = crate::obs::driving_slots(&self.sim);
        let mut buf = vec![0f32; slots.len() * stride];
        crate::obs::observe_flat(&mut self.sim, &slots, &mut buf, v3);
        for (k, &s) in slots.iter().enumerate() { out[s * stride..(s + 1) * stride].copy_from_slice(&buf[k * stride..(k + 1) * stride]); }
        stride
    }
    /// crash_flags(): crashed (stopped wreck / stranded) or disabled, per slot.
    pub fn crash_flags(&self) -> Vec<bool> {
        let vs = &self.sim.vs;
        (0..vs.high).map(|i| vs.state[i] == crate::vehicle::STATE_CRASHED || vs.flags[i] & crate::vehicle::FLAG_DISABLED != 0).collect()
    }
    /// state(): named per-slot columns (f64 for uniform numpy conversion; the store keeps the JS element types).
    pub fn state_columns(&self) -> Vec<(&'static str, Vec<f64>)> {
        let vs = &self.sim.vs;
        let h = vs.high;
        macro_rules! col { ($($n:literal => $f:ident),*) => { vec![$(($n, (0..h).map(|i| vs.$f[i] as f64).collect::<Vec<f64>>())),*] }; }
        col!("edge" => edge, "state" => state, "flags" => flags, "route_pos" => route_pos, "s" => s, "speed" => speed, "accel" => accel, "lat" => lat,
            "heading_rel" => heading_rel, "steer" => steer_angle, "odometer" => odometer, "fuel" => fuel, "gas" => gas, "brake" => brake, "yaw_rate" => yaw_rate,
            "v_lat" => v_lat, "lane" => lane, "x" => x, "y" => y, "z" => z, "heading" => heading, "commuter" => commuter, "leader" => leader, "gap" => gap)
    }
    /// stats(): counters (same keys as the GPU backend where they exist).
    pub fn stats(&self) -> Vec<(&'static str, f64)> {
        let s = &self.sim; let c = &s.counters; let l = &s.live;
        vec![("driving", l.driving as f64), ("crashed_on_scene", l.crashed as f64), ("arrived", c.arrived as f64), ("spawned", c.spawned as f64), ("stuck", c.stuck as f64),
            ("crash_events", c.crashes as f64), ("crash_scenes", c.disabling_crashes as f64), ("crashed_vehicles", c.crashed_vehicles as f64), ("minor_contacts", c.minor_contacts as f64),
            ("stranded", c.stranded as f64), ("towed", c.towed as f64), ("cleared", c.cleared as f64), ("missed_exits", c.missed_exits as f64), ("reroutes", c.reroutes as f64),
            ("signal_violations", c.signal_violations as f64), ("stop_violations", c.stop_violations as f64), ("arrival_rate", s.arrival_rate()),
            ("avg_speed", if l.driving > 0 { l.sum_speed / l.driving as f64 } else { 0.0 }), ("active_scenes", s.active.len() as f64), ("time", s.time)]
    }
}

impl Env {
    /// createTrainingEnv({ density: n, trafficMode, seed }) on the real world in `data_dir`, with no node involved:
    /// bayenv_world::World::load -> World::from_bayenv (share the returned Arc<World> across envs), population as
    /// env.js generates it, dt 0.1, the external-action driver for every car, bayenv_world::nav routing.
    pub fn create_real(world: Arc<World>, data_dir: &std::path::Path, n: usize, seed: f64, traffic_mode: TrafficMode) -> Result<Env, String> {
        let pop = Arc::new(Population::generate_like_env(&world, "real", n, seed, Some(data_dir))?);
        let opts = SimOpts { dt: 0.1, seed, traffic_mode, wrecks: if traffic_mode == TrafficMode::Sparse { 40 } else { 0 }, ..SimOpts::default() };
        let cfg = EnvConfig { opts, start_time: None, static_obstacles: Vec::new(), scenario_obstacles: Vec::new() };
        Ok(Env::new(world, pop, cfg, Box::new(|w| crate::harness::router_for(&w))))
    }
    /// One-call world load for create_real.
    pub fn load_world(data_dir: &std::path::Path) -> Result<Arc<World>, String> {
        let bw = bayenv_world::World::load(data_dir).map_err(|e| e.to_string())?;
        Ok(Arc::new(World::from_bayenv(&bw)))
    }
}
