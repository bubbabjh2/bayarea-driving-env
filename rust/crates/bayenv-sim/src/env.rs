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
