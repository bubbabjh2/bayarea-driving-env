//! Startup cost: World::load (verified), the sim World conversion, one env (population + Sim::new)
use std::time::Instant;
fn main() {
    let data = std::env::args().nth(1).unwrap_or_else(|| "data".into());
    let n: usize = std::env::args().nth(2).map(|s| s.parse().unwrap()).unwrap_or(2000);
    let t = Instant::now(); let w = bayenv_world::World::load(&data).unwrap(); let t_load = t.elapsed().as_secs_f64();
    let t = Instant::now(); let sw = std::sync::Arc::new(bayenv_sim::world::World::from_bayenv(&w)); let t_conv = t.elapsed().as_secs_f64();
    let t = Instant::now(); let pop = std::sync::Arc::new(bayenv_sim::population::Population::generate_like_env(&sw, "real", n, 1.0, Some(std::path::Path::new(&data))).unwrap()); let t_pop = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let cfg = bayenv_sim::env::EnvConfig { opts: bayenv_sim::sim::SimOpts::default(), start_time: Some(25200.0), static_obstacles: Vec::new(), scenario_obstacles: Vec::new(), obstacles: None };
    let _e = bayenv_sim::env::Env::new(sw.clone(), pop, cfg, Box::new(|w| bayenv_sim::harness::router_for(&w)));
    let t_env = t.elapsed().as_secs_f64();
    println!("World::load {t_load:.2}s, sim World::from_bayenv {t_conv:.2}s, population n={n} {t_pop:.2}s, Env::new {t_env:.2}s");
}
