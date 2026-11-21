//! CPU reference sim (bayenv-sim) vs the GPU sim with the default model (GpuOptions.cpu_connector_join = true):
//! rust/tools/compare_backends.py --spawn-default as a test. 5,000 commuters (population.js, seed 42) from 07:00,
//! dt 0.05, 1,200 steps, every driving car commanded [0, 0.3, 0]; GPU slots that are not driving get the CPU's default
//! action [0, 0, 0.3] (a car spawning during a step brakes on its first decision, as on the CPU). Aggregate event
//! counts must agree within tolerances around the measured values (rust/docs/kerb-investigation.md: contacts 17 / 12,
//! crashed 28 / 27, missed exits 120 / 123, violations 1,627 / 1,627; the legacy GPU model gives 1,336 contacts).
//! Skips without the world data or a GPU adapter.
use bayenv_gpu::{defines::ObsMode, sim::{GpuOptions, GpuPopulation}};
use std::{path::Path, sync::Arc};

const N: usize = 5000;
const STEPS: usize = 1200;
const START: f64 = 7.0 * 3600.0;
// event kinds (bayenv_world::events): 1 arrived, 2 crashed, 3 contact, 6 spawned, 7 missed exit, 8 violation
fn counts(log: &bayenv_world::events::EventLog, c: &mut [usize; 10], barrier: &mut usize) {
    for (k, o) in log.kind.iter().zip(&log.other) { c[*k as usize] += 1; if *k == 3 && *o == -1 { *barrier += 1; } }
}

#[test]
fn cpu_and_gpu_agree_in_aggregate() {
    let data = Path::new("../../../data");
    let world = match bayenv_world::World::load(data) { Ok(w) => Arc::new(w), Err(e) => { eprintln!("skip: world: {e}"); return } };
    let cap = (N + 1023) / 1024 * 1024;
    let mut o = GpuOptions { capacity: cap, start_time: START, seed: 42, obs: ObsMode::None, refuel_budget_ms: f64::INFINITY, ..Default::default() };
    assert!(o.cpu_connector_join, "the CPU connector entry is the default GPU model");
    o.features.accidents = true;
    let mut gpu = match bayenv_gpu::env::create(world.clone(), o) { Ok(s) => s, Err(e) => { eprintln!("skip: {e}"); return } };
    gpu.set_population(Arc::new(GpuPopulation::generate(&world, N, 42.0, true).unwrap()), true);
    let sw = Arc::new(bayenv_sim::world::World::from_bayenv(&world));
    let pop = Arc::new(bayenv_sim::population::Population::generate_like_env(&sw, "real", N, 42.0, Some(data)).unwrap());
    let mut opts = bayenv_sim::sim::SimOpts::default(); opts.dt = 0.05; opts.seed = 42.0;
    let cfg = bayenv_sim::env::EnvConfig { opts, start_time: Some(START), static_obstacles: Vec::new(), scenario_obstacles: Vec::new(), obstacles: None };
    let mut cpu = bayenv_sim::env::Env::new(sw, pop, cfg, Box::new(|w| bayenv_sim::harness::router_for(&w)));
    let (mut cc, mut gc, mut cb, mut gb) = ([0usize; 10], [0usize; 10], 0, 0);
    let mut acts = vec![0f32; cap * 3];
    for _ in 0..STEPS {
        let rows: Vec<f32> = (0..cpu.sim.vs.high).flat_map(|_| [0.0, 0.3, 0.0]).collect();
        cpu.act_rows(&rows, 0); cpu.step(1); counts(&cpu.take_records(), &mut cc, &mut cb);
        let st = gpu.read_state();
        for i in 0..cap { let driving = st.veh[i * bayenv_gpu::layout::VEH_WORDS + 7] == 1; acts[3 * i..3 * i + 3].copy_from_slice(if driving { &[0.0, 0.3, 0.0] } else { &[0.0, 0.0, 0.3] }); }
        gpu.set_actions(&acts, 0); gpu.run_steps(1); counts(&gpu.take_records(), &mut gc, &mut gb);
    }
    let name = ["", "arrived", "crashed", "contact", "stranded", "cleared", "spawned", "missed_exit", "violation", ""];
    for k in [1, 2, 3, 6, 7, 8] { eprintln!("{:12} cpu {:6} gpu {:6}", name[k], cc[k], gc[k]); }
    eprintln!("barrier rows cpu {cb} gpu {gb}");
    let near = |k: usize, rel: f64, abs: f64| { let (c, g) = (cc[k] as f64, gc[k] as f64); assert!((c - g).abs() <= abs.max(rel * c), "{}: cpu {c} gpu {g}", name[k]); };
    assert!(cc[6] > 500, "spawned {}", cc[6]);
    assert_eq!(cc[6], gc[6], "spawned");
    near(1, 0.5, 3.0);      // arrived (4 / 4)
    near(2, 0.25, 6.0);     // crashed (28 / 27)
    near(3, 0.5, 12.0);     // contact rows (17 / 12; legacy GPU 1,336)
    assert!(gb <= cb + 12, "barrier strikes: cpu {cb} gpu {gb}");
    near(7, 0.15, 10.0);    // missed exits (120 / 123)
    near(8, 0.03, 20.0);    // violations (1,627 / 1,627)
}
