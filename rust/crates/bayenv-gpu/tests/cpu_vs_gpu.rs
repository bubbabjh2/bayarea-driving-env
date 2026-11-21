//! CPU reference sim (bayenv-sim) vs the GPU sim with the default model (GpuOptions: cpu_connector_join, contact_tangent
//! and the physics-audit fixes on, curvature_dynamics off; the CPU SimOpts default is the same model), and once more with
//! curvature_dynamics on for both backends, on the smoothed frame geometry with the re-fitted junction turns
//! (smooth_geometry + junction_radii, DECISIONS.md 33 / 34: both sims built from the same world, so the GPU blob carries
//! the smoothed lane vertices and the moved connectors):
//! rust/tools/compare_backends.py --spawn-default as a test. 5,000 commuters (population.js, seed 42) from 07:00,
//! dt 0.05, 1,200 steps, every driving car commanded [0, 0.3, 0]; GPU slots that are not driving get the CPU's default
//! action [0, 0, 0.3] (a car spawning during a step brakes on its first decision, as on the CPU). Aggregate event
//! counts must agree within tolerances. Before the physics-audit fixes (rust/docs/kerb-investigation.md): contacts
//! 17 / 12, crashed 28 / 27, missed exits 120 / 123, violations 1,627 / 1,627 (the legacy GPU model gives 1,336 contacts).
//! Skips without the world data or a GPU adapter.
use bayenv_gpu::{defines::ObsMode, sim::{GpuOptions, GpuPopulation}};
use std::{path::Path, sync::Arc};

const N: usize = 5000;
const STEPS: usize = 1200;
const START: f64 = 7.0 * 3600.0;
const NAME: [&str; 10] = ["", "arrived", "crashed", "contact", "stranded", "cleared", "spawned", "missed_exit", "violation", ""];
// event kinds (bayenv_world::events): 1 arrived, 2 crashed, 3 contact, 6 spawned, 7 missed exit, 8 violation
fn counts(log: &bayenv_world::events::EventLog, c: &mut [usize; 10], barrier: &mut usize) {
    for (k, o) in log.kind.iter().zip(&log.other) { c[*k as usize] += 1; if *k == 3 && *o == -1 { *barrier += 1; } }
}

/// (cpu counts, gpu counts, cpu barrier rows, gpu barrier rows), or None when skipped
fn run(curvature: bool, smooth: bool, jr: bool) -> Option<([usize; 10], [usize; 10], usize, usize)> {
    let data = Path::new("../../../data");
    let world = match bayenv_world::World::load_with(data, &bayenv_world::LoadOptions { smooth_geometry: smooth, junction_radii: jr, ..Default::default() }) { Ok(w) => Arc::new(w), Err(e) => { bayenv_world::oracle::skip("cpu_vs_gpu", &format!("world: {e}")); return None } };
    let cap = (N + 1023) / 1024 * 1024;
    let mut o = GpuOptions { capacity: cap, start_time: START, seed: 42, obs: ObsMode::None, refuel_budget_ms: f64::INFINITY, ..Default::default() };
    assert!(o.cpu_connector_join && o.contact_tangent, "the CPU connector entry and the f32 contact tangent are the default GPU model");
    assert_eq!(o.model_fixes(), bayenv_gpu::kernels::ModelFixes::default_model(), "the physics-audit model fixes are on by default on the GPU, except curvature_dynamics");
    assert!(!o.curvature_dynamics, "curvature_dynamics is opt-in (default off) on the GPU");
    o.curvature_dynamics = curvature; o.smooth_geometry = smooth; o.junction_radii = jr;
    o.features.accidents = true;
    let mut gpu = match bayenv_gpu::env::create(world.clone(), o) { Ok(s) => s, Err(e) => { eprintln!("cpu_vs_gpu: no GPU ({e}): not run"); return None } };
    gpu.set_population(Arc::new(GpuPopulation::generate(&world, N, 42.0, true).unwrap()), true);
    let sw = Arc::new(bayenv_sim::world::World::from_bayenv(&world));
    let pop = Arc::new(bayenv_sim::population::Population::generate_like_env(&sw, "real", N, 42.0, Some(data)).unwrap());
    let mut opts = bayenv_sim::sim::SimOpts::default(); opts.dt = 0.05; opts.seed = 42.0;
    assert!(opts.replan_at_missed_event, "the CPU re-plans a missed turn at the event (crash-divergence cause A) by default");
    assert!(!opts.curvature_dynamics, "curvature_dynamics is opt-in (default off) on the CPU");
    opts.curvature_dynamics = curvature; opts.smooth_geometry = smooth; opts.junction_radii = jr;
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
    eprintln!("curvature_dynamics {curvature} smooth_geometry {smooth} junction_radii {jr}");
    for k in [1, 2, 3, 6, 7, 8] { eprintln!("{:12} cpu {:6} gpu {:6}", NAME[k], cc[k], gc[k]); }
    eprintln!("barrier rows cpu {cb} gpu {gb}");
    assert!(cc[6] > 500, "spawned {}", cc[6]);
    assert_eq!(cc[6], gc[6], "spawned");
    Some((cc, gc, cb, gb))
}
fn near(cc: &[usize; 10], gc: &[usize; 10], k: usize, rel: f64, abs: f64) {
    let (c, g) = (cc[k] as f64, gc[k] as f64); assert!((c - g).abs() <= abs.max(rel * c), "{}: cpu {c} gpu {g}", NAME[k]);
}

/// The default model (curvature_dynamics off). Measured: arrived 4 / 4, crashed 31 / 31, contacts 16 / 16 (barrier 11 / 11),
/// missed exits 114 / 123, violations 1,626 / 1,625.
#[test]
fn cpu_and_gpu_agree_in_aggregate() {
    let Some((cc, gc, cb, gb)) = run(false, false, false) else { return };
    near(&cc, &gc, 1, 0.5, 3.0);      // arrived
    near(&cc, &gc, 2, 0.25, 6.0);     // crashed
    near(&cc, &gc, 3, 0.5, 12.0);     // contact rows (legacy GPU 1,336)
    assert!(gb <= cb + 12, "barrier strikes: cpu {cb} gpu {gb}");
    near(&cc, &gc, 7, 0.15, 10.0);    // missed exits
    near(&cc, &gc, 8, 0.03, 20.0);    // violations
}

/// curvature_dynamics on (both backends) with smooth_geometry and junction_radii: a car held at steer 0 no longer
/// follows the road, so most cars of this policy hit a kerb. Measured: arrived 0 / 0, crashed 734 / 734, contacts
/// 2,023 / 2,015 (barrier 2,019 / 2,011), missed exits 13 / 15, violations 337 / 337. (Smoothed geometry only: crashed
/// 733 / 733, contacts 1,909 / 1,903; raw geometry: crashed 751 / 751, contacts 1,804 / 1,805.)
#[test]
fn cpu_and_gpu_agree_with_curvature_dynamics() {
    let Some((cc, gc, cb, gb)) = run(true, true, true) else { return };
    near(&cc, &gc, 1, 0.5, 3.0);
    near(&cc, &gc, 2, 0.05, 6.0);
    near(&cc, &gc, 3, 0.05, 12.0);
    assert!(gb.abs_diff(cb) <= 12.max(cb / 20), "barrier strikes: cpu {cb} gpu {gb}");
    near(&cc, &gc, 7, 0.3, 6.0);
    near(&cc, &gc, 8, 0.03, 20.0);
}
