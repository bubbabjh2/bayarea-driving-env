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

/// Where a car started: (route position, edge, s) right after its spawn step, per (agent, nth spawn of the agent).
type Starts = std::collections::HashMap<(i64, u32), (i32, i32, f32)>;
fn record_spawns(log: &bayenv_world::events::EventLog, seen: &mut std::collections::HashMap<i64, u32>, out: &mut Starts, at: impl Fn(usize) -> (i32, i32, f32)) {
    for k in 0..log.len() { if log.kind[k] == 6 { let a = log.agent[k]; let n = seen.entry(a).or_insert(0); out.insert((a, *n), at(log.slot[k] as usize)); *n += 1; } }
}
/// spawn agreement: (spawns on both, same route position and edge, |ds| <= 2 m among those, cpu moved (pos > 0), gpu moved)
fn spawn_agreement(c: &Starts, g: &Starts) -> (usize, usize, usize, usize, usize) {
    let (mut both, mut same, mut near) = (0, 0, 0);
    for (k, a) in c { if let Some(b) = g.get(k) { both += 1; if a.0 == b.0 && a.1 == b.1 { same += 1; if (a.2 - b.2).abs() <= 2.0 { near += 1; } } } }
    (both, same, near, c.values().filter(|v| v.0 > 0).count(), g.values().filter(|v| v.0 > 0).count())
}

/// (cpu counts, gpu counts, cpu barrier rows, gpu barrier rows), or None when skipped
fn run(curvature: bool, smooth: bool, jr: bool) -> Option<([usize; 10], [usize; 10], usize, usize)> { run_sc(curvature, smooth, jr, false).map(|r| r.0) }
#[allow(clippy::type_complexity)]
fn run_sc(curvature: bool, smooth: bool, jr: bool, sc: bool) -> Option<(([usize; 10], [usize; 10], usize, usize), (usize, usize, usize, usize, usize))> {
    let data = Path::new("../../../data");
    let world = match bayenv_world::World::load_with(data, &bayenv_world::LoadOptions { smooth_geometry: smooth, junction_radii: jr, ..Default::default() }) { Ok(w) => Arc::new(w), Err(e) => { bayenv_world::oracle::skip("cpu_vs_gpu", &format!("world: {e}")); return None } };
    let cap = (N + 1023) / 1024 * 1024;
    let mut o = GpuOptions { capacity: cap, start_time: START, seed: 42, obs: ObsMode::None, refuel_budget_ms: f64::INFINITY, ..Default::default() };
    assert!(o.cpu_connector_join && o.contact_tangent, "the CPU connector entry and the f32 contact tangent are the default GPU model");
    assert_eq!(o.model_fixes(), bayenv_gpu::kernels::ModelFixes::default_model(), "the physics-audit model fixes are on by default on the GPU, except curvature_dynamics");
    assert!(!o.curvature_dynamics, "curvature_dynamics is opt-in (default off) on the GPU");
    o.curvature_dynamics = curvature; o.smooth_geometry = smooth; o.junction_radii = jr; o.spawn_clearance = sc;
    o.features.accidents = true;
    let mut gpu = match bayenv_gpu::env::create(world.clone(), o) { Ok(s) => s, Err(e) => { eprintln!("cpu_vs_gpu: no GPU ({e}): not run"); return None } };
    gpu.set_population(Arc::new(GpuPopulation::generate(&world, N, 42.0, true).unwrap()), true);
    let sw = Arc::new(bayenv_sim::world::World::from_bayenv(&world));
    let pop = Arc::new(bayenv_sim::population::Population::generate_like_env(&sw, "real", N, 42.0, Some(data)).unwrap());
    let mut opts = bayenv_sim::sim::SimOpts::default(); opts.dt = 0.05; opts.seed = 42.0;
    assert!(opts.replan_at_missed_event, "the CPU re-plans a missed turn at the event (crash-divergence cause A) by default");
    assert!(!opts.curvature_dynamics, "curvature_dynamics is opt-in (default off) on the CPU");
    opts.curvature_dynamics = curvature; opts.smooth_geometry = smooth; opts.junction_radii = jr; opts.spawn_clearance = sc;
    let cfg = bayenv_sim::env::EnvConfig { opts, start_time: Some(START), static_obstacles: Vec::new(), scenario_obstacles: Vec::new(), obstacles: None };
    let mut cpu = bayenv_sim::env::Env::new(sw, pop, cfg, Box::new(|w| bayenv_sim::harness::router_for(&w)));
    let (mut cc, mut gc, mut cb, mut gb) = ([0usize; 10], [0usize; 10], 0, 0);
    let mut acts = vec![0f32; cap * 3];
    let (mut cs, mut gs, mut cn, mut gn) = (Starts::new(), Starts::new(), Default::default(), Default::default());
    for _ in 0..STEPS {
        let rows: Vec<f32> = (0..cpu.sim.vs.high).flat_map(|_| [0.0, 0.3, 0.0]).collect();
        cpu.act_rows(&rows, 0); cpu.step(1);
        let log = cpu.take_records(); counts(&log, &mut cc, &mut cb);
        { let vs = &cpu.sim.vs; record_spawns(&log, &mut cn, &mut cs, |i| (vs.route_pos[i], vs.edge[i], vs.s[i])); }
        let st = gpu.read_state();
        for i in 0..cap { let driving = st.veh[i * bayenv_gpu::layout::VEH_WORDS + 7] == 1; acts[3 * i..3 * i + 3].copy_from_slice(if driving { &[0.0, 0.3, 0.0] } else { &[0.0, 0.0, 0.3] }); }
        gpu.set_actions(&acts, 0); gpu.run_steps(1);
        let log = gpu.take_records(); counts(&log, &mut gc, &mut gb);
        if log.kind.contains(&6) { let st = gpu.read_state(); record_spawns(&log, &mut gn, &mut gs, |i| (st.u(i, 8) as i32, st.u(i, 0) as i32, st.f(i, 1))); }
    }
    let sa = spawn_agreement(&cs, &gs);
    eprintln!("curvature_dynamics {curvature} smooth_geometry {smooth} junction_radii {jr}");
    for k in [1, 2, 3, 6, 7, 8] { eprintln!("{:12} cpu {:6} gpu {:6}", NAME[k], cc[k], gc[k]); }
    eprintln!("barrier rows cpu {cb} gpu {gb}");
    eprintln!("spawn_clearance {sc}: spawns on both {}, same route position + edge {}, of those |ds| <= 2 m {}, started past route edge 0: cpu {} gpu {}", sa.0, sa.1, sa.2, sa.3, sa.4);
    assert!(cc[6] > 500, "spawned {}", cc[6]);
    assert_eq!(cc[6], gc[6], "spawned");
    Some(((cc, gc, cb, gb), sa))
}
fn near(cc: &[usize; 10], gc: &[usize; 10], k: usize, rel: f64, abs: f64) {
    let (c, g) = (cc[k] as f64, gc[k] as f64); assert!((c - g).abs() <= abs.max(rel * c), "{}: cpu {c} gpu {g}", NAME[k]);
}

/// The default model (curvature_dynamics off; static_hold, idle_creep, steer_delay 0.15 s, mountable_kerbs). Measured:
/// arrived 4 / 4, crashed 32 / 32, contacts 11 / 11 (barrier 5 / 5), missed exits 116 / 124, violations 1,632 / 1,629.
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

/// curvature_dynamics on (both backends) with smooth_geometry, junction_radii and spawn_clearance: a car held at steer 0
/// no longer follows the road, so most cars of this policy hit a kerb. Measured (with static_hold, idle_creep,
/// steer_delay, mountable_kerbs): arrived 0 / 0, crashed 679 / 680, contacts 1,326 / 1,328 (barrier 1,175 / 1,168),
/// missed exits 15 / 19, violations 399 / 399; spawns: all 834 start on the same route position and edge (127 past route edge 0 on both).
#[test]
fn cpu_and_gpu_agree_with_curvature_dynamics() {
    let Some(((cc, gc, cb, gb), sa)) = run_sc(true, true, true, true) else { return };
    // spawn_clearance: the host picks the same start edge as the CPU (DECISIONS.md 34)
    assert!(sa.3 > 0 && sa.3 == sa.4, "spawns moved: cpu {} gpu {}", sa.3, sa.4);
    assert!(sa.1 + 2 >= sa.0, "start route position / edge: {} of {} agree", sa.1, sa.0);
    near(&cc, &gc, 1, 0.5, 3.0);
    near(&cc, &gc, 2, 0.05, 6.0);
    near(&cc, &gc, 3, 0.05, 12.0);
    assert!(gb.abs_diff(cb) <= 12.max(cb / 20), "barrier strikes: cpu {cb} gpu {gb}");
    near(&cc, &gc, 7, 0.3, 6.0);
    near(&cc, &gc, 8, 0.03, 20.0);
}
