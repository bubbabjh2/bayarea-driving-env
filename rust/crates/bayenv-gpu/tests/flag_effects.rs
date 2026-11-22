//! Every model flag reaches the kernels and changes behaviour (GpuOptions -> model_fixes() -> mf_consts -> WGSL, the
//! fused resolve and the sorted variants included), on both backends: a population run with a varied external policy
//! (every slot cycles through gas / brake / steer / coasting phases, so cars pull away, creep, brake, scrape kerbs and
//! crash), once with the default model and once with one flag flipped. Per flag: the run's state hash must change on
//! the GPU and on the CPU (the same flag in SimOpts), and the event counts of the two backends must move the same way.
//! Skips without the world data or a GPU.
use bayenv_gpu::{defines::ObsMode, sim::{GpuOptions, GpuPopulation}};
use std::{path::Path, sync::Arc};

const N: usize = 3000;
const STEPS: usize = 900;
const START: f64 = 7.0 * 3600.0;
const WRECKS: usize = 1500;

fn action(slot: usize, step: usize) -> [f32; 3] {
    let h = (slot as u32).wrapping_mul(2654435761).wrapping_add((step / 40) as u32 * 40503);
    let h = (h ^ (h >> 13)).wrapping_mul(0x5bd1e995); let ph = (h >> 8) % 8;
    let steer = (((h >> 20) % 7) as f32 - 3.0) * 0.1;
    match ph {
        0 | 1 | 2 => [steer * 0.3, 0.35, 0.0],   // drive
        3 => [steer, 0.15, 0.0],                 // weave
        4 => [0.0, 0.0, 0.0],                    // coast / creep
        5 => [0.0, 0.0, 0.04],                   // light brake
        6 => [0.0, 0.02, 0.03],                  // gas + brake mix
        _ => [0.0, 0.0, 0.7],                    // hard brake
    }
}
fn fnv(h: &mut u64, w: u32) { for b in w.to_le_bytes() { *h ^= b as u64; *h = h.wrapping_mul(0x100000001b3); } }

/// (state hash, event counts by kind)
fn run_gpu(w: &Arc<bayenv_world::World>, f: &dyn Fn(&mut GpuOptions), wrecks: usize) -> Option<(u64, [usize; 10])> {
    let cap = (N + 1023) / 1024 * 1024;
    let mut o = GpuOptions { capacity: cap, start_time: START, seed: 42, obs: ObsMode::None, refuel_budget_ms: f64::INFINITY, ..Default::default() };
    o.features.accidents = false;
    f(&mut o);
    let mut s = match bayenv_gpu::env::create(w.clone(), o) { Ok(s) => s, Err(e) => { eprintln!("flag_effects: no GPU ({e}): not run"); return None } };
    s.set_population(Arc::new(GpuPopulation::generate(w, N, 42.0, true).unwrap()), true);
    if wrecks > 0 { s.place_wrecks(wrecks); }
    let mut c = [0usize; 10]; let mut acts = vec![0f32; cap * 3];
    for t in 0..STEPS {
        for i in 0..cap { acts[3 * i..3 * i + 3].copy_from_slice(&action(i, t)); }
        s.set_actions(&acts, 0); s.run_steps(1);
        for k in s.take_records().kind { c[k as usize] += 1; }
    }
    let st = s.read_state(); let mut h = 0xcbf29ce484222325u64;
    for &x in &st.veh { fnv(&mut h, x); }
    Some((h, c))
}
fn run_cpu(w: &Arc<bayenv_world::World>, f: &dyn Fn(&mut bayenv_sim::sim::SimOpts), wrecks: usize) -> (u64, [usize; 10]) {
    let data = Path::new("../../../data");
    let sw = Arc::new(bayenv_sim::world::World::from_bayenv(w));
    let pop = Arc::new(bayenv_sim::population::Population::generate_like_env(&sw, "real", N, 42.0, Some(data)).unwrap());
    let mut opts = bayenv_sim::sim::SimOpts::default(); opts.dt = 0.05; opts.seed = 42.0;
    f(&mut opts);
    let cfg = bayenv_sim::env::EnvConfig { opts, start_time: Some(START), static_obstacles: Vec::new(), scenario_obstacles: Vec::new(), obstacles: None };
    let mut cpu = bayenv_sim::env::Env::new(sw, pop, cfg, Box::new(|w| bayenv_sim::harness::router_for(&w)));
    if wrecks > 0 { cpu.sim.place_wrecks(wrecks); }
    let mut c = [0usize; 10];
    for t in 0..STEPS {
        let rows: Vec<f32> = (0..cpu.sim.vs.high).flat_map(|i| action(i, t)).collect();
        cpu.act_rows(&rows, 0); cpu.step(1);
        for k in cpu.take_records().kind { c[k as usize] += 1; }
    }
    let vs = &cpu.sim.vs; let mut h = 0xcbf29ce484222325u64;
    for i in 0..vs.high { for x in [vs.s[i].to_bits(), vs.speed[i].to_bits(), vs.lat[i].to_bits(), vs.heading_rel[i].to_bits(), vs.edge[i] as u32, vs.state[i] as u32] { fnv(&mut h, x); } }
    (h, c)
}

type Flip = (&'static str, fn(&mut GpuOptions), fn(&mut bayenv_sim::sim::SimOpts));
const FLIPS: &[Flip] = &[
    ("wreck_real_mass", |o| o.wreck_real_mass = false, |s| s.wreck_real_mass = false),
    ("curvature_dynamics", |o| o.curvature_dynamics = true, |s| s.curvature_dynamics = true),
    ("wreck_end_stop", |o| o.wreck_end_stop = false, |s| s.wreck_end_stop = false),
    ("contact_episodes", |o| o.contact_episodes = false, |s| s.contact_episodes = false),
    ("connector_barrier", |o| o.connector_barrier = false, |s| s.connector_barrier = false),
    ("brake_hold", |o| o.brake_hold = true, |s| s.brake_hold = true),
    ("static_hold", |o| o.static_hold = false, |s| s.static_hold = false),
    ("idle_creep", |o| o.idle_creep = false, |s| s.idle_creep = false),
    ("steer_delay", |o| o.steer_delay = -1.0, |s| s.steer_delay = -1.0),
    ("heavy_brakes", |o| o.heavy_brakes = false, |s| s.heavy_brakes = false),
    ("barrier_soft_push", |o| o.barrier_soft_push = false, |s| s.barrier_soft_push = false),
    ("impact_reverse", |o| o.impact_reverse = false, |s| s.impact_reverse = false),
    ("mountable_kerbs", |o| o.mountable_kerbs = false, |s| s.mountable_kerbs = false),
    ("spawn_clearance", |o| o.spawn_clearance = true, |s| s.spawn_clearance = true),
];

#[test]
fn every_model_flag_changes_behaviour_on_both_backends() {
    let w = match bayenv_world::World::load(Path::new("../../../data")) { Ok(w) => Arc::new(w), Err(e) => { bayenv_world::oracle::skip("flag_effects", &format!("world: {e}")); return } };
    let Some((g0, gc0)) = run_gpu(&w, &|_| {}, 0) else { return };
    let (c0, cc0) = run_cpu(&w, &|_| {}, 0);
    // wreck_real_mass needs resting wrecks that cars run into: its own pair of runs with placed wrecks
    let (gw0, gwc0) = run_gpu(&w, &|_| {}, WRECKS).unwrap();
    let (cw0, cwc0) = run_cpu(&w, &|_| {}, WRECKS);
    eprintln!("with {WRECKS} wrecks: gpu crashed {} contacts {}; cpu crashed {} contacts {}", gwc0[2], gwc0[3], cwc0[2], cwc0[3]);
    // GPU-only fixes: their effect on the GPU
    for (name, f) in [("cpu_connector_join", (|o: &mut GpuOptions| o.cpu_connector_join = false) as fn(&mut GpuOptions)), ("contact_tangent", |o| o.contact_tangent = false)] {
        let (g, gc) = run_gpu(&w, &f, 0).unwrap();
        eprintln!("{name:18} gpu changed {} (crashed {} -> {}, contacts {} -> {})", g != g0, gc0[2], gc[2], gc0[3], gc[3]);
        assert_ne!(g, g0, "{name}: no effect on the GPU");
    }
    let mut fails = Vec::new();
    for (name, fg, fc) in FLIPS {
        let k = if *name == "wreck_real_mass" { WRECKS } else { 0 };
        let (g, gc) = run_gpu(&w, fg, k).unwrap();
        let (c, cc) = run_cpu(&w, fc, k);
        let (g0, gc0, c0, cc0) = if k > 0 { (gw0, gwc0, cw0, cwc0) } else { (g0, gc0, c0, cc0) };
        let d = |a: &[usize; 10], b: &[usize; 10], k: usize| b[k] as i64 - a[k] as i64;
        eprintln!("{name:18} gpu changed {:5} cpu changed {:5} | delta crashed gpu {:+} cpu {:+}, contacts gpu {:+} cpu {:+}, arrived gpu {:+} cpu {:+}, missed gpu {:+} cpu {:+}",
            g != g0, c != c0, d(&gc0, &gc, 2), d(&cc0, &cc, 2), d(&gc0, &gc, 3), d(&cc0, &cc, 3), d(&gc0, &gc, 1), d(&cc0, &cc, 1), d(&gc0, &gc, 7), d(&cc0, &cc, 7));
        if g == g0 { fails.push(format!("{name}: no effect on the GPU")); }
        if c == c0 { fails.push(format!("{name}: no effect on the CPU")); }
    }
    eprintln!("default: gpu crashed {} contacts {} arrived {}; cpu crashed {} contacts {} arrived {}", gc0[2], gc0[3], gc0[1], cc0[2], cc0[3], cc0[1]);
    assert!(fails.is_empty(), "{fails:?}");
}
