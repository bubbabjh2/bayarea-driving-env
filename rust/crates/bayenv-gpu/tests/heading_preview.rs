//! GPU lane-heading preview (GpuOptions::heading_preview, kernels/heading_preview.wgsl) vs the CPU's
//! bayenv-sim obs::heading_preview on identical states. A GPU population runs for a while, then every driving car's
//! (edge, connector, s, lane, route position, route words as the GPU sees them) is copied into a CPU sim's vehicle
//! store over the same world, and the CPU function is evaluated on it. Raw world (curvature off) and the smoothed world
//! with re-fitted junction turns (curvature + smooth_geometry + junction_radii). Measured: max |gpu - cpu| 2.1e-6 rad
//! (495 cars, raw) and 1.4e-6 rad (269 cars, smoothed + junction_radii). Skips without the world data / a GPU.
use bayenv_gpu::{defines::ObsMode, layout as L, sim::{GpuOptions, GpuPopulation}};
use std::{path::Path, sync::Arc};

const N: usize = 3000;
const START: f64 = 7.0 * 3600.0;

/// (cars compared, max |gpu - cpu| rad, rows with an error > 1e-4)
fn compare(curvature: bool, smooth: bool, jr: bool, steps: usize) -> Option<(usize, f64, usize)> {
    let data = Path::new("../../../data");
    let world = match bayenv_world::World::load_with(data, &bayenv_world::LoadOptions { smooth_geometry: smooth, junction_radii: jr, ..Default::default() }) { Ok(w) => Arc::new(w), Err(e) => { bayenv_world::oracle::skip("heading_preview", &format!("world: {e}")); return None } };
    let cap = (N + 1023) / 1024 * 1024;
    let mut o = GpuOptions { capacity: cap, start_time: START, seed: 42, obs: ObsMode::None, refuel_budget_ms: f64::INFINITY, heading_preview: true, ..Default::default() };
    o.curvature_dynamics = curvature; o.smooth_geometry = smooth; o.junction_radii = jr;
    let mut gpu = match bayenv_gpu::env::create(world.clone(), o) { Ok(s) => s, Err(e) => { eprintln!("heading_preview: no GPU ({e}): not run"); return None } };
    gpu.set_population(Arc::new(GpuPopulation::generate(&world, N, 42.0, true).unwrap()), true);
    let acts: Vec<f32> = (0..cap).flat_map(|_| [0.0, 0.3, 0.0]).collect();
    gpu.set_actions(&acts, 0);
    gpu.run_steps(steps);
    let hp = gpu.heading_preview().unwrap();
    let st = gpu.read_state();
    let routes = gpu.read_words("routes");
    // CPU sim over the same world with a slot pool as large as the GPU's, used only as a vehicle store
    let sw = Arc::new(bayenv_sim::world::World::from_bayenv(&world));
    let pop = Arc::new(bayenv_sim::population::Population::generate_like_env(&sw, "real", 10, 42.0, Some(data)).unwrap());
    let mut opts = bayenv_sim::sim::SimOpts::default(); opts.slot_pool = cap; opts.curvature_dynamics = curvature; opts.smooth_geometry = smooth; opts.junction_radii = jr;
    let cfg = bayenv_sim::env::EnvConfig { opts, start_time: Some(START), static_obstacles: Vec::new(), scenario_obstacles: Vec::new(), obstacles: None };
    let mut cpu = bayenv_sim::env::Env::new(sw, pop, cfg, Box::new(|w| bayenv_sim::harness::router_for(&w)));
    let has_j = world.junctions.is_some();
    let (mut n, mut worst, mut bad) = (0usize, 0.0f64, 0usize);
    for i in 0..cap {
        if st.u(i, 7) != 1 { assert!(hp[4 * i..4 * i + 4].iter().all(|&x| x == 0.0), "slot {i}: not driving, preview not 0"); continue; }
        let (off, len) = (st.par[i * L::PAR_WORDS] as usize, st.par[i * L::PAR_WORDS + 1] as usize);
        let vs = &mut cpu.sim.vs;
        vs.edge[i] = st.u(i, 0) as i32; vs.s[i] = st.f(i, 1); vs.route_pos[i] = st.u(i, 8) as i32;
        vs.lane[i] = (st.u(i, 14) & 255) as i8;
        vs.conn[i] = if has_j { st.f(i, 30) as i32 - 1 } else { -1 };
        vs.routes[i] = routes[off..off + len].iter().map(|&w| (w & 0xFFFFF) as i32).collect();
        let c = bayenv_sim::obs::heading_preview(&cpu.sim, i);
        for k in 0..4 {
            let d = (hp[4 * i + k] as f64 - c[k]).sin().atan2((hp[4 * i + k] as f64 - c[k]).cos()).abs();
            if d > worst { worst = d; }
            if d > 1e-4 { bad += 1; if bad <= 5 { eprintln!("slot {i} k {k}: gpu {} cpu {} (edge {} conn {} s {} pos {})", hp[4 * i + k], c[k], vs_edge(&cpu, i), cpu.sim.vs.conn[i], cpu.sim.vs.s[i], cpu.sim.vs.route_pos[i]); } }
        }
        n += 1;
    }
    eprintln!("heading_preview curvature {curvature} smooth {smooth} junction_radii {jr}: {n} driving cars, max |gpu - cpu| {worst:.3e} rad, {bad} values > 1e-4");
    Some((n, worst, bad))
}
fn vs_edge(cpu: &bayenv_sim::env::Env, i: usize) -> i32 { cpu.sim.vs.edge[i] }

#[test]
fn gpu_heading_preview_matches_cpu() {
    let Some((n, worst, bad)) = compare(false, false, false, 600) else { return };
    assert!(n > 300, "driving cars {n}");
    assert!(bad == 0, "{bad} preview values differ by more than 1e-4 rad (max {worst})");
}

#[test]
fn gpu_heading_preview_matches_cpu_smoothed_junction_radii() {
    let Some((n, worst, bad)) = compare(true, true, true, 300) else { return };
    assert!(n > 200, "driving cars {n}");   // curvature on: many cars of this policy are wrecks by now
    assert!(bad == 0, "{bad} preview values differ by more than 1e-4 rad (max {worst})");
}
