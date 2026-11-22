//! GPU mirror of bayenv-sim tests/physics_fixes.rs `hill_rollback_and_hill_starts` (kernels/model_integrate.wgsl
//! MF_ROLLBACK), against the CPU on the same hill: trip 818's (edge 124811, 17 % at s 15 rising to 23 % at s 30). One
//! sedan from rest at (s0, gas, brake), reaction delay one step on both backends, no other traffic. Skips without the
//! world data or a GPU.
use bayenv_gpu::{defines::ObsMode, layout as L, sim::GpuOptions};
use std::{path::Path, sync::Arc};

const E: usize = 124811;

/// (max s on the edge, final s, min speed, left the edge forwards)
type Run = (f64, f64, f64, bool);

fn cpu_run(sw: &Arc<bayenv_sim::world::World>, rollback: bool, s0: f64, gas: f64, brake: f64, secs: f64) -> Run {
    use bayenv_sim::{sim::*, vehicle::*};
    let opts = SimOpts { dt: 0.05, start_time: 7.0 * 3600.0, fuel: false, hill_rollback: rollback, traffic_mode: TrafficMode::Empty, ..SimOpts::default() };
    let mut sim = Sim::new(sw.clone(), Arc::new(bayenv_sim::population::Population::default()), opts, bayenv_sim::harness::router_for(sw));
    let (time, dt, step) = (sim.time, sim.dt, sim.step_count);
    let i = sim.vs.alloc(sw, AllocArgs { type_idx: 0, traits: DEFAULT_TRAITS, route: &[E as i32], commuter: -1, leg: 0, time, learner: false, dt, step, lane: 0 });
    sim.vs.s[i] = s0 as f32; sim.vs.speed[i] = 0.0; sim.vs.set_reaction_time(i, 0.05, dt);
    sim.vs.update_lanes(i, sw, E); sim.vs.update_pose(i, sw); sim.vs.save_pose(i);
    if sim.drivers.len() <= i { sim.drivers.resize(i + 1, Driver::None); }
    if sim.actions.len() <= i { sim.actions.resize(i + 1, None); }
    sim.drivers[i] = Driver::External;
    let (mut smax, mut umin, mut left) = (s0, 0.0f64, false);
    for _ in 0..(secs / dt) as usize {
        if sim.vs.state[i] != STATE_DRIVING || sim.vs.edge[i] as usize != E { left = true; break; }
        sim.actions[i] = Some(Action { steer: Some(0.0), steer_rate: None, gas, brake }); sim.step();
        if sim.vs.state[i] != STATE_DRIVING || sim.vs.edge[i] as usize != E { left = true; break; }
        smax = smax.max(sim.vs.s[i] as f64); umin = umin.min(sim.vs.speed[i] as f64);
    }
    (smax, sim.vs.s[i] as f64, umin, left)
}

fn gpu_run(w: &Arc<bayenv_world::World>, rollback: bool, s0: f32, gas: f32, brake: f32, secs: f64) -> Option<Run> {
    let mut o = GpuOptions { capacity: 1024, dt: 0.05, start_time: 7.0 * 3600.0, obs: ObsMode::None, hill_rollback: rollback, ..Default::default() };
    o.features.empty = true; o.features.fuel = false; o.features.accidents = false;
    let mut gpu = match bayenv_gpu::env::create(w.clone(), o) { Ok(s) => s, Err(e) => { eprintln!("hill: no GPU ({e}): not run"); return None } };
    bayenv_gpu::env::scatter(&mut gpu, 1, 4);
    let acts: Vec<f32> = (0..gpu.cap).flat_map(|_| [0.0, gas, brake]).collect();
    gpu.set_actions(&acts, 0);
    let st = gpu.read_state();
    let p = &bayenv_sim::physics::VEHICLE_PARAMS[0];
    let mut pr = st.par[..L::PAR_WORDS].to_vec();
    pr[2] = 0; pr[3] = 1; pr[4] = (p.length as f32).to_bits(); pr[5] = (p.width as f32).to_bits();   // a sedan, one step of reaction delay
    gpu.gpu.write("par", bytemuck::cast_slice(&pr), 0);
    let mut v = st.veh[..L::VEH_WORDS].to_vec();
    v[0] = E as u32; v[1] = s0.to_bits(); v[2] = 0f32.to_bits(); v[4] = (0.5 * w.lanes.lane_width(E, s0 as f64) as f32).to_bits(); v[5] = 0f32.to_bits();
    for k in [6, 25, 33, 34, 35, 30] { v[k] = 0f32.to_bits(); }
    v[14] = L::pack_lanes(0, 0, 0, -1);
    for _ in 0..3 { gpu.gpu.write("veh", bytemuck::cast_slice(&v), 0); gpu.run_steps(1); }   // the command fills the ring
    gpu.gpu.write("veh", bytemuck::cast_slice(&v), 0);
    let (mut smax, mut umin, mut left, mut s_end) = (s0 as f64, 0.0f64, false, s0 as f64);
    for _ in 0..(secs / 0.05) as usize {
        gpu.run_steps(1);
        let st = gpu.read_state();
        if st.u(0, 7) != 1 || st.u(0, 0) as usize != E { left = true; break; }
        s_end = st.f(0, 1) as f64; smax = smax.max(s_end); umin = umin.min(st.f(0, 2) as f64);
    }
    Some((smax, s_end, umin, left))
}

#[test]
fn gpu_hill_rollback_and_hill_starts_match_cpu() {
    let w = match bayenv_world::World::load(Path::new("../../../data")) { Ok(w) => Arc::new(w), Err(e) => { bayenv_world::oracle::skip("hill", &format!("world: {e}")); return } };
    let sw = Arc::new(bayenv_sim::world::World::from_bayenv(&w));
    let cases: [(&str, bool, f64, f64, f64, f64); 6] = [
        ("gas 0.546 from s 14.72", true, 14.72, 0.546, 0.0, 40.0),
        ("gas 0.45 from s 20", true, 20.0, 0.45, 0.0, 10.0),
        ("gas 0.45 from s 20, JS clamp", false, 20.0, 0.45, 0.0, 10.0),
        ("gas 0.65 from s 14.72", true, 14.72, 0.65, 0.0, 40.0),
        ("s 25, brake 0.6", true, 25.0, 0.0, 0.6, 20.0),
        ("s 25, no pedals", true, 25.0, 0.0, 0.0, 5.0),
    ];
    let mut res = Vec::new();
    for (name, rb, s0, gas, brake, secs) in cases {
        let Some(g) = gpu_run(&w, rb, s0 as f32, gas as f32, brake as f32, secs) else { return };
        let c = cpu_run(&sw, rb, s0, gas, brake, secs);
        eprintln!("{name:30}: cpu peak s {:.2} end {:.2} min u {:.2} left {} | gpu peak s {:.2} end {:.2} min u {:.2} left {}", c.0, c.1, c.2, c.3, g.0, g.1, g.2, g.3);
        assert_eq!(c.3, g.3, "{name}: left the edge");
        // (the CPU car starts with an empty control ring: one step without the command)
        if !c.3 { assert!((c.1 - g.1).abs() < 0.3 + 0.01 * c.1 && (c.0 - g.0).abs() < 0.3 + 0.01 * c.0, "{name}: cpu {c:?} gpu {g:?}"); }
        res.push(g);
    }
    // the physics_fixes expectations, on the GPU
    assert!(res[0].1 > res[0].0 - 2.0, "gas 0.546 stalls at the balance point");
    assert!(res[1].2 < -0.3 && res[1].1 < res[1].0 - 2.0, "an under-powered hill start rolls back");
    assert!(res[2].2 == 0.0 && (res[2].1 - res[2].0).abs() < 1e-3, "the JS stops dead");
    assert!(res[3].3 || res[3].0 > 60.0, "0.65 gas climbs");
    assert!((res[4].1 - 25.0).abs() < 0.02 && res[4].2 > -0.15, "the brake holds");
    assert!(res[5].1 < 24.0 && res[5].2 < -0.5, "without brake it rolls back");
}
