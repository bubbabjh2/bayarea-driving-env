//! GPU mirror of bayenv-sim tests/physics_fixes.rs `mountable_kerbs_scrape_not_crash` (kernels/model_contacts.wgsl
//! MF_MOUNTABLE_KERBS): a car at 3.5 m/s, heading 0.7 rad into the right kerb of a non-rigid edge (not the freeway
//! system, not a bridge) while steering into it. With rigid kerbs the contact episode adds up to a disabling crash;
//! with mountable kerbs it is one minor contact and the car keeps driving. The same scenario runs on the CPU sim on the
//! same edge for comparison; both backends agree (on this real edge 0.45 rad, the CPU fixture's angle, grinds without
//! disabling on both; at 0.7 rad rigid disables on both after 3 minor contacts, and with mountable kerbs the car keeps
//! driving with no contact row on both: the grind's steps no longer add up, and its first touch is below touch_kmh). The car is one of a few scattered cars on the
//! real world, chosen alone on a long non-rigid edge; its state is written into the veh buffer. Skips without the
//! world data or a GPU.
use bayenv_gpu::{defines::ObsMode, layout as L, sim::GpuOptions};
use std::{path::Path, sync::Arc};

const N: usize = 400;
const STEER: f32 = -0.15; const GAS: f32 = 0.1; const U0: f32 = 3.5; const HR: f32 = 0.7;
// veh words: edge 0, s 1, speed 2, lat 4, headingRel 5, steerAngle 6, state 7, vLat 33, yawRate 34
fn world() -> Option<Arc<bayenv_world::World>> {
    match bayenv_world::World::load(Path::new("../../../data")) { Ok(w) => Some(Arc::new(w)), Err(e) => { bayenv_world::oracle::skip("kerbs", &format!("world: {e}")); None } }
}

/// (disabled, minor contacts of the car, still driving, its edge, its s)
fn run(w: &Arc<bayenv_world::World>, mountable: bool) -> Option<(bool, usize, bool, usize, f64)> {
    let o = GpuOptions { capacity: 1024, obs: ObsMode::None, mountable_kerbs: mountable, ..Default::default() };
    let mut s = match bayenv_gpu::env::create(w.clone(), o) { Ok(s) => s, Err(e) => { eprintln!("kerbs: no GPU ({e}): not run"); return None } };
    bayenv_gpu::env::scatter(&mut s, N, 3);
    let g = &w.graph;
    let st = s.read_state();
    // the car: alone on a long non-rigid edge (class > 3 = not the freeway system, no bridge), mid-edge
    let rigid = |e: usize| g.class[e] <= 3 || g.has(e, bayenv_world::graph::ef::BRIDGE);
    let on_edge = |e: u32| (0..N).filter(|&j| st.u(j, 7) == 1 && st.u(j, 0) == e).count();
    let i = (0..N).find(|&i| {
        let e = st.u(i, 0) as usize; let sx = st.f(i, 1) as f64;
        st.u(i, 7) == 1 && !rigid(e) && g.length[e] > 150.0 && sx > g.s_exit[e] as f64 + 20.0 && sx < g.s_stop[e] as f64 - 60.0 && on_edge(e as u32) == 1
    })?;
    // everyone else brakes; the car holds steer -0.15 (into the right kerb), gas 0.1
    let mut a: Vec<f32> = (0..s.cap).flat_map(|_| [0.0, 0.0, 1.0]).collect();
    a[3 * i..3 * i + 3].copy_from_slice(&[STEER, GAS, 0.0]);
    s.set_actions(&a, 0);
    s.run_steps(30);   // the command reaches the end of the control ring
    let _ = s.take_records();
    let mut v = s.read_state().veh[i * L::VEH_WORDS..(i + 1) * L::VEH_WORDS].to_vec();
    v[2] = U0.to_bits(); v[4] = 1.0f32.to_bits(); v[5] = (-HR).to_bits(); v[6] = 0f32.to_bits(); v[25] = 0f32.to_bits(); v[35] = 0f32.to_bits(); v[33] = 0f32.to_bits(); v[34] = 0f32.to_bits();
    s.gpu.write("veh", bytemuck::cast_slice(&v), (i * L::VEH_WORDS * 4) as u64);
    let (mut minor, mut disabled) = (0, false);
    for _ in 0..60 {
        s.run_steps(1);
        let log = s.take_records();
        for k in 0..log.len() { if log.slot[k] as usize == i { if log.kind[k] == 3 { minor += 1; } if log.kind[k] == 2 { disabled = true; } } }
        let st = s.read_state();
        if std::env::var("KERB_DEBUG").is_ok() { eprintln!("u {:.2} lat {:.2} hr {:.3} steer {:.3} state {} ev {:?}", st.f(i, 2), st.f(i, 4), st.f(i, 5), st.f(i, 6), st.u(i, 7), (0..log.len()).filter(|&k| log.slot[k] as usize == i).map(|k| (log.kind[k], log.dv_kmh[k])).collect::<Vec<_>>()); }
        if st.u(i, 7) != 1 { break; }
    }
    let driving = s.read_state().u(i, 7) == 1;
    Some((disabled, minor, driving, st.u(i, 0) as usize, st.f(i, 1) as f64))
}

/// The same scenario on the CPU sim (bayenv-sim, same world, the car placed by hand as physics_fixes.rs does):
/// (crashed vehicles, minor contacts, still driving)
fn run_cpu(w: &Arc<bayenv_world::World>, mountable: bool, e: usize, s0: f64) -> (u64, u64, bool) {
    use bayenv_sim::{sim::*, vehicle::*};
    let g = Arc::new(bayenv_sim::world::World::from_bayenv(w));
    let opts = SimOpts { dt: 0.05, start_time: 7.0 * 3600.0, fuel: false, mountable_kerbs: mountable, ..SimOpts::default() };
    let mut sim = Sim::new(g.clone(), Arc::new(bayenv_sim::population::Population::default()), opts, bayenv_sim::harness::router_for(&g));
    let (time, dt, step) = (sim.time, sim.dt, sim.step_count);
    let i = sim.vs.alloc(&g, AllocArgs { type_idx: 0, traits: DEFAULT_TRAITS, route: &[e as i32], commuter: -1, leg: 0, time, learner: false, dt, step, lane: 0 });
    sim.vs.s[i] = s0 as f32; sim.vs.speed[i] = U0; sim.vs.set_reaction_time(i, 0.05, dt);
    sim.vs.lat[i] = 1.0; sim.vs.heading_rel[i] = -HR;
    sim.vs.update_lanes(i, &g, e); sim.vs.update_pose(i, &g); sim.vs.save_pose(i);
    if sim.drivers.len() <= i { sim.drivers.resize(i + 1, Driver::None); }
    if sim.actions.len() <= i { sim.actions.resize(i + 1, None); }
    sim.drivers[i] = Driver::External;
    for _ in 0..60 { if sim.vs.state[i] != STATE_DRIVING { break; } sim.actions[i] = Some(Action { steer: Some(STEER as f64), steer_rate: None, gas: GAS as f64, brake: 0.0 }); sim.step(); }
    (sim.counters.crashed_vehicles, sim.counters.minor_contacts, sim.vs.state[i] == STATE_DRIVING)
}

#[test]
fn gpu_mountable_kerbs_scrape_not_crash() {
    let Some(w) = world() else { return };
    let Some((rd, rm, rdrv, e, s0)) = run(&w, false) else { return };
    let (kd, km, kdrv, _, _) = run(&w, true).unwrap();
    let (crc, crm, crd) = run_cpu(&w, false, e, s0);
    let (ckc, ckm, ckd) = run_cpu(&w, true, e, s0);
    eprintln!("{U0} m/s at {HR} rad steering into a kerb (edge {e}, s {s0:.0}): GPU rigid disabled {rd} (minor {rm}, driving {rdrv}), mountable disabled {kd}, minor {km}, driving {kdrv}; CPU rigid crashed {crc} (minor {crm}, driving {crd}), mountable crashed {ckc}, minor {ckm}, driving {ckd}");
    assert!(rd && !rdrv, "rigid: the grind adds up to a disabling crash");
    assert!(!kd && km <= 1 && kdrv, "mountable: at most one minor scrape, still driving");
    assert!(ckc == 0 && ckm <= 1 && ckd, "CPU mountable: at most one minor scrape, still driving");
    assert_eq!(km, ckm as usize, "mountable: GPU and CPU minor contacts");
    assert_eq!(rdrv, crd, "rigid: GPU and CPU agree on the outcome");
    assert_eq!(rd, crc > 0, "rigid: GPU and CPU agree on the outcome");
}
