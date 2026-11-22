//! Starts from a stop, CPU vs GPU (static_hold, idle_creep, brake_hold): cars scattered at rest on the real world (so on
//! real grades and surfaces), each with its own (gas, brake) pair from a grid, held at rest under full brake until the
//! control ring is full, then 20 s on the pair. The same cars (edge, s, lane, route, reaction delay) are placed in a CPU
//! sim. Traffic mode "empty" on both (no contacts: each car drives alone). Per car: distance and final speed on both
//! backends. Skips without the world data or a GPU.
use bayenv_gpu::{defines::ObsMode, layout as L, sim::GpuOptions};
use std::{path::Path, sync::Arc};

const N: usize = 800;
const GAS: [f32; 6] = [0.0, 0.01, 0.02, 0.05, 0.1, 0.3];
const BRAKE: [f32; 6] = [0.0, 0.01, 0.03, 0.07, 0.1, 0.3];
fn pair(i: usize) -> (f32, f32) { (GAS[i % 6], BRAKE[(i / 6) % 6]) }

fn world() -> Option<Arc<bayenv_world::World>> {
    match bayenv_world::World::load(Path::new("../../../data")) { Ok(w) => Some(Arc::new(w)), Err(e) => { bayenv_world::oracle::skip("pull_away", &format!("world: {e}")); None } }
}

/// per slot (distance, speed) for the GPU and the CPU; None for slots not compared
#[allow(clippy::type_complexity)]
fn run(w: &Arc<bayenv_world::World>, tune: impl Fn(&mut GpuOptions, &mut bayenv_sim::sim::SimOpts)) -> Option<Vec<Option<((f64, f64), (f64, f64))>>> {
    let mut o = GpuOptions { capacity: 1024, obs: ObsMode::None, ..Default::default() };
    o.features.empty = true; o.features.fuel = false; o.features.accidents = false;
    let mut so = bayenv_sim::sim::SimOpts { dt: 0.05, start_time: 7.0 * 3600.0, fuel: false, traffic_mode: bayenv_sim::sim::TrafficMode::Empty, ..Default::default() };
    tune(&mut o, &mut so);
    let mut gpu = match bayenv_gpu::env::create(w.clone(), o) { Ok(s) => s, Err(e) => { eprintln!("pull_away: no GPU ({e}): not run"); return None } };
    bayenv_gpu::env::scatter(&mut gpu, N, 5);
    let brake: Vec<f32> = (0..gpu.cap).flat_map(|_| [0.0, 0.0, 1.0]).collect();
    gpu.set_actions(&brake, 0); gpu.run_steps(45);
    let st0 = gpu.read_state();
    let routes = gpu.read_words("routes");
    // CPU: the same cars
    use bayenv_sim::{sim::*, vehicle::*};
    let g = Arc::new(bayenv_sim::world::World::from_bayenv(w));
    let mut cpu = Sim::new(g.clone(), Arc::new(bayenv_sim::population::Population::default()), so, bayenv_sim::harness::router_for(&g));
    let mut slot = vec![usize::MAX; N];
    let mut routes_k: Vec<Vec<i32>> = vec![Vec::new(); N];
    for i in 0..N {
        if st0.u(i, 7) != 1 { continue; }
        let (off, len, pos) = (st0.par[i * L::PAR_WORDS] as usize, st0.par[i * L::PAR_WORDS + 1] as usize, st0.u(i, 8) as usize);
        routes_k[i] = routes[off + pos..off + len].iter().map(|&x| (x & 0xFFFFF) as i32).collect();
        if st0.f(i, 30) >= 1.0 { continue; }   // on a junction connector: skip (placing on a connector by hand is not supported here)
        let e = st0.u(i, 0) as usize; let lane = (st0.u(i, 14) & 255) as i32;
        let (time, dt, step) = (cpu.time, cpu.dt, cpu.step_count);
        let k = cpu.vs.alloc(&g, AllocArgs { type_idx: st0.par[i * L::PAR_WORDS + 2] as u8, traits: DEFAULT_TRAITS, route: &routes_k[i], commuter: -1, leg: 0, time, learner: false, dt, step, lane });
        cpu.vs.s[k] = st0.f(i, 1); cpu.vs.lat[k] = st0.f(i, 4); cpu.vs.heading_rel[k] = st0.f(i, 5);
        cpu.vs.delay_steps[k] = st0.par[i * L::PAR_WORDS + 3];
        assert_eq!(cpu.vs.edge[k] as usize, e);
        cpu.vs.update_lanes(k, &g, e); cpu.vs.update_pose(k, &g); cpu.vs.save_pose(k);
        if cpu.drivers.len() <= k { cpu.drivers.resize(k + 1, Driver::None); }
        if cpu.actions.len() <= k { cpu.actions.resize(k + 1, None); }
        cpu.drivers[k] = Driver::External;
        slot[i] = k;
    }
    let act = |cpu: &mut Sim, f: &dyn Fn(usize) -> (f32, f32)| { for i in 0..N { let k = slot[i]; if k == usize::MAX || cpu.vs.state[k] != STATE_DRIVING { continue; } let (ga, br) = f(i); cpu.actions[k] = Some(Action { steer: Some(0.0), steer_rate: None, gas: ga as f64, brake: br as f64 }); } };
    for _ in 0..45 { act(&mut cpu, &|_| (0.0, 1.0)); cpu.step(); }
    let o0: Vec<f32> = (0..N).map(|i| if slot[i] == usize::MAX { 0.0 } else { cpu.vs.odometer[slot[i]] }).collect();
    let mut a: Vec<f32> = (0..gpu.cap).flat_map(|_| [0.0, 0.0, 1.0]).collect();
    for i in 0..N { let (ga, br) = pair(i); a[3 * i + 1] = ga; a[3 * i + 2] = br; }
    gpu.set_actions(&a, 0); gpu.run_steps(400);
    for _ in 0..400 { act(&mut cpu, &|i| pair(i)); cpu.step(); }
    let st = gpu.read_state();
    Some((0..N).map(|i| {
        let k = slot[i];
        if k == usize::MAX || st.u(i, 7) != 1 || cpu.vs.state[k] != STATE_DRIVING || st.u(i, 0) as i32 != cpu.vs.edge[k] && st.u(i, 8) == 0 { return None; }
        Some((((st.f(i, 13) - st0.f(i, 13)) as f64, st.f(i, 2) as f64), ((cpu.vs.odometer[k] - o0[i]) as f64, cpu.vs.speed[k] as f64)))
    }).collect())
}

fn summary(tag: &str, r: &[Option<((f64, f64), (f64, f64))>]) -> (usize, usize, f64) {
    let (mut n, mut bad, mut worst) = (0, 0, 0.0f64);
    let mut cells = std::collections::BTreeMap::<(u32, u32), (usize, f64, f64, usize, usize)>::new();
    for (i, x) in r.iter().enumerate() {
        let Some(((dg, ug), (dc, uc))) = x else { continue };
        n += 1;
        let d = (dg - dc).abs(); if d > worst { worst = d; }
        if d > 0.5 + 0.02 * dc.abs() || (ug - uc).abs() > 0.1 { bad += 1; if bad <= 6 { eprintln!("{tag}: slot {i} pair {:?}: gpu {dg:.2} m {ug:.2} m/s, cpu {dc:.2} m {uc:.2} m/s", pair(i)); } }
        let (ga, br) = pair(i); let c = cells.entry(((ga * 100.0) as u32, (br * 100.0) as u32)).or_default();
        c.0 += 1; c.1 += dg; c.2 += dc; if *ug < 0.05 { c.3 += 1; } if *uc < 0.05 { c.4 += 1; }
    }
    eprintln!("{tag}: {n} cars, {bad} differ (> 0.5 m + 2 % or > 0.1 m/s), worst |dd| {worst:.2} m");
    for ((ga, br), (k, dg, dc, sg, sc)) in &cells { eprintln!("  gas {ga:>2}% brake {br:>2}%: {k:>3} cars, mean distance gpu {:.1} cpu {:.1} m, stopped gpu {sg} cpu {sc}", dg / *k as f64, dc / *k as f64); }
    (n, bad, worst)
}

#[test]
fn pull_away_from_rest_matches_cpu() {
    let Some(w) = world() else { return };
    let Some(r) = run(&w, |_, _| {}) else { return };
    let (n, bad, _) = summary("default (static_hold + idle_creep)", &r);
    let r2 = run(&w, |o, s| { o.static_hold = false; o.idle_creep = false; s.static_hold = false; s.idle_creep = false; }).unwrap();
    let (n2, bad2, _) = summary("static_hold + idle_creep off", &r2);
    assert!(n > 400 && n2 > 400, "cars compared {n} {n2}");
    assert!(bad <= n / 50 && bad2 <= n2 / 50, "GPU and CPU starts from rest differ: {bad} / {n}, {bad2} / {n2}");
}
