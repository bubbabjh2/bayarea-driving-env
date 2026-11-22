//! CPU vs GPU replay: the same population on both backends, every car commanded a policy-like action that depends only
//! on (trip, step) (noisy per-decision steer in [-1, 1] with large swings, gas / brake mixes, light brakes, coasting), so
//! both backends get identical commands. Per trip, the pose is compared every step; prints the first divergence (step,
//! trip, fields) and the outcome counts on both backends.
//!   cargo run --release -p bayenv-gpu --example replay -- <data> [n=1000] [steps=6000] [steer_delay=0.15] [static_hold=1] [idle_creep=1] [tol=0.05]
use bayenv_gpu::{defines::ObsMode, sim::{GpuOptions, GpuPopulation}};
use std::{collections::HashMap, path::Path, sync::Arc};

fn hash(a: u64, b: u64) -> u64 { let mut h = a.wrapping_mul(0x9e3779b97f4a7c15) ^ b.wrapping_mul(0xc2b2ae3d27d4eb4f); h ^= h >> 31; h = h.wrapping_mul(0xbf58476d1ce4e5b9); h ^ (h >> 29) }
fn unit(h: u64) -> f32 { (h >> 40) as f32 / (1u64 << 24) as f32 }
/// policy-like action of trip `k` at step `t`
pub fn action(k: i64, t: usize) -> [f32; 3] {
    let k = k as u64;
    let swing = unit(hash(k, t as u64 / 6));            // a new steer target every 0.3 s
    let wob = unit(hash(k ^ 77, t as u64)) - 0.5;       // per-step noise
    let steer = ((swing * 2.0 - 1.0) * if unit(hash(k, t as u64 / 40 + 9)) < 0.2 { 1.0 } else { 0.15 } + 0.1 * wob).clamp(-1.0, 1.0);
    let ph = (hash(k, t as u64 / 60 + 1000) % 10) as u32;
    match ph {
        0..=4 => [steer, 0.25 + 0.2 * unit(hash(k, t as u64)), 0.0],
        5 => [steer, 0.0, 0.03],        // light brake (waiting)
        6 => [steer, 0.02, 0.02],       // gas + brake mix
        7 => [steer, 0.0, 0.0],         // coast / creep
        8 => [steer, 0.0, 0.5],
        _ => [steer, 0.05, 0.0],
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let arg = |i: usize, d: &str| a.get(i).cloned().unwrap_or(d.to_string());
    let data = arg(1, "../data");
    let n: usize = arg(2, "1000").parse().unwrap();
    let steps: usize = arg(3, "6000").parse().unwrap();
    let sd: f64 = arg(4, "0.15").parse().unwrap();
    let hold = arg(5, "1") != "0"; let creep = arg(6, "1") != "0";
    let tol: f32 = arg(7, "0.05").parse().unwrap();
    let start = 7.0 * 3600.0;
    let world = Arc::new(bayenv_world::World::load(Path::new(&data)).unwrap());
    let cap = (n.min(131072) + 1023) / 1024 * 1024;
    let mut o = GpuOptions { capacity: cap, start_time: start, seed: 777, obs: ObsMode::None, refuel_budget_ms: f64::INFINITY, ..Default::default() };
    o.steer_delay = sd; o.static_hold = hold; o.idle_creep = creep;
    let mut gpu = bayenv_gpu::env::create(world.clone(), o).unwrap();
    gpu.set_population(Arc::new(GpuPopulation::generate(&world, n, 777.0, true).unwrap()), true);
    let sw = Arc::new(bayenv_sim::world::World::from_bayenv(&world));
    let pop = Arc::new(bayenv_sim::population::Population::generate_like_env(&sw, "real", n, 777.0, Some(Path::new(&data))).unwrap());
    let mut opts = bayenv_sim::sim::SimOpts::default(); opts.dt = 0.05; opts.seed = 777.0;
    opts.steer_delay = sd; opts.static_hold = hold; opts.idle_creep = creep;
    let cfg = bayenv_sim::env::EnvConfig { opts, start_time: Some(start), static_obstacles: Vec::new(), scenario_obstacles: Vec::new(), obstacles: None };
    let mut cpu = bayenv_sim::env::Env::new(sw, pop, cfg, Box::new(|w| bayenv_sim::harness::router_for(&w)));
    let mut acts = vec![0f32; cap * 3];
    let (mut cc, mut gc, mut cb, mut gb) = ([0usize; 10], [0usize; 10], 0usize, 0usize);
    let mut first: Option<String> = None; let mut diverged = 0usize;
    let mut div_trips: HashMap<i64, usize> = HashMap::new();
    for t in 0..steps {
        let vs = &cpu.sim.vs;
        let rows: Vec<f32> = (0..vs.high).flat_map(|i| action(2 * vs.commuter[i] as i64 + vs.leg[i] as i64, t)).collect();
        cpu.act_rows(&rows, 0); cpu.step(1);
        let clog = cpu.take_records();
        for (k, o) in clog.kind.iter().zip(&clog.other) { cc[*k as usize] += 1; if *k == 3 && *o == -1 { cb += 1; } }
        let st = gpu.read_state();
        for i in 0..cap {
            let driving = st.u(i, 7) == 1; let tr = gpu.slot_trip[i];
            acts[3 * i..3 * i + 3].copy_from_slice(&if driving && tr >= 0 { action(tr as i64, t) } else { [0.0, 0.0, 0.3] });
        }
        gpu.set_actions(&acts, 0); gpu.run_steps(1);
        let log = gpu.take_records();
        for (k, o) in log.kind.iter().zip(&log.other) { gc[*k as usize] += 1; if *k == 3 && *o == -1 { gb += 1; } }
        // compare poses per trip
        let st = gpu.read_state();
        let vs = &cpu.sim.vs;
        let mut cpos: HashMap<i64, usize> = HashMap::new();
        for i in 0..vs.high { if vs.state[i] == bayenv_sim::vehicle::STATE_DRIVING { cpos.insert(2 * vs.commuter[i] as i64 + vs.leg[i] as i64, i); } }
        for g in 0..cap {
            if st.u(g, 7) != 1 { continue; } let tr = gpu.slot_trip[g] as i64; if tr < 0 { continue; }
            let Some(&c) = cpos.get(&tr) else { continue };
            // Frenet state (the GPU pose carries the blob's f32 vertex rounding, ~4 mm)
            let same = st.u(g, 0) as i32 == vs.edge[c] && st.f(g, 30) as i32 - 1 == vs.conn[c];
            let d = if same { (st.f(g, 1) - vs.s[c]).abs() + (st.f(g, 4) - vs.lat[c]).abs() + (st.f(g, 2) - vs.speed[c]).abs() + (st.f(g, 5) - vs.heading_rel[c]).abs() } else { 1e9 };
            if d > tol && !div_trips.contains_key(&tr) {
                div_trips.insert(tr, t); diverged += 1;
                let msg = format!("step {t} trip {tr}: |d| {d:.4} | cpu edge {} conn {} s {:.3} lat {:.3} u {:.3} hr {:.4} steer {:.4} sl {:.4} | gpu edge {} conn {} s {:.3} lat {:.3} u {:.3} hr {:.4} steer {:.4} sl {:.4}",
                    vs.edge[c], vs.conn[c], vs.s[c], vs.lat[c], vs.speed[c], vs.heading_rel[c], vs.steer_angle[c], vs.steer_sl[c],
                    st.u(g, 0), st.f(g, 30) as i32 - 1, st.f(g, 1), st.f(g, 4), st.f(g, 2), st.f(g, 5), st.f(g, 6), st.f(g, 25));
                if diverged <= 8 { println!("{msg}"); }
                if first.is_none() { first = Some(msg); }
            }
        }
    }
    // cars stopped > 5 min at the end
    let st = gpu.read_state(); let now = gpu.time;
    let gs = (0..cap).filter(|&g| st.u(g, 7) == 1 && st.f(g, 12) >= 0.0 && now - st.f(g, 12) as f64 > 300.0).count();
    let vs = &cpu.sim.vs; let cnow = cpu.sim.time;
    let cs = (0..vs.high).filter(|&i| vs.state[i] == bayenv_sim::vehicle::STATE_DRIVING && vs.stopped_since[i] >= 0.0 && cnow - vs.stopped_since[i] as f64 > 300.0).count();
    println!("steer_delay {sd} static_hold {hold} idle_creep {creep}: {diverged} trips diverged (> {tol} m)");
    println!("  arrived cpu {} gpu {} | crashed cpu {} gpu {} | contacts cpu {} gpu {} (barrier cpu {cb} gpu {gb}) | spawned cpu {} gpu {} | stopped > 5 min cpu {cs} gpu {gs}", cc[1], gc[1], cc[2], gc[2], cc[3], gc[3], cc[6], gc[6]);
}
