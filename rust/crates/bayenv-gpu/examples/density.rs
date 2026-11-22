//! Density closed loop on one backend (the obs-v2 driver of examples/snap_obs.rs: lane keeper + IDM-ish pedals + stops),
//! the same population (n commuters, seed 777, 07:00, dt 0.05); every 5 sim min: event rows by kind (arrived; CRASHED /
//! CONTACT, pair / barrier), crash events / minor contacts / crashed vehicles, and on the GPU the junction candidate-list
//! overflow stats 62 / 63. GPU flags from SNAP_FLAGS ("name=0,...").
//!   cargo run --release -p bayenv-gpu --example density -- <data> <cpu|gpu> [n=10000] [hours=1]
use bayenv_gpu::{defines::ObsMode, sim::{GpuOptions, GpuPopulation}};
use std::{path::Path, sync::Arc};

const W: usize = 70;

fn policy(o: &[f32]) -> [f32; 3] {
    let v = o[0] * 40.0; let vl = (o[1] * 40.0).max(5.0); let gap = o[8] * 120.0; let rel = o[9] * 20.0;
    let mut a = 0.8 * (0.9 * vl - v);
    let safe = 2.0 + 1.5 * v;
    if gap < safe + 15.0 { a = a.min(0.6 * (gap - safe) + 0.8 * rel); }
    let stopc = o[53] > 0.5 || o[52] > 0.5 || (o[47] > 0.5 && o[61] < 0.5);
    let d = o[56] * 250.0;
    let need = stopc && d < v * v / 3.0 + 8.0;
    if need { a = a.min(-(v * v) / (2.0 * (d - 1.5).max(0.5)) - 0.3); }
    let eps: f32 = std::env::var("POL_EPS").ok().and_then(|x| x.parse().ok()).unwrap_or(0.0);   // a perturbation of the steering gain (chaos check)
    let steer = ((-0.6 * o[4] + 0.6 * o[5]) * (1.0 + eps)).clamp(-1.0, 1.0);
    let (mut g, mut b) = ((a / 3.0).clamp(0.0, 1.0), (-a / 6.5).clamp(0.0, 1.0));
    if v < 0.5 && (need || gap < 4.0) { g = 0.0; b = 0.3; }
    [steer, g, b]
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = a[1].clone(); let backend = a[2].clone();
    let n: usize = a.get(3).map_or(10000, |x| x.parse().unwrap());
    let hours: f64 = a.get(4).map_or(1.0, |x| x.parse().unwrap());
    let start = 7.0 * 3600.0; let dt = 0.05;
    let steps = (hours * 3600.0 / dt) as usize; let every = (300.0 / dt) as usize;
    let w = Arc::new(bayenv_world::World::load(Path::new(&data)).unwrap());
    let mut ev = [0u64; 5]; // arrived, crash pair, crash bar, contact pair, contact bar
    let mut per: std::collections::HashMap<i64, u64> = Default::default();
    let count = |ev: &mut [u64; 5], per: &mut std::collections::HashMap<i64, u64>, r: &bayenv_world::events::EventLog| { for k in 0..r.len() { if r.kind[k] == 3 && r.other[k] < 0 { *per.entry(r.agent[k]).or_default() += 1; } match (r.kind[k], r.other[k] >= 0) { (1, _) => ev[0] += 1, (2, true) => ev[1] += 1, (2, false) => ev[2] += 1, (3, true) => ev[3] += 1, (3, false) => ev[4] += 1, _ => {} } } };
    let t0 = std::time::Instant::now();
    let top = |per: &std::collections::HashMap<i64, u64>| { let mut v: Vec<_> = per.iter().map(|(a, c)| (*c, *a)).collect(); v.sort_by(|a, b| b.cmp(a)); let tot: u64 = v.iter().map(|x| x.0).sum(); let t20: u64 = v.iter().take(20).map(|x| x.0).sum(); format!("barrier contact rows {tot} from {} trips (top 20: {t20}): {:?}", v.len(), &v[..v.len().min(20)]) };
    if backend == "cpu" {
        let sw = Arc::new(bayenv_sim::world::World::from_bayenv(&w));
        let pop = Arc::new(bayenv_sim::population::Population::generate_like_env(&sw, "real", n, 777.0, Some(Path::new(&data))).unwrap());
        let mut opts = bayenv_sim::sim::SimOpts::default(); opts.dt = dt; opts.seed = 777.0;
        let cfg = bayenv_sim::env::EnvConfig { opts, start_time: Some(start), static_obstacles: Vec::new(), scenario_obstacles: Vec::new(), obstacles: None };
        let mut cpu = bayenv_sim::env::Env::new(sw.clone(), pop, cfg, Box::new(|w| bayenv_sim::harness::router_for(&w)));
        let mut enc = bayenv_sim::obs_v2::V2Encoder::new(&sw);
        let mut cobs: Vec<f32> = Vec::new();
        for t in 0..steps {
            let vs = &cpu.sim.vs;
            let rows: Vec<f32> = (0..vs.high).flat_map(|i| if vs.state[i] == 1 && cobs.len() >= (i + 1) * W { policy(&cobs[i * W..(i + 1) * W]) } else { [0.0, 0.0, 0.3] }).collect();
            cpu.act_rows(&rows, 0); enc.record_actions(&cpu.sim); cpu.sim.step(); let r = cpu.take_records(); count(&mut ev, &mut per, &r);
            for x in cpu.sim.actions.iter_mut() { *x = None; }
            let h = cpu.sim.vs.high; cobs.clear(); cobs.resize(h * W, 0.0);
            for i in 0..h { if cpu.sim.vs.state[i] == 1 { enc.encode_v2(&mut cpu.sim, i, &mut cobs[i * W..(i + 1) * W]); } }
            if (t + 1) % every == 0 {
                let c = &cpu.sim.counters;
                println!("cpu t+{:4.0} min arrived {} crash_pair {} crash_bar {} contact_pair {} contact_bar {} | crash_events {} minor {} crashed_vehicles {} wall {:.0}s", (t + 1) as f64 * dt / 60.0, ev[0], ev[1], ev[2], ev[3], ev[4], c.crashes, c.minor_contacts, c.crashed_vehicles, t0.elapsed().as_secs_f64());
                if (t + 1) % (6 * every) == 0 { println!("   {}", top(&per)); }
            }
        }
    } else {
        let cap = 16384;
        let mut o = GpuOptions { capacity: cap, start_time: start, seed: 777, obs: ObsMode::V2, refuel_budget_ms: f64::INFINITY, dt, ..Default::default() };
        if let Ok(v) = std::env::var("SNAP_FLAGS") { for kv in v.split(',').filter(|x| !x.is_empty()) { let (k, b) = kv.split_once('=').unwrap_or((kv, "1")); o.set_flag(k, b != "0").unwrap(); } }
        let mut gpu = bayenv_gpu::env::create(w.clone(), o).unwrap();
        gpu.set_population(Arc::new(GpuPopulation::generate(&w, n, 777.0, true).unwrap()), true);
        let mut gobs = vec![0f32; cap * W];
        let (mut o62, mut o63) = (0u64, 0u64);
        for t in 0..steps {
            let st = gpu.read_state();
            let rows: Vec<f32> = (0..cap).flat_map(|i| if st.u(i, 7) == 1 { policy(&gobs[i * W..(i + 1) * W]) } else { [0.0, 0.0, 0.3] }).collect();
            gpu.set_actions(&rows, 0); gpu.run_steps(1);
            let r = gpu.take_records(); count(&mut ev, &mut per, &r);
            let s = gpu.stats(); o62 += s[62] as u64; o63 += s[63] as u64;
            gpu.read_obs_into(&mut gobs);
            if (t + 1) % every == 0 {
                let c = &gpu.counters;
                println!("gpu t+{:4.0} min arrived {} crash_pair {} crash_bar {} contact_pair {} contact_bar {} | crash_events {} minor {} crashed_vehicles {} | cand overflow swept {o62} path {o63} wall {:.0}s", (t + 1) as f64 * dt / 60.0, ev[0], ev[1], ev[2], ev[3], ev[4], c.crash_events, c.minor_contacts, c.crashed_vehicles, t0.elapsed().as_secs_f64());
                if (t + 1) % (6 * every) == 0 { println!("   {}", top(&per)); }
            }
        }
    }
}
