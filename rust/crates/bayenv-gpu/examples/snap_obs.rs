//! CPU state transplanted into the GPU, then one step on both with the same actions, obs v2 compared per index.
//! The CPU runs a population closed loop (a lane keeper + IDM-ish pedals on its own obs v2) to time t; its vehicle
//! store is exported as a GpuSim snapshot (`import_snapshot`), both backends take one step with the CPU policy's
//! actions, and every driving car's 70 inputs are diffed (per index: rows over 1e-3, and for stopped cars in boxes).
//! Crash scenes, control side records and junction state are not transplanted (inputs 56..61 may differ there).
//!   cargo run --release -p bayenv-gpu --example snap_obs -- <data> [n=10000] [minutes=60]
use bayenv_gpu::{defines::ObsMode, sim::{GpuOptions, GpuPopulation, Snapshot}};
use std::{collections::HashMap, path::Path, sync::Arc};

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
    let steer = (-0.6 * o[4] + 0.6 * o[5]).clamp(-1.0, 1.0);
    let (mut g, mut b) = ((a / 3.0).clamp(0.0, 1.0), (-a / 6.5).clamp(0.0, 1.0));
    if v < 0.5 && (need || gap < 4.0) { g = 0.0; b = 0.3; }
    [steer, g, b]
}

fn snapshot(sim: &bayenv_sim::sim::Sim) -> Snapshot {
    let vs = &sim.vs; let h = vs.high;
    let mut arrays: HashMap<String, Vec<f64>> = HashMap::new();
    let mut ring = Vec::new();
    for (name, dt, stride, bytes) in vs.fields() {
        let vals: Vec<f64> = match dt {
            "u8" => bytes.iter().map(|&b| b as f64).collect(), "i8" => bytes.iter().map(|&b| b as i8 as f64).collect(),
            "i32" => bytes.chunks(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f64).collect(),
            "u32" => bytes.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap()) as f64).collect(),
            "f32" => bytes.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap()) as f64).collect(),
            _ => continue,
        };
        if name == "ctrl_ring" { ring = vals.iter().map(|&x| x as f32).collect(); continue; }
        if stride != 1 { continue; }
        arrays.insert(bayenv_sim::vehicle::js_name(name), vals);
    }
    // routes: the slot's route edges, concatenated; the junction connector in `taper` (GPU vConn = taper - 1)
    let mut flat = Vec::new(); let (mut off, mut len) = (vec![0.0; h], vec![0.0; h]);
    for i in 0..h { off[i] = flat.len() as f64; len[i] = vs.routes[i].len() as f64; flat.extend_from_slice(&vs.routes[i]); }
    arrays.insert("routeOff".into(), off); arrays.insert("routeLen".into(), len);
    let taper: Vec<f64> = (0..h).map(|i| if vs.conn[i] >= 0 { vs.conn[i] as f64 + 1.0 } else { 0.0 }).collect();
    arrays.insert("taper".into(), taper);
    Snapshot { version: 4, time: sim.time, step: sim.step_count as u64, dt: sim.dt, high: h, arrays, ctrl_ring: ring, route_flat: flat, work_start: Some(sim.pop.work_start.clone()) }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = a.get(1).cloned().unwrap_or("../data".into());
    let n: usize = a.get(2).map_or(10000, |x| x.parse().unwrap());
    let minutes: f64 = a.get(3).map_or(60.0, |x| x.parse().unwrap());
    let start = 7.0 * 3600.0; let dt = 0.05;
    let w = Arc::new(bayenv_world::World::load(Path::new(&data)).unwrap());
    let sw = Arc::new(bayenv_sim::world::World::from_bayenv(&w));
    let pop = Arc::new(bayenv_sim::population::Population::generate_like_env(&sw, "real", n, 777.0, Some(Path::new(&data))).unwrap());
    let mut opts = bayenv_sim::sim::SimOpts::default(); opts.dt = dt; opts.seed = 777.0;
    let cfg = bayenv_sim::env::EnvConfig { opts, start_time: Some(start), static_obstacles: Vec::new(), scenario_obstacles: Vec::new(), obstacles: None };
    let mut cpu = bayenv_sim::env::Env::new(sw.clone(), pop, cfg, Box::new(|w| bayenv_sim::harness::router_for(&w)));
    let mut enc = bayenv_sim::obs_v2::V2Encoder::new(&sw);
    let mut cobs: Vec<f32> = Vec::new();
    let steps = (minutes * 60.0 / dt) as usize;
    let rows_for = |cpu: &bayenv_sim::env::Env, cobs: &[f32]| -> Vec<f32> { (0..cpu.sim.vs.high).flat_map(|i| if cpu.sim.vs.state[i] == 1 && cobs.len() >= (i + 1) * W { policy(&cobs[i * W..(i + 1) * W]) } else { [0.0, 0.0, 0.3] }).collect() };
    let encode = |cpu: &mut bayenv_sim::env::Env, enc: &mut bayenv_sim::obs_v2::V2Encoder, cobs: &mut Vec<f32>| {
        let n = cpu.sim.vs.high; cobs.clear(); cobs.resize(n * W, 0.0);
        for i in 0..n { if cpu.sim.vs.state[i] == 1 { enc.encode_v2(&mut cpu.sim, i, &mut cobs[i * W..(i + 1) * W]); } }
    };
    for _ in 0..steps {
        let rows = rows_for(&cpu, &cobs); cpu.act_rows(&rows, 0); enc.record_actions(&cpu.sim); cpu.sim.step(); let _ = cpu.take_records();
        for x in cpu.sim.actions.iter_mut() { *x = None; }
        encode(&mut cpu, &mut enc, &mut cobs);
    }
    // transplant
    let cap = ((cpu.sim.vs.high + 1023) / 1024 * 1024).max(1024);
    let mut o = GpuOptions { capacity: cap, start_time: start, seed: 777, obs: ObsMode::V2, refuel_budget_ms: f64::INFINITY, dt, ..Default::default() };
    o.features.accidents = false;
    if let Ok(v) = std::env::var("SNAP_FLAGS") { for kv in v.split(',').filter(|x| !x.is_empty()) { let (k, b) = kv.split_once('=').unwrap_or((kv, "1")); o.set_flag(k, b != "0").unwrap(); } }
    let mut gpu = bayenv_gpu::env::create(w.clone(), o).unwrap();
    gpu.set_population(Arc::new(GpuPopulation::generate(&w, n, 777.0, true).unwrap()), true);
    gpu.import_snapshot(&snapshot(&cpu.sim)).unwrap();
    // one step on both with the CPU policy's actions
    let rows = rows_for(&cpu, &cobs);
    let mut ga = vec![0f32; cap * 3]; ga[..rows.len()].copy_from_slice(&rows); for i in rows.len() / 3..cap { ga[3 * i + 2] = 0.3; }
    cpu.act_rows(&rows, 0); enc.record_actions(&cpu.sim); cpu.sim.step(); let _ = cpu.take_records(); for x in cpu.sim.actions.iter_mut() { *x = None; }
    encode(&mut cpu, &mut enc, &mut cobs);
    gpu.set_actions(&ga, 0); gpu.run_steps(1);
    let mut gobs = vec![0f32; cap * W]; gpu.read_obs_into(&mut gobs);
    let st = gpu.read_state(); let vs = &cpu.sim.vs;
    let (mut over, mut over_box, mut rows_n, mut box_n) = (vec![0usize; W], vec![0usize; W], 0, 0);
    let mut samples: Vec<String> = Vec::new();
    for i in 0..vs.high {
        if vs.state[i] != 1 || st.u(i, 7) != 1 { continue; }
        let same = st.u(i, 0) as i32 == vs.edge[i] && (st.f(i, 1) - vs.s[i]).abs() < 0.01 && (st.f(i, 2) - vs.speed[i]).abs() < 0.01;
        if !same { continue; }
        rows_n += 1;
        let inbox = vs.conn[i] >= 0 && vs.speed[i] < 0.1; if inbox { box_n += 1; }
        for k in 0..W {
            let (c, g) = (cobs[i * W + k], gobs[i * W + k]);
            if (c - g).abs() > 1e-3 { over[k] += 1; if inbox { over_box[k] += 1; } if (inbox || std::env::var("SNAP_ALL").is_ok()) && samples.len() < 40 { samples.push(format!("slot {i} conn {} edge {} s {:.1} v {:.1} k {k}: cpu {c:.3} gpu {g:.3}", vs.conn[i], vs.edge[i], vs.s[i], vs.speed[i])); } }
        }
    }
    for i in 0..vs.high {
        if vs.state[i] != 1 || st.u(i, 7) != 1 || vs.conn[i] < 0 { continue; }
        let (c, g) = (cobs[i * W + 8], gobs[i * W + 8]); if (c - g).abs() < 1e-3 { continue; }
        let gl = st.u(i, 20) as i32; let gg = st.f(i, 21);
        let d = |j: i32| if j >= 0 { let j = j as usize; format!("{j} (edge {} conn {} s {:.1} lane {} st {} v {:.2} | its leader cpu {} gap {:.2} gpu {} gap {:.2})", vs.edge[j], vs.conn[j], vs.s[j], vs.lane[j], vs.state[j], vs.speed[j], vs.leader[j], vs.gap[j], st.u(j, 20) as i32, st.f(j, 21)) } else { format!("{j}") };
        println!("  box row slot {i} edge {} conn {} s {:.1} lane {} lo/hi {}/{} lat {:.2} gpu lat {:.2} lanes {:08x}: cpu o8 {:.2} m leader {} gap {:.2} | gpu o8 {:.2} m leader {} gap {:.2}", vs.edge[i], vs.conn[i], vs.s[i], vs.lane[i], vs.lane_lo[i], vs.lane_hi[i], vs.lat[i], st.f(i, 4), st.u(i, 14), c * 120.0, d(vs.leader[i]), vs.gap[i], g * 120.0, d(gl), gg);
    }
    println!("t {} min, {} driving cars in step compared ({} stopped in a box)", minutes, rows_n, box_n);
    for k in 0..W { if over[k] > 0 { println!("  input {k:2}: {} rows over 1e-3 ({} of the stopped-in-box rows)", over[k], over_box[k]); } }
    for s in samples { println!("    {s}"); }
}
