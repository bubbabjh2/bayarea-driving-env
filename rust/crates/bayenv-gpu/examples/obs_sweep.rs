//! Input parity sweep: the CPU's vehicle store imported into the GPU (`import_snapshot`), one step on both with the
//! same actions, every input index compared (obs v2 or v3). The CPU runs a closed loop (examples/snap_obs.rs driver
//! on inputs 0..70) to each check time; each check re-imports. Compared: driving cars present at the import on both
//! backends whose edge / s / speed agree after the step, not on an edge where either backend spawned a car in the
//! step (the spawn queue is not imported). Printed: every index over 1e-3 in more than 0.1 % of the rows, with counts
//! and samples (SWEEP_SAMPLES=k for k samples per index).
//!   cargo run --release -p bayenv-gpu --example obs_sweep -- <data> [n=10000] [v2|v3] [minutes=10,30,60]
use bayenv_gpu::{defines::ObsMode, sim::{GpuOptions, GpuPopulation, Snapshot}};
use std::{collections::{HashMap, HashSet}, path::Path, sync::Arc};

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
    let v3 = a.get(3).map_or(false, |x| x == "v3");
    let mins: Vec<f64> = a.get(4).map_or("10,30,60".into(), |x| x.clone()).split(',').map(|x| x.parse().unwrap()).collect();
    let nsamp: usize = std::env::var("SWEEP_SAMPLES").ok().and_then(|x| x.parse().ok()).unwrap_or(3);
    let ow = if v3 { bayenv_sim::obs::OBS3 } else { W };
    let start = 7.0 * 3600.0; let dt = 0.05;
    let w = Arc::new(bayenv_world::World::load(Path::new(&data)).unwrap());
    let sw = Arc::new(bayenv_sim::world::World::from_bayenv(&w));
    let pop = Arc::new(bayenv_sim::population::Population::generate_like_env(&sw, "real", n, 777.0, Some(Path::new(&data))).unwrap());
    let mut opts = bayenv_sim::sim::SimOpts::default(); opts.dt = dt; opts.seed = 777.0;
    let cfg = bayenv_sim::env::EnvConfig { opts, start_time: Some(start), static_obstacles: Vec::new(), scenario_obstacles: Vec::new(), obstacles: None };
    let mut cpu = bayenv_sim::env::Env::new(sw.clone(), pop, cfg, Box::new(|w| bayenv_sim::harness::router_for(&w)));
    let mut e2 = bayenv_sim::obs_v2::V2Encoder::new(&sw); let mut e3 = bayenv_sim::obs_v3::V3Encoder::new(&sw);
    let mut cobs: Vec<f32> = Vec::new();
    let rows_for = |cpu: &bayenv_sim::env::Env, cobs: &[f32]| -> Vec<f32> { (0..cpu.sim.vs.high).flat_map(|i| if cpu.sim.vs.state[i] == 1 && cobs.len() >= (i + 1) * ow { policy(&cobs[i * ow..i * ow + W]) } else { [0.0, 0.0, 0.3] }).collect() };
    let step_cpu = |cpu: &mut bayenv_sim::env::Env, cobs: &mut Vec<f32>, rows: &[f32], e2: &mut bayenv_sim::obs_v2::V2Encoder, e3: &mut bayenv_sim::obs_v3::V3Encoder| {
        cpu.act_rows(rows, 0); if v3 { e3.record_actions(&cpu.sim) } else { e2.record_actions(&cpu.sim) }
        cpu.sim.step(); let _ = cpu.take_records(); for x in cpu.sim.actions.iter_mut() { *x = None; }
        let h = cpu.sim.vs.high; cobs.clear(); cobs.resize(h * ow, 0.0);
        for i in 0..h { if cpu.sim.vs.state[i] == 1 { if v3 { e3.encode(&mut cpu.sim, i, &mut cobs[i * ow..(i + 1) * ow]) } else { e2.encode_v2(&mut cpu.sim, i, &mut cobs[i * ow..(i + 1) * ow]) } } }
    };
    let cap = 16384;
    let mut o = GpuOptions { capacity: cap, start_time: start, seed: 777, obs: if v3 { ObsMode::V3 } else { ObsMode::V2 }, refuel_budget_ms: f64::INFINITY, dt, ..Default::default() };
    o.features.accidents = false;
    if let Ok(v) = std::env::var("SNAP_FLAGS") { for kv in v.split(',').filter(|x| !x.is_empty()) { let (k, b) = kv.split_once('=').unwrap_or((kv, "1")); o.set_flag(k, b != "0").unwrap(); } }
    let mut gpu = bayenv_gpu::env::create(w.clone(), o).unwrap();
    gpu.set_population(Arc::new(GpuPopulation::generate(&w, n, 777.0, true).unwrap()), true);
    let mut gobs = vec![0f32; cap * ow];
    let mut t = 0usize;
    for &m in &mins {
        let target = (m * 60.0 / dt) as usize;
        while t < target { let rows = rows_for(&cpu, &cobs); step_cpu(&mut cpu, &mut cobs, &rows, &mut e2, &mut e3); t += 1; }
        let snap = snapshot(&cpu.sim);
        assert!(cpu.sim.vs.high <= cap);
        gpu.import_snapshot(&snap).unwrap();
        // the CPU's active crash / breakdown scenes (their edge blocks are scene obstacles for the leaders pass)
        let scenes: Vec<bayenv_gpu::sim::SceneImport> = cpu.sim.active.iter().map(|&k| &cpu.sim.crashes[k]).filter(|r| r.edge >= 0).map(|r| bayenv_gpu::sim::SceneImport {
            time: r.time, vehicles: r.vehicles.iter().map(|&v| v as u32).collect(), lane_mask: r.lane_mask as u32, edge: r.edge as u32, s: r.s, x: r.x, y: r.y, z: r.z,
            block_end: r.block_end, clear_end: r.clear_end, squeeze_at: r.squeeze_at, squeeze: r.squeeze,
            response: match r.response { bayenv_sim::sim::Response::Active => 0, bayenv_sim::sim::Response::Shoulder => 1, _ => 2 }, is_crash: r.is_crash }).collect();
        if std::env::var("SWEEP_NO_SCENES").is_err() { gpu.import_scenes(&scenes); }
        if v3 && std::env::var("SWEEP_NO_LHIST").is_err() {
            // the CPU encoder's leader histories (inputs 94..100)
            let recs: Vec<(usize, [f32; 45])> = e3.v2.slots.iter().enumerate().filter_map(|(i, t)| {
                let t = t.as_ref()?; if i >= cpu.sim.vs.high || cpu.sim.vs.state[i] != 1 || t.key.2 != cpu.sim.vs.spawn_time[i].to_bits() { return None; }
                let h = &t.lhist; let mut r = [0f32; 45];
                r[0] = cpu.sim.vs.spawn_time[i]; r[1] = h.n as f32; r[2] = h.head as f32; r[3] = h.last_leader as f32; r[4] = h.since_change as f32;
                r[5..25].copy_from_slice(&h.g); r[25..45].copy_from_slice(&h.r); Some((i, r)) }).collect();
            gpu.import_leader_history(&recs);
        }
        let rows = rows_for(&cpu, &cobs);
        let mut ga = vec![0f32; cap * 3]; ga[..rows.len()].copy_from_slice(&rows); for i in rows.len() / 3..cap { ga[3 * i + 2] = 0.3; }
        step_cpu(&mut cpu, &mut cobs, &rows, &mut e2, &mut e3); t += 1;
        gpu.set_actions(&ga, 0); gpu.run_steps(1); let _ = gpu.take_records();
        gpu.read_obs_into(&mut gobs);
        let st = gpu.read_state(); let vs = &cpu.sim.vs;
        let sst = &snap.arrays["state"]; let ssp = &snap.arrays["spawnTime"];
        let fresh = |i: usize, sp: f32| i >= sst.len() || sst[i] == 0.0 || ssp[i] as f32 != sp;
        let mut new_edges = HashSet::new();
        for i in 0..cap { if st.u(i, 7) == 1 && fresh(i, st.f(i, 18)) { new_edges.insert(st.u(i, 0) as i32); } }
        for i in 0..vs.high { if vs.state[i] == 1 && fresh(i, vs.spawn_time[i]) { new_edges.insert(vs.edge[i]); } }
        // positions of the cars spawned in the step (either backend): rows within 150 m are skipped (their neighbours,
        // junction conflicts and yields can see a car the other backend does not have)
        let mut fresh_pts: Vec<(f64, f64)> = Vec::new();
        for i in 0..cap { if st.u(i, 7) == 1 && fresh(i, st.f(i, 18)) { let p = sw.edge_point_dir(st.u(i, 0) as usize, st.f(i, 1) as f64, st.u(i, 9) as i32); fresh_pts.push((p.x, p.z)); } }
        for i in 0..vs.high { if vs.state[i] == 1 && fresh(i, vs.spawn_time[i]) { fresh_pts.push((vs.x[i] as f64, vs.z[i] as f64)); } }
        let near_fresh = |i: usize| fresh_pts.iter().any(|&(x, z)| (x - vs.x[i] as f64).hypot(z - vs.z[i] as f64) < 150.0);
        let mut over = vec![0usize; ow]; let mut samples: HashMap<usize, Vec<String>> = HashMap::new(); let mut rows_n = 0usize;
        for i in 0..vs.high {
            if vs.state[i] != 1 || st.u(i, 7) != 1 || fresh(i, vs.spawn_time[i]) || new_edges.contains(&vs.edge[i]) || near_fresh(i) { continue; }
            if !(st.u(i, 0) as i32 == vs.edge[i] && (st.f(i, 1) - vs.s[i]).abs() < 0.01 && (st.f(i, 2) - vs.speed[i]).abs() < 0.01) { continue; }
            rows_n += 1;
            for k in 0..ow {
                let (c, g) = (cobs[i * ow + k], gobs[i * ow + k]);
                if (c - g).abs() > 1e-3 { over[k] += 1; let sv = samples.entry(k).or_default(); if sv.len() < nsamp { sv.push({ let rp = vs.route_pos[i] as usize; let cum: f64 = vs.routes[i][..rp.min(vs.routes[i].len())].iter().map(|&q| sw.e_length[q as usize]).sum(); format!("slot {i} edge {} conn {} s {:.1} v {:.1} lane {} rp {rp} pos-odo {:.2} | leader cpu {} {:.2} gpu {} {:.2}: cpu {c:.4} gpu {g:.4}", vs.edge[i], vs.conn[i], vs.s[i], vs.speed[i], vs.lane[i], cum + vs.s[i] as f64 - vs.route_dist[i] as f64, vs.leader[i], vs.gap[i], st.u(i, 20) as i32, st.f(i, 21)) + &{ let l = vs.leader[i]; if l >= 0 { let l = l as usize; format!(" [leader st {} s {:.2} len {:.2} w {:.2} | raw {:.2} | lat/hr cpu me {:.3}/{:.3} L {:.3}/{:.3} gpu me {:.3}/{:.3} L {:.3}/{:.3} | w me {:.2}]", vs.state[l], vs.s[l], vs.length[l], vs.width[l], vs.s[l] - vs.s[i] - 0.5 * (vs.length[i] + vs.length[l]), vs.lat[i], vs.heading_rel[i], vs.lat[l], vs.heading_rel[l], st.f(i, 4), st.f(i, 5), st.f(l, 4), st.f(l, 5), vs.width[i]) } else { String::new() } } }); } }
            }
        }
        if let Ok(dk) = std::env::var("SWEEP_DUMP") {
            // the bodies block of every row where input `dk` differs: cpu / gpu per body, with the slots near
            let dk: usize = dk.parse().unwrap();
            for i in 0..vs.high {
                if vs.state[i] != 1 || st.u(i, 7) != 1 || (cobs[i * ow + dk] - gobs[i * ow + dk]).abs() <= 1e-3 { continue; }
                if fresh(i, vs.spawn_time[i]) || new_edges.contains(&vs.edge[i]) || near_fresh(i) { continue; }
                if !(st.u(i, 0) as i32 == vs.edge[i] && (st.f(i, 1) - vs.s[i]).abs() < 0.01) { continue; }
                println!("DUMP slot {i} edge {} s {:.2} lat {:.2} v {:.2} x {:.2} z {:.2}", vs.edge[i], vs.s[i], vs.lat[i], vs.speed[i], vs.x[i], vs.z[i]);
                for b in 0..16 { let at = i * ow + 102 + 16 * b; let c = &cobs[at..at + 16]; let g = &gobs[at..at + 16]; if c.iter().zip(g).any(|(a, b)| (a - b).abs() > 1e-3) || c[15] > 0.0 {
                    println!("   body {b:2} cpu {:?}\n           gpu {:?}", c.iter().map(|x| (x * 1000.0).round() / 1000.0).collect::<Vec<_>>(), g.iter().map(|x| (x * 1000.0).round() / 1000.0).collect::<Vec<_>>()); } }
                for j in 0..vs.high { if j != i && (vs.state[j] == 1 || vs.state[j] == 2) && ((vs.x[j] - vs.x[i]).hypot(vs.z[j] - vs.z[i])) < 60.0 { println!("     near {j} st {} edge {} s {:.2} lat {:.2} v {:.2} d {:.1} | gpu st {} edge {} s {:.2} lat {:.2}", vs.state[j], vs.edge[j], vs.s[j], vs.lat[j], vs.speed[j], (vs.x[j] - vs.x[i]).hypot(vs.z[j] - vs.z[i]), st.u(j, 7), st.u(j, 0), st.f(j, 1), st.f(j, 4)); } }
                break;
            }
        }
        println!("== {} {} commuters, t {m} min: {rows_n} rows compared", if v3 { "v3" } else { "v2" }, n);
        for k in 0..ow { if over[k] > 0 { let flag = if over[k] * 1000 > rows_n { "  ***" } else { "" }; println!("  input {k:3}: {:5} rows ({:.2} %){flag}", over[k], 100.0 * over[k] as f64 / rows_n as f64); if !flag.is_empty() { for s in &samples[&k] { println!("        {s}"); } } } }
    }
}
