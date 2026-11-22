//! The junction-conflict obstacle in the own-ahead obs slot (GpuOptions::junction_obstacle, DECISIONS.md 45): a CPU
//! closed loop (3,000 commuters, seed 777, 07:00, a lane keeper + IDM-ish pedals on its own obs v2) runs 30 min; every
//! 2 min from 10 min its vehicle store is transplanted into the GPU (`import_snapshot`), both take one step with the same actions, and for
//! every driving car with a junction conflict ahead (CPU input 65 < 1) the own-ahead gap (input 8) is compared. The
//! CPU encoder puts max(0, conflict distance - 2) there when the conflicting body is stopped or reaches the conflict
//! point first; with the flag the GPU must too (no row where the CPU sees the obstacle and the GPU a freer lane), and
//! js_model's setting (the JS GPU, no obstacle) must show such rows. GpuOptions::waits_prev_step: no car in a box may
//! take as its leader a stopped car whose leader (at the step's start) is the car itself (the CPU's slot-order "waits
//! for me" skips it). GpuOptions::box_standstill: no row may have a smaller own-ahead gap than the CPU (mutual
//! standstill pairs in boxes, bodies behind on my edge); without the flag some box rows do. Skips without data / a GPU.
use bayenv_gpu::{defines::ObsMode, sim::{GpuOptions, GpuPopulation, Snapshot}};
use std::{collections::HashMap, path::Path, sync::Arc};

const W: usize = 70;
const N: usize = 3000;

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

#[test]
fn junction_obstacle_in_own_ahead_slot_matches_cpu() {
    let data = "../../../data".to_string();
    let w = match bayenv_world::World::load(Path::new(&data)) { Ok(w) => Arc::new(w), Err(e) => { bayenv_world::oracle::skip("junction_obstacle", &format!("world: {e}")); return } };
    let start = 7.0 * 3600.0; let dt = 0.05;
    let sw = Arc::new(bayenv_sim::world::World::from_bayenv(&w));
    let pop = Arc::new(bayenv_sim::population::Population::generate_like_env(&sw, "real", N, 777.0, Some(Path::new(&data))).unwrap());
    let mut opts = bayenv_sim::sim::SimOpts::default(); opts.dt = dt; opts.seed = 777.0;
    let cfg = bayenv_sim::env::EnvConfig { opts, start_time: Some(start), static_obstacles: Vec::new(), scenario_obstacles: Vec::new(), obstacles: None };
    let mut cpu = bayenv_sim::env::Env::new(sw.clone(), pop, cfg, Box::new(|w| bayenv_sim::harness::router_for(&w)));
    let mut enc = bayenv_sim::obs_v2::V2Encoder::new(&sw);
    let mut cobs: Vec<f32> = Vec::new();
    let rows_for = |cpu: &bayenv_sim::env::Env, cobs: &[f32]| -> Vec<f32> { (0..cpu.sim.vs.high).flat_map(|i| if cpu.sim.vs.state[i] == 1 && cobs.len() >= (i + 1) * W { policy(&cobs[i * W..(i + 1) * W]) } else { [0.0, 0.0, 0.3] }).collect() };
    let encode = |cpu: &mut bayenv_sim::env::Env, enc: &mut bayenv_sim::obs_v2::V2Encoder, cobs: &mut Vec<f32>| {
        let n = cpu.sim.vs.high; cobs.clear(); cobs.resize(n * W, 0.0);
        for i in 0..n { if cpu.sim.vs.state[i] == 1 { enc.encode_v2(&mut cpu.sim, i, &mut cobs[i * W..(i + 1) * W]); } }
    };
    let cap = 4096;
    let mut gpus = Vec::new();
    for (on, wp, bs) in [(true, true, true), (false, true, true), (true, false, true), (true, true, false)] {
        let mut o = GpuOptions { capacity: cap, start_time: start, seed: 777, obs: ObsMode::V2, refuel_budget_ms: f64::INFINITY, dt, ..Default::default() };
        o.features.accidents = false; o.junction_obstacle = on; o.waits_prev_step = wp; o.box_standstill = bs;
        let mut gpu = match bayenv_gpu::env::create(w.clone(), o) { Ok(s) => s, Err(e) => { eprintln!("junction_obstacle: no GPU ({e}): not run"); return } };
        gpu.set_population(Arc::new(GpuPopulation::generate(&w, N, 777.0, true).unwrap()), true);
        gpus.push(gpu);
    }
    let mut res = vec![(0usize, 0usize, 0usize); 4]; let mut extra = [0usize; 4]; let mut follows = [0usize; 4]; let mut box_below = [0usize; 4]; let mut box_led = 0usize;
    let per = (2.0 * 60.0 / dt) as usize;
    for k in 0..(30.0 * 60.0 / dt) as usize {
        let rows = rows_for(&cpu, &cobs);
        let check = k >= 5 * per && k % per == 0;
        let snap = if check { Some(snapshot(&cpu.sim)) } else { None };
        cpu.act_rows(&rows, 0); enc.record_actions(&cpu.sim); cpu.sim.step(); let _ = cpu.take_records();
        for x in cpu.sim.actions.iter_mut() { *x = None; }
        encode(&mut cpu, &mut enc, &mut cobs);
        let Some(snap) = snap else { continue };
        assert!(cpu.sim.vs.high <= cap);
        for (gk, gpu) in gpus.iter_mut().enumerate() {
            gpu.import_snapshot(&snap).unwrap();
            let mut ga = vec![0f32; cap * 3]; ga[..rows.len()].copy_from_slice(&rows); for i in rows.len() / 3..cap { ga[3 * i + 2] = 0.3; }
            gpu.set_actions(&ga, 0); gpu.run_steps(1);
            let mut gobs = vec![0f32; cap * W]; gpu.read_obs_into(&mut gobs);
            let st = gpu.read_state(); let vs = &cpu.sim.vs;
            let prev = &snap.arrays["leader"];
            for i in 0..vs.high {
                // waits_prev_step: a GPU leader that is stopped and was (at the step's start) following me
                if vs.state[i] == 1 && st.u(i, 7) == 1 && vs.conn[i] >= 0 {
                    let l = st.u(i, 20) as i32;
                    if gk == 0 && l >= 0 { box_led += 1; }
                    if gobs[i * W + 8] < cobs[i * W + 8] - 5e-3 { box_below[gk] += 1; }
                    if l >= 0 && (l as usize) < prev.len() && prev[l as usize] as i32 == i as i32 && vs.state[l as usize] == 1 && st.f(l as usize, 2) < 0.5 { follows[gk] += 1; }
                }
                if vs.state[i] != 1 || st.u(i, 7) != 1 { continue; }
                if !(st.u(i, 0) as i32 == vs.edge[i] && (st.f(i, 1) - vs.s[i]).abs() < 0.01 && (st.f(i, 2) - vs.speed[i]).abs() < 0.01) { continue; }
                let (c, g) = (&cobs[i * W..(i + 1) * W], &gobs[i * W..(i + 1) * W]);
                if g[8] < c[8] - 5e-3 { extra[gk] += 1; }
                if c[65] >= 1.0 || (c[65] - g[65]).abs() > 1e-3 || (c[67] - g[67]).abs() > 1e-3 { continue; }
                res[gk].0 += 1;
                let jgap = (c[65] as f64 * 60.0 - 2.0).max(0.0) / 120.0;
                if (c[8] as f64 - jgap).abs() < 2e-3 && c[8] < 1.0 { res[gk].1 += 1; if g[8] > c[8] + 5e-3 { res[gk].2 += 1; } }
            }
        }
    }
    for (gk, r) in res.iter().enumerate() { eprintln!("variant {gk} (junction_obstacle {}, waits_prev_step {}, box_standstill {}): {} cars with a junction conflict ahead, {} with it as the CPU's own-ahead obstacle, GPU missed {}; rows with a GPU own-ahead gap below the CPU's: {}; box cars following a stopped car that waits for them: {} (of {box_led} box cars with a car leader); box rows with a GPU own-ahead gap below the CPU's: {}", gk != 1, gk != 2, gk != 3, r.0, r.1, r.2, extra[gk], follows[gk], box_below[gk]); }
    let ((c1, o1, m1), (_, _, m0)) = (res[0], res[1]);
    assert!(follows[0] == 0, "waits_prev_step on: {} box cars follow a stopped car that waits for them", follows[0]);
    // (waits_prev_step off: no such pair at 3,000 commuters; examples/snap_obs.rs shows them at 10,000, e.g. 30 min)
    assert!(extra[0] == 0, "all on: {} rows where the GPU's own-ahead gap is below the CPU's", extra[0]);
    assert!(box_below[3] > 0, "box_standstill off shows no box row below the CPU's gap: the test does not exercise the flag");
    assert!(c1 >= 20 && o1 >= 5, "too few cases: {c1} conflicts, {o1} obstacles");
    assert!(m1 == 0, "junction_obstacle on: {m1} of {o1} rows where the GPU misses the CPU's junction obstacle");
    assert!(m0 > 0, "junction_obstacle off (the JS GPU) shows no missed obstacle: the test does not exercise the flag");
}
