//! Contact events at density, CPU vs GPU from the same state: the CPU closed loop (examples/snap_obs.rs policy) runs to
//! t; its vehicle store goes into the GPU (`import_snapshot`); then both run K more steps closed loop on their own obs
//! v2 with the same policy, and every CRASHED / CONTACT event row is classified (pair / barrier; barrier rows by where
//! the car is: on a connector, the box part of its edge, the edge proper; dv <= 3 km/h vs above). The first 2 steps
//! are skipped (contact / touch state is not transplanted).
//!   cargo run --release -p bayenv-gpu --example contact_diff -- <data> [n=10000] [minutes=20] [steps=600]
use bayenv_gpu::{defines::ObsMode, sim::{GpuOptions, GpuPopulation, Snapshot}};
use std::{collections::{BTreeMap, HashMap}, path::Path, sync::Arc};

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
    let minutes: f64 = a.get(3).map_or(20.0, |x| x.parse().unwrap());
    let k_steps: usize = a.get(4).map_or(600, |x| x.parse().unwrap());
    let start = 7.0 * 3600.0; let dt = 0.05;
    let w = Arc::new(bayenv_world::World::load(Path::new(&data)).unwrap());
    let sw = Arc::new(bayenv_sim::world::World::from_bayenv(&w));
    let pop = Arc::new(bayenv_sim::population::Population::generate_like_env(&sw, "real", n, 777.0, Some(Path::new(&data))).unwrap());
    let mut opts = bayenv_sim::sim::SimOpts::default(); opts.dt = dt; opts.seed = 777.0;
    let cfg = bayenv_sim::env::EnvConfig { opts, start_time: Some(start), static_obstacles: Vec::new(), scenario_obstacles: Vec::new(), obstacles: None };
    let mut cpu = bayenv_sim::env::Env::new(sw.clone(), pop, cfg, Box::new(|w| bayenv_sim::harness::router_for(&w)));
    let mut enc = bayenv_sim::obs_v2::V2Encoder::new(&sw);
    let mut cobs: Vec<f32> = Vec::new();
    let rows_for = |high: usize, alive: &dyn Fn(usize) -> bool, obs: &[f32]| -> Vec<f32> { (0..high).flat_map(|i| if alive(i) && obs.len() >= (i + 1) * W { policy(&obs[i * W..(i + 1) * W]) } else { [0.0, 0.0, 0.3] }).collect() };
    let encode = |cpu: &mut bayenv_sim::env::Env, enc: &mut bayenv_sim::obs_v2::V2Encoder, cobs: &mut Vec<f32>| {
        let n = cpu.sim.vs.high; cobs.clear(); cobs.resize(n * W, 0.0);
        for i in 0..n { if cpu.sim.vs.state[i] == 1 { enc.encode_v2(&mut cpu.sim, i, &mut cobs[i * W..(i + 1) * W]); } }
    };
    let cstep = |cpu: &mut bayenv_sim::env::Env, enc: &mut bayenv_sim::obs_v2::V2Encoder, cobs: &mut Vec<f32>| -> bayenv_world::events::EventLog {
        let vs = &cpu.sim.vs; let rows = rows_for(vs.high, &|i| vs.state[i] == 1, cobs);
        cpu.act_rows(&rows, 0); enc.record_actions(&cpu.sim); cpu.sim.step(); let r = cpu.take_records();
        for x in cpu.sim.actions.iter_mut() { *x = None; }
        encode(cpu, enc, cobs); r
    };
    for _ in 0..(minutes * 60.0 / dt) as usize { let _ = cstep(&mut cpu, &mut enc, &mut cobs); }
    let cap = ((cpu.sim.vs.high + 2047) / 1024 * 1024).max(1024);
    let mut o = GpuOptions { capacity: cap, start_time: start, seed: 777, obs: ObsMode::V2, refuel_budget_ms: f64::INFINITY, dt, ..Default::default() };
    if let Ok(v) = std::env::var("SNAP_FLAGS") { for kv in v.split(',').filter(|x| !x.is_empty()) { let (k, b) = kv.split_once('=').unwrap_or((kv, "1")); o.set_flag(k, b != "0").unwrap(); } }
    let mut gpu = bayenv_gpu::env::create(w.clone(), o).unwrap();
    gpu.set_population(Arc::new(GpuPopulation::generate(&w, n, 777.0, true).unwrap()), true);
    gpu.import_snapshot(&snapshot(&cpu.sim)).unwrap();
    let _ = gpu.take_records();
    // only cars present at the transplant (the spawn queue is not transplanted: later spawns differ)
    let h0 = cpu.sim.vs.high; let sp0: Vec<f32> = (0..h0).map(|i| if cpu.sim.vs.state[i] != 0 { cpu.sim.vs.spawn_time[i] } else { f32::NAN }).collect();
    let g = &sw;
    let place = |edge: i32, conn: i32, s: f32| -> &'static str {
        if conn >= 0 { return "conn"; }
        let e = edge as usize; let s = s as f64;
        if s > g.e_s_stop[e] || s < g.e_s_exit[e] { "box" } else { "edge" }
    };
    let mut cnt: [BTreeMap<String, usize>; 2] = [BTreeMap::new(), BTreeMap::new()];
    let mut diverged = vec![false; h0]; let mut prev_od = vec![String::new(); h0];
    let mut per: [BTreeMap<usize, (usize, i32, f32, f32)>; 2] = [BTreeMap::new(), BTreeMap::new()];
    let mut gobs = vec![0f32; cap * W];
    gobs[..cobs.len()].copy_from_slice(&cobs);   // obs after the import are not computed yet: the first GPU actions come from the CPU obs
    if std::env::var("ONE").is_ok() {
        // one step from the transplanted state with the CPU policy's actions on both: barrier touches and outcomes
        for _rep in 0..std::env::var("ONE").unwrap().parse::<usize>().unwrap_or(1) {
        let vs = &cpu.sim.vs; let rows = rows_for(vs.high, &|i| vs.state[i] == 1, &cobs);
        let pre: Vec<(i32, f32, f32)> = (0..vs.high).map(|i| (vs.state[i] as i32, vs.lat[i], vs.speed[i])).collect();
        let mut ga = vec![0f32; cap * 3]; ga[..rows.len()].copy_from_slice(&rows); for i in rows.len() / 3..cap { ga[3 * i + 2] = 0.3; }
        gpu.import_snapshot(&snapshot(&cpu.sim)).unwrap();
        gpu.set_actions(&ga, 0); gpu.run_steps(1);
        let cr = cstep(&mut cpu, &mut enc, &mut cobs); let gr = gpu.take_records();
        let st = gpu.read_state(); let vs = &cpu.sim.vs;
        let (mut both, mut conly, mut gonly) = (0, 0, 0);
        for i in 0..vs.high.min(pre.len()) {
            if pre[i].0 != 1 { continue; }
            let ct = cpu.sim.touching_barrier(i); let gt = st.u(i, 10) & 2048 != 0;
            let cev: Vec<String> = (0..cr.len()).filter(|&k| cr.slot[k] as usize == i && cr.other[k] < 0 && (cr.kind[k] == 2 || cr.kind[k] == 3)).map(|k| format!("{:.2}", cr.dv_kmh[k])).collect();
            let gev: Vec<String> = (0..gr.len()).filter(|&k| gr.slot[k] as usize == i && gr.other[k] < 0 && (gr.kind[k] == 2 || gr.kind[k] == 3)).map(|k| format!("{:.2}", gr.dv_kmh[k])).collect();
            if ct && gt { both += 1; } else if ct { conly += 1; } else if gt { gonly += 1; }
            if ct || gt { println!("  slot {i} edge {} s {:.2} lat {:.3}->cpu {:.3} gpu {:.3} | u {:.3}->cpu {:.3} gpu {:.3} | vlat cpu {:+.3} gpu {:+.3} | touch cpu {} gpu {} | ev cpu {:?} gpu {:?}", vs.edge[i], vs.s[i], pre[i].1, vs.lat[i], st.f(i, 4), pre[i].2, vs.speed[i], st.f(i, 2), vs.v_lat[i], st.f(i, 33), ct as u8, gt as u8, cev, gev); }
        }
        println!("barrier touches: both {both}, cpu only {conly}, gpu only {gonly}");
        }
        return;
    }
    for step in 0..k_steps {
        // GPU: actions from its own obs (state before the step)
        let st = gpu.read_state();
        let grows = rows_for(cap, &|i| st.u(i, 7) == 1, &gobs);
        gpu.set_actions(&grows, 0); gpu.run_steps(1);
        let gr = gpu.take_records();
        gpu.read_obs_into(&mut gobs);
        let st2 = gpu.read_state();
        let cr = cstep(&mut cpu, &mut enc, &mut cobs);
        if let Ok(ts) = std::env::var("TRACE_SLOT") { let q: usize = ts.parse().unwrap(); if step < 3600 {
            let ce = (0..cr.len()).filter(|&k| cr.slot[k] as usize == q && (cr.kind[k] == 2 || cr.kind[k] == 3)).map(|k| format!("{}:{:.1}/{}", cr.kind[k], cr.dv_kmh[k], cr.other[k])).collect::<Vec<_>>().join(",");
            let ge = (0..gr.len()).filter(|&k| gr.slot[k] as usize == q && (gr.kind[k] == 2 || gr.kind[k] == 3)).map(|k| format!("{}:{:.1}/{}", gr.kind[k], gr.dv_kmh[k], gr.other[k])).collect::<Vec<_>>().join(",");
            let vs = &cpu.sim.vs;
            let od: Vec<String> = if cobs.len() >= (q + 1) * W { (0..W).filter(|&k| (cobs[q * W + k] - gobs[q * W + k]).abs() > 1e-3).map(|k| format!("{k}:{:.3}/{:.3}", cobs[q * W + k], gobs[q * W + k])).collect() } else { vec![] };
            if !od.is_empty() {
                let mut jo = bayenv_sim::obs::JunctionObs { mouth_distance: 0.0, in_box: false, distance: 0.0, time_to_conflict: 0.0, stopped: false };
                bayenv_sim::obs::junction_obs(&cpu.sim, q, &mut jo);
                let jb = gpu.gpu.read("jobst", (q * 8) as u64, 8); let jg = f32::from_le_bytes(jb[0..4].try_into().unwrap()); let js = u32::from_le_bytes(jb[4..8].try_into().unwrap());
                println!("{step:4}   obs diff {} | cpu jn dist {:.2} ttc {:.2} stopped {} | gpu jobst {jg:.2} step {js} | cpu leader {} gap {:.2} gpu leader {} gap {:.2}", od.join(" "), jo.distance, jo.time_to_conflict, jo.stopped, vs.leader[q], vs.gap[q], st2.u(q, 20) as i32, st2.f(q, 21));
            }
            println!("{step:4} cpu s {:7.2} lat {:6.3} hr {:+.3} u {:5.2} vlat {:+.3} w{} e{} {ce:12} | gpu s {:7.2} lat {:6.3} hr {:+.3} u {:5.2} vlat {:+.3} w{} e{} {ge}", vs.s[q], vs.lat[q], vs.heading_rel[q], vs.speed[q], vs.v_lat[q], cpu.sim.touching_barrier(q) as u8, vs.edge[q], st2.f(q, 1), st2.f(q, 4), st2.f(q, 5), st2.f(q, 2), st2.f(q, 33), (st2.u(q, 10) & 2048 != 0) as u8, st2.u(q, 0));
        } }
        if std::env::var("DIVERGE").is_ok() {
            let vs = &cpu.sim.vs;
            for q in 0..h0 {
                if diverged[q] || vs.state[q] != 1 || st2.u(q, 7) != 1 || vs.spawn_time[q] != sp0[q] || st2.f(q, 18) != sp0[q] { continue; }
                let bad = vs.edge[q] as u32 != st2.u(q, 0) || (vs.speed[q] - st2.f(q, 2)).abs() > 0.05 || (vs.lat[q] - st2.f(q, 4)).abs() > 0.02;
                if bad {
                    diverged[q] = true;
                    println!("DIV step {step} slot {q} edge {} conn {} s {:.1} u cpu {:.2} gpu {:.2} lat cpu {:.3} gpu {:.3} | prev obs diff {}", vs.edge[q], vs.conn[q], vs.s[q], vs.speed[q], st2.f(q, 2), vs.lat[q], st2.f(q, 4), prev_od[q]);
                }
            }
            for q in 0..h0.min(cobs.len() / W) { prev_od[q] = (0..W).filter(|&k| (cobs[q * W + k] - gobs[q * W + k]).abs() > 1e-3).map(|k| format!("{k}:{:.3}/{:.3}", cobs[q * W + k], gobs[q * W + k])).collect::<Vec<_>>().join(" "); }
        }
        if step < 2 { continue; }
        for (b, r) in [(0usize, &cr), (1, &gr)] {
            for k in 0..r.len() {
                let kd = r.kind[k]; if kd != 2 && kd != 3 { continue; }
                let sl = r.slot[k] as usize;
                let orig = |q: usize| q < h0 && (if b == 0 { cpu.sim.vs.spawn_time[q] } else { st2.f(q, 18) }) == sp0[q];
                if !orig(sl) || (r.other[k] >= 0 && !orig(r.other[k] as usize)) { continue; }
                let key = if r.other[k] >= 0 { format!("{} pair", if kd == 2 { "CRASHED" } else { "CONTACT" }) } else {
                    let (e, c, s) = if b == 0 { (cpu.sim.vs.edge[sl], cpu.sim.vs.conn[sl], cpu.sim.vs.s[sl]) } else { (st2.u(sl, 0) as i32, st2.f(sl, 30) as i32 - 1, st2.f(sl, 1)) };
                    format!("{} barrier {} {}", if kd == 2 { "CRASHED" } else { "CONTACT" }, place(e, c, s), if r.dv_kmh[k] <= 3.0 { "dv<=3" } else { "dv>3" })
                };
                if r.other[k] < 0 { let (e, ss, la) = if b == 0 { (cpu.sim.vs.edge[sl], cpu.sim.vs.s[sl], cpu.sim.vs.lat[sl]) } else { (st2.u(sl, 0) as i32, st2.f(sl, 1), st2.f(sl, 4)) }; let x = per[b].entry(sl).or_insert((0, e, ss, la)); x.0 += 1; }
                *cnt[b].entry(format!("w{} {key}", step / 600)).or_default() += 1;
            }
        }
    }
    for b in 0..2 {
        let mut v: Vec<_> = per[b].iter().map(|(k, x)| (x.0, *k, x.1, x.2, x.3)).collect(); v.sort_by(|a, b| b.0.cmp(&a.0));
        println!("{} barrier rows: {} slots, top: {:?}", ["cpu", "gpu"][b], v.len(), &v[..v.len().min(12)]);
        for &(c, sl, ..) in v.iter().take(6) { let o = per[1 - b].get(&sl).map_or(0, |x| x.0); println!("   slot {sl}: {c} here, {o} on the other"); }
    }
    let keys: std::collections::BTreeSet<String> = cnt[0].keys().chain(cnt[1].keys()).cloned().collect();
    println!("t {minutes} min + {k_steps} steps, {} commuters: event rows (cpu / gpu)", n);
    for k in keys { println!("  {k:32} {:6} {:6}", cnt[0].get(&k).copied().unwrap_or(0), cnt[1].get(&k).copied().unwrap_or(0)); }
}
