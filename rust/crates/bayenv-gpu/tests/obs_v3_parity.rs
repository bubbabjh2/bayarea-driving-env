//! CPU input-version-3 rows (bayenv-sim obs_v3.rs V3Encoder) vs the GPU's (GpuOptions::obs = ObsMode::V3: sense.wgsl +
//! senseV3, perceive.wgsl, kernels/model_sense.wgsl), per input index. The same commuter population on both backends
//! (population.js, seed 42, from 07:30), every car driven open loop by the same per-commuter command (a slow steering
//! swing, light gas), so most cars stay in step for the first half minute; the rows of cars whose state and route agree
//! on both backends (|ds|, |dlat| < 2 mm, |dv| < 2 mm/s, ...), with no car out of step within 150 m, are compared at
//! steps 5, 45, 200, 600 (1,000 and 10,000 commuters, steer_delay 0.15 and -1). DECISIONS.md 42 has the numbers.
//! The CPU row is encoded as bayenv-py obs="v3" does (record_actions, step, encode every driving car).
//! Run with OBS_V3_TABLE=1 for the per-index table. Skips without the world data or a GPU.
use bayenv_gpu::{defines::ObsMode, sim::{GpuOptions, GpuPopulation}};
use bayenv_sim::obs::OBS3;
use std::{collections::HashMap, path::Path, sync::Arc};

const START: f64 = 7.5 * 3600.0;
const TOL: f32 = 1e-3;
/// Inputs exempt from the per-index check. None: with obs_after_step the GPU computes 62 / 65..69 (jyield.wgsl) after
/// integrate, with the rest of the row (they showed the state before the step until DECISIONS.md 44); the diagnostic
/// "GPU 65..69 = the CPU's of the state before the step" is printed only.
const GPU_STALE: [usize; 0] = [];
/// Rows of an index allowed over TOL (sim-state differences between the backends, DECISIONS.md 42): 1 %, at least 3
fn allowed(rows: usize) -> usize { (rows / 100).max(3) }
/// state agreement of a car in step (m, m/s)
const SYNC: f32 = 2e-3;

fn command(agent: i32, t: usize) -> [f32; 3] {
    let a = agent as f32;
    [0.12 * (0.7 * a + 0.04 * t as f32).sin(), 0.25 + 0.1 * (0.3 * a).sin().abs(), 0.0]
}

/// obs v2 inputs 65..69 of slot i now (obs_v2.rs encode_core)
fn junction_inputs(sim: &bayenv_sim::sim::Sim, i: usize) -> [f32; 5] {
    let mut jn = bayenv_sim::obs::JunctionObs::default();
    bayenv_sim::obs::junction_obs(sim, i, &mut jn);
    let jd = jn.distance < f64::INFINITY;
    let c = |x: f64| x.max(0.0).min(1.0) as f32;
    [if jd { c(jn.distance / 60.0) } else { 1.0 }, if jd && jn.time_to_conflict < f64::INFINITY { c(jn.time_to_conflict / 10.0) } else { 1.0 },
     if jd && jn.stopped { 1.0 } else { 0.0 }, if jn.in_box { 1.0 } else { 0.0 }, if jn.mouth_distance < f64::INFINITY { c(jn.mouth_distance / 60.0) } else { 1.0 }]
}

#[derive(Default)]
struct Diff { rows: usize, pre_n: usize, pre_ok: usize, clean_rows: usize, max: Vec<f32>, over: Vec<usize>, worst: Vec<(usize, f32, f32)> }

/// per check step: (compared rows, per-index max |gpu - cpu|, per-index rows over TOL)
fn run(w: &Arc<bayenv_world::World>, n: usize, steer_delay: f64, checks: &[usize]) -> Option<Vec<(usize, Diff)>> {
    let data = Path::new("../../../data");
    let cap = (n + 1023) / 1024 * 1024;
    let mut o = GpuOptions { capacity: cap, start_time: START, seed: 42, obs: ObsMode::V3, refuel_budget_ms: f64::INFINITY, ..Default::default() };
    o.steer_delay = steer_delay;
    // diagnostics: OBS_V3_BASE=1 = both backends without the v3 junction fixes (the rows before DECISIONS.md 46)
    let base = std::env::var("OBS_V3_BASE").is_ok();
    if base { o.guide_in_box = false; o.junction_obstacle = false; }
    assert!(o.obs_after_step);
    let mut gpu = match bayenv_gpu::env::create(w.clone(), o) { Ok(s) => s, Err(e) => { eprintln!("obs_v3_parity: no GPU ({e}): not run"); return None } };
    gpu.set_population(Arc::new(GpuPopulation::generate(w, n, 42.0, true).unwrap()), true);
    let sw = Arc::new(bayenv_sim::world::World::from_bayenv(w));
    let pop = Arc::new(bayenv_sim::population::Population::generate_like_env(&sw, "real", n, 42.0, Some(data)).unwrap());
    let mut opts = bayenv_sim::sim::SimOpts::default(); opts.dt = 0.05; opts.seed = 42.0; opts.steer_delay = steer_delay;
    // diagnostics: OBS_V3_CPU_JS=1 = the CPU without the v3 junction fixes (the GPU keeps them), to see they are exercised
    if base || std::env::var("OBS_V3_CPU_JS").is_ok() { opts.v3_guide_in_box = false; opts.v3_junction_obstacle = false; }
    let cfg = bayenv_sim::env::EnvConfig { opts, start_time: Some(START), static_obstacles: Vec::new(), scenario_obstacles: Vec::new(), obstacles: None };
    let mut cpu = bayenv_sim::env::Env::new(sw.clone(), pop, cfg, Box::new(|w| bayenv_sim::harness::router_for(&w)));
    let mut enc = bayenv_sim::obs_v3::V3Encoder::new(&sw);
    let mut cobs: Vec<f32> = Vec::new(); let mut gobs = vec![0f32; cap * OBS3]; let mut acts = vec![0f32; cap * 3];
    let last = *checks.iter().max().unwrap();
    let mut out = Vec::new();
    for t in 0..=last {
        // CPU (bayenv-py CpuBackend obs="v3"): act, record, step, encode every driving car
        let nh = cpu.sim.vs.high;
        let rows: Vec<f32> = (0..nh).flat_map(|i| command(cpu.sim.vs.commuter[i], t)).collect();
        cpu.act_rows(&rows, 0);
        enc.record_actions(&cpu.sim);
        // the junction inputs 65..69 of the state before this step (diagnostic: the GPU's jyield writes them before
        // act / integrate)
        let pre_j: HashMap<usize, [f32; 5]> = if checks.contains(&t) { (0..cpu.sim.vs.high).filter(|&i| cpu.sim.vs.state[i] == 1).map(|i| (i, junction_inputs(&cpu.sim, i))).collect() } else { HashMap::new() };
        cpu.sim.step();
        for a in cpu.sim.actions.iter_mut() { *a = None; }
        let _ = cpu.take_records();
        let slots = bayenv_sim::obs::driving_slots(&cpu.sim);
        cobs.clear(); cobs.resize(slots.len() * OBS3, 0.0);
        enc.encode_batch(&mut cpu.sim, &slots, &mut cobs);
        // GPU: the same command per commuter; free slots get the CPU's default action (a car spawning in the step)
        let st = gpu.read_state();
        for i in 0..cap {
            let a = if st.u(i, 7) == 1 && gpu.slot_trip[i] >= 0 { command(gpu.slot_trip[i] / 2, t) } else { [0.0, 0.0, 0.3] };
            acts[3 * i..3 * i + 3].copy_from_slice(&a);
        }
        gpu.set_actions(&acts, 0); gpu.run_steps(1);
        let _ = gpu.take_records();
        if !checks.contains(&t) { continue; }
        gpu.read_obs_into(&mut gobs);
        let st = gpu.read_state();
        let routes = gpu.read_words("routes");
        let has_j = w.junctions.is_some();
        let live = |i: usize| st.u(i, 7) == 1 || st.u(i, 7) == 2;
        let by_trip: HashMap<i32, usize> = (0..cap).filter(|&i| live(i) && gpu.slot_trip[i] >= 0).map(|i| (gpu.slot_trip[i], i)).collect();
        let mut d = Diff { max: vec![0.0; OBS3], over: vec![0; OBS3], worst: vec![(0, 0.0, 0.0); OBS3], ..Default::default() };
        let vs = &cpu.sim.vs;
        // a car is in step when both backends have it in the same state; the GPU twin of CPU slot i
        let twin = |i: usize| -> Option<usize> {
            let &j = by_trip.get(&(vs.commuter[i] * 2 + vs.leg[i] as i32))?;
            let same = st.u(j, 7) as u8 == vs.state[i] && st.u(j, 0) as i32 == vs.edge[i] && (st.u(j, 14) & 255) as i8 == vs.lane[i]
                && (st.f(j, 1) - vs.s[i]).abs() < SYNC && (st.f(j, 4) - vs.lat[i]).abs() < SYNC && (st.f(j, 2) - vs.speed[i]).abs() < SYNC
                && (st.f(j, 5) - vs.heading_rel[i]).abs() < 1e-4 && st.u(j, 24) as u8 == vs.refuel_state[i]
                && st.u(j, 8) as i32 == vs.route_pos[i] && (if has_j { st.f(j, 30) as i32 - 1 } else { -1 }) == vs.conn[i] && {
                    let (off, len) = (st.par[j * 20] as usize, st.par[j * 20 + 1] as usize);
                    len == vs.routes[i].len() && routes[off..off + len].iter().zip(&vs.routes[i]).all(|(&w, &e)| (w & 0xFFFFF) as i32 == e) };
            if same { Some(j) } else { None }
        };
        // cars out of step (either backend), by position: rows within NEAR m of one are not compared
        const NEAR: f64 = 150.0;
        let mut bad: Vec<(f64, f64)> = Vec::new();
        let mut matched = std::collections::HashSet::new();
        for i in 0..vs.high {
            if vs.state[i] != 1 && vs.state[i] != 2 { continue; }
            match twin(i) { Some(j) => { matched.insert(j); } None => bad.push((vs.x[i] as f64, vs.z[i] as f64)) }
        }
        for j in 0..cap { if live(j) && !matched.contains(&j) { bad.push((st.pose[4 * j] as f64, st.pose[4 * j + 2] as f64)); } }
        let mut grid: HashMap<(i64, i64), Vec<(f64, f64)>> = HashMap::new();
        for &(x, z) in &bad { grid.entry(((x / NEAR).floor() as i64, (z / NEAR).floor() as i64)).or_default().push((x, z)); }
        let clean = |x: f64, z: f64| { let (cx, cz) = ((x / NEAR).floor() as i64, (z / NEAR).floor() as i64);
            !(-1..=1).any(|a| (-1..=1).any(|b| grid.get(&(cx + a, cz + b)).map_or(false, |v| v.iter().any(|p| (p.0 - x).hypot(p.1 - z) < NEAR)))) };
        let (mut out_of_step, mut near_bad, mut pre_n, mut pre_ok) = (0, 0, 0, 0);
        for (r, &i) in slots.iter().enumerate() {
            let Some(j) = twin(i) else { out_of_step += 1; continue };
            if !clean(vs.x[i] as f64, vs.z[i] as f64) { near_bad += 1; continue; }
            d.rows += 1;
            let (c, g) = (&cobs[r * OBS3..(r + 1) * OBS3], &gobs[j * OBS3..(j + 1) * OBS3]);
            if let Ok(ks) = std::env::var("OBS_V3_DEBUG") {
                for k in ks.split(',').map(|x| x.parse::<usize>().unwrap()) {
                    if (c[k] - g[k]).abs() > TOL {
                        let e = vs.edge[i] as usize;
                        let diffs: Vec<String> = (0..102).filter(|&q| (c[q] - g[q]).abs() > TOL).map(|q| format!("{q}: {} / {}", c[q], g[q])).collect();
                        eprintln!("DBG k {k} slot {i}/{j}: edge {} s {} lat {} v {} conn {} rpos {}/{} leader {} {} gpu {} {} | {diffs:?}", vs.edge[i], vs.s[i], vs.lat[i], vs.speed[i], vs.conn[i], vs.route_pos[i], vs.routes[i].len(), vs.leader[i], vs.gap[i], st.u(j, 20) as i32, st.f(j, 21));
                    }
                }
            }
            if let Some(p) = pre_j.get(&i) { pre_n += 1; if (65..70).all(|k| (p[k - 65] - g[k]).abs() <= TOL) { pre_ok += 1; } }
            if (0..OBS3).all(|k| GPU_STALE.contains(&k) || (c[k] - g[k]).abs() <= TOL) { d.clean_rows += 1; }
            for k in 0..OBS3 {
                let e = (c[k] - g[k]).abs();
                if !(e <= TOL) { d.over[k] += 1; }
                if e > d.max[k] || (e.is_nan() && !d.max[k].is_nan()) { d.max[k] = e; d.worst[k] = (i, c[k], g[k]); }
            }
        }
        let unsynced = format!("out of step {out_of_step}, within {NEAR} m of a car out of step {near_bad}; GPU 65..69 = the CPU's of the state before the step: {pre_ok} / {pre_n}");
        eprintln!("n {n} steer_delay {steer_delay} step {t}: cpu driving {}, compared {} (not in step {unsynced})", slots.len(), d.rows);
        d.pre_n = pre_n; d.pre_ok = pre_ok;
        out.push((t, d));
    }
    Some(out)
}

fn report(n: usize, sd: f64, res: &[(usize, Diff)]) -> Vec<String> {
    let table = std::env::var("OBS_V3_TABLE").is_ok();
    let mut bad = Vec::new();
    for (t, d) in res {
        let worst = d.max.iter().cloned().fold(0.0f32, f32::max);
        let over: Vec<usize> = (0..OBS3).filter(|&k| d.over[k] > 0).collect();
        eprintln!("n {n} sd {sd} step {t}: {} rows, max |diff| {worst:.2e}, indices over {TOL}: {:?}", d.rows,
            over.iter().map(|&k| format!("{k}({}/{})", d.over[k], d.rows)).collect::<Vec<_>>());
        if table {
            for k in 0..OBS3 { if d.max[k] > 0.0 { let (i, c, g) = d.worst[k]; eprintln!("   [{k:3}] max {:.3e} over {:4}  (slot {i}: cpu {c} gpu {g})", d.max[k], d.over[k]); } }
        }
        let block = |a: usize, b: usize| (a..b).map(|k| d.max[k]).fold(0.0f32, f32::max);
        eprintln!("   max |diff| by block: 0..69 {:.1e} (without 62, 65..69: {:.1e}) | 70..78 {:.1e} | 79..86 {:.1e} | 87..93 {:.1e} | 94..101 {:.1e} | bodies {:.1e}; rows within {TOL} everywhere but 62, 65..69: {} / {}; GPU 65..69 = CPU before the step: {} / {}",
            block(0, 70), (0..70).filter(|k| !GPU_STALE.contains(k)).map(|k| d.max[k]).fold(0.0f32, f32::max), block(70, 79), block(79, 87), block(87, 94), block(94, 102), block(102, OBS3), d.clean_rows, d.rows, d.pre_ok, d.pre_n);
        for &k in &over { if !GPU_STALE.contains(&k) && d.over[k] > allowed(d.rows) { bad.push(format!("n {n} sd {sd} step {t} index {k}: {} / {} rows, max {:.3e}", d.over[k], d.rows, d.max[k])); } }
        if d.clean_rows * 100 < d.rows * 97 { bad.push(format!("n {n} sd {sd} step {t}: only {} / {} rows within {TOL}", d.clean_rows, d.rows)); }
    }
    bad
}

#[test]
fn cpu_v3_rows_match_gpu() {
    let w = match bayenv_world::World::load(Path::new("../../../data")) { Ok(w) => Arc::new(w), Err(e) => { bayenv_world::oracle::skip("obs_v3_parity", &format!("world: {e}")); return } };
    let mut bad = Vec::new();
    for n in [1000, 10000] {
        for sd in [0.15, -1.0] {
            if let Ok(only) = std::env::var("OBS_V3_ONLY") { if only != format!("{n},{sd}") { continue; } }
            let Some(res) = run(&w, n, sd, &[5, 45, 200, 600]) else { return };
            for (t, d) in &res { if *t >= 45 { assert!(d.rows >= if n >= 10000 { 300 } else { 30 }, "too few cars compared: {}", d.rows); } }
            bad.extend(report(n, sd, &res));
        }
    }
    assert!(bad.is_empty(), "{bad:#?}");
}
