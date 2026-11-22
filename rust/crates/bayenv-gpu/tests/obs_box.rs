//! obs v2 inside junction boxes, CPU (bayenv-sim V2Encoder) vs GPU (sense.wgsl with GpuOptions::guide_in_box): the same
//! population (1,000 commuters, seed 777, 07:00) on both backends, every car braking; after 60 s, a few driving trips are
//! moved 15 m before a turning junction on their own route and crawl into the box (s past the in-edge's end), then
//! brake to a standstill there. All 70 inputs must agree within 1e-3 (the JS GPU guidance reported the
//! junction being crossed with a negative distance and its side: inputs 20..32 differed). Skips without data / a GPU.
use bayenv_gpu::{defines::ObsMode, layout as L, sim::{GpuOptions, GpuPopulation}};
use std::{path::Path, sync::Arc};

const N: usize = 1000;
const W: usize = 70;

#[test]
fn obs_v2_in_junction_boxes_matches_cpu() {
    let data = "../../../data".to_string();
    let w = match bayenv_world::World::load(Path::new(&data)) { Ok(w) => Arc::new(w), Err(e) => { bayenv_world::oracle::skip("obs_box", &format!("world: {e}")); return } };
    let start = 7.0 * 3600.0; let dt = 0.05;
    let cap = 1024;
    let mut o = GpuOptions { capacity: cap, start_time: start, seed: 777, obs: ObsMode::V2, refuel_budget_ms: f64::INFINITY, dt, ..Default::default() };
    o.features.accidents = false;
    let mut gpu = match bayenv_gpu::env::create(w.clone(), o) { Ok(s) => s, Err(e) => { eprintln!("obs_box: no GPU ({e}): not run"); return } };
    gpu.set_population(Arc::new(GpuPopulation::generate(&w, N, 777.0, true).unwrap()), true);
    let sw = Arc::new(bayenv_sim::world::World::from_bayenv(&w));
    let pop = Arc::new(bayenv_sim::population::Population::generate_like_env(&sw, "real", N, 777.0, Some(Path::new(&data))).unwrap());
    let mut opts = bayenv_sim::sim::SimOpts::default(); opts.dt = dt; opts.seed = 777.0;
    let cfg = bayenv_sim::env::EnvConfig { opts, start_time: Some(start), static_obstacles: Vec::new(), scenario_obstacles: Vec::new(), obstacles: None };
    let mut cpu = bayenv_sim::env::Env::new(sw.clone(), pop, cfg, Box::new(|w| bayenv_sim::harness::router_for(&w)));
    let mut enc = bayenv_sim::obs_v2::V2Encoder::new(&sw);
    let mut gacts = vec![0f32; cap * 3]; for i in 0..cap { gacts[3 * i + 2] = 0.5; }
    let mut cobs = vec![0f32; W]; let mut gobs = vec![0f32; cap * W];
    let mut t = 0usize;
    let mut cstep = |cpu: &mut bayenv_sim::env::Env, enc: &mut bayenv_sim::obs_v2::V2Encoder, special: Option<(usize, [f32; 3])>| {
        let n = cpu.sim.vs.high;
        let rows: Vec<f32> = (0..n).flat_map(|i| match special { Some((k, a)) if k == i => a, _ => [0.0, 0.0, 0.5] }).collect();
        cpu.act_rows(&rows, 0); enc.record_actions(&cpu.sim); cpu.sim.step(); let _ = cpu.take_records();
        for x in cpu.sim.actions.iter_mut() { *x = None; }
    };
    // 60 s of braking traffic, then the cases: driving trips with a turn (type 3 / 4) on their route ahead
    for _ in 0..1200 { cstep(&mut cpu, &mut enc, None); gpu.set_actions(&gacts, 0); gpu.run_steps(1); t += 1; }
    let st = gpu.read_state(); let routes = gpu.read_words("routes");
    let mut cases = Vec::new();
    for gi in 0..cap {
        if cases.len() >= 8 { break; }
        if st.u(gi, 7) != 1 || gpu.slot_trip[gi] < 0 { continue; }
        let (off, len, pos) = (st.par[gi * L::PAR_WORDS] as usize, st.par[gi * L::PAR_WORDS + 1] as usize, st.u(gi, 8) as usize);
        for k in pos + 1..len.saturating_sub(2) {
            let c = (routes[off + k] >> 20) & 2047; let ty = (c >> 1) & 15; let e = (routes[off + k] & 0xFFFFF) as usize;
            if c & 1 != 0 && (ty == 3 || ty == 4) && w.graph.length[e] >= 40.0 && w.graph.lanes[e] == 1 { cases.push((gpu.slot_trip[gi] as i64, e, w.graph.length[e] as f32 + 1.5, 0u32, 0.0f32, 1.0f32)); break; }
        }
    }
    assert!(cases.len() >= 3, "cases {}", cases.len());
    let mut fails = Vec::new(); let mut compared = 0;
    for (trip, e, s, lane, gas, brake) in cases {
        // the trip's car on both backends
        let ci = (0..cpu.sim.vs.high).find(|&i| cpu.sim.vs.state[i] == bayenv_sim::vehicle::STATE_DRIVING && 2 * cpu.sim.vs.commuter[i] as i64 + cpu.sim.vs.leg[i] as i64 == trip);
        let gi = (0..cap).find(|&i| gpu.slot_trip[i] as i64 == trip && gpu.read_state().u(i, 7) == 1);
        let (Some(ci), Some(gi)) = (ci, gi) else { eprintln!("trip {trip}: not driving on both (cpu {ci:?} gpu {gi:?})"); continue };
        let route = cpu.sim.vs.routes[ci].clone();
        let Some(rp) = route.iter().position(|&x| x as usize == e) else { continue };
        // the trace's s is in the in-edge frame (past the edge length inside a junction box): start 15 m before it on
        // the edge, drive in at a crawl until s reaches it, then hold the trace's pedals
        let s_tgt = s; let s = (s_tgt.min(w.graph.length[e] as f32 - 1.0) - 15.0).max(1.0);
        let lw = w.lanes.lane_width(e, s as f64);
        let rdist: f64 = route[..rp].iter().map(|&q| w.graph.length[q as usize]).sum::<f64>() + s as f64;   // route distance at the spot
        // CPU: move it
        { let vs = &mut cpu.sim.vs; vs.edge[ci] = e as i32; vs.s[ci] = s; vs.lat[ci] = ((lane as f64 + 0.5) * lw) as f32; vs.route_pos[ci] = rp as i32; vs.conn[ci] = -1; vs.route_dist[ci] = rdist as f32;
          vs.speed[ci] = 0.0; vs.v_lat[ci] = 0.0; vs.yaw_rate[ci] = 0.0; vs.heading_rel[ci] = 0.0; vs.steer_angle[ci] = 0.0; vs.taper[ci] = 0.0;
          vs.update_lanes(ci, &sw, e); vs.update_pose(ci, &sw); vs.save_pose(ci); }
        // GPU: the same (veh words), its route position from the GPU's route words
        let st = gpu.read_state(); let routes = gpu.read_words("routes");
        let (off, len) = (st.par[gi * L::PAR_WORDS] as usize, st.par[gi * L::PAR_WORDS + 1] as usize);
        let groute: Vec<i32> = routes[off..off + len].iter().map(|&x| (x & 0xFFFFF) as i32).collect();
        let grp = groute.iter().position(|&x| x as usize == e).unwrap();
        let mut v = st.veh[gi * L::VEH_WORDS..(gi + 1) * L::VEH_WORDS].to_vec();
        v[0] = e as u32; v[1] = s.to_bits(); v[4] = (((lane as f64 + 0.5) * lw) as f32).to_bits(); v[8] = grp as u32; v[30] = 0f32.to_bits(); v[19] = (rdist as f32).to_bits();
        for k in [2usize, 5, 6, 25, 33, 34, 35] { v[k] = 0f32.to_bits(); }
        v[14] = L::pack_lanes(lane, lane, lane, -1);
        gpu.gpu.write("veh", bytemuck::cast_slice(&v), (gi * L::VEH_WORDS * 4) as u64);
        let mut ga = gacts.clone(); ga[3 * gi..3 * gi + 3].copy_from_slice(&[0.0, 0.12, 0.0]);
        let (mut cdone, mut gdone) = (false, false);
        for _ in 0..600 {
            if !cdone && (cpu.sim.vs.s[ci] >= s_tgt || cpu.sim.vs.edge[ci] as usize != e) { cdone = true; }
            let gs = gpu.read_state(); if !gdone && (gs.f(gi, 1) >= s_tgt || gs.u(gi, 0) as usize != e) { gdone = true; ga[3 * gi..3 * gi + 3].copy_from_slice(&[0.0, gas, brake]); }
            if cdone && gdone { break; }
            cstep(&mut cpu, &mut enc, Some((ci, if cdone { [0.0, gas, brake] } else { [0.0, 0.12, 0.0] }))); gpu.set_actions(&ga, 0); gpu.run_steps(1); t += 1;
        }
        ga[3 * gi..3 * gi + 3].copy_from_slice(&[0.0, gas, brake]);
        for _ in 0..100 { cstep(&mut cpu, &mut enc, Some((ci, [0.0, gas, brake]))); gpu.set_actions(&ga, 0); gpu.run_steps(1); t += 1; }
        enc.encode_v2(&mut cpu.sim, ci, &mut cobs);
        gpu.read_obs_into(&mut gobs);
        let g = &gobs[gi * W..(gi + 1) * W];
        let st = gpu.read_state();
        eprintln!("trip {trip} at t {:.1}: cpu edge {} conn {} s {:.2} u {:.3} route_pos {} | gpu edge {} conn {} s {:.2} u {:.3} route_pos {} (route same: {})", cpu.sim.time,
            cpu.sim.vs.edge[ci], cpu.sim.vs.conn[ci], cpu.sim.vs.s[ci], cpu.sim.vs.speed[ci], cpu.sim.vs.route_pos[ci], st.u(gi, 0), st.f(gi, 30) as i32 - 1, st.f(gi, 1), st.f(gi, 2), st.u(gi, 8), route == groute);
        let mut nd = 0;
        for k in 0..W { if (cobs[k] - g[k]).abs() > 1e-3 { nd += 1; eprintln!("   input {k:2}: cpu {:+.4} gpu {:+.4}", cobs[k], g[k]); } }
        if st.f(gi, 1) > w.graph.length[e] as f32 + 1.0 { compared += 1; }
        if nd > 0 { fails.push(format!("trip {trip}: {nd} inputs differ")); }
    }
    assert!(compared >= 2, "cars compared inside a box past the in-edge end: {compared}");
    assert!(fails.is_empty(), "{fails:?}");
}
