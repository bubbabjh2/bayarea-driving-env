//! GPU check of `lane_drop_hop` (kernels/model_integrate.wgsl MF_LANE_DROP_HOP, DECISIONS.md 39), mirroring bayenv-sim
//! physics_fixes.rs `lane_drop_hop_takes_the_edge_the_lane_lands_nearest`: a motorway diverge where a lane continues
//! onto no out-edge (6 lanes -> a 4-lane mainline + a 1-lane ramp). One car in the dropped far-left lane, routed onto the
//! ramp, at 25 m/s with steer 0. Flag off (the JS): it takes the ramp with the strict lat and lands ~16.6 m past the
//! ramp's barrier line. Flag on: it stays on the mainline (EV_MISSED), inside the barrier lines. The same placement on
//! the CPU sim must agree. The car is one of a few scattered cars, its route and state written into the GPU buffers.
//! Skips without the world data or a GPU.
use bayenv_gpu::{defines::ObsMode, layout as L, sim::GpuOptions};
use bayenv_sim::world::World as SimWorld;
use std::{path::Path, sync::Arc};

const N: usize = 200;
const U0: f32 = 25.0;
const BEFORE: f64 = 60.0; // m before the diverge node

fn world() -> Option<Arc<bayenv_world::World>> {
    match bayenv_world::World::load(Path::new("../../../data")) { Ok(w) => Some(Arc::new(w)), Err(e) => { bayenv_world::oracle::skip("lane_drop", &format!("world: {e}")); None } }
}

/// Every (edge, lane, ramp) whose lane maps onto no strict out-edge (the first is the CPU test's case); the ramp is
/// the out-edge the lane lands furthest from.
fn find_cases(g: &SimWorld) -> Vec<(usize, i32, usize)> {
    let mut out = Vec::new();
    for e in 0..g.edge_count {
        if !g.is_shoulder_road(e) || g.e_length[e] < 200.0 || (g.j.is_some() && g.e_s_stop[e] < g.e_length[e]) { continue; }
        let to = g.e_to[e] as usize;
        let outs: Vec<usize> = (g.out_start[to]..g.out_start[to + 1]).map(|q| g.out_edges[q as usize] as usize).filter(|&o| o as i32 != g.e_rev[e] && g.hop(e, o).strict).collect();
        if outs.len() < 2 { continue; }
        for k in 0..g.e_lanes[e] {
            if outs.iter().all(|&o| { let l = g.map_lane(e, o, k); l < 0 || l >= g.e_lanes[o] }) {
                let far = *outs.iter().max_by_key(|&&o| { let l = g.map_lane(e, o, k); if l < 0 { -l } else { l - g.e_lanes[o] + 1 } }).unwrap();
                out.push((e, k, far));
            }
        }
    }
    out
}

/// How far a sedan's footprint at `lat` (heading 0) is past edge `e`'s barrier lines at s (> 0: in the barrier).
fn depth(g: &SimWorld, e: usize, s: f64, lat: f64) -> f64 {
    let o = bayenv_sim::sim::SimOpts::default();
    let hw = 0.5 * bayenv_sim::physics::VEHICLE_PARAMS[0].width;
    let w = g.lane_w(e, s); let n = g.e_lanes[e] as f64;
    let (pr, pl) = match &g.lg { Some(lg) => { let b = lg.bounds(e, s); (b.r - b.right, b.r - b.left) } None => (0.0, n * w) };
    let right = pr - if g.is_shoulder_road(e) { o.freeway_shoulder } else { o.shoulder };
    let left = pl + if g.e_rev[e] >= 0 { (if g.e_opp[e] != 0 { g.e_opp[e] as f64 } else { n }) * w + o.opposite_margin } else { o.shoulder };
    (right - (lat - hw)).max(lat + hw - left)
}

/// Result of one run: (edge after the hop, missed-exit events of the car, depth past the barrier lines on the first
/// step after the hop, barrier / crash event rows of the car)
type Out = (usize, usize, f64, usize);

fn run_gpu(w: &Arc<bayenv_world::World>, g: &SimWorld, case: (usize, i32, usize), on: bool) -> Option<Out> {
    let (e, lane, ramp) = case;
    let o = GpuOptions { capacity: 1024, obs: ObsMode::None, lane_drop_hop: on, ..Default::default() };
    let mut s = match bayenv_gpu::env::create(w.clone(), o) { Ok(s) => s, Err(err) => { eprintln!("lane_drop: no GPU ({err}): not run"); return None } };
    bayenv_gpu::env::scatter(&mut s, N, 5);
    let st = s.read_state();
    // the car: a driving slot with a route buffer of at least 2 words; everyone else brakes
    let i = (0..N).find(|&i| st.u(i, 7) == 1 && st.par[i * L::PAR_WORDS + 1] >= 2).expect("a driving car");
    let mut a: Vec<f32> = (0..s.cap).flat_map(|_| [0.0, 0.0, 1.0]).collect();
    a[3 * i..3 * i + 3].copy_from_slice(&[0.0, 0.3, 0.0]);
    s.set_actions(&a, 0);
    s.run_steps(30); // the command reaches the end of the control ring
    let _ = s.take_records();
    // route [e, ramp] and the car's state at BEFORE m before the node, lane `lane`, heading 0
    let packer = bayenv_gpu::pack::RoutePacker::new(&w.graph);
    let words = packer.pack_route(&w.graph, &[e as i32, ramp as i32]);
    let st = s.read_state();
    let off = st.par[i * L::PAR_WORDS] as u64;
    s.gpu.write("routes", bytemuck::cast_slice(&words[..2]), off * 4);
    s.gpu.write("par", bytemuck::cast_slice(&[2u32]), ((i * L::PAR_WORDS + 1) * 4) as u64);
    let len = g.e_length[e];
    let s0 = (len - BEFORE) as f32;
    let lat0 = ((lane as f64 + 0.5) * g.lane_w(e, s0 as f64)) as f32;
    let mut v = st.veh[i * L::VEH_WORDS..(i + 1) * L::VEH_WORDS].to_vec();
    v[0] = e as u32; v[1] = s0.to_bits(); v[2] = U0.to_bits(); v[4] = lat0.to_bits(); v[5] = 0f32.to_bits(); v[6] = 0f32.to_bits();
    v[8] = 0; v[9] = 0; v[25] = 0f32.to_bits(); v[30] = 0f32.to_bits(); v[33] = 0f32.to_bits(); v[34] = 0f32.to_bits(); v[35] = 0f32.to_bits();
    s.gpu.write("veh", bytemuck::cast_slice(&v), (i * L::VEH_WORDS * 4) as u64);
    let (mut missed, mut contacts, mut first): (usize, usize, Option<(usize, f64)>) = (0, 0, None);
    for _ in 0..80 {
        s.run_steps(1);
        let log = s.take_records();
        for k in 0..log.len() { if log.slot[k] as usize == i { match log.kind[k] { 7 => missed += 1, 2 | 3 => contacts += 1, _ => {} } } }
        let st = s.read_state();
        let ed = st.u(i, 0) as usize;
        if std::env::var("LANE_DROP_DEBUG").is_ok() { eprintln!("gpu e{ed} s {:.2} lat {:.2} hr {:.3} u {:.2} state {}", st.f(i, 1), st.f(i, 4), st.f(i, 5), st.f(i, 2), st.u(i, 7)); }
        if ed != e && first.is_none() {
            // the barrier acted in this step already when the car was past it (pushed back by at most its correction):
            // the depth before the push is at least the depth after it
            first = Some((ed, depth(g, ed, st.f(i, 1) as f64, st.f(i, 4) as f64)));
        }
        if st.u(i, 7) != 1 || (ed != e && st.f(i, 1) > 20.0) { break; }
    }
    let (ed, d) = first.expect("the car hopped");
    Some((ed, missed, d, contacts))
}

fn run_cpu(g: &Arc<SimWorld>, case: (usize, i32, usize), on: bool) -> Out {
    use bayenv_sim::{sim::*, vehicle::*};
    let (e, lane, ramp) = case;
    let opts = SimOpts { dt: 0.05, start_time: 7.0 * 3600.0, fuel: false, lane_drop_hop: on, ..SimOpts::default() };
    let mut sim = Sim::new(g.clone(), Arc::new(bayenv_sim::population::Population::default()), opts, bayenv_sim::harness::router_for(g));
    let (time, dt, step) = (sim.time, sim.dt, sim.step_count);
    let route = [e as i32, ramp as i32];
    let i = sim.vs.alloc(g, AllocArgs { type_idx: 0, traits: DEFAULT_TRAITS, route: &route, commuter: -1, leg: 0, time, learner: false, dt, step, lane });
    sim.vs.s[i] = (g.e_length[e] - BEFORE) as f32; sim.vs.speed[i] = U0; sim.vs.set_reaction_time(i, 0.05, dt);
    sim.vs.update_lanes(i, g, e); sim.vs.update_pose(i, g); sim.vs.save_pose(i);
    if sim.drivers.len() <= i { sim.drivers.resize(i + 1, Driver::None); }
    if sim.actions.len() <= i { sim.actions.resize(i + 1, None); }
    sim.drivers[i] = Driver::External;
    sim.barrier_dbg = Some(Vec::new());
    let (mut contacts, mut first): (usize, Option<(usize, f64)>) = (0, None);
    let missed0 = sim.counters.missed_exits;
    for _ in 0..80 {
        if sim.vs.state[i] != STATE_DRIVING { break; }
        sim.actions[i] = Some(Action { steer: Some(0.0), steer_rate: None, gas: 0.3, brake: 0.0 });
        sim.step();
        let dbg: Vec<_> = sim.barrier_dbg.as_mut().unwrap().drain(..).collect();
        contacts += sim.event_log.drain(..).filter(|ev| ev.slot as usize == i && (ev.kind == EV_CRASHED || ev.kind == EV_CONTACT)).count();
        let ed = sim.vs.edge[i] as usize;
        if ed != e && first.is_none() {
            // the barrier's pre-push depth when it acted this step, else the depth now
            let d = dbg.iter().find(|b| b.slot == i).map(|b| b.depth).unwrap_or_else(|| depth(g, ed, sim.vs.s[i] as f64, sim.vs.lat[i] as f64));
            first = Some((ed, d));
        }
        if ed != e && sim.vs.s[i] > 20.0 { break; }
    }
    let (ed, d) = first.expect("the car hopped");
    (ed, (sim.counters.missed_exits - missed0) as usize, d, contacts)
}

#[test]
fn gpu_lane_drop_hop_matches_cpu() {
    let Some(w) = world() else { return };
    let g = Arc::new(SimWorld::from_bayenv(&w));
    // the first case, and the first 6-lane one (6 -> 4 + 1, the ~16.6 m landings of the map-wide scan)
    let all = find_cases(&g);
    let mut cases = vec![*all.first().expect("a lane dropped at a diverge")];
    if let Some(c) = all.iter().find(|c| g.e_lanes[c.0] == 6) { cases.push(*c); }
    for case in cases {
        let (e, lane, ramp) = case;
        let Some(gon) = run_gpu(&w, &g, case, true) else { return };
        let goff = run_gpu(&w, &g, case, false).unwrap();
        let (con, coff) = (run_cpu(&g, case, true), run_cpu(&g, case, false));
        eprintln!("e{e} ({} lanes) lane {lane} -> planned ramp e{ramp}  (edge after the hop, missed exits, depth past the barrier lines, contact rows)", g.e_lanes[e]);
        eprintln!("  flag on : gpu {gon:?} cpu {con:?}");
        eprintln!("  flag off: gpu {goff:?} cpu {coff:?}");
        // flag off (the JS): the ramp, 12-17 m past its barrier line
        assert_eq!(goff.0, ramp, "gpu, off: the planned ramp");
        assert!(goff.2 > 12.0 && goff.2 < 17.0, "gpu, off: deep in the ramp's barrier: {}", goff.2);
        // flag on: the edge the lane physically leads to, a missed exit, no barrier contact
        assert_ne!(gon.0, ramp, "gpu, on: not the ramp");
        assert!(gon.1 >= 1, "gpu, on: a missed exit is reported");
        assert!(gon.2 <= 0.0 && gon.3 == 0, "gpu, on: no barrier contact: depth {} rows {}", gon.2, gon.3);
        // the CPU on the same placement
        assert_eq!((gon.0, goff.0), (con.0, coff.0), "same edges on both backends");
        assert!(con.1 >= 1 && con.2 <= 0.0 && con.3 == 0, "cpu, on: {con:?}");
        assert!((goff.2 - coff.2).abs() < 0.5, "same depth off: gpu {} cpu {}", goff.2, coff.2);
    }
}
