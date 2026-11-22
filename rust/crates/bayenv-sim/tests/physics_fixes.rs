//! The physics model fixes of rust/docs/physics-audit.md (SimOpts flags, default on; SimOpts::js() turns them off).
//! Each test runs the scenario with the fix and with the JS model, and checks the fix against the physics it restores.
//! Worlds: tests/common/gen_worlds.mjs ("road3": a 3 km 3-lane one-way motorway a -> b, then a 90-degree turn onto a
//! residential road b -> c; the cars are placed by hand on an empty population, as js_port_collisions.rs).
mod common;
use bayenv_sim::population::Population;
use bayenv_sim::sim::*;
use bayenv_sim::vehicle::*;
use bayenv_sim::world::World;
use std::sync::Arc;

const SEDAN: u8 = 0;
const TRUCK: u8 = 2;
const BUS: u8 = 3;

fn sim_with(g: &Arc<World>, opts: SimOpts) -> Sim { sim_dt(g, opts, 0.05) }
fn sim_dt(g: &Arc<World>, opts: SimOpts, dt: f64) -> Sim {
    let opts = SimOpts { dt, start_time: 7.0 * 3600.0, fuel: false, ..opts };
    Sim::new(g.clone(), Arc::new(Population::default()), opts, bayenv_sim::harness::router_for(g))
}
fn place(sim: &mut Sim, ty: u8, route: &[i32], s: f64, lane: i32, speed: f64) -> usize {
    let g = sim.g.clone();
    let (time, dt, step) = (sim.time, sim.dt, sim.step_count);
    let i = sim.vs.alloc(&g, AllocArgs { type_idx: ty, traits: DEFAULT_TRAITS, route, commuter: -1, leg: 0, time, learner: false, dt, step, lane });
    sim.vs.s[i] = s as f32; sim.vs.speed[i] = speed as f32; sim.vs.set_reaction_time(i, 0.05, dt);
    let e = route[0] as usize;
    sim.vs.update_lanes(i, &g, e); sim.vs.update_pose(i, &g); sim.vs.save_pose(i);
    if sim.drivers.len() <= i { sim.drivers.resize(i + 1, Driver::None); }
    if sim.actions.len() <= i { sim.actions.resize(i + 1, None); }
    sim.drivers[i] = Driver::External;
    i
}
/// a resting wreck (as a parked obstacle) of type `ty` on edge 0 at s, lane 0
fn wreck(sim: &mut Sim, ty: u8, s: f64) -> usize { sim.place_static_obstacles(&[(0, s, Some(0), None, ty)], true)[0] }
/// the defaults plus curvature_dynamics (default off until the lane geometry is smoothed, DECISIONS.md 23)
fn curv() -> SimOpts { SimOpts { curvature_dynamics: true, ..SimOpts::default() } }
/// curv() on the smoothed frame geometry (DECISIONS.md 33)
fn curv_smooth() -> SimOpts { SimOpts { smooth_geometry: true, ..curv() } }
fn mass(ty: u8) -> f64 { bayenv_sim::physics::VEHICLE_PARAMS[ty as usize].m }
/// world velocity along x (edge 0 runs along +x)
fn vx(sim: &Sim, i: usize) -> f64 { let h = sim.vs.heading[i] as f64; sim.vs.speed[i] as f64 * h.cos() + sim.vs.v_lat[i] as f64 * h.sin() }
/// this step's pedals / position steer for an externally driven car
fn hold(sim: &mut Sim, i: usize, steer: f64, gas: f64, brake: f64) { sim.actions[i] = Some(Action { steer: Some(steer), steer_rate: None, gas, brake }); }

/// Drive car `a` (holding gas `gas`) into the wreck; returns (a's x speed before, a's x speed after, wreck x speed after)
/// over the step of first contact.
fn rear_end(opts: SimOpts, ty_a: u8, ty_b: u8, kmh: f64) -> Option<(f64, f64, f64)> {
    let g = common::shared_world("road3")?;
    // a short step: the locked-wheel friction acting on the wrecks after the impulse within the step stays small
    let mut sim = sim_dt(&g, opts, 0.01);
    let a = place(&mut sim, ty_a, &[0], 100.0, 0, kmh / 3.6);
    let b = wreck(&mut sim, ty_b, 125.0);
    for _ in 0..1000 {
        hold(&mut sim, a, 0.0, 0.0, 0.0);
        let (va, vb) = (vx(&sim, a), vx(&sim, b));
        sim.step();
        if sim.collided_last.contains(&a) { let _ = vb; return Some((va, vx(&sim, a), vx(&sim, b))); }
    }
    panic!("no contact");
}

#[test]
fn wreck_real_mass_sedan_into_parked_sedan() {
    let Some((v0, va, vb)) = rear_end(SimOpts::default(), SEDAN, SEDAN, 50.0) else { return bayenv_world::oracle::skip("physics_fixes", "fixtures") };
    let (_, ja, jb) = rear_end(SimOpts::js(), SEDAN, SEDAN, 50.0).unwrap();
    let dv = (v0 - va) * 3.6;
    eprintln!("fix: dv {dv:.1} km/h, wreck now {:.1} km/h; js: dv {:.1} km/h, wreck {:.1} km/h", vb * 3.6, (v0 - ja) * 3.6, jb * 3.6);
    // equal masses, restitution 0.2: the striking car loses (1 + e) / 2 of 50 km/h = 30, the wreck gains it
    assert!((dv - 30.0).abs() < 4.0, "delta-v {dv}");
    assert!((vb * 3.6 - 30.0).abs() < 4.0, "the wreck is pushed: {vb}");
    // momentum along the road is conserved by the impulse
    let (p0, p1) = (mass(SEDAN) * v0, mass(SEDAN) * (va + vb));
    assert!((p1 - p0).abs() / p0 < 0.02, "momentum {p0} -> {p1}");
    // the JS: the wreck is immovable, the car takes the whole 50 km/h plus the rebound (clamped to 0)
    assert!(jb.abs() < 1e-6 && (v0 - ja) * 3.6 > 45.0, "js {ja} {jb}");
}

#[test]
fn wreck_real_mass_bus_into_parked_sedan() {
    let Some((v0, va, vb)) = rear_end(SimOpts::default(), BUS, SEDAN, 50.0) else { return bayenv_world::oracle::skip("physics_fixes", "fixtures") };
    let (_, ja, _) = rear_end(SimOpts::js(), BUS, SEDAN, 50.0).unwrap();
    eprintln!("bus: {:.1} -> {:.1} km/h (sedan {:.1}); js -> {:.1}", v0 * 3.6, va * 3.6, vb * 3.6, ja * 3.6);
    // a 12 t bus barely slows hitting a 1.5 t car; the JS stopped it dead
    assert!(va > 0.75 * v0, "bus keeps most of its speed: {va}");
    assert!(vb > v0, "the sedan is thrown faster than the bus");
    let (p0, p1) = (mass(BUS) * v0, mass(BUS) * va + mass(SEDAN) * vb);
    assert!((p1 - p0).abs() / p0 < 0.03, "momentum {p0} -> {p1}");
    assert!(ja < 0.05 * v0, "js: bus stopped dead ({ja})");
}

/// data/scenarios/<name> (a small real-map cut with junctions), None (skip) when the data dir is missing
fn scenario_world(name: &str) -> Option<Arc<World>> { scenario_world_s(name, false) }
fn scenario_world_s(name: &str, smooth: bool) -> Option<Arc<World>> {
    let d = bayenv_sim::population_gen::default_data_dir().join("scenarios").join(name);
    if !d.join("roads.meta.json").exists() { return None; }
    Some(bayenv_sim::env::Env::load_world_with(&d, false, smooth).expect("scenario world"))
}


/// A car approaching a corner at `kmh` with steer `policy(sim, i)`: on "road3" the 90-degree kink b (no arc), on a
/// scenario world the first junction turn movement (|turn| > 1.2 rad, in-edge >= 150 m). Returns (max world lateral accel in g while
/// driving, drove 30 m down the out-edge, any barrier / contact).
fn corner(world: &str, opts: SimOpts, kmh: f64, policy: impl Fn(&Sim, usize) -> f64) -> Option<(f64, bool, bool)> {
    let g = if world == "road3" { common::shared_world(world)? } else { scenario_world_s(world, opts.smooth_geometry)? };
    let mut sim = sim_with(&g, opts);
    let (route, s0, lane) = if world == "road3" { ([0, 1], g.e_length[0] - 120.0, 0) } else {
        let j = g.j.as_ref().expect("junctions");
        let m = (0..j.move_count).find(|&m| j.turn(m).abs() > 1.2 && g.e_length[j.info(m, 0) as usize] >= 150.0).expect("a turn");
        ([j.info(m, 0) as i32, j.info(m, 2) as i32], j.s_in(m) - 100.0, j.info(m, 1) as i32)
    };
    let a = place(&mut sim, SEDAN, &route, s0, lane, kmh / 3.6);
    let out = route[1];
    let (mut peak, mut reached) = (0.0f64, false);
    let mut h0 = sim.vs.heading[a] as f64;
    for _ in 0..600 {
        if sim.vs.state[a] != STATE_DRIVING { break; }
        let u = sim.vs.speed[a] as f64;
        let st = policy(&sim, a);
        hold(&mut sim, a, st, ((kmh / 3.6 - u) * 0.5).clamp(0.0, 1.0), 0.0);
        sim.step();
        if sim.vs.state[a] != STATE_DRIVING { break; }
        let h = sim.vs.heading[a] as f64;
        let d = (h - h0).sin().atan2((h - h0).cos());
        peak = peak.max(sim.vs.speed[a] as f64 * d.abs() / sim.dt / 9.81);
        h0 = h;
        if std::env::var("CORNER_DEBUG").is_ok() { eprintln!("e {} c {} s {:.1} lat {:.2} hr {:.2} u {:.1} x {:.1} z {:.1}", sim.vs.edge[a], sim.vs.conn[a], sim.vs.s[a], sim.vs.lat[a], sim.vs.heading_rel[a], sim.vs.speed[a], sim.vs.x[a], sim.vs.z[a]); }
        // made the turn: 30 m down the new road, still on it
        if sim.vs.edge[a] == out && sim.vs.conn[a] < 0 && sim.vs.s[a] > 30.0 { reached = true; break; }
    }
    Some((peak, reached, sim.counters.crashes + sim.counters.minor_contacts > 0))
}

#[test]
fn curvature_dynamics_steer_zero_does_not_turn() {
    let Some((peak, reached, _)) = corner("road3", curv(), 108.0, |_, _| 0.0) else { return bayenv_world::oracle::skip("physics_fixes", "fixtures") };
    let (jpeak, jreached, _) = corner("road3", SimOpts::js(), 108.0, |_, _| 0.0).unwrap();
    eprintln!("108 km/h, steer 0: fix peak {peak:.2} g reached {reached}; js peak {jpeak:.2} g reached {jreached}");
    // the JS turned the car with the road: a freeway-speed right angle with the wheel straight
    assert!(jreached && jpeak > 3.0, "js {jpeak} {jreached}");
    // now the car goes straight on (and off the road)
    assert!(!reached, "a car with the wheel straight must not take the corner");
    assert!(peak < 0.2, "no lateral acceleration without steering: {peak}");
}

#[test]
fn curvature_dynamics_steered_corner_is_physical() {
    // a simple heading + offset tracker at 30 km/h through a junction turn (the connector keeps the in-edge lat)
    let pol = |sim: &Sim, i: usize| {
        let hr = sim.vs.heading_rel[i] as f64;
        let e = sim.vs.edge[i] as usize;
        let err = sim.vs.lat[i] as f64 - (sim.vs.lane[i].max(0) as f64 + 0.5) * sim.g.lane_w(e, (sim.vs.s[i] as f64).min(sim.g.e_length[e]));
        let mx = bayenv_sim::physics::VEHICLE_PARAMS[0].max_steer;
        ((-1.2 * hr - (0.08 * err).clamp(-0.1, 0.1)) / mx).clamp(-1.0, 1.0)
    };
    let Some((peak, reached, touched)) = corner("junction-s1", curv(), 30.0, pol) else { return bayenv_world::oracle::skip("physics_fixes", "fixtures") };
    eprintln!("30 km/h steered: peak {peak:.2} g, reached {reached}, touched {touched}");
    assert!(reached, "a steered car takes the corner");
    assert!(peak < 1.1, "within tyre grip: {peak} g");
}

/// The heading + offset tracker of `curvature_dynamics_steered_corner_is_physical`.
fn tracker(sim: &Sim, i: usize) -> f64 {
    let hr = sim.vs.heading_rel[i] as f64;
    let e = sim.vs.edge[i] as usize;
    let err = sim.vs.lat[i] as f64 - (sim.vs.lane[i].max(0) as f64 + 0.5) * sim.g.lane_w(e, (sim.vs.s[i] as f64).min(sim.g.e_length[e]));
    let mx = bayenv_sim::physics::VEHICLE_PARAMS[0].max_steer;
    ((-1.2 * hr - (0.08 * err).clamp(-0.1, 0.1)) / mx).clamp(-1.0, 1.0)
}

/// The sharpest kink (< 1.6 rad) at a pass-through join (no junction box, no other edge; in-edge >= 70 m, out-edge
/// >= 40 m, same lanes, speed limit <= 72 km/h)
/// among a few scenario cuts: (scenario, in-edge, out-edge, turn).
fn sharpest_join() -> Option<(&'static str, usize, usize, f64)> {
    let mut best: Option<(&'static str, usize, usize, f64)> = None;
    for name in ["curve-s1", "sbend-s1", "urban-s1", "grid-s1", "hill-s1", "junction-s1", "lanes-s1", "interchange-s1", "roundabout-s1", "weave-s1", "pack-mixed-s3"] {
        let Some(g) = scenario_world(name) else { continue };
        let lg = g.lg.as_ref()?;
        for a in 0..g.edge_count {
            let n = g.e_to[a] as usize;
            if g.j.as_ref().map_or(false, |j| j.j_of_node[n] >= 0) || g.e_length[a] < 70.0 { continue; }
            let outs: Vec<i32> = (g.out_start[n]..g.out_start[n + 1]).map(|q| g.out_edges[q as usize]).filter(|&o| o != g.e_rev[a]).collect();
            if outs.len() != 1 { continue; }
            let b = outs[0] as usize;
            let ins = (g.in_start[n]..g.in_start[n + 1]).map(|q| g.in_edges[q as usize]).filter(|&i| i != g.e_rev[b]).count();
            if ins != 1 || g.e_length[b] < 40.0 || g.e_lanes[a] != g.e_lanes[b] || g.e_speed[a] > 20.0 { continue; }
            let (p, q) = (lg.offset_point(a, g.e_length[a], 0.0, 0.0), lg.offset_point(b, 0.0, 0.0, 0.0));
            let t = (q.hz.atan2(q.hx) - p.hz.atan2(p.hx)).sin().atan2((q.hz.atan2(q.hx) - p.hz.atan2(p.hx)).cos());
            if t.abs() < 1.6 && best.map_or(true, |x| t.abs() > x.3.abs()) { best = Some((name, a, b, t)); }
        }
    }
    best
}

/// A car at `kmh` on route [a, b] from 60 m before the join, steered by `policy`: (peak world lateral g, drove 30 m
/// down b, any barrier / contact).
fn join_corner(g: &Arc<World>, a: usize, b: usize, opts: SimOpts, kmh: f64, policy: impl Fn(&Sim, usize) -> f64) -> (f64, bool, bool) {
    let mut sim = sim_with(g, opts);
    let i = place(&mut sim, SEDAN, &[a as i32, b as i32], g.e_length[a] - 60.0, 0, kmh / 3.6);
    let (mut peak, mut reached) = (0.0f64, false);
    let mut h0 = sim.vs.heading[i] as f64;
    for _ in 0..400 {
        if sim.vs.state[i] != STATE_DRIVING { break; }
        let u = sim.vs.speed[i] as f64;
        let st = policy(&sim, i);
        hold(&mut sim, i, st, ((kmh / 3.6 - u) * 0.5).clamp(0.0, 1.0), 0.0);
        sim.step();
        if sim.vs.state[i] != STATE_DRIVING { break; }
        let h = sim.vs.heading[i] as f64; let d = (h - h0).sin().atan2((h - h0).cos());
        peak = peak.max(sim.vs.speed[i] as f64 * d.abs() / sim.dt / 9.81); h0 = h;
        if sim.vs.edge[i] as usize == b && sim.vs.s[i] > 30.0 { reached = true; break; }
    }
    (peak, reached, sim.counters.crashes + sim.counters.minor_contacts + sim.counters.touches > 0)
}

/// smooth_geometry (DECISIONS.md 33): the sharpest pass-through join kink of the scenario cuts becomes a fillet arc.
/// With curvature_dynamics the frame turns by the whole kink at one point on the raw geometry; on the smoothed
/// geometry it turns by at most one arc sample there, and a heading tracker takes it within grip. Steer 0 still goes
/// straight on. (The sharp joins of the real map are checked in bayenv-world tests/smooth_world.rs.)
#[test]
fn smooth_geometry_smooths_a_join_kink() {
    let Some((name, a, b, turn)) = sharpest_join() else { return bayenv_world::oracle::skip("physics_fixes", "scenario data") };
    let (raw, sm) = (scenario_world(name).unwrap(), scenario_world_s(name, true).unwrap());
    // the frame heading step across the join: the whole kink raw, ~0 smoothed
    let step = |g: &Arc<World>| { let lg = g.lg.as_ref().unwrap(); let (p, q) = (lg.offset_point(a, g.e_length[a], 0.0, 0.0), lg.offset_point(b, 0.0, 0.0, 0.0)); let d = q.hz.atan2(q.hx) - p.hz.atan2(p.hx); d.sin().atan2(d.cos()).abs() };
    let (sr, ss) = (step(&raw), step(&sm));
    let (rp, rr, rt) = join_corner(&raw, a, b, curv(), 30.0, tracker);
    let (sp, sr2, st) = join_corner(&sm, a, b, curv_smooth(), 30.0, tracker);
    let (zp, _, _) = join_corner(&sm, a, b, curv_smooth(), 30.0, |_, _| 0.0);
    eprintln!("{name}: join {a} -> {b} turns {turn:.2} rad; frame step at the join raw {sr:.3} rad, smoothed {ss:.4} rad; tracker at 30 km/h: raw peak {rp:.2} g reached {rr} touched {rt}, smoothed peak {sp:.2} g reached {sr2} touched {st}; steer 0 smoothed: peak {zp:.2} g");
    // the smoothed join lies inside a fillet arc: the frame steps by at most one arc sample (arc_step) there
    assert!(ss <= bayenv_world::smooth::SmoothCfg::default().arc_step + 1e-6 && ss < sr, "frame heading step at the smoothed join: {sr} -> {ss}");
    assert!(sr2 && !st, "a steered car takes the smoothed join");
    assert!(sp < 1.0 && sp <= rp + 1e-9, "within grip, no harsher than raw: {sp} vs {rp}");
    assert!(zp < 0.2, "steer 0 does not turn: {zp}");
}

/// The junction-s1 turn of `curvature_dynamics_steered_corner_is_physical` on the smoothed geometry (the connector is
/// re-fitted to the smoothed lanes): still taken, within grip.
#[test]
fn smooth_geometry_steered_junction_turn() {
    let Some((peak, reached, touched)) = corner("junction-s1", curv_smooth(), 30.0, tracker) else { return bayenv_world::oracle::skip("physics_fixes", "fixtures") };
    eprintln!("junction-s1, 30 km/h steered, smoothed: peak {peak:.2} g, reached {reached}, touched {touched}");
    assert!(reached, "a steered car takes the corner");
    assert!(peak < 1.1, "within tyre grip: {peak} g");
}

#[test]
#[should_panic(expected = "SimOpts.smooth_geometry")]
fn smooth_geometry_must_match_the_world() {
    let Some(g) = common::shared_world("road3") else { panic!("SimOpts.smooth_geometry (fixtures unavailable)") };
    let _ = Sim::new(g.clone(), Arc::new(Population::default()), curv_smooth(), bayenv_sim::harness::router_for(&g));
}

#[test]
fn wreck_end_stop_no_frozen_speed() {
    let Some(g) = common::shared_world("road3") else { return bayenv_world::oracle::skip("physics_fixes", "fixtures") };
    let run = |opts: SimOpts| {
        let mut sim = sim_with(&g, opts);
        let a = place(&mut sim, SEDAN, &[0], g.e_length[0] - 15.0, 1, 25.0);
        sim.vs.state[a] = STATE_CRASHED; sim.vs.flags[a] |= FLAG_DISABLED;
        let mut frozen = 0;
        for _ in 0..60 {
            let s0 = sim.vs.s[a];
            sim.step();
            if sim.vs.s[a] == s0 && sim.vs.speed[a] > 0.5 { frozen += 1; }
        }
        frozen
    };
    let (fix, js) = (run(SimOpts::default()), run(SimOpts::js()));
    eprintln!("steps pinned at the edge end with speed: fix {fix}, js {js}");
    assert!(js > 5, "js: the wreck keeps its speed at a fixed position");
    assert_eq!(fix, 0);
}

#[test]
fn brake_hold_stops_creep() {
    let Some(g) = common::shared_world("road3") else { return bayenv_world::oracle::skip("physics_fixes", "fixtures") };
    let run = |opts: SimOpts| {
        let mut sim = sim_with(&g, opts);
        let a = place(&mut sim, SEDAN, &[0], 100.0, 0, 0.015);
        for _ in 0..1200 { hold(&mut sim, a, 0.0, 0.0, 1.0); sim.step(); }
        (sim.vs.s[a] as f64 - 100.0, sim.vs.speed[a] as f64)
    };
    let ((ds, u), (jds, ju)) = (run(SimOpts::default()), run(SimOpts::js()));
    eprintln!("60 s on the brake from 0.015 m/s: fix crept {ds:.3} m (u {u}), js {jds:.3} m (u {ju})");
    assert!(jds > 0.5 && ju > 0.0, "js creeps");
    assert!(ds < 0.01 && u == 0.0);
}

#[test]
fn heavy_brakes_stopping_distance() {
    let Some(g) = common::shared_world("road3") else { return bayenv_world::oracle::skip("physics_fixes", "fixtures") };
    let run = |opts: SimOpts, ty: u8| {
        let mut sim = sim_with(&g, opts);
        let a = place(&mut sim, ty, &[0], 100.0, 1, 100.0 / 3.6);
        for _ in 0..600 { hold(&mut sim, a, 0.0, 0.0, 1.0); sim.step(); if sim.vs.speed[a] <= 0.0 { break; } }
        sim.vs.s[a] as f64 - 100.0
    };
    for ty in [TRUCK, BUS] {
        let (d, jd) = (run(SimOpts::default(), ty), run(SimOpts::js(), ty));
        eprintln!("type {ty} 100-0 km/h: fix {d:.1} m, js {jd:.1} m");
        assert!((jd - 58.0).abs() < 4.0, "js {jd}");
        assert!((74.0..=88.0).contains(&d), "about 80 m: {d}");
    }
    // cars are unchanged
    let (d, jd) = (run(SimOpts::default(), SEDAN), run(SimOpts::js(), SEDAN));
    assert!((d - jd).abs() < 0.5, "sedan {d} vs {jd}");
}

#[test]
fn impact_reverse_light_car_pushed_back() {
    // a bus hits a stationary sedan head-on (the sedan faces the bus); the sedan becomes a wreck sliding backwards
    let Some(g) = common::shared_world("road3") else { return bayenv_world::oracle::skip("physics_fixes", "fixtures") };
    let run = |opts: SimOpts| {
        let mut sim = sim_with(&g, opts);
        let a = place(&mut sim, BUS, &[0], 100.0, 0, 50.0 / 3.6);
        let b = place(&mut sim, SEDAN, &[0], 125.0, 0, 0.0);
        sim.vs.heading_rel[b] = std::f32::consts::PI; sim.vs.update_pose(b, &g); sim.vs.save_pose(b);
        let mut min_u = 0.0f64;
        for _ in 0..80 { hold(&mut sim, a, 0.0, 0.0, 0.0); hold(&mut sim, b, 0.0, 0.0, 0.0); sim.step(); min_u = min_u.min(sim.vs.speed[b] as f64); }
        (min_u, sim.vs.state[b])
    };
    let ((u, st), (ju, _)) = (run(SimOpts::default()), run(SimOpts::js()));
    eprintln!("sedan speed after a head-on bus strike: fix min {u:.2} m/s (state {st}), js min {ju:.2}");
    assert_eq!(ju, 0.0, "js clamps backward speed");
    assert!(u < -3.0 && st == STATE_CRASHED, "pushed back: {u}");
}

#[test]
fn barrier_soft_push_caps_correction() {
    let Some(g) = common::shared_world("road3") else { return bayenv_world::oracle::skip("physics_fixes", "fixtures") };
    let run = |opts: SimOpts| {
        let mut sim = sim_with(&g, opts);
        let a = place(&mut sim, SEDAN, &[0], 100.0, 0, 10.0);
        // 1.2 m into the right-hand barrier, no lateral speed
        let lat = sim.vs.lat[a] as f64;
        let w = g.lane_w(0, 100.0);
        sim.vs.lat[a] = (lat - 0.5 * w - 1.0 - 3.0 - 1.2 + sim.vs.hw[a] as f64) as f32;
        sim.vs.update_lanes(a, &g, 0); sim.vs.update_pose(a, &g); sim.vs.save_pose(a);
        let l0 = sim.vs.lat[a] as f64;
        hold(&mut sim, a, 0.0, 0.0, 0.0); sim.step();
        sim.vs.lat[a] as f64 - l0
    };
    let (d, jd) = (run(SimOpts::default()), run(SimOpts::js()));
    eprintln!("one-step lateral jump out of the barrier: fix {d:.2} m, js {jd:.2} m");
    assert!(jd > 0.8, "js removes the whole depth: {jd}");
    assert!(d > 0.0 && d < 0.35, "capped: {d}");
}

#[test]
fn contact_episodes_second_impact_reports() {
    // car a leans on the right-hand barrier (steering into it) for 5 s, touching it when it takes a 6 m/s
    // lateral strike into the same barrier. With contact_episodes the strike is an event (its
    // delta-v adds to the episode); the JS reported only the first touch of the episode.
    let Some(g) = common::shared_world("road3") else { return bayenv_world::oracle::skip("physics_fixes", "fixtures") };
    let run = |opts: SimOpts| {
        let mut sim = sim_with(&g, opts);
        let a = place(&mut sim, SEDAN, &[0], 100.0, 0, 10.0);
        let ev = |sim: &Sim| sim.counters.crashes + sim.counters.minor_contacts;
        let mut touching = 0;
        for _ in 0..100 { hold(&mut sim, a, -0.03, 0.1, 0.0); sim.step(); if sim.touching_barrier(a) { touching += 1; } }
        let lean = ev(&sim);
        let still = sim.touching_barrier(a);
        sim.vs.v_lat[a] = -6.0;
        // the strike step itself (later steps may bounce off and re-touch: a new episode in both models)
        let n0 = ev(&sim);
        hold(&mut sim, a, -0.03, 0.1, 0.0); sim.step();
        (lean, touching, still, ev(&sim) - n0, sim.crashes.last().map(|c| c.delta_v_kmh.clone()))
    };
    let (fix, js) = (run(SimOpts::default()), run(SimOpts::js()));
    eprintln!("lean events / touching steps / touching at strike / strike events / last dv: fix {fix:?}, js {js:?}");
    assert!(fix.2 && js.2, "a is leaning on the barrier when struck");
    assert!(fix.3 >= 1, "the strike during the contact reports");
    assert_eq!(js.3, 0, "js: nothing while the touch lasts");
}

#[test]
fn connector_barrier_walls_junction_and_no_exit_teleport() {
    // steer 0 into a junction turn at 30 km/h: the car runs wide inside the junction box. With connector_barrier it
    // meets the box's wall there; the JS let it wander and snapped lat back at the exit (a teleport).
    let Some(g) = scenario_world("junction-s1") else { return bayenv_world::oracle::skip("physics_fixes", "no data dir") };
    let j = g.j.as_ref().expect("junctions");
    let m = (0..j.move_count).find(|&m| j.turn(m).abs() > 1.2 && g.e_length[j.info(m, 0) as usize] >= 150.0).expect("a turn");
    let route = [j.info(m, 0) as i32, j.info(m, 2) as i32];
    let run = |opts: SimOpts| {
        let mut sim = sim_with(&g, opts);
        let a = place(&mut sim, SEDAN, &route, j.s_in(m) - 40.0, j.info(m, 1) as i32, 30.0 / 3.6);
        let (mut jump, mut wall_on_conn) = (0.0f64, false);
        for _ in 0..200 {
            if sim.vs.state[a] != STATE_DRIVING { break; }
            let (x0, z0, u0, conn0) = (sim.vs.x[a] as f64, sim.vs.z[a] as f64, sim.vs.speed[a] as f64, sim.vs.conn[a]);
            hold(&mut sim, a, 0.0, ((30.0 / 3.6 - u0) * 0.5).clamp(0.0, 1.0), 0.0);
            sim.step();
            if sim.touching_barrier(a) && (conn0 >= 0 || sim.vs.conn[a] >= 0) { wall_on_conn = true; }
            if sim.vs.state[a] != STATE_DRIVING { break; }
            let d = ((sim.vs.x[a] as f64 - x0).powi(2) + (sim.vs.z[a] as f64 - z0).powi(2)).sqrt();
            jump = jump.max(d - 1.5 * u0 * sim.dt - 0.5);
        }
        (jump, wall_on_conn)
    };
    // with curvature_dynamics (the frame no longer turns the car, so a steer-0 car runs wide)
    let ((jump, wall), (jjump, jwall)) = (run(curv()), run(SimOpts { curvature_dynamics: true, ..SimOpts::js() }));
    eprintln!("turn {:.2} rad, steer 0: fix teleport excess {jump:.2} m, wall in the box {wall}; js teleport excess {jjump:.2} m, wall {jwall}", j.turn(m));
    assert!(wall, "the junction box has walls");
    assert!(jump <= 0.0, "no teleport: {jump}");
    assert!(!jwall, "without connector_barrier the box has no walls");
}

#[test]
#[should_panic(expected = "SimOpts.dt")]
fn dt_zero_rejected() {
    let Some(g) = common::shared_world("road3") else { panic!("SimOpts.dt (fixtures unavailable)") };
    let _ = Sim::new(g.clone(), Arc::new(Population::default()), SimOpts { dt: 0.0, ..SimOpts::default() }, bayenv_sim::harness::router_for(&g));
}

#[test]
fn heading_preview_sees_the_corner_ahead() {
    let Some(g) = common::shared_world("road3") else { return bayenv_world::oracle::skip("physics_fixes", "fixtures") };
    let mut sim = sim_with(&g, SimOpts::default());
    // 15 m before the 90-degree kink at b, route a -> b -> c
    let a = place(&mut sim, SEDAN, &[0, 1], g.e_length[0] - 15.0, 0, 10.0);
    let p = bayenv_sim::obs::heading_preview(&sim, a);
    eprintln!("preview before the kink: {p:?}");
    assert!(p[0].abs() < 1e-6 && p[1].abs() < 1e-6, "straight for the first 10 m");
    assert!((p[2].abs() - std::f64::consts::FRAC_PI_2).abs() < 0.1 && (p[3] - p[2]).abs() < 1e-6, "the right angle at 20 and 40 m");
    // on a junction connector of a real-map cut: the preview follows the connector's turn
    let Some(w) = scenario_world("junction-s1") else { return };
    let j = w.j.as_ref().unwrap();
    let m = (0..j.move_count).find(|&m| j.turn(m).abs() > 1.2 && w.e_length[j.info(m, 0) as usize] >= 150.0).unwrap();
    let mut sim = sim_with(&w, SimOpts::default());
    let c = place(&mut sim, SEDAN, &[j.info(m, 0) as i32, j.info(m, 2) as i32], j.s_in(m) - 2.0, j.info(m, 1) as i32, 5.0);
    let p = bayenv_sim::obs::heading_preview(&sim, c);
    eprintln!("preview 2 m before a {:.2} rad turn: {p:?}", j.turn(m));
    assert!((p[3] - j.turn(m)).abs() < 0.25, "the 40 m preview sees the whole turn");
    assert!(p[3].signum() == j.turn(m).signum());
}

/// spawn_clearance (DECISIONS.md 34): the spawn edge chosen for a route leaves at least spawn_clearance_m between its
/// spawn point (s_exit) and the junction entry the route takes next, unless it is the route's last edge; without the
/// flag (the JS) cars spawn on the first edge however close the junction. Checked for every two-edge route through a
/// junction of the real-map scenario cuts.
#[test]
fn spawn_clearance_keeps_spawns_off_junction_entries() {
    let (mut close, mut moved, mut kept) = (0usize, 0usize, 0usize);
    for name in ["urban-s1", "grid-s1", "junction-s1", "roundabout-s1", "interchange-s1", "pack-mixed-s3", "weave-s1", "merge-s1", "diverge-s1"] {
    let Some(w) = scenario_world(name) else { return bayenv_world::oracle::skip("physics_fixes", "scenario data") };
    // the scenario cuts have long blocks: a 50 m clearance makes the rule bite (the default is 18 m)
    let sim = sim_with(&w, SimOpts { spawn_clearance: true, spawn_clearance_m: 50.0, ..SimOpts::default() });
    let j = w.j.as_ref().unwrap();
    let entry = |e: usize, nx: i32| (0..w.e_lanes[e]).filter_map(|l| { let m = j.movement(e, l, nx); (m >= 0).then(|| j.s_in(m as usize)) }).fold(f64::INFINITY, f64::min);
    for m in 0..j.move_count {
        let (a, b) = (j.info(m, 0) as usize, j.info(m, 2) as i32);
        let s0 = w.e_s_exit[a].max(0.0);
        let gap = entry(a, b) - s0;
        if gap < sim.opts.spawn_clearance_m { close += 1; }
        // route [a, b, b's continuation if any]
        let to = w.e_to[b as usize] as usize;
        let mut route = vec![a as i32, b];
        if let Some(q) = (w.out_start[to]..w.out_start[to + 1]).map(|q| w.out_edges[q as usize]).find(|&o| o != w.e_rev[b as usize]) { route.push(q); }
        let k = sim.spawn_edge_index(&route);
        if gap < sim.opts.spawn_clearance_m { assert!(k >= 1, "movement {m}: moved on"); moved += 1; } else { assert_eq!(k, 0); kept += 1; }
        if k + 1 < route.len() {
            let e = route[k] as usize;
            if w.e_s_stop[e] < w.e_length[e] { let g2 = entry(e, route[k + 1]) - w.e_s_exit[e].max(0.0); assert!(!g2.is_finite() || g2 >= sim.opts.spawn_clearance_m, "route {route:?}: spawn edge {k} gap {g2}"); }
        }
    }
    }
    eprintln!("scenario cuts, movements: {close} have their in-edge's spawn point within 50 m of the entry ({moved} spawn further along with spawn_clearance), {kept} keep the first edge");
    assert!(close > 0 && moved == close);
}

#[test]
fn static_hold_and_idle_creep_pull_away() {
    // ENV-REQUESTS #17: the first brake_hold counted rolling resistance as holding force, so a car pulling away with
    // a little gas (or gas + some brake) stayed parked for ever
    let Some(g) = common::shared_world("road3") else { return bayenv_world::oracle::skip("physics_fixes", "fixtures") };
    let run = |opts: SimOpts, gas: f64, brake: f64| {
        let mut sim = sim_with(&g, opts);
        let a = place(&mut sim, SEDAN, &[0], 100.0, 0, 0.0);
        for _ in 0..400 { hold(&mut sim, a, 0.0, gas, brake); sim.step(); }
        (sim.vs.s[a] as f64 - 100.0, sim.vs.speed[a] as f64)
    };
    let v1 = SimOpts { brake_hold: true, static_hold: false, idle_creep: false, ..SimOpts::default() };
    let (d_v1, _) = run(v1, 0.02, 0.0);
    let (d_new, _) = run(SimOpts::default(), 0.02, 0.0);
    let (_, u_creep) = run(SimOpts::default(), 0.0, 0.0);
    let (d_brk, _) = run(SimOpts::default(), 0.0, 0.1);
    let (d_js, _) = run(SimOpts::js(), 0.0, 0.0);
    eprintln!("20 s from rest: gas 2 %: brake_hold v1 {d_v1:.2} m, now {d_new:.1} m; no pedals: creep {u_creep:.2} m/s (js {d_js:.2} m); brake 10 %: {d_brk:.3} m");
    assert!(d_v1 < 0.01, "v1 strands the car");
    assert!(d_new > 10.0, "gas 2 % pulls away");
    assert!((1.2..=1.8).contains(&u_creep), "idle creep about 1.5 m/s: {u_creep}");
    assert!(d_brk < 0.01, "a light brake holds against the creep");
    assert!(d_js < 0.01, "the JS: no creep");
}

#[test]
fn steer_delay_is_short_pedals_keep_reaction_time() {
    // ENV-REQUESTS #16: steering waits steer_delay (0.15 s), the pedals the reaction time
    let Some(g) = common::shared_world("road3") else { return bayenv_world::oracle::skip("physics_fixes", "fixtures") };
    let run = |opts: SimOpts| {
        let mut sim = sim_with(&g, opts);
        let a = place(&mut sim, SEDAN, &[0], 100.0, 1, 10.0);
        sim.vs.set_reaction_time(a, 1.0, sim.dt);
        for _ in 0..30 { hold(&mut sim, a, 0.0, 0.0, 0.0); sim.step(); }
        let (mut t_steer, mut t_brake) = (-1.0, -1.0);
        let u0 = sim.vs.speed[a] as f64;
        for k in 0..60 {
            hold(&mut sim, a, 0.3, 0.0, 1.0); sim.step();
            let t = (k + 1) as f64 * sim.dt;
            if t_steer < 0.0 && sim.vs.steer_angle[a].abs() > 1e-4 { t_steer = t; }
            if t_brake < 0.0 && (sim.vs.speed[a] as f64) < u0 - 0.5 { t_brake = t; }
        }
        (t_steer, t_brake)
    };
    let ((s, b), (js, jb)) = (run(SimOpts::default()), run(SimOpts::js()));
    eprintln!("first response (s): steer {s:.2}, brake {b:.2}; js steer {js:.2}, brake {jb:.2}");
    assert!(s <= 0.25 && s > 0.0, "steering after about 0.15 s: {s}");
    assert!(b >= 1.0, "pedals keep the 1 s reaction time: {b}");
    assert!(js >= 1.0, "the JS delays steering by the reaction time: {js}");
}

#[test]
fn mountable_kerbs_scrape_not_crash() {
    // audit finding 10 / the teacher's low-speed junction kerb touches: a car at walking pace steering into a city kerb.
    // Rigid kerb + contact_episodes: every step's push is an impact, the episode delta-v adds up to a disabling
    // "crash"; with mountable_kerbs it is one minor contact and the car is not disabled.
    let Some(g) = common::shared_world("road3") else { return bayenv_world::oracle::skip("physics_fixes", "fixtures") };
    let run = |opts: SimOpts, u: f64, hr: f64| {
        let mut sim = sim_with(&g, opts);
        let i = place(&mut sim, SEDAN, &[1], 100.0, 0, u);
        sim.vs.lat[i] = 1.0; sim.vs.heading_rel[i] = -hr as f32;
        sim.vs.update_lanes(i, &g, 1); sim.vs.update_pose(i, &g); sim.vs.save_pose(i);
        for _ in 0..60 { if sim.vs.state[i] != STATE_DRIVING { break; } hold(&mut sim, i, -0.15, 0.1, 0.0); sim.step(); }
        (sim.counters.crashed_vehicles, sim.counters.minor_contacts, sim.vs.state[i])
    };
    let rigid = SimOpts { mountable_kerbs: false, ..SimOpts::default() };
    let (rc, _, _) = run(rigid, 3.5, 0.45);
    let (kc, km, ks) = run(SimOpts::default(), 3.5, 0.45);
    eprintln!("3.5 m/s at 0.45 rad steering into the kerb: rigid {rc} crashed; mountable {kc} crashed, {km} minor contacts");
    assert_eq!(rc, 1, "rigid: the grind adds up to a disabling crash");
    assert!(kc == 0 && km == 1 && ks == STATE_DRIVING, "mountable: one minor scrape");
    // residential road 1 is not rigid; the motorway (edge 0) is
    assert!(!g.rigid_edge(1) && g.rigid_edge(0));
}

#[test]
fn box_barrier_walls_the_junction_box_part_of_an_edge() {
    // the "phantom wall" (DECISIONS.md 39): a car crossing the junction-box part of an edge (s < sExit + 1) at a large
    // heading error, as after a spin or with an untrained policy. The JS has no walls there: the car drives sideways off
    // the road, then the first barrier check past the box meets it tens of metres deep, with a delta-v near its speed.
    // With box_barrier it meets a wall at the box's edge (the edge's barrier lines + BOX_MARGIN), still in the box.
    let Some(g) = scenario_world("junction-s1") else { return bayenv_world::oracle::skip("physics_fixes", "no data dir") };
    let e = (0..g.edge_count).filter(|&e| g.e_s_exit[e] >= 4.0 && g.e_length[e] >= 60.0 && g.e_rev[e] < 0)
        .chain((0..g.edge_count).filter(|&e| g.e_s_exit[e] >= 4.0 && g.e_length[e] >= 60.0)).next().expect("an edge with a start box");
    let run = |opts: SimOpts| {
        let mut sim = sim_with(&g, opts);
        sim.barrier_dbg = Some(Vec::new());
        let i = place(&mut sim, SEDAN, &[e as i32], 0.5, 0, 20.0);
        sim.vs.heading_rel[i] = 1.45; // nearly across the road, toward the left
        sim.vs.update_lanes(i, &g, e); sim.vs.update_pose(i, &g); sim.vs.save_pose(i);
        let (mut max_lat, mut box_touch, mut first_depth) = (0.0f64, false, None);
        for _ in 0..200 {
            if sim.vs.state[i] != STATE_DRIVING { break; }
            hold(&mut sim, i, 0.0, 0.3, 0.0);
            sim.step();
            max_lat = max_lat.max(sim.vs.lat[i] as f64);
            for d in sim.barrier_dbg.as_mut().unwrap().drain(..) { if d.slot == i { if d.in_box { box_touch = true; } if first_depth.is_none() { first_depth = Some(d.depth); } } }
            if first_depth.is_some() { break; }
        }
        (max_lat, box_touch, first_depth.unwrap_or(0.0))
    };
    let (lat, touch, depth) = run(SimOpts::default());
    let (jlat, jtouch, jdepth) = run(SimOpts::js());
    let lanes_w = g.e_lanes[e] as f64 * g.lane_w(e, 0.0);
    eprintln!("edge {e} (sExit {:.1}, {lanes_w:.1} m of lanes): box_barrier max lat {lat:.1} m, wall in the box {touch}, first depth {depth:.2} m; js max lat {jlat:.1} m, wall in the box {jtouch}, first depth {jdepth:.2} m", g.e_s_exit[e]);
    assert!(touch, "the box part has walls");
    assert!(depth < 3.0 && lat < lanes_w + 15.0, "the wall stops the car near the box: lat {lat}, depth {depth}");
    assert!(!jtouch, "the JS: no walls in the box");
    assert!(jlat > lanes_w + 20.0 && jdepth > 10.0, "the JS: the car runs far off the road and meets a phantom wall: lat {jlat}, depth {jdepth}");
}

#[test]
fn lane_drop_hop_takes_the_edge_the_lane_lands_nearest() {
    // DECISIONS.md 39: a 6-lane motorway diverging into a 4-lane mainline and a 1-lane ramp drops its far-left lane at
    // the node. A car still in that lane, routed onto the ramp: the JS kept the ramp and the strict lat (lane 5 of 6),
    // 16.6 m past the ramp's barrier line. With lane_drop_hop it stays on the mainline (a missed exit), about one lane out.
    let d = bayenv_sim::population_gen::default_data_dir();
    if !d.join("roads.meta.json").exists() { return bayenv_world::oracle::skip("physics_fixes", "no data dir"); }
    let g = bayenv_sim::env::Env::load_world(&d).expect("world");
    // the first (edge, lane, ramp) whose lane maps onto no strict out-edge
    let mut case = None;
    'f: for e in 0..g.edge_count {
        if !g.is_shoulder_road(e) || g.e_length[e] < 200.0 || (g.j.is_some() && g.e_s_stop[e] < g.e_length[e]) { continue; }
        let to = g.e_to[e] as usize;
        let outs: Vec<usize> = (g.out_start[to]..g.out_start[to + 1]).map(|q| g.out_edges[q as usize] as usize).filter(|&o| o as i32 != g.e_rev[e] && g.hop(e, o).strict).collect();
        if outs.len() < 2 { continue; }
        for k in 0..g.e_lanes[e] {
            if outs.iter().all(|&o| { let l = g.map_lane(e, o, k); l < 0 || l >= g.e_lanes[o] }) {
                // the out-edge it lands furthest from (the ramp)
                let far = *outs.iter().max_by_key(|&&o| { let l = g.map_lane(e, o, k); if l < 0 { -l } else { l - g.e_lanes[o] + 1 } }).unwrap();
                case = Some((e, k, far)); break 'f;
            }
        }
    }
    let (e, lane, ramp) = case.expect("a lane dropped at a diverge");
    let run = |opts: SimOpts| {
        let mut sim = sim_dt(&g, opts, 0.1);
        sim.barrier_dbg = Some(Vec::new());
        let i = place(&mut sim, SEDAN, &[e as i32, ramp as i32], g.e_length[e] - 60.0, lane, 25.0);
        let mut first: Option<(i32, f64)> = None;
        for _ in 0..60 {
            if sim.vs.state[i] != STATE_DRIVING { break; }
            let gas = ((25.0 - sim.vs.speed[i] as f64) * 0.5).clamp(0.0, 1.0); hold(&mut sim, i, 0.0, gas, 0.0);
            sim.step();
            for b in sim.barrier_dbg.as_mut().unwrap().drain(..) { if b.slot == i && first.is_none() { first = Some((b.edge, b.depth)); } }
            if sim.vs.edge[i] as usize != e && sim.vs.s[i] > 20.0 { break; }
        }
        (sim.vs.edge[i] as usize, first)
    };
    let (fe, fb) = run(SimOpts::default());
    let (je, jb) = run(SimOpts::js());
    eprintln!("e{e} lane {lane} -> planned ramp e{ramp}: fix on e{fe}, first barrier {fb:?}; js on e{je}, first barrier {jb:?}");
    assert_eq!(je, ramp, "the JS takes the planned ramp");
    assert!(jb.map(|b| b.1 > 10.0).unwrap_or(false), "the JS lands far past the ramp's barrier: {jb:?}");
    assert_ne!(fe, ramp, "lane_drop_hop: the car goes where its lane physically is");
    assert!(fb.map(|b| b.1 < 3.0).unwrap_or(true), "about one lane out at most: {fb:?}");
}


/// the real map (data/), None (skip) when missing
fn real_world() -> Option<Arc<World>> {
    let d = bayenv_sim::population_gen::default_data_dir();
    if !d.join("roads.meta.json").exists() { return None; }
    Some(bayenv_sim::env::Env::load_world(&d).expect("world"))
}
/// a sedan from rest at (edge e, s0) holding (gas, brake) for `secs`: (max s, final s, min speed, final speed)
fn hill_run(g: &Arc<World>, opts: SimOpts, e: usize, s0: f64, gas: f64, brake: f64, secs: f64) -> (f64, f64, f64, f64) {
    let mut sim = sim_with(g, opts);
    let a = place(&mut sim, SEDAN, &[e as i32], s0, 0, 0.0);
    let (mut smax, mut umin) = (s0, 0.0f64);
    for _ in 0..(secs / sim.dt) as usize {
        if sim.vs.state[a] != STATE_DRIVING { break; }
        hold(&mut sim, a, 0.0, gas, brake); sim.step();
        smax = smax.max(sim.vs.s[a] as f64); umin = umin.min(sim.vs.speed[a] as f64);
    }
    (smax, sim.vs.s[a] as f64, umin, sim.vs.speed[a] as f64)
}

#[test]
fn hill_rollback_and_hill_starts() {
    // trip 818's hill (edge 124811: 17 % at s 15 rising to 23 % at s 30)
    let Some(g) = real_world() else { return bayenv_world::oracle::skip("physics_fixes", "no data dir") };
    let e = 124811;
    // gas 0.546 (trip 818's): drive ~3.06 kN against ~3.17 kN of grade + ~0.2 kN rolling resistance at s ~26: it
    // stalls at the balance point (rolling resistance holds it); gas 0.45 from rest at s 20 (20 %) can't hold the grade
    let (smax0, s_end0, umin0, _) = hill_run(&g, SimOpts::default(), e, 14.72, 0.546, 0.0, 40.0);
    eprintln!("gas 0.546 from s 14.72: peak s {smax0:.2}, end {s_end0:.2}, min speed {umin0:.2}");
    assert!(s_end0 > smax0 - 2.0, "at the balance point it stalls, it doesn't run away backwards");
    let (smax, s_end, umin, _) = hill_run(&g, SimOpts::default(), e, 20.0, 0.45, 0.0, 10.0);
    let (jmax, js_end, jumin, _) = hill_run(&g, SimOpts::js(), e, 20.0, 0.45, 0.0, 10.0);
    eprintln!("gas 0.45 from s 20: rollback peak s {smax:.2} -> {s_end:.2} (min speed {umin:.2}); js peak {jmax:.2} -> {js_end:.2} (min speed {jumin:.2})");
    assert!(jumin == 0.0 && (js_end - jmax).abs() < 1e-3, "the JS stops dead");
    assert!(umin < -0.3 && s_end < smax - 2.0, "an under-powered hill start rolls back (to where the grade eases)");
    let (smax2, _, _, _) = hill_run(&g, SimOpts::default(), e, 14.72, 0.65, 0.0, 40.0);
    eprintln!("gas 0.65: reaches s {smax2:.1}");
    assert!(smax2 > 60.0, "0.65 gas climbs");
    // stopped on the 22 % part with gas off: the brake holds it, no brake rolls back
    let (_, sb, ub, _) = hill_run(&g, SimOpts::default(), e, 25.0, 0.0, 0.6, 20.0);
    let (_, sn, un, _) = hill_run(&g, SimOpts::default(), e, 25.0, 0.0, 0.0, 5.0);
    eprintln!("at s 25, gas 0: brake 0.6 -> s {sb:.3} (min u {ub:.2}); no brake -> s {sn:.2} (min u {un:.2})");
    // (the first step runs before the brake command leaves the control ring)
    assert!((sb - 25.0).abs() < 0.02 && ub > -0.15, "the brake holds");
    assert!(sn < 24.0 && un < -0.5, "without brake it rolls back");
}

#[test]
fn full_gas_climbs_the_steepest_streets() {
    // a sedan at full gas from rest climbs the steepest 30 m of street on the map (SF tops out near 31 %)
    let Some(g) = real_world() else { return bayenv_world::oracle::skip("physics_fixes", "no data dir") };
    let (mut best, mut be, mut bs) = (0.0f64, 0usize, 0.0f64);
    for e in 0..g.edge_count {
        if g.e_length[e] < 40.0 || g.is_shoulder_road(e) { continue; }
        let mut s = 0.0;
        while s + 30.0 < g.e_length[e] {
            let m = (0..=6).map(|k| g.edge_grade(e, s + 5.0 * k as f64, -1)).fold(f64::INFINITY, f64::min);
            if m > best { best = m; be = e; bs = s; }
            s += 5.0;
        }
    }
    let (smax, _, _, _) = hill_run(&g, SimOpts::default(), be, bs, 1.0, 0.0, 30.0);
    eprintln!("steepest 30 m: edge {be} at s {bs:.0}, sustained grade {:.1} %; full gas reaches s {smax:.1}", 100.0 * best);
    assert!(best > 0.25, "the map has SF's steep streets");
    assert!(smax > bs + 30.0, "full gas climbs them");
}
