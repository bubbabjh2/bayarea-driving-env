//! Port of test/sim/collisions.test.js (3 of 3 JS test cases -> 3 #[test]s).
//! `road(3)` (3-lane one-way 3 km motorway + a residential spur) is exported by tests/common/gen_worlds.mjs ("road3").
//! The JS "no population" Simulation({commuters: []}) becomes a Sim over an empty Population (so nothing ever spawns, which
//! replaces the JS `sim._spawn = () => {}`); cars are placed by hand with VehicleStore::alloc exactly like the JS `place()`.
//! Cars without a driver hold SimOpts.default_action in both implementations (JS: drivers[i] == null; Rust: Driver::None).
mod common;
use bayenv_sim::contacts::severity_of;
use bayenv_sim::population::Population;
use bayenv_sim::sim::*;
use bayenv_sim::vehicle::*;
use bayenv_sim::world::World;
use std::sync::Arc;

fn empty_sim(g: &Arc<World>) -> Sim {
    let opts = SimOpts { dt: 0.05, start_time: 7.0 * 3600.0, fuel: false, ..SimOpts::default() };
    Sim::new(g.clone(), Arc::new(Population::default()), opts, bayenv_sim::harness::router_for(g))
}
fn place(sim: &mut Sim, edge: usize, s: f64, lane: i32, speed: f64) -> usize {
    let g = sim.g.clone();
    let (time, dt, step) = (sim.time, sim.dt, sim.step_count);
    let i = sim.vs.alloc(&g, AllocArgs { type_idx: 0, traits: DEFAULT_TRAITS, route: &[edge as i32], commuter: -1, leg: 0, time, learner: false, dt, step, lane });
    sim.vs.s[i] = s as f32; sim.vs.speed[i] = speed as f32; sim.vs.set_reaction_time(i, 0.05, dt);
    sim.vs.update_lanes(i, &g, edge); sim.vs.update_pose(i, &g); sim.vs.save_pose(i);
    if sim.drivers.len() <= i { sim.drivers.resize(i + 1, Driver::None); }
    if sim.actions.len() <= i { sim.actions.resize(i + 1, None); }
    i
}

#[test]
fn severity_tiers_by_delta_v() {
    assert_eq!(severity_of(5.0), "minor"); assert_eq!(severity_of(15.0), "moderate");
    assert_eq!(severity_of(30.0), "severe"); assert_eq!(severity_of(70.0), "very severe");
}

#[test]
fn rear_end_at_50kmh_closing_crash_scene_blocks_lane_then_clears() {
    let g = need!(common::shared_world("road3"));
    let mut sim = empty_sim(&g);
    let (a, b) = (place(&mut sim, 0, 100.0, 0, 22.0), place(&mut sim, 0, 112.0, 0, 8.0));
    let c = place(&mut sim, 0, 20.0, 0, 20.0);
    // nobody brakes: hold the pedals by writing commands directly (no driver steps for these slots)
    let mut crashed_at: i64 = -1;
    let mut k = 0;
    while k < 40 && crashed_at < 0 {
        for i in [a, b] { let st = sim.step_count + 1; sim.vs.command(i, 0.0, 0.3, 0.0, st); }
        sim.step();
        if sim.counters.crashes > 0 { crashed_at = k; }
        k += 1;
    }
    assert!(crashed_at >= 0, "the cars touched and a crash was recorded");
    let rec = sim.crashes[0].clone();
    assert!(rec.delta_v_kmh.iter().all(|&d| d > 10.0), "delta-v {:?}", rec.delta_v_kmh);
    assert!((rec.delta_v_kmh[0] - rec.delta_v_kmh[1]).abs() < 3.0, "equal masses share the delta-v");
    assert_eq!(sim.vs.state[a], STATE_CRASHED); assert_eq!(sim.vs.state[b], STATE_CRASHED);
    assert!(rec.lane_mask & 1 != 0, "lane 0 is blocked by the hitboxes");
    // the disabled cars slide to rest under tyre friction and stay there
    for _ in 0..200 { sim.step(); }
    assert!(sim.vs.speed[a] < 0.05 && sim.vs.speed[b] < 0.05);
    // the follower stops or steers around without touching
    let crashes0 = sim.counters.crashes;
    for _ in 0..20 * 90 { sim.step(); }
    assert_eq!(sim.counters.crashes, crashes0, "no secondary crash");
    assert!(sim.vs.state[c] == STATE_DRIVING || sim.vs.state[c] == STATE_FREE);
    sim.time = sim.crashes[0].clear_end + 1.0; sim.step();
    assert_eq!(sim.crashes[0].response, Response::Cleared);
    assert_eq!(sim.vs.state[a], STATE_FREE); assert_eq!(sim.vs.state[b], STATE_FREE);
}

#[test]
fn slow_touch_below_1kmh_delta_v_is_physics_only_no_crash_event() {
    let g = need!(common::shared_world("road3"));
    let mut sim = empty_sim(&g);
    let (a, b) = (place(&mut sim, 0, 100.0, 1, 10.2), place(&mut sim, 0, 104.4, 1, 10.0));
    for _ in 0..4 {
        for i in [a, b] { let st = sim.step_count + 1; sim.vs.command(i, 0.0, 0.0, 0.0, st); }
        sim.step();
    }
    assert_eq!(sim.counters.crashes, 0);
    assert!(sim.counters.touches >= 1);
}
