//! Port of test/sim/sim.test.js: 2 of 5 JS test cases ported (2 #[test]s).
//!   ported : "one simulated hour with 500 vehicles (no driver: default action)" and "scale-crash options on/off".
//!            JS `new Simulation(g, {population, ...})` -> Env over the exported synthetic world + a Rust-generated population
//!            (PopConfig::env_synthetic + workStart window, returnTrips false). Cars without commands hold the default action.
//!            getStats() fields are rebuilt from Sim: spawned/arrived/cleared/towed counters, active = vs.count, avgSpeed from `live`,
//!            arrivalRate and the failure breakdown from `outcome`.
//!            Not asserted: stats.density.freewayLaneKm (Sim::fwy_lane_km is pub(crate)) and `typeof failures.* === number` (always true in Rust).
//!   skipped: "clock: advance() honours the speed multiplier and step cap" (Simulation.advance / speed / lagging wall-clock driver, JS-only),
//!            "trip stats and genome export/import round trip" (tripStats / exportGenomes / importGenomes / Genome objects, JS-only),
//!            "runHeadless reports arrivalRate ..." (JS CLI harness around Simulation + makeSynthGraph),
//!            "route.js routingPack" (worker routing-graph packing + structuredClone; routing lives in bayenv_world::nav in Rust).
mod common;
use bayenv_sim::population_gen::*;
use bayenv_sim::sim::*;
use bayenv_sim::vehicle::*;
use serde_json::json;
use std::sync::Arc;

fn busy_pop(g: &Arc<bayenv_sim::world::World>, n: usize) -> Arc<bayenv_sim::population::Population> {
    let mut c = PopConfig::env_synthetic(n, 1.0);
    c.apply_json(&json!({ "workStart": { "mean": 8.0 * 3600.0, "sd": 15.0 * 60.0, "min": 7.7 * 3600.0, "max": 8.3 * 3600.0 } }));
    Arc::new(generate_population(g, &c, None).unwrap().pop)
}

#[test]
fn one_simulated_hour_500_vehicles_no_nans_everyone_accounted_for() {
    let g = need!(common::shared_world("synth"));
    let pop = busy_pop(&g, 500);
    let mut env = common::env_with(g.clone(), pop, SimOpts { dt: 0.05, ..SimOpts::js() }, 7.0 * 3600.0);
    let mut steps = 0;
    while env.sim.time < 8.5 * 3600.0 {
        env.step(1); steps += 1;
        if steps % 200 == 0 {
            let vs = &env.sim.vs;
            for i in 0..vs.high {
                if vs.state[i] == STATE_FREE { continue; }
                for (k, v) in [("x", vs.x[i]), ("z", vs.z[i]), ("y", vs.y[i]), ("s", vs.s[i]), ("speed", vs.speed[i]), ("accel", vs.accel[i]), ("heading", vs.heading[i])] {
                    assert!(v.is_finite(), "{k} finite at slot {i}");
                }
                assert!(vs.speed[i] >= 0.0 && vs.speed[i] <= vs.max_speed[i] + 1e-4);
                let e = vs.edge[i] as usize;
                assert!(vs.s[i] as f64 >= 0.0 && vs.s[i] as f64 <= g.e_length[e] + 1e-3);
                assert!(vs.lane[i] >= 0 && (vs.lane[i] as i32) < g.e_lanes[e]
                    && ((vs.lane_lo[i] <= vs.lane[i] && vs.lane_hi[i] >= vs.lane[i]) || vs.lane_hi[i] == vs.lane_lo[i] - 1), "footprint lanes (or off the travel lanes)");
                assert!(vs.lat[i].is_finite() && vs.heading_rel[i].is_finite() && vs.v_lat[i].is_finite());
            }
        }
    }
    let sim = &env.sim; let c = &sim.counters;
    assert!(c.spawned > 0 && c.spawned <= 500, "spawned {}", c.spawned);
    assert_eq!(c.spawned, c.arrived + sim.vs.count as u64 + c.cleared + c.towed,
        "every spawned vehicle is arrived, active (driving / on scene / refuelling / waiting), cleared or towed");
    let avg = if sim.live.driving > 0 { sim.live.sum_speed / sim.live.driving as f64 } else { 0.0 };
    assert!(avg >= 0.0 && avg.is_finite());
    let ar = sim.arrival_rate();
    assert!((0.0..=1.0).contains(&ar));
    let count = |code: u8| sim.outcome.iter().filter(|&&o| o == code).count();
    assert_eq!(count(FAIL_PENDING) + count(FAIL_ARRIVED) + count(FAIL_CRASH) + count(FAIL_STRANDED) + count(FAIL_STUCK) + count(FAIL_NOROUTE), sim.outcome.len());
}

#[test]
fn scale_crash_options_on_and_off_keep_state_finite() {
    let g = need!(common::shared_world("synth"));
    let pop = busy_pop(&g, 400);
    let run = |on: bool| {
        let f = if on { 1.0 } else { 0.0 };
        let opts = SimOpts { dt: 0.05, junction_sight: f, path_sight: f, deadlock_break: f, spawn_lane_route: f, ..SimOpts::js() };
        let mut env = common::env_with(g.clone(), pop.clone(), opts, 7.0 * 3600.0);
        while env.sim.time < 8.5 * 3600.0 { env.step(1); }
        let vs = &env.sim.vs;
        for i in 0..vs.high { if vs.state[i] != STATE_FREE { for (k, v) in [("x", vs.x[i]), ("z", vs.z[i]), ("s", vs.s[i]), ("speed", vs.speed[i]), ("gap", vs.gap[i])] { assert!(!v.is_nan(), "{k} at slot {i}"); } } }
        env.sim.counters.spawned
    };
    let (off, on) = (run(false), run(true));
    assert!(on > 0 && off > 0);
}
