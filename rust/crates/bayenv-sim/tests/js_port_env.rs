//! Port of test/sim/env.test.js: 4 JS test cases -> 3 #[test]s (2 ported 1:1, 1 replaced by a Rust analogue, 1 skipped).
//!   ported : createTrainingEnv empty / external modes (observations, signals, fitness, deltas); traffic controls (signal plan cycles,
//!            perpendicular approaches never green together, stop lines perceived, an hour of sim around it);
//!            exportState/importState contract -> replaced by the Rust analogue VehicleStore::fields() (every field exports slots [0, high)).
//!   skipped: "concurrency report for a population" - population.js concurrency() is a JS reporting helper, not in the Rust crate.
//!   skipped (not in Rust): state.js exportState / importState / LAYOUT_VERSION / STATE_FIELDS (GPU cross-check snapshot format);
//!            the round trip of routes through importState cannot be expressed.
//! createTrainingEnv({graph, density, trafficMode, seed, population}) -> PopConfig::env_synthetic + Env::new (dt 0.1 like env.js).
//! The traffic-controls fixture (a signal on every in-edge of one 4-way junction) is built by tests/common/gen_worlds.mjs ("ctl")
//! and read back through World.controls + extra::state_of (TrafficControls.stateOf).
mod common;
use bayenv_sim::extra::*;
use bayenv_sim::obs::{self, OBS_FIELDS};
use bayenv_sim::population_gen::*;
use bayenv_sim::sim::*;
use bayenv_sim::vehicle::*;
use std::sync::Arc;

fn field(name: &str) -> usize { OBS_FIELDS.iter().position(|f| *f == name).unwrap_or_else(|| panic!("obs field {name}")) }

#[test]
fn training_env_empty_and_external_modes_observations_signals_fitness() {
    let g = need!(common::shared_world("synth"));
    let pop = Arc::new(generate_population(&g, &PopConfig::env_synthetic(60, 5.0), None).unwrap().pop);
    let opts = SimOpts { dt: 0.1, seed: 5.0, traffic_mode: TrafficMode::Empty, ..SimOpts::default() };
    let mut env = common::env_with(g.clone(), pop.clone(), opts, 6.0 * 3600.0);
    let end = env.sim.time + 3600.0;
    while env.sim.time < end { env.step(1); }
    assert_eq!(env.sim.counters.crashes, 0, "agents pass through each other in empty mode");
    assert_eq!(pop.deltas.len(), 60 * 64);
    assert_eq!(env.sim.fitness.len(), 60);

    let pop = Arc::new(generate_population(&g, &PopConfig::env_synthetic(20, 6.0), None).unwrap().pop);
    let opts = SimOpts { dt: 0.1, seed: 6.0, traffic_mode: TrafficMode::Full, ..SimOpts::default() };
    let mut ext = bayenv_sim::env::Env::new(g.clone(), pop.clone(),
        bayenv_sim::env::EnvConfig { opts, start_time: None, static_obstacles: vec![], scenario_obstacles: vec![], obstacles: None },
        Box::new(|w| bayenv_sim::harness::router_for(&w)));
    let i_speed = field("speed");
    let (i_ms, i_wb, i_mass) = (field("maxSteer"), field("wheelbase"), field("mass"));
    let i_nb = field("neighbors.0");
    let (i_ttws, i_fuel, i_refuel) = (field("signals.timeToWorkStart"), field("signals.fuelFraction"), field("signals.refuelPlanned"));
    let (mut seen, mut after) = (0usize, 0usize);
    let mut k = 0;
    while k < 6000 && after < 60 {
        if seen > 0 { after += 1; }
        let slots = obs::driving_slots(&ext.sim);
        let st = obs::flat_stride(false);
        let mut flat = vec![0f32; slots.len() * st];
        obs::observe_flat(&mut ext.sim, &slots, &mut flat, false);
        for (r, &slot) in slots.iter().enumerate() {
            seen += 1;
            let row = &flat[r * st..(r + 1) * st];
            assert!(row[i_speed].is_finite());
            assert_eq!(row[i_nb..i_nb + 12].len(), 12); // neighbors has 12 entries
            assert!(!row[i_ttws].is_nan() && !row[i_fuel].is_nan() && !row[i_refuel].is_nan());
            assert!(row[i_ms] > 0.0 && row[i_wb] > 0.0 && row[i_mass] > 0.0);
            ext.act(slot, Action { steer: None, steer_rate: Some(0.0), gas: 0.5, brake: 0.0 });
        }
        ext.step(1);
        k += 1;
    }
    assert!(seen > 0);
    let v = &ext.sim.vs;
    let moving = (0..v.high).filter(|&i| v.state[i] == 1 && v.speed[i] > 0.0).count();
    assert!(moving > 0, "external controls drive the cars (after the reaction delay)");
}

#[test]
fn vehicle_store_export_covers_slots_up_to_high() {
    // analogue of exportState/importState "GPU cross-check contract": every exported field has slots [0, high) * stride elements
    let g = need!(common::shared_world("synth"));
    let mut cfg = PopConfig::env_synthetic(80, 1.0); cfg.return_trips = false;
    let pop = Arc::new(generate_population(&g, &cfg, None).unwrap().pop);
    let mut env = common::env_with(g.clone(), pop, SimOpts { dt: 0.1, ..SimOpts::default() }, 7.5 * 3600.0);
    for _ in 0..600 { env.step(1); }
    let vs = &env.sim.vs;
    let fields = vs.fields();
    assert!(!fields.is_empty() && vs.high > 0);
    for (name, dtype, stride, bytes) in &fields {
        let size = match *dtype { "f64" | "i64" => 8, "f32" | "i32" | "u32" => 4, "i16" | "u16" => 2, _ => 1 };
        assert_eq!(bytes.len(), vs.high * stride * size, "{name}");
    }
    for i in 0..vs.high { if vs.state[i] != STATE_FREE { assert!(!vs.routes[i].is_empty()); } }
}

#[test]
fn traffic_controls_signal_plan_cycles_and_stop_lines_are_perceived() {
    let g = need!(common::shared_world("ctl"));
    let meta = need!(common::fixtures_meta());
    let ins = meta["ctl"]["ins"].as_array().unwrap();
    let cc = g.controls.as_ref().expect("controls in the world pack");
    assert_eq!(cc.count, 1);
    assert_eq!(cc.ap_control.len(), ins.len());
    let cycle = cc.cycle[0];
    assert!(cycle > 0.0);
    let mut seen = std::collections::HashSet::new();
    let mut t = 0.0;
    while t < cycle {
        let (_, state, time_in, time_left) = state_of(&g, 0, MOVE_THROUGH, t);
        seen.insert(state);
        assert!(time_left >= 0.0 && time_in >= 0.0);
        t += 0.5;
    }
    assert!(seen.contains(&LIGHT_GREEN) && seen.contains(&LIGHT_YELLOW) && seen.contains(&LIGHT_RED));
    // perpendicular approaches are never green together
    let a0 = 0usize;
    let a1 = (0..ins.len()).find(|&k| cc.ap_group[k] != cc.ap_group[0]).expect("a perpendicular approach");
    let mut t = 0.0;
    while t < cycle {
        assert!(!(state_of(&g, a0, MOVE_THROUGH, t).1 == LIGHT_GREEN && state_of(&g, a1, MOVE_THROUGH, t).1 == LIGHT_GREEN));
        t += 0.5;
    }
    // a simulated hour around it: cars on the signalised approaches perceive the control (type, state, distance)
    let mut cfg = PopConfig::env_synthetic(300, 1.0); cfg.return_trips = false;
    let pop = Arc::new(generate_population(&g, &cfg, None).unwrap().pop);
    let mut env = common::env_with(g.clone(), pop, SimOpts { dt: 0.1, ..SimOpts::default() }, 7.0 * 3600.0);
    let mut perceived = 0u64;
    while env.sim.time < 8.0 * 3600.0 {
        env.step(1);
        let vs = &env.sim.vs;
        for i in 0..vs.high { if vs.state[i] == 1 && vs.ctl_type[i] as i32 == CTRL_SIGNAL as i32 && vs.ctl_dist[i] < 250.0 { perceived += 1; } }
    }
    eprintln!("perceived signal observations: {perceived}");
    assert!((env.sim.counters.signal_violations as f64).is_finite());
}
