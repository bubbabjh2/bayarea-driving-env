//! Differential harness against JS traces (tools/rust/trace.mjs): builds the same env from the exported world /
//! population / run.json, drives it with the same deterministic policy and compares every VehicleStore field.
use crate::env::{Env, EnvConfig, StaticObstacle};
use crate::pack::Pack;
use crate::population::Population;
use crate::rng::Rng;
use crate::sim::{Action, SimOpts, TrafficMode};
use crate::vehicle::{js_name, STATE_DRIVING};
use crate::world::World;
use serde_json::Value;
use std::sync::Arc;

pub fn sim_opts_from(v: &Value) -> SimOpts {
    let mut o = SimOpts::default();
    let n = |k: &str, d: f64| -> f64 { match &v[k] { Value::Number(x) => x.as_f64().unwrap(), Value::String(s) if s == "Infinity" => f64::INFINITY, Value::Bool(b) => if *b { 1.0 } else { 0.0 }, _ => d } };
    o.dt = n("dt", o.dt); o.start_time = n("startTime", o.start_time); o.learner_fraction = n("learnerFraction", o.learner_fraction);
    o.shoulder = n("shoulder", o.shoulder); o.freeway_shoulder = n("freewayShoulder", o.freeway_shoulder); o.opposite_margin = n("oppositeMargin", o.opposite_margin);
    o.touch_kmh = n("touchKmh", o.touch_kmh); o.layer_gap = n("layerGap", o.layer_gap); o.stuck_timeout = n("stuckTimeout", o.stuck_timeout);
    o.pass_wrecks = n("passWrecks", 0.0) != 0.0; o.spawn_blocked_timeout = n("spawnBlockedTimeout", o.spawn_blocked_timeout);
    o.cell_size = n("cellSize", o.cell_size); o.table_bits = n("tableBits", o.table_bits as f64) as u32;
    o.leader_lookahead = n("leaderLookahead", o.leader_lookahead); o.merge_lookahead = n("mergeLookahead", o.merge_lookahead);
    o.lane_end_lookahead = n("laneEndLookahead", o.lane_end_lookahead); o.control_lookahead = n("controlLookahead", o.control_lookahead);
    o.reroute = n("reroute", 1.0) != 0.0; o.reroute_sight = n("rerouteSight", o.reroute_sight); o.squeeze_delay = n("squeezeDelay", o.squeeze_delay);
    o.squeeze_speed = n("squeezeSpeed", o.squeeze_speed); o.density_interval = n("densityInterval", o.density_interval);
    o.path_sight = n("pathSight", 0.0); o.deadlock_break = n("deadlockBreak", 0.0); o.spawn_lane_route = n("spawnLaneRoute", 0.0); o.junction_sight = n("junctionSight", 0.0);
    o.position_mode = v["actionMode"].as_str() != Some("rate");
    o.traffic_mode = match v["trafficMode"].as_str() { Some("sparse") => TrafficMode::Sparse, Some("empty") => TrafficMode::Empty, _ => TrafficMode::Full };
    o.fuel = n("fuel", 1.0) != 0.0; o.seed = n("seed", 1.0); o.wrecks = n("wrecks", 0.0) as usize; o.slot_pool = n("slotPool", 0.0) as usize;
    o
}

/// The JS policies of tools/rust/trace.mjs makePolicy.
pub struct Policy { name: String, r: Rng }
impl Policy {
    pub fn new(name: &str, seed: f64) -> Policy { Policy { name: name.to_string(), r: Rng::from_u32(crate::jm::to_u32(seed * 7919.0 + 13.0)) } }
    pub fn act(&mut self, env: &mut Env) {
        if self.name == "idle" { return; }
        let g = env.world.clone();
        let clamp = |x: f64| if x > 1.0 { 1.0 } else if x < -1.0 { -1.0 } else { x };
        for i in 0..env.sim.vs.high {
            if env.sim.vs.state[i] != STATE_DRIVING { continue; }
            let a = match self.name.as_str() {
                "const" => Action { steer: Some(0.0), steer_rate: None, gas: 0.3, brake: 0.0 },
                "random" => { let a = self.r.random(); let b = self.r.random(); let c = self.r.random(); Action { steer: Some(a * 0.2 - 0.1), steer_rate: None, gas: b, brake: c * 0.2 } }
                _ => {
                    let vs = &env.sim.vs;
                    let e = vs.edge[i] as usize;
                    let lw = g.lane_w(e, vs.s[i] as f64);
                    let off = vs.lat[i] as f64 - (vs.lane[i] as f64 + 0.5) * lw;
                    let mut steer = clamp((-0.08 * off - 1.2 * vs.heading_rel[i] as f64) / vs.max_steer[i] as f64);
                    let mut gas = if (vs.speed[i] as f64) < 0.8 * g.e_speed[e] { 0.4 } else { 0.0 };
                    let mut brake = if vs.gap[i] < 15.0 { 0.6 } else { 0.0 };
                    if self.name == "cruisenoise" { let a = self.r.random(); let b = self.r.random(); steer = clamp(steer + (a - 0.5) * 0.1); if b < 0.05 { gas = 1.0; brake = 0.0; } else if b < 0.08 { brake = 0.0; } }
                    Action { steer: Some(steer), steer_rate: None, gas, brake }
                }
            };
            env.act(i, a);
        }
    }
}

pub fn obstacles_from(v: &Value) -> Vec<StaticObstacle> {
    v.as_array().map(|a| a.iter().map(|o| StaticObstacle { edge: o["edge"].as_i64().unwrap() as i32, at: o["at"].as_f64().unwrap_or(0.0),
        lane: o["lane"].as_i64().map(|x| x as i32), lat: o["lat"].as_f64(), type_idx: o["type"].as_u64().unwrap_or(0) as u8 }).collect()).unwrap_or_default()
}

pub fn env_from_dir(dir: &str) -> (Env, Value) {
    let run: Value = serde_json::from_str(&std::fs::read_to_string(format!("{dir}/run.json")).unwrap()).unwrap();
    let world = Arc::new(World::load(&format!("{dir}/world.pack")));
    let pop = Arc::new(Population::load(&format!("{dir}/pop.pack")));
    let mut opts = sim_opts_from(&run["simOpts"]);
    let da = &run["defaultAction"];
    opts.default_action = Action { steer: da["steer"].as_f64(), steer_rate: da["steerRate"].as_f64(), gas: da["gas"].as_f64().unwrap_or(0.0), brake: da["brake"].as_f64().unwrap_or(0.3) };
    let cfg = EnvConfig { start_time: Some(run["startTime"].as_f64().unwrap()), static_obstacles: obstacles_from(&run["staticObstacles"]),
        scenario_obstacles: obstacles_from(&run["obstacles"]), opts };
    let env = Env::new(world, pop, cfg, Box::new(|w| Box::new(crate::route_stub::AStar::new(w, 1.25))));
    (env, run)
}

#[derive(Debug)]
pub struct Mismatch { pub step: usize, pub field: String, pub slot: usize, pub js: f64, pub rust: f64 }

fn val(dtype: &str, b: &[u8]) -> f64 {
    match dtype { "f32" => f32::from_le_bytes(b.try_into().unwrap()) as f64, "f64" => f64::from_le_bytes(b.try_into().unwrap()), "i32" => i32::from_le_bytes(b.try_into().unwrap()) as f64,
        "u32" => u32::from_le_bytes(b.try_into().unwrap()) as f64, "i8" => b[0] as i8 as f64, "u8" => b[0] as f64, _ => 0.0 }
}

/// Run the trace in `dir`; returns (steps checked, first mismatches (up to `max`)) and the per-step wall time.
pub fn check(dir: &str, max: usize, verbose: bool) -> (usize, Vec<Mismatch>, f64) {
    let (mut env, run) = env_from_dir(dir);
    let tr = Pack::read(&format!("{dir}/trace.pack")).unwrap();
    let every = run["every"].as_u64().unwrap_or(1) as usize;
    let steps = run["steps"].as_u64().unwrap() as usize;
    let fields: Vec<String> = tr.meta["fields"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
    let highs = tr.u32s("high");
    let mut pol = Policy::new(run["policy"].as_str().unwrap(), run["seed"].as_f64().unwrap());
    let mut offs: std::collections::HashMap<String, usize> = Default::default();
    let mut bad = Vec::new();
    let mut rec = 0usize;
    let t0 = std::time::Instant::now();
    let mut sim_time = 0.0;
    for k in 1..=steps {
        pol.act(&mut env);
        let ts = std::time::Instant::now();
        env.step(1);
        sim_time += ts.elapsed().as_secs_f64();
        if k % every != 0 { continue; }
        let h = highs[rec] as usize;
        rec += 1;
        let mine: std::collections::HashMap<String, (&'static str, usize, Vec<u8>)> = env.sim.vs.fields().into_iter().map(|(n, d, s, b)| (js_name(n), (d, s, b))).collect();
        if env.sim.vs.high != h { bad.push(Mismatch { step: k, field: "high".into(), slot: 0, js: h as f64, rust: env.sim.vs.high as f64 }); }
        for fname in &fields {
            let sec = &tr.sections[&format!("vs.{fname}")];
            let dt = match sec.dtype { crate::pack::DType::F32 => "f32", crate::pack::DType::F64 => "f64", crate::pack::DType::I32 => "i32", crate::pack::DType::U32 => "u32", crate::pack::DType::I8 => "i8", crate::pack::DType::U8 => "u8", _ => "?" };
            let sz = sec.dtype.size();
            let stride = tr.meta["strides"][fname].as_u64().unwrap() as usize;
            let off = offs.entry(fname.clone()).or_insert(0);
            let js = &sec.bytes[*off * sz..(*off + h * stride) * sz];
            *off += h * stride;
            let Some((_, rs, rb)) = mine.get(fname) else { if k == every && verbose { eprintln!("field {fname} not in the Rust store"); } continue };
            assert_eq!(*rs, stride, "{fname} stride");
            let n = (h * stride).min(rb.len() / sz);
            for q in 0..n {
                let (a, b) = (&js[q * sz..(q + 1) * sz], &rb[q * sz..(q + 1) * sz]);
                if a != b {
                    let (x, y) = (val(dt, a), val(dt, b));
                    if x.is_nan() && y.is_nan() { continue; }
                    bad.push(Mismatch { step: k, field: fname.clone(), slot: q / stride, js: x, rust: y });
                    if bad.len() >= max { return (k, bad, sim_time); }
                }
            }
        }
        if !bad.is_empty() { return (k, bad, sim_time); }
    }
    if verbose { eprintln!("wall {:.2}s (sim {:.3}s) for {steps} steps", t0.elapsed().as_secs_f64(), sim_time); }
    (steps, bad, sim_time)
}
