//! Rust side of rust/tools/oracle/gpu_parity_run.html: commuter population (JS pop pack, or generated natively) and/or
//! training obstacles, constant gas, `steps` steps in batches; writes veh + pose (compare_state format) and a JSON of
//! host counters / slot tables.
//!   cargo run --release -p bayenv-gpu --example parity_run -- <data> <oracleDir> <steps> <batch> <out-name>
//!       [cap=32768] [start=25200] [pop=1|native|0] [obstacles=0] [oseed=3]
use bayenv_gpu::layout as L;
use bayenv_gpu::sim::{GpuOptions, GpuPopulation, ObstacleSpec};
use std::{path::Path, sync::Arc, time::Instant};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (data, dir) = (Path::new(&a[1]), Path::new(&a[2]));
    let p = |i: usize, d: &str| a.get(i).cloned().unwrap_or(d.to_string());
    let (steps, batch, name): (usize, usize, String) = (p(3, "4000").parse().unwrap(), p(4, "10").parse().unwrap(), p(5, "rs"));
    let (cap, start): (usize, f64) = (p(6, "32768").parse().unwrap(), p(7, "25200").parse().unwrap());
    let (popm, obst, oseed): (String, usize, f64) = (p(8, "1"), p(9, "0").parse().unwrap(), p(10, "3").parse().unwrap());
    let world = Arc::new(bayenv_world::World::load(data).expect("world"));
    let mut o = GpuOptions { capacity: cap, start_time: start, refuel_budget_ms: f64::INFINITY, deterministic_contacts: false, deterministic_spawn: false, deterministic_events: false, deterministic_leaders: false, ..GpuOptions::default().js_model() };   // JS oracle: legacy model
    o.features.accidents = true;
    o.apply_env();   // BAYENV_GPU_OPTS (e.g. cpu_connector_join=1)
    let mut sim = bayenv_gpu::env::create(world.clone(), o).expect("sim");
    let mut acts = vec![0f32; cap * 3]; for i in 0..cap { acts[3 * i + 1] = 0.3; } sim.set_actions(&acts, 0);
    if popm != "0" {
        let pop = if popm == "native" {
            let m: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("pop.json")).unwrap()).unwrap();
            let t = Instant::now(); let r = GpuPopulation::generate(&world, m["n"].as_u64().unwrap() as usize, m["seed"].as_f64().unwrap(), true).unwrap();
            println!("native population: {:.0} ms", t.elapsed().as_secs_f64() * 1e3); r
        } else { GpuPopulation::load_pack(&dir.join("pop")).expect("pop pack") };
        sim.set_population(Arc::new(pop), true);
    }
    if obst > 0 { let pl = sim.place_obstacles(&ObstacleSpec { count: obst, seed: oseed, ..Default::default() }); println!("obstacles {}", pl.len()); }
    let t = Instant::now();
    let mut done = 0; while done < steps { let k = batch.min(steps - done); sim.run_steps(k); done += k; }
    let ms = t.elapsed().as_secs_f64() * 1e3;
    let st = sim.read_state();
    let mut b = Vec::new(); b.extend_from_slice(bytemuck::cast_slice(&st.veh)); b.extend_from_slice(bytemuck::cast_slice(&st.pose));
    std::fs::write(dir.join(format!("{name}.bin")), b).unwrap();
    let ag = sim.read_agents();
    let s = sim.stats(); let c = &sim.counters;
    let j = serde_json::json!({ "msPerStep": ms / steps as f64, "stats": {
        "driving": s[L::stat::LIVE_DRIVING], "spawned": s[L::stat::SPAWNED], "arrived": s[L::stat::ARRIVED], "crashes": c.crash_events, "crashScenes": c.crashes, "activeCrashes": sim.active.len(),
        "stranded": c.stranded, "refuelPlanned": c.refuel_planned, "refuelPlanFailed": c.refuel_plan_failed, "refuelled": c.refuelled, "reentered": c.reentered, "minorContacts": c.minor_contacts,
        "missedExits": c.missed_exits, "reroutes": c.reroutes, "counterSpawned": c.spawned, "noRoute": c.no_route, "lot": sim.lot(), "queueHigh": sim.queue_high(), "tripPtr": sim.x.trip_ptr,
        "free": sim.free_len(), "time": sim.time, "step": sim.step },
        "slotTrip": sim.slot_trip, "slotStation": sim.slot_station(), "outcome": sim.x.outcome, "fuelLeft": sim.x.fuel_left, "agentOutcome": &ag.outcome[..64.min(ag.outcome.len())] });
    std::fs::write(dir.join(format!("{name}.json")), j.to_string()).unwrap();
    println!("{steps} steps {:.3} ms/step: {}", ms / steps as f64, j["stats"]);
}
