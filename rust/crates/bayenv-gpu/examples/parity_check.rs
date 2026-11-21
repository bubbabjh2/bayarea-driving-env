//! Rust host vs the JS GPUSim mock run of rust/tools/oracle/gpu_parity_mock.mjs (populations + spawn queue, refuel
//! plans, stations / pumps, arrivals + return legs; obstacles / wrecks; snapshot import). Every compared array must be
//! bit-identical.
//!   cargo run --release -p bayenv-gpu --example parity_check -- <data> <oracleDir> [native-pop]
use bayenv_gpu::layout as L;
use bayenv_gpu::sim::{GpuOptions, GpuPopulation, ObstacleKind, ObstacleSpec, Snapshot, StaticObstacle};
use std::{path::Path, sync::Arc, time::Instant};

struct Dump { meta: serde_json::Value, bin: Vec<u8> }
impl Dump {
    fn open(base: &Path) -> Dump {
        Dump { meta: serde_json::from_str(&std::fs::read_to_string(base.with_extension("json")).unwrap()).unwrap(), bin: std::fs::read(base.with_extension("bin")).unwrap() }
    }
    /// raw 32-bit words of a section (u8 sections widened)
    fn words(&self, k: &str) -> Vec<u32> {
        let s = &self.meta["sections"][k]; assert!(!s.is_null(), "no section {k}");
        let (o, n, t) = (s["byteOffset"].as_u64().unwrap() as usize, s["length"].as_u64().unwrap() as usize, s["type"].as_str().unwrap());
        match t { "Uint8Array" => self.bin[o..o + n].iter().map(|&x| x as u32).collect(), "Int8Array" => self.bin[o..o + n].iter().map(|&x| x as i8 as i32 as u32).collect(),
            "Float64Array" => bytemuck::pod_collect_to_vec(&self.bin[o..o + 8 * n]),
            _ => bytemuck::pod_collect_to_vec(&self.bin[o..o + 4 * n]) }
    }
    fn u(&self, k: &str) -> u64 { self.meta[k].as_u64().unwrap_or_else(|| panic!("meta {k}")) }
    fn f(&self, k: &str) -> f64 { self.meta[k].as_f64().unwrap_or_else(|| panic!("meta {k}")) }
}

fn cmp(name: &str, js: &[u32], rs: &[u32], bad: &mut usize) {
    let n = js.len().min(rs.len());
    let d: Vec<usize> = (0..n).filter(|&i| js[i] != rs[i]).collect();
    if js.len() != rs.len() || !d.is_empty() {
        *bad += 1;
        println!("  {name}: MISMATCH len js {} rs {}, {} differing words{}", js.len(), rs.len(), d.len(), d.first().map_or(String::new(), |&i| format!(", first [{i}] js {:#x} rs {:#x}", js[i], rs[i])));
    } else { println!("  {name}: {} words identical", js.len()); }
}
fn eq<T: PartialEq + std::fmt::Debug>(name: &str, js: T, rs: T, bad: &mut usize) { if js != rs { *bad += 1; println!("  {name}: MISMATCH js {js:?} rs {rs:?}"); } else { println!("  {name}: {rs:?}"); } }

fn opts(cap: usize, start: f64) -> GpuOptions {
    let mut o = GpuOptions { capacity: cap, start_time: start, refuel_budget_ms: f64::INFINITY, ..Default::default() };
    o.features.accidents = true; o
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (data, dir) = (Path::new(&a[1]), Path::new(&a[2]));
    let native = a.get(3).map_or(false, |s| s == "native-pop");
    let t = Instant::now();
    let world = Arc::new(bayenv_world::World::load(data).expect("world"));
    println!("world {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
    let m = Dump::open(&dir.join("mock"));
    let mut bad = 0;
    if !m.meta["refuelRequests"].is_null() {
        // ---------------------------------------------------------------- population scenario
        let (n, cap, start) = (m.u("n") as usize, m.u("cap") as usize, m.f("start"));
        let mut sim = bayenv_gpu::env::create(world.clone(), opts(cap, start)).expect("sim");
        let mut acts = vec![0f32; cap * 3]; for i in 0..cap { acts[3 * i + 1] = 0.3; } sim.set_actions(&acts, 0);
        let js_pop = GpuPopulation::load_pack(&dir.join("pop")).expect("pop pack");
        if native {
            let t = Instant::now();
            let rp = GpuPopulation::generate(&world, n, m.u("seed") as f64, true).expect("population");
            println!("native population {n}: {:.0} ms (JS popParallel {} ms)", t.elapsed().as_secs_f64() * 1e3, Dump::open(&dir.join("pop")).u("ms"));
            let b = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
            cmp("pop routeFlat", &bytemuck::cast_slice(&js_pop.route_flat).to_vec(), bytemuck::cast_slice(&rp.route_flat), &mut bad);
            cmp("pop routeOffset", &js_pop.route_offset, &rp.route_offset, &mut bad);
            cmp("pop vehicleType", &js_pop.vehicle_type.iter().map(|&x| x as u32).collect::<Vec<_>>(), &rp.vehicle_type.iter().map(|&x| x as u32).collect::<Vec<_>>(), &mut bad);
            cmp("pop depart", &b(&js_pop.depart), &b(&rp.depart), &mut bad);
            cmp("pop returnDepart", &b(&js_pop.return_depart), &b(&rp.return_depart), &mut bad);
            cmp("pop workStart", &b(&js_pop.work_start), &b(&rp.work_start), &mut bad);
            cmp("pop fuelFill", &b(js_pop.fuel_fill.as_ref().unwrap()), &b(rp.fuel_fill.as_ref().unwrap()), &mut bad);
            cmp("pop deltas", &b(js_pop.deltas.as_ref().unwrap()), &b(rp.deltas.as_ref().unwrap()), &mut bad);
        }
        let t = Instant::now();
        sim.set_population(Arc::new(js_pop), true);
        println!("set_population {:.0} ms (JS {} ms)", t.elapsed().as_secs_f64() * 1e3, m.u("setPopMs"));
        eq("nAgents", m.u("nAgents") as usize, sim.n_agents(), &mut bad);
        eq("noRoute", m.u("noRoute"), sim.counters.no_route, &mut bad);
        cmp("population route words", &m.words("routesPop"), &sim.route_mirror[..sim.route_words], &mut bad);
        cmp("trips", &m.words("trips"), &sim.x.trips.iter().map(|&x| x as u32).collect::<Vec<_>>(), &mut bad);
        cmp("tripDepart", &m.words("tripDepart"), &sim.x.trip_depart.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), &mut bad);
        // first frame: runSteps(10)'s host part
        let t = Instant::now();
        let t_end = sim.time + 10.0 * sim.dt; sim.enqueue_spawns(t_end);
        println!("enqueue_spawns {} entries {:.1} ms (JS {} ms)", sim.queue_high(), t.elapsed().as_secs_f64() * 1e3, m.u("enqueueMs"));
        sim.write_params_now(0); sim.time = t_end; sim.step += 10;
        eq("queueHigh", m.u("queueHigh") as usize, sim.queue_high(), &mut bad);
        eq("free", m.u("freeLen") as usize, sim.free_len(), &mut bad);
        let qh = sim.queue_high();
        cmp("spawn queue entries", &m.words("queue1"), &sim.read_words("queue")[..qh * L::SPAWN_WORDS], &mut bad);
        cmp("params", &m.words("params1"), &sim.params_words()[..], &mut bad);
        cmp("slotTrip", &m.words("slotTrip1"), &sim.slot_trip.iter().map(|&x| x as u32).collect::<Vec<_>>(), &mut bad);
        cmp("slotRoute", &m.words("slotRoute1"), &sim.slot_route, &mut bad);
        // refuel plans
        let req = m.words("refuelReq");
        for p in req.chunks(2) { sim.queue_refuel(p[0], p[1]); }
        let rw0 = sim.route_words; let t = Instant::now();
        sim.run_refuels();
        println!("{} refuel plans {:.0} ms (JS {} ms)", req.len() / 2, t.elapsed().as_secs_f64() * 1e3, m.u("refuelMs"));
        eq("refuelPlanned", m.u("refuelPlanned"), sim.counters.refuel_planned, &mut bad);
        eq("refuelPlanFailed", m.u("refuelPlanFailed"), sim.counters.refuel_plan_failed, &mut bad);
        cmp("refuel ops (SET_ROUTE_REFUEL)", &m.words("refuelOps"), &sim.take_ops(), &mut bad);
        cmp("refuel route blocks", &m.words("routesRefuel"), &sim.route_mirror[rw0..sim.route_words], &mut bad);
        cmp("slotStation", &m.words("slotStation2"), &sim.slot_station().iter().map(|&x| x as u32).collect::<Vec<_>>(), &mut bad);
        // stations, pumps, re-entry
        let st: Vec<[u32; 4]> = req.chunks(2).map(|p| [L::ev::STATION, p[0], 0, 0]).collect();
        sim.ingest_events(&st);
        cmp("station ops", &m.words("stationOps"), &sim.take_ops(), &mut bad);
        eq("lot", m.u("lot3") as u32, sim.lot(), &mut bad);
        let mut pops = Vec::new();
        for _ in 0..=10 { sim.run_pumps(); pops.extend(sim.take_ops()); pops.push(0xFFFF_FFFF); sim.time += 60.0; }
        cmp("pump ops (PUMPED, by frame)", &m.words("pumpOps"), &pops, &mut bad);
        { let s = sim.stations().unwrap(); eq("refuelled / maxQueue / waitN", (m.u("refuelled"), m.u("maxQueue") as usize, m.u("waitN")), (sim.counters.refuelled, s.max_queue, s.wait_n), &mut bad);
          eq("waitSum", m.f("waitSum"), s.wait_sum, &mut bad); }
        let re: Vec<[u32; 4]> = req.chunks(2).take(req.len() / 4).map(|p| [L::ev::REENTER, p[0], 0, 0]).collect();
        sim.ingest_events(&re);
        eq("lot after REENTER / reentered", (m.u("lot4") as u32, m.u("reentered")), (sim.lot(), sim.counters.reentered), &mut bad);
        // spawned, arrivals (fuel to the return leg), evening enqueue
        let sp: Vec<[u32; 4]> = (0..qh as u32).map(|q| [L::ev::SPAWN, 0, q, 0]).collect();
        sim.ingest_events(&sp);
        let arr: Vec<[u32; 4]> = m.words("arriveEv").chunks(4).map(|c| [c[0], c[1], c[2], c[3]]).collect();
        sim.ingest_events(&arr);
        sim.time = 18.0 * 3600.0; let t_end = sim.time + 0.5; sim.enqueue_spawns(t_end);
        eq("queueHigh / tripPtr / free", (m.u("queueHigh5") as usize, m.u("tripPtr5") as usize, m.u("freeLen5") as usize), (sim.queue_high(), sim.x.trip_ptr, sim.free_len()), &mut bad);
        cmp("evening spawn queue", &m.words("queue5"), &sim.read_words("queue")[..sim.queue_high() * L::SPAWN_WORDS], &mut bad);
        cmp("fuelLeft", &m.words("fuelLeft5"), &sim.x.fuel_left.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), &mut bad);
        cmp("outcome", &m.words("outcome5"), &sim.x.outcome.iter().map(|&x| x as u32).collect::<Vec<_>>(), &mut bad);
        cmp("slotTrip", &m.words("slotTrip5"), &sim.slot_trip.iter().map(|&x| x as u32).collect::<Vec<_>>(), &mut bad);
    } else if !m.meta["placed"].is_null() {
        // ---------------------------------------------------------------- obstacles / wrecks
        let cap = m.u("cap") as usize;
        let mut o = opts(cap, 8.0 * 3600.0); o.seed = 5;
        let mut sim = bayenv_gpu::env::create(world.clone(), o).expect("sim");
        let t = Instant::now();
        let placed = sim.place_obstacles(&ObstacleSpec { count: m.u("count") as usize, seed: m.f("oseed"), ..Default::default() });
        println!("place_obstacles {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
        let w = sim.place_wrecks(m.u("nw") as usize);
        let list: Vec<StaticObstacle> = m.meta["list"].as_array().unwrap().iter().map(|o| StaticObstacle { edge: o["edge"].as_i64().unwrap(), at: o["at"].as_f64().unwrap(),
            lane: o["lane"].as_f64(), lat: o["lat"].as_f64(), ty: o["type"].as_u64().unwrap_or(0) as u32 }).collect();
        let placed2 = sim.place_static_obstacles(&list);
        let f64s = |k: &str| -> Vec<u64> { m.words(k).chunks(2).map(|c| c[0] as u64 | (c[1] as u64) << 32).collect() };
        let rows = |v: &serde_json::Value, sk: &str| -> Vec<(u32, String, u32, u64)> { let ss = f64s(sk); v.as_array().unwrap().iter().zip(ss).map(|(p, s)| (p[0].as_u64().unwrap() as u32, p[1].as_str().unwrap().to_string(), p[2].as_u64().unwrap() as u32, s)).collect() };
        let mine = |v: &[bayenv_gpu::sim::Placed]| -> Vec<(u32, String, u32, u64)> { v.iter().map(|p| (p.slot, p.kind.name().to_string(), p.edge, p.s.to_bits())).collect() };
        eq("placeObstacles(spec)", rows(&m.meta["placed"], "placedS"), mine(&placed), &mut bad);
        eq("placeWrecks", m.u("wrecks") as usize, w, &mut bad);
        eq("placeObstacles(list)", rows(&m.meta["placed2"], "placed2S"), mine(&placed2), &mut bad);
        let kinds: std::collections::BTreeMap<&str, usize> = placed.iter().fold(Default::default(), |mut a, p| { *a.entry(p.kind.name()).or_default() += 1; a });
        println!("  kinds {kinds:?}"); let _ = ObstacleKind::Static;
        cmp("veh", &m.words("veh"), &sim.read_words("veh"), &mut bad);
        cmp("par", &m.words("par"), &sim.read_words("par"), &mut bad);
        cmp("ctl (slot records)", &m.words("ctl"), &sim.read_words("ctl")[..cap * L::CTL_WORDS], &mut bad);
        let blk = sim.blob.off.blk as usize;
        cmp("scene blocks", &m.words("blocks"), &sim.read_words("world")[blk..blk + 256 * L::BLOCK_WORDS], &mut bad);
        cmp("edge heads", &m.words("edgeHead"), sim.edge_head(), &mut bad);
        cmp("routes", &m.words("routes"), &sim.route_mirror[..sim.route_words], &mut bad);
        cmp("slotTrip", &m.words("slotTrip"), &sim.slot_trip.iter().map(|&x| x as u32).collect::<Vec<_>>(), &mut bad);
        let sf = f64s("sceneF");
        let js_sc: Vec<(u32, u32, u64, i64, u64, String)> = m.meta["scene0"].as_array().unwrap().iter().enumerate().map(|(k, r)| (r[0].as_u64().unwrap() as u32, r[1].as_u64().unwrap() as u32, sf[2 * k], r[3].as_i64().unwrap(), sf[2 * k + 1], r[5].as_str().unwrap().to_string())).collect();
        let rs_sc: Vec<(u32, u32, u64, i64, u64, String)> = sim.scene_rows().into_iter().take(50).map(|r| (r.0, r.1, r.2.to_bits(), r.3 as i64, r.4.to_bits(), r.5.to_string())).collect();
        eq("scenes (edge, laneMask, s, block, squeezeAt, cause)", js_sc.len(), rs_sc.len(), &mut bad);
        if js_sc != rs_sc { bad += 1; println!("  scene records differ: js {:?} rs {:?}", js_sc.first(), rs_sc.first()); } else { println!("  scene records identical"); }
    } else {
        // ---------------------------------------------------------------- snapshot import
        let cap = m.u("cap") as usize;
        let mut sim = bayenv_gpu::env::create(world.clone(), opts(cap, 5.0 * 3600.0)).expect("sim");
        let snap = Snapshot::load(&dir.join("snap")).expect("snapshot");
        let t = Instant::now(); sim.import_snapshot(&snap).expect("import");
        println!("import_snapshot (high {}) {:.0} ms", snap.high, t.elapsed().as_secs_f64() * 1e3);
        eq("time / step / dt", (m.f("time").to_bits(), m.u("step"), m.f("dt").to_bits()), (sim.time.to_bits(), sim.step, sim.dt.to_bits()), &mut bad);
        eq("free", m.u("free") as usize, sim.free_len(), &mut bad);
        cmp("veh", &m.words("veh"), &sim.read_words("veh"), &mut bad);
        cmp("par", &m.words("par"), &sim.read_words("par"), &mut bad);
        cmp("ring", &m.words("ring"), &sim.read_words("ring"), &mut bad);
        cmp("pose", &m.words("pose"), &sim.read_words("pose"), &mut bad);
        cmp("routes", &m.words("routes"), &sim.route_mirror[..sim.route_words], &mut bad);
        cmp("slotTrip", &m.words("slotTrip"), &sim.slot_trip.iter().map(|&x| x as u32).collect::<Vec<_>>(), &mut bad);
    }
    println!("{}", if bad == 0 { "ALL IDENTICAL".to_string() } else { format!("{bad} MISMATCHES") });
    std::process::exit(if bad == 0 { 0 } else { 1 });
}
