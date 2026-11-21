//! Streaming route prefetch (RouteProvider workers) vs serial routing at departure.
//!
//!   cargo run --release -p bayenv-gpu --example stream_prefetch -- <data> route [n=1380000] [threads=cores] [from_h=5] [to_h=10] [modes=0,threads (threads[:cache],...)]
//!     routes every departure of [from_h - 1 h, to_h) (the window setStreaming takes from a from_h start) serially,
//!     then with `threads` prefetch workers; checks that both give the same route for every commuter.
//!   cargo run --release -p bayenv-gpu --example stream_prefetch -- <data> sim [n] [threads] [start_h=5] [hours=1] [steps=20] [capacity] [cache=0] [chunk_h=1]
//!     GPU sim with streaming on (threads = 0: serial routes at departure); per simulated hour: wall time, cars,
//!     route stats; the spawn hash (every placed departure: commuter, step, slot, queue entry, route) at the end.
use bayenv_gpu::sim::{CommuterTable, GpuOptions, RouteProvider};
use std::{path::Path, sync::Arc, time::Instant};

fn fnv(h: &mut u64, v: u32) { for b in v.to_le_bytes() { *h ^= b as u64; *h = h.wrapping_mul(0x100000001b3); } }
fn arg<T: std::str::FromStr>(a: &[String], i: usize, d: T) -> T { a.get(i).and_then(|s| s.parse().ok()).unwrap_or(d) }
fn cores() -> usize { std::thread::available_parallelism().map_or(1, |n| n.get()) }
fn load() -> String { std::process::Command::new("sysctl").args(["-n", "vm.loadavg"]).output().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default() }

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = Path::new(&a[1]);
    let mode = a[2].as_str();
    let n: usize = arg(&a, 3, 1_380_000);
    let threads: usize = arg(&a, 4, cores());
    let t = Instant::now();
    let world = Arc::new(bayenv_world::World::load(data).expect("world"));
    println!("world {:.0} ms, load {}", t.elapsed().as_secs_f64() * 1e3, load());
    let t = Instant::now();
    // BAYENV_TABLE_CACHE=<file>: load the table if the file exists, else generate and save it
    let cache_file = std::env::var("BAYENV_TABLE_CACHE").ok().map(std::path::PathBuf::from);
    let table = match &cache_file {
        Some(f) if f.exists() => CommuterTable::load(f).expect("table cache"),
        _ => { let t = CommuterTable::generate(&world, n, 42.0).expect("table"); if let Some(f) = &cache_file { t.save(f).expect("save table"); } t }
    };
    assert_eq!(table.n, n, "table cache size");
    println!("CommuterTable {n}: {:.1} s (generation routes once, parallel), load {}", t.elapsed().as_secs_f64(), load());
    match mode {
        "route" => {
            let (h0, h1): (f64, f64) = (arg(&a, 5, 5.0), arg(&a, 6, 10.0));
            let (k0, k1) = (table.index_at(h0 * 3600.0 - 3600.0), table.index_at(h1 * 3600.0));
            let ids: Vec<usize> = table.order[k0..k1].iter().map(|&i| i as usize).collect();
            println!("departures {}..{} h: {} commuters", h0 - 1.0, h1, ids.len());
            let run = |threads: usize, cache: usize| {
                let mut p = RouteProvider::new(&world, threads, cache);
                let t = Instant::now();
                let mut h = 0xcbf29ce484222325u64; let mut none = 0;
                if threads > 0 {
                    // the sim's pattern without the GPU: the clock moves through the departures in 1 s frames,
                    // prefetching up to 300 s ahead; each departure takes its route when the clock reaches it
                    let (mut pf, mut k) = (0usize, 0usize);
                    let mut clock = table.sorted_depart[k0] as f64;
                    while k < ids.len() {
                        while pf < ids.len() && table.sorted_depart[k0 + pf] as f64 <= clock + 300.0 { let i = ids[pf]; if !p.prefetch(table.home[i], table.work[i]) { break; } pf += 1; }
                        p.flush();
                        while k < ids.len() && table.sorted_depart[k0 + k] as f64 <= clock + 2.0 {
                            let i = ids[k]; match p.get(table.home[i], table.work[i]) { Some(r) => { fnv(&mut h, r.len() as u32); for &e in r.iter() { fnv(&mut h, e as u32); } } None => { none += 1; fnv(&mut h, u32::MAX); } }
                            k += 1;
                        }
                        clock += 1.0;
                    }
                } else {
                    for &i in &ids { match p.get(table.home[i], table.work[i]) { Some(r) => { fnv(&mut h, r.len() as u32); for &e in r.iter() { fnv(&mut h, e as u32); } } None => { none += 1; fnv(&mut h, u32::MAX); } } }
                }
                let s = t.elapsed().as_secs_f64();
                let st = p.stats();
                let computed = st.misses + st.prefetched;
                println!("threads {threads:>2}: {:.1} s, {:.0} departures/s, {} A* ({:.0} routes/s), cache {cache}: OD hit rate {:.1} % ({} served from the cache, {} waited for a worker, {:.1} s waiting), {} no route, cache {} entries / {:.0} MB of edges, sim-thread bookkeeping + locks {:.2} s, worker A* {:.2} ms/route, sim-thread A* {:.2} ms/route, load {}, route hash {h:016x}",
                    s, ids.len() as f64 / s, computed, computed as f64 / s, 100.0 * (1.0 - computed as f64 / st.lookups as f64), st.cache_hits, st.waited, st.wait_ns as f64 * 1e-9, none,
                    st.cache_len, st.cache_edges as f64 * 4.0 / 1048576.0, (st.lock_ns - st.wait_ns) as f64 * 1e-9, st.worker_ns as f64 * 1e-6 / st.prefetched.max(1) as f64, st.route_ns as f64 * 1e-6 / st.misses.max(1) as f64, load());
                h
            };
            // modes: threads[:cache entries], comma-separated
            let modes: Vec<(usize, usize)> = a.get(7).map_or(vec![(0, 0), (threads, 0)], |s| s.split(',').map(|x| { let mut p = x.split(':'); (p.next().unwrap().parse().unwrap(), p.next().map_or(0, |c| c.parse().unwrap())) }).collect());
            let hs: Vec<u64> = modes.iter().map(|&(t, c)| run(t, c)).collect();
            println!("routes identical across {:?}: {}", modes, hs.windows(2).all(|w| w[0] == w[1]));
        }
        "sim" => {
            let (h0, hours, steps): (f64, f64, usize) = (arg(&a, 5, 5.0), arg(&a, 6, 1.0), arg(&a, 7, 20));
            let cap: usize = arg(&a, 8, n.min(131072).div_ceil(1024) * 1024);
            let cache: usize = arg(&a, 9, 0);
            let chunk_h: f64 = arg(&a, 10, 1.0);
            let o = GpuOptions { capacity: cap, start_time: h0 * 3600.0, refuel_budget_ms: f64::INFINITY, route_threads: threads, route_cache: cache, ..Default::default() };
            let mut sim = bayenv_gpu::env::create(world.clone(), o).expect("sim");
            sim.set_streaming(Arc::new(table), true, None);
            // reroute ODs (diagnostic for an OD cache on reroutes): plain = cacheable by OD; others keyed by the
            // 60 s traffic-penalty epoch
            let mut seen_q: std::collections::HashSet<(u32, u64)> = Default::default();
            let (mut rr_n, mut rr_plain) = (0u64, 0u64);
            let mut rr_keys: std::collections::HashSet<(u32, u32, i64)> = Default::default();
            let mut rr_plain_keys: std::collections::HashSet<(u32, u32)> = Default::default();
            println!("sim: capacity {cap}, start {h0} h, {hours} h, {steps} steps per call, route threads {threads}");
            let per_hour = (chunk_h * 3600.0 / sim.dt).round() as usize / steps;
            let total = Instant::now();
            let total_calls = (hours / chunk_h * per_hour as f64).round() as usize;
            for hr in 0..total_calls.div_ceil(per_hour) {
                let t = Instant::now(); let s0 = sim.route_stats().unwrap();
                for _ in 0..per_hour.min(total_calls - hr * per_hour) {
                    sim.sync();
                    for (f, g, plain, slot, tb) in sim.reroute_ods() {
                        if !seen_q.insert((slot, tb)) { continue; }
                        rr_n += 1;
                        if plain { rr_plain += 1; rr_plain_keys.insert((f, g)); } else { rr_keys.insert((f, g, (sim.time / 60.0).floor() as i64)); }
                    }
                    sim.run_steps(steps);
                }
                sim.sync();
                let s1 = sim.route_stats().unwrap(); let ss = sim.x.stream_stats;
                println!("  {:>5.2}-{:>5.2} h: {:.1} s wall, cars on road {}, routed {} total, A* this hour {} sim-thread / {} workers, sim thread routing {:.1} s + waiting {:.1} s, pool full {}, load {}",
                    h0 + hr as f64 * chunk_h, h0 + (hr + 1) as f64 * chunk_h, t.elapsed().as_secs_f64(), cap - sim.free_len(), ss.routed, s1.misses - s0.misses, s1.prefetched - s0.prefetched,
                    (s1.route_ns - s0.route_ns) as f64 * 1e-9, (s1.wait_ns - s0.wait_ns) as f64 * 1e-9, ss.pool_full, load());
            }
            let st = sim.route_stats().unwrap();
            println!("total {:.1} s; spawn hash {:016x} routed {} no route {}; OD lookups {} hits {} ({:.1} %)", total.elapsed().as_secs_f64(), sim.x.stream_stats.spawn_hash, sim.x.stream_stats.routed, sim.counters.no_route,
                st.lookups, st.hits, 100.0 * st.hits as f64 / st.lookups.max(1) as f64);
            println!("reroutes queued {rr_n} (reroutes applied {}): plain {rr_plain} with {} distinct ODs (OD-cache hit rate {:.1} %); penalised {} with {} distinct (OD, 60 s epoch) (hit rate {:.1} %)",
                sim.counters.reroutes, rr_plain_keys.len(), 100.0 * (1.0 - rr_plain_keys.len() as f64 / rr_plain.max(1) as f64), rr_n - rr_plain, rr_keys.len(), 100.0 * (1.0 - rr_keys.len() as f64 / (rr_n - rr_plain).max(1) as f64));
        }
        "spawn" => {
            // host-only spawn table, every field: frames of 20 steps without GPU stepping; each spawn-queue entry is
            // consumed at once (SPAWN event) and each car arrives (ARRIVE) after length / 12 m/s, so slots, queue entries
            // and arena offsets are recycled deterministically. Hash of (commuter, step, slot, queue entry, route
            // offset + words) for every placed departure.
            let (h0, hours): (f64, f64) = (arg(&a, 5, 5.0), arg(&a, 6, 5.0));
            let cap: usize = arg(&a, 7, 131072);
            let o = GpuOptions { capacity: cap, start_time: h0 * 3600.0, refuel_budget_ms: f64::INFINITY, route_threads: threads, ..Default::default() };
            let mut sim = bayenv_gpu::env::create(world.clone(), o).expect("sim");
            sim.set_streaming(Arc::new(table), false, None);
            let g = world.clone();
            let mut arrivals: std::collections::BTreeMap<u64, Vec<u32>> = Default::default();
            let mut h = 0xcbf29ce484222325u64; let mut placed = 0u64;
            let t0 = Instant::now(); let frames = (hours * 3600.0) as u64;
            for f in 0..frames {
                let t_end = sim.time + 20.0 * sim.dt;
                let qf: std::collections::HashSet<u32> = sim.queue_free_list().iter().copied().collect();
                let before: Vec<i32> = sim.slot_trip.clone();
                sim.enqueue_spawns(t_end);
                let qa: std::collections::HashSet<u32> = sim.queue_free_list().iter().copied().collect();
                let used: Vec<u32> = qf.difference(&qa).copied().collect::<std::collections::BTreeSet<u32>>().into_iter().collect();
                for q in &used { fnv(&mut h, *q); }
                for s in 0..cap { if sim.slot_trip[s] >= 0 && sim.slot_trip[s] != before[s] {
                    let (off, len) = (sim.slot_route[2 * s] as usize, sim.slot_route[2 * s + 1] as usize);
                    for v in [sim.slot_trip[s] as u32, sim.step as u32, s as u32, off as u32, len as u32] { fnv(&mut h, v); }
                    let mut m = 0.0; for &w in &sim.route_mirror[off..off + len] { fnv(&mut h, w); }
                    let _ = &g; m += len as f64 * 30.0; // ~30 m per edge, 12 m/s
                    arrivals.entry(f + 1 + (m / 12.0) as u64).or_default().push(s as u32); placed += 1;
                } }
                let mut ev: Vec<[u32; 4]> = used.iter().map(|&q| [bayenv_gpu::layout::ev::SPAWN, 0, q, 0]).collect();
                if let Some(v) = arrivals.remove(&f) { for s in v { ev.push([bayenv_gpu::layout::ev::ARRIVE, s, 0, 10f32.to_bits()]); } }
                sim.ingest_events(&ev); sim.take_ops();
                sim.time = t_end; sim.step += 20;
            }
            let st = sim.route_stats().unwrap();
            println!("spawn table {h0}-{} h, threads {threads}: {:.1} s, {placed} placed, pool full {}, route words {}, waited {} ({:.1} s), sim-thread A* {}, hash {h:016x}",
                h0 + hours, t0.elapsed().as_secs_f64(), sim.x.stream_stats.pool_full, sim.route_words, st.waited, st.wait_ns as f64 * 1e-9, st.misses);
        }
        "prof" => {
            // step-time split of a streaming population: host submit (enqueue / scenes / refuels / reroutes / encode),
            // GPU wait (poll), host finish (readback, events, records). Env PROF_OPTS: record_events=0, stations=0, reroute=0
            let (h0, warm_min, min, steps): (f64, f64, f64, usize) = (arg(&a, 5, 7.5), arg(&a, 6, 2.0), arg(&a, 7, 3.0), arg(&a, 8, 20));
            let cap: usize = arg(&a, 9, 131072);
            let mut o = GpuOptions { capacity: cap, start_time: h0 * 3600.0, refuel_budget_ms: f64::INFINITY, route_threads: threads, ..Default::default() };
            let po = std::env::var("PROF_OPTS").unwrap_or_default();
            let flag = |k: &str| po.split(',').find_map(|kv| kv.strip_prefix(&format!("{k}=")).map(|v| v != "0"));
            if let Some(v) = flag("record_events") { o.record_events = v; }
            if let Some(v) = flag("reroute") { o.reroute = v; }
            if let Some(v) = flag("collisions") { o.features.collisions = v; }
            if let Some(v) = flag("accidents") { o.features.accidents = v; }
            let stations = flag("stations").unwrap_or(true);
            o.apply_env();   // BAYENV_GPU_OPTS (e.g. deterministic_leaders=0)
            let mut sim = bayenv_gpu::env::create(world.clone(), o).expect("sim");
            sim.set_streaming(Arc::new(table), stations, None);
            let dt = sim.dt; let calls = |m: f64| (m * 60.0 / dt).round() as usize / steps;
            let tw = Instant::now();
            for _ in 0..calls(warm_min) { sim.run_steps(steps); }
            sim.sync();
            println!("prof: start {h0} h, warm-up {warm_min} sim min in {:.1} s, cars on road {}, PROF_OPTS '{po}', steps/call {steps}, capacity {cap}, load {}", tw.elapsed().as_secs_f64(), cap - sim.free_len(), load());
            if let Some(p) = sim.profiler.as_mut() { p.reset(); }   // BAYENV_GPU_OPTS=profile=1: per-kernel GPU time
            let (mut ts, mut tw, mut tf) = (0f64, 0f64, 0f64); let k = calls(min); let (r0, e0) = (sim.counters.reroutes, sim.rec.len());
            let t_all = Instant::now();
            for _ in 0..k {
                let t0 = Instant::now(); let p = sim.submit_steps(steps, 0, 0); let t1 = Instant::now();
                sim.gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll"); let t2 = Instant::now();
                sim.finish_steps(p, None, None); let t3 = Instant::now();
                ts += (t1 - t0).as_secs_f64(); tw += (t2 - t1).as_secs_f64(); tf += (t3 - t2).as_secs_f64();
            }
            let n = (k * steps) as f64;
            println!("prof: {} steps in {:.1} s = {:.2} ms/step: submit {:.2}, GPU wait {:.2}, finish {:.2} ms/step; cars on road {}, reroutes {}, records {}, load {}",
                k * steps, t_all.elapsed().as_secs_f64(), t_all.elapsed().as_secs_f64() * 1e3 / n, ts * 1e3 / n, tw * 1e3 / n, tf * 1e3 / n, cap - sim.free_len(), sim.counters.reroutes - r0, sim.rec.len() - e0, load());
            if let Some(p) = &sim.profiler {
                let mut rows = p.report(); rows.sort_by(|a, b| b.ms.partial_cmp(&a.ms).unwrap());
                println!("prof profile: {} profiled steps, GPU step {:.3} ms", p.steps, p.step_ms());
                for r in rows.iter().take(40) { println!("  {:10} {:24} {:8.3} ms {:5}", r.stage, r.kernel, r.ms, r.dispatches); }
            }
            // state hash (bit-identity checks between builds): GPU state, slot tables, route mirror, records, counters
            let stt = sim.read_state(); let mut h = 0xcbf29ce484222325u64;
            for &w in stt.veh.iter().chain(&stt.par) { fnv(&mut h, w); } for &f in &stt.pose { fnv(&mut h, f.to_bits()); }
            let hs = h; let mut h2 = 0xcbf29ce484222325u64;
            for &x in &sim.slot_trip { fnv(&mut h2, x as u32); } for &x in &sim.slot_route { fnv(&mut h2, x); } for &x in &sim.route_mirror[..sim.route_words] { fnv(&mut h2, x); }
            let mut h3 = 0xcbf29ce484222325u64; let r = &sim.rec;
            for i in 0..r.len() { for v in [r.step[i] as u32, r.kind[i] as u32, r.slot[i] as u32, r.agent[i] as u32, r.other[i] as u32, r.progress_m[i].to_bits(), r.late_by[i].to_bits(), r.dv_kmh[i].to_bits(), r.time[i].to_bits() as u32] { fnv(&mut h3, v); } }
            let c = &sim.counters;
            println!("prof hash: state {hs:016x} tables {h2:016x} records {h3:016x} ({} rows) spawn {:016x}; reroutes {} failed {} checked {} crashes {} arrived-spawned {} route_words {}",
                r.len(), sim.x.stream_stats.spawn_hash, c.reroutes, c.reroute_failed, bayenv_gpu::sim::GpuSim::reroute_checked(), c.crashes, sim.stats()[bayenv_gpu::layout::stat::ARRIVED], sim.route_words);
        }
        _ => panic!("mode route | sim | spawn | prof"),
    }
}
