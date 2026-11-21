//! Streaming route prefetch (RouteProvider workers) vs serial routing at departure.
//!
//!   cargo run --release -p bayenv-gpu --example stream_prefetch -- <data> route [n=1380000] [threads=cores] [from_h=5] [to_h=10]
//!     routes every departure of [from_h - 1 h, to_h) (the window setStreaming takes from a from_h start) serially,
//!     then with `threads` prefetch workers; checks that both give the same route for every commuter.
//!   cargo run --release -p bayenv-gpu --example stream_prefetch -- <data> sim [n] [threads] [start_h=5] [hours=1] [steps=20] [capacity]
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
    let table = CommuterTable::generate(&world, n, 42.0).expect("table");
    println!("CommuterTable {n}: {:.1} s (generation routes once, parallel), load {}", t.elapsed().as_secs_f64(), load());
    match mode {
        "route" => {
            let (h0, h1): (f64, f64) = (arg(&a, 5, 5.0), arg(&a, 6, 10.0));
            let (k0, k1) = (table.index_at(h0 * 3600.0 - 3600.0), table.index_at(h1 * 3600.0));
            let ids: Vec<usize> = table.order[k0..k1].iter().map(|&i| i as usize).collect();
            println!("departures {}..{} h: {} commuters", h0 - 1.0, h1, ids.len());
            let run = |threads: usize| {
                let mut p = RouteProvider::new(&world, threads);
                let t = Instant::now();
                let mut h = 0xcbf29ce484222325u64; let mut none = 0;
                if threads > 0 {
                    // the sim's pattern without the GPU: the clock moves through the departures in 1 s frames,
                    // prefetching up to 300 s ahead; each departure takes its route when the clock reaches it
                    let (mut pf, mut k) = (0usize, 0usize);
                    let mut clock = table.sorted_depart[k0] as f64;
                    while k < ids.len() {
                        while pf < ids.len() && table.sorted_depart[k0 + pf] as f64 <= clock + 300.0 { let i = ids[pf]; p.prefetch(table.home[i], table.work[i]); pf += 1; }
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
                println!("threads {threads:>2}: {:.1} s, {:.0} departures/s, {} A* ({:.0} routes/s), OD cache hit rate {:.1} % ({} served from cache incl. prefetched, {} waited for a worker, {:.1} s waiting), {} no route, cache {} entries / {:.0} MB of edges, worker A* {:.2} ms/route, sim-thread A* {:.2} ms/route, load {}, route hash {h:016x}",
                    s, ids.len() as f64 / s, computed, computed as f64 / s, 100.0 * (1.0 - computed as f64 / st.lookups as f64), st.hits, st.waited, st.wait_ns as f64 * 1e-9, none,
                    st.cache_len, st.cache_edges as f64 * 4.0 / 1048576.0, st.worker_ns as f64 * 1e-6 / st.prefetched.max(1) as f64, st.route_ns as f64 * 1e-6 / st.misses.max(1) as f64, load());
                h
            };
            let modes: Vec<usize> = a.get(7).map_or(vec![0, threads], |s| s.split(',').map(|x| x.parse().unwrap()).collect());
            let hs: Vec<u64> = modes.iter().map(|&t| run(t)).collect();
            println!("routes identical across {:?}: {}", modes, hs.windows(2).all(|w| w[0] == w[1]));
        }
        "sim" => {
            let (h0, hours, steps): (f64, f64, usize) = (arg(&a, 5, 5.0), arg(&a, 6, 1.0), arg(&a, 7, 20));
            let cap: usize = arg(&a, 8, n.min(131072).div_ceil(1024) * 1024);
            let o = GpuOptions { capacity: cap, start_time: h0 * 3600.0, refuel_budget_ms: f64::INFINITY, route_threads: threads, ..Default::default() };
            let mut sim = bayenv_gpu::env::create(world.clone(), o).expect("sim");
            sim.set_streaming(Arc::new(table), true, None);
            println!("sim: capacity {cap}, start {h0} h, {hours} h, {steps} steps per call, route threads {threads}");
            let per_hour = (3600.0 / sim.dt).round() as usize / steps;
            let total = Instant::now();
            let total_calls = (hours * per_hour as f64).round() as usize;
            for hr in 0..total_calls.div_ceil(per_hour) {
                let t = Instant::now(); let s0 = sim.route_stats().unwrap();
                for _ in 0..per_hour.min(total_calls - hr * per_hour) { sim.run_steps(steps); }
                sim.sync();
                let s1 = sim.route_stats().unwrap(); let ss = sim.x.stream_stats;
                println!("  hour {:>4.1}-{:>4.1}: {:.1} s wall, cars on road {}, routed {} total, A* this hour {} sim-thread / {} workers, sim thread routing {:.1} s + waiting {:.1} s, pool full {}, load {}",
                    h0 + hr as f64, h0 + hr as f64 + 1.0, t.elapsed().as_secs_f64(), cap - sim.free_len(), ss.routed, s1.misses - s0.misses, s1.prefetched - s0.prefetched,
                    (s1.route_ns - s0.route_ns) as f64 * 1e-9, (s1.wait_ns - s0.wait_ns) as f64 * 1e-9, ss.pool_full, load());
            }
            let st = sim.route_stats().unwrap();
            println!("total {:.1} s; spawn hash {:016x} routed {} no route {}; OD lookups {} hits {} ({:.1} %)", total.elapsed().as_secs_f64(), sim.x.stream_stats.spawn_hash, sim.x.stream_stats.routed, sim.counters.no_route,
                st.lookups, st.hits, 100.0 * st.hits as f64 / st.lookups.max(1) as f64);
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
        _ => panic!("mode route | sim | spawn"),
    }
}
