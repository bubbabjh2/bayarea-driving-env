//! Same-seed determinism: two GPU sims with the same options, stepped in lockstep in one process; after every call of
//! `steps` steps the GPU state (veh, par, pose, ctl), host tables (slot trips / routes, route mirror) and the event records
//! are hashed and compared. On the first difference: the call, the buffer, and (veh) the first differing slots / fields.
//!
//!   cargo run --release -p bayenv-gpu --example determinism -- <data> scatter|pop|stream [n] [start_h=7.5] [calls=60] [steps=20] [threads]
//!     scatter: n cars scattered (seed 1); pop: population of n (seed 42, routes at load); stream: CommuterTable of n
//!     (BAYENV_TABLE_CACHE=<file> to reuse); threads = route prefetch workers (stream). Env BAYENV_GPU_OPTS applies.
//!     DET_OPTS: comma list of k=v overrides: refuel_ms (default: GpuOptions default; inf), stations=0|1, collisions=0|1,
//!     events=0|1 (record_events).
use bayenv_gpu::layout as L;
use bayenv_gpu::sim::{CommuterTable, GpuOptions, GpuPopulation, GpuSim};
use std::{path::Path, sync::Arc};

fn fnv(h: &mut u64, v: u32) { for b in v.to_le_bytes() { *h ^= b as u64; *h = h.wrapping_mul(0x100000001b3); } }
fn hash(ws: &[u32]) -> u64 { let mut h = 0xcbf29ce484222325u64; for &w in ws { fnv(&mut h, w); } h }
fn arg<T: std::str::FromStr>(a: &[String], i: usize, d: T) -> T { a.get(i).and_then(|s| s.parse().ok()).unwrap_or(d) }

struct Snap { parts: Vec<(&'static str, Vec<u32>)> }
fn snap(s: &GpuSim) -> Snap {
    let st = s.read_state();
    let r = &s.rec; let mut rec = Vec::new();
    for i in 0..r.len() { rec.extend([r.step[i] as u32, r.kind[i] as u32, r.slot[i] as u32, r.agent[i] as u32, r.other[i] as u32, r.dv_kmh[i].to_bits(), r.progress_m[i].to_bits(), r.late_by[i].to_bits()]); }
    let trips: Vec<u32> = s.slot_trip.iter().map(|&x| x as u32).collect();
    Snap { parts: vec![("veh", st.veh), ("par", st.par), ("pose", bytemuck::cast_slice(&st.pose).to_vec()), ("ctl", s.read_words("ctl")),
        ("slot_trip", trips), ("slot_route", s.slot_route.clone()), ("route_mirror", s.route_mirror[..s.route_words].to_vec()), ("records", rec)] }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = Path::new(&a[1]); let mode = a[2].clone();
    let n: usize = arg(&a, 3, 10000); let h0: f64 = arg(&a, 4, 7.5); let calls: usize = arg(&a, 5, 60); let steps: usize = arg(&a, 6, 20);
    let threads: usize = arg(&a, 7, std::thread::available_parallelism().map_or(1, |n| n.get()));
    let det = std::env::var("DET_OPTS").unwrap_or_default();
    let dopt = |k: &str| det.split(',').find_map(|kv| kv.strip_prefix(&format!("{k}=")).map(|v| v.to_string()));
    let world = Arc::new(bayenv_world::World::load(data).expect("world"));
    let cap = n.min(131072).div_ceil(1024) * 1024;
    let mut o = GpuOptions { capacity: cap, start_time: h0 * 3600.0, route_threads: threads, ..Default::default() };
    if let Some(v) = dopt("refuel_ms") { o.refuel_budget_ms = if v == "inf" { f64::INFINITY } else { v.parse().unwrap() }; }
    o.apply_env();
    if let Some(v) = dopt("collisions") { o.features.collisions = v != "0"; }
    if let Some(v) = dopt("events") { o.record_events = v != "0"; }
    let stations = dopt("stations").map_or(true, |v| v != "0");
    let table = if mode == "stream" {
        let cf = std::env::var("BAYENV_TABLE_CACHE").ok().map(std::path::PathBuf::from);
        Some(Arc::new(match &cf { Some(f) if f.exists() => CommuterTable::load(f).expect("table"), _ => { let t = CommuterTable::generate(&world, n, 42.0).expect("table"); if let Some(f) = &cf { t.save(f).expect("save"); } t } }))
    } else { None };
    let pop = if mode == "pop" { Some(Arc::new(GpuPopulation::generate(&world, n, 42.0, true).expect("pop"))) } else { None };
    let make = || {
        let mut s = bayenv_gpu::env::create(world.clone(), o.clone()).expect("sim");
        match mode.as_str() { "scatter" => { bayenv_gpu::env::scatter(&mut s, n, 1); } "pop" => s.set_population(pop.clone().unwrap(), stations), "stream" => s.set_streaming(table.clone().unwrap(), stations, None), _ => panic!("mode") }
        s
    };
    let (mut x, mut y) = (make(), make());
    println!("determinism {mode} n {n} cap {cap} start {h0} h, {calls} x {steps} steps, refuel_budget_ms {}, route_threads {threads}", o.refuel_budget_ms);
    let mut ctl_seen = 0;
    for c in 0..calls {
        x.run_steps(steps); y.run_steps(steps); x.sync(); y.sync();
        let (sx, sy) = (snap(&x), snap(&y));
        // ctl (perception side record + junction list links + per-step counters) is reported, not a stop condition:
        // words that are list links hold an order that need not be unique
        if let (Some(p), Some(q)) = (sx.parts.iter().find(|p| p.0 == "ctl"), sy.parts.iter().find(|p| p.0 == "ctl")) {
            let mut w = std::collections::BTreeMap::new();
            for (k, (u, v)) in p.1.iter().zip(&q.1).enumerate() { if u != v { *w.entry(if k < cap * L::CTL_WORDS { format!("slot word {}", k % L::CTL_WORDS) } else { format!("tail {}", k - cap * L::CTL_WORDS) }).or_insert(0usize) += 1; } }
            if !w.is_empty() && ctl_seen < 3 { ctl_seen += 1; println!("  call {} (step {}): ctl differs: {:?}", c + 1, x.step, w.iter().take(12).collect::<Vec<_>>()); }
        }
        let bad: Vec<&str> = sx.parts.iter().zip(&sy.parts).filter(|(p, q)| p.0 != "ctl" && p.1 != q.1).map(|(p, _)| p.0).collect();
        if bad.is_empty() { if c % 10 == 9 || c + 1 == calls { println!("  call {:>4} (step {:>6}): identical, cars {}, records {}, hash {:016x}", c + 1, x.step, cap - x.free_len(), x.rec.len(), hash(&sx.parts[0].1)); } continue; }
        println!("FIRST DIFFERENCE after call {} (steps {}..{}): {:?}", c + 1, x.step as usize - steps + 1, x.step, bad);
        for (p, q) in sx.parts.iter().zip(&sy.parts) {
            if p.1 == q.1 { continue; }
            if p.0 == "ctl" { continue; }
            if p.0 == "veh" {
                let mut k = 0;
                for i in 0..cap { let (u, v) = (&p.1[i * L::VEH_WORDS..(i + 1) * L::VEH_WORDS], &q.1[i * L::VEH_WORDS..(i + 1) * L::VEH_WORDS]);
                    if u != v { let f: Vec<String> = (0..L::VEH_WORDS).filter(|&f| u[f] != v[f]).map(|f| format!("{}:{:#x}/{:#x}", L::VEH_FIELDS[f], u[f], v[f])).collect();
                        println!("    slot {i} trip {}/{}: {}", x.slot_trip[i], y.slot_trip[i], f.join(" ")); k += 1;
                        // a different leader: both candidates as run x sees them (edge, s, lat, lanes, state, length)
                        if u[L::veh::LEADER] != v[L::veh::LEADER] { for &c in &[u[L::veh::LEADER], v[L::veh::LEADER], i as u32] { if (c as i32) >= 0 { let w = &p.1[c as usize * L::VEH_WORDS..(c as usize + 1) * L::VEH_WORDS];
                            println!("      slot {c}: edge {} s {} lat {} lanes {:#x} state {} gap {} (me: car {i})", w[0], f32::from_bits(w[1]), f32::from_bits(w[4]), w[14], w[7], f32::from_bits(w[21])); } } }
                        if k >= 12 { break; } } }
            } else {
                let i = p.1.iter().zip(&q.1).position(|(u, v)| u != v).unwrap_or(p.1.len().min(q.1.len()));
                println!("  {}: len {} / {}, first differing word {i}", p.0, p.1.len(), q.1.len());
            }
        }
        std::process::exit(1);
    }
    println!("DETERMINISTIC over {} steps", calls * steps);
}
