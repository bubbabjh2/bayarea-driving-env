//! Router benchmark: the pre-optimisation router (nav::router_ref) vs nav::Router on the nav oracle's queries
//! (`rust/oracle/nav/nav.json`: 3000 plain OD pairs w 1.25, 750 w 1, 750 avoid, 750 penalty), single-threaded,
//! then route_batch throughput on all cores and the OD-cache hit cost.
//!   cargo run --release -p bayenv-world --example router_bench -- <dataDir> <nav.json> [rounds]
use bayenv_world::nav::router_ref::RefRouter;
use bayenv_world::nav::{RouteQuery, Router, RouterOpts};
use bayenv_world::World;
use std::time::Instant;

struct Q { s: usize, g: usize, w1: bool, avoid: Vec<u32>, pen: Vec<(u32, f64)> }

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let w = World::load(&a[1]).expect("world");
    let rounds: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&a[2]).unwrap()).unwrap();
    let u32s = |v: &serde_json::Value| -> Vec<u32> { v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect() };
    let qs: Vec<Q> = v["queries"].as_array().unwrap().iter().map(|q| Q {
        s: q["s"].as_u64().unwrap() as usize, g: q["g"].as_u64().unwrap() as usize, w1: q["w"].as_f64().unwrap() == 1.0,
        avoid: q.get("avoid").map(u32s).unwrap_or_default(),
        pen: q.get("pen").map(|p| p.as_array().unwrap().iter().map(|x| (x[0].as_u64().unwrap() as u32, x[1].as_f64().unwrap())).collect()).unwrap_or_default(),
    }).collect();
    let kind = |q: &Q| if !q.avoid.is_empty() { 2 } else if !q.pen.is_empty() { 3 } else if q.w1 { 1 } else { 0 };
    let names = ["plain w=1.25", "plain w=1", "avoid w=1.25", "penalty w=1.25"];
    let g = &w.graph;
    let o125 = RouterOpts { weight: 1.25, cache_max: 0, ..Default::default() };
    let o1 = RouterOpts { weight: 1.0, cache_max: 0, ..Default::default() };
    let (mut ra, mut rb) = (RefRouter::new(g, &o125), RefRouter::new(g, &o1));
    let (mut na, mut nb) = (Router::new(g, &o125), Router::new(g, &o1));
    eprintln!("nodes {} edges {} queries {} threads {}", g.node_count, g.edge_count, qs.len(), rayon::current_num_threads());
    // single-threaded, per kind; ref and new alternate per query (same machine load), min over rounds
    let mut best = [[f64::INFINITY; 4]; 2]; let mut cnt = [0usize; 4];
    for q in &qs { cnt[kind(q)] += 1; }
    let mut out = Vec::new();
    for _ in 0..rounds {
        let mut sum = [[0.0f64; 4]; 2];
        for q in &qs {
            let k = kind(q); let rq = RouteQuery { avoid: &q.avoid, penalty: &q.pen };
            let t = Instant::now();
            std::hint::black_box(if q.w1 { rb.route_into(q.s, q.g, &rq, &mut out) } else { ra.route_into(q.s, q.g, &rq, &mut out) });
            let t1 = Instant::now();
            std::hint::black_box(if q.w1 { nb.route_into(q.s, q.g, &rq, &mut out) } else { na.route_into(q.s, q.g, &rq, &mut out) });
            let t2 = Instant::now();
            sum[0][k] += (t1 - t).as_secs_f64(); sum[1][k] += (t2 - t1).as_secs_f64();
        }
        for i in 0..2 { for k in 0..4 { best[i][k] = best[i][k].min(sum[i][k]); } }
    }
    println!("| query kind | before (ms) | after (ms) | speed-up |");
    println!("|---|---|---|---|");
    for k in 0..4 {
        let (b, n) = (best[0][k] * 1e3 / cnt[k] as f64, best[1][k] * 1e3 / cnt[k] as f64);
        println!("| {} ({} queries) | {b:.3} | {n:.3} | {:.2}x |", names[k], cnt[k], b / n);
    }
    println!("tie fallbacks: {}", na.fallbacks() + nb.fallbacks());
    // batch throughput: every w 1.25 query (plain, avoid, penalty), all cores, cache off
    let batch: Vec<(usize, usize, RouteQuery)> = qs.iter().filter(|q| !q.w1).map(|q| (q.s, q.g, RouteQuery { avoid: &q.avoid, penalty: &q.pen })).collect();
    let single_s = (best[1][0] + best[1][2] + best[1][3]) / batch.len() as f64;
    let mut bt = f64::INFINITY;
    for _ in 0..rounds { let t = Instant::now(); std::hint::black_box(na.route_batch(&batch)); bt = bt.min(t.elapsed().as_secs_f64()); }
    let ref_s = (best[0][0] + best[0][2] + best[0][3]) / batch.len() as f64;
    println!("batch ({} w 1.25 queries, {} threads): {:.0} queries/s ({:.3} ms wall per query); single-thread new {:.0} q/s, ref {:.0} q/s",
        batch.len(), rayon::current_num_threads(), batch.len() as f64 / bt, bt * 1e3 / batch.len() as f64, 1.0 / single_s, 1.0 / ref_s);
    // OD cache: plain w 1.25 queries, each OD twice (second = hit)
    let mut cr = Router::new(g, &RouterOpts { weight: 1.25, ..Default::default() });
    let plain: Vec<&Q> = qs.iter().filter(|q| kind(q) == 0).collect();
    let t = Instant::now(); for q in &plain { std::hint::black_box(cr.route(q.s, q.g)); } let miss = t.elapsed().as_secs_f64() / plain.len() as f64;
    let t = Instant::now(); for q in &plain { std::hint::black_box(cr.route(q.s, q.g)); } let hit = t.elapsed().as_secs_f64() / plain.len() as f64;
    let pb: Vec<(usize, usize, RouteQuery)> = plain.iter().map(|q| (q.s, q.g, RouteQuery::default())).collect();
    let t = Instant::now(); std::hint::black_box(cr.route_batch(&pb)); let bhit = t.elapsed().as_secs_f64() / pb.len() as f64;
    println!("OD cache: miss {:.3} ms, hit {:.2} us (single), batch all-hit {:.2} us wall per query", miss * 1e3, hit * 1e6, bhit * 1e6);
}
