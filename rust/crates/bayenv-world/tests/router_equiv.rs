//! nav::Router (4-ary heap + goal-bound skipping + tie fallback, SCC no-path check, batch, shared OD cache) vs
//! the pre-optimisation router (nav::router_ref, itself checked against the JS by nav_oracle.rs): identical paths
//! and "no path" results on random queries. `ROUTER_EQ_N=<OD pairs>` (default 600; each pair gives 6 queries:
//! plain w 1.25, plain w 1, avoid, penalty, avoid + penalty, penalty with +inf / NaN values).
//! Skips when the data are absent.
use bayenv_world::nav::router_ref::RefRouter;
use bayenv_world::nav::{RouteQuery, Router, RouterOpts};
use bayenv_world::graph::{Bounds, SimGraph};
use bayenv_world::World;
use rayon::prelude::*;
use std::path::Path;
use std::time::Instant;

struct Q { s: usize, g: usize, w1: bool, avoid: Vec<u32>, pen: Vec<(u32, f64)> }

#[test]
fn router_equals_reference() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    if !root.join("data/roads.meta.json").exists() { eprintln!("skip: no data"); return; }
    let w = World::load(root.join("data")).expect("world");
    let g = &w.graph;
    let n_od: usize = std::env::var("ROUTER_EQ_N").ok().and_then(|v| v.parse().ok()).unwrap_or(600);
    let e_n = g.edge_count as u64;
    // OD pairs: starts with an out-edge, goals with an in-edge (some unreachable: exercises "no path")
    let starts: Vec<usize> = (0..g.node_count).filter(|&i| g.out_degree(i) > 0).collect();
    let goals: Vec<usize> = (0..g.node_count).filter(|&i| !g.in_of(i).is_empty()).collect();
    let mut s = 0x2545F4914F6CDD1Du64;
    let mut rnd = move || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
    let ods: Vec<(usize, usize, u64)> = (0..n_od).map(|_| { let a = starts[(rnd() % starts.len() as u64) as usize]; let b = goals[(rnd() % goals.len() as u64) as usize]; (a, b, rnd()) }).collect();
    let o125 = RouterOpts { weight: 1.25, cache_max: 0, ..Default::default() };
    let o1 = RouterOpts { weight: 1.0, cache_max: 0, ..Default::default() };

    let (base125, base1) = (Router::new(g, &o125), Router::new(g, &o1));
    let cached125 = Router::new(g, &RouterOpts { weight: 1.25, ..Default::default() });
    let cached1 = Router::new(g, &RouterOpts { weight: 1.0, ..Default::default() });
    let (mut bad, mut bad_batch, mut fb, mut n_none, mut n_q, mut t_ref, mut t_new) = (0usize, 0usize, 0u64, 0usize, 0usize, 0.0f64, 0.0f64);
    // blocks of 1500 OD pairs (bounded memory)
    for blk in ods.chunks(1500) {
        // reference results + the derived queries
        let t = Instant::now();
        let chunk = blk.len().div_ceil(rayon::current_num_threads() * 4).max(1);
        let groups: Vec<Vec<(Q, Option<Vec<u32>>)>> = blk.par_chunks(chunk).map(|ods| {
            let (mut ra, mut rb) = (RefRouter::new(g, &o125), RefRouter::new(g, &o1));
            let mut v = Vec::new();
            for &(s, gl, seed) in ods {
                let mut x = seed | 1; let mut r = move || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; x };
                let plain = ra.route_with(s, gl, &RouteQuery::default());
                let p1 = rb.route_with(s, gl, &RouteQuery::default());
                let base = plain.clone().unwrap_or_default();
                let mut avoid: Vec<u32> = base.iter().enumerate().filter(|(k, _)| k % 3 == 1).map(|(_, &e)| e).collect();
                for _ in 0..20 { avoid.push((r() % e_n) as u32); }
                avoid.push(e_n as u32 + 7);
                let mut pen: Vec<(u32, f64)> = base.iter().map(|&e| (e, (r() % 240) as f64 * 0.25)).collect();
                for _ in 0..50 { pen.push(((r() % e_n) as u32, (r() % 40) as f64 * 0.25)); }
                if let Some(&e) = base.first() { pen.push((e, 3.0)); }
                let mut pen_odd = pen.clone();
                if let Some(&e) = base.get(base.len() / 2) { pen_odd.push((e, f64::INFINITY)); }
                pen_odd.push(((r() % e_n) as u32, f64::NAN));
                let av_small: Vec<u32> = avoid.iter().step_by(2).copied().collect();
                let qs = [(false, avoid.clone(), vec![]), (false, vec![], pen.clone()), (false, av_small, pen.clone()), (false, vec![], pen_odd)];
                v.push((Q { s, g: gl, w1: false, avoid: vec![], pen: vec![] }, plain));
                v.push((Q { s, g: gl, w1: true, avoid: vec![], pen: vec![] }, p1));
                for (w1, avoid, pen) in qs {
                    let res = ra.route_with(s, gl, &RouteQuery { avoid: &avoid, penalty: &pen });
                    v.push((Q { s, g: gl, w1, avoid, pen }, res));
                }
            }
            v
        }).collect();
        let all: Vec<(Q, Option<Vec<u32>>)> = groups.into_iter().flatten().collect();
        t_ref += t.elapsed().as_secs_f64();
        n_none += all.iter().filter(|q| q.1.is_none()).count();

        // single queries on forked routers (one shared graph)
        let t = Instant::now();
        let res: Vec<(usize, u64)> = all.par_chunks(chunk * 6).enumerate().map(|(ci, qs)| {
            let (mut a, mut b) = (base125.fork(), base1.fork());
            // every 5th chunk: the exact (JS heap) search alone, so the tie fallback path is checked on real data too
            if ci % 5 == 4 { a.set_exact_only(true); b.set_exact_only(true); }
            let mut out = Vec::new(); let mut bad = 0;
            for (q, want) in qs {
                let r = if q.w1 { &mut b } else { &mut a };
                let ok = r.route_into(q.s, q.g, &RouteQuery { avoid: &q.avoid, penalty: &q.pen }, &mut out);
                let got = if ok { Some(out.clone()) } else { None };
                if &got != want { bad += 1; if bad < 4 { eprintln!("mismatch {}->{} w1={} avoid={} pen={}", q.s, q.g, q.w1, q.avoid.len(), q.pen.len()); } }
            }
            (bad, a.fallbacks() + b.fallbacks())
        }).collect();
        t_new += t.elapsed().as_secs_f64();
        bad += res.iter().map(|r| r.0).sum::<usize>(); fb += res.iter().map(|r| r.1).sum::<u64>(); n_q += all.len();

        // batch API with the shared OD cache: twice (second pass hits the cache for plain queries)
        for pass in 0..2 {
            for (w1, r) in [(false, &cached125), (true, &cached1)] {
                let idx: Vec<usize> = (0..all.len()).filter(|&i| all[i].0.w1 == w1).collect();
                let qs: Vec<(usize, usize, RouteQuery)> = idx.iter().map(|&i| { let q = &all[i].0; (q.s, q.g, RouteQuery { avoid: &q.avoid, penalty: &q.pen }) }).collect();
                let got = r.route_batch(&qs);
                for (k, &i) in idx.iter().enumerate() { if got[k] != all[i].1 { bad_batch += 1; if bad_batch < 4 { eprintln!("batch mismatch pass {pass} query {i}"); } } }
            }
        }
    }
    let c = cached125.cache().unwrap();
    eprintln!("router_equiv: {} queries ({} OD pairs, {n_none} no path), {bad} single / {bad_batch} batch mismatches, {fb} tie fallbacks; ref {:.1} s, new {:.1} s (all threads); cache hits {} misses {}",
        n_q, n_od, t_ref, t_new, c.hits.load(std::sync::atomic::Ordering::Relaxed), c.misses.load(std::sync::atomic::Ordering::Relaxed));
    assert_eq!(bad, 0);
    assert_eq!(bad_batch, 0);
}

/// Synthetic grid with equal edge costs: many exact key ties, so the fast search falls back often. Banned hops,
/// one-way stubs and an unreachable island included. Router (fast + fallback, and exact-only) vs RefRouter.
#[test]
fn ties_fall_back_and_match_reference() {
    let (w, h) = (40usize, 40usize);
    let n = w * h + 3; // + island of 3 nodes
    let mut g = SimGraph { node_count: n, ..Default::default() };
    for i in 0..n { let (x, z) = if i < w * h { ((i % w) as f64 * 10.0, (i / w) as f64 * 10.0) } else { (1000.0 + i as f64, 1000.0) }; g.nx.push(x); g.nz.push(z); }
    let mut s = 0x1234_5678_9ABC_DEF1u64;
    let mut rnd = move || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
    let mut edges: Vec<(u32, u32)> = Vec::new();
    for y in 0..h { for x in 0..w {
        let i = (y * w + x) as u32;
        if x + 1 < w { edges.push((i, i + 1)); if rnd() % 7 != 0 { edges.push((i + 1, i)); } }
        if y + 1 < h { edges.push((i, i + w as u32)); if rnd() % 7 != 0 { edges.push((i + w as u32, i)); } }
    } }
    let b = (w * h) as u32; edges.extend([(b, b + 1), (b + 1, b + 2), (b + 2, b)]);
    g.edge_count = edges.len();
    for &(a, c) in &edges {
        g.from.push(a); g.to.push(c);
        g.length.push(10.0); g.speed_limit.push([10.0, 20.0, 12.5][(rnd() % 3) as usize]); g.rank.push((rnd() % 3) as u8);
    }
    g.out_start = vec![0; n + 1]; g.in_start = vec![0; n + 1];
    for &(a, c) in &edges { g.out_start[a as usize + 1] += 1; g.in_start[c as usize + 1] += 1; }
    for i in 0..n { g.out_start[i + 1] += g.out_start[i]; g.in_start[i + 1] += g.in_start[i]; }
    g.out_edges = vec![0; edges.len()]; g.in_edges = vec![0; edges.len()];
    let (mut fo, mut fi) = (g.out_start.clone(), g.in_start.clone());
    for (e, &(a, c)) in edges.iter().enumerate() { g.out_edges[fo[a as usize] as usize] = e as u32; fo[a as usize] += 1; g.in_edges[fi[c as usize] as usize] = e as u32; fi[c as usize] += 1; }
    // banned hops: (in, via) -> [out], random
    for _ in 0..300 {
        let via = (rnd() % edges.len() as u64) as usize; let pre = g.from[via] as usize; let post = g.to[via] as usize;
        let ins = g.in_of(pre).to_vec(); let outs = g.out_of(post).to_vec();
        if ins.is_empty() || outs.is_empty() { continue; }
        let inn = ins[(rnd() % ins.len() as u64) as usize]; let out = outs[(rnd() % outs.len() as u64) as usize];
        g.banned_hops.entry(inn as u64 * 4194304 + via as u64).or_default().push(out);
    }
    g.bounds = Bounds { min_x: 0.0, max_x: 1010.0 + n as f64, min_z: 0.0, max_z: 1000.0 };
    g.max_speed = 20.0;
    let e_n = edges.len() as u64;
    let (mut fb, mut cnt, mut none) = (0u64, 0usize, 0usize);
    for weight in [1.25, 1.0] {
        let o = RouterOpts { weight, cache_max: 0, ..Default::default() };
        let mut r0 = RefRouter::new(&g, &o);
        let mut r1 = Router::new(&g, &o);
        let mut r2 = r1.fork(); r2.set_exact_only(true);
        for k in 0..3000 {
            let (s, t) = ((rnd() % n as u64) as usize, (rnd() % n as u64) as usize);
            let avoid: Vec<u32> = if k % 3 == 1 { (0..30).map(|_| (rnd() % e_n) as u32).collect() } else { vec![] };
            let pen: Vec<(u32, f64)> = if k % 3 == 2 { (0..60).map(|_| ((rnd() % e_n) as u32, (rnd() % 4) as f64 * 0.5)).collect() } else { vec![] };
            let q = RouteQuery { avoid: &avoid, penalty: &pen };
            let want = r0.route_with(s, t, &q);
            assert_eq!(r1.route_with(s, t, &q), want, "fast {s}->{t} w {weight} k {k}");
            assert_eq!(r2.route_with(s, t, &q), want, "exact {s}->{t} w {weight} k {k}");
            cnt += 1; if want.is_none() { none += 1; }
        }
        let qs: Vec<(usize, usize, RouteQuery)> = (0..500).map(|_| ((rnd() % n as u64) as usize, (rnd() % n as u64) as usize, RouteQuery::default())).collect();
        let got = r1.route_batch(&qs);
        for (k, &(s, t, _)) in qs.iter().enumerate() { assert_eq!(got[k], r0.route_with(s, t, &RouteQuery::default())); }
        fb += r1.fallbacks() + r1.batch_fallbacks();
    }
    eprintln!("ties: {cnt} queries ({none} no path), {fb} fast-path tie fallbacks");
    assert!(fb > 100, "the grid should produce ties");
}

/// A negative penalty is clamped to 0 (the JS would loop forever on this negative-cost cycle): the query returns
/// promptly with the route of the same query without that penalty. Debug builds assert instead.
#[test]
#[cfg_attr(debug_assertions, should_panic(expected = "negative route penalty"))]
fn negative_penalty_cycle_returns() {
    // 0 <-> 1 <-> 2 line, both directions, 10 m at 10 m/s
    let n = 3;
    let mut g = SimGraph { node_count: n, nx: vec![0.0, 10.0, 20.0], nz: vec![0.0; 3], max_speed: 10.0, ..Default::default() };
    let edges = [(0u32, 1u32), (1, 0), (1, 2), (2, 1)];
    g.edge_count = edges.len();
    for &(a, b) in &edges { g.from.push(a); g.to.push(b); g.length.push(10.0); g.speed_limit.push(10.0); g.rank.push(0); }
    g.out_start = vec![0, 1, 3, 4]; g.out_edges = vec![0, 1, 2, 3];
    g.in_start = vec![0, 1, 3, 4]; g.in_edges = vec![1, 0, 3, 2];
    g.bounds = Bounds { min_x: 0.0, max_x: 20.0, min_z: 0.0, max_z: 0.0 };
    let o = RouterOpts { weight: 1.25, cache_max: 0, ..Default::default() };
    let mut r = Router::new(&g, &o);
    let pen = [(0u32, -100.0), (1, -100.0), (3, f64::NEG_INFINITY)]; // cycle 0 -> 1 -> 0 costs -196 s
    let t = Instant::now();
    let got = r.route_with(0, 2, &RouteQuery { avoid: &[], penalty: &pen });
    assert!(t.elapsed().as_secs_f64() < 1.0);
    assert_eq!(got, Some(vec![0, 2]));
    let qs = [(0usize, 2usize, RouteQuery { avoid: &[], penalty: &pen }); 8];
    assert!(r.route_batch(&qs).iter().all(|p| p.as_deref() == Some(&[0u32, 2][..])));
}
