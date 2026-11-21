//! Population-generation profile: time in routing vs the rest, routes per commuter, pops per route.
//! usage: pop_prof <world.pack> <n> [region preset] [seed]
use bayenv_sim::population_gen::*;
use bayenv_sim::sim::Router;
use bayenv_sim::world::World;
use std::sync::Arc;
use std::time::Instant;

struct Timed { r: bayenv_world::nav::router::Router, ns: u128, calls: usize, pops: usize, none: usize, len: usize }
impl Router for Timed {
    fn route(&mut self, s: i32, g: i32, _a: Option<&std::collections::HashSet<i32>>, _p: Option<&std::collections::HashMap<i32, f64>>) -> Option<Vec<i32>> {
        let t = Instant::now();
        let r = self.r.route(s as usize, g as usize);
        self.ns += t.elapsed().as_nanos(); self.calls += 1; self.pops += self.r.last_pops;
        match &r { None => self.none += 1, Some(v) => self.len += v.len() }
        r.map(|v| v.into_iter().map(|x| x as i32).collect())
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let w = Arc::new(World::load(&a[1]));
    let n: usize = a[2].parse().unwrap();
    let seed: f64 = a.get(4).map(|s| s.parse().unwrap()).unwrap_or(1.0);
    let mut cfg = PopConfig::env(n, seed);
    cfg.return_trips = std::env::var("POP_RET").map(|v| v != "0").unwrap_or(true);
    if let Some(p) = a.get(3) { if p != "-" { cfg.set_region(Some(Region::preset(p))); } }
    let sg = bayenv_sim::route_stub::sim_graph_of(&w);
    let t0 = Instant::now();
    let ctx = Context::new(&*w, &cfg).unwrap();
    let tctx = t0.elapsed().as_secs_f64();
    if let Ok(t) = std::env::var("POP_THREADS") {
        let t: usize = t.parse().unwrap();
        let t0 = Instant::now();
        let g = generate_population(&w, &cfg, Some(t)).unwrap();
        let el = t0.elapsed().as_secs_f64();
        println!("generate_population n {n} threads {t}: {:.2}s ({:.1} us/commuter wall, {} route edges)", el, el * 1e6 / n as f64, g.pop.route_flat.len());
        return;
    }
    if let Ok(q) = std::env::var("POP_CHECK_NN") {
        // PoolIndex vs ring search on random points (bounds + 20 km), plus points snapped onto nodes (exact ties)
        let q: usize = q.parse().unwrap();
        let (mut lo_x, mut hi_x, mut lo_z, mut hi_z) = (f64::INFINITY, -f64::INFINITY, f64::INFINITY, -f64::INFINITY);
        for i in 0..w.node_count { lo_x = lo_x.min(w.nx[i]); hi_x = hi_x.max(w.nx[i]); lo_z = lo_z.min(w.nz[i]); hi_z = hi_z.max(w.nz[i]); }
        let mut s = 0x9E3779B97F4A7C15u64;
        let mut rnd = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s >> 11) as f64 / (1u64 << 53) as f64 };
        let (mut bad, mut ts, mut tf) = (0, 0.0, 0.0);
        for i in 0..q {
            let (x, z) = if i % 4 == 0 { let k = (rnd() * w.node_count as f64) as usize; (w.nx[k] + if i % 8 == 0 { 0.0 } else { 125.0 }, w.nz[k]) }
                else { (lo_x - 20000.0 + rnd() * (hi_x - lo_x + 40000.0), lo_z - 20000.0 + rnd() * (hi_z - lo_z + 40000.0)) };
            let t = Instant::now(); let a = ctx.nearest.nearest(&*w, x, z, |k| ctx.in_pool[k] == 1); ts += t.elapsed().as_secs_f64();
            let t = Instant::now(); let b = ctx.pool_index.nearest(&ctx.nearest, &*w, x, z); tf += t.elapsed().as_secs_f64();
            if a != b { bad += 1; if bad < 10 { println!("MISMATCH ({x}, {z}): ring {a} kd {b}"); } }
        }
        println!("nearest check: {q} queries, {bad} mismatches; ring {:.1} us/q, kd {:.2} us/q", ts * 1e6 / q as f64, tf * 1e6 / q as f64);
        return;
    }
    let o = bayenv_world::nav::router::RouterOpts { weight: cfg.astar_weight, ..Default::default() };
    let mut r = Timed { r: bayenv_world::nav::router::Router::new(&sg, &o), ns: 0, calls: 0, pops: 0, none: 0, len: 0 };
    let t1 = Instant::now();
    let mut att = 0u64;
    for id in 0..n { let rec = generate_commuter(&*w, &cfg, &ctx, &mut r, id).unwrap(); att += rec.attempts as u64; }
    let tot = t1.elapsed().as_secs_f64();
    let rt = r.ns as f64 * 1e-9;
    println!("nodes {} edges {}  ctx {:.2}s", w.node_count, w.edge_count, tctx);
    println!("n {n}: total {:.2}s ({:.3} ms/commuter); routing {:.2}s ({:.0}%), other {:.2}s", tot, tot * 1e3 / n as f64, rt, 100.0 * rt / tot, tot - rt);
    println!("routes {} ({:.2}/commuter, {} none), attempts {:.2}/commuter, pops/route {:.0}, us/route {:.1}, ns/pop {:.1}, path edges/route {:.0}",
        r.calls, r.calls as f64 / n as f64, r.none, att as f64 / n as f64, r.pops as f64 / r.calls as f64, rt * 1e6 / r.calls as f64, r.ns as f64 / r.pops as f64, r.len as f64 / r.calls as f64);
}
