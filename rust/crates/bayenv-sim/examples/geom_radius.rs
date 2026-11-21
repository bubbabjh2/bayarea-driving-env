//! Minimum-radius census of the geometry the car dynamics follow (curvature_dynamics, DECISIONS.md 23 / 33).
//!   cargo run --release -p bayenv-sim --example geom_radius -- <data dir> [--smooth]
//!
//! The radius at a vertex is that of the largest fillet arc that fits between its neighbouring segments:
//! r = min(half of each adjacent segment) / tan(turn / 2). On a sampled arc of radius R this is R cos(step / 2).
//!
//! 1. "centreline": the SimGraph polylines (px / pz), vertices turning > 0.05 rad (the census of DECISIONS.md 23).
//! 2. The frame the dynamics follow (vertices turning > 0.01 rad):
//!    "lane": the lane-geometry reference polylines (LaneGeometry vx / vz), radius at the innermost lane centre
//!       (reference radius minus the lane-centre offset on the inside of the turn);
//!    "join": the vertex between consecutive edges at a pass-through node (no junction box, no other edge);
//!    "connector": the junction connector polylines (JunctionSet.pts).
//!    With --smooth the world is loaded with LoadOptions::smooth_geometry (bayenv_world::smooth).
//! 3. Continuity: lane-point jumps at edge -> edge hops (strict lane map, nodes without a junction box) and at
//!    connector entry / exit, frame heading jumps there, and s consistency.
use bayenv_world::{LoadOptions, World};
use std::collections::BTreeMap;
use std::path::Path;

fn fillet(l1: f64, l2: f64, turn: f64) -> f64 { if turn.abs() < 1e-9 { f64::INFINITY } else { 0.5 * l1.min(l2) / (turn.abs() / 2.0).tan() } }
fn wrap(a: f64) -> f64 { a.sin().atan2(a.cos()) }
const BINS: [f64; 9] = [3.0, 5.0, 10.0, 15.0, 30.0, 60.0, 120.0, 250.0, 500.0];

#[derive(Default)]
struct Hist(BTreeMap<String, (u64, Vec<u64>)>);
impl Hist {
    fn add(&mut self, k: String, r: f64) { let h = self.0.entry(k).or_insert((0, vec![0; BINS.len()])); h.0 += 1; for (b, &lim) in BINS.iter().enumerate() { if r < lim { h.1[b] += 1; } } }
    fn print(&self, title: &str, min_n: u64) {
        println!("{:40} {:>9} {}", title, "count", BINS.iter().map(|b| format!("{:>8}", format!("<{b}m"))).collect::<String>());
        for (k, (n, h)) in &self.0 { if *n < min_n { continue; } println!("{k:40} {n:>9} {}", h.iter().map(|&c| format!("{:>7.1}%", 100.0 * c as f64 / *n as f64)).collect::<String>()); }
    }
}

/// The connector heading the sim uses (bayenv-sim world.rs `connector_point`: segment headings blended towards the
/// neighbouring segments' at the vertices).
fn conn_heading(j: &bayenv_world::JunctionSet, m: usize, t: f64) -> f64 {
    let (first, n) = (j.m_info[7 * m + 4] as usize, j.m_info[7 * m + 5] as usize);
    let p = |k: usize, o: usize| j.pts[4 * (first + k) + o] as f64;
    let mut k = 0; while k + 2 < n && p(k + 1, 3) < t { k += 1; }
    let d = { let d = p(k + 1, 3) - p(k, 3); if d == 0.0 { 1.0 } else { d } };
    let u = ((t - p(k, 3)) / d).clamp(0.0, 1.0);
    let dir = |a: usize, b: usize| { let (x, z) = (p(b, 0) - p(a, 0), p(b, 2) - p(a, 2)); let l = x.hypot(z).max(1e-12); (x / l, z / l) };
    let c = dir(k, k + 1);
    let (mut a, mut b) = (c, c);
    if k > 0 { let q = dir(k - 1, k); a = (a.0 + q.0, a.1 + q.1); }
    if k + 2 < n { let q = dir(k + 1, k + 2); b = (b.0 + q.0, b.1 + q.1); }
    let (al, bl) = (a.0.hypot(a.1).max(1e-12), b.0.hypot(b.1).max(1e-12));
    let (hx, hz) = ((1.0 - u) * a.0 / al + u * b.0 / bl, (1.0 - u) * a.1 / al + u * b.1 / bl);
    hz.atan2(hx)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let dir = a.get(1).filter(|s| !s.starts_with("--")).map(|s| s.as_str()).unwrap_or("../data");
    let smooth = a.iter().any(|s| s == "--smooth");
    let o = LoadOptions { verify: false, smooth_geometry: smooth, ..Default::default() };
    let t0 = std::time::Instant::now();
    let w = World::load_with(Path::new(dir), &o).expect("world");
    println!("world {dir}: {} edges, smoothed: {smooth} (load {:.1} s)", w.graph.edge_count, t0.elapsed().as_secs_f64());
    if let Some(st) = &w.smoothing { println!("smoothing: {st:#?}"); }
    let g = &w.graph; let lg = &w.lanes; let lgr = &w.lane_graph;
    let cname = |e: usize| bayenv_world::roadpack::CLASSES_ALL.get(g.class[e] as usize).copied().unwrap_or("?").to_string();

    // ---- 1. SimGraph centrelines (unchanged by smoothing)
    if !smooth {
        let mut h = Hist::default();
        let seg = |e: usize, k: usize| { let v = g.pt_start[e] as usize + k; let (dx, dz) = (g.px[v + 1] - g.px[v], g.pz[v + 1] - g.pz[v]); ((dx * dx + dz * dz).sqrt(), dz.atan2(dx)) };
        let nseg = |e: usize| (g.pt_start[e + 1] - g.pt_start[e]) as usize - 1;
        for e in 0..g.edge_count {
            for k in 1..nseg(e) { let ((l1, h1), (l2, h2)) = (seg(e, k - 1), seg(e, k)); let t = wrap(h2 - h1); if t.abs() > 0.05 { h.add(format!("centreline {}", cname(e)), fillet(l1, l2, t)); } }
        }
        h.print("1. SimGraph centreline vertices (> 0.05 rad)", 50);
    }

    // ---- 2. frame geometry
    let mut h = Hist::default();
    let jnodes: Vec<u32> = w.junctions.as_ref().map(|j| j.j_node.to_vec()).unwrap_or_default();
    let joins = bayenv_world::smooth::pass_through_joins(lgr, &jnodes);
    let vseg = |a: usize, b: usize| { let (dx, dz) = (lg.vx[b] - lg.vx[a], lg.vz[b] - lg.vz[a]); ((dx * dx + dz * dz).sqrt(), dz.atan2(dx)) };
    let inner = |e: usize, t: f64| { let w = lg.lane_w[e] as f64; let rd = lg.rd[e] as f64; let n = lgr.lanes[e].max(1) as f64; if t > 0.0 { (rd - 0.5 * w).max(0.0) } else { (-(rd - n * w + 0.5 * w)).max(0.0) } };
    let mut n_lane = 0u64;
    for e in 0..g.edge_count {
        let (v0, v1) = (lg.v_first[e] as usize, lg.v_first[e + 1] as usize);
        for v in v0 + 1..v1 - 1 {
            let ((l1, h1), (l2, h2)) = (vseg(v - 1, v), vseg(v, v + 1)); let t = wrap(h2 - h1);
            if t.abs() > 0.01 { n_lane += 1; h.add(format!("lane {}", cname(e)), (fillet(l1, l2, t) - inner(e, t)).max(0.0)); }
        }
    }
    for &(ea, eb) in &joins {
        let (ea, eb) = (ea as usize, eb as usize);
        let (a1, b0) = (lg.v_first[ea + 1] as usize - 1, lg.v_first[eb] as usize);
        let ((l1, h1), (l2, h2)) = (vseg(a1 - 1, a1), vseg(b0, b0 + 1)); let t = wrap(h2 - h1);
        if t.abs() > 0.01 { h.add(format!("join {}", cname(ea)), (fillet(l1, l2, t) - inner(ea, t).max(inner(eb, t))).max(0.0)); }
    }
    if let Some(j) = &w.junctions {
        for m in 0..j.move_count() {
            let (first, n) = (j.m_info[7 * m + 4] as usize, j.m_info[7 * m + 5] as usize);
            let p = |k: usize| (j.pts[4 * (first + k)] as f64, j.pts[4 * (first + k) + 2] as f64);
            for k in 1..n.saturating_sub(1) {
                let (a0, a1, a2) = (p(k - 1), p(k), p(k + 1));
                let (l1, h1) = ((a1.0 - a0.0).hypot(a1.1 - a0.1), (a1.1 - a0.1).atan2(a1.0 - a0.0));
                let (l2, h2) = ((a2.0 - a1.0).hypot(a2.1 - a1.1), (a2.1 - a1.1).atan2(a2.0 - a1.0));
                let t = wrap(h2 - h1);
                if t.abs() > 0.01 { h.add(format!("connector {}", cname(j.m_info[7 * m] as usize)), fillet(l1, l2, t)); }
            }
        }
    }
    h.print(&format!("2. frame geometry{} (> 0.01 rad)", if smooth { ", smoothed" } else { ", raw" }), 50);
    println!("lane vertices turning > 0.01 rad: {n_lane}; joins: {}", joins.len());

    // ---- 2b. heading rate along the path (sampling-independent): r = W / |heading change over W|, W = 2 m, sampled
    //      every metre of lane reference line (radius at the innermost lane centre), across each join (the metre
    //      before and after the node) and every 0.5 m of connector (its blended `connector_point` heading); shares are of path length. Per connector
    //      turning > 0.3 rad: its tightest r.
    {
        #[derive(Default)] struct Lh(BTreeMap<String, (f64, Vec<f64>)>);
        impl Lh { fn add(&mut self, k: String, r: f64, w: f64) { let h = self.0.entry(k).or_insert((0.0, vec![0.0; BINS.len()])); h.0 += w; for (b, &lim) in BINS.iter().enumerate() { if r < lim { h.1[b] += w; } } } }
        let mut lh = Lh::default();
        let head = |e: usize, s: f64| { let p = lg.offset_point(e, s, 0.0, 0.0); p.hz.atan2(p.hx) };
        for e in 0..g.edge_count {
            let l = lgr.length[e] as f64; if l < 2.0 { continue; }
            let mut s = 1.0;
            while s <= l - 1.0 { let d = wrap(head(e, s + 1.0) - head(e, s - 1.0)); let r = if d.abs() < 1e-9 { f64::INFINITY } else { (2.0 / d.abs() - inner(e, d)).max(0.0) }; lh.add(format!("lane {}", cname(e)), r, 1.0); s += 1.0; }
        }
        for &(ea, eb) in &joins {
            let (ea, eb) = (ea as usize, eb as usize); let la = lgr.length[ea] as f64;
            if la < 1.0 || (lgr.length[eb] as f64) < 1.0 { continue; }
            let d = wrap(head(eb, 1.0) - head(ea, la - 1.0)); let r = if d.abs() < 1e-9 { f64::INFINITY } else { (2.0 / d.abs() - inner(ea, d).max(inner(eb, d))).max(0.0) };
            lh.add(format!("join {}", cname(ea)), r, 1.0);
        }
        let mut cmin = Hist::default();
        if let Some(j) = &w.junctions {
            for m in 0..j.move_count() {
                let len = j.m_f[4 * m + 2] as f64; if len < 2.0 { continue; }
                let hc = |t: f64| conn_heading(j, m, t);
                let (mut t, mut rmin) = (1.0, f64::INFINITY);
                while t <= len - 1.0 { let d = wrap(hc(t + 1.0) - hc(t - 1.0)); let r = if d.abs() < 1e-9 { f64::INFINITY } else { 2.0 / d.abs() }; rmin = rmin.min(r); lh.add(format!("connector {}", cname(j.m_info[7 * m] as usize)), r, 0.5); t += 0.5; }
                if (j.m_f[4 * m + 3] as f64).abs() > 0.3 { cmin.add(format!("connector-min {}", cname(j.m_info[7 * m] as usize)), rmin); }
            }
        }
        println!("{:40} {:>9} {}", format!("2b. heading rate{} (km of path)", if smooth { ", smoothed" } else { ", raw" }), "km", BINS.iter().map(|b| format!("{:>8}", format!("<{b}m"))).collect::<String>());
        for (k, (n, h)) in &lh.0 { if *n < 50.0 { continue; } println!("{k:40} {:>9.1} {}", n / 1e3, h.iter().map(|&c| format!("{:>7.2}%", 100.0 * c / *n)).collect::<String>()); }
        cmin.print("2c. tightest point per connector (> 0.3 rad)", 50);
    }

    // ---- 3. continuity and s consistency
    let mut s_bad = 0usize;
    for e in 0..g.edge_count {
        let (v0, v1) = (lg.v_first[e] as usize, lg.v_first[e + 1] as usize - 1);
        let bad = lg.vs[v0] != 0.0 || (lg.vs[v1] - lgr.length[e] as f64).abs() > 1e-6 * (1.0 + lg.vs[v1]) || (v0..v1).any(|v| !(lg.vs[v + 1] > lg.vs[v]));
        if bad { s_bad += 1; if s_bad <= 3 { println!("   s-bad edge {e}: length {} vs {:?}", lgr.length[e], &lg.vs[v0..=v1.min(v0 + 12)]); } }
    }
    let ht = bayenv_world::hops::HopTable::new(lgr);
    let mut jsorted = jnodes.clone(); jsorted.sort_unstable();
    let (mut hop_n, mut hop_jump, mut hop_turn, mut hop_j5, mut hop_t1) = (0usize, 0f64, 0f64, 0usize, 0usize);
    for n in 0..lgr.node_count {
        if jsorted.binary_search(&(n as u32)).is_ok() { continue; }
        for &ea in lgr.in_edges(n) { for &eb in lgr.out_edges(n) {
            let (ea, eb) = (ea as usize, eb as usize);
            if eb as i32 == lgr.twin[ea] { continue; }
            let hp = ht.get(lgr, ea, eb); if !hp.strict { continue; }
            let la = lgr.length[ea] as f64;
            for l in 0..lgr.lanes[ea] as i32 {
                let l2 = l - hp.shift; if l2 < 0 || l2 >= lgr.lanes[eb] as i32 { continue; }
                let p = lg.lat_point(ea, la, (l as f64 + 0.5) * lg.lane_width(ea, la));
                let q = lg.lat_point(eb, 0.0, (l2 as f64 + 0.5) * lg.lane_width(eb, 0.0));
                let (dj, dh) = ((p.x - q.x).hypot(p.z - q.z), wrap(q.hz.atan2(q.hx) - p.hz.atan2(p.hx)).abs());
                hop_n += 1; hop_jump = hop_jump.max(dj); hop_turn = hop_turn.max(dh);
                if dj > 0.05 { hop_j5 += 1; if hop_j5 <= 3 { println!("   hop jump {dj:.3} m: edge {ea} lane {l} -> {eb} lane {l2} (shift {})", hp.shift); } } if dh > 0.1 { hop_t1 += 1; }
            }
        } }
    }
    let (mut c_in, mut c_out, mut c_hin, mut c_hout, mut c_s) = (0f64, 0f64, 0f64, 0f64, 0usize);
    if let Some(j) = &w.junctions {
        for m in 0..j.move_count() {
            let d = j.describe(m);
            if d.length < 1.0 { continue; } // degenerate (e.g. a 0.6 m self-loop turn-back)
            let pa = lg.lat_point(d.in_edge as usize, d.s_in as f64, (d.in_lane as f64 + 0.5) * lg.lane_width(d.in_edge as usize, d.s_in as f64));
            let pb = lg.lat_point(d.out_edge as usize, d.s_out as f64, (d.out_lane as f64 + 0.5) * lg.lane_width(d.out_edge as usize, d.s_out as f64));
            let (a0, a1) = (j.arc_point(m, 0.0), j.arc_point(m, d.length as f64));
            c_in = c_in.max((a0.x - pa.x).hypot(a0.z - pa.z)); c_out = c_out.max((a1.x - pb.x).hypot(a1.z - pb.z));
            let dh = wrap(a0.hz.atan2(a0.hx) - pa.hz.atan2(pa.hx)).abs(); if dh > c_hin && dh > 0.2 { println!("   connector {m} entry heading {dh:.3}: {:?}", d); }
            c_hin = c_hin.max(wrap(a0.hz.atan2(a0.hx) - pa.hz.atan2(pa.hx)).abs()); let dh1 = wrap(a1.hz.atan2(a1.hx) - pb.hz.atan2(pb.hx)).abs(); if dh1 > c_hout && dh1 > 0.2 { println!("   connector {m} exit heading {dh1:.3}: {:?}", d); } c_hout = c_hout.max(dh1);
            let (first, n) = (d.first_pt as usize, d.pt_count as usize);
            if (j.pts[4 * (first + n - 1) + 3] - d.length).abs() > 1e-3 || j.pts[4 * first + 3] != 0.0 { c_s += 1; }
            if (0..n - 1).any(|k| !(j.pts[4 * (first + k + 1) + 3] > j.pts[4 * (first + k) + 3])) { c_s += 1; if c_s <= 2 { println!("   connector {m} s: {:?}", (0..n.min(12)).map(|k| j.pts[4 * (first + k) + 3]).collect::<Vec<_>>()); } }
        }
    }
    println!("3. continuity: {hop_n} strict lane hops at nodes without a junction box: max lane-point jump {hop_jump:.4} m ({hop_j5} > 5 cm), max frame heading jump {hop_turn:.3} rad ({hop_t1} > 0.1 rad)");
    println!("   connectors: max entry jump {c_in:.4} m / heading {c_hin:.4} rad, exit jump {c_out:.4} m / heading {c_hout:.4} rad");
    println!("   s: edges with vs[0] != 0, vs[last] != length or non-increasing vs: {s_bad}; connectors with s not 0 .. length increasing: {c_s}");
}
