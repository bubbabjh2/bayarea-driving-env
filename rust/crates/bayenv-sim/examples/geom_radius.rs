//! Minimum-radius census of the geometry the car dynamics follow (curvature_dynamics, DECISIONS.md 23):
//! edge centreline vertices, edge-to-edge joins without a junction, and junction connector polylines.
//! The radius at a vertex is that of the largest fillet arc that fits between its neighbouring segments:
//! r = min(half of each adjacent segment) / tan(turn / 2).
//!   cargo run --release -p bayenv-sim --example geom_radius -- <data dir>
use bayenv_sim::env::Env;
use std::collections::BTreeMap;
use std::path::Path;

fn fillet(l1: f64, l2: f64, turn: f64) -> f64 { if turn.abs() < 1e-6 { f64::INFINITY } else { 0.5 * l1.min(l2) / (turn.abs() / 2.0).tan() } }
fn wrap(a: f64) -> f64 { a.sin().atan2(a.cos()) }

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let g = Env::load_world_unverified(Path::new(a.get(1).map(|s| s.as_str()).unwrap_or("../data"))).expect("world");
    let bins = [3.0, 5.0, 10.0, 15.0, 30.0, 60.0, 120.0, 250.0];
    // per class: vertex count with turn > 0.05 rad, histogram of fillet radius
    let mut hist: BTreeMap<String, (u64, Vec<u64>)> = BTreeMap::new();
    let mut add = |k: String, r: f64| { let h = hist.entry(k).or_insert((0, vec![0; bins.len()])); h.0 += 1; for (b, &lim) in bins.iter().enumerate() { if r < lim { h.1[b] += 1; } } };
    let name = |e: usize| g.highway_names.get(g.e_class[e] as usize).cloned().unwrap_or_default();
    let seg = |e: usize, k: usize| { let v = g.v_start[e] as usize + k; let (dx, dz) = (g.px[v + 1] - g.px[v], g.pz[v + 1] - g.pz[v]); ((dx * dx + dz * dz).sqrt(), dz.atan2(dx)) };
    let nseg = |e: usize| (g.v_start[e + 1] - g.v_start[e]) as usize - 1;
    for e in 0..g.edge_count {
        let n = nseg(e);
        for k in 1..n {
            let ((l1, h1), (l2, h2)) = (seg(e, k - 1), seg(e, k));
            let t = wrap(h2 - h1);
            if t.abs() > 0.05 { add(format!("vertex {}", name(e)), fillet(l1, l2, t)); }
        }
        // joins to successor edges at nodes without a junction
        let node = g.e_to[e] as usize;
        let jn = g.j.as_ref().map_or(false, |j| j.j_of_node.get(node).copied().unwrap_or(-1) >= 0);
        if !jn && n > 0 {
            for q in g.out_start[node]..g.out_start[node + 1] {
                let f = g.out_edges[q as usize] as usize;
                if f as i32 == g.e_rev[e] || nseg(f) == 0 { continue; }
                let ((l1, h1), (l2, h2)) = (seg(e, n - 1), seg(f, 0));
                let t = wrap(h2 - h1);
                if t.abs() > 0.05 { add(format!("join {}", name(e)), fillet(l1, l2, t)); }
            }
        }
    }
    if let Some(j) = &g.j {
        for m in 0..j.move_count {
            let (first, n) = (j.info(m, 4) as usize, j.info(m, 5) as usize);
            let p = |k: usize| (j.pts[4 * (first + k)], j.pts[4 * (first + k) + 2]);
            for k in 1..n.saturating_sub(1) {
                let (a0, a1, a2) = (p(k - 1), p(k), p(k + 1));
                let (l1, h1) = (((a1.0 - a0.0).powi(2) + (a1.1 - a0.1).powi(2)).sqrt(), (a1.1 - a0.1).atan2(a1.0 - a0.0));
                let (l2, h2) = (((a2.0 - a1.0).powi(2) + (a2.1 - a1.1).powi(2)).sqrt(), (a2.1 - a1.1).atan2(a2.0 - a1.0));
                let t = wrap(h2 - h1);
                if t.abs() > 0.05 { add(format!("connector {}", name(j.info(m, 0) as usize)), fillet(l1, l2, t)); }
            }
        }
    }
    println!("{:34} {:>9} {}", "kind / class (turn > 0.05 rad)", "count", bins.iter().map(|b| format!("{:>8}", format!("<{b}m"))).collect::<String>());
    for (k, (n, h)) in &hist {
        if *n < 50 { continue; }
        println!("{k:34} {n:>9} {}", h.iter().map(|&c| format!("{:>7.1}%", 100.0 * c as f64 / *n as f64)).collect::<String>());
    }
}
