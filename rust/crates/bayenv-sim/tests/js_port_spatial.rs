//! Port of test/sim/spatial.test.js: 2 of 3 JS test cases ported (2 #[test]s).
//!   ported : spatial hash neighbour queries match brute force; footprint lane index (ahead/behind, neighborsAt, stale buckets).
//!   skipped: "raycast hits the nearest vehicle box along the ray" - SpatialHash.raycast / rayRect are not in the Rust crate
//!            (the Rust obs port senses bodies through sense_bodies instead; no standalone raycast).
//! The JS store is filled without a graph; Rust VehicleStore::alloc needs a World (edges 0..50 must exist), so the synthetic grid is used.
mod common;
use bayenv_sim::rng::Rng;
use bayenv_sim::spatial::*;
use bayenv_sim::vehicle::*;
use bayenv_sim::world::World;

fn random_store(g: &World, n: usize, seed: f64) -> VehicleStore {
    let mut rng = Rng::new(seed);
    let mut vs = VehicleStore::new(n);
    for _ in 0..n {
        let e = rng.int(50) as i32;
        let s = vs.alloc(g, AllocArgs { type_idx: 0, traits: DEFAULT_TRAITS, route: &[e], commuter: -1, leg: 0, time: 0.0, learner: false, dt: 0.05, step: 0, lane: 0 });
        vs.x[s] = rng.range(-500.0, 500.0) as f32; vs.z[s] = rng.range(-500.0, 500.0) as f32; vs.heading[s] = rng.range(-std::f64::consts::PI, std::f64::consts::PI) as f32;
        vs.edge[s] = rng.int(50) as i32; vs.lane[s] = rng.int(3) as i8; vs.s[s] = rng.range(0.0, 400.0) as f32;
        vs.lane_lo[s] = vs.lane[s];
        let straddle = if rng.random() < 0.3 { 1 } else { 0 };
        vs.lane_hi[s] = 3.min(vs.lane[s] + straddle); // some straddle two lanes
    }
    vs
}

#[test]
fn spatial_hash_neighbour_queries_match_brute_force() {
    let g = need!(common::shared_world("synth"));
    let vs = random_store(&g, 1500, 1.0);
    let mut hash = SpatialHash::new(25.0, 14, 1500);
    hash.rebuild(&vs);
    let mut rng = Rng::new(9.0);
    let mut out = Vec::new();
    for _ in 0..200 {
        let (x, z, r) = (rng.range(-520.0, 520.0), rng.range(-520.0, 520.0), rng.range(5.0, 80.0));
        hash.near_into(&vs.x, &vs.z, x, z, r, &mut out);
        let mut got = out.clone(); got.sort();
        let want: Vec<usize> = (0..vs.high).filter(|&i| (vs.x[i] as f64 - x).powi(2) + (vs.z[i] as f64 - z).powi(2) <= r * r).collect();
        assert_eq!(got, want);
    }
}

#[test]
fn footprint_lane_index_lists_each_overlapped_lane_sorted_by_s() {
    let g = need!(common::shared_world("synth"));
    let mut vs = random_store(&g, 800, 3.0);
    let mut li = LaneIndex::new(60, 800);
    li.rebuild(&vs);
    let in_lane = |vs: &VehicleStore, j: usize, l: i32| l >= vs.lane_lo[j] as i32 && l <= vs.lane_hi[j] as i32;
    let mut straddlers = 0;
    for i in 0..vs.high {
        if vs.lane_hi[i] > vs.lane_lo[i] { straddlers += 1; }
        for l in vs.lane_lo[i] as i32..=vs.lane_hi[i] as i32 {
            let (a, b) = (li.ahead(&vs, i, l), li.behind(&vs, i, l));
            let (mut best_a, mut best_b) = (-1i32, -1i32);
            for j in 0..vs.high {
                if j == i || vs.edge[j] != vs.edge[i] || !in_lane(&vs, j, l) { continue; }
                let after = vs.s[j] > vs.s[i] || (vs.s[j] == vs.s[i] && j > i);
                if after { if best_a < 0 || vs.s[j] < vs.s[best_a as usize] || (vs.s[j] == vs.s[best_a as usize] && (j as i32) < best_a) { best_a = j as i32; } }
                else if best_b < 0 || vs.s[j] > vs.s[best_b as usize] || (vs.s[j] == vs.s[best_b as usize] && j as i32 > best_b) { best_b = j as i32; }
            }
            assert_eq!(a, best_a); assert_eq!(b, best_b);
        }
    }
    assert!(straddlers > 100);
    let s0 = vs.s[0] as f64 + 0.001;
    let (ah, bh) = li.neighbors_at(&vs, vs.edge[0], vs.lane[0] as i32, s0, -1);
    if ah >= 0 { assert!(vs.s[ah as usize] as f64 >= s0); }
    if bh >= 0 { assert!((vs.s[bh as usize] as f64) < s0); }
    // rebuild after moving everyone to other keys leaves no stale buckets
    for i in 0..vs.high { vs.edge[i] = 55; }
    li.rebuild(&vs);
    for e in 0..50 { for l in 0..4 { assert_eq!(li.first(e, l), -1); } }
}
