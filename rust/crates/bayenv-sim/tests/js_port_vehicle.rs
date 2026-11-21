//! Port of test/sim/vehicle.test.js (3 of 3 JS test cases -> 3 #[test]s).
//! The graph is the JS smallGraph() (synthetic grid) exported by tests/common/gen_worlds.mjs. JS roadGraph helpers are mapped to
//! their Rust World equivalents: laneOffset(e, l) -> World.lane_off[l_start[e] + l], edgePoint -> World::edge_point,
//! hopMap -> World::hop, mapLat -> World::map_lat.
mod common;
use bayenv_sim::vehicle::*;

fn alloc(vs: &mut VehicleStore, g: &bayenv_sim::world::World, route: &[i32], rt: Option<f64>, lane: i32) -> usize {
    let mut traits = DEFAULT_TRAITS;
    if let Some(r) = rt { traits.reaction_time = r; }
    vs.alloc(g, AllocArgs { type_idx: 0, traits, route, commuter: -1, leg: 0, time: 0.0, learner: false, dt: 0.05, step: 0, lane })
}

#[test]
fn vehicle_advances_along_edge_and_hops_to_next_route_edge() {
    let g = need!(common::shared_world("synth"));
    let (mut e0, mut e1) = (-1i32, -1i32);
    for e in 0..g.edge_count {
        let to = g.e_to[e] as usize;
        let cand = (g.out_start[to]..g.out_start[to + 1]).map(|q| g.out_edges[q as usize])
            .find(|&k| k != g.e_rev[e] && (g.e_head_start[k as usize] - g.e_head_end[e]).abs() < 0.1);
        if let Some(k) = cand { if g.e_length[e] > 100.0 { e0 = e as i32; e1 = k; break; } }
    }
    assert!(e0 >= 0 && e1 >= 0);
    let mut vs = VehicleStore::new(16);
    let slot = alloc(&mut vs, &g, &[e0, e1], None, 0);
    assert_eq!(vs.state[slot], STATE_DRIVING);
    vs.delay_steps[slot] = 0;
    let len0 = g.e_length[e0 as usize];
    let (mut t, mut hopped, mut arrived) = (0.0, false, 0);
    let mut k = 0i64;
    while k < 20000 && arrived == 0 {
        vs.command(slot, 0.0, 0.4, 0.0, k);
        arrived = vs.integrate(slot, 0.05, t, &g, k, false); t += 0.05;
        vs.update_pose(slot, &g);
        assert!(vs.x[slot].is_finite() && vs.z[slot].is_finite() && vs.speed[slot].is_finite());
        if vs.edge[slot] == e1 && !hopped { hopped = true; assert!(vs.route_dist[slot] as f64 >= len0 - 0.01); assert_eq!(vs.route_pos[slot], 1); }
        k += 1;
    }
    assert!(hopped, "vehicle hopped to the second edge");
    assert_eq!(arrived, 1, "vehicle reached the end of its route");
    assert!(vs.speed[slot] <= vs.max_speed[slot] + 1e-6);
    assert!(vs.fuel[slot] < vs.tank[slot], "fuel is burnt");
}

#[test]
fn lane_offsets_right_hand_traffic_and_hops() {
    let g = need!(common::shared_world("synth"));
    let lane_off = |e: usize, l: usize| g.lane_off[g.l_start[e] as usize + l];
    let tw = (0..g.edge_count).find(|&e| g.e_rev[e] >= 0 && g.e_lanes[e] >= 2).expect("two-way multi-lane edge");
    assert!(lane_off(tw, 0) > lane_off(tw, 1), "lane 0 is further right");
    assert!(lane_off(tw, g.e_lanes[tw] as usize - 1) > 0.0, "all lanes of a two-way edge are right of centre");
    assert!((g.e_right[tw] - (lane_off(tw, 0) + 1.85)).abs() < 1e-6);
    // right of +x is +z: the ported test's hand-made 2-point edge becomes an east-pointing edge of the road3 world
    let r3 = need!(common::shared_world("road3"));
    let p = r3.edge_point(0, 5.0, 2.0, 0);
    assert!((p.z - 2.0).abs() < 1e-9 && (p.x - 5.0).abs() < 1e-9, "right of +x is +z ({}, {})", p.x, p.z);
    // strict hops keep the physical lane, junction hops keep the world position inside the carriageway
    for e in 0..g.edge_count {
        let to = g.e_to[e] as usize;
        for q in g.out_start[to]..g.out_start[to + 1] {
            let en = g.out_edges[q as usize] as usize;
            if en as i32 == g.e_rev[e] { continue; }
            let h = g.hop(e, en);
            let l2 = g.map_lat(e, en, 1.85, 1.0);
            if h.strict { assert!((l2 - (1.85 - 3.7 * h.shift as f64)).abs() < 1e-9); }
            else { assert!(l2 >= 0.0 && l2 <= g.e_lanes[en] as f64 * 3.7); }
        }
    }
}

#[test]
fn controls_go_through_the_reaction_time_ring() {
    let g = need!(common::shared_world("synth"));
    let e = (0..g.edge_count).find(|&x| g.e_length[x] > 300.0).unwrap();
    let mut vs = VehicleStore::new(4);
    let slot = alloc(&mut vs, &g, &[e as i32], Some(0.5), 0);
    assert_eq!(vs.delay_steps[slot], 10);
    assert!(CTRL_RING > vs.delay_steps[slot] as usize);
    let mut step = 0i64;
    macro_rules! tick { ($s:expr, $gas:expr, $br:expr) => {{ vs.command(slot, $s, $gas, $br, step); vs.integrate(slot, 0.05, step as f64 * 0.05, &g, step, false); step += 1; }}; }
    for _ in 0..10 { tick!(0.0, 1.0, 0.0); }
    assert_eq!(vs.speed[slot], 0.0, "no acceleration before the reaction time");
    for _ in 0..50 { tick!(0.0, 1.0, 0.0); }
    assert!(vs.speed[slot] > 3.0 && vs.gas[slot] == 1.0);
    let v_before = vs.speed[slot];
    for _ in 0..10 { tick!(0.0, 0.0, 1.0); }
    assert!(vs.speed[slot] > v_before, "still accelerating during the reaction delay after braking is commanded");
    for _ in 0..100 { tick!(0.0, 0.0, 1.0); }
    assert_eq!(vs.speed[slot], 0.0);
    // steering: the wheel angle follows the commanded steer after the delay and is clamped
    for _ in 0..10 { tick!(1.0, 0.0, 0.0); }
    assert_eq!(vs.steer_angle[slot], 0.0);
    for _ in 0..100 { tick!(1.0, 0.0, 0.0); }
    assert!((vs.steer_angle[slot] - vs.max_steer[slot]).abs() < 1e-6);
    // held control: no new command on a step keeps the last one
    vs.command(slot, 0.0, 1.0, 0.0, step);
    for _ in 0..30 { vs.integrate(slot, 0.05, step as f64 * 0.05, &g, step, false); step += 1; }
    assert!(vs.speed[slot] > 0.0);
}
