//! cargo run --release -p bayenv-sim --bin world_check -- <data dir> <world.pack exported by JS>
//! Compares World::from_bayenv(bayenv_world::World::load(dir)) with the JS snapshot, array by array.
use bayenv_sim::world::World;
fn cmp<T: PartialEq + std::fmt::Debug + Copy>(name: &str, a: &[T], b: &[T], bad: &mut usize) {
    if a.len() != b.len() { println!("{name}: length {} vs {}", a.len(), b.len()); *bad += 1; return; }
    let n = a.iter().zip(b).filter(|(x, y)| x != y && !(format!("{x:?}") == "NaN" && format!("{y:?}") == "NaN")).count();
    if n > 0 { let k = a.iter().zip(b).position(|(x, y)| x != y).unwrap(); println!("{name}: {n} mismatches, first [{k}] {:?} vs {:?}", a[k], b[k]); *bad += 1; }
}
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let t = std::time::Instant::now();
    let bw = bayenv_world::World::load(&a[1]).expect("load");
    let t1 = t.elapsed().as_secs_f64();
    let w = World::from_bayenv(&bw);
    let t2 = t.elapsed().as_secs_f64();
    let p = bayenv_sim::pack::Pack::read(&a[2]).unwrap();
    let r = World::from_pack(&p);
    println!("bayenv_world load {:.2}s, convert {:.2}s", t1, t2 - t1);
    let mut bad = 0;
    macro_rules! c { ($($f:ident),*) => { $(cmp(stringify!($f), &w.$f, &r.$f, &mut bad);)* }; }
    c!(nx, nz, node_y, e_from, e_to, e_rev, e_lanes, e_opp, e_speed, e_length, e_rank, e_hw, e_is_link, e_has_ys, e_head_start, e_head_end, e_y, e_right,
       e_s_exit, e_s_stop, e_gore, e_surf, e_gnd_r, e_gnd_l, v_start, px, pz, cum, l_start, lane_off, out_start, out_edges, in_start, in_edges);
    // ys only matters where has_ys
    let ysm = (0..w.edge_count).filter(|&e| w.e_has_ys[e]).flat_map(|e| w.v_start[e]..w.v_start[e + 1]).filter(|&v| w.ys[v as usize] != r.ys[v as usize]).count();
    if ysm > 0 { println!("ys: {ysm} mismatches"); bad += 1; }
    let names = |x: &World| (0..x.edge_count).map(|e| x.highway(e).to_string()).collect::<Vec<_>>();
    if names(&w) != names(&r) { println!("highway names differ"); bad += 1; }
    let (a, b) = (w.lg.as_ref().unwrap(), r.lg.as_ref().unwrap());
    macro_rules! l { ($($f:ident),*) => { $(cmp(concat!("lg.", stringify!($f)), &a.$f, &b.$f, &mut bad);)* }; }
    let flat = |x: &bayenv_sim::world::LaneGeom| x.vtx.iter().flatten().copied().collect::<Vec<f64>>();
    cmp("lg.vtx", &flat(a), &flat(b), &mut bad);
    l!(v_first, vs, rd, rend, tau, x_r, x_l, lane_w, lane_w_end, tap_r, tap_l, g_length, g_lanes, g_opp, g_twin, g_cls, g_first, g_count, g_poly, g_xyz, g_out_start, g_out_list, g_in_start, g_in_list);
    match (&w.j, &r.j) { (Some(a), Some(b)) => { macro_rules! j { ($($f:ident),*) => { $(cmp(concat!("j.", stringify!($f)), &a.$f, &b.$f, &mut bad);)* }; } j!(mouth_s, gore_s, j_node, j_first_m, j_count_m, m_info, m_f, m_len, pts, j_of_node, conf_start); }
        _ => { println!("junctions presence differs"); bad += 1; } }
    match (&w.controls, &r.controls) { (Some(a), Some(b)) => { macro_rules! k { ($($f:ident),*) => { $(cmp(concat!("c.", stringify!($f)), &a.$f, &b.$f, &mut bad);)* }; } k!(ty, flags, cycle, offset, ap_control, ap_stop_at, ap_edge, ap_group, ap_prot_left, cluster_of, edge_ap_start, edge_ap_list, phase_start);
        let pa: Vec<f64> = a.phases.iter().flatten().copied().collect(); let pb: Vec<f64> = b.phases.iter().flatten().copied().collect(); cmp("c.phases", &pa, &pb, &mut bad); }
        _ => { println!("controls presence differs"); bad += 1; } }
    let sa: Vec<(i32, i32, f64, f64)> = w.stations.iter().map(|s| (s.node, s.pumps, s.x, s.z)).collect();
    let sb: Vec<(i32, i32, f64, f64)> = r.stations.iter().map(|s| (s.node, s.pumps, s.x, s.z)).collect();
    cmp("stations", &sa, &sb, &mut bad);
    if w.contact_pairs != r.contact_pairs { println!("contact pairs differ"); bad += 1; }
    if w.banned_hops != r.banned_hops { println!("banned hops differ"); bad += 1; }
    if w.surfaces != r.surfaces { println!("surfaces flag differs"); bad += 1; }
    println!("hop table vs JS hopRef: {} mismatches", w.check_hops(&p));
    println!("{}", if bad == 0 { "WORLD OK" } else { "WORLD MISMATCH" });
}
