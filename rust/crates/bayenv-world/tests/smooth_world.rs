//! The smoothed frame geometry (bayenv_world::smooth, DECISIONS.md 33) on the real world: s consistency, continuity
//! at edge -> edge hops and connector entry / exit, small sideways moves on straight-ish roads, and the radii it
//! removes. Skips without the world data.
use bayenv_world::{LoadOptions, World};
use std::path::Path;

fn wrap(a: f64) -> f64 { a.sin().atan2(a.cos()) }

fn load(smooth: bool) -> Option<World> { load_jr(smooth, false) }
fn load_jr(smooth: bool, jr: bool) -> Option<World> {
    let d = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../data");
    match World::load_with(&d, &LoadOptions { verify: false, smooth_geometry: smooth, junction_radii: jr, ..Default::default() }) {
        Ok(w) => Some(w),
        Err(e) => { bayenv_world::oracle::skip("smooth_world", &format!("world: {e}")); None }
    }
}

/// share of lane reference-line length (sampled every metre, window 2 m) of class `cls` whose heading rate means a
/// radius below `r` at the innermost lane centre
fn tight_share(w: &World, cls: u8, r: f64) -> f64 {
    let (lg, g) = (&w.lanes, &w.lane_graph);
    let head = |e: usize, s: f64| { let p = lg.offset_point(e, s, 0.0, 0.0); p.hz.atan2(p.hx) };
    let (mut n, mut k) = (0u64, 0u64);
    for e in 0..g.edge_count {
        if g.cls[e] != cls { continue; }
        let l = g.length[e] as f64; let mut s = 1.0;
        let (wd, rd, nl) = (lg.lane_w[e] as f64, lg.rd[e] as f64, g.lanes[e].max(1) as f64);
        while s <= l - 1.0 {
            let d = wrap(head(e, s + 1.0) - head(e, s - 1.0));
            let inner = if d > 0.0 { (rd - 0.5 * wd).max(0.0) } else { (-(rd - nl * wd + 0.5 * wd)).max(0.0) };
            n += 1; if d.abs() > 1e-9 && 2.0 / d.abs() - inner < r { k += 1; }
            s += 1.0;
        }
    }
    k as f64 / n.max(1) as f64
}

#[test]
fn smoothed_world_is_consistent_and_continuous() {
    let Some(raw) = load(false) else { return };
    let w = load(true).unwrap();
    let st = w.smoothing.clone().expect("smoothed");
    eprintln!("{st:#?}");
    let (lg, g, rl) = (&w.lanes, &w.lane_graph, &raw.lanes);
    // s: every edge runs 0 .. length, strictly increasing; the right-edge / width arrays are untouched
    for e in 0..g.edge_count {
        let (v0, v1) = (lg.v_first[e] as usize, lg.v_first[e + 1] as usize - 1);
        assert!(v1 > v0, "edge {e} keeps >= 2 vertices");
        assert_eq!(lg.vs[v0], 0.0, "edge {e} starts at s 0");
        assert_eq!(lg.vs[v1], rl.vs[rl.v_first[e + 1] as usize - 1], "edge {e} ends at its length");
        assert!((v0..v1).all(|v| lg.vs[v + 1] > lg.vs[v]), "edge {e}: s increases");
    }
    assert_eq!(&lg.rd[..], &rl.rd[..]); assert_eq!(&lg.rend[..], &rl.rend[..]); assert_eq!(&lg.lane_w[..], &rl.lane_w[..]);
    // junction data other than the connector points is untouched; connectors run 0 .. length in s
    let (j, rj) = (w.junctions.as_ref().unwrap(), raw.junctions.as_ref().unwrap());
    assert_eq!(&j.m_f[..], &rj.m_f[..]); assert_eq!(&j.mouth_s[..], &rj.mouth_s[..]); assert_eq!(&j.conf_s[..], &rj.conf_s[..]);
    let (mut entry, mut exit, mut hin, mut hout) = (0f64, 0f64, 0f64, 0f64);
    for m in 0..j.move_count() {
        let d = j.describe(m);
        let (f, n) = (d.first_pt as usize, d.pt_count as usize);
        assert_eq!(j.pts[4 * f + 3], 0.0); assert!((j.pts[4 * (f + n - 1) + 3] - rj.pts[4 * (rj.m_info[7 * m + 4] + rj.m_info[7 * m + 5] - 1) as usize + 3]).abs() < 1e-4, "connector {m} keeps its length");
        assert!((0..n - 1).all(|k| j.pts[4 * (f + k + 1) + 3] > j.pts[4 * (f + k) + 3]), "connector {m}: s increases");
        if d.length < 1.0 { continue; } // degenerate self-loop turn-backs
        // entry / exit on the smoothed lane centres, tangent to them
        let pa = lg.lat_point(d.in_edge as usize, d.s_in as f64, (d.in_lane as f64 + 0.5) * lg.lane_width(d.in_edge as usize, d.s_in as f64));
        let pb = lg.lat_point(d.out_edge as usize, d.s_out as f64, (d.out_lane as f64 + 0.5) * lg.lane_width(d.out_edge as usize, d.s_out as f64));
        let (a0, a1) = (j.arc_point(m, 0.0), j.arc_point(m, d.length as f64));
        entry = entry.max((a0.x - pa.x).hypot(a0.z - pa.z)); exit = exit.max((a1.x - pb.x).hypot(a1.z - pb.z));
        hin = hin.max(wrap(a0.hz.atan2(a0.hx) - pa.hz.atan2(pa.hx)).abs()); hout = hout.max(wrap(a1.hz.atan2(a1.hx) - pb.hz.atan2(pb.hx)).abs());
    }
    eprintln!("connector entry {entry:.4} m / {hin:.3} rad, exit {exit:.4} m / {hout:.3} rad");
    assert!(entry < 0.01 && exit < 0.01, "connector ends on the lanes: {entry} {exit}");
    assert!(hin < 0.12 && hout < 0.12, "connector ends tangent to the lanes: {hin} {hout}");
    // strict lane hops at nodes without a junction box: no new position jumps, and none at all at smoothed joins
    let ht = bayenv_world::hops::HopTable::new(g);
    let mut jn = vec![false; g.node_count]; for &n in &j.j_node[..] { jn[n as usize] = true; }
    let joins: std::collections::HashSet<(u32, u32)> = bayenv_world::smooth::pass_through_joins(g, &j.j_node).into_iter().collect();
    let (mut worse, mut join_max, mut hops) = (0usize, 0f64, 0usize);
    for n in 0..g.node_count {
        if jn[n] { continue; }
        for &a in g.in_edges(n) { for &b in g.out_edges(n) {
            let (a, b) = (a as usize, b as usize);
            if b as i32 == g.twin[a] { continue; }
            let h = ht.get(g, a, b); if !h.strict { continue; }
            let la = g.length[a] as f64;
            for l in 0..g.lanes[a] as i32 {
                let l2 = l - h.shift; if l2 < 0 || l2 >= g.lanes[b] as i32 { continue; }
                let jump = |lg: &bayenv_world::LaneGeometry| { let p = lg.lat_point(a, la, (l as f64 + 0.5) * lg.lane_width(a, la)); let q = lg.lat_point(b, 0.0, (l2 as f64 + 0.5) * lg.lane_width(b, 0.0)); (p.x - q.x).hypot(p.z - q.z) };
                let (js, jr) = (jump(lg), jump(rl));
                hops += 1;
                if js > jr + 1e-6 { worse += 1; }
                if joins.contains(&(a as u32, b as u32)) { join_max = join_max.max(js); }
            }
        } }
    }
    eprintln!("{hops} strict lane hops: {worse} with a larger jump than raw; max jump at a smoothed join {join_max:.2e} m");
    assert_eq!(worse, 0, "no hop gets a position jump");
    assert!(join_max < 1e-6, "smoothed joins are continuous: {join_max}");
    // straight-ish roads (fillets turning < 0.1 rad) move at most dev_max sideways; no fillet more than dev_hard
    let cfg = bayenv_world::smooth::SmoothCfg::default();
    assert!(st.max_dev_straightish <= cfg.dev_max + 1e-6, "{}", st.max_dev_straightish);
    assert!(st.max_dev <= cfg.dev_hard + 1e-6, "{}", st.max_dev);
    // tight radii removed: residential (11) lane length under 5 m / 10 m at the innermost lane centre
    let (r5, s5, r10, s10) = (tight_share(&raw, 11, 5.0), tight_share(&w, 11, 5.0), tight_share(&raw, 11, 10.0), tight_share(&w, 11, 10.0));
    eprintln!("residential lane length under 5 m: {:.3} % -> {:.3} %; under 10 m: {:.3} % -> {:.3} %", 100.0 * r5, 100.0 * s5, 100.0 * r10, 100.0 * s10);
    assert!(s5 < 0.5 * r5 && s10 < r10, "smoothing removes tight radii");
    // deterministic: a second smoothing gives the same arrays
    let w2 = load(true).unwrap();
    assert_eq!(&w2.lanes.vx[..], &lg.vx[..]); assert_eq!(&w2.junctions.as_ref().unwrap().pts[..], &j.pts[..]);
}

/// junction_radii (DECISIONS.md 34): re-fitted turning connectors start and end on the smoothed lanes at their new
/// s_in / s_out, tangent to them, with s running 0 .. length (= mF length); s_in / s_out stay on their edges; mouths,
/// turn angles and the conflict list are unchanged and conflict positions lie on the connectors; far fewer turning
/// connectors are below 6 m.
#[test]
fn junction_radii_consistent() {
    let Some(sm) = load(true) else { return };
    let w = load_jr(true, true).unwrap();
    let st = w.junction_radii.clone().expect("re-fitted");
    eprintln!("{st:#?}");
    let (lg, g) = (&w.lanes, &w.lane_graph);
    let (j, j0) = (w.junctions.as_ref().unwrap(), sm.junctions.as_ref().unwrap());
    assert_eq!(&j.mouth_s[..], &j0.mouth_s[..]); assert_eq!(&j.gore_s[..], &j0.gore_s[..]); assert_eq!(&j.conf[..], &j0.conf[..]);
    let (mut moved, mut entry, mut hin, mut exit, mut hout) = (0usize, 0f64, 0f64, 0f64, 0f64);
    for m in 0..j.move_count() {
        let (d, d0) = (j.describe(m), j0.describe(m));
        assert_eq!(d.turn, d0.turn); assert_eq!((d.in_edge, d.in_lane, d.out_edge, d.out_lane), (d0.in_edge, d0.in_lane, d0.out_edge, d0.out_lane));
        let (f, n) = (d.first_pt as usize, d.pt_count as usize);
        assert_eq!(j.pts[4 * f + 3], 0.0);
        assert!((j.pts[4 * (f + n - 1) + 3] - d.length).abs() < 1e-3, "movement {m}: length = last point s");
        assert!((0..n - 1).all(|k| j.pts[4 * (f + k + 1) + 3] > j.pts[4 * (f + k) + 3]), "movement {m}: s increases");
        assert!(d.s_in >= 0.0 && d.s_in <= g.length[d.in_edge as usize] && d.s_out >= 0.0 && d.s_out <= g.length[d.out_edge as usize]);
        if d.s_in == d0.s_in && d.s_out == d0.s_out { continue; }
        moved += 1;
        assert!(d.s_in <= d0.s_in && d.s_out >= d0.s_out, "movement {m}: only moves s_in back and s_out forward");
        let pa = lg.lat_point(d.in_edge as usize, d.s_in as f64, (d.in_lane as f64 + 0.5) * lg.lane_width(d.in_edge as usize, d.s_in as f64));
        let pb = lg.lat_point(d.out_edge as usize, d.s_out as f64, (d.out_lane as f64 + 0.5) * lg.lane_width(d.out_edge as usize, d.s_out as f64));
        let (a0, a1) = (j.arc_point(m, 0.0), j.arc_point(m, d.length as f64));
        entry = entry.max((a0.x - pa.x).hypot(a0.z - pa.z)); exit = exit.max((a1.x - pb.x).hypot(a1.z - pb.z));
        hin = hin.max(wrap(a0.hz.atan2(a0.hx) - pa.hz.atan2(pa.hx)).abs()); hout = hout.max(wrap(a1.hz.atan2(a1.hx) - pb.hz.atan2(pb.hx)).abs());
    }
    for c in 0..j.conflict_count() {
        let (a, b) = (j.conf[3 * c] as usize, j.conf[3 * c + 1] as usize);
        assert!(j.conf_s[2 * c] <= j.m_f[4 * a + 2] + 0.5 && j.conf_s[2 * c + 1] <= j.m_f[4 * b + 2] + 0.5, "conflict {c} on its connectors");
    }
    eprintln!("{moved} movements moved; entry {entry:.4} m / {hin:.3} rad, exit {exit:.4} m / {hout:.3} rad");
    assert!(moved > 100_000 && moved == st.at_target + st.fit_limited);
    // (heading: the 0.3 m lead segments in f32 coordinates)
    assert!(entry < 0.01 && exit < 0.01 && hin < 0.03 && hout < 0.03, "re-fitted connectors start / end on the lanes, tangent");
    assert!(st.below_floor * 2 < st.below_floor_before, "below 6 m: {} -> {}", st.below_floor_before, st.below_floor);
}
