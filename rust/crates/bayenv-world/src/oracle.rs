//! Comparison against JS oracle dumps (`rust/tools/oracle/dump_world.mjs`): element-for-element, bit-exact.
use crate::blob::Pack;
use crate::world::World;
use std::path::Path;

/// One comparison result: (name, elements compared, mismatches, first mismatch index).
pub type Cmp = (String, usize, usize, Option<usize>);

fn cmp<T: PartialEq + Copy + bytemuck::Pod + Bits>(name: &str, js: &[T], rs: &[T]) -> Cmp {
    let n = js.len().min(rs.len()); let mut bad = if js.len() != rs.len() { 1 } else { 0 }; let mut first = None;
    for i in 0..n { if js[i].bits() != rs[i].bits() { bad += 1; if first.is_none() { first = Some(i); } } }
    (name.to_string(), n, bad, first)
}
pub trait Bits { fn bits(self) -> u64; }
macro_rules! bits { ($($t:ty => $e:expr),*) => { $( impl Bits for $t { #[inline] fn bits(self) -> u64 { let f: fn($t) -> u64 = $e; f(self) } } )* } }
bits!(f64 => |x| if x.is_nan() { u64::MAX } else { x.to_bits() }, f32 => |x| if x.is_nan() { u64::MAX } else { x.to_bits() as u64 },
      u32 => |x| x as u64, i32 => |x| x as u32 as u64, u8 => |x| x as u64, i8 => |x| x as u8 as u64);

/// Compare the loaded world with `<oracle>/graph.*` and `<oracle>/controls.*`.
pub fn check_world(w: &World, oracle: &Path) -> std::io::Result<Vec<Cmp>> {
    let mut out = Vec::new();
    if let Ok(p) = Pack::open(oracle, "graph.meta.json") {
        let g = &w.graph;
        macro_rules! c { ($name:literal, $t:ty, $v:expr) => { out.push(cmp::<$t>($name, &p.req::<$t>($name)?, &$v)); } }
        c!("from", u32, g.from); c!("to", u32, g.to); c!("reverse", i32, g.reverse); c!("lanes", u8, g.lanes); c!("rank", u8, g.rank); c!("navRank", u8, g.nav_rank);
        c!("speedLimit", f64, g.speed_limit); c!("length", f64, g.length); c!("headingStart", f64, g.heading_start); c!("headingEnd", f64, g.heading_end);
        c!("y", f32, g.y); c!("rightEdge", f64, g.right_edge); c!("surf", u8, g.surf); c!("gndR", u8, g.gnd_r); c!("gndL", u8, g.gnd_l);
        c!("sStop", f32, g.s_stop); c!("sExit", f32, g.s_exit); c!("gore", f32, g.gore); c!("ptStart", u32, g.pt_start);
        c!("px", f64, g.px); c!("pz", f64, g.pz); c!("cum", f64, g.cum); c!("ys", f32, g.ys); c!("laneStart", u32, g.lane_start); c!("laneOff", f32, g.lane_off);
        c!("outStart", u32, g.out_start); c!("outEdges", u32, g.out_edges); c!("inStart", u32, g.in_start); c!("inEdges", u32, g.in_edges);
        c!("nx", f64, g.nx); c!("nz", f64, g.nz); c!("nodeY", f32, g.node_y);
        let st: Vec<u32> = w.stations.iter().map(|s| s.node as u32).collect(); c!("stationNode", u32, st);
    }
    if let (Ok(p), Some(cc)) = (Pack::open(oracle, "controls.meta.json"), &w.controls) {
        macro_rules! c { ($name:literal, $t:ty, $v:expr) => { out.push(cmp::<$t>($name, &p.req::<$t>($name)?, &$v)); } }
        c!("type", i8, cc.ty); c!("cycle", f32, cc.cycle); c!("offset", f32, cc.offset); c!("apGroup", i8, cc.ap_group); c!("apProtLeft", u8, cc.ap_prot_left);
        c!("edgeApStart", i32, cc.edge_ap_start); c!("edgeApList", i32, cc.edge_ap_list); c!("clusterOf", i32, cc.cluster_of);
        let n = cc.count; let mut ph = vec![f64::NAN; n * 16]; let mut pn = vec![0u8; n];
        for c in 0..n { pn[c] = cc.phases[c].len() as u8; for (j, q) in cc.phases[c].iter().take(4).enumerate() { ph[16 * c + 4 * j] = q.g; ph[16 * c + 4 * j + 1] = q.y; ph[16 * c + 4 * j + 2] = q.r; } }
        c!("phases", f64, ph); c!("phN", u8, pn);
        let ts: Vec<f64> = p.meta["stateTimes"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        let na = cc.approach_count; let mut s = vec![0f64; na * 27]; let mut ss = vec![0u8; na * 9];
        for a in 0..na { if cc.ap_edge[a] < 0 { continue; } for ti in 0..3 { for m in 0..3 { let o = cc.state_of(a, m as u8, ts[ti]); let q = a * 9 + ti * 3 + m; ss[q] = o.state; s[3 * q] = o.time_in; s[3 * q + 1] = o.time_left; } } }
        // stateT slot 3*q+2 is unused (0) in the dump
        c!("stateT", f64, s); c!("stateS", u8, ss);
    }
    if let Ok(p) = Pack::open(oracle, "lg.meta.json") {
        let l = &w.lanes;
        macro_rules! c { ($name:literal, $t:ty, $v:expr) => { out.push(cmp::<$t>(concat!("lg.", $name), &p.req::<$t>($name)?, &$v)); } }
        c!("vFirst", u32, l.v_first); c!("vx", f64, l.vx); c!("vy", f32, l.vy); c!("vz", f64, l.vz); c!("vs", f64, l.vs); c!("mx", f64, l.mx); c!("mz", f64, l.mz);
        c!("Rd", f32, l.rd); c!("Rend", f32, l.rend); c!("tau", f32, l.tau); c!("tapR", u8, l.tap_r); c!("tapL", u8, l.tap_l); c!("xR", u8, l.x_r); c!("xL", u8, l.x_l);
        c!("laneW", f32, l.lane_w); c!("laneWEnd", f32, l.lane_w_end);
    }
    Ok(out)
}

/// Seed the lane-geometry bake from a JS dump (`<oracle>/lg.*`) until the Rust builder is complete.
pub fn import_lanes(data: &Path, oracle: &Path, bake: &Path) -> std::io::Result<()> {
    let p = Pack::open(oracle, "lg.meta.json")?;
    let lg = crate::lanes::LaneGeometry {
        v_first: p.req("vFirst")?, vx: p.req("vx")?, vy: p.req("vy")?, vz: p.req("vz")?, vs: p.req("vs")?, mx: p.req("mx")?, mz: p.req("mz")?,
        rd: p.req("Rd")?, rend: p.req("Rend")?, tau: p.req("tau")?, tap_r: p.req("tapR")?, tap_l: p.req("tapL")?, x_r: p.req("xR")?, x_l: p.req("xL")?,
        lane_w: p.req("laneW")?, lane_w_end: p.req("laneWEnd")?, lw: p.req::<f64>("LW")?[0], ..Default::default()
    };
    crate::world::write_lanes_bake(bake, crate::world::lanes_bake_key(data)?, &lg)
}

/// `rust/oracle` of this checkout (gitignored; `rust/tools/make_oracle.sh` generates it from the JS with node).
pub fn oracle_root() -> std::path::PathBuf { Path::new(env!("CARGO_MANIFEST_DIR")).join("../../oracle") }

/// An oracle-dependent test that cannot run: prints `SKIPPED <test>: <reason>` straight to the terminal (stderr,
/// bypassing the test harness's output capture, so it shows without --nocapture) and returns, or panics when
/// `BAYENV_REQUIRE_ORACLE=1` (a skipped JS-equivalence check must not count as a pass there).
pub fn skip(test: &str, reason: &str) {
    use std::io::Write;
    let msg = format!("SKIPPED {test}: {reason} (generate the JS oracle with rust/tools/make_oracle.sh)\n");
    let _ = std::io::stderr().write_all(msg.as_bytes());
    if std::env::var("BAYENV_REQUIRE_ORACLE").as_deref() == Ok("1") { panic!("BAYENV_REQUIRE_ORACLE=1: {}", msg.trim_end()); }
}

/// Subdirectories of `dir` holding `file` (sorted); empty if `dir` is missing.
pub fn dirs_with(dir: &Path, file: &str) -> Vec<std::path::PathBuf> {
    let mut v: Vec<_> = std::fs::read_dir(dir).map(|r| r.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.join(file).exists()).collect()).unwrap_or_default();
    v.sort(); v
}
