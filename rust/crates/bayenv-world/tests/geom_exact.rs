//! Bit-exact checks of the geometry builders against the JS reference.
//!  * lane geometry vs `rust/oracle/lg.*` (`node --max-old-space-size=14000 rust/tools/oracle/dump_world.mjs data rust/oracle lg`)
//!  * junctions vs `rust/oracle/ref/junctions.bin` (a copy of `npm run build:junctions` output), when present
//! Skipped (with a message) when the data or the oracle files are missing.
//! Run: `cargo test --release -p bayenv-world --test geom_exact -- --nocapture --test-threads 1`
use bayenv_world::blob::Pack;
use bayenv_world::lanegraph::graph_from_road_sets;
use bayenv_world::RoadSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn root() -> PathBuf { Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..").canonicalize().unwrap() }

fn bits_eq<T: bytemuck::Pod>(a: &[T], b: &[T]) -> (usize, Option<usize>) {
    let (ab, bb): (&[u8], &[u8]) = (bytemuck::cast_slice(a), bytemuck::cast_slice(b));
    let sz = std::mem::size_of::<T>();
    let n = a.len().min(b.len());
    let mut bad = if a.len() != b.len() { 1 } else { 0 }; let mut first = None;
    for i in 0..n { if ab[i * sz..(i + 1) * sz] != bb[i * sz..(i + 1) * sz] { bad += 1; if first.is_none() { first = Some(i); } } }
    (bad, first)
}

#[test]
fn lane_geometry_matches_js() {
    let data = root().join("data"); let oracle = root().join("rust/oracle");
    if !data.join("roads.meta.json").exists() || !oracle.join("lg.meta.json").exists() { eprintln!("skip: no data / oracle"); return; }
    let main = RoadSet::load(&data, "roads.meta.json").unwrap();
    let local = RoadSet::load(&data, "roads-local.meta.json").unwrap();
    let g = graph_from_road_sets(&main, Some(&local));
    let t = Instant::now();
    let (lg, stats) = bayenv_world::lanes_build::build_lane_geometry_stats(&g);
    eprintln!("buildLaneGeometry (rust): {:.0} ms; {:?}", t.elapsed().as_secs_f64() * 1e3, stats);
    let p = Pack::open(&oracle, "lg.meta.json").unwrap();
    eprintln!("js stats: {}", p.meta["stats"]);
    let mut fail = 0;
    macro_rules! c { ($name:literal, $t:ty, $v:expr) => {{
        let js = p.req::<$t>($name).unwrap(); let (bad, first) = bits_eq::<$t>(&js, &$v);
        if bad > 0 { fail += 1; }
        eprintln!("{} lg.{:10} {:>9} elems, {} mismatches{}", if bad == 0 { "OK  " } else { "FAIL" }, $name, js.len(), bad,
            first.map_or(String::new(), |i| format!(" (first at {i}: js {:?} rust {:?})", js[i], $v[i])));
    }}; }
    c!("vFirst", u32, lg.v_first); c!("vx", f64, lg.vx); c!("vy", f32, lg.vy); c!("vz", f64, lg.vz); c!("vs", f64, lg.vs);
    c!("mx", f64, lg.mx); c!("mz", f64, lg.mz); c!("Rd", f32, lg.rd); c!("Rend", f32, lg.rend); c!("tau", f32, lg.tau);
    c!("tapR", u8, lg.tap_r); c!("tapL", u8, lg.tap_l); c!("xR", u8, lg.x_r); c!("xL", u8, lg.x_l);
    c!("laneW", f32, lg.lane_w); c!("laneWEnd", f32, lg.lane_w_end);
    assert_eq!(fail, 0, "lane geometry differs from the JS dump");
}

#[test]
fn junctions_match_js_file() {
    let data = root().join("data"); let refd = root().join("rust/oracle/ref");
    if !data.join("roads.meta.json").exists() || !refd.join("junctions.bin").exists() || !data.join("controls.json").exists() {
        eprintln!("skip: no data / rust/oracle/ref/junctions.bin (copy of npm run build:junctions output) / data/controls.json"); return;
    }
    use bayenv_world::jsjson;
    use bayenv_world::junctions_build::{build_junctions_full, crosswalk_legs, ControlLegs};
    let main = RoadSet::load(&data, "roads.meta.json").unwrap();
    let local = RoadSet::load(&data, "roads-local.meta.json").unwrap();
    let g = graph_from_road_sets(&main, Some(&local));
    let lg = bayenv_world::lanes_build::build_lane_geometry(&g);
    // legs as tools/build_junctions.mjs computes them (the JSON of the reference build: rust/oracle/ref/controls.json)
    let ctl = jsjson::parse(&std::fs::read_to_string(refd.join("controls.json")).unwrap()).unwrap();
    let cw = jsjson::parse(&std::fs::read_to_string(data.join("crosswalks.json")).unwrap()).unwrap();
    let controls: Vec<ControlLegs> = ctl.get("controls").unwrap().as_arr().unwrap().iter().map(|c| ControlLegs {
        node: c.get("node").and_then(|v| v.as_f64()).unwrap_or(f64::NAN), ty: c.get("type").and_then(|v| v.as_str()).unwrap_or("").into(),
        all_way: c.get("allWay").map_or(false, |v| v.truthy()),
        approach_edges: c.get("approaches").unwrap().as_arr().unwrap().iter().filter_map(|a| a.get("edge").and_then(|v| v.as_f64())).filter(|&e| e >= 0.0).map(|e| e as u32).collect(),
    }).collect();
    let points: Vec<[f64; 2]> = cw.get("points").unwrap().as_arr().unwrap().iter().map(|p| { let a = p.as_arr().unwrap(); [a[0].as_f64().unwrap(), a[1].as_f64().unwrap()] }).collect();
    let legs = crosswalk_legs(&g, Some(&controls), Some(&points));
    let t = Instant::now();
    let (j, st) = build_junctions_full(&lg, &g, Some(&legs));
    eprintln!("buildJunctions (rust): {:.0} ms; {:?}", t.elapsed().as_secs_f64() * 1e3, st);
    let p = Pack::open(&refd, "junctions.meta.json").unwrap();
    let mut fail = 0;
    macro_rules! c { ($name:literal, $t:ty, $v:expr) => {{
        let js = p.req::<$t>($name).unwrap(); let (bad, first) = bits_eq::<$t>(&js, &$v);
        if bad > 0 { fail += 1; }
        eprintln!("{} j.{:8} {:>9} elems, {} mismatches{}", if bad == 0 { "OK  " } else { "FAIL" }, $name, js.len(), bad, first.map_or(String::new(), |i| format!(" (first at {i})")));
    }}; }
    c!("mouthS", f32, j.mouth_s); c!("goreS", f32, j.gore_s); c!("jNode", u32, j.j_node); c!("jFirstM", u32, j.j_first_m); c!("jCountM", u32, j.j_count_m);
    c!("jFirstC", u32, j.j_first_c); c!("jCountC", u32, j.j_count_c); c!("jFirstH", u32, j.j_first_h); c!("jCountH", u32, j.j_count_h);
    c!("mInfo", u32, j.m_info); c!("mF", f32, j.m_f); c!("pts", f32, j.pts); c!("conf", u32, j.conf); c!("confS", f32, j.conf_s); c!("hull", f32, j.hull);
    assert_eq!(p.meta["crossingFix"], j.crossing_fix.clone().unwrap(), "crossingFix stats");
    assert_eq!(std::fs::read(refd.join("junctions.bin")).unwrap(), j.to_bin(), "junctions.bin bytes");
    assert_eq!(fail, 0);
}
