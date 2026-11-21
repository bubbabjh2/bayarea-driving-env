//! `World`: everything static the CPU and GPU sims read, loaded from `data/` (port of `src/sim/data.js loadWorld`).
//!
//! Fast path: a bake (`<data>/build/rust/world.bake`, see [`crate::bake`]) holds the derived arrays that are
//! expensive to compute (lane geometry; junctions when `data/junctions.bin` is absent). It is keyed by an
//! xxh3 of its inputs and rebuilt automatically when they change.
use crate::bake::{files_key, Bake, BakeWriter};
use crate::controls::{ControlsPack, TrafficControls};
use crate::extras::{attach_surfaces, load_stations, ContactPairs, Dem, NodeGrid, Station};
use crate::graph::SimGraph;
use crate::junctions::JunctionSet;
use crate::lanegraph::{graph_from_road_sets, LaneGraph};
use crate::lanes::LaneGeometry;
use crate::roadpack::RoadSet;
use std::{io, path::{Path, PathBuf}, time::Instant};

#[derive(Clone, Debug)]
pub struct LoadOptions {
    pub local: bool, pub junctions: bool, pub controls: bool, pub stations: bool, pub dem: bool,
    /// where the bake lives (default `<data>/build/rust/world.bake`); None = no bake read / write
    pub bake: Option<PathBuf>,
    pub verify_bake: bool,
}
impl Default for LoadOptions {
    fn default() -> Self { LoadOptions { local: true, junctions: true, controls: true, stations: true, dem: false, bake: None, verify_bake: false } }
}

pub struct World {
    pub dir: PathBuf,
    pub main: RoadSet,
    pub local: Option<RoadSet>,
    pub graph: SimGraph,
    pub lane_graph: LaneGraph,
    pub lanes: LaneGeometry,
    pub junctions: Option<JunctionSet>,
    pub surfaces: bool,
    pub controls_pack: Option<ControlsPack>,
    pub controls: Option<TrafficControls>,
    pub stations: Vec<Station>,
    pub contact_pairs: Option<ContactPairs>,
    pub dem: Option<Dem>,
    pub node_grid: NodeGrid,
    /// load timings (ms) per stage
    pub timings: Vec<(&'static str, f64)>,
}

pub const LG_SECTIONS: [&str; 17] = ["vFirst", "vx", "vy", "vz", "vs", "mx", "mz", "Rd", "Rend", "tau", "tapR", "tapL", "xR", "xL", "laneW", "laneWEnd", "LW"];

impl World {
    pub fn load(dir: impl AsRef<Path>) -> io::Result<World> { Self::load_with(dir, &LoadOptions::default()) }

    pub fn load_with(dir: impl AsRef<Path>, o: &LoadOptions) -> io::Result<World> {
        let dir = dir.as_ref().to_path_buf(); let mut tm = Vec::new(); let mut t = Instant::now();
        let mut lap = |name: &'static str, tm: &mut Vec<(&'static str, f64)>| { tm.push((name, t.elapsed().as_secs_f64() * 1e3)); t = Instant::now(); };
        let main = RoadSet::load(&dir, "roads.meta.json")?;
        let local = if o.local { RoadSet::load(&dir, "roads-local.meta.json").ok() } else { None };
        lap("packs", &mut tm);
        let mut graph = SimGraph::from_road_sets(&main, local.as_ref());
        let lane_graph = graph_from_road_sets(&main, local.as_ref());
        lap("graph", &mut tm);
        let bake_path = o.bake.clone().unwrap_or_else(|| dir.join("build/rust").join(if local.is_some() { "world.bake" } else { "world-main.bake" }));
        let key = files_key(&[&dir.join("roads.bin"), &dir.join("roads-local.bin")])?;
        let lanes = match Bake::open(&bake_path, o.verify_bake) {
            Ok(b) if b.key == key && b.has("lg.vx") => lanes_from_bake(&b, &lane_graph)?,
            _ => {
                let lg = LaneGeometry::build(&lane_graph);
                let _ = write_lanes_bake(&bake_path, key, &lg);
                lg
            }
        };
        lap("lanes", &mut tm);
        let junctions = if o.junctions {
            match JunctionSet::load(&dir) {
                Ok(j) if j.mouth_s.len() == 2 * graph.edge_count => Some(j),
                _ => Some(junctions_from_bake_or_build(&dir, &bake_path.with_file_name("junctions.bake"), &lanes, &lane_graph)?),
            }
        } else { None };
        if let Some(j) = &junctions { for e in 0..graph.edge_count { graph.s_exit[e] = j.mouth_s[2 * e]; graph.s_stop[e] = j.mouth_s[2 * e + 1]; graph.gore[e] = j.gore_s[e]; } }
        lap("junctions", &mut tm);
        let surfaces = attach_surfaces(&dir, &mut graph, main.edge_count, local.as_ref().map(|l| l.edge_count)).unwrap_or(false);
        let node_grid = NodeGrid::new(&graph);
        let stations = if o.stations { load_stations(&dir, &graph, &node_grid).unwrap_or_default() } else { Vec::new() };
        let contact_pairs = ContactPairs::load(&dir, main.edge_count).ok().flatten();
        lap("surfaces+stations+crossings", &mut tm);
        let (controls_pack, controls) = if o.controls {
            match ControlsPack::load(&dir) {
                Ok(p) if p.main_edge_count() == Some(main.edge_count as u64) => { let c = TrafficControls::new(&graph, &p); (Some(p), Some(c)) }
                _ => (None, None),
            }
        } else { (None, None) };
        lap("controls", &mut tm);
        let dem = if o.dem { Dem::load(&dir).ok() } else { None };
        lap("dem", &mut tm);
        Ok(World { dir, main, local, graph, lane_graph, lanes, junctions, surfaces, controls_pack, controls, stations, contact_pairs, dem, node_grid, timings: tm })
    }
}

fn lanes_from_bake(b: &Bake, g: &LaneGraph) -> io::Result<LaneGeometry> {
    Ok(LaneGeometry {
        v_first: b.get("lg.vFirst")?, vx: b.get("lg.vx")?, vy: b.get("lg.vy")?, vz: b.get("lg.vz")?, vs: b.get("lg.vs")?, mx: b.get("lg.mx")?, mz: b.get("lg.mz")?,
        rd: b.get("lg.Rd")?, rend: b.get("lg.Rend")?, tau: b.get("lg.tau")?, tap_r: b.get("lg.tapR")?, tap_l: b.get("lg.tapL")?,
        x_r: b.get("lg.xR")?, x_l: b.get("lg.xL")?, lane_w: b.get("lg.laneW")?, lane_w_end: b.get("lg.laneWEnd")?,
        lw: b.get::<f64>("lg.LW")?.first().copied().unwrap_or(3.7),
        length: g.length.clone().into(), lanes: g.lanes.clone().into(),
    })
}

/// Write the lane-geometry bake (also used by `rust/tools/oracle` to seed it from the JS builder).
pub fn write_lanes_bake(path: &Path, key: u64, lg: &LaneGeometry) -> io::Result<()> {
    let mut w = BakeWriter::default();
    w.put("lg.vFirst", &lg.v_first).put("lg.vx", &lg.vx).put("lg.vy", &lg.vy).put("lg.vz", &lg.vz).put("lg.vs", &lg.vs)
        .put("lg.mx", &lg.mx).put("lg.mz", &lg.mz).put("lg.Rd", &lg.rd).put("lg.Rend", &lg.rend).put("lg.tau", &lg.tau)
        .put("lg.tapR", &lg.tap_r).put("lg.tapL", &lg.tap_l).put("lg.xR", &lg.x_r).put("lg.xL", &lg.x_l)
        .put("lg.laneW", &lg.lane_w).put("lg.laneWEnd", &lg.lane_w_end).put("lg.LW", &[lg.lw]);
    w.write(path, key)
}

/// The bake key for a data dir (inputs of the lane geometry).
pub fn lanes_bake_key(dir: &Path) -> io::Result<u64> { files_key(&[&dir.join("roads.bin"), &dir.join("roads-local.bin")]) }

/// No (matching) `data/junctions.bin`: build the junctions as `npm run build:junctions` does (crosswalk legs from
/// controls.json + crosswalks.json; the same bytes as the file) and keep them in a bake keyed by those inputs.
/// (The JS loader's in-memory fallback builds without crosswalk legs; this builds what the file would contain.)
fn junctions_from_bake_or_build(dir: &Path, path: &Path, lanes: &LaneGeometry, g: &LaneGraph) -> io::Result<JunctionSet> {
    let key = files_key(&[&dir.join("roads.bin"), &dir.join("roads-local.bin"), &dir.join("controls.json"), &dir.join("crosswalks.json")])?;
    const S: [&str; 15] = crate::junctions::JUNCTION_SECTIONS;
    if let Ok(b) = Bake::open(path, false) { if b.key == key && b.has("j.mouthS") {
        let f = |k: &str| b.get::<f32>(&format!("j.{k}")); let u = |k: &str| b.get::<u32>(&format!("j.{k}"));
        return Ok(JunctionSet { mouth_s: f(S[0])?, gore_s: f(S[1])?, j_node: u(S[2])?, j_first_m: u(S[3])?, j_count_m: u(S[4])?, j_first_c: u(S[5])?, j_count_c: u(S[6])?,
            j_first_h: u(S[7])?, j_count_h: u(S[8])?, m_info: u(S[9])?, m_f: f(S[10])?, pts: f(S[11])?, conf: u(S[12])?, conf_s: f(S[13])?, hull: f(S[14])?, crossing_fix: None });
    } }
    let legs = crosswalk_legs_from_files(dir, g);
    let j = crate::junctions::build_junctions(lanes, g, Some(&legs));
    let mut w = BakeWriter::default();
    w.put("j.mouthS", &j.mouth_s).put("j.goreS", &j.gore_s).put("j.jNode", &j.j_node).put("j.jFirstM", &j.j_first_m).put("j.jCountM", &j.j_count_m)
        .put("j.jFirstC", &j.j_first_c).put("j.jCountC", &j.j_count_c).put("j.jFirstH", &j.j_first_h).put("j.jCountH", &j.j_count_h).put("j.mInfo", &j.m_info)
        .put("j.mF", &j.m_f).put("j.pts", &j.pts).put("j.conf", &j.conf).put("j.confS", &j.conf_s).put("j.hull", &j.hull);
    let _ = w.write(path, key);
    Ok(j.into_set())
}

/// `crosswalkLegs(LG, controls.json controls, crosswalks.json points)` as tools/build_junctions.mjs feeds it.
pub fn crosswalk_legs_from_files(dir: &Path, g: &LaneGraph) -> crate::junctions::CrosswalkLegs {
    use crate::jsjson;
    use crate::junctions_build::{crosswalk_legs, ControlLegs};
    let read = |p: PathBuf| std::fs::read_to_string(&p).ok().and_then(|t| jsjson::parse(&t).ok());
    let ctl_doc = read(dir.join("controls.json")); let cw_doc = read(dir.join("crosswalks.json"));
    let controls: Option<Vec<ControlLegs>> = ctl_doc.as_ref().and_then(|d| d.get("controls")).and_then(|c| c.as_arr()).map(|cs| cs.iter().map(|c| ControlLegs {
        node: c.get("node").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
        ty: c.get("type").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        all_way: c.get("allWay").map_or(false, |v| v.truthy()),
        approach_edges: c.get("approaches").and_then(|a| a.as_arr()).map(|a| a.iter().filter_map(|ap| ap.get("edge").and_then(|v| v.as_f64())).filter(|&e| e >= 0.0).map(|e| e as u32).collect()).unwrap_or_default(),
    }).collect());
    let points: Option<Vec<[f64; 2]>> = cw_doc.as_ref().and_then(|d| d.get("points")).and_then(|p| p.as_arr())
        .map(|ps| ps.iter().map(|p| { let a = p.as_arr().unwrap(); [a[0].as_f64().unwrap_or(f64::NAN), a.get(1).and_then(|v| v.as_f64()).unwrap_or(f64::NAN)] }).collect());
    crosswalk_legs(g, controls.as_deref(), points.as_deref())
}
