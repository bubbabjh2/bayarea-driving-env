//! `build-junctions [dataDir] [--out <dir>]`: Rust port of `tools/build_junctions.mjs`. OWNER: geom port agent.
//!
//! Builds the lane geometry and the junctions (crosswalk legs from `controls.json` + `crosswalks.json`), writes
//! `junctions.bin` + `junctions.meta.json` and rewrites every junction control's `approachStopAt` to the mouth's
//! sStop in `controls.bin` / `controls.json` / `controls.meta.json`. Output is byte-identical to the JS tool
//! (meta files: except `buildMs`). Reads from `dataDir` (default `data`); writes to `--out` (default: `dataDir`).
use bayenv_world::jsjson::{self, Json};
use bayenv_world::junctions::JUNCTION_SECTIONS;
use bayenv_world::junctions_build::{build_junctions_full, crosswalk_legs, ControlLegs};
use bayenv_world::lanegraph::graph_from_road_sets;
use bayenv_world::{jsmath, RoadSet};
use std::path::PathBuf;
use std::time::Instant;

fn die(m: String) -> ! { eprintln!("build-junctions: {m}"); std::process::exit(1) }

fn read_json(p: &std::path::Path) -> Json {
    let t = std::fs::read_to_string(p).unwrap_or_else(|e| die(format!("{}: {e}", p.display())));
    jsjson::parse(&t).unwrap_or_else(|e| die(format!("{}: {e}", p.display())))
}

fn obj(kv: Vec<(&str, Json)>) -> Json { Json::Obj(kv.into_iter().map(|(k, v)| (k.to_string(), v)).collect()) }
fn s(x: &str) -> Json { Json::Str(x.to_string()) }
fn n(x: f64) -> Json { Json::Num(x) }

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut dir: Option<PathBuf> = None; let mut out: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => { out = Some(PathBuf::from(args.get(i + 1).unwrap_or_else(|| die("--out needs a directory".into())))); i += 1; }
            "-h" | "--help" => { eprintln!("usage: build-junctions [dataDir] [--out <dir>]"); return; }
            a => dir = Some(PathBuf::from(a)),
        }
        i += 1;
    }
    let dir = dir.unwrap_or_else(|| PathBuf::from("data"));
    let out = out.unwrap_or_else(|| dir.clone());
    std::fs::create_dir_all(&out).unwrap_or_else(|e| die(format!("{}: {e}", out.display())));

    let t0 = Instant::now();
    let main = RoadSet::load(&dir, "roads.meta.json").unwrap_or_else(|e| die(e.to_string()));
    let local = RoadSet::load(&dir, "roads-local.meta.json").unwrap_or_else(|e| die(e.to_string()));
    let g = graph_from_road_sets(&main, Some(&local));
    let lg = bayenv_world::lanes_build::build_lane_geometry(&g);
    let t_lg = t0.elapsed().as_secs_f64();
    // crosswalk legs: controls (signals, all-way stops) + painted / signalised OSM crossings
    let (ctl_path, cw_path) = (dir.join("controls.json"), dir.join("crosswalks.json"));
    // (the JSON trees are dropped during the build and controls.json is parsed again for the patch: peak memory)
    let ctl_doc = if ctl_path.exists() { Some(read_json(&ctl_path)) } else { None };
    let cw_doc = if cw_path.exists() { Some(read_json(&cw_path)) } else { None };
    let controls: Option<Vec<ControlLegs>> = ctl_doc.as_ref().and_then(|d| d.get("controls")).and_then(|c| c.as_arr()).map(|cs| cs.iter().map(|c| ControlLegs {
        node: c.get("node").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
        ty: c.get("type").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        all_way: c.get("allWay").map_or(false, |v| v.truthy()),
        approach_edges: c.get("approaches").and_then(|a| a.as_arr()).map(|a| a.iter().filter_map(|ap| ap.get("edge").and_then(|v| v.as_f64())).filter(|&e| e >= 0.0).map(|e| e as u32).collect()).unwrap_or_default(),
    }).collect());
    let points: Option<Vec<[f64; 2]>> = cw_doc.as_ref().and_then(|d| d.get("points")).and_then(|p| p.as_arr())
        .map(|ps| ps.iter().map(|p| { let a = p.as_arr().unwrap(); [a[0].as_f64().unwrap_or(f64::NAN), a.get(1).and_then(|v| v.as_f64()).unwrap_or(f64::NAN)] }).collect());
    let legs = crosswalk_legs(&g, controls.as_deref(), points.as_deref());
    let has_ctl = ctl_doc.is_some();
    drop((ctl_doc, cw_doc, controls, points));
    let (j, st) = build_junctions_full(&lg, &g, Some(&legs));
    let build_ms = t0.elapsed().as_millis() as f64;
    eprintln!("lane geometry {:.2} s, junctions {:.2} s ({} builds)", t_lg, t0.elapsed().as_secs_f64() - t_lg, st.builds);

    // controls must stop at the junction mouths: patch approachStopAt (json + bin) for junction controls
    let mut patched = 0;
    if has_ctl {
        let mut doc_v = read_json(&ctl_path); let doc = &mut doc_v;
        let mut cmeta = read_json(&dir.join("controls.meta.json"));
        let file = cmeta.get("file").and_then(|f| f.as_str()).unwrap_or_else(|| die("controls.meta.json: no file".into())).to_string();
        let mut cb = std::fs::read(dir.join(&file)).unwrap_or_else(|e| die(format!("{file}: {e}")));
        let sec = cmeta.get("sections").and_then(|x| x.get("approachStopAt")).unwrap_or_else(|| die("controls.meta.json: no approachStopAt".into()));
        let (off, len) = (sec.get("byteOffset").and_then(|v| v.as_f64()).unwrap() as usize, sec.get("length").and_then(|v| v.as_f64()).unwrap() as usize);
        let rd = |cb: &[u8], a: usize| f32::from_le_bytes(cb[off + 4 * a..off + 4 * a + 4].try_into().unwrap());
        let mut a = 0usize;
        for c in doc.get_mut("controls").and_then(|c| c.as_arr_mut()).unwrap_or_else(|| die("controls.json: no controls".into())).iter_mut() {
            let node = c.get("node").and_then(|v| v.as_f64()).unwrap_or(f64::NAN);
            let is_meter = c.get("type").and_then(|v| v.as_str()) == Some("ramp_meter");
            let Some(aps) = c.get_mut("approaches").and_then(|x| x.as_arr_mut()) else { continue };
            for ap in aps.iter_mut() {
                let edge = ap.get("edge").and_then(|v| v.as_f64()).unwrap_or(f64::NAN);
                // a junction control on a node with no junction box (only plain continuations) keeps its own stop line
                if node >= 0.0 && !is_meter && edge >= 0.0 && (edge as usize) < g.edge_count {
                    let e = edge as usize;
                    let v = j.mouth_s[2 * e + 1];
                    if (v as f64) < g.length[e] as f64 - 0.5 {
                        assert!(a < len, "more approaches than approachStopAt entries");
                        if (rd(&cb, a) as f64 - v as f64).abs() > 0.01 { patched += 1; }
                        cb[off + 4 * a..off + 4 * a + 4].copy_from_slice(&v.to_le_bytes());
                        ap.set("stopAt", Json::Num(jsmath::round(v as f64 * 100.0) / 100.0));
                    }
                }
                a += 1;
            }
        }
        cmeta.set("stopLinesFrom", s("data/junctions.bin mouthS (tools/build_junctions.mjs)"));
        let w = |name: &str, bytes: &[u8]| std::fs::write(out.join(name), bytes).unwrap_or_else(|e| die(format!("{name}: {e}")));
        w(&file, &cb);
        w("controls.json", doc.stringify().as_bytes());
        w("controls.meta.json", cmeta.stringify_indent(1).as_bytes());
    }
    println!("crosswalk legs: {} stop mouths, {} exit mouths; controls stop lines patched: {patched}", legs.crosswalk_stop.len(), legs.crosswalk_exit.len());

    let bin = j.to_bin();
    let lens: [(usize, &str); 15] = [(j.mouth_s.len(), "Float32"), (j.gore_s.len(), "Float32"), (j.j_node.len(), "Uint32"), (j.j_first_m.len(), "Uint32"),
        (j.j_count_m.len(), "Uint32"), (j.j_first_c.len(), "Uint32"), (j.j_count_c.len(), "Uint32"), (j.j_first_h.len(), "Uint32"), (j.j_count_h.len(), "Uint32"),
        (j.m_info.len(), "Uint32"), (j.m_f.len(), "Float32"), (j.pts.len(), "Float32"), (j.conf.len(), "Uint32"), (j.conf_s.len(), "Float32"), (j.hull.len(), "Float32")];
    let mut sections = Vec::new(); let mut off = 0usize;
    for (k, (len, ty)) in JUNCTION_SECTIONS.iter().zip(lens) {
        sections.push((k.to_string(), obj(vec![("byteOffset", n(off as f64)), ("byteLength", n(4.0 * len as f64)), ("type", s(ty)), ("length", n(len as f64))])));
        off += 4 * len;
    }
    assert_eq!(off, bin.len());
    std::fs::write(out.join("junctions.bin"), &bin).unwrap_or_else(|e| die(format!("junctions.bin: {e}")));
    let (mut conflicts, mut cross, mut merge, mut diverge) = (0, 0, 0, 0);
    for c in 0..j.conf.len() / 3 { conflicts += 1; match j.conf[3 * c + 2] { 0 => cross += 1, 1 => merge += 1, _ => diverge += 1 } }
    let gores = j.gore_s.iter().filter(|&&x| x >= 0.0).count();
    let counts = obj(vec![("junctions", n(j.j_node.len() as f64)), ("movements", n((j.m_info.len() / 7) as f64)), ("connectorPoints", n((j.pts.len() / 4) as f64)),
        ("conflicts", n(conflicts as f64)), ("crossing", n(cross as f64)), ("merge", n(merge as f64)), ("sameEntry", n(diverge as f64)), ("rampGores", n(gores as f64))]);
    let crossing_fix = obj(vec![("crossingsBefore", n(st.crossings_before as f64)), ("crossingsAfter", n(st.crossings_after as f64)),
        ("edgesAdjusted", n(st.edges_adjusted as f64)), ("rounds", n(st.rounds as f64)), ("crossConflicts", n(st.cross_conflicts as f64))]);
    let layout = obj(vec![
        ("mouthS", s("Float32 x2 per edge: [sExit at the start junction, sStop at the end junction] (m along the edge; 0 / length where the node is not a junction)")),
        ("goreS", s("Float32 per edge: ramp wait point (m) where the ramp lane centre clears the mainline pavement by 1.3 m; -1 if the ramp is not a no-added-lane merge")),
        ("jNode", s("Uint32 per junction: merged-graph node")), ("jFirstM/jCountM", s("movements of the junction")), ("jFirstC/jCountC", s("conflicts of the junction")),
        ("jFirstH/jCountH", s("junction-box hull points (x,z pairs in hull)")),
        ("mInfo", s("Uint32 x7 per movement: [inEdge, inLane, outEdge, outLane, firstPt, ptCount, preferred]")),
        ("mF", s("Float32 x4 per movement: [sIn (leave inEdge here), sOut (join outEdge here), connector length, turn angle rad (>0 right)]")),
        ("pts", s("Float32 x4 per connector point: [x, y, z, metres along the connector]")),
        ("conf", s("Uint32 x3 per conflict: [movementA, movementB, type 0 crossing / 1 merge (exits within a car width)]; movements sharing an entry lane are not listed (an ordinary queue). Entries after the last junction range (jFirstC/jCountC) are cross-junction conflicts between connectors of neighbouring junctions (crossingFix)")),
        ("confS", s("Float32 x2 per conflict: [metres along A, metres along B] where the paths meet")),
        ("hull", s("Float32 x2 per hull point")),
    ]);
    let meta = obj(vec![
        ("format", s("bay-junctions/1")), ("file", s("junctions.bin")), ("byteLength", n(off as f64)), ("sections", Json::Obj(sections)),
        ("edgeIds", s("merged graph: main e, local mainEdgeCount + e (= src/sim fromRoadSet, src/geom)")), ("mainEdgeCount", n(main.edge_count as f64)),
        ("nodeIds", s("merged-graph node indices (graphFromRoadSets); they change with every rebuild, edge ids do not")),
        ("layout", layout), ("counts", counts.clone()), ("crossingFix", crossing_fix.clone()), ("buildMs", n(build_ms)),
    ]);
    std::fs::write(out.join("junctions.meta.json"), meta.stringify_indent(1)).unwrap_or_else(|e| die(format!("junctions.meta.json: {e}")));
    println!("{} {} {:.1} MB, {} ms", counts.stringify(), crossing_fix.stringify(), off as f64 / 1e6, build_ms);
}
