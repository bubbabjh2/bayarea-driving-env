//! Router + guidance route words vs the JS oracle (`node rust/tools/oracle/nav_dump.mjs data rust/oracle/nav`).
//! Skips when the oracle or the data are absent.
use bayenv_world::nav::{self, ManeuverTable, RouteQuery, Router, RouterOpts};
use bayenv_world::World;
use std::path::Path;
use std::time::Instant;

fn setup() -> Option<(World, serde_json::Value)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let f = root.join("rust/oracle/nav/nav.json");
    if !f.exists() || !root.join("data/roads.meta.json").exists() { eprintln!("skip: no oracle {}", f.display()); return None; }
    let w = World::load(root.join("data")).expect("world");
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(f).unwrap()).unwrap();
    Some((w, v))
}

fn u32s(v: &serde_json::Value) -> Vec<u32> { v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect() }

#[test]
fn router_and_pack_match_js() {
    let Some((w, v)) = setup() else { return };
    let g = &w.graph;
    let mut r125 = Router::new(g, &RouterOpts { weight: 1.25, cache_max: 0, ..Default::default() });
    let mut r1 = Router::new(g, &RouterOpts { weight: 1.0, cache_max: 0, ..Default::default() });
    let qs = v["queries"].as_array().unwrap();
    let (mut bad, mut n_av, mut n_pen) = (0, 0, 0);
    // timing per kind: plain w1.25, plain w1, avoid, penalty
    let mut tk = [0.0f64; 4]; let mut nk = [0usize; 4]; let mut pops = [0usize; 4];
    let mut out = Vec::new();
    for q in qs {
        let (s, gl) = (q["s"].as_u64().unwrap() as usize, q["g"].as_u64().unwrap() as usize);
        let avoid: Vec<u32> = q.get("avoid").map(u32s).unwrap_or_default();
        let pen: Vec<(u32, f64)> = q.get("pen").map(|p| p.as_array().unwrap().iter().map(|x| (x[0].as_u64().unwrap() as u32, x[1].as_f64().unwrap())).collect()).unwrap_or_default();
        if !avoid.is_empty() { n_av += 1; } if !pen.is_empty() { n_pen += 1; }
        let r = if q["w"].as_f64().unwrap() == 1.0 { &mut r1 } else { &mut r125 };
        let t = Instant::now();
        let ok = r.route_into(s, gl, &RouteQuery { avoid: &avoid, penalty: &pen }, &mut out);
        let kind = if !avoid.is_empty() { 2 } else if !pen.is_empty() { 3 } else if q["w"].as_f64().unwrap() == 1.0 { 1 } else { 0 };
        tk[kind] += t.elapsed().as_secs_f64(); nk[kind] += 1; pops[kind] += r.last_pops;
        let js: Option<Vec<u32>> = if q["r"].is_null() { None } else { Some(u32s(&q["r"])) };
        let rs = if ok { Some(out.clone()) } else { None };
        if js != rs { bad += 1; if bad < 5 { eprintln!("route mismatch {s}->{gl} w={} avoid={} pen={}: js {:?} rs {:?}", q["w"], avoid.len(), pen.len(), js.as_ref().map(|x| x.len()), rs.as_ref().map(|x| x.len())); } }
    }
    eprintln!("router: {} queries ({n_av} avoid, {n_pen} penalty), {bad} mismatches", qs.len());
    for (k, name) in ["plain w=1.25", "plain w=1", "avoid w=1.25", "penalty w=1.25"].iter().enumerate() {
        eprintln!("  {name:16} {:5} queries {:.3} ms/query, {:.0} pops/query", nk[k], tk[k] * 1e3 / nk[k] as f64, pops[k] as f64 / nk[k] as f64);
    }
    assert_eq!(bad, 0);

    // packRoutes
    let flat: Vec<i32> = v["flat"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap() as i32).collect();
    let offset = u32s(&v["offset"]);
    let js = u32s(&v["words"]);
    let t = Instant::now(); let tab = ManeuverTable::new(g); let t_tab = t.elapsed().as_secs_f64();
    let t = Instant::now(); let words = nav::pack_routes(g, &tab, &flat, &offset); let t_pool = t.elapsed().as_secs_f64();
    let mismatch = js.iter().zip(&words).filter(|(a, b)| a != b).count() + js.len().abs_diff(words.len());
    if mismatch > 0 {
        let i = js.iter().zip(&words).position(|(a, b)| a != b).unwrap_or(0);
        eprintln!("first word mismatch at {i}: js {:08x} rs {:08x}", js[i], words[i]);
    }
    // per-route timing (spawn / reroute path)
    let t = Instant::now(); let mut buf = vec![0u32; 256];
    for r in 0..offset.len() - 1 { let (a, b) = (offset[r] as usize, offset[r + 1] as usize); if buf.len() < b - a { buf.resize(b - a, 0); } nav::pack_route_words(g, &tab, &flat[a..b], &mut buf[..b - a]); }
    let t_each = t.elapsed().as_secs_f64() / (offset.len() - 1) as f64;
    eprintln!("packRoutes: {} words, {mismatch} mismatches; table {:.1} ms, pool {:.2} ms, {:.2} us/route (JS {})", words.len(), t_tab * 1e3, t_pool * 1e3, t_each * 1e6, v["timings"]);
    // full guidance port agrees with the table path on every route
    for r in 0..offset.len() - 1 {
        let (a, b) = (offset[r] as usize, offset[r + 1] as usize);
        if flat[a..b].iter().any(|&e| e < 0) { continue; }
        let ids: Vec<u32> = flat[a..b].iter().map(|&e| e as u32).collect();
        let ms = nav::maneuvers(g, &ids).unwrap();
        let mut code = vec![0u32; b - a];
        for m in &ms { let k = if m.ty == nav::man::ARRIVE { b - a - 1 } else { m.edge_pos - 1 }; if code[k] == 0 { code[k] = 1 | (m.ty as u32) << 1 | (m.side as u32) << 5; } }
        for k in 0..b - a { assert_eq!((words[a + k] >> 20) & 127, code[k], "route {r} entry {k}"); }
    }
    assert_eq!(mismatch, 0);
}
