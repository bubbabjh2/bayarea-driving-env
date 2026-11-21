//! Population generator vs the JS oracle (tools/rust/export_pop.mjs or trace.mjs runs). Every subdirectory of
//! $BAYENV_POP_DIR holding world.pack + pop.pack is checked: all Population arrays bit-exact (f64 / f32 bits, routes).
//! Region runs (popOpts.region, e.g. export_pop.mjs --opts '{"region":{"preset":"sf"}}') also check that every home
//! and workplace is inside the region. Skips when BAYENV_POP_DIR is unset. Timings: serial (1 thread) vs rayon vs the JS time stored in pop.pack meta.ms.
use bayenv_sim::pack::Pack;
use bayenv_sim::population::Population;
use bayenv_sim::population_gen::*;
use bayenv_sim::world::World;
use std::sync::Arc;
use std::time::Instant;

fn diff_f64(name: &str, a: &[f64], b: &[f64]) -> usize {
    if a.len() != b.len() { eprintln!("  {name}: len {} vs {}", a.len(), b.len()); return 1; }
    let bad: Vec<usize> = (0..a.len()).filter(|&i| a[i].to_bits() != b[i].to_bits()).collect();
    if let Some(&i) = bad.first() { eprintln!("  {name}: {} mismatches, first [{i}] rust {:?} js {:?}", bad.len(), a[i], b[i]); }
    bad.len()
}
fn diff<T: PartialEq + std::fmt::Debug>(name: &str, a: &[T], b: &[T]) -> usize {
    if a.len() != b.len() { eprintln!("  {name}: len {} vs {}", a.len(), b.len()); return 1; }
    let bad: Vec<usize> = (0..a.len()).filter(|&i| a[i] != b[i]).collect();
    if let Some(&i) = bad.first() { eprintln!("  {name}: {} mismatches, first [{i}] rust {:?} js {:?}", bad.len(), a[i], b[i]); }
    bad.len()
}

pub fn compare(r: &Population, j: &Population) -> usize {
    let mut bad = 0;
    if r.n != j.n { eprintln!("  n {} vs {}", r.n, j.n); return 1; }
    bad += diff("vehicleType", &r.vehicle_type, &j.vehicle_type);
    bad += diff("home", &r.home, &j.home);
    bad += diff("work", &r.work, &j.work);
    bad += diff_f64("depart", &r.depart, &j.depart);
    bad += diff_f64("returnDepart", &r.return_depart, &j.return_depart);
    bad += diff_f64("workStart", &r.work_start, &j.work_start);
    bad += diff_f64("fuelFill", &r.fuel_fill, &j.fuel_fill);
    bad += diff_f64("L", &r.route_length, &j.route_length);
    bad += diff_f64("expectedTravel", &r.expected_travel, &j.expected_travel);
    bad += diff("routeOffset", &r.route_offset, &j.route_offset);
    bad += diff("routeFlat", &r.route_flat, &j.route_flat);
    let (rb, jb): (Vec<u32>, Vec<u32>) = (r.deltas.iter().map(|x| x.to_bits()).collect(), j.deltas.iter().map(|x| x.to_bits()).collect());
    bad += diff("deltas", &rb, &jb);
    bad
}

#[test]
fn population_matches_js() {
    // BAYENV_POP_DIR, else rust/oracle/sim/pop (make_oracle.sh)
    let root = std::env::var("BAYENV_POP_DIR").unwrap_or_else(|_| bayenv_world::oracle::oracle_root().join("sim/pop").to_string_lossy().to_string());
    if bayenv_world::oracle::dirs_with(std::path::Path::new(&root), "pop.pack").is_empty() { bayenv_world::oracle::skip("population_gen", &format!("no */pop.pack under {root} (BAYENV_POP_DIR)")); return; }
    let only = std::env::var("BAYENV_POP_ONLY").ok();
    let mut dirs: Vec<_> = std::fs::read_dir(&root).unwrap().filter_map(|e| e.ok()).map(|e| e.path())
        .filter(|p| p.join("pop.pack").exists() && p.join("world.pack").exists()).collect();
    dirs.sort();
    let mut failed = Vec::new();
    let mut worlds: std::collections::HashMap<String, Arc<World>> = Default::default();
    for d in dirs {
        let name = d.file_name().unwrap().to_string_lossy().to_string();
        if let Some(o) = &only { if !name.contains(o.as_str()) { continue; } }
        let pp = Pack::read(d.join("pop.pack").to_str().unwrap()).unwrap();
        let js = Population::from_pack(&pp);
        let m = &pp.meta;
        let wp = std::fs::canonicalize(d.join("world.pack")).unwrap().to_string_lossy().to_string();
        let world = worlds.entry(wp.clone()).or_insert_with(|| Arc::new(World::load(&wp))).clone();
        // trace.mjs runs carry run.json instead of export_pop meta
        let (wname, seed, opts) = if m.get("world").is_some() { (m["world"].as_str().unwrap().to_string(), m["seed"].as_f64().unwrap(), m["popOpts"].clone()) } else {
            let run: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(d.join("run.json")).unwrap()).unwrap();
            (run["world"].as_str().unwrap().to_string(), run["seed"].as_f64().unwrap(), serde_json::json!({ "n": run["n"] }))
        };
        let js_ms = m.get("ms").and_then(|x| x.as_f64());
        let max_speed: f64 = world.e_speed.iter().cloned().fold(0.0, f64::max);
        if max_speed.to_bits() != world.max_speed.to_bits() { eprintln!("  NOTE {name}: world.max_speed {:?} != max(e_speed) {:?}", world.max_speed, max_speed); }
        let (r1, t1, tp) = if let Some(sc) = wname.strip_prefix("scenario:") {
            let txt = std::fs::read_to_string(default_data_dir().join("scenarios").join(sc).join("scenario.json")).unwrap();
            let scn: serde_json::Value = serde_json::from_str(&txt).unwrap();
            let t = Instant::now();
            let g = scenario_population(&world, &scn, &ScenarioOpts { seed: Some(seed), ..Default::default() }).unwrap();
            (g.pop, t.elapsed().as_secs_f64() * 1e3, None)
        } else {
            let mut cfg = if wname == "synthetic" { PopConfig::env_synthetic(0, seed) } else { PopConfig::env(0, seed) };
            cfg.apply_json(&opts);
            if js.n != cfg.n { cfg.n = js.n; }
            let t = Instant::now();
            let s = generate_population(&world, &cfg, Some(1)).unwrap();
            let t1 = t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            let p = generate_population(&world, &cfg, None).unwrap();
            let tp = t.elapsed().as_secs_f64() * 1e3;
            let b = compare(&p.pop, &s.pop);
            assert_eq!(b, 0, "{name}: rayon != serial");
            // region runs: every home and workplace lies inside the region (regionFilter)
            if let Some(f) = Context::new(&*world, &cfg).unwrap().region {
                let out = s.pop.home.iter().chain(s.pop.work.iter()).filter(|&&k| !f.contains(world.nx[k as usize], world.nz[k as usize])).count();
                assert_eq!(out, 0, "{name}: {out} homes / workplaces outside the region");
                eprintln!("  {name}: region {:?}: all {} homes / workplaces inside", cfg.region, 2 * s.pop.n);
            }
            (s.pop, t1, Some(tp))
        };
        let bad = compare(&r1, &js);
        eprintln!("{} {name}: world {wname} n {} seed {seed} routes {} | js {:?} ms, rust 1 thread {:.0} ms, rayon ({} threads) {:?} ms",
            if bad == 0 { "ok  " } else { "FAIL" }, js.n, js.route_flat.len(), js_ms, t1, rayon::current_num_threads(), tp.map(|x| x.round()));
        if bad != 0 { failed.push(name); }
    }
    assert!(failed.is_empty(), "failed: {failed:?}");
}
