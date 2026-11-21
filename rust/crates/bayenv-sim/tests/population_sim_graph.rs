//! Population generation on `bayenv_world::graph::SimGraph` (PopGraph impl, NavRouter::from_sim_graph) must give the
//! same output, bit for bit, as on the sim `World` copied from it (`World::routing_from_bayenv`, the path the GPU host
//! used before). Real world: n = 2000 seed 1 (serial + rayon); every data/scenarios/<name> that loads: scenario.json.
//! Skips when the data dir is missing.
use bayenv_sim::population_gen::*;
use bayenv_sim::world::World;
use std::sync::Arc;
use std::time::Instant;

fn same(a: &Generated, b: &Generated) -> Result<(), String> {
    let (p, q) = (&a.pop, &b.pop);
    macro_rules! eq { ($($f:ident).+) => { if a.$($f).+ != b.$($f).+ { return Err(stringify!($($f).+).into()); } } }
    macro_rules! eqf { ($v:expr, $w:expr, $n:expr) => { if $v.len() != $w.len() || $v.iter().zip($w.iter()).any(|(x, y)| x.to_bits() != y.to_bits()) { return Err($n.into()); } } }
    if p.n != q.n { return Err("n".into()); }
    eq!(pop.vehicle_type); eq!(pop.home); eq!(pop.work); eq!(pop.route_offset); eq!(pop.route_flat);
    eqf!(p.depart, q.depart, "depart"); eqf!(p.return_depart, q.return_depart, "returnDepart"); eqf!(p.work_start, q.work_start, "workStart");
    eqf!(p.fuel_fill, q.fuel_fill, "fuelFill"); eqf!(p.route_length, q.route_length, "L"); eqf!(p.expected_travel, q.expected_travel, "expectedTravel");
    eqf!(p.deltas, q.deltas, "deltas");
    let (x, y) = (&a.extras, &b.extras);
    eq!(extras.kind); eq!(extras.length_ok); eq!(extras.attempts); eq!(extras.crosses_bay); eq!(extras.trip_tile); eq!(extras.trip_spec);
    eqf!(x.intended, y.intended, "intended"); eqf!(x.slack, y.slack, "slack"); eqf!(x.reaction_time, y.reaction_time, "reactionTime");
    Ok(())
}

/// Heap bytes of the graph-only World copy (routing_from_bayenv).
fn routing_world_bytes(w: &World) -> usize {
    let v8 = |n: usize| n * 8; let v4 = |n: usize| n * 4;
    v8(w.nx.len()) + v8(w.nz.len()) + v4(w.e_from.len()) + v4(w.e_to.len()) + v4(w.e_rev.len()) + v4(w.e_lanes.len())
        + v8(w.e_speed.len()) + v8(w.e_length.len()) + v4(w.e_rank.len()) + w.e_class.len() + v8(w.e_head_start.len()) + v8(w.e_head_end.len())
        + v4(w.out_start.len()) + v4(w.out_edges.len()) + v4(w.in_start.len()) + v4(w.in_edges.len())
        + w.banned_hops.iter().map(|(_, v)| 8 + 24 + v4(v.len())).sum::<usize>()
}

fn data_dir() -> std::path::PathBuf { default_data_dir() }

#[test]
fn sim_graph_population_matches_world() {
    let dir = data_dir();
    if !dir.join("roads.meta.json").exists() { eprintln!("skip: no data dir {}", dir.display()); return; }
    let t = Instant::now();
    let bw = bayenv_world::World::load(&dir).unwrap();
    eprintln!("bayenv_world::World::load {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
    let t = Instant::now();
    let rw = Arc::new(World::routing_from_bayenv(&bw));
    eprintln!("routing_from_bayenv {:.0} ms, ~{:.1} MB (no longer needed on the SimGraph path)", t.elapsed().as_secs_f64() * 1e3, routing_world_bytes(&rw) as f64 / 1e6);
    let cfg = PopConfig { n: 2000, seed: 1.0, ..Default::default() };
    for threads in [Some(1), None] {
        let t = Instant::now();
        let a = generate_population(&rw, &cfg, threads).unwrap();
        let ta = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let b = generate_population_sim_graph(&bw.graph, &cfg, threads).unwrap();
        let tb = t.elapsed().as_secs_f64() * 1e3;
        same(&a, &b).unwrap_or_else(|f| panic!("real n 2000 seed 1 threads {threads:?}: {f} differs"));
        eprintln!("ok real n 2000 seed 1 threads {threads:?}: routes {} | World {ta:.0} ms, SimGraph {tb:.0} ms", a.pop.route_flat.len());
    }
    // env-style config (returnTrips false) too
    let cfg = PopConfig::env(2000, 1.0);
    same(&generate_population(&rw, &cfg, None).unwrap(), &generate_population_sim_graph(&bw.graph, &cfg, None).unwrap()).unwrap();
    eprintln!("ok real env cfg n 2000 seed 1");
    drop(rw); drop(bw);

    let mut scn: Vec<_> = std::fs::read_dir(dir.join("scenarios")).map(|r| r.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.join("scenario.json").exists()).collect()).unwrap_or_default();
    scn.sort();
    let mut checked = 0;
    for d in scn {
        let name = d.file_name().unwrap().to_string_lossy().to_string();
        let bw = match bayenv_world::World::load(&d) { Ok(w) => w, Err(e) => { eprintln!("skip scenario {name}: load failed: {e}"); continue; } };
        let sc: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(d.join("scenario.json")).unwrap()).unwrap();
        let rw = Arc::new(World::routing_from_bayenv(&bw));
        let o = ScenarioOpts { seed: Some(1.0), ..Default::default() };
        let (a, b) = (scenario_population(&rw, &sc, &o), scenario_population_sim_graph(&bw.graph, &sc, &o));
        match (a, b) {
            (Ok(a), Ok(b)) => { same(&a, &b).unwrap_or_else(|f| panic!("scenario {name}: {f} differs")); eprintln!("ok scenario {name}: n {}", a.pop.n); checked += 1; }
            (Err(x), Err(y)) => { assert_eq!(x, y); eprintln!("scenario {name}: both error ({x})"); }
            (a, b) => panic!("scenario {name}: World {:?} vs SimGraph {:?}", a.err(), b.err()),
        }
        // the generic synth-axis path as well (Axis::synth / NearestNode / largest_scc over both impls)
        let cfg = PopConfig::env_synthetic(200, 1.0);
        match (generate_population(&rw, &cfg, Some(1)), generate_population_sim_graph(&bw.graph, &cfg, Some(1))) {
            (Ok(a), Ok(b)) => same(&a, &b).unwrap_or_else(|f| panic!("scenario {name} synth gen: {f} differs")),
            (Err(x), Err(y)) => assert_eq!(x, y),
            (a, b) => panic!("scenario {name} synth gen: World {:?} vs SimGraph {:?}", a.err(), b.err()),
        }
    }
    eprintln!("{checked} scenario worlds checked");
}
