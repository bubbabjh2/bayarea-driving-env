use bayenv_sim::{pack::Pack, world::World};

fn scratch() -> String { std::env::var("BAYENV_TRACE_DIR").unwrap_or_else(|_| bayenv_world::oracle::oracle_root().join("sim/traces").to_string_lossy().to_string()) }

#[test]
fn hops_match_js() {
    for d in ["junction-s1", "synthetic"] {
        let path = format!("{}/{}/world.pack", scratch(), d);
        if !std::path::Path::new(&path).exists() { bayenv_world::oracle::skip("world_hops", &format!("no {path} (BAYENV_TRACE_DIR)")); continue; }
        let p = Pack::read(&path).unwrap();
        let w = World::from_pack(&p);
        let bad = w.check_hops(&p);
        eprintln!("{d}: edges {} hop mismatches {bad}", w.edge_count);
        assert_eq!(bad, 0);
    }
}
