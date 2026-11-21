use bayenv_sim::{pack::Pack, world::World};

fn scratch() -> String { std::env::var("BAYENV_TRACE_DIR").unwrap_or_else(|_| "/private/tmp/claude-501/-Users-USER-Documents-youtube-driving/bbb58b25-79a2-4e79-bb3c-772de0095e7a/scratchpad".into()) }

#[test]
fn hops_match_js() {
    for d in ["tr-junction-s1", "tr-synthetic"] {
        let path = format!("{}/{}/world.pack", scratch(), d);
        if !std::path::Path::new(&path).exists() { eprintln!("skip {path}"); continue; }
        let p = Pack::read(&path).unwrap();
        let w = World::from_pack(&p);
        let bad = w.check_hops(&p);
        eprintln!("{d}: edges {} hop mismatches {bad}", w.edge_count);
        assert_eq!(bad, 0);
    }
}
