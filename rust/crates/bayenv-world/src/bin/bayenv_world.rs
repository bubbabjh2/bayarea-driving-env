//! `bayenv-world info <data>` | `check <data> <oracleDir>` | `import-lg <data> <oracleDir>`
use bayenv_world::{oracle, World};
use std::path::Path;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let cmd = a.get(1).map(|s| s.as_str()).unwrap_or("info"); let data = Path::new(a.get(2).map(|s| s.as_str()).unwrap_or("data"));
    match cmd {
        "import-lg" => {
            let bake = data.join("build/rust/world.bake");
            oracle::import_lanes(data, Path::new(&a[3]), &bake).expect("import"); println!("wrote {}", bake.display());
        }
        "info" | "check" => {
            let t = std::time::Instant::now();
            let w = World::load(data).expect("load");
            println!("loaded in {:.1} ms: {} nodes, {} edges, {} junction movements, {} controls, {} stations",
                t.elapsed().as_secs_f64() * 1e3, w.graph.node_count, w.graph.edge_count, w.junctions.as_ref().map_or(0, |j| j.move_count()),
                w.controls.as_ref().map_or(0, |c| c.count), w.stations.len());
            for (k, ms) in &w.timings { println!("  {k:28} {ms:8.1} ms"); }
            if cmd == "check" {
                let mut fail = 0;
                for (name, n, bad, first) in oracle::check_world(&w, Path::new(&a[3])).expect("oracle") {
                    if bad > 0 { fail += 1; }
                    println!("{} {name:16} {n:>10} elems, {bad} mismatches{}", if bad == 0 { "OK  " } else { "FAIL" }, first.map_or(String::new(), |i| format!(" (first at {i})")));
                }
                std::process::exit(if fail > 0 { 1 } else { 0 });
            }
        }
        _ => eprintln!("usage: bayenv-world info|check|import-lg <data> [oracleDir]"),
    }
}
