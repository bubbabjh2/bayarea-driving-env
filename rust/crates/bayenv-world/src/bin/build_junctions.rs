//! `build-junctions [dataDir]`: Rust port of `tools/build_junctions.mjs` (writes data/junctions.{bin,meta.json}
//! and patches the controls stop lines). OWNER: geom port agent.
fn main() {
    let dir = std::env::args().nth(1).unwrap_or_else(|| "data".into());
    eprintln!("build-junctions {dir}: port in progress");
    std::process::exit(2);
}
