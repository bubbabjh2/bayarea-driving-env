//! Shared fixtures for the js_port_*.rs tests (port of test/sim/*.test.js).
//!
//! The JS tests build their worlds in-process (buildGraph(makeSynthGraph(..)), custom roads); Rust has no synthGraph /
//! buildGraph port, so `tests/common/gen_worlds.mjs` builds the same worlds with the JS sim and exports them as world packs
//! into `$CARGO_TARGET_TMPDIR/js_port_fixtures/` (once per machine / until that dir is deleted). When `node` (or the repo's
//! src/) is unavailable every test that needs a world skips itself with an eprintln (`fixtures!()`).
#![allow(dead_code)]
use bayenv_sim::env::{Env, EnvConfig};
use bayenv_sim::population::Population;
use bayenv_sim::sim::SimOpts;
use bayenv_sim::world::World;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

pub const KMH: f64 = 1.0 / 3.6;
pub fn kmh(v: f64) -> f64 { v / 3.6 }

fn repo_root() -> PathBuf { PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..") }

/// Directory holding <name>/world.pack + fixtures.json, generating it with node on first use. None: node unavailable.
pub fn fixture_dir() -> Option<&'static PathBuf> {
    static DIR: OnceLock<Option<PathBuf>> = OnceLock::new();
    DIR.get_or_init(|| {
        let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("js_port_fixtures");
        if dir.join("fixtures.json").exists() { return Some(dir); }
        let root = repo_root();
        if !root.join("src/sim/roadGraph.js").exists() { bayenv_world::oracle::skip("js_port", &format!("repo src/sim not found at {}", root.display())); return None; }
        // build into a private dir, then move into place (cargo may run several test binaries / threads)
        let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("js_port_fixtures.tmp{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let st = std::process::Command::new("node").arg(root.join("rust/crates/bayenv-sim/tests/common/gen_worlds.mjs")).arg(&tmp).current_dir(&root).status();
        match st {
            Ok(s) if s.success() => {
                if std::fs::rename(&tmp, &dir).is_err() { let _ = std::fs::remove_dir_all(&tmp); }
                if dir.join("fixtures.json").exists() { Some(dir) } else { None }
            }
            Ok(s) => { bayenv_world::oracle::skip("js_port", &format!("gen_worlds.mjs failed ({s})")); None }
            Err(e) => { bayenv_world::oracle::skip("js_port", &format!("node unavailable ({e})")); None }
        }
    }).as_ref()
}

/// Parsed fixtures.json (edge indices of the generated worlds).
pub fn fixtures_meta() -> Option<serde_json::Value> {
    let d = fixture_dir()?;
    serde_json::from_str(&std::fs::read_to_string(d.join("fixtures.json")).ok()?).ok()
}

/// A fresh (owned, mutable) copy of fixture world `name`: synth | synth40 | road3 | road30k | surfcar | ctl.
pub fn load_world(name: &str) -> Option<World> {
    let d = fixture_dir()?;
    Some(World::load(d.join(name).join("world.pack").to_str().unwrap()))
}
/// Shared read-only world (cached per process).
pub fn shared_world(name: &str) -> Option<Arc<World>> {
    static CACHE: OnceLock<Mutex<std::collections::HashMap<String, Arc<World>>>> = OnceLock::new();
    let c = CACHE.get_or_init(Default::default);
    let mut m = c.lock().unwrap();
    if let Some(w) = m.get(name) { return Some(w.clone()); }
    let w = Arc::new(load_world(name)?);
    m.insert(name.to_string(), w.clone());
    Some(w)
}

/// `let Some(x) = fixtures!(load_world, "synth") else { return }` style helper.
#[macro_export]
macro_rules! need { ($e:expr) => { match $e { Some(x) => x, None => { bayenv_world::oracle::skip(&format!("{}:{}", file!(), line!()), "node fixtures unavailable"); return; } } }; }

/// synthAxis population options of test/sim/helpers.js popOpts: PopConfig::env_synthetic (axis synth, long 5-16 km, departures 'spec').
pub fn env_with(world: Arc<World>, pop: Arc<Population>, opts: SimOpts, start: f64) -> Env {
    let cfg = EnvConfig { opts, start_time: Some(start), static_obstacles: Vec::new(), scenario_obstacles: Vec::new(), obstacles: None };
    Env::new(world, pop, cfg, Box::new(|w| bayenv_sim::harness::router_for(&w)))
}
