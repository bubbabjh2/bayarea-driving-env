//! A fresh clone of the public repo has only the git-tracked data (no junctions.bin, no unpacked controls.json):
//! World::load must verify the inputs, unpack / build what is missing from verified inputs and arrive at exactly the
//! same world as the full checkout. Copies the tracked data files into a temp dir.
use std::path::{Path, PathBuf};

fn repo() -> PathBuf { Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..").canonicalize().unwrap() }

#[test]
fn fresh_clone_loads_the_same_world() {
    let root = repo();
    let out = std::process::Command::new("git").arg("-C").arg(&root).args(["ls-files", "data"]).output();
    let Ok(out) = out else { eprintln!("skip: no git"); return };
    let files: Vec<String> = String::from_utf8_lossy(&out.stdout).lines().map(|s| s.to_string())
        .filter(|f| { let r = &f[5..]; !r.contains('/') || r.starts_with("dem/compact/") }).collect();
    if files.is_empty() { eprintln!("skip: no tracked data"); return; }
    let tmp = std::env::temp_dir().join(format!("bayenv-fresh-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    for f in &files {
        let dst = tmp.join(&f[5..]); std::fs::create_dir_all(dst.parent().unwrap()).unwrap();
        std::fs::copy(root.join(f), &dst).unwrap();
    }
    assert!(!tmp.join("junctions.bin").exists() && !tmp.join("controls.json").exists(), "fresh clone has no build outputs");
    let fresh = bayenv_world::World::load(&tmp).expect("fresh-clone load");
    let full = bayenv_world::World::load(root.join("data")).expect("checkout load");
    assert_eq!(fresh.content_hash(), full.content_hash(), "fresh clone built a different world");
    // second load uses the caches (junctions bake keyed by the input hashes)
    let t = std::time::Instant::now(); let again = bayenv_world::World::load(&tmp).unwrap();
    assert_eq!(again.content_hash(), full.content_hash());
    eprintln!("fresh clone: same world (hash {:016x}); warm reload {:.0} ms", full.content_hash(), t.elapsed().as_secs_f64() * 1e3);
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn tampered_input_fails_loudly() {
    let root = repo();
    let tmp = std::env::temp_dir().join(format!("bayenv-tamper-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp); std::fs::create_dir_all(&tmp).unwrap();
    for f in ["roads.meta.json", "roads.bin", "roads-local.meta.json", "roads-local.bin"] { std::fs::copy(root.join("data").join(f), tmp.join(f)).unwrap(); }
    let mut b = std::fs::read(tmp.join("roads.bin")).unwrap(); b[1000] ^= 1; std::fs::write(tmp.join("roads.bin"), b).unwrap();
    let e = bayenv_world::World::load(&tmp).err().expect("must fail").to_string();
    assert!(e.contains("roads.bin") && e.contains("git checkout"), "{e}");
    let _ = std::fs::remove_dir_all(&tmp);
}
