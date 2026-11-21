//! Input verification for `World::load`: every data file the world is derived from is checked against
//! `tools/osm/world-manifest.json` (sha256, embedded at compile time), missing unpacked JSON copies are restored from
//! the committed `.json.gz`, and nothing is ever silently substituted: a missing or mismatched file is an error that
//! names the command that fixes it. Hashes are cached per (path, size, mtime) in `<data>/build/rust/sha-cache.json`
//! so a warm load does not re-hash.
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

pub const MANIFEST: &str = include_str!("../../../../tools/osm/world-manifest.json");

/// Files `World::load` reads (manifest names, relative to the data dir).
pub const INPUTS: [&str; 7] = ["roads.bin", "roads-local.bin", "controls.bin", "stations.bin", "surfaces.bin", "crossings.json", "crosswalks.json"];

pub fn manifest_sha(name: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(MANIFEST).ok()?;
    v["files"][name]["sha256"].as_str().map(|s| s.to_string())
}

fn bad(msg: String) -> io::Error { io::Error::new(io::ErrorKind::InvalidData, msg) }

static CACHE_LOCK: Mutex<()> = Mutex::new(());

/// sha256 of a file, cached by (size, mtime) in `<dir>/build/rust/sha-cache.json`.
pub fn sha256_file(dir: &Path, rel: &str) -> io::Result<String> {
    let p = dir.join(rel);
    let md = std::fs::metadata(&p)?;
    let mt = md.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_nanos() as u64);
    let key = format!("{}:{}", md.len(), mt);
    let _g = CACHE_LOCK.lock().unwrap();
    let cpath = dir.join("build/rust/sha-cache.json");
    let mut cache: BTreeMap<String, (String, String)> = std::fs::read_to_string(&cpath).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
    if let Some((k, h)) = cache.get(rel) { if *k == key { return Ok(h.clone()); } }
    let mut f = std::fs::File::open(&p)?; let mut h = Sha256::new(); let mut buf = vec![0u8; 1 << 20];
    loop { let n = f.read(&mut buf)?; if n == 0 { break; } h.update(&buf[..n]); }
    let hex: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
    cache.insert(rel.to_string(), (key, hex.clone()));
    if let Some(d) = cpath.parent() { let _ = std::fs::create_dir_all(d); }
    let _ = std::fs::write(&cpath, serde_json::to_string(&cache).unwrap_or_default());
    Ok(hex)
}

/// Verify one committed input against the manifest (error with a fix command when missing / different).
pub fn check_input(dir: &Path, rel: &str) -> io::Result<String> {
    let want = manifest_sha(rel).ok_or_else(|| bad(format!("{rel}: not in tools/osm/world-manifest.json")))?;
    if !dir.join(rel).exists() {
        return Err(bad(format!("{}: missing. It is committed: restore it with `git checkout -- data/{rel}` (or pass LoadOptions {{ verify: false }} for a custom world)", dir.join(rel).display())));
    }
    let got = sha256_file(dir, rel)?;
    if got != want {
        return Err(bad(format!("{}: sha256 {got} does not match tools/osm/world-manifest.json ({want}). Restore the committed file (`git checkout -- data/{rel}`), or rebuild the world (`npm run build:world-derived`, docs/WORLD-BUILD.md) and the manifest", dir.join(rel).display())));
    }
    Ok(got)
}

/// An unpacked JSON copy (`controls.json`, gitignored): restored from `<rel>.gz` when missing, then verified.
pub fn ensure_unpacked(dir: &Path, rel: &str) -> io::Result<String> {
    let p = dir.join(rel);
    if !p.exists() {
        let gz = dir.join(format!("{rel}.gz"));
        if !gz.exists() { return Err(bad(format!("{}: missing and no {}.gz to unpack: `git checkout -- data/{rel}.gz && npm run unpack`", p.display(), rel))); }
        let mut out = Vec::new(); flate2::read::GzDecoder::new(std::fs::File::open(&gz)?).read_to_end(&mut out)?;
        let tmp: PathBuf = p.with_extension("json.tmp"); std::fs::write(&tmp, &out)?; std::fs::rename(&tmp, &p)?;
    }
    let want = manifest_sha(rel).ok_or_else(|| bad(format!("{rel}: not in the manifest")))?;
    let got = sha256_file(dir, rel)?;
    if got != want {
        return Err(bad(format!("{}: sha256 {got} does not match the manifest ({want}): re-unpack it (`rm data/{rel} && npm run unpack`)", p.display())));
    }
    Ok(got)
}

/// sha256 of a byte buffer (hex).
pub fn sha256_bytes(b: &[u8]) -> String { Sha256::digest(b).iter().map(|x| format!("{x:02x}")).collect() }
