//! World blob + scatter route words vs the JS mock-WebGPU oracle (`rust/oracle/gpu10k`, `rust/tools/oracle/gpu_mock.mjs`).
//! Skips when the oracle or the data are absent.
use bayenv_gpu::pack::{build_world_blob, RoutePacker, MAX_BLOCKS};
use bayenv_world::World;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::time::Instant;

fn read_buf(dir: &Path, meta: &serde_json::Value, name: &str) -> Option<Vec<u8>> {
    let b = meta["buffers"].as_array()?.iter().find(|b| b["name"] == name)?;
    let size = b["size"].as_u64()? as usize; let off = b["offset"].as_i64()?;
    if off < 0 { return Some(vec![0; size]); }
    let mut f = std::fs::File::open(dir.join("gpu.bin")).ok()?;
    f.seek(SeekFrom::Start(off as u64)).ok()?;
    let mut v = vec![0u8; size]; f.read_exact(&mut v).ok()?; Some(v)
}

#[test]
fn world_blob_and_routes_match_js() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let dir = root.join("rust/oracle/gpu10k");
    if !dir.join("gpu.meta.json").exists() || !root.join("data/roads.meta.json").exists() { bayenv_world::oracle::skip("pack_oracle", "no rust/oracle/gpu10k/gpu.meta.json or data/roads.meta.json"); return; }
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("gpu.meta.json")).unwrap()).unwrap();
    let w = World::load(root.join("data")).expect("world");
    let t = Instant::now();
    let blob = build_world_blob(&w, MAX_BLOCKS);
    eprintln!("build_world_blob: {:.0} ms, {} words", t.elapsed().as_secs_f64() * 1e3, blob.words.len());
    // offsets + counts
    for (k, v) in blob.off.entries() { assert_eq!(meta["layoutOff"][k].as_u64(), Some(v as u64), "layoutOff.{k}"); }
    assert_eq!(meta["world"]["E"].as_u64(), Some(blob.e as u64)); assert_eq!(meta["world"]["N"].as_u64(), Some(blob.n as u64));
    assert_eq!(meta["world"]["P"].as_u64(), Some(blob.p as u64)); assert_eq!(meta["world"]["K"].as_u64(), Some(blob.k as u64));
    assert_eq!(meta["pairN"].as_u64(), Some(blob.pair_n as u64)); assert_eq!(meta["jM"].as_u64(), Some(blob.j_m.unwrap_or(0) as u64));
    let js = read_buf(&dir, &meta, "world").unwrap();
    let rs = blob.bytes();
    assert_eq!(js.len(), rs.len(), "world buffer size");
    let jw: &[u32] = bytemuck::cast_slice(&js);
    // per-section mismatch report
    let mut secs: Vec<(&str, usize)> = vec![("edges", 0)]; secs.extend(blob.off.entries().iter().map(|&(k, v)| (k, v)));
    let mut total = 0;
    for (i, &(name, a)) in secs.iter().enumerate() {
        let b = secs.get(i + 1).map_or(jw.len(), |s| s.1);
        let bad: Vec<usize> = (a..b).filter(|&k| jw[k] != blob.words[k]).collect();
        total += bad.len();
        eprintln!("{} {name:12} [{a}, {b}) {} words, {} mismatches{}", if bad.is_empty() { "OK  " } else { "FAIL" }, b - a, bad.len(),
            bad.first().map_or(String::new(), |&k| format!(" (first at +{}: js {:08x} rs {:08x})", k - a, jw[k], blob.words[k])));
    }
    // routes: the first routeWords words = packRoutes of the scatter pool; boundaries = entries with the arrive code
    let rw = meta["scatter"]["routeWords"].as_u64().unwrap() as usize;
    let rb = read_buf(&dir, &meta, "routes").unwrap(); let rj: &[u32] = &bytemuck::cast_slice::<u8, u32>(&rb)[..rw];
    let packer = RoutePacker::new(&w.graph);
    let (mut flat, mut offset) = (Vec::with_capacity(rw), vec![0u32]);
    for (k, &x) in rj.iter().enumerate() { flat.push((x & 0xFFFFF) as i32); if (x >> 20) & 1 == 1 && (x >> 21) & 15 == 7 { offset.push(k as u32 + 1); } }
    assert_eq!(*offset.last().unwrap() as usize, rw);
    let t = Instant::now(); let words = packer.pack_routes(&w.graph, &flat, &offset);
    let bad_r = rj.iter().zip(&words).filter(|(a, b)| a != b).count();
    eprintln!("routes: {} routes, {rw} words, {bad_r} mismatches, {:.2} ms", offset.len() - 1, t.elapsed().as_secs_f64() * 1e3);
    assert_eq!(bad_r, 0);
    assert_eq!(total, 0, "world blob mismatches");
}
