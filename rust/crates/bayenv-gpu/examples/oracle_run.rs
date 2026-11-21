//! Native GPU sim vs JS oracle (mock run): setup buffers, dispatch lists, params, then N steps -> state dump.
//!   cargo run --release -p bayenv-gpu --example oracle_run -- <data> <oracleDir> <steps> <batch> <out.bin> [native]
use bayenv_gpu::{oracle::OracleDump, sim::{GpuOptions, GpuSim}, layout as L};
use std::{path::Path, sync::Arc, time::Instant};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (data, odir) = (Path::new(&a[1]), Path::new(&a[2]));
    let steps: usize = a.get(3).map_or(200, |s| s.parse().unwrap()); let batch: usize = a.get(4).map_or(10, |s| s.parse().unwrap());
    let out = a.get(5).cloned(); let native = a.get(6).map_or(false, |s| s == "native");
    let t = Instant::now();
    let world = Arc::new(bayenv_world::World::load(data).expect("world"));
    let o = OracleDump::open(odir).expect("oracle");
    println!("world {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
    let cap = std::env::var("BAYENV_CAP").ok().map(|v| v.parse().unwrap()).unwrap_or(o.meta["cap"].as_u64().unwrap() as usize);
    let mut opts = GpuOptions { capacity: cap, ..Default::default() };
    opts.features.accidents = true; opts.start_time = o.meta["time"].as_f64().unwrap() - o.meta["k"].as_f64().unwrap() * opts.dt;
    let t = Instant::now();
    let full = a.get(6).map_or(false, |s| s == "full");
    let native = native || full;
    let mut sim = if full { bayenv_gpu::env::create(world.clone(), opts).expect("sim") } else { GpuSim::with_blob(world.clone(), o.world_blob(), opts).expect("sim") };
    if full {
        let rs: Vec<u32> = sim.read_words("world"); let js: &[u32] = bytemuck::cast_slice(o.buffer("world").unwrap());
        let bad = (0..js.len().min(rs.len())).filter(|&i| js[i] != rs[i]).count();
        println!("native world blob: {} words (js {}), {} mismatching words", rs.len(), js.len(), bad);
    }
    println!("sim create {:.0} ms on {}", t.elapsed().as_secs_f64() * 1e3, sim.gpu.adapter_name());
    // params vs JS
    let jp: &[u32] = bytemuck::cast_slice(o.buffer("params").unwrap()); if std::env::var("BAYENV_CAP").is_ok() { println!("(capacity override: params / scatter comparisons expected to differ)"); }
    let bad: Vec<String> = (0..128).filter(|&i| jp[i] != sim.params_words()[i]).map(|i| format!("{}: js {:#x} rs {:#x}", L::PARAM_FIELDS[i].0, jp[i], sim.params_words()[i])).collect();
    println!("params: {} mismatches {:?}", bad.len(), bad);
    let gas = o.meta["gas"].as_f64().unwrap() as f32;
    if native {
        let mut acts = vec![0f32; cap * 3]; for i in 0..cap { acts[3 * i + 1] = gas; } sim.set_actions(&acts, 0);
        let n = o.meta["n"].as_u64().unwrap() as usize; let seed = o.meta["seed"].as_u64().unwrap() as u32;
        let t = Instant::now();
        let info = if full { bayenv_gpu::env::scatter(&mut sim, n, seed) } else { sim.scatter(n, seed, &|r: &[i32]| r.iter().map(|&e| e as u32).collect()) };
        println!("native scatter {:?} in {:.0} ms (route words: edge ids only)", info, t.elapsed().as_secs_f64() * 1e3);
        let st = sim.read_state();
        for (name, rs) in [("veh", bytemuck::cast_slice::<u32, u8>(&st.veh).to_vec()), ("par", bytemuck::cast_slice::<u32, u8>(&st.par).to_vec()), ("pose", bytemuck::cast_slice::<f32, u8>(&st.pose).to_vec())] {
            let js: &[u32] = bytemuck::cast_slice(o.buffer(name).unwrap()); let rs: &[u32] = bytemuck::cast_slice(&rs);
            let w = if name == "veh" { L::VEH_WORDS } else if name == "par" { L::PAR_WORDS } else { 4 };
            let mut bad = vec![0usize; w]; for i in 0..js.len().min(rs.len()) { if js[i] != rs[i] { bad[i % w] += 1; } }
            println!("scatter {name}: mismatching words per field {:?}", bad);
        }
        // the route words of the pool come from the JS run (guidance codes) until crate::pack lands
        if full { let js: &[u32] = bytemuck::cast_slice(o.buffer("routes").unwrap()); let used = o.meta["scatter"]["routeWords"].as_u64().unwrap() as usize;
            let bad = (0..used).filter(|&i| js[i] != sim.route_mirror[i]).count(); println!("native route words: {used}, {bad} mismatches"); }
        else if let Some(r) = o.buffer("routes") { let words: &[u32] = bytemuck::cast_slice(r); let used = o.meta["scatter"]["routeWords"].as_u64().unwrap() as usize; sim.route_words = 0; sim.append_route(&words[..used]); }
    } else {
        sim.load_oracle_snapshot(odir).expect("snapshot");
    }
    // dispatch lists of the first two steps vs the JS command stream
    let js = o.passes();
    for (k, first) in [(1usize, true), (2, false)] {
        let rs = sim.peek_stages(first, 0); let jsl: Vec<(String, u32)> = js[k].clone();
        let rsl: Vec<(String, u32)> = rs.iter().map(|(a, b)| (a.to_string(), *b)).collect();
        println!("step {k} dispatches: {} js / {} rs, equal: {}", jsl.len(), rsl.len(), jsl == rsl);
        if jsl != rsl { for i in 0..jsl.len().max(rsl.len()) { if jsl.get(i) != rsl.get(i) { println!("  first diff at {i}: js {:?} rs {:?}", jsl.get(i), rsl.get(i)); break; } } }
    }
    let t = Instant::now();
    let mut done = 0; while done < steps { let k = batch.min(steps - done); sim.run_steps(k); done += k; }
    let ms = t.elapsed().as_secs_f64() * 1e3;
    println!("{steps} steps in batches of {batch}: {:.1} ms ({:.3} ms/step); stats driving {} crashes {} contacts {} events dropped {}", ms, ms / steps as f64,
        sim.stats()[L::stat::LIVE_DRIVING], sim.counters.crashes, sim.counters.crash_events, sim.counters.events_dropped);
    if let Some(p) = out {
        let st = sim.read_state(); let mut b = Vec::new();
        b.extend_from_slice(bytemuck::cast_slice(&st.veh)); b.extend_from_slice(bytemuck::cast_slice(&st.pose));
        std::fs::write(&p, b).unwrap(); println!("state -> {p}");
    }
}
