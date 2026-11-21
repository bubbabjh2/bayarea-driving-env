//! ms/step on the real world, test/gpu/bench.html protocol: scatter n (seed 11), constant gas 0.3, 100 warm-up
//! steps, then `steps` steps in batches of 20.   cargo run --release -p bayenv-gpu --example bench -- ../data 10000,100000 400 [v2|v3|none] [batch] [--profile]
//! `--profile`: GPU timestamps per dispatch of the first step of each batch (GpuOptions.profile), per stage and kernel,
//! plus the GPU time of the second step encoded normally (one pass); the timed ms/step then includes the profiling cost.
use bayenv_gpu::{defines::ObsMode, sim::GpuOptions};
use std::{sync::Arc, time::Instant};
fn main() {
    let all: Vec<String> = std::env::args().collect();
    let profile = all.iter().any(|x| x == "--profile");
    let a: Vec<String> = all.into_iter().filter(|x| !x.starts_with("--")).collect();
    let data = a.get(1).cloned().unwrap_or("../data".into());
    let sizes: Vec<usize> = a.get(2).map_or("10000,100000".into(), |s| s.clone()).split(',').map(|s| s.parse().unwrap()).collect();
    let steps: usize = a.get(3).map_or(400, |s| s.parse().unwrap());
    let obs = match a.get(4).map(|s| s.as_str()) { Some("v3") => ObsMode::V3, Some("none") => ObsMode::None, _ => ObsMode::V2 };
    let batch: usize = a.get(5).map_or(20, |s| s.parse().unwrap());
    let t = Instant::now(); let world = Arc::new(bayenv_world::World::load(&data).unwrap()); println!("world load {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
    for n in sizes {
        let t = Instant::now();
        let cap = match std::env::var("BAYENV_CAP").as_deref() { Ok("auto") => (n + 1023) / 1024 * 1024, Ok(v) => v.parse().unwrap(), Err(_) => 131072usize.max(n) };
        let mut o = GpuOptions { capacity: cap, obs, profile, ..Default::default() }; o.features.accidents = true; o.apply_env();
        let mut sim = bayenv_gpu::env::create(world.clone(), o).unwrap();
        let acts: Vec<f32> = (0..sim.cap * 3).map(|i| if i % 3 == 1 { 0.3 } else { 0.0 }).collect(); sim.set_actions(&acts, 0);
        let info = bayenv_gpu::env::scatter(&mut sim, n, 11);
        let setup = t.elapsed().as_secs_f64() * 1e3;
        for _ in 0..(100 / batch).max(1) { sim.run_steps(batch); }
        if let Some(p) = sim.profiler.as_mut() { p.reset(); }
        let t = Instant::now(); let mut k = 0; let (mut c46, mut c47, mut m46, mut m47) = (0u64, 0u64, 0u32, 0u32);
        while k < steps { sim.run_steps(batch); k += batch; let s = sim.stats(); c46 += s[46] as u64; c47 += s[47] as u64; m46 = m46.max(s[46]); m47 = m47.max(s[47]); }
        let ms = t.elapsed().as_secs_f64() * 1e3 / steps as f64;
        println!("n {n} placed {} setup {setup:.0} ms: {ms:.3} ms/step ({:.0} steps/s, batch {batch}), driving {}, contacts {}", info.placed, 1000.0 / ms, sim.stats()[32], sim.counters.crash_events);
        let nb = (steps / batch).max(1) as u64;
        println!("  contacts found per step (last of each batch): mean {} max {}; left to the serial pass: mean {} max {}", c46 / nb, m46, c47 / nb, m47);
        if let Some(p) = &sim.profiler {
            let rows = p.report(); let tot: f64 = rows.iter().map(|r| r.ms).sum();
            println!("  profile ({} timestamps, {} profiled steps): GPU step (one pass) {:.3} ms; per-dispatch sum {:.3} ms in {} dispatches",
                if p.inside { "inside-pass" } else { "pass-per-dispatch" }, p.steps, p.step_ms(), tot, rows.iter().map(|r| r.dispatches).sum::<u64>());
            let mut stages: Vec<(&str, f64, u64)> = Vec::new();
            for r in &rows { match stages.iter_mut().find(|s| s.0 == r.stage) { Some(s) => { s.1 += r.ms; s.2 += r.dispatches; } None => stages.push((r.stage, r.ms, r.dispatches)) } }
            println!("  {:10} {:>8} {:>5}", "stage", "ms", "disp");
            for (st, ms, nd) in &stages { println!("  {:10} {:8.3} {:5}", st, ms, nd); }
            println!("  {:10} {:24} {:>8} {:>5}", "stage", "kernel", "ms", "disp");
            for r in &rows { println!("  {:10} {:24} {:8.3} {:5}", r.stage, r.kernel, r.ms, r.dispatches); }
        }
    }
}
