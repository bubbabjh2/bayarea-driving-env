//! Host-optimisation check: two sims with different `GpuOptions` flags, same world / scatter / actions, stepped
//! side by side; per buffer the number of differing words after N steps (veh, par, pose, inputs = obs, ctl) and the
//! aggregate counters.
//!   BAYENV_GPU_OPTS_A="fused_resolve=0,sorted_order=0" BAYENV_GPU_OPTS="sorted_order=1" \
//!   cargo run --release -p bayenv-gpu --example opt_check -- <data> [n=10000] [steps=200] [batch=10]
//! Flags A default to the JS dispatch structure (every host optimisation off); B to the defaults + BAYENV_GPU_OPTS.
use bayenv_gpu::{layout as L, sim::GpuOptions};
use std::sync::Arc;

fn opts(cap: usize, var: &str, js: bool) -> GpuOptions {
    // both sims on the legacy model (= the JS kernels) unless a flag list sets cpu_connector_join / contact_tangent
    let mut o = GpuOptions { capacity: cap, deterministic_contacts: false, deterministic_spawn: false, deterministic_events: false, deterministic_leaders: false, ..GpuOptions::default().js_model() }; o.features.accidents = true;
    if js { for f in ["indirect_resolve", "fused_resolve", "sorted_order", "fast_scan", "fast_search"] { o.set_flag(f, false).unwrap(); } o.zero_init_workgroup = true; }
    if let Ok(v) = std::env::var(var) { for kv in v.split(',').filter(|x| !x.is_empty()) {
        let (k, b) = kv.split_once('=').unwrap_or((kv, "1")); o.set_flag(k.trim(), b.trim() != "0").unwrap(); } }
    o
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = a.get(1).cloned().unwrap_or("../data".into());
    let n: usize = a.get(2).map_or(10000, |s| s.parse().unwrap());
    let steps: usize = a.get(3).map_or(200, |s| s.parse().unwrap());
    let batch: usize = a.get(4).map_or(10, |s| s.parse().unwrap());
    let world = Arc::new(bayenv_world::World::load(&data).unwrap());
    let cap = (n + 1023) / 1024 * 1024;
    let mut sims: Vec<_> = [opts(cap, "BAYENV_GPU_OPTS_A", true), opts(cap, "BAYENV_GPU_OPTS", false)].into_iter().map(|o| {
        println!("indirect_resolve {} fused_resolve {} sorted_order {} fast_scan {} zero_init_workgroup {}", o.indirect_resolve, o.fused_resolve, o.sorted_order, o.fast_scan, o.zero_init_workgroup);
        let mut s = bayenv_gpu::env::create(world.clone(), o).unwrap();
        let acts: Vec<f32> = (0..s.cap * 3).map(|i| if i % 3 == 1 { 0.3 } else { 0.0 }).collect(); s.set_actions(&acts, 0);
        bayenv_gpu::env::scatter(&mut s, n, 11); s
    }).collect();
    let mut done = 0;
    while done < steps { let k = batch.min(steps - done); for s in sims.iter_mut() { s.run_steps(k); } done += k; }
    let mut total = 0;
    for name in ["veh", "par", "pose", "inputs", "ctl"] {
        let (x, y) = (sims[0].read_words(name), sims[1].read_words(name));
        let d = x.iter().zip(&y).filter(|(p, q)| p != q).count();
        // ctl holds the junction linked lists (atomicExchange heads): their order, not their content, varies run to run
        if name != "ctl" { total += d; }
        println!("{name:7} {} words, {d} differ{}", x.len(), if name == "ctl" { " (list order is atomic-order dependent, also between identical runs)" } else { "" });
    }
    let (s0, s1) = (sims[0].stats(), sims[1].stats());
    let sd: Vec<usize> = (0..64).filter(|&i| s0[i] != s1[i] && i != 46 && i != 47).collect();
    println!("stats differ at {:?}; driving {} / {}, contacts {} / {}, crashes {} / {}", sd, s0[L::stat::LIVE_DRIVING], s1[L::stat::LIVE_DRIVING],
        sims[0].counters.crash_events, sims[1].counters.crash_events, sims[0].counters.crashes, sims[1].counters.crashes);
    println!("{}", if total == 0 && sd.is_empty() { "BIT-IDENTICAL" } else { "DIFFERENT" });
}
