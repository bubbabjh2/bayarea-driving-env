//! cargo run --release -p bayenv-sim --bin trace_check -- <trace dir> [max mismatches]
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let dir = &a[1];
    let max = a.get(2).map(|s| s.parse().unwrap()).unwrap_or(20);
    let (steps, bad, sim) = bayenv_sim::harness::check(dir, max, true);
    if bad.is_empty() { println!("OK {dir}: {steps} steps bit-exact (rust sim {:.3}s, {:.0} steps/s)", sim, steps as f64 / sim); return; }
    println!("MISMATCH {dir} at step {steps}:");
    for m in &bad { println!("  step {} {}[{}]: js {} rust {}", m.step, m.field, m.slot, m.js, m.rust); }
    std::process::exit(1);
}
