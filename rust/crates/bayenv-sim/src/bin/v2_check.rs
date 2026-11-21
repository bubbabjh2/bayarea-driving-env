//! cargo run --release -p bayenv-sim --bin v2_check -- <v2_oracle dir> [max mismatches]
//! Compares obs_v2::V2Encoder against tools/rust/v2_oracle.mjs bit for bit, the 70 inputs and (when the pack has v2.near)
//! the driver's eyes.near fields (+ checks that observing leaves the sim unchanged). BAYENV_V2_BATCH=1: encode_batch_near.
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let dir = &a[1];
    let max = a.get(2).map(|s| s.parse().unwrap()).unwrap_or(20);
    let r = bayenv_sim::obs_v2::check_v2(dir, max, true);
    for m in &r.near_mismatches { println!("  NEAR step {} slot {} {}: js {:?} rust {:?}", m.step, m.slot, m.field, m.js, m.rust); }
    if r.mismatches.is_empty() && r.near_mismatches.is_empty() {
        println!("OK {dir}: {} steps, {} v2 rows bit-exact, {} eyes.near rows bit-exact ({} fields), encode {:.0} ns/agent ({} encodes; observe_one alone {:.0} ns), sim state unchanged by observing: {:?}", r.steps, r.rows, r.near_rows, bayenv_sim::obs_v2::NEAR_N, r.encode_ns, r.encodes, r.observe_ns, r.state_unchanged);
        if r.state_unchanged == Some(false) { std::process::exit(2); }
        return;
    }
    println!("MISMATCH {dir} at step {} ({} rows ok):", r.steps, r.rows);
    for m in &r.mismatches { println!("  step {} slot {} v2[{}]: js {} rust {}", m.step, m.slot, m.index, m.js, m.rust); }
    std::process::exit(1);
}
