//! v2 encoder check: BAYENV_V2_TRACE=<dir of `node tools/rust/v2_oracle.mjs ...`> (several dirs: ':'-separated)
//! cargo test --release -p bayenv-sim --test obs_v2_trace -- --nocapture
#[test]
fn v2_bit_exact() {
    let Ok(dirs) = std::env::var("BAYENV_V2_TRACE") else { eprintln!("BAYENV_V2_TRACE unset: skip"); return };
    for dir in dirs.split(':').filter(|d| !d.is_empty()) {
        let r = bayenv_sim::obs_v2::check_v2(dir, 20, true);
        for m in &r.mismatches { eprintln!("  step {} slot {} v2[{}]: js {} rust {}", m.step, m.slot, m.index, m.js, m.rust); }
        for m in &r.near_mismatches { eprintln!("  NEAR step {} slot {} {}: js {:?} rust {:?}", m.step, m.slot, m.field, m.js, m.rust); }
        assert!(r.mismatches.is_empty(), "{dir}: v2 mismatch at step {}", r.steps);
        assert!(r.near_mismatches.is_empty(), "{dir}: eyes.near mismatch at step {}", r.steps);
        assert_eq!(r.state_unchanged, Some(true), "{dir}: observing changed the sim state");
        eprintln!("{dir}: {} steps, {} rows bit-exact ({} with eyes.near), {:.0} ns/agent", r.steps, r.rows, r.near_rows, r.encode_ns);
    }
}
