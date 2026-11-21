//! v2 encoder check: BAYENV_V2_TRACE=<dir of `node tools/rust/v2_oracle.mjs ...`> (several dirs: ':'-separated)
//! cargo test --release -p bayenv-sim --test obs_v2_trace -- --nocapture
#[test]
fn v2_bit_exact() {
    // BAYENV_V2_TRACE, else every run under rust/oracle/sim/v2 (make_oracle.sh)
    let dirs = std::env::var("BAYENV_V2_TRACE").unwrap_or_else(|_| bayenv_world::oracle::dirs_with(&bayenv_world::oracle::oracle_root().join("sim/v2"), "run.json").iter().map(|p| p.to_string_lossy().to_string()).collect::<Vec<_>>().join(":"));
    if dirs.is_empty() { bayenv_world::oracle::skip("obs_v2_trace", "BAYENV_V2_TRACE unset and no rust/oracle/sim/v2/*"); return; }
    for dir in dirs.split(':').filter(|d| !d.is_empty()) {
        let r = bayenv_sim::obs_v2::check_v2(dir, 20, true);
        for m in &r.mismatches { eprintln!("  step {} slot {} v2[{}]: js {} rust {}", m.step, m.slot, m.index, m.js, m.rust); }
        for m in &r.near_mismatches { eprintln!("  NEAR step {} slot {} {}: js {:?} rust {:?}", m.step, m.slot, m.field, m.js, m.rust); }
        assert!(r.mismatches.is_empty(), "{dir}: v2 mismatch at step {}", r.steps);
        for m in &r.teacher_mismatches { eprintln!("  TEACHER step {} slot {} {}: js {:?} rust {:?}", m.step, m.slot, m.field, m.js, m.rust); }
        assert!(r.teacher_mismatches.is_empty(), "{dir}: TeacherObs mismatch at step {}", r.steps);
        assert!(r.near_mismatches.is_empty(), "{dir}: eyes.near mismatch at step {}", r.steps);
        assert_eq!(r.state_unchanged, Some(true), "{dir}: observing changed the sim state");
        eprintln!("{dir}: {} steps, {} rows bit-exact ({} with eyes.near, {} with TeacherObs), {:.0} ns/agent", r.steps, r.rows, r.near_rows, r.teacher_rows, r.encode_ns);
    }
}
