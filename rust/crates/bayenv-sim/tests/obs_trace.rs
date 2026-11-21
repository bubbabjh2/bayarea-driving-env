//! Observation port check: BAYENV_OBS_TRACE=<dir of `node tools/rust/trace.mjs ... --obs 1`> (several dirs: ':'-separated)
//! cargo test --release -p bayenv-sim --test obs_trace -- --nocapture
#[test]
fn obs_bit_exact() {
    // BAYENV_OBS_TRACE, else every trace under rust/oracle/sim/traces (make_oracle.sh)
    let dirs = std::env::var("BAYENV_OBS_TRACE").unwrap_or_else(|_| bayenv_world::oracle::dirs_with(&bayenv_world::oracle::oracle_root().join("sim/traces"), "trace.pack").iter().map(|p| p.to_string_lossy().to_string()).collect::<Vec<_>>().join(":"));
    if dirs.is_empty() { bayenv_world::oracle::skip("obs_trace", "BAYENV_OBS_TRACE unset and no rust/oracle/sim/traces/*/trace.pack"); return; }
    for dir in dirs.split(':').filter(|d| !d.is_empty()) {
        let tr = bayenv_sim::pack::Pack::read(&format!("{dir}/trace.pack")).expect("trace.pack");
        assert!(tr.has("obs.n"), "{dir}: trace has no obs sections (record with --obs 1)");
        let (steps, bad, _) = bayenv_sim::harness::check(dir, 20, true);
        for m in &bad { eprintln!("  step {} {}[{}]: js {} rust {}", m.step, m.field, m.slot, m.js, m.rust); }
        assert!(bad.is_empty(), "{dir}: mismatch at step {steps}");
        eprintln!("{dir}: {steps} steps bit-exact (state + obs)");
    }
    // flat API smoke test on the last dir: rows match obs::Obs::scalars cast to f32
    let dir = dirs.split(':').filter(|d| !d.is_empty()).last().unwrap();
    let (mut env, run) = bayenv_sim::harness::env_from_dir(dir);
    let mut pol = bayenv_sim::harness::Policy::new(run["policy"].as_str().unwrap(), run["seed"].as_f64().unwrap());
    for _ in 0..200 { pol.act(&mut env); env.step(1); }
    let slots = bayenv_sim::obs::driving_slots(&env.sim);
    let st = bayenv_sim::obs::flat_stride(true);
    let mut flat = vec![0f32; slots.len() * st];
    bayenv_sim::obs::observe_flat(&mut env.sim, &slots, &mut flat, true);
    for (r, &i) in slots.iter().enumerate() { assert_eq!(flat[r * st], i as f32); }
}
