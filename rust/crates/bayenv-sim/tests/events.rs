//! Event log sanity: CRASHED rows == crashed_vehicles, ARRIVED == arrived, STRANDED == stranded, MISSED_EXIT ==
//! missed_exits, rows of a step sorted by (slot, kind). Needs a trace dir (tools/rust/trace.mjs) in BAYENV_EVENTS_TRACE.
use bayenv_sim::harness::{env_from_dir, Policy};
#[test]
fn event_counts_match_counters() {
    // BAYENV_EVENTS_TRACE, else rust/oracle/sim/traces/real2k (make_oracle.sh)
    let dir = std::env::var("BAYENV_EVENTS_TRACE").unwrap_or_else(|_| bayenv_world::oracle::oracle_root().join("sim/traces/real2k").to_string_lossy().to_string());
    if !std::path::Path::new(&dir).join("run.json").exists() { bayenv_world::oracle::skip("events", &format!("no trace at {dir} (BAYENV_EVENTS_TRACE)")); return; }
    let (mut env, run) = env_from_dir(&dir);
    let mut pol = Policy::new(run["policy"].as_str().unwrap(), run["seed"].as_f64().unwrap());
    let _ = env.take_records();
    let (mut crashed, mut arrived, mut stranded, mut missed) = (0u64, 0u64, 0u64, 0u64);
    for _ in 0..run["steps"].as_u64().unwrap() {
        pol.act(&mut env); env.step(1);
        let r = env.take_records();
        for k in 1..r.len() { assert!((r.slot[k - 1], r.kind[k - 1]) <= (r.slot[k], r.kind[k])); }
        for &kd in &r.kind { match kd { 2 => crashed += 1, 1 => arrived += 1, 4 => stranded += 1, 7 => missed += 1, _ => {} } }
    }
    let c = &env.sim.counters;
    eprintln!("crashed {crashed} arrived {arrived} stranded {stranded} missed {missed} (counters {} {} {} {})", c.crashed_vehicles, c.arrived, c.stranded, c.missed_exits);
    assert_eq!(crashed, c.crashed_vehicles); assert_eq!(arrived, c.arrived); assert_eq!(stranded, c.stranded); assert_eq!(missed, c.missed_exits);
}
