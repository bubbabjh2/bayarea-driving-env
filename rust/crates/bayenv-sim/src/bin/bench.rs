//! cargo run --release -p bayenv-sim --bin bench -- <trace dir (world/pop/run.json)> [steps] [warmup]
//! Runs the trace's env + policy without comparison; prints steps/s and agent-steps/s.
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let dir = &a[1];
    let steps: usize = a.get(2).map(|s| s.parse().unwrap()).unwrap_or(3000);
    let warm: usize = a.get(3).map(|s| s.parse().unwrap()).unwrap_or(0);
    let (mut env, run) = bayenv_sim::harness::env_from_dir(dir);
    let mut pol = bayenv_sim::harness::Policy::new(run["policy"].as_str().unwrap(), run["seed"].as_f64().unwrap());
    for _ in 0..warm { pol.act(&mut env); env.step(1); }
    let mut agent_steps = 0usize;
    let t = std::time::Instant::now();
    for _ in 0..steps { agent_steps += env.driving().count(); pol.act(&mut env); env.step(1); }
    let s = t.elapsed().as_secs_f64();
    if env.sim.profile { let names = ["spawn","hash","contacts","lanes","leaders","controls","drive","integrate","scenes","traffic"]; let tot: u64 = env.sim.phase_ns.iter().sum(); for (k, n) in names.iter().enumerate() { eprintln!("{n:10} {:6.1}%  {:8.3} ms/step", 100.0 * env.sim.phase_ns[k] as f64 / tot as f64, env.sim.phase_ns[k] as f64 / 1e6 / (steps + warm) as f64); } }
    println!("{{\"dir\":\"{dir}\",\"steps\":{steps},\"stepsPerSec\":{:.1},\"agentStepsPerSec\":{:.0},\"meanAgents\":{:.1}}}", steps as f64 / s, agent_steps as f64 / s, agent_steps as f64 / steps as f64);
}
