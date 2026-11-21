//! cargo run --release -p bayenv-sim --bin bench_vec -- <trace dir> <envs> [steps] [warmup] [threads]
//! N independent copies of the trace's env (seeds 1..N), stepped in parallel with the trace policy.
use bayenv_sim::harness::{env_from_dir, Policy};
use bayenv_sim::vec_env::VecEnv;
use rayon::prelude::*;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let dir = &a[1];
    let n: usize = a[2].parse().unwrap();
    let steps: usize = a.get(3).map(|s| s.parse().unwrap()).unwrap_or(1000);
    let warm: usize = a.get(4).map(|s| s.parse().unwrap()).unwrap_or(0);
    if let Some(t) = a.get(5) { rayon::ThreadPoolBuilder::new().num_threads(t.parse().unwrap()).build_global().unwrap(); }
    let (env0, run) = env_from_dir(dir);
    let pname = run["policy"].as_str().unwrap().to_string();
    let mut envs = vec![env0];
    for k in 1..n { let mut e = bayenv_sim::harness::env_from_parts(&run, envs[0].world.clone(), envs[0].pop.clone()); e.reset(Some((k + 1) as f64)); envs.push(e); }
    let mut pols: Vec<Policy> = (0..n).map(|k| Policy::new(&pname, (k + 1) as f64)).collect();
    let mut v = VecEnv::new(envs);
    // BENCH_SYNC=1: one pool pass per step over all envs (the Python VecEnv's pattern: a barrier every step); else
    // every env runs all its steps in one task
    let sync = std::env::var("BENCH_SYNC").map(|v| v == "1").unwrap_or(false);
    let go = |v: &mut VecEnv, pols: &mut Vec<Policy>, steps: usize| -> usize {
        if sync { return (0..steps).map(|_| v.envs.par_iter_mut().zip(pols.par_iter_mut()).map(|(e, p)| { let c = e.driving().count(); p.act(e); e.step(1); c }).sum::<usize>()).sum(); }
        v.envs.par_iter_mut().zip(pols.par_iter_mut()).map(|(e, p)| { let mut c = 0; for _ in 0..steps { c += e.driving().count(); p.act(e); e.step(1); } c }).sum()
    };
    go(&mut v, &mut pols, warm);
    let t = std::time::Instant::now();
    let agent_steps = go(&mut v, &mut pols, steps);
    let s = t.elapsed().as_secs_f64();
    println!("{{\"envs\":{n},\"threads\":{},\"steps\":{steps},\"envStepsPerSec\":{:.0},\"agentStepsPerSec\":{:.0},\"meanAgentsPerEnv\":{:.1}}}", rayon::current_num_threads(), (n * steps) as f64 / s, agent_steps as f64 / s, agent_steps as f64 / (n * steps) as f64);
}
