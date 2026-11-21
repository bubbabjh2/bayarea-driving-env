//! v2 encoding cost next to the sim step (the CPU training path encodes every driving car after every step):
//! encode_prof <trace dir> [steps] [warmup]; ENCODE_ONLY=1 loops the encoding of one state (for profiling)
use bayenv_sim::obs::driving_slots;
use bayenv_sim::obs_v2::V2Encoder;
use std::time::Instant;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let steps: usize = a.get(2).map(|s| s.parse().unwrap()).unwrap_or(1000);
    let warm: usize = a.get(3).map(|s| s.parse().unwrap()).unwrap_or(500);
    let (mut env, run) = bayenv_sim::harness::env_from_dir(&a[1]);
    let mut pol = bayenv_sim::harness::Policy::new(run["policy"].as_str().unwrap(), run["seed"].as_f64().unwrap());
    let mut enc = V2Encoder::new(&env.world);
    let mut buf = vec![0f32; 70];
    for _ in 0..warm { pol.act(&mut env); enc.record_actions(&env.sim); env.sim.step(); for i in driving_slots(&env.sim) { enc.encode_v2(&mut env.sim, i, &mut buf); } }
    if std::env::var("ENCODE_ONLY").is_ok() {
        let slots = driving_slots(&env.sim);
        let t = Instant::now(); let mut n = 0usize;
        while t.elapsed().as_secs_f64() < 10.0 { for &i in &slots { enc.encode_v2(&mut env.sim, i, &mut buf); n += 1; } }
        println!("encode only: {:.0} ns/agent ({} encodes)", t.elapsed().as_secs_f64() * 1e9 / n as f64, n);
        return;
    }
    let (mut ts, mut te, mut agents) = (0.0, 0.0, 0usize);
    for _ in 0..steps {
        pol.act(&mut env); enc.record_actions(&env.sim);
        let t = Instant::now(); env.sim.step(); ts += t.elapsed().as_secs_f64();
        let t = Instant::now(); let sl = driving_slots(&env.sim); agents += sl.len(); for i in sl { enc.encode_v2(&mut env.sim, i, &mut buf); } te += t.elapsed().as_secs_f64();
    }
    println!("per agent-step: sim {:.0} ns, v2 encode {:.0} ns ({:.0} agents/step)", ts * 1e9 / agents as f64, te * 1e9 / agents as f64, agents as f64 / steps as f64);
}
