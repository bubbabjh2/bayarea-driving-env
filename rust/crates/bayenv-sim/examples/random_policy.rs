//! Rust counterpart of examples/random_policy.mjs (no node): real world from data/, every car an agent.
//!   cargo run --release -p bayenv-sim --example random_policy -- [data dir] [n] [steps]
use bayenv_sim::env::Env;
use bayenv_sim::sim::TrafficMode;
use std::path::Path;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = a.get(1).map(|s| s.as_str()).unwrap_or("../data");
    let n: usize = a.get(2).map(|s| s.parse().unwrap()).unwrap_or(2000);
    let steps: usize = a.get(3).map(|s| s.parse().unwrap()).unwrap_or(3000);
    let t0 = std::time::Instant::now();
    let world = Env::load_world(Path::new(data)).expect("world");
    let t1 = t0.elapsed().as_secs_f64();
    let mut env = Env::create_real(world, Path::new(data), n, 1.0, TrafficMode::Full).expect("env");
    println!("world {:.2}s, population + env {:.2}s", t1, t0.elapsed().as_secs_f64() - t1);
    let mut obs = Vec::new();
    let mut rng = bayenv_sim::rng::Rng::new(1.0);
    let t = std::time::Instant::now();
    let mut agent_steps = 0usize;
    for _ in 0..steps {
        let stride = env.observe_into(false, &mut obs);
        let rows = env.n_slots();
        let mut act = vec![0f32; rows * 3];
        for r in 0..rows {
            let o = &obs[r * stride..(r + 1) * stride];
            if o[5] == 0.0 && o[12] == 0.0 { continue; }            // not driving (speed, speedLimit 0)
            agent_steps += 1;
            act[3 * r] = (rng.random() * 0.2 - 0.1) as f32;          // steer
            act[3 * r + 1] = if o[5] < 0.8 * o[12] { 0.4 } else { 0.0 }; // gas
            act[3 * r + 2] = if o[42] < 15.0 { 0.6 } else { 0.0 };   // brake on a close leader (neighbors[0])
        }
        env.act_rows(&act, 0);
        env.step(1);
    }
    let s = t.elapsed().as_secs_f64();
    let st: std::collections::HashMap<_, _> = env.stats().into_iter().collect();
    println!("{steps} steps in {s:.2}s: {:.0} steps/s, {:.0} agent-steps/s (incl. observe), arrived {}, crashes {}", steps as f64 / s, agent_steps as f64 / s, st["arrived"], st["crash_events"]);
}
