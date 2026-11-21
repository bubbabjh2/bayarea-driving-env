//! Batched independent envs for RL throughput: each env is single-threaded and deterministic; `step` runs the
//! envs on the rayon pool (one env per task), so results do not depend on the thread count.
use crate::env::Env;
use rayon::prelude::*;

pub struct VecEnv { pub envs: Vec<Env> }

impl VecEnv {
    pub fn new(envs: Vec<Env>) -> VecEnv { VecEnv { envs } }
    pub fn len(&self) -> usize { self.envs.len() }
    pub fn is_empty(&self) -> bool { self.envs.is_empty() }
    /// Step every env k times in parallel.
    pub fn step(&mut self, k: usize) { self.envs.par_iter_mut().for_each(|e| e.step(k)); }
    /// Run `f(env_index, env)` on every env in parallel (act / observe / step loops written by the caller).
    pub fn for_each(&mut self, f: impl Fn(usize, &mut Env) + Sync + Send) { self.envs.par_iter_mut().enumerate().for_each(|(i, e)| f(i, e)); }
}
