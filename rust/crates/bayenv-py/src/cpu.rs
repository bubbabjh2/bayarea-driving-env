//! CPU backend (crate bayenv-sim) behind the same Python calls / shapes as the GPU backend.
//! obs: "v2" = the 70-float vector (bayenv_sim::obs_v2::V2Encoder, encoded after every step as the JS driver loop
//! senses), "v3" = the 358-float v3 vector (obs.rs fill_obs3), "flat" = the 77-float structured obs (CPU only),
//! "none". Rows = slots 0..n_slots (non-driving rows 0).
use bayenv_sim::env::Env as CEnv;
use bayenv_sim::obs::{flat_stride, observe_flat, driving_slots, OBS_N};
use bayenv_sim::obs_v2::V2Encoder;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CpuObs { V2, V3, Flat, None }
impl CpuObs { pub fn width(self) -> usize { match self { CpuObs::V2 => 70, CpuObs::V3 => bayenv_sim::obs::OBS3, CpuObs::Flat => OBS_N, CpuObs::None => 0 } } }

pub struct CpuBackend { pub env: CEnv, pub obs: CpuObs, enc: Option<V2Encoder>, buf: Vec<f32>, rows: usize, fresh: bool, scratch: Vec<f32> }

impl CpuBackend {
    pub fn new(env: CEnv, obs: CpuObs) -> CpuBackend {
        let enc = if obs == CpuObs::V2 { Some(V2Encoder::new(&env.world)) } else { None };
        CpuBackend { env, obs, enc, buf: Vec::new(), rows: 0, fresh: false, scratch: Vec::new() }
    }
    pub fn n_slots(&self) -> usize { self.env.sim.vs.high }
    /// k steps; the v2 encoder records the commands before and senses every driving car after each step
    pub fn step(&mut self, k: usize) {
        if let Some(enc) = self.enc.as_mut() {
            for _ in 0..k {
                enc.record_actions(&self.env.sim);
                self.env.sim.step();
                let n = self.env.sim.vs.high; let w = 70;
                self.buf.clear(); self.buf.resize(n * w, 0.0);
                for i in driving_slots(&self.env.sim) { enc.encode_v2(&mut self.env.sim, i, &mut self.buf[i * w..(i + 1) * w]); }
                self.rows = n;
            }
            for a in self.env.sim.actions.iter_mut() { *a = None; }
            self.fresh = true;
        } else { self.env.step(k); self.fresh = false; }
    }
    /// Observations into `out` (n_slots x width).
    pub fn observe_into(&mut self, out: &mut [f32]) {
        let w = self.obs.width(); let n = self.n_slots();
        match self.obs {
            CpuObs::V2 => { if !self.fresh || self.rows != n {   // not stepped yet: sense now (as Driver.sense would)
                    let enc = self.enc.as_mut().unwrap(); self.buf.clear(); self.buf.resize(n * w, 0.0);
                    for i in driving_slots(&self.env.sim) { enc.encode_v2(&mut self.env.sim, i, &mut self.buf[i * w..(i + 1) * w]); }
                    self.rows = n; self.fresh = true; }
                out[..n * w].copy_from_slice(&self.buf[..n * w]); }
            CpuObs::Flat | CpuObs::V3 => {
                let v3 = self.obs == CpuObs::V3; let st = flat_stride(v3);
                let slots = driving_slots(&self.env.sim);
                self.scratch.clear(); self.scratch.resize(slots.len() * st, 0.0);
                observe_flat(&mut self.env.sim, &slots, &mut self.scratch, v3);
                out[..n * w].fill(0.0);
                let off = if v3 { OBS_N } else { 0 };
                for (r, &i) in slots.iter().enumerate() { out[i * w..(i + 1) * w].copy_from_slice(&self.scratch[r * st + off..r * st + off + w]); }
            }
            CpuObs::None => {}
        }
    }
}
