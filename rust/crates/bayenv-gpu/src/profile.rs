//! GPU timestamp profiler (`GpuOptions.profile`), the counterpart of `gpuSim.js` `opts.profile` / `Profiler`.
//!
//! With profiling on, the first step of every `run_steps` batch is encoded with one timestamp pair per dispatch:
//! `pass.write_timestamp` between dispatches of a single pass when the adapter has `TIMESTAMP_QUERY_INSIDE_PASSES`
//! (dispatch-boundary counters), otherwise one compute pass per dispatch with `timestamp_writes` (Apple GPUs only
//! sample at stage boundaries). The second step of the batch is encoded as usual (one pass) with a timestamp pair
//! around the pass: that is the real GPU time of a step, the per-dispatch sum includes the per-pass overheads.
//! Results are summed over all profiled batches; `report()` gives ms per profiled step per (stage, kernel).
use std::collections::HashMap;

pub struct Profiler {
    pub qs: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    read: wgpu::Buffer,
    pub cap: u32,
    pub inside: bool,
    period_ns: f64,
    /// (stage, kernel) of the queries written this batch (2 per entry, then 2 for the whole-step pass)
    pub pending: Vec<(&'static str, &'static str)>,
    pub step_pair: bool,
    rows: Vec<ProfileRow>,
    index: HashMap<(&'static str, &'static str), usize>,
    pub steps: u64,
    pub step_ms: f64,
    pub step_n: u64,
}

#[derive(Clone, Debug)]
pub struct ProfileRow { pub stage: &'static str, pub kernel: &'static str, pub ms: f64, pub dispatches: u64 }

impl Profiler {
    pub const CAP: u32 = 512;
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, inside: bool) -> Profiler {
        let cap = Self::CAP;
        let qs = device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("profile"), ty: wgpu::QueryType::Timestamp, count: cap });
        let resolve = device.create_buffer(&wgpu::BufferDescriptor { label: Some("profile.resolve"), size: cap as u64 * 8, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let read = device.create_buffer(&wgpu::BufferDescriptor { label: Some("profile.read"), size: cap as u64 * 8, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        Profiler { qs, resolve, read, cap, inside, period_ns: queue.get_timestamp_period() as f64, pending: Vec::new(), step_pair: false, rows: Vec::new(), index: HashMap::new(), steps: 0, step_ms: 0.0, step_n: 0 }
    }
    /// Index of the whole-step query pair (queries 0, 1; dispatch i uses 2 + 2i, 3 + 2i).
    pub fn step_index(&self) -> u32 { 0 }
    pub fn dispatch_index(i: usize) -> u32 { 2 + 2 * i as u32 }
    /// Max dispatches per profiled step.
    pub fn max_dispatches(&self) -> usize { (self.cap as usize - 2) / 2 }
    /// Resolve this batch's queries into the read buffer (after the batch's passes).
    pub fn resolve(&self, enc: &mut wgpu::CommandEncoder) {
        if self.pending.is_empty() && !self.step_pair { return; }
        let n = 2 + (self.pending.len() * 2) as u32;
        enc.resolve_query_set(&self.qs, 0..n, &self.resolve, 0); enc.copy_buffer_to_buffer(&self.resolve, 0, &self.read, 0, n as u64 * 8);
    }
    /// After the submit completed: read the timestamps and accumulate.
    pub fn collect(&mut self, device: &wgpu::Device) {
        if self.pending.is_empty() && !self.step_pair { return; }
        let sl = self.read.slice(..);
        sl.map_async(wgpu::MapMode::Read, |r| r.expect("map profile"));
        device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        {
            let m = sl.get_mapped_range().expect("profile range"); let t: &[u64] = bytemuck::cast_slice(&m);
            let ms = |a: u64, b: u64| b.saturating_sub(a) as f64 * self.period_ns * 1e-6;
            for (i, &(stage, kernel)) in self.pending.iter().enumerate() {
                let q = Self::dispatch_index(i) as usize; let d = ms(t[q], t[q + 1]);
                let j = *self.index.entry((stage, kernel)).or_insert_with(|| { self.rows.push(ProfileRow { stage, kernel, ms: 0.0, dispatches: 0 }); self.rows.len() - 1 });
                self.rows[j].ms += d; self.rows[j].dispatches += 1;
            }
            if !self.pending.is_empty() { self.steps += 1; }
            if self.step_pair { self.step_ms += ms(t[0], t[1]); self.step_n += 1; }
        }
        self.read.unmap();
        self.pending.clear(); self.step_pair = false;
    }
    pub fn reset(&mut self) { self.rows.clear(); self.index.clear(); self.steps = 0; self.step_ms = 0.0; self.step_n = 0; }
    /// Per (stage, kernel): ms per profiled step (summed over the dispatches of that kernel in the step), dispatches per step.
    pub fn report(&self) -> Vec<ProfileRow> {
        let s = self.steps.max(1) as f64;
        self.rows.iter().map(|r| ProfileRow { stage: r.stage, kernel: r.kernel, ms: r.ms / s, dispatches: (r.dispatches as f64 / s).round() as u64 }).collect()
    }
    /// Real GPU ms of a whole step encoded as one pass (mean over the profiled batches).
    pub fn step_ms(&self) -> f64 { self.step_ms / self.step_n.max(1) as f64 }
}
