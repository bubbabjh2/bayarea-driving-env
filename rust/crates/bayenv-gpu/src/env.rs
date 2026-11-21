//! One-call setup: world -> packed blob -> `GpuSim` (-> optional scatter population).
use crate::pack::{self, RoutePacker};
use crate::sim::{GpuOptions, GpuSim, ScatterInfo, WorldBlob};
use bayenv_world::World;
use std::sync::Arc;

/// World + GPU sim, packed natively.
pub fn create(world: Arc<World>, opts: GpuOptions) -> Result<GpuSim, String> { create_on(crate::gpu::Gpu::new_with(opts.gpu_features())?, world, opts) }

/// As `create` on a given context (`sim.gpu.share()` for several sims on one device).
pub fn create_on(gpu: crate::gpu::Gpu, world: Arc<World>, opts: GpuOptions) -> Result<GpuSim, String> {
    let blob: WorldBlob = pack::build_world_blob(&world, opts.max_blocks).into();
    let mut sim = GpuSim::with_blob_on(gpu, world.clone(), blob, opts)?;
    sim.packer = Some(Arc::new(RoutePacker::new(&world.graph)));
    Ok(sim)
}

/// `scatter(n, {seed})` with the native route packer.
pub fn scatter(sim: &mut GpuSim, n: usize, seed: u32) -> ScatterInfo {
    let packer = sim.packer.get_or_insert_with(|| Arc::new(RoutePacker::new(&sim.world.graph))).clone();
    let world = sim.world.clone();
    sim.scatter(n, seed, &|r: &[i32]| packer.pack_route(&world.graph, r)[..r.len()].to_vec())
}

/// `n` independent sims on one device sharing the world (the blob is packed once; each sim gets its own buffers,
/// since crash scenes write into its world buffer).
pub fn create_many(world: Arc<World>, opts: &GpuOptions, n: usize) -> Result<Vec<GpuSim>, String> {
    let blob: WorldBlob = pack::build_world_blob(&world, opts.max_blocks).into();
    let packer = Arc::new(RoutePacker::new(&world.graph));
    let base = crate::gpu::Gpu::new_with(opts.gpu_features())?;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let mut sim = GpuSim::with_blob_on(base.share(), world.clone(), blob.clone(), opts.clone())?;
        sim.packer = Some(packer.clone());
        out.push(sim);
    }
    Ok(out)
}

/// Step several sims on one device: submit all, wait once, then process each (obs / veh outputs optional per sim).
pub fn step_many(sims: &mut [GpuSim], k: usize, mut obs: Vec<Option<&mut [f32]>>, mut veh: Vec<Option<&mut [u32]>>) {
    if sims.is_empty() { return; }
    for s in sims.iter_mut() { s.sync(); }
    obs.resize_with(sims.len(), || None); veh.resize_with(sims.len(), || None);
    let pend: Vec<_> = sims.iter_mut().zip(obs.iter().zip(veh.iter())).map(|(s, (o, v))| s.submit_steps(k, o.as_ref().map_or(0, |x| x.len()), v.as_ref().map_or(0, |x| x.len()))).collect();
    sims[0].gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    for ((s, p), (o, v)) in sims.iter_mut().zip(pend).zip(obs.into_iter().zip(veh.into_iter())) { s.finish_steps(p, o, v); }
}

/// As `step_many`, the observation rows of sim i written to `buf` at byte `i * rows * obs_size * fmt.elem_bytes()`
/// (one `ObsRing` slot for all sims). `lazy`: return after the submits, each sim keeping its batch pending (`sync`);
/// otherwise wait once and finish every sim. Request the map of `buf` between the submits and the wait: `on_submitted`.
pub fn step_many_obs(sims: &mut [GpuSim], k: usize, buf: &std::sync::Arc<wgpu::Buffer>, rows: usize, fmt: crate::obs::ObsFmt, lazy: bool, on_submitted: impl FnOnce()) {
    if sims.is_empty() { on_submitted(); return; }
    for s in sims.iter_mut() { s.sync(); }
    let per = (rows * sims[0].obs_size * fmt.elem_bytes()) as u64;
    let pend: Vec<_> = sims.iter_mut().enumerate().map(|(i, s)| {
        let d = crate::obs::ObsDst { buf: buf.clone(), offset: i as u64 * per, rows, fmt };
        if k == 0 { s.submit_obs(&d); None } else { Some(s.submit_steps_obs(k, 0, Some(&d), 0)) }
    }).collect();
    on_submitted();
    if lazy { for (s, p) in sims.iter_mut().zip(pend) { if let Some(p) = p { s.defer(p); } } return; }
    sims[0].gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    for (s, p) in sims.iter_mut().zip(pend) { if let Some(p) = p { s.finish_steps(p, None, None); } }
}
