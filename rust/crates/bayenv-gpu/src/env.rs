//! One-call setup: world -> packed blob -> `GpuSim` (-> optional scatter population).
use crate::pack::{self, RoutePacker};
use crate::sim::{GpuOptions, GpuSim, ScatterInfo, WorldBlob};
use bayenv_world::World;
use std::sync::Arc;

/// World + GPU sim, packed natively.
pub fn create(world: Arc<World>, opts: GpuOptions) -> Result<GpuSim, String> {
    let blob: WorldBlob = pack::build_world_blob(&world, opts.max_blocks).into();
    let mut sim = GpuSim::with_blob(world.clone(), blob, opts)?;
    sim.packer = Some(Arc::new(RoutePacker::new(&world.graph)));
    Ok(sim)
}

/// `scatter(n, {seed})` with the native route packer.
pub fn scatter(sim: &mut GpuSim, n: usize, seed: u32) -> ScatterInfo {
    let packer = sim.packer.get_or_insert_with(|| Arc::new(RoutePacker::new(&sim.world.graph))).clone();
    let world = sim.world.clone();
    sim.scatter(n, seed, &|r: &[i32]| packer.pack_route(&world.graph, r)[..r.len()].to_vec())
}
