// bayenv-gpu model patch of the v3 body perception (src/gpu/kernels/perceive.wgsl), with GpuOptions::sector_snap (the
// host prepends `const MF_SECTOR_SNAP: bool`). A body straight ahead or behind in my lane sits on a sector boundary
// (atan2(left, fwd) = 0 or +-pi): the sign of a lateral offset of ~1e-3 m (f32 world coordinates on the GPU, f64 on
// the CPU) decided between sectors 3 / 4 or 0 / 7, so the CPU and GPU filed the same car in different body slots
// (inputs 102..357, 3..4 % of the rows at 10,000 commuters) and the mirror clearances (77 / 78) on different sides.
// Both backends now snap |left| < 1 cm to +0 (bayenv-sim SimOpts::v3_sector_snap, obs.rs sense_bodies).
//#orig sector-fn
fn sectorOf3(fwd: f32, left: f32) -> u32 {
//#end
//#patch sector-fn
fn snapLeft3(l: f32) -> f32 { return select(l, 0.0, MF_SECTOR_SNAP && abs(l) < 0.01); }
fn sectorOf3(fwd: f32, left: f32) -> u32 {
//#end
//#orig sector-offer
        let fwd = dx * c + dz * s; let left = dx * s - dz * c;
//#end
//#patch sector-offer
        let fwd = dx * c + dz * s; let left = snapLeft3(dx * s - dz * c);
//#end
//#orig sector-force
        let b0 = sectorOf3(dx * c + dz * s, dx * s - dz * c) * 2u;
//#end
//#patch sector-force
        let b0 = sectorOf3(dx * c + dz * s, snapLeft3(dx * s - dz * c)) * 2u;
//#end
