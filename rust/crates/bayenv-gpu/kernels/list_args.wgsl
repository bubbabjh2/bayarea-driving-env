// bayenv-gpu helper (not part of src/gpu/kernels): indirect dispatch sizes for the lane-list-order kernels
// (kernels/sorted.wgsl): starts[K] lane-list entries + cap slots, at 128 (largs[0..3]) and 256 (largs[4..7])
// threads per workgroup. Runs after build.sortLanes.
struct LP { K: u32, cap: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<uniform> LA: LP;
@group(0) @binding(1) var<storage, read> starts: array<u32>;
@group(0) @binding(2) var<storage, read_write> largs: array<u32>;
@compute @workgroup_size(1)
fn main() {
  let n = starts[LA.K] + LA.cap;
  largs[0] = (n + 127u) / 128u; largs[1] = 1u; largs[2] = 1u;
  largs[4] = (n + 255u) / 256u; largs[5] = 1u; largs[6] = 1u;
}
