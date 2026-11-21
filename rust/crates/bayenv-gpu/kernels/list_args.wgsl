// bayenv-gpu helper (not part of src/gpu/kernels): indirect dispatch sizes for the lane-list-order kernels
// (kernels/sorted.wgsl): starts[K] lane-list entries + cap slots, at 32 / 64 / 128 / 256 threads per workgroup
// (largs[4 w ..], w = 0..3). Runs after build.sortLanes.
struct LP { K: u32, cap: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<uniform> LA: LP;
@group(0) @binding(1) var<storage, read> starts: array<u32>;
@group(0) @binding(2) var<storage, read_write> largs: array<u32>;
@compute @workgroup_size(1)
fn main() {
  let n = starts[LA.K] + LA.cap;
  for (var w = 0u; w < 4u; w = w + 1u) { let s = 32u << w; largs[4u * w] = (n + s - 1u) / s; largs[4u * w + 1u] = 1u; largs[4u * w + 2u] = 1u; }
}
