// Exclusive prefix sum (in place) over u32, 512 elements per workgroup, recursive
// over block sums (driven from gpuSim.js). Used to turn lane / cell counts into starts.

struct ScanParams { n: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<uniform> SP: ScanParams;
@group(0) @binding(1) var<storage, read_write> data: array<u32>;
@group(0) @binding(2) var<storage, read_write> sums: array<u32>;

var<workgroup> tmp: array<u32, 512>;

@compute @workgroup_size(256)
fn scanBlocks(@builtin(local_invocation_id) l: vec3u, @builtin(workgroup_id) w: vec3u) {
  let base = w.x * 512u;
  let t = l.x;
  let i0 = base + 2u * t;
  var a = 0u; var b = 0u;
  if (i0 < SP.n) { a = data[i0]; }
  if (i0 + 1u < SP.n) { b = data[i0 + 1u]; }
  tmp[2u * t] = a; tmp[2u * t + 1u] = b;
  // Blelloch up-sweep
  var off = 1u;
  for (var d = 256u; d > 0u; d = d >> 1u) {
    workgroupBarrier();
    if (t < d) {
      let ai = off * (2u * t + 1u) - 1u;
      let bi = off * (2u * t + 2u) - 1u;
      tmp[bi] = tmp[bi] + tmp[ai];
    }
    off = off << 1u;
  }
  workgroupBarrier();
  if (t == 0u) { sums[w.x] = tmp[511]; tmp[511] = 0u; }
  // down-sweep
  for (var d = 1u; d < 512u; d = d << 1u) {
    off = off >> 1u;
    workgroupBarrier();
    if (t < d) {
      let ai = off * (2u * t + 1u) - 1u;
      let bi = off * (2u * t + 2u) - 1u;
      let x = tmp[ai];
      tmp[ai] = tmp[bi];
      tmp[bi] = tmp[bi] + x;
    }
  }
  workgroupBarrier();
  if (i0 < SP.n) { data[i0] = tmp[2u * t]; }
  if (i0 + 1u < SP.n) { data[i0 + 1u] = tmp[2u * t + 1u]; }
}

@compute @workgroup_size(256)
fn addSums(@builtin(local_invocation_id) l: vec3u, @builtin(workgroup_id) w: vec3u) {
  let s = sums[w.x];
  let i0 = w.x * 512u + 2u * l.x;
  if (i0 < SP.n) { data[i0] = data[i0] + s; }
  if (i0 + 1u < SP.n) { data[i0 + 1u] = data[i0 + 1u] + s; }
}
