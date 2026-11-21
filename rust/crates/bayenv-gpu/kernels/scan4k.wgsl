// bayenv-gpu kernel (replaces scan.wgsl + the count clears when GpuOptions.fast_scan is on).
// Changed vs the original (gpuSim.js: clear kernel over n, atomic counts, scan.wgsl in place at 512 elements per
// workgroup -> 3 scan + 2 add dispatches for the 2^18 hash cells / K lane keys):
//  * 4096 elements per workgroup (256 threads x 16), so n <= 16.7M needs 3 dispatches: scanFirst, scanInPlace
//    over the block sums, addSums;
//  * scanFirst reads the counts from a separate accumulation buffer (`src`, where hash.hcount / build.count
//    atomically count), writes the exclusive prefix sum to `dst` (the buffer the JS kernels read as starts) and
//    zeroes `src` for the next step, which replaces hash.hclear / build.clear.
// Integer sums: the starts are identical to the original.
struct ScanParams { n: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<uniform> SP: ScanParams;
@group(0) @binding(1) var<storage, read_write> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
@group(0) @binding(3) var<storage, read_write> sums: array<u32>;

const B: u32 = 4096u;
var<workgroup> tmp: array<u32, 4096>;
var<workgroup> tot: array<u32, 256>;

// exclusive scan of tmp[0..4096) in place; returns the block total
fn blockScan(t: u32) -> u32 {
  var acc = 0u;
  for (var j = 0u; j < 16u; j = j + 1u) { let x = tmp[t * 16u + j]; tmp[t * 16u + j] = acc; acc = acc + x; }
  tot[t] = acc;
  workgroupBarrier();
  // Hillis-Steele inclusive scan over the 256 thread totals
  for (var d = 1u; d < 256u; d = d * 2u) {
    var x = 0u; if (t >= d) { x = tot[t - d]; }
    workgroupBarrier();
    tot[t] = tot[t] + x;
    workgroupBarrier();
  }
  let off = tot[t] - acc;   // exclusive prefix of this thread's chunk
  for (var j = 0u; j < 16u; j = j + 1u) { tmp[t * 16u + j] = tmp[t * 16u + j] + off; }
  let total = tot[255];
  workgroupBarrier();
  return total;
}

@compute @workgroup_size(256)
fn scanFirst(@builtin(local_invocation_id) l: vec3u, @builtin(workgroup_id) w: vec3u) {
  let base = w.x * B; let t = l.x;
  for (var j = 0u; j < 16u; j = j + 1u) { let i = base + j * 256u + t; var x = 0u; if (i < SP.n) { x = src[i]; src[i] = 0u; } tmp[j * 256u + t] = x; }
  workgroupBarrier();
  let total = blockScan(t);
  if (t == 0u) { sums[w.x] = total; }
  for (var j = 0u; j < 16u; j = j + 1u) { let i = base + j * 256u + t; if (i < SP.n) { dst[i] = tmp[j * 256u + t]; } }
}

// in place over src (dst / sums: the next level's sums)
@compute @workgroup_size(256)
fn scanInPlace(@builtin(local_invocation_id) l: vec3u, @builtin(workgroup_id) w: vec3u) {
  let base = w.x * B; let t = l.x;
  for (var j = 0u; j < 16u; j = j + 1u) { let i = base + j * 256u + t; var x = 0u; if (i < SP.n) { x = src[i]; } tmp[j * 256u + t] = x; }
  workgroupBarrier();
  let total = blockScan(t);
  if (t == 0u) { sums[w.x] = total; }
  for (var j = 0u; j < 16u; j = j + 1u) { let i = base + j * 256u + t; if (i < SP.n) { src[i] = tmp[j * 256u + t]; } }
}

// dst[block w] += sums[w]
@compute @workgroup_size(256)
fn addSums(@builtin(local_invocation_id) l: vec3u, @builtin(workgroup_id) w: vec3u) {
  let s = sums[w.x];
  if (s == 0u) { return; }
  let base = w.x * B;
  for (var j = 0u; j < 16u; j = j + 1u) { let i = base + j * 256u + l.x; if (i < SP.n) { dst[i] = dst[i] + s; } }
}
