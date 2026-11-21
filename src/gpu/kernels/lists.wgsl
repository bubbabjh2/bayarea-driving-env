// Read-only footprint lane lists built by build.wgsl this step (spatial.js LaneIndex).
// Key = edge.laneBase + lane; starts[key] .. starts[key+1] indexes `order`, sorted by s
// (ties: lower slot first). A car is listed in every lane laneLo..laneHi its body
// overlaps; vidx[i][lane - laneLo] is its position in that lane's list.
// Including kernels must declare `veh` (and `par`).

@group(2) @binding(0) var<storage, read> starts: array<u32>;
@group(2) @binding(1) var<storage, read> order: array<u32>;
@group(2) @binding(2) var<storage, read> vidx: array<vec4u>;

fn lkey(e: u32, l: u32) -> u32 { return eLaneBase(e) + l; }

fn laneFirst(e: u32, l: u32) -> i32 {
  let k = lkey(e, l);
  let a = starts[k];
  if (starts[k + 1u] > a) { return i32(order[a]); }
  return -1;
}
fn laneLast(e: u32, l: u32) -> i32 {
  let k = lkey(e, l);
  let b = starts[k + 1u];
  if (b > starts[k]) { return i32(order[b - 1u]); }
  return -1;
}
// next car ahead of slot i in `lane` of i's edge (-1 if i is not listed there / none)
fn aheadIn(i: u32, lane: u32) -> i32 {
  let v = veh[i];
  if (lane < vLo(v) || lane > vHi(v)) { return -1; }
  let k = lkey(v.edge, lane);
  let p = vidx[i][lane - vLo(v)] + 1u;
  if (p < starts[k + 1u]) { return i32(order[p]); }
  return -1;
}
fn behindIn(i: u32, lane: u32) -> i32 {
  let v = veh[i];
  if (lane < vLo(v) || lane > vHi(v)) { return -1; }
  let k = lkey(v.edge, lane);
  let p = vidx[i][lane - vLo(v)];
  if (p > starts[k]) { return i32(order[p - 1u]); }
  return -1;
}
// [ahead (first with s_j >= s), behind (last with s_j < s)] in (e, l), skipping `self`
fn neighborsAt(e: u32, l: u32, s: f32, self_: u32) -> vec2i {
  let k = lkey(e, l);
  let a = starts[k];
  let b = starts[k + 1u];
  var lo = a;
  var hi = b;
  loop {
    if (lo >= hi) { break; }
    let m = (lo + hi) >> 1u;
    if (veh[order[m]].s < s) { lo = m + 1u; } else { hi = m; }
  }
  var p = lo;
  loop { if (p >= b || order[p] != self_) { break; } p = p + 1u; }
  var q = i32(lo) - 1;
  loop { if (q < i32(a) || order[u32(q)] != self_) { break; } q = q - 1; }
  var r = vec2i(-1, -1);
  if (p < b) { r.x = i32(order[p]); }
  if (q >= i32(a)) { r.y = i32(order[u32(q)]); }
  return r;
}
// first car at or ahead of `a` in `lane` that physically blocks it (skips shoulder cars)
fn blocking(a0: i32, lane: u32) -> i32 {
  var a = a0;
  for (var n = 0; n < 64; n = n + 1) {
    // layout v8: body-based - skip a car only when its body is off the lanes (laneHi < laneLo)
    if (a < 0 || vHi(veh[u32(a)]) >= vLo(veh[u32(a)])) { break; }
    a = aheadIn(u32(a), lane);
  }
  return a;
}
