// Stage (a): per-(edge, lane) occupancy lists from the body FOOTPRINT (spatial.js LaneIndex):
// every DRIVING / CRASHED car is listed in each lane laneLo..laneHi it overlaps (up to
// MAX_OCC = 4). Cars in a station lot (REFUELING) are not on the road.
//   tick      : advance the clock, reset per-step stats              (1 thread)
//   clear     : counts[0 .. K+1) = 0
//   count     : atomic count per lane key, rank-in-bucket per entry (kept in vidx)
//   (scan.wgsl: exclusive prefix sum over counts -> starts)
//   scatter   : items[start + rank] = slot (unsorted)
//   sortLanes : rank sort inside each bucket by (s, slot) -> order[]; vidx[i][l - laneLo] = position
// #include common.wgsl

@group(1) @binding(0) var<storage, read> veh: array<Veh>;
@group(1) @binding(1) var<storage, read> par: array<VPar>;
@group(1) @binding(2) var<storage, read_write> counts: array<atomic<u32>>;
@group(1) @binding(3) var<storage, read_write> items: array<u32>;
@group(1) @binding(4) var<storage, read_write> order: array<u32>;
@group(1) @binding(5) var<storage, read_write> vidx: array<vec4u>;

fn listed(st: u32) -> bool { return st == DRIVING || st == CRASHED; }

@compute @workgroup_size(1)
fn tick() {
  atomicAdd(&io[IO_STEP], 1u);
  for (var k = 32u; k < 64u; k = k + 1u) { atomicStore(&io[IO_STATS + k], 0u); }
}

@compute @workgroup_size(256)
fn clear(@builtin(global_invocation_id) g: vec3u) {
  if (g.x < P.K + 1u) { atomicStore(&counts[g.x], 0u); }
}

@compute @workgroup_size(256)
fn count(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= P.cap) { return; }
  let v = veh[i];
  if (!listed(v.state)) { return; }
  let base = eLaneBase(v.edge);
  var r = vec4u(0u);
  for (var l = vLo(v); l <= vHi(v); l = l + 1u) { r[l - vLo(v)] = atomicAdd(&counts[base + l], 1u); }
  vidx[i] = r;
}

@compute @workgroup_size(256)
fn scatter(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= P.cap) { return; }
  let v = veh[i];
  if (!listed(v.state)) { return; }
  let base = eLaneBase(v.edge);
  let r = vidx[i];
  for (var l = vLo(v); l <= vHi(v); l = l + 1u) { items[atomicLoad(&counts[base + l]) + r[l - vLo(v)]] = i; }
}

@compute @workgroup_size(256)
fn sortLanes(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= P.cap) { return; }
  let v = veh[i];
  if (!listed(v.state)) { return; }
  let base = eLaneBase(v.edge);
  let si = v.s;
  var out = vec4u(0u);
  for (var l = vLo(v); l <= vHi(v); l = l + 1u) {
    let a = atomicLoad(&counts[base + l]);
    let b = atomicLoad(&counts[base + l + 1u]);
    var r = 0u;
    for (var k = a; k < b; k = k + 1u) {
      let j = items[k];
      let sj = veh[j].s;
      if (sj < si || (sj == si && j < i)) { r = r + 1u; }
    }
    order[a + r] = i;
    out[l - vLo(v)] = a + r;
  }
  vidx[i] = out;
}
