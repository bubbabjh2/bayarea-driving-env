// Freeway density sample : number of DRIVING cars per edge
// on motorway / trunk edges (worldPack flag 16). clear zeroes edgeCount, count adds one per car.
// The CPU turns the counts into veh/km/lane per edge (veh / km / lane).
// #include common.wgsl

@group(1) @binding(0) var<storage, read> veh: array<Veh>;
@group(1) @binding(1) var<storage, read_write> edgeCount: array<atomic<u32>>;
@group(1) @binding(2) var<storage, read_write> edgeSpeed: array<atomic<u32>>;   // cm/s sums (sim.js _trafficSample)

@compute @workgroup_size(256)
fn clearCounts(@builtin(global_invocation_id) g: vec3u) {
  if (g.x < P.E) { atomicStore(&edgeCount[g.x], 0u); }
}

@compute @workgroup_size(256)
fn countCars(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= P.cap) { return; }
  let v = veh[i];
  if (v.state != DRIVING || (eFlags(v.edge) & 16u) == 0u) { return; }
  atomicAdd(&edgeCount[v.edge], 1u);
}

// sim.js _trafficSample: speed sum (cm/s) and count of the DRIVING cars per edge (any road class)
@compute @workgroup_size(256)
fn clearTraffic(@builtin(global_invocation_id) g: vec3u) { if (g.x < P.E) { atomicStore(&edgeCount[g.x], 0u); atomicStore(&edgeSpeed[g.x], 0u); } }
@compute @workgroup_size(256)
fn countTraffic(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= P.cap) { return; }
  let v = veh[i]; if (v.state != DRIVING) { return; }
  atomicAdd(&edgeCount[v.edge], 1u); atomicAdd(&edgeSpeed[v.edge], u32(max(v.speed, 0.0) * 100.0 + 0.5));
}
