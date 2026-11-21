// Spatial hash for contacts (src/sim/spatial.js SpatialHash): 25 m cells wrapped into a 2^tableBits
// table, DRIVING + CRASHED cars at their current (end-of-last-integration) positions.
//   hclear, hcount (atomic count + rank per car), [scan.wgsl over hcounts], hscatter
// #include common.wgsl

@group(1) @binding(0) var<storage, read> veh: array<Veh>;
@group(1) @binding(1) var<storage, read> pose: array<vec4f>;
@group(1) @binding(2) var<storage, read_write> hcounts: array<atomic<u32>>;
@group(1) @binding(3) var<storage, read_write> hitems: array<u32>;
@group(1) @binding(4) var<storage, read_write> hidx: array<vec2u>;

fn cellOfXZ(x: f32, z: f32) -> u32 {
  let cx = i32(floor(x / P.cellSize)); let cz = i32(floor(z / P.cellSize));
  return (bitcast<u32>(cx) & P.wMask) | ((bitcast<u32>(cz) & P.hMask) << P.wBits);
}
@compute @workgroup_size(256)
fn hclear(@builtin(global_invocation_id) g: vec3u) { if (g.x < P.T + 1u) { atomicStore(&hcounts[g.x], 0u); } }
@compute @workgroup_size(256)
fn hcount(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= P.cap) { return; }
  let st = veh[i].state;
  if (st != DRIVING && st != CRASHED) { hidx[i] = vec2u(NONE, 0u); return; }
  let p = pose[i]; let c = cellOfXZ(p.x, p.z);
  hidx[i] = vec2u(c, atomicAdd(&hcounts[c], 1u));
}
@compute @workgroup_size(256)
fn hscatter(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= P.cap) { return; }
  let h = hidx[i]; if (h.x == NONE) { return; }
  hitems[atomicLoad(&hcounts[h.x]) + h.y] = i;
}
