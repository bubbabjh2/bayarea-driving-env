// bayenv-gpu helper (not part of src/gpu/kernels): the training signal "collided at sense time" (JS external protocol:
// sig.collision = FLAG_COLLIDED when the policy senses; act.wgsl clears the flag right after). Dispatched between
// sense and act: per slot, cflag[2 i] = steps of the current batch with the flag set, cflag[2 i + 1] = the flag at the
// last sense. veh is read raw (VEH_WORDS = 40: state word 7, flags word 10).
struct CP { cap: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<uniform> CC: CP;
@group(0) @binding(1) var<storage, read> vehw: array<u32>;
@group(0) @binding(2) var<storage, read_write> cflag: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= CC.cap) { return; }
  let st = vehw[i * 40u + 7u]; let f = vehw[i * 40u + 10u];
  let c = select(0u, 1u, st == 1u && (f & 2u) != 0u);
  cflag[2u * i] = cflag[2u * i] + c; cflag[2u * i + 1u] = c;
}
