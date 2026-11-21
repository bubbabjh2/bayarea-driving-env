// bayenv-gpu helper (not part of src/gpu/kernels): opt-in compact observations. Packs the first 2 * HP.n floats of
// `inputs` into binary16 pairs (pack2x16float, round to nearest even), clamped to the finite f16 range so values
// beyond +-65504 saturate instead of becoming indeterminate. Grid-stride loop over HP.n words.
struct HalfP { n: u32, stride: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<uniform> HP: HalfP;
@group(0) @binding(1) var<storage, read> inputs: array<f32>;
@group(0) @binding(2) var<storage, read_write> obsHalf: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3u) {
  for (var i = g.x; i < HP.n; i = i + HP.stride) {
    let v = clamp(vec2f(inputs[2u * i], inputs[2u * i + 1u]), vec2f(-65504.0), vec2f(65504.0));
    obsHalf[i] = pack2x16float(v);
  }
}
