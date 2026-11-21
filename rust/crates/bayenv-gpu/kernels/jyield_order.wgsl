// bayenv-gpu patch (GpuOptions.deterministic_leaders, default on; off = the JS kernel): text replacements in the
// resolved source of src/gpu/kernels/jyield.wgsl (kernels.rs `patch_text`). See rust/docs/determinism.md.
// jyield writes its own leader / gap (passing a wreck) while prioJ reads another car's leader / gap ("held at its
// mouth by an obstacle"), so the answer depends on whether that car's thread already ran.
// The patch reads the other cars' values from a snapshot taken right before this pass (courtesy_order.wgsl commit,
// or ysnap after the deadlock passes; ctl words
// 12 / 13, unused by the JS kernels; courtesy_order.wgsl uses them only inside the courtesy stage), so every car sees
// the values of the start of the pass.
//#orig snap
@compute @workgroup_size(128)
fn jyield(@builtin(global_invocation_id) g: vec3u) {
//#end
//#patch snap
@compute @workgroup_size(256)
fn ysnap(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= P.cap) { return; }
  ctl[i * CW + 12u] = bitcast<u32>(veh[i].leader); ctl[i * CW + 13u] = bitcast<u32>(veh[i].gap);
}
@compute @workgroup_size(128)
fn jyield(@builtin(global_invocation_id) g: vec3u) {
//#end
//#orig held
vj.leader != -1 && vj.gap < d2 + 3.0
//#end
//#patch held
bitcast<i32>(ctl[j * CW + 12u]) != -1 && bitcast<f32>(ctl[j * CW + 13u]) < d2 + 3.0
//#end
