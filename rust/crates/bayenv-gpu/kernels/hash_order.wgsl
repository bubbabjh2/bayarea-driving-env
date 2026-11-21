// bayenv-gpu patch (GpuOptions.deterministic_leaders, default on; off = the JS kernel): text replacement in the
// resolved source of src/gpu/kernels/hash.wgsl (kernels.rs `patch_text`). See rust/docs/determinism.md.
// hcount ranks the cars of a cell by atomicAdd, so a cell's items come in a run-dependent order; perception walks the
// cells (junction.wgsl jperceive: first 32 candidates, first hit; pathBodies) and keeps the first of equal gaps. The
// patch adds hrank: after the first hscatter every car takes as rank the number of cars of its cell with a smaller
// slot, and the host runs hscatter again, so every cell lists its cars in slot order.
//#orig hrank
@compute @workgroup_size(256)
fn hscatter(@builtin(global_invocation_id) g: vec3u) {
//#end
//#patch hrank
@compute @workgroup_size(256)
fn hrank(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= P.cap) { return; }
  let h = hidx[i]; if (h.x == NONE) { return; }
  let a = atomicLoad(&hcounts[h.x]); let b = atomicLoad(&hcounts[h.x + 1u]);
  var r = 0u; for (var k = a; k < b; k = k + 1u) { if (hitems[k] < i) { r = r + 1u; } }
  hidx[i] = vec2u(h.x, r);
}
@compute @workgroup_size(256)
fn hscatter(@builtin(global_invocation_id) g: vec3u) {
//#end
