// bayenv-gpu model patch (GpuOptions.contact_tangent, default off): text replacement in the resolved source of
// src/gpu/kernels/collidefns.wgsl resolveContact (included by contacts.wgsl and kernels/resolve_fused.wgsl; kernels.rs
// `patch_text`, each //#orig snippet must occur exactly once). See rust/docs/crash-divergence.md.
// Changed vs the JS kernel: the tangent of the Coulomb friction impulse. collide.js computes t = rv - vn n in f64; for
// a collinear contact (aligned rear-end, both cars at lat = lane centre, no yaw) |t| ~ 1e-15 < 1e-6 and there is no
// friction impulse. In f32 the subtraction cancels to rounding noise of |rv| (ulp(38 m/s) ~ 4e-6): |t| passes the
// 1e-6 test, the normalised "tangent" points in a noise direction with a large component ALONG n, rv.t ~ vn, and jt
// saturates at mu jn. Both cars get a spurious sideways impulse and the same-sign yaw (~1 rad/s at 8 m/s closing), and
// slide / spin on locked wheels to the kerb instead of stopping in their lane. The patch removes the normal component
// of t again (exact orthogonality) and treats a slip below the f32 noise of rv (1e-5 |rv|) as none.
//#orig tangent
  var tx = rvx - vn * c.nx; var tz = rvz - vn * c.nz;
  let tl = sqrt(tx * tx + tz * tz);
  var jt = 0.0;
  if (tl > 1e-6) {
//#end
//#patch tangent
  var tx = rvx - vn * c.nx; var tz = rvz - vn * c.nz;
  let tn = tx * c.nx + tz * c.nz; tx = tx - tn * c.nx; tz = tz - tn * c.nz;   // exactly orthogonal to n (f32 cancellation)
  let tl = sqrt(tx * tx + tz * tz);
  var jt = 0.0;
  if (tl > max(1e-6, 1e-5 * sqrt(rvx * rvx + rvz * rvz))) {   // slip below the f32 noise of rv = none (collide.js in f64: |t| ~ 1e-15)
//#end
