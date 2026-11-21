// bayenv-gpu function patches (GpuOptions.fast_search): replacements for functions of src/gpu/kernels/common.wgsl,
// substituted by the host after the includes are resolved (kernels.rs `patch`; the original text is checked so a
// changed upstream function is not silently replaced).
// Changed vs the original: the linear scans for the polyline segment containing s (first k in [0, n - 2) with
// w(k + 1) >= s, else n - 2) are binary searches. The cumulative distances w are non-decreasing (checked for every
// edge of the world blob / connector before this patch is used), so both find the same k and the results are
// identical; log2(n) dependent loads instead of up to n. Same for the junction connector points (arcPointJ,
// connPointJ: jSegK; the connector distances are checked too).
//#orig yAlongEdge
fn yAlongEdge(b: u32, n: u32, s: f32) -> f32 {
  var k = 0u;
  loop { if (k >= n - 2u || pt(b + k + 1u).w >= s) { break; } k = k + 1u; }
  let p0 = pt(b + k); let p1 = pt(b + k + 1u);
  let d = p1.w - p0.w;
  var t = 0.0; if (d > 0.0) { t = clamp((s - p0.w) / d, 0.0, 1.0); }
  return p0.z + (p1.z - p0.z) * t;
}
//#end
//#patch yAlongEdge
fn yAlongEdge(b: u32, n: u32, s: f32) -> f32 {
  var lo = 0u; var hi = n - 2u;
  loop { if (lo >= hi) { break; } let m = (lo + hi) >> 1u; if (pt(b + m + 1u).w >= s) { hi = m; } else { lo = m + 1u; } }
  let k = lo;
  let p0 = pt(b + k); let p1 = pt(b + k + 1u);
  let d = p1.w - p0.w;
  var t = 0.0; if (d > 0.0) { t = clamp((s - p0.w) / d, 0.0, 1.0); }
  return p0.z + (p1.z - p0.z) * t;
}
//#end
//#orig arcPointJ
fn arcPointJ(m: u32, t: f32) -> GeoPt {
  let first = movW(m, 2u); let n = movW(m, 3u);
  var k = 0u;
  loop { if (k >= n - 2u || bitcast<f32>(world[P.jPtsOff + 4u * (first + k + 1u) + 3u]) >= t) { break; } k = k + 1u; }
  let a = P.jPtsOff + 4u * (first + k); let b = a + 4u;
  let ax = bitcast<f32>(world[a]); let az = bitcast<f32>(world[a + 1u]); let ay = bitcast<f32>(world[a + 2u]); let ac = bitcast<f32>(world[a + 3u]);
  let bx = bitcast<f32>(world[b]); let bz = bitcast<f32>(world[b + 1u]); let by = bitcast<f32>(world[b + 2u]); let bc = bitcast<f32>(world[b + 3u]);
  var d = bc - ac; if (d == 0.0) { d = 1.0; }
  let u = clamp((t - ac) / d, 0.0, 1.0);
  let hx = bx - ax; let hz = bz - az; var L = sqrt(hx * hx + hz * hz); if (L == 0.0) { L = 1.0; }
  return GeoPt(ax + (bx - ax) * u, ay + (by - ay) * u, az + (bz - az) * u, hx / L, hz / L);
}
//#end
//#patch arcPointJ
// first k in [0, n - 2) with t(k + 1) >= t, else n - 2 (the linear scan of arcPointJ / connPointJ)
fn jSegK(first: u32, n: u32, t: f32) -> u32 {
  var lo = 0u; var hi = n - 2u;
  loop { if (lo >= hi) { break; } let m = (lo + hi) >> 1u; if (bitcast<f32>(world[P.jPtsOff + 4u * (first + m + 1u) + 3u]) >= t) { hi = m; } else { lo = m + 1u; } }
  return lo;
}
fn arcPointJ(m: u32, t: f32) -> GeoPt {
  let first = movW(m, 2u); let n = movW(m, 3u);
  let k = jSegK(first, n, t);
  let a = P.jPtsOff + 4u * (first + k); let b = a + 4u;
  let ax = bitcast<f32>(world[a]); let az = bitcast<f32>(world[a + 1u]); let ay = bitcast<f32>(world[a + 2u]); let ac = bitcast<f32>(world[a + 3u]);
  let bx = bitcast<f32>(world[b]); let bz = bitcast<f32>(world[b + 1u]); let by = bitcast<f32>(world[b + 2u]); let bc = bitcast<f32>(world[b + 3u]);
  var d = bc - ac; if (d == 0.0) { d = 1.0; }
  let u = clamp((t - ac) / d, 0.0, 1.0);
  let hx = bx - ax; let hz = bz - az; var L = sqrt(hx * hx + hz * hz); if (L == 0.0) { L = 1.0; }
  return GeoPt(ax + (bx - ax) * u, ay + (by - ay) * u, az + (bz - az) * u, hx / L, hz / L);
}
//#end
//#orig connPointJ
fn connPointJ(m: u32, t: f32) -> GeoPt {
  var o = arcPointJ(m, t);
  let first = movW(m, 2u); let n = movW(m, 3u);
  if (n < 3u) { return o; }
  var k = 0u;
  loop { if (k >= n - 2u || bitcast<f32>(world[P.jPtsOff + 4u * (first + k + 1u) + 3u]) >= t) { break; } k = k + 1u; }
  let A = P.jPtsOff + 4u * (first + k); let B = A + 4u;
  let pa = vec2f(bitcast<f32>(world[A]), bitcast<f32>(world[A + 1u])); let pb = vec2f(bitcast<f32>(world[B]), bitcast<f32>(world[B + 1u]));
  var d = bitcast<f32>(world[B + 3u]) - bitcast<f32>(world[A + 3u]); if (d == 0.0) { d = 1.0; }
  let u = clamp((t - bitcast<f32>(world[A + 3u])) / d, 0.0, 1.0);
  var c = pb - pa; var cl = length(c); if (cl == 0.0) { cl = 1.0; }
  var a = c / cl; var b = a;
  if (k > 0u) { let pp = vec2f(bitcast<f32>(world[A - 4u]), bitcast<f32>(world[A - 3u])); var q = pa - pp; var ql = length(q); if (ql == 0.0) { ql = 1.0; } a = a + q / ql; }
  if (k < n - 2u) { let pn = vec2f(bitcast<f32>(world[B + 4u]), bitcast<f32>(world[B + 5u])); var q = pn - pb; var ql = length(q); if (ql == 0.0) { ql = 1.0; } b = b + q / ql; }
  var al = length(a); if (al == 0.0) { al = 1.0; } var bl = length(b); if (bl == 0.0) { bl = 1.0; }
  var h = (1.0 - u) * a / al + u * b / bl; var hl = length(h); if (hl == 0.0) { hl = 1.0; }
  o.hx = h.x / hl; o.hz = h.y / hl;
  return o;
}
//#end
//#patch connPointJ
fn connPointJ(m: u32, t: f32) -> GeoPt {
  var o = arcPointJ(m, t);
  let first = movW(m, 2u); let n = movW(m, 3u);
  if (n < 3u) { return o; }
  let k = jSegK(first, n, t);
  let A = P.jPtsOff + 4u * (first + k); let B = A + 4u;
  let pa = vec2f(bitcast<f32>(world[A]), bitcast<f32>(world[A + 1u])); let pb = vec2f(bitcast<f32>(world[B]), bitcast<f32>(world[B + 1u]));
  var d = bitcast<f32>(world[B + 3u]) - bitcast<f32>(world[A + 3u]); if (d == 0.0) { d = 1.0; }
  let u = clamp((t - bitcast<f32>(world[A + 3u])) / d, 0.0, 1.0);
  var c = pb - pa; var cl = length(c); if (cl == 0.0) { cl = 1.0; }
  var a = c / cl; var b = a;
  if (k > 0u) { let pp = vec2f(bitcast<f32>(world[A - 4u]), bitcast<f32>(world[A - 3u])); var q = pa - pp; var ql = length(q); if (ql == 0.0) { ql = 1.0; } a = a + q / ql; }
  if (k < n - 2u) { let pn = vec2f(bitcast<f32>(world[B + 4u]), bitcast<f32>(world[B + 5u])); var q = pn - pb; var ql = length(q); if (ql == 0.0) { ql = 1.0; } b = b + q / ql; }
  var al = length(a); if (al == 0.0) { al = 1.0; } var bl = length(b); if (bl == 0.0) { bl = 1.0; }
  var h = (1.0 - u) * a / al + u * b / bl; var hl = length(h); if (hl == 0.0) { hl = 1.0; }
  o.hx = h.x / hl; o.hz = h.y / hl;
  return o;
}
//#end
