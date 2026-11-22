// bayenv-gpu kernel (GpuOptions.heading_preview, default off; not part of the JS kernels): the lane-heading preview of
// bayenv-sim obs::heading_preview, per slot after the step. hpreview[4 i + k] = the change of the road frame's world
// heading (rad, wrapped to (-pi, pi], positive = the frame turns towards +heading) between the car's position and
// HP_D[k] = 5, 10, 20, 40 m further along its planned path: the current edge or junction connector, then the route
// (the connector its lane takes into the next route edge, any lane as fallback, or a plain hop). 0 for slots that are
// not DRIVING. Frame heading: connector tangent (connPointJ) inside a box, else the edge direction (roadDirAt = sim.js
// edgePointDir). Mirrors the CPU walk step for step (same pieces, same route-position bookkeeping, at most 16 hops).
// Edge directions come from hpdir (host, one per lane-geometry vertex: the angle of the chord to the next vertex, computed
// in f64 from the same vertices as the CPU): the blob's f32 vertex coordinates (ulp ~4-8 mm at 30-60 km from the origin)
// would put up to ~7e-3 rad of noise on a short segment's direction.
// #include common.wgsl

@group(1) @binding(0) var<storage, read> veh: array<Veh>;
@group(1) @binding(1) var<storage, read> par: array<VPar>;
@group(1) @binding(2) var<storage, read_write> hpreview: array<f32>;
@group(1) @binding(3) var<storage, read> hpdir: array<f32>;

// direction of the lane-geometry segment at s (offsetPointAt's segment search), else roadDirAt
fn hpEdgeDir(e: u32, s: f32) -> f32 {
  if (P.hasLG == 0u) { let d = roadDirAt(e, s); return atan2(d.y, d.x); }
  let v0 = geoU(e, 0u); let v1 = geoU(e + 1u, 0u) - 1u;
  var i = v0;
  if (s <= 0.0) { i = v0; }
  else if (s >= lgv(v1, 3u)) { i = v1 - 1u; }
  else {
    var lo = v0; var hi = v1;
    loop { if (hi - lo <= 1u) { break; } let m = (lo + hi) >> 1u; if (lgv(m, 3u) <= s) { lo = m; } else { hi = m; } }
    i = lo;
  }
  return hpdir[i];
}
fn hpDir(e: u32, m: i32, s: f32) -> f32 {
  if (m >= 0) { let p = connPointJ(u32(m), s - movSIn(u32(m))); return atan2(p.hz, p.hx); }
  return hpEdgeDir(e, max(0.0, min(s, eLength(e))));
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= P.cap) { return; }
  let v = veh[i];
  if (v.state != DRIVING) { for (var k = 0u; k < 4u; k = k + 1u) { hpreview[4u * i + k] = 0.0; } return; }
  let pi = par[i];
  var e = v.edge; var m = vConn(v); var s = v.s;
  let h0 = hpDir(e, m, s);
  var pos = v.routePos;
  let lane = i32(vLane(v));
  var walked = 0.0; var hops = 0u;
  for (var k = 0u; k < 4u; k = k + 1u) {
    let d = select(select(select(40.0, 20.0, k == 2u), 10.0, k == 1u), 5.0, k == 0u);
    loop {
      let need = d - walked;
      var end = 0.0; var hasNext = false; var ne = 0u; var nm = -1; var ns = 0.0;
      if (m >= 0) {
        let mu = u32(m);
        end = movSIn(mu) + movLen(mu); hasNext = true; ne = movOutEdge(mu); nm = -1; ns = movSOut(mu);
      } else {
        var mv = -1;
        let hasNx = pos + 1u < pi.routeLen;
        if (hasNx && P.hasJ != 0u) {
          let n = routeEdge(pi, pos + 1u);
          let nl = i32(eLanes(e));
          let ln = max(min(lane, nl - 1), 0);
          mv = jMovement(e, u32(ln), n);
          var l2 = 0; loop { if (mv >= 0 || l2 >= nl) { break; } mv = jMovement(e, u32(l2), n); l2 = l2 + 1; }
        }
        if (mv >= 0) { end = movSIn(u32(mv)); hasNext = true; ne = e; nm = mv; ns = movSIn(u32(mv)); }
        else { end = eLength(e); hasNext = hasNx; if (hasNx) { ne = routeEdge(pi, pos + 1u); nm = -1; ns = 0.0; } }
      }
      let left = max(0.0, end - s);
      if (need <= left || !hasNext || hops > 16u) { s = min(end, s + need); walked = d; break; }
      walked = walked + left;
      if ((m < 0 && nm < 0) || m >= 0) { if (ne != e) { pos = pos + 1u; } }
      e = ne; m = nm; s = ns; hops = hops + 1u;
    }
    let dh = hpDir(e, m, s) - h0;
    hpreview[4u * i + k] = atan2(sin(dh), cos(dh));
  }
}
