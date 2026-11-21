// sim.js _courtesy (zipper courtesy, end of _findLeaders): a slow car k that has to leave its lane
// (lane end within 80 m, or a target lane while blocked by a wreck / lane end / a close leader) is taken as
// leader by the first car b behind it in the target lane, when 0.5 <= gap <= 40 m and smaller than b's gap.
// propose: atomicMin of the gap (f32 bits, positive) into b's ctl word 29; apply: the k whose gap won writes it.
// #include common.wgsl
// #include lists.wgsl

@group(1) @binding(0) var<storage, read_write> veh: array<Veh>;
@group(1) @binding(1) var<storage, read> par: array<VPar>;
@group(1) @binding(2) var<storage, read_write> ctl: array<atomic<u32>>;

fn courtesyOf(k: u32) -> vec2i {   // (b, gap bits) or (-1, 0)
  let v = veh[k];
  if (v.state != DRIVING || vConn(v) >= 0 || v.speed > 6.0) { return vec2i(-1, 0); }
  var side = 0;
  let lane = i32(vLane(v)); let tl = vTarget(v);
  if (v.laneEndSide != 0 && v.laneEndDist < 80.0) { side = v.laneEndSide; }
  else if (tl >= 0 && tl != lane) {
    let L = v.leader;
    if ((L >= 0 && veh[u32(L)].state == CRASHED) || L == -2 || v.gap < 15.0) { side = select(-1, 1, tl > lane); }
  }
  if (side == 0) { return vec2i(-1, 0); }
  let t = lane + side;
  if (t < 0 || t >= i32(eLanes(v.edge))) { return vec2i(-1, 0); }
  let nb = neighborsAt(v.edge, u32(t), v.s, k);
  let b = nb.y; if (b < 0) { return vec2i(-1, 0); }
  let vb = veh[u32(b)];
  if (vb.state != DRIVING || vConn(vb) >= 0) { return vec2i(-1, 0); }
  let gg = v.s - vb.s - 0.5 * par[k].length - 0.5 * par[u32(b)].length;
  if (gg < 0.5 || gg > 40.0 || gg >= vb.gap) { return vec2i(-1, 0); }
  return vec2i(b, bitcast<i32>(gg));
}
@compute @workgroup_size(128)
fn propose(@builtin(global_invocation_id) g: vec3u) {
  let k = g.x; if (k >= P.cap) { return; }
  let t = courtesyOf(k); if (t.x < 0) { return; }
  atomicMin(&ctl[u32(t.x) * CTL_W + 29u], bitcast<u32>(t.y));
}
@compute @workgroup_size(128)
fn apply(@builtin(global_invocation_id) g: vec3u) {
  let k = g.x; if (k >= P.cap) { return; }
  let t = courtesyOf(k); if (t.x < 0) { return; }
  let b = u32(t.x);
  if (atomicLoad(&ctl[b * CTL_W + 29u]) == bitcast<u32>(t.y)) { veh[b].gap = bitcast<f32>(t.y); veh[b].leader = i32(k); }
}
