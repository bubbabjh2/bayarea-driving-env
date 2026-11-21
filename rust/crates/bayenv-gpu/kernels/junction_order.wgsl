// bayenv-gpu patch (GpuOptions.deterministic_leaders, default on; off = the JS kernel): text replacements in the
// resolved source of src/gpu/kernels/junction.wgsl (kernels.rs `patch_text`). See rust/docs/determinism.md.
// jperceive writes its own leader / gap / ctl word 28 (conflict partner) while other threads read them ("is j waiting for me?"),
// so the answer depends on whether j's thread already ran.
// The patch reads the other cars' values from the snapshot the leaders pass writes at its end (leaders_order.wgsl;
// ctl words 12 / 13, unused by the JS kernels; courtesy_order.wgsl uses them only inside the courtesy stage), so every car sees
// the values of the start of the pass. jsortA / jsortB (below) make the per-movement, wreck-passer and off-lane lists
// that jlink builds with atomicExchange slot-ordered; jsortB also takes the control-record snapshot of leaders_order.wgsl.
//#orig snap
@compute @workgroup_size(128)
fn jperceive(@builtin(global_invocation_id) g: vec3u) {
//#end
//#patch snap
// a driving car's leader / conflict partner as the leaders pass left them (snapshot ctl 12 / 13, leaders_order.wgsl);
// other cars' are not written by this pass
fn snapLeader(j: u32) -> i32 { if (veh[j].state == DRIVING) { return bitcast<i32>(atomicLoad(&ctl[j * CW + 12u])); } return veh[j].leader; }
fn snapPartner(j: u32) -> u32 { if (veh[j].state == DRIVING) { return atomicLoad(&ctl[j * CW + 13u]); } return atomicLoad(&ctl[j * CW + 28u]); }
fn snapWaits(j: u32, i: u32) -> bool { let L = snapLeader(j); return L == i32(i) || (L == -3 && snapPartner(j) == i); }
@compute @workgroup_size(128)
fn jperceive(@builtin(global_invocation_id) g: vec3u) {
//#end
//#orig w64
(vjn.leader == i32(i) || (vjn.leader == -3 && atomicLoad(&ctl[j * CW + 28u]) == i))
//#end
//#patch w64
snapWaits(j, i)
//#end
//#orig w150
vj.state == DRIVING && vj.leader == i32(i)) {
//#end
//#patch w150
vj.state == DRIVING && snapLeader(j) == i32(i)) {
//#end
//#orig w192
(vjw.leader == i32(i) || (vjw.leader == -3 && atomicLoad(&ctl[j * CW + 28u]) == i))
//#end
//#patch w192
snapWaits(j, i)
//#end
//#orig w255
(vj.leader == i32(i) || (vj.leader == -3 && atomicLoad(&ctl[j * CW + 28u]) == i))
//#end
//#patch w255
snapWaits(j, i)
//#end
//#orig jsort
fn cellOfC(cx: i32, cz: i32) -> u32 {
//#end
//#patch jsort
// deterministic_leaders: jlink builds its lists with atomicExchange (run-dependent order); jsortA / jsortB relink every
// list in slot order (head = smallest slot, next = the next larger slot), ctl words 12 / 13 / 14 as scratch
fn listMinAbove(head: u32, i: u32, w: u32) -> u32 {
  var best = NONE; var j = head;
  for (var gq = 0u; gq < 4096u; gq = gq + 1u) { if (j == NONE) { break; } if (j > i && j < best) { best = j; } j = atomicLoad(&ctl[j * CW + w]); }
  return best;
}
fn isPasser(i: u32, v: Veh) -> bool { return v.state == DRIVING && atomicLoad(&ctl[i * CW + 8u]) == bitcast<u32>(v.spawnTime) && bitcast<i32>(atomicLoad(&ctl[i * CW + 20u])) >= 0; }
@compute @workgroup_size(256)
fn jsortA(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= P.cap) { return; }
  let v = veh[i];
  if (v.state != DRIVING && v.state != CRASHED) { return; }
  if (vHi(v) < vLo(v)) { atomicStore(&ctl[i * CW + 12u], listMinAbove(atomicLoad(&ctl[offHeadIdx(v.edge)]), i, 30u)); }
  if (isPasser(i, v)) { atomicStore(&ctl[i * CW + 13u], listMinAbove(atomicLoad(&ctl[P.jHeadOff + P.jM + v.edge]), i, 21u)); }
  let m = vConn(v); if (m >= 0) { atomicStore(&ctl[i * CW + 14u], listMinAbove(atomicLoad(&ctl[P.jHeadOff + u32(m)]), i, 18u)); }
}
@compute @workgroup_size(256)
fn jsortB(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= P.cap) { return; }
  let v = veh[i];
  // leaders_order.wgsl ctlSnap (the control-record snapshot the leaders pass reads, of driving cars), folded in here
  let o = i * CW;
  if (v.state == DRIVING) {
  atomicStore(&ctl[o + 25u], select(bitcast<u32>(-1), atomicLoad(&ctl[o + 1u]), atomicLoad(&ctl[o + 8u]) == bitcast<u32>(v.spawnTime)));
  atomicStore(&ctl[o + 26u], atomicLoad(&ctl[o + 2u])); atomicStore(&ctl[o + 27u], atomicLoad(&ctl[o + 5u])); atomicStore(&ctl[o + 15u], atomicLoad(&ctl[o + 6u]));
  }
  if (v.state != DRIVING && v.state != CRASHED) { return; }
  if (vHi(v) < vLo(v)) { atomicStore(&ctl[i * CW + 30u], atomicLoad(&ctl[i * CW + 12u])); atomicMin(&ctl[offHeadIdx(v.edge)], i); }
  if (isPasser(i, v)) { atomicStore(&ctl[i * CW + 21u], atomicLoad(&ctl[i * CW + 13u])); atomicMin(&ctl[P.jHeadOff + P.jM + v.edge], i); }
  let m = vConn(v); if (m >= 0) { atomicStore(&ctl[i * CW + 18u], atomicLoad(&ctl[i * CW + 14u])); atomicMin(&ctl[P.jHeadOff + u32(m)], i); }
}
fn cellOfC(cx: i32, cz: i32) -> u32 {
//#end
