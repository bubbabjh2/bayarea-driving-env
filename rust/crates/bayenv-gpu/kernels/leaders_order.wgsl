// bayenv-gpu patch (GpuOptions.deterministic_leaders, default on; off = the JS kernel): text replacements in the
// resolved source of src/gpu/kernels/leaders.wgsl (kernels.rs `patch_text`). See rust/docs/determinism.md.
// In the JS kernel thread i reads other cars' control records (ctl words 1, 2, 5, 6, 8: signedL / mustStopL in the
// zipper merge, convergeBodies; meant as "the previous step's controlObs, as the CPU order") while their threads
// rewrite them in the same pass (the new-occupant reset at the top of `main`, controlObs at the end), so i sees this
// step's or the previous step's values depending on thread timing. The patch reads a snapshot taken before the pass
// (ctlSnap below, or junction_order.wgsl jsortB with junctions; ctl words 25, 26, 27, 15, unused by the JS kernels).
// It also writes the snapshot of each car's final leader / conflict partner (ctl 12 / 13) that junction_order.wgsl
// jperceive reads.
//#orig fns
fn ctlOwnL(j: u32) -> bool { return ctl[j * CTL_W + 8u] == bitcast<u32>(veh[j].spawnTime); }
fn signedL(j: u32) -> bool { if (P.hasCtl == 0u || !ctlOwnL(j)) { return false; } let t = bitcast<i32>(ctl[j * CTL_W + 1u]); return (t == 1 || t == 2) && bitcast<f32>(ctl[j * CTL_W + 5u]) < 60.0; }
fn mustStopL(j: u32) -> bool {
  if (P.hasCtl == 0u || !ctlOwnL(j)) { return false; }
  let t = bitcast<i32>(ctl[j * CTL_W + 1u]);
  if (t < 0 || bitcast<f32>(ctl[j * CTL_W + 5u]) > 60.0) { return false; }
  return ctl[j * CTL_W + 2u] == 2u || (t == 1 && ((ctl[j * CTL_W + 6u] >> 8u) & 1u) == 0u);
}
//#end
//#patch fns
fn ctlOwnL(j: u32) -> bool { return ctl[j * CTL_W + 8u] == bitcast<u32>(veh[j].spawnTime); }
// deterministic_leaders: other cars' control records from the snapshot of the start of this pass (ctl 25 type, -1 when
// the record is not this occupant's / 26 light / 27 dist / 15 move | stopped << 8; driving cars, the only ones read),
// written by ctlSnap or junction jsortB
fn snapType(j: u32) -> i32 { return bitcast<i32>(ctl[j * CTL_W + 25u]); }
fn signedL(j: u32) -> bool { if (P.hasCtl == 0u) { return false; } let t = snapType(j); return (t == 1 || t == 2) && bitcast<f32>(ctl[j * CTL_W + 27u]) < 60.0; }
fn mustStopL(j: u32) -> bool {
  if (P.hasCtl == 0u) { return false; }
  let t = snapType(j);
  if (t < 0 || bitcast<f32>(ctl[j * CTL_W + 27u]) > 60.0) { return false; }
  return ctl[j * CTL_W + 26u] == 2u || (t == 1 && ((ctl[j * CTL_W + 15u] >> 8u) & 1u) == 0u);
}
//#end
//#orig conv
        let own = ctlOwnL(j); let ct = select(-1, bitcast<i32>(ctl[j * CTL_W + 1u]), own); let cd = bitcast<f32>(ctl[j * CTL_W + 5u]); let cs = ctl[j * CTL_W + 2u];
//#end
//#patch conv
        let ct = snapType(j); let cd = bitcast<f32>(ctl[j * CTL_W + 27u]); let cs = ctl[j * CTL_W + 26u];
//#end
//#orig end
  veh[i].leader = leader;
  veh[i].gap = gap;
//#end
//#patch end
  veh[i].leader = leader;
  veh[i].gap = gap;
  ctl[i * CTL_W + 12u] = bitcast<u32>(leader); ctl[i * CTL_W + 13u] = ctl[i * CTL_W + 28u];   // junction_order.wgsl snapshot
//#end
//#orig tail
// ---------------------------------------------------------------- traffic controls (sim.js _controlObs)
//#end
//#patch tail
fn ctlSnapOf(j: u32) {
  let o = j * CTL_W;
  ctl[o + 25u] = select(bitcast<u32>(-1), ctl[o + 1u], ctl[o + 8u] == bitcast<u32>(veh[j].spawnTime));
  ctl[o + 26u] = ctl[o + 2u]; ctl[o + 27u] = ctl[o + 5u]; ctl[o + 15u] = ctl[o + 6u];
}
@compute @workgroup_size(256)
fn ctlSnap(@builtin(global_invocation_id) g: vec3u) { if (g.x < P.cap && veh[g.x].state == DRIVING) { ctlSnapOf(g.x); } }
// ---------------------------------------------------------------- traffic controls (sim.js _controlObs)
//#end
