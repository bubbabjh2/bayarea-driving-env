// bayenv-gpu patch (GpuOptions.deterministic_leaders, default on; off = the JS kernel): text replacements in the
// resolved source of src/gpu/kernels/courtesy.wgsl (kernels.rs `patch_text`). See rust/docs/determinism.md.
// In the JS kernel `apply` recomputes courtesyOf(k), which reads veh[b].gap / leader of cars whose gap / leader other
// `apply` threads are writing in the same pass, and two proposers with the same (minimum) gap both write b's leader:
// the result depends on thread timing. The patch keeps each car's proposal from `propose` (ctl words 12 = b + 1,
// 13 = gap bits; unused by the JS kernels), lets the winners of b's minimum gap bid with atomicMin(k) (ctl word 14 of b,
// cleared by b's own propose thread) and writes b's leader / gap in a third pass, `commit`, from b's thread; commit
// also takes the leader / gap snapshot of jyield_order.wgsl.
//#orig propose
  let k = g.x; if (k >= P.cap) { return; }
  let t = courtesyOf(k); if (t.x < 0) { return; }
  atomicMin(&ctl[u32(t.x) * CTL_W + 29u], bitcast<u32>(t.y));
}
//#end
//#patch propose
  let k = g.x; if (k >= P.cap) { return; }
  atomicStore(&ctl[k * CTL_W + 14u], NONE);
  let t = courtesyOf(k);
  atomicStore(&ctl[k * CTL_W + 12u], select(0u, u32(t.x) + 1u, t.x >= 0)); atomicStore(&ctl[k * CTL_W + 13u], bitcast<u32>(t.y));
  if (t.x < 0) { return; }
  atomicMin(&ctl[u32(t.x) * CTL_W + 29u], bitcast<u32>(t.y));
}
//#end
//#orig apply
  let k = g.x; if (k >= P.cap) { return; }
  let t = courtesyOf(k); if (t.x < 0) { return; }
  let b = u32(t.x);
  if (atomicLoad(&ctl[b * CTL_W + 29u]) == bitcast<u32>(t.y)) { veh[b].gap = bitcast<f32>(t.y); veh[b].leader = i32(k); }
}
//#end
//#patch apply
  let k = g.x; if (k >= P.cap) { return; }
  let pb = atomicLoad(&ctl[k * CTL_W + 12u]); if (pb == 0u) { return; }
  let b = pb - 1u;
  if (atomicLoad(&ctl[b * CTL_W + 29u]) == atomicLoad(&ctl[k * CTL_W + 13u])) { atomicMin(&ctl[b * CTL_W + 14u], k); }
}
@compute @workgroup_size(128)
fn commit(@builtin(global_invocation_id) g: vec3u) {
  let b = g.x; if (b >= P.cap) { return; }
  let k = atomicLoad(&ctl[b * CTL_W + 14u]);
  if (k != NONE) { veh[b].gap = bitcast<f32>(atomicLoad(&ctl[b * CTL_W + 29u])); veh[b].leader = i32(k); }
  // jyield_order.wgsl: the leader / gap snapshot jyield reads (here when no deadlock passes run in between)
  atomicStore(&ctl[b * CTL_W + 12u], bitcast<u32>(veh[b].leader)); atomicStore(&ctl[b * CTL_W + 13u], bitcast<u32>(veh[b].gap));
}
//#end
