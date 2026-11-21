// bayenv-gpu kernel wrapper (not part of src/gpu/kernels): runs a per-car entry point in lane-list order.
// Changed vs the original: the host renames the kernel's entry `fn ENTRY(@builtin(global_invocation_id) g: vec3u)`
// to a plain function `ENTRY_at(g)` (body unchanged) and adds the entry below. Thread t < total (lane-list
// entries this step, starts[K]) handles car order[t] when t is that car's first entry (vidx[i].x == t; cars
// spanning two lanes are listed twice); threads total .. total + cap handle the cars not in any list (slot
// order). Every slot is handled exactly once, so a kernel whose cars do not depend on each other's writes in
// the same dispatch gives the same result, with neighbouring threads on the same lane (cache locality).
// Entries with a local_invocation_index (workgroup reductions, barriers) get ENTRY_at(i or cap when idle, li)
// called by every thread. Dispatched indirectly from kernels/list_args.wgsl. Kernels without lists.wgsl get it
// appended by the host (its bindings are in group 2, unused by those kernels).
fn sortedCar(t: u32) -> u32 {
  let total = starts[P.K];
  if (t < total) { let i = order[t]; if (vidx[i].x == t) { return i; } return NONE; }
  let i = t - total; if (i >= P.cap) { return NONE; }
  let p = vidx[i].x; if (p < total && order[p] == i) { return NONE; }
  return i;
}
//#if G
@compute @workgroup_size(WG)
fn ENTRY_sorted(@builtin(global_invocation_id) g: vec3u) {
  let i = sortedCar(g.x);
  if (i != NONE) { ENTRY_at(vec3u(i, 0u, 0u)); }
}
//#endif
//#if LI
@compute @workgroup_size(WG)
fn ENTRY_sorted(@builtin(global_invocation_id) g: vec3u, @builtin(local_invocation_index) li: u32) {
  let i = sortedCar(g.x);
  ENTRY_at(vec3u(select(i, P.cap, i == NONE), 0u, 0u), li);
}
//#endif
