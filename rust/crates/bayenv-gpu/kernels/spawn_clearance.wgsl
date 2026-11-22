// bayenv-gpu model patch (GpuOptions.spawn_clearance, default off; off = the JS kernel): text replacements in the resolved
// source of src/gpu/kernels/spawn.wgsl (kernels.rs `patch_text`; applied before kernels/spawn_order.wgsl, whose snippets it
// does not touch). Mirrors bayenv-sim SimOpts::spawn_clearance (sim.rs try_spawn / spawn_edge_index, DECISIONS.md 34).
// The host picks the start route index k0 when it writes the spawn-queue entry (host.rs write_entry: the first route edge
// that leaves spawn_clearance_m between its spawn point s_exit and the junction entry the route takes next) and stores it
// in the entry's spare word p0, with the route distance of the skipped edges (f32 bits, summed in f64 as the CPU does)
// in p1. The kernel spawns on route edge k0 (entry lane chosen against route edge k0 + 1) at its s_exit, with routePos k0.
// Difference to the CPU: the CPU runs its synchronous refuel check first and keeps the car on edge 0 when the refuel
// plan replaced the route; the GPU refuel plan is asynchronous (host process_refuels), so the car always starts on k0
// and the plan is made from route edge k0.
//#orig start-edge
  let e = routes[q.par.routeOff] & ROUTE_EDGE_MASK;
  var nx = -1; if (q.par.routeLen > 1u) { nx = i32(routes[q.par.routeOff + 1u] & ROUTE_EDGE_MASK); }
//#end
//#patch start-edge
  let k0 = min(q.p0, max(q.par.routeLen, 1u) - 1u);   // spawn_clearance start index (host)
  let e = routes[q.par.routeOff + k0] & ROUTE_EDGE_MASK;
  var nx = -1; if (q.par.routeLen > k0 + 1u) { nx = i32(routes[q.par.routeOff + k0 + 1u] & ROUTE_EDGE_MASK); }
//#end
//#orig start-pos
  v.routePos = 0u; v.seg = 0u;
//#end
//#patch start-pos
  v.routePos = k0; v.seg = 0u;
//#end
//#orig start-dist
v.spawnTime = now; v.routeDist = 0.0;
//#end
//#patch start-dist
v.spawnTime = now; v.routeDist = select(0.0, bitcast<f32>(q.p1), k0 > 0u);
//#end
