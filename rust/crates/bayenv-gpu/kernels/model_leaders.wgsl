// bayenv-gpu model patch of the leaders stage (src/gpu/kernels/leaders.wgsl), with GpuOptions::waits_prev_step (the
// host prepends `const MF_WAITS_PREV: bool`; applied before leaders_order.wgsl, whose `end` snippet it leaves intact).
// The junction perception (junction.wgsl jperceive: node / conflict / path bodies) skips a body that "waits for me"
// (stopped, its leader me, or -3 with me as its conflict partner). bayenv-sim (and sim.js) run that perception in
// slot order, so for a car j after me it reads j's leader / partner as the previous step left them (j's final
// perception, junction conflicts and path bodies included). deterministic_leaders' snapshot (ctl 12 / 13, written
// at the end of this pass) held j's leader of this pass, before its junction perception: a car stopped in a box
// for a conflicting car (leader set by jperceive) never "waited" for it on the GPU, and both held (mutual block).
// With the flag the snapshot is the value at the start of this pass (the previous step's final leader / partner),
// as the CPU reads it for every later slot (for an earlier slot the CPU reads this step's final value: the same in
// a standstill, where the test applies).
//#orig waits-prev-start
  let v = veh[i];
  if (v.state != DRIVING) { return; }
  let pi = par[i];
//#end
//#patch waits-prev-start
  let v = veh[i];
  if (v.state != DRIVING) { return; }
  let prevPartner = ctl[i * CTL_W + 28u];
  let pi = par[i];
//#end
//#orig waits-prev-end
  veh[i].leader = leader;
  veh[i].gap = gap;
//#end
//#patch waits-prev-end
  veh[i].leader = leader;
  veh[i].gap = gap;
  if (MF_WAITS_PREV) { ctl[i * CTL_W + 12u] = bitcast<u32>(v.leader); ctl[i * CTL_W + 13u] = prevPartner; }
//#end
