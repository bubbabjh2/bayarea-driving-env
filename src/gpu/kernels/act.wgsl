// External policy actions -> the reaction-time ring (replaces the built-in driver and the net of the
// original project). The caller writes actions[i * 3 + {0, 1, 2}] = [steer, gas, brake] per slot
// (GPUSim.setActions) before advancing:
//   steer: target wheel position as a fraction of maxSteer in [-1, 1] (layout v9 position mode); with
//          P.steerMode bit 1 (opts.actionInput 'rate') a steer RATE in [-1, 1] of maxSteerRate that is integrated
//          into the target (target += rate x maxSteerRate x dt / maxSteer, as src/sim/sim.js steerCommand)
//   gas, brake: pedals in [0, 1]
// Every DRIVING car with F_POLICY that is due a decision this step (P.decideEvery) gets its command written
// at the current step; integrate.wgsl applies it delaySteps later (reaction time), exactly like a CPU driver.
// No reflexes, no speed caps: the policy is fully in charge.
// #include common.wgsl

@group(1) @binding(0) var<storage, read_write> veh: array<Veh>;
@group(1) @binding(1) var<storage, read> par: array<VPar>;
@group(1) @binding(2) var<storage, read_write> ring: array<f32>;
@group(1) @binding(3) var<storage, read> actions: array<f32>;

var<workgroup> wPolicy: atomic<u32>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) g: vec3u, @builtin(local_invocation_index) li: u32) {
  if (li == 0u) { atomicStore(&wPolicy, 0u); }
  workgroupBarrier();
  let i = g.x;
  if (i < P.cap && hasFeat(FEAT_POLICY)) {
    let v = veh[i];
    let due = v.state == DRIVING && (v.flags & F_POLICY) != 0u && (P.decideEvery <= 1u || (simStep() + i) % P.decideEvery == 0u);
    if (due) {
      atomicAdd(&wPolicy, 1u);
      let step = simStep();
      let o = (i * CTRL_RING + (step % CTRL_RING)) * CTRL_STRIDE;
      var sc = clamp(actions[i * 3u], -1.0, 1.0);
      if ((P.steerMode & 3u) == 3u) {
        let pi = par[i];
        let po = (i * CTRL_RING + u32(((v.cmdStep % i32(CTRL_RING)) + i32(CTRL_RING)) % i32(CTRL_RING))) * CTRL_STRIDE;
        sc = clamp(ring[po] + sc * pi.maxSteerRate * P.dt / pi.maxSteer, -1.0, 1.0);
      }
      ring[o] = sc;
      ring[o + 1u] = clamp(actions[i * 3u + 1u], 0.0, 1.0);
      ring[o + 2u] = clamp(actions[i * 3u + 2u], 0.0, 1.0);
      veh[i].cmdStep = i32(step);
      veh[i].flags = v.flags & ~F_COLLIDED;
    }
  }
  workgroupBarrier();
  if (li == 0u) { statAdd(S_LIVE_POLICY, atomicLoad(&wPolicy)); }
}
