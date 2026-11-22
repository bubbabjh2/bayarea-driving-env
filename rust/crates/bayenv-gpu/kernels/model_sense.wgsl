// bayenv-gpu model patches of the observation stage (src/gpu/kernels/sense.wgsl), applied by kernels.rs `model_patches` with
// GpuOptions::steer_delay >= 0 (the host prepends `const MF_STEER_DELAY: i32`). adjustInputs (adjust.js delay
// compensation) predicts lane offset / heading error / steer angle (obs 4 / 5 / 41) by replaying the steering commands
// issued but not applied yet. In the JS model every command waits the reaction delay d, so these are ring steps
// t - d .. t - 1. With steer_delay (kernels/model_integrate.wgsl) steering waits only sds = MF_STEER_DELAY steps when
// sds < d: the commands t - d .. t - sds - 1 are already applied, and replaying them counted them twice (the predicted
// heading error / lane offset overshot by d - sds steps of steering). Only t - sds .. t - 1 are pending now.
// The external-policy pipeline adds one more: sense runs at the start of step t (state after integrate t - 1), but the
// command the host uploaded for this step (actions[i], written into the ring by act after sense) is pending too, and
// the action computed from this observation applies one step later. bayenv-sim's V2Encoder observes after the step
// with that command already in its pending window; to predict the same horizon, the replay appends the command act
// writes this step (actions[i] when the car is due a decision, else the held ring entry t - 1). Position mode only
// (rate mode keeps the JS window). With steer_delay -1 nothing changes (JS-exact).
//#orig bindings
@group(1) @binding(3) var<storage, read> ring: array<f32>;
//#end
//#patch bindings
@group(1) @binding(3) var<storage, read> ring: array<f32>;
@group(1) @binding(4) var<storage, read> actions: array<f32>;   // steer_delay: the command act writes this step
//#end
//#orig compensation-window
  let d = min(pi.delaySteps, CTRL_RING - 1u);
//#end
//#patch compensation-window
  var d = min(pi.delaySteps, CTRL_RING - 1u);
  let sdOn = MF_STEER_DELAY >= 0 && MF_STEER_DELAY < i32(pi.delaySteps);
  if (sdOn) { d = u32(max(MF_STEER_DELAY, 0)); }   // steer_delay: pending = last sds steps
//#end
//#orig compensation-loop
  for (var k = d; k >= 1u; k = k - 1u) {
    let sr = clamp(ring[(i * CTRL_RING + (step + CTRL_RING - k) % CTRL_RING) * CTRL_STRIDE], -1.0, 1.0);
//#end
//#patch compensation-loop
  // steer_delay (position mode): k = 0 is the command act writes this step
  let kLo = select(1, 0, sdOn && posM && !netRate);
  let due = v.state == DRIVING && (v.flags & F_POLICY) != 0u && hasFeat(FEAT_POLICY) && (P.decideEvery <= 1u || (step + i) % P.decideEvery == 0u);
  for (var kk = i32(d); kk >= kLo; kk = kk - 1) {
    let k = u32(kk);
    var sr = clamp(ring[(i * CTRL_RING + (step + CTRL_RING - max(k, 1u)) % CTRL_RING) * CTRL_STRIDE], -1.0, 1.0);
    if (k == 0u && due) { sr = clamp(actions[i * 3u], -1.0, 1.0); }
//#end
// v3 own commands (perceive.js senseCommands, 87..92): with steer_delay the steer applied now (87) comes from S - sds
// (S - 1 when sds = 0), as bayenv-sim obs.rs sense_commands; the pedals (88, 89) and "last" (90..92) are unchanged.
//#orig commands-now
  inputs[b + 87u] = ring[on]; inputs[b + 88u] = ring[on + 1u]; inputs[b + 89u] = ring[on + 2u];
//#end
//#patch commands-now
  var ons = on;
  if (MF_STEER_DELAY >= 0 && MF_STEER_DELAY < i32(pi.delaySteps)) { ons = (i * CTRL_RING + select(S - u32(max(MF_STEER_DELAY, 0)), S - 1u, MF_STEER_DELAY == 0) % CTRL_RING) * CTRL_STRIDE; }
  inputs[b + 87u] = ring[ons]; inputs[b + 88u] = ring[on + 1u]; inputs[b + 89u] = ring[on + 2u];
//#end
