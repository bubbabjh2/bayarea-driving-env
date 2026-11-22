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
// MF_OBS_AFTER (GpuOptions::obs_after_step): sense runs after act + integrate of the step (as bayenv-sim observes after
// Sim::step); the pending commands are then ring t - d + 1 .. t (the newest one written by act this step), so the window
// shifts by one and nothing is appended.
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
  // steer_delay (position mode, sense before act): k = 0 is the command act writes this step; MF_OBS_AFTER: the
  // window is ring t - d + 1 .. t (k = d - 1 .. 0)
  let kHi = select(i32(d), i32(d) - 1, MF_OBS_AFTER);
  let kLo = select(select(1, 0, sdOn && posM && !netRate), 0, MF_OBS_AFTER);
  let due = v.state == DRIVING && (v.flags & F_POLICY) != 0u && hasFeat(FEAT_POLICY) && (P.decideEvery <= 1u || (step + i) % P.decideEvery == 0u);
  for (var kk = kHi; kk >= kLo; kk = kk - 1) {
    let k = u32(kk);
    var sr = clamp(ring[(i * CTRL_RING + (step + CTRL_RING - select(max(k, 1u), k, MF_OBS_AFTER)) % CTRL_RING) * CTRL_STRIDE], -1.0, 1.0);
    if (!MF_OBS_AFTER && k == 0u && due) { sr = clamp(actions[i * 3u], -1.0, 1.0); }
//#end
// v3 own commands (perceive.js senseCommands, 87..92): with steer_delay the steer applied now (87) comes from S - sds
// (S - 1 when sds = 0), as bayenv-sim obs.rs sense_commands; the pedals (88, 89) and "last" (90..92) are unchanged.
//#orig commands-now
  inputs[b + 87u] = ring[on]; inputs[b + 88u] = ring[on + 1u]; inputs[b + 89u] = ring[on + 2u];
//#end
//#patch commands-now
  var ons = on;
  if (MF_STEER_DELAY >= 0 && MF_STEER_DELAY < i32(pi.delaySteps)) { ons = (i * CTRL_RING + (S + CTRL_RING - select(u32(max(MF_STEER_DELAY, 0)), 1u, MF_STEER_DELAY == 0)) % CTRL_RING) * CTRL_STRIDE; }
  inputs[b + 87u] = ring[ons]; inputs[b + 88u] = ring[on + 1u]; inputs[b + 89u] = ring[on + 2u];
//#end
// MF_GUIDE_BOX (GpuOptions::guide_in_box): inside a junction box the car's s (in-edge frame) runs past the in-edge's
// length, and the JS guidance kept reporting the junction being crossed, with a negative distance (input 20 < 0) and its
// type / side (21..32). bayenv-sim's guidance (guidance.js at(routeDistance)) counts a maneuver as passed once the car
// is more than 1 m past it and reports the next one, at a distance >= 0. The patch skips the current entry's maneuver
// when the car is more than 1 m past the in-edge's end and clamps the distance at 0.
//#orig guide-passed
    if ((code & 1u) != 0u) { typ = (code >> 1u) & 15u; sd = (code >> 5u) & 3u; found = true; break; }
//#end
//#patch guide-passed
    if ((code & 1u) != 0u && !(MF_GUIDE_BOX && d < -1.0)) { typ = (code >> 1u) & 15u; sd = (code >> 5u) & 3u; found = true; break; }
//#end
//#orig guide-distance
  inputs[b + OFF_MAN] = min(d / 2000.0, 1.0);
//#end
//#patch guide-distance
  inputs[b + OFF_MAN] = min(select(d, max(d, 0.0), MF_GUIDE_BOX) / 2000.0, 1.0);
//#end
// MF_RING_FIX (GpuOptions::obs_ring_fix): senseV3's "now" command index S - d is u32; while the step counter is below
// the car's delay (the first <= 39 steps of a sim) it wraps (2^32 is not a multiple of the 40-entry ring) and inputs
// 87..89 read a written entry instead of the unwritten (0) one bayenv-sim's modular index reads. The patch indexes
// (S + ring - d) % ring.
//#orig ring-now
  var sn = S - 1u; if (dl >= 1u) { sn = S - dl; }
//#end
//#patch ring-now
  var sn = S - 1u; if (dl >= 1u) { sn = S - dl; }
  if (MF_RING_FIX) { sn = S + CTRL_RING - 1u; if (dl >= 1u) { sn = S + CTRL_RING - dl; } }
//#end
// MF_JUNC_OBST (GpuOptions::junction_obstacle): the junction-conflict obstacle (kernels/model_jyield.wgsl) into the
// own-ahead slot (input 8 / 9), as bayenv-sim's encoder puts it, when computed in this step.
//#orig junction-obstacle-bind
@group(1) @binding(3) var<storage, read> ring: array<f32>;
//#end
//#patch junction-obstacle-bind
@group(1) @binding(3) var<storage, read> ring: array<f32>;
@group(1) @binding(5) var<storage, read> jobst: array<u32>;   // junction_obstacle: gap / step per slot (kernels/model_jyield.wgsl)
//#end
//#orig junction-obstacle-put
  if (leftEx) { let e = laneEndAhead(i, v, pi, lane + 1); if (e < 1e8) { putEnd(b + OFF_NB + 4u, max(0.0, e - half), rel); } }
//#end
//#patch junction-obstacle-put
  if (MF_JUNC_OBST && jobst[2u * i + 1u] == step) { let jg = bitcast<f32>(jobst[2u * i]); if (jg < 1e8) { putEnd(b + OFF_NB, jg, rel); } }
  if (leftEx) { let e = laneEndAhead(i, v, pi, lane + 1); if (e < 1e8) { putEnd(b + OFF_NB + 4u, max(0.0, e - half), rel); } }
//#end

// MF_LIST_BEHIND (GpuOptions::sense_list_behind): with obs_after_step the row is sensed after integrate, but the
// footprint lane lists (vidx: the car's position in each lane laneLo..laneHi of the list build) are from before it.
// behindIn indexes vidx with the car's new laneLo, so after a footprint change (a lane change) it read another lane's
// slot: own-behind gap 0 (input 10) for a step where bayenv-sim (its lane index snapshot) has none. The patch looks
// the car up in the list of the lane itself (the vidx entry that points at it there), as the CPU's snapshot does.
//#orig list-behind
  let B = behindIn(i, vLane(v));
//#end
//#patch list-behind
  var B = behindIn(i, vLane(v));
  if (MF_LIST_BEHIND) {
    B = -1; let k = lkey(v.edge, vLane(v)); let a = starts[k]; let z = starts[k + 1u];
    for (var q = 0u; q < 4u; q = q + 1u) { let p = vidx[i][q]; if (p >= a && p < z && order[p] == i) { if (p > a) { B = i32(order[p - 1u]); } break; } }
  }
//#end
// MF_GUIDE_ODO (GpuOptions::guide_odometer): bayenv-sim's encoder (sim.js / driver.js: guidance.at(obs.routeDistance))
// places the car on its route by the route odometer (routeDist: 0 at spawn, + ds), the JS GPU by route position + s.
// The odometer lags the position by the spawn s (about 4 m, more after spawn_clearance), so the CPU's maneuver
// distance (input 20) and guidance lane ends (inputs 12 / 16) were that much longer, and for the first metres of
// an edge it still saw the previous edge's maneuver. The patch evaluates guidance and lane ends at the odometer's
// route position, as Guidance.edge_pos_at / at do.
//#orig guide-odo-fns
fn laneEndAhead(i: u32, v: Veh, pi: VPar, lane: i32) -> f32 {
//#end
//#patch guide-odo-fns
struct OdoPos { k: u32, cum: f32 }
// the route index whose start (cumulative edge length) is at or before the odometer; `fwd`: also past the current
// edge (Guidance.edge_pos_at), else only back from routePos (the maneuver scan starts there)
fn odoPos(v: Veh, pi: VPar, fwd: bool) -> OdoPos {
  var cum = 0.0;
  for (var q = 0u; q < v.routePos; q = q + 1u) { cum = cum + eLength(routeEdge(pi, q)); }
  var kk = v.routePos;
  loop { if (kk == 0u || cum <= v.routeDist) { break; } kk = kk - 1u; cum = cum - eLength(routeEdge(pi, kk)); }
  if (fwd) { loop { if (kk + 1u >= pi.routeLen || cum + eLength(routeEdge(pi, kk)) > v.routeDist) { break; } cum = cum + eLength(routeEdge(pi, kk)); kk = kk + 1u; } }
  var r: OdoPos; r.k = kk; r.cum = cum; return r;
}
fn laneEndAhead(i: u32, v0: Veh, pi: VPar, lane: i32) -> f32 {
  var v = v0;
  if (MF_GUIDE_ODO) { let o = odoPos(v0, pi, true); v.edge = routeEdge(pi, o.k); v.s = v0.routeDist - o.cum; v.routePos = o.k; }
//#end
//#orig guide-odo-start
  var d = e.length - v.s;
  var k = v.routePos;
//#end
//#patch guide-odo-start
  var d = e.length - v.s;
  var k = v.routePos;
  if (MF_GUIDE_ODO) { let o = odoPos(v, pi, false); k = o.k; d = o.cum + eLength(routeEdge(pi, o.k)) - v.routeDist; }
//#end
