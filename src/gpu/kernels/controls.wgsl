// Traffic controls (src/sim/controls.js TrafficControls.stateOf + sim.js _controlObs), layout in
// worldPack.js packControls. Perceived, never enforced: leaders.wgsl writes each driving car's
// observation of the next control on its route into ctl[], drive.wgsl decides to stop.
// Included by leaders.wgsl / drive.wgsl, which bind `ctl` (CTL_W words per slot):
//   0 approach (i32, -1)  1 type (i32, -1)  2 light state  3 timeIn  4 timeLeft  5 dist (stop line - front bumper)
//   6 move | stopped << 8   7 mergeCap (m/s, sim.js _laneEnd)   8 owner tag (bitcast spawnTime)
//   9 leader2 (i32, -1)   10 gap2 (m)   (sim.js _findLeaders: the car in front of my leader)   11 pullUntil (sim time, f32)
//   12..14 (unused: 0)
//   16 inside signal cluster (i32, -1)   17 inside since (f32)   18 next car on my junction movement   19 junction bits
//   20 passWreck (i32, -1: the wreck I pass through the opposing lane)
// CTL_W (words per slot) is injected by gpuSim.js (CTL_WORDS)
const CTL_SIGNAL: i32 = 0; const CTL_STOP: i32 = 1; const CTL_GIVE_WAY: i32 = 2; const CTL_RAMP: i32 = 3; const CTL_CROSSING: i32 = 4;
const LIGHT_GREEN: u32 = 0u; const LIGHT_YELLOW: u32 = 1u; const LIGHT_RED: u32 = 2u; const LIGHT_SIGN: u32 = 3u; const LIGHT_DARK: u32 = 4u;
const MOVE_THROUGH: u32 = 0u; const MOVE_LEFT: u32 = 1u; const MOVE_RIGHT: u32 = 2u;
const CTL_REC: u32 = 24u;   // per control: type, cycle, offset, nPhases, 4 x (group | left << 8, g, y, r), cluster
const AP_REC: u32 = 4u;     // per approach: control, stopAt, group | protLeft << 8, edge

// stop lines of edge e: approaches world[ctlListOff + k], k in [edgeApBegin(e), edgeApBegin(e + 1)), sorted by stopAt
fn edgeApBegin(e: u32) -> u32 { return world[P.ctlEdgeOff + e]; }
fn edgeApAt(k: u32) -> u32 { return world[P.ctlListOff + k]; }
fn clusterOf(c: u32) -> u32 { return world[P.ctlCOff + c * CTL_REC + 20u]; }
fn apControl(a: u32) -> u32 { return world[P.ctlApOff + a * AP_REC]; }
fn apStopAt(a: u32) -> f32 { return bitcast<f32>(world[P.ctlApOff + a * AP_REC + 1u]); }
fn apEdge(a: u32) -> u32 { return world[P.ctlApOff + a * AP_REC + 3u]; }

fn movementOf(e: u32, en: u32) -> u32 {
  var d = bitcast<f32>(world[en * 16u + 12u]) - bitcast<f32>(world[e * 16u + 13u]);
  loop { if (d <= 3.14159265) { break; } d = d - 6.2831853; }
  loop { if (d >= -3.14159265) { break; } d = d + 6.2831853; }
  if (d < -0.5) { return MOVE_LEFT; }
  if (d > 0.5) { return MOVE_RIGHT; }
  return MOVE_THROUGH;
}

struct CtlOut { ty: i32, state: u32, timeIn: f32, timeLeft: f32 };
struct Phase { grp: u32, left: bool, g: f32, y: f32, r: f32 };
fn phaseAt(c: u32, k: u32) -> Phase {
  let o = P.ctlCOff + c * CTL_REC + 4u + k * 4u;
  let w = world[o];
  return Phase(w & 255u, ((w >> 8u) & 1u) != 0u, bitcast<f32>(world[o + 1u]), bitcast<f32>(world[o + 2u]), bitcast<f32>(world[o + 3u]));
}
fn phLen(p: Phase) -> f32 { return p.g + p.y + p.r; }

/** controls.js stateOf(a, move, t) */
fn ctlStateOf(a: u32, mv: u32, t: f32) -> CtlOut {
  let c = apControl(a); let co = P.ctlCOff + c * CTL_REC;
  let ty = i32(world[co]);
  var out = CtlOut(ty, LIGHT_SIGN, 0.0, 1e9);
  if (ty == CTL_STOP || ty == CTL_GIVE_WAY) { return out; }
  let C = bitcast<f32>(world[co + 1u]); let off = bitcast<f32>(world[co + 2u]);
  let tc = ((t + off) % C + C) % C;
  if (ty == CTL_RAMP) {
    let tod = t % 86400.0;
    if (tod < P.rampStart || tod > P.rampEnd) { out.state = LIGHT_DARK; return out; }
    if (tc < 2.0) { out.state = LIGHT_GREEN; out.timeIn = tc; out.timeLeft = 2.0 - tc; } else { out.state = LIGHT_RED; out.timeIn = tc - 2.0; out.timeLeft = C - tc; }
    return out;
  }
  if (ty == CTL_CROSSING) {
    if (tc < 75.0) { out.state = LIGHT_GREEN; out.timeIn = tc; out.timeLeft = 75.0 - tc; }
    else if (tc < 78.0) { out.state = LIGHT_YELLOW; out.timeIn = tc - 75.0; out.timeLeft = 78.0 - tc; }
    else { out.state = LIGHT_RED; out.timeIn = tc - 78.0; out.timeLeft = C - tc; }
    return out;
  }
  let n = world[co + 3u];
  let ag = world[P.ctlApOff + a * AP_REC + 2u];
  let grp = ag & 255u; let prot = ((ag >> 8u) & 1u) != 0u;
  var t0 = 0.0; var cur = -1; var inPhase = 0.0;
  for (var k = 0u; k < n; k = k + 1u) { let L = phLen(phaseAt(c, k)); if (tc < t0 + L) { cur = i32(k); inPhase = tc - t0; break; } t0 = t0 + L; }
  if (cur < 0) { cur = i32(n) - 1; inPhase = 0.0; }
  let p = phaseAt(c, u32(cur));
  let mineP = p.grp == grp && select(!p.left, p.left, mv == MOVE_LEFT && prot);
  if (mineP && inPhase < p.g) { out.state = LIGHT_GREEN; out.timeIn = inPhase; out.timeLeft = p.g - inPhase; return out; }
  if (mineP && inPhase < p.g + p.y) { out.state = LIGHT_YELLOW; out.timeIn = inPhase - p.g; out.timeLeft = p.g + p.y - inPhase; return out; }
  out.state = LIGHT_RED;
  var until = phLen(p) - inPhase; var k = (u32(cur) + 1u) % n; var guard = 0u;
  loop {
    let q = phaseAt(c, k);
    if ((q.grp == grp && select(!q.left, q.left, mv == MOVE_LEFT && prot)) || guard >= n) { break; }
    guard = guard + 1u; until = until + phLen(q); k = (k + 1u) % n;
  }
  out.timeLeft = until;
  var since = select(inPhase, inPhase - p.g - p.y, mineP); k = (u32(cur) + n - 1u) % n; guard = 0u;
  loop {
    let q = phaseAt(c, k);
    if ((q.grp == grp && select(!q.left, q.left, mv == MOVE_LEFT && prot)) || guard >= n) { break; }
    guard = guard + 1u; since = since + phLen(q); k = (k + n - 1u) % n;
  }
  let qk = phaseAt(c, k);
  if ((qk.grp == grp && select(!qk.left, qk.left, mv == MOVE_LEFT && prot)) && i32(k) != cur) { since = since + qk.r; }
  out.timeIn = max(0.0, since);
  return out;
}
