// Stages (d) reaction-time ring, (e) dynamics + edge transitions, (g) fuel, (i) pose,
// and the per-step part of (h) stats. Mirrors src/sim/vehicle.js integrate / updateLanes /
// updatePose and sim.js _integrate:
//   ring: 40 records x {steerRate, gas, brake}; the command decided at step k acts at
//         k + delaySteps; steps without a new decision hold the previous record
//   layout v4: savePose, then src/sim/physics stepDynamics (dynamic bicycle, 10 ms sub-steps,
//   BSFC fuel) on the delayed controls; hops via mapLat (strict lane / junction re-projection);
//   disabled wrecks that still move slide with locked wheels; pose from src/geom latPoint
// Vehicle-type constants come in as consts from gpuSim.js (VT_* arrays, fuel model).
// #include common.wgsl
// #include physics.wgsl

@group(1) @binding(0) var<storage, read_write> veh: array<Veh>;
@group(1) @binding(1) var<storage, read> par: array<VPar>;
@group(1) @binding(2) var<storage, read_write> ring: array<f32>;
@group(1) @binding(3) var<storage, read_write> pose: array<vec4f>;
// per agent (commuter = trip >> 1): x rewardSum (f32 bits), y outcome (leg0 | leg1 << 8; 1 arrived,
// 2 crash, 3 stranded, 4 stuck), z trip seconds of the last finished leg (f32 bits), w crashes
@group(1) @binding(4) var<storage, read_write> agents: array<vec4u>;
@group(1) @binding(5) var<storage, read_write> telemetry: array<atomic<u32>>;
@group(1) @binding(6) var<storage, read_write> ctl: array<u32>;   // controls.wgsl side record (word 22: stoppedSince already labelled stuck)   // src/telemetry/gpuLayout.js W

var<workgroup> wDriving: atomic<u32>;
var<workgroup> wCrashed: atomic<u32>;
var<workgroup> wRefuel: atomic<u32>;
var<workgroup> wSumSpeed: atomic<u32>;
var<workgroup> wVehCm: atomic<u32>;
var<workgroup> wSpeedCm: atomic<u32>;
var<workgroup> wSpeedSteps: atomic<u32>;
var<workgroup> wSteps: atomic<u32>;

fn agentOf(i: u32) -> u32 { let a = par[i].trip >> 1u; return select(NONE, a, a < P.nAgents); }
fn addReward(a: u32, r: f32) { if (a != NONE) { agents[a].x = bitcast<u32>(bitcast<f32>(agents[a].x) + r); } }
// agents[a].y: outbound outcome | return outcome << 8 | outbound progress (routeDist / routeTotal, 16 bits) << 16
fn finishLeg(a: u32, leg: u32, code: u32, tripS: f32, prog: f32) {
  if (a == NONE) { return; }
  var o = agents[a].y;
  let sh = leg * 8u;
  if (((o >> sh) & 255u) == 0u) {
    o = o | (code << sh);
    if (leg == 0u) { o = (o & 0xffffu) | (u32(clamp(prog, 0.0, 1.0) * 65535.0) << 16u); }
  }
  agents[a].y = o; agents[a].z = bitcast<u32>(tripS);
}
fn release(i: u32) { veh[i].state = FREE; pose[i] = vec4f(0.0, -1000.0, 0.0, 0.0); }

// src/sim/surface.js carSurface: road-surface factors (pk, sk, crr) under the right / left wheels. Edge flag bits
// 16..19 pavement code, 20..23 ground right, 24..27 ground left (worldPack.js); SF_* per-code factor tables.
fn surfFactors(e: u32, s: f32, lat: f32, hw: f32, conn: i32, taper: f32) -> vec3f {
  let fl = eFlags(e);
  let cs = (fl >> 16u) & 15u; let gR = (fl >> 20u) & 15u; let gL = (fl >> 24u) & 15u;
  var cR = cs; var cL = cs;
  let inBox = P.hasJ != 0u && (s < eSExit(e) + 1.0 || s > eSStop(e));
  if (conn < 0 && !inBox && (gR != cs || gL != cs)) {
    let W = laneWAt(e, s); let latR = lat - hw; let latL = lat + hw; let nW = f32(eLanes(e)) * W;
    if (latR < 0.0 || latL > nW) {
      var padR = 0.0; var padL = nW;
      if (P.hasLG != 0u) { let bd = boundsAt(e, s); padR = bd.z - bd.x; padL = bd.z - bd.y; }
      else if (taper > s) { padR = padR - W; padL = padL + W; }
      let sh = select(0.0, PAVED_SH_FW, (fl & 64u) != 0u);
      var left = padL + sh;
      if (eRev(e) >= 0) { left = padL + f32(eLanes(u32(eRev(e)))) * W; }
      if (latR < padR - sh) { cR = gR; }
      if (latL > left) { cL = gL; }
    }
  }
  return vec3f(0.5 * (SF_PK[cR] + SF_PK[cL]), 0.5 * (SF_SK[cR] + SF_SK[cL]), 0.5 * (SF_CRR[cR] + SF_CRR[cL]));
}

fn updatePose(i: u32) { pose[i] = poseOf(veh[i]); }
fn savePose(i: u32) { let p = pose[i]; veh[i].px = p.x; veh[i].pz = p.z; veh[i].pyaw = p.w; }

fn clearControls(i: u32) {
  for (var k = 0u; k < CTRL_RING * CTRL_STRIDE; k = k + 1u) { ring[i * CTRL_RING * CTRL_STRIDE + k] = 0.0; }
  veh[i].steerSl = veh[i].steerAngle; veh[i].gas = 0.0; veh[i].brake = 0.0;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3u, @builtin(local_invocation_index) li: u32) {
  if (li == 0u) {
    atomicStore(&wDriving, 0u); atomicStore(&wCrashed, 0u); atomicStore(&wRefuel, 0u); atomicStore(&wSumSpeed, 0u);
    atomicStore(&wVehCm, 0u); atomicStore(&wSpeedCm, 0u); atomicStore(&wSpeedSteps, 0u); atomicStore(&wSteps, 0u);
  }
  workgroupBarrier();
  let i = g.x;
  let now = simTime();
  let dt = P.dt;
  if (i < P.cap) {
    var v = veh[i];
    if (v.state == CRASHED) {
      atomicAdd(&wCrashed, 1u);
      savePose(i);
      if (v.speed > 0.01 || abs(v.vLat) > 0.01 || abs(v.yawRate) > 0.01) {
        let pc = par[i];
        var st: DynState;
        st.u = v.speed; st.v = v.vLat; st.r = v.yawRate; st.steer = v.steerAngle; st.sl = v.steerSl; st.ax = v.axPrev; st.hr = v.headingRel; st.lat = v.lat; st.s = v.s;
        st = stepDynamicsS(st, 0.0, 0.0, 1.0, pc.typeIdx, dt, edgeGrade(v.edge, v.s), true, surfFactors(v.edge, v.s, v.lat, 0.5 * pc.width, vConn(v), v.taper));
        veh[i].speed = st.u; veh[i].vLat = st.v; veh[i].yawRate = st.r; veh[i].steerAngle = st.steer; veh[i].axPrev = st.ax;
        veh[i].headingRel = st.hr; veh[i].lat = st.lat; veh[i].s = clamp(st.s, 0.0, frameEnd(v));
        veh[i].lanes = derivedLanes(v.edge, veh[i].s, st.lat, halfExtent(veh[i], pc), vTarget(v));
        updatePose(i);
      }
      if ((v.flags & F_SCORED) == 0u && (v.flags & (F_COLLIDED | F_STRANDED)) != 0u) {
        // reward.js: crash or out of fuel = -20, the trip has failed (rewardVersion 2, P.crashOn: a crash costs the
        // severity penalty contacts.wgsl crashCharge stashed in eventTime, capped; stranding stays -20)
        let a = agentOf(i);
        var pen = -20.0;
        if (P.crashOn != 0u && (v.flags & F_COLLIDED) != 0u && (v.flags & F_STRANDED) == 0u) {
          pen = select(-min(P.crashCap, P.crashBase + P.crashFail), max(v.eventTime, -P.crashCap), v.eventTime < 0.0);
          veh[i].eventTime = 0.0;
        }
        addReward(a, pen);
        if (a != NONE && (v.flags & F_COLLIDED) != 0u) { agents[a].w = agents[a].w + 1u; }
        finishLeg(a, par[i].trip & 1u, select(2u, 3u, (v.flags & F_STRANDED) != 0u), now - v.spawnTime, v.routeDist / max(v.routeTotal, 1.0));
        veh[i].flags = v.flags | F_SCORED;
      }
    }
    else if (v.state == REFUELING) { atomicAdd(&wRefuel, 1u); }
    else if (v.state == DRIVING) {
      let pi = par[i];
      let step = simStep();
      // rewardVersion 2: a minor crash (car still driving) charged here, on the step after the contact (contacts.wgsl crashCharge)
      if (P.crashOn != 0u && v.eventTime < 0.0) { addReward(agentOf(i), max(v.eventTime, -P.crashCap)); v.eventTime = 0.0; veh[i].eventTime = 0.0; }
      if (!hasFeat(FEAT_FUEL)) { v.fuel = v.tank; }
      // ---- (d) control delay ring
      let base = i * CTRL_RING;
      let cur = (base + step % CTRL_RING) * CTRL_STRIDE;
      if (v.cmdStep != i32(step)) {
        let prev = (base + (step + CTRL_RING - 1u) % CTRL_RING) * CTRL_STRIDE;
        ring[cur] = ring[prev]; ring[cur + 1u] = ring[prev + 1u]; ring[cur + 2u] = ring[prev + 2u];
      } else if (P.rlOn != 0u && P.jerkW > 0.0 && (v.flags & F_POLICY) != 0u && (P.steerMode & 1u) != 0u) {
        // steering smoothness (src/sim/reward.js JERK, driver.js lastJerk / lastJerkScale): - jerkW * dSteer^2 *
        // min(4, (v/20)^2) * ease(time to collision with the own-lane leader / lane-end wall: 0 below 1 s, 1 above 3 s).
        // dSteer = this decision's command - the ring entry of the step before. (CPU difference: the CPU leader also
        // includes the junction-yield virtual obstacle of adjust.js; not here. Only during RL rollouts: the leader's speed
        // is read while other threads integrate, so the ease is approximate by one step; it only touches the reward.)
        let prevO = (base + (step + CTRL_RING - 1u) % CTRL_RING) * CTRL_STRIDE;
        let dS = ring[cur] - ring[prevO];
        var ttc = 1e9;
        let Lj = v.leader;
        if ((Lj >= 0 || Lj == -2) && v.gap < 1e8) {
          var vl = 0.0; if (Lj >= 0 && veh[u32(Lj)].state == DRIVING) { vl = veh[u32(Lj)].speed; }
          let cl = v.speed - vl; if (cl > 0.1) { ttc = max(0.0, v.gap) / cl; }
        }
        let sv = min(4.0, (v.speed / 20.0) * (v.speed / 20.0));
        let ease = clamp((ttc - 1.0) / 2.0, 0.0, 1.0);
        addReward(agentOf(i), -P.jerkW * dS * dS * sv * ease);
      }
      let d = min(pi.delaySteps, CTRL_RING - 1u);
      let o = (base + (step + CTRL_RING - d) % CTRL_RING) * CTRL_STRIDE;
      let sr = ring[o]; var gas = ring[o + 1u]; var brake = ring[o + 2u];
      if (v.fuel <= 0.0) { gas = 0.0; if (brake < 0.3) { brake = 0.3; } }
      v.gas = gas; v.brake = brake;
      // ---- (e) physics (src/sim/physics/dynamics.js) from the start-of-step pose
      savePose(i); v.px = veh[i].px; v.pz = veh[i].pz; v.pyaw = veh[i].pyaw;
      let u0 = v.speed;
      let s0 = v.s;
      var st: DynState;
      st.u = v.speed; st.v = v.vLat; st.r = v.yawRate; st.steer = v.steerAngle; st.sl = v.steerSl; st.ax = v.axPrev; st.hr = v.headingRel; st.lat = v.lat; st.s = v.s;
      st = stepDynamicsS(st, sr, gas, brake, pi.typeIdx, dt, select(edgeGrade(v.edge, v.s), 0.0, vConn(v) >= 0), false, surfFactors(v.edge, v.s, v.lat, 0.5 * pi.width, vConn(v), v.taper));
      v.speed = st.u; v.vLat = st.v; v.yawRate = st.r; v.steerAngle = st.steer; v.steerSl = st.sl; v.axPrev = st.ax; v.headingRel = st.hr; v.lat = st.lat; v.power = st.power;
      let nv = st.u;
      var a = (st.u - u0) / dt;
      v.accel = a;
      if (v.fuel > 0.0) { v.fuel = max(0.0, v.fuel - st.fuelL * P.fuelScale); }
      let ds = st.s - s0;
      v.odometer = v.odometer + abs(ds);
      v.routeDist = v.routeDist + ds;
      addReward(agentOf(i), P.progressW * ds / max(eSpeedLimit(v.edge), 1.0));   // reward.js progress term
      if (st.u < 0.2 && abs(st.v) < 0.2) { if (v.stoppedSince < 0.0) { v.stoppedSince = now; } } else { v.stoppedSince = -1.0; }
      var s = max(0.0, st.s);
      var len = eLength(v.edge);
      var ret = 0u;
      var tgtLane = vTarget(v);
      loop {
        // ---- junction connectors (vehicle.js, layout v8): past the mouth's stop point a car follows the
        // connector of its movement (in-edge, lane, out-edge) through the box; s keeps counting in the in-edge
        // frame (t = s - sIn), lat keeps its offset from the in-lane centre
        if (P.hasJ != 0u) {
          let m = vConn(v);
          if (m >= 0) {
            let mu = u32(m);
            let end = movSIn(mu) + movLen(mu);
            if (s < end) { break; }
            let en = movOutEdge(mu);
            let off = v.lat - (f32(movInLane(mu)) + 0.5) * laneWAt(v.edge, movSIn(mu));
            v.routeDist = v.routeDist + len - end + movSOut(mu);   // route distance counts edge lengths
            s = movSOut(mu) + (s - end);
            v.lat = (f32(movOutLane(mu)) + 0.5) * laneWAt(en, movSOut(mu)) + off;
            if (tgtLane >= 0) { tgtLane = i32(movOutLane(mu)); }
            let bIn = eBridge(en);
            if (bIn != 0u && bIn != eBridge(v.edge)) { atomicAdd(&telemetry[TW_BRIDGE_ENTRIES + bIn], 1u); }
            v.routePos = v.routePos + 1u; v.edge = en; v.seg = 0u; v.taper = 0.0;
            len = eLength(en);
            continue;
          }
          let sStop = eSStop(v.edge);
          // vehicle.js: the connector leaves e at its movement's own sIn (<= sStop since the junctions rebuild)
          if (sStop < len && !(v.refuelState == 1u && i32(v.routePos) == v.stopPos)) {
            let nxt = v.routePos + 1u;
            if (nxt < pi.routeLen) {
              let la = u32(clamp(floor(v.lat / laneWAt(v.edge, s)), 0.0, f32(eLanes(v.edge)) - 1.0));
              var mv = jMovement(v.edge, la, routeEdge(pi, nxt));
              if (mv >= 0 && s < movSIn(u32(mv))) { mv = -2; }
              // a turn is made from its lane: a car in a lane that is not the turn's lane goes the way its lane
              // leads (the straightest preferred movement from that lane) - a missed turn the CPU re-plans; only
              // where there was room to get into lane (edge >= 80 m), and not again within 5 min (ctl word 31)
              let lastM = bitcast<f32>(ctl[i * CTL_W + 31u]);
              let ownC = ctl[i * CTL_W + 8u] == bitcast<u32>(v.spawnTime);
              if (mv >= 0 && ((movW(u32(mv), 0u) >> 28u) & 1u) == 0u && eLanes(v.edge) > 1u && len >= 80.0 && !(ownC && now - lastM < 300.0)) {
                var best = -1; var bt = 1e9;
                let nd = eTo(v.edge);
                for (var q = outStart(nd); q < outStart(nd + 1u); q = q + 1u) {
                  let o = outEdge(q); if (i32(o) == eRev(v.edge)) { continue; }
                  let m2 = jMovement(v.edge, la, o);
                  if (m2 >= 0 && ((movW(u32(m2), 0u) >> 28u) & 1u) != 0u && abs(movTurn(u32(m2))) < bt) { bt = abs(movTurn(u32(m2))); best = m2; }
                }
                if (best >= 0) {
                  let took = movOutEdge(u32(best));
                  routes[pi.routeOff + nxt] = took;
                  emit(EV_MISSED, i, took | 0x80000000u, v.edge);   // via a junction connector: b = in-edge
                  mv = best;
                  if (ownC) { ctl[i * CTL_W + 31u] = bitcast<u32>(now); }
                }
              }
              if (mv >= 0 && s >= movSIn(u32(mv))) {
                let mu = u32(mv); let sIn = movSIn(mu);
                if (s > sIn + 0.3 && P.hasLG != 0u) {
                  // already past the entry point (back-to-back boxes on a short edge): join the connector at the
                  // point nearest my body, keeping world position and heading
                  let p = latPointAt(v.edge, min(s, len), v.lat);
                  let fOld = atan2(p.hz, p.hx); let L = min(movLen(mu), s - sIn + 2.0);
                  // nearest in position and heading (a heading error costs like 6 m of distance per rad)
                  let hOld = fOld - v.headingRel;
                  var tb = 0.0; var bd = 1e18;
                  for (var t = 0.0; t <= L; t = t + 0.5) {
                    let c = connPointJ(mu, t); var dp = atan2(c.hz, c.hx) - hOld; dp = atan2(sin(dp), cos(dp));
                    let d = (c.x - p.x) * (c.x - p.x) + (c.z - p.z) * (c.z - p.z) + 36.0 * dp * dp; if (d < bd) { bd = d; tb = t; }
                  }
                  let c = connPointJ(mu, tb);
                  let off = (p.x - c.x) * c.hz - (p.z - c.z) * c.hx;
                  s = sIn + tb; v.lat = (f32(movInLane(mu)) + 0.5) * laneWAt(v.edge, sIn) + off;
                  var dh = atan2(c.hz, c.hx) - fOld; dh = atan2(sin(dh), cos(dh));
                  v.headingRel = v.headingRel + dh;
                }
                v.taper = connWord(mv); if (tgtLane >= 0) { tgtLane = i32(la); } continue;
              }
            }
          }
        }
        if (s < len) { break; }
        if (v.refuelState == 1u && i32(v.routePos) == v.stopPos) { v.routeDist = v.routeDist - (s - len); s = len; ret = 2u; break; }
        let nxt = v.routePos + 1u;
        if (nxt >= pi.routeLen) { v.routeDist = v.routeDist - (s - len); s = len; ret = 1u; break; }
        s = s - len;
        var en = routeEdge(pi, nxt);
        // a strict diverge is physical (vehicle.js, layout v8): if my lane does not continue onto the planned
        // edge but onto another out-edge, that is where I go (missed exit; patched in place, the CPU re-plans)
        if (P.hasLG != 0u && hopMap(v.edge, en).x == 1) {
          let ln = i32(clamp(floor(v.lat / laneWAt(v.edge, len)), 0.0, f32(eLanes(v.edge)) - 1.0));
          let ml = mapLane(v.edge, en, ln);
          if (ml < 0 || ml >= i32(eLanes(en))) {
            let nd = eTo(v.edge);
            for (var q = outStart(nd); q < outStart(nd + 1u); q = q + 1u) {
              let o = outEdge(q);
              if (o == en || i32(o) == eRev(v.edge) || hopMap(v.edge, o).x != 1) { continue; }
              let l3 = mapLane(v.edge, o, ln);
              if (l3 >= 0 && l3 < i32(eLanes(o))) {
                routes[pi.routeOff + nxt] = o;   // no maneuver code: the CPU replaces the route from here
                emit(EV_MISSED, i, o, pi.trip);
                en = o; break;
              }
            }
          }
        }
        let heH = halfExtent(v, pi);
        var l2 = mapLat(v.edge, en, v.lat, heH);
        // vehicle.js: joining a carriageway with the body on the edge line -> start on the nearest lane centre
        let LWn = laneWAt(en, s); let nW = f32(eLanes(en)) * LWn;
        if (l2 >= 0.0 && l2 <= nW && (l2 - heH < 0.3 || l2 + heH > nW - 0.3) && (hopMap(v.edge, en).x == 0 || eLanes(v.edge) != eLanes(en) || (eFlags(en) & 64u) != 0u)) {
          l2 = (clamp(floor(l2 / LWn), 0.0, f32(eLanes(en)) - 1.0) + 0.5) * LWn;
        }
        v.lat = l2;
        if (tgtLane >= 0) { tgtLane = clamp(mapLane(v.edge, en, tgtLane), 0, i32(eLanes(en)) - 1); }
        if (P.hasJ == 0u) { v.taper = select(0.0, TAPER, P.hasLG == 0u && (v.lat < 0.0 || v.lat > nW)); }   // vehicle.js: with src/geom the taper is drawn before the node (taper holds the connector with junctions)
        let bIn = eBridge(en);
        if (bIn != 0u && bIn != eBridge(v.edge)) { atomicAdd(&telemetry[TW_BRIDGE_ENTRIES + bIn], 1u); }
        v.routePos = nxt; v.edge = en; v.seg = 0u;
        len = eLength(en);
      }
      v.s = s;
      v.lanes = derivedLanes(v.edge, v.s, v.lat, halfExtent(v, pi), tgtLane);
      // stats
      let cm = u32(nv * dt * 100.0 + 0.5);
      atomicAdd(&wVehCm, cm); atomicAdd(&wSteps, 1u);
      if (nv > 1.1 * eSpeedLimit(v.edge)) { v.flags = v.flags | F_SPEEDING; atomicAdd(&wSpeedCm, cm); atomicAdd(&wSpeedSteps, 1u); }
      else { v.flags = v.flags & ~F_SPEEDING; }
      if (ret == 0u) {
        let rem = len - v.s;
        let stopHere = v.refuelState == 1u && v.stopPos == i32(v.routePos);
        if ((v.routePos + 1u >= pi.routeLen || stopHere) && (rem < 2.5 || (rem < 8.0 && nv < 0.3))) { ret = select(1u, 2u, stopHere); }
      }
      veh[i] = v;
      if (false) {
      } else if (ret == 1u) {
        let leg = pi.trip & 1u;
        let late = now - pi.workStart;
        emit(EV_TRIPX, i, bitcast<u32>(v.odometer), bitcast<u32>(now - v.spawnTime));
    emit(EV_ARRIVE, i, pi.trip, bitcast<u32>(v.fuel));
        statAdd(S_ARRIVED, 1u);
        let ag = agentOf(i);
        let lateBy = select(0.0, late, leg == 0u);
        addReward(ag, 10.0 + select(max(-10.0, -0.5 * lateBy / 60.0), 2.0, lateBy <= 0.0));
        finishLeg(ag, leg, 1u, now - v.spawnTime, 1.0);
        if (leg == 0u) {
          statAdd(S_ARRIVED_OUT, 1u);
          if (late <= 0.0) { statAdd(S_ONTIME, 1u); }
          statAdd64(S_LATESUM_LO, u32(max(0.0, late)));
        }
        release(i);
      } else if (ret == 2u) {
        updatePose(i);
        veh[i].state = REFUELING; veh[i].speed = 0.0; veh[i].accel = 0.0; veh[i].refuelState = 2u; veh[i].eventTime = now;
        clearControls(i);
        emit(EV_STATION, i, pi.trip, 0u);
        atomicAdd(&wRefuel, 1u);
      } else if (v.fuel <= 0.0 && nv < 0.05) {
        updatePose(i);
        veh[i].state = CRASHED; veh[i].speed = 0.0; veh[i].accel = 0.0; veh[i].flags = veh[i].flags | F_STRANDED;
        clearControls(i);
        emitWreck(i, veh[i], pi.typeIdx);
        emit(EV_STRANDED, i, pi.trip, v.edge);
        statAdd(S_STRANDED, 1u);
        atomicAdd(&wCrashed, 1u);
      } else {
        // layout v8: waiting is normal traffic - a car stopped > stuckTimeout is only labelled (once per stop)
        if (v.stoppedSince >= 0.0 && now - v.stoppedSince > P.stuckTimeout && ctl[i * CTL_W + 22u] != bitcast<u32>(v.stoppedSince)) {
          ctl[i * CTL_W + 22u] = bitcast<u32>(v.stoppedSince);
          emit(EV_STUCK, i, pi.trip, bitcast<u32>(now - v.stoppedSince));
          statAdd(S_STUCK, 1u);
        }
        if (hasFeat(FEAT_FUEL) && now - v.lastFuelCheck > P.fuelCheckInterval) {
          veh[i].lastFuelCheck = now;
          if (v.refuelState == 0u && (v.flags & F_FUELREQ) == 0u && needsFuel(pi.typeIdx, v.fuel, v.tank, v.routeTotal - v.routeDist)) {
            veh[i].flags = v.flags | F_FUELREQ;
            emit(EV_LOWFUEL, i, v.edge, 0u);
            statAdd(S_LOWFUEL, 1u);
          }
        }
        updatePose(i);
        atomicAdd(&wDriving, 1u);
        atomicAdd(&wSumSpeed, u32(nv * 100.0 + 0.5));
      }
    }
  }
  workgroupBarrier();
  if (li == 0u) {
    statAdd(S_LIVE_DRIVING, atomicLoad(&wDriving));
    statAdd(S_LIVE_CRASHED, atomicLoad(&wCrashed));
    statAdd(S_LIVE_DWELL, atomicLoad(&wRefuel));
    statAdd(S_LIVE_SUMSPEED, atomicLoad(&wSumSpeed));
    statAdd64(S_VEHM_LO, atomicLoad(&wVehCm));
    statAdd64(S_SPEEDM_LO, atomicLoad(&wSpeedCm));
    statAdd64(S_SPEEDSTEPS_LO, atomicLoad(&wSpeedSteps));
    statAdd64(S_VEHSTEPS_LO, atomicLoad(&wSteps));
  }
}

// pose + start pose of cars that entered the road this step (spawn / station re-entry)
@compute @workgroup_size(256)
fn newPose(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= P.cap) { return; }
  let v = veh[i];
  if ((v.flags & F_NEW) == 0u) { return; }
  updatePose(i);
  let p = pose[i];
  veh[i].px = p.x; veh[i].pz = p.z; veh[i].pyaw = p.w;
  veh[i].flags = v.flags & ~F_NEW;
}
