// bayenv-gpu model patches of the integrate stage (src/gpu/kernels/integrate.wgsl + the included physics.wgsl), applied by
// kernels.rs `model_patches` (after kernels/cpu_connector_join.wgsl) when any of the GpuOptions flags below is on; each
// //#orig snippet must occur exactly once in the resolved source. The host prepends `const MF_*: bool` for every flag; a flag
// that is off takes the JS code path. Mirrors bayenv-sim SimOpts (physics.rs Dyn, vehicle.rs integrate_post, sim.rs
// integrate eligibility); see rust/docs/physics-audit.md.
//   MF_HEAVY_BRAKES     heavy_brakes: vehicles over 5,000 kg brake at 4.7 m/s^2 (JS: PT_BRAKE 6.5)
//   MF_BRAKE_HOLD       brake_hold: below 0.05 m/s brake + rolling resistance >= |drive - grade| holds the car at 0
//   MF_IMPACT_REVERSE   impact_reverse: a locked (wreck) car may slide backwards (u < 0), its (u, v) rotated exactly by r h,
//                       stop test |u| < 0.05; a wreck with
//                       |speed| > 0.01 still integrates
//   MF_WRECK_END_STOP   wreck_end_stop: a sliding wreck clamped at the end of its edge / connector (or at s < 0 moving
//                       backwards) gets speed 0
//   MF_IDLE_CREEP       idle_creep: the drive force never drops below 4 x the rolling resistance at rest, fading to 0 at
//                       2 m/s (an automatic's idle creep, ~1.5 m/s on the flat)
//   MF_STATIC_HOLD      static_hold: below 0.05 m/s the brake holds the car at 0 when the brake force alone beats drive
//                       minus grade; with no drive and no brake, rolling resistance holds it against a smaller grade force
//   MF_STEER_DELAY      steer_delay (i32 steps, -1 = the JS): steering read from the control ring this many steps back
//                       when that is shorter than the car's reaction delay; the pedals keep the reaction delay
//   MF_LANE_DROP_HOP    lane_drop_hop: a strict hop from a lane that continues onto no out-edge takes the strict out-edge
//                       its lane overshoots least (EV_MISSED if not the planned one; JS: the planned edge, strict lat)
//   MF_ROLLBACK         hill_rollback: a driving car that drive + brake + rolling resistance can't hold on the grade rolls
//                       backwards (brakes push forwards while rolling back, hold it at rest against the grade; rolling
//                       resistance opposes it); clamped at s 0 of its edge with speed 0
//   MF_CURVATURE        curvature_dynamics: the world heading changes by the car's own yaw only (headingRel takes up the
//                       road frame's rotation between the start and the end pose, connector / edge transitions included)
//#orig brake-fb
    let Fb = bk * m * PT_BRAKE[t];
//#end
//#patch brake-fb
    let Fb = bk * m * select(PT_BRAKE[t], 4.7, MF_HEAVY_BRAKES && m > 5000.0);
//#end
//#orig idle-creep
    if (locked || u >= PT_VMAX[t]) { Fd = 0.0; }
//#end
//#patch idle-creep
    if (locked || u >= PT_VMAX[t]) { Fd = 0.0; }
    if (MF_IDLE_CREEP && !locked && u < 2.0) {   // bayenv-sim IDLE_CREEP_ROLL = 4, IDLE_CREEP_U = 2
      let fi = 4.0 * PT_CRR[t] * sf.z * m * G_ACC * cosG * (1.0 - max(0.0, u) / 2.0);
      if (fi > Fd) { Fd = fi; }
    }
//#end
//#orig steer-delay
      let sr = ring[o]; var gas = ring[o + 1u]; var brake = ring[o + 2u];
//#end
//#patch steer-delay
      var sr = ring[o]; var gas = ring[o + 1u]; var brake = ring[o + 2u];
      if (MF_STEER_DELAY >= 0 && MF_STEER_DELAY < i32(pi.delaySteps)) { sr = ring[(base + (step + CTRL_RING - u32(max(MF_STEER_DELAY, 0))) % CTRL_RING) * CTRL_STRIDE]; }
//#end
//#orig u-clamp
    var un = u + (FxBody / m + v * r) * h;
    if (un < 0.0) { un = 0.0; }
    var vn = v + (FyBody / m - u * r) * h;
//#end
//#patch u-clamp
    // impact_reverse wrecks (locked, u may be < 0): the body-frame rotation terms (v r, -u r) as an exact rotation by r h
    // (explicit Euler grows |(u, v)| by sqrt(1 + (r h)^2) per sub-step: a runaway for a spinning wreck; bayenv-sim physics.rs)
    let exactRot = MF_IMPACT_REVERSE && locked;
    var ur = u; var vr = v;
    if (exactRot) { let cr = cos(r * h); let sr2 = sin(r * h); ur = u * cr + v * sr2; vr = -u * sr2 + v * cr; }
    var un = select(u + (FxBody / m + v * r) * h, ur + FxBody / m * h, exactRot);
    if (MF_ROLLBACK && !locked) {
      // hill_rollback: a driving car rolls back only when gravity beats drive + brake + rolling resistance at rest;
      // a backward-rolling car that the forces bring to a stop stops (no reverse drive)
      let roll0 = PT_CRR[t] * sf.z * m * G_ACC * cosG;
      if (u >= -0.02 && un < 0.0 && Wg <= Fd + Fb + roll0) { un = 0.0; }
      if (u < -0.02 && un > 0.0) { un = 0.0; }
    } else if (un < 0.0 && !(MF_IMPACT_REVERSE && locked)) { un = 0.0; }
    if (MF_BRAKE_HOLD && !locked && u <= 0.05) {
      let roll = PT_CRR[t] * sf.z * m * G_ACC * cosG;
      if (Fb + roll >= abs(Fd - Wg) && un <= 0.05) { un = 0.0; }
    }
    if (MF_STATIC_HOLD && !locked && u <= 0.05 && u >= -0.05 && un <= 0.05 && un >= -0.05) {
      let roll = PT_CRR[t] * sf.z * m * G_ACC * cosG;
      if ((b_ > 0.0 && Fb >= abs(Fd - Wg)) || (Fd == 0.0 && b_ == 0.0 && abs(Wg) <= roll)) { un = 0.0; }
    }
    var vn = select(v + (FyBody / m - u * r) * h, vr + FyBody / m * h, exactRot);
//#end
//#orig locked-stop
    if (locked && un < 0.05 && abs(vn) < 0.05) {
//#end
//#patch locked-stop
    if (locked && select(un, abs(un), MF_IMPACT_REVERSE) < 0.05 && abs(vn) < 0.05) {
//#end
//#orig wreck-moving
      if (v.speed > 0.01 || abs(v.vLat) > 0.01 || abs(v.yawRate) > 0.01) {
//#end
//#patch wreck-moving
      if (select(v.speed, abs(v.speed), MF_IMPACT_REVERSE) > 0.01 || abs(v.vLat) > 0.01 || abs(v.yawRate) > 0.01) {
//#end
//#orig wreck-clamp
        veh[i].headingRel = st.hr; veh[i].lat = st.lat; veh[i].s = clamp(st.s, 0.0, frameEnd(v));
//#end
//#patch wreck-clamp
        veh[i].headingRel = st.hr; veh[i].lat = st.lat; veh[i].s = clamp(st.s, 0.0, frameEnd(v));
        if (MF_WRECK_END_STOP && (st.s >= frameEnd(v) || (st.s < 0.0 && st.u < 0.0))) { veh[i].speed = 0.0; }
//#end
//#orig wreck-pose
        updatePose(i);
      }
      if ((v.flags & F_SCORED) == 0u
//#end
//#patch wreck-pose
        if (MF_CURVATURE) {
          let w = veh[i]; let d = (poseOf(w).w + w.headingRel) - (w.pyaw - (st.hr - v.headingRel));
          veh[i].headingRel = atan2(sin(d), cos(d));
        }
        updatePose(i);
      }
      if ((v.flags & F_SCORED) == 0u
//#end
//#orig drive-yaw
      v.speed = st.u; v.vLat = st.v; v.yawRate = st.r;
//#end
//#patch drive-yaw
      let yawDh = st.hr - v.headingRel;   // curvature_dynamics: the physics-integrated heading change
      v.speed = st.u; v.vLat = st.v; v.yawRate = st.r;
//#end
//#orig drive-store
      veh[i] = v;
      if (false) {
//#end
//#patch drive-store
      if (MF_CURVATURE) { let d = (poseOf(v).w + v.headingRel) - (v.pyaw - yawDh); v.headingRel = atan2(sin(d), cos(d)); }
      veh[i] = v;
      if (false) {
//#end
//#orig lane-drop-hop
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
//#end
//#patch lane-drop-hop
            let nd = eTo(v.edge);
            var found = false;
            for (var q = outStart(nd); q < outStart(nd + 1u); q = q + 1u) {
              let o = outEdge(q);
              if (o == en || i32(o) == eRev(v.edge) || hopMap(v.edge, o).x != 1) { continue; }
              let l3 = mapLane(v.edge, o, ln);
              if (l3 >= 0 && l3 < i32(eLanes(o))) {
                routes[pi.routeOff + nxt] = o;   // no maneuver code: the CPU replaces the route from here
                emit(EV_MISSED, i, o, pi.trip);
                en = o; found = true; break;
              }
            }
            // lane_drop_hop (bayenv-sim vehicle.rs): my lane continues nowhere -> the strict out-edge it overshoots least
            if (!found && MF_LANE_DROP_HOP) {
              var best = en; let l0 = mapLane(v.edge, en, ln); var bo = select(select(0, l0 - i32(eLanes(en)) + 1, l0 >= i32(eLanes(en))), -l0, l0 < 0);
              for (var q = outStart(nd); q < outStart(nd + 1u); q = q + 1u) {
                let o = outEdge(q);
                if (o == en || i32(o) == eRev(v.edge) || hopMap(v.edge, o).x != 1) { continue; }
                let lo2 = mapLane(v.edge, o, ln); let ov = select(select(0, lo2 - i32(eLanes(o)) + 1, lo2 >= i32(eLanes(o))), -lo2, lo2 < 0);
                if (ov < bo) { bo = ov; best = o; }
              }
              if (best != en) { routes[pi.routeOff + nxt] = best; emit(EV_MISSED, i, best, pi.trip); en = best; }
            }
//#end
//#orig rollback-brakes
    if (moving) { Fxf = Fxf - bf; Fxr = Fxr - br; }
    else { let hold = min(Fb, max(0.0, Fd - Wg)); let hf = min(bf, hold * shareF); Fxf = Fxf - hf; Fxr = Fxr - min(br, hold - hf); }
//#end
//#patch rollback-brakes
    let back = MF_ROLLBACK && !locked && u < -0.02;
    if (back) { Fxf = Fxf + bf; Fxr = Fxr + br; }
    else if (moving) { Fxf = Fxf - bf; Fxr = Fxr - br; }
    else {
      let hold = min(Fb, max(0.0, Fd - Wg)); let hf = min(bf, hold * shareF); Fxf = Fxf - hf; Fxr = Fxr - min(br, hold - hf);
      if (MF_ROLLBACK && !locked) { let hb = min(Fb, max(0.0, Wg - Fd)); let hf2 = min(bf, hb * shareF); Fxf = Fxf + hf2; Fxr = Fxr + min(br, hb - hf2); }
    }
//#end
//#orig rollback-res
    let res = resistanceS(t, u, grade, sf.z);
//#end
//#patch rollback-res
    var res = resistanceS(t, u, grade, sf.z);
    if (back && u < -0.05) { res = res - PT_CRR[t] * sf.z * m * G_ACC; }   // hill_rollback: rolling resistance opposes rolling back
//#end
//#orig rollback-s0
      var s = max(0.0, st.s);
//#end
//#patch rollback-s0
      var s = max(0.0, st.s);
      if (MF_ROLLBACK && st.s < 0.0 && v.speed < 0.0) { v.speed = 0.0; }   // hill_rollback: rolled back to the edge start
//#end
