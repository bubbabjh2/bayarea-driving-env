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
//   MF_CURVATURE        curvature_dynamics: the world heading changes by the car's own yaw only (headingRel takes up the
//                       road frame's rotation between the start and the end pose, connector / edge transitions included)
//#orig brake-fb
    let Fb = bk * m * PT_BRAKE[t];
//#end
//#patch brake-fb
    let Fb = bk * m * select(PT_BRAKE[t], 4.7, MF_HEAVY_BRAKES && m > 5000.0);
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
    if (un < 0.0 && !(MF_IMPACT_REVERSE && locked)) { un = 0.0; }
    if (MF_BRAKE_HOLD && !locked && u <= 0.05) {
      let roll = PT_CRR[t] * sf.z * m * G_ACC * cosG;
      if (Fb + roll >= abs(Fd - Wg) && un <= 0.05) { un = 0.0; }
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
