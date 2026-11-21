// bayenv-gpu model patch (GpuOptions.cpu_connector_join, default off): text replacements in the resolved source of
// src/gpu/kernels/integrate.wgsl (kernels.rs `patch_text`; each //#orig snippet must occur exactly once, so a changed
// upstream kernel is not silently patched). See rust/docs/kerb-investigation.md.
// Changed vs the JS kernel: the "already past the movement's entry point" branch (join the junction connector at the
// point nearest the body and add the heading difference to headingRel) tests where the car WAS on the edge at the start
// of the step (or where it arrived on the edge this step), as src/sim/vehicle.js integrate does (`sArrive > sIn + 0.3`,
// sArrive = s0 / the connector's sOut / 0 after a hop). The JS kernel tests the end-of-step s (`s > sIn + 0.3`), which
// is true for almost every car that crosses sIn at speed (u dt > 0.3 m above 6 m/s), so ordinary box entries were
// re-projected: headingRel picked up the connector-vs-edge heading difference (|dh| up to ~0.1 rad) with yaw rate 0,
// and a car holding steer 0 then drifted sideways at u sin(dh) until it hit the kerb.
//#orig sArr-init
      var tgtLane = vTarget(v);
//#end
//#patch sArr-init
      var tgtLane = vTarget(v);
      var sArr = s0;   // vehicle.js sArrive: where I was on the current edge at the start of the step (or where I arrived on it)
//#end
//#orig sArr-conn-exit
            s = movSOut(mu) + (s - end);
//#end
//#patch sArr-conn-exit
            s = movSOut(mu) + (s - end); sArr = movSOut(mu);
//#end
//#orig sArr-test
                if (s > sIn + 0.3 && P.hasLG != 0u) {
//#end
//#patch sArr-test
                if (sArr > sIn + 0.3 && P.hasLG != 0u) {
//#end
//#orig sArr-hop
        v.routePos = nxt; v.edge = en; v.seg = 0u;
//#end
//#patch sArr-hop
        v.routePos = nxt; v.edge = en; v.seg = 0u; sArr = 0.0;
//#end
