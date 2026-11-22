// bayenv-gpu model patch of the junction perception (src/gpu/kernels/junction.wgsl jperceiveCar), with
// GpuOptions::box_standstill (the host prepends `const MF_BOX_STILL: bool`; applied before junction_order.wgsl, whose
// snippets it leaves intact). bayenv-sim's junction_p (junction.rs) skips, besides the JS kernel's filters:
//   swept path (1.): a driving body on my edge behind me, and a driving body on a connector when we are both on
//     connectors, both stuck (stopped > 20 s) and my slot is the lower (the mutual standstill rule pathBodies has);
//   conflicting movements (2.): the same mutual standstill when I am on a connector.
// Without them a pair stopped across each other in a box both held for good on the GPU (leader -3 / the other car,
// gap ~0) where the CPU let the lower slot creep out.
//#orig box-still-path
          if (!(vj.state == CRASHED || vj.speed < 2.0 || (otherBox && vj.speed < 7.0))) { continue; }
//#end
//#patch box-still-path
          if (!(vj.state == CRASHED || vj.speed < 2.0 || (otherBox && vj.speed < 7.0))) { continue; }
          if (MF_BOX_STILL && vj.state == DRIVING && vj.edge == e && vj.s < v.s) { continue; }
          if (MF_BOX_STILL && vj.state == DRIVING && vConn(v) >= 0 && vConn(vj) >= 0 && i < j && stuckJ(i) && stuckJ(j)) { continue; }
//#end
//#orig box-still-conf
      // it is itself waiting for me (I am its leader / the conflict it yields to): I go
//#end
//#patch box-still-conf
      // it is itself waiting for me (I am its leader / the conflict it yields to): I go
      let stillPair = MF_BOX_STILL && j != i && vjw.state == DRIVING && vConn(v) >= 0 && i < j && stuckJ(i) && stuckJ(j);
//#end
//#orig box-still-skip
      if (j != i && !waitsMe) {
        let vj = veh[j]; let tj = vj.s - movSIn(other);
//#end
//#patch box-still-skip
      if (j != i && !waitsMe && !stillPair) {
        let vj = veh[j]; let tj = vj.s - movSIn(other);
//#end
// MF_CAND_CAP (GpuOptions::junction_cand_cap): the swept-path (jperceiveCar, 32) and path-body (pathBodies, 48)
// candidate lists of the JS kernel drop every candidate past their size, where bayenv-sim has no limit. With the flag
// they hold 128; either way a candidate dropped for want of room (after the filters) is counted in GPU stat 62 / 63
// (the cap check moves after the filters: the kept list is the same).
//#orig cand-swept-decl
    var cand: array<u32, 32>; var nc = 0u;
//#end
//#patch cand-swept-decl
    var cand: array<u32, CAND_SWEPT>; var nc = 0u;
//#end
//#orig cand-swept-check
          if (j == i || nc >= 32u) { continue; }
//#end
//#patch cand-swept-check
          if (j == i) { continue; }
//#end
//#orig cand-swept-push
          if (dx * dx + dz * dz > r * r) { continue; }
          cand[nc] = j; nc = nc + 1u;
//#end
//#patch cand-swept-push
          if (dx * dx + dz * dz > r * r) { continue; }
          if (nc >= CAND_SWEPT) { statAdd(62u, 1u); continue; }
          cand[nc] = j; nc = nc + 1u;
//#end
//#orig cand-path-decl
  var cand: array<u32, 48>; var nc = 0u;
//#end
//#patch cand-path-decl
  var cand: array<u32, CAND_PATH>; var nc = 0u;
//#end
//#orig cand-path-check
        if (j == i || nc >= 48u) { continue; }
//#end
//#patch cand-path-check
        if (j == i) { continue; }
//#end
//#orig cand-path-push
        if (vj.state == DRIVING && i < j && stI && stuckJ(j)) { continue; }        // mutual standstill: the lower slot creeps
        cand[nc] = j; nc = nc + 1u;
//#end
//#patch cand-path-push
        if (vj.state == DRIVING && i < j && stI && stuckJ(j)) { continue; }        // mutual standstill: the lower slot creeps
        if (nc >= CAND_PATH) { statAdd(63u, 1u); continue; }
        cand[nc] = j; nc = nc + 1u;
//#end
