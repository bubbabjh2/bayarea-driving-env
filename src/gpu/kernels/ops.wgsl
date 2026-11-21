// CPU -> GPU command list, applied at the start of the first step of a frame.
// Each op = (op, slot, a, b). Codes: src/gpu/layout.js OP.
// #include common.wgsl

@group(1) @binding(0) var<storage, read_write> veh: array<Veh>;
@group(1) @binding(1) var<storage, read_write> par: array<VPar>;
@group(1) @binding(2) var<storage, read> ops: array<vec4u>;
@group(1) @binding(3) var<storage, read_write> pose: array<vec4f>;
@group(1) @binding(4) var<storage, read_write> ctl: array<u32>;   // controls.wgsl side record (word 11: pullUntil)

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let k = g.x;
  if (k >= P.opCount) { return; }
  let o = ops[k];
  let i = o.y;
  if (i >= P.cap) { return; }
  switch o.x {
    case 1u: { veh[i].state = FREE; pose[i] = vec4f(0.0, -1000.0, 0.0, 0.0); }        // FREE (scene cleared / towed)
    case 2u: { veh[i].state = o.z; }                                                    // SET_STATE
    case 3u: { veh[i].flags = veh[i].flags | o.z; }                                     // SET_FLAGS (shoulder, squeeze)
    case 4u: { veh[i].flags = veh[i].flags & ~o.z; }                                    // CLEAR_FLAGS
    case 7u: { veh[i].fuel = bitcast<f32>(o.z); }                                       // SET_FUEL
    case 9u: {
      // SET_ROUTE_REFUEL: a = routeOff, b = routeLen; routes[a-2] = stop index (route index whose
      // end node is the station), routes[a-1] = route length (f32 bits). route[0] must be the
      // car's current edge, else (it moved on meanwhile) the request is re-armed.
      if ((routes[o.z] & ROUTE_EDGE_MASK) == veh[i].edge && veh[i].state == DRIVING && veh[i].refuelState == 0u) {
        par[i].routeOff = o.z; par[i].routeLen = o.w; veh[i].routePos = 0u;
        veh[i].stopPos = bitcast<i32>(routes[o.z - 2u]);
        veh[i].routeTotal = bitcast<f32>(routes[o.z - 1u]);
        veh[i].routeDist = veh[i].s;
        veh[i].refuelState = 1u;
        veh[i].flags = (veh[i].flags | F_REFUEL) & ~F_FUELREQ;
      } else {
        veh[i].flags = veh[i].flags & ~F_FUELREQ;
      }
    }
    case 10u: {                                                                         // PUMPED: tank full, wait to re-enter
      if (veh[i].state == REFUELING) { veh[i].fuel = veh[i].tank; veh[i].refuelState = 4u; veh[i].eventTime = bitcast<f32>(o.z); }
    }
    case 11u: { veh[i].flags = veh[i].flags & ~F_FUELREQ; }                            // FUEL_PLAN_FAILED (retry at the next check)
    case 12u: {                                                                         // TOW_SHOULDER (layout v6): wreck onto the right shoulder
      if (veh[i].state == CRASHED) {
        let hw = 0.5 * par[i].width;
        // back in the in-edge frame (off any junction connector), >= 15 m from either node
        veh[i].taper = 0.0;
        veh[i].s = min(eLength(veh[i].edge) - 15.0, max(15.0, veh[i].s));
        veh[i].lat = -(hw + 0.3); veh[i].headingRel = 0.0; veh[i].speed = 0.0; veh[i].vLat = 0.0; veh[i].yawRate = 0.0;
        veh[i].lanes = derivedLanes(veh[i].edge, veh[i].s, veh[i].lat, hw, -1);
        veh[i].flags = veh[i].flags | F_NEW | F_SHOULDER;                               // newPose: pose + start pose
      }
    }
    case 14u: {
      // SET_ROUTE (re-route around a closure): a = routeOff, b = routeLen; routes[a - 1] = route length (f32 bits).
      // route[0] must still be the car's current edge, and it must not be committed to a junction connector
      // on a connector: only a route continuing with the connector's out-edge (missed turn re-plan)
      let cn = vConn(veh[i]);
      if ((routes[o.z] & ROUTE_EDGE_MASK) == veh[i].edge && veh[i].state == DRIVING && veh[i].refuelState == 0u && (cn < 0 || (o.w > 1u && (routes[o.z + 1u] & ROUTE_EDGE_MASK) == movOutEdge(u32(cn))))) {
        par[i].routeOff = o.z; par[i].routeLen = o.w; veh[i].routePos = 0u;
        veh[i].routeTotal = bitcast<f32>(routes[o.z - 1u]); veh[i].routeDist = veh[i].s;
        let v = veh[i]; let t = vTarget(v); if (t >= 0) { veh[i].lanes = packLanes(vLane(v), vLo(v), vHi(v), i32(vLane(v))); }
      }
    }
    case 13u: { ctl[i * CTL_W + 11u] = o.z; }                                             // SET_PULL: pullUntil (f32 bits), sim.js minor freeway crash
    default: {}
  }
}
