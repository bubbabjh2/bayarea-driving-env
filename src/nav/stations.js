// Gas stations: loading data/stations.json (format bay-stations/1) onto the
// nav graph and routing a trip through a station near its route.
//
//   loadStations(json, graph, router) -> [{id, node, x, z, pumps, tier, osmNode, nearInterchange}]
//     node = graph node index: (1) the station's tier-local `node` through
//     graph.tierNode(tier, node) (bay-roads JSON and binary packs), else (2) its
//     osmNode id (JSON graphs only), else (3) the nearest node to x,z (needs router).
//   routeViaStation(graph, router, fromNode, goalNode, stations, opts)
//     -> {route, station, direct, detourTime, detourLength, stopEdgePos} | null
//     Candidates = the `maxCandidates` stations closest to the direct route
//     (distance to the route's node chain); the min-time route via a candidate
//     wins. `route` is a Router result whose edges pass through station.node;
//     feed it to `new Guidance(graph, route.edges, { stops: [{ node: station.node,
//     type: 'refuel', station }] })` to get the "refuel" maneuver.
export function loadStations(json, graph, router) {
  const list = json.stations || json;
  const out = [];
  for (const s of list) {
    let node = s.tier != null && s.node != null && graph.tierNode ? graph.tierNode(s.tier, s.node) : -1;
    if ((node == null || node < 0) && s.osmNode != null && graph.nodeIdsAreOsm !== false) node = graph.indexOf(s.osmNode);
    if ((node == null || node < 0) && router) node = router.nearestNode(s.x, s.z, { requireOut: true, requireIn: true });
    if (node == null || node < 0) continue;
    out.push({ id: s.id, node, x: s.x, z: s.z, pumps: s.pumps ?? 4, tier: s.tier, osmNode: s.osmNode, nearInterchange: !!s.nearInterchange });
  }
  return out;
}

/** Squared distance from a point to the closest node of a route (brute force, routes are short lists). */
function distToRoute2(x, z, nx, nz, routeNodes, stride) {
  let best = Infinity;
  for (let i = 0; i < routeNodes.length; i += stride) {
    const dx = nx[routeNodes[i]] - x, dz = nz[routeNodes[i]] - z;
    const d2 = dx * dx + dz * dz;
    if (d2 < best) best = d2;
  }
  return best;
}

export function routeViaStation(graph, router, fromNode, goalNode, stations, { maxCandidates = 6, maxDetourTime = Infinity, direct = null } = {}) {
  direct = direct || router.route(fromNode, goalNode);
  if (!direct) return null;
  const nx = router.nx, nz = router.nz;
  const stride = Math.max(1, Math.floor(direct.nodes.length / 400));
  const scored = [];
  const inMain = router.largestSCC ? router.largestSCC().inLargest : null;
  for (const s of stations) {
    if (s.node === fromNode || s.node === goalNode) continue;
    if (inMain && !inMain[s.node]) continue; // a station on a stub could strand the trip
    scored.push({ s, d2: distToRoute2(s.x, s.z, nx, nz, direct.nodes, stride) });
  }
  scored.sort((a, b) => a.d2 - b.d2);
  let best = null;
  for (let k = 0; k < Math.min(maxCandidates, scored.length); k++) {
    const s = scored[k].s;
    const r = router.routeVia([fromNode, s.node, goalNode]);
    if (!r) continue;
    if (!best || r.time < best.route.time) best = { route: r, station: s };
  }
  if (!best) return null;
  const detourTime = best.route.time - direct.time;
  if (detourTime > maxDetourTime) return null;
  // position of the stop in the route: the first edge index whose `to` is the station node
  let stopEdgePos = -1;
  for (let i = 0; i < best.route.edges.length; i++) if (graph.edges[best.route.edges[i]].to === best.station.node) { stopEdgePos = i; break; }
  return { route: best.route, station: best.station, direct, detourTime, detourLength: best.route.length - direct.length, stopEdgePos };
}
