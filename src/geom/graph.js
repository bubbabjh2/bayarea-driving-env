// LaneGraph: the minimal directed-graph view the lane-geometry module works on.
// Built from one or two RoadSets (src/roadpack.js). Numbering matches src/sim/roadGraph.js
// fromRoadSet(): main-tier nodes and edges keep their ids, local-tier edges follow at
// mainEdgeCount + e, local nodes linked to the main tier (linkNodes) reuse the main index,
// the other local nodes are appended in order.

export const LANE_W = 3.7;
export const CLASSES_ALL = ['motorway', 'motorway_link', 'trunk', 'trunk_link', 'primary', 'primary_link',
  'secondary', 'secondary_link', 'tertiary', 'tertiary_link', 'unclassified', 'residential', 'living_street'];
export const FREEWAY = new Set(['motorway', 'motorway_link', 'trunk', 'trunk_link']);
/** Default lane width (m) per class index (CLASSES_ALL order) when the data has no edgeLaneWidth. */
export const LANE_WIDTH_CLASS = [3.7, 3.7, 3.7, 3.6, 3.6, 3.5, 3.5, 3.4, 3.3, 3.3, 3.1, 3.1, 3.0];

export class LaneGraph {
  /**
   * @param {object} a
   *  xyz Float32Array|Float64Array (3 per node), first Uint32Array, count Uint32Array (per edge,
   *  into poly), poly Uint32Array (node ids), lanes/opp Uint8Array, twin Int32Array (-1 none),
   *  cls Uint8Array (index into CLASSES_ALL), length Float32Array (data edge length, m),
   *  flags Uint32Array (pack flag bits, optional)
   */
  constructor(a) {
    Object.assign(this, a);
    this.nodeCount = a.xyz.length / 3;
    this.edgeCount = a.first.length;
    const N = this.nodeCount, E = this.edgeCount;
    const oc = new Uint32Array(N + 1), ic = new Uint32Array(N + 1);
    for (let e = 0; e < E; e++) { oc[this.from(e) + 1]++; ic[this.to(e) + 1]++; }
    for (let n = 0; n < N; n++) { oc[n + 1] += oc[n]; ic[n + 1] += ic[n]; }
    const of = oc.slice(0, N), inf = ic.slice(0, N);
    this.outStart = oc; this.inStart = ic;
    this.outList = new Uint32Array(E); this.inList = new Uint32Array(E);
    for (let e = 0; e < E; e++) { this.outList[of[this.from(e)]++] = e; this.inList[inf[this.to(e)]++] = e; }
  }
  from(e) { return this.poly[this.first[e]]; }
  to(e) { return this.poly[this.first[e] + this.count[e] - 1]; }
  nodesOf(e) { return this.poly.subarray(this.first[e], this.first[e] + this.count[e]); }
  outEdges(n) { return this.outList.subarray(this.outStart[n], this.outStart[n + 1]); }
  inEdges(n) { return this.inList.subarray(this.inStart[n], this.inStart[n + 1]); }
  className(e) { return CLASSES_ALL[this.cls[e]]; }
  isFreeway(e) { return this.cls[e] <= 3; }
  x(n) { return this.xyz[3 * n]; }
  y(n) { return this.xyz[3 * n + 1]; }
  z(n) { return this.xyz[3 * n + 2]; }
}

/** LaneGraph over main (+ optional local) RoadSets, numbered like the sim's fromRoadSet(). */
export function graphFromRoadSets(main, local = null) {
  const sets = local ? [main, local] : [main];
  let N = 0;
  const maps = [];
  for (let si = 0; si < sets.length; si++) {
    const rs = sets[si], map = new Int32Array(rs.nodeCount).fill(-1);
    if (si > 0 && rs.linkNodes) for (let k = 0; k < rs.linkNodes.length; k += 2) map[rs.linkNodes[k]] = rs.linkNodes[k + 1];
    for (let i = 0; i < rs.nodeCount; i++) if (map[i] < 0) map[i] = N++;
    maps.push(map);
  }
  const xyz = new Float64Array(3 * N);
  let E = 0, P = 0;
  for (let si = 0; si < sets.length; si++) {
    const rs = sets[si], map = maps[si];
    for (let i = 0; i < rs.nodeCount; i++) {
      const k = map[i];
      xyz[3 * k] = rs.x(i); xyz[3 * k + 1] = rs.y(i); xyz[3 * k + 2] = rs.z(i);
    }
    E += rs.edgeCount; P += rs.poly.length;
  }
  const first = new Uint32Array(E), count = new Uint32Array(E), poly = new Uint32Array(P);
  const lanes = new Uint8Array(E), opp = new Uint8Array(E), twin = new Int32Array(E), cls = new Uint8Array(E);
  const length = new Float32Array(E), flags = new Uint32Array(E), laneW = new Float32Array(E);
  let base = 0, p = 0;
  for (let si = 0; si < sets.length; si++) {
    const rs = sets[si], map = maps[si];
    for (let e = 0; e < rs.edgeCount; e++) {
      const g = base + e, nodes = rs.edgeNodes(e);
      first[g] = p; count[g] = nodes.length;
      for (let i = 0; i < nodes.length; i++) poly[p++] = map[nodes[i]];
      lanes[g] = rs.lanes(e); opp[g] = rs.oppLanes(e);
      const t = rs.twin(e); twin[g] = t >= 0 ? base + t : -1;
      cls[g] = rs.classId(e); length[g] = rs.edgeLength[e]; flags[g] = rs.flags(e);
      laneW[g] = rs.edgeLaneWidth ? rs.edgeLaneWidth[e] : LANE_WIDTH_CLASS[cls[g]];
    }
    base += rs.edgeCount;
  }
  const g = new LaneGraph({ xyz, first, count, poly, lanes, opp, twin, cls, length, flags, laneW });
  g.mainEdgeCount = main.edgeCount;
  g.tierNodeMap = maps;
  return g;
}
