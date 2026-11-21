// RoadSet: typed-array view of a road tier, loaded from data/roads.bin +
// data/roads.meta.json (or converted from the readable data/roads.json).
// Layout is documented in data/roads.schema.md.  All arrays are little-endian
// and can be uploaded straight into GPU storage buffers.
//
//   const roads = await loadRoadPack('data/roads.meta.json');
//   roads.xyz          Float32Array  3*N   (x, y, z)  y = elevation
//   roads.edges        Uint32Array   8*E   [firstNode,nodeCount,lanesFwd,lanesBack,classId,speedCmps,flags,twin]
//   roads.poly         Uint32Array         node indices, travel order
//   roads.edgeLength   Float32Array  E     metres
//   roads.csrOffsets   Uint32Array   N+1
//   roads.csrEdges     Uint32Array   E     outgoing edge ids grouped by from-node
//   roads.linkNodes    Uint32Array   2*K   (localNodeIdx, mainNodeIdx)  local tier only
//   roads.bannedHops   Uint32Array   3*B   (inEdge, viaEdge, outEdge) prohibited lane paths, main tier only
//   roads.edgeLaneWidth Float32Array E    lane width (m) per edge (use src/geom laneWidth(e, s))

export const EDGE_STRIDE = 8;
export const F = { firstNode: 0, nodeCount: 1, lanesFwd: 2, lanesBack: 3, classId: 4, speedCmps: 5, flags: 6, twin: 7 };
export const FLAG = { ONEWAY: 1, BRIDGE: 2, TUNNEL: 4, LANES_FALLBACK: 8, SPEED_DEFAULT: 16, LINK: 32,
  LANES_CLAMPED: 64, LANES_SMOOTHED: 128, AUX_LANE: 256, LANE_DROP: 512 };
export const NO_TWIN = 0xFFFFFFFF;
export const CLASSES_ALL = ['motorway', 'motorway_link', 'trunk', 'trunk_link', 'primary', 'primary_link',
  'secondary', 'secondary_link', 'tertiary', 'tertiary_link', 'unclassified', 'residential', 'living_street'];

const TYPES = { Float32: Float32Array, Uint32: Uint32Array, Uint16: Uint16Array, Uint8: Uint8Array };

export class RoadSet {
  constructor(meta, arrays) {
    this.meta = meta;
    this.classesAll = meta.classesAll || CLASSES_ALL;
    this.laneWidth = meta.laneWidth || 3.7;
    Object.assign(this, arrays);
    this.nodeCount = this.xyz.length / 3;
    this.edgeCount = this.edges.length / EDGE_STRIDE;
    this._self = this;   // views made with Object.create(roadSet) still reach the real set (geometry cache)
  }
  // ---- edge accessors
  firstNode(e) { return this.edges[e * EDGE_STRIDE + F.firstNode]; }
  nodeCountOf(e) { return this.edges[e * EDGE_STRIDE + F.nodeCount]; }
  lanes(e) { return this.edges[e * EDGE_STRIDE + F.lanesFwd]; }
  oppLanes(e) { return this.edges[e * EDGE_STRIDE + F.lanesBack]; }
  classId(e) { return this.edges[e * EDGE_STRIDE + F.classId]; }
  className(e) { return this.classesAll[this.classId(e)]; }
  speedMps(e) { return this.edges[e * EDGE_STRIDE + F.speedCmps] / 100; }
  flags(e) { return this.edges[e * EDGE_STRIDE + F.flags]; }
  has(e, flag) { return (this.flags(e) & flag) !== 0; }
  twin(e) { const t = this.edges[e * EDGE_STRIDE + F.twin]; return t === NO_TWIN ? -1 : t; }
  from(e) { return this.poly[this.firstNode(e)]; }
  to(e) { const f = this.firstNode(e); return this.poly[f + this.nodeCountOf(e) - 1]; }
  /** Uint32Array subarray of node indices in travel order (no copy). */
  edgeNodes(e) { const f = this.firstNode(e); return this.poly.subarray(f, f + this.nodeCountOf(e)); }
  width(e) { return (this.lanes(e) + this.oppLanes(e)) * this.laneWidth; }
  laneOffsets(e) {
    const n = this.lanes(e), o = this.oppLanes(e), out = new Float32Array(n);
    for (let k = 0; k < n; k++) out[k] = ((o - n) / 2 + k + 0.5) * this.laneWidth;
    return out;
  }
  /** true for one edge of each physical road segment (skip twins). */
  isPrimaryDir(e) { const t = this.twin(e); return t < 0 || e < t; }
  outEdges(n) { return this.csrEdges.subarray(this.csrOffsets[n], this.csrOffsets[n + 1]); }
  // ---- node accessors
  x(n) { return this.xyz[n * 3]; }
  y(n) { return this.xyz[n * 3 + 1]; }
  z(n) { return this.xyz[n * 3 + 2]; }
  /** Position/height s metres along edge e (clamped). Returns {x,y,z}. */
  pointAlong(e, s) {
    const nodes = this.edgeNodes(e);
    let acc = 0;
    for (let i = 1; i < nodes.length; i++) {
      const a = nodes[i - 1], b = nodes[i];
      const dx = this.x(b) - this.x(a), dz = this.z(b) - this.z(a);
      const L = Math.hypot(dx, dz);
      if (s <= acc + L || i === nodes.length - 1) {
        const t = L > 0 ? Math.max(0, Math.min(1, (s - acc) / L)) : 0;
        return { x: this.x(a) + dx * t, y: this.y(a) + (this.y(b) - this.y(a)) * t, z: this.z(a) + dz * t };
      }
      acc += L;
    }
    const n0 = nodes[0];
    return { x: this.x(n0), y: this.y(n0), z: this.z(n0) };
  }
}

/** Load the binary pack described by a *.meta.json file (bin path is relative to it). */
export async function loadRoadPack(metaUrl) {
  const r = await fetch(metaUrl);
  if (!r.ok) throw new Error(`${metaUrl}: HTTP ${r.status}`);
  const meta = await r.json();
  const binUrl = metaUrl.replace(/[^/]*$/, '') + meta.file;
  const rb = await fetch(binUrl);
  if (!rb.ok) throw new Error(`${binUrl}: HTTP ${rb.status}`);
  const buf = await rb.arrayBuffer();
  const arrays = {};
  for (const [name, s] of Object.entries(meta.sections)) {
    const T = TYPES[s.type];
    arrays[sectionKey(name)] = new T(buf, s.byteOffset, s.length);
  }
  const rs = new RoadSet(meta, arrays);
  // remember the tiers so the lane geometry is built on the merged graph (like the sim)
  if (meta.tier === 'main') lastMain = rs;
  else if (meta.tier === 'local' && lastMain) { rs._mainPartner = lastMain; lastMain._localPartner = rs; }
  return rs;
}
let lastMain = null;

function sectionKey(name) { return name === 'nodeXYZ' ? 'xyz' : name; }

/** Convert the readable data/roads.json document into a RoadSet (same layout). */
export function roadSetFromJSON(doc) {
  const N = doc.nodes.length, E = doc.edges.length;
  const classesAll = CLASSES_ALL;
  const xyz = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    const n = doc.nodes[i];
    xyz[i * 3] = n.x; xyz[i * 3 + 1] = n.y || 0; xyz[i * 3 + 2] = n.z;
  }
  let polyLen = 0;
  for (const e of doc.edges) polyLen += e.nodes.length;
  const edges = new Uint32Array(E * EDGE_STRIDE);
  const poly = new Uint32Array(polyLen);
  const edgeLength = new Float32Array(E);
  const outCount = new Uint32Array(N + 1);
  let p = 0;
  const defSpeed = doc.meta.defaultSpeedKph || {};
  for (let i = 0; i < E; i++) {
    const e = doc.edges[i];
    const o = i * EDGE_STRIDE;
    edges[o + F.firstNode] = p;
    edges[o + F.nodeCount] = e.nodes.length;
    for (const n of e.nodes) poly[p++] = n;
    edges[o + F.lanesFwd] = e.lanes;
    edges[o + F.lanesBack] = e.oppLanes;
    edges[o + F.classId] = Math.max(0, classesAll.indexOf(e.class));
    const kph = e.maxspeed ?? defSpeed[e.class] ?? 50;
    edges[o + F.speedCmps] = Math.round((kph / 3.6) * 100);
    let fl = 0;
    if (e.oneway) fl |= FLAG.ONEWAY;
    if (e.bridge) fl |= FLAG.BRIDGE;
    if (e.tunnel) fl |= FLAG.TUNNEL;
    if (e.lanesFallback) fl |= FLAG.LANES_FALLBACK;
    if (e.maxspeed == null) fl |= FLAG.SPEED_DEFAULT;
    if (e.class.endsWith('_link')) fl |= FLAG.LINK;
    edges[o + F.flags] = fl;
    edges[o + F.twin] = e.twin >= 0 ? e.twin : NO_TWIN;
    edgeLength[i] = e.length;
    outCount[e.from + 1]++;
  }
  const csrOffsets = new Uint32Array(N + 1);
  for (let i = 0; i < N; i++) csrOffsets[i + 1] = csrOffsets[i] + outCount[i + 1];
  const fill = csrOffsets.slice();
  const csrEdges = new Uint32Array(E);
  for (let i = 0; i < E; i++) csrEdges[fill[doc.edges[i].from]++] = i;
  const extra = doc.bannedHops ? { bannedHops: Uint32Array.from(doc.bannedHops.flat()) } : {};
  if (doc.edges.length && doc.edges[0].laneWidth !== undefined) extra.edgeLaneWidth = Float32Array.from(doc.edges, (e) => e.laneWidth);
  return new RoadSet(Object.assign({ classesAll }, doc.meta), { xyz, edges, poly, edgeLength, csrOffsets, csrEdges, ...extra });
}
