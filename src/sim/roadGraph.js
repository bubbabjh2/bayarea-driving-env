// Directed road graph used by the simulation (SimGraph).
//
// Sources:
//   * `bay-roads/1` (data/roads.json + data/roads-local.json, see data/roads.schema.md):
//     edges are already directed (one per travel direction), `from`/`to`/`nodes` are
//     node indices, `twin` is the opposite direction, `lanes` is per direction,
//     `laneOffsets` are lane centrelines (positive = right of travel, [0] = leftmost),
//     `maxspeed` is km/h. Tiers are merged by OSM node `id`. Edge indices of the
//     main tier are preserved (local-tier edges are appended).
//   * generic {nodes:[{id,x,z}], edges:[{nodes:[ids], highway, oneway, lanes, ...}]}
//     (used by the synthetic test graph): split at shared interior nodes, two-way
//     edges become two directed edges.
//   * fromNavGraph(): adapt a src/nav/graph.js Graph (edge ids preserved).
//
// Every SimGraph is also duck-type compatible with the nav Graph for the brain's
// Guidance/Driver: edges carry `id, points (interleaved), cum, length, speed, lanes,
// highway, isLink, rank, name, ref, from, to, headingStart, headingEnd, laneLat`
// and the graph has `pointOnEdge(e, d)`, `laneWidth`, `laneLateral(e, k)`.
//
// Elevation matches src/roads.js: tunnels 0, bridges 15 m (+10 m per extra layer),
// elevated non-bridge roads 8 m per layer; node height is the minimum of its edges
// and bridge ends ramp to it over 140 m (per-vertex `ys`).

export const HIGHWAY_RANK = {
  motorway: 7, motorway_link: 6, trunk: 6, trunk_link: 5, primary: 5, primary_link: 4,
  secondary: 4, secondary_link: 3, tertiary: 3, tertiary_link: 2, unclassified: 2,
  residential: 1, living_street: 0, service: 0,
};
/** Same ordering as src/nav/graph.js CLASS_RANK (used by the brain's guidance). */
export const CLASS_RANK = {
  motorway: 9, trunk: 8, motorway_link: 7, trunk_link: 6, primary: 5, primary_link: 4,
  secondary: 3, secondary_link: 3, tertiary: 2, tertiary_link: 2, unclassified: 1,
  residential: 1, living_street: 0, service: 0,
};

export const DEFAULT_SPEED_MPH = {
  motorway: 65, motorway_link: 45, trunk: 55, trunk_link: 40, primary: 40, primary_link: 30,
  secondary: 35, secondary_link: 25, tertiary: 30, tertiary_link: 25, unclassified: 25,
  residential: 25, living_street: 15, service: 15,
};
export const DEFAULT_LANES = {
  motorway: 4, motorway_link: 1, trunk: 2, trunk_link: 1, primary: 2, primary_link: 1,
  secondary: 2, secondary_link: 1, tertiary: 1, tertiary_link: 1, unclassified: 1,
  residential: 1, living_street: 1, service: 1,
};

export const LANE_WIDTH = 3.7;   // synthetic graphs / fallback only: real edges use laneW(e, s)
/** Lane width (m) of edge e at s: src/geom per-edge width (blended near strict nodes), else 3.7. */
export function laneW(e, s) { return e.LG ? e.LG.laneWidth(e.idx, s < 0 ? 0 : s > e.length ? e.length : s) : LANE_WIDTH; }
export const RAMP_M = 140;
const MPH = 0.44704;

export function parseMaxspeed(v, highway) {
  if (v == null || v === '') return (DEFAULT_SPEED_MPH[highway] ?? 30) * MPH;
  if (typeof v === 'number') return v * MPH; // generic format: bare number = mph
  const s = String(v).trim().toLowerCase();
  const m = s.match(/([\d.]+)\s*(mph|km\/h|kmh|kph)?/);
  if (!m) return (DEFAULT_SPEED_MPH[highway] ?? 30) * MPH;
  const n = parseFloat(m[1]);
  if (m[2] && m[2] !== 'mph') return n / 3.6;
  return n * MPH;
}

/** Elevation rule shared with src/roads.js edgeElevation(). */
export function edgeElevation(e) {
  if (e.tunnel) return 0;
  if (e.bridge) return 15 + Math.max(0, (e.layer | 0) - 1) * 10;
  if ((e.layer | 0) > 0) return e.layer * 8;
  return 0;
}

function isOneway(v) {
  return v === true || v === 1 || v === 'yes' || v === 'true' || v === '1' || v === -1 || v === '-1';
}
function isBayRoads(doc) {
  if (doc?.meta?.format && String(doc.meta.format).startsWith('bay-roads')) return true;
  const e = doc?.edges?.[0];
  return !!(e && typeof e.from === 'number' && typeof e.to === 'number' && e.class && Array.isArray(e.nodes));
}

/** Build a SimGraph from one or more road json documents. */
export function buildGraph(datasets, opts = {}) {
  const list = Array.isArray(datasets) ? datasets : [datasets];
  const bay = list.length && list.every(isBayRoads);
  return bay ? buildBayRoads(list, opts) : buildGeneric(list, opts);
}

// --------------------------------------------------------------------- shared
function makeEdge(g, idx, from, to, px, pz, props) {
  const n = px.length;
  const cum = new Float64Array(n);
  let L = 0;
  for (let i = 1; i < n; i++) { L += Math.hypot(px[i] - px[i - 1], pz[i] - pz[i - 1]); cum[i] = L; }
  const points = new Float64Array(n * 2);
  for (let i = 0; i < n; i++) { points[2 * i] = px[i]; points[2 * i + 1] = pz[i]; }
  const highway = props.highway;
  const lanes = Math.max(1, Math.min(8, props.lanes | 0));
  const e = {
    idx, id: idx, from, to, px, pz, cum, points, length: Math.max(L, 0.01),
    lanes, oneway: !!props.oneway, reverse: -1,
    speedLimit: props.speedLimit, speed: props.speedLimit,
    highway, rank: HIGHWAY_RANK[highway] ?? 1, navRank: CLASS_RANK[highway] ?? 1, isLink: /_link$/.test(highway),
    bridge: !!props.bridge, tunnel: !!props.tunnel, layer: props.layer | 0,
    y: 0, ys: null, laneOff: null, laneLat: null,
    name: props.name || '', ref: props.ref || '',
    headingStart: Math.atan2(pz[1] - pz[0], px[1] - px[0]),
    headingEnd: Math.atan2(pz[n - 1] - pz[n - 2], px[n - 1] - px[n - 2]),
    nodeIdx: props.nodeIdx || null, meta: props.meta || null, srcEdgeId: props.srcEdgeId ?? idx, tier: props.tier || 'main',
  };
  e.y = edgeElevation(e);
  g.edges[idx] = e;
  return e;
}

function finishGraph(g, opts) {
  const { edges } = g, N = g.nodeCount, E = edges.length;
  g.edgeCount = E;
  // lane offsets (right-positive, lane 0 = rightmost) and brain-side laneLat (+left)
  for (const e of edges) {
    if (!e.laneOff) {
      const off = new Float32Array(e.lanes);
      for (let l = 0; l < e.lanes; l++) {
        const fromRight = l + 0.5; // lane centre distance from the right kerb, in lanes
        off[l] = e.reverse >= 0 ? (e.lanes - fromRight) * LANE_WIDTH : (e.lanes / 2 - fromRight) * LANE_WIDTH;
      }
      e.laneOff = off;
    }
    const lat = new Float32Array(e.lanes);
    for (let l = 0; l < e.lanes; l++) lat[l] = -e.laneOff[l];
    e.laneLat = lat;
    e.rightEdge = e.laneOff[0] + LANE_WIDTH / 2; // right boundary of the carriageway, m right of the polyline
  }
  // CSR adjacency
  const outStart = new Int32Array(N + 1), inStart = new Int32Array(N + 1);
  for (const e of edges) { outStart[e.from + 1]++; inStart[e.to + 1]++; }
  for (let i = 0; i < N; i++) { outStart[i + 1] += outStart[i]; inStart[i + 1] += inStart[i]; }
  const outEdges = new Int32Array(E), inEdges = new Int32Array(E);
  const oc = outStart.slice(0, N), ic = inStart.slice(0, N);
  for (const e of edges) { outEdges[oc[e.from]++] = e.idx; inEdges[ic[e.to]++] = e.idx; }
  Object.assign(g, { outStart, outEdges, inStart, inEdges });
  // node ranks
  const nodeRank = new Int8Array(N).fill(-1), nodeLowRank = new Int8Array(N).fill(127);
  for (const e of edges) for (const k of [e.from, e.to]) {
    if (e.rank > nodeRank[k]) nodeRank[k] = e.rank;
    if (e.rank < nodeLowRank[k]) nodeLowRank[k] = e.rank;
  }
  g.nodeRank = nodeRank; g.nodeLowRank = nodeLowRank;
  // elevation with ramps (src/roads.js prepareRoads)
  const elevation = opts.elevation;
  const baked = g.nodeY && g.nodeY.length === N ? g.nodeY : null; // data with y per node (bay-roads/2, roads.bin)
  const nodeElev = baked || new Float32Array(N).fill(Infinity);
  if (!baked) {
    for (const e of edges) { if (elevation) e.y = elevation(e); nodeElev[e.from] = Math.min(nodeElev[e.from], e.y); nodeElev[e.to] = Math.min(nodeElev[e.to], e.y); }
    for (let i = 0; i < N; i++) if (nodeElev[i] === Infinity) nodeElev[i] = 0;
  }
  g.nodeY = nodeElev;
  for (const e of edges) {
    if (baked && e.nodeIdx) {
      const n = e.nodeIdx.length; let flat = true; const y0 = baked[e.nodeIdx[0]];
      for (let i = 1; i < n; i++) if (baked[e.nodeIdx[i]] !== y0) { flat = false; break; }
      e.y = y0;
      if (!flat) { const ys = new Float32Array(n); for (let i = 0; i < n; i++) ys[i] = baked[e.nodeIdx[i]]; e.ys = ys; }
      continue;
    }
    const y0 = nodeElev[e.from], y1 = nodeElev[e.to], Y = e.y;
    if (y0 === Y && y1 === Y) continue;
    const n = e.px.length, L = e.length || 1, ys = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const fa = Math.min(1, e.cum[i] / RAMP_M), fb = Math.min(1, (L - e.cum[i]) / RAMP_M);
      ys[i] = Math.min(y0 + (Y - y0) * fa, y1 + (Y - y1) * fb);
    }
    e.ys = ys;
  }
  // bounds, maxSpeed
  let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity, maxSpeed = 0;
  for (let i = 0; i < N; i++) {
    const x = g.nx[i], z = g.nz[i];
    if (x < minX) minX = x; if (x > maxX) maxX = x; if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
  }
  for (const e of edges) if (e.speedLimit > maxSpeed) maxSpeed = e.speedLimit;
  g.bounds = N ? { minX, maxX, minZ, maxZ } : { minX: 0, maxX: 0, minZ: 0, maxZ: 0 };
  g.maxSpeed = maxSpeed;
  g.laneWidth = LANE_WIDTH;
  g.outDegree = (n) => outStart[n + 1] - outStart[n];
  g.inDegree = (n) => inStart[n + 1] - inStart[n];
  g.edgeBetween = (a, b) => { for (let i = outStart[a]; i < outStart[a + 1]; i++) if (edges[outEdges[i]].to === b) return outEdges[i]; return -1; };
  g.nearestNode = makeNearestNode(g);
  // nav Graph duck-typing for the brain
  g.pointOnEdge = (e, d) => { const o = edgePoint(e, d, 0, { x: 0, z: 0, y: 0, hx: 1, hz: 0, seg: 0 }, 0); return [o.x, o.z, Math.atan2(o.hz, o.hx)]; };
  g.laneLateral = (e, k) => e.laneLat[Math.min(Math.max(k, 0), e.lanes - 1)];
  g.nodeAt = (i) => ({ id: g.nodeIds ? g.nodeIds[i] : i, x: g.nx[i], z: g.nz[i], y: nodeElev[i] });
  g.nodes = new Proxy({}, { get: (t, k) => (k === 'length' ? N : typeof k === 'string' && /^\d+$/.test(k) ? g.nodeAt(+k) : undefined) });
  Object.defineProperty(g, 'out', { get() { if (!this._out) { this._out = []; for (let i = 0; i < N; i++) this._out.push(Array.from(outEdges.subarray(outStart[i], outStart[i + 1]))); } return this._out; }, configurable: true });
  Object.defineProperty(g, 'in', { get() { if (!this._in) { this._in = []; for (let i = 0; i < N; i++) this._in.push(Array.from(inEdges.subarray(inStart[i], inStart[i + 1]))); } return this._in; }, configurable: true });
  g.nodeCount = N;
  g.indexOf = (id) => g.idToIdx.get(id);
  return g;
}

// --------------------------------------------------------------------- bay-roads/1
function buildBayRoads(docs, opts) {
  const idToIdx = new Map();
  const nxA = [], nzA = [], nyA = [], ids = [];
  let hasY = false;
  const localMaps = [];
  for (const d of docs) {
    const map = new Int32Array(d.nodes.length);
    for (let i = 0; i < d.nodes.length; i++) {
      const n = d.nodes[i];
      let k = idToIdx.get(n.id);
      if (k === undefined) { k = nxA.length; idToIdx.set(n.id, k); nxA.push(n.x); nzA.push(n.z); nyA.push(n.y || 0); if (n.y != null) hasY = true; ids.push(n.id); }
      map[i] = k;
    }
    localMaps.push(map);
  }
  const g = { nodeCount: nxA.length, nx: Float64Array.from(nxA), nz: Float64Array.from(nzA), idToIdx, nodeIds: ids, edges: [], meta: docs[0].meta || null };
  if (hasY) g.nodeY = Float32Array.from(nyA);
  let base = 0;
  for (let di = 0; di < docs.length; di++) {
    const d = docs[di], map = localMaps[di], meta = d.meta || {};
    const tier = meta.tier || (di === 0 ? 'main' : 'local');
    const defKph = meta.defaultSpeedKph || {};
    const edgeBase = base;
    for (let k = 0; k < d.edges.length; k++) {
      const raw = d.edges[k];
      const nodes = raw.nodes || [raw.from, raw.to];
      const px = new Float64Array(nodes.length), pz = new Float64Array(nodes.length);
      const nodeIdx = new Int32Array(nodes.length);
      for (let i = 0; i < nodes.length; i++) { const gi = map[nodes[i]]; nodeIdx[i] = gi; px[i] = nxA[gi]; pz[i] = nzA[gi]; }
      const highway = raw.class || raw.highway || 'unclassified';
      let kph = raw.maxspeed;
      if (kph == null) kph = defKph[highway];
      const speedLimit = kph != null ? +kph / 3.6 : (DEFAULT_SPEED_MPH[highway] ?? 25) * MPH;
      const e = makeEdge(g, edgeBase + (raw.id ?? k), map[raw.from], map[raw.to], px, pz, {
        highway, lanes: raw.lanes || DEFAULT_LANES[highway] || 1, oneway: !!raw.oneway, speedLimit,
        bridge: raw.bridge, tunnel: raw.tunnel, layer: raw.layer, name: raw.name, ref: raw.ref, nodeIdx, meta: raw, srcEdgeId: raw.id ?? k, tier,
      });
      if (Array.isArray(raw.laneOffsets) && raw.laneOffsets.length === e.lanes) {
        const off = new Float32Array(e.lanes);
        for (let l = 0; l < e.lanes; l++) off[l] = raw.laneOffsets[e.lanes - 1 - l]; // lane 0 = rightmost
        e.laneOff = off;
      }
      e.twinLocal = raw.twin;
      e.way = raw.way; e.oppLanes = raw.oppLanes || 0;
    }
    for (let k = edgeBase; k < g.edges.length; k++) {
      const e = g.edges[k]; if (!e) continue;
      if (e.twinLocal != null && e.twinLocal >= 0) e.reverse = edgeBase + e.twinLocal;
      delete e.twinLocal;
    }
    base = g.edges.length;
  }
  // defensive: fill holes if ids were sparse
  for (let i = 0; i < g.edges.length; i++) if (!g.edges[i]) throw new Error(`bay-roads: missing edge id ${i}`);
  return finishGraph(g, opts);
}

// --------------------------------------------------------------------- generic
function buildGeneric(list, opts) {
  const idToIdx = new Map();
  const nxA = [], nzA = [], ids = [];
  for (const d of list) for (const n of d.nodes || []) {
    if (idToIdx.has(n.id)) continue;
    idToIdx.set(n.id, nxA.length); nxA.push(n.x); nzA.push(n.z); ids.push(n.id);
  }
  const useCount = new Int32Array(nxA.length);
  const raws = [];
  for (const d of list) for (const e of d.edges || []) {
    const idx = [];
    for (const id of e.nodes || []) { const k = idToIdx.get(id); if (k === undefined) { idx.length = 0; break; } idx.push(k); }
    if (idx.length < 2) continue;
    raws.push({ meta: e, idx });
    for (const k of idx) useCount[k]++;
  }
  const g = { nodeCount: nxA.length, nx: Float64Array.from(nxA), nz: Float64Array.from(nzA), idToIdx, nodeIds: ids, edges: [], meta: list[0].meta || null };
  const push = (meta, pts, reverse) => {
    const n = pts.length, px = new Float64Array(n), pz = new Float64Array(n), nodeIdx = new Int32Array(n);
    for (let i = 0; i < n; i++) { const k = reverse ? pts[n - 1 - i] : pts[i]; nodeIdx[i] = k; px[i] = nxA[k]; pz[i] = nzA[k]; }
    const highway = meta.highway || 'residential';
    let lanes = Number(meta.lanes); if (!Number.isFinite(lanes) || lanes < 1) lanes = DEFAULT_LANES[highway] ?? 1;
    return makeEdge(g, g.edges.length, nodeIdx[0], nodeIdx[n - 1], px, pz, {
      highway, lanes, oneway: isOneway(meta.oneway), speedLimit: parseMaxspeed(meta.maxspeed, highway),
      bridge: meta.bridge, tunnel: meta.tunnel, layer: meta.layer, name: meta.name, ref: meta.ref, nodeIdx, meta,
    });
  };
  for (const { meta, idx } of raws) {
    let start = 0;
    for (let i = 1; i < idx.length; i++) {
      const last = i === idx.length - 1;
      if (!last && useCount[idx[i]] <= 1) continue;
      const pts = idx.slice(start, i + 1);
      const ow = meta.oneway;
      if (isOneway(ow) && (ow === -1 || ow === '-1')) push(meta, pts, true);
      else if (isOneway(ow)) push(meta, pts, false);
      else { const a = push(meta, pts, false), b = push(meta, pts, true); a.reverse = b.idx; b.reverse = a.idx; }
      start = i;
    }
  }
  return finishGraph(g, opts);
}

// --------------------------------------------------------------------- nav adapter
/** Adapt a src/nav/graph.js Graph; edge ids are preserved. */
export function fromNavGraph(ng, opts = {}) {
  const N = ng.nodes.length;
  const nx = new Float64Array(N), nz = new Float64Array(N), ids = new Array(N);
  const idToIdx = new Map();
  for (let i = 0; i < N; i++) { nx[i] = ng.nodes[i].x; nz[i] = ng.nodes[i].z; ids[i] = ng.nodes[i].id; idToIdx.set(ng.nodes[i].id, i); }
  const g = { nodeCount: N, nx, nz, idToIdx, nodeIds: ids, edges: new Array(ng.edges.length), meta: null, nav: ng };
  const bySrc = new Map();
  for (const e of ng.edges) {
    const n = e.points.length / 2, px = new Float64Array(n), pz = new Float64Array(n);
    for (let i = 0; i < n; i++) { px[i] = e.points[2 * i]; pz[i] = e.points[2 * i + 1]; }
    const ed = makeEdge(g, e.id, e.from, e.to, px, pz, {
      highway: e.highway || 'residential', lanes: e.lanes, oneway: !!e.oneway, speedLimit: e.speed,
      bridge: e.bridge, tunnel: e.tunnel, layer: e.layer, name: e.name, ref: e.ref, nodeIdx: e.nodeIdx, meta: e, srcEdgeId: e.srcEdgeId, tier: e.tier,
    });
    if (e.laneLat && e.laneLat.length === ed.lanes) { ed.laneOff = Float32Array.from(e.laneLat, v => -v); }
    if (e.tier != null && e.srcEdgeId != null) bySrc.set(`${e.tier}:${e.srcEdgeId}:${e.from}:${e.to}`, ed);
  }
  for (const e of ng.edges) {
    const ed = g.edges[e.id];
    if (e.twinSrc != null && e.twinSrc >= 0) { const t = bySrc.get(`${e.tier}:${e.twinSrc}:${e.to}:${e.from}`); if (t) { ed.reverse = t.idx; t.reverse = ed.idx; } }
    else if (!e.oneway && ed.reverse < 0) { for (const k of ng.out[e.to]) { const o = ng.edges[k]; if (o.to === e.from && o.srcId === e.srcId) { ed.reverse = o.id; g.edges[o.id].reverse = ed.idx; break; } } }
  }
  return finishGraph(g, opts);
}

// --------------------------------------------------------------------- RoadSet (roads.bin)
/**
 * Build from the world's RoadSet (src/roadpack.js: loadRoadPack / roadSetFromJSON).
 * Node y is baked in. A second (local tier) RoadSet can be merged through its
 * linkNodes (local node -> main node) table. Edge ids of the main tier are kept.
 */
export function fromRoadSet(main, local = null, opts = {}) {
  const sets = local ? [main, local] : [main];
  let N = 0;
  const maps = [];
  for (let si = 0; si < sets.length; si++) {
    const rs = sets[si], map = new Int32Array(rs.nodeCount).fill(-1);
    if (si > 0 && rs.linkNodes) for (let k = 0; k < rs.linkNodes.length; k += 2) map[rs.linkNodes[k]] = rs.linkNodes[k + 1];
    for (let i = 0; i < rs.nodeCount; i++) if (map[i] < 0) map[i] = N++;
    maps.push(map);
  }
  const nx = new Float64Array(N), nz = new Float64Array(N), ny = new Float32Array(N);
  for (let si = 0; si < sets.length; si++) {
    const rs = sets[si], map = maps[si];
    for (let i = 0; i < rs.nodeCount; i++) { const k = map[i]; nx[k] = rs.x(i); nz[k] = rs.z(i); ny[k] = rs.y(i); }
  }
  const idToIdx = new Map(); for (let i = 0; i < N; i++) idToIdx.set(i, i);
  const g = { nodeCount: N, nx, nz, nodeY: ny, idToIdx, nodeIds: null, edges: [], meta: main.meta || null, roadSet: main, tierNodeMap: maps };
  const classes = main.classesAll || ['motorway', 'motorway_link', 'trunk', 'trunk_link', 'primary', 'primary_link', 'secondary', 'secondary_link', 'tertiary', 'tertiary_link', 'unclassified', 'residential', 'living_street'];
  let base = 0;
  for (let si = 0; si < sets.length; si++) {
    const rs = sets[si], map = maps[si], tier = si === 0 ? 'main' : 'local';
    for (let e = 0; e < rs.edgeCount; e++) {
      const nodes = rs.edgeNodes(e), n = nodes.length;
      const px = new Float64Array(n), pz = new Float64Array(n), nodeIdx = new Int32Array(n);
      for (let i = 0; i < n; i++) { const k = map[nodes[i]]; nodeIdx[i] = k; px[i] = nx[k]; pz[i] = nz[k]; }
      const flags = rs.flags(e);
      const highway = classes[rs.classId(e)] || 'unclassified';
      const ed = makeEdge(g, base + e, nodeIdx[0], nodeIdx[n - 1], px, pz, {
        highway, lanes: rs.lanes(e), oneway: (flags & 1) !== 0, speedLimit: rs.speedMps(e),
        bridge: (flags & 2) !== 0, tunnel: (flags & 4) !== 0, layer: 0, name: '', ref: '', nodeIdx, meta: null, srcEdgeId: e, tier,
      });
      const offs = rs.laneOffsets(e);
      const off = new Float32Array(ed.lanes); for (let l = 0; l < ed.lanes; l++) off[l] = offs[ed.lanes - 1 - l];
      ed.laneOff = off; ed.oppLanes = rs.oppLanes(e);
      const t = rs.twin(e); if (t >= 0) ed.reverse = base + t;
    }
    base += rs.edgeCount;
  }
  // banned lane paths (main tier: in -> via -> out needs too many lane changes inside a short via)
  if (main.bannedHops && main.bannedHops.length) {
    const bh = new Map();
    for (let k = 0; k + 2 < main.bannedHops.length; k += 3) { const key = main.bannedHops[k] * 4194304 + main.bannedHops[k + 1]; let l = bh.get(key); if (!l) bh.set(key, (l = [])); l.push(main.bannedHops[k + 2]); }
    g.bannedHops = bh; g.bannedHopCount = main.bannedHops.length / 3;
  }
  return finishGraph(g, opts);
}

// --------------------------------------------------------------------- nearest node
function makeNearestNode(g) {
  const { nx, nz, nodeCount: N, bounds } = g;
  if (N === 0) return () => -1;
  const cell = 250;
  const cols = Math.max(1, Math.ceil((bounds.maxX - bounds.minX) / cell) + 1);
  const rows = Math.max(1, Math.ceil((bounds.maxZ - bounds.minZ) / cell) + 1);
  const start = new Int32Array(cols * rows + 1);
  const cellOf = (x, z) => {
    const cx = Math.min(cols - 1, Math.max(0, Math.floor((x - bounds.minX) / cell)));
    const cz = Math.min(rows - 1, Math.max(0, Math.floor((z - bounds.minZ) / cell)));
    return cz * cols + cx;
  };
  for (let i = 0; i < N; i++) start[cellOf(nx[i], nz[i]) + 1]++;
  for (let i = 0; i < cols * rows; i++) start[i + 1] += start[i];
  const items = new Int32Array(N);
  const fill = start.slice(0, cols * rows);
  for (let i = 0; i < N; i++) items[fill[cellOf(nx[i], nz[i])]++] = i;
  return function nearestNode(x, z, filter) {
    const cx0 = Math.floor((x - bounds.minX) / cell), cz0 = Math.floor((z - bounds.minZ) / cell);
    let best = -1, bestD = Infinity;
    const maxR = Math.max(cols, rows) + 1;
    for (let r = 0; r <= maxR; r++) {
      if (best >= 0 && Math.sqrt(bestD) <= (r - 1) * cell) break;
      for (let cz = cz0 - r; cz <= cz0 + r; cz++) {
        if (cz < 0 || cz >= rows) continue;
        for (let cx = cx0 - r; cx <= cx0 + r; cx++) {
          if (cx < 0 || cx >= cols) continue;
          if (Math.abs(cx - cx0) !== r && Math.abs(cz - cz0) !== r) continue;
          const c = cz * cols + cx;
          for (let i = start[c]; i < start[c + 1]; i++) {
            const k = items[i];
            if (filter && !filter(k)) continue;
            const d = (nx[k] - x) ** 2 + (nz[k] - z) ** 2;
            if (d < bestD) { bestD = d; best = k; }
          }
        }
      }
    }
    return best;
  };
}

// --------------------------------------------------------------------- geometry
/**
 * Point on directed edge e at distance s, offset to the right by `offset` m.
 * Writes {x,z,y,hx,hz,seg} into out. segHint speeds up repeated lookups.
 */
export function edgePoint(e, s, offset, out, segHint = 0) {
  const { px, pz, cum } = e;
  const n = px.length;
  let i = segHint;
  if (i < 0) i = 0; if (i > n - 2) i = n - 2;
  while (i > 0 && cum[i] > s) i--;
  while (i < n - 2 && cum[i + 1] < s) i++;
  const segLen = cum[i + 1] - cum[i];
  const t = segLen > 0 ? Math.min(1, Math.max(0, (s - cum[i]) / segLen)) : 0;
  let dx = px[i + 1] - px[i], dz = pz[i + 1] - pz[i];
  const L = Math.hypot(dx, dz) || 1;
  dx /= L; dz /= L;
  // right-hand normal (x east, z south, looking down): right of (dx,dz) is (-dz, dx)
  out.x = px[i] + dx * t * segLen - dz * offset;
  out.z = pz[i] + dz * t * segLen + dx * offset;
  out.y = e.ys ? e.ys[i] + (e.ys[i + 1] - e.ys[i]) * t : e.y;
  out.hx = dx; out.hz = dz;
  out.seg = i;
  return out;
}

import { hopMap as geomHopMap } from '../geom/hops.js';

const FREEWAY = new Set(['motorway', 'motorway_link', 'trunk', 'trunk_link']);

/**
 * Attach the shared lane geometry (src/geom LaneGeometry built on the same RoadSets, same
 * edge / node numbering): edgePoint, hop lane mapping and pavement bounds then come from it,
 * so the sim, the renderer and the GPU agree on where every lane is.
 */
export function attachLaneGeometry(g, LG) {
  if (!LG || LG.G.edgeCount !== g.edgeCount) return g;
  g.laneGeom = LG;
  for (const e of g.edges) e.LG = LG;
  return g;
}
/**
 * Attach physical junctions (src/geom junctions.js JunctionSet): g.J, and per edge
 * e.sStop (front-bumper stop / yield point at the mouth of the junction at its end; = length
 * when that node is no junction), e.sExit (where e resumes after the box at its start; 0 if
 * none) and e.gore (ramp wait point outside the mainline lanes, -1 none).
 */
export function attachJunctions(g, J) {
  if (!J || J.mouthS.length !== 2 * g.edgeCount) return g;
  g.J = J;
  for (const e of g.edges) { e.sExit = J.mouthS[2 * e.idx]; e.sStop = J.mouthS[2 * e.idx + 1]; e.gore = J.goreS[e.idx]; }
  return g;
}
function turnAngle(a, b) { let d = b.headingStart - a.headingEnd; while (d > Math.PI) d -= 2 * Math.PI; while (d < -Math.PI) d += 2 * Math.PI; return d; } // > 0 = right turn

/**
 * Lane mapping for the hop e -> en (cached on e). Two kinds of hop:
 *  STRICT (a continuous carriageway: both edges on the freeway system and |turn| < 0.5 rad,
 *   or a plain 1-in/1-out continuation anywhere): lane l of e continues as lane l - shift
 *   of en and lat shifts by -shift * LANE_WIDTH. Lanes mapping outside en END at the node
 *   (taper: cars must steer out of them before the node, or they leave the road).
 *   shift: diverge (several out-edges, U-turn excluded): out-edges sorted from the
 *   sharpest right turn to the sharpest left take consecutive lane blocks from the right
 *   (exit right = lane 0, continuation = the lanes left of it), clamped to fit inside e;
 *   merge (several in-edges at en.from): in-edges sorted the same way (traffic joining
 *   from the right gets en's right lanes), clamped to fit in en; single continuation with
 *   a lane-count change: left-aligned (the right lanes start/end, auxiliary-lane convention).
 *  JUNCTION (an intersection turn / non-freeway junction): the junction box lets a car
 *   reach any lane: lat is rescaled across the carriageway, lat' = lat * en.lanes / e.lanes,
 *   so every lane continues (lane l -> floor((l + 0.5) en.lanes / e.lanes)).
 * Returns {strict, shift, scale}.
 */
export function hopMap(g, e, en) {
  let m = e._hop; if (!m) m = e._hop = new Map();
  let h = m.get(en.idx); if (h) return h;
  if (g.laneGeom) { const gh = geomHopMap(g.laneGeom.G, e.idx, en.idx); h = { strict: gh.strict, shift: gh.shift, scale: en.lanes / e.lanes }; m.set(en.idx, h); return h; }
  const outs = [], ins = [];
  for (let q = g.outStart[e.to]; q < g.outStart[e.to + 1]; q++) { const o = g.edges[g.outEdges[q]]; if (o.idx !== e.reverse) outs.push(o); }
  for (let q = g.inStart[en.from]; q < g.inStart[en.from + 1]; q++) { const o = g.edges[g.inEdges[q]]; if (o.idx !== en.reverse) ins.push(o); }
  const simple = outs.length <= 1 && ins.length <= 1;
  const strict = simple || (FREEWAY.has(e.highway) && FREEWAY.has(en.highway) && Math.abs(turnAngle(e, en)) < 0.5);
  let shift = 0;
  if (strict) {
    if (simple) shift = e.lanes - en.lanes;
    else {
      let div = 0, mer = 0;
      if (outs.length > 1) {
        outs.sort((a, b) => turnAngle(e, b) - turnAngle(e, a));
        for (const o of outs) { if (o.idx === en.idx) break; div += o.lanes; }
        div = Math.max(0, Math.min(div, e.lanes - en.lanes));
      }
      if (ins.length > 1) {
        ins.sort((a, b) => turnAngle(b, en) - turnAngle(a, en));
        for (const o of ins) { if (o.idx === e.idx) break; mer += o.lanes; }
        mer = Math.max(0, Math.min(mer, en.lanes - e.lanes));
      }
      shift = div - mer;
    }
  }
  h = { strict, shift, scale: en.lanes / e.lanes };
  m.set(en.idx, h);
  return h;
}
/** Lane index on en that lane l of e continues as (may be out of range for a strict lane end). */
export function mapLane(g, e, en, l) {
  const h = hopMap(g, e, en);
  return h.strict ? l - h.shift : Math.min(en.lanes - 1, Math.floor((l + 0.5) * h.scale));
}
/**
 * lat on en of a car at lat on e. Strict hops: same physical lane (lat - shift * 3.7).
 * Junction hops: the car's world position at the node re-projected onto en's first
 * segment (no sideways jump); if that lies outside en's carriageway it is brought inside
 * (half-extent he from the edge), since the junction box connects every lane.
 */
const _q1 = {}, _q2 = {};
export function mapLat(g, e, en, lat, he = 1) {
  const h = hopMap(g, e, en);
  if (h.strict) return lat - h.shift * laneW(e, e.length);
  const W = en.lanes * laneW(en, 0);
  if (g.laneGeom) {
    const LG = g.laneGeom, p = LG.latPoint(e.idx, e.length, lat, _q1), q = LG.offsetPoint(en.idx, 0, 0, 0, _q2);
    let l2 = LG.rightEdge(en.idx, 0) - ((p.x - q.x) * -q.hz + (p.z - q.z) * q.hx);
    if (l2 < he) l2 = Math.min(he, W / 2); else if (l2 > W - he) l2 = Math.max(W - he, W / 2);
    return l2;
  }
  const n = e.px.length;
  let dx = e.px[n - 1] - e.px[n - 2], dz = e.pz[n - 1] - e.pz[n - 2]; let L = Math.hypot(dx, dz) || 1; dx /= L; dz /= L;
  const o = e.rightEdge - lat;
  const wx = e.px[n - 1] - dz * o, wz = e.pz[n - 1] + dx * o;
  let ex = en.px[1] - en.px[0], ez = en.pz[1] - en.pz[0]; L = Math.hypot(ex, ez) || 1; ex /= L; ez /= L;
  const o2 = (wx - en.px[0]) * -ez + (wz - en.pz[0]) * ex;
  let l2 = en.rightEdge - o2;
  if (l2 < he) l2 = Math.min(he, W / 2); else if (l2 > W - he) l2 = Math.max(W - he, W / 2);
  return l2;
}

/** Lateral offset (m, right-positive) of a lane centre; lane 0 = rightmost. */
export function laneOffset(e, lane) {
  const off = e.laneOff;
  return off[lane < 0 ? 0 : lane >= off.length ? off.length - 1 : lane];
}

/** Load roads json files (browser fetch or node fs) and build the graph. */
export async function loadRoadGraph(urls, opts = {}) {
  const docs = [];
  for (const u of urls) {
    try {
      if (typeof window !== 'undefined' || opts.fetch) {
        const r = await (opts.fetch || fetch)(u);
        if (!r.ok) throw Object.assign(new Error(`fetch ${u}: ${r.status}`), { code: 'ENOENT' });
        docs.push(await r.json());
      } else {
        const fs = await import('node:fs/promises');
        docs.push(JSON.parse(await fs.readFile(u, 'utf8')));
      }
    } catch (err) {
      if (opts.optional && opts.optional.includes(u)) continue;
      throw err;
    }
  }
  return buildGraph(docs, opts);
}
