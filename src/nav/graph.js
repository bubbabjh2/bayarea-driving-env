// Road graph loader: converts data/roads.json (+ optional data/roads-local.json)
// into a directed graph with lengths, lane counts, speed limits and adjacency.
//
// INPUT SCHEMA (as planned by the world agent; this loader is tolerant):
//   nodes: [{id, x, z}]                       meters, x east, z south
//   edges: [{nodes:[ids], highway, name, ref, oneway, lanes, bridge, tunnel,
//            layer, maxspeed, ...}]
//
// REAL FORMAT `bay-roads/1` (data/roads.schema.md) is auto-detected via
// meta.format and handled natively:
//  * edges are ALREADY directed (one per travel direction, `twin` = opposite);
//    no reverse edges are synthesised, `oneway` is only kept as metadata.
//  * edge.nodes / from / to are INDICES into that file's nodes array; node.id
//    (OSM id) is the join key when merging roads.json + roads-local.json.
//  * `class` is the highway class; `maxspeed` is km/h (null -> meta.defaultSpeedKph
//    -> class default); `lanes` is per direction; `laneOffsets` (positive = right,
//    [0] = leftmost) are converted to edge.laneLat (positive = LEFT, [0] = RIGHTMOST,
//    the brain's convention, see src/brain/lanes.js); meta.laneWidth -> graph.laneWidth.
//  * every node becomes a vertex (splitAtAllNodes), each of our directed edges
//    remembers `srcEdgeId` (their edge id), `way` and `twinSrc` for the sim.
//
// ASSUMPTIONS / TOLERANCE for the planned (generic) schema:
//  * Node ids may be numbers or strings; they are keyed by String(id).
//  * `nodes` may also be an object {id: {x,z}} or [[id,x,z]]; `edges` may use
//    `nodes`, `nodeIds`, `path` or `refs` for the node list.
//  * By default (splitAtAllNodes: true) every node becomes a graph vertex and
//    each polyline segment a directed edge, so ANY node is a valid route
//    origin/destination and mid-way junctions always connect. With
//    splitAtAllNodes: false only shared (junction) nodes split a way and the
//    other nodes stay shape points of a longer edge.
//  * oneway: true/false/"yes"/"no"/"-1" (reverse). motorway, motorway_link,
//    trunk_link, primary_link... default to oneway=true when unspecified; all
//    other classes default to two-way. Two-way edges become two directed edges.
//  * lanes: for one-way roads `lanes` is the lane count. For two-way roads we
//    look for per-direction fields (lanesForward/lanesBackward, lanes:forward,
//    lanes_forward, forwardLanes...) and otherwise use max(1, round(lanes/2)).
//    Defaults by class when missing (see LANE_DEFAULTS).
//  * maxspeed: "65 mph" | "100 km/h" | 65 (bare number = mph, US data) | missing
//    -> class default (SPEED_DEFAULTS_MPH). Stored in m/s as `speed`.
//  * Unknown highway classes are kept (speed 25 mph, 1 lane) rather than dropped.
//  * Two-way roads get bay-roads-style lane laterals (each direction's lanes to the
//    right of the shared polyline); one-way roads are centred on it.
//  * A second file with the same schema (roads-local.json: tertiary, residential,
//    ...) is merged by shared node ids via loadGraph([json1, json2]).
import { MPH_TO_MS, heading, dist } from './geom.js';

export const SPEED_DEFAULTS_MPH = {
  motorway: 65, motorway_link: 45,
  trunk: 55, trunk_link: 40,
  primary: 40, primary_link: 30,
  secondary: 35, secondary_link: 30,
  tertiary: 30, tertiary_link: 30,
  unclassified: 25, residential: 25, living_street: 15,
  service: 15,
};
export const LANE_DEFAULTS = {
  motorway: 4, motorway_link: 1, trunk: 2, trunk_link: 1, primary: 2,
  primary_link: 1, secondary: 1, secondary_link: 1, tertiary: 1,
  tertiary_link: 1, unclassified: 1, residential: 1, living_street: 1, service: 1,
};
/** Higher = more major. Used by guidance to decide what counts as a fork. */
export const CLASS_RANK = {
  motorway: 9, trunk: 8, motorway_link: 7, trunk_link: 6, primary: 5,
  primary_link: 4, secondary: 3, secondary_link: 3, tertiary: 2,
  tertiary_link: 2, unclassified: 1, residential: 1, living_street: 0, service: 0,
};
const ONEWAY_BY_DEFAULT = new Set(['motorway', 'motorway_link', 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link']);

export function parseMaxspeedMph(v, highway) {
  if (v == null || v === '') return SPEED_DEFAULTS_MPH[highway] ?? 25;
  if (typeof v === 'number') return v;
  const s = String(v).toLowerCase();
  const m = s.match(/([\d.]+)/);
  if (!m) return SPEED_DEFAULTS_MPH[highway] ?? 25;
  const n = parseFloat(m[1]);
  if (s.includes('km')) return n / 1.609344;
  return n; // "65 mph" or bare "65"
}

function parseOneway(v) {
  if (v === true || v === 1) return 1;
  if (v === false || v === 0 || v == null) return 0;
  const s = String(v).toLowerCase();
  if (s === 'yes' || s === 'true' || s === '1') return 1;
  if (s === '-1' || s === 'reverse') return -1;
  return 0;
}

function pick(obj, keys) {
  for (const k of keys) if (obj[k] != null && obj[k] !== '') return obj[k];
  return undefined;
}

function normalizeNodes(nodes) {
  const out = [];
  if (Array.isArray(nodes)) {
    for (const n of nodes) {
      if (Array.isArray(n)) out.push({ id: String(n[0]), x: +n[1], z: +n[2] });
      else out.push({ id: String(n.id), x: +(n.x ?? n.lon ?? 0), z: +(n.z ?? n.y ?? 0) });
    }
  } else if (nodes && typeof nodes === 'object') {
    for (const [id, n] of Object.entries(nodes)) out.push({ id, x: +n.x, z: +(n.z ?? n.y) });
  }
  return out;
}

export class Graph {
  constructor() {
    this.nodes = [];          // {id, x, z}
    this.nodeIndex = new Map(); // String(id) -> index
    this.edges = [];          // directed edges, see _addDirectedEdge
    this.out = [];            // node index -> [edge index]
    this.in = [];
    this.maxSpeed = 0;        // m/s, for the A* heuristic
    this.laneWidth = 3.7;     // meters (meta.laneWidth when present)
    this.bounds = { minX: Infinity, maxX: -Infinity, minZ: Infinity, maxZ: -Infinity };
    // tier name -> Int32Array(tier-local node index -> graph node index); filled for bay-roads
    // JSON and binary packs (stations.json `node` is a tier-local index, see nav/stations.js)
    this.tierNodes = {};
    this.nodeIdsAreOsm = true; // false for binary packs (node ids are pack indices)
  }
  /** Graph node index of node `i` of tier `tier` ('main' | 'local'), or -1. */
  tierNode(tier, i) { const m = this.tierNodes[tier]; return m && i >= 0 && i < m.length ? m[i] : -1; }
  nodeAt(i) { return this.nodes[i]; }
  indexOf(id) { return this.nodeIndex.get(String(id)); }
  edgeCount() { return this.edges.length; }
  nodeCount() { return this.nodes.length; }
  /** Lateral position (+left, metres from the polyline) of lane k (0 = rightmost) on edge e. */
  laneLateral(e, k) {
    if (e.laneLat) return e.laneLat[Math.min(Math.max(k, 0), e.laneLat.length - 1)];
    return (k - (e.lanes - 1) / 2) * this.laneWidth;
  }
  /** Point on edge e at distance d from its start: [x, z, headingRad]. */
  pointOnEdge(e, d) {
    const p = e.points;
    const cum = e.cum;
    if (d <= 0) return [p[0], p[1], e.headingStart];
    if (d >= e.length) return [p[p.length - 2], p[p.length - 1], e.headingEnd];
    let lo = 0, hi = cum.length - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (cum[m] <= d) lo = m; else hi = m; }
    const seg = cum[hi] - cum[lo] || 1;
    const t = (d - cum[lo]) / seg;
    const x0 = p[2 * lo], z0 = p[2 * lo + 1], x1 = p[2 * hi], z1 = p[2 * hi + 1];
    return [x0 + (x1 - x0) * t, z0 + (z1 - z0) * t, Math.atan2(z1 - z0, x1 - x0)];
  }
}

/**
 * Build a Graph from one json document or an array of them (merged by node id).
 * @param {object|object[]} jsons
 * @returns {Graph}
 */
export function loadGraph(jsons, opts = {}) {
  const docs = Array.isArray(jsons) ? jsons : [jsons];
  const splitAll = opts.splitAtAllNodes ?? true;
  const g = new Graph();
  const docLocal = []; // per doc: local node index -> global index (bay-roads format)
  // 1. nodes
  for (const doc of docs) {
    const bay = isBayRoads(doc, opts);
    const local = [];
    if (bay && doc.meta?.laneWidth) g.laneWidth = doc.meta.laneWidth;
    for (const n of normalizeNodes(doc.nodes)) {
      let gi = g.nodeIndex.get(n.id);
      if (gi === undefined) { gi = g.nodes.length; g.nodeIndex.set(n.id, gi); g.nodes.push(n); }
      local.push(gi);
      if (gi !== g.nodes.length - 1) continue; // shared node between files
      const b = g.bounds;
      if (n.x < b.minX) b.minX = n.x; if (n.x > b.maxX) b.maxX = n.x;
      if (n.z < b.minZ) b.minZ = n.z; if (n.z > b.maxZ) b.maxZ = n.z;
    }
    docLocal.push(local);
    if (bay) g.tierNodes[doc.meta?.tier || (docLocal.length === 1 ? 'main' : 'local')] = Int32Array.from(local);
  }
  for (let i = 0; i < g.nodes.length; i++) { g.out.push([]); g.in.push([]); }
  // 2. collect raw ways (as node index arrays) and count node usage
  const ways = [];
  const useCount = new Int32Array(g.nodes.length);
  let srcId = 0;
  docs.forEach((doc, di) => {
    const bay = isBayRoads(doc, opts);
    const local = docLocal[di];
    const meta = doc.meta || {};
    for (const raw of doc.edges || doc.ways || []) {
      const ids = pick(raw, ['nodes', 'nodeIds', 'path', 'refs']);
      if (!Array.isArray(ids) || ids.length < 2) continue;
      const idx = [];
      for (const id of ids) {
        const i = bay ? local[id] : g.nodeIndex.get(String(id));
        if (i === undefined) continue; // dangling reference: skip the node
        if (idx.length && idx[idx.length - 1] === i) continue;
        idx.push(i);
      }
      if (idx.length < 2) continue;
      for (const i of idx) useCount[i]++;
      ways.push({ raw, idx, srcId: srcId++, bay, meta, tier: meta.tier || di });
    }
  });
  // 3. split ways at shared interior nodes, create directed edges
  for (const w of ways) {
    const { raw, idx, bay } = w;
    const highway = String(raw.highway || raw.class || 'unclassified');
    const isLink = highway.endsWith('_link');
    let ow = parseOneway(raw.oneway);
    if (raw.oneway == null && ONEWAY_BY_DEFAULT.has(highway)) ow = 1;
    if (bay) {
      // already directed: one edge per direction; lanes per direction; km/h
      const lanes = Math.max(1, Math.round(+raw.lanes || LANE_DEFAULTS[highway] || 1));
      let kph = raw.maxspeed;
      if (kph == null) kph = w.meta.defaultSpeedKph?.[highway];
      const speedMph = kph != null ? +kph / 1.609344 : (SPEED_DEFAULTS_MPH[highway] ?? 25);
      let laneLat = null;
      if (Array.isArray(raw.laneOffsets) && raw.laneOffsets.length === lanes) {
        laneLat = new Float64Array(lanes);
        for (let k = 0; k < lanes; k++) laneLat[k] = -raw.laneOffsets[lanes - 1 - k];
      }
      const meta = {
        highway, isLink, name: raw.name || '', ref: raw.ref || '',
        bridge: !!raw.bridge, tunnel: !!raw.tunnel, layer: +(raw.layer || 0),
        speedMph, speed: speedMph * MPH_TO_MS, srcId: raw.way ?? w.srcId, rank: CLASS_RANK[highway] ?? 1,
        srcEdgeId: raw.id, way: raw.way, twinSrc: raw.twin, tier: w.tier, laneLat, oppLanes: raw.oppLanes || 0,
      };
      let start = 0;
      for (let k = 1; k < idx.length; k++) {
        if (k === idx.length - 1 || splitAll || useCount[idx[k]] > 1) { addDirected(g, idx.slice(start, k + 1), meta, lanes, !!raw.oneway); start = k; }
      }
      continue;
    }
    // generic schema: lay lanes out like bay-roads does (two-way: this direction's lanes
    // sit to the RIGHT of the polyline; one-way: centred), index 0 = rightmost, + = left
    const laneLatFor = (lanes, lanesBack) => { const a = new Float64Array(lanes); for (let j = 0; j < lanes; j++) { const k = lanes - 1 - j; a[j] = -(((lanesBack - lanes) / 2 + k + 0.5) * g.laneWidth); } return a; };
    const totalLanes = raw.lanes != null ? Math.max(1, Math.round(+raw.lanes)) : undefined;
    let lanesF, lanesB;
    if (ow !== 0) {
      lanesF = lanesB = totalLanes ?? LANE_DEFAULTS[highway] ?? 1;
    } else {
      const f = pick(raw, ['lanesForward', 'lanes:forward', 'lanes_forward', 'forwardLanes', 'lanesPerDirection', 'lanes_per_direction']);
      const b = pick(raw, ['lanesBackward', 'lanes:backward', 'lanes_backward', 'backwardLanes', 'lanesPerDirection', 'lanes_per_direction']);
      const half = totalLanes != null ? Math.max(1, Math.round(totalLanes / 2)) : (LANE_DEFAULTS[highway] ?? 1);
      lanesF = f != null ? Math.max(1, Math.round(+f)) : half;
      lanesB = b != null ? Math.max(1, Math.round(+b)) : half;
    }
    const speedMph = parseMaxspeedMph(raw.maxspeed ?? raw.maxSpeed ?? raw.speed, highway);
    const meta = {
      highway, isLink, name: raw.name || '', ref: raw.ref || '',
      bridge: !!raw.bridge, tunnel: !!raw.tunnel, layer: +(raw.layer || 0),
      speedMph, speed: speedMph * MPH_TO_MS, srcId: raw.way ?? w.srcId, rank: CLASS_RANK[highway] ?? 1,
      srcEdgeId: raw.id, way: raw.way, twinSrc: -1, tier: w.tier, laneLat: null, oppLanes: 0,
    };
    const metaF = ow === 0 ? { ...meta, laneLat: laneLatFor(lanesF, lanesB), oppLanes: lanesB } : meta;
    const metaB = ow === 0 ? { ...meta, laneLat: laneLatFor(lanesB, lanesF), oppLanes: lanesF } : meta;
    // split points
    let start = 0;
    for (let k = 1; k < idx.length; k++) {
      const last = k === idx.length - 1;
      if (last || splitAll || useCount[idx[k]] > 1) {
        const piece = idx.slice(start, k + 1);
        if (ow >= 0) addDirected(g, piece, metaF, lanesF, ow === 0 ? false : true);
        if (ow <= 0) addDirected(g, piece.slice().reverse(), metaB, lanesB, ow === 0 ? false : true);
        start = k;
      }
    }
  }
  for (const e of g.edges) if (e.speed > g.maxSpeed) g.maxSpeed = e.speed;
  return g;
}

function isBayRoads(doc, opts) {
  if (opts.format === 'bay-roads') return true;
  if (opts.format === 'generic') return false;
  return typeof doc?.meta?.format === 'string' && doc.meta.format.startsWith('bay-roads');
}

function addDirected(g, idx, meta, lanes, oneway) {
  const n = idx.length;
  const points = new Float64Array(n * 2);
  const cum = new Float64Array(n);
  let len = 0;
  for (let i = 0; i < n; i++) {
    const nd = g.nodes[idx[i]];
    points[2 * i] = nd.x; points[2 * i + 1] = nd.z;
    if (i > 0) len += dist(points[2 * i - 2], points[2 * i - 1], nd.x, nd.z);
    cum[i] = len;
  }
  const e = {
    id: g.edges.length,
    from: idx[0], to: idx[n - 1],
    nodeIdx: idx,
    points, cum, length: Math.max(len, 0.01),
    lanes, oneway,
    headingStart: heading(points[0], points[1], points[2], points[3]),
    headingEnd: heading(points[2 * n - 4], points[2 * n - 3], points[2 * n - 2], points[2 * n - 1]),
    ...meta,
  };
  g.edges.push(e);
  g.out[e.from].push(e.id);
  g.in[e.to].push(e.id);
  return e;
}

/**
 * Load the world agent's binary packs (format bay-roads-bin/1):
 *   loadGraphBin(meta, buffer)                       one tier (data/roads.bin + roads.meta.json)
 *   loadGraphBin([{meta, buffer}, {meta, buffer}])   both tiers merged (main first, then local)
 * Tiers are joined with the local pack's `linkNodes` section (pairs localIndex, mainIndex),
 * so the merged graph has the SAME node order and edges as loadGraph([roads.json,
 * roads-local.json]) (main nodes, then the local nodes not shared with main; edges in pack
 * order, split at every node). The packs carry no names/refs/OSM ids: guidance text falls
 * back to "the road", node ids are `${tier}:${index}` strings, graph.nodeIdsAreOsm = false,
 * and graph.tierNode(tier, i) maps a pack node index (e.g. stations.json `node`) to the graph.
 * Each node also keeps its baked elevation `y`.
 */
export function loadGraphBin(packsOrMeta, buffer, opts = {}) {
  let packs;
  if (Array.isArray(packsOrMeta)) { packs = packsOrMeta; if (buffer) opts = buffer; } // (packs, opts)
  else packs = [{ meta: packsOrMeta, buffer }];                                      // (meta, buffer, opts)
  const splitAll = opts.splitAtAllNodes ?? true;
  const g = new Graph();
  g.nodeIdsAreOsm = false;
  // main tier first so the shared-node join resolves into existing indices
  const order = packs.map((p, i) => ({ ...p, buffer: toArrayBuffer(p.buffer), tier: p.meta.tier || (i === 0 ? 'main' : 'local'), i }))
    .sort((a, b) => (a.tier === 'main' ? 0 : 1) - (b.tier === 'main' ? 0 : 1) || a.i - b.i);
  const maps = [];
  for (const pk of order) {
    const { meta } = pk, sec = meta.sections;
    const view = (name, T) => new T(pk.buffer, sec[name].byteOffset, sec[name].length);
    const xyz = view('nodeXYZ', Float32Array);
    const n = sec.nodeXYZ.count;
    const map = new Int32Array(n).fill(-1);
    if (sec.linkNodes && g.tierNodes.main) {
      const ln = view('linkNodes', Uint32Array), mainMap = g.tierNodes.main;
      for (let k = 0; k + 1 < ln.length; k += 2) if (ln[k + 1] < mainMap.length) map[ln[k]] = mainMap[ln[k + 1]];
    }
    for (let i = 0; i < n; i++) {
      if (map[i] >= 0) continue;
      const gi = g.nodes.length;
      const nd = { id: `${pk.tier}:${i}`, x: xyz[3 * i], y: xyz[3 * i + 1], z: xyz[3 * i + 2] };
      g.nodeIndex.set(nd.id, gi); g.nodes.push(nd); g.out.push([]); g.in.push([]);
      map[i] = gi;
      const b = g.bounds;
      if (nd.x < b.minX) b.minX = nd.x; if (nd.x > b.maxX) b.maxX = nd.x;
      if (nd.z < b.minZ) b.minZ = nd.z; if (nd.z > b.maxZ) b.maxZ = nd.z;
    }
    g.tierNodes[pk.tier] = map;
    maps.push(map);
    if (meta.laneWidth) g.laneWidth = meta.laneWidth;
  }
  order.forEach((pk, pi) => {
    const { meta } = pk, sec = meta.sections, map = maps[pi];
    const E = new Uint32Array(pk.buffer, sec.edges.byteOffset, sec.edges.length);
    const poly = new Uint32Array(pk.buffer, sec.poly.byteOffset, sec.poly.length);
    const stride = meta.edgeStride || 8;
    const classes = meta.classesAll || meta.classes;
    const flag = meta.flagBits || { ONEWAY: 1, BRIDGE: 2, TUNNEL: 4, LANES_FALLBACK: 8, SPEED_DEFAULT: 16, LINK: 32 };
    const noTwin = meta.noTwin ?? 0xFFFFFFFF;
    const lw = meta.laneWidth || g.laneWidth;
    const m = sec.edges.count;
    for (let e = 0; e < m; e++) {
      const b = e * stride;
      const first = E[b], count = E[b + 1], lanes = Math.max(1, E[b + 2]), lanesBack = E[b + 3];
      const highway = classes[E[b + 4]] || 'unclassified';
      const speed = E[b + 5] / 100, flags = E[b + 6], twin = E[b + 7];
      const idx = [];
      for (let k = 0; k < count; k++) { const ni = map[poly[first + k]]; if (ni >= 0 && (!idx.length || idx[idx.length - 1] !== ni)) idx.push(ni); }
      if (idx.length < 2) continue;
      const laneLat = new Float64Array(lanes);
      for (let j = 0; j < lanes; j++) { const k = lanes - 1 - j; laneLat[j] = -(((lanesBack - lanes) / 2 + k + 0.5) * lw); }
      const md = {
        highway, isLink: !!(flags & flag.LINK) || highway.endsWith('_link'), name: '', ref: '',
        bridge: !!(flags & flag.BRIDGE), tunnel: !!(flags & flag.TUNNEL), layer: 0,
        speedMph: speed / MPH_TO_MS, speed, srcId: `${pk.tier}:${e}`, rank: CLASS_RANK[highway] ?? 1,
        srcEdgeId: e, way: undefined, twinSrc: twin === noTwin ? -1 : twin, tier: pk.tier, laneLat, oppLanes: lanesBack,
      };
      let start = 0;
      for (let k = 1; k < idx.length; k++) if (k === idx.length - 1 || splitAll) { addDirected(g, idx.slice(start, k + 1), md, lanes, !!(flags & flag.ONEWAY)); start = k; }
    }
  });
  for (const e of g.edges) if (e.speed > g.maxSpeed) g.maxSpeed = e.speed;
  return g;
}

function toArrayBuffer(b) {
  if (b instanceof ArrayBuffer) return b;
  if (b && b.buffer instanceof ArrayBuffer) return b.byteOffset === 0 && b.byteLength === b.buffer.byteLength ? b.buffer : b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
  throw new Error('loadGraphBin: buffer must be an ArrayBuffer or typed array');
}

/** Browser/node helper: fetch both tiers' packs and merge them. `base` = URL of the data/ folder. */
export async function fetchGraphBin(base = 'data/', files = ['roads', 'roads-local'], fetchFn = globalThis.fetch) {
  const packs = [];
  for (const f of files) {
    const metaRes = await fetchFn(`${base}${f}.meta.json`);
    if (!metaRes.ok) continue;
    const meta = await metaRes.json();
    const buf = await (await fetchFn(`${base}${meta.file || f + '.bin'}`)).arrayBuffer();
    packs.push({ meta, buffer: buf });
  }
  return packs.length ? loadGraphBin(packs) : null;
}

/** Load from files under node (not used in the browser). */
export async function loadGraphFromFiles(paths) {
  const fs = await import('node:fs/promises');
  const docs = [], packs = [];
  for (const p of paths) {
    try {
      if (p.endsWith('.bin')) {
        const meta = JSON.parse(await fs.readFile(p.replace(/\.bin$/, '.meta.json'), 'utf8'));
        packs.push({ meta, buffer: await fs.readFile(p) });
      } else docs.push(JSON.parse(await fs.readFile(p, 'utf8')));
    } catch (err) { if (err.code !== 'ENOENT') throw err; }
  }
  if (packs.length) return loadGraphBin(packs); // all .bin paths are merged (json paths are ignored then)
  if (!docs.length) return null;
  return loadGraph(docs);
}
