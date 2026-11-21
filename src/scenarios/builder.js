// Scenario world builder: author a small road network in metres, emit it in EXACTLY the real
// world's data format (data/roads.schema.md): two road tiers (roads.bin / roads-local.bin + meta
// + readable JSON copies), stations, controls, crosswalks, crossings (grade separations).
// Junctions (data/junctions.*) are then built by the world's own tools/build_junctions.mjs
// (see write.js), which also rewrites the junction controls' stop lines to the junction mouths.
//
//   const w = new World();
//   const a = w.node(0, 0), b = w.node(1000, 0);            // junction / end nodes (x east, z south, y up)
//   w.road([a, [500, 0, 2], b], { cls: 'primary', lanes: 2, opp: 2 });   // [x, z, y?] = shape points
//   w.control('signal', b);                                   // every approach of node b
//   w.trip({ from: a, to: b, count: 20, depart: [7 * 3600, 7.2 * 3600] });
//   const doc = w.finish();                                   // packs (typed arrays + JSON) + scenario doc
//
// Plain ES module (node + browser); no rendering imports.
import { META_TEMPLATE } from './metaTemplate.js';

export const CLASSES_ALL = ['motorway', 'motorway_link', 'trunk', 'trunk_link', 'primary', 'primary_link',
  'secondary', 'secondary_link', 'tertiary', 'tertiary_link', 'unclassified', 'residential', 'living_street'];
export const MAIN_CLASSES = new Set(CLASSES_ALL.slice(0, 8));
export const FLAG = { ONEWAY: 1, BRIDGE: 2, TUNNEL: 4, LANES_FALLBACK: 8, SPEED_DEFAULT: 16, LINK: 32,
  LANES_CLAMPED: 64, LANES_SMOOTHED: 128, AUX_LANE: 256, LANE_DROP: 512 };
export const LANE_WIDTH_BY_CLASS = { motorway: 3.7, motorway_link: 3.7, trunk: 3.7, trunk_link: 3.6, primary: 3.6, primary_link: 3.5,
  secondary: 3.5, secondary_link: 3.4, tertiary: 3.3, tertiary_link: 3.3, unclassified: 3.1, residential: 3.1, living_street: 3.0 };
export const DEFAULT_SPEED_KPH = { ...META_TEMPLATE.main.defaultSpeedKph, ...META_TEMPLATE.local.defaultSpeedKph };
const NO_TWIN = 0xFFFFFFFF;
const CONTROL_TYPES = ['signal', 'stop', 'give_way', 'ramp_meter', 'crossing_signal'];
const FREEWAY_SYS = new Set(['motorway', 'motorway_link', 'trunk', 'trunk_link']);

export class World {
  constructor({ name = 'scenario', family = 'custom', seed = 1, params = {} } = {}) {
    this.name = name; this.family = family; this.seed = seed; this.params = params;
    this.nx = []; this.ny = []; this.nz = []; this.nname = [];
    this.roads = [];        // {pts: [nodeId...], opts}
    this.controls = [];     // {type, node, approaches: 'all' | [fromNodeId...], allWay, midBlock: {from, to, at}}
    this.trips = [];        // {from, to, via, count, depart, ...}
    this.obstacles = [];    // {from, to, at (m along), lane | lat, type}
    this.stations = [];     // {node, pumps}
    this.notes = [];
    this.forced = new Set();   // nodes that always split roads (degree-2 attribute changes)
    this.view = null;          // {x, z, span}: where the interesting part is (viewer / screenshots)
    this.tiles = null;         // world packs: [{name, family, params, notes, offset, bbox}]
  }

  /** Bounding box of the nodes used by roads. */
  bbox() {
    let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
    for (const r of this.roads) for (const p of r.pts) { minX = Math.min(minX, this.nx[p]); maxX = Math.max(maxX, this.nx[p]); minZ = Math.min(minZ, this.nz[p]); maxZ = Math.max(maxZ, this.nz[p]); }
    return { minX, maxX, minZ, maxZ };
  }

  /** Copy another World into this one shifted by (dx, dz) as tile `tile` (world packs: disjoint, unconnected). */
  append(o, dx, dz, tile) {
    const base = this.nx.length;
    for (let i = 0; i < o.nx.length; i++) this.node(o.nx[i] + dx, o.nz[i] + dz, o.ny[i], o.nname[i] != null ? `${o.name}/${o.nname[i]}` : null);
    const m = (id) => (id == null || id < 0 ? id : id + base);
    for (const r of o.roads) this.roads.push({ pts: r.pts.map(m), opts: r.opts, shape: r.shape });
    for (const c of o.controls) this.controls.push({ ...c, node: m(c.node), approaches: c.approaches === 'all' ? 'all' : c.approaches.map(m),
      midBlock: c.midBlock ? { ...c.midBlock, from: m(c.midBlock.from), to: m(c.midBlock.to) } : null });
    for (const t of o.trips) this.trips.push({ ...t, from: m(t.from), to: m(t.to), via: (t.via || []).map(m), tile });
    for (const ob of o.obstacles) this.obstacles.push({ ...ob, from: m(ob.from), to: m(ob.to), tile });
    for (const s of o.stations) this.stations.push({ ...s, node: m(s.node) });
    for (const f of o.forced) this.forced.add(m(f));
    const bb = o.bbox();
    (this.tiles || (this.tiles = [])).push({ name: o.name, family: o.family, seed: o.seed, params: o.params, notes: o.notes, offset: [dx, dz], view: o.view ? { ...o.view, x: o.view.x + dx, z: o.view.z + dz } : null,
      bbox: { minX: bb.minX + dx, maxX: bb.maxX + dx, minZ: bb.minZ + dz, maxZ: bb.maxZ + dz } });
    return this;
  }

  /** A graph node (junction / end / forced split point). y = elevation (m). */
  node(x, z, y = 0, name = null) {
    this.nx.push(x); this.nz.push(z); this.ny.push(y); this.nname.push(name);
    return this.nx.length - 1;
  }
  split(id) { this.forced.add(id); return id; }

  /**
   * A road through points: node ids (numbers) or [x, z, y?] shape points (interior only).
   * opts: cls, lanes (in travel direction), opp (lanes the other way; 0 / omitted with oneway = one-way),
   *       oneway (default: opp == 0), speedKph, laneWidth, bridge, tunnel, layer, name, ref, aux (AUX_LANE)
   * Two-way roads give an edge pair (twins); roads split at every node used by another road.
   */
  road(pts, opts = {}) {
    const ids = pts.map((p) => (typeof p === 'number' ? p : this.node(p[0], p[1], p[2] ?? 0)));
    if (typeof pts[0] !== 'number' || typeof pts[pts.length - 1] !== 'number') throw new Error('road: first and last points must be node ids');
    const o = { cls: 'residential', lanes: 1, ...opts };
    if (o.opp == null) o.opp = o.oneway ? 0 : o.lanes;
    o.oneway = o.opp === 0;
    if (!CLASSES_ALL.includes(o.cls)) throw new Error('road: unknown class ' + o.cls);
    this.roads.push({ pts: ids, opts: o, shape: pts.map((p) => typeof p !== 'number') });
    return this.roads.length - 1;
  }

  /** Traffic control at node `node` (type signal | stop | give_way | ramp_meter). approaches: 'all' or list of from-node ids
   *  (the in-edge coming from that neighbour node along its road). midBlock: {from, to, at} = control along edge (from->to) at s = at. */
  control(type, node, { approaches = 'all', allWay = false, midBlock = null } = {}) {
    if (!CONTROL_TYPES.includes(type)) throw new Error('control: unknown type ' + type);
    this.controls.push({ type, node, approaches, allWay, midBlock });
  }

  /** Trips: from node -> to node (optional via nodes); count trips departing uniformly in [depart[0], depart[1]] s. */
  trip(t) { this.trips.push({ count: 1, depart: [7 * 3600, 7 * 3600 + 600], via: [], ...t }); }
  /** Static obstacle (stopped / disabled car) on the edge from node `from` to node `to`, `at` m along, in lane `lane`. */
  obstacle(o) { this.obstacles.push({ lane: 0, type: 0, ...o }); }
  station(node, pumps = 8) { this.stations.push({ node, pumps }); }

  // ------------------------------------------------------------------ geometry helpers
  /** Shape points on a straight line a->b every `step` m (excl. ends), elevation from yFn(t) (t 0..1) or linear. */
  static line(ax, az, bx, bz, step = 25, yFn = null) {
    const L = Math.hypot(bx - ax, bz - az), n = Math.max(1, Math.round(L / step)), out = [];
    for (let k = 1; k < n; k++) { const t = k / n; out.push([ax + (bx - ax) * t, az + (bz - az) * t, yFn ? yFn(t) : 0]); }
    return out;
  }
  /** Arc shape points: centre (cx, cz), radius R, from angle a0 to a1 (rad, x-z plane), every <= dA rad / <= step m (excl. ends). */
  static arc(cx, cz, R, a0, a1, { dA = 0.06, step = 12, yFn = null } = {}) {
    const n = Math.max(2, Math.ceil(Math.max(Math.abs(a1 - a0) / dA, (Math.abs(a1 - a0) * R) / step))), out = [];
    for (let k = 1; k < n; k++) { const t = k / n, a = a0 + (a1 - a0) * t; out.push([cx + R * Math.cos(a), cz + R * Math.sin(a), yFn ? yFn(t) : 0]); }
    return out;
  }

  // ------------------------------------------------------------------ emit
  /**
   * Split roads into directed edges and pack both tiers. Returns
   * { main: {meta, bin: ArrayBuffer, json}, local: {...}, stations, controls, crosswalks, crossings, scenario, nodeMap, edgeOf(fromNode, toNode) }.
   * nodeMap[builderNode] = merged-graph node index (the sim's / src/geom numbering); edges are merged ids.
   */
  finish() {
    const N = this.nx.length;
    // node use counts (endpoints and nodes shared by roads split)
    const use = new Uint16Array(N), isEnd = new Uint8Array(N);
    for (const r of this.roads) { const seen = new Set(); for (const p of r.pts) if (!seen.has(p)) { use[p]++; seen.add(p); } isEnd[r.pts[0]] = 1; isEnd[r.pts[r.pts.length - 1]] = 1; }
    for (const f of this.forced) isEnd[f] = 1;
    // directed pieces
    const pieces = [];   // {nodes: [], opts, roadIdx}
    for (let ri = 0; ri < this.roads.length; ri++) {
      const r = this.roads[ri];
      let cur = [r.pts[0]];
      for (let k = 1; k < r.pts.length; k++) {
        const p = r.pts[k];
        cur.push(p);
        if (k === r.pts.length - 1 || isEnd[p] || use[p] > 1) { pieces.push({ nodes: cur, opts: r.opts, roadIdx: ri }); cur = [p]; }
      }
    }
    // edges per tier: forward (+ twin)
    const tierEdges = [[], []];
    for (const pc of pieces) {
      const o = pc.opts, tier = MAIN_CLASSES.has(o.cls) ? 0 : 1;
      const fwd = { nodes: pc.nodes, lanes: o.lanes, back: o.opp, o, roadIdx: pc.roadIdx };
      tierEdges[tier].push(fwd);
      if (!o.oneway) { const bwd = { nodes: pc.nodes.slice().reverse(), lanes: o.opp, back: o.lanes, o, roadIdx: pc.roadIdx }; fwd.twinRef = bwd; bwd.twinRef = fwd; tierEdges[tier].push(bwd); }
    }
    // tier node tables
    const tierNodes = [[], []], tierIdx = [new Int32Array(N).fill(-1), new Int32Array(N).fill(-1)];
    for (let t = 0; t < 2; t++) {
      const used = new Uint8Array(N);
      for (const e of tierEdges[t]) for (const p of e.nodes) used[p] = 1;
      for (let i = 0; i < N; i++) if (used[i]) { tierIdx[t][i] = tierNodes[t].length; tierNodes[t].push(i); }
    }
    // merged numbering (= graphFromRoadSets / fromRoadSet): main nodes, then local-only nodes in local order
    const nodeMap = new Int32Array(N).fill(-1);
    for (const i of tierNodes[0]) nodeMap[i] = tierIdx[0][i];
    let nm = tierNodes[0].length;
    for (const i of tierNodes[1]) if (nodeMap[i] < 0) nodeMap[i] = nm++;
    const mainEdgeCount = tierEdges[0].length;
    tierEdges[0].forEach((e, k) => { e.id = k; e.tier = 0; });
    tierEdges[1].forEach((e, k) => { e.id = mainEdgeCount + k; e.tier = 1; e.tierEdge = k; });
    const allEdges = [...tierEdges[0], ...tierEdges[1]];
    const len2d = (nodes) => { let L = 0; for (let k = 1; k < nodes.length; k++) L += Math.hypot(this.nx[nodes[k]] - this.nx[nodes[k - 1]], this.nz[nodes[k]] - this.nz[nodes[k - 1]]); return L; };
    for (const e of allEdges) e.length = len2d(e.nodes);
    // LANE_DROP: plain continuation (one way on, one feeder) into fewer lanes
    const outOf = new Map(), inOf = new Map();
    for (const e of allEdges) { const a = e.nodes[0], b = e.nodes[e.nodes.length - 1]; (outOf.get(a) || outOf.set(a, []).get(a)).push(e); (inOf.get(b) || inOf.set(b, []).get(b)).push(e); }
    for (const e of allEdges) {
      const b = e.nodes[e.nodes.length - 1];
      const outs = (outOf.get(b) || []).filter((o) => o !== e.twinRef), ins = (inOf.get(b) || []).filter((i) => i !== e.twinRef?.twinRef && i.twinRef !== e && true);
      const feeders = (inOf.get(b) || []).filter((i) => !(outs.length === 1 && i === outs[0].twinRef));
      if (outs.length === 1 && feeders.length === 1 && outs[0].lanes < e.lanes) e.laneDrop = true;
      void ins;
    }
    const packs = [0, 1].map((t) => this._packTier(t, tierEdges[t], tierNodes[t], tierIdx, mainEdgeCount));
    const edgeOfMap = new Map();
    for (const e of allEdges) edgeOfMap.set(e.nodes[0] + ':' + e.nodes[e.nodes.length - 1] + ':' + e.roadIdx, e.id);
    const edgeOf = (a, b) => { for (const e of allEdges) if (e.nodes[0] === a && e.nodes[e.nodes.length - 1] === b) return e.id; return -1; };
    // controls (stop lines = node minus half the widest crossing road + 1.5 m; tools/build_junctions.mjs moves them to the junction mouths)
    const ctl = this._controls(allEdges, inOf, nodeMap, tierIdx, mainEdgeCount);
    const st = this._stations(tierIdx);
    const crossings = this._crossings(allEdges, mainEdgeCount);
    let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
    for (let i = 0; i < N; i++) { minX = Math.min(minX, this.nx[i]); maxX = Math.max(maxX, this.nx[i]); minZ = Math.min(minZ, this.nz[i]); maxZ = Math.max(maxZ, this.nz[i]); }
    const scenario = {
      format: 'bay-scenario/1', name: this.name, family: this.family, seed: this.seed, params: this.params, notes: this.notes,
      bbox: { minX, maxX, minZ, maxZ }, nodeCount: nm, edgeCount: allEdges.length, mainEdgeCount,
      nodeNames: Object.fromEntries(this.nname.map((n, i) => [n, nodeMap[i]]).filter(([n, k]) => n != null && k >= 0)),
      trips: this.trips.map((t) => ({ ...t, from: nodeMap[t.from], to: nodeMap[t.to], via: (t.via || []).map((v) => nodeMap[v]) })),
      obstacles: this.obstacles.map((o) => ({ ...o, edge: edgeOf(o.from, o.to), from: undefined, to: undefined })),
      tiles: this.tiles || [{ name: this.name, family: this.family, seed: this.seed, params: this.params, notes: this.notes, offset: [0, 0], view: this.view, bbox: { minX, maxX, minZ, maxZ } }],
    };
    for (const t of scenario.trips) if (t.from < 0 || t.to < 0 || t.via.some((v) => v < 0)) throw new Error(`${this.name}: trip node not on any road`);
    for (const o of scenario.obstacles) if (o.edge < 0) throw new Error(`${this.name}: obstacle edge not found`);
    return { main: packs[0], local: packs[1], stations: st, controls: ctl, crossings, scenario, nodeMap, edgeOf,
      crosswalks: { meta: { format: 'bay-crosswalks/1', count: 0, kinds: ['painted', 'signalised'], source: 'synthetic scenario' }, points: [] } };
  }

  _packTier(t, edges, nodes, tierIdx, mainEdgeCount) {
    const tierName = t === 0 ? 'main' : 'local', idx = tierIdx[t];
    const nN = nodes.length, nE = edges.length;
    const xyz = new Float32Array(3 * nN);
    nodes.forEach((i, k) => { xyz[3 * k] = this.nx[i]; xyz[3 * k + 1] = this.ny[i]; xyz[3 * k + 2] = this.nz[i]; });
    let P = 0; for (const e of edges) P += e.nodes.length;
    const E = new Uint32Array(8 * nE), poly = new Uint32Array(P), len = new Float32Array(nE), lw = new Float32Array(nE);
    const jsonEdges = [];
    let p = 0;
    const local = (e) => (t === 0 ? e.id : e.id - mainEdgeCount);
    edges.forEach((e, k) => {
      const o = e.o;
      E[8 * k] = p; E[8 * k + 1] = e.nodes.length;
      for (const n of e.nodes) poly[p++] = idx[n];
      E[8 * k + 2] = e.lanes; E[8 * k + 3] = e.back; E[8 * k + 4] = CLASSES_ALL.indexOf(o.cls);
      const kph = o.speedKph ?? DEFAULT_SPEED_KPH[o.cls];
      E[8 * k + 5] = Math.round((kph / 3.6) * 100);
      let fl = 0;
      if (o.oneway) fl |= FLAG.ONEWAY; if (o.bridge) fl |= FLAG.BRIDGE; if (o.tunnel) fl |= FLAG.TUNNEL;
      if (o.speedKph == null) fl |= FLAG.SPEED_DEFAULT; if (o.cls.endsWith('_link')) fl |= FLAG.LINK;
      if (o.aux) fl |= FLAG.AUX_LANE; if (e.laneDrop) fl |= FLAG.LANE_DROP;
      E[8 * k + 6] = fl;
      E[8 * k + 7] = e.twinRef ? local(e.twinRef) : NO_TWIN;
      len[k] = Math.round(e.length * 100) / 100;
      lw[k] = o.laneWidth ?? LANE_WIDTH_BY_CLASS[o.cls];
      const W = (e.lanes + e.back) * 3.7;
      jsonEdges.push({ id: k, from: idx[e.nodes[0]], to: idx[e.nodes[e.nodes.length - 1]], nodes: e.nodes.map((n) => idx[n]), class: o.cls, name: o.name ?? null, ref: o.ref ?? null,
        oneway: o.oneway, twin: e.twinRef ? local(e.twinRef) : -1, lanes: e.lanes, oppLanes: e.back,
        laneOffsets: Array.from({ length: e.lanes }, (_, j) => +(((e.back - e.lanes) / 2 + j + 0.5) * 3.7).toFixed(2)), width: +W.toFixed(2),
        lanesFallback: false, lanesRaw: e.lanes, oppLanesRaw: e.back, laneWidth: lw[k], sidewalk: null, parking: null, laneFlags: fl & (FLAG.AUX_LANE | FLAG.LANE_DROP),
        bridge: !!o.bridge, tunnel: !!o.tunnel, layer: o.layer ?? (o.bridge ? 1 : 0), maxspeed: o.speedKph ?? null, length: len[k], way: e.roadIdx + 1 });
    });
    // CSR over this tier's own edges
    const off = new Uint32Array(nN + 1);
    edges.forEach((e) => off[idx[e.nodes[0]] + 1]++);
    for (let i = 0; i < nN; i++) off[i + 1] += off[i];
    const fill = off.slice(0, nN), csr = new Uint32Array(nE);
    edges.forEach((e, k) => { csr[fill[idx[e.nodes[0]]]++] = k; });
    const sections = [['nodeXYZ', xyz, 3], ['edges', E, 8], ['poly', poly, 1], ['edgeLength', len, 1], ['csrOffsets', off, 1], ['csrEdges', csr, 1], ['edgeLaneWidth', lw, 1]];
    if (t === 1) {
      const link = [];
      nodes.forEach((i, k) => { if (tierIdx[0][i] >= 0) link.push(k, tierIdx[0][i]); });
      sections.push(['linkNodes', Uint32Array.from(link), 2]);
    } else sections.push(['bannedHops', new Uint32Array(0), 3]);
    const meta = JSON.parse(JSON.stringify(META_TEMPLATE[tierName]));
    meta.generated = new Date(0).toISOString().replace('.000', '');   // deterministic bytes
    meta.scenario = this.name;
    let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
    for (let k = 0; k < nN; k++) { minX = Math.min(minX, xyz[3 * k]); maxX = Math.max(maxX, xyz[3 * k]); minZ = Math.min(minZ, xyz[3 * k + 2]); maxZ = Math.max(maxZ, xyz[3 * k + 2]); }
    meta.bbox = nN ? { minX: +minX.toFixed(2), maxX: +maxX.toFixed(2), minZ: +minZ.toFixed(2), maxZ: +maxZ.toFixed(2) } : { minX: 0, maxX: 0, minZ: 0, maxZ: 0 };
    meta.nodeCount = nN; meta.edgeCount = nE;
    meta.file = t === 0 ? 'roads.bin' : 'roads-local.bin'; meta.json = t === 0 ? 'roads.json' : 'roads-local.json';
    if (t === 0) meta.stations = 'stations.meta.json';
    const { bin, secs } = packSections(sections);
    meta.byteLength = bin.byteLength; meta.sections = secs;
    const json = { meta: { format: 'bay-roads/2', tier: tierName, classes: meta.classes, classIds: meta.classIds, projection: meta.projection, laneWidth: 3.7,
      laneDefaults: meta.laneDefaults, defaultSpeedKph: meta.defaultSpeedKph, elevation: meta.elevation, laneWidthByClass: meta.laneWidthByClass, bbox: meta.bbox, nodeCount: nN, edgeCount: nE, scenario: this.name },
    nodes: nodes.map((i) => ({ id: i + 1, x: +this.nx[i].toFixed(2), y: +this.ny[i].toFixed(2), z: +this.nz[i].toFixed(2) })), edges: jsonEdges };
    if (t === 0) json.bannedHops = [];
    return { meta, bin, json, arrays: Object.fromEntries(sections.map(([n, a]) => [n === 'nodeXYZ' ? 'xyz' : n, a])) };
  }

  _controls(allEdges, inOf, nodeMap, tierIdx, mainEdgeCount) {
    const recs = [];
    const half = (e) => ((e.lanes + e.back) * (e.o.laneWidth ?? LANE_WIDTH_BY_CLASS[e.o.cls])) / 2;
    for (const c of this.controls) {
      let aps = [], node = -1, tier = null, x, y, z;
      if (c.midBlock) {
        const e = allEdges.find((q) => q.nodes[0] === c.midBlock.from && q.nodes[q.nodes.length - 1] === c.midBlock.to);
        if (!e) throw new Error('control midBlock edge not found');
        aps = [{ e, stopAt: c.midBlock.at }];
        const pt = pointAlongNodes(this, e.nodes, c.midBlock.at); x = pt.x; y = pt.y; z = pt.z;
      } else {
        const ins = inOf.get(c.node) || [];
        const sel = c.approaches === 'all' ? ins : ins.filter((e) => c.approaches.includes(e.nodes[e.nodes.length - 2]) || c.approaches.includes(firstNodeBack(this, e)));
        const widest = Math.max(...ins.map(half));
        aps = sel.map((e) => ({ e, stopAt: Math.max(1, e.length - (widest + 1.5)) }));
        node = c.node; tier = tierIdx[0][node] >= 0 ? 'main' : 'local';
        x = this.nx[node]; y = this.ny[node]; z = this.nz[node];
      }
      recs.push({ id: recs.length, type: c.type, tier, node: node >= 0 ? (tier === 'main' ? tierIdx[0][node] : tierIdx[1][node]) : -1, mergedNode: node >= 0 ? nodeMap[node] : -1,
        junctionOsm: null, osmNodes: [], x: +x.toFixed(2), y: +y.toFixed(2), z: +z.toFixed(2), allWay: !!c.allWay, rampMeter: c.type === 'ramp_meter', midBlock: !!c.midBlock, crossing: c.type === 'crossing_signal',
        approaches: aps.map(({ e, stopAt }) => ({ edge: e.id, tier: e.tier === 0 ? 'main' : 'local', tierEdge: e.tier === 0 ? e.id : e.id - mainEdgeCount, stopAt: Math.round(stopAt * 100) / 100,
          fromOsm: null, toOsm: null, class: e.o.cls })) });
    }
    const nA = recs.reduce((s, r) => s + r.approaches.length, 0);
    const info = new Uint32Array(6 * recs.length), xyz = new Float32Array(3 * recs.length), apE = new Uint32Array(nA), apS = new Float32Array(nA);
    let a = 0;
    recs.forEach((r, k) => {
      const flags = (r.allWay ? 1 : 0) | (r.rampMeter ? 2 : 0) | (r.midBlock ? 4 : 0);
      info.set([CONTROL_TYPES.indexOf(r.type), flags, r.tier === 'main' ? 0 : r.tier === 'local' ? 1 : 0xFFFFFFFF, r.node >= 0 ? r.node : 0xFFFFFFFF, a, r.approaches.length], 6 * k);
      xyz.set([r.x, r.y, r.z], 3 * k);
      for (const ap of r.approaches) { apE[a] = ap.edge; apS[a] = ap.stopAt; a++; }
    });
    const byType = {}; for (const r of recs) byType[r.type] = (byType[r.type] || 0) + 1;
    const { bin, secs } = packSections([['controlInfo', info, 6], ['controlXYZ', xyz, 3], ['approachEdge', apE, 1], ['approachStopAt', apS, 1]]);
    const meta = { format: 'bay-controls/1', count: recs.length, types: CONTROL_TYPES, flagBits: { ALL_WAY: 1, RAMP_METER: 2, MIDBLOCK: 4 },
      edgeIds: 'merged graph: main e, local mainEdgeCount + e (= src/sim fromRoadSet, src/geom)', mainEdgeCount, byType,
      allWay: recs.filter((r) => r.allWay).length, midBlock: recs.filter((r) => r.midBlock).length, source: 'synthetic scenario (src/scenarios)', build: {},
      file: 'controls.bin', json: 'controls.json', byteLength: bin.byteLength, sections: secs };
    for (const r of recs) delete r.mergedNode;
    const json = { meta: { format: 'bay-controls/1', count: recs.length, types: CONTROL_TYPES, byType, mainEdgeCount, build: {} }, controls: recs };
    return { meta, bin, json };
  }

  _stations(tierIdx) {
    const xyz = new Float32Array(3 * this.stations.length), info = new Uint32Array(4 * this.stations.length), list = [];
    this.stations.forEach((s, k) => {
      const tier = tierIdx[0][s.node] >= 0 ? 0 : 1, ti = tierIdx[tier][s.node];
      xyz.set([this.nx[s.node], this.ny[s.node], this.nz[s.node]], 3 * k); info.set([ti, tier, s.pumps, 0], 4 * k);
      list.push({ id: k, tier: tier ? 'local' : 'main', node: ti, osmNode: null, x: this.nx[s.node], z: this.nz[s.node], y: this.ny[s.node], pumps: s.pumps, nearInterchange: false });
    });
    const { bin, secs } = packSections([['stationXYZ', xyz, 3], ['stationInfo', info, 4]]);
    const meta = { format: 'bay-stations-bin/1', file: 'stations.bin', byteLength: bin.byteLength, endianness: 'little', count: list.length, sections: secs,
      stationInfoFields: ['nodeIndex', 'tierId', 'pumps', 'flags'], tierIds: { main: 0, local: 1 }, flagBits: { NEAR_INTERCHANGE: 1 }, json: 'stations.json' };
    const json = { meta: { format: 'bay-stations/1', count: list.length, seed: this.seed, tiers: ['main', 'local'], tierIds: { main: 0, local: 1 } }, stations: list };
    return { meta, bin, json };
  }

  /** Plan crossings without a shared node where one road is on the freeway system (as tools/build_crossings.py). */
  _crossings(allEdges, mainEdgeCount) {
    const segs = [];
    for (const e of allEdges) {
      if (e.twinRef && e.twinRef.id < e.id) continue;   // one direction per road
      for (let k = 1; k < e.nodes.length; k++) segs.push({ e, k, s0: 0 });
    }
    // cumulative s per segment start
    for (const e of allEdges) { e._cum = [0]; for (let k = 1; k < e.nodes.length; k++) e._cum.push(e._cum[k - 1] + Math.hypot(this.nx[e.nodes[k]] - this.nx[e.nodes[k - 1]], this.nz[e.nodes[k]] - this.nz[e.nodes[k - 1]])); }
    const out = [], seen = new Set();
    for (let i = 0; i < segs.length; i++) for (let j = i + 1; j < segs.length; j++) {
      const A = segs[i], B = segs[j];
      if (A.e === B.e) continue;
      const ea = A.e, eb = B.e;
      if (!FREEWAY_SYS.has(ea.o.cls) && !FREEWAY_SYS.has(eb.o.cls)) continue;
      const ends = [ea.nodes[0], ea.nodes[ea.nodes.length - 1]], endsB = [eb.nodes[0], eb.nodes[eb.nodes.length - 1]];
      if (ends.some((n) => endsB.includes(n))) continue;
      const a0 = ea.nodes[A.k - 1], a1 = ea.nodes[A.k], b0 = eb.nodes[B.k - 1], b1 = eb.nodes[B.k];
      const hit = segX(this.nx[a0], this.nz[a0], this.nx[a1], this.nz[a1], this.nx[b0], this.nz[b0], this.nx[b1], this.nz[b1]);
      if (!hit) continue;
      const key = ea.id + ':' + eb.id; if (seen.has(key)) continue; seen.add(key);
      const ya = this.ny[a0] + (this.ny[a1] - this.ny[a0]) * hit.t, yb = this.ny[b0] + (this.ny[b1] - this.ny[b0]) * hit.u;
      const sa = ea._cum[A.k - 1] + hit.t * (ea._cum[A.k] - ea._cum[A.k - 1]), sb = eb._cum[B.k - 1] + hit.u * (eb._cum[B.k] - eb._cum[B.k - 1]);
      const dy = Math.abs(ya - yb);
      out.push({ a: ea.id, aTwin: ea.twinRef ? ea.twinRef.id : -1, sa: +sa.toFixed(2), aClass: ea.o.cls, b: eb.id, bTwin: eb.twinRef ? eb.twinRef.id : -1, sb: +sb.toFixed(2), bClass: eb.o.cls,
        x: +hit.x.toFixed(1), z: +hit.z.toFixed(1), ya: +ya.toFixed(2), yb: +yb.toFixed(2), dy: +dy.toFixed(2), reason: dy < 4.5 ? 'at-grade (no OSM junction)' : null });
    }
    for (const e of allEdges) delete e._cum;
    const below = out.filter((c) => c.dy < 4.5).length;
    return { meta: { format: 'bay-crossings/1', count: out.length, below4_5m: below, byReason: below ? { 'at-grade (no OSM junction)': below } : {}, edgeIds: 'merged graph: main e, local mainEdgeCount + e', mainEdgeCount,
      note: 'synthetic scenario: all crossings in plan with no shared node that involve the freeway system' }, crossings: out, overlaps: [] };
  }
}

function firstNodeBack(w, e) { return e.nodes[0]; }
function pointAlongNodes(w, nodes, s) {
  let acc = 0;
  for (let k = 1; k < nodes.length; k++) {
    const a = nodes[k - 1], b = nodes[k], L = Math.hypot(w.nx[b] - w.nx[a], w.nz[b] - w.nz[a]);
    if (acc + L >= s || k === nodes.length - 1) { const t = L > 0 ? Math.min(1, Math.max(0, (s - acc) / L)) : 0; return { x: w.nx[a] + (w.nx[b] - w.nx[a]) * t, y: w.ny[a] + (w.ny[b] - w.ny[a]) * t, z: w.nz[a] + (w.nz[b] - w.nz[a]) * t }; }
    acc += L;
  }
  return { x: w.nx[nodes[0]], y: w.ny[nodes[0]], z: w.nz[nodes[0]] };
}
function segX(ax, az, bx, bz, cx, cz, dx, dz) {
  const rx = bx - ax, rz = bz - az, sx = dx - cx, sz = dz - cz, den = rx * sz - rz * sx;
  if (Math.abs(den) < 1e-9) return null;
  const t = ((cx - ax) * sz - (cz - az) * sx) / den, u = ((cx - ax) * rz - (cz - az) * rx) / den;
  if (t < 0 || t > 1 || u < 0 || u > 1) return null;
  return { t, u, x: ax + rx * t, z: az + rz * t };
}

/** Pack typed arrays into one little-endian blob with 4-byte-aligned sections (roads.schema.md convention). */
export function packSections(list) {
  let off = 0; const secs = {}, parts = [];
  for (const [name, arr, stride] of list) {
    const pad = (4 - (off % 4)) % 4; off += pad; if (pad) parts.push(new Uint8Array(pad));
    const type = arr instanceof Float32Array ? 'Float32' : arr instanceof Uint32Array ? 'Uint32' : arr instanceof Int32Array ? 'Int32' : 'Uint8';
    secs[name] = { byteOffset: off, byteLength: arr.byteLength, type, length: arr.length, stride, count: stride ? arr.length / stride : arr.length };
    parts.push(new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength)); off += arr.byteLength;
  }
  const bin = new Uint8Array(off); let p = 0;
  for (const q of parts) { bin.set(q, p); p += q.byteLength; }
  return { bin: bin.buffer, secs };
}
