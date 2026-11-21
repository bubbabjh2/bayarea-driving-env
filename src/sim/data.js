// Real-data loading for the sim, in node (fs) or the browser (fetch):
// the world's binary road packs (data/roads.bin + data/roads-local.bin, schema in
// data/roads.schema.md) merged into one two-tier SimGraph through the local tier's
// linkNodes table, plus the gas stations (data/stations.bin) mapped onto graph nodes.
//
//   const { graph, stations, main, local } = await loadWorld({ base: 'data/' });
//   stations = [{id, node (SimGraph node), x, y, z, pumps, nearInterchange, tier}]

import { RoadSet } from '../roadpack.js';
import { fromRoadSet, attachLaneGeometry, attachJunctions } from './roadGraph.js';
import { ensureLaneGeometry, buildJunctions, JunctionSet } from '../geom/index.js';
import { TrafficControls, loadControlsPack } from './controls.js';
import { attachSurfaces } from './surface.js';

const TYPES = { Float32: Float32Array, Uint32: Uint32Array, Uint16: Uint16Array, Uint8: Uint8Array, Int32: Int32Array };
const isNode = typeof process !== 'undefined' && !!process.versions?.node && typeof window === 'undefined';

async function readJSON(url) {
  if (isNode) { const fs = await import('node:fs/promises'); return JSON.parse(await fs.readFile(toPath(url), 'utf8')); }
  const r = await fetch(url); if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`); return r.json();
}
async function readBin(url) {
  if (isNode) { const fs = await import('node:fs/promises'); const b = await fs.readFile(toPath(url)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); }
  const r = await fetch(url); if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`); return r.arrayBuffer();
}
function toPath(url) { return url instanceof URL ? url.pathname : String(url); }
function join(base, f) { return base instanceof URL ? new URL(f, base) : (base.endsWith('/') || base === '' ? base + f : base + '/' + f); }

/** Default data directory: <repo>/data/ relative to this module. */
export function defaultDataBase() { return new URL('../../data/', import.meta.url); }

function sections(meta, buf) {
  const out = {};
  for (const [name, s] of Object.entries(meta.sections)) out[name === 'nodeXYZ' ? 'xyz' : name] = new TYPES[s.type](buf, s.byteOffset, s.length);
  return out;
}

export async function loadRoadPackAny(base, metaFile) {
  const meta = await readJSON(join(base, metaFile));
  const buf = await readBin(join(base, meta.file));
  return new RoadSet(meta, sections(meta, buf));
}

export async function loadStationPack(base, metaFile = 'stations.meta.json') {
  const meta = await readJSON(join(base, metaFile));
  const buf = await readBin(join(base, meta.file));
  const s = sections(meta, buf);
  return { meta, count: meta.count, xyz: s.stationXYZ, info: s.stationInfo };
}

/**
 * Map stations onto SimGraph nodes. Station node indices are per tier; the merged
 * graph keeps main-tier node indices and maps local nodes through graph.tierNodeMap.
 */
export function stationsOnGraph(pack, graph) {
  const out = [];
  if (!pack) return out;
  for (let i = 0; i < pack.count; i++) {
    const x = pack.xyz[i * 3], y = pack.xyz[i * 3 + 1], z = pack.xyz[i * 3 + 2];
    const tierNode = pack.info[i * 4], tier = pack.info[i * 4 + 1], pumps = pack.info[i * 4 + 2], flags = pack.info[i * 4 + 3];
    let node = graph.tierNodeMap && graph.tierNodeMap[tier] ? graph.tierNodeMap[tier][tierNode] : -1;
    if (!(node >= 0) || Math.hypot(graph.nx[node] - x, graph.nz[node] - z) > 5) node = graph.nearestNode(x, z, (k) => graph.outDegree(k) > 0 && graph.inDegree(k) > 0);
    if (node < 0) continue;
    out.push({ id: i, node, x, y, z, pumps: Math.max(1, pumps), nearInterchange: (flags & 1) !== 0, tier: tier === 0 ? 'main' : 'local' });
  }
  return out;
}

/**
 * Junctions for lane geometry LG: data/junctions.bin (tools/build_junctions.mjs: crosswalk set-backs,
 * the same sStop as data/controls) when it matches LG's edge count, else built in memory.
 */
async function loadJunctionPack(base, LG) {
  try {
    const meta = await readJSON(join(base, 'junctions.meta.json'));
    const buf = await readBin(join(base, meta.file));
    const T = { Float32: Float32Array, Uint32: Uint32Array }, a = {};
    const ab = buf.buffer ? buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) : buf;
    for (const [k, sec] of Object.entries(meta.sections)) a[k] = new T[sec.type](ab, sec.byteOffset, sec.length);
    if (a.mouthS.length === 2 * LG.G.edgeCount) return new JunctionSet(LG.G, LG, a);
    console.warn('data/junctions.bin does not match the road graph: building junctions in memory');
  } catch (e) { /* no pack: build */ }
  return buildJunctions(LG);
}

/** Load both road tiers (local optional), the stations and the traffic controls; returns {graph, stations, controls, main, local}. */
export async function loadWorld({ base = defaultDataBase(), local = true, stations = true, controls = true, geom: opts_geom = true, junctions = true } = {}) {
  const main = await loadRoadPackAny(base, 'roads.meta.json');
  let loc = null;
  if (local) { try { loc = await loadRoadPackAny(base, 'roads-local.meta.json'); } catch (e) { /* optional tier */ } }
  const graph = fromRoadSet(main, loc);
  // shared lane geometry (src/geom, world-owned): mitred / filleted lanes, tapers, strict-hop lane mapping
  if (opts_geom !== false) {
    const LG = ensureLaneGeometry(main, loc);
    attachLaneGeometry(graph, LG);
    // physical junctions (src/geom junctions.js): mouths (stop / exit points), connectors, conflicts, gores
    if (junctions !== false && graph.laneGeom) {
      const key = loc ? 'junctionsMerged' : 'junctionsMain';
      if (!main[key]) main[key] = await loadJunctionPack(base, LG);
      attachJunctions(graph, main[key]);
    }
  }
  // road surfaces (data/surfaces.bin, tools/surfaces/build.mjs): per-edge pavement + roadside ground codes for the
  // tyre friction (src/sim/surface.js). Optional: without it every edge is dry asphalt.
  try {
    const sm = await readJSON(join(base, 'surfaces.meta.json'));
    if (sm.edgeCount && sm.edgeCount.main === main.edgeCount && (!loc || sm.edgeCount.local === loc.edgeCount)) {
      attachSurfaces(graph, new Uint8Array(await readBin(join(base, sm.file))).subarray(0, 2 * graph.edges.length));
    }
  } catch (e) { /* optional */ }
  let st = [];
  if (stations) { try { st = stationsOnGraph(await loadStationPack(base), graph); } catch (e) { /* optional */ } }
  // grade separations (data/crossings.json): freeway-system edge pairs that cross in plan without a shared
  // node can only touch where the build reports them at (nearly) the same height, or as pavement overlaps
  try {
    const cr = await readJSON(join(base, 'crossings.json'));
    if (cr.meta && cr.meta.mainEdgeCount === main.edgeCount) {
      const set = new Set(), key = (a, b) => (a < b ? a * 4194304 + b : b * 4194304 + a);
      const add = (a, b) => { if (a >= 0 && b >= 0) set.add(key(a, b)); };
      for (const c of cr.crossings) if (c.dy < 2.5) for (const a of [c.a, c.aTwin]) for (const b of [c.b, c.bTwin]) add(a, b);
      for (const o of cr.overlaps || []) add(o.a, o.b);
      graph.contactPairs = set; graph.contactPairKey = key; graph.crossingsLoaded = cr.crossings.length;
    }
  } catch (e) { /* optional */ }
  let ctl = null;
  if (controls) {
    try {
      const pack = await loadControlsPack(base, (b, f) => readJSON(join(b, f)), (b, f) => readBin(join(b, f)));
      if (pack.meta.mainEdgeCount === main.edgeCount) ctl = new TrafficControls(graph, pack);
      else console.warn(`controls pack was built for ${pack.meta.mainEdgeCount} main edges, roads.bin has ${main.edgeCount}: controls ignored`);
    } catch (e) { /* optional */ }
  }
  return { graph, stations: st, controls: ctl, main, local: loc };
}
