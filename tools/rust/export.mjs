// Export a loaded world (SimGraph + src/geom lane geometry + junctions + controls + stations) and a population
// into BAYPACK1 files (tools/rust/pack.mjs) for the Rust port (rust/bayenv-sim). The world pack is a
// snapshot of exactly what the JS sim sees after loading, so the Rust sim can be checked against the JS
// one without depending on the world builders; the same pack is a reference for the Rust world loaders.
import { PackWriter } from './pack.mjs';
import { hopMap as simHopMap } from '../../src/sim/roadGraph.js';

const f64 = (n) => new Float64Array(n), i32 = (n) => new Int32Array(n), u8 = (n) => new Uint8Array(n);

export function exportWorld(graph, { stations = [], controls = null, path, extraMeta = {} }) {
  const w = new PackWriter(), E = graph.edges.length, N = graph.nodeCount;
  const names = [];
  const hw = (s) => { let k = names.indexOf(s); if (k < 0) { names.push(s); k = names.length - 1; } return k; };
  w.add('nx', Float64Array.from(graph.nx)).add('nz', Float64Array.from(graph.nz));
  w.add('nodeY', Float32Array.from(graph.nodeY || new Float32Array(N)));
  const from = i32(E), to = i32(E), rev = i32(E), lanes = u8(E), opp = i32(E), speed = f64(E), length = f64(E), rank = i32(E), cls = u8(E);
  const flags = u8(E), hS = f64(E), hE = f64(E), ey = f64(E), right = f64(E), sExit = f64(E), sStop = f64(E), gore = f64(E);
  const surf = u8(E), gndR = u8(E), gndL = u8(E), tier = u8(E);
  const vStart = new Uint32Array(E + 1), lStart = new Uint32Array(E + 1);
  let nv = 0, nl = 0;
  for (const e of graph.edges) { nv += e.px.length; nl += e.laneOff.length; }
  const px = f64(nv), pz = f64(nv), cum = f64(nv), ys = new Float32Array(nv), laneOff = new Float32Array(nl);
  let v = 0, l = 0;
  for (let i = 0; i < E; i++) {
    const e = graph.edges[i];
    if (e.idx !== i) throw new Error('edge idx mismatch');
    from[i] = e.from; to[i] = e.to; rev[i] = e.reverse; lanes[i] = e.lanes; opp[i] = e.oppLanes ?? 0; speed[i] = e.speedLimit; length[i] = e.length;
    rank[i] = e.rank; cls[i] = hw(e.highway);
    flags[i] = (e.isLink ? 1 : 0) | (e.oneway ? 2 : 0) | (e.bridge ? 4 : 0) | (e.tunnel ? 8 : 0) | (e.ys ? 16 : 0);
    hS[i] = e.headingStart; hE[i] = e.headingEnd; ey[i] = e.y; right[i] = e.rightEdge;
    sExit[i] = e.sExit ?? NaN; sStop[i] = e.sStop ?? NaN; gore[i] = e.gore ?? NaN;
    surf[i] = e.surf ?? 0; gndR[i] = e.gndR ?? 0; gndL[i] = e.gndL ?? 0; tier[i] = e.tier === 'local' ? 1 : 0;
    vStart[i] = v;
    for (let k = 0; k < e.px.length; k++, v++) { px[v] = e.px[k]; pz[v] = e.pz[k]; cum[v] = e.cum[k]; ys[v] = e.ys ? e.ys[k] : 0; }
    lStart[i] = l; for (let k = 0; k < e.laneOff.length; k++) laneOff[l++] = e.laneOff[k];
  }
  vStart[E] = v; lStart[E] = l;
  w.add('eFrom', from).add('eTo', to).add('eRev', rev).add('eLanes', lanes).add('eOpp', opp).add('eSpeed', speed).add('eLength', length)
    .add('eRank', rank).add('eClass', cls).add('eFlags', flags).add('eHeadStart', hS).add('eHeadEnd', hE).add('eY', ey).add('eRight', right)
    .add('eSExit', sExit).add('eSStop', sStop).add('eGore', gore).add('eSurf', surf).add('eGndR', gndR).add('eGndL', gndL).add('eTier', tier)
    .add('vStart', vStart).add('px', px).add('pz', pz).add('cum', cum).add('ys', ys).add('lStart', lStart).add('laneOff', laneOff)
    .add('outStart', Int32Array.from(graph.outStart)).add('outEdges', Int32Array.from(graph.outEdges))
    .add('inStart', Int32Array.from(graph.inStart)).add('inEdges', Int32Array.from(graph.inEdges));
  if (graph.bannedHops) { const t = []; for (const [k, list] of graph.bannedHops) for (const x of list) t.push(Math.floor(k / 4194304), k % 4194304, x); w.add('bannedHops', Int32Array.from(t)); }
  if (graph.contactPairs) w.add('contactPairs', Float64Array.from([...graph.contactPairs].sort((a, b) => a - b)));
  // hop table for every (in, out) pair at every node (reference for the Rust hopMap port)
  const hops = [];
  for (let n = 0; n < N; n++) for (let a = graph.inStart[n]; a < graph.inStart[n + 1]; a++) for (let b = graph.outStart[n]; b < graph.outStart[n + 1]; b++) {
    const e = graph.edges[graph.inEdges[a]], en = graph.edges[graph.outEdges[b]], h = simHopMap(graph, e, en);
    hops.push(e.idx, en.idx, h.strict ? 1 : 0, h.shift);
  }
  w.add('hopRef', Int32Array.from(hops));
  const meta = { kind: 'world', nodeCount: N, edgeCount: E, highway: names, maxSpeed: graph.maxSpeed, bounds: graph.bounds, laneGeom: !!graph.laneGeom, junctions: !!graph.J, surfaces: !!graph.surfaces, ...extraMeta };
  // lane geometry (src/geom LaneGeometry + its LaneGraph)
  const LG = graph.laneGeom;
  if (LG) {
    for (const [k, a] of Object.entries(LG)) if (ArrayBuffer.isView(a)) w.add('LG.' + k, a);
    for (const [k, a] of Object.entries(LG.G)) if (ArrayBuffer.isView(a)) w.add('G.' + k, a);
    meta.LG = { LW: LG.LW, classes: ['motorway', 'motorway_link', 'trunk', 'trunk_link', 'primary', 'primary_link', 'secondary', 'secondary_link', 'tertiary', 'tertiary_link', 'unclassified', 'residential', 'living_street'] };
  }
  const J = graph.J;
  if (J) { for (const k of ['mouthS', 'goreS', 'jNode', 'jFirstM', 'jCountM', 'jFirstC', 'jCountC', 'mInfo', 'mF', 'pts', 'conf', 'confS', 'mLen']) w.add('J.' + k, J[k]); }
  // traffic controls: raw pack inputs are not kept by TrafficControls, so export its derived arrays (+ phases flattened)
  if (controls) {
    const C = controls;
    for (const k of ['type', 'flags', 'cycle', 'offset', 'apControl', 'apStopAt', 'apEdge', 'apGroup', 'apProtLeft', 'edgeApproach', 'clusterOf', 'edgeApStart', 'edgeApList']) w.add('C.' + k, C[k]);
    const pStart = new Uint32Array(C.count + 1), pf = [];
    for (let c = 0; c < C.count; c++) { pStart[c] = pf.length / 5; const ph = C.phases[c]; if (ph) for (const p of ph) pf.push(p.group, p.left ? 1 : 0, p.g, p.y, p.r); }
    pStart[C.count] = pf.length / 5;
    w.add('C.phaseStart', pStart).add('C.phases', Float64Array.from(pf));
    meta.controls = { count: C.count, approachCount: C.approachCount, rampStart: C.rampStart, rampEnd: C.rampEnd };
  }
  const S = stations.length;
  w.add('stNode', Int32Array.from(stations.map((s) => s.node))).add('stPumps', Int32Array.from(stations.map((s) => s.pumps)))
    .add('stX', Float64Array.from(stations.map((s) => s.x))).add('stZ', Float64Array.from(stations.map((s) => s.z)));
  meta.stations = S;
  w.meta = meta;
  w.write(path);
  return meta;
}

/** Population (generatePopulation / scenarioPopulation result) -> BAYPACK1. */
export function exportPopulation(pop, path, extraMeta = {}) {
  const w = new PackWriter(), C = pop.commuters, n = C.length;
  const num = (f) => Float64Array.from(C.map(f));
  w.add('vehicleType', Uint8Array.from(C.map((c) => c.vehicleType))).add('home', Int32Array.from(C.map((c) => c.home))).add('work', Int32Array.from(C.map((c) => c.work)))
    .add('depart', num((c) => c.depart)).add('returnDepart', num((c) => c.returnDepart)).add('workStart', num((c) => c.workStart))
    .add('fuelFill', num((c) => c.fuelFill ?? 1)).add('L', num((c) => c.L)).add('expectedTravel', num((c) => c.expectedTravel))
    .add('routeFlat', Int32Array.from(pop.routes.flat)).add('routeOffset', Uint32Array.from(pop.routes.offset)).add('deltas', Float32Array.from(pop.deltas));
  w.meta = { kind: 'population', n, ...extraMeta };
  w.write(path);
}
