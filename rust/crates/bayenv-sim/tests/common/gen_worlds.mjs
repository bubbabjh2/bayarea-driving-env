// Fixture generator for the js_port_*.rs cargo tests (see tests/common/mod.rs). Rust has no synthGraph / buildGraph port, so the
// worlds the JS unit tests build in-process (test/sim/helpers.js smallGraph, collisions.test.js road(), physics.test.js lane-change
// road, a controls fixture) are built here with the JS sim and exported to world.pack files (tools/rust/export.mjs).
//   node gen_worlds.mjs <outDir>
// Writes <outDir>/{synth,synth40,road3,road30k,surfcar,ctl}/world.pack and <outDir>/fixtures.json (edge indices etc.).
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../../..');
const imp = (p) => import(path.join(root, p));
const { buildGraph } = await imp('src/sim/roadGraph.js');
const { makeSynthGraph } = await imp('src/sim/synthGraph.js');
const { exportWorld } = await imp('tools/rust/export.mjs');
const { TrafficControls, CTRL } = await imp('src/sim/controls.js');

const out = process.argv[2];
fs.mkdirSync(out, { recursive: true });
const meta = {};
const write = (name, graph, extra = {}) => {
  fs.mkdirSync(`${out}/${name}`, { recursive: true });
  exportWorld(graph, { path: `${out}/${name}/world.pack`, ...extra });
};

const smallGraph = (o = {}) => buildGraph(makeSynthGraph({ blocks: 12, cross: 6, bridges: 2, ...o }));
write('synth', smallGraph());
write('synth40', smallGraph({ blocks: 40, cross: 4 }));

// collisions.test.js road(3)
write('road3', buildGraph({ nodes: [{ id: 'a', x: 0, z: 0 }, { id: 'b', x: 3000, z: 0 }, { id: 'c', x: 3000, z: 500 }],
  edges: [{ nodes: ['a', 'b'], highway: 'motorway', oneway: true, lanes: 3, maxspeed: '65 mph' }, { nodes: ['b', 'c'], highway: 'residential', oneway: false, lanes: 1, maxspeed: '25 mph' }] }));
// physics.test.js lane-change road
write('road30k', buildGraph({ nodes: [{ id: 'a', x: 0, z: 0 }, { id: 'b', x: 30000, z: 0 }], edges: [{ nodes: ['a', 'b'], highway: 'motorway', oneway: true, lanes: 3, maxspeed: '65 mph' }] }));
// surface.test.js carSurface: a 2-lane one-way motorway (edge 0) + a two-way 1-lane residential pair (edges 1, 2)
const sc = buildGraph({ nodes: [{ id: 'a', x: 0, z: 0 }, { id: 'b', x: 500, z: 0 }, { id: 'c', x: 0, z: 100 }, { id: 'd', x: 200, z: 100 }],
  edges: [{ nodes: ['a', 'b'], highway: 'motorway', oneway: true, lanes: 2, maxspeed: '65 mph' }, { nodes: ['c', 'd'], highway: 'residential', oneway: false, lanes: 1, maxspeed: '25 mph' }] });
write('surfcar', sc);
meta.surfcar = { edges: sc.edges.map((e) => ({ idx: e.idx, highway: e.highway, reverse: e.reverse, lanes: e.lanes, length: e.length })) };

// env.test.js traffic-controls fixture: a signal on every in-edge of one 4-way junction of the synthetic grid
{
  const g = smallGraph();
  let node = -1;
  for (let n = 0; n < g.nodeCount && node < 0; n++) if (g.inStart[n + 1] - g.inStart[n] === 4) node = n;
  const ins = Array.from(g.inEdges.subarray(g.inStart[node], g.inStart[node + 1]));
  const pack = { info: Uint32Array.from([CTRL.SIGNAL, 0, 0, node, 0, ins.length]), xyz: Float32Array.from([g.nx[node], 0, g.nz[node]]),
    approachEdge: Uint32Array.from(ins), approachStopAt: Float32Array.from(ins.map((e) => g.edges[e].length - 2)), meta: {} };
  const C = new TrafficControls(g, pack);
  write('ctl', g, { controls: C });
  meta.ctl = { node, ins, cycle: C.cycle[0] };
}
fs.writeFileSync(`${out}/fixtures.json`, JSON.stringify(meta));
