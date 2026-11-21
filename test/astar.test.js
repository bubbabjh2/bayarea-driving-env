import { test } from 'node:test';
import assert from 'node:assert/strict';
import { loadGraph } from '../src/nav/graph.js';
import { Router } from '../src/nav/astar.js';
import { makeGridGraph } from '../src/nav/synthetic.js';
import { synthetic, idx } from './helpers.js';

test('A* picks the fastest path, not the shortest', () => {
  // A -> B direct is 1000 m of residential (25 mph); A -> C -> B is 1400 m of motorway (65 mph)
  const g = loadGraph({
    nodes: [{ id: 'A', x: 0, z: 0 }, { id: 'B', x: 1000, z: 0 }, { id: 'C', x: 500, z: 500 }],
    edges: [
      { nodes: ['A', 'B'], highway: 'residential', oneway: true },
      { nodes: ['A', 'C'], highway: 'motorway' }, { nodes: ['C', 'B'], highway: 'motorway' },
    ],
  });
  const r = new Router(g);
  const res = r.route(g.indexOf('A'), g.indexOf('B'));
  assert.ok(res);
  assert.deepEqual(res.nodes.map((i) => g.nodes[i].id), ['A', 'C', 'B']);
  assert.ok(Math.abs(res.length - 2 * Math.hypot(500, 500)) < 1e-6);
  assert.ok(res.time < 1000 / (25 * 0.44704));
  assert.equal(r.route(g.indexOf('B'), g.indexOf('A')), null, 'no reverse path on one-way edges');
  assert.equal(r.route(0, 0).edges.length, 0);
});

test('routeVia concatenates legs and nearestNode uses the spatial grid', () => {
  const { graph: g, router: r } = synthetic();
  const res = r.routeVia([idx(g, 'E0'), idx(g, 'MS200'), idx(g, 'E20')]);
  assert.ok(res && res.edges.length > 5);
  const names = res.edges.map((e) => g.edges[e].name);
  assert.ok(names.includes('Main St exit') && names.includes('US 101 N on-ramp'));
  const n = r.nearestNode(5010, 20);
  assert.equal(g.nodes[n].id, 'E5');
  const nIn = r.nearestNode(5390, 2010, { requireOut: false, requireIn: true });
  assert.equal(g.nodes[nIn].id, 'MS2000');
  assert.equal(r.nearestNode(1e9, 1e9, { maxDist: 100 }), -1);
});

test('A* on a 100k-node grid finishes well under a second', () => {
  const g = loadGraph(makeGridGraph(320, 320, 100));
  assert.ok(g.nodeCount() >= 100000);
  const r = new Router(g);
  const t0 = performance.now();
  const res = r.route(0, g.nodeCount() - 1);
  const ms = performance.now() - t0;
  assert.ok(res, 'route found');
  assert.ok(Math.abs(res.length - (319 + 319) * 100) < 1e-6, 'manhattan length');
  assert.ok(ms < 1000, `took ${ms.toFixed(1)} ms`);
  const t1 = performance.now();
  const res2 = r.route(160 * 320 + 10, 170 * 320 + 300);
  assert.ok(res2 && performance.now() - t1 < 500);
});

test('largestSCC finds the strongly connected core; nearestNode({scc: true}) never snaps to a one-way stub', async () => {
  const { loadGraph } = await import('../src/nav/graph.js');
  const { Router } = await import('../src/nav/astar.js');
  // A <-> B <-> C two-way, plus a one-way spur C -> D (D is a dead end) and E -> A (E unreachable)
  const g = loadGraph({ nodes: [{ id: 'A', x: 0, z: 0 }, { id: 'B', x: 100, z: 0 }, { id: 'C', x: 200, z: 0 }, { id: 'D', x: 300, z: 0 }, { id: 'E', x: -100, z: 0 }],
    edges: [{ nodes: ['A', 'B', 'C'], highway: 'residential' }, { nodes: ['C', 'D'], highway: 'residential', oneway: 'yes' }, { nodes: ['E', 'A'], highway: 'residential', oneway: 'yes' }] });
  const r = new Router(g);
  const s = r.largestSCC();
  assert.equal(s.size, 3);
  assert.deepEqual(['A', 'B', 'C', 'D', 'E'].map((id) => s.inLargest[g.indexOf(id)]), [1, 1, 1, 0, 0]);
  assert.equal(r.nearestNode(310, 0, { requireOut: false, requireIn: true }), g.indexOf('D'));
  assert.equal(r.nearestNode(310, 0, { requireOut: false, requireIn: true, scc: true }), g.indexOf('C'));
  assert.equal(r.nearestNode(-110, 0, { scc: true }), g.indexOf('A'));
});

test('rerouteAround avoids a blocked edge when an alternative exists, keeps the current edge', async () => {
  const { loadGraph } = await import('../src/nav/graph.js');
  const { Router } = await import('../src/nav/astar.js');
  // A -> B -> D (fast) and A -> B -> C -> D (detour); block B->D
  const g = loadGraph({ nodes: [{ id: 'A', x: 0, z: 0 }, { id: 'B', x: 100, z: 0 }, { id: 'C', x: 150, z: 80 }, { id: 'D', x: 200, z: 0 }],
    edges: [{ nodes: ['A', 'B'], highway: 'residential', oneway: 'yes' }, { nodes: ['B', 'D'], highway: 'residential', oneway: 'yes' }, { nodes: ['B', 'C', 'D'], highway: 'residential', oneway: 'yes' }] });
  const r = new Router(g);
  const route = r.route(g.indexOf('A'), g.indexOf('D')).edges;
  assert.equal(route.length, 2);
  const blocked = route[1];
  const rr = r.rerouteAround(route, 0, [blocked]);
  assert.ok(rr.changed && rr.edges[0] === route[0] && !rr.edges.includes(blocked) && rr.edges.length === 3);
  assert.equal(g.edges[rr.edges[rr.edges.length - 1]].to, g.indexOf('D'));
});
