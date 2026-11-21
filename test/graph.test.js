import { test } from 'node:test';
import assert from 'node:assert/strict';
import { loadGraph, parseMaxspeedMph } from '../src/nav/graph.js';
import { MPH_TO_MS } from '../src/nav/geom.js';

test('generic schema: two-way roads become two directed edges, oneway stays one', () => {
  const g = loadGraph({
    nodes: [{ id: 1, x: 0, z: 0 }, { id: 2, x: 100, z: 0 }, { id: 3, x: 200, z: 0 }],
    edges: [
      { nodes: [1, 2], highway: 'primary', oneway: false, lanes: 4 },
      { nodes: [2, 3], highway: 'motorway', lanes: 3 }, // oneway by default
    ],
  });
  assert.equal(g.nodeCount(), 3);
  assert.equal(g.edgeCount(), 3);
  const ab = g.edges.find((e) => e.from === 0 && e.to === 1), ba = g.edges.find((e) => e.from === 1 && e.to === 0);
  assert.ok(ab && ba);
  assert.equal(ab.lanes, 2); // 4 total -> 2 per direction
  assert.equal(ab.length, 100);
  assert.ok(Math.abs(ab.speed - 40 * MPH_TO_MS) < 1e-9);
  const m = g.edges.find((e) => e.highway === 'motorway');
  assert.equal(m.lanes, 3); assert.equal(m.oneway, true);
  assert.ok(Math.abs(m.speedMph - 65) < 1e-9);
});

test('generic schema: maxspeed parsing, oneway=-1, per-direction lanes, merging two docs by node id', () => {
  assert.equal(parseMaxspeedMph('55 mph', 'primary'), 55);
  assert.equal(parseMaxspeedMph(45, 'primary'), 45);
  assert.ok(Math.abs(parseMaxspeedMph('80 km/h', 'primary') - 49.7) < 0.1);
  assert.equal(parseMaxspeedMph(null, 'residential'), 25);
  const main = { nodes: [{ id: 'a', x: 0, z: 0 }, { id: 'b', x: 0, z: 100 }], edges: [{ nodes: ['a', 'b'], highway: 'secondary', oneway: '-1', maxspeed: '30 mph' }] };
  const local = { nodes: [{ id: 'b', x: 0, z: 100 }, { id: 'c', x: 50, z: 100 }], edges: [{ nodes: ['b', 'c'], highway: 'residential', 'lanes:forward': 2, 'lanes:backward': 1 }] };
  const g = loadGraph([main, local]);
  assert.equal(g.nodeCount(), 3); // 'b' shared
  const rev = g.edges.find((e) => e.highway === 'secondary');
  assert.equal(g.nodes[rev.from].id, 'b'); assert.equal(g.nodes[rev.to].id, 'a'); // reversed
  const f = g.edges.find((e) => e.highway === 'residential' && g.nodes[e.from].id === 'b');
  const b = g.edges.find((e) => e.highway === 'residential' && g.nodes[e.from].id === 'c');
  assert.equal(f.lanes, 2); assert.equal(b.lanes, 1);
  assert.ok(Math.abs(f.speedMph - 25) < 1e-9);
});

test('bay-roads/1 schema: directed edges, node indices, km/h, laneOffsets -> laneLat, tier merge by id', () => {
  const main = {
    meta: { format: 'bay-roads/1', tier: 'main', laneWidth: 3.7, defaultSpeedKph: { motorway: 105 } },
    nodes: [{ id: 10, x: 0, z: 0 }, { id: 11, x: 500, z: 0 }, { id: 12, x: 1000, z: 0 }],
    edges: [
      { id: 0, from: 0, to: 2, nodes: [0, 1, 2], class: 'motorway', name: 'Fwy', ref: 'US 101', oneway: true, twin: -1, lanes: 3, oppLanes: 0, laneOffsets: [-3.7, 0, 3.7], maxspeed: null, way: 1, length: 1000 },
      { id: 1, from: 2, to: 0, nodes: [2, 1, 0], class: 'primary', name: 'Rd', ref: null, oneway: false, twin: 2, lanes: 2, oppLanes: 2, laneOffsets: [1.85, 5.55], maxspeed: 56.3, way: 2, length: 1000 },
      { id: 2, from: 0, to: 2, nodes: [0, 1, 2], class: 'primary', name: 'Rd', ref: null, oneway: false, twin: 1, lanes: 2, oppLanes: 2, laneOffsets: [1.85, 5.55], maxspeed: 56.3, way: 2, length: 1000 },
    ],
  };
  const local = { meta: { format: 'bay-roads/1', tier: 'local', laneWidth: 3.7 }, nodes: [{ id: 12, x: 1000, z: 0 }, { id: 13, x: 1000, z: 300 }], edges: [{ id: 0, from: 0, to: 1, nodes: [0, 1], class: 'residential', name: 'Elm', oneway: false, twin: 1, lanes: 1, oppLanes: 1, laneOffsets: [1.85], maxspeed: null, way: 3 }] };
  const g = loadGraph([main, local]);
  assert.equal(g.nodeCount(), 4);
  assert.equal(g.laneWidth, 3.7);
  // no reverse edges synthesised: motorway 2 segments + primary 2x2 + residential 1 = 7
  assert.equal(g.edgeCount(), 7);
  const fwy = g.edges.filter((e) => e.highway === 'motorway');
  assert.equal(fwy.length, 2);
  assert.ok(Math.abs(fwy[0].speedMph - 105 / 1.609344) < 1e-6, 'default kph used');
  assert.deepEqual(Array.from(fwy[0].laneLat), [-3.7, 0, 3.7].map((v) => -v).reverse()); // rightmost first, +left
  const rd = g.edges.find((e) => e.highway === 'primary');
  assert.ok(Math.abs(rd.speedMph - 35) < 0.05);
  assert.deepEqual(Array.from(rd.laneLat), [-5.55, -1.85]);
  const res = g.edges.find((e) => e.highway === 'residential');
  assert.equal(g.nodes[res.from].id, '12'); // joined through shared OSM id
  assert.ok(Math.abs(res.speedMph - 25) < 1e-9);
});
