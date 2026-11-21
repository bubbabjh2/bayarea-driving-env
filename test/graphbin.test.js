import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { loadGraph, loadGraphBin, loadGraphFromFiles } from '../src/nav/graph.js';
import { Router } from '../src/nav/astar.js';
import { loadStations } from '../src/nav/stations.js';

const root = new URL('../data/', import.meta.url);
const files = ['roads.json', 'roads.bin', 'roads.meta.json', 'roads-local.json', 'roads-local.bin', 'roads-local.meta.json'];
const have = files.every((f) => fs.existsSync(new URL(f, root)));
const readJson = (f) => JSON.parse(fs.readFileSync(new URL(f, root), 'utf8'));
const pack = (name) => ({ meta: readJson(`${name}.meta.json`), buffer: fs.readFileSync(new URL(`${name}.bin`, root)) });

let cached = null;
const graphs = () => cached || (cached = { gj: loadGraph([readJson('roads.json'), readJson('roads-local.json')]), gb: loadGraphBin([pack('roads'), pack('roads-local')]) });

test('both binary tiers merge into the same graph as both JSON tiers (nodes, edges, lanes, speeds)', { skip: !have && 'data/roads{,-local}.{json,bin,meta.json} not present' }, () => {
  const { gj, gb } = graphs();
  assert.equal(gb.nodeCount(), gj.nodeCount(), 'node count (shared nodes joined via linkNodes)');
  assert.equal(gb.edgeCount(), gj.edgeCount(), 'edge count');
  for (let i = 0; i < gj.nodeCount(); i += 97) {
    assert.ok(Math.abs(gb.nodes[i].x - gj.nodes[i].x) < 0.1 && Math.abs(gb.nodes[i].z - gj.nodes[i].z) < 0.1, `node ${i} position (JSON is rounded to 0.1 m)`);
  }
  let lenErr = 0;
  for (let e = 0; e < gj.edgeCount(); e++) {
    const a = gj.edges[e], b = gb.edges[e];
    if (a.from !== b.from || a.to !== b.to) assert.fail(`edge ${e}: ${a.from}->${a.to} vs ${b.from}->${b.to}`);
    if (a.lanes !== b.lanes || a.highway !== b.highway || a.isLink !== b.isLink || a.tier !== b.tier) assert.fail(`edge ${e} attributes differ`);
    if (Math.abs(a.speed - b.speed) > 0.02) assert.fail(`edge ${e} speed ${a.speed} vs ${b.speed}`);
    lenErr = Math.max(lenErr, Math.abs(a.length - b.length));
    for (let k = 0; k < a.lanes; k++) if (Math.abs(a.laneLat[k] - b.laneLat[k]) > 1e-6) assert.fail(`edge ${e} lane ${k} lateral`);
  }
  assert.ok(lenErr < 0.25, `max edge length diff ${lenErr} m (JSON coordinates rounded to 0.1 m)`);
  assert.equal(gb.laneWidth, gj.laneWidth);
  // tier maps agree: tier-local node i -> same graph node in both loaders
  for (const t of ['main', 'local']) {
    assert.equal(gb.tierNodes[t].length, gj.tierNodes[t].length);
    for (let i = 0; i < gj.tierNodes[t].length; i += 53) assert.equal(gb.tierNodes[t][i], gj.tierNodes[t][i], `${t} node ${i}`);
  }
  // a cross-tier route has the same length on both
  const rj = new Router(gj), rb = new Router(gb);
  const a = rj.nearestNode(22900, 24400), b = rj.nearestNode(-23800, -24400, { requireOut: false, requireIn: true }); // downtown San Jose -> downtown SF
  const pj = rj.route(a, b), pb = rb.route(a, b);
  assert.ok(pj && pb && Math.abs(pj.length - pb.length) < 5, `route ${pj?.length} vs ${pb?.length}`);
});

test('single main pack + loadGraphFromFiles merge, stations resolve through tier maps on both loaders', { skip: !have && 'data not present' }, async () => {
  const one = loadGraphBin(readJson('roads.meta.json'), fs.readFileSync(new URL('roads.bin', root)));
  assert.equal(one.nodeCount(), readJson('roads.meta.json').nodeCount);
  const gf = await loadGraphFromFiles([new URL('roads.bin', root).pathname, new URL('roads-local.bin', root).pathname]);
  const { gj, gb } = graphs();
  assert.equal(gf.edgeCount(), gb.edgeCount());
  if (!fs.existsSync(new URL('stations.json', root))) return;
  const sj = loadStations(readJson('stations.json'), gj, new Router(gj));
  const sb = loadStations(readJson('stations.json'), gb, null);
  assert.equal(sb.length, sj.length);
  for (let i = 0; i < sj.length; i++) assert.equal(sb[i].node, sj[i].node, `station ${i}`);
});
