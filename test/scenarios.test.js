// Synthetic training worlds (src/scenarios): real data format, loads through the sim's own loader,
// geometry checks pass, packs are disjoint, a training env runs on them.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { FAMILIES, makeScenario, packWorld, loadScenario, scenarioReport, attachScenario, parseScenarioLevel } from '../src/scenarios/index.js';
import { writeScenario } from '../src/scenarios/write.js';
import { RoadSet } from '../src/roadpack.js';
import { graphFromRoadSets, buildLaneGeometry, hopMap } from '../src/geom/index.js';
import { createTrainingEnv } from '../src/sim/env.js';

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'scn-test-'));
test.after(() => fs.rmSync(tmp, { recursive: true, force: true }));

function loadPack(dir, metaName) {
  const meta = JSON.parse(fs.readFileSync(path.join(dir, metaName), 'utf8'));
  const buf = fs.readFileSync(path.join(dir, meta.file)), ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  const T = { Float32: Float32Array, Uint32: Uint32Array }, a = {};
  for (const [n, s] of Object.entries(meta.sections)) a[n === 'nodeXYZ' ? 'xyz' : n] = new T[s.type](ab, s.byteOffset, s.length);
  return new RoadSet(meta, a);
}

test('every family builds, writes the real file set and passes the lane-geometry continuity checks', () => {
  const real = JSON.parse(fs.readFileSync(new URL('../data/roads.meta.json', import.meta.url), 'utf8'));
  for (const fam of Object.keys(FAMILIES)) {
    const w = makeScenario(fam, 5);
    const dir = path.join(tmp, w.name);
    writeScenario(w.finish(), dir);
    for (const f of ['roads.bin', 'roads.meta.json', 'roads.json', 'roads-local.bin', 'roads-local.meta.json', 'stations.bin', 'stations.meta.json',
      'controls.bin', 'controls.meta.json', 'controls.json', 'crosswalks.json', 'crossings.json', 'junctions.bin', 'junctions.meta.json', 'scenario.json'])
      assert.ok(fs.existsSync(path.join(dir, f)), `${fam}: ${f}`);
    const main = loadPack(dir, 'roads.meta.json'), local = loadPack(dir, 'roads-local.meta.json');
    assert.deepEqual(Object.keys(main.meta.sections).filter((k) => k !== 'bannedHops'), Object.keys(real.sections).filter((k) => k !== 'bannedHops'), `${fam}: same sections as data/roads.bin`);
    assert.equal(main.meta.edgeFields.join(), real.edgeFields.join());
    const G = graphFromRoadSets(main, local), LG = buildLaneGeometry(G), a = {}, b = {};
    let bad = 0;
    for (let e = 0; e < G.edgeCount; e++) for (const en of G.outEdges(G.to(e))) {
      if (en === G.twin[e]) continue;
      const h = hopMap(G, e, en); if (!h.strict) continue;
      for (let l = 0; l < G.lanes[e]; l++) { const l2 = l - h.shift; if (l2 < 0 || l2 >= G.lanes[en]) continue; LG.lanePoint(e, G.length[e], l, a); LG.lanePoint(en, 0, l2, b); if (Math.hypot(a.x - b.x, a.z - b.z) >= 0.05) bad++; }
    }
    assert.equal(bad, 0, `${fam}: strict-hop lane jumps`);
    for (let e = 0; e < G.edgeCount; e++) for (const s of LG.sampleS(e)) { const bd = LG.bounds(e, s); assert.ok(bd.right >= bd.left - 1e-6, `${fam}: inverted pavement`); }
  }
});

test('a scenario loads through loadWorld and an external lane-keeping policy reaches 100% on an empty straight road', async () => {
  const w = makeScenario('straight', 2, { lanes: 2, rate: 4, win: 120 });
  const dir = path.join(tmp, 'straight-env'); writeScenario(w.finish(), dir);
  const sw = await loadScenario(dir);
  assert.ok(sw.graph.laneGeom && sw.graph.J, 'lane geometry + junctions attached by the sim loader');
  const env = await createTrainingEnv({ ...sw.envOpts(), trafficMode: 'empty' });
  attachScenario(env, sw);
  // external policy (test fixture): lane-centre P steering + cruise below the limit
  const end = env.sim.time + 1800;
  while (env.sim.time < end) {
    for (const { slot, obs } of env.observe()) env.act(slot, { steer: Math.max(-1, Math.min(1, (-0.08 * obs.laneOffset - 1.2 * obs.headingRel) / obs.maxSteer)), gas: obs.speed < 0.8 * obs.speedLimit ? 0.4 : 0, brake: 0 });
    env.step();
    if (env.sim.isIdle) break;
  }
  const st = env.stats();
  assert.equal(st.arrivedOutbound, env.n);
  const rep = scenarioReport(env.sim, sw);
  assert.equal(rep.families[0].arrival, 1);
});

test('junction controls get their stop lines from the junction mouths (tools/build_junctions.mjs)', async () => {
  const w = makeScenario('junction', 3, { control: 'signal', shape: 'cross' });
  const dir = path.join(tmp, 'junction-sig'); writeScenario(w.finish(), dir);
  const sw = await loadScenario(dir);
  const J = sw.graph.J, ctl = JSON.parse(fs.readFileSync(path.join(dir, 'controls.json'), 'utf8'));
  assert.equal(ctl.controls.length, 1);
  for (const ap of ctl.controls[0].approaches) assert.ok(Math.abs(ap.stopAt - J.mouthS[2 * ap.edge + 1]) < 0.01);
  assert.ok(sw.controls && sw.controls.count === 1);
});

test('world packs are disjoint: no edge joins two tiles, trips stay in their tile', async () => {
  const worlds = ['straight', 'junction', 'merge', 'roundabout'].map((f, k) => makeScenario(f, 10 + k));
  const pack = packWorld(worlds, { name: 'pack-test' });
  const doc = pack.finish();
  const dir = path.join(tmp, 'pack'); writeScenario(doc, dir);
  const sw = await loadScenario(dir);
  const g = sw.graph, tiles = sw.scenario.tiles;
  const tileOf = (x, z) => tiles.findIndex((t) => x >= t.bbox.minX - 1 && x <= t.bbox.maxX + 1 && z >= t.bbox.minZ - 1 && z <= t.bbox.maxZ + 1);
  for (const e of g.edges) assert.equal(tileOf(g.nx[e.from], g.nz[e.from]), tileOf(g.nx[e.to], g.nz[e.to]));
  for (let a = 0; a < tiles.length; a++) for (let b = a + 1; b < tiles.length; b++) {
    const A = tiles[a].bbox, B = tiles[b].bbox;
    assert.ok(A.maxX + 300 < B.minX || B.maxX + 300 < A.minX || A.maxZ + 300 < B.minZ || B.maxZ + 300 < A.minZ, 'tiles >= 300 m apart');
  }
  const pop = sw.population();
  for (let i = 0; i < pop.commuters.length; i++) { const c = pop.commuters[i]; assert.equal(tileOf(g.nx[c.home], g.nz[c.home]), pop.tripTile[i]); }
});

test('level spec parsing', () => {
  assert.deepEqual(parseScenarioLevel('junction+merge/40/0.5'), { families: ['junction', 'merge'], tiles: 40, rate: 0.5 });
  assert.equal(parseScenarioLevel('all/10').families.length, Object.keys(FAMILIES).length);
  assert.throws(() => parseScenarioLevel('nope/3'));
});
