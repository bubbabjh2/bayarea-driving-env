import test from 'node:test';
import assert from 'node:assert/strict';
import { STATE_CRASHED, STATE_FREE, STATE_DRIVING } from '../../src/sim/vehicle.js';
import { Simulation } from '../../src/sim/sim.js';
import { buildGraph } from '../../src/sim/roadGraph.js';
import { severityOf } from '../../src/sim/crashModel.js';

/** A straight 3-lane one-way road, 3 km, plus a dummy commuter set (sim needs a population). */
function road(lanes = 3) {
  return buildGraph({ nodes: [{ id: 'a', x: 0, z: 0 }, { id: 'b', x: 3000, z: 0 }, { id: 'c', x: 3000, z: 500 }],
    edges: [{ nodes: ['a', 'b'], highway: 'motorway', oneway: true, lanes, maxspeed: '65 mph' }, { nodes: ['b', 'c'], highway: 'residential', oneway: false, lanes: 1, maxspeed: '25 mph' }] });
}
function emptySim(g) {
  return new Simulation(g, { commuters: [], startTime: 7 * 3600, dt: 0.05, fuel: false });
}
function place(sim, edge, s, lane, speed, typeIdx = 0) {
  const vs = sim.vs, g = sim.graph;
  const i = vs.alloc({ route: Int32Array.of(edge), time: sim.time, graph: g, dt: sim.dt, typeIdx, lane });
  vs.s[i] = s; vs.speed[i] = speed; vs.setReactionTime(i, 0.05, sim.dt);
  vs.updateLanes(i, g.edges[edge]); vs.updatePose(i, g, {}); vs.savePose(i);
  return i;
}

test('severity tiers by delta-v', () => {
  assert.equal(severityOf(5), 'minor'); assert.equal(severityOf(15), 'moderate');
  assert.equal(severityOf(30), 'severe'); assert.equal(severityOf(70), 'very severe');
});

test('rear-end at 50 km/h closing: physical contact -> crash, struck car is pushed, scene blocks the lane, then clears', () => {
  const g = road(), sim = emptySim(g), vs = sim.vs;
  const a = place(sim, 0, 100, 0, 22), b = place(sim, 0, 112, 0, 8);
  const c = place(sim, 0, 20, 0, 20);
  // nobody brakes: hold the pedals by writing commands directly (no driver steps for these slots)
  let crashedAt = -1;
  for (let k = 0; k < 40 && crashedAt < 0; k++) {
    for (const i of [a, b]) vs.command(i, 0, 0.3, 0, sim.stepCount + 1);
    sim._spawn = () => {}; // no population
    sim.step();
    if (sim.counters.crashes > 0) crashedAt = k;
  }
  assert.ok(crashedAt >= 0, 'the cars touched and a crash was recorded');
  const rec = sim.crashes[0];
  assert.ok(rec.deltaVKmh.every(d => d > 10), `delta-v ${rec.deltaVKmh.map(d => d.toFixed(1))}`);
  assert.ok(Math.abs(rec.deltaVKmh[0] - rec.deltaVKmh[1]) < 3, 'equal masses share the delta-v');
  assert.equal(vs.state[a], STATE_CRASHED); assert.equal(vs.state[b], STATE_CRASHED);
  assert.ok(rec.laneMask & 1, 'lane 0 is blocked by the hitboxes');
  // the disabled cars slide to rest under tyre friction and stay there
  for (let k = 0; k < 200; k++) sim.step();
  assert.ok(vs.speed[a] < 0.05 && vs.speed[b] < 0.05);
  // the follower (built-in driver) stops or steers around without touching
  const crashes0 = sim.counters.crashes;
  for (let k = 0; k < 20 * 90; k++) sim.step();
  assert.equal(sim.counters.crashes, crashes0, 'no secondary crash');
  assert.ok(vs.state[c] === STATE_DRIVING || vs.state[c] === STATE_FREE);
  sim.time = rec.clearEnd + 1; sim.step();
  assert.equal(rec.responseState, 'cleared');
  assert.equal(vs.state[a], STATE_FREE); assert.equal(vs.state[b], STATE_FREE);
});

test('a slow touch below 1 km/h delta-v is resolved by physics without a crash event', () => {
  const g = road(), sim = emptySim(g), vs = sim.vs;
  sim._spawn = () => {};
  const a = place(sim, 0, 100, 1, 10.2), b = place(sim, 0, 104.4, 1, 10.0);
  for (let k = 0; k < 4; k++) { for (const i of [a, b]) vs.command(i, 0, 0, 0, sim.stepCount + 1); sim.step(); }
  assert.equal(sim.counters.crashes, 0);
  assert.ok(sim.counters.touches >= 1);
});

