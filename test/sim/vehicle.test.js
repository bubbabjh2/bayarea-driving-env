import test from 'node:test';
import assert from 'node:assert/strict';
import { VehicleStore, STATE_DRIVING, CTRL_RING } from '../../src/sim/vehicle.js';
import { laneOffset, edgePoint, mapLat, hopMap } from '../../src/sim/roadGraph.js';
import { smallGraph } from './helpers.js';

test('vehicle advances along an edge and transitions to the next route edge (physics)', () => {
  const g = smallGraph();
  let e0 = -1, e1 = -1;
  for (const e of g.edges) {
    const nxt = Array.from(g.outEdges.subarray(g.outStart[e.to], g.outStart[e.to + 1]));
    const cand = nxt.find(k => k !== e.reverse && Math.abs(g.edges[k].headingStart - e.headingEnd) < 0.1);
    if (cand !== undefined && e.length > 100) { e0 = e.idx; e1 = cand; break; }
  }
  assert.ok(e0 >= 0 && e1 >= 0);
  const vs = new VehicleStore(16);
  const slot = vs.alloc({ route: Int32Array.from([e0, e1]), time: 0, graph: g, dt: 0.05 });
  assert.equal(vs.state[slot], STATE_DRIVING);
  vs.delaySteps[slot] = 0;
  const len0 = g.edges[e0].length;
  let t = 0, hopped = false, arrived = 0;
  for (let k = 0; k < 20000 && !arrived; k++) {
    vs.command(slot, 0, 0.4, 0, k);
    arrived = vs.integrate(slot, 0.05, t, g, k); t += 0.05;
    vs.updatePose(slot, g, {});
    assert.ok(Number.isFinite(vs.x[slot]) && Number.isFinite(vs.z[slot]) && Number.isFinite(vs.speed[slot]));
    if (vs.edge[slot] === e1 && !hopped) { hopped = true; assert.ok(vs.routeDist[slot] >= len0 - 0.01); assert.equal(vs.routePos[slot], 1); }
  }
  assert.ok(hopped, 'vehicle hopped to the second edge');
  assert.equal(arrived, 1, 'vehicle reached the end of its route');
  assert.ok(vs.speed[slot] <= vs.maxSpeed[slot] + 1e-6);
  assert.ok(vs.fuel[slot] < vs.tank[slot], 'fuel is burnt');
});

test('lane offsets: right-hand traffic, lane 0 rightmost, two-way lanes right of centreline', () => {
  const g = smallGraph();
  const twoWay = g.edges.find(e => e.reverse >= 0 && e.lanes >= 2);
  assert.ok(laneOffset(twoWay, 0) > laneOffset(twoWay, 1), 'lane 0 is further right');
  assert.ok(laneOffset(twoWay, twoWay.lanes - 1) > 0, 'all lanes of a two-way edge are right of centre');
  assert.ok(Math.abs(twoWay.rightEdge - (laneOffset(twoWay, 0) + 1.85)) < 1e-6);
  const p0 = edgePoint({ px: Float64Array.from([0, 10]), pz: Float64Array.from([0, 0]), cum: Float64Array.from([0, 10]), y: 0, ys: null }, 5, 2, {});
  assert.ok(Math.abs(p0.z - 2) < 1e-9 && Math.abs(p0.x - 5) < 1e-9, 'right of +x is +z');
  // strict hops keep the physical lane, junction hops keep the world position inside the carriageway
  for (const e of g.edges) for (let q = g.outStart[e.to]; q < g.outStart[e.to + 1]; q++) {
    const en = g.edges[g.outEdges[q]]; if (en.idx === e.reverse) continue;
    const h = hopMap(g, e, en), l2 = mapLat(g, e, en, 1.85, 1);
    if (h.strict) assert.ok(Math.abs(l2 - (1.85 - 3.7 * h.shift)) < 1e-9);
    else assert.ok(l2 >= 0 && l2 <= en.lanes * 3.7);
  }
});

test('controls go through the reaction-time ring; pedals and steering act after the delay', () => {
  const g = smallGraph();
  const e = g.edges.find(x => x.length > 300);
  const vs = new VehicleStore(4);
  const slot = vs.alloc({ route: Int32Array.from([e.idx]), time: 0, traits: { reactionTime: 0.5 }, dt: 0.05, step: 0, graph: g });
  assert.equal(vs.delaySteps[slot], 10);
  assert.ok(CTRL_RING > vs.delaySteps[slot]);
  let step = 0;
  for (; step < 10; step++) { vs.command(slot, 0, 1, 0, step); vs.integrate(slot, 0.05, step * 0.05, g, step); }
  assert.equal(vs.speed[slot], 0, 'no acceleration before the reaction time');
  for (; step < 60; step++) { vs.command(slot, 0, 1, 0, step); vs.integrate(slot, 0.05, step * 0.05, g, step); }
  assert.ok(vs.speed[slot] > 3 && vs.gas[slot] === 1);
  const vBefore = vs.speed[slot];
  for (let k = 0; k < 10; k++, step++) { vs.command(slot, 0, 0, 1, step); vs.integrate(slot, 0.05, step * 0.05, g, step); }
  assert.ok(vs.speed[slot] > vBefore, 'still accelerating during the reaction delay after braking is commanded');
  for (let k = 0; k < 100; k++, step++) { vs.command(slot, 0, 0, 1, step); vs.integrate(slot, 0.05, step * 0.05, g, step); }
  assert.equal(vs.speed[slot], 0);
  // steering: the wheel angle integrates steerRate x maxSteerRate after the delay and is clamped
  for (let k = 0; k < 10; k++, step++) { vs.command(slot, 1, 0, 0, step); vs.integrate(slot, 0.05, step * 0.05, g, step); }
  assert.equal(vs.steerAngle[slot], 0);
  for (let k = 0; k < 100; k++, step++) { vs.command(slot, 1, 0, 0, step); vs.integrate(slot, 0.05, step * 0.05, g, step); }
  assert.ok(Math.abs(vs.steerAngle[slot] - vs.maxSteer[slot]) < 1e-6);
  // held control: no new command on a step keeps the last one
  vs.command(slot, 0, 1, 0, step); for (let k = 0; k < 30; k++, step++) vs.integrate(slot, 0.05, step * 0.05, g, step);
  assert.ok(vs.speed[slot] > 0);
});
