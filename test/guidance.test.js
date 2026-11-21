import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Guidance, formatDistance } from '../src/nav/guidance.js';
import { synthetic, idx } from './helpers.js';

const route = (a, b) => { const { graph, router } = synthetic(); const r = router.route(idx(graph, a), idx(graph, b)); assert.ok(r, `route ${a}->${b}`); return new Guidance(graph, r.edges); };
const types = (g) => g.maneuvers.map((m) => m.type);

test('exit + merge are detected with the right side and destination', () => {
  const g = route('E0', 'MS2000');
  assert.deepEqual(types(g), ['exit', 'merge', 'arrive']);
  const ex = g.maneuvers[0];
  assert.equal(ex.side, 'right');
  assert.equal(ex.toName, 'Main St');
  assert.ok(Math.abs(ex.atDistance - 5000) < 1e-6);
  assert.match(ex.text, /^In 3\.1 mi, take the right exit toward Main St/);
  assert.equal(g.maneuvers[1].side, 'right');
  assert.equal(g.maneuvers[2].type, 'arrive');
  assert.ok(Math.abs(g.maneuvers[2].atDistance - g.totalLength) < 1e-9);
});

test('turns on surface streets, merge from a ramp, keep-left at a freeway fork', () => {
  const g = route('MS-3000', 'E15');
  assert.deepEqual(types(g), ['turn-left', 'merge', 'keep-left', 'arrive']);
  const g2 = route('E0', 'C4');
  assert.deepEqual(types(g2), ['keep-right', 'arrive']);
  assert.equal(g2.maneuvers[0].toRef, 'CA 237');
  const g3 = route('ELM5000', 'PINE5000'); // residential -> residential
  assert.deepEqual(types(g3), ['turn-right', 'turn-right', 'arrive']);
  assert.equal(g3.maneuvers[0].toName, 'Oak Ave');
  const g4 = route('PINE5000', 'ELM5000'); // same streets, opposite way -> left turns
  assert.deepEqual(types(g4), ['turn-left', 'turn-left', 'arrive']);
});

test('at(d): next maneuver, distance, lane hint, eta and arrival', () => {
  const g = route('E0', 'MS2000');
  const far = g.at(1000);
  assert.equal(far.nextManeuver.type, 'exit');
  assert.ok(Math.abs(far.distanceToNext - 4000) < 1e-6);
  assert.equal(far.laneHint, 'none', 'hint only inside the horizon');
  assert.equal(far.currentRoad.ref, 'US 101');
  assert.match(far.liveText, /^In 2\.5 mi, take the right exit/);
  const near = g.at(3500);
  assert.equal(near.laneHint, 'right');
  assert.ok(near.etaSeconds < far.etaSeconds && near.remainingDistance < far.remainingDistance);
  const onRamp = g.at(5100);
  assert.equal(onRamp.currentRoad.isLink, true);
  assert.equal(onRamp.nextManeuver.type, 'merge');
  const end = g.at(g.totalLength);
  assert.equal(end.nextManeuver.type, 'arrive');
  assert.equal(end.arrived, true);
  assert.equal(end.etaSeconds, 0);
  assert.equal(formatDistance(100), '350 ft');
  assert.equal(formatDistance(1609.344 * 2.3), '2.3 mi');
});

test('advance() tracks progress from world coordinates', () => {
  const g = route('E0', 'E20');
  const [x, z] = g.poseAt(2345);
  const a = g.advance(2000, x + 1, z - 2);
  assert.ok(Math.abs(a.d - 2345) < 2 && a.onRoute);
  const off = g.advance(2000, x, z + 500);
  assert.equal(off.onRoute, false);
});
