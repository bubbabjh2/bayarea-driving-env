import test from 'node:test';
import assert from 'node:assert/strict';
import { generatePopulation, tripsOf, synthAxis } from '../../src/sim/population.js';
import { smallGraph, popOpts } from './helpers.js';

test('population is deterministic for a seed and differs across seeds', () => {
  const g = smallGraph();
  const a = generatePopulation(g, { n: 300, seed: 7, ...popOpts(g) });
  const b = generatePopulation(g, { n: 300, seed: 7, ...popOpts(g) });
  const c = generatePopulation(g, { n: 300, seed: 8, ...popOpts(g) });
  for (let i = 0; i < 300; i++) {
    assert.equal(a.commuters[i].home, b.commuters[i].home);
    assert.equal(a.commuters[i].work, b.commuters[i].work);
    assert.equal(a.commuters[i].depart, b.commuters[i].depart);
    assert.deepEqual(Array.from(a.commuters[i].routeOut), Array.from(b.commuters[i].routeOut));
  }
  assert.ok(a.commuters.some((x, i) => x.home !== c.commuters[i].home));
});

test('local/long mixture, cross-bay share and trip-length histogram', () => {
  const g = smallGraph({ blocks: 40, cross: 4 });
  const p = generatePopulation(g, { n: 2000, seed: 3, ...popOpts(g) });
  const s = p.stats;
  assert.ok(s.localShare > 0.75 && s.localShare < 0.85, `localShare ${s.localShare}`);
  assert.ok(s.longCrossBayShare > 0.25 && s.longCrossBayShare < 0.6, `long cross share ${s.longCrossBayShare}`);
  assert.ok(s.crossBayShare > 0.05, `cross share ${s.crossBayShare}`);
  assert.ok(s.histogram.length >= 2 && s.histogram.reduce((a, b) => a + b.count, 0) === 2000);
  assert.ok(s.departuresPer15min.length > 4);
  // every commuter has a route and a sane departure in the morning window
  for (const c of p.commuters) {
    assert.ok(c.routeOut.length > 0);
    assert.ok(c.depart >= 5 * 3600 && c.depart <= 10.5 * 3600);
    assert.ok(c.returnDepart > c.depart + 3600);
    assert.ok(c.genome && c.genome.traits && c.genome.traits.maxAccel > 0);
  }
  const trips = tripsOf(p.commuters);
  assert.equal(trips.length, 4000);
  for (let i = 1; i < trips.length; i++) assert.ok(trips[i].depart >= trips[i - 1].depart);
});

test('side classifier splits the synthetic shores', () => {
  const g = smallGraph();
  const axis = synthAxis(g);
  let east = 0, west = 0;
  for (let i = 0; i < g.nodeCount; i++) { if (axis.sideOf(g.nx[i], g.nz[i])) east++; else west++; }
  assert.equal(east, west);
});
