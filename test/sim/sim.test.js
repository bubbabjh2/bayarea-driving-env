import test from 'node:test';
import assert from 'node:assert/strict';
import { Simulation, runHeadless } from '../../src/sim/sim.js';
import { STATE_FREE, STATE_DRIVING, STATE_CRASHED } from '../../src/sim/vehicle.js';
import { smallGraph, popOpts } from './helpers.js';

test('one simulated hour with 500 vehicles (no driver: default action): no NaNs, every vehicle accounted for', () => {
  const g = smallGraph();
  // all 500 depart within the hour so that the sim is busy
  const sim = new Simulation(g, { population: { n: 500, ...popOpts(g), returnTrips: false, workStart: { mean: 8 * 3600, sd: 15 * 60, min: 7.7 * 3600, max: 8.3 * 3600 } }, startTime: 7 * 3600 });
  const vs = sim.vs;
  let steps = 0;
  while (sim.time < 8.5 * 3600) {
    sim.step(); steps++;
    if (steps % 200 === 0) for (let i = 0; i < vs.high; i++) {
      if (vs.state[i] === STATE_FREE) continue;
      for (const k of ['x', 'z', 'y', 's', 'speed', 'accel', 'heading']) assert.ok(Number.isFinite(vs[k][i]), `${k} finite at slot ${i}`);
      assert.ok(vs.speed[i] >= 0 && vs.speed[i] <= vs.maxSpeed[i] + 1e-4);
      const e = g.edges[vs.edge[i]];
      assert.ok(vs.s[i] >= 0 && vs.s[i] <= e.length + 1e-3);
      assert.ok(vs.lane[i] >= 0 && vs.lane[i] < e.lanes && ((vs.laneLo[i] <= vs.lane[i] && vs.laneHi[i] >= vs.lane[i]) || vs.laneHi[i] === vs.laneLo[i] - 1), 'footprint lanes (or off the travel lanes)');
      assert.ok(Number.isFinite(vs.lat[i]) && Number.isFinite(vs.headingRel[i]) && Number.isFinite(vs.vLat[i]));
    }
  }
  const s = sim.getStats();
  assert.ok(s.spawned > 0 && s.spawned <= 500, `spawned ${s.spawned}`);   // spawning waits for room (cars without a driver hold the brake)
  assert.equal(s.spawned, s.arrived + s.active + s.cleared + s.towed, 'every spawned vehicle is arrived, active (driving / on scene / refuelling / waiting), cleared or towed (stuck is only a label)');
  assert.ok(s.avgSpeed >= 0 && Number.isFinite(s.avgSpeed));
  assert.ok(s.arrivalRate >= 0 && s.arrivalRate <= 1);
  assert.ok(typeof s.failures.crash === 'number' && typeof s.failures.stranded === 'number');
  assert.ok(s.density && s.density.freewayLaneKm > 0);
});

test('clock: advance() honours the speed multiplier and step cap', () => {
  const g = smallGraph();
  const sim = new Simulation(g, { population: { n: 10, ...popOpts(g) }, speed: 60, dt: 0.05, maxStepsPerFrame: 50 });
  const t0 = sim.time;
  const steps = sim.advance(1 / 60);
  assert.equal(steps, 20);
  assert.ok(Math.abs(sim.time - t0 - 1) < 1e-6);
  sim.speed = 600; sim.advance(1); // would need 12000 steps -> capped
  assert.equal(sim.lagging, true);
});

test('trip stats and genome export/import round trip', () => {
  const g = smallGraph();
  const sim = new Simulation(g, { population: { n: 100, ...popOpts(g), returnTrips: false, workStart: { mean: 7.5 * 3600, sd: 5 * 60, min: 7.4 * 3600, max: 7.6 * 3600 } }, startTime: 6.5 * 3600 });
  while (sim.time < 8.5 * 3600) sim.step();
  const ts = sim.tripStats();
  assert.ok(ts && typeof ts === 'object');
  const exported = sim.exportGenomes();
  assert.equal(exported.length, 100);
  assert.equal(exported[0].genome.delta.length, 64);
  const sim2 = new Simulation(g, { population: { n: 100, ...popOpts(g) } });
  assert.equal(sim2.importGenomes(exported), 100);
  assert.equal(sim2.commuters[0].genome.fitness, exported[0].genome.fitness);
});

test('runHeadless reports arrivalRate and failure breakdown', () => {
  const { stats } = runHeadless(0.5, 50, { log: null, synth: { blocks: 8, cross: 4, bridges: 1 }, population: { workStart: { mean: 5.3 * 3600, sd: 60, min: 5.2 * 3600, max: 5.4 * 3600 } }, startTime: 5 * 3600 });
  assert.ok(stats.arrivalRate >= 0 && stats.arrivalRate <= 1);
  for (const k of ['crash', 'stranded', 'stuck', 'unfinished']) assert.ok(k in stats.failures, k);
  assert.ok('onTimeRate' in stats && 'meanLatenessMin' in stats && 'speedingRate' in stats && 'crashRatePerMillionKm' in stats);
  assert.ok(stats.msPerStep > 0);
});

test('scale-crash options (junctionSight / pathSight / deadlockBreak / spawnLaneRoute, docs/scale-crashes.md): finite state with the options on and off', () => {
  const g = smallGraph();
  const run = (on) => {
    const f = on ? 1 : 0;
    const sim = new Simulation(g, { population: { n: 400, ...popOpts(g), returnTrips: false, workStart: { mean: 8 * 3600, sd: 15 * 60, min: 7.7 * 3600, max: 8.3 * 3600 } }, startTime: 7 * 3600,
      junctionSight: f, pathSight: f, deadlockBreak: f, spawnLaneRoute: f });
    const vs = sim.vs;
    while (sim.time < 8.5 * 3600) sim.step();
    for (let i = 0; i < vs.high; i++) if (vs.state[i] !== STATE_FREE) for (const k of ['x', 'z', 's', 'speed', 'gap']) assert.ok(!Number.isNaN(vs[k][i]), `${k} at slot ${i}`);
    return sim.getStats();
  };
  const off = run(false), on = run(true);
  assert.ok(on.spawned > 0 && off.spawned > 0);
});

test('route.js routingPack (worker routing graph, docs/scale-crashes.md): identical routes to the full graph, survives structured clone', async () => {
  const { makeRouter, routingPack, liteGraph } = await import('../../src/sim/route.js');
  const g = smallGraph();
  const pack = structuredClone(routingPack(g)), a = makeRouter(g, { weight: 1.25, cacheMax: 0 }), b = makeRouter(pack, { weight: 1.25, cacheMax: 0 }), c = makeRouter(liteGraph(pack), { weight: 1.25, cacheMax: 0 });
  let n = 0;
  for (let f = 0; f < g.nodeCount; f += 7) for (let t = 3; t < g.nodeCount; t += 11) {
    const ra = a(f, t), rb = b(f, t), rc = c(f, t);
    assert.deepEqual(rb && Array.from(rb), ra && Array.from(ra)); assert.deepEqual(rc && Array.from(rc), ra && Array.from(ra)); n++;
  }
  assert.ok(n > 100);
});
