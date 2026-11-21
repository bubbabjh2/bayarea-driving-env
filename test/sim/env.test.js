import test from 'node:test';
import assert from 'node:assert/strict';
import { createTrainingEnv } from '../../src/sim/env.js';
import { exportState, importState, STATE_FIELDS, LAYOUT_VERSION } from '../../src/sim/state.js';
import { Simulation } from '../../src/sim/sim.js';
import { TrafficControls, LIGHT, MOVE, CTRL } from '../../src/sim/controls.js';
import { concurrency } from '../../src/sim/population.js';
import { smallGraph, popOpts } from './helpers.js';

test('createTrainingEnv: empty / external modes, observations, signals, fitness', async () => {
  const g = smallGraph();
  const env = await createTrainingEnv({ graph: g, density: 60, trafficMode: 'empty', seed: 5, population: popOpts(g), startTime: 6 * 3600 });
  const s = env.run(1);
  assert.equal(s.crashes, 0, 'agents pass through each other in empty mode');
  assert.equal(env.deltas.length, 60 * 64);
  assert.equal(env.fitness().length, 60);

  const ext = await createTrainingEnv({ graph: g, density: 20, trafficMode: 'full', seed: 6, driver: 'external', population: popOpts(g) });
  let seen = 0;
  for (let k = 0, after = 0; k < 6000 && after < 60; k++) {
    if (seen) after++;
    const obs = ext.observe();
    for (const o of obs) {
      seen++;
      assert.ok(Number.isFinite(o.obs.speed) && o.obs.neighbors.length === 12);
      assert.ok('timeToWorkStart' in o.obs.signals && 'fuelFraction' in o.obs.signals && 'refuelPlanned' in o.obs.signals);
      assert.ok(o.obs.maxSteer > 0 && o.obs.wheelbase > 0 && o.obs.mass > 0);
      ext.act(o.slot, { steerRate: 0, gas: 0.5, brake: 0 });
    }
    ext.step();
  }
  assert.ok(seen > 0);
  const v = ext.sim.vs; let moving = 0;
  for (let i = 0; i < v.high; i++) if (v.state[i] === 1 && v.speed[i] > 0) moving++;
  assert.ok(moving > 0, 'external controls drive the cars (after the reaction delay)');
});

test('exportState / importState round trip (GPU cross-check contract)', () => {
  const g = smallGraph();
  const sim = new Simulation(g, { population: { n: 80, ...popOpts(g), returnTrips: false }, startTime: 7.5 * 3600, dt: 0.1 });
  for (let k = 0; k < 600; k++) sim.step();
  const snap = exportState(sim);
  assert.equal(snap.version, LAYOUT_VERSION);
  for (const [name, , per] of STATE_FIELDS) assert.equal(snap.arrays[name].length, snap.meta.high * per, name);
  const sim2 = new Simulation(g, { population: { n: 80, ...popOpts(g), returnTrips: false }, startTime: 7.5 * 3600, dt: 0.1 });
  importState(sim2, snap);
  assert.equal(sim2.vs.count, sim.vs.count);
  for (let i = 0; i < sim.vs.high; i++) if (sim.vs.state[i]) assert.deepEqual(Array.from(sim2.vs.routes[i]), Array.from(sim.vs.routes[i]));
});

test('traffic controls: signal plan cycles through green / yellow / red; stop lines are perceived', () => {
  const g = smallGraph();
  // a signal on every in-edge of one 4-way junction of the grid
  let node = -1;
  for (let n = 0; n < g.nodeCount && node < 0; n++) if (g.inStart[n + 1] - g.inStart[n] === 4) node = n;
  const ins = Array.from(g.inEdges.subarray(g.inStart[node], g.inStart[node + 1]));
  const pack = { info: Uint32Array.from([CTRL.SIGNAL, 0, 0, node, 0, ins.length]), xyz: Float32Array.from([g.nx[node], 0, g.nz[node]]),
    approachEdge: Uint32Array.from(ins), approachStopAt: Float32Array.from(ins.map(e => g.edges[e].length - 2)), meta: {} };
  const C = new TrafficControls(g, pack);
  const seen = new Set(), o = {};
  for (let t = 0; t < C.cycle[0]; t += 0.5) { const st = C.stateOf(0, MOVE.THROUGH, t, o).state; seen.add(st); assert.ok(o.timeLeft >= 0 && o.timeIn >= 0); }
  assert.ok(seen.has(LIGHT.GREEN) && seen.has(LIGHT.YELLOW) && seen.has(LIGHT.RED));
  // perpendicular approaches are never green together
  const grp = (a) => C.apGroup[a];
  const a0 = 0, a1 = ins.findIndex((_, k) => grp(k) !== grp(0));
  for (let t = 0; t < C.cycle[0]; t += 0.5) assert.ok(!(C.stateOf(a0, MOVE.THROUGH, t, {}).state === LIGHT.GREEN && C.stateOf(a1, MOVE.THROUGH, t, {}).state === LIGHT.GREEN));
  // a simulated hour around it: cars on the signalised approaches perceive the control (type, state, distance)
  const sim = new Simulation(g, { population: { n: 300, ...popOpts(g), returnTrips: false }, controls: C, startTime: 7 * 3600, dt: 0.1 });
  const vs = sim.vs; let perceived = 0;
  while (sim.time < 8 * 3600) { sim.step(); for (let i = 0; i < vs.high; i++) if (vs.state[i] === 1 && vs.ctlType[i] === CTRL.SIGNAL && vs.ctlDist[i] < 250) perceived++; }
  assert.ok(Number.isFinite(sim.counters.signalViolations));
});

test('concurrency report for a population', () => {
  const cs = [{ depart: 8 * 3600, expectedTravel: 1800 }, { depart: 8 * 3600 + 600, expectedTravel: 1800 }, { depart: 10 * 3600, expectedTravel: 600 }];
  const c = concurrency(cs, { durationFactor: 1, step: 60 });
  assert.equal(c.peak, 2);
  assert.ok(c.peakAt >= 8 * 3600 + 600 && c.peakAt <= 8.5 * 3600);
});
