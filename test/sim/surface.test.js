import test from 'node:test';
import assert from 'node:assert/strict';
import { createState, stepDynamics, VEHICLE_PARAMS } from '../../src/sim/physics/index.js';
import { SURFACES, SURFACE_CODES as C, SURF_PK, SURF_SK, SURF_CRR, carSurface, attachSurfaces, osmSurfaceCode } from '../../src/sim/surface.js';
import { Simulation } from '../../src/sim/sim.js';
import { smallGraph, popOpts } from './helpers.js';

const SEDAN = VEHICLE_PARAMS[0], dt = 0.05;
const uniform = (code) => ({ pk: SURF_PK[code], sk: SURF_SK[code], crr: SURF_CRR[code] });
function brakeDistance(code, { locked = false, v0 = 100 / 3.6 } = {}) {
  const st = createState({ u: v0 }); let t = 0;
  const sf = code == null ? null : uniform(code);
  while (st.u > 0 && t < 60) { stepDynamics(st, 0, 0, 1, SEDAN, dt, 0, locked, undefined, 'rate', sf); t += dt; }
  return st.s;
}

test('surface table: dry asphalt is the identity (pk 1, sk 0.8, crr 1); OSM tag mapping', () => {
  assert.equal(SURF_PK[C.asphalt], 1); assert.equal(SURF_SK[C.asphalt], 0.8); assert.equal(SURF_CRR[C.asphalt], 1);
  for (const s of SURFACES) {
    assert.ok(s.muSlideDry <= s.muPeakDry && s.muSlideWet <= s.muPeakWet && s.muPeakWet <= s.muPeakDry + 1e-9, s.name);
    assert.ok(s.crr > 0, s.name);
  }
  assert.equal(osmSurfaceCode('concrete:plates'), C.concrete);
  assert.equal(osmSurfaceCode('unhewn_cobblestone'), C.sett);
  assert.equal(osmSurfaceCode('metal'), C.metal_grate);
  assert.equal(osmSurfaceCode('fine_gravel'), C.gravel);
  assert.equal(osmSurfaceCode('nonsense'), -1);
});

test('braking 100-0 km/h: asphalt surf factors are bit-identical to the single-mu model; lower mu stops longer, ~ 1/mu', () => {
  const base = brakeDistance(null), asph = brakeDistance(C.asphalt);
  assert.equal(asph, base, 'dry asphalt = previous model');
  const out = {};
  for (const n of ['concrete', 'paving_stones', 'sett', 'gravel', 'dirt', 'grass', 'metal_grate']) {
    const d = brakeDistance(C[n]);
    out[n] = d;
    const pk = SURF_PK[C[n]];
    if (pk < 1) assert.ok(d > asph * 1.05, `${n} ${d.toFixed(1)} m > asphalt ${asph.toFixed(1)} m`);
    // tyre-limited (ABS) braking: distance ~ v^2 / (2 * 0.95 mu g); aero + rolling make it a bit shorter
    const ratio = d / asph;
    assert.ok(ratio > 0.85 / pk && ratio < 1.05 / pk, `${n}: ratio ${ratio.toFixed(3)} vs 1/pk ${(1 / pk).toFixed(3)}`);
  }
  assert.ok(out.grass > out.gravel && out.gravel > asph);
  // sand: low mu and very high rolling resistance (the rolling drag alone is ~0.3 g)
  assert.ok(brakeDistance(C.sand) > asph);
  // locked wheels (disabled car) slide on the sliding coefficient: distance ~ 1 / sk
  const la = brakeDistance(C.asphalt, { locked: true }), lg = brakeDistance(C.grass, { locked: true });
  const r = lg / la, want = SURF_SK[C.asphalt] / SURF_SK[C.grass];
  assert.ok(r > 0.85 * want && r < 1.05 * want, `locked grass/asphalt ${r.toFixed(3)} vs ${want.toFixed(3)}`);
});

test('rolling resistance follows the surface: coasting on grass decelerates faster than on asphalt', () => {
  const a = createState({ u: 15 }), g = createState({ u: 15 });
  for (let k = 0; k < 100; k++) { stepDynamics(a, 0, 0, 0, SEDAN, dt, 0, false, undefined, 'rate', uniform(C.asphalt)); stepDynamics(g, 0, 0, 0, SEDAN, dt, 0, false, undefined, 'rate', uniform(C.grass)); }
  const dA = 15 - a.u, dG = 15 - g.u;
  assert.ok(dG > 3 * dA, `grass speed loss ${dG.toFixed(2)} vs asphalt ${dA.toFixed(2)} m/s`);
});

test('carSurface: wheels on the pavement read the edge surface, past the paved width the side ground class', () => {
  const e = { idx: 0, surf: C.concrete, gndR: C.grass, gndL: C.gravel, lanes: 2, highway: 'motorway', reverse: -1, length: 500, LG: null };
  const g = { J: null, edges: [e] }, o = {}, hw = 0.9;
  carSurface(g, e, 100, 3.7, hw, -1, 0, o);
  assert.deepEqual([o.codeR, o.codeL, o.pk], [C.concrete, C.concrete, 1]);
  carSurface(g, e, 100, -1.5, hw, -1, 0, o);      // on the 3 m paved shoulder: still concrete
  assert.equal(o.codeR, C.concrete);
  carSurface(g, e, 100, -3.5, hw, -1, 0, o);      // right wheels past the shoulder: grass
  assert.equal(o.codeR, C.grass); assert.equal(o.codeL, C.concrete);
  assert.ok(Math.abs(o.pk - 0.5 * (SURF_PK[C.grass] + 1)) < 1e-12);
  carSurface(g, e, 100, 2 * 3.7 + 3.5, hw, -1, 0, o);   // left wheels past the left shoulder: gravel median
  assert.equal(o.codeL, C.gravel);
  carSurface(g, e, 100, -3.5, hw, 0, 0, o);       // inside a junction box (connector): paved
  assert.equal(o.codeR, C.concrete);
  // two-way surface street: the left side is the opposing lanes, then that side's ground class
  const a = { idx: 0, surf: C.asphalt, gndR: C.curb, gndL: C.curb, lanes: 1, highway: 'residential', reverse: 1, length: 200, LG: null };
  const b = { ...a, idx: 1, reverse: 0 };
  carSurface({ J: null, edges: [a, b] }, a, 50, 3.7 + 1.5, hw, -1, 0, o);
  assert.equal(o.codeL, C.asphalt);
});

test('sim: an all-asphalt surface map is bit-identical to no map; an all-grass map changes the dynamics', () => {
  const run = (code) => {
    const g = smallGraph();
    if (code != null) { const b = new Uint8Array(2 * g.edges.length); for (let i = 0; i < g.edges.length; i++) { b[2 * i] = code; b[2 * i + 1] = code | (code << 4); } assert.ok(attachSurfaces(g, b)); }
    const sim = new Simulation(g, { population: { n: 60, ...popOpts(g), returnTrips: false }, startTime: 7.5 * 3600, dt: 0.1, seed: 3, defaultAction: { steer: 0.04, gas: 0.6, brake: 0 } });
    for (let k = 0; k < 900; k++) sim.step();
    const v = sim.vs; let h = 0, codes = new Set();
    for (let i = 0; i < v.high; i++) { h = (h * 31 + Math.round(v.s[i] * 1000) + Math.round(v.speed[i] * 1000)) % 1e9; if (v.state[i]) codes.add(v.surfR[i]); }
    return { h, codes, odo: Array.from(v.odometer.subarray(0, v.high)) };
  };
  const none = run(null), asph = run(C.asphalt), grass = run(C.grass);
  assert.equal(asph.h, none.h); assert.deepEqual(asph.odo, none.odo);
  assert.notEqual(grass.h, none.h);
  assert.ok(grass.codes.has(C.grass));
});
