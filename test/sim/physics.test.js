import test from 'node:test';
import assert from 'node:assert/strict';
import { createState, stepDynamics, VEHICLE_PARAMS, pedalsFor, resistance, RHO_AIR, G,
  obbContact, resolveContact, separate, sweptContact, boxInertia } from '../../src/sim/physics/index.js';
import { buildGraph } from '../../src/sim/roadGraph.js';
import { VehicleStore } from '../../src/sim/vehicle.js';
import { steerToward } from './steerFixture.js';

const [SEDAN, SUV, TRUCK] = VEHICLE_PARAMS;
const dt = 0.05;
const kmh = (v) => v / 3.6;

function accelTime(P, target) { const st = createState(); let t = 0; while (st.u < target && t < 200) { stepDynamics(st, 0, 1, 0, P, dt); t += dt; } return t; }

test('longitudinal: 0-100 km/h, braking distance, coastdown', () => {
  const ts = accelTime(SEDAN, kmh(100)), tt = accelTime(TRUCK, kmh(100));
  assert.ok(ts > 7.5 && ts < 10.5, `sedan 0-100 ${ts.toFixed(1)} s`);
  assert.ok(tt > 2.5 * ts, `truck 0-100 ${tt.toFixed(1)} s is much slower`);
  const st = createState({ u: kmh(100) }); let t = 0;
  while (st.u > 0 && t < 20) { stepDynamics(st, 0, 0, 1, SEDAN, dt); t += dt; }
  assert.ok(st.s > 34 && st.s < 42, `100-0 in ${st.s.toFixed(1)} m`);
  // coastdown: deceleration over a short window equals (aero + rolling) / m
  const c = createState({ u: 25 }); stepDynamics(c, 0, 0, 0, SEDAN, 0.5);
  const expect = (0.5 * RHO_AIR * SEDAN.CdA * 25 * 25 + SEDAN.Crr * SEDAN.m * G) / SEDAN.m;
  const got = (25 - c.u) / 0.5;
  assert.ok(Math.abs(got - expect) / expect < 0.03, `coastdown ${got.toFixed(3)} vs ${expect.toFixed(3)} m/s^2`);
  // grade: uphill coasting decelerates by g sin(theta) more
  const up = createState({ u: 25 }); stepDynamics(up, 0, 0, 0, SEDAN, 0.5, 0.05);
  assert.ok((c.u - up.u) / 0.5 > 0.45 && (c.u - up.u) / 0.5 < 0.52);
  assert.ok(Math.abs(resistance(SEDAN, 0, 0)) < 1e-9);
});

test('lateral: ~1 g before sliding, understeer gradient, sliding and spinning', () => {
  const ped = {};
  let st = createState({ u: 20 }), slideAy = null;
  for (let k = 0; k < 3000 && slideAy == null; k++) {
    pedalsFor(SEDAN, (20 - st.u) * 0.8, st.u, 0, ped);
    stepDynamics(st, 0.004, ped.gas, ped.brake, SEDAN, dt);
    if (Math.abs(Math.atan2(st.v, st.u)) > 0.08) slideAy = Math.abs(st.u * st.r) / G;
  }
  assert.ok(slideAy > 0.85 && slideAy < 1.1, `slides at ${slideAy && slideAy.toFixed(2)} g`);
  // steady-state steer angle grows faster than the kinematic L/R (understeer)
  const K = [];
  for (const R of [150, 60]) {
    const s = createState({ u: 20 });
    for (let k = 0; k < 1500; k++) { pedalsFor(SEDAN, 20 - s.u, s.u, 0, ped); stepDynamics(s, Math.max(-1, Math.min(1, (20 / R - s.r) * 2)), ped.gas, ped.brake, SEDAN, dt); }
    K.push(((s.steer - SEDAN.L / R) * 57.3) / (s.u * s.r / G));
  }
  for (const k of K) assert.ok(k > 0.5 && k < 6, `understeer gradient ${k.toFixed(2)} deg/g`);
  // hard swerve at 120 km/h: full lock left then right -> the car can lose it
  st = createState({ u: kmh(120) }); let maxBeta = 0;
  for (let k = 0; k < 80; k++) { const t = k * dt; stepDynamics(st, t < 1 ? 1 : -1, 0, 0, SEDAN, dt); maxBeta = Math.max(maxBeta, Math.abs(Math.atan2(st.v, Math.max(st.u, 0.1)))); }
  assert.ok(maxBeta > 0.3, `swerve slip angle ${maxBeta.toFixed(2)} rad`);
});

test('lane change at 100 km/h with the delayed steering controller is stable', () => {
  const g = buildGraph({ nodes: [{ id: 'a', x: 0, z: 0 }, { id: 'b', x: 30000, z: 0 }], edges: [{ nodes: ['a', 'b'], highway: 'motorway', oneway: true, lanes: 3, maxspeed: '65 mph' }] });
  for (const rt of [0.6, 0.9, 1.8]) {
    const vs = new VehicleStore(4); vs._dt = 0.1;
    const i = vs.alloc({ route: Int32Array.of(0), graph: g, dt: 0.1, traits: { reactionTime: rt }, lane: 0 });
    vs.speed[i] = kmh(100);
    const ped = {}; let maxHr = 0, maxOver = 0;
    for (let k = 0; k < 300; k++) {
      const sr = steerToward(vs, i, 1.5 * 3.7);
      pedalsFor(SEDAN, (kmh(100) - vs.speed[i]) * 0.5, vs.speed[i], 0, ped);
      vs.command(i, sr, ped.gas, ped.brake, k);
      vs.integrate(i, 0.1, k * 0.1, g, k);
      maxHr = Math.max(maxHr, Math.abs(vs.headingRel[i])); maxOver = Math.max(maxOver, vs.lat[i] - 1.5 * 3.7);
    }
    assert.ok(Math.abs(vs.lat[i] - 1.5 * 3.7) < 0.2, `rt ${rt}: settles in lane 1 (lat ${vs.lat[i].toFixed(2)})`);
    assert.ok(maxHr < 0.1 && maxOver < 0.6, `rt ${rt}: max heading ${maxHr.toFixed(3)}, overshoot ${maxOver.toFixed(2)}`);
  }
});

function body(x, z, yaw, vx, vz, P = SEDAN, w = 0) {
  return { x, z, yaw, vx, vz, w, hl: P.length / 2, hw: P.width / 2, invMass: 1 / P.m, invInertia: 1 / boxInertia(P.m, P.length, P.width) };
}
const momentum = (...bs) => bs.reduce((m, b) => [m[0] + (b.invMass ? b.vx / b.invMass : 0), m[1] + (b.invMass ? b.vz / b.invMass : 0)], [0, 0]);

test('collisions: head-on, rear-end, T-bone, side-swipe conserve momentum with the expected delta-v', () => {
  const e = 0.2, c = {}, r = {};
  // head-on, equal sedans at 50 km/h each, slight overlap
  let A = body(0, 0, 0, kmh(50), 0), B = body(4.4, 0, Math.PI, -kmh(50), 0);
  const p0 = momentum(A, B);
  assert.ok(obbContact(A, B, c)); resolveContact(A, B, c, e, 0.5, r);
  let p1 = momentum(A, B);
  assert.ok(Math.abs(p1[0] - p0[0]) < 1e-6 && Math.abs(p1[1] - p0[1]) < 1e-6, 'momentum conserved');
  assert.ok(Math.abs(r.dvA * 3.6 - (1 + e) * 50) < 1, `head-on delta-v ${(r.dvA * 3.6).toFixed(1)} km/h`);
  assert.ok(Math.abs(r.dvA - r.dvB) < 1e-6);
  // rear-end: 60 km/h into a stopped car -> each changes by (1+e) * 30 km/h
  A = body(0, 0, 0, kmh(60), 0); B = body(4.4, 0, 0, 0, 0);
  obbContact(A, B, c); resolveContact(A, B, c, e, 0.5, r);
  assert.ok(Math.abs(r.dvB * 3.6 - (1 + e) * 30) < 1, `rear-end delta-v ${(r.dvB * 3.6).toFixed(1)}`);
  assert.ok(B.vx > 0, 'a car at rest gets pushed');
  // rear-end of a truck by a sedan: mass ratio -> the sedan takes most of the delta-v
  A = body(0, 0, 0, kmh(60), 0); B = body(6.7, 0, 0, 0, 0, TRUCK);
  obbContact(A, B, c); resolveContact(A, B, c, e, 0.5, r);
  assert.ok(r.dvA > 5 * r.dvB, 'light car absorbs the delta-v');
  // T-bone: car A (east, 50 km/h) into the side of B (heading north) -> B spins
  A = body(0, 0, 0, kmh(50), 0); B = body(3.1, 1.0, -Math.PI / 2, 0, -kmh(30));
  const q0 = momentum(A, B);
  assert.ok(obbContact(A, B, c)); resolveContact(A, B, c, e, 0.5, r);
  const q1 = momentum(A, B);
  assert.ok(Math.hypot(q1[0] - q0[0], q1[1] - q0[1]) < 1e-6, 'T-bone momentum conserved');
  assert.ok(Math.abs(B.w) > 0.1, 'off-centre impact spins the struck car');
  // glancing side-swipe: parallel, 5 km/h lateral closing -> small delta-v
  A = body(0, 0, 0, kmh(100), kmh(5)); B = body(1.0, 1.8, 0, kmh(100), 0);
  assert.ok(obbContact(A, B, c)); resolveContact(A, B, c, e, 0.5, r);
  assert.ok(r.dvA * 3.6 < 10 && r.dvA > 0, `side-swipe delta-v ${(r.dvA * 3.6).toFixed(1)} km/h is minor`);
  // separation removes the penetration
  A = body(0, 0, 0, 0, 0); B = body(4.0, 0, 0, 0, 0); obbContact(A, B, c); const d0 = c.depth; separate(A, B, c);
  assert.ok(!obbContact(A, B, c) || c.depth < 0.3 * d0);
});

test('no tunnelling at 150 km/h closing speed', () => {
  // a coarse step (0.5 s) of two cars closing at 150 km/h: they swap places, end poses do not overlap
  const v = kmh(75), c = {};
  const A0 = body(-5, 0, 0, v, 0), B0 = body(5, 0, Math.PI, -v, 0);
  const A = body(-5 + v * 0.5, 0, 0, v, 0), B = body(5 - v * 0.5, 0, Math.PI, -v, 0);
  assert.equal(obbContact(A, B, c), null, 'end poses alone do not overlap');
  const t = sweptContact(A0, A, B0, B, c, 0.5);
  assert.ok(t > 0 && t < 1, 'swept test finds the contact');
});

// ---- real terrain (2026-10-24, docs/elevation.md): steep hills
test('hills: a braked car stands still on a 30 % downhill; a locked wreck too', async () => {
  const { stepDynamics, createState, VEHICLE_PARAMS } = await import('../../src/sim/physics/index.js');
  for (const P of VEHICLE_PARAMS) {
    const st = createState();
    for (let k = 0; k < 600; k++) stepDynamics(st, 0, 0, 0.6, P, 0.05, -0.30);
    assert.ok(st.s < 0.01 && st.u < 0.01, `${P.name} crept ${st.s.toFixed(3)} m with the brake held`);
    const w = createState();
    for (let k = 0; k < 600; k++) stepDynamics(w, 0, 0, 0, P, 0.05, -0.30, true);
    assert.ok(w.s < 0.01, `${P.name} wreck slid ${w.s.toFixed(3)} m`);
  }
});
test('hills: every vehicle type starts and climbs a 25 % grade; coasting downhill accelerates', async () => {
  const { stepDynamics, createState, VEHICLE_PARAMS } = await import('../../src/sim/physics/index.js');
  for (const P of VEHICLE_PARAMS) {
    const st = createState();
    for (let k = 0; k < 400; k++) stepDynamics(st, 0, 1, 0, P, 0.05, 0.25); // 20 s full throttle from rest
    assert.ok(st.u > 4 && st.s > 30, `${P.name} on 25 %: u ${st.u.toFixed(2)} m/s after 20 s, s ${st.s.toFixed(1)} m`);
    const d = createState({ u: 5 });
    for (let k = 0; k < 40; k++) stepDynamics(d, 0, 0, 0, P, 0.05, -0.2);
    assert.ok(d.u > 5 + 1.5, `${P.name} coasting down 20 % did not speed up (${d.u.toFixed(2)})`);
  }
});
