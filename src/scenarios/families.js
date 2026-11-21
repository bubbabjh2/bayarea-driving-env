// Scenario families: small synthetic road networks with their trips, parameterised and randomised
// by seed. Every family builds into a World (builder.js), so it comes out in the real data format.
//
//   const w = makeScenario('junction', 7, { control: 'signal', movement: 'left' });   // World
//   FAMILIES[name] = { desc, variants?, build(w, R, o) }   (o = randomised params + overrides)
//
// Coordinates: metres, x east, z south, y up; a scenario fits in about 3 x 3 km around (0, 0).
// Trips depart from 07:00 (signals run, ramp meters are on 06:00-10:00).
import { World } from './builder.js';
import { makeRng, hashSeed } from '../sim/rng.js';

const T0 = 7 * 3600;
const H = Math.PI / 2;

// ------------------------------------------------------------------ helpers
const lerp = (a, b, t) => a + (b - a) * t;
/** Straight road between two existing nodes, shape points every `step` m, elevation interpolated (or yFn(t)). */
function straight(w, A, B, opts, { step = 40, yFn = null } = {}) {
  const ya = w.ny[A], yb = w.ny[B];
  return w.road([A, ...World.line(w.nx[A], w.nz[A], w.nx[B], w.nz[B], step, yFn || ((t) => lerp(ya, yb, t))), B], opts);
}
/** New end node at `len` m from node N in direction (dx, dz); straight road end -> N (or N -> end with out=true). */
function stub(w, N, dx, dz, len, opts, { out = false, name = null, y = null } = {}) {
  const d = Math.hypot(dx, dz) || 1, E = w.node(w.nx[N] + (dx / d) * len, w.nz[N] + (dz / d) * len, y ?? w.ny[N], name);
  if (out) straight(w, N, E, opts); else straight(w, E, N, opts);
  return E;
}
/** Smooth crest/sag profile with max grade g over length L: y(t) = H sin^2(pi t), H = g L / pi. */
const bump = (g, L, sign = 1) => (t) => sign * ((g * L) / Math.PI) * Math.sin(Math.PI * t) ** 2;
/** Advisory speed on a curve of radius R (lateral 2.0-2.5 m/s^2), capped by the class default, multiple of 5 km/h. */
const curveKph = (R, cap) => Math.max(15, Math.min(cap, Math.floor((Math.sqrt(R * 2.2) * 3.6) / 5) * 5));
const pick = (R, arr) => arr[R.int(arr.length)];
/** count of trips for `rate` veh/min over a `win` s window. */
const cnt = (rate, win) => Math.max(1, Math.round((rate * win) / 60));

// ------------------------------------------------------------------ families
export const FAMILIES = {
  straight: {
    desc: 'straight road, 1-5 lanes per direction (1-2: two-way arterial / local, 3-5: one-way freeway carriageway)',
    params: (R) => ({ lanes: 1 + R.int(5), length: 1500 + R.int(1500), rate: 6 + R.int(20), win: 600 }),
    build(w, R, o) {
      const fw = o.lanes >= 3;
      const cls = fw ? 'motorway' : o.lanes === 2 ? 'primary' : pick(R, ['residential', 'tertiary', 'secondary']);
      const a = w.node(-o.length / 2, 0, 0, 'west'), b = w.node(o.length / 2, 0, 0, 'east');
      straight(w, a, b, { cls, lanes: o.lanes, opp: fw ? 0 : o.lanes }, { step: 50 });
      const n = cnt(o.rate * o.lanes / 2, o.win);
      w.trip({ from: a, to: b, count: n, depart: [T0, T0 + o.win] });
      if (!fw) w.trip({ from: b, to: a, count: n, depart: [T0, T0 + o.win] });
      w.view = { x: 0, z: 0, span: 500 };
    },
  },
  curve: {
    desc: 'single curve of radius 30-800 m and 45-180 deg sweep between straights; advisory speed from the radius; optional grade up to 6 %',
    params: (R) => ({ radius: Math.round(Math.exp(lerp(Math.log(30), Math.log(800), R.random()))), sweep: lerp(0.8, Math.PI, R.random()), dir: R.random() < 0.5 ? 1 : -1,
      lanes: 1 + R.int(2), grade: R.random() < 0.4 ? lerp(0.02, 0.06, R.random()) : 0, rate: 4 + R.int(10), win: 600 }),
    build(w, R, o) {
      const cls = o.radius < 80 ? 'residential' : o.radius < 200 ? 'tertiary' : o.radius < 450 ? 'secondary' : 'primary';
      const kph = curveKph(o.radius, { residential: 40, tertiary: 50, secondary: 65, primary: 80 }[cls]);
      const lanes = o.radius < 80 ? 1 : o.lanes, appr = 400, Rr = o.radius;
      // curve centred so that it starts at (0, 0) heading +x; dir +1 = right turn (toward +z)
      const cz = o.dir * Rr, a0 = o.dir > 0 ? -H : H, a1 = a0 + o.dir * o.sweep;
      const P0 = w.node(0, 0, 0), endX = Rr * Math.cos(a1), endZ = cz + Rr * Math.sin(a1);
      const rise = Math.min(30, o.grade * Rr * o.sweep);
      const P1 = w.node(endX, endZ, rise);
      w.road([P0, ...World.arc(0, cz, Rr, a0, a1, { yFn: (t) => rise * t }), P1], { cls, lanes, opp: lanes, speedKph: kph });
      const hx = -Math.sin(a1) * o.dir, hz = Math.cos(a1) * o.dir;   // exit tangent
      const A = stub(w, P0, -1, 0, appr, { cls, lanes, opp: lanes, speedKph: Math.min(kph + 20, 80) }, { name: 'start' });
      const B = stub(w, P1, hx, hz, appr, { cls, lanes, opp: lanes, speedKph: Math.min(kph + 20, 80) }, { out: true, name: 'end' });
      w.notes.push(`radius ${Rr} m, sweep ${(o.sweep * 180 / Math.PI).toFixed(0)} deg, ${cls} ${kph} km/h, grade ${(o.grade * 100).toFixed(1)} %`);
      w.trip({ from: A, to: B, count: cnt(o.rate, o.win), depart: [T0, T0 + o.win] });
      w.trip({ from: B, to: A, count: cnt(o.rate, o.win), depart: [T0, T0 + o.win] });
      w.view = { x: (w.nx[P0] + w.nx[P1]) / 2, z: (w.nz[P0] + w.nz[P1]) / 2, span: Math.max(250, Math.min(1500, Rr * 2.2)) };
    },
  },
  sbend: {
    desc: 'S-bend: two reversing curves (radius 40-400 m) with a short tangent',
    params: (R) => ({ radius: Math.round(lerp(40, 400, R.random() ** 1.5)), sweep: lerp(0.6, 1.5, R.random()), tangent: R.int(80), lanes: 1 + R.int(2), rate: 4 + R.int(10), win: 600 }),
    build(w, R, o) {
      const Rr = o.radius, cls = Rr < 100 ? 'residential' : Rr < 250 ? 'tertiary' : 'secondary';
      const kph = curveKph(Rr, { residential: 40, tertiary: 50, secondary: 65 }[cls]), lanes = Rr < 100 ? 1 : o.lanes;
      const opts = { cls, lanes, opp: lanes, speedKph: kph };
      // right curve from (0,0) heading +x, then a tangent, then a left curve
      const P0 = w.node(0, 0);
      const a1 = -H + o.sweep, p1 = [Rr * Math.cos(a1), Rr + Rr * Math.sin(a1)];
      const hx = Math.cos(o.sweep), hz = Math.sin(o.sweep);
      const P1 = w.node(p1[0], p1[1]);
      w.road([P0, ...World.arc(0, Rr, Rr, -H, a1), P1], opts);
      const P2 = w.node(p1[0] + hx * o.tangent, p1[1] + hz * o.tangent);
      if (o.tangent > 5) straight(w, P1, P2, opts, { step: 20 }); else w.notes.push('no tangent');
      const Q = o.tangent > 5 ? P2 : P1;
      // left curve: centre to the left of heading (hx, hz): left normal = (hz, -hx)
      const cx = w.nx[Q] + hz * Rr, cz = w.nz[Q] - hx * Rr, b0 = Math.atan2(w.nz[Q] - cz, w.nx[Q] - cx), b1 = b0 - o.sweep;
      const P3 = w.node(cx + Rr * Math.cos(b1), cz + Rr * Math.sin(b1));
      w.road([Q, ...World.arc(cx, cz, Rr, b0, b1), P3], opts);
      const A = stub(w, P0, -1, 0, 350, { ...opts, speedKph: Math.min(80, kph + 15) }, { name: 'start' });
      const B = stub(w, P3, 1, 0, 350, { ...opts, speedKph: Math.min(80, kph + 15) }, { out: true, name: 'end' });
      w.notes.push(`radius ${Rr} m, sweep ${(o.sweep * 180 / Math.PI).toFixed(0)} deg, tangent ${o.tangent} m, ${kph} km/h`);
      w.trip({ from: A, to: B, count: cnt(o.rate, o.win), depart: [T0, T0 + o.win] });
      w.trip({ from: B, to: A, count: cnt(o.rate, o.win), depart: [T0, T0 + o.win] });
      w.view = { x: w.nx[Q], z: w.nz[Q], span: Math.max(300, Rr * 3) };
    },
  },
  hill: {
    desc: 'crest and sag: smooth vertical profile with max grade 3-8 % (limited sight over the crest); optional horizontal curve on the crest',
    params: (R) => ({ grade: lerp(0.03, 0.08, R.random()), length: 900 + R.int(900), lanes: 1 + R.int(2), curve: R.random() < 0.4, rate: 5 + R.int(10), win: 600 }),
    build(w, R, o) {
      const cls = o.grade > 0.06 ? 'residential' : 'secondary', L = o.length;
      const opts = { cls, lanes: o.grade > 0.06 ? 1 : o.lanes, opp: o.grade > 0.06 ? 1 : o.lanes };
      const A = w.node(-L - 300, 0, 0, 'start'), P = w.node(-L, 0, 0), M = w.node(0, 0, 0), Q = w.node(L, 0, 0), B = w.node(L + 300, 0, 0, 'end');
      straight(w, A, P, opts);
      straight(w, P, M, opts, { yFn: bump(o.grade, L), step: 25 });          // crest
      let end = B;
      if (o.curve) {   // a bend (radius 200 m, 60 deg) right behind the crest, then flat
        const Rr = 200, sw = 1.05, E1 = w.node(Rr * Math.sin(sw), Rr - Rr * Math.cos(sw));
        w.road([M, ...World.arc(0, Rr, Rr, -H, -H + sw), E1], { ...opts, speedKph: curveKph(Rr, 60) });
        end = stub(w, E1, Math.cos(sw), Math.sin(sw), 400, opts, { out: true, name: 'end' });
      } else {
        straight(w, M, Q, opts, { yFn: bump(o.grade, L, -1), step: 25 });    // sag (a dip below grade)
        straight(w, Q, B, opts);
      }
      w.notes.push(`crest${o.curve ? ' + bend behind it' : ' + sag'}, max grade ${(o.grade * 100).toFixed(1)} %, height ${(o.grade * L / Math.PI).toFixed(1)} m`);
      w.trip({ from: A, to: end, count: cnt(o.rate, o.win), depart: [T0, T0 + o.win] });
      w.trip({ from: end, to: A, count: cnt(o.rate, o.win), depart: [T0, T0 + o.win] });
      w.view = { x: -L / 2, z: 0, span: 900 };
    },
  },
  lanes: {
    desc: 'lane add / lane drop (merge taper) on a one-way carriageway: n -> n-1 (LANE_DROP) or n -> n+1, arterial or freeway',
    params: (R) => ({ kind: R.random() < 0.6 ? 'drop' : 'add', lanes: 2 + R.int(3), freeway: R.random() < 0.5, rate: 8 + R.int(12), win: 600 }),
    build(w, R, o) {
      const cls = o.freeway ? 'motorway' : 'primary', n1 = o.lanes, n2 = o.kind === 'drop' ? n1 - 1 : n1 + 1;
      const A = w.node(-1200, 0, 0, 'start'), M = w.split(w.node(0, 0)), B = w.node(1200, 0, 0, 'end');
      straight(w, A, M, { cls, lanes: n1, opp: 0 }); straight(w, M, B, { cls, lanes: n2, opp: 0 });
      w.notes.push(`${n1} -> ${n2} lanes (${cls})`);
      w.trip({ from: A, to: B, count: cnt(o.rate * n1 / 2, o.win), depart: [T0, T0 + o.win] });
      w.view = { x: 0, z: 0, span: 450 };
    },
  },
  merge: {
    desc: 'freeway on-ramp: 1-lane ramp joining a 2-4 lane mainline, with a 300 m acceleration lane (as the real build synthesises) or a plain gore merge',
    params: (R) => ({ lanes: 2 + R.int(3), accel: R.random() < 0.6, rate: 10 + R.int(20), rampRate: 4 + R.int(10), win: 600 }),
    build(w, R, o) {
      const n = o.lanes, fw = { cls: 'motorway', lanes: n, opp: 0 }, ramp = { cls: 'motorway_link', lanes: 1, opp: 0 };
      const A = w.node(-1500, 0, 0, 'start'), M = w.node(0, 0), B = w.node(1500, 0, 0, 'end');
      straight(w, A, M, fw);
      if (o.accel) { const D = w.split(w.node(300, 0)); straight(w, M, D, { ...fw, lanes: n + 1, aux: true }); straight(w, D, B, fw); }
      else straight(w, M, B, fw);
      // ramp from the right (+z side for eastbound): starts 450 m back, 120 m off, joins tangentially
      const RS = w.node(-600, 150, 0, 'ramp'), r1 = w.node(-250, 40, 0);
      straight(w, RS, r1, ramp);
      w.road([r1, [-180, 22], [-120, 10], [-60, 3], M], ramp);
      w.notes.push(`${n}-lane mainline, ${o.accel ? '300 m acceleration lane' : 'gore merge (no added lane)'}`);
      w.trip({ from: A, to: B, count: cnt(o.rate * n / 3, o.win), depart: [T0, T0 + o.win] });
      w.trip({ from: RS, to: B, count: cnt(o.rampRate, o.win), depart: [T0, T0 + o.win] });
      w.view = { x: -50, z: 20, span: 650 };
    },
  },
  diverge: {
    desc: 'freeway off-ramp: exit from a 2-4 lane mainline (option lane, or a 300 m added exit lane); through and exiting trips',
    params: (R) => ({ lanes: 2 + R.int(3), exitLane: R.random() < 0.5, rate: 10 + R.int(20), exitShare: lerp(0.2, 0.5, R.random()), win: 600 }),
    build(w, R, o) {
      const n = o.lanes, fw = { cls: 'motorway', lanes: n, opp: 0 }, ramp = { cls: 'motorway_link', lanes: 1, opp: 0 };
      const A = w.node(-1500, 0, 0, 'start'), X = w.node(0, 0), B = w.node(1500, 0, 0, 'end');
      if (o.exitLane) { const D = w.split(w.node(-300, 0)); straight(w, A, D, fw); straight(w, D, X, { ...fw, lanes: n + 1, aux: true }); }
      else straight(w, A, X, fw);
      straight(w, X, B, fw);
      const r1 = w.node(250, 40, 0), RE = w.node(600, 150, 0, 'exit');
      w.road([X, [60, 3], [120, 10], [180, 22], r1], ramp);
      straight(w, r1, RE, ramp);
      const N = cnt(o.rate * n / 3, o.win), ne = Math.max(1, Math.round(N * o.exitShare));
      w.trip({ from: A, to: B, count: N - ne, depart: [T0, T0 + o.win] });
      w.trip({ from: A, to: RE, count: ne, depart: [T0, T0 + o.win] });
      w.notes.push(`${n}-lane mainline, ${o.exitLane ? 'added exit lane' : 'option-lane exit'}, exit share ${(o.exitShare * 100).toFixed(0)} %`);
      w.view = { x: 50, z: 20, span: 650 };
    },
  },
  weave: {
    desc: 'weave: on-ramp followed 300-700 m later by an off-ramp, joined by an auxiliary lane; ramp->mainline, mainline->exit and ramp->exit trips',
    params: (R) => ({ lanes: 2 + R.int(3), weave: 300 + R.int(400), rate: 10 + R.int(15), rampRate: 4 + R.int(8), win: 600 }),
    build(w, R, o) {
      const n = o.lanes, fw = { cls: 'motorway', lanes: n, opp: 0 }, ramp = { cls: 'motorway_link', lanes: 1, opp: 0 };
      const A = w.node(-1500, 0, 0, 'start'), M = w.node(0, 0), X = w.node(o.weave, 0), B = w.node(o.weave + 1500, 0, 0, 'end');
      straight(w, A, M, fw); straight(w, M, X, { ...fw, lanes: n + 1, aux: true }); straight(w, X, B, fw);
      const RS = w.node(-600, 150, 0, 'ramp'), r1 = w.node(-250, 40, 0);
      straight(w, RS, r1, ramp); w.road([r1, [-180, 22], [-120, 10], [-60, 3], M], ramp);
      const r2 = w.node(o.weave + 250, 40, 0), RE = w.node(o.weave + 600, 150, 0, 'exit');
      w.road([X, [o.weave + 60, 3], [o.weave + 120, 10], [o.weave + 180, 22], r2], ramp); straight(w, r2, RE, ramp);
      const N = cnt(o.rate * n / 3, o.win), q = cnt(o.rampRate, o.win);
      w.trip({ from: A, to: B, count: Math.round(N * 0.7), depart: [T0, T0 + o.win] });
      w.trip({ from: A, to: RE, count: Math.max(1, Math.round(N * 0.3)), depart: [T0, T0 + o.win] });
      w.trip({ from: RS, to: B, count: Math.max(1, Math.round(q * 0.7)), depart: [T0, T0 + o.win] });
      w.trip({ from: RS, to: RE, count: Math.max(1, Math.round(q * 0.3)), depart: [T0, T0 + o.win] });
      w.notes.push(`${n}+1 lane weave, ${o.weave} m`);
      w.view = { x: o.weave / 2, z: 30, span: o.weave + 600 };
    },
  },
  rampmeter: {
    desc: 'metered on-ramp: ramp meter (6 s cycle, one car per green, 06:00-10:00) mid-ramp, then merge with an acceleration lane',
    params: (R) => ({ lanes: 2 + R.int(3), rate: 10 + R.int(20), rampRate: 4 + R.int(6), win: 600 }),
    build(w, R, o) {
      const n = o.lanes, fw = { cls: 'motorway', lanes: n, opp: 0 }, ramp = { cls: 'motorway_link', lanes: 1, opp: 0 };
      const A = w.node(-1500, 0, 0, 'start'), M = w.node(0, 0), D = w.split(w.node(300, 0)), B = w.node(1500, 0, 0, 'end');
      straight(w, A, M, fw); straight(w, M, D, { ...fw, lanes: n + 1, aux: true }); straight(w, D, B, fw);
      const RS = w.node(-750, 190, 0, 'ramp'), r1 = w.node(-250, 40, 0);
      straight(w, RS, r1, ramp); w.road([r1, [-180, 22], [-120, 10], [-60, 3], M], ramp);
      w.control('ramp_meter', -1, { midBlock: { from: RS, to: r1, at: 330 } });
      w.trip({ from: A, to: B, count: cnt(o.rate * n / 3, o.win), depart: [T0, T0 + o.win] });
      w.trip({ from: RS, to: B, count: cnt(o.rampRate, o.win), depart: [T0, T0 + o.win] });
      w.notes.push(`${n}-lane mainline, metered ramp`);
      w.view = { x: -250, z: 60, span: 800 };
    },
  },
  junction: {
    desc: '4-way or T junction of a major and a minor road; control signal | allway | twoway (stop on minor) | giveway | none; movements left / right / through with cross traffic; protected lefts on 2-lane primaries (signal plan), permissive otherwise',
    variants: { control: ['signal', 'allway', 'twoway', 'giveway', 'none'], movement: ['left', 'right', 'through', 'mixed'], shape: ['cross', 'tee'], major: ['primary2', 'secondary1', 'residential1'] },
    params: (R) => ({ control: pick(R, ['signal', 'allway', 'twoway', 'giveway', 'none']), movement: pick(R, ['left', 'right', 'through', 'mixed']), shape: R.random() < 0.7 ? 'cross' : 'tee',
      major: pick(R, ["primary2", "secondary1", "residential1"]), angle: lerp(-0.25, 0.25, R.random()), rate: 1 + R.int(3), crossRate: 1 + R.int(4), win: 600 }),
    build(w, R, o) {
      if (o.control === 'allway' && o.major === 'primary2') o.major = 'secondary1';
      const maj = { primary2: { cls: 'primary', lanes: 2 }, secondary1: { cls: 'secondary', lanes: 1 }, residential1: { cls: 'residential', lanes: 1 } }[o.major];
      const min = o.major === 'residential1' || o.control === 'allway' ? { cls: maj.cls, lanes: 1 } : { cls: pick(R, ['residential', 'tertiary']), lanes: 1 };
      const L = 450, C = w.node(0, 0);
      const ends = {};
      ends.W = stub(w, C, -1, 0, L, { ...maj, opp: maj.lanes }, { name: 'W' });
      ends.E = stub(w, C, 1, 0, L, { ...maj, opp: maj.lanes }, { name: 'E' });
      const ax = Math.sin(o.angle), az = Math.cos(o.angle);       // minor road axis (skew up to +-14 deg)
      ends.S = stub(w, C, ax, az, L, { ...min, opp: min.lanes }, { name: 'S' });
      if (o.shape === 'cross') ends.N = stub(w, C, -ax, -az, L, { ...min, opp: min.lanes }, { name: 'N' });
      const minorLegs = [ends.S, ends.N].filter((x) => x != null);
      if (o.control === 'signal') w.control('signal', C);
      else if (o.control === 'allway') w.control('stop', C, { allWay: true });
      else if (o.control === 'twoway') w.control('stop', C, { approaches: minorLegs });
      else if (o.control === 'giveway') w.control('give_way', C, { approaches: minorLegs });
      // test movements: from the south (minor) approach + the major road's own turns; cross traffic on the major road
      const win = o.win, d = [T0, T0 + win];
      const mv = o.movement;
      const n = cnt(o.rate, win), xn = cnt(o.crossRate, win);
      if (mv === 'left' || mv === 'mixed') { w.trip({ from: ends.S, to: ends.W, count: n, depart: d }); w.trip({ from: ends.E, to: ends.S, count: n, depart: d }); }
      if (mv === 'right' || mv === 'mixed') { w.trip({ from: ends.S, to: ends.E, count: n, depart: d }); w.trip({ from: ends.W, to: ends.S, count: n, depart: d }); }
      if ((mv === 'through' || mv === 'mixed') && ends.N != null) { w.trip({ from: ends.S, to: ends.N, count: n, depart: d }); w.trip({ from: ends.N, to: ends.S, count: n, depart: d }); }
      if (mv === 'through' && ends.N == null) { w.trip({ from: ends.S, to: ends.W, count: n, depart: d }); }
      w.trip({ from: ends.W, to: ends.E, count: xn, depart: d });
      w.trip({ from: ends.E, to: ends.W, count: xn, depart: d });
      if (ends.N != null && mv !== 'through') w.trip({ from: ends.N, to: ends.S, count: Math.max(1, xn >> 1), depart: d });
      w.notes.push(`${o.shape} ${o.control} ${mv}; major ${maj.cls} x${maj.lanes}, minor ${min.cls}, skew ${(o.angle * 180 / Math.PI).toFixed(0)} deg`);
      w.view = { x: 0, z: 0, span: 170 };
    },
  },
  grid: {
    desc: 'grid of blocks (150-250 m): arterials every 2-3 streets, signals at arterial x arterial, stop signs on the locals at arterials, all-way stops between locals; random trips across the grid',
    params: (R) => ({ nx: 4 + R.int(3), nz: 4 + R.int(3), block: 150 + R.int(100), every: 2 + R.int(2), trips: 60 + R.int(80), win: 900, urban: false }),
    build(w, R, o) { buildGrid(w, R, o); },
  },
  urban: {
    desc: 'dense urban grid: 2-lane primaries and secondaries, a signal at every intersection (protected lefts where the plan gives them), short blocks, heavy demand',
    params: (R) => ({ nx: 5 + R.int(3), nz: 5 + R.int(3), block: 110 + R.int(60), every: 1, trips: 150 + R.int(150), win: 900, urban: true }),
    build(w, R, o) { buildGrid(w, R, o); },
  },
  interchange: {
    desc: 'diamond interchange: freeway (two one-way carriageways) under an arterial bridge (7 m deck), four ramps, ramp terminals signalised or stop-controlled; through, exiting and entering trips',
    params: (R) => ({ lanes: 2 + R.int(2), terminal: R.random() < 0.6 ? 'signal' : 'stop', rate: 10 + R.int(15), rampRate: 3 + R.int(5), win: 900 }),
    build(w, R, o) {
      const n = o.lanes, fw = { cls: 'motorway', lanes: n, opp: 0 }, ramp = { cls: 'motorway_link', lanes: 1, opp: 0 };
      const art = { cls: 'primary', lanes: 2, opp: 2 }, D = 9 + n * 1.85;   // carriageway offset from the median line
      // freeway: EB at z = +D (right-hand side), WB at z = -D
      const EB0 = w.node(-1800, D, 0, 'EBstart'), EBx = w.node(-520, D), EBm = w.node(520, D), EBa = w.split(w.node(820, D)), EB1 = w.node(1800, D, 0, 'EBend');
      const WB0 = w.node(1800, -D, 0, 'WBstart'), WBx = w.node(520, -D), WBm = w.node(-520, -D), WBa = w.split(w.node(-820, -D)), WB1 = w.node(-1800, -D, 0, 'WBend');
      straight(w, EB0, EBx, fw); straight(w, EBx, EBm, fw); straight(w, EBm, EBa, { ...fw, lanes: n + 1, aux: true }); straight(w, EBa, EB1, fw);
      straight(w, WB0, WBx, fw); straight(w, WBx, WBm, fw); straight(w, WBm, WBa, { ...fw, lanes: n + 1, aux: true }); straight(w, WBa, WB1, fw);
      // arterial along z over the freeway: deck 7 m between the terminals, 6 % approaches
      const tz = 150, Y = 7;
      const TS = w.node(0, tz, Y), TN = w.node(0, -tz, Y);
      const AS = w.node(0, tz + 600, 0, 'Sart'), AN = w.node(0, -tz - 600, 0, 'Nart');
      const prof = (t) => { const z = lerp(tz, tz + 600, t) - tz; return Math.max(0, Y - z * 0.06); };
      w.road([AS, ...World.line(0, tz + 600, 0, tz, 25, (t) => prof(1 - t)), TS], art);
      w.road([TS, ...World.line(0, tz, 0, -tz, 25, () => Y), TN], { ...art, bridge: true, layer: 1 });
      w.road([TN, ...World.line(0, -tz, 0, -tz - 600, 25, (t) => prof(t)), AN], art);
      // ramps (climb 0 -> 7 m), EB off to the south terminal, EB on from it; WB mirrored on the north side
      const rampPts = (x0, z0, x1, z1, y0, y1, bend) => { const out = []; for (let k = 1; k < 16; k++) { const t = k / 16, u = t * t * (3 - 2 * t); out.push([lerp(x0, x1, t), lerp(z0, z1, u) + bend * Math.sin(Math.PI * t), lerp(y0, y1, t)]); } return out; };
      w.road([EBx, ...rampPts(-520, D, -40, tz, 0, Y, 25), TS], ramp);
      w.road([TS, ...rampPts(40, tz, 520, D, Y, 0, 25), EBm], ramp);
      w.road([WBx, ...rampPts(520, -D, 40, -tz, 0, Y, -25), TN], ramp);
      w.road([TN, ...rampPts(-40, -tz, -520, -D, Y, 0, -25), WBm], ramp);
      if (o.terminal === 'signal') { w.control('signal', TS); w.control('signal', TN); }
      else { w.control('stop', TS, { approaches: [EBx] }); w.control('stop', TN, { approaches: [WBx] }); }
      const d = [T0, T0 + o.win], N = cnt(o.rate * n / 3, o.win), q = cnt(o.rampRate, o.win);
      w.trip({ from: EB0, to: EB1, count: N, depart: d }); w.trip({ from: WB0, to: WB1, count: N, depart: d });
      w.trip({ from: EB0, to: AN, count: q, depart: d }); w.trip({ from: EB0, to: AS, count: q, depart: d });
      w.trip({ from: WB0, to: AS, count: q, depart: d }); w.trip({ from: AS, to: EB1, count: q, depart: d });
      w.trip({ from: AN, to: WB1, count: q, depart: d }); w.trip({ from: AN, to: EB1, count: q, depart: d });
      w.trip({ from: AS, to: AN, count: q, depart: d }); w.trip({ from: AN, to: AS, count: q, depart: d });
      w.notes.push(`${n}-lane freeway, ${o.terminal} terminals, arterial deck 7 m`);
      w.view = { x: 0, z: 0, span: 1000 };
    },
  },
  overpass: {
    desc: 'grade separation: an arterial or local road crossing a freeway (or another road) on a 7 m bridge with 6-8 % approaches, no connection; traffic on both',
    params: (R) => ({ under: R.random() < 0.6 ? 'motorway' : 'primary', over: pick(R, ['secondary', 'residential', 'primary']), angle: lerp(-0.5, 0.5, R.random()), rate: 6 + R.int(10), win: 600 }),
    build(w, R, o) {
      const under = o.under === 'motorway' ? { cls: 'motorway', lanes: 3, opp: 0 } : { cls: 'primary', lanes: 2, opp: 2 };
      const over = { cls: o.over, lanes: o.over === 'primary' ? 2 : 1, opp: o.over === 'primary' ? 2 : 1 };
      const U0 = w.node(-1200, 0, 0, 'Ustart'), U1 = w.node(1200, 0, 0, 'Uend');
      straight(w, U0, U1, under, { step: 50 });
      const g = o.over === 'residential' ? 0.08 : 0.06, Y = 7, ramp = Y / g, ax = Math.sin(o.angle), az = Math.cos(o.angle);
      const L = 800, deck = 40;
      const pt = (s) => [ax * s, az * s];
      const yAt = (s) => { const a = Math.abs(s); return a <= deck ? Y : Math.max(0, Y - (a - deck) * g); };
      const O0 = w.node(...pt(-L), 0, 'Ostart'), O1 = w.node(...pt(L), 0, 'Oend');
      const Da = w.split(w.node(...pt(-deck), Y)), Db = w.split(w.node(...pt(deck), Y));
      const seg = (s0, s1) => { const out = []; const n = Math.max(1, Math.round(Math.abs(s1 - s0) / 20)); for (let k = 1; k < n; k++) { const s = lerp(s0, s1, k / n); out.push([...pt(s), yAt(s)]); } return out; };
      w.road([O0, ...seg(-L, -deck), Da], over); w.road([Da, ...seg(-deck, deck), Db], { ...over, bridge: true, layer: 1 }); w.road([Db, ...seg(deck, L), O1], over);
      const d = [T0, T0 + o.win];
      w.trip({ from: U0, to: U1, count: cnt(o.rate, o.win), depart: d });
      if (under.opp) w.trip({ from: U1, to: U0, count: cnt(o.rate, o.win), depart: d });
      w.trip({ from: O0, to: O1, count: cnt(o.rate * 0.6, o.win), depart: d }); w.trip({ from: O1, to: O0, count: cnt(o.rate * 0.6, o.win), depart: d });
      w.notes.push(`${over.cls} over ${under.cls}, ${(g * 100).toFixed(0)} % approaches, ramp ${ramp.toFixed(0)} m`);
      w.view = { x: 0, z: 0, span: 450 };
    },
  },
  roundabout: {
    desc: 'single-lane roundabout (radius 22-40 m), 3-4 legs, give-way on entry; all movements',
    params: (R) => ({ radius: 22 + R.int(18), legs: R.random() < 0.75 ? 4 : 3, rate: 3 + R.int(5), win: 600 }),
    build(w, R, o) {
      const Rr = o.radius, ring = { cls: 'residential', lanes: 1, opp: 0, speedKph: 25 }, leg = { cls: 'tertiary', lanes: 1, opp: 1 };
      const angs = o.legs === 4 ? [H, 0, -H, Math.PI] : [H, 0, Math.PI];       // south, east, north, west (atan2(z, x))
      const nodes = angs.map((a) => w.node(Rr * Math.cos(a), Rr * Math.sin(a)));
      // circulate counter-clockwise on the map (right-hand traffic): angle decreasing
      for (let k = 0; k < nodes.length; k++) {
        const a0 = angs[k], a1n = angs[(k + 1) % angs.length];
        let a1 = a1n; while (a1 >= a0) a1 -= 2 * Math.PI;
        w.road([nodes[k], ...World.arc(0, 0, Rr, a0, a1, { dA: 0.12, step: 8 }), nodes[(k + 1) % nodes.length]], ring);
      }
      const ends = angs.map((a, k) => stub(w, nodes[k], Math.cos(a), Math.sin(a), 400, leg, { name: 'leg' + k }));
      for (let k = 0; k < nodes.length; k++) w.control('give_way', nodes[k], { approaches: [ends[k]] });
      const d = [T0, T0 + o.win];
      for (let i = 0; i < ends.length; i++) for (let j = 0; j < ends.length; j++) if (i !== j) w.trip({ from: ends[i], to: ends[j], count: cnt(o.rate / (ends.length - 1), o.win), depart: d });
      w.notes.push(`radius ${Rr} m, ${o.legs} legs`);
      w.view = { x: 0, z: 0, span: Rr * 7 };
    },
  },
  obstacle: {
    desc: 'stopped (disabled) cars in the travel lanes: multi-lane one-way road (steer around into a free lane) or two-lane two-way road (pass through the opposing lane when clear)',
    params: (R) => ({ lanes: 1 + R.int(3), count: 1 + R.int(3), rate: 5 + R.int(10), win: 600 }),
    build(w, R, o) {
      const twoWay = o.lanes === 1, cls = twoWay ? 'secondary' : 'primary';
      const A = w.node(-1000, 0, 0, 'start'), B = w.node(1000, 0, 0, 'end');
      straight(w, A, B, { cls, lanes: o.lanes, opp: twoWay ? 1 : 0 }, { step: 50 });
      for (let k = 0; k < o.count; k++) w.obstacle({ from: A, to: B, at: 500 + k * (1000 / o.count) + R.int(80), lane: R.int(o.lanes) });
      w.view0 = -1000 + w.obstacles[0].at;
      w.trip({ from: A, to: B, count: cnt(o.rate, o.win), depart: [T0, T0 + o.win] });
      if (twoWay) w.trip({ from: B, to: A, count: cnt(o.rate * 0.5, o.win), depart: [T0, T0 + o.win] });
      w.notes.push(`${o.count} stopped car(s), ${o.lanes} lane(s) ${twoWay ? 'two-way' : 'one-way'}`);
      w.view = { x: w.view0 ?? 0, z: 0, span: 500 };
    },
  },
};

function buildGrid(w, R, o) {
  const { nx, nz, block } = o, art = (i) => i % o.every === 0;
  const ids = [];
  for (let i = 0; i < nx; i++) { ids.push([]); for (let j = 0; j < nz; j++) ids[i].push(w.node((i - (nx - 1) / 2) * block, (j - (nz - 1) / 2) * block)); }
  const kind = (isArt) => (o.urban ? (isArt ? { cls: 'primary', lanes: 2 } : { cls: 'secondary', lanes: 2 }) : isArt ? { cls: 'secondary', lanes: 2 } : { cls: 'residential', lanes: 1 });
  // streets along x (index j) and along z (index i)
  for (let j = 0; j < nz; j++) { const k = kind(art(j)); for (let i = 0; i + 1 < nx; i++) straight(w, ids[i][j], ids[i + 1][j], { ...k, opp: k.lanes }); }
  for (let i = 0; i < nx; i++) { const k = kind(art(i)); for (let j = 0; j + 1 < nz; j++) straight(w, ids[i][j], ids[i][j + 1], { ...k, opp: k.lanes }); }
  // stubs out of the boundary (origins / destinations); zEnd[i] = [north end, south end], xEnd[j] = [west, east]
  const ends = [], zEnd = [], xEnd = [];
  for (let i = 0; i < nx; i++) { const k = kind(art(i)); zEnd[i] = [stub(w, ids[i][0], 0, -1, 250, { ...k, opp: k.lanes }), stub(w, ids[i][nz - 1], 0, 1, 250, { ...k, opp: k.lanes })]; ends.push(...zEnd[i]); }
  for (let j = 0; j < nz; j++) { const k = kind(art(j)); xEnd[j] = [stub(w, ids[0][j], -1, 0, 250, { ...k, opp: k.lanes }), stub(w, ids[nx - 1][j], 1, 0, 250, { ...k, opp: k.lanes })]; ends.push(...xEnd[j]); }
  // controls: street i runs along z, street j along x
  for (let i = 0; i < nx; i++) for (let j = 0; j < nz; j++) {
    const N = ids[i][j], ai = art(i), aj = art(j);
    if (o.urban || (ai && aj)) w.control('signal', N);
    else if (!ai && !aj) w.control('stop', N, { allWay: true });
    else {   // the local street stops at the arterial: its two approaches (neighbour junction or boundary stub)
      const ap = ai ? [i > 0 ? ids[i - 1][j] : xEnd[j][0], i + 1 < nx ? ids[i + 1][j] : xEnd[j][1]]
                    : [j > 0 ? ids[i][j - 1] : zEnd[i][0], j + 1 < nz ? ids[i][j + 1] : zEnd[i][1]];
      w.control('stop', N, { approaches: ap });
    }
  }
  const n = o.trips, d = [T0, T0 + o.win];
  const inner = ids.flat();
  for (let k = 0; k < n; k++) {
    const a = ends[R.int(ends.length)]; let b = R.random() < 0.7 ? ends[R.int(ends.length)] : inner[R.int(inner.length)];
    if (b === a) b = ends[(ends.indexOf(a) + 1 + R.int(ends.length - 1)) % ends.length];
    w.trip({ from: a, to: b, count: 1, depart: d });
  }
  w.notes.push(`${nx} x ${nz} grid, ${block} m blocks, ${o.urban ? 'all signals' : `arterial every ${o.every}`}, ${n} trips`);
}

/** Scenario names: `${family}-s${seed}` (+ `-${variant values}` when overridden). */
export function makeScenario(family, seed = 1, overrides = {}, { name = null, rateScale = 1 } = {}) {
  const F = FAMILIES[family];
  if (!F) throw new Error(`unknown scenario family '${family}' (have: ${Object.keys(FAMILIES).join(', ')})`);
  const R = makeRng(hashSeed(`scenario:${family}:${seed}`));
  const o = { ...F.params(R), ...overrides };
  if (rateScale !== 1) { for (const k of ['rate', 'rampRate', 'crossRate']) if (o[k] != null) o[k] *= rateScale; if (o.trips) o.trips = Math.round(o.trips * rateScale); }
  const tag = Object.entries(overrides).map(([k, v]) => `${v}`).join('-');
  const w = new World({ name: name || `${family}-s${seed}${tag ? '-' + tag : ''}`, family, seed, params: o });
  F.build(w, R, o);
  return w;
}
