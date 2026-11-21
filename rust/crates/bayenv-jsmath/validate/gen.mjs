// Generate bit-exact Math.* reference vectors from the running node.
//
//   node gen.mjs gen   <outdir> <N> [fn,fn,...]   -> writes <outdir>/<fn>.bin
//   node gen.mjs check <outdir> [fn,fn,...]       -> recomputes outputs from the
//        stored inputs with *this* node process (run it with --jitless or
//        --no-turbofan) and counts bit differences vs the stored outputs.
//
// File format: little-endian f64 records [in_0 .. in_{arity-1}, out], no header.
// Integer-valued results (imul, |0, >>>0) are stored as the JS number (f64).
import fs from 'node:fs';
import path from 'node:path';

const PI = Math.PI;
const f64 = new Float64Array(1);
const u32 = new Uint32Array(f64.buffer);
const rnd = Math.random;
const uni = (a, b) => a + (b - a) * rnd();
const sgn = () => (rnd() < 0.5 ? -1 : 1);
const pick = (arr) => arr[(rnd() * arr.length) | 0];
const randBits = () => {
  u32[0] = (rnd() * 4294967296) >>> 0;
  u32[1] = (rnd() * 4294967296) >>> 0;
  return f64[0];
};
// log-uniform magnitude in [10^lo, 10^hi]
const logu = (lo, hi) => Math.pow(10, uni(lo, hi));
const slogu = (lo, hi) => sgn() * logu(lo, hi);
// nudge x by a random number of ulps in [-k, k] (via bit pattern arithmetic)
const nudge = (x, k) => {
  if (!isFinite(x) || x === 0) return x;
  f64[0] = x;
  const d = Math.round(uni(-k, k));
  let lo = u32[0] + d;
  let hi = u32[1];
  if (lo < 0) { lo += 4294967296; hi -= 1; }
  if (lo >= 4294967296) { lo -= 4294967296; hi += 1; }
  u32[0] = lo; u32[1] = hi;
  return f64[0];
};
const nearMult = (step, kmax) => {
  const k = Math.round(uni(-kmax, kmax));
  const x = k * step;
  return rnd() < 0.6 ? nudge(x, 64) : x * (1 + uni(-1e-9, 1e-9));
};
const subnormal = () => sgn() * rnd() * 2.2250738585072014e-308;
const intish = (m) => Math.round(uni(-m, m));
const halfish = (m) => nudge(Math.round(uni(-m, m)) + 0.5, 3);

const SPECIAL = [
  0, -0, Infinity, -Infinity, NaN, Number.MIN_VALUE, -Number.MIN_VALUE,
  Number.MAX_VALUE, -Number.MAX_VALUE, 2.2250738585072014e-308, -2.2250738585072014e-308,
  2.225073858507201e-308, 1, -1, 0.5, -0.5, 2, -2, 0.49999999999999994, -0.49999999999999994,
  1.5, -1.5, 2.5, -2.5, PI, -PI, PI / 2, -PI / 2, PI / 4, -PI / 4, 3 * PI / 4, 2 * PI, -2 * PI,
  1e-8, -1e-8, 1e-300, 1e300, 709.782712893384, 709.7827128933841, -745.1332191019411,
  -745.1332191019412, 710, -746, 1 + 2 ** -52, 1 - 2 ** -53, 2 ** 31, -(2 ** 31), 2 ** 32,
  2 ** 32 + 0.5, 2 ** 53, 2 ** 53 + 2, 2 ** 52 + 1, 2 ** 63, 2 ** 64, 4503599627370495.5,
  -4503599627370495.5, 1e16, 1e22, 1e23, 22, -22, 0.6744, 0.4375, 2.4375, 1.1875, 0.6875,
  0.975, -0.975, 3.725290298461914e-9, 7.450580596923828e-9, 1.4901161193847656e-8,
  5.551115123125783e-17, 2 ** 19 * PI / 2, 2 ** 20 * PI / 2, 2 ** 66, -(2 ** 66), 0.1, 0.3,
  1e7, 1e-7, 1e-310, 0.78125, 0.28125, 0.3465735902799726, 1.0397207708399179, 56 * Math.LN2,
];
for (let k = -1074; k <= 1023; k += 7) SPECIAL.push(2 ** k, -(2 ** k));
for (let k = 1; k <= 64; k++) { SPECIAL.push(k * PI / 2, -k * PI / 2, nudge(k * PI / 2, 1)); }

// Weighted mixture of generators.
const mix = (...parts) => {
  const tot = parts.reduce((s, p) => s + p[0], 0);
  const cum = [];
  let c = 0;
  for (const p of parts) { c += p[0] / tot; cum.push([c, p[1]]); }
  return () => {
    const r = rnd();
    for (const [cc, g] of cum) if (r < cc) return g();
    return cum[cum.length - 1][1]();
  };
};

const TRIG = mix(
  [30, () => uni(-2 * PI, 2 * PI)],
  [25, () => uni(-1e4, 1e4)],
  [10, () => nearMult(PI / 2, 1e5)],
  [3, () => nearMult(PI / 2, 64)],
  [5, () => slogu(-320, -8)],
  [5, () => uni(-1e-8, 1e-8)],
  [5, () => uni(-1e6, 1e6)],
  [7, () => slogu(-8, 308)],
  [10, randBits],
);
const GEN = {
  sin: TRIG, cos: TRIG, tan: TRIG,
  atan: mix([30, () => uni(-10, 10)], [20, () => uni(-3, 3)], [20, () => slogu(-320, 308)],
    [10, () => uni(-1e4, 1e4)], [10, randBits], [10, () => nudge(pick([0.4375, 0.6875, 1.1875, 2.4375, 1]) * sgn(), 1e6)]),
  asin: mix([50, () => uni(-1, 1)], [15, () => sgn() * nudge(1, 1e12)], [10, () => slogu(-320, 0)],
    [10, () => uni(-1.2, 1.2)], [5, randBits], [10, () => sgn() * nudge(pick([0.5, 0.975]), 1e9)]),
  exp: mix([45, () => uni(-50, 50)], [20, () => uni(-750, 750)], [10, () => slogu(-320, -1)],
    [10, () => uni(-2, 2)], [5, randBits], [10, () => nudge(pick([709.78, -745.13, 0.34657, 1.03972]) * 1, 1e9)]),
  log: mix([40, () => uni(0, 1e7)], [20, () => logu(-320, 308)], [15, () => nudge(1, 1e7)],
    [10, () => uni(0, 2)], [10, randBits], [5, () => subnormal()]),
  tanh: mix([50, () => uni(-25, 25)], [15, () => uni(-1, 1)], [15, () => slogu(-320, 3)], [10, randBits],
    [10, () => sgn() * nudge(pick([1, 22, 3.725290298461914e-9]), 1e10)]),
  pow: () => {
    const r = rnd();
    if (r < 0.25) return [uni(-100, 100), uni(-10, 10)];
    if (r < 0.40) return [uni(-100, 100), intish(60)];
    if (r < 0.50) return [logu(-300, 300), uni(-5, 5)];
    if (r < 0.58) return [nudge(1, 1e8) * sgn(), uni(-1e10, 1e10)];
    if (r < 0.66) return [uni(0, 10), pick([0.5, 2, 3, -1, -2, 1 / 3, 1.5, 0.25, 4, -0.5, 10])];
    if (r < 0.72) return [pick([10, 2, Math.E, 0.5, -2, -10]), uni(-400, 400)];
    if (r < 0.80) return [uni(-2, 2), uni(-2000, 2000)];
    if (r < 0.90) return [randBits(), randBits()];
    if (r < 0.95) return [pick(SPECIAL), pick(SPECIAL)];
    return [logu(-320, 308) * sgn(), intish(1100) + pick([0, 0.5])];
  },
  atan2: () => {
    const comp = () => {
      const r = rnd();
      if (r < 0.40) return uni(-1e3, 1e3);
      if (r < 0.55) return uni(-1, 1);
      if (r < 0.70) return slogu(-320, 308);
      if (r < 0.75) return pick([0, -0]);
      if (r < 0.78) return pick([Infinity, -Infinity]);
      if (r < 0.79) return NaN;
      if (r < 0.85) return randBits();
      return uni(-1e5, 1e5);
    };
    const r = rnd();
    if (r < 0.02) return [pick([0, -0]), pick([0, -0])];
    if (r < 0.06) { const x = comp(); return [x * uni(0.999, 1.001), x]; }
    if (r < 0.08) { return [1, comp()]; }
    if (r < 0.10) { return [comp(), 1]; }
    return [comp(), comp()];
  },
  hypot2: () => {
    const comp = () => {
      const r = rnd();
      if (r < 0.40) return uni(-1e3, 1e3);
      if (r < 0.55) return uni(-1, 1);
      if (r < 0.70) return slogu(-320, 308);
      if (r < 0.75) return pick([0, -0]);
      if (r < 0.77) return pick([Infinity, -Infinity]);
      if (r < 0.78) return NaN;
      if (r < 0.85) return randBits();
      return uni(-1e5, 1e5);
    };
    if (rnd() < 0.2) { const a = comp(); return [a, a * slogu(-20, 0)]; }
    return [comp(), comp()];
  },
};
GEN.acos = GEN.asin;
GEN.log2 = GEN.log;
GEN.expm1 = GEN.exp;
GEN.powop = GEN.pow;
GEN.hypot3 = () => { const [a, b] = GEN.hypot2(); const [c] = GEN.hypot2(); return [a, b, c]; };
GEN.round = mix([30, () => uni(-1e3, 1e3)], [20, () => halfish(1e4)], [10, () => uni(-1, 1)],
  [10, () => slogu(-320, 308)], [10, randBits], [10, () => nudge(sgn() * 0.5, 1e3)],
  [10, () => uni(-1e16, 1e16)]);
GEN.sign = mix([50, () => uni(-1e3, 1e3)], [25, randBits], [25, () => slogu(-320, 308)]);
GEN.fround = mix([30, () => uni(-1e3, 1e3)], [30, randBits], [20, () => slogu(-320, 308)],
  [20, () => nudge(Math.fround(uni(-1e3, 1e3)), 1 << 28)]);
GEN.sqrt = mix([40, () => uni(0, 1e6)], [30, () => logu(-320, 308)], [30, randBits]);
GEN.floor = GEN.round; GEN.ceil = GEN.round;
const INTSRC = mix([25, () => uni(-1e10, 1e10)], [20, randBits], [20, () => slogu(-1, 25)],
  [15, () => intish(2 ** 32) + pick([0, 0.5, -0.5, 0.25])], [10, () => uni(-3, 3)],
  [10, () => sgn() * nudge(pick([2 ** 31, 2 ** 32, 2 ** 53, 2 ** 63, 2 ** 64, 2 ** 84, 2 ** 85]), 1e3)]);
GEN.toint32 = INTSRC; GEN.touint32 = INTSRC;
GEN.imul = () => [INTSRC(), INTSRC()];
const MMSRC = mix([50, () => uni(-10, 10)], [10, () => pick([0, -0])], [5, () => NaN],
  [5, () => pick([Infinity, -Infinity])], [15, randBits], [15, () => intish(3)]);
GEN.max2 = () => [MMSRC(), MMSRC()]; GEN.min2 = GEN.max2;
GEN.max3 = () => [MMSRC(), MMSRC(), MMSRC()]; GEN.min3 = GEN.max3;

const EXPR = {
  sin: 'Math.sin(a)', cos: 'Math.cos(a)', tan: 'Math.tan(a)', atan: 'Math.atan(a)',
  asin: 'Math.asin(a)', acos: 'Math.acos(a)', exp: 'Math.exp(a)', expm1: 'Math.expm1(a)',
  log: 'Math.log(a)', log2: 'Math.log2(a)', tanh: 'Math.tanh(a)',
  pow: 'Math.pow(a, b)', powop: 'a ** b', atan2: 'Math.atan2(a, b)',
  hypot2: 'Math.hypot(a, b)', hypot3: 'Math.hypot(a, b, c)', round: 'Math.round(a)',
  sign: 'Math.sign(a)', fround: 'Math.fround(a)', sqrt: 'Math.sqrt(a)', floor: 'Math.floor(a)',
  ceil: 'Math.ceil(a)', toint32: 'a | 0', touint32: 'a >>> 0', imul: 'Math.imul(a, b)',
  max2: 'Math.max(a, b)', min2: 'Math.min(a, b)', max3: 'Math.max(a, b, c)', min3: 'Math.min(a, b, c)',
};
const ARITY = {};
for (const k of Object.keys(EXPR)) ARITY[k] = /\bc\b/.test(EXPR[k]) ? 3 : /\bb\b/.test(EXPR[k]) ? 2 : 1;

function makeKernel(name) {
  const ar = ARITY[name];
  const loads = ['const a = A[i * S];', ar > 1 ? 'const b = A[i * S + 1];' : '', ar > 2 ? 'const c = A[i * S + 2];' : ''].join(' ');
  // Records are interleaved [inputs..., out]; S = arity + 1.
  return new Function('A', 'N', `const S = ${ar + 1}; for (let i = 0; i < N; i++) { ${loads} A[i * S + ${ar}] = ${EXPR[name]}; }`);
}

function gen(outdir, N, names) {
  fs.mkdirSync(outdir, { recursive: true });
  for (const name of names) {
    const ar = ARITY[name];
    const S = ar + 1;
    const g = GEN[name];
    // specials (cartesian for arity 2, sampled for 3) at start and end
    const sp = [];
    if (ar === 1) for (const x of SPECIAL) sp.push([x]);
    else if (ar === 2) for (const x of SPECIAL) for (const y of SPECIAL) sp.push([x, y]);
    else for (let i = 0; i < 100000; i++) sp.push([pick(SPECIAL), pick(SPECIAL), pick(SPECIAL)]);
    const total = N + 2 * sp.length;
    const A = new Float64Array(total * S);
    let r = 0;
    const put = (v) => { for (let j = 0; j < ar; j++) A[r * S + j] = v[j]; r++; };
    for (const v of sp) put(v);
    for (let i = 0; i < N; i++) put(ar === 1 ? [g()] : g());
    for (const v of sp) put(v);
    const t0 = Date.now();
    makeKernel(name)(A, total);
    const dt = Date.now() - t0;
    fs.writeFileSync(path.join(outdir, `${name}.bin`), new Uint8Array(A.buffer));
    console.log(`${name}: ${total} records (arity ${ar}) in ${dt} ms`);
  }
}

function check(outdir, names) {
  let bad = 0;
  for (const name of names) {
    const file = path.join(outdir, `${name}.bin`);
    if (!fs.existsSync(file)) continue;
    const buf = fs.readFileSync(file);
    const A = new Float64Array(buf.buffer, buf.byteOffset, buf.byteLength / 8);
    const ar = ARITY[name];
    const S = ar + 1;
    const total = A.length / S;
    const want = new Float64Array(total);
    for (let i = 0; i < total; i++) want[i] = A[i * S + ar];
    makeKernel(name)(A, total);
    const wb = new BigUint64Array(want.buffer);
    let mism = 0;
    for (let i = 0; i < total; i++) {
      const got = A[i * S + ar];
      if (Number.isNaN(got) && Number.isNaN(want[i])) continue;
      f64[0] = got;
      const gb = new BigUint64Array(f64.buffer)[0];
      if (gb !== wb[i]) { if (mism < 3) console.log(`  ${name}[${i}] stored ${want[i]} now ${got}`); mism++; }
    }
    bad += mism;
    console.log(`${name}: ${total} records, ${mism} differ (flags: ${process.execArgv.join(' ') || 'default'})`);
  }
  process.exitCode = bad ? 1 : 0;
}

const [mode, outdir, a3, a4] = process.argv.slice(2);
const ALL = Object.keys(EXPR);
if (mode === 'gen') gen(outdir, Number(a3 ?? 10_000_000), a4 ? a4.split(',') : ALL);
else if (mode === 'check') check(outdir, a3 ? a3.split(',') : ALL);
else { console.error('usage: node gen.mjs gen <dir> <N> [fns] | check <dir> [fns]'); process.exit(2); }
