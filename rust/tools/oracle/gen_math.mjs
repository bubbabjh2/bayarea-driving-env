// JS oracle for bayenv_world::jsmath: evaluates V8's Math.* on generated inputs and dumps (a, b, f(a, b)).
//   node rust/tools/oracle/gen_math.mjs [outFile=rust/oracle/math.bin] [--n=400000] [--seed=12345]
// Format (little endian): "JSMATH01", u32 function count, then per function:
//   u32 name length, name bytes (ascii), u32 n, n * 3 f64 [a, b, result]  (b = 0 for unary functions)
// Inputs mix: sim-typical ranges, wide log-uniform magnitudes of both signs, raw random bit patterns (huge, tiny,
// subnormal), special values, domain edges, and multiples of pi/2 (and pi/4) near argument-reduction boundaries.
import { writeFileSync, mkdirSync } from 'node:fs';
import path from 'node:path';

const args = process.argv.slice(2);
const opt = (k, d) => { const a = args.find((s) => s.startsWith(`--${k}=`)); return a ? Number(a.split('=')[1]) : d; };
const outFile = args.find((s) => !s.startsWith('--')) ?? 'rust/oracle/math.bin';
const N = opt('n', 400000);
let seed = opt('seed', 12345) >>> 0;

// xorshift128+ on BigInt-free 32-bit halves: deterministic, independent of Math.random
let s0 = seed ^ 0x9e3779b9, s1 = (seed * 2654435761) >>> 0, s2 = 0x6a09e667, s3 = 0xbb67ae85;
function u32() { // xoshiro128**
  const r = Math.imul(rotl(Math.imul(s1, 5), 7), 9) >>> 0;
  const t = s1 << 9;
  s2 ^= s0; s3 ^= s1; s1 ^= s2; s0 ^= s3; s2 ^= t; s3 = rotl(s3, 11);
  return r;
}
function rotl(x, k) { return ((x << k) | (x >>> (32 - k))) >>> 0; }
for (let i = 0; i < 16; i++) u32();
const u = () => (u32() + (u32() >>> 11) / 2097152) / 4294967296; // [0,1), 53-ish bits
const pick = (a) => a[u32() % a.length];
const sign = () => (u32() & 1 ? -1 : 1);
const buf = new ArrayBuffer(8), f64 = new Float64Array(buf), w32 = new Uint32Array(buf);
const bits = () => { w32[0] = u32(); w32[1] = u32(); return f64[0]; }; // any double incl. NaN/inf/subnormal
const finiteBits = () => { let x; do x = bits(); while (!Number.isFinite(x)); return x; };
const logu = (lo, hi) => Math.pow(10, lo + (hi - lo) * u()); // log-uniform magnitude 10^lo..10^hi
const wide = () => sign() * logu(-320, 308);
const nudge = (x) => { // x moved by a few ulps
  f64[0] = x; const k = (u32() % 9) - 4; if (x === 0 || !Number.isFinite(x)) return x;
  const lo = w32[0] + k; w32[0] = lo >>> 0; if (lo < 0) w32[1]--; else if (lo > 0xffffffff) w32[1]++;
  return f64[0];
};
const SPECIAL = [0, -0, Infinity, -Infinity, NaN, 1, -1, 2, -2, 0.5, -0.5, 3, -3, 10, -10, 1e-300, -1e-300, 5e-324,
  -5e-324, 2.2250738585072014e-308, -2.2250738585072014e-308, Number.MAX_VALUE, -Number.MAX_VALUE, Math.PI, -Math.PI,
  Math.PI / 2, -Math.PI / 2, Math.PI / 4, Math.E, Math.LN2, 709.782712893384, 709.7827128933841, -745.1332191019411,
  -745.1332191019412, 0.9999999999999999, 1.0000000000000002, 2 ** 52, 2 ** 53, 2 ** 31, 2 ** -28, 2 ** -27, 22, -22,
  0.6931471805599453, 0.34657359027997264, 1.0397207708399179, 2 ** 1023, 2 ** -1022, 2 ** -1074];
const special = () => pick(SPECIAL);
const PIO2 = Math.PI / 2;
const nearPiMultiple = () => { // k * pi/4 (and k * pi/2) for small and huge k, nudged by a few ulps
  const big = u32() % 4;
  const k = big === 0 ? (u32() % 64) - 32 : big === 1 ? Math.floor(u() * 2e6) - 1e6 : big === 2 ? sign() * Math.floor(logu(6, 22)) : sign() * logu(22, 308);
  return nudge(k * (u32() & 1 ? PIO2 : PIO2 / 2));
};
const angle = () => (u() - 0.5) * 4 * Math.PI;
const smallD = () => sign() * pick([() => logu(-6, -1), () => logu(-1, 2), () => u() * 2, () => logu(-15, -6)])();

// generic unary mix; `sim` supplies the function-specific realistic range
function unary(sim) {
  const c = u32() % 100;
  if (c < 40) return sim();
  if (c < 55) return wide();
  if (c < 65) return bits();
  if (c < 72) return special();
  if (c < 85) return (u() - 0.5) * pick([2, 20, 200, 2000, 2e5]);
  return nudge(sim());
}
const oneMinus = () => 1 - logu(-17, 0); // (0,1) clustered near 1
const fns = {
  atan2: [(a, b) => Math.atan2(a, b), () => {
    const c = u32() % 10;
    if (c < 5) return [smallD(), smallD()];
    if (c < 6) return [(u() - 0.5) * 200, (u() - 0.5) * 200];
    if (c < 7) return [wide(), wide()];
    if (c < 8) return [bits(), bits()];
    if (c < 9) return [special(), pick([special, smallD, wide])()];
    const a = smallD(); return [a, nudge(a * pick([1, -1, 1e-16, 1e16, 1e-300, 1e300]))];
  }],
  hypot: [(a, b) => Math.hypot(a, b), () => (u32() % 4 === 0 ? [special(), pick([special, smallD, wide])()] : u32() % 3 === 0 ? [wide(), wide()] : [smallD() * 100, smallD() * 100])],
  atan: [Math.atan, () => unary(() => smallD() * pick([1, 1, 10, 1e3, 1e8]))],
  sin: [Math.sin, () => unary(() => pick([angle, nearPiMultiple, nearPiMultiple, () => (u() - 0.5) * 1e5])())],
  cos: [Math.cos, () => unary(() => pick([angle, nearPiMultiple, nearPiMultiple, () => (u() - 0.5) * 1e5])())],
  tan: [Math.tan, () => unary(() => pick([angle, nearPiMultiple, nearPiMultiple, () => (u() - 0.5) * 1e5])())],
  exp: [Math.exp, () => unary(() => pick([() => -logu(-12, 2.9), () => (u() - 0.5) * 1500, () => (u() - 0.5) * 2, () => -u() * 50])())],
  log: [Math.log, () => unary(() => pick([() => u(), oneMinus, () => logu(-320, 0), () => logu(0, 308), () => 1 + (u() - 0.5) * 1e-3])())],
  acos: [Math.acos, () => unary(() => pick([() => (u() - 0.5) * 2, () => sign() * oneMinus(), () => sign() * logu(-20, 0)])())],
  asin: [Math.asin, () => unary(() => pick([() => (u() - 0.5) * 2, () => sign() * oneMinus(), () => sign() * logu(-20, 0)])())],
  pow: [(a, b) => Math.pow(a, b), () => {
    const c = u32() % 12;
    if (c < 3) return [u() * 100, (u() - 0.5) * 8];
    if (c < 4) return [sign() * Math.floor(u() * 1000), Math.floor((u() - 0.5) * 40)];
    if (c < 5) return [u() * 2, pick([0.5, 2, 3, -1, -2, 1 / 3, 1.5, -0.5, 4, 10])];
    if (c < 6) return [oneMinus() * pick([1, 1, -1]), sign() * logu(0, 20)];
    if (c < 7) return [logu(-300, 300), (u() - 0.5) * 4];
    if (c < 8) return [wide(), (u() - 0.5) * 20];
    if (c < 9) return [bits(), bits()];
    if (c < 10) return [special(), special()];
    if (c < 11) return [pick([special, wide, () => u() * 10])(), pick([special, () => Math.floor((u() - 0.5) * 100)])()];
    return [10, Math.floor((u() - 0.5) * 660)];
  }],
  cbrt: [Math.cbrt, () => unary(() => sign() * pick([() => u() * 1000, () => logu(-320, 308)])())],
  log2: [Math.log2, () => unary(() => pick([() => u(), oneMinus, () => logu(-320, 308), () => 2 ** Math.floor((u() - 0.5) * 2140), () => Math.floor(u() * 1e6)])())],
  log10: [Math.log10, () => unary(() => pick([() => u(), oneMinus, () => logu(-320, 308), () => 10 ** Math.floor((u() - 0.5) * 640), () => Math.floor(u() * 1e6)])())],
  expm1: [Math.expm1, () => unary(() => pick([() => -logu(-12, 2.9), () => (u() - 0.5) * 1500, () => sign() * logu(-20, 0), () => -u() * 50])())],
  log1p: [Math.log1p, () => unary(() => pick([() => u() * 10 - 0.9, () => -oneMinus(), () => sign() * logu(-20, 0), () => logu(0, 308)])())],
  sinh: [Math.sinh, () => unary(() => pick([() => (u() - 0.5) * 40, () => (u() - 0.5) * 1500, () => sign() * logu(-20, 0)])())],
  cosh: [Math.cosh, () => unary(() => pick([() => (u() - 0.5) * 40, () => (u() - 0.5) * 1500, () => sign() * logu(-20, 0)])())],
  tanh: [Math.tanh, () => unary(() => pick([() => (u() - 0.5) * 50, () => sign() * logu(-20, 1.5)])())],
  atanh: [Math.atanh, () => unary(() => pick([() => (u() - 0.5) * 2, () => sign() * oneMinus(), () => sign() * logu(-20, 0)])())],
};

const names = Object.keys(fns);
const enc = new TextEncoder();
let total = 8 + 4; for (const n of names) total += 4 + enc.encode(n).length + 4 + N * 24;
const out = Buffer.alloc(total); let o = 0;
out.write('JSMATH01', 0, 'ascii'); o = 8; out.writeUInt32LE(names.length, o); o += 4;
for (const name of names) {
  const [f, gen] = fns[name]; const nb = enc.encode(name);
  out.writeUInt32LE(nb.length, o); o += 4; out.set(nb, o); o += nb.length; out.writeUInt32LE(N, o); o += 4;
  for (let i = 0; i < N; i++) {
    let v = gen(); const [a, b] = Array.isArray(v) ? v : [v, 0];
    out.writeDoubleLE(a, o); out.writeDoubleLE(b, o + 8); out.writeDoubleLE(f(a, b), o + 16); o += 24;
  }
}
mkdirSync(path.dirname(outFile), { recursive: true });
writeFileSync(outFile, out);
console.log(`${outFile}: ${names.length} functions x ${N} (${(total / 1e6).toFixed(0)} MB), node ${process.version}, v8 ${process.versions.v8}, ${process.arch}`);
