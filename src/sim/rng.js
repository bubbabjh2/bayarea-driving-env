// Deterministic seeded PRNG (mulberry32) plus a few sampling helpers.
// Plain ES module, works in browser and node.

export function makeRng(seed = 1) {
  let a = (seed >>> 0) || 0x9e3779b9;
  const rng = {
    seed,
    /** uniform [0,1) */
    random() {
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    },
    /** integer in [0, n) */
    int(n) { return Math.floor(rng.random() * n); },
    /** uniform in [lo, hi) */
    range(lo, hi) { return lo + (hi - lo) * rng.random(); },
    pick(arr) { return arr[Math.floor(rng.random() * arr.length)]; },
    /** standard normal via Box-Muller */
    gaussian() {
      let u = 0, v = 0;
      while (u === 0) u = rng.random();
      while (v === 0) v = rng.random();
      return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    },
    /** index sampled proportional to weights (Float64Array or array). cum = prefix sums */
    weightedIndex(cum) {
      const total = cum[cum.length - 1];
      const r = rng.random() * total;
      let lo = 0, hi = cum.length - 1;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (cum[mid] <= r) lo = mid + 1; else hi = mid;
      }
      return lo;
    },
    /** exponential with mean m */
    exp(m) { return -Math.log(1 - rng.random()) * m; },
  };
  return rng;
}

export function prefixSums(weights) {
  const cum = new Float64Array(weights.length);
  let acc = 0;
  for (let i = 0; i < weights.length; i++) { acc += weights[i]; cum[i] = acc; }
  return cum;
}

/** Hash a string to a 32-bit int (for deriving sub-seeds). */
export function hashSeed(str) {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 16777619); }
  return h >>> 0;
}
