// Per-commuter driver profile vector (64 floats, uploaded to the GPU as one Float32Array(n * 64)).
//   0..5   traits read by the ENV: aggression, patience, reactionTime (control delay of the reaction ring, s),
//          preferredHeadway, maxAccel, maxDecel (driver limits); population.js samples them per commuter.
//   6..46  per-agent "personalisation" floats: random noise drawn by population.js (kept so populations are
//          bit-identical to the original project's); the env never reads them -- free for a policy to use or ignore.
//   47..63 reserved (0).
// Moved here (verbatim math) from the original project's policy code so the env has no model dependency.
export const DELTA_SIZE = 64;
export const D_TRAITS = 0, D_LEARN_START = 6, D_LEARN_END = 47;
export const TRAIT_NAMES = ['aggression', 'patience', 'reactionTime', 'preferredHeadway', 'maxAccel', 'maxDecel'];
export function traitsOf(delta, into = {}) {
  for (let i = 0; i < TRAIT_NAMES.length; i++) into[TRAIT_NAMES[i]] = delta[D_TRAITS + i];
  return into;
}
export function setTraits(delta, traits) {
  for (let i = 0; i < TRAIT_NAMES.length; i++) if (traits[TRAIT_NAMES[i]] != null) delta[D_TRAITS + i] = traits[TRAIT_NAMES[i]];
  return delta;
}

export const DEFAULT_TRAITS = {
  aggression: 0.5,        // 0..1: desired speed vs limit, gap acceptance, accel use
  patience: 0.5,          // 0..1: how long a slow leader is tolerated before overtaking
  reactionTime: 0.9,      // seconds of control latency (median human; see sampleReactionTime)
  preferredHeadway: 1.4,  // seconds (IDM T)
  maxAccel: 2.5,          // m/s^2 (vehicle + driver limit)
  maxDecel: 5.0,          // m/s^2 braking limit
};
export const TRAIT_RANGES = {
  aggression: [0, 1], patience: [0, 1], reactionTime: [0.6, 1.8],
  preferredHeadway: [0.6, 3.0], maxAccel: [1.0, 4.5], maxDecel: [2.0, 9.0],
};

/** Deterministic PRNG (mulberry32) with a gaussian helper. */
export function makeRng(seed = 1) {
  let a = seed >>> 0;
  const rnd = () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  rnd.gauss = () => { const u = rnd() || 1e-12, v = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
  return rnd;
}

/**
 * Human-like reaction time: log-normal with median 0.9 s and 95th percentile
 * ~1.4 s (sigma = ln(1.4/0.9)/1.645), hard-clipped to [0.6, 1.8] s so 1.8 s is
 * a rare distracted-driver tail, not a common value.
 */
export const REACTION_LOGNORMAL = { mu: Math.log(0.9), sigma: Math.log(1.4 / 0.9) / 1.645 };
export function sampleReactionTime(rng) {
  const v = Math.exp(REACTION_LOGNORMAL.mu + REACTION_LOGNORMAL.sigma * rng.gauss());
  return v < 0.6 ? 0.6 : v > 1.8 ? 1.8 : v;
}

export function clampTraits(traits) {
  const out = {};
  for (const k of TRAIT_NAMES) {
    const [lo, hi] = TRAIT_RANGES[k];
    const v = traits[k] ?? DEFAULT_TRAITS[k];
    out[k] = v < lo ? lo : v > hi ? hi : v;
  }
  return out;
}

export function clampDelta(delta) {
  for (let i = 0; i < TRAIT_NAMES.length; i++) {
    const [lo, hi] = TRAIT_RANGES[TRAIT_NAMES[i]];
    const v = delta[D_TRAITS + i];
    delta[D_TRAITS + i] = v < lo ? lo : v > hi ? hi : v;
  }
  for (let i = D_LEARN_START; i < D_LEARN_END; i++) { const v = delta[i]; delta[i] = v < -2 ? -2 : v > 2 ? 2 : v; }
  for (let i = D_LEARN_END; i < DELTA_SIZE; i++) delta[i] = 0;
  // size cap (coordinator 2026-10-20: the base carries the competence, deltas are personalisation):
  // the learnable part's L2 norm is projected onto a ball of radius DELTA_NORM_CAP
  let q = 0; for (let i = D_LEARN_START; i < D_LEARN_END; i++) q += delta[i] * delta[i];
  if (q > DELTA_NORM_CAP * DELTA_NORM_CAP) { const k = DELTA_NORM_CAP / Math.sqrt(q); for (let i = D_LEARN_START; i < D_LEARN_END; i++) delta[i] *= k; }
  return delta;
}
/** L2 cap on the learnable delta part [6, 47) (personalisation stays small next to the base). */
export let DELTA_NORM_CAP = 0.5;
export function setDeltaNormCap(c) { DELTA_NORM_CAP = c; }

export class Genome {
  constructor(delta) { this.delta = delta; this.fitness = 0; this.generation = 0; }
  /** Alias for sim code that treats the per-agent vector as opaque "weights". */
  get weights() { return this.delta; }
  get traits() { return traitsOf(this.delta); }
  set traits(t) { setTraits(this.delta, clampTraits({ ...this.traits, ...t })); }
  clone() { return new Genome(new Float32Array(this.delta)); }
  toJSON() { return { delta: Array.from(this.delta) }; }
  static fromJSON(j) { return new Genome(new Float32Array(j.delta)); }
}

/** A genome from traits (adapter part zero) or from an existing delta. */
export function makeGenome(traitsOrDelta = {}) {
  if (traitsOrDelta instanceof Float32Array) return new Genome(clampDelta(new Float32Array(traitsOrDelta)));
  if (traitsOrDelta instanceof Genome) return traitsOrDelta.clone();
  const d = new Float32Array(DELTA_SIZE);
  setTraits(d, clampTraits({ ...DEFAULT_TRAITS, ...traitsOrDelta }));
  return new Genome(d);
}

/** Gaussian mutation: traits by traitSigma * range, adapter floats by weightSigma (absolute). */
export function mutate(genome, sigma = 0.05, rng = makeRng(), { weightSigma = sigma, traitSigma = sigma } = {}) {
  const d = new Float32Array(genome.delta);
  for (let i = 0; i < TRAIT_NAMES.length; i++) {
    const k = TRAIT_NAMES[i], [lo, hi] = TRAIT_RANGES[k];
    if (k === 'reactionTime') d[D_TRAITS + i] *= Math.exp(rng.gauss() * traitSigma * 2 * REACTION_LOGNORMAL.sigma); // log-space, stays human-like
    else d[D_TRAITS + i] += rng.gauss() * traitSigma * (hi - lo);
  }
  for (let i = D_LEARN_START; i < D_LEARN_END; i++) d[i] += rng.gauss() * weightSigma;
  return new Genome(clampDelta(d)); // clampDelta keeps reactionTime in [0.6, 1.8]
}

