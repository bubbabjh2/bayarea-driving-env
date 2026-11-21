// Per-commuter driver genome. Mirrors the shape the brain agent documents in
// src/brain/evolve.js: { weights, traits: { aggression, patience, reactionTime,
// preferredHeadway, maxAccel, maxDecel } }. The sim only *reads* traits (vehicle
// limits, reaction delay, headway); weights are opaque and handed to the driver.
// If src/brain/evolve.js exists, pass its functions in as `genomeOps` to the
// population generator and these fall back to being unused.

export const DEFAULT_TRAITS = Object.freeze({
  aggression: 0.5,        // 0..1, scales desired speed over the limit and accepted gaps
  patience: 0.5,          // 0..1, lane-change eagerness is (1 - patience)
  reactionTime: 1.0,      // seconds of control delay (population samples 0.6-1.8)
  preferredHeadway: 1.5,  // seconds
  maxAccel: 2.5,          // m/s^2 (comfort)
  maxDecel: 5.0,          // m/s^2 braking limit (negative throttle * maxDecel)
});

export const TRAIT_BOUNDS = Object.freeze({
  aggression: [0, 1], patience: [0, 1], reactionTime: [0.3, 1.8],
  preferredHeadway: [0.6, 3.0], maxAccel: [1.0, 4.5], maxDecel: [3.0, 9.0],
});

export function defaultGenome(weightCount = 0) {
  return { weights: new Float32Array(weightCount), traits: { ...DEFAULT_TRAITS }, fitness: 0, generation: 0 };
}

export function mutateGenome(genome, rng, { traitSigma = 0.25, weightSigma = 0.1 } = {}) {
  const traits = {};
  for (const k of Object.keys(DEFAULT_TRAITS)) {
    const [lo, hi] = TRAIT_BOUNDS[k];
    const base = genome.traits?.[k] ?? DEFAULT_TRAITS[k];
    const v = base + rng.gaussian() * traitSigma * (hi - lo) * 0.5;
    traits[k] = Math.min(hi, Math.max(lo, v));
  }
  const src = genome.weights || [];
  const weights = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) weights[i] = src[i] + rng.gaussian() * weightSigma;
  return { weights, traits, fitness: 0, generation: (genome.generation || 0) + 1 };
}

export function cloneGenome(g) {
  return { weights: Float32Array.from(g.weights || []), traits: { ...DEFAULT_TRAITS, ...(g.traits || {}) },
    fitness: g.fitness || 0, generation: g.generation || 0 };
}

/** JSON-friendly form (Float32Array -> plain array). */
export function genomeToJSON(g) {
  return { weights: Array.from(g.weights || []), traits: { ...g.traits }, fitness: g.fitness || 0, generation: g.generation || 0 };
}
export function genomeFromJSON(j) { return cloneGenome(j); }
