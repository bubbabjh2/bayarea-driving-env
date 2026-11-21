// Crash severity and scene clearance (SPEC "Crashes: real physics only").
// A crash is a physical contact (src/sim/physics/collide.js). Its severity is the larger
// per-vehicle delta-v of the contact impulse:
//
//   tier         delta-v        vehicles                 scene clearance (log-normal, sigma 0.4)
//   minor        < 10 km/h      drivable, keep going     15-30 min (median 20) - rubbernecking only
//   moderate     10-25 km/h     disabled, tow            30-60 min (median 45)
//   severe       25-50 km/h     disabled                 1-3 h     (median 110 min)
//   very severe  > 50 km/h      disabled                 2-5 h     (median 200 min)
//
// A vehicle is disabled when ITS OWN delta-v is >= 10 km/h (a truck nudged by a sedan stays
// drivable). Disabled vehicles stay where physics leaves them and block whatever lanes their
// hitbox covers until the scene clears; lanes are never closed by rule.

export const DISABLE_KMH = 10;
export const CLEARANCE = Object.freeze({
  minor: { median: 20 * 60, sigma: 0.4 },
  moderate: { median: 45 * 60, sigma: 0.4 },
  severe: { median: 110 * 60, sigma: 0.4 },
  'very severe': { median: 200 * 60, sigma: 0.4 },
});
export const CLEAR_RANGE = Object.freeze({ minor: [15, 30], moderate: [30, 60], severe: [60, 180], 'very severe': [120, 300] }); // minutes

export function severityOf(dvKmh) { return dvKmh < 10 ? 'minor' : dvKmh < 25 ? 'moderate' : dvKmh < 50 ? 'severe' : 'very severe'; }

/**
 * Incident timeline (docs/realism.md section 7: TRR 1748-17, MUTCD classes, Caltrans quick clearance):
 * CHP notified 3.4 min after the crash and on scene 6.0 min later; scene clearance by class:
 * PDO / disabled vehicle 10-30 min (median 20) -> tier moderate; injury 45-75 min (median 50) -> severe;
 * major / fatal 2-4 h (median 150 min) -> very severe. Lanes stay blocked until the wrecks are moved to
 * the shoulder: PDO ~5 min after responders arrive, injury ~80 % of the clearance, major ~90 %.
 * Returns {notify, arrive, block, clear} in seconds after the crash.
 */
export const INCIDENT = Object.freeze({
  notify: 3.4 * 60, arriveAfter: 6.0 * 60,
  moderate: { median: 20 * 60, sigma: 0.4, range: [10, 30], blockFrac: null, towAfterArrival: 5 * 60 },
  severe: { median: 50 * 60, sigma: 0.25, range: [45, 75], blockFrac: 0.8 },
  'very severe': { median: 150 * 60, sigma: 0.3, range: [120, 240], blockFrac: 0.9 },
  minor: { median: 20 * 60, sigma: 0.4, range: [15, 30], blockFrac: 0 },
});
export function incidentTimeline(severity, rng) {
  const c = INCIDENT[severity] || INCIDENT.moderate, [lo, hi] = c.range;
  const notify = INCIDENT.notify * Math.exp(0.3 * rng.gaussian()), arrive = notify + INCIDENT.arriveAfter * Math.exp(0.3 * rng.gaussian());
  const clear = Math.min(hi * 60, Math.max(lo * 60, c.median * Math.exp(c.sigma * rng.gaussian())));
  let block = c.blockFrac == null ? arrive + c.towAfterArrival * Math.exp(0.3 * rng.gaussian()) : c.blockFrac * clear;
  block = Math.min(block, clear);
  return { notify, arrive, block, clear: Math.max(clear, block + 60) };
}

/** Clearance time (s) for a severity tier: log-normal around the median, clipped to the tier's range. */
export function clearanceFor(severity, rng) {
  const c = CLEARANCE[severity] || CLEARANCE.moderate, [lo, hi] = CLEAR_RANGE[severity] || [30, 60];
  const t = c.median * Math.exp(c.sigma * rng.gaussian());
  return Math.min(hi * 60, Math.max(lo * 60, t));
}

// Driver-behaviour effect near active scenes (not a crash rule): desired-speed factor.
export const RUBBERNECK = Object.freeze({ dist: 120, factor: 0.8 });

// ---- legacy exports (layout v3: the rule-based crash model, kept so v3 consumers such as
// src/gpu keep loading; NOT used by the sim any more) ----
export const MASS_BY_TYPE = [1500, 2100, 15000, 13000];
export const CRASH_TABLE = Object.freeze({
  minor: { blockMedian: 5 * 60, clearMedian: 20 * 60, sigma: 0.4, lanes: 1 },
  moderate: { blockMedian: 45 * 60, clearMedian: 45 * 60, sigma: 0.4, lanes: 2 },
  severe: { blockMedian: 110 * 60, clearMedian: 110 * 60, sigma: 0.4, lanes: 3 },
  fatal: { blockMedian: 200 * 60, clearMedian: 200 * 60, sigma: 0.35, lanes: 99 },
});
export const SEVERITY_THRESHOLDS = Object.freeze({ minorMph: 15, moderateMph: 40, pFatalSevere: 0.08, pFatalModerate: 0.01 });
export const SECONDARY = Object.freeze({ queueRiskWindow: 300, queueRiskDist: 500, queueRiskFactor: 6, rubberneckDist: RUBBERNECK.dist, rubberneckFactor: RUBBERNECK.factor });
