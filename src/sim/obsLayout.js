// Input version 3 observation layout (moved from the original project's policy code; env-side constants only).
//
// OBSERVATION VECTOR (OBS3 = 358 floats) = RAW3 (102) | NB (16 bodies x 16 features = 256)
//   0..69    the version-2 inputs (eyes.js layout, grade in 41..43)
//   70..78   mirror / blind spot (src/sim/perceive.js mirrorInputs, GPU perceive.wgsl)
//            70 / 71 left / right alongside (0/1)  72 / 73 rear-left / rear-right closing TTC / 6 (1 = none)
//            74 / 75 rear-left / rear-right nearest gap / 60 (1 = none)  76 own-lane rear TTC / 6 (1 = none)
//            77 / 78 lateral clearance left / right / 3.7 (1 = none)
//   79..86   controls: 79 permissive left  80 cross street signed (they yield to me)  81 all-way stop
//            82 2nd control distance / 250 (1 = none)  83 / 84 / 85 its type signal / stop-or-give-way / other  86 red or yellow
//   87..93   own commands: 87..89 being applied now (ring at step - delaySteps), 90..92 last command (ring at step - 1),
//            93 history visible (1; 0 = dropped out / ablated)
//   94..100  own-lane leader history: (input 8, input 9) 5, 10, 20 decisions ago; 100 leader changed in the last
//            20 decisions (0/1)
//   101      reserved (0)
//   102..357 NB: bodies k = 0..15 (8 ego-frame sectors x 2 nearest, sector order), features d = 0..15 (FEATURES)
// The net (net3.js) pools NB through a shared MLP into 64 floats; its trunk sees RAW3 (102) + pooled (64) = 166.
export const RAW3 = 102;
export const NB_K = 16;
export const NB_D = 16;
export const NB_OFF = RAW3;
export const OBS3 = RAW3 + NB_K * NB_D;     // 358
export const MIRROR_OFF = 70;
export const CTL3_OFF = 79;
export const HIST_OFF = 87;
export const HIST_VISIBLE = 93;
export const LHIST_OFF = 94;
export const SECTORS = 8;
export const PER_SECTOR = NB_K / SECTORS;  // 2
export const NB_RANGE = 100;               // m, centre distance
export const LAYER_GAP = 2.5;              // m, |dy| for the same road level (sim layerGap)
export const LHIST_N = 20;                 // decisions kept in the leader-history ring
export const LHIST_AGO = [5, 10, 20];
export const FEATURES = ['fwd', 'left', 'gap', 'vFwd', 'vLeft', 'cosDh', 'sinDh', 'length', 'width', 'brake', 'indLeft', 'indRight', 'crashed', 'crossesMyPath', 'closestApproach', 'valid'];
export const INDICATOR_DIST = 60;          // m (or 4 s at speed) before the edge end
export const CONFLICT_DIST = 150;          // m before the edge end: my / its movement through the junction ahead

/** ego-frame sector 0..7 of a point (fwd, left): angle atan2(left, fwd) in [-pi, pi) split in 45 deg sectors */
export function sectorOf(fwd, left) {
  const a = Math.atan2(left, fwd);
  let s = Math.floor((a + Math.PI) / (Math.PI / 4));
  return s < 0 ? 0 : s > 7 ? 7 : s;
}

/**
 * Selection: keep the PER_SECTOR nearest (squared centre distance, ties: lower slot) per sector.
 * sel = { d2: Float64Array(16), j: Int32Array(16) } reset by resetSelection. Returns nothing.
 */
export function resetSelection(sel) { sel.d2.fill(Infinity); sel.j.fill(-1); }
export function offerBody(sel, sector, d2, j) {
  const b = sector * PER_SECTOR;
  const better = (k) => d2 < sel.d2[k] || (d2 === sel.d2[k] && j < sel.j[k]);
  if (!better(b + 1)) return;
  if (better(b)) { sel.d2[b + 1] = sel.d2[b]; sel.j[b + 1] = sel.j[b]; sel.d2[b] = d2; sel.j[b] = j; }
  else { sel.d2[b + 1] = d2; sel.j[b + 1] = j; }
}
/** guaranteed member (my own-lane leader): if absent, it takes the second place of its sector (pair re-sorted) */
export function forceBody(sel, sector, d2, j) {
  for (let k = 0; k < NB_K; k++) if (sel.j[k] === j) return;
  const b = sector * PER_SECTOR;
  sel.d2[b + 1] = d2; sel.j[b + 1] = j;
  if (sel.j[b] < 0 || d2 < sel.d2[b] || (d2 === sel.d2[b] && j < sel.j[b])) {
    const t = sel.d2[b], tj = sel.j[b]; sel.d2[b] = sel.d2[b + 1]; sel.j[b] = sel.j[b + 1]; sel.d2[b + 1] = t; sel.j[b + 1] = tj;
  }
}

/**
 * The 16 features of one body in my frame. me / o: {x, z, h (heading), v (speed, 0 when not driving), hl, hw, length,
 * width}; o also {brake (pedal), indL, indR, crashed, conflict}. Writes out[at .. at + 16).
 * Frame: f = (cos h, sin h), l = (sin h, -cos h) (geom.js leftOf, HANDEDNESS 1).
 */
export function bodyFeatures(me, o, out, at) {
  const c = Math.cos(me.h), s = Math.sin(me.h);
  const dx = o.x - me.x, dz = o.z - me.z;
  const fwd = dx * c + dz * s, left = dx * s - dz * c;
  const co = Math.cos(o.h), so = Math.sin(o.h);
  const rvx = o.v * co - me.v * c, rvz = o.v * so - me.v * s;
  const vf = rvx * c + rvz * s, vl = rvx * s - rvz * c;
  const d = Math.sqrt(dx * dx + dz * dz);
  // oriented-box gap along the line between the centres
  let ux = 1, uz = 0; if (d > 1e-6) { ux = dx / d; uz = dz / d; }
  const eMe = me.hl * Math.abs(ux * c + uz * s) + me.hw * Math.abs(ux * s - uz * c);
  const eO = o.hl * Math.abs(ux * co + uz * so) + o.hw * Math.abs(ux * so - uz * co);
  const gap = Math.max(0, d - eMe - eO);
  let dh = o.h - me.h; dh = Math.atan2(Math.sin(dh), Math.cos(dh));
  // closest approach of the relative motion
  const vv = vf * vf + vl * vl; let ca = 1;
  if (vv > 1e-4) {
    const t = -(fwd * vf + left * vl) / vv;
    if (t > 0) { const mx = fwd + vf * t, ml = left + vl * t; if (mx * mx + ml * ml <= 9) ca = Math.min(1, t / 10); }
  }
  out[at] = fwd / NB_RANGE; out[at + 1] = left / NB_RANGE; out[at + 2] = Math.min(gap / NB_RANGE, 1);
  out[at + 3] = Math.max(-1, Math.min(1, vf / 20)); out[at + 4] = Math.max(-1, Math.min(1, vl / 20));
  out[at + 5] = Math.cos(dh); out[at + 6] = Math.sin(dh);
  out[at + 7] = o.length / 20; out[at + 8] = o.width / 3;
  out[at + 9] = o.crashed ? 0 : (o.brake > 0.05 ? 1 : 0);
  out[at + 10] = o.indL ? 1 : 0; out[at + 11] = o.indR ? 1 : 0;
  out[at + 12] = o.crashed ? 1 : 0; out[at + 13] = o.conflict ? 1 : 0;
  out[at + 14] = ca; out[at + 15] = 1;
}

/**
 * Leader history (inputs 94..100), kept per car by its driver (CPU) / the GPU `lhist` buffer. Call once per
 * decision with the final inputs 8 / 9 and the leader id; reads the history BEFORE pushing the current values.
 */
export class LeaderHistory {
  constructor() { this.g = new Float32Array(LHIST_N); this.r = new Float32Array(LHIST_N); this.n = 0; this.head = 0; this.lastLeader = -1e9; this.sinceChange = 1e6; }
  step(input, leaderId) {
    for (let q = 0; q < 3; q++) {
      const ago = LHIST_AGO[q], o = LHIST_OFF + 2 * q;
      if (this.n === 0) { input[o] = input[8]; input[o + 1] = input[9]; continue; }
      const k = Math.min(ago, this.n), idx = (this.head - k + LHIST_N) % LHIST_N;
      input[o] = this.g[idx]; input[o + 1] = this.r[idx];
    }
    if (this.n > 0 && leaderId !== this.lastLeader) this.sinceChange = 0; else this.sinceChange = Math.min(this.sinceChange + 1, 1e6);
    input[LHIST_OFF + 6] = this.sinceChange < LHIST_N ? 1 : 0;
    this.lastLeader = leaderId;
    this.g[this.head] = input[8]; this.r[this.head] = input[9];
    this.head = (this.head + 1) % LHIST_N; if (this.n < LHIST_N) this.n++;
    input[101] = 0;
  }
}
