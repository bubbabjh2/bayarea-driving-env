// Traffic control (SPEC "Traffic control"): real OSM signals, stop / yield signs and ramp
// meters from data/controls.* (schema: data/roads.schema.md), with signal phase plans run by
// the sim. Controls are PERCEIVED, never enforced: each driving car gets an observation of
// the next control on its route (type, distance to the stop line, state, time in state, time
// to change); obeying is driver behaviour. Running a red is possible; physics decides what
// happens. Crossing a stop line on red / rolling a stop sign is logged as an event.
//
// Signal plans (fixed time, per signalised junction):
//   approaches are grouped into two axes by their heading into the junction (+-45 deg);
//   phases: [A left (protected, big arterials only)] A through+right (+permissive left)
//           [B left] B through+right; each phase = green, yellow, all-red
//   yellow = 1.0 s + v / (2 x 3.05 m/s^2) (ITE, v = approach speed limit), clipped [3, 5] s; all-red 1.5 s
//   cycle by size: 60 s (two minor axes), 90 s (an arterial), 120 s (protected lefts);
//   green split proportional to the axis' road class weight (major road gets more)
//   coordination: offset = (position along the major axis direction) / (progression speed) mod
//   cycle, so consecutive signals along a straight arterial open in a green wave (one direction)
// Mid-block pedestrian crossing signals: 90 s cycle, road green 75 s, yellow 3 s, red 12 s.
// Ramp meters: 6 s cycle (green 2 s, red 4 s, one car per green) from 06:00 to 10:00, else dark (green).

import { LANE_WIDTH } from './roadGraph.js';

export const CTRL = Object.freeze({ NONE: -1, SIGNAL: 0, STOP: 1, GIVE_WAY: 2, RAMP_METER: 3, CROSSING: 4 });
export const LIGHT = Object.freeze({ GREEN: 0, YELLOW: 1, RED: 2, SIGN: 3, DARK: 4 });
export const MOVE = Object.freeze({ THROUGH: 0, LEFT: 1, RIGHT: 2 });
const TYPES = ['signal', 'stop', 'give_way', 'ramp_meter', 'crossing_signal'];
const CLASS_W = { motorway: 4, trunk: 4, motorway_link: 2, trunk_link: 2, primary: 3, primary_link: 2, secondary: 2.5, secondary_link: 1.5, tertiary: 1.5, tertiary_link: 1, unclassified: 1, residential: 1, living_street: 1 };

/** Movement through a node from edge e into edge en. */
export function movementOf(e, en) {
  let d = en.headingStart - e.headingEnd; while (d > Math.PI) d -= 2 * Math.PI; while (d < -Math.PI) d += 2 * Math.PI;
  return d < -0.5 ? MOVE.LEFT : d > 0.5 ? MOVE.RIGHT : MOVE.THROUGH;   // heading delta < 0 = left (x east, z south)
}

export class TrafficControls {
  /**
   * @param graph SimGraph (merged numbering, same as data/controls.*)
   * @param pack  {info Uint32(6/control), xyz Float32(3/control), approachEdge Uint32, approachStopAt Float32, meta}
   */
  constructor(graph, pack, opts = {}) {
    this.graph = graph;
    const nC = pack.info.length / 6, nA = pack.approachEdge.length, E = graph.edgeCount;
    this.count = nC; this.approachCount = nA;
    this.type = new Int8Array(nC); this.flags = new Uint8Array(nC);
    this.cycle = new Float32Array(nC); this.offset = new Float32Array(nC);
    this.phases = new Array(nC);           // signals: [{group, left: bool, g, y, r}]
    this.apControl = new Int32Array(nA); this.apStopAt = new Float32Array(nA); this.apEdge = new Int32Array(nA);
    this.apGroup = new Int8Array(nA);      // signal axis 0 / 1
    this.apProtLeft = new Uint8Array(nA);
    this.edgeApproach = new Int32Array(E).fill(-1);
    this.rampStart = opts.rampStart ?? 6 * 3600; this.rampEnd = opts.rampEnd ?? 10 * 3600;
    let skipped = 0;
    const sigAps = new Map();   // signal control -> its approaches (planned per cluster below)
    for (let c = 0; c < nC; c++) {
      const t = pack.info[c * 6], fl = pack.info[c * 6 + 1], first = pack.info[c * 6 + 4], cnt = pack.info[c * 6 + 5];
      this.type[c] = t; this.flags[c] = fl;
      const aps = [];
      for (let a = first; a < first + cnt; a++) {
        const e = pack.approachEdge[a];
        this.apControl[a] = c; this.apEdge[a] = e < E ? e : -1; this.apStopAt[a] = pack.approachStopAt[a];
        if (e >= E) { skipped++; continue; }
        if (this.edgeApproach[e] < 0 || pack.approachStopAt[a] < this.apStopAt[this.edgeApproach[e]]) this.edgeApproach[e] = a; // first stop line along the edge
        aps.push(a);
      }
      if (t === CTRL.SIGNAL) sigAps.set(c, aps);
      else if (t === CTRL.CROSSING) { this.cycle[c] = 90; this.offset[c] = (c * 37) % 90; this.phases[c] = null; }
      else if (t === CTRL.RAMP_METER) { this.cycle[c] = 6; this.offset[c] = (c * 1.7) % 6; }
    }
    // signals closer than clusterDist m (both halves of a divided-road intersection, OSM nodes on
    // each corner) are one intersection run by one controller: plan them together
    const cl = opts.clusterDist ?? 60, cell = new Map(), parent = new Map();
    const find = (c) => { while (parent.get(c) !== c) { parent.set(c, parent.get(parent.get(c))); c = parent.get(c); } return c; };
    for (const c of sigAps.keys()) parent.set(c, c);
    for (const c of sigAps.keys()) {
      const x = pack.xyz[c * 3], z = pack.xyz[c * 3 + 2], cx = Math.floor(x / cl), cz = Math.floor(z / cl);
      for (let dx = -1; dx <= 1; dx++) for (let dz = -1; dz <= 1; dz++) for (const o of cell.get((cx + dx) * 100003 + cz + dz) || [])
        if (Math.hypot(pack.xyz[o * 3] - x, pack.xyz[o * 3 + 2] - z) < cl) parent.set(find(o), find(c));
      const k = cx * 100003 + cz; let l = cell.get(k); if (!l) cell.set(k, (l = [])); l.push(c);
    }
    const clusters = new Map();
    for (const c of sigAps.keys()) { const r = find(c); let l = clusters.get(r); if (!l) clusters.set(r, (l = [])); l.push(c); }
    this.clusterOf = new Int32Array(nC).map((_, c) => c);
    for (const [root, members] of clusters) for (const c of members) this.clusterOf[c] = root;
    for (const members of clusters.values()) {
      const aps = members.flatMap(c => sigAps.get(c));
      this._planSignal(members[0], aps, pack.xyz);
      for (const c of members) { this.cycle[c] = this.cycle[members[0]]; this.offset[c] = this.offset[members[0]]; this.phases[c] = this.phases[members[0]]; }
    }
    this.signalClusters = clusters.size;
    this.skippedApproaches = skipped;
    // every stop line per edge, sorted by position (an edge can carry a mid-block crossing AND its junction)
    const per = new Int32Array(E + 1);
    for (let a = 0; a < nA; a++) if (this.apEdge[a] >= 0) per[this.apEdge[a] + 1]++;
    for (let e = 0; e < E; e++) per[e + 1] += per[e];
    const fill = per.slice(0, E), list = new Int32Array(per[E]);
    for (let a = 0; a < nA; a++) if (this.apEdge[a] >= 0) list[fill[this.apEdge[a]]++] = a;
    for (let e = 0; e < E; e++) { const sub = list.subarray(per[e], per[e + 1]); if (sub.length > 1) sub.sort((x, y) => this.apStopAt[x] - this.apStopAt[y]); }
    this.edgeApStart = per; this.edgeApList = list;
    this.stats = { signals: 0, stops: 0, giveWays: 0, rampMeters: 0, crossings: 0, protectedLefts: 0 };
    for (let c = 0; c < nC; c++) { const t = this.type[c]; if (t === 0) this.stats.signals++; else if (t === 1) this.stats.stops++; else if (t === 2) this.stats.giveWays++; else if (t === 3) this.stats.rampMeters++; else if (t === 4) this.stats.crossings++; }
  }

  _planSignal(c, aps, xyz) {
    const g = this.graph;
    if (!aps.length) { this.cycle[c] = 60; this.phases[c] = [{ group: 0, left: false, g: 54, y: 4, r: 2 }]; return; }
    const ax0 = axisOf(g.edges[this.apEdge[aps[0]]].headingEnd);
    const w = [0, 0], spd = [0, 0], big = [false, false];
    for (const a of aps) {
      const e = g.edges[this.apEdge[a]];
      let d = Math.abs(axisOf(e.headingEnd) - ax0); if (d > Math.PI / 2) d = Math.PI - d;
      const grp = d < Math.PI / 4 ? 0 : 1;
      this.apGroup[a] = grp;
      w[grp] = Math.max(w[grp], CLASS_W[e.highway] || 1);
      spd[grp] = Math.max(spd[grp], e.speedLimit);
      if ((e.highway === 'primary' || e.highway === 'trunk' || e.highway === 'secondary') && e.lanes >= 2 && aps.length >= 3) big[grp] = true;
    }
    const groups = w[1] > 0 ? [0, 1] : [0];
    const yel = (v) => Math.min(5, Math.max(3, 1.0 + v / (2 * 3.05))), allRed = 1.5;
    const phases = [];
    for (const k of groups) {
      if (big[k]) phases.push({ group: k, left: true, g: 0, y: 3, r: allRed });
      phases.push({ group: k, left: false, g: 0, y: yel(spd[k]), r: allRed });
    }
    const anyBig = big[0] || big[1], arterial = Math.max(w[0], w[1]) >= 2.5;
    const C = anyBig ? 120 : arterial ? 90 : 60;
    let lost = 0; for (const p of phases) lost += p.y + p.r;
    const avail = Math.max(10 * phases.length, C - lost);
    let leftShare = 0; for (const p of phases) if (p.left) leftShare += 0.12;
    const throughAvail = avail * (1 - leftShare);
    const wsum = groups.reduce((s, k) => s + w[k], 0);
    for (const p of phases) p.g = p.left ? Math.max(7, avail * 0.12) : Math.max(10, throughAvail * w[p.group] / wsum);
    let tot = 0; for (const p of phases) tot += p.g + p.y + p.r;
    this.cycle[c] = tot;
    // coordination: green wave along the major axis (progression at its speed limit)
    const major = w[1] > w[0] ? 1 : 0;
    const maj = aps.find(a => this.apGroup[a] === major);
    const h = g.edges[this.apEdge[maj]].headingEnd, x = xyz[c * 3], z = xyz[c * 3 + 2];
    const along = x * Math.cos(h) + z * Math.sin(h), v = Math.max(8, spd[major] * 0.9);
    this.offset[c] = ((along / v) % tot + tot) % tot;
    for (const a of aps) if (big[this.apGroup[a]]) this.apProtLeft[a] = 1;
    this.phases[c] = phases;
  }

  /** First stop line on edge e at or ahead of position s (front bumper), -1 if none. */
  approachAhead(e, s) {
    for (let k = this.edgeApStart[e]; k < this.edgeApStart[e + 1]; k++) { const a = this.edgeApList[k]; if (this.apStopAt[a] - s > -1) return a; }
    return -1;
  }

  /**
   * State of approach a for a movement at time t: writes {state (LIGHT), timeIn, timeLeft} into out.
   * Signs return LIGHT.SIGN; dark ramp meters LIGHT.DARK.
   */
  stateOf(a, move, t, out) {
    const c = this.apControl[a], ty = this.type[c];
    out.type = ty; out.timeIn = 0; out.timeLeft = 1e9;
    if (ty === CTRL.STOP || ty === CTRL.GIVE_WAY) { out.state = LIGHT.SIGN; return out; }
    const C = this.cycle[c];
    let tc = ((t + this.offset[c]) % C + C) % C;
    if (ty === CTRL.RAMP_METER) {
      const tod = t % 86400;
      if (tod < this.rampStart || tod > this.rampEnd) { out.state = LIGHT.DARK; return out; }
      if (tc < 2) { out.state = LIGHT.GREEN; out.timeIn = tc; out.timeLeft = 2 - tc; } else { out.state = LIGHT.RED; out.timeIn = tc - 2; out.timeLeft = C - tc; }
      return out;
    }
    if (ty === CTRL.CROSSING) {
      if (tc < 75) { out.state = LIGHT.GREEN; out.timeIn = tc; out.timeLeft = 75 - tc; }
      else if (tc < 78) { out.state = LIGHT.YELLOW; out.timeIn = tc - 75; out.timeLeft = 78 - tc; }
      else { out.state = LIGHT.RED; out.timeIn = tc - 78; out.timeLeft = C - tc; }
      return out;
    }
    // signal: walk the phase ring; my movement is green in its group's through phase (left: in the
    // protected-left phase if any, else permissive in the through phase)
    const ph = this.phases[c], grp = this.apGroup[a], prot = this.apProtLeft[a] === 1;
    const mine = (p) => p.group === grp && (move === MOVE.LEFT && prot ? p.left : !p.left);
    let t0 = 0, cur = -1, inPhase = 0;
    for (let k = 0; k < ph.length; k++) { const len = ph[k].g + ph[k].y + ph[k].r; if (tc < t0 + len) { cur = k; inPhase = tc - t0; break; } t0 += len; }
    if (cur < 0) { cur = ph.length - 1; inPhase = 0; }
    const p = ph[cur];
    if (mine(p) && inPhase < p.g) { out.state = LIGHT.GREEN; out.timeIn = inPhase; out.timeLeft = p.g - inPhase; return out; }
    if (mine(p) && inPhase < p.g + p.y) { out.state = LIGHT.YELLOW; out.timeIn = inPhase - p.g; out.timeLeft = p.g + p.y - inPhase; return out; }
    // red: time since my last yellow ended / until my next green
    out.state = LIGHT.RED;
    let until = (p.g + p.y + p.r) - inPhase, k = (cur + 1) % ph.length, guard = 0;
    while (!mine(ph[k]) && guard++ < ph.length) { until += ph[k].g + ph[k].y + ph[k].r; k = (k + 1) % ph.length; }
    out.timeLeft = until;
    let since = mine(p) ? inPhase - p.g - p.y : inPhase; k = (cur - 1 + ph.length) % ph.length; guard = 0;
    while (!mine(ph[k]) && guard++ < ph.length) { since += ph[k].g + ph[k].y + ph[k].r; k = (k - 1 + ph.length) % ph.length; }
    if (mine(ph[k]) && k !== cur) since += ph[k].r;
    out.timeIn = Math.max(0, since);
    return out;
  }
}

function axisOf(h) { let a = h % Math.PI; if (a < 0) a += Math.PI; return a; }

/** Load data/controls.* (node fs or fetch) into the arrays TrafficControls takes. */
export async function loadControlsPack(base, readJSON, readBin) {
  const meta = await readJSON(base, 'controls.meta.json');
  const buf = await readBin(base, meta.file);
  const s = meta.sections, view = (n, T) => new T(buf, s[n].byteOffset, s[n].length);
  return { meta, info: view('controlInfo', Uint32Array), xyz: view('controlXYZ', Float32Array), approachEdge: view('approachEdge', Uint32Array), approachStopAt: view('approachStopAt', Float32Array) };
}

export { TYPES as CONTROL_TYPE_NAMES, LANE_WIDTH };
