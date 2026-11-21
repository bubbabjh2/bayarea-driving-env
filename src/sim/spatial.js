// Spatial structures rebuilt every step from the vehicle store:
//  * SpatialHash: uniform grid, cells hashed into a fixed power-of-two table,
//    counting-sorted into CSR arrays. carsNear and raycast are O(cells touched).
//  * LaneIndex: vehicles ordered by s within each (edge, lane) so "car ahead in my
//    lane" is a single array lookup.
// All storage is typed arrays; nothing allocates per step after warm-up.

import { STATE_FREE, STATE_DRIVING, STATE_CRASHED, MAX_LANES, MAX_OCC } from './vehicle.js';

export class SpatialHash {
  constructor({ cellSize = 25, tableBits = 18, capacity = 4096 } = {}) {
    this.cellSize = cellSize;
    this.tableSize = 1 << tableBits;
    this.mask = this.tableSize - 1;
    // wrapped 2D grid: nearby cells never share a bucket (collisions only across the wrap)
    this.wBits = Math.ceil(tableBits / 2); this.wMask = (1 << this.wBits) - 1; this.hMask = (1 << (tableBits - this.wBits)) - 1;
    this.start = new Int32Array(this.tableSize + 1);
    this.fill = new Int32Array(this.tableSize);
    this.items = new Int32Array(capacity);
    this.cellOfItem = new Int32Array(capacity);
    this.n = 0;
    this.vs = null;
  }
  hash(cx, cz) { return (cx & this.wMask) | ((cz & this.hMask) << this.wBits); }
  cellX(x) { return Math.floor(x / this.cellSize); }

  rebuild(vs) {
    this.vs = vs;
    const high = vs.high;
    if (this.items.length < high) { this.items = new Int32Array(high * 2); this.cellOfItem = new Int32Array(high * 2); }
    const { start, fill, cellOfItem, items } = this;
    start.fill(0);
    const inv = 1 / this.cellSize;
    for (let i = 0; i < high; i++) {
      if (vs.state[i] !== STATE_DRIVING && vs.state[i] !== STATE_CRASHED) { cellOfItem[i] = -1; continue; }
      const c = this.hash(Math.floor(vs.x[i] * inv), Math.floor(vs.z[i] * inv));
      cellOfItem[i] = c; start[c + 1]++;
    }
    for (let c = 0; c < this.tableSize; c++) start[c + 1] += start[c];
    fill.set(start.subarray(0, this.tableSize));
    for (let i = 0; i < high; i++) { const c = cellOfItem[i]; if (c >= 0) items[fill[c]++] = i; }
    this.n = start[this.tableSize];
  }

  /** Call fn(slot) for every vehicle within radius r of (x,z). Returns count. */
  forEachNear(x, z, r, fn) {
    const vs = this.vs; if (!vs) return 0;
    const inv = 1 / this.cellSize;
    const cx0 = Math.floor((x - r) * inv), cx1 = Math.floor((x + r) * inv);
    const cz0 = Math.floor((z - r) * inv), cz1 = Math.floor((z + r) * inv);
    const r2 = r * r; let count = 0;
    for (let cz = cz0; cz <= cz1; cz++) for (let cx = cx0; cx <= cx1; cx++) {
      const c = this.hash(cx, cz);
      for (let k = this.start[c]; k < this.start[c + 1]; k++) {
        const i = this.items[k];
        const dx = vs.x[i] - x, dz = vs.z[i] - z;
        if (dx * dx + dz * dz <= r2) { count++; if (fn(i, dx * dx + dz * dz) === false) return count; }
      }
    }
    return count;
  }

  /** Array of slots within r of (x,z) (allocates; prefer forEachNear in hot code). */
  near(x, z, r) { const out = []; this.forEachNear(x, z, r, (i) => { out.push(i); }); return out; }

  /**
   * Cast a ray from (ox,oz) along unit (dx,dz) up to maxDist against vehicle
   * oriented rectangles. Returns {hit, dist, slot}. `skip` = slot to ignore (self).
   */
  raycast(ox, oz, dx, dz, maxDist, skip = -1, out = { hit: false, dist: 0, slot: -1 }) {
    const vs = this.vs; out.hit = false; out.dist = maxDist; out.slot = -1;
    if (!vs) return out;
    const cs = this.cellSize;
    // DDA over cells
    let cx = Math.floor(ox / cs), cz = Math.floor(oz / cs);
    const stepX = dx > 0 ? 1 : -1, stepZ = dz > 0 ? 1 : -1;
    const tDeltaX = dx !== 0 ? Math.abs(cs / dx) : Infinity, tDeltaZ = dz !== 0 ? Math.abs(cs / dz) : Infinity;
    let tMaxX = dx !== 0 ? ((dx > 0 ? (cx + 1) * cs - ox : ox - cx * cs) / Math.abs(dx)) : Infinity;
    let tMaxZ = dz !== 0 ? ((dz > 0 ? (cz + 1) * cs - oz : oz - cz * cs) / Math.abs(dz)) : Infinity;
    let t = 0;
    const visited = this._visited || (this._visited = new Int32Array(64)); let nv = 0;
    const testCell = (ccx, ccz) => {
      // check 3x3 around to catch long vehicles straddling cell borders
      for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) {
        const c = this.hash(ccx + a, ccz + b);
        let seen = false; for (let q = 0; q < nv; q++) if (visited[q] === c) { seen = true; break; }
        if (seen) continue; if (nv < 64) visited[nv++] = c;
        for (let k = this.start[c]; k < this.start[c + 1]; k++) {
          const i = this.items[k]; if (i === skip) continue;
          const d = rayRect(ox, oz, dx, dz, vs.x[i], vs.z[i], vs.heading[i], vs.length[i], vs.width[i]);
          if (d >= 0 && d < out.dist) { out.dist = d; out.slot = i; out.hit = true; }
        }
      }
    };
    while (t <= maxDist) {
      testCell(cx, cz);
      if (out.hit && out.dist < t) break;
      if (tMaxX < tMaxZ) { t = tMaxX; tMaxX += tDeltaX; cx += stepX; } else { t = tMaxZ; tMaxZ += tDeltaZ; cz += stepZ; }
    }
    return out;
  }
}

/** Ray vs oriented rectangle (2D slab test in the rectangle's frame). Returns distance or -1. */
export function rayRect(ox, oz, dx, dz, cx, cz, heading, length, width) {
  const c = Math.cos(heading), s = Math.sin(heading);
  // to local: forward axis (c,s), right axis (-s,c)
  const rx = ox - cx, rz = oz - cz;
  const lox = rx * c + rz * s, loz = -rx * s + rz * c;
  const ldx = dx * c + dz * s, ldz = -dx * s + dz * c;
  const hx = length / 2, hz = width / 2;
  let tmin = 0, tmax = Infinity;
  if (Math.abs(ldx) < 1e-9) { if (lox < -hx || lox > hx) return -1; }
  else { let t1 = (-hx - lox) / ldx, t2 = (hx - lox) / ldx; if (t1 > t2) { const q = t1; t1 = t2; t2 = q; } tmin = Math.max(tmin, t1); tmax = Math.min(tmax, t2); if (tmin > tmax) return -1; }
  if (Math.abs(ldz) < 1e-9) { if (loz < -hz || loz > hz) return -1; }
  else { let t1 = (-hz - loz) / ldz, t2 = (hz - loz) / ldz; if (t1 > t2) { const q = t1; t1 = t2; t2 = q; } tmin = Math.max(tmin, t1); tmax = Math.min(tmax, t2); if (tmin > tmax) return -1; }
  return tmin;
}

/**
 * Per-(edge, lane) vehicle lists sorted by s, built from the body FOOTPRINT: a car
 * is listed in every lane laneLo..laneHi it overlaps (a straddling car is in both).
 * key = edge * MAX_LANES + lane. Entries: order[] holds slots grouped by key and
 * sorted by s (ties by slot); entPos[slot * MAX_OCC + (lane - laneLo)] is the index of
 * that slot's entry for `lane`. Vehicles in lists: DRIVING and CRASHED (obstacles).
 */
export class LaneIndex {
  constructor(edgeCount, capacity = 4096) {
    this.keys = edgeCount * MAX_LANES;
    this.lstart = new Int32Array(this.keys);
    this.lend = new Int32Array(this.keys);
    this.count = new Int32Array(this.keys);
    this.touched = new Int32Array(capacity * 2);
    this.order = new Int32Array(capacity * 2);
    this.entPos = new Int32Array(capacity * MAX_OCC);
    this.nTouched = 0;
    this.nEntries = 0;
    this.vs = null;
  }
  rebuild(vs) {
    this.vs = vs;
    const high = vs.high;
    let total = 0;
    for (let i = 0; i < high; i++) { const st = vs.state[i]; if (st === STATE_DRIVING || st === STATE_CRASHED) total += vs.laneHi[i] - vs.laneLo[i] + 1; }
    if (this.order.length < total) { this.order = new Int32Array(total * 2); }
    if (this.touched.length < total) { const t = new Int32Array(total * 2); t.set(this.touched.subarray(0, this.nTouched)); this.touched = t; }
    if (this.entPos.length < high * MAX_OCC) this.entPos = new Int32Array(high * MAX_OCC * 2);
    const { count, touched, lstart, lend, order, entPos } = this;
    for (let q = 0; q < this.nTouched; q++) { const k = touched[q]; lstart[k] = 0; lend[k] = 0; }
    let nt = 0;
    for (let i = 0; i < high; i++) {
      const st = vs.state[i]; if (st !== STATE_DRIVING && st !== STATE_CRASHED) continue;
      const base = vs.edge[i] * MAX_LANES;
      for (let l = vs.laneLo[i]; l <= vs.laneHi[i]; l++) { const k = base + l; if (count[k] === 0) touched[nt++] = k; count[k]++; }
    }
    const tk = touched.subarray(0, nt); tk.sort();
    let acc = 0;
    for (let q = 0; q < nt; q++) { const k = tk[q]; lstart[k] = acc; acc += count[k]; lend[k] = lstart[k]; }
    for (let i = 0; i < high; i++) {
      const st = vs.state[i]; if (st !== STATE_DRIVING && st !== STATE_CRASHED) continue;
      const base = vs.edge[i] * MAX_LANES;
      for (let l = vs.laneLo[i]; l <= vs.laneHi[i]; l++) order[lend[base + l]++] = i;
    }
    const s = vs.s;
    for (let q = 0; q < nt; q++) {
      const k = tk[q], a = lstart[k], b = lend[k], lane = k % MAX_LANES;
      for (let i = a + 1; i < b; i++) {
        const v = order[i], sv = s[v]; let j = i - 1;
        while (j >= a && (s[order[j]] > sv || (s[order[j]] === sv && order[j] > v))) { order[j + 1] = order[j]; j--; }
        order[j + 1] = v;
      }
      for (let i = a; i < b; i++) { const v = order[i]; entPos[v * MAX_OCC + lane - vs.laneLo[v]] = i; }
      count[k] = 0;
    }
    this.nTouched = nt; this.nEntries = acc;
  }
  /** slot of the next car ahead of slot i in `lane` (default: i's centre lane) on i's edge, or -1 */
  ahead(i, lane = this.vs.lane[i]) {
    const vs = this.vs; if (lane < vs.laneLo[i] || lane > vs.laneHi[i]) return -1;
    const k = vs.edge[i] * MAX_LANES + lane, p = this.entPos[i * MAX_OCC + lane - vs.laneLo[i]] + 1;
    return p < this.lend[k] ? this.order[p] : -1;
  }
  behind(i, lane = this.vs.lane[i]) {
    const vs = this.vs; if (lane < vs.laneLo[i] || lane > vs.laneHi[i]) return -1;
    const k = vs.edge[i] * MAX_LANES + lane, p = this.entPos[i * MAX_OCC + lane - vs.laneLo[i]] - 1;
    return p >= this.lstart[k] ? this.order[p] : -1;
  }
  first(edge, lane) { const k = edge * MAX_LANES + lane; return this.lend[k] > this.lstart[k] ? this.order[this.lstart[k]] : -1; }
  last(edge, lane) { const k = edge * MAX_LANES + lane; return this.lend[k] > this.lstart[k] ? this.order[this.lend[k] - 1] : -1; }
  countOn(edge, lane) { const k = edge * MAX_LANES + lane; return this.lend[k] - this.lstart[k]; }
  /** Range [start, end) of `order` for (edge, lane). */
  range(edge, lane, out) { const k = edge * MAX_LANES + lane; out[0] = this.lstart[k]; out[1] = this.lend[k]; return out; }
  /**
   * Cars in (edge, lane) around position s: [first with s_j >= s (ahead), last with s_j < s (behind)]
   * (-1 if none), skipping `self`. Binary search on s.
   */
  neighborsAt(edge, lane, s, out = [-1, -1], self = -1) {
    const k = edge * MAX_LANES + lane, a = this.lstart[k], b = this.lend[k];
    const vs = this.vs, order = this.order;
    let lo = a, hi = b;
    while (lo < hi) { const m = (lo + hi) >> 1; if (vs.s[order[m]] < s) lo = m + 1; else hi = m; }
    let p = lo; while (p < b && order[p] === self) p++;
    let q = lo - 1; while (q >= a && order[q] === self) q--;
    out[0] = p < b ? order[p] : -1; out[1] = q >= a ? order[q] : -1;
    return out;
  }
}
