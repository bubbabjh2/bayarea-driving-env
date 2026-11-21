// Packed route storage (GPU-uploadable): one Int32Array `flat` of directed edge ids.
// [0, baseLen) is the precomputed population pack (route of trip k = 2*commuter+leg is
// flat[popOffset[k] .. popOffset[k+1])). Routes created during the run (refuel detours,
// reroutes) are bump-allocated after it; when the dynamic region is full it is compacted
// (only routes still referenced by a vehicle are kept) and the callback re-points every
// vehicle's (routeOff, routes[i]) view.

export class RouteStore {
  constructor(popRoutes = null, dynamicCapacity = 1 << 20) {
    const base = popRoutes ? popRoutes.flat : new Int32Array(0);
    this.baseLen = base.length;
    this.flat = new Int32Array(this.baseLen + dynamicCapacity);
    this.flat.set(base);
    this.used = this.baseLen;
    this.popOffset = popRoutes ? popRoutes.offset : null;
    this.compactions = 0;
  }
  /** View of a population route (no copy). */
  popRoute(commuter, leg) {
    const o = this.popOffset; if (!o) return null;
    const k = 2 * commuter + leg; return this.flat.subarray(o[k], o[k + 1]);
  }
  /** Offset of a view that lives in this store, or -1. */
  offsetOf(view) {
    return view && view.buffer === this.flat.buffer ? view.byteOffset / 4 : -1;
  }
  /**
   * Store a new route; returns {off, view}. `live(callback)` must enumerate
   * (slot, off, len) of every route still in use so compaction can relocate them.
   */
  add(edges, live, relocate) {
    const n = edges.length;
    if (this.used + n > this.flat.length) this._compact(n, live, relocate);
    const off = this.used;
    this.flat.set(edges, off);
    this.used += n;
    return { off, view: this.flat.subarray(off, off + n) };
  }
  _compact(need, live, relocate) {
    const keep = [];
    live((slot, off, len) => { if (off >= this.baseLen) keep.push([slot, off, len]); });
    let total = this.baseLen; for (const [, , len] of keep) total += len;
    const cap = Math.max(this.flat.length, total + need + (1 << 18));
    const flat = new Int32Array(cap);
    flat.set(this.flat.subarray(0, this.baseLen));
    // identical routes shared by several slots keep sharing
    const moved = new Map();
    let p = this.baseLen;
    for (const [slot, off, len] of keep) {
      let no = moved.get(off);
      if (no === undefined) { no = p; flat.set(this.flat.subarray(off, off + len), p); p += len; moved.set(off, no); }
      relocate(slot, no, flat.subarray(no, no + len));
    }
    this.flat = flat; this.used = p; this.compactions++;
    // population views must point into the new buffer too
    this.onRebase && this.onRebase(flat);
  }
}
