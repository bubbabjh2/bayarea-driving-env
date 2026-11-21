// Binary min-heap over (key: float, value: int) pairs backed by typed arrays.
// Used by the A* router. Lazy deletion: callers re-check gScore on pop.
export class MinHeap {
  constructor(capacity = 1024) {
    this.keys = new Float64Array(capacity);
    this.vals = new Int32Array(capacity);
    this.size = 0;
  }
  clear() { this.size = 0; }
  _grow() {
    const n = this.keys.length * 2;
    const k = new Float64Array(n); k.set(this.keys); this.keys = k;
    const v = new Int32Array(n); v.set(this.vals); this.vals = v;
  }
  push(key, val) {
    if (this.size === this.keys.length) this._grow();
    let i = this.size++;
    const keys = this.keys, vals = this.vals;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (keys[p] <= key) break;
      keys[i] = keys[p]; vals[i] = vals[p];
      i = p;
    }
    keys[i] = key; vals[i] = val;
  }
  /** Pops the min element; returns its value. Read `this.lastKey` for its key. */
  pop() {
    const keys = this.keys, vals = this.vals;
    const topVal = vals[0];
    this.lastKey = keys[0];
    const n = --this.size;
    if (n === 0) return topVal;
    const key = keys[n], val = vals[n];
    let i = 0;
    for (;;) {
      let c = 2 * i + 1;
      if (c >= n) break;
      if (c + 1 < n && keys[c + 1] < keys[c]) c++;
      if (keys[c] >= key) break;
      keys[i] = keys[c]; vals[i] = vals[c];
      i = c;
    }
    keys[i] = key; vals[i] = val;
    return topVal;
  }
}
