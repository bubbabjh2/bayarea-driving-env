// Even-odd scanline rasteriser for data/water.json, per tile.
// One scanline structure per polygon (even-odd inside a polygon, union across polygons,
// exactly like the canvas fill in src/water.js).
export class WaterRaster {
  constructor(water, band = 250) {
    this.parts = (water.polygons || []).map((p) => new PolyRaster({ polygons: [p] }, band));
  }
  tile(minX, minZ, size, res) {
    let bits = null, count = 0;
    for (const p of this.parts) {
      const t = p.tile(minX, minZ, size, res);
      if (!t) continue;
      if (t.full) return { full: true, res };
      if (!bits) bits = new Uint8Array(t.bits.length);
      for (let i = 0; i < bits.length; i++) bits[i] |= t.bits[i];
    }
    if (!bits) return null;
    for (let i = 0; i < bits.length; i++) { let v = bits[i]; while (v) { count += v & 1; v >>= 1; } }
    if (count === res * res) return { full: true, res };
    return { full: false, bits, res };
  }
}

class PolyRaster {
  constructor(water, band = 250) {
    this.band = band;
    const edges = [];
    let minZ = Infinity, maxZ = -Infinity;
    const ring = (r) => {
      for (let i = 0; i < r.length; i++) {
        const a = r[i], b = r[(i + 1) % r.length];
        if (a[1] === b[1]) continue;
        edges.push([a[0], a[1], b[0], b[1]]);
        minZ = Math.min(minZ, a[1], b[1]); maxZ = Math.max(maxZ, a[1], b[1]);
      }
    };
    for (const p of water.polygons || []) { ring(p.outer); (p.holes || []).forEach(ring); }
    this.z0 = minZ; this.nb = Math.max(1, Math.ceil((maxZ - minZ) / band) + 1);
    this.buckets = Array.from({ length: this.nb }, () => []);
    for (const e of edges) {
      const z1 = Math.min(e[1], e[3]), z2 = Math.max(e[1], e[3]);
      const b1 = Math.floor((z1 - minZ) / band), b2 = Math.floor((z2 - minZ) / band);
      for (let b = b1; b <= b2; b++) this.buckets[b].push(e);
    }
    this.rings = water.polygons || [];
  }
  /** Returns null (all land), {full:true} (all water) or {bits, res}. Row 0 = min z (north). */
  tile(minX, minZ, size, res) {
    const bits = new Uint8Array(Math.ceil(res * res / 8));
    const px = size / res;
    let count = 0;
    const xs = [];
    for (let r = 0; r < res; r++) {
      const z = minZ + (r + 0.5) * px;
      const b = Math.floor((z - this.z0) / this.band);
      if (b < 0 || b >= this.nb) continue;
      xs.length = 0;
      for (const e of this.buckets[b]) {
        const [x1, z1, x2, z2] = e;
        if ((z1 <= z) !== (z2 <= z)) xs.push(x1 + (z - z1) * (x2 - x1) / (z2 - z1));
      }
      if (xs.length < 2) continue;
      xs.sort((a, b) => a - b);
      for (let k = 0; k + 1 < xs.length; k += 2) {
        let c0 = Math.ceil((xs[k] - minX) / px - 0.5), c1 = Math.floor((xs[k + 1] - minX) / px - 0.5);
        if (c0 < 0) c0 = 0; if (c1 > res - 1) c1 = res - 1;
        for (let c = c0; c <= c1; c++) { const i = r * res + c; bits[i >> 3] |= 1 << (i & 7); count++; }
      }
    }
    if (count === 0) return null;
    if (count === res * res) return { full: true, res };
    return { full: false, bits, res };
  }
}
