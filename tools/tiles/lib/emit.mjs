// Tile buckets: geometry is generated in world space and every triangle
// (centroid) / line segment (midpoint) is assigned to exactly one tile of the
// level. Vertices are quantised on a grid aligned to the level origin, so a
// vertex shared by triangles in two neighbouring tiles decodes to the same
// world position in both: no cracks, no overlaps.
import { MAGIC, VERSION, HEADER_BYTES, PRIM, CHUNK_FLAG, XZ_UNITS, Y_STEP, ROLE, ROLES,
  LANE_ALONG_SCALE, LANE_ALONG_WRAP, LANE_ACROSS_SCALE } from '../../../src/stream/format.js';

class Grow {
  constructor(T, n = 256) { this.T = T; this.a = new T(n); this.n = 0; }
  ensure(k) { if (this.n + k > this.a.length) { let m = this.a.length * 2; while (m < this.n + k) m *= 2; const b = new this.T(m); b.set(this.a.subarray(0, this.n)); this.a = b; } }
  push(v) { if (this.n >= this.a.length) this.ensure(1); this.a[this.n++] = v; }
  view() { return this.a.subarray(0, this.n); }
}

class RoleBuf {
  constructor(prim, lane) {
    this.prim = prim;
    this.pos = new Grow(Float64Array, 768); this.cls = new Grow(Uint8Array, 256);
    this.lane = lane ? new Grow(Float32Array, 512) : null;
    this.idx = prim === PRIM.TRI ? new Grow(Uint32Array, 768) : null;
  }
  get nv() { return this.cls.n; }
  vert(x, y, z, c, la, lb) {
    this.pos.push(x); this.pos.push(y); this.pos.push(z); this.cls.push(c);
    if (this.lane) { this.lane.push(la || 0); this.lane.push(lb || 0); }
    return this.cls.n - 1;
  }
}

export class Level {
  constructor(L, root, spec) {
    this.L = L; this.spec = spec;
    this.n = 1 << L; this.size = root.size / this.n;
    this.x0 = root.x0; this.z0 = root.z0;
    this.tiles = new Map();
  }
  key(ix, iy) { return ix * 65536 + iy; }
  tileAt(x, z) {
    let ix = Math.floor((x - this.x0) / this.size), iy = Math.floor((z - this.z0) / this.size);
    ix = ix < 0 ? 0 : ix >= this.n ? this.n - 1 : ix; iy = iy < 0 ? 0 : iy >= this.n ? this.n - 1 : iy;
    return this.tile(ix, iy);
  }
  tile(ix, iy) {
    const k = this.key(ix, iy);
    let t = this.tiles.get(k);
    if (!t) { t = { ix, iy, roles: new Map(), water: null }; this.tiles.set(k, t); }
    return t;
  }
  buf(t, role, prim, lane) {
    const r = ROLE[role]; if (r === undefined) throw new Error('unknown role ' + role);
    let b = t.roles.get(r);
    if (!b) { b = new RoleBuf(prim, lane); t.roles.set(r, b); }
    return b;
  }
  /**
   * Triangles. V = {x,y,z} arrays; cls: number or per-vertex array;
   * lane: null or per-vertex [across, along] flat array; tris: flat vertex indices.
   */
  tris(role, V, cls, lane, tris) {
    const maps = new Map();
    const hasLane = !!lane;
    for (let i = 0; i < tris.length; i += 3) {
      const a = tris[i], b = tris[i + 1], c = tris[i + 2];
      const t = this.tileAt((V.x[a] + V.x[b] + V.x[c]) / 3, (V.z[a] + V.z[b] + V.z[c]) / 3);
      const buf = this.buf(t, role, PRIM.TRI, hasLane);
      let m = maps.get(buf);
      if (!m) { m = new Int32Array(V.x.length).fill(-1); maps.set(buf, m); }
      for (const v of [a, b, c]) {
        if (m[v] < 0) m[v] = buf.vert(V.x[v], V.y[v], V.z[v], typeof cls === 'number' ? cls : cls[v], hasLane ? lane[v * 2] : 0, hasLane ? lane[v * 2 + 1] : 0);
        buf.idx.push(m[v]);
      }
    }
  }
  /** Line strip through V (consecutive points), one segment per tile by midpoint. */
  lines(role, V, cls) {
    for (let i = 1; i < V.x.length; i++) {
      const t = this.tileAt((V.x[i - 1] + V.x[i]) / 2, (V.z[i - 1] + V.z[i]) / 2);
      const buf = this.buf(t, role, PRIM.LINES, false);
      buf.vert(V.x[i - 1], V.y[i - 1], V.z[i - 1], cls);
      buf.vert(V.x[i], V.y[i], V.z[i], cls);
    }
  }
  tileBounds(ix, iy) {
    return { minX: this.x0 + ix * this.size, minZ: this.z0 + iy * this.size, maxX: this.x0 + (ix + 1) * this.size, maxZ: this.z0 + (iy + 1) * this.size };
  }
}

const pad4 = (n) => (n + 3) & ~3;

/** Serialise one tile. Returns { bytes: Uint8Array, bounds, overflow, verts, tris }. */
export function encodeTile(level, t) {
  const step = level.size / XZ_UNITS;
  const ox = Math.round((t.ix + 0.5) * XZ_UNITS), oz = Math.round((t.iy + 0.5) * XZ_UNITS);
  const cx = level.x0 + ox * step, cz = level.z0 + oz * step;
  const chunks = [];
  const tb = level.tileBounds(t.ix, t.iy);
  const bounds = { minX: Infinity, minY: Infinity, minZ: Infinity, maxX: -Infinity, maxY: -Infinity, maxZ: -Infinity };
  let overflow = 0, verts = 0, tris = 0;
  const roleIds = [...t.roles.keys()].sort((a, b) => a - b);
  for (const r of roleIds) {
    const b = t.roles.get(r);
    const nv = b.nv; if (!nv) continue;
    const nIdx = b.idx ? b.idx.n : 0;
    const idx32 = nv > 65535;
    let flags = 0;
    if (b.lane) flags |= CHUNK_FLAG.HAS_LANE;
    if (idx32) flags |= CHUNK_FLAG.IDX32;
    const size = 12 + pad4(nv * 6) + pad4(nv) + (b.lane ? nv * 4 : 0) + pad4(nIdx * (idx32 ? 4 : 2));
    const buf = new ArrayBuffer(size), dv = new DataView(buf);
    dv.setUint8(0, r); dv.setUint8(1, b.prim); dv.setUint8(2, flags); dv.setUint32(4, nv, true); dv.setUint32(8, nIdx, true);
    let o = 12;
    const pos = new Int16Array(buf, o, nv * 3); o += pad4(nv * 6);
    const P = b.pos.a;
    for (let i = 0; i < nv; i++) {
      const x = P[i * 3], y = P[i * 3 + 1], z = P[i * 3 + 2];
      let qx = Math.round((x - level.x0) / step) - ox, qz = Math.round((z - level.z0) / step) - oz, qy = Math.round(y / Y_STEP);
      if (qx < -32767 || qx > 32767 || qz < -32767 || qz > 32767 || qy < -32767 || qy > 32767) overflow++;
      qx = Math.max(-32767, Math.min(32767, qx)); qz = Math.max(-32767, Math.min(32767, qz)); qy = Math.max(-32767, Math.min(32767, qy));
      pos[i * 3] = qx; pos[i * 3 + 1] = qy; pos[i * 3 + 2] = qz;
      const wx = cx + qx * step, wy = qy * Y_STEP, wz = cz + qz * step;
      if (wx < bounds.minX) bounds.minX = wx; if (wx > bounds.maxX) bounds.maxX = wx;
      if (wy < bounds.minY) bounds.minY = wy; if (wy > bounds.maxY) bounds.maxY = wy;
      if (wz < bounds.minZ) bounds.minZ = wz; if (wz > bounds.maxZ) bounds.maxZ = wz;
    }
    new Uint8Array(buf, o, nv).set(b.cls.view()); o += pad4(nv);
    if (b.lane) {
      const la = new Uint16Array(buf, o, nv * 2), L = b.lane.a;
      for (let i = 0; i < nv; i++) {
        la[i * 2] = Math.max(0, Math.min(65535, Math.round(L[i * 2] * LANE_ACROSS_SCALE)));
        const al = ((L[i * 2 + 1] % LANE_ALONG_WRAP) + LANE_ALONG_WRAP) % LANE_ALONG_WRAP;
        la[i * 2 + 1] = Math.min(65535, Math.round(al * LANE_ALONG_SCALE));
      }
      o += nv * 4;
    }
    if (nIdx) {
      (idx32 ? new Uint32Array(buf, o, nIdx) : new Uint16Array(buf, o, nIdx)).set(b.idx.view());
      tris += nIdx / 3;
    }
    verts += nv;
    chunks.push(new Uint8Array(buf));
  }
  if (t.water) {
    const w = t.water;
    const bits = w.full ? 0 : pad4(w.bits.length);
    const buf = new ArrayBuffer(12 + bits), dv = new DataView(buf);
    dv.setUint8(0, ROLE.water); dv.setUint8(1, PRIM.WATER); dv.setUint8(2, w.full ? CHUNK_FLAG.FULL : 0);
    dv.setUint32(4, w.res, true); dv.setUint32(8, w.res, true);
    if (!w.full) new Uint8Array(buf, 12, w.bits.length).set(w.bits);
    chunks.push(new Uint8Array(buf));
    bounds.minX = Math.min(bounds.minX, tb.minX); bounds.maxX = Math.max(bounds.maxX, tb.maxX);
    bounds.minZ = Math.min(bounds.minZ, tb.minZ); bounds.maxZ = Math.max(bounds.maxZ, tb.maxZ);
    bounds.minY = Math.min(bounds.minY, 0); bounds.maxY = Math.max(bounds.maxY, 0.5);
  }
  const total = HEADER_BYTES + chunks.reduce((s, c) => s + c.length, 0);
  const out = new Uint8Array(total), dv = new DataView(out.buffer);
  dv.setUint32(0, MAGIC, true); dv.setUint16(4, VERSION, true); dv.setUint8(6, level.L); dv.setUint8(7, chunks.length);
  dv.setFloat32(8, cx, true); dv.setFloat32(12, cz, true); dv.setFloat32(16, step, true); dv.setFloat32(20, Y_STEP, true);
  dv.setUint32(24, t.ix, true); dv.setUint32(28, t.iy, true);
  let o = HEADER_BYTES;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return { bytes: out, bounds, overflow, verts, tris };
}

export { ROLES };
