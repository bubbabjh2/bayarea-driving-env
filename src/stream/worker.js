// Tile fetch + gunzip + decode worker (module worker, no three.js).
// in:  {id, url} | {palette}
// out: {id, ok, bytes, header, chunks:[{role, prim, pos, cls, lane, idx, color}], water}
import { gunzipMaybe } from './gunzip.js';
import { MAGIC, HEADER_BYTES, PRIM, CHUNK_FLAG, ROLES, LANE_ACROSS_SCALE, LANE_ALONG_SCALE } from './format.js';

let palette = null;   // {roleName: {byClass: [[r,g,b]...] , dim}} colours for vertex-coloured roles


const pad4 = (n) => (n + 3) & ~3;

/** Area-weighted vertex normals (curb strips are emitted unshared, so edges stay sharp). Normals face up/outward. */
function faceNormals(pos, idx) {
  const n = new Float32Array(pos.length);
  for (let t = 0; t < idx.length; t += 3) {
    const a = idx[t] * 3, b = idx[t + 1] * 3, c = idx[t + 2] * 3;
    const ux = pos[b] - pos[a], uy = pos[b + 1] - pos[a + 1], uz = pos[b + 2] - pos[a + 2];
    const vx = pos[c] - pos[a], vy = pos[c + 1] - pos[a + 1], vz = pos[c + 2] - pos[a + 2];
    let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    if (ny < -1e-9) { nx = -nx; ny = -ny; nz = -nz; }
    for (const v of [a, b, c]) { n[v] += nx; n[v + 1] += ny; n[v + 2] += nz; }
  }
  for (let i = 0; i < n.length; i += 3) { const l = Math.hypot(n[i], n[i + 1], n[i + 2]) || 1; n[i] /= l; n[i + 1] /= l; n[i + 2] /= l; }
  return n;
}

function decode(buf) {
  const dv = new DataView(buf);
  if (dv.getUint32(0, true) !== MAGIC) throw new Error('bad tile magic');
  const header = { version: dv.getUint16(4, true), level: dv.getUint8(6), chunks: dv.getUint8(7),
    cx: dv.getFloat32(8, true), cz: dv.getFloat32(12, true), step: dv.getFloat32(16, true), yStep: dv.getFloat32(20, true),
    ix: dv.getUint32(24, true), iy: dv.getUint32(28, true) };
  const chunks = [], transfer = [];
  let water = null, o = HEADER_BYTES;
  for (let c = 0; c < header.chunks; c++) {
    const role = dv.getUint8(o), prim = dv.getUint8(o + 1), flags = dv.getUint8(o + 2);
    if (prim === PRIM.WATER) {
      const w = dv.getUint32(o + 4, true), h = dv.getUint32(o + 8, true);
      o += 12;
      if (flags & CHUNK_FLAG.FULL) { water = { full: true, w, h }; continue; }
      const nb = Math.ceil(w * h / 8), bits = new Uint8Array(buf, o, nb), px = new Uint8Array(w * h);
      for (let i = 0; i < w * h; i++) px[i] = (bits[i >> 3] >> (i & 7)) & 1 ? 255 : 0;
      o += pad4(nb);
      water = { full: false, w, h, data: px };
      transfer.push(px.buffer);
      continue;
    }
    const nv = dv.getUint32(o + 4, true), nIdx = dv.getUint32(o + 8, true);
    o += 12;
    const q = new Int16Array(buf, o, nv * 3); o += pad4(nv * 6);
    const pos = new Float32Array(nv * 3);
    for (let i = 0; i < nv; i++) { pos[i * 3] = q[i * 3] * header.step; pos[i * 3 + 1] = q[i * 3 + 1] * header.yStep; pos[i * 3 + 2] = q[i * 3 + 2] * header.step; }
    const cls = new Uint8Array(nv); cls.set(new Uint8Array(buf, o, nv)); o += pad4(nv);
    let lane = null, uv = null;
    if (flags & CHUNK_FLAG.HAS_LANE) {
      // laneCoord = (across, along) for the look's road shader; uv = (along, across) in metres for street materials
      const l = new Uint16Array(buf, o, nv * 2); lane = new Float32Array(nv * 2); uv = new Float32Array(nv * 2);
      for (let i = 0; i < nv; i++) {
        const ac = l[i * 2] / LANE_ACROSS_SCALE, al = l[i * 2 + 1] / LANE_ALONG_SCALE;
        lane[i * 2] = ac; lane[i * 2 + 1] = al; uv[i * 2] = al; uv[i * 2 + 1] = ac;
      }
      o += nv * 4;
    }
    let idx = null;
    if (prim === PRIM.TRI) {
      const T = flags & CHUNK_FLAG.IDX32 ? Uint32Array : Uint16Array;
      idx = new T(nIdx); idx.set(new T(buf, o, nIdx)); o += pad4(nIdx * T.BYTES_PER_ELEMENT);
    }
    const roleName = ROLES[role];
    let normal = null;
    if (roleName === 'curb' && idx) normal = faceNormals(pos, idx);
    let color = null;
    const pal = palette && palette[roleName];
    if (pal) {
      color = new Float32Array(nv * 3);
      for (let i = 0; i < nv; i++) {
        const c = cls[i], rgb = pal.byClass[c & 15] || pal.byClass[0], d = c & 16 ? pal.tunnelDim : 1;
        color[i * 3] = rgb[0] * d; color[i * 3 + 1] = rgb[1] * d; color[i * 3 + 2] = rgb[2] * d;
      }
      transfer.push(color.buffer);
    }
    transfer.push(pos.buffer, cls.buffer);
    if (lane) transfer.push(lane.buffer, uv.buffer);
    if (normal) transfer.push(normal.buffer);
    if (idx) transfer.push(idx.buffer);
    chunks.push({ role: roleName, prim, pos, cls, lane, uv, normal, idx, color });
  }
  return { header, chunks, water, transfer };
}

self.onmessage = async (ev) => {
  const m = ev.data;
  if (m.palette) { palette = m.palette; return; }
  try {
    const r = await fetch(m.url);
    if (!r.ok) throw new Error(`${m.url}: HTTP ${r.status}`);
    const raw = await r.arrayBuffer();
    const t0 = performance.now();
    const buf = await gunzipMaybe(raw);
    const { header, chunks, water, transfer } = decode(buf);
    self.postMessage({ id: m.id, ok: true, bytes: raw.byteLength, header, chunks, water, decodeMs: performance.now() - t0 }, transfer);
  } catch (err) {
    self.postMessage({ id: m.id, ok: false, error: String(err && err.message || err) });
  }
};
