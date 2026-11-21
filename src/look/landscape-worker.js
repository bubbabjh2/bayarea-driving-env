// Landscape tile worker (module worker, no three.js): fetch + gunzip + decode building / tree tiles and build
// the building meshes (extruded footprints: flat-shaded walls + ear-clipped roofs) off the main thread.
// in:  {id, kind: 'bld' | 'tree', url, cx, cz, tile}
// out: {id, ok, ...arrays} (transferred)

async function fetchGz(url) {
  const r = await fetch(url);
  if (r.status === 404) return null;
  if (!r.ok) throw new Error(url + ' ' + r.status);
  return new Uint8Array(await new Response(r.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
}

// building wall / roof palettes (sRGB 0..255): soft clay pastels; towers in cool light greys
const WALLS = [[236, 230, 218], [228, 220, 206], [240, 236, 228], [222, 214, 204], [232, 222, 210], [214, 210, 204], [238, 226, 212],
  [218, 222, 220], [226, 214, 196], [210, 216, 214], [242, 232, 214], [224, 206, 196], [214, 222, 206], [232, 216, 204]];
const TOWERS = [[214, 220, 226], [200, 208, 216], [226, 228, 230], [190, 198, 208], [220, 214, 206]];
const ROOFS = [[196, 188, 178], [180, 174, 168], [206, 196, 184], [170, 162, 156], [188, 170, 158], [200, 192, 182], [176, 180, 178]];

function earClip(xs, zs) {
  // CCW polygon (positive area in x-z) -> triangle index list (local); falls back to a fan
  const n = xs.length, idx = [];
  const V = Array.from({ length: n }, (_, i) => i);
  const area2 = (a, b, c) => (xs[b] - xs[a]) * (zs[c] - zs[a]) - (zs[b] - zs[a]) * (xs[c] - xs[a]);
  const inside = (p, a, b, c) => area2(a, b, p) >= 0 && area2(b, c, p) >= 0 && area2(c, a, p) >= 0;
  let guard = 0;
  while (V.length > 3 && guard++ < n * n) {
    let clipped = false;
    for (let i = 0; i < V.length; i++) {
      const a = V[(i + V.length - 1) % V.length], b = V[i], c = V[(i + 1) % V.length];
      if (area2(a, b, c) <= 1e-9) continue;
      let ear = true;
      for (const p of V) { if (p === a || p === b || p === c) continue; if (inside(p, a, b, c)) { ear = false; break; } }
      if (!ear) continue;
      idx.push(a, b, c); V.splice(i, 1); clipped = true; break;
    }
    if (!clipped) break;
  }
  if (V.length === 3) idx.push(V[0], V[1], V[2]);
  else if (V.length > 3) for (let i = 1; i + 1 < V.length; i++) idx.push(V[0], V[i], V[i + 1]);
  return idx;
}

function buildBuildings(u8, q) {
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const count = dv.getUint32(0, true);
  let o = 4;
  // first pass: sizes
  const recs = [];
  let nv = 0, ni = 0;
  for (let b = 0; b < count; b++) {
    const n = dv.getUint16(o, true), h = dv.getUint16(o + 2, true) / 10, style = u8[o + 4], flags = u8[o + 5];
    o += 6;
    const xs = new Float32Array(n), zs = new Float32Array(n);
    let px = 0, pz = 0;
    for (let i = 0; i < n; i++) { px += dv.getInt16(o, true); pz += dv.getInt16(o + 2, true); o += 4; xs[i] = px / q; zs[i] = pz / q; }
    recs.push({ n, h, style, flags, xs, zs });
    nv += n * 4 + n; ni += n * 6 + (n - 2) * 3;
  }
  const pos = new Float32Array(nv * 3), col = new Uint8Array(nv * 4), info = new Float32Array(nv * 4);
  const index = nv > 65535 ? new Uint32Array(ni) : new Uint16Array(ni);
  let v = 0, t = 0;
  const BURY = 6;   // walls continue below grade (hidden by the ground; covers sloped sites)
  for (const r of recs) {
    const { n, h, style, xs, zs } = r;
    let cx = 0, cz = 0; for (let i = 0; i < n; i++) { cx += xs[i]; cz += zs[i]; } cx /= n; cz /= n;
    const hsh = ((style * 2654435761) >>> 0) / 4294967296;
    const tall = r.flags & 2;
    const W = tall ? TOWERS[style % TOWERS.length] : WALLS[style % WALLS.length];
    const Rf = tall ? W.map((c) => c * 0.86) : ROOFS[(style >> 3) % ROOFS.length];
    // walls
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      const base = v;
      const P = [[xs[i], -BURY, zs[i]], [xs[j], -BURY, zs[j]], [xs[j], h, zs[j]], [xs[i], h, zs[i]]];
      for (const p of P) {
        pos[v * 3] = p[0]; pos[v * 3 + 1] = p[1]; pos[v * 3 + 2] = p[2];
        col[v * 4] = W[0]; col[v * 4 + 1] = W[1]; col[v * 4 + 2] = W[2]; col[v * 4 + 3] = 0;   // a = 0: wall
        info[v * 4] = cx; info[v * 4 + 1] = cz; info[v * 4 + 2] = hsh; info[v * 4 + 3] = h;
        v++;
      }
      // CCW footprint seen from above (x east, z south): outward faces wind (0, 2, 1), (0, 3, 2)
      index[t++] = base; index[t++] = base + 2; index[t++] = base + 1;
      index[t++] = base; index[t++] = base + 3; index[t++] = base + 2;
    }
    // roof
    const rb = v;
    for (let i = 0; i < n; i++) {
      pos[v * 3] = xs[i]; pos[v * 3 + 1] = h; pos[v * 3 + 2] = zs[i];
      col[v * 4] = Rf[0]; col[v * 4 + 1] = Rf[1]; col[v * 4 + 2] = Rf[2]; col[v * 4 + 3] = 255;   // a = 255: roof
      info[v * 4] = cx; info[v * 4 + 1] = cz; info[v * 4 + 2] = hsh; info[v * 4 + 3] = h;
      v++;
    }
    const tri = earClip(xs, zs);
    // seen from above, CCW in x-z (z south) is clockwise on screen: emit (a, c, b) so the roof faces up
    for (let k = 0; k + 2 < tri.length; k += 3) { index[t++] = rb + tri[k]; index[t++] = rb + tri[k + 2]; index[t++] = rb + tri[k + 1]; }
  }
  return { count, pos: pos.subarray(0, v * 3), col: col.subarray(0, v * 4), info: info.subarray(0, v * 4), index: index.subarray(0, t) };
}

function decodeTrees(u8) {
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const count = dv.getUint32(0, true);
  const out = new Float32Array(count * 4);
  for (let i = 0; i < count; i++) {
    const o = 4 + i * 6;
    out[i * 4] = dv.getInt16(o, true) / 16; out[i * 4 + 1] = dv.getInt16(o + 2, true) / 16;
    out[i * 4 + 2] = u8[o + 5] * 0.05; out[i * 4 + 3] = u8[o + 4];
  }
  return out;
}

self.onmessage = async (ev) => {
  const m = ev.data;
  try {
    const u8 = await fetchGz(m.url);
    if (!u8) { self.postMessage({ id: m.id, ok: true, empty: true }); return; }
    if (m.kind === 'bld') {
      const r = buildBuildings(u8, m.quant || 32);
      self.postMessage({ id: m.id, ok: true, ...r }, [r.pos.buffer, r.col.buffer, r.info.buffer, r.index.buffer]);
    } else {
      const trees = decodeTrees(u8);
      self.postMessage({ id: m.id, ok: true, trees }, [trees.buffer]);
    }
  } catch (e) { self.postMessage({ id: m.id, ok: false, error: String(e.message || e) }); }
};
