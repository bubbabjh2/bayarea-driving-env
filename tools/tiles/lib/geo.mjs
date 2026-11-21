// Polyline helpers for the tiler (plain arrays of numbers, world metres).

/** Insert points so no segment is longer than maxLen (y interpolated). */
export function subdivide(P, maxLen) {
  const out = { x: [P.x[0]], y: [P.y[0]], z: [P.z[0]] };
  for (let i = 1; i < P.x.length; i++) {
    const dx = P.x[i] - P.x[i - 1], dz = P.z[i] - P.z[i - 1];
    const L = Math.hypot(dx, dz);
    const k = Math.ceil(L / maxLen);
    for (let j = 1; j < k; j++) {
      const t = j / k;
      out.x.push(P.x[i - 1] + dx * t); out.z.push(P.z[i - 1] + dz * t);
      out.y.push(P.y[i - 1] + (P.y[i] - P.y[i - 1]) * t);
    }
    out.x.push(P.x[i]); out.y.push(P.y[i]); out.z.push(P.z[i]);
  }
  return out;
}

/** Douglas-Peucker in xz (and y, so ramps keep their shape). Endpoints kept. */
export function simplify(P, tol) {
  const n = P.x.length;
  if (n <= 2 || tol <= 0) return P;
  const keep = new Uint8Array(n); keep[0] = keep[n - 1] = 1;
  const stack = [0, n - 1];
  const t2 = tol * tol;
  while (stack.length) {
    const b = stack.pop(), a = stack.pop();
    const ax = P.x[a], az = P.z[a], ay = P.y[a];
    const dx = P.x[b] - ax, dz = P.z[b] - az, dy = P.y[b] - ay;
    const L2 = dx * dx + dz * dz;
    let best = -1, bi = -1;
    for (let i = a + 1; i < b; i++) {
      let t = L2 > 0 ? ((P.x[i] - ax) * dx + (P.z[i] - az) * dz) / L2 : 0;
      t = t < 0 ? 0 : t > 1 ? 1 : t;
      const ex = P.x[i] - (ax + dx * t), ez = P.z[i] - (az + dz * t), ey = P.y[i] - (ay + dy * t);
      const d = ex * ex + ez * ez + ey * ey * 4;   // elevation errors count double
      if (d > best) { best = d; bi = i; }
    }
    if (best > t2) { keep[bi] = 1; stack.push(a, bi, bi, b); }
  }
  const out = { x: [], y: [], z: [] };
  for (let i = 0; i < n; i++) if (keep[i]) { out.x.push(P.x[i]); out.y.push(P.y[i]); out.z.push(P.z[i]); }
  return out;
}

export function length(P) {
  let s = 0;
  for (let i = 1; i < P.x.length; i++) s += Math.hypot(P.x[i] - P.x[i - 1], P.z[i] - P.z[i - 1]);
  return s;
}

/** Cut d0 metres off the start and d1 off the end. Returns null if nothing is left. */
export function trim(P, d0, d1) {
  const L = length(P);
  if (d0 + d1 >= L - 0.05) return null;
  const out = { x: [], y: [], z: [] };
  const a = d0, b = L - d1;
  let s = 0;
  const at = (i, t) => { out.x.push(P.x[i] + (P.x[i + 1] - P.x[i]) * t); out.y.push(P.y[i] + (P.y[i + 1] - P.y[i]) * t); out.z.push(P.z[i] + (P.z[i + 1] - P.z[i]) * t); };
  for (let i = 0; i < P.x.length - 1; i++) {
    const l = Math.hypot(P.x[i + 1] - P.x[i], P.z[i + 1] - P.z[i]);
    const s1 = s + l;
    if (s1 > a && s < b) {
      if (out.x.length === 0) at(i, l > 0 ? Math.max(0, (a - s) / l) : 0);
      if (s1 < b) { out.x.push(P.x[i + 1]); out.y.push(P.y[i + 1]); out.z.push(P.z[i + 1]); }
      else { at(i, l > 0 ? (b - s) / l : 1); break; }
    }
    s = s1;
  }
  return out.x.length >= 2 ? out : null;
}

/**
 * Per-vertex right vectors (unit right of travel scaled by the mitre factor,
 * clamped like src/roads.js) and cumulative distance.
 */
export function frames(P) {
  const n = P.x.length, rx = new Float64Array(n), rz = new Float64Array(n), s = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let dx = 0, dz = 0;
    if (i > 0) { const ax = P.x[i] - P.x[i - 1], az = P.z[i] - P.z[i - 1], l = Math.hypot(ax, az) || 1; dx += ax / l; dz += az / l; s[i] = s[i - 1] + Math.hypot(ax, az); }
    if (i < n - 1) { const bx = P.x[i + 1] - P.x[i], bz = P.z[i + 1] - P.z[i], l = Math.hypot(bx, bz) || 1; dx += bx / l; dz += bz / l; }
    const len = Math.hypot(dx, dz) || 1; dx /= len; dz /= len;
    let scale = 1;
    if (i > 0 && i < n - 1) {
      const ax = P.x[i] - P.x[i - 1], az = P.z[i] - P.z[i - 1], al = Math.hypot(ax, az) || 1;
      scale = Math.min(2.0, 1 / Math.max(0.5, (ax / al) * dx + (az / al) * dz));
    }
    rx[i] = -dz * scale; rz[i] = dx * scale;
  }
  return { rx, rz, s };
}

/** Convex hull (monotone chain) of [x,z] points; CCW when viewed from above (+y). */
export function hull(pts) {
  const p = pts.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (p.length < 3) return p;
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lo = [], up = [];
  for (const q of p) { while (lo.length >= 2 && cross(lo[lo.length - 2], lo[lo.length - 1], q) <= 0) lo.pop(); lo.push(q); }
  for (let i = p.length - 1; i >= 0; i--) { const q = p[i]; while (up.length >= 2 && cross(up[up.length - 2], up[up.length - 1], q) <= 0) up.pop(); up.push(q); }
  up.pop(); lo.pop();
  return lo.concat(up);
}
