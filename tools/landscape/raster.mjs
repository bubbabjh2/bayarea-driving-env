// Scanline polygon rasteriser (pixel-centre sampling, even-odd rule across all rings of one shape).
// rings: array of flat arrays [x0, y0, x1, y1, ...] in PIXEL coordinates (closed implicitly).
// paint(rowIndex, x0, x1) is called for every covered span [x0, x1) (clipped to 0..W).
export function scanFill(rings, W, H, paint) {
  // edge list: (ymin, ymax, x at ymin, dx/dy), ignoring horizontal edges
  let n = 0;
  for (const r of rings) n += r.length >> 1;
  const ey0 = new Float64Array(n), ey1 = new Float64Array(n), ex0 = new Float64Array(n), edx = new Float64Array(n);
  let m = 0, ymin = Infinity, ymax = -Infinity;
  for (const r of rings) {
    const k = r.length >> 1; if (k < 3) continue;
    for (let i = 0; i < k; i++) {
      const j = (i + 1) % k;
      let xa = r[i * 2], ya = r[i * 2 + 1], xb = r[j * 2], yb = r[j * 2 + 1];
      if (ya === yb || !Number.isFinite(ya) || !Number.isFinite(yb)) continue;
      if (ya > yb) { let t = xa; xa = xb; xb = t; t = ya; ya = yb; yb = t; }
      ey0[m] = ya; ey1[m] = yb; ex0[m] = xa; edx[m] = (xb - xa) / (yb - ya); m++;
      if (ya < ymin) ymin = ya; if (yb > ymax) ymax = yb;
    }
  }
  if (!m) return;
  const r0 = Math.max(0, Math.ceil(ymin - 0.5)), r1 = Math.min(H - 1, Math.floor(ymax - 0.5));
  if (r0 > r1) return;
  // bucket edges by first row
  const order = new Int32Array(m); for (let i = 0; i < m; i++) order[i] = i;
  order.sort((a, b) => ey0[a] - ey0[b]);
  let next = 0; const active = []; const xs = [];
  for (let row = r0; row <= r1; row++) {
    const yc = row + 0.5;
    while (next < m && ey0[order[next]] <= yc) { active.push(order[next]); next++; }
    xs.length = 0;
    for (let a = active.length - 1; a >= 0; a--) {
      const e = active[a];
      if (ey1[e] <= yc) { active[a] = active[active.length - 1]; active.pop(); continue; }
      if (ey0[e] <= yc) xs.push(ex0[e] + (yc - ey0[e]) * edx[e]);
    }
    if (xs.length < 2) continue;
    xs.sort((a, b) => a - b);
    for (let i = 0; i + 1 < xs.length; i += 2) {
      const a = Math.max(0, Math.ceil(xs[i] - 0.5)), b = Math.min(W, Math.ceil(xs[i + 1] - 0.5));
      if (b > a) paint(row, a, b);
    }
  }
}

/** Thick polyline (flat caps) -> quad rings, for runways / taxiways. pts flat [x,y,...], width in the same units. */
export function lineToRings(pts, width) {
  const out = [], hw = width / 2;
  for (let i = 0; i + 3 < pts.length; i += 2) {
    const ax = pts[i], ay = pts[i + 1], bx = pts[i + 2], by = pts[i + 3];
    const dx = bx - ax, dy = by - ay, L = Math.hypot(dx, dy); if (L < 1e-9) continue;
    const nx = -dy / L * hw, ny = dx / L * hw, ex = dx / L * hw * 0.5, ey = dy / L * hw * 0.5;
    out.push([ax + nx - ex, ay + ny - ey, bx + nx + ex, by + ny + ey, bx - nx + ex, by - ny + ey, ax - nx - ex, ay - ny - ey]);
  }
  return out;
}
