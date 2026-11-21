// Procedural, tileable canvas textures for the look pack. Nothing here runs at
// import time: every generator is called from applyLook() (src/look/index.js),
// so training / headless sim runs never pay for it.
//
// Budget (video preset): asphalt 1024² (5.6 MB with mips) + grass 512² (1.4 MB)
// + 3 × 512² coast/urban masks (no mips, 3 MB) ≈ 10 MB, well under 32 MB.

/** Small deterministic PRNG (mulberry32). */
function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Tileable value noise: lattice of `cells` × `cells` random values, wrapped. */
function tileNoise(size, cells, rand) {
  const g = new Float32Array(cells * cells);
  for (let i = 0; i < g.length; i++) g[i] = rand();
  const out = new Float32Array(size * size);
  const s = cells / size;
  for (let y = 0; y < size; y++) {
    const fy = y * s, iy = Math.floor(fy), ty = fy - iy, uy = ty * ty * (3 - 2 * ty);
    const y0 = (iy % cells) * cells, y1 = ((iy + 1) % cells) * cells;
    for (let x = 0; x < size; x++) {
      const fx = x * s, ix = Math.floor(fx), tx = fx - ix, ux = tx * tx * (3 - 2 * tx);
      const x0 = ix % cells, x1 = (ix + 1) % cells;
      const a = g[y0 + x0], b = g[y0 + x1], c = g[y1 + x0], d = g[y1 + x1];
      out[y * size + x] = (a + (b - a) * ux) + ((c + (d - c) * ux) - (a + (b - a) * ux)) * uy;
    }
  }
  return out;
}

function fbmTile(size, baseCells, octaves, rand, gain = 0.5) {
  const out = new Float32Array(size * size);
  let amp = 1, norm = 0, cells = baseCells;
  for (let o = 0; o < octaves; o++) {
    const n = tileNoise(size, Math.min(cells, size), rand);
    for (let i = 0; i < out.length; i++) out[i] += (n[i] - 0.5) * amp;
    norm += amp; amp *= gain; cells *= 2;
  }
  for (let i = 0; i < out.length; i++) out[i] /= norm;
  return out;   // roughly in [-0.5, 0.5]
}

/** Stamp a soft wrapped disc into a float buffer. */
function stamp(buf, size, cx, cy, r, v) {
  const r2 = r * r, ir = Math.ceil(r);
  for (let dy = -ir; dy <= ir; dy++) for (let dx = -ir; dx <= ir; dx++) {
    const d2 = dx * dx + dy * dy;
    if (d2 > r2) continue;
    const x = (cx + dx + size) % size, y = (cy + dy + size) % size;
    const k = 1 - d2 / (r2 + 1e-6);
    buf[y * size + x] += v * k;
  }
}

function toCanvas(size, fill) {
  const cv = document.createElement('canvas');
  cv.width = cv.height = size;
  const ctx = cv.getContext('2d');
  const img = ctx.createImageData(size, size);
  fill(img.data);
  ctx.putImageData(img, 0, 0);
  return cv;
}

/**
 * Asphalt detail: neutral grey around sRGB ~0.5 (shader divides by the mean,
 * so the texture only carries *variation*; per-class tone is a uniform).
 * Layers: soft mottling, fine grain, light aggregate stones, dark pits, a few
 * hairline cracks. Tileable.
 */
export function makeAsphaltCanvas(size = 1024, seed = 7) {
  const rand = rng(seed);
  const mott = fbmTile(size, 4, 4, rand, 0.55);
  const grain = tileNoise(size, size / 2, rand);
  const v = new Float32Array(size * size);
  for (let i = 0; i < v.length; i++) v[i] = mott[i] * 0.22 + (grain[i] - 0.5) * 0.16 + (rand() - 0.5) * 0.10;
  const px = size / 1024;   // texel scale relative to the 1024 version
  // aggregate: light stones and dark pits
  const nStones = Math.round(9000 * px * px);
  for (let k = 0; k < nStones; k++) {
    const r = (0.6 + rand() * 1.6) * px;
    stamp(v, size, (rand() * size) | 0, (rand() * size) | 0, Math.max(0.7, r), (rand() < 0.62 ? 1 : -1) * (0.12 + rand() * 0.22));
  }
  // hairline cracks: short random walks
  for (let c = 0; c < 10; c++) {
    let x = rand() * size, y = rand() * size, a = rand() * Math.PI * 2;
    const steps = (60 + rand() * 160) * px;
    for (let s = 0; s < steps; s++) {
      a += (rand() - 0.5) * 0.6;
      x += Math.cos(a); y += Math.sin(a);
      stamp(v, size, Math.round(x), Math.round(y), 0.8 * px + 0.3, -0.10);
    }
  }
  return toCanvas(size, (d) => {
    for (let i = 0; i < v.length; i++) {
      const g = Math.max(0, Math.min(255, 128 + v[i] * 255));
      const warm = mott[i] * 6;   // slight warm/cool tint in the mottling
      d[i * 4] = Math.max(0, Math.min(255, g + warm));
      d[i * 4 + 1] = g;
      d[i * 4 + 2] = Math.max(0, Math.min(255, g - warm * 0.6));
      d[i * 4 + 3] = 255;
    }
  });
}

/**
 * Grass/vegetation detail (tileable): luminance + hue variation centred on mid
 * grey, used as a multiplier at close range. R = brightness detail, G = dry
 * blade mask (blends towards straw), B = clump darkness.
 */
export function makeGrassCanvas(size = 512, seed = 11) {
  const rand = rng(seed);
  const clumps = fbmTile(size, 8, 4, rand, 0.6);
  const fine = tileNoise(size, size / 2, rand);
  const dry = fbmTile(size, 4, 3, rand, 0.5);
  const v = new Float32Array(size * size);
  for (let i = 0; i < v.length; i++) v[i] = clumps[i] * 0.5 + (fine[i] - 0.5) * 0.35 + (rand() - 0.5) * 0.2;
  // blade strokes: short bright/dark dashes
  for (let k = 0; k < size * size / 40; k++) {
    let x = rand() * size, y = rand() * size;
    const a = rand() * Math.PI * 2, len = 2 + rand() * 4, sgn = rand() < 0.5 ? 0.12 : -0.12;
    for (let s = 0; s < len; s++) { x += Math.cos(a); y += Math.sin(a); stamp(v, size, Math.round(x), Math.round(y), 0.7, sgn); }
  }
  return toCanvas(size, (d) => {
    for (let i = 0; i < v.length; i++) {
      d[i * 4] = Math.max(0, Math.min(255, 128 + v[i] * 200));
      d[i * 4 + 1] = Math.max(0, Math.min(255, 128 + dry[i] * 400));
      d[i * 4 + 2] = Math.max(0, Math.min(255, 128 - clumps[i] * 300));
      d[i * 4 + 3] = 255;
    }
  });
}

/** Downsample + gaussian-blur a mask canvas (white = 1) into a new canvas. */
export function blurMask(src, size, blurPx) {
  // 1) resample into a padded canvas whose border replicates the edge pixels (clamp-to-edge),
  //    so the blur does not darken the mask along its rectangle edges
  const pad = Math.ceil(blurPx * 3) + 2, P = size + pad * 2;
  const tmp = document.createElement('canvas');
  tmp.width = tmp.height = P;
  const t = tmp.getContext('2d');
  t.drawImage(src, pad, pad, size, size);
  t.drawImage(tmp, pad, pad, 1, size, 0, pad, pad, size);                 // left
  t.drawImage(tmp, pad + size - 1, pad, 1, size, pad + size, pad, pad, size);   // right
  t.drawImage(tmp, 0, pad, P, 1, 0, 0, P, pad);                           // top
  t.drawImage(tmp, 0, pad + size - 1, P, 1, 0, pad + size, P, pad);       // bottom
  // 2) blur, crop back to size
  const cv = document.createElement('canvas');
  cv.width = cv.height = size;
  const ctx = cv.getContext('2d');
  ctx.filter = blurPx > 0 ? `blur(${blurPx}px)` : 'none';
  ctx.drawImage(tmp, -pad, -pad);
  ctx.filter = 'none';
  return cv;
}

/**
 * Road-density field from line segments (Float32Array position pairs) over a
 * rectangle, blurred: high where there are many streets (towns), ~0 in hills.
 */
export function densityCanvas(segArrays, rect, size = 512, blurPx = 3) {
  const grid = new Float32Array(size * size);
  const sx = size / rect.w, sz = size / rect.h;
  for (const a of segArrays) {
    for (let i = 0; i + 5 < a.length; i += 6) {
      const mx = (a[i] + a[i + 3]) * 0.5, mz = (a[i + 2] + a[i + 5]) * 0.5;
      const gx = ((mx - rect.x) * sx) | 0, gz = ((mz - rect.z) * sz) | 0;
      if (gx < 0 || gz < 0 || gx >= size || gz >= size) continue;
      grid[gz * size + gx] += Math.hypot(a[i + 3] - a[i], a[i + 5] - a[i + 2]);
    }
  }
  const cellKm2 = (rect.w / size) * (rect.h / size) / 1e6;
  const raw = toCanvas(size, (d) => {
    for (let i = 0; i < grid.length; i++) {
      const kmPerKm2 = grid[i] / 1000 / cellKm2;          // suburbs ~15-25 km/km²
      const v = 255 * (1 - Math.exp(-kmPerKm2 / 9));
      d[i * 4] = d[i * 4 + 1] = d[i * 4 + 2] = v; d[i * 4 + 3] = 255;
    }
  });
  return blurMask(raw, size, blurPx);
}

/** Incremental road-density grid (streamed tiles add their line segments as they arrive). */
export function accumulateDensity(grid, a, rect, size, ox = 0, oz = 0) {
  const sx = size / rect.w, sz = size / rect.h;
  for (let i = 0; i + 5 < a.length; i += 6) {
    const mx = (a[i] + a[i + 3]) * 0.5 + ox, mz = (a[i + 2] + a[i + 5]) * 0.5 + oz;
    const gx = ((mx - rect.x) * sx) | 0, gz = ((mz - rect.z) * sz) | 0;
    if (gx < 0 || gz < 0 || gx >= size || gz >= size) continue;
    grid[gz * size + gx] += Math.hypot(a[i + 3] - a[i], a[i + 5] - a[i + 2]);
  }
}
export function densityFromGrid(grid, rect, size = 512, blurPx = 3) {
  const cellKm2 = (rect.w / size) * (rect.h / size) / 1e6;
  const raw = toCanvas(size, (d) => {
    for (let i = 0; i < grid.length; i++) {
      const v = 255 * (1 - Math.exp(-(grid[i] / 1000 / cellKm2) / 9));
      d[i * 4] = d[i * 4 + 1] = d[i * 4 + 2] = v; d[i * 4 + 3] = 255;
    }
  });
  return blurMask(raw, size, blurPx);
}
