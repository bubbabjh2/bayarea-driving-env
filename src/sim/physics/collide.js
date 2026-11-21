// Planar rigid-body contact physics (SPEC "Crashes: real physics only").
//
// A crash happens if and only if two hitboxes (oriented boxes in world x/z) overlap,
// or a hitbox reaches a road-edge barrier. There are no crash rules: detection is an
// exact separating-axis test with swept sub-steps (no tunnelling), the response is an
// impulse with restitution, Coulomb friction and angular terms plus positional
// correction, and severity is the per-vehicle delta-v of that impulse.
//
// Body (plain object, reused by the sim):
//   {x, z, yaw, vx, vz, w,  hl, hw,  invMass, invInertia}
//   x, z     centre (m); yaw = heading atan2(dz, dx) (rad); vx, vz (m/s); w = d(yaw)/dt (rad/s)
//   hl, hw   half length / half width of the hitbox (m)
//   invMass, invInertia  (0 = static: barrier, disabled car at rest)
// Contact: {depth, nx, nz, px, pz}: normal n points from A to B, p is the contact point.

export const PHYS = Object.freeze({
  restitution: 0.2,     // car-car (0.1-0.3)
  friction: 0.5,        // Coulomb coefficient at a car-car contact (sheet metal)
  wallRestitution: 0.15,
  wallFriction: 0.4,
  slop: 0.01,           // m of penetration left uncorrected
  correction: 0.8,      // fraction of the penetration removed per contact
  mu: 0.9,              // tyre-road friction (dry), the friction-circle radius is mu * g
  g: 9.81,
});

/** Yaw inertia of a uniform box (kg m^2). */
export function boxInertia(mass, length, width) { return mass * (length * length + width * width) / 12; }

function axes(b, out) { const c = Math.cos(b.yaw), s = Math.sin(b.yaw); out[0] = c; out[1] = s; out[2] = -s; out[3] = c; return out; }
const _a = new Float64Array(4), _b = new Float64Array(4);

/** Projection radius of box b onto unit axis (ux, uz). */
function radius(b, ax, ux, uz) { return Math.abs(ax[0] * ux + ax[1] * uz) * b.hl + Math.abs(ax[2] * ux + ax[3] * uz) * b.hw; }

/**
 * Separating-axis test between oriented boxes A and B. Returns null (separated) or a
 * contact {depth, nx, nz, px, pz} into `out`: n = minimum-penetration axis oriented
 * from A to B, p = the deepest vertex of the incident box (or the midpoint of the two
 * deepest vertices for edge-edge contact).
 */
export function obbContact(A, B, out = {}) {
  axes(A, _a); axes(B, _b);
  const dx = B.x - A.x, dz = B.z - A.z;
  let best = Infinity, bnx = 0, bnz = 0, fromA = true;
  for (let k = 0; k < 4; k++) {
    const src = k < 2 ? _a : _b, ux = src[(k & 1) * 2], uz = src[(k & 1) * 2 + 1];
    const d = dx * ux + dz * uz;
    const o = radius(A, _a, ux, uz) + radius(B, _b, ux, uz) - Math.abs(d);
    if (o <= 0) return null;
    if (o < best) { best = o; const sg = d < 0 ? -1 : 1; bnx = ux * sg; bnz = uz * sg; fromA = k < 2; }
  }
  // incident box = the one whose face is NOT the reference; its deepest vertex along the normal
  const inc = fromA ? B : A, ax = fromA ? _b : _a, sgn = fromA ? -1 : 1; // B's vertices go most negative along n into A
  let bestP = Infinity, px = 0, pz = 0, secondP = Infinity, qx = 0, qz = 0;
  for (let sx = -1; sx <= 1; sx += 2) for (let sz = -1; sz <= 1; sz += 2) {
    const vx = inc.x + ax[0] * sx * inc.hl + ax[2] * sz * inc.hw, vz = inc.z + ax[1] * sx * inc.hl + ax[3] * sz * inc.hw;
    const p = sgn * (vx * bnx + vz * bnz) * -1; // smaller = deeper
    if (p < bestP) { secondP = bestP; qx = px; qz = pz; bestP = p; px = vx; pz = vz; }
    else if (p < secondP) { secondP = p; qx = vx; qz = vz; }
  }
  if (secondP - bestP < 0.05) { px = 0.5 * (px + qx); pz = 0.5 * (pz + qz); } // edge-edge: middle of the contact edge
  out.depth = best; out.nx = bnx; out.nz = bnz; out.px = px; out.pz = pz;
  return out;
}

/**
 * Impulse response for contact c between A and B (n from A to B). Updates vx, vz, w of
 * both (static bodies untouched) and returns {jn, jt, dvA, dvB, closing} with the
 * normal / tangential impulse (N s), each body's delta-v (m/s) and the closing speed
 * along the normal before the impulse (m/s, > 0 = approaching).
 */
export function resolveContact(A, B, c, e = PHYS.restitution, mu = PHYS.friction, out = {}) {
  const rax = c.px - A.x, raz = c.pz - A.z, rbx = c.px - B.x, rbz = c.pz - B.z;
  // contact point velocities (2D cross: w x r = (-w rz, w rx))
  const vax = A.vx - A.w * raz, vaz = A.vz + A.w * rax;
  const vbx = B.vx - B.w * rbz, vbz = B.vz + B.w * rbx;
  const rvx = vbx - vax, rvz = vbz - vaz;
  const vn = rvx * c.nx + rvz * c.nz;
  out.jn = 0; out.jt = 0; out.dvA = 0; out.dvB = 0; out.closing = -vn;
  if (vn >= 0) return out; // separating already
  const raN = rax * c.nz - raz * c.nx, rbN = rbx * c.nz - rbz * c.nx;
  const kN = A.invMass + B.invMass + raN * raN * A.invInertia + rbN * rbN * B.invInertia;
  if (kN <= 0) return out;
  const jn = -(1 + e) * vn / kN;
  // tangent
  let tx = rvx - vn * c.nx, tz = rvz - vn * c.nz;
  const tl = Math.hypot(tx, tz);
  let jt = 0;
  if (tl > 1e-6) {
    tx /= tl; tz /= tl;
    const raT = rax * tz - raz * tx, rbT = rbx * tz - rbz * tx;
    const kT = A.invMass + B.invMass + raT * raT * A.invInertia + rbT * rbT * B.invInertia;
    jt = -(rvx * tx + rvz * tz) / kT;
    const lim = mu * jn; if (jt > lim) jt = lim; else if (jt < -lim) jt = -lim;
  } else { tx = 0; tz = 0; }
  const Px = jn * c.nx + jt * tx, Pz = jn * c.nz + jt * tz;  // impulse on B (A gets -P)
  const ax0 = A.vx, az0 = A.vz, bx0 = B.vx, bz0 = B.vz;
  A.vx -= Px * A.invMass; A.vz -= Pz * A.invMass; A.w -= (rax * Pz - raz * Px) * A.invInertia;
  B.vx += Px * B.invMass; B.vz += Pz * B.invMass; B.w += (rbx * Pz - rbz * Px) * B.invInertia;
  out.jn = jn; out.jt = jt;
  out.dvA = Math.hypot(A.vx - ax0, A.vz - az0); out.dvB = Math.hypot(B.vx - bx0, B.vz - bz0);
  return out;
}

/** Push A and B apart along the contact normal (split by inverse mass). Returns [dxA, dzA, dxB, dzB]. */
export function separate(A, B, c, out = [0, 0, 0, 0]) {
  const k = A.invMass + B.invMass;
  out[0] = out[1] = out[2] = out[3] = 0;
  if (k <= 0) return out;
  const corr = Math.max(0, c.depth - PHYS.slop) * PHYS.correction / k;
  out[0] = -c.nx * corr * A.invMass; out[1] = -c.nz * corr * A.invMass;
  out[2] = c.nx * corr * B.invMass; out[3] = c.nz * corr * B.invMass;
  A.x += out[0]; A.z += out[1]; B.x += out[2]; B.z += out[3];
  return out;
}

/**
 * Swept test between poses: A0/B0 = poses at the start of the step, A/B = at the end.
 * Samples the relative motion so that no sample moves more than `maxStep` m, returns the
 * earliest overlapping fraction t in (0, 1] with its contact in `out` (bodies' x/z/yaw are
 * left at the end pose), or -1.
 */
export function sweptContact(A0, A, B0, B, out, maxStep = 0.5, tmpA = {}, tmpB = {}) {
  const rel = Math.hypot((A.x - A0.x) - (B.x - B0.x), (A.z - A0.z) - (B.z - B0.z));
  const n = Math.max(1, Math.ceil(rel / maxStep));
  tmpA.hl = A.hl; tmpA.hw = A.hw; tmpB.hl = B.hl; tmpB.hw = B.hw;
  for (let k = 1; k <= n; k++) {
    const t = k / n;
    tmpA.x = A0.x + (A.x - A0.x) * t; tmpA.z = A0.z + (A.z - A0.z) * t; tmpA.yaw = lerpAngle(A0.yaw, A.yaw, t);
    tmpB.x = B0.x + (B.x - B0.x) * t; tmpB.z = B0.z + (B.z - B0.z) * t; tmpB.yaw = lerpAngle(B0.yaw, B.yaw, t);
    if (obbContact(tmpA, tmpB, out)) return t;
  }
  return -1;
}
function lerpAngle(a, b, t) { let d = b - a; while (d > Math.PI) d -= 2 * Math.PI; while (d < -Math.PI) d += 2 * Math.PI; return a + d * t; }

/** Severity tier from the larger per-vehicle delta-v (SPEC: km/h thresholds 10 / 25 / 50). */
export function severityOf(dvKmh) {
  return dvKmh < 10 ? 'minor' : dvKmh < 25 ? 'moderate' : dvKmh < 50 ? 'severe' : 'very severe';
}
