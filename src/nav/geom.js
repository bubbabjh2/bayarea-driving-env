// Small 2D geometry helpers shared by nav + brain.
//
// COORDINATE CONVENTION (matches a Three.js y-up world where x = east):
//   x = east, z = south (north is -z). Headings are radians measured from +x
//   toward +z, i.e. heading = atan2(dz, dx). Forward vector = (cos h, sin h).
//   With y up, the LEFT-hand side of a car heading (fx, fz) is (fz, -fx).
//   A heading delta < 0 is a LEFT turn, delta > 0 is a RIGHT turn.
// If the world agent ends up with z = north instead, flip HANDEDNESS to -1 and
// every left/right decision in nav + brain flips with it.
export const HANDEDNESS = 1;

export const TAU = Math.PI * 2;
export const MPH_TO_MS = 0.44704;
export const MS_TO_MPH = 1 / MPH_TO_MS;

export function wrapAngle(a) {
  a = a % TAU;
  if (a > Math.PI) a -= TAU;
  else if (a < -Math.PI) a += TAU;
  return a;
}

export function heading(x0, z0, x1, z1) {
  return Math.atan2(z1 - z0, x1 - x0);
}

export function dist(x0, z0, x1, z1) {
  const dx = x1 - x0, dz = z1 - z0;
  return Math.sqrt(dx * dx + dz * dz);
}

/** Left-hand unit vector for a forward vector (fx, fz). */
export function leftOf(fx, fz) {
  return [HANDEDNESS * fz, -HANDEDNESS * fx];
}

/** Signed lateral offset of point p relative to origin o facing heading h (+ = left). */
export function lateralOffset(ox, oz, h, px, pz) {
  const fx = Math.cos(h), fz = Math.sin(h);
  const [lx, lz] = leftOf(fx, fz);
  return (px - ox) * lx + (pz - oz) * lz;
}

/** 'left' | 'right' | 'straight' for a heading delta, with a threshold in radians. */
export function turnSide(delta, threshold = 0) {
  const d = delta * HANDEDNESS;
  if (d < -threshold) return 'left';
  if (d > threshold) return 'right';
  return 'straight';
}

export const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
export const deg = (r) => (r * 180) / Math.PI;
export const rad = (d) => (d * Math.PI) / 180;
