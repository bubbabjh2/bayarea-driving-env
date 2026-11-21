// Pure rigid-body contact maths, 1:1 with src/sim/physics/collide.js (obbContact, resolveContact,
// lerpAngle). No bindings: included by contacts.wgsl and test/gpu/physics.html.

struct Body { x: f32, z: f32, yaw: f32, vx: f32, vz: f32, w: f32, hl: f32, hw: f32, invMass: f32, invInertia: f32 };
struct Contact { depth: f32, nx: f32, nz: f32, px: f32, pz: f32, ok: bool };
struct Res { jn: f32, dvA: f32, dvB: f32, closing: f32 };

fn radiusOn(b: Body, ux: f32, uz: f32) -> f32 {
  let c = cos(b.yaw); let s = sin(b.yaw);
  return abs(c * ux + s * uz) * b.hl + abs(-s * ux + c * uz) * b.hw;
}
// collide.js obbContact
fn obbContact(A: Body, B: Body) -> Contact {
  var out: Contact; out.ok = false;
  let ca = cos(A.yaw); let sa = sin(A.yaw); let cb = cos(B.yaw); let sb = sin(B.yaw);
  let ax = array<vec2f, 4>(vec2f(ca, sa), vec2f(-sa, ca), vec2f(cb, sb), vec2f(-sb, cb));
  let dx = B.x - A.x; let dz = B.z - A.z;
  var best = 1e30; var bn = vec2f(0.0); var fromA = true;
  for (var k = 0u; k < 4u; k = k + 1u) {
    let u = ax[k];
    let d = dx * u.x + dz * u.y;
    let o = radiusOn(A, u.x, u.y) + radiusOn(B, u.x, u.y) - abs(d);
    if (o <= 0.0) { return out; }
    if (o < best) { best = o; let sg = select(1.0, -1.0, d < 0.0); bn = u * sg; fromA = k < 2u; }
  }
  var inc = B; if (!fromA) { inc = A; }
  let ic = cos(inc.yaw); let is_ = sin(inc.yaw);
  let sgn = select(1.0, -1.0, fromA);
  var bestP = 1e30; var p = vec2f(0.0); var secondP = 1e30; var q = vec2f(0.0);
  for (var sx = -1.0; sx <= 1.0; sx = sx + 2.0) {
    for (var sz = -1.0; sz <= 1.0; sz = sz + 2.0) {
      let vx = inc.x + ic * sx * inc.hl - is_ * sz * inc.hw; let vz = inc.z + is_ * sx * inc.hl + ic * sz * inc.hw;
      let pp = sgn * (vx * bn.x + vz * bn.y) * -1.0;
      if (pp < bestP) { secondP = bestP; q = p; bestP = pp; p = vec2f(vx, vz); }
      else if (pp < secondP) { secondP = pp; q = vec2f(vx, vz); }
    }
  }
  if (secondP - bestP < 0.05) { p = 0.5 * (p + q); }
  out.depth = best; out.nx = bn.x; out.nz = bn.y; out.px = p.x; out.pz = p.y; out.ok = true;
  return out;
}
fn lerpAngle(a: f32, b: f32, t: f32) -> f32 { return a + wrapAngle(b - a) * t; }
// collide.js resolveContact (A, B updated in place through pointers)
fn resolveContact(A: ptr<function, Body>, B: ptr<function, Body>, c: Contact, e: f32, mu: f32) -> Res {
  var r: Res; r.jn = 0.0; r.dvA = 0.0; r.dvB = 0.0;
  let rax = c.px - (*A).x; let raz = c.pz - (*A).z; let rbx = c.px - (*B).x; let rbz = c.pz - (*B).z;
  let vax = (*A).vx - (*A).w * raz; let vaz = (*A).vz + (*A).w * rax;
  let vbx = (*B).vx - (*B).w * rbz; let vbz = (*B).vz + (*B).w * rbx;
  let rvx = vbx - vax; let rvz = vbz - vaz;
  let vn = rvx * c.nx + rvz * c.nz;
  r.closing = -vn;
  if (vn >= 0.0) { return r; }
  let raN = rax * c.nz - raz * c.nx; let rbN = rbx * c.nz - rbz * c.nx;
  let kN = (*A).invMass + (*B).invMass + raN * raN * (*A).invInertia + rbN * rbN * (*B).invInertia;
  if (kN <= 0.0) { return r; }
  let jn = -(1.0 + e) * vn / kN;
  var tx = rvx - vn * c.nx; var tz = rvz - vn * c.nz;
  let tl = sqrt(tx * tx + tz * tz);
  var jt = 0.0;
  if (tl > 1e-6) {
    tx = tx / tl; tz = tz / tl;
    let raT = rax * tz - raz * tx; let rbT = rbx * tz - rbz * tx;
    let kT = (*A).invMass + (*B).invMass + raT * raT * (*A).invInertia + rbT * rbT * (*B).invInertia;
    jt = clamp(-(rvx * tx + rvz * tz) / kT, -mu * jn, mu * jn);
  } else { tx = 0.0; tz = 0.0; }
  let Px = jn * c.nx + jt * tx; let Pz = jn * c.nz + jt * tz;
  let ax0 = (*A).vx; let az0 = (*A).vz; let bx0 = (*B).vx; let bz0 = (*B).vz;
  (*A).vx = (*A).vx - Px * (*A).invMass; (*A).vz = (*A).vz - Pz * (*A).invMass; (*A).w = (*A).w - (rax * Pz - raz * Px) * (*A).invInertia;
  (*B).vx = (*B).vx + Px * (*B).invMass; (*B).vz = (*B).vz + Pz * (*B).invMass; (*B).w = (*B).w + (rbx * Pz - rbz * Px) * (*B).invInertia;
  r.jn = jn;
  r.dvA = length(vec2f((*A).vx - ax0, (*A).vz - az0)); r.dvB = length(vec2f((*B).vx - bx0, (*B).vz - bz0));
  return r;
}
