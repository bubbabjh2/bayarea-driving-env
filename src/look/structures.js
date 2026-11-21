// Bridge structures of the landscape layer (render only): concrete deck girders + piers under every elevated road,
// and a clay Golden Gate Bridge (towers with portal struts, stiffening truss, main cables, suspenders, pedestals,
// anchorages) aligned to the road deck. Data: data/landscape/structures.json (tools/landscape/build_structures.mjs).
// Thin cables are camera-facing ribbons with a minimum pixel width and alpha fade (no shimmer at any distance);
// piers / girders dissolve per instance far away (no sub-pixel clutter in the zoom-out).
import * as THREE from 'three';
import { GLSL_COMMON } from './materials.js';
import { GLSL_TERRAIN } from './terrain-glsl.js';

const HEAD_V = '#include <common>\n#include <fog_pars_vertex>\n#include <logdepthbuf_pars_vertex>\n';
const HEAD_F = '#include <common>\n#include <fog_pars_fragment>\n#include <logdepthbuf_pars_fragment>\n';
const TAIL_F = '\n  #include <tonemapping_fragment>\n  #include <colorspace_fragment>\n  #include <fog_fragment>\n';
const BAYER = /* glsl */`
float st_bayer(vec2 p) { ivec2 q = ivec2(mod(p, 4.0)); int i = q.x + q.y * 4;
  int b[16] = int[16](0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5); return (float(b[i]) + 0.5) / 16.0; }`;
const SHADE = /* glsl */`
vec3 st_shade(vec3 alb, vec3 n) {
  vec3 sunH = normalize(vec3(uSunDir.x, max(uSunDir.y, 0.2), uSunDir.z));
  float ndl = clamp((dot(n, sunH) + 0.15) / 1.15, 0.0, 1.0);
  return alb * (uSunIrr * ndl * 0.95 + mix(uGroundIrr, uSkyIrr, 0.5 + 0.5 * n.y) * 0.8);
}`;

// ---- instanced boxes: kind 0 = pier column / cap (x, z, yBottomMode, yTop | w, d, heading, hash), kind 1 = girder (a -> b)
function boxMaterial(U, L, P, color, mode) {
  const vs = HEAD_V + GLSL_TERRAIN + /* glsl */`
attribute vec4 iA; attribute vec4 iB;
uniform float uLsOn, uFadeNear, uFadeFar, uBFade;
varying vec3 vW; varying vec3 vRel; varying float vA;
void main() {
  vec3 p = position;   // unit box: x, z in [-0.5, 0.5], y in [0, 1]
  vec3 w; float hsh;
#if ${mode} == 0
  // column: iA = (x, z, yTop, height-below-top or -1 = to the ground), iB = (sizeX, sizeZ, heading, hash)
  float c = cos(iB.z), s = sin(iB.z);
  vec2 xz = vec2(p.x * iB.x, p.z * iB.y);
  xz = vec2(c * xz.x - s * xz.y, s * xz.x + c * xz.y) + iA.xy;
  float g = uLsOn > 0.5 ? ls_heightRaw(iA.xy) * uLsExag + uLsFlatY - 2.0 : -3.0;
  float bot = iA.w < 0.0 ? g : iA.z - iA.w;
  w = vec3(xz.x, mix(bot, iA.z, p.y), xz.y); hsh = iB.w;
#else
  // girder: iA = (ax, az, ay, width), iB = (bx, bz, by, hash); box top 0.3 m under the deck, 1.5 m deep
  vec2 d = iB.xy - iA.xy; float L = length(d); vec2 u = d / max(L, 1e-3), r = vec2(-u.y, u.x);
  float t = p.x + 0.5;
  vec2 xz = mix(iA.xy, iB.xy, t) + r * p.z * iA.w;
  float y = mix(iA.z, iB.z, t) - 0.3 - (1.0 - p.y) * 1.5;
  w = vec3(xz.x, y, xz.y); hsh = iB.w;
#endif
  float dist = length(w - cameraPosition);
  float a = (1.0 - smoothstep(uFadeNear, uFadeFar, dist)) * uBFade;
  vA = clamp((a - hsh * 0.9) * 8.0, 0.0, 1.0);
  vW = w; vRel = w - cameraPosition;
  vec4 mvPosition = viewMatrix * vec4(w, 1.0);
  gl_Position = vA <= 0.0 ? vec4(2.0, 2.0, 2.0, 1.0) : projectionMatrix * mvPosition;
  #include <logdepthbuf_vertex>
  #include <fog_vertex>
}`;
  const fs = HEAD_F + GLSL_COMMON + BAYER + SHADE + /* glsl */`
uniform vec3 uCol;
varying vec3 vW; varying vec3 vRel; varying float vA;
void main() {
  if (vA < 1.0 && vA <= st_bayer(gl_FragCoord.xy)) discard;
  #include <logdepthbuf_fragment>
  vec3 cr = cross(dFdx(vW), dFdy(vW)); vec3 n = dot(cr, cr) > 1e-12 ? normalize(cr) : vec3(0.0, 1.0, 0.0); if (dot(n, vRel) > 0.0) n = -n;
  gl_FragColor = vec4(st_shade(uCol, n), 1.0);
` + TAIL_F + '}';
  const m = new THREE.ShaderMaterial({ uniforms: Object.assign(THREE.UniformsUtils.clone(THREE.UniformsLib.fog), U, L, P, { uCol: { value: new THREE.Color(color) } }),
    vertexShader: vs, fragmentShader: fs, fog: true });
  m.extensions = { derivatives: true };
  return m;
}
function unitBox() {
  const g = new THREE.BoxGeometry(1, 1, 1).toNonIndexed(); g.translate(0, 0.5, 0);
  const ig = new THREE.InstancedBufferGeometry(); ig.setAttribute('position', g.attributes.position); return ig;
}
function instanced(U, L, P, color, mode, A, B, name) {
  const g = unitBox();
  g.setAttribute('iA', new THREE.InstancedBufferAttribute(A, 4)); g.setAttribute('iB', new THREE.InstancedBufferAttribute(B, 4));
  g.instanceCount = A.length / 4;
  const m = new THREE.Mesh(g, boxMaterial(U, L, P, color, mode)); m.frustumCulled = false; m.name = name;
  return m;
}

// ---- cables: camera-facing ribbons with a minimum pixel width (alpha-faded when thinner than that)
function cableMaterial(U, L, P, color) {
  const vs = HEAD_V + /* glsl */`
attribute vec2 quad; attribute vec4 cA; attribute vec4 cB;   // (x, y, z, radius)
uniform float uViewportH, uMinPx;
varying float vSide; varying float vAlpha; varying vec3 vW;
void main() {
  vec3 P0 = mix(cA.xyz, cB.xyz, quad.x);
  vec3 ax = cB.xyz - cA.xyz;
  vec3 side = normalize(cross(ax, P0 - cameraPosition) + vec3(1e-6));
  vec4 mvC = viewMatrix * vec4(P0, 1.0);
  float px = 2.0 * max(-mvC.z, 0.1) / (projectionMatrix[1][1] * uViewportH);
  float r = cA.w, rw = max(r, uMinPx * px * 0.5);
  vAlpha = r / rw; vSide = quad.y;
  vec3 w = P0 + side * quad.y * rw;
  vW = w;
  vec4 mvPosition = viewMatrix * vec4(w, 1.0);
  gl_Position = projectionMatrix * mvPosition;
  #include <logdepthbuf_vertex>
  #include <fog_vertex>
}`;
  const fs = HEAD_F + GLSL_COMMON + /* glsl */`
uniform vec3 uCol; uniform float uBFade;
varying float vSide; varying float vAlpha; varying vec3 vW;
void main() {
  float aa = clamp((1.0 - abs(vSide)) / max(fwidth(vSide), 1e-4), 0.0, 1.0);
  float a = aa * vAlpha * uBFade;
  if (a < 0.01) discard;
  #include <logdepthbuf_fragment>
  vec3 n = vec3(0.0, 1.0, 0.0);
  gl_FragColor = vec4(uCol * (uSunIrr * 0.45 + uSkyIrr * 0.55), a);
` + TAIL_F + '}';
  const m = new THREE.ShaderMaterial({ uniforms: Object.assign(THREE.UniformsUtils.clone(THREE.UniformsLib.fog), U, P, { uCol: { value: new THREE.Color(color) }, uMinPx: { value: 1.1 } }),
    vertexShader: vs, fragmentShader: fs, fog: true, transparent: false, depthWrite: false, side: THREE.DoubleSide,
    blending: THREE.CustomBlending, blendSrc: THREE.SrcAlphaFactor, blendDst: THREE.OneMinusSrcAlphaFactor });
  m.extensions = { derivatives: true };
  return m;
}
function cables(U, L, P, color, segs, name, minPx) {
  const g = new THREE.InstancedBufferGeometry();
  g.setAttribute('quad', new THREE.Float32BufferAttribute([0, -1, 1, -1, 0, 1, 1, 1], 2)); g.setIndex([0, 2, 1, 1, 2, 3]);
  const A = new Float32Array(segs.length * 4), B = new Float32Array(segs.length * 4);
  segs.forEach(([a, b, r], i) => { A.set([a[0], a[1], a[2], r], i * 4); B.set([b[0], b[1], b[2], r], i * 4); });
  g.setAttribute('cA', new THREE.InstancedBufferAttribute(A, 4)); g.setAttribute('cB', new THREE.InstancedBufferAttribute(B, 4));
  g.instanceCount = segs.length;
  const mat = cableMaterial(U, L, P, color); mat.uniforms.uMinPx.value = minPx;
  const m = new THREE.Mesh(g, mat); m.frustumCulled = false; m.name = name; m.renderOrder = 2;
  return m;
}

// ---- solid boxes in world space (Golden Gate parts), merged into one geometry per colour
function addBox(arr, c, ax, ay, sx, sy, sz, rot, y0, y1) {
  // centre c = [x, z], half sizes along the deck axis (sx) and across (sz), from y0 to y1; rot = deck direction [ux, uz]
  const [ux, uz] = rot, rx = -uz, rz = ux;
  const P = (i, j, k) => [c[0] + ux * sx * i + rx * sz * j, k ? y1 : y0, c[1] + uz * sx * i + rz * sz * j];
  const v = [P(-1, -1, 0), P(1, -1, 0), P(1, 1, 0), P(-1, 1, 0), P(-1, -1, 1), P(1, -1, 1), P(1, 1, 1), P(-1, 1, 1)];
  for (const [a, b, cc, d] of [[0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1], [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0]]) arr.push(...v[a], ...v[b], ...v[cc], ...v[a], ...v[cc], ...v[d]);
}
function solidMaterial(U, P, color) {
  const vs = HEAD_V + 'varying vec3 vW; varying vec3 vRel;\nvoid main() { vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; vRel = w.xyz - cameraPosition;\n vec4 mvPosition = viewMatrix * w; gl_Position = projectionMatrix * mvPosition;\n #include <logdepthbuf_vertex>\n #include <fog_vertex>\n}';
  const fs = HEAD_F + GLSL_COMMON + SHADE + /* glsl */`
uniform vec3 uCol; uniform float uBFade;
varying vec3 vW; varying vec3 vRel;
void main() {
  if (uBFade < 0.999 && uBFade <= fract(sin(dot(floor(gl_FragCoord.xy), vec2(12.9898, 78.233))) * 43758.5453)) discard;
  #include <logdepthbuf_fragment>
  vec3 cr = cross(dFdx(vW), dFdy(vW)); vec3 n = dot(cr, cr) > 1e-12 ? normalize(cr) : vec3(0.0, 1.0, 0.0); if (dot(n, vRel) > 0.0) n = -n;
  gl_FragColor = vec4(st_shade(uCol, n), 1.0);
` + TAIL_F + '}';
  const m = new THREE.ShaderMaterial({ uniforms: Object.assign(THREE.UniformsUtils.clone(THREE.UniformsLib.fog), U, P, { uCol: { value: new THREE.Color(color) } }), vertexShader: vs, fragmentShader: fs, fog: true, side: THREE.DoubleSide });
  m.extensions = { derivatives: true };
  return m;
}

function goldenGate(U, P, gg, group) {
  const [ax, az] = gg.a, [bx, bz] = gg.b, L = Math.hypot(bx - ax, bz - az), u = [(bx - ax) / L, (bz - az) / L], r = [-u[1], u[0]];
  const at = (s, t = 0) => [ax + u[0] * s + r[0] * t, az + u[1] * s + r[1] * t];
  const deckY = (s) => {
    const d = gg.deck; if (!d.length) return 20;
    if (s <= d[0][0]) return d[0][1]; if (s >= d[d.length - 1][0]) return d[d.length - 1][1];
    let i = 0; while (d[i + 1][0] < s) i++;
    const f = (s - d[i][0]) / (d[i + 1][0] - d[i][0]); return d[i][1] + (d[i + 1][1] - d[i][1]) * f;
  };
  const [t0, t1] = gg.towers, [a0, a1] = gg.anchors, mid = (t0 + t1) / 2;
  const off = deckY(mid) - gg.deckRealM;   // real heights above water -> world y
  const Y = (h) => off + h;
  const orange = [], grey = [];
  const legT = gg.width / 2 + 3.2;
  for (const ts of [t0, t1]) {
    // pedestal / pier (concrete), tower legs in three set-back stages, portal struts
    addBox(grey, at(ts), 0, 0, 24, 1, legT + 12, u, Y(-8), Y(14));
    for (const side of [-1, 1]) {
      const c = at(ts, side * legT);
      addBox(orange, c, 0, 0, 5.4, 1, 5.0, u, Y(14), Y(100));
      addBox(orange, c, 0, 0, 4.6, 1, 4.3, u, Y(100), Y(160));
      addBox(orange, c, 0, 0, 3.9, 1, 3.6, u, Y(160), Y(213));
      addBox(orange, c, 0, 0, 3.3, 1, 3.1, u, Y(213), Y(227));
    }
    for (const [h0, h1] of [[56, 63], [104, 113], [150, 158], [190, 197], [217, 227]]) addBox(orange, at(ts), 0, 0, 3.0, 1, legT - 2, u, Y(h0), Y(h1));
  }
  // anchorages + pylons at the span ends (concrete)
  for (const as of [a0, a1]) addBox(grey, at(as), 0, 0, 22, 1, legT + 6, u, Y(-6), deckY(as) - 0.8);
  // stiffening truss under the deck between the anchorages: segmented to follow the deck profile
  for (let s = a0; s < a1; s += 25) {
    const s1 = Math.min(a1, s + 25), sm = (s + s1) / 2, y = deckY(sm);
    addBox(orange, at(sm), 0, 0, (s1 - s) / 2 + 0.05, 1, gg.width / 2, u, y - 7.6, y - 0.35);
  }
  const mk = (arr, col, name) => {
    const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.Float32BufferAttribute(arr, 3));
    g.computeBoundingSphere();
    const m = new THREE.Mesh(g, solidMaterial(U, P, col)); m.name = name; group.add(m); return m;
  };
  mk(orange, 0xc0472f, 'look-goldengate-steel');
  mk(grey, 0xb8b2a8, 'look-goldengate-concrete');
  // main cables: parabolic main span, side spans down to the anchorages; suspenders every 15.2 m
  const cableY = (s) => {
    const sad = Y(gg.saddleRealM), low = Y(gg.cableLowRealM);
    if (s >= t0 && s <= t1) { const x = (s - mid) / ((t1 - t0) / 2); return low + (sad - low) * x * x; }
    const [ts, as] = s < t0 ? [t0, a0] : [t1, a1];
    const f = (s - ts) / (as - ts), end = deckY(as) + 3;
    return sad + (end - sad) * f - 18 * 4 * f * (1 - f) * 0;   // side spans: nearly straight backstays
  };
  const segs = [], hangers = [];
  for (const side of [-1, 1]) {
    const t = side * gg.cableGap / 2;
    let prev = null;
    for (let s = a0; s <= a1 + 0.1; s += 8) {
      const p = at(s, t), q = [p[0], cableY(s), p[1]];
      if (prev) segs.push([prev, q, 0.55]);
      prev = q;
    }
    for (let s = a0 + 7.6; s < a1; s += 15.24) {
      if (Math.abs(s - t0) < 6 || Math.abs(s - t1) < 6) continue;
      const p = at(s, t), cy = cableY(s), dy = deckY(s) + 0.6;
      if (cy - dy > 1) hangers.push([[p[0], dy, p[1]], [p[0], cy, p[1]], 0.16]);
    }
  }
  group.add(cables(U, null, P, 0xa8392a, segs, 'look-goldengate-cables', 1.2));
  group.add(cables(U, null, P, 0xa8392a, hangers, 'look-goldengate-suspenders', 0.7));
}

export function createStructures({ U, L, renderer }) {
  const group = new THREE.Group(); group.name = 'look-structures';
  const P = { uFadeNear: { value: 4500 }, uFadeFar: { value: 7000 }, uBFade: { value: 1 }, uViewportH: U.uViewportH };
  let state = 'loading';
  const ready = fetch('data/landscape/structures.json').then((r) => (r.ok ? r.json() : null)).then((d) => {
    if (!d) { state = 'none'; return; }
    // piers: one column per pier (two for wide decks) + a cap beam
    const cols = [], capsA = [], capsB = [];
    const h = (i) => ((Math.imul(i + 1, 2654435761) >>> 0) / 4294967296);
    d.piers.forEach(([x, z, yTop, w, hd], i) => {
      const c = Math.cos(hd), s = Math.sin(hd);   // heading: along-deck direction (x = sin, z = -cos); across = (cos, sin)
      const across = [Math.cos(hd), Math.sin(hd)];
      const off = w > 12 ? [-(w / 2 - 2.2), w / 2 - 2.2] : [0];
      for (const o of off) cols.push(x + across[0] * o, z + across[1] * o, yTop - 1.2, -1, 1.9, 1.9, hd, h(i));
      capsA.push(x, z, yTop, 1.3); capsB.push(Math.max(2.5, w - 0.6), 2.0, hd, h(i));
      void c; void s;
    });
    const colA = [], colB = [];
    for (let i = 0; i < cols.length; i += 8) { colA.push(cols[i], cols[i + 1], cols[i + 2], cols[i + 3]); colB.push(cols[i + 4], cols[i + 5], cols[i + 6], cols[i + 7]); }
    if (colA.length) group.add(instanced(U, L, P, 0xb5b0a6, 0, new Float32Array(colA), new Float32Array(colB), 'look-piers'));
    if (capsA.length) group.add(instanced(U, L, P, 0xb5b0a6, 0, new Float32Array(capsA), new Float32Array(capsB), 'look-pier-caps'));
    const gA = [], gB = [];
    d.girders.forEach(([ax, az, ay, bx, bz, by, w], i) => { gA.push(ax, az, ay, w); gB.push(bx, bz, by, h(i + 7)); });
    if (gA.length) group.add(instanced(U, L, P, 0xaea99f, 1, new Float32Array(gA), new Float32Array(gB), 'look-girders'));
    if (d.goldengate) goldenGate(U, P, d.goldengate, group);
    state = 'ready';
  }).catch((e) => { state = 'none'; console.warn('landscape structures', e.message); });
  return {
    group, ready, get state() { return state; },
    setFade(k) { P.uBFade.value = k; },
    dispose() { group.traverse((o) => { if (o.geometry) o.geometry.dispose(); if (o.material) o.material.dispose(); }); },
  };
}
