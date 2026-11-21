// Look-pack materials. Every factory takes the shared uniform block `U` (sun,
// sky, camera helpers; updated once per frame by the sky dome) so lighting is
// consistent across custom ShaderMaterials and the stock three.js materials
// used for cars and stations (their HemisphereLight/DirectionalLight get the
// same values * PI, see index.js).
//
// All ShaderMaterials include three's fog, log-depth, tone-mapping and
// colour-space chunks, so they work with plain depth and with
// logarithmicDepthBuffer: true.
import * as THREE from 'three';

// ------------------------------------------------------------------ GLSL
export const GLSL_COMMON = /* glsl */`
uniform vec3 uSunDir;
uniform vec3 uSunIrr;
uniform vec3 uSkyIrr;
uniform vec3 uGroundIrr;
uniform vec2 uCamMod;      // camera.xz mod 512 (computed in double precision on the CPU)
uniform float uTime;
float lk_hash(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
float lk_noise(vec2 p) {
  vec2 i = floor(p), f = fract(p), u = f * f * (3.0 - 2.0 * f);
  float a = lk_hash(i), b = lk_hash(i + vec2(1.0, 0.0)), c = lk_hash(i + vec2(0.0, 1.0)), d = lk_hash(i + vec2(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
// band-limited fbm: octaves finer than the pixel footprint fade to their mean (no shimmer)
float lk_fbm(vec2 p, float fw, int oct) {
  float s = 0.0, a = 0.5, n = 0.0;
  for (int i = 0; i < 6; i++) {
    if (i >= oct) break;
    float fade = 1.0 - smoothstep(0.2, 0.6, fw);
    s += a * (lk_noise(p) - 0.5) * fade; n += a;
    p = p * 2.03 + vec2(17.1, 9.2); fw *= 2.03; a *= 0.5;
  }
  return s / n;
}
vec3 lk_light(vec3 albedo, vec3 n) {
  float nl = max(dot(n, uSunDir), 0.0);
  return albedo * (uSunIrr * nl + mix(uGroundIrr, uSkyIrr, 0.5 + 0.5 * n.y));
}
`;

const VERT_HEAD = /* glsl */`
#include <common>
#include <fog_pars_vertex>
#include <logdepthbuf_pars_vertex>
`;
const VERT_TAIL = /* glsl */`
  #include <logdepthbuf_vertex>
  #include <fog_vertex>
`;
const FRAG_HEAD = /* glsl */`
#include <common>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
`;
const FRAG_TAIL = /* glsl */`
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
  #include <fog_fragment>
`;

// clone() keeps the uniform objects shared (live sun/sky/camera values, no texture copies); only
// the uniforms map itself is new, so e.g. the stream's per-clone fade uniform stays per clone
// Only the look's live uniforms (the shared U block) and texture uniforms are shared; anything
// else (e.g. src/stream's per-clone uStreamFade) is deep-copied so a clone never drives its base.
function shareOnClone(m, U) {
  const shared = new Set(Object.values(U));
  m.clone = function () {
    const c = THREE.ShaderMaterial.prototype.clone.call(this);   // deep-copied uniforms
    for (const k of Object.keys(this.uniforms)) {
      const u = this.uniforms[k];
      if (shared.has(u) || (u && u.value && u.value.isTexture)) c.uniforms[k] = u;
    }
    shareOnClone(c, U);
    return c;
  };
  return m;
}
function shaderMat(U, extraUniforms, vertexShader, fragmentShader, params = {}) {
  return shareOnClone(new THREE.ShaderMaterial({
    // shared uniform objects (U) are referenced, not cloned, so per-frame updates reach every material
    uniforms: Object.assign(THREE.UniformsUtils.clone(THREE.UniformsLib.fog), U, extraUniforms),
    vertexShader, fragmentShader, fog: true, ...params,
  }), U);
}

// "opaque pass but alpha-blended": transparent:false keeps the object in the
// opaque list (ordered by renderOrder, before cars/markings), CustomBlending
// still blends its anti-aliased edges.
const BLEND_IN_OPAQUE = {
  transparent: false, blending: THREE.CustomBlending,
  blendSrc: THREE.SrcAlphaFactor, blendDst: THREE.OneMinusSrcAlphaFactor,
  blendSrcAlpha: THREE.OneFactor, blendDstAlpha: THREE.OneMinusSrcAlphaFactor,
};

// ------------------------------------------------------------------ pavement surfaces
// Surface group (stream tiles: bits 5-7 of the class byte, from data/surfaces.bin; docs/SURFACES.md):
// 0 asphalt (albedo unchanged), 1 concrete, 2 pavers (paving_stones / sett), 3 gravel / compacted, 4 dirt,
// 5 steel grate, 6 wood, 7 grass / sand. st = (metres along, metres across) when hasSt > 0.5, else world xz.
const SURFACE_FN = /* glsl */`
float lk_jointAA(float x, float period, float halfW, float fw) {
  float f = fract(x / period), d = min(f, 1.0 - f) * period;
  return 1.0 - smoothstep(halfW, halfW + fw, d);
}
vec3 lk_surface(vec3 alb, float surf, vec2 w, float fw, vec3 det, vec2 st, float hasSt) {
  if (surf < 0.5) return alb;
  vec2 q = hasSt > 0.5 ? st : w;
  vec2 qfw = max(fwidth(q), vec2(1e-4));
  float fine = 1.0 - smoothstep(0.04, 0.25, fw);          // fades small-scale detail before it aliases
  float n1 = lk_fbm(w / 6.0 + 7.0, fw / 6.0, 2), n2 = lk_noise(w * 6.0);
  vec3 c;
  if (surf < 1.5) {
    // concrete: pale grey slabs, transverse joints every 4.8 m, longitudinal joints every lane (3.7 m)
    c = vec3(0.30, 0.297, 0.285) * mix(vec3(1.0), det, 0.3) * (1.0 + n1 * 0.10);
    vec2 sid = floor(q / vec2(4.8, 3.7));
    c *= 1.0 + (lk_hash(sid + 3.0) - 0.5) * 0.06;
    float j = max(lk_jointAA(q.x, 4.8, 0.012, 1.5 * qfw.x), hasSt > 0.5 ? lk_jointAA(q.y, 3.7, 0.012, 1.5 * qfw.y) : 0.0);
    c *= 1.0 - 0.4 * j * (1.0 - smoothstep(0.05, 0.25, max(qfw.x, qfw.y)));
  } else if (surf < 2.5) {
    // pavers / sett: running-bond blocks 0.2 x 0.1 m, per-block tone, sand joints
    vec2 b = q / vec2(0.2, 0.1);
    b.x += 0.5 * mod(floor(b.y), 2.0);
    vec2 bid = floor(b), bf = abs(fract(b) - 0.5), bfw = fwidth(b);
    c = vec3(0.27, 0.215, 0.18) * (1.0 + (lk_hash(bid) - 0.5) * 0.25 * fine) * (1.0 + n1 * 0.12);
    float j = max(1.0 - smoothstep(0.08, 0.08 + bfw.x, 0.5 - bf.x), 1.0 - smoothstep(0.08, 0.08 + bfw.y, 0.5 - bf.y));
    c = mix(c, vec3(0.30, 0.28, 0.24), j * 0.6 * (1.0 - smoothstep(0.2, 0.6, max(bfw.x, bfw.y))));
  } else if (surf < 3.5) {
    // gravel / compacted: light grey-brown, speckled
    c = vec3(0.34, 0.31, 0.27) * (1.0 + n1 * 0.15) * (1.0 + (n2 - 0.5) * 0.35 * fine);
  } else if (surf < 4.5) {
    // dirt
    c = vec3(0.31, 0.24, 0.17) * (1.0 + n1 * 0.2) * (1.0 + (n2 - 0.5) * 0.2 * fine);
  } else if (surf < 5.5) {
    // steel grate deck: dark steel, 0.12 m grid of bars over open cells
    vec2 g = q / 0.12, gf = abs(fract(g) - 0.5), gfw = fwidth(g);
    float bar = max(1.0 - smoothstep(0.32, 0.32 + gfw.x, gf.x), 1.0 - smoothstep(0.32, 0.32 + gfw.y, gf.y));
    float grid = mix(0.45, bar, 1.0 - smoothstep(0.15, 0.5, max(gfw.x, gfw.y)));
    c = mix(vec3(0.03, 0.032, 0.035), vec3(0.20, 0.205, 0.21) * (1.0 + n1 * 0.1), grid);
  } else if (surf < 6.5) {
    // wood planks across the deck, 0.25 m wide
    float pid = floor(q.x / 0.25);
    c = vec3(0.27, 0.19, 0.12) * (1.0 + (lk_hash(vec2(pid, 1.0)) - 0.5) * 0.3 * fine) * (1.0 + n1 * 0.1);
    c *= 1.0 - 0.5 * lk_jointAA(q.x, 0.25, 0.008, 1.5 * qfw.x) * (1.0 - smoothstep(0.03, 0.1, qfw.x));
  } else {
    // grass / sand track
    c = vec3(0.16, 0.23, 0.09) * (1.0 + n1 * 0.25) * (1.0 + (n2 - 0.5) * 0.3 * fine);
  }
  return c;
}
`;

// ------------------------------------------------------------------ asphalt ribbons
// lookClass: 0..12 = CLASSES_ALL index, +16 = tunnel, +32 * surface group. Set by index.js (from a
// `classId` attribute when the world provides one, else matched from vertex colours).
const ASPHALT_FN = SURFACE_FN + /* glsl */`
uniform sampler2D uAsphalt;
uniform float uAsphaltMean;
uniform float uTone[13];
uniform vec3 uConcrete;
uniform float uBridgeMix;
vec3 lk_asphalt(vec3 wpos, vec3 rel, float clsF, vec2 st, float hasSt, out float elev) {
  float cls = floor(clsF + 0.5);
  float surf = floor(cls / 32.0);                         // pavement surface group (bits 5-7), 0 = asphalt
  cls = mod(cls, 32.0);
  float tunnel = step(15.5, cls);
  cls = mod(cls, 16.0);
  float tone = uTone[int(clamp(cls, 0.0, 12.0))];
  vec2 pf = rel.xz + uCamMod;
  vec3 t1 = texture2D(uAsphalt, pf / 8.0).rgb;
  vec3 t2 = texture2D(uAsphalt, vec2(pf.y, -pf.x) / 32.0).rgb;
  vec3 det = (t1 * 0.7 + t2 * 0.3) / uAsphaltMean;
  vec2 w = wpos.xz;
  float fw = length(fwidth(w));
  float macro = lk_fbm(w / 45.0, fw / 45.0, 3);
  float patchN = lk_fbm(w / 22.0 + 31.0, fw / 22.0, 2);
  float patchM = smoothstep(0.17, 0.2, patchN) * (1.0 - smoothstep(0.3, 0.8, fw / 22.0));
  vec3 alb = vec3(tone) * vec3(1.0, 0.99, 0.98) * det * (1.0 + macro * 0.45);
  alb = mix(alb, alb * 0.72, patchM * (1.0 - step(0.5, surf)) * 0.7);   // fresh-asphalt repair patches
  alb = lk_surface(alb, surf, w, fw, det, st, hasSt);
  elev = smoothstep(5.0, 11.0, wpos.y) * uBridgeMix;      // bridges / flyovers: concrete deck
  alb = mix(alb, uConcrete * mix(vec3(1.0), det, 0.45) * (1.0 + macro * 0.3), elev);
  alb *= mix(1.0, 0.4, tunnel);
  return alb;
}
`;

export function makeRoadMaterial(U, tex) {
  const vs = VERT_HEAD + /* glsl */`
attribute float lookClass;
#ifdef LOOK_LANECOORD
attribute vec2 laneCoord;   // (metres along the edge, metres across from the left pavement edge)
varying vec2 vLane;
#endif
varying vec3 vW; varying vec3 vRel; varying float vCls;
void main() {
  vec4 w = modelMatrix * vec4(position, 1.0);
  vW = w.xyz; vRel = w.xyz - cameraPosition; vCls = lookClass;
#ifdef LOOK_LANECOORD
  vLane = laneCoord;
#endif
  vec4 mvPosition = viewMatrix * w;
  gl_Position = projectionMatrix * mvPosition;
` + VERT_TAIL + '}';
  const fs = FRAG_HEAD + GLSL_COMMON + ASPHALT_FN + /* glsl */`
varying vec3 vW; varying vec3 vRel; varying float vCls;
#ifdef LOOK_LANECOORD
varying vec2 vLane;
#endif
void main() {
  #include <logdepthbuf_fragment>
  float elev;
  vec2 st = vec2(0.0); float hasSt = 0.0;
#ifdef LOOK_LANECOORD
#ifdef LOOK_LANE_ACROSS_X
  st = vec2(vLane.y, vLane.x); hasSt = 1.0;
#else
  st = vLane; hasSt = 1.0;
#endif
#endif
  vec3 alb = lk_asphalt(vW, vRel, vCls, st, hasSt, elev);
#ifdef LOOK_LANECOORD
#ifdef LOOK_LANE_ACROSS_X
  float f = fract(vLane.x / 3.7);   // stream tiles: laneCoord = (across, along)
#else
  float f = fract(vLane.y / 3.7);   // src/roads.js: laneCoord = (metres along, metres across from the left pavement edge)
#endif
  float wheel = exp(-pow((f - 0.27) / 0.07, 2.0)) + exp(-pow((f - 0.73) / 0.07, 2.0));
  float drip = exp(-pow((f - 0.5) / 0.05, 2.0));
  alb *= 1.0 - 0.13 * wheel * (1.0 - elev) - 0.06 * drip;
#endif
  if (!gl_FrontFacing) alb = uConcrete * 0.75;           // deck underside (soffit), seen from a lower deck
  vec3 col = lk_light(alb, vec3(0.0, 1.0, 0.0));
  // low-sun sheen on worn asphalt (cheap Blinn lobe, no env map)
  vec3 V = normalize(-vRel);
  vec3 H = normalize(V + uSunDir);
  col += uSunIrr * pow(max(H.y, 0.0), 80.0) * 0.05;
  gl_FragColor = vec4(col, 1.0);
` + FRAG_TAIL + '}';
  return shaderMat(U, {
    uAsphalt: { value: tex.asphalt }, uAsphaltMean: { value: tex.asphaltMean },
    uTone: { value: tex.tones }, uConcrete: { value: new THREE.Color(0.20, 0.195, 0.185) }, uBridgeMix: { value: 1 },
  }, vs, fs, { side: THREE.DoubleSide });
}

// ------------------------------------------------------------------ 1-px road lines (far view)
export function makeRoadLineMaterial(U, { opacity = 0.6, fadeNear = 0, fadeFar = 0, lo = 0.05, hi = 0.30 } = {}) {
  const vs = VERT_HEAD + /* glsl */`
attribute vec3 color;
varying float vLum; varying float vDist;
void main() {
  vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mvPosition;
  vLum = dot(color, vec3(0.3, 0.59, 0.11));
  vDist = -mvPosition.z;
` + VERT_TAIL + '}';
  const fs = FRAG_HEAD + GLSL_COMMON + /* glsl */`
uniform float uOpacity, uFadeNear, uFadeFar, uLo, uHi;
varying float vLum; varying float vDist;
void main() {
  #include <logdepthbuf_fragment>
  float a = uOpacity * (uFadeFar > 0.0 ? smoothstep(uFadeNear, uFadeFar, vDist) : 1.0);
  if (a < 0.01) discard;
  vec3 alb = vec3(mix(uLo, uHi, smoothstep(0.08, 0.7, vLum)));
  gl_FragColor = vec4(lk_light(alb, vec3(0.0, 1.0, 0.0)), a);
` + FRAG_TAIL + '}';
  return shaderMat(U, {
    uOpacity: { value: opacity }, uFadeNear: { value: fadeNear }, uFadeFar: { value: fadeFar }, uLo: { value: lo }, uHi: { value: hi },
  }, vs, fs, { transparent: true, depthWrite: false });
}

// ------------------------------------------------------------------ segment quads (paint, local ribbons)
// One instanced quad per line segment, reading the segment end points straight
// out of the source LineSegments' position array (zero copy). Width is clamped
// to >= ~0.75 px with alpha scaled down accordingly, so thin paint and far
// streets fade out instead of aliasing; edges are anti-aliased with fwidth.
const QUAD_VS = VERT_HEAD + /* glsl */`
attribute vec2 quad;          // x: 0..1 along, y: -1..1 across
attribute vec3 aA; attribute vec3 aB;
#ifdef SEG_PER_INSTANCE
attribute vec2 aWC;           // width (m), class id
#endif
uniform float uWidth, uExtend, uViewportH, uMinPx;
varying float vSide; varying float vAlpha; varying vec3 vW; varying vec3 vRel; varying float vCls;
void main() {
#ifdef SEG_PER_INSTANCE
  float wdt = aWC.x; vCls = aWC.y;
#else
  float wdt = uWidth; vCls = 0.0;
#endif
  vec3 d = aB - aA;
  float L = length(d.xz);
  vec2 dir = L > 1e-4 ? d.xz / L : vec2(1.0, 0.0);
  vec2 perp = vec2(-dir.y, dir.x);
  vec3 P = mix(aA, aB, quad.x);
  vec4 mvC = modelViewMatrix * vec4(P, 1.0);
  float px = 2.0 * max(-mvC.z, 0.1) / (projectionMatrix[1][1] * uViewportH);  // metres per pixel here
  float hw = 0.5 * wdt;
  float hwE = max(hw, uMinPx * px);
  vAlpha = hw / hwE;
  P.xz += perp * quad.y * hwE + dir * (quad.x * 2.0 - 1.0) * hwE * uExtend;
  vec4 w = modelMatrix * vec4(P, 1.0);
  vW = w.xyz; vRel = w.xyz - cameraPosition;
  vSide = quad.y;
  vec4 mvPosition = viewMatrix * w;
  gl_Position = projectionMatrix * mvPosition;
` + VERT_TAIL + '}';

export function makeQuadGeometry(posArray, perInstance = null) {
  const g = new THREE.InstancedBufferGeometry();
  g.setAttribute('quad', new THREE.Float32BufferAttribute([0, -1, 1, -1, 0, 1, 1, 1], 2));
  g.setIndex([0, 2, 1, 1, 2, 3]);   // counter-clockwise seen from above
  const ib = new THREE.InstancedInterleavedBuffer(posArray, 6, 1);
  g.setAttribute('aA', new THREE.InterleavedBufferAttribute(ib, 3, 0));
  g.setAttribute('aB', new THREE.InterleavedBufferAttribute(ib, 3, 3));
  if (perInstance) g.setAttribute('aWC', new THREE.InstancedBufferAttribute(perInstance, 2));
  g.instanceCount = Math.floor(posArray.length / 6);
  return g;
}

export function makePaintMaterial(U, { color, width = 0.15, opacity = 1 }) {
  const fs = FRAG_HEAD + GLSL_COMMON + /* glsl */`
uniform vec3 uPaint; uniform float uOpacity, uGlow;
varying float vSide; varying float vAlpha; varying vec3 vW; varying vec3 vRel; varying float vCls;
void main() {
  #include <logdepthbuf_fragment>
  float aa = clamp((1.0 - abs(vSide)) / max(fwidth(vSide), 1e-4), 0.0, 1.0);
  float a = aa * vAlpha * uOpacity;
  if (a < 0.004) discard;
  // worn paint: slight breakup close up
  vec2 pf = vRel.xz + uCamMod;
  float wear = lk_noise(pf * 1.7) * lk_noise(pf * 0.31 + 7.0);
  vec3 alb = uPaint * (0.86 + 0.14 * smoothstep(0.05, 0.4, wear));
  vec3 col = lk_light(alb, vec3(0.0, 1.0, 0.0)) + alb * uGlow;
  gl_FragColor = vec4(col, a);
` + FRAG_TAIL + '}';
  return shaderMat(U, {
    uPaint: { value: new THREE.Color(color) }, uOpacity: { value: opacity }, uGlow: { value: 0.12 },
    uWidth: { value: width }, uExtend: { value: 0.5 }, uMinPx: { value: 0.6 },
  }, QUAD_VS, fs, { depthWrite: false, ...BLEND_IN_OPAQUE });
}

export function makeLocalRibbonMaterial(U, tex) {
  const fs = FRAG_HEAD + GLSL_COMMON + ASPHALT_FN + /* glsl */`
varying float vSide; varying float vAlpha; varying vec3 vW; varying vec3 vRel; varying float vCls;
void main() {
  #include <logdepthbuf_fragment>
  float aa = clamp((1.0 - abs(vSide)) / max(fwidth(vSide), 1e-4), 0.0, 1.0);
  float a = aa * vAlpha;
  if (a < 0.004) discard;
  float elev;
  vec3 alb = lk_asphalt(vW, vRel, vCls, vec2(0.0), 0.0, elev);
  alb *= 1.0 - 0.18 * smoothstep(0.75, 1.0, abs(vSide));   // gutter / curb darkening
  gl_FragColor = vec4(lk_light(alb, vec3(0.0, 1.0, 0.0)), a);
` + FRAG_TAIL + '}';
  const m = shaderMat(U, {
    uAsphalt: { value: tex.asphalt }, uAsphaltMean: { value: tex.asphaltMean },
    uTone: { value: tex.tones }, uConcrete: { value: new THREE.Color(0.20, 0.195, 0.185) }, uBridgeMix: { value: 1 },
    uWidth: { value: 7 }, uExtend: { value: 1.0 }, uMinPx: { value: 0.5 },
  }, QUAD_VS, fs, { depthWrite: false, ...BLEND_IN_OPAQUE });
  m.defines = { SEG_PER_INSTANCE: '' };
  return m;
}

// ------------------------------------------------------------------ land
export function makeGroundMaterial(U, tex, q) {
  const vs = VERT_HEAD + /* glsl */`
varying vec3 vW; varying vec3 vRel;
void main() {
  vec4 w = modelMatrix * vec4(position, 1.0);
  vW = w.xyz; vRel = w.xyz - cameraPosition;
  vec4 mvPosition = viewMatrix * w;
  gl_Position = projectionMatrix * mvPosition;
` + VERT_TAIL + '}';
  const fs = FRAG_HEAD + GLSL_COMMON + /* glsl */`
uniform sampler2D uGrass; uniform sampler2D uCoastS; uniform sampler2D uUrban;
uniform vec4 uMaskRect;     // minX, minZ, W, H of the coast / urban masks
uniform float uHasUrban, uHasCoast, uDetail;
uniform vec3 uGreenA, uGreenB, uGold, uStraw, uTree, uLawn, uRoof, uRoofDark, uRoofTerra, uRoofLight, uMarsh;
varying vec3 vW; varying vec3 vRel;
void main() {
  #include <logdepthbuf_fragment>
  vec2 w = vW.xz;
  float fw = length(fwidth(w));
  vec2 muv = vec2((w.x - uMaskRect.x) / uMaskRect.z, 1.0 - (w.y - uMaskRect.y) / uMaskRect.w);
  float inRect = step(0.0, muv.x) * step(muv.x, 1.0) * step(0.0, muv.y) * step(muv.y, 1.0);
  muv = clamp(muv, 0.0, 1.0);
  float urban = uHasUrban > 0.5 ? texture2D(uUrban, muv).r * inRect : 0.0;
  float shore = uHasCoast > 0.5 ? texture2D(uCoastS, muv).r * inRect : 0.0;
  float town = smoothstep(0.15, 0.7, urban);
  float region = lk_fbm(w / 12000.0, fw / 12000.0, 3);
  float mid = lk_fbm(w / 1500.0, fw / 1500.0, 3);
  float fine = lk_fbm(w / 160.0, fw / 160.0, LOOK_OCT);
  // hills: October gold grass, greener where the regional noise says wetter (north slopes, creeks)
  float wet = smoothstep(0.02, 0.22, region + mid * 0.45);
  vec3 grass = mix(uGold, uStraw, clamp(0.5 + fine * 1.6, 0.0, 1.0));
  grass = mix(grass, mix(uGreenA, uGreenB, clamp(0.5 + fine * 1.5, 0.0, 1.0)), wet * 0.85);
  // oak / scrub: dense dark-green dots, clustered by the mid noise (fades to an average tint when sub-pixel)
  float tf = 1.0 - smoothstep(0.35, 0.9, fw / 45.0);
  float tn = lk_noise(w / 45.0 + 3.7) * 0.65 + lk_noise(w / 13.0) * 0.35;   // 0..1
  float treeCover = clamp(0.12 + mid * 0.8 + wet * 0.25, 0.02, 0.6);       // fraction of ground under canopy
  float trees = mix(treeCover, smoothstep(1.0 - treeCover - 0.06, 1.0 - treeCover + 0.06, tn), tf);
  vec3 hills = mix(grass, uTree, trees * 0.8);
  // towns: organic mix of roofs/pavement (varied roof colours per 14 m cell), lawns and tree
  // canopy; every term falls back to its area average when it gets sub-pixel (no shimmer)
  float f30 = 1.0 - smoothstep(0.3, 0.8, fw / 30.0), f18 = 1.0 - smoothstep(0.3, 0.8, fw / 18.0), f14 = 1.0 - smoothstep(0.3, 0.8, fw / 14.0);
  // organic (grid-free) mix: lawns, paved / roofed patches with drifting tone, canopy on top
  vec3 roofAvg = (uRoofDark + uRoofTerra + uRoofLight + uRoof) * 0.25;
  float rt = lk_fbm(w / 60.0 + 21.0, fw / 60.0, 2);
  vec3 roof = mix(roofAvg, rt > 0.0 ? mix(uRoof, uRoofTerra, smoothstep(0.0, 0.25, rt)) : mix(uRoof, uRoofDark, smoothstep(0.0, -0.25, rt)), f30);
  float built = mix(0.42, smoothstep(-0.04, 0.12, lk_fbm(w / 24.0 + 11.0, fw / 24.0, LOOK_OCT)), f30);
  vec3 lot = mix(uLawn, roof, built * 0.7);
  float canopy = mix(0.3, smoothstep(0.06, 0.16, lk_fbm(w / 18.0 + 4.0, fw / 18.0, LOOK_OCT)), f18);
  lot = mix(lot, uTree, canopy * 0.85);
  vec3 c = mix(hills, lot, town);
#ifdef LOOK_VERGE
  // planting strip / yard: irrigated lawn with mowing patches and some dry spots
  float vf = lk_fbm(w / 3.0, fw / 3.0, 3), vd = lk_fbm(w / 11.0 + 2.0, fw / 11.0, 2);
  c = mix(uLawn, uGreenA, clamp(0.5 + vf * 1.4, 0.0, 1.0));
  c = mix(c, uGold, smoothstep(0.08, 0.25, vd) * 0.45);
  c *= 0.92;
#endif
  // tidal marsh / mud band along the shore
  c = mix(c, uMarsh, smoothstep(0.2, 0.5, shore) * 0.45);
  // close-range grass detail (two scales, faded out by distance)
  if (uDetail > 0.5) {
    vec2 pf = vRel.xz + uCamMod;
    vec3 g1 = texture2D(uGrass, pf / 4.0).rgb;
    vec3 g2 = texture2D(uGrass, vec2(pf.y, -pf.x) / 16.0).rgb;
    float df = 1.0 - smoothstep(600.0, 2500.0, length(vRel));
    float lum = (g1.r * 0.6 + g2.r * 0.4) * 2.0 - 1.0;
    c *= 1.0 + lum * 0.45 * df;
    c = mix(c, uStraw, clamp((g1.g - 0.5) * 1.2, 0.0, 1.0) * 0.3 * df * (1.0 - town) * (1.0 - wet));
  }
  gl_FragColor = vec4(lk_light(c, vec3(0.0, 1.0, 0.0)), 1.0);
` + FRAG_TAIL + '}';
  const C = (h) => new THREE.Color(h);
  const m = shaderMat(U, {
    uGrass: { value: tex.grass }, uCoastS: { value: tex.coastS }, uUrban: { value: tex.urban },
    uMaskRect: { value: tex.maskRect }, uHasUrban: { value: 0 }, uHasCoast: { value: tex.coastS ? 1 : 0 }, uDetail: { value: q.groundDetail ? 1 : 0 },
    uGreenA: { value: C(0x5a6c36) }, uGreenB: { value: C(0x76844a) },
    uGold: { value: C(0x9e8a5e) }, uStraw: { value: C(0xb8a47c) },
    uTree: { value: C(0x354629) }, uLawn: { value: C(0x66773f) }, uRoof: { value: C(0x8d8a84) },
    uRoofDark: { value: C(0x55565a) }, uRoofTerra: { value: C(0x8f7666) }, uRoofLight: { value: C(0xbdb9b0) },
    uMarsh: { value: C(0x6e6a4c) },
  }, vs, fs, { depthWrite: false });
  m.defines = { LOOK_OCT: q.groundOctaves };
  return m;
}

// ------------------------------------------------------------------ water
export function makeWaterMaterial(U, tex, q, tile = null) {
  const vs = VERT_HEAD + /* glsl */`
varying vec3 vW; varying vec2 vTileUv;
void main() {
  vTileUv = uv;
  vec4 w = modelMatrix * vec4(position, 1.0);
  vW = w.xyz;
  vec4 mvPosition = viewMatrix * w;
  gl_Position = projectionMatrix * mvPosition;
` + VERT_TAIL + '}';
  const fs = FRAG_HEAD + GLSL_COMMON + /* glsl */`
uniform sampler2D uMask; uniform sampler2D uCoastL;
uniform vec4 uMaskRect;
uniform vec3 uDeep, uShallow, uZenith, uHorizon;
varying vec3 vW;
#ifdef LOOK_TILE_MASK
uniform sampler2D uTileMask;
uniform float uTileChannel;   // 0 = .r (R8 DataTexture), 1 = .g (RGBA canvas)
#endif
varying vec2 vTileUv;
float maskAt(vec2 w) {
#if defined(LOOK_TILE_FULL)
  return 1.0;
#elif defined(LOOK_TILE_MASK)
  vec4 tm = texture2D(uTileMask, vTileUv);
  return mix(tm.r, tm.g, uTileChannel);
#endif
  vec2 uv = vec2((w.x - uMaskRect.x) / uMaskRect.z, 1.0 - (w.y - uMaskRect.y) / uMaskRect.w);
  if (uv.x < 0.002) return 1.0;                     // Pacific west of the mask
  if (uv.x > 1.0 || uv.y > 1.0) return 0.0;          // inland east / north: land (v = 1 is north)
  return texture2D(uMask, clamp(uv, 0.0, 1.0)).g;    // south: continue the coastline straight
}
void main() {
  #include <logdepthbuf_fragment>
  vec2 w = vW.xz;
  float m = maskAt(w);
  float fm = fwidth(m);
  float a = smoothstep(0.5 - max(fm, 0.02), 0.5 + max(fm, 0.02), m);
  if (a < 0.01) discard;
  vec2 muv = clamp(vec2((w.x - uMaskRect.x) / uMaskRect.z, 1.0 - (w.y - uMaskRect.y) / uMaskRect.w), 0.0, 1.0);
  vec2 ruv = vec2((w.x - uMaskRect.x) / uMaskRect.z, 1.0 - (w.y - uMaskRect.y) / uMaskRect.w);
  float off = texture2D(uCoastL, muv).r;            // ~1 far offshore, ~0.5 at the coast
  off = max(off, 1.0 - smoothstep(-0.0, 0.04, ruv.x));   // open Pacific beyond the mask's west edge
  float fw = length(fwidth(w));
  // gentle swell + chop as a sum of directional waves, band-limited by pixel footprint
  vec2 grad = vec2(0.0);
  float t = uTime;
  for (int i = 0; i < LOOK_WAVES; i++) {
    float fi = float(i);
    float ang = fi * 2.399 + 0.6;
    vec2 dir = vec2(cos(ang), sin(ang));
    float lambda = 38.0 / pow(1.9, fi);
    float k = 6.2832 / lambda;
    float amp = 0.035 * lambda / 38.0 * (0.8 + 0.4 * lk_noise(w / 400.0 + fi));
    float fade = 1.0 - smoothstep(0.15, 0.5, fw / lambda);
    grad += dir * k * amp * cos(dot(dir, w) * k + t * sqrt(9.8 * k)) * fade;
  }
  float nf = 1.0 - smoothstep(0.2, 0.6, fw / 6.0);
  grad += (vec2(lk_noise(w / 6.0 + t * 0.15), lk_noise(w / 6.0 + 19.0 - t * 0.12)) - 0.5) * 0.08 * nf;
  vec3 n = normalize(vec3(-grad.x, 1.0, -grad.y));
  vec3 V = normalize(cameraPosition - vW);
  float ndv = max(dot(n, V), 0.0);
  float fres = 0.02 + 0.98 * pow(1.0 - ndv, 5.0);
  vec3 R = reflect(-V, n);
  vec3 sky = mix(uHorizon, uZenith, pow(clamp(R.y, 0.0, 1.0), 0.5));
  float shallow = 1.0 - smoothstep(0.55, 0.92, off);
  vec3 body = mix(uDeep, uShallow, shallow);
  vec3 col = body * (uSkyIrr * 0.7 + uSunIrr * max(uSunDir.y, 0.0) * 0.5);
  col = mix(col, sky * 0.85, fres * 0.9);
  float sd = max(dot(R, uSunDir), 0.0);
  col += uSunIrr * (pow(sd, 600.0) * 2.5 + pow(sd, 60.0) * 0.06) * (0.3 + 0.7 * nf);
  // soft foam/wet line right at the coast
  col += vec3(0.05) * smoothstep(0.62, 0.5, off) * (1.0 - smoothstep(0.3, 0.8, fw / 25.0)) * lk_noise(w / 5.0 + t * 0.3);
  gl_FragColor = vec4(col, a);
` + FRAG_TAIL + '}';
  const m = shaderMat(U, {
    uMask: { value: tex.mask }, uCoastL: { value: tex.coastL }, uMaskRect: { value: tex.maskRect },
    uDeep: { value: new THREE.Color(0x1b4b5e) }, uShallow: { value: new THREE.Color(0x4f7a6c) },
    uZenith: U.uZenith, uHorizon: U.uHorizon,
    uTileMask: { value: tile && tile.mask ? tile.mask : null }, uTileChannel: { value: tile ? tile.channel : 0 },
  }, vs, fs, { depthWrite: false, ...BLEND_IN_OPAQUE });
  m.defines = { LOOK_WAVES: q.waterWaves };
  if (tile) { if (tile.full) m.defines.LOOK_TILE_FULL = ''; else m.defines.LOOK_TILE_MASK = ''; }
  return m;
}

// ------------------------------------------------------------------ sky dome
export function makeSkyMaterial(U) {
  return new THREE.ShaderMaterial({
    uniforms: U,
    vertexShader: /* glsl */`
varying vec3 vDir;
void main() {
  vDir = position;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  gl_Position.z = gl_Position.w * 0.999999;
}`,
    fragmentShader: /* glsl */`
uniform vec3 uZenith, uHorizon, uSunDir, uSunGlow;
varying vec3 vDir;
void main() {
  vec3 d = normalize(vDir);
  float h = d.y;
  vec3 c = mix(uHorizon, uZenith, pow(smoothstep(-0.02, 1.0, h), 0.45));
  c = mix(c, uHorizon * 0.93, smoothstep(0.0, -0.25, h));
  float sd = max(dot(d, uSunDir), 0.0);
  c += uSunGlow * (pow(sd, 8.0) * 0.16 + pow(sd, 400.0) * 0.35) + uSunGlow * smoothstep(0.99994, 0.99997, sd) * 2.0;
  gl_FragColor = vec4(c, 1.0);
  #include <colorspace_fragment>
}`,
    side: THREE.BackSide, depthTest: false, depthWrite: false, fog: false, toneMapped: false,
  });
}

// ------------------------------------------------------------------ cars
/** Patch a car material: darkened underside (fake AO), rim light, glowing state colours. */
export function patchCarMaterial(mat, U, q) {
  mat.onBeforeCompile = (sh) => {
    sh.uniforms.uRimCol = U.uRimCol;
    sh.uniforms.uStateGlow = U.uStateGlow;
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', '#include <common>\nvarying float vCarH;')
      .replace('#include <begin_vertex>', `#include <begin_vertex>
vCarH = position.y;
#ifdef USE_INSTANCING
  transformed.y -= step(position.y, 0.001) * 0.27 / max(instanceMatrix[1][1], 0.1);   // skirt down to the road
#endif`);
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', '#include <common>\nvarying float vCarH;\nuniform vec3 uRimCol;\nuniform float uStateGlow;')
      .replace('#include <color_fragment>', `#include <color_fragment>
  float lkAO = mix(0.42, 1.0, smoothstep(0.0, 0.45, vCarH));
  vec3 lkBase = diffuseColor.rgb;
  float lkSat = max(lkBase.r, max(lkBase.g, lkBase.b)) - min(lkBase.r, min(lkBase.g, lkBase.b));
  float lkState = step(0.55, lkSat) * step(0.6, lkBase.r) * step(lkBase.b, 0.35);   // braking amber / crashed red
  diffuseColor.rgb *= lkAO;`)
      .replace('#include <emissivemap_fragment>', `#include <emissivemap_fragment>
  totalEmissiveRadiance += lkBase * uStateGlow * lkState;
  ${q.carRim ? 'float lkRim = pow(1.0 - clamp(dot(normal, normalize(vViewPosition)), 0.0, 1.0), 3.0);\n  totalEmissiveRadiance += uRimCol * lkRim * lkAO;' : ''}`);
  };
  mat.customProgramCacheKey = () => 'look-car-' + (q.carRim ? 1 : 0);
  return mat;
}

/** Soft contact-shadow blob under every car: shares the car mesh's instanceMatrix. */
export function makeCarShadowMaterial(U) {
  return new THREE.ShaderMaterial({
    uniforms: Object.assign({ uLift: { value: 0.27 }, uStrength: { value: 0.6 } }, U),
    vertexShader: /* glsl */`
#include <common>
#include <logdepthbuf_pars_vertex>
uniform vec3 uSunDir; uniform float uLift;
varying vec2 vQ;
void main() {
  vQ = position.xz;
  vec3 T = instanceMatrix[3].xyz;
  vec3 ax = instanceMatrix[0].xyz, az = instanceMatrix[2].xyz;
  vec2 sh = -normalize(uSunDir.xz + vec2(1e-4)) * min(0.9, 0.3 / max(uSunDir.y, 0.2));
  vec3 wp = T + ax * position.x * 0.58 + az * position.z * 0.66;
  wp.xz += sh;
  wp.y = T.y - uLift;
  vec4 mvPosition = viewMatrix * modelMatrix * vec4(wp, 1.0);
  gl_Position = projectionMatrix * mvPosition;
  #include <logdepthbuf_vertex>
}`,
    fragmentShader: /* glsl */`
#include <common>
#include <logdepthbuf_pars_fragment>
uniform float uStrength;
varying vec2 vQ;
void main() {
  #include <logdepthbuf_fragment>
  float d = length(vQ);
  float a = (1.0 - smoothstep(0.45, 1.0, d)) * uStrength;
  if (a < 0.01) discard;
  gl_FragColor = vec4(0.0, 0.0, 0.0, a);
}`,
    depthWrite: false, toneMapped: false, fog: false, ...BLEND_IN_OPAQUE,
  });
}

export function makeCarShadowGeometry() {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute([-1, 0, -1, 1, 0, -1, -1, 0, 1, 1, 0, 1], 3));
  g.setIndex([0, 2, 1, 1, 2, 3]);
  return g;
}

// ------------------------------------------------------------------ far-view road quads (from roadLines)
// Drawn under the ribbons (renderOrder -0.5, no depth write): invisible close up where the
// ribbons cover them, and an anti-aliased >= 1-2 px stroke per class from map altitude.
export function makeFarRoadMaterial(U, tex) {
  const fs = FRAG_HEAD + GLSL_COMMON + /* glsl */`
uniform float uTone[13];
varying float vSide; varying float vAlpha; varying vec3 vW; varying vec3 vRel; varying float vCls;
void main() {
  #include <logdepthbuf_fragment>
  float aa = clamp((1.0 - abs(vSide)) / max(fwidth(vSide), 1e-4), 0.0, 1.0);
  float a = aa * mix(0.55, 1.0, vAlpha) * smoothstep(1500.0, 4000.0, length(vRel));   // ribbons take over close up
  if (a < 0.004) discard;
  float cls = mod(floor(vCls + 0.5), 16.0);
  float tone = uTone[int(clamp(cls, 0.0, 12.0))] * 0.7;
  gl_FragColor = vec4(lk_light(vec3(tone), vec3(0.0, 1.0, 0.0)), a);
` + FRAG_TAIL + '}';
  const m = shaderMat(U, {
    uTone: { value: tex.tones }, uWidth: { value: 10 }, uExtend: { value: 1.0 }, uMinPx: { value: 0.75 },
  }, QUAD_VS.replace('float hwE = max(hw, uMinPx * px);', 'float hwE = max(hw, uMinPx * px * (vCls < 0.5 ? 1.5 : vCls < 2.5 ? 1.2 : 0.85));'), fs, { depthWrite: false, ...BLEND_IN_OPAQUE });
  m.defines = { SEG_PER_INSTANCE: '' };
  return m;
}

// ------------------------------------------------------------------ paint on world-built marking meshes
// For the rebuilt world (markings as triangle quads): keep their geometry, swap in lit
// paint. Colour comes from the mesh's `color` attribute (LOOK_VCOLOR) or the old material.
export function makePaintMeshMaterial(U, color, vertexColors) {
  const vs = VERT_HEAD + /* glsl */`
#ifdef LOOK_VCOLOR
attribute vec3 color;
#endif
varying vec3 vCol; varying vec3 vRel;
uniform vec3 uPaint;
void main() {
#ifdef LOOK_VCOLOR
  vCol = color;
#else
  vCol = uPaint;
#endif
  vec4 w = modelMatrix * vec4(position, 1.0);
  vRel = w.xyz - cameraPosition;
  vec4 mvPosition = viewMatrix * w;
  gl_Position = projectionMatrix * mvPosition;
` + VERT_TAIL + '}';
  const fs = FRAG_HEAD + GLSL_COMMON + /* glsl */`
uniform float uGlow;
varying vec3 vCol; varying vec3 vRel;
void main() {
  #include <logdepthbuf_fragment>
  vec2 pf = vRel.xz + uCamMod;
  float wear = lk_noise(pf * 1.7) * lk_noise(pf * 0.31 + 7.0);
  vec3 alb = min(vCol, vec3(0.92)) * (0.86 + 0.14 * smoothstep(0.05, 0.4, wear));
  gl_FragColor = vec4(lk_light(alb, vec3(0.0, 1.0, 0.0)) + alb * uGlow, 1.0);
#ifdef USE_LOGDEPTHBUF
  gl_FragDepthEXT -= 4e-7;    // log depth ignores polygonOffset: pull paint just in front of its own deck
#endif
` + FRAG_TAIL + '}';
  const m = shaderMat(U, { uPaint: { value: new THREE.Color(color) }, uGlow: { value: 0.12 } }, vs, fs,
    { polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -4, side: THREE.FrontSide, depthTest: true, depthWrite: true });
  if (vertexColors) m.defines = { LOOK_VCOLOR: '' };
  return m;
}

// ------------------------------------------------------------------ local streets (streaming tiles)
// One shader, five surface kinds (define LOOK_STREET): 0 residential asphalt, 1 junction,
// 2 sidewalk, 3 curb, 4 driveway. Optional `uv` attribute in METRES (x = along the street,
// y = across) aligns joints/patches with the street; without it everything is world-xz.
// Normals come from screen-space derivatives (curb tops vs faces), so no normal attribute
// is needed. All fine detail is band-limited by the pixel footprint (no shimmer at 1.2 m eye height).
export function makeStreetMaterial(U, tex, kind) {
  const vs = VERT_HEAD + /* glsl */`
// (three's prefix already declares \`attribute vec2 uv\` and \`attribute vec3 normal\`)
attribute float classId;   // stream tiles: class byte, surface group in bits 5-7
varying vec3 vW; varying vec3 vRel; varying vec2 vSt; varying vec3 vN; varying float vSurf;
void main() {
  vSurf = floor(floor(classId + 0.5) / 32.0);
  vN = normalize(mat3(modelMatrix) * normal);
  vec4 w = modelMatrix * vec4(position, 1.0);
  vW = w.xyz; vRel = w.xyz - cameraPosition;
#ifdef LOOK_UV
  vSt = uv;
#else
  vSt = w.xz;
#endif
  vec4 mvPosition = viewMatrix * w;
  gl_Position = projectionMatrix * mvPosition;
` + VERT_TAIL + '}';
  const fs = FRAG_HEAD + GLSL_COMMON + SURFACE_FN + /* glsl */`
uniform sampler2D uAsphalt; uniform float uAsphaltMean;
uniform float uResTone; uniform vec3 uConcreteSW; uniform vec3 uCurbCol;
varying vec3 vW; varying vec3 vRel; varying vec2 vSt; varying vec3 vN; varying float vSurf;
float lineAA(float d, float halfW, float fwd) { return 1.0 - smoothstep(halfW - fwd, halfW + fwd, abs(d)); }
void main() {
  #include <logdepthbuf_fragment>
  vec3 n = vec3(0.0, 1.0, 0.0);
#if LOOK_STREET == 3 && defined(LOOK_NORMAL)
  n = normalize(vN); if (n.y < 0.0) n = -n;
#elif LOOK_STREET == 3
  vec3 cr = cross(dFdx(vW), dFdy(vW));
  float cl = length(cr);
  if (cl > 1e-10) { n = cr / cl; if (n.y < 0.0) n = -n; }
#endif
  vec2 w = vW.xz, st = vSt;
  float fw = length(fwidth(w));
  float dist = length(vRel);
  vec2 pf = vRel.xz + uCamMod;
  vec3 det = (texture2D(uAsphalt, pf / 8.0).rgb * 0.7 + texture2D(uAsphalt, vec2(pf.y, -pf.x) / 32.0).rgb * 0.3) / uAsphaltMean;
  vec3 alb;
#if LOOK_STREET <= 1
  // older, lighter, greyer asphalt; colour drifts block to block
  float block = lk_fbm(w / 160.0 + 5.0, fw / 160.0, 2);
  alb = vec3(uResTone) * vec3(1.0, 0.995, 0.985) * mix(vec3(1.0), det, 0.8) * (1.0 + block * 0.35 + lk_fbm(w / 9.0, fw / 9.0, 2) * 0.18);
  // rectangular utility-cut patches (darker, fresher), aligned with the street frame
  vec2 pc = floor(st / vec2(11.0, 4.0) + vec2(0.0, 0.5));
  vec2 pp = fract(st / vec2(11.0, 4.0) + vec2(0.0, 0.5)) - 0.5;
  float ph = lk_hash(pc + ${kind === 1 ? '17.0' : '3.0'});
  float pf2 = 1.0 - smoothstep(0.3, 0.9, fw / 3.0);
  float patchR = (1.0 - smoothstep(0.0, fw / 4.0 + 0.01, max(abs(pp.x) - 0.22 * (0.5 + ph), abs(pp.y) - 0.16))) * step(0.94, ph);
  alb = mix(alb, alb * 0.84, patchR * mix(0.25, 1.0, pf2));
  // crack-seal: meandering dark tar lines (iso-lines of noise), fade to a faint tint far away
  // crack-seal: thin transverse cracks across the street (street frame from the metre uv), each
  // with a little wiggle and a ragged end; never closed loops. No street frame (world xz) -> none.
  float pxm = max(fw * 0.7071, 1e-4);
  float crack = 0.0;
#ifdef LOOK_UV
  float cell = floor(st.x / 7.0);
  float ch = lk_hash(vec2(cell, 3.3));
  float xc = (cell + 0.2 + 0.6 * lk_hash(vec2(cell, 7.1))) * 7.0 + (lk_noise(vec2(st.y * 0.9, cell)) - 0.5) * 0.35;
  float dmc = abs(st.x - xc);
  float span = step(0.55, ch) * smoothstep(0.0, 0.6, lk_noise(vec2(st.y * 0.35 + cell * 1.7, 2.0)) );   // broken along its length
  crack = (1.0 - smoothstep(0.012, 0.012 + pxm, dmc)) * span;
  crack *= 1.0 - smoothstep(0.03, 0.08, pxm);                   // sub-pixel: fade instead of shimmer
#endif
  alb = mix(alb, alb * 0.35, crack * (1.0 - step(0.5, vSurf)) * 0.8);
#ifdef LOOK_UV
  alb = lk_surface(alb, vSurf, w, fw, det, st, 1.0);
#else
  alb = lk_surface(alb, vSurf, w, fw, det, w, 0.0);
#endif
  // manhole / valve covers on a 26 m cell grid
  vec2 mc = floor(w / 26.0);
  float mh = lk_hash(mc + 1.7);
  vec2 mpos = (mc + 0.25 + 0.5 * vec2(lk_hash(mc + 4.1), lk_hash(mc + 8.3))) * 26.0;
  float md = length(w - mpos);
  float mr = mh < 0.5 ? 0.33 : 0.14;
  float cover = (1.0 - smoothstep(mr - fw, mr + fw, md)) * step(0.55, mh) * (1.0 - smoothstep(0.3, 0.8, fw / 0.3));
  float ring = lineAA(md - mr, 0.025, fw) * cover;
  float grid = step(0.5, fract((w.x - mpos.x) * 6.0)) * 0.12;
  alb = mix(alb, vec3(0.06, 0.058, 0.055) + grid * 0.05, cover);
  alb = mix(alb, vec3(0.02), ring * 0.7);
#elif LOOK_STREET == 3
  // curb: light concrete, bright top edge, slightly darker vertical face
  alb = uCurbCol * (1.0 + lk_fbm(w / 4.0, fw / 4.0, 2) * 0.12 + lk_fbm(w / 0.7, fw / 0.7, 2) * 0.08);
  alb *= mix(0.82, 1.08, smoothstep(0.6, 0.95, n.y));
#else
  // sidewalk / driveway concrete: slabs with saw-cut joints, per-slab tone, stains
  vec2 slab = st / vec2(${kind === 4 ? '3.0, 3.0' : '1.5, 1.5'});
  vec2 sid = floor(slab);
  float tone = 1.0 + (lk_hash(sid + 2.0) - 0.5) * 0.08;
  alb = uConcreteSW * tone * mix(vec3(1.0), det, 0.35) * (1.0 + lk_fbm(w / 7.0 + 2.0, fw / 7.0, 3) * 0.25);
  vec2 fs2 = abs(fract(slab) - 0.5);
  vec2 jfw = fwidth(slab);
  float joint = max(1.0 - smoothstep(0.012, 0.012 + 1.5 * jfw.x, 0.5 - fs2.x), 1.0 - smoothstep(0.012, 0.012 + 1.5 * jfw.y, 0.5 - fs2.y));
  joint *= 1.0 - smoothstep(0.08, 0.2, max(jfw.x, jfw.y));
  alb *= 1.0 - 0.45 * joint;
  alb *= 1.0 - 0.15 * smoothstep(0.62, 0.8, lk_noise(w / 3.0 + 13.0)) * (1.0 - smoothstep(0.3, 0.8, fw / 3.0));
#endif
  vec3 col = lk_light(alb, n);
  gl_FragColor = vec4(col, 1.0);
` + FRAG_TAIL + '}';
  const m = shaderMat(U, {
    uAsphalt: { value: tex.asphalt }, uAsphaltMean: { value: tex.asphaltMean },
    uResTone: { value: 0.17 }, uConcreteSW: { value: new THREE.Color(0xb3aea4) }, uCurbCol: { value: new THREE.Color(0xc4c0b8) },
  }, vs, fs, { side: THREE.DoubleSide });
  m.defines = { LOOK_STREET: kind };
  m.extensions = { derivatives: true };
  return m;
}

// ------------------------------------------------------------------ signal / sign posts (stream role 'signals')
// classId per vertex: 0 signal, 1 stop, 2 give_way, 3 ramp_meter, 4 crossing signal.
export function makeSignalMaterial(U, hasNormal) {
  const vs = VERT_HEAD + /* glsl */`
attribute float classId;
varying vec3 vW; varying float vCls; varying vec3 vN; varying float vH;
void main() {
  vCls = classId; vN = normalize(mat3(modelMatrix) * normal); vH = position.y;
  vec4 w = modelMatrix * vec4(position, 1.0);
  vW = w.xyz;
  vec4 mvPosition = viewMatrix * w;
  gl_Position = projectionMatrix * mvPosition;
` + VERT_TAIL + '}';
  const fs = FRAG_HEAD + GLSL_COMMON + /* glsl */`
varying vec3 vW; varying float vCls; varying vec3 vN; varying float vH;
void main() {
  #include <logdepthbuf_fragment>
#ifdef LOOK_NORMAL
  vec3 n = normalize(vN);
#else
  vec3 cr = cross(dFdx(vW), dFdy(vW)); vec3 n = length(cr) > 1e-10 ? normalize(cr) : vec3(0.0, 1.0, 0.0);
#endif
  int c = int(floor(mod(vCls, 16.0) + 0.5));
  // pole: galvanised grey (signals, meters) or the sign colour (stop red, yield white-red)
  vec3 alb = vec3(0.32, 0.33, 0.34); vec3 glow = vec3(0.0);
  if (c == 0) { alb = vec3(0.10, 0.11, 0.10); glow = vec3(0.9, 0.12, 0.05) * 0.5; }        // signal head: dark with red lamp
  else if (c == 1) { alb = vec3(0.55, 0.03, 0.04); glow = alb * 0.25; }                       // stop sign red
  else if (c == 2) { alb = vec3(0.70, 0.68, 0.66); glow = vec3(0.6, 0.05, 0.05) * 0.15; }     // yield: white / red
  else if (c == 3) { alb = vec3(0.12, 0.12, 0.12); glow = vec3(1.0, 0.5, 0.05) * 0.5; }       // ramp meter: amber lamp
  else if (c == 4) { alb = vec3(0.12, 0.12, 0.12); glow = vec3(1.0, 0.45, 0.1) * 0.35; }      // crossing signal: hand
  vec3 col = lk_light(alb, n) + glow;
  gl_FragColor = vec4(col, 1.0);
` + FRAG_TAIL + '}';
  const m = shaderMat(U, {}, vs, fs, { side: THREE.DoubleSide });
  if (hasNormal) m.defines = { LOOK_NORMAL: '' };
  return m;
}
