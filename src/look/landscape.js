// Landscape layer of the look pack (render only; SPEC "Rendering vs training": never imported by the sim,
// training, tests or src/gpu). Created by applyLook() when the landscape mode is 'full' (default).
//
//   ground : one CDLOD terrain surface (quadtree of 32x32-cell patches, geomorphed between levels, so LOD changes
//            never pop or crack), drawn first, writing depth with a small distance-relative bias so roads, paint,
//            buildings and cars always win where they touch it. Colour = OSM land cover in a clay palette
//            (data/landscape/lc-global.png, 25 m/px, plus 6.45 m/px detail tiles streamed into an atlas through a
//            page table, cross-faded in), hillshade + cavity shading from the USGS 3DEP heightfield (live sun),
//            water with a shallow -> deep gradient and a soft shore band. Optional true relief (terrain=1).
//   buildings / trees : see buildings.js (phase 3).
//
// Data: data/landscape/* built by tools/landscape/*.mjs (see docs/landscape.md).
import * as THREE from 'three';
import { GLSL_COMMON } from './materials.js';
import { GLSL_TERRAIN } from './terrain-glsl.js';
import { createBuildingsAndTrees } from './buildings.js';
import { createStructures } from './structures.js';
export { GLSL_TERRAIN };

const BASE = 'data/landscape/';
const GRID_M = 32;              // cells per CDLOD patch side
const LEAF = 75;                // finest patch size (m): cell 2.3 m
const LEVELS = 13;              // 75 m .. 307.2 km
const RANGE_K = 4.0;            // patch at level l is used out to RANGE_K * size_l
const MORPH_START = 0.9;        // fraction of the range where the morph to the next level starts

// --------------------------------------------------------------------------------------------- GLSL
const VERT_HEAD = /* glsl */`
#include <common>
#include <fog_pars_vertex>
#include <logdepthbuf_pars_vertex>
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

function groundMaterial(U, L) {
  const vs = VERT_HEAD + GLSL_TERRAIN + /* glsl */`
attribute vec2 grid;
attribute vec4 aNode;            // x0, z0, size, level
uniform float uMorphStart[${LEVELS}], uMorphEnd[${LEVELS}];
uniform float uLsOn;             // 1 = displaced relief
varying vec3 vW; varying vec3 vRel; varying float vMorph;
float hAt(vec2 xz) { return uLsOn > 0.5 ? ls_heightRaw(xz) * uLsExag + uLsFlatY : uLsFlatY; }
void main() {
  float cell = aNode.z / ${GRID_M}.0;
  vec2 xz = aNode.xy + grid * cell;
  float d = distance(cameraPosition, vec3(xz.x, hAt(xz), xz.y));
  int l = int(aNode.w + 0.5);
  float mk = clamp((d - uMorphStart[l]) / (uMorphEnd[l] - uMorphStart[l]), 0.0, 1.0);
  vec2 fr = fract(grid * 0.5) * 2.0;
  xz = aNode.xy + (grid - fr * mk) * cell;
  vec4 w = vec4(xz.x, hAt(xz), xz.y, 1.0);
  vW = w.xyz; vRel = w.xyz - cameraPosition; vMorph = mk;
  vec4 mvPosition = viewMatrix * w;
  gl_Position = projectionMatrix * mvPosition;
  #include <logdepthbuf_vertex>
  #include <fog_vertex>
}`;
  const fs = FRAG_HEAD + GLSL_COMMON + /* glsl */`
uniform sampler2D uLsHeight; uniform vec4 uLsRoot; uniform float uLsShade;
uniform sampler2D uGlobal, uAtlas, uPage, uShore, uGrass, uOuter;
uniform vec4 uOuterRect;
uniform vec4 uPageInfo;          // tiles per side, tile size (m), slot px, gutter px
uniform float uAtlasSlots;       // slots per side
uniform float uFade; uniform vec3 uFadeCol; uniform float uLogBias;
uniform vec3 uDeep, uShallow, uShoreCol, uWetSand, uZenith, uHorizon;
uniform float uDetail, uAlbedoGain, uLsShadow, uLsDebug;
varying vec3 vW; varying vec3 vRel; varying float vMorph;
float hTex(vec2 uv) { return textureLod(uLsHeight, uv, 0.0).r; }
float gDbgA = -1.0, gDbgLod = 0.0;
vec4 albedoAt(vec2 xz, out float detailW) {
  vec2 uv = (xz - uLsRoot.xy) / uLsRoot.z;
  bool inside = uv.x >= 0.0 && uv.y >= 0.0 && uv.x <= 1.0 && uv.y <= 1.0;
  detailW = 0.0;
  // screen derivatives first, in uniform control flow (derivatives after a divergent branch are undefined)
  vec2 duvx = dFdx(uv), duvy = dFdy(uv);
  // past the data square: the outer ring (150 m/px, DEM-derived clay colour); blended in over the last 1.5 km
  vec2 eo = max(-uv, uv - 1.0) * uLsRoot.z;
  float outK = smoothstep(-1500.0, 0.0, max(eo.x, eo.y));
  vec4 ov = texture2D(uOuter, (xz - uOuterRect.xy) / uOuterRect.z);
  if (!inside) return ov;
  vec4 g = mix(texture2D(uGlobal, uv), ov, outK);
  vec2 tc = uv * uPageInfo.x;
  vec2 tile = floor(tc);
  vec4 pg = texelFetch(uPage, ivec2(tile), 0);
  gDbgA = pg.a;
  if (pg.a <= 0.0) return g;
  vec2 inTile = (tc - tile) * (uPageInfo.z - 2.0 * uPageInfo.w) + uPageInfo.w;   // px inside the slot
  vec2 slot = floor(pg.rg * 255.0 + 0.5);
  vec2 auv = (slot * uPageInfo.z + inTile) / (uAtlasSlots * uPageInfo.z);
  // derivatives from the continuous coordinate (tile seams have no derivative jump); limit to mip 2.5 (8 px gutter)
  vec2 dx = duvx * uPageInfo.x * (uPageInfo.z - 2.0 * uPageInfo.w) / (uAtlasSlots * uPageInfo.z);
  vec2 dy = duvy * uPageInfo.x * (uPageInfo.z - 2.0 * uPageInfo.w) / (uAtlasSlots * uPageInfo.z);
  float texels = max(length(dx), length(dy)) * uAtlasSlots * uPageInfo.z;
  float lod = log2(max(texels, 1e-6));
  gDbgLod = lod;
  float k = min(1.0, 5.6 / max(texels, 1e-6));
  vec4 d = textureGrad(uAtlas, auv, dx * k, dy * k);
  if (lod < 0.5) {
    // magnified: crisp anti-aliased class edges from the 4 nearest texels (argmax of bilinear weight per colour);
    // smooth gradients (all four colours close) keep plain bilinear
    vec2 tp = auv * uAtlasSlots * uPageInfo.z - 0.5;
    ivec2 i0 = ivec2(floor(tp)); vec2 f = fract(tp);
    vec4 c0 = texelFetch(uAtlas, i0, 0), c1 = texelFetch(uAtlas, i0 + ivec2(1, 0), 0), c2 = texelFetch(uAtlas, i0 + ivec2(0, 1), 0), c3 = texelFetch(uAtlas, i0 + ivec2(1, 1), 0);
    vec4 w = vec4((1.0 - f.x) * (1.0 - f.y), f.x * (1.0 - f.y), (1.0 - f.x) * f.y, f.x * f.y);
    // two-colour edge reconstruction (like an SDF): the most different pair of corners gives the two classes A, B;
    // every corner's coverage t = its position between A and B (texels on an edge are anti-aliased mixes), the
    // bilinear coverage is thresholded at 0.5 with a one-pixel ramp -> straight, crisp, anti-aliased class edges
    vec4 A = c0, B = c1; float dm = dot(abs(c0 - c1), vec4(1.0));
    float q;
    q = dot(abs(c0 - c2), vec4(1.0)); if (q > dm) { dm = q; A = c0; B = c2; }
    q = dot(abs(c0 - c3), vec4(1.0)); if (q > dm) { dm = q; A = c0; B = c3; }
    q = dot(abs(c1 - c2), vec4(1.0)); if (q > dm) { dm = q; A = c1; B = c2; }
    q = dot(abs(c1 - c3), vec4(1.0)); if (q > dm) { dm = q; A = c1; B = c3; }
    q = dot(abs(c2 - c3), vec4(1.0)); if (q > dm) { dm = q; A = c2; B = c3; }
    if (dm > 0.03) {
      vec4 ab = B - A; float ab2 = max(dot(ab, ab), 1e-6);
      vec4 tq = clamp(vec4(dot(c0 - A, ab), dot(c1 - A, ab), dot(c2 - A, ab), dot(c3 - A, ab)) / ab2, 0.0, 1.0);
      float T = dot(tq, w);
      float aa = max(texels, 1e-3);
      vec4 sharp = mix(A, B, clamp((T - 0.5) / aa + 0.5, 0.0, 1.0));
      d = mix(sharp, d, smoothstep(-0.5, 0.5, lod));
    }
  }
  detailW = pg.a * (1.0 - smoothstep(1.2, 2.2, lod));
  return mix(g, d, detailW);
}
void main() {
  #include <logdepthbuf_fragment>
#ifdef USE_LOGDEPTHBUF
  gl_FragDepthEXT += uLogBias;   // ground sits a hair behind everything that touches it (relative bias, any altitude)
#endif
  vec2 xz = vW.xz;
  float dist = length(vRel);
  float fw = length(fwidth(xz));
  float dW;
  vec4 a = albedoAt(xz, dW);
  vec3 alb = a.rgb;
  // ---- land shading: hillshade from the heightfield (live sun) + cavity darkening in valleys
  vec2 uv = (xz - uLsRoot.xy) / uLsRoot.z;
  float t = 2.0 / uLsRoot.w, cm = 2.0 * uLsRoot.z / uLsRoot.w;   // normals over +-50 m: smooth relief, no street-conform ridges
  float hl = hTex(uv - vec2(t, 0.0)), hr = hTex(uv + vec2(t, 0.0)), hd = hTex(uv - vec2(0.0, t)), hu = hTex(uv + vec2(0.0, t));
  float hc = hTex(uv);
  vec3 n = normalize(vec3(-(hr - hl) * uLsShade / (2.0 * cm), 1.0, -(hu - hd) * uLsShade / (2.0 * cm)));
  float hl2 = hTex(uv - vec2(3.0 * t, 0.0)), hr2 = hTex(uv + vec2(3.0 * t, 0.0)), hd2 = hTex(uv - vec2(0.0, 3.0 * t)), hu2 = hTex(uv + vec2(0.0, 3.0 * t));
  float lap = (hl2 + hr2 + hd2 + hu2 - 4.0 * hc) / (9.0 * cm * cm) * 25.0;    // + in valleys
  float cav = clamp(1.0 - lap * 6.0, 0.78, 1.06);
  // relief only inside the data square (outside, the clamped texture would smear into stripes); fade over 2 km
  vec2 eo = max(-uv, uv - 1.0) * uLsRoot.z;
  float inR = 1.0 - smoothstep(-2000.0, 0.0, max(eo.x, eo.y));
  n = normalize(mix(vec3(0.0, 1.0, 0.0), n, inR)); cav = mix(1.0, cav, inR); hc *= inR;
  // close-range texture (lawn / dry grass / soil grain), only on vegetated albedo, faded by distance
  if (uDetail > 0.5) {
    float veg = clamp((alb.g - max(alb.r, alb.b)) * 9.0, 0.0, 1.0);
    vec2 pf = vRel.xz + uCamMod;
    vec3 g1 = texture2D(uGrass, pf / 4.0).rgb, g2 = texture2D(uGrass, vec2(pf.y, -pf.x) / 16.0).rgb;
    float df = 1.0 - smoothstep(250.0, 1200.0, dist);
    float lum = (g1.r * 0.6 + g2.r * 0.4) * 2.0 - 1.0;
    alb *= 1.0 + lum * mix(0.10, 0.32, veg) * df;
  }
  // clay hillshade: wrapped sun term relative to flat ground (pure light/dark modulation, no colour cast)
  vec3 sunH = normalize(vec3(uSunDir.x, max(uSunDir.y, 0.25), uSunDir.z));
  float wrapN = clamp((dot(n, sunH) + 0.45) / 1.45, 0.0, 1.0), wrapF = (sunH.y + 0.45) / 1.45;
  float shade = clamp(mix(1.0, wrapN / wrapF, 0.85), 0.5, 1.4);
  // long terrain shadows (low sun): march the heightfield toward the sun, soft penumbra
  if (uLsShadow > 0.0 && uSunDir.y > 0.0 && uSunDir.y < 0.6 && hc > 1.0) {
    vec2 sd2 = normalize(uSunDir.xz + vec2(1e-5)); float tanE = uSunDir.y / max(length(uSunDir.xz), 1e-3);
    float occ = 0.0, t3 = 30.0;
    for (int i = 0; i < 14; i++) {
      vec2 q = uv + sd2 * t3 / uLsRoot.z;
      float rise = hTex(q) * uLsShade * 0.6 - (hc * uLsShade * 0.6 + t3 * tanE);
      occ = max(occ, clamp(rise / (t3 * 0.08 + 4.0), 0.0, 1.0));
      t3 *= 1.45;
    }
    shade *= 1.0 - occ * uLsShadow;
  }
  // built-up land (light, unsaturated albedo): softer relief shading so streets do not look dirty
  float lumA = dot(alb, vec3(0.3, 0.59, 0.11)), satA = max(alb.r, max(alb.g, alb.b)) - min(alb.r, min(alb.g, alb.b));
  float urb = smoothstep(0.5, 0.62, lumA) * (1.0 - smoothstep(0.06, 0.14, satA));
  shade = mix(shade, 1.0 + (shade - 1.0) * 0.45, urb); cav = mix(cav, 1.0 + (cav - 1.0) * 0.3, urb);
  vec3 land = lk_light(alb * uAlbedoGain * cav, vec3(0.0, 1.0, 0.0)) * shade;
  // ---- water: crisp anti-aliased shoreline from the (bilinear) coverage, shallow -> deep by distance offshore
  float cov = a.a;
  float aw = max(fwidth(cov), 0.02);
  float wmask = smoothstep(0.5 - aw, 0.5 + aw, cov);
  float shoreD = mix(texture2D(uShore, clamp(uv, 0.0, 1.0)).r * 255.0 * 16.0, 4000.0, 1.0 - inR);          // metres offshore (bay / ocean)
  float off = smoothstep(80.0, 3200.0, shoreD);
  vec3 body = mix(uShallow, uDeep, off);
  // gentle sky reflection + soft sun glint (band-limited; nothing sub-pixel)
  vec3 V = normalize(-vRel);
  float t2 = uTime;
  float nf = 1.0 - smoothstep(0.2, 0.6, fw / 9.0);
  vec2 wg = (vec2(lk_noise(xz / 9.0 + t2 * 0.12), lk_noise(xz / 9.0 + 31.0 - t2 * 0.1)) - 0.5) * 0.06 * nf;
  vec2 sw = (vec2(lk_noise(xz / 70.0 + t2 * 0.03), lk_noise(xz / 70.0 + 7.0 - t2 * 0.025)) - 0.5) * 0.05 * (1.0 - smoothstep(0.2, 0.6, fw / 70.0));
  vec3 wn = normalize(vec3(wg.x + sw.x, 1.0, wg.y + sw.y));
  float fres = 0.02 + 0.98 * pow(1.0 - max(dot(wn, V), 0.0), 5.0);
  vec3 R = reflect(-V, wn);
  vec3 sky = mix(uHorizon, uZenith, pow(clamp(R.y, 0.0, 1.0), 0.5));
  vec3 water = body * (uSkyIrr * 0.75 + uSunIrr * max(uSunDir.y, 0.0) * 0.45);
  water = mix(water, sky * 0.8, fres * 0.6);
  float sd = max(dot(R, uSunDir), 0.0);
  water += uSunIrr * (pow(sd, 300.0) * 0.25 + pow(sd, 40.0) * 0.03) * (0.3 + 0.7 * nf);   // soft glint (no sparkle noise)
  // wet sand just above the waterline
  float wet = smoothstep(0.22, 0.5, cov) * (1.0 - wmask);
  land = mix(land, lk_light(uWetSand * uAlbedoGain, vec3(0.0, 1.0, 0.0)) * shade, wet * 0.7);
  vec3 col = mix(land, water, wmask);
  col = mix(uFadeCol, col, uFade);
  if (uLsDebug > 0.5) col = vec3(dW, max(gDbgA, 0.0), gDbgA < 0.0 ? 1.0 : clamp(gDbgLod / 4.0 + 0.5, 0.0, 1.0));
  gl_FragColor = vec4(col, 1.0);
` + FRAG_TAIL + '}';
  const C = (h) => new THREE.Color(h);
  const uniforms = Object.assign(THREE.UniformsUtils.clone(THREE.UniformsLib.fog), U, L, {
    uMorphStart: { value: new Array(LEVELS).fill(0) }, uMorphEnd: { value: new Array(LEVELS).fill(1) },
    uDeep: { value: C(0x3d7d93) }, uShallow: { value: C(0x6fb0b4) }, uShoreCol: { value: C(0x9ccfc6) }, uWetSand: { value: C(0xcfc29c) },
    uZenith: U.uZenith, uHorizon: U.uHorizon,
  });
  const m = new THREE.ShaderMaterial({ uniforms, vertexShader: vs, fragmentShader: fs, fog: true, depthWrite: true, depthTest: true,
    polygonOffset: true, polygonOffsetFactor: 2, polygonOffsetUnits: 8 });
  m.extensions = { derivatives: true, shaderTextureLOD: true };
  return m;
}

// --------------------------------------------------------------------------------------------- helpers
const halfBuf = new Float32Array(1), halfInt = new Int32Array(halfBuf.buffer);
function toHalf(v) {
  halfBuf[0] = v;
  const x = halfInt[0];
  const sign = (x >>> 16) & 0x8000;
  let e = ((x >>> 23) & 0xff) - 127 + 15, m = x & 0x7fffff;
  if (e <= 0) return sign;
  if (e >= 31) return sign | 0x7c00;
  return sign | (e << 10) | ((m + 0x1000) >>> 13);   // round to nearest
}
async function fetchGz(url) {
  const r = await fetch(url); if (!r.ok) throw new Error(url + ' ' + r.status);
  const ds = r.body.pipeThrough(new DecompressionStream('gzip'));
  return new Uint8Array(await new Response(ds).arrayBuffer());
}
async function fetchBitmap(url) {
  const r = await fetch(url); if (!r.ok) throw new Error(url + ' ' + r.status);
  return createImageBitmap(await r.blob(), { premultiplyAlpha: 'none', colorSpaceConversion: 'none' });
}

/**
 * opts: { U (look uniforms), q (look quality), renderer, scene, terrain (bool), exaggeration, grassTex, addTo, own, clock() }
 * Returns a handle (see bottom). Loads asynchronously; until the data is in, nothing is drawn (the page ground stays).
 */
export function createLandscape(opts) {
  const { U, renderer, scene } = opts;
  const L = {
    uLsHeight: { value: null }, uLsRoot: { value: new THREE.Vector4(-53000, -57900, 102400, 4096) },
    uLsExag: { value: opts.exaggeration ?? 1.0 }, uLsFlatY: { value: opts.terrain === true ? 0 : -0.3 }, uLsOn: { value: opts.terrain === true ? 1 : 0 },
    uHAtlas: { value: null }, uHPage: { value: null }, uHInfo: { value: new THREE.Vector4(32, 3200, 324, 2) }, uHSlots: { value: 8 }, uHOn: { value: 0 },
    uLsShade: { value: opts.shade ?? 1.6 },
    uOuter: { value: null }, uOuterRect: { value: new THREE.Vector4(-155400, -160300, 307200, 0) }, uGlobal: { value: null }, uAtlas: { value: null }, uPage: { value: null }, uShore: { value: null }, uGrass: { value: opts.grassTex || null },
    uPageInfo: { value: new THREE.Vector4(32, 3200, 512, 8) }, uAtlasSlots: { value: 8 },
    uFade: { value: 1 }, uLsDebug: { value: typeof location !== 'undefined' && /[?&]lsdebug=1/.test(location.search) ? 1 : 0 }, uLsShadow: { value: opts.sky === 'golden' ? 0.45 : opts.sky === 'overcast' ? 0 : 0.22 }, uAlbedoGain: { value: opts.albedoGain ?? 0.62 }, uFadeCol: { value: new THREE.Color(0x070a14) }, uLogBias: { value: 0 }, uDetail: { value: opts.q?.groundDetail === false ? 0 : 1 },
  };
  const group = new THREE.Group(); group.name = 'look-landscape';
  const mat = groundMaterial(U, L);
  // CDLOD patch: (M+1)^2 grid, instanced per selected node
  const geo = new THREE.InstancedBufferGeometry();
  {
    const M = GRID_M, g = new Float32Array((M + 1) * (M + 1) * 2), idx = [];
    for (let j = 0; j <= M; j++) for (let i = 0; i <= M; i++) { g[(j * (M + 1) + i) * 2] = i; g[(j * (M + 1) + i) * 2 + 1] = j; }
    for (let j = 0; j < M; j++) for (let i = 0; i < M; i++) {
      const a = j * (M + 1) + i, b = a + 1, c = a + M + 1, d = c + 1;
      // alternate the diagonal so the morph never creates sliver folds
      if ((i + j) & 1) idx.push(a, c, b, b, c, d); else idx.push(a, c, d, a, d, b);
    }
    geo.setAttribute('grid', new THREE.BufferAttribute(g, 2));
    geo.setIndex(idx);
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array((M + 1) * (M + 1) * 3), 3));   // unused (three wants one)
  }
  const MAXN = 2048;
  const nodeAttr = new THREE.InstancedBufferAttribute(new Float32Array(MAXN * 4), 4); nodeAttr.setUsage(THREE.DynamicDrawUsage);
  geo.setAttribute('aNode', nodeAttr);
  geo.instanceCount = 0;
  const ground = new THREE.Mesh(geo, mat);
  ground.name = 'look-landscape-ground'; ground.frustumCulled = false; ground.renderOrder = -3; ground.visible = false;
  group.add(ground);

  // buildings + trees (phase 3)
  const objects = (opts.buildings !== false || opts.trees !== false)
    ? createBuildingsAndTrees({ U, L, renderer, buildings: opts.buildings !== false, trees: opts.trees !== false }) : null;
  if (objects) { objects.group.visible = false; group.add(objects.group); }
  const structures = opts.structures !== false ? createStructures({ U, L, renderer }) : null;
  if (structures) group.add(structures.group);

  // ranges per level
  const sizeOf = (l) => LEAF * Math.pow(2, l);
  const range = (l) => RANGE_K * sizeOf(l);
  for (let l = 0; l < LEVELS; l++) { mat.uniforms.uMorphEnd.value[l] = range(l); mat.uniforms.uMorphStart.value[l] = range(l) * MORPH_START; }

  // ---- data
  let heights = null;        // Float32Array N*N (m) for CPU lookups
  let N = 4096, meta = null, lcMeta = null, maxH = 1400;
  const state = { ready: false, error: null, tiles: new Map(), pending: 0, loads: 0 };
  const atlasSlots = 8, slotPx = 512;
  let atlas = null, page = null, pageData = null;
  const slots = new Array(atlasSlots * atlasSlots).fill(null);   // slot -> tile key
  const tileAvail = new Set();
  // height detail residency
  const hSlots = 8; let hDetail = null, hAtlas = null, hPage = null, hPageData = null;
  const hAvail = new Set(), hState = new Map(), hSlotKey = new Array(hSlots * hSlots).fill(null), hCpu = new Map();
  let hInflight = 0;
  const wantH = new Map();

  let terrainAuto = null;
  const ready = (async () => {
    try {
      const [tm, lm] = await Promise.all([fetch(BASE + 'terrain.json').then((r) => r.json()), fetch(BASE + 'landcover.json').then((r) => r.json())]);
      meta = tm; lcMeta = lm; N = tm.n;
      if (opts.terrain === 'auto') {
        // real road heights (build) -> the ground must follow the relief, or roads float
        const rm = await fetch('data/roads.meta.json').then((r) => (r.ok ? r.json() : null)).catch(() => null);
        const on = !!(rm && rm.elevation && rm.elevation.model === 'terrain/1');
        L.uLsOn.value = on ? 1 : 0; L.uLsFlatY.value = on ? 0 : -0.3; terrainAuto = on;
      }
      L.uLsRoot.value.set(tm.root.x0, tm.root.z0, tm.root.size, N);
      maxH = tm.heightRange[1];
      for (const [tx, tz] of lm.tiles) tileAvail.add(tx + '_' + tz);
      L.uPageInfo.value.set(lm.grid, lm.tile, lm.tileSize, lm.gutter);
      const [hb, sb, gbm, obm, om] = await Promise.all([fetchGz(BASE + 'terrain.bin.gz'), fetchGz(BASE + 'shore.bin.gz'), fetchBitmap(BASE + 'lc-global.png'),
        fetchBitmap(BASE + 'outer.png').catch(() => null), fetch(BASE + 'outer.json').then((r) => (r.ok ? r.json() : null)).catch(() => null)]);
      {
        const ot = new THREE.Texture(obm || gbm);
        ot.colorSpace = THREE.SRGBColorSpace; ot.flipY = false; ot.premultiplyAlpha = false; ot.generateMipmaps = true;
        ot.minFilter = THREE.LinearMipmapLinearFilter; ot.anisotropy = 4; ot.wrapS = ot.wrapT = THREE.ClampToEdgeWrapping; ot.needsUpdate = true;
        L.uOuter.value = ot;
        if (om) L.uOuterRect.value.set(om.x0, om.z0, om.size, 0); else L.uOuterRect.value.set(tm.root.x0, tm.root.z0, tm.root.size, 0);
      }
      // heights: delta-coded Int16 dm -> Float32 (CPU) + half float (GPU)
      const d = new Int16Array(hb.buffer, hb.byteOffset, N * N);
      heights = new Float32Array(N * N);
      const half = new Uint16Array(N * N);
      for (let j = 0; j < N; j++) { let p = 0; for (let i = 0; i < N; i++) { p += d[j * N + i]; const v = p / 10; heights[j * N + i] = v; half[j * N + i] = toHalf(v); } }
      const ht = new THREE.DataTexture(half, N, N, THREE.RedFormat, THREE.HalfFloatType);
      ht.minFilter = THREE.LinearFilter; ht.magFilter = THREE.LinearFilter; ht.generateMipmaps = false; ht.needsUpdate = true;
      L.uLsHeight.value = ht;
      const st = new THREE.DataTexture(sb, tm.shoreN, tm.shoreN, THREE.RedFormat, THREE.UnsignedByteType);
      st.minFilter = THREE.LinearFilter; st.magFilter = THREE.LinearFilter; st.generateMipmaps = false; st.needsUpdate = true;
      L.uShore.value = st;
      const gt = new THREE.Texture(gbm);
      gt.colorSpace = THREE.SRGBColorSpace; gt.flipY = false; gt.premultiplyAlpha = false;
      gt.generateMipmaps = true; gt.minFilter = THREE.LinearMipmapLinearFilter; gt.magFilter = THREE.LinearFilter;
      gt.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy()); gt.wrapS = gt.wrapT = THREE.ClampToEdgeWrapping; gt.needsUpdate = true;
      L.uGlobal.value = gt;
      // detail atlas (allocated once; tiles are copied in with copyTextureToTexture, mips regenerated per copy)
      atlas = new THREE.DataTexture(new Uint8Array(atlasSlots * slotPx * atlasSlots * slotPx * 4), atlasSlots * slotPx, atlasSlots * slotPx, THREE.RGBAFormat, THREE.UnsignedByteType);
      atlas.colorSpace = THREE.SRGBColorSpace; atlas.flipY = false; atlas.premultiplyAlpha = false; atlas.unpackAlignment = 4;
      atlas.generateMipmaps = true; atlas.minFilter = THREE.LinearMipmapLinearFilter; atlas.magFilter = THREE.LinearFilter;
      atlas.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy()); atlas.needsUpdate = true;
      atlas.onUpdate = () => { atlas.image.data = null; };   // free the 64 MB staging copy once uploaded
      L.uAtlas.value = atlas;
      pageData = new Uint8Array(lm.grid * lm.grid * 4);
      page = new THREE.DataTexture(pageData, lm.grid, lm.grid, THREE.RGBAFormat, THREE.UnsignedByteType);
      page.minFilter = THREE.NearestFilter; page.magFilter = THREE.NearestFilter; page.generateMipmaps = false; page.needsUpdate = true;
      L.uPage.value = page;
      renderer.initTexture(atlas);
      // 10 m height detail (road-conformed ground grid) atlas + page table
      if (tm.detail && tm.detail.tiles.length) {
        const D = tm.detail; hDetail = D;
        for (const [tx, tz] of D.tiles) hAvail.add(tx + '_' + tz);
        L.uHInfo.value.set(D.grid, D.tile, D.n, D.gutter);
        hAtlas = new THREE.DataTexture(new Float32Array(hSlots * D.n * hSlots * D.n), hSlots * D.n, hSlots * D.n, THREE.RedFormat, THREE.FloatType);
        hAtlas.minFilter = THREE.NearestFilter; hAtlas.magFilter = THREE.NearestFilter; hAtlas.generateMipmaps = false; hAtlas.flipY = false; hAtlas.unpackAlignment = 4; hAtlas.needsUpdate = true;
        hAtlas.onUpdate = () => { hAtlas.image.data = null; };
        hPageData = new Uint8Array(D.grid * D.grid * 4);
        hPage = new THREE.DataTexture(hPageData, D.grid, D.grid, THREE.RGBAFormat, THREE.UnsignedByteType);
        hPage.minFilter = THREE.NearestFilter; hPage.magFilter = THREE.NearestFilter; hPage.generateMipmaps = false; hPage.needsUpdate = true;
        L.uHAtlas.value = hAtlas; L.uHPage.value = hPage; L.uHOn.value = 1;
        renderer.initTexture(hAtlas);
      } else {
        // the shader samples uHAtlas / uHPage even when off: give it 1x1 placeholders
        const z = new THREE.DataTexture(new Float32Array(1), 1, 1, THREE.RedFormat, THREE.FloatType); z.needsUpdate = true;
        const zp = new THREE.DataTexture(new Uint8Array(4), 1, 1, THREE.RGBAFormat, THREE.UnsignedByteType); zp.needsUpdate = true;
        L.uHAtlas.value = z; L.uHPage.value = zp;
      }
      state.ready = true;
      ground.visible = true;
      if (objects) objects.group.visible = true;
      opts.onReady && opts.onReady();
    } catch (e) { state.error = e; console.warn('landscape: data unavailable (' + e.message + '); keeping the basic ground'); }
  })();

  // ---- CPU height (same Catmull-Rom as the shader: global 25 m + resident 10 m detail), metres (unscaled)
  function crw(t) { const t2 = t * t, t3 = t2 * t; return [-0.5 * t3 + t2 - 0.5 * t, 1.5 * t3 - 2.5 * t2 + 1, -1.5 * t3 + 2 * t2 + 0.5 * t, 0.5 * t3 - 0.5 * t2]; }
  function cr(arr, n, px, pz, x0 = 0, z0 = 0) {
    const ix = Math.floor(px), iz = Math.floor(pz), wx = crw(px - ix), wz = crw(pz - iz);
    let h = 0;
    for (let b = 0; b < 4; b++) {
      const jj = Math.min(n - 1, Math.max(0, iz - 1 + b)) + z0; let row = 0;
      for (let a = 0; a < 4; a++) row += arr[jj * (x0 ? x0 : n) + Math.min(n - 1, Math.max(0, ix - 1 + a))] * wx[a];
      h += row * wz[b];
    }
    return h;
  }
  function heightAt(x, z) {
    if (!heights) return 0;
    const R = L.uLsRoot.value;
    const u = (x - R.x) / R.z, v = (z - R.y) / R.z;
    if (u < 0 || v < 0 || u > 1 || v > 1) return 0;
    let h = cr(heights, N, u * N - 0.5, v * N - 0.5);
    if (hDetail) {
      const g = hDetail.grid, tcx = u * g, tcz = v * g, tx = Math.floor(tcx), tz = Math.floor(tcz);
      const t = hCpu.get(tx + '_' + tz), st = hState.get(tx + '_' + tz);
      if (t && st && st.fade > 0) {
        const n = hDetail.n, per = n - 2 * hDetail.gutter;
        const d = cr(t, n, (tcx - tx) * per + hDetail.gutter - 0.5, (tcz - tz) * per + hDetail.gutter - 0.5);
        h += (d - h) * st.fade;
      }
    }
    const e = Math.min(u, v, 1 - u, 1 - v) * R.z;
    const s2 = Math.min(1, Math.max(0, e / 2000)); return h * s2 * s2 * (3 - 2 * s2);
  }

  // ---- 10 m height tiles: fetch, invert the planar predictor, copy into the float atlas, fade in
  function setHPage(key, slot, a) {
    const [t, u] = key.split('_').map(Number), g = hDetail.grid, o = (u * g + t) * 4;
    hPageData[o] = slot % hSlots; hPageData[o + 1] = Math.floor(slot / hSlots); hPageData[o + 3] = a; hPage.needsUpdate = true;
  }
  async function loadH(key, slot) {
    hInflight++; state.pending++;
    const rec = { slot, fade: 0, loading: true, lastWanted: frameNo }; hState.set(key, rec); hSlotKey[slot] = key;
    try {
      const u8 = await fetchGz(BASE + 'h10/' + key + '.bin.gz');
      if (hState.get(key) !== rec) return;
      const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
      if (dv.getUint32(0, true) !== 0x00303148) throw new Error('bad h10 tile');
      const base = dv.getFloat32(4, true), n = dv.getUint16(8, true), unit = dv.getFloat32(12, true);
      if (n !== hDetail.n) throw new Error('h10 size');
      const q = new Int16Array(u8.buffer.slice(u8.byteOffset + 16, u8.byteOffset + 16 + n * n * 2));
      const v = new Int32Array(n * n), f = new Float32Array(n * n);
      for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
        const k = j * n + i, L0 = i ? v[k - 1] : 0, U0 = j ? v[k - n] : 0, UL = i && j ? v[k - n - 1] : 0;
        const pred = i && j ? L0 + U0 - UL : i ? L0 : U0;
        v[k] = (((q[k] + pred) << 16) >> 16);   // Int16 wrap, as encoded
        f[k] = base + v[k] * unit;
      }
      const src = new THREE.DataTexture(f, n, n, THREE.RedFormat, THREE.FloatType); src.flipY = false; src.unpackAlignment = 4;
      renderer.copyTextureToTexture(new THREE.Vector2((slot % hSlots) * n, Math.floor(slot / hSlots) * n), src, hAtlas);
      hCpu.set(key, f);
      rec.loading = false; setHPage(key, slot, 1);
    } catch (e) { hState.delete(key); hSlotKey[slot] = null; console.warn('landscape h10', key, e.message); }
    finally { hInflight--; state.pending--; }
  }
  function updateHTiles() {
    if (!hDetail) return;
    for (const [k, rec] of hState) if (!rec.loading && rec.fade < 1) { rec.fade = Math.min(1, rec.fade + 1 / 12); setHPage(k, rec.slot, Math.max(1, Math.round(rec.fade * 255))); }
    const want = [...wantH.entries()].sort((a, b) => a[1] - b[1]).slice(0, hSlots * hSlots);
    for (const [k] of want) { const r = hState.get(k); if (r) r.lastWanted = frameNo; }
    for (const [k] of want) {
      if (hState.has(k)) continue;
      if (hInflight >= 3) break;
      let s = hSlotKey.indexOf(null);
      if (s < 0) {
        let best = -1, bt = Infinity;
        for (let i = 0; i < hSlotKey.length; i++) { const r = hState.get(hSlotKey[i]); if (r && !r.loading && r.lastWanted < frameNo && r.lastWanted < bt) { bt = r.lastWanted; best = i; } }
        if (best < 0) break;
        const old = hSlotKey[best]; hState.delete(old); hCpu.delete(old); setHPage(old, 0, 0); hSlotKey[best] = null; s = best;
      }
      loadH(k, s);
    }
  }

  // ---- CDLOD selection
  const frustum = new THREE.Frustum(), pm = new THREE.Matrix4(), box = new THREE.Box3(), v3 = new THREE.Vector3();
  let nSel = 0;
  const wantTiles = new Map();   // key -> priority (distance)
  let pxAngle = 0.0008;
  function boxOf(x0, z0, S) {
    const top = L.uLsOn.value > 0.5 ? maxH * L.uLsExag.value + 2 : 1;
    box.min.set(x0, -2, z0); box.max.set(x0 + S, top, z0 + S);
    return box;
  }
  function distToBox(cam, b) {
    const dx = Math.max(b.min.x - cam.x, 0, cam.x - b.max.x), dy = Math.max(b.min.y - cam.y, 0, cam.y - b.max.y), dz = Math.max(b.min.z - cam.z, 0, cam.z - b.max.z);
    return Math.hypot(dx, dy, dz);
  }
  function add(x0, z0, S, l, d) {
    if (nSel >= MAXN) return;
    const a = nodeAttr.array; a[nSel * 4] = x0; a[nSel * 4 + 1] = z0; a[nSel * 4 + 2] = S; a[nSel * 4 + 3] = l; nSel++;
    // 10 m height detail near the camera (where roads sit on it)
    if (hDetail && L.uLsOn.value > 0.5 && d < 6000) {
      const R = L.uLsRoot.value, T = hDetail.tile;
      const t0 = Math.max(0, Math.floor((x0 - R.x) / T)), t1 = Math.min(hDetail.grid - 1, Math.floor((x0 + S - R.x - 1e-3) / T));
      const u0 = Math.max(0, Math.floor((z0 - R.y) / T)), u1 = Math.min(hDetail.grid - 1, Math.floor((z0 + S - R.y - 1e-3) / T));
      for (let u = u0; u <= u1; u++) for (let t = t0; t <= t1; t++) { const k = t + '_' + u; if (!hAvail.has(k)) continue; const p = wantH.get(k); if (p === undefined || d < p) wantH.set(k, d); }
    }
    // detail land-cover tiles wanted where a pixel covers < ~20 m of ground
    if (d * pxAngle < 20 && lcMeta) {
      const R = L.uLsRoot.value, T = lcMeta.tile;
      const t0 = Math.max(0, Math.floor((x0 - R.x) / T)), t1 = Math.min(lcMeta.grid - 1, Math.floor((x0 + S - R.x - 1e-3) / T));
      const u0 = Math.max(0, Math.floor((z0 - R.y) / T)), u1 = Math.min(lcMeta.grid - 1, Math.floor((z0 + S - R.y - 1e-3) / T));
      for (let u = u0; u <= u1; u++) for (let t = t0; t <= t1; t++) {
        const k = t + '_' + u; if (!tileAvail.has(k)) continue;
        const p = wantTiles.get(k); if (p === undefined || d < p) wantTiles.set(k, d);
      }
    }
  }
  function select(cam, x0, z0, l) {
    const S = sizeOf(l);
    const b = boxOf(x0, z0, S);
    const d = distToBox(cam, b);
    if (!frustum.intersectsBox(b)) return;
    if (l === 0 || d > range(l - 1)) { add(x0, z0, S, l, d); return; }
    const h = S / 2;
    select(cam, x0, z0, l - 1); select(cam, x0 + h, z0, l - 1); select(cam, x0, z0 + h, l - 1); select(cam, x0 + h, z0 + h, l - 1);
  }

  // ---- detail tile residency (atlas slots, LRU, cross-fade in)
  const tileState = new Map();   // key -> {slot, fade, lastWanted, loading}
  let frameNo = 0, inflight = 0;
  function setPage(key, slot, a) {
    const [t, u] = key.split('_').map(Number), g = lcMeta.grid, o = (u * g + t) * 4;
    pageData[o] = slot % atlasSlots; pageData[o + 1] = Math.floor(slot / atlasSlots); pageData[o + 2] = 0; pageData[o + 3] = a;
    page.needsUpdate = true;
  }
  function freeSlot() {
    let s = slots.indexOf(null);
    if (s >= 0) return s;
    // evict the least recently wanted resident tile not wanted this frame
    let best = -1, bestT = Infinity;
    for (let i = 0; i < slots.length; i++) {
      const ts = tileState.get(slots[i]);
      if (ts && ts.lastWanted < frameNo && ts.lastWanted < bestT) { bestT = ts.lastWanted; best = i; }
    }
    if (best < 0) return -1;
    const k = slots[best]; const ts = tileState.get(k); tileState.delete(k); setPage(k, 0, 0); slots[best] = null;
    return best;
  }
  async function loadTile(key, slot) {
    inflight++; state.pending++;
    const ts = { slot, fade: 0, t0: 0, lastWanted: frameNo, loading: true };
    tileState.set(key, ts); slots[slot] = key;
    try {
      const bm = await fetchBitmap(BASE + 'lc/' + key + '.png');
      if (tileState.get(key) !== ts) { bm.close && bm.close(); return; }
      const src = new THREE.Texture(bm); src.flipY = false; src.premultiplyAlpha = false;
      renderer.copyTextureToTexture(new THREE.Vector2((slot % atlasSlots) * slotPx, Math.floor(slot / atlasSlots) * slotPx), src, atlas);
      bm.close && bm.close();
      ts.loading = false; state.loads++;
      setPage(key, slot, 1);
    } catch (e) { tileState.delete(key); slots[slot] = null; console.warn('landscape tile', key, e.message); }
    finally { inflight--; state.pending--; }
  }
  function updateTiles() {
    // fades
    // cross-fade in over ~12 rendered frames (frame-counted, so frame-exact captures are deterministic)
    for (const [k, ts] of tileState) if (!ts.loading && ts.fade < 1) {
      ts.fade = Math.min(1, ts.fade + 1 / (opts.fadeFrames ?? 12));
      setPage(k, ts.slot, Math.max(1, Math.round(ts.fade * 255)));
    }
    const want = [...wantTiles.entries()].sort((a, b) => a[1] - b[1]).slice(0, atlasSlots * atlasSlots);
    for (const [k] of want) { const ts = tileState.get(k); if (ts) ts.lastWanted = frameNo; }
    for (const [k] of want) {
      if (tileState.has(k)) continue;
      if (inflight >= 4) break;
      const s = freeSlot(); if (s < 0) break;
      loadTile(k, s);
    }
  }

  function update(camera, vpH) {
    if (!state.ready) return;
    frameNo++;
    camera.updateMatrixWorld();
    pm.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    frustum.setFromProjectionMatrix(pm);
    pxAngle = 2 * Math.tan((camera.fov || 50) * Math.PI / 360) / Math.max(200, vpH || 1080);
    nSel = 0; wantTiles.clear(); wantH.clear();
    const cam = camera.position, R = L.uLsRoot.value, top = LEVELS - 1, S = sizeOf(top);
    // 3x3 super-root around the data square (ground continues past it: ocean west, flat land elsewhere)
    const cx = R.x + R.z / 2, cz = R.y + R.z / 2;
    select(cam, cx - S / 2, cz - S / 2, top);
    nodeAttr.needsUpdate = true; nodeAttr.addUpdateRange ? null : null;
    geo.instanceCount = nSel;
    updateTiles();
    updateHTiles();
    if (objects) objects.update(camera);
    // log-depth bias: push the ground back by ~0.15 % of its distance
    const far = camera.far || 400000;
    L.uLogBias.value = Math.log2(1.0015) * (2.0 / Math.log2(far + 1.0)) * 0.5;
  }

  function settled(timeoutMs = 20000) {
    const t0 = performance.now();
    return new Promise((res) => {
      const tick = () => {
        let fading = false; for (const ts of tileState.values()) if (ts.loading || ts.fade < 1) fading = true; for (const r of hState.values()) if (r.loading || r.fade < 1) fading = true;
        if ((state.ready || state.error) && !state.pending && !fading && !(objects && objects.pending)) return res(true);
        if (performance.now() - t0 > timeoutMs) return res(false);
        setTimeout(tick, 30);
      };
      ready.then(tick);
    });
  }

  return {
    group, ground, material: mat, uniforms: L, ready, update, settled, heightAt,
    get isReady() { return state.ready; },
    /** 0..1: 1 = full landscape, 0 = faded to the fade colour (dark by default). */
    setFade(k, col) { L.uFade.value = Math.max(0, Math.min(1, k)); if (col != null) L.uFadeCol.value.set(col); if (objects) objects.setFade(L.uFade.value); if (structures) structures.setFade(L.uFade.value); },
    objects,
    /** Hide landscape buildings / trees in oriented rectangles (see buildings.js setExclusion), e.g. for src/look/hero-block.js. */
    setExclusion(list) { if (objects) objects.setExclusion(list); },
    setTerrain(on, exag) { L.uLsOn.value = on ? 1 : 0; L.uLsFlatY.value = on ? 0 : -0.3; if (exag != null) L.uLsExag.value = exag; },
    get terrainMeta() { return meta; },
    get terrainOn() { return L.uLsOn.value > 0.5; },
    debugNodes() { const a = nodeAttr.array, o = []; for (let i = 0; i < nSel; i++) o.push([a[i * 4], a[i * 4 + 1], a[i * 4 + 2], a[i * 4 + 3]]); return { nodes: o, want: [...wantTiles.entries()], avail: tileAvail.size }; },
    stats() { return { nodes: nSel, wanted: wantTiles.size, pxAngle: +pxAngle.toFixed(5), tiles: [...tileState.values()].filter((t) => !t.loading).length, loads: state.loads, pending: state.pending, ...(objects ? objects.stats() : {}) }; },
    dispose() { if (objects) objects.dispose(); if (structures) structures.dispose(); geo.dispose(); mat.dispose(); for (const k of ['uLsHeight', 'uGlobal', 'uAtlas', 'uPage', 'uShore']) L[k].value && L[k].value.dispose(); },
  };
}
