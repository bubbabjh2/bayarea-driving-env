// Buildings + trees of the landscape layer (render only). Created by landscape.js.
//   buildings: OSM footprints extruded in a worker (landscape-worker.js), one mesh per 1.6 km tile, loaded within
//              ~3.4 km of the camera; each building dissolves individually (hash threshold + 4x4 dither) between
//              2.4 and 3.2 km, so there is no per-building cost further out (the land-cover texture already
//              carries the roofs as a flat tint) and no popping.
//   trees:     instanced low-poly blobs / cones (two draws) from data/landscape/trees/*, within ~1.7 km.
// Both follow the terrain when relief is on (anchor height per building / tree, so nothing shears).
import * as THREE from 'three';
import { GLSL_COMMON } from './materials.js';
import { GLSL_TERRAIN } from './terrain-glsl.js';

const BASE = 'data/landscape/';
const HEAD_V = /* glsl */`
#include <common>
#include <fog_pars_vertex>
#include <logdepthbuf_pars_vertex>
`;
const HEAD_F = /* glsl */`
#include <common>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
`;
const TAIL_F = /* glsl */`
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
  #include <fog_fragment>
`;
const BAYER = /* glsl */`
float ls_bayer(vec2 p) {
  ivec2 q = ivec2(mod(p, 4.0));
  int i = q.x + q.y * 4;
  int b[16] = int[16](0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5);
  return (float(b[i]) + 0.5) / 16.0;
}`;

function buildingMaterial(U, L, P) {
  const vs = HEAD_V + GLSL_TERRAIN + /* glsl */`

uniform vec4 uExA[2], uExB[2]; uniform float uExN;
bool ls_excluded(vec2 p) {
  for (int i = 0; i < 2; i++) {
    if (float(i) >= uExN) break;
    vec2 d = p - uExA[i].xy, al = vec2(uExA[i].z, -uExA[i].w), rt = vec2(uExA[i].w, uExA[i].z);
    float s = dot(d, al), t = dot(d, rt);
    if (s >= uExB[i].x && s <= uExB[i].y && t >= uExB[i].z && t <= uExB[i].w) return true;
  }
  return false;
}
attribute vec4 info;      // anchor x, z (tile local), hash, height
attribute vec4 color;     // sRGB wall / roof colour, a: 0 wall 1 roof
uniform float uLsOn, uBFadeNear, uBFadeFar, uBFade;
varying vec3 vW; varying vec3 vRel; varying vec3 vCol; varying float vRoof; varying float vY; varying float vA; varying float vH;
void main() {
  vec4 w = modelMatrix * vec4(position, 1.0);
  vec2 anchor = (modelMatrix * vec4(info.x, 0.0, info.y, 1.0)).xz;
  float base = uLsOn > 0.5 ? ls_heightRaw(anchor) * uLsExag + uLsFlatY : 0.0;
  w.y += base;
  float d = length(vec3(anchor.x, base, anchor.y) - cameraPosition);
  float a = (1.0 - smoothstep(uBFadeNear, uBFadeFar, d)) * uBFade;
  vA = clamp((a - info.z * 0.9) * 8.0, 0.0, 1.0);   // each building dissolves on its own (no wave front)
  if (ls_excluded(anchor)) vA = 0.0;
  vCol = pow(color.rgb, vec3(2.2)); vRoof = color.a; vY = position.y; vH = info.w;
  vW = w.xyz; vRel = w.xyz - cameraPosition;
  vec4 mvPosition = viewMatrix * w;
  gl_Position = vA <= 0.0 ? vec4(2.0, 2.0, 2.0, 1.0) : projectionMatrix * mvPosition;
  #include <logdepthbuf_vertex>
  #include <fog_vertex>
}`;
  const fs = HEAD_F + GLSL_COMMON + BAYER + /* glsl */`
uniform float uGain;
varying vec3 vW; varying vec3 vRel; varying vec3 vCol; varying float vRoof; varying float vY; varying float vA; varying float vH;
void main() {
  if (vA < 1.0 && vA <= ls_bayer(gl_FragCoord.xy)) discard;
  #include <logdepthbuf_fragment>
  vec3 cr = cross(dFdx(vW), dFdy(vW)); vec3 n = dot(cr, cr) > 1e-12 ? normalize(cr) : vec3(0.0, 1.0, 0.0);
  if (dot(n, vRel) > 0.0) n = -n;
  vec3 alb = vCol * uGain;
  // clay AO: walls darken toward the street, roofs a touch lighter at the rim
  float ao = vRoof > 0.5 ? 1.0 : mix(0.68, 1.0, smoothstep(-0.5, min(6.0, vH * 0.6 + 1.0), vY));
  // tall buildings: very faint floor bands for scale (band-limited)
  float fw = fwidth(vY);
  float band = vH > 24.0 && vRoof < 0.5 ? (1.0 - smoothstep(0.08, 0.08 + fw * 1.5, abs(fract(vY / 3.6) - 0.5) * 2.0 - 0.8)) * (1.0 - smoothstep(0.3, 0.9, fw)) : 0.0;
  alb *= ao * (1.0 - 0.06 * band);
  vec3 sunH = normalize(vec3(uSunDir.x, max(uSunDir.y, 0.2), uSunDir.z));
  float ndl = clamp((dot(n, sunH) + 0.15) / 1.15, 0.0, 1.0);
  vec3 col = alb * (uSunIrr * ndl * 0.95 + mix(uGroundIrr, uSkyIrr, 0.5 + 0.5 * n.y) * 0.8);
  gl_FragColor = vec4(col, 1.0);
` + TAIL_F + '}';
  const m = new THREE.ShaderMaterial({
    uniforms: Object.assign(THREE.UniformsUtils.clone(THREE.UniformsLib.fog), U, L, P),
    vertexShader: vs, fragmentShader: fs, fog: true, side: THREE.FrontSide,
  });
  m.extensions = { derivatives: true };
  return m;
}

function treeMaterial(U, L, P) {
  const vs = HEAD_V + GLSL_TERRAIN + /* glsl */`

uniform vec4 uExA[2], uExB[2]; uniform float uExN;
bool ls_excluded(vec2 p) {
  for (int i = 0; i < 2; i++) {
    if (float(i) >= uExN) break;
    vec2 d = p - uExA[i].xy, al = vec2(uExA[i].z, -uExA[i].w), rt = vec2(uExA[i].w, uExA[i].z);
    float s = dot(d, al), t = dot(d, rt);
    if (s >= uExB[i].x && s <= uExB[i].y && t >= uExB[i].z && t <= uExB[i].w) return true;
  }
  return false;
}
attribute vec4 aTree;     // world x, z, canopy radius, kind + hash (kind in the integer part)
attribute float part;     // 0 trunk, 1 canopy
uniform float uLsOn, uTFadeNear, uTFadeFar, uBFade;
varying vec3 vW; varying vec3 vRel; varying vec3 vN; varying float vPart; varying float vA; varying float vHash; varying float vKind;
void main() {
  float kind = floor(aTree.w), hsh = fract(aTree.w);
  vec2 xz = aTree.xy;
  float base = uLsOn > 0.5 ? ls_heightRaw(xz) * uLsExag + uLsFlatY : uLsFlatY;
  float r = aTree.z;
  vec3 p = position * r;
  if (kind > 1.5) { p *= 0.9; if (part < 0.5) p *= 0.0; p.y -= r * 0.9; }   // bush: no trunk, sits low
  // a little per-tree variety: squash / stretch and a random turn
  float ang = hsh * 6.2832, c = cos(ang), s = sin(ang);
  p.xz = mat2(c, -s, s, c) * p.xz;
  p.y *= 0.9 + 0.3 * fract(hsh * 7.31);
  vec4 w = vec4(xz.x + p.x, base + p.y, xz.y + p.z, 1.0);
  float d = length(vec3(xz.x, base, xz.y) - cameraPosition);
  float a = (1.0 - smoothstep(uTFadeNear, uTFadeFar, d)) * uBFade;
  vA = clamp((a - hsh * 0.9) * 8.0, 0.0, 1.0);
  if (ls_excluded(xz)) vA = 0.0;
  vN = normal; vPart = part; vHash = hsh; vKind = kind;
  vW = w.xyz; vRel = w.xyz - cameraPosition;
  vec4 mvPosition = viewMatrix * w;
  gl_Position = vA <= 0.0 ? vec4(2.0, 2.0, 2.0, 1.0) : projectionMatrix * mvPosition;
  #include <logdepthbuf_vertex>
  #include <fog_vertex>
}`;
  const fs = HEAD_F + GLSL_COMMON + BAYER + /* glsl */`
uniform vec3 uLeafA, uLeafB, uLeafC, uConifer, uBush, uTrunk; uniform float uGain;
varying vec3 vW; varying vec3 vRel; varying vec3 vN; varying float vPart; varying float vA; varying float vHash; varying float vKind;
void main() {
  if (vA < 1.0 && vA <= ls_bayer(gl_FragCoord.xy)) discard;
  #include <logdepthbuf_fragment>
  vec3 cr = cross(dFdx(vW), dFdy(vW)); vec3 n = dot(cr, cr) > 1e-12 ? normalize(cr) : vec3(0.0, 1.0, 0.0);
  if (dot(n, vRel) > 0.0) n = -n;
  vec3 leaf = vHash < 0.33 ? mix(uLeafA, uLeafB, vHash * 3.0) : vHash < 0.66 ? mix(uLeafB, uLeafC, vHash * 3.0 - 1.0) : mix(uLeafC, uLeafA, vHash * 3.0 - 2.0);
  vec3 alb = vPart < 0.5 ? uTrunk : vKind > 1.5 ? uBush : vKind > 0.5 ? uConifer : leaf;
  alb *= uGain * mix(0.75, 1.0, smoothstep(-0.6, 0.6, n.y));   // soft self-shadow underneath
  vec3 sunH = normalize(vec3(uSunDir.x, max(uSunDir.y, 0.2), uSunDir.z));
  float ndl = clamp((dot(n, sunH) + 0.4) / 1.4, 0.0, 1.0);
  vec3 col = alb * (uSunIrr * ndl * 0.8 + mix(uGroundIrr, uSkyIrr, 0.5 + 0.5 * n.y) * 1.1);
  gl_FragColor = vec4(col, 1.0);
` + TAIL_F + '}';
  const C = (h) => new THREE.Color(h);
  const m = new THREE.ShaderMaterial({
    uniforms: Object.assign(THREE.UniformsUtils.clone(THREE.UniformsLib.fog), U, L, P, {
      uLeafA: { value: C(0x6f9a58) }, uLeafB: { value: C(0x82a95f) }, uLeafC: { value: C(0x5e8a50) },
      uConifer: { value: C(0x4c7a4c) }, uBush: { value: C(0x7f9a5c) }, uTrunk: { value: C(0x7a6450) },
    }),
    vertexShader: vs, fragmentShader: fs, fog: true,
  });
  m.extensions = { derivatives: true };
  return m;
}

function treeGeometry(kind) {
  // unit canopy radius; trunk + canopy merged, flat-shaded low poly (clay)
  const trunk = new THREE.CylinderGeometry(0.11, 0.15, kind === 1 ? 1.4 : 1.6, 6, 1);
  trunk.translate(0, kind === 1 ? -0.0 : 0.1, 0);
  let canopy;
  if (kind === 1) { canopy = new THREE.ConeGeometry(0.85, 3.0, 7, 1); canopy.translate(0, 2.0, 0); }
  else { canopy = new THREE.IcosahedronGeometry(1, 0); canopy.scale(1, 0.88, 1); canopy.translate(0, 1.75, 0); }
  const ni = (g) => (g.index ? g.toNonIndexed() : g);
  const parts = [[ni(trunk), 0], [ni(canopy), 1]];
  let n = 0; for (const [g] of parts) n += g.attributes.position.count;
  const pos = new Float32Array(n * 3), part = new Float32Array(n);
  let o = 0;
  for (const [g, p] of parts) { pos.set(g.attributes.position.array, o * 3); part.fill(p, o, o + g.attributes.position.count); o += g.attributes.position.count; g.dispose(); }
  const geo = new THREE.InstancedBufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('part', new THREE.BufferAttribute(part, 1));
  geo.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
  return geo;
}

export function createBuildingsAndTrees({ U, L, renderer, buildings = true, trees = true, onPending }) {
  const group = new THREE.Group(); group.name = 'look-landscape-objects';
  const P = { uExA: { value: [new THREE.Vector4(), new THREE.Vector4()] }, uExB: { value: [new THREE.Vector4(), new THREE.Vector4()] }, uExN: { value: 0 }, uBFadeNear: { value: 2400 }, uBFadeFar: { value: 3200 }, uTFadeNear: { value: 1200 }, uTFadeFar: { value: 1700 }, uBFade: { value: 1 }, uGain: { value: 0.66 } };
  const bMat = buildingMaterial(U, L, P);
  const tMat = treeMaterial(U, L, P);
  const worker = new Worker(new URL('./landscape-worker.js', import.meta.url), { type: 'module' });
  let msgId = 0; const waiting = new Map();
  worker.onmessage = (ev) => { const w = waiting.get(ev.data.id); if (w) { waiting.delete(ev.data.id); w(ev.data); } };
  const job = (m) => new Promise((res) => { const id = ++msgId; waiting.set(id, res); worker.postMessage({ ...m, id }); });
  let bMeta = null, tMeta = null;
  const bAvail = new Map(), tAvail = new Map();
  const ready = Promise.all([
    buildings ? fetch(BASE + 'buildings.json').then((r) => (r.ok ? r.json() : null)).catch(() => null) : null,
    trees ? fetch(BASE + 'landcover.json').then((r) => (r.ok ? r.json() : null)).catch(() => null) : null,
  ]).then(([b, t]) => {
    bMeta = b; if (b) for (const [tx, tz, n] of b.tiles) bAvail.set(tx + '_' + tz, n);
    tMeta = t && t.trees ? t : null; if (tMeta) for (const [tx, tz, n] of tMeta.trees.tiles) tAvail.set(tx + '_' + tz, n);
  });

  // ---- buildings: tile meshes
  const bTiles = new Map();   // key -> {mesh|null, loading, last}
  let pending = 0, frame = 0;
  const maxTerrain = () => (L.uLsOn.value > 0.5 ? 1400 * L.uLsExag.value : 0);
  function loadB(key) {
    const [tx, tz] = key.split('_').map(Number), T = bMeta.tile;
    const cx = bMeta.root.x0 + (tx + 0.5) * T, cz = bMeta.root.z0 + (tz + 0.5) * T;
    const rec = { mesh: null, loading: true, last: frame }; bTiles.set(key, rec); pending++; onPending && onPending(1);
    job({ kind: 'bld', url: new URL(BASE + 'bld/' + key + '.bin.gz', document.baseURI).href, quant: bMeta.quant }).then((r) => {
      pending--; onPending && onPending(-1);
      rec.loading = false;
      if (!r.ok || r.empty || bTiles.get(key) !== rec) { if (!r.ok) console.warn('landscape buildings', key, r.error); return; }
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(r.pos, 3));
      g.setAttribute('color', new THREE.BufferAttribute(r.col, 4, true));
      g.setAttribute('info', new THREE.BufferAttribute(r.info, 4));
      g.setIndex(new THREE.BufferAttribute(r.index, 1));
      const mt = maxTerrain();
      g.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, mt / 2 + 100, 0), Math.SQRT2 * T / 2 + 60 + mt / 2 + 250);
      const m = new THREE.Mesh(g, bMat);
      m.name = 'look-buildings'; m.position.set(cx, 0, cz); m.matrixAutoUpdate = false; m.updateMatrix(); m.renderOrder = 0;
      rec.mesh = m; group.add(m);
    });
  }
  function updateBuildings(cam) {
    if (!bMeta) return;
    const T = bMeta.tile, R = bMeta.root, loadR = 3500, keepR = 4600;
    const camH = Math.max(0, cam.y - (L.uLsOn.value > 0.5 ? 0 : 0));
    if (camH > 3400 || P.uBFade.value <= 0) {
      // nothing within reach: keep what's loaded (cheap: vertex shader culls) but load nothing new
    } else {
      const r = Math.sqrt(Math.max(0, loadR * loadR - camH * camH));
      const t0 = Math.floor((cam.x - r - R.x0) / T), t1 = Math.floor((cam.x + r - R.x0) / T);
      const u0 = Math.floor((cam.z - r - R.z0) / T), u1 = Math.floor((cam.z + r - R.z0) / T);
      const want = [];
      for (let u = u0; u <= u1; u++) for (let t = t0; t <= t1; t++) {
        const k = t + '_' + u; if (!bAvail.has(k)) continue;
        const cx = R.x0 + (t + 0.5) * T, cz = R.z0 + (u + 0.5) * T;
        const dx = Math.max(0, Math.abs(cam.x - cx) - T / 2), dz = Math.max(0, Math.abs(cam.z - cz) - T / 2);
        const d = Math.hypot(dx, dz, camH); if (d > loadR) continue;
        const rec = bTiles.get(k); if (rec) { rec.last = frame; continue; }
        want.push([d, k]);
      }
      want.sort((a, b) => a[0] - b[0]);
      for (const [, k] of want) { if (pending >= 3) break; loadB(k); }
    }
    // unload far tiles
    for (const [k, rec] of bTiles) {
      if (rec.loading) continue;
      const [t, u] = k.split('_').map(Number);
      const cx = R.x0 + (t + 0.5) * T, cz = R.z0 + (u + 0.5) * T;
      const d = Math.hypot(Math.max(0, Math.abs(cam.x - cx) - T / 2), Math.max(0, Math.abs(cam.z - cz) - T / 2), Math.max(0, cam.y));
      if (d > keepR) { if (rec.mesh) { group.remove(rec.mesh); rec.mesh.geometry.dispose(); } bTiles.delete(k); }
    }
  }

  // ---- trees: decoded tiles, gathered into two instanced meshes around the camera
  const tTiles = new Map();   // key -> {data|null, loading}
  const CAP = 160000;
  const mk = (kind) => {
    const geo = treeGeometry(kind);
    const attr = new THREE.InstancedBufferAttribute(new Float32Array(CAP * 4), 4); attr.setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute('aTree', attr); geo.instanceCount = 0;
    const m = new THREE.Mesh(geo, tMat); m.frustumCulled = false; m.name = 'look-trees-' + kind; group.add(m);
    return { geo, attr, mesh: m };
  };
  const round = trees ? mk(0) : null, cone = trees ? mk(1) : null;
  let lastGather = null, treesDirty = false;
  function loadT(key) {
    const rec = { data: null, loading: true }; tTiles.set(key, rec); pending++; onPending && onPending(1);
    job({ kind: 'tree', url: new URL(BASE + 'trees/' + key + '.bin.gz', document.baseURI).href }).then((r) => {
      pending--; onPending && onPending(-1); rec.loading = false;
      if (!r.ok || r.empty || tTiles.get(key) !== rec) return;
      const [tx, tz] = key.split('_').map(Number), T = tMeta.tile;
      rec.data = r.trees; rec.cx = tMeta.root.x0 + (tx + 0.5) * T; rec.cz = tMeta.root.z0 + (tz + 0.5) * T; treesDirty = true;
    });
  }
  function updateTrees(cam) {
    if (!tMeta || !round) return;
    const T = tMeta.tile, R = tMeta.root, loadR = 2000;
    const camH = Math.max(0, cam.y);
    if (camH < 1900 && P.uBFade.value > 0) {
      const r = Math.sqrt(Math.max(0, loadR * loadR - camH * camH));
      const t0 = Math.floor((cam.x - r - R.x0) / T), t1 = Math.floor((cam.x + r - R.x0) / T);
      const u0 = Math.floor((cam.z - r - R.z0) / T), u1 = Math.floor((cam.z + r - R.z0) / T);
      for (let u = u0; u <= u1; u++) for (let t = t0; t <= t1; t++) {
        const k = t + '_' + u; if (!tAvail.has(k) || tTiles.has(k)) continue;
        if (pending < 4) loadT(k);
      }
    }
    for (const [k, rec] of tTiles) if (!rec.loading && rec.data && (Math.abs(cam.x - rec.cx) > T * 1.6 || Math.abs(cam.z - rec.cz) > T * 1.6)) { tTiles.delete(k); treesDirty = true; }
    const moved = !lastGather || Math.hypot(cam.x - lastGather.x, cam.z - lastGather.z) > 120 || Math.abs(cam.y - lastGather.y) > 150;
    if (!moved && !treesDirty) return;
    lastGather = { x: cam.x, y: cam.y, z: cam.z }; treesDirty = false;
    const rr = (P.uTFadeFar.value + 250) ** 2;
    let nr = 0, nc = 0; const ar = round.attr.array, ac = cone.attr.array;
    for (const rec of tTiles.values()) {
      if (!rec.data) continue;
      const d = rec.data;
      for (let i = 0; i < d.length; i += 4) {
        const x = rec.cx + d[i], z = rec.cz + d[i + 1];
        const dx = x - cam.x, dz = z - cam.z; if (dx * dx + dz * dz + cam.y * cam.y > rr) continue;
        const kind = d[i + 3], h = ((Math.imul((x * 8) | 0, 73856093) ^ Math.imul((z * 8) | 0, 19349663)) >>> 0) / 4294967296;
        const tgt = kind === 1 ? ac : ar; const k = kind === 1 ? nc++ : nr++;
        if (k >= CAP) continue;
        tgt[k * 4] = x; tgt[k * 4 + 1] = z; tgt[k * 4 + 2] = d[i + 2]; tgt[k * 4 + 3] = kind + Math.min(0.999, h);
      }
    }
    round.geo.instanceCount = Math.min(nr, CAP); cone.geo.instanceCount = Math.min(nc, CAP);
    round.attr.needsUpdate = true; cone.attr.needsUpdate = true;
  }

  function update(cam) {
    frame++;
    if (buildings) updateBuildings(cam.position);
    if (trees) updateTrees(cam.position);
  }
  return {
    group, ready, update, uniforms: P,
    get pending() { return pending; },
    setFade(k) { P.uBFade.value = k; },
    /** Hide buildings / trees inside up to 2 oriented rectangles: [{x, z, heading (deg, compass), s0, s1, t0, t1}] or null.
     *  along = (sin h, -cos h), right = (cos h, sin h) in world x/z; s / t ranges in metres from (x, z). */
    setExclusion(list) {
      const l = (list || []).slice(0, 2); P.uExN.value = l.length;
      l.forEach((r, i) => { const h = (r.heading || 0) * Math.PI / 180; P.uExA.value[i].set(r.x, r.z, Math.sin(h), Math.cos(h)); P.uExB.value[i].set(r.s0, r.s1, r.t0, r.t1); });
    },
    stats() {
      let tris = 0, nb = 0; for (const r of bTiles.values()) if (r.mesh) { nb++; tris += r.mesh.geometry.index.count / 3; }
      return { buildingMeta: bAvail.size, buildingReq: bTiles.size, treeMeta: tAvail.size, buildingTiles: nb, buildingTris: tris, treeInstances: (round ? round.geo.instanceCount : 0) + (cone ? cone.geo.instanceCount : 0) };
    },
    dispose() { worker.terminate(); for (const r of bTiles.values()) r.mesh && r.mesh.geometry.dispose(); bMat.dispose(); tMat.dispose(); round && round.geo.dispose(); cone && cone.geo.dispose(); },
  };
}
