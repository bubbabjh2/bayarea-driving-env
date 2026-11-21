// Look pack: runtime materials/lighting for the Bay Area world.
//
//   import { applyLook } from './src/look/index.js';
//   const look = applyLook(scene, renderer, { preset: 'video' });   // or 'perf'
//
// Strictly opt-in and lazy (SPEC "Rendering vs training"): importing this file
// does nothing; textures are generated inside applyLook(). Training / headless
// sim code never imports it. look.remove() restores the original materials.
//
// Objects are recognised by name / userData.lookRole (see README.md role table),
// never by vertex positions, so the pack survives road-geometry rebuilds. The
// scene is re-scanned every 30 frames, so meshes added later (local tier, cars,
// streamed tiles) are picked up automatically.
import * as THREE from 'three';
import * as packMod from '../roadpack.js';      // read-only: CLASSES_ALL order
import { CLASS_STYLE } from '../stream/classStyle.js';   // stream tile vertex colours -> class matching
import { makeAsphaltCanvas, makeGrassCanvas, blurMask, accumulateDensity, densityFromGrid } from './textures.js';
import * as M from './materials.js';
import { createSignals } from './signals.js';
import { createLandscape } from './landscape.js';

export const PRESETS = {
  // best quality that holds 60 fps with ~20k visible cars on Apple Silicon
  video: { asphaltSize: 1024, anisotropy: 8, groundDetail: true, groundOctaves: 4, waterWaves: 5, carShadows: true, carRim: true, carMaterial: 'standard', localRibbons: true, signals: true },
  // 100k cars: cheaper car shading, no blob shadows, no extra local-street ribbons
  perf: { asphaltSize: 512, anisotropy: 4, groundDetail: false, groundOctaves: 2, waterWaves: 2, carShadows: false, carRim: false, carMaterial: 'lambert', localRibbons: false, signals: true },
};

const FALLBACK_CLASSES = ['motorway', 'motorway_link', 'trunk', 'trunk_link', 'primary', 'primary_link',
  'secondary', 'secondary_link', 'tertiary', 'tertiary_link', 'unclassified', 'residential', 'living_street'];
const FALLBACK_STYLE = {
  motorway: 0xffb340, motorway_link: 0xc98a2c, trunk: 0xff7d6b, trunk_link: 0xb55a4d, primary: 0xe8e2c8,
  primary_link: 0x9c9888, secondary: 0x7f8bb0, secondary_link: 0x5e6886, tertiary: 0x55617f, tertiary_link: 0x55617f,
  unclassified: 0x3a435c, residential: 0x363e56, living_street: 0x2f364c,
};
// linear albedo of each class's asphalt: freeways dark and fresh, local streets lighter and greyer
const TONES = [0.10, 0.11, 0.11, 0.115, 0.125, 0.125, 0.135, 0.135, 0.15, 0.15, 0.155, 0.16, 0.17];
// local-street ribbon width (m) by class when built from the local lines
const LOCAL_WIDTH = [22, 7, 18, 7, 14, 7, 12, 7, 10, 7, 6.5, 8, 5];

// ------------------------------------------------------------------ time of day
// key frames by sun elevation (deg): sky colours (display sRGB), sun colour + irradiance
const SKY_KEYS = [
  { el: -4, zen: 0x34466e, hor: 0xd8a993, sun: 0xff9a6a, sunI: 0.0, skyI: 0.30, glow: 0xff8a50 },
  { el: 4,  zen: 0x4a6ea8, hor: 0xf1c6a0, sun: 0xffad70, sunI: 1.0, skyI: 0.48, glow: 0xffb070 },
  { el: 16, zen: 0x4c84c6, hor: 0xe3e0d8, sun: 0xffe4c4, sunI: 1.9, skyI: 0.62, glow: 0xffd8a0 },
  { el: 45, zen: 0x3b7bcf, hor: 0xcadcea, sun: 0xfff5e8, sunI: 2.3, skyI: 0.72, glow: 0xfff0d8 },
];
/** Sim clock hour -> sun elevation/azimuth (deg). 5:30 dawn ... 10:00 high morning. */
export function sunForHour(h) {
  const el = -4 + (h - 5.5) * 10.4;
  const az = 72 + (h - 5.5) * 13;          // ENE at dawn -> SE by 10:00 (clockwise from north)
  return { el: Math.max(-6, Math.min(50, el)), az };
}

function lerpKeys(el) {
  let a = SKY_KEYS[0], b = SKY_KEYS[SKY_KEYS.length - 1];
  for (let i = 0; i < SKY_KEYS.length - 1; i++) if (el >= SKY_KEYS[i].el && el <= SKY_KEYS[i + 1].el) { a = SKY_KEYS[i]; b = SKY_KEYS[i + 1]; }
  const t = b.el === a.el ? 0 : Math.max(0, Math.min(1, (el - a.el) / (b.el - a.el)));
  const c = (x, y) => new THREE.Color(x).lerp(new THREE.Color(y), t);
  return { zen: c(a.zen, b.zen), hor: c(a.hor, b.hor), sun: c(a.sun, b.sun), glow: c(a.glow, b.glow),
    sunI: a.sunI + (b.sunI - a.sunI) * t, skyI: a.skyI + (b.skyI - a.skyI) * t };
}

// ------------------------------------------------------------------ helpers
function classTable() {
  const names = packMod.CLASSES_ALL || FALLBACK_CLASSES;
  const style = CLASS_STYLE;
  return names.map((n) => {
    const hex = style[n] ? style[n].color : FALLBACK_STYLE[n] ?? 0x808080;
    const c = new THREE.Color(hex);
    return [c.r, c.g, c.b];
  });
}

/** Per-vertex class id (+16 = tunnel) from vertex colours. Distinct colours are cached. */
function classFromColors(colorAttr, table, stride = 1) {
  const n = colorAttr.count, a = colorAttr.array, out = new Float32Array(n);
  const cache = new Map();
  const dims = [[1, 0], [0.3, 0], [0.35, 16], [0.105, 16]];
  for (let i = 0; i < n; i += stride) {
    const r = a[i * 3], g = a[i * 3 + 1], b = a[i * 3 + 2];
    const key = ((r * 4096) | 0) + ':' + ((g * 4096) | 0) + ':' + ((b * 4096) | 0);
    let v = cache.get(key);
    if (v === undefined) {
      let best = Infinity; v = 0;
      for (let c = 0; c < table.length; c++) for (const [d, add] of dims) {
        const t = table[c];
        const e = Math.abs(r - t[0] * d) + Math.abs(g - t[1] * d) + Math.abs(b - t[2] * d);
        const rel = e / (d * (t[0] + t[1] + t[2]) + 1e-6);
        if (rel < best) { best = rel; v = c + add; }
      }
      cache.set(key, v);
    }
    for (let k = 0; k < stride && i + k < n; k++) out[i + k] = v;
  }
  return out;
}

function linearMeanOfCanvas(cv) {
  const d = cv.getContext('2d').getImageData(0, 0, cv.width, cv.height).data;
  let s = 0;
  for (let i = 0; i < d.length; i += 4) { const v = (d[i] + d[i + 1] + d[i + 2]) / 765; s += Math.pow(v, 2.2); }
  return s / (d.length / 4);
}

function canvasTex(cv, { srgb = true, repeat = true, mips = true, aniso = 1 } = {}) {
  const t = new THREE.CanvasTexture(cv);
  t.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
  t.wrapS = t.wrapT = repeat ? THREE.RepeatWrapping : THREE.ClampToEdgeWrapping;
  t.generateMipmaps = mips;
  t.minFilter = mips ? THREE.LinearMipmapLinearFilter : THREE.LinearFilter;
  t.anisotropy = aniso;
  t.needsUpdate = true;
  return t;
}

function blackTex() {
  const t = new THREE.DataTexture(new Uint8Array([0, 0, 0, 255]), 1, 1);
  t.needsUpdate = true;
  return t;
}

const isCars = (o) => o.isInstancedMesh && (o.name === 'cars' || o.userData.lookRole === 'cars' ||
  (o.instanceColor && o.geometry && o.geometry.attributes.position && o.geometry.attributes.position.count === 48 && !o.material.map));

// ------------------------------------------------------------------ main entry
export function applyLook(scene, renderer, opts = {}) {
  const presetName = opts.preset || 'video';
  const q = Object.assign({}, PRESETS[presetName] || PRESETS.video, opts.overrides || {});
  const aniso = Math.min(q.anisotropy, renderer.capabilities.getMaxAnisotropy());
  const t0 = performance.now();
  // landscape toggles (docs/landscape.md): opts win, then URL params, then defaults (full landscape, flat world)
  const urlP = typeof location !== 'undefined' ? new URLSearchParams(location.search) : new URLSearchParams();
  const pick = (k, d) => (opts[k] !== undefined ? opts[k] : urlP.has(k) ? urlP.get(k) : d);
  const onOff = (v) => !(v === false || v === 0 || v === '0' || v === 'off' || v === 'false');
  const LS = {
    mode: String(pick('landscape', 'full')) === 'basic' ? 'basic' : 'full',
    // terrain: 'auto' (default) = on when the roads carry real heights (data/roads.meta.json elevation.model 'terrain/1')
    terrain: String(pick('terrain', 'auto')) === 'auto' ? 'auto' : onOff(pick('terrain', false)),
    exaggeration: +pick('exaggeration', 1.0),
    buildings: onOff(pick('buildings', true)),
    trees: onOff(pick('trees', true)),
    structures: onOff(pick('structures', true)),
    sky: String(pick('sky', 'clear')),
  };

  // ---- shared uniforms
  const U = {
    uSunDir: { value: new THREE.Vector3(0.6, 0.4, 0.2).normalize() },
    uSunIrr: { value: new THREE.Color(1, 1, 1) },
    uSkyIrr: { value: new THREE.Color(0.5, 0.55, 0.6) },
    uGroundIrr: { value: new THREE.Color(0.1, 0.1, 0.08) },
    uZenith: { value: new THREE.Color() },
    uHorizon: { value: new THREE.Color() },
    uSunGlow: { value: new THREE.Color() },
    uCamMod: { value: new THREE.Vector2() },
    uViewportH: { value: 1000 },
    uTime: { value: 0 },
    uRimCol: { value: new THREE.Color(0.10, 0.09, 0.08) },
    uStateGlow: { value: 0.55 },
  };

  // ---- textures (generated here, never at import)
  const asphaltCv = makeAsphaltCanvas(q.asphaltSize);
  const grassCv = makeGrassCanvas(512);
  const tex = {
    asphalt: canvasTex(asphaltCv, { aniso }),
    asphaltMean: linearMeanOfCanvas(asphaltCv),
    grass: canvasTex(grassCv, { srgb: false, aniso }),
    tones: Float32Array.from(TONES),
    mask: blackTex(), coastS: null, coastL: blackTex(), urban: blackTex(),
    maskRect: new THREE.Vector4(-50000, -50000, 100000, 100000),
  };
  const texBytes = () => {
    let b = q.asphaltSize * q.asphaltSize * 4 * 4 / 3 + 512 * 512 * 4 * 4 / 3;
    if (tex.coastS) b += 2 * 512 * 512 * 4;
    if (tex.urban.image && tex.urban.image.width > 1) b += 512 * 512 * 4;
    return b;
  };
  const table = classTable();

  // ---- bookkeeping for remove()
  // keyed by object, so repeated styling of shared materials never piles up records and a
  // streamed tile's records can be dropped when it is evicted (onTileDispose)
  const swapped = new Map();   // obj -> Map(prop -> original value)   ('__attr:name' = attribute we added)
  const added = new Map();     // child we added -> parent
  const disposables = new Set();
  const seen = new WeakSet();
  const roles = {};            // role -> live count (for info())
  const roleOf = new Map();    // obj -> role label
  const heuristics = new Set();
  const setProp = (obj, prop, v) => {
    let m = swapped.get(obj); if (!m) swapped.set(obj, (m = new Map()));
    if (!m.has(prop)) m.set(prop, obj[prop]);
    if (!prop.startsWith('__')) obj[prop] = v;
  };
  const addTo = (parent, child) => { parent.add(child); added.set(child, parent); };
  const own = (x) => { disposables.add(x); return x; };
  const role = (r, o) => {
    if (o) { if (roleOf.has(o)) return; roleOf.set(o, r); }
    roles[r] = (roles[r] || 0) + 1;
  };

  // ---- materials (shared)
  const mats = {};
  const roadMat = (lc) => {
    const k = 'road' + lc;
    if (!mats[k]) {
      mats[k] = own(M.makeRoadMaterial(U, tex));
      if (lc) mats[k].defines = { LOOK_LANECOORD: '' };
      if (lc === 'AX') mats[k].defines.LOOK_LANE_ACROSS_X = '';
    }
    return mats[k];
  };
  mats.localLines = own(M.makeRoadLineMaterial(U, { opacity: 0.55, fadeNear: q.localRibbons ? 500 : 0, fadeFar: q.localRibbons ? 1800 : 0, lo: 0.09, hi: 0.16 }));
  mats.paintWhite = own(M.makePaintMaterial(U, { color: 0xf2f2ee, width: 0.15 }));
  mats.paintDash = own(M.makePaintMaterial(U, { color: 0xf2f2ee, width: 0.13 }));
  mats.paintYellow = own(M.makePaintMaterial(U, { color: 0xf0b323, width: 0.14 }));
  mats.localRibbon = own(M.makeLocalRibbonMaterial(U, tex));
  mats.ground = null;       // built once the coast mask is known
  mats.water = null;
  mats.station = own(new THREE.MeshStandardMaterial({ color: 0x29c9a0, emissive: 0x0a4a3a, roughness: 0.5, metalness: 0.1 }));
  mats.stationCanopy = own(new THREE.MeshStandardMaterial({ color: 0xe6ebe8, emissive: 0x0b3b30, roughness: 0.55, metalness: 0.15 }));
  // traffic-control posts: grey painted metal; the live heads (signals.js) carry the state
  mats.controls = own(new THREE.MeshStandardMaterial({ color: 0x6d7175, roughness: 0.45, metalness: 0.55 }));
  mats.controls.onBeforeCompile = (sh) => { sh.fragmentShader = sh.fragmentShader.replace('#include <color_fragment>', ''); };   // posts stay grey; the heads show the state
  mats.controls.customProgramCacheKey = () => 'look-control-posts';
  mats.car = own(M.patchCarMaterial(q.carMaterial === 'standard'
    ? new THREE.MeshStandardMaterial({ color: 0xd8d8d8, roughness: 0.6, metalness: 0.0 })
    : new THREE.MeshLambertMaterial({ color: 0xd8d8d8 }), U, q));
  mats.carShadow = own(M.makeCarShadowMaterial(U));
  const shadowGeo = own(M.makeCarShadowGeometry());

  // ---- lights + sky
  const lights = new THREE.Group(); lights.name = 'look-lights';
  const hemi = new THREE.HemisphereLight(0xffffff, 0x444433, Math.PI);
  const sun = new THREE.DirectionalLight(0xffffff, Math.PI);
  lights.add(hemi, sun, sun.target);
  addTo(scene, lights);

  const sky = new THREE.Mesh(own(new THREE.SphereGeometry(1, 32, 16)), own(M.makeSkyMaterial(U)));
  sky.name = 'look-sky'; sky.frustumCulled = false; sky.renderOrder = -100;
  addTo(scene, sky);
  const fog = new THREE.FogExp2(0xffffff, 1e-5);

  // ---- renderer / scene state
  setProp(renderer, 'toneMapping', THREE.ACESFilmicToneMapping);
  setProp(renderer, 'toneMappingExposure', opts.exposure ?? 1.0);
  setProp(renderer, 'outputColorSpace', THREE.SRGBColorSpace);
  setProp(scene, 'background', null);
  setProp(scene, 'fog', fog);

  let hour = opts.timeOfDay ?? 7.6;
  let hourSetAt = performance.now(), hourChangedAt = -1e9;
  // sky presets (docs/landscape.md): 'clear' follows the clock; 'golden' holds a low warm sun (long shadows);
  // 'overcast' is flat, soft light under a grey sky
  let skyPreset = ['clear', 'golden', 'overcast'].includes(LS.sky) ? LS.sky : 'clear';
  let fogMul = 1;
  function setTimeOfDay(h) {
    if (h !== hour) hourChangedAt = performance.now();
    hour = h; hourSetAt = performance.now();
    const s = sunForHour(h);
    if (skyPreset === 'golden') s.el = Math.min(s.el, 6.5);
    const el = THREE.MathUtils.degToRad(Math.max(s.el, -2)), az = THREE.MathUtils.degToRad(s.az);
    U.uSunDir.value.set(Math.cos(el) * Math.sin(az), Math.sin(el), -Math.cos(el) * Math.cos(az)).normalize();
    const k = lerpKeys(s.el);
    fogMul = 1;
    if (skyPreset === 'golden') { k.sun.lerp(new THREE.Color(0xffa860), 0.35); k.hor.lerp(new THREE.Color(0xf3c39a), 0.35); k.glow.lerp(new THREE.Color(0xffa050), 0.4); k.sunI *= 1.1; }
    if (skyPreset === 'overcast') {
      k.zen.set(0x8f99a6); k.hor.set(0xc8cdd2); k.glow.set(0x000000); k.sun.set(0xf2f0ec);
      k.sunI *= 0.22; k.skyI *= 1.45; fogMul = 1.8;
    }
    U.uZenith.value.copy(k.zen); U.uHorizon.value.copy(k.hor); U.uSunGlow.value.copy(k.glow).multiplyScalar(Math.max(0, Math.min(1, (s.el + 4) / 6)));
    U.uSunIrr.value.copy(k.sun).multiplyScalar(k.sunI);
    U.uSkyIrr.value.copy(k.zen).lerp(k.hor, 0.45).multiplyScalar(k.skyI * 1.15);
    U.uGroundIrr.value.set(0.42, 0.40, 0.30).multiplyScalar(k.skyI * 0.5);
    hemi.color.copy(U.uSkyIrr.value); hemi.groundColor.copy(U.uGroundIrr.value); hemi.intensity = Math.PI;
    sun.color.copy(k.sun); sun.intensity = k.sunI * Math.PI;
    sun.position.copy(U.uSunDir.value).multiplyScalar(1000);
    // fog colour is mixed after colour-space conversion in three r160, so store display sRGB numbers raw
    const hs = k.hor.clone().convertLinearToSRGB();
    fog.color.setRGB(hs.r, hs.g, hs.b, THREE.LinearSRGBColorSpace);
    U.uRimCol.value.copy(k.hor).multiplyScalar(0.12);
  }
  setTimeOfDay(hour);

  // ---- live traffic-control heads (src/look/signals.js)
  const signals = q.signals ? createSignals(U) : null;
  if (signals) addTo(scene, signals.group);
  let wantSignals = 0, signalsLoading = false, signalsT0 = performance.now();
  async function autoAttachSignals() {
    // no controller handed over by the page: build the sim's controller from the same data
    signalsLoading = true;
    const t0 = performance.now();
    console.info('look: building the traffic-control controller from data/ …');
    try {
      const { loadWorld } = await import('../sim/data.js');
      const w = await loadWorld({ local: true, stations: false, controls: true, geom: false, junctions: false });
      if (w.controls && !signals.attached) { await signals.attach({ controls: w.controls }); hidePosts(); heuristics.add('signals: controller built from data/ (page did not attach one)'); }
      console.info(`look: traffic controls ready in ${Math.round(performance.now() - t0)} ms (${w.controls ? w.controls.approachCount : 0} approaches)`);
    } catch (e) { console.warn('look: traffic controls unavailable', e.message); }
  }

  // once live heads exist, the world's placeholder posts (controlPosts / stream 'signals') would stand
  // in front of them at the same spot: hide those materials (shared, so tiles streamed later follow)
  function hidePosts() {
    mats.controls.visible = false;
    for (const k of ['signalstrue', 'signalsfalse']) if (mats[k]) mats[k].visible = false;
  }

  // ---- landscape (src/look/landscape.js): land cover, hillshade, water, optional relief
  let landscape = null, landscapeLive = false, lsFade = 1;
  const hiddenForLandscape = [];
  function hideForLandscape(o, prop = 'visible') { hiddenForLandscape.push([o, prop]); if (landscapeLive) setProp(o, prop, false); }
  if (LS.mode === 'full') {
    landscape = createLandscape({ U, renderer, scene, q, terrain: LS.terrain, exaggeration: LS.exaggeration, grassTex: tex.grass, buildings: LS.buildings, trees: LS.trees, structures: LS.structures, sky: LS.sky,
      onReady: () => { landscapeLive = true; for (const [o, prop] of hiddenForLandscape) setProp(o, prop, false); } });
    addTo(scene, landscape.group);
    const prevSBR = scene.onBeforeRender;
    setProp(scene, 'onBeforeRender', function (r, sc, cam, rt) {
      if (prevSBR) prevSBR.call(this, r, sc, cam, rt);
      if (landscape && cam.isPerspectiveCamera) { r.getDrawingBufferSize(vpLs); landscape.update(cam, vpLs.y); }
    });
  }
  const vpLs = new THREE.Vector2();

  let clockOverride = null;
  let frame = 0;
  const vp = new THREE.Vector2();
  sky.onBeforeRender = (r, sc, cam) => {
    const p = cam.position;
    sky.position.copy(p);
    sky.scale.setScalar((cam.far || 1e5) * 0.5);
    sky.updateMatrixWorld();
    U.uCamMod.value.set(((p.x % 512) + 512) % 512, ((p.z % 512) + 512) % 512);
    r.getDrawingBufferSize(vp);
    U.uViewportH.value = vp.y;
    U.uTime.value = clockOverride ?? performance.now() / 1000;
    fog.density = fogMul / Math.max(28000, p.y * 2.4);
    if (signals) {
      // sim clock if the page gives one (setSimTime), else the time of day; on static pages it runs in real time
      const nowMs = performance.now();
      const live = nowMs - hourChangedAt < 1000;
      signals.update(cam, hour * 3600 + (live ? 0 : (nowMs - Math.max(hourSetAt, signalsT0)) / 1000));
      if (!wantSignals && cam.position.y < 3000) wantSignals = nowMs;   // close enough for heads to matter
      if (!signals.attached && wantSignals && !signalsLoading && nowMs - wantSignals > 2000) autoAttachSignals();
    }
    if (++frame % 30 === 0) scan();
  };

  // ---- role handlers
  let waterBuilt = false, groundSeen = null, urbanBuilt = false, localRibbonMesh = null;

  function buildCoast(waterMesh) {
    const am = waterMesh.material && waterMesh.material.alphaMap;
    const gp = waterMesh.geometry && waterMesh.geometry.parameters;
    if (!am || !am.image || !gp || !gp.width) { heuristics.add('water: no alphaMap/PlaneGeometry; coast mask unavailable'); return; }
    const W = gp.width, H = gp.height;
    tex.maskRect.set(waterMesh.position.x - W / 2, waterMesh.position.z - H / 2, W, H);
    tex.mask = am;   // read-only reuse of the world's coastline mask
    tex.coastS = own(canvasTex(blurMask(am.image, 512, 2), { srgb: false, repeat: false, mips: false }));
    tex.coastL = own(canvasTex(blurMask(am.image, 512, 9), { srgb: false, repeat: false, mips: false }));
  }

  function ensureGround() {
    if (!mats.ground) mats.ground = own(M.makeGroundMaterial(U, tex, q));
    return mats.ground;
  }

  function ensureWaterPlane(y) {
    if (waterBuilt) return;
    waterBuilt = true;
    mats.water = own(M.makeWaterMaterial(U, tex, q));
    const r = tex.maskRect, pad = 70000;
    const g = own(new THREE.PlaneGeometry(r.z + pad * 2, r.w + pad * 2));
    g.rotateX(-Math.PI / 2);
    const wm = new THREE.Mesh(g, mats.water);
    wm.position.set(r.x + r.z / 2, y, r.y + r.w / 2);
    wm.name = 'look-water'; wm.renderOrder = -2; wm.frustumCulled = false;
    addTo(scene, wm);
    if (landscape) hideForLandscape(wm);   // the landscape ground draws the water itself (one surface, no coplanar layers)
  }

  function styleRoadRibbon(o) {
    const g = o.geometry;
    if (!g.attributes.lookClass) {
      if (g.attributes.classId) g.setAttribute('lookClass', g.attributes.classId);
      else if (g.attributes.color) { g.setAttribute('lookClass', new THREE.BufferAttribute(classFromColors(g.attributes.color, table, 2), 1)); heuristics.add(`${o.name}: class from vertex colours`); }
      else g.setAttribute('lookClass', new THREE.BufferAttribute(new Float32Array(g.attributes.position.count).fill(o.name === 'localRibbons' ? 11 : 4), 1));
      setProp(g, '__attr:lookClass');
    }
    const lc = g.attributes.laneCoord ? (o.userData.role ? 'AX' : 'XA') : '';   // stream tiles store (across, along)
    setProp(o, 'material', roadMat(lc));
    role('road ribbon', o);
  }

  function styleMarkings(group) {
    const o = group;
    let n = 0;
    for (const ls of group.children) {
      if (seen.has(ls)) continue;
      if (ls.isMesh && !ls.isInstancedMesh) {
        // rebuilt world: markings already triangle quads -> keep geometry, lit paint material
        seen.add(ls);
        const c0 = ls.material.color ? ls.material.color.getHex() : 0xffffff;
        const vc = !!ls.geometry.attributes.color;
        setProp(ls, 'material', own(M.makePaintMeshMaterial(U, c0, vc)));
        setProp(ls, 'renderOrder', 1);
        n++;
        continue;
      }
      if (!ls.isLineSegments) continue;
      seen.add(ls);
      const pos = ls.geometry.attributes.position;
      if (!pos || pos.count < 2) continue;
      const nm = ls.name;
      const c = ls.material.color || new THREE.Color(1, 1, 1);
      const yellow = nm === 'laneCenter' || (c.r > 0.6 && c.b < 0.4);
      if (!['laneEdges', 'laneCenter', 'laneDashes'].includes(nm)) heuristics.add(`laneMarkings child "${nm}" matched by colour`);
      const mat = yellow ? mats.paintYellow : nm === 'laneDashes' ? mats.paintDash : mats.paintWhite;
      const quads = new THREE.Mesh(own(M.makeQuadGeometry(pos.array)), mat);
      quads.name = 'look-paint-' + nm; quads.frustumCulled = false; quads.renderOrder = 1;
      addTo(ls, quads);
      setProp(ls.material, 'visible', false);
      n++;
    }
    if (n) { role('lane paint', o); changed = true; }
  }

  function styleLocalGroup(group) {
    const o = group;
    group.updateMatrixWorld();
    const ox = group.matrixWorld.elements[12], oz = group.matrixWorld.elements[14];   // stream tiles: tile centre
    let n = 0;
    const wantRibbons = q.localRibbons && !localRibbonMesh;
    for (const ls of group.children) {
      if (!ls.isLineSegments || seen.has(ls)) continue;
      seen.add(ls);
      setProp(ls, 'material', mats.localLines);
      n++;
      const pos = ls.geometry.attributes.position, col = ls.geometry.attributes.color;
      if (!pos) continue;
      addUrban(pos.array, ox, oz);
      if (wantRibbons) {
        const n = Math.floor(pos.count / 2);
        const wc = new Float32Array(n * 2);
        const cls = col ? classFromColors(col, table, 2) : null;
        for (let i = 0; i < n; i++) {
          const c = cls ? cls[i * 2] % 16 : 11;
          wc[i * 2] = LOCAL_WIDTH[c] ?? 7; wc[i * 2 + 1] = c;
        }
        const rib = new THREE.Mesh(own(M.makeQuadGeometry(pos.array, wc)), mats.localRibbon);
        rib.name = 'look-localRibbon'; rib.frustumCulled = false; rib.renderOrder = -1;
        addTo(ls, rib);
      }
    }
    if (n) { role('local streets', o); changed = true; }
  }

  // town density: an incremental grid of street length (streamed tiles keep adding to it);
  // the texture is rebuilt at most every 2 s, and only once the coast-mask rectangle is known
  const urbanGrid = new Float32Array(512 * 512), urbanArrays = new WeakSet();
  let urbanDirty = false, urbanPending = [], urbanAt = -1e9;
  function addUrban(arr, ox, oz) {
    if (urbanArrays.has(arr)) return;
    urbanArrays.add(arr);
    urbanPending.push([arr, ox, oz]);
    urbanDirty = true;
  }
  function maybeRebuildUrban() {
    if (!urbanDirty || !(waterBuilt || frame > 180) || performance.now() - urbanAt < 2000) return;
    const r = tex.maskRect, rect = { x: r.x, z: r.y, w: r.z, h: r.w };
    for (const [a, ox, oz] of urbanPending) accumulateDensity(urbanGrid, a, rect, 512, ox, oz);
    urbanPending = []; urbanDirty = false; urbanAt = performance.now();
    const cv = densityFromGrid(urbanGrid, rect, 512, 3);
    if (tex.urban && tex.urban.isCanvasTexture) { tex.urban.image = cv; tex.urban.needsUpdate = true; }
    else tex.urban = own(canvasTex(cv, { srgb: false, repeat: false, mips: false }));
    const gm = ensureGround();
    gm.uniforms.uUrban.value = tex.urban; gm.uniforms.uHasUrban.value = 1;
    urbanBuilt = true;
  }

  // streamed water tile: keep the tile (its mask is up to 3 m/px at L6), draw it with the look's water
  let streamWater = false;
  function styleWaterTile(o) {
    let mat;
    if (!o.material.alphaMap) mat = mats.waterFull || (mats.waterFull = own(M.makeWaterMaterial(U, tex, q, { full: true })));
    else { mat = own(M.makeWaterMaterial(U, tex, q, { mask: o.material.alphaMap, channel: o.material.alphaMap.isDataTexture ? 0 : 1 })); o.userData.__lookOwnMat = mat; }
    setProp(o, 'material', mat);
    setProp(o, 'renderOrder', -2);
    role('water tiles', o);
    if (landscape) hideForLandscape(o);
  }

  /** Streamed tile evicted: drop every record about it and free what the look added under it. */
  function forget(obj) {
    seen.delete(obj);
    const r = roleOf.get(obj);
    if (r) { roleOf.delete(obj); if (--roles[r] <= 0) delete roles[r]; }
    swapped.delete(obj);
    if (obj.geometry) swapped.delete(obj.geometry);
    for (const c of [...obj.children]) {
      if (added.get(c) !== obj) continue;
      obj.remove(c); added.delete(c);
      if (c.geometry && disposables.has(c.geometry)) { c.geometry.dispose(); disposables.delete(c.geometry); }
    }
    const om = obj.userData && obj.userData.__lookOwnMat;
    if (om) { om.dispose(); disposables.delete(om); delete obj.userData.__lookOwnMat; }
  }

  // main-tier 1-px lines -> anti-aliased far-view quads (class width, min pixel width by class)
  const FAR_WIDTH = [20, 6, 16, 6, 13, 6, 11, 6, 9, 6, 7, 7, 5];
  function styleRoadLines(o) {
    const pos = o.geometry.attributes.position, col = o.geometry.attributes.color;
    if (!pos) return;
    const n = Math.floor(pos.count / 2);
    const cls = col ? classFromColors(col, table, 2) : null;
    const wc = new Float32Array(n * 2);
    for (let i = 0; i < n; i++) { const c = cls ? cls[i * 2] % 16 : 4; wc[i * 2] = FAR_WIDTH[c] ?? 8; wc[i * 2 + 1] = c; }
    if (!mats.farRoad) mats.farRoad = own(M.makeFarRoadMaterial(U, tex));
    const quads = new THREE.Mesh(own(M.makeQuadGeometry(pos.array, wc)), mats.farRoad);
    quads.name = 'look-farRoads'; quads.frustumCulled = false; quads.renderOrder = -0.5;
    addTo(o, quads);
    setProp(o.material, 'visible', false);
    role('road lines -> far quads', o);
  }

  // ---- streaming local-street tiles (src/stream): roles by userData.lookRole or name
  const STREET_KIND = { 'local-asphalt': 0, junction: 1, sidewalk: 2, curb: 3, driveway: 4 };
  function streetRole(k) {
    if (!k || typeof k !== 'string') return null;
    if (k in STREET_KIND || k === 'verge' || k === 'planting' || k.startsWith('marking-')) return 'street';
    return null;
  }
  function styleStreet(o) {
    const k = o.userData.lookRole || o.name;
    const hasUv = !!o.geometry.attributes.uv;
    let mat;
    if (k.startsWith('marking-')) {
      const yellow = /yellow|center|centre/.test(k);
      const key = 'paint-' + (yellow ? 'y' : 'w') + (o.geometry.attributes.color ? 'c' : '');
      mat = mats[key] || (mats[key] = own(M.makePaintMeshMaterial(U, yellow ? 0xf0b323 : 0xf2f2ee, !!o.geometry.attributes.color)));
    } else if (k === 'verge' || k === 'planting') {
      if (!mats.verge) { mats.verge = own(M.makeGroundMaterial(U, tex, q)); mats.verge.depthWrite = true; mats.verge.side = THREE.DoubleSide; mats.verge.uniforms.uHasUrban.value = 0; mats.verge.uniforms.uHasCoast.value = 0; mats.verge.defines.LOOK_VERGE = ''; }
      mat = mats.verge;
    } else {
      const kind = STREET_KIND[k];
      const hasN = !!o.geometry.attributes.normal;
      const key = 'street' + kind + (hasUv ? 'uv' : '') + (hasN ? 'n' : '');
      if (!mats[key]) {
        mats[key] = own(M.makeStreetMaterial(U, tex, kind));
        if (hasUv) mats[key].defines.LOOK_UV = '';
        if (hasN) mats[key].defines.LOOK_NORMAL = '';
      }
      mat = mats[key];
    }
    setProp(o, 'material', mat);
    role('street: ' + (k.startsWith('marking-') ? 'marking-*' : k), o);
  }

  function styleCars(o) {
    if (o.material && (o.material.isShaderMaterial || o.material.isRawShaderMaterial || o.userData.lookKeepMaterial)) {
      // custom car shader (e.g. GPU backend reading storage buffers): never swap it, only lighting applies
      role('cars (custom shader kept)', o); heuristics.add('cars: custom ShaderMaterial kept as-is');
      return;
    }
    setProp(o, 'material', mats.car);
    role('cars', o);
    if (o.name !== 'cars' && o.userData.lookRole !== 'cars') heuristics.add('cars: InstancedMesh matched by geometry (48 verts + instanceColor)');
    if (q.carShadows) {
      const sh = new THREE.InstancedMesh(shadowGeo, mats.carShadow, 1);
      sh.instanceMatrix = o.instanceMatrix;          // shared buffer, no copy, no CPU cost
      sh.count = 0; sh.frustumCulled = false; sh.renderOrder = 1.5; sh.name = 'look-carShadows';
      sh.onBeforeRender = () => { sh.count = o.count; };
      addTo(o, sh);
    }
  }

  function classify(o) {
    const r = o.userData && o.userData.lookRole;
    const n = o.name;
    if (n.startsWith('look-') || o === lights) return 'skip';
    if (o.isLight && !lights.children.includes(o)) return 'light';
    const sr = streetRole(r || n);
    if (sr && o.isMesh) return sr;
    if (r) return r;
    if (o.isMesh && !o.isInstancedMesh && (n === 'roads' || n === 'localRibbons')) return 'road';
    if (n === 'roadGlow') return 'glow';
    if (n === 'roadLines' && o.isLineSegments) return 'roadLines';
    if (n === 'localRoads') return 'local';
    if (n === 'laneMarkings' && o.isMesh) return 'markingMesh';
    if (n === 'laneMarkings') return 'markings';
    if (n === 'ground' && o.isMesh) return 'ground';
    if (n === 'water' && o.isMesh) return 'water';
    if (n === 'shoreline') return 'shoreline';
    if (n === 'stations') return 'stations';
    if ((n === 'controls' || n === 'controlPosts') && o.isInstancedMesh) return 'controls';   // world: Group 'controls' of per-km 'controlPosts'
    if ((n === 'signals' || o.userData.role === 'signals') && o.isMesh) return 'signals';
    if (isCars(o)) return 'cars';
    return null;
  }

  let changed = false;
  function scan(root = scene) {
    changed = false;
    const found = [];
    root.traverse((o) => {
      if (seen.has(o)) return;
      const r = classify(o);
      if (r === 'skip') return;
      // groups whose children arrive with them: re-check children each scan
      if (r === 'local' || r === 'markings' || r === 'stations') { found.push([r, o]); return; }
      if (r) { seen.add(o); found.push([r, o]); }
    });
    if (!localRibbonMesh) localRibbonMesh = found.find(([r, o]) => r === 'road' && o.name === 'localRibbons') ? true : null;
    // water first (it defines the coast mask the ground uses). Streamed water: the root tile (L0)
    // gives the coast mask; every tile then draws itself with its own finer mask (no global plane).
    for (const [r, o] of found) if (r === 'water' && !waterBuilt) {
      if (o.userData.level === undefined) { buildCoast(o); ensureWaterPlane(o.position.y || 0.3); }
      else if (o.userData.level === 0 && o.material.alphaMap) { buildCoast(o); waterBuilt = true; streamWater = true; }
    }
    for (const [r, o] of found) {
      switch (r) {
        case 'road': styleRoadRibbon(o); break;
        case 'glow': setProp(o.material, 'visible', false); role('glow (hidden)', o); break;
        case 'roadLines': styleRoadLines(o); break;
        case 'local': styleLocalGroup(o); break;
        case 'markings': styleMarkings(o); break;
        case 'markingMesh': setProp(o, 'material', own(M.makePaintMeshMaterial(U, o.material.color ? o.material.color.getHex() : 0xffffff, !!o.geometry.attributes.color))); role('lane paint (mesh)', o); break;
        case 'ground': groundSeen = o; setProp(o, 'material', ensureGround()); setProp(o, 'renderOrder', -3); role('ground', o); if (landscape) hideForLandscape(o); break;
        case 'water':
          if (o.userData.level !== undefined) styleWaterTile(o);
          else { setProp(o.material, 'visible', false); role('water (replaced)', o); }
          break;
        case 'shoreline': setProp(o.material, 'visible', false); role('shoreline (hidden)', o); break;
        case 'signals': if (!mats['signals' + !!o.geometry.attributes.normal]) { mats['signals' + !!o.geometry.attributes.normal] = own(M.makeSignalMaterial(U, !!o.geometry.attributes.normal)); if (signals && signals.attached) hidePosts(); }
          setProp(o, 'material', mats['signals' + !!o.geometry.attributes.normal]); role('signal / sign posts', o); if (!wantSignals) wantSignals = performance.now(); break;
        case 'stations':
          // streamed station tiles are single merged meshes (no stationBoxes / stationPins children)
          for (const c of o.children) if (c.isMesh && !seen.has(c)) {
            seen.add(c);
            let sm = c.name === 'stationPins' ? mats.station : mats.stationCanopy;   // stationBoxes = canopy
            if (!c.geometry.attributes.normal) {   // streamed station meshes carry no normals: flat-shade from derivatives (else black)
              const k = sm === mats.station ? 'stationFlat' : 'stationCanopyFlat';
              if (!mats[k]) { mats[k] = own(sm.clone()); mats[k].flatShading = true; }
              sm = mats[k];
            }
            setProp(c, 'material', sm);
            role('stations', c); changed = true;
          }
          break;
        case 'cars': styleCars(o); break;
        case 'light': setProp(o, 'intensity', 0); role('page light (muted)', o); break;
        case 'street': styleStreet(o); break;
        case 'controls': setProp(o, 'material', mats.controls); role('traffic control posts', o); if (!wantSignals) wantSignals = performance.now(); break;
        default: break;
      }
      if (r && r !== 'local' && r !== 'markings' && r !== 'stations') changed = true;
    }
    maybeRebuildUrban();
    if (root !== scene) { if (changed) root.traverse((o) => { if (o.material && !Array.isArray(o.material)) o.material.needsUpdate = true; }); return; }
    // no land surface in the page: add one (grass where land, the water plane covers the rest)
    if (!groundSeen && (waterBuilt || roles['road ribbon'])) {
      const r = tex.maskRect, pad = 70000;
      const g = own(new THREE.PlaneGeometry(r.z + pad * 2, r.w + pad * 2));
      g.rotateX(-Math.PI / 2);
      const gm = new THREE.Mesh(g, ensureGround());
      gm.position.set(r.x + r.z / 2, 0, r.y + r.w / 2);
      gm.name = 'look-ground'; gm.renderOrder = -3; gm.frustumCulled = false;
      addTo(scene, gm);
      if (landscape) hideForLandscape(gm);
      groundSeen = gm;
      heuristics.add('no "ground" mesh: added look-ground');
    }
    if (changed) scene.traverse((o) => { if (o.material && !Array.isArray(o.material)) o.material.needsUpdate = true; });
  }

  scan();
  // first scan also refreshes programs of untouched materials (tone mapping changed)
  scene.traverse((o) => { if (o.material && !Array.isArray(o.material)) o.material.needsUpdate = true; });

  const handle = {
    preset: presetName,
    quality: q,
    uniforms: U,
    setTimeOfDay,
    get timeOfDay() { return hour; },
    /** Sky / light preset: 'clear' (default, follows the clock), 'golden' (low warm sun), 'overcast' (flat, soft). */
    setSky(name) { skyPreset = ['clear', 'golden', 'overcast'].includes(name) ? name : 'clear'; LS.sky = skyPreset; setTimeOfDay(hour); if (landscape) landscape.uniforms.uLsShadow.value = skyPreset === 'golden' ? 0.45 : skyPreset === 'overcast' ? 0.0 : 0.22; },
    get skyPreset() { return skyPreset; },
    /** Frame-exact capture: drive water animation from the sim clock (seconds). null = wall clock. */
    setClock(sec) { clockOverride = sec; },
    /** Force a re-scan now (it also runs every 30 rendered frames). */
    refresh: () => scan(scene),
    /** Streaming tiles: call with each new tile mesh/group (styled immediately, before its first frame). */
    onTileMesh(mesh) { if (mesh) scan(mesh); return mesh; },
    /** Live signal heads: hand over the sim's TrafficControls (or {graph, pack} to build one). */
    attachSignals(o) { if (!signals) return Promise.resolve(false); signalsLoading = true; return signals.attach(o).then((ok) => { if (ok) hidePosts(); return ok; }); },
    /** Sim clock (s since midnight) for the signal lamps; call each frame on sim pages. */
    setSimTime(t) { if (signals) signals.setTime(t); },
    get signals() { return signals; },
    /** Streaming tiles: call before a tile's meshes are disposed (stream onTileDispose). */
    onTileDispose(obj) { if (obj) forget(obj); },
    /** Landscape layer (land cover, relief, water, buildings, trees); null in landscape=basic mode. */
    get landscape() { return landscape; },
    landscapeOptions: LS,
    /**
     * Fade every landscape layer: 1 = full landscape, 0 = gone (ground and water blend to the fade colour, dark
     * #070a14 by default, i.e. the roads-only synthetic look; buildings / trees dissolve). Smooth and frame-exact:
     * call it every frame with any 0..1 curve (e.g. by altitude).
     */
    landscapeFade(k, color) { lsFade = Math.max(0, Math.min(1, k)); if (landscape) landscape.setFade(lsFade, color); return lsFade; },
    /** Terrain height (m, before exaggeration) at world x, z; 0 until the landscape data is loaded or when flat. */
    terrainHeight(x, z) { return landscape ? landscape.heightAt(x, z) : 0; },
    info() {
      return { preset: presetName, roles: { ...roles }, heuristics: [...heuristics], textureMB: +(texBytes() / 1048576).toFixed(1), buildMs: Math.round(buildMs),
        landscape: landscape ? { ...LS, terrainOn: landscape.terrainOn, ready: landscape.isReady, ...landscape.stats() } : { mode: 'basic' } };
    },
    remove() {
      if (landscape) { landscape.dispose(); landscape = null; }
      for (const [child, parent] of added) parent.remove(child);
      for (const [obj, props] of swapped) for (const [prop, v] of props) {
        if (prop.startsWith('__attr:')) obj.deleteAttribute(prop.slice(7)); else obj[prop] = v;
      }
      for (const d of disposables) d.dispose && d.dispose();
      if (signals) signals.dispose();
      for (const t of [tex.asphalt, tex.grass]) t.dispose();
      scene.traverse((o) => { if (o.material && !Array.isArray(o.material)) o.material.needsUpdate = true; });
      sky.onBeforeRender = () => {};
    },
  };
  const buildMs = performance.now() - t0;
  return handle;
}
