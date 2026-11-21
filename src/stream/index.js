// View-dependent tile streaming for the Bay Area world (rendering only; never
// imported by the sim, brain, nav or GPU compute code).
//
//   import { createStream } from './src/stream/index.js';
//   const stream = createStream({ scene, camera, renderer, baseUrl: 'data/tiles/' });
//   await stream.ready;                 // root tile on screen
//   renderer.setAnimationLoop(() => { stream.update(); renderer.render(scene, camera); });
//   stream.stats()                      // {tilesLoaded, tilesVisible, gpuMB, fetchedMB, drawCalls, ...}
//
// Selection: quadtree from the root; a tile is refined when its screen-space
// error  err * (viewportH / (2 tan(fov/2))) / distance  exceeds `sse` px (and
// stays refined until it drops below sse * hysteresis). Children replace their
// parent only once every visible child is decoded and built, so there are never
// holes; the swap is instant. Fetch + gunzip + decode run in Web Workers,
// requests are prioritised (coarse levels first, then larger error), stale
// requests are dropped, and an LRU evicts tiles over the GPU-memory budget.
import * as THREE from 'three';
import { CLASS_STYLE } from './classStyle.js';
import { CLASSES_ALL } from '../roadpack.js';
import { LOOK_ROLE, PRIM, tileUrl, tileKey } from './format.js';
import { gunzipMaybe } from './gunzip.js';

const LEVEL_COLORS = [0xff3b3b, 0xff9f1c, 0xffe14d, 0x5be65b, 0x2ec4ff, 0x6c63ff, 0xe05bff];
export const WATER_COLOR = 0x0e2c5c, LAND_COLOR = 0x141824;

function palette() {
  const byClass = CLASSES_ALL.map((c) => { const col = new THREE.Color((CLASS_STYLE[c] || CLASS_STYLE.secondary).color); return [col.r, col.g, col.b]; });
  const dimmed = byClass.map((c) => c.map((v) => v * 0.3));   // ribbon fill dim
  return {
    'main-asphalt': { byClass: dimmed, tunnelDim: 0.35 },
    'road-strip': { byClass: dimmed, tunnelDim: 0.35 },
    'road-lines': { byClass, tunnelDim: 0.35 },
    'local-lines': { byClass, tunnelDim: 0.35 },
  };
}

function makeMaterials() {
  const basic = (o) => new THREE.MeshBasicMaterial(Object.assign({ side: THREE.DoubleSide }, o));
  const paint = (color, off) => basic({ color, polygonOffset: true, polygonOffsetFactor: off, polygonOffsetUnits: off });
  const water = new THREE.MeshBasicMaterial({ color: WATER_COLOR, depthWrite: false, transparent: false });
  return {
    'main-asphalt': basic({ vertexColors: true }),
    'road-strip': basic({ vertexColors: true }),
    'local-asphalt': basic({ color: 0x2c313d }),
    'junction': basic({ color: 0x2c313d, polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -1 }),
    'curb': basic({ color: 0x7a808c }),
    'sidewalk': basic({ color: 0x4a505c }),
    'verge': basic({ color: 0x2f4a2a }),
    'driveway': basic({ color: 0x55596a, polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -1 }),
    'road-lines': new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.55 }),
    'local-lines': new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.9 }),
    'coast': new THREE.LineBasicMaterial({ color: 0x27405f, transparent: true, opacity: 0.7 }),
    'marking-white': paint(0xf4f6ff, -2), 'marking-yellow': paint(0xffd23f, -2), 'marking-dash': paint(0xffffff, -2),
    'marking-stopbar': paint(0xf4f6ff, -2), 'marking-crosswalk': paint(0xe8ecf5, -2),
    'stations': basic({ color: 0x5ff2c9 }),
    'signals': basic({ color: 0x3b4250 }),
    'water-full': water,
  };
}

/** Water mask material: single-channel (R8) alpha map; the stock shader reads .g, so patch it to .r. */
function maskWaterMaterial(tex) {
  const m = new THREE.MeshBasicMaterial({ color: WATER_COLOR, alphaMap: tex, alphaTest: 0.5, depthWrite: false });
  m.onBeforeCompile = (s) => { s.fragmentShader = s.fragmentShader.replace('#include <alphamap_fragment>', 'diffuseColor.a *= texture2D( alphaMap, vAlphaMapUv ).r;'); };
  m.customProgramCacheKey = () => 'stream-water-r8';
  return m;
}

// ---- cross-fade: complementary screen-door dither (no blending, no sorting, works with any material,
// including the look's ShaderMaterials). Injected once per material; a fading tile draws with a
// clone of its material (same program, so no shader compile) carrying the fade value.
const SD_GLSL = `
uniform float uStreamFade;
uniform float uStreamFadeOut;
float streamBayer(vec2 p) {
  ivec2 q = ivec2(mod(p, 4.0));
  const float m[16] = float[16](0., 8., 2., 10., 12., 4., 14., 6., 3., 11., 1., 9., 15., 7., 13., 5.);
  return (m[q.x + q.y * 4] + 0.5) / 16.0;
}
`;
export function injectDither(mat) {
  if (!mat || mat.userData.__sd) return;
  mat.userData.__sd = true;
  const prev = mat.onBeforeCompile, prevKey = mat.customProgramCacheKey.call(mat);
  mat.onBeforeCompile = function (sh, r) {
    if (prev && prev !== mat.onBeforeCompile) prev.call(this, sh, r);
    sh.uniforms.uStreamFade = this.userData.__sdF || (this.userData.__sdF = { value: 1 });
    sh.uniforms.uStreamFadeOut = this.userData.__sdO || (this.userData.__sdO = { value: 0 });
    sh.fragmentShader = sh.fragmentShader.replace(/void\s+main\s*\(\s*\)\s*\{/, SD_GLSL +
      'void main() {\n  if (uStreamFade < 0.999) { float sdB = streamBayer(gl_FragCoord.xy); if (uStreamFadeOut > 0.5 ? sdB < uStreamFade : sdB >= uStreamFade) discard; }\n');
  };
  mat.customProgramCacheKey = () => prevKey + '|streamDither';
  mat.needsUpdate = true;
}
function fadeClone(base) {
  const m = base.clone();
  m.onBeforeCompile = base.onBeforeCompile; m.customProgramCacheKey = base.customProgramCacheKey;
  m.userData.__sd = true; m.userData.__sdF = { value: 0 }; m.userData.__sdO = { value: 0 };
  return m;
}

function workerPool(n) {
  const url = new URL('./worker.js', import.meta.url);
  const workers = [], pending = new Map();
  let next = 0, seq = 0;
  for (let i = 0; i < n; i++) {
    const w = new Worker(url, { type: 'module' });
    w.onmessage = (ev) => { const p = pending.get(ev.data.id); if (p) { pending.delete(ev.data.id); p(ev.data); } };
    workers.push(w);
  }
  return {
    post(msg) { for (const w of workers) w.postMessage(msg); },
    load(url) { return new Promise((res) => { const id = ++seq; pending.set(id, res); workers[next++ % n].postMessage({ id, url }); }); },
    terminate() { workers.forEach((w) => w.terminate()); },
  };
}

/**
 * opts:
 *   scene, camera, renderer, baseUrl='data/tiles/'
 *   sse=6 (px), hysteresis=0.75, maxLevel=Infinity, gpuBudgetMB=384, maxConcurrent=4, workers=2..4
 *   buildMsPerFrame=6        main-thread time budget for building meshes from decoded tiles
 *   fadeMs=350               cross-fade (dither dissolve) between LOD levels; 0 = instant swap
 *   now()                    clock for fades (default performance.now; pass a fixed clock for frame-exact capture)
 *   ground=false             add a land plane under everything (pages that have their own ground leave it off)
 *   onTileMesh(mesh, tile)   called for every mesh/line/group a tile creates (look-pack hook)
 *   onTileDispose(mesh, tile) called before a tile's meshes are disposed (evicted)
 *   roles: optional Set of role names to build (others skipped), e.g. without markings
 */
export function createStream(opts) {
  const { scene, camera, renderer } = opts;
  // absolute, so the workers (whose base URL is src/stream/) resolve it the same way
  const baseUrl = new URL((opts.baseUrl || 'data/tiles/').replace(/\/?$/, '/'), document.baseURI).href;
  const cfg = {
    sse: opts.sse ?? 6, hysteresis: opts.hysteresis ?? 0.75, maxLevel: opts.maxLevel ?? Infinity,
    gpuBudget: (opts.gpuBudgetMB ?? 384) * 1048576, maxConcurrent: opts.maxConcurrent ?? 4,
    buildMsPerFrame: opts.buildMsPerFrame ?? 6, debugLevels: false, frozen: false, fadeMs: opts.fadeMs ?? 350,
  };
  const root = new THREE.Group(); root.name = 'streamTiles';
  scene.add(root);
  const mats = makeMaterials();
  const levelMats = LEVEL_COLORS.map((c) => ({ mesh: new THREE.MeshBasicMaterial({ color: c, side: THREE.DoubleSide }), line: new THREE.LineBasicMaterial({ color: c }) }));
  if ((opts.fadeMs ?? 350) > 0) { for (const m of Object.values(mats)) injectDither(m); for (const l of levelMats) { injectDither(l.mesh); injectDither(l.line); } }
  const pool = workerPool(opts.workers ?? Math.max(2, Math.min(4, (navigator.hardwareConcurrency || 4) >> 1)));
  pool.post({ palette: palette() });

  const nodes = new Map();
  let rootNode = null, manifest = null, frame = 0;
  const queue = new Set(), built = [];
  let inflight = 0;
  const counters = { fetchedBytes: 0, fetchedTiles: 0, builtTiles: 0, evicted: 0, errors: 0, gpuBytes: 0, decodeMs: 0, buildMs: 0 };
  const frustum = new THREE.Frustum(), projView = new THREE.Matrix4(), tmpV = new THREE.Vector3();
  let rendered = [], visibleDraws = 0;
  const fading = new Map();      // node -> {mode: 'in'|'out', t0, key}
  const fadeMats = new Map();    // `${baseMat.id}|${mode}|${t0}` -> clone
  const clock = opts.now || (() => performance.now());
  let ground = null;

  // ---------------------------------------------------------------- manifest
  const ready = (async () => {
    const t0 = performance.now();
    let r = await fetch(baseUrl + 'manifest.json.gz');
    let m;
    if (r.ok) m = JSON.parse(new TextDecoder().decode(await gunzipMaybe(await r.arrayBuffer())));
    else { r = await fetch(baseUrl + 'manifest.json'); if (!r.ok) throw new Error('no tile manifest at ' + baseUrl); m = await r.json(); }
    manifest = m;
    const F = Object.fromEntries(m.tileFields.map((f, i) => [f, i]));
    for (const t of m.tiles) {
      const L = t[F.level], ix = t[F.ix], iy = t[F.iy];
      const n = {
        key: tileKey(L, ix, iy), L, ix, iy, url: baseUrl + tileUrl(L, ix, iy), gzBytes: t[F.gzBytes],
        err: m.levels[L].err, size: m.levels[L].size,
        box: new THREE.Box3(new THREE.Vector3(t[F.minX], t[F.minY] - 1, t[F.minZ]), new THREE.Vector3(t[F.maxX], t[F.maxY] + 2, t[F.maxZ])),
        childMask: t[F.childMask], children: [], parent: null, state: 'none', group: null, gpuBytes: 0,
        lastUsed: -1, wanted: -1, refined: false, sse: 0, pri: 0,
      };
      nodes.set(n.key, n);
    }
    for (const n of nodes.values()) {
      for (let q = 0; q < 4; q++) if (n.childMask & (1 << q)) {
        const c = nodes.get(tileKey(n.L + 1, n.ix * 2 + (q & 1), n.iy * 2 + (q >> 1)));
        if (c) { n.children.push(c); c.parent = n; }
      }
    }
    rootNode = nodes.get(tileKey(0, 0, 0));
    if (opts.ground) {
      // subdivided (huge 2-triangle planes lose depth precision near the camera and cover the roads)
      // and pushed back with polygon offset so pavement at +2..3 cm always wins
      const R = m.root, g = new THREE.PlaneGeometry(R.size * 3, R.size * 3, 96, 96); g.rotateX(-Math.PI / 2);
      ground = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: LAND_COLOR, polygonOffset: true, polygonOffsetFactor: 2, polygonOffsetUnits: 8 }));
      ground.position.set(R.x0 + R.size / 2, -0.15, R.z0 + R.size / 2); ground.name = 'ground'; ground.renderOrder = -3;
      scene.add(ground);
    }
    await new Promise((res) => { rootNode.onReady = res; request(rootNode, 1e9); pump(); });
    // also make sure the whole coarse level 1 is resident (cheap; keeps zoom-outs hole-free)
    for (const c of rootNode.children) request(c, 1e8);
    counters.firstTileMs = performance.now() - t0;
    return api;
  })();

  // ---------------------------------------------------------------- loading
  function request(n, pri) {
    n.wanted = frame;
    n.pri = pri;
    if (n.state === 'none' && !(n.retryAt > performance.now())) { n.state = 'queued'; queue.add(n); }
  }
  function pump() {
    if (inflight >= cfg.maxConcurrent || !queue.size) return;
    const cand = [];
    for (const n of queue) {
      if (frame - n.wanted > 30 && n !== rootNode && n.L > 1) { queue.delete(n); n.state = 'none'; continue; }   // stale
      cand.push(n);
    }
    cand.sort((a, b) => b.pri - a.pri);
    // over budget with nothing evictable: the working set is bigger than the budget. Stop refining
    // (coarser tiles stay on screen) instead of thrashing; coverage loads (L<=1) still go through.
    const full = counters.gpuBytes > cfg.gpuBudget;
    counters.budgetLimited = full;
    for (const n of cand) {
      if (inflight >= cfg.maxConcurrent) break;
      if (full && n.L > 1) continue;
      queue.delete(n); n.state = 'loading'; inflight++;
      pool.load(n.url).then((res) => {
        inflight--;
        if (!res.ok) {
          counters.errors++;
          n.retries = (n.retries || 0) + 1;
          if (n.retries <= 4 && !/HTTP 404/.test(res.error)) { n.state = 'none'; n.retryAt = performance.now() + 250 * 2 ** n.retries; }   // transient: retry with backoff
          else { n.state = 'error'; console.warn('[stream]', res.error); }
          return;
        }
        counters.fetchedBytes += res.bytes; counters.fetchedTiles++; counters.decodeMs += res.decodeMs;
        built.push([n, res]);
      });
    }
  }
  function buildPending() {
    const t0 = performance.now();
    while (built.length && performance.now() - t0 < cfg.buildMsPerFrame) {
      // build coarse first
      let bi = 0; for (let i = 1; i < built.length; i++) if (built[i][0].pri > built[bi][0].pri) bi = i;
      const [n, res] = built.splice(bi, 1)[0];
      buildTile(n, res);
    }
    counters.buildMs += performance.now() - t0;
  }

  function buildTile(n, res) {
    const h = res.header, g = new THREE.Group();
    // the group stays at the origin and every top-level object carries the tile offset itself, so
    // object.position is in world units (src/look reads water.position as the mask rectangle centre)
    g.name = 'tile-' + n.key;
    g.userData = { stream: true, level: n.L, key: n.key };
    let bytes = 0;
    const want = opts.roles;
    const sphere = new THREE.Sphere();
    n.box.getBoundingSphere(sphere); sphere.center.x -= h.cx; sphere.center.z -= h.cz;
    for (const c of res.chunks) {
      if (want && !want.has(c.role)) continue;
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(c.pos, 3));
      geo.setAttribute('classId', new THREE.BufferAttribute(c.cls, 1));
      if (c.lane) { geo.setAttribute('laneCoord', new THREE.BufferAttribute(c.lane, 2)); geo.setAttribute('uv', new THREE.BufferAttribute(c.uv, 2)); }
      if (c.normal) geo.setAttribute('normal', new THREE.BufferAttribute(c.normal, 3));
      if (c.color) geo.setAttribute('color', new THREE.BufferAttribute(c.color, 3));
      if (c.idx) geo.setIndex(new THREE.BufferAttribute(c.idx, 1));
      geo.boundingSphere = sphere.clone();
      bytes += c.pos.byteLength + c.cls.byteLength + (c.lane ? c.lane.byteLength * 2 : 0) + (c.normal ? c.normal.byteLength : 0) + (c.color ? c.color.byteLength : 0) + (c.idx ? c.idx.byteLength : 0);
      let obj;
      if (c.prim === PRIM.LINES) obj = new THREE.LineSegments(geo, mats[c.role]);
      else obj = new THREE.Mesh(geo, mats[c.role]);
      obj.name = c.role;
      obj.userData.role = c.role;
      obj.userData.level = n.L;
      if (LOOK_ROLE[c.role]) obj.userData.lookRole = LOOK_ROLE[c.role];
      obj.userData.baseMaterial = obj.material;
      obj.position.set(h.cx, 0, h.cz); obj.updateMatrix();
      obj.matrixAutoUpdate = false;
      if (c.role === 'coast' || c.role === 'road-lines' || c.role === 'local-lines') obj.renderOrder = -0.5;
      // look-pack group conventions: local lines + stations are groups whose children are styled
      if (c.role === 'local-lines' || c.role === 'stations') {
        const wrap = new THREE.Group(); wrap.name = c.role === 'local-lines' ? 'localRoads' : 'stations';
        wrap.userData = { role: c.role, lookRole: LOOK_ROLE[c.role], level: n.L };
        delete obj.userData.lookRole;
        obj.position.set(0, 0, 0); obj.updateMatrix();
        wrap.position.set(h.cx, 0, h.cz);
        wrap.add(obj); g.add(wrap);
      } else g.add(obj);
    }
    if (res.water && (!want || want.has('water'))) {
      const size = n.size, geo = new THREE.PlaneGeometry(size, size); geo.rotateX(-Math.PI / 2);
      let mat;
      if (res.water.full) mat = mats['water-full'];
      else if (n.L === 0) {
        // root: RGBA canvas texture so src/look can reuse it as the coastline mask
        const { w, h: hh, data } = res.water, cv = document.createElement('canvas'); cv.width = w; cv.height = hh;
        const ctx = cv.getContext('2d'), img = ctx.createImageData(w, hh);
        for (let i = 0; i < w * hh; i++) { const v = data[i]; img.data[i * 4] = img.data[i * 4 + 1] = img.data[i * 4 + 2] = img.data[i * 4 + 3] = v; }
        ctx.putImageData(img, 0, 0);
        const tex = new THREE.CanvasTexture(cv); tex.colorSpace = THREE.NoColorSpace; tex.anisotropy = 4;
        mat = new THREE.MeshBasicMaterial({ color: WATER_COLOR, alphaMap: tex, alphaTest: 0.5, depthWrite: false });
        bytes += w * hh * 4;
      } else {
        const tex = new THREE.DataTexture(res.water.data, res.water.w, res.water.h, THREE.RedFormat, THREE.UnsignedByteType);
        tex.flipY = false; tex.magFilter = THREE.LinearFilter; tex.minFilter = THREE.LinearMipmapLinearFilter; tex.generateMipmaps = true;
        tex.needsUpdate = true;
        mat = maskWaterMaterial(tex);
        bytes += res.water.w * res.water.h * 1.34;
      }
      // PlaneGeometry uv v=1 at -z after rotateX; mask row 0 is min z (north): flip v for DataTexture (flipY false)
      if (!res.water.full && n.L !== 0) { const uv = geo.attributes.uv; for (let i = 0; i < uv.count; i++) uv.setY(i, 1 - uv.getY(i)); }
      const wm = new THREE.Mesh(geo, mat);
      wm.name = 'water'; wm.userData = { role: 'water', lookRole: 'water', level: n.L, baseMaterial: mat, ownMaterial: mat !== mats['water-full'] };
      wm.position.set(h.cx, 0.3, h.cz); wm.renderOrder = -1; wm.updateMatrix(); wm.matrixAutoUpdate = false;
      g.add(wm);
    }
    g.updateMatrixWorld(true);
    g.visible = false;
    n.group = g; n.gpuBytes = bytes; n.state = 'ready'; n.lastUsed = frame;   // fresh tiles are not evictable this frame
    counters.gpuBytes += bytes; counters.builtTiles++;
    root.add(g);
    if (cfg.debugLevels) applyDebug(n);
    // look hook: called once per top-level tile object (mesh, lines, or role group) before its first frame
    if (opts.onTileMesh) for (const o of g.children) opts.onTileMesh(o, n);
    if (cfg.fadeMs > 0) g.traverse((o) => { if (o.material && !Array.isArray(o.material)) injectDither(o.material); });
    if (n.onReady) { n.onReady(); n.onReady = null; }
  }

  function disposeTile(n) {
    if (!n.group) return;
    if (fading.has(n)) endFade(n);
    if (opts.onTileDispose) n.group.traverse((o) => { if (o !== n.group) opts.onTileDispose(o, n); });
    n.group.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      if (o.userData.ownMaterial && o.userData.baseMaterial) { const m = o.userData.baseMaterial; if (m.alphaMap) m.alphaMap.dispose(); m.dispose(); }
    });
    root.remove(n.group);
    counters.gpuBytes -= n.gpuBytes;
    n.group = null; n.gpuBytes = 0; n.state = 'none'; n.refined = false;
    counters.evicted++;
  }

  function evict() {
    if (counters.gpuBytes <= cfg.gpuBudget) return;
    const cands = [];
    for (const n of nodes.values()) if (n.state === 'ready' && n.L > 1 && n.lastUsed < frame) cands.push(n);
    cands.sort((a, b) => a.lastUsed - b.lastUsed || b.L - a.L);
    for (const n of cands) { if (counters.gpuBytes <= cfg.gpuBudget * 0.9) break; disposeTile(n); }
  }

  // ---------------------------------------------------------------- selection
  function distTo(box, p) {
    const dx = Math.max(box.min.x - p.x, 0, p.x - box.max.x), dy = Math.max(box.min.y - p.y, 0, p.y - box.max.y), dz = Math.max(box.min.z - p.z, 0, p.z - box.max.z);
    return Math.max(1, Math.hypot(dx, dy, dz));
  }
  // ---------------------------------------------------------------- cross-fade
  const isAncestor = (a, b) => { for (let p = b.parent; p; p = p.parent) if (p === a) return true; return false; };
  function related(n, list) { for (const m of list) if (m !== n && (isAncestor(n, m) || isAncestor(m, n))) return true; return false; }
  function setFadeMaterials(n, f) {
    const st = fading.get(n);
    n.group.traverse((o) => {
      if (!o.material || Array.isArray(o.material) || !(o.isMesh || o.isLine)) return;
      if (f === null) { if (o.userData.__fadeBase) { o.material = o.userData.__fadeBase; o.userData.__fadeBase = null; } return; }
      const base = o.userData.__fadeBase || o.material;
      if (!base.userData.__sd) return;
      const key = base.id + '|' + st.key;
      let m = fadeMats.get(key);
      if (!m) { m = fadeClone(base); fadeMats.set(key, m); }
      m.userData.__sdF.value = f; m.userData.__sdO.value = st.mode === 'out' ? 1 : 0;
      if (m.uniforms) { if (m.uniforms.uStreamFade) m.uniforms.uStreamFade.value = f; if (m.uniforms.uStreamFadeOut) m.uniforms.uStreamFadeOut.value = st.mode === 'out' ? 1 : 0; }
      if (o.material !== m) { o.userData.__fadeBase = base; o.material = m; }
    });
  }
  function endFade(n) {
    const st = fading.get(n); fading.delete(n);
    if (n.group) { setFadeMaterials(n, null); if (st.mode === 'out' && !n.selected) n.group.visible = false; }
    // drop clones no fade uses any more
    const live = new Set([...fading.values()].map((v) => v.key));
    for (const [k, m] of fadeMats) if (!live.has(k.slice(k.indexOf('|') + 1))) { m.dispose(); fadeMats.delete(k); }
  }
  function updateFades(prev) {
    for (const n of prev) n.selected = false;
    for (const n of rendered) n.selected = true;
    const now = clock();
    if (cfg.fadeMs > 0) {
      const prevSet = new Set(prev);
      const key = 't' + Math.round(now);
      for (const n of rendered) {
        if (prevSet.has(n)) continue;
        if (fading.has(n)) { if (fading.get(n).mode === 'out') endFade(n); continue; }   // came back: show solid
        if (related(n, prev)) { fading.set(n, { mode: 'in', t0: now, key: 'in' + key }); counters.fades = (counters.fades || 0) + 1; }
      }
      for (const n of prev) {
        if (n.selected || fading.has(n) || !n.group) continue;
        if (related(n, rendered)) fading.set(n, { mode: 'out', t0: now, key: 'out' + key });
      }
    }
    for (const [n, st] of [...fading]) {
      if (!n.group || (st.mode === 'in' && !n.selected)) { endFade(n); continue; }
      const f = (now - st.t0) / cfg.fadeMs;
      if (f >= 1) endFade(n); else setFadeMaterials(n, Math.max(0.001, f));
    }
  }

  function update() {
    if (!rootNode || rootNode.state !== 'ready') { pump(); buildPending(); return; }
    frame++;
    if (!cfg.frozen) {
      camera.updateMatrixWorld();
      projView.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
      frustum.setFromProjectionMatrix(projView);
      const vh = renderer.domElement.height || 1000;
      const k = camera.isPerspectiveCamera ? vh / (2 * Math.tan((camera.fov * Math.PI) / 360)) : vh / ((camera.top - camera.bottom) / camera.zoom);
      const cam = camera.getWorldPosition(tmpV);
      const prev = rendered;
      rendered = [];
      const visit = (n) => {
        if (!frustum.intersectsBox(n.box)) return;
        n.lastUsed = frame;
        const sse = camera.isPerspectiveCamera ? (n.err * k) / distTo(n.box, cam) : n.err * k;
        n.sse = sse;
        const can = n.children.length && n.L < cfg.maxLevel;
        const want = can && (sse > cfg.sse || (n.refined && sse > cfg.sse * cfg.hysteresis));
        if (want) {
          let allReady = true;
          for (const c of n.children) {
            if (!frustum.intersectsBox(c.box)) continue;
            if (c.state !== 'ready' && c.state !== 'error') { allReady = false; request(c, -c.L * 1e6 + Math.min(sse, 1e6 - 1)); }
            else { c.wanted = frame; c.lastUsed = frame; }   // ready sibling waiting for the others: keep it
          }
          if (allReady || n.state !== 'ready') { n.refined = true; for (const c of n.children) if (c.state === 'ready') visit(c); return; }
        } else n.refined = false;
        if (n.state === 'ready') rendered.push(n);
        else request(n, -n.L * 1e6 + sse);
      };
      visit(rootNode);
      updateFades(prev);
      visibleDraws = 0;
      for (const n of prev) if (n.group && !n.selected && !fading.has(n)) n.group.visible = false;
      for (const n of rendered) { n.group.visible = true; }
      for (const n of fading.keys()) if (n.group) { n.group.visible = true; n.lastUsed = frame; }
      for (const n of rendered) n.group.traverse((o) => { if (o.isMesh || o.isLine) visibleDraws++; });
    }
    pump();
    buildPending();
    evict();
  }

  // ---------------------------------------------------------------- debug + stats
  function applyDebug(n) {
    if (!n.group) return;
    n.group.traverse((o) => {
      if (!o.userData.baseMaterial || o.userData.role === 'water') return;
      if (cfg.debugLevels) { if (!o.userData.debugSaved) o.userData.debugSaved = o.material; o.material = o.isLine ? levelMats[n.L % 7].line : levelMats[n.L % 7].mesh; }
      else if (o.userData.debugSaved) { o.material = o.userData.debugSaved; o.userData.debugSaved = null; }
    });
  }
  function setDebugLevels(on) { cfg.debugLevels = !!on; for (const n of nodes.values()) applyDebug(n); }

  function stats() {
    const byLevel = {};
    let maxL = 0;
    for (const n of rendered) { byLevel[n.L] = (byLevel[n.L] || 0) + 1; maxL = Math.max(maxL, n.L); }
    let loaded = 0; for (const n of nodes.values()) if (n.state === 'ready') loaded++;
    return {
      tilesTotal: nodes.size, tilesLoaded: loaded, tilesVisible: rendered.length, visibleByLevel: byLevel, maxVisibleLevel: maxL,
      queued: queue.size, inflight, pendingBuild: built.length,
      gpuMB: +(counters.gpuBytes / 1048576).toFixed(1), fetchedMB: +(counters.fetchedBytes / 1048576).toFixed(2), fetchedTiles: counters.fetchedTiles,
      drawCalls: visibleDraws, evicted: counters.evicted, errors: counters.errors,
      decodeMs: Math.round(counters.decodeMs), buildMs: Math.round(counters.buildMs), firstTileMs: counters.firstTileMs && Math.round(counters.firstTileMs),
      budgetLimited: !!counters.budgetLimited, fading: fading.size, fadesStarted: counters.fades || 0,
      idle: (queue.size === 0 || !!counters.budgetLimited) && inflight === 0 && built.length === 0,
    };
  }

  function dispose() {
    for (const n of nodes.values()) disposeTile(n);
    scene.remove(root); if (ground) scene.remove(ground);
    pool.terminate();
  }

  const api = {
    ready, update, stats, dispose, setDebugLevels, config: cfg, group: root, nodes,
    get manifest() { return manifest; },
    /** Visible tiles this frame (array of {key, L, ix, iy, group}). */
    get visibleTiles() { return rendered; },
    /** Promise resolving once nothing is queued, loading or waiting to build (after at least one update). */
    settled(timeoutMs = 30000) {
      // idle for 250 ms of rendered frames (a single idle sample right after a camera jump can precede the new requests)
      const t0 = performance.now(), f0 = frame;
      let idleSince = -1;
      return new Promise((res) => {
        const tick = () => {
          const s = stats(), now = performance.now();
          if (s.idle && frame > f0 + 5) { if (idleSince < 0) idleSince = now; } else idleSince = -1;
          if ((idleSince >= 0 && now - idleSince > 250 && !fading.size) || now - t0 > timeoutMs) res(s); else setTimeout(tick, 30);
        };
        tick();
      });
    },
  };
  return api;
}
