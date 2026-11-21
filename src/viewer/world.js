// world.html: the Bay Area world, streamed in tiles (src/stream) and styled by the look pack (src/look).
// Render only: nothing in src/sim or src/gpu imports this file.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { makeProjection } from '../proj.js';
import { createStream } from '../stream/index.js';
import { createCarLOD } from '../stream/cars.js';

const P = new URLSearchParams(location.search);
const BG = 0x070a14;
const canvas = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
const scene = new THREE.Scene();
scene.background = new THREE.Color(BG);
const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 2, 400000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.dampingFactor = 0.08;
controls.maxPolarAngle = Math.PI * 0.495; controls.minDistance = 8; controls.maxDistance = 250000;
controls.screenSpacePanning = false;
scene.add(new THREE.AmbientLight(0x8899bb, 0.9));
const sun = new THREE.DirectionalLight(0xffffff, 0.8); sun.position.set(-40000, 70000, -30000); scene.add(sun);

const proj = makeProjection(null);
// look-at (lat, lon), camera height (m), horizontal distance back (m), azimuth (deg, 0 = camera south looking north)
const PRESETS = {
  'Whole Bay': { lat: 37.585, lon: -122.17, h: 88000, d: 32000, az: 10 },
  'SF': { lat: 37.77, lon: -122.42, h: 6000, d: 6000, az: 10 },
  'Downtown SF': { lat: 37.7905, lon: -122.4005, h: 260, d: 520, az: 200 },
  'Street (SF)': { lat: 37.7735, lon: -122.4135, h: 40, d: 100, az: -20 },
  'Bay Bridge': { lat: 37.815, lon: -122.34, h: 1800, d: 3000, az: 35 },
  'Oakland': { lat: 37.80, lon: -122.27, h: 5000, d: 6000, az: 0 },
  'Residential (PA)': { lat: 37.4405, lon: -122.1475, h: 70, d: 160, az: 15 },
  'Interchange 101/280': { lat: 37.3395, lon: -121.8495, h: 500, d: 700, az: 40 },
  'San Jose': { lat: 37.345, lon: -121.92, h: 7000, d: 7000, az: 20 },
  'Street (SJ)': { lat: 37.3355, lon: -121.8875, h: 45, d: 110, az: 30 },
};
const KEYS = Object.keys(PRESETS);
function viewOf(p) {
  const t = proj.toXZ(p.lat, p.lon), a = (p.az * Math.PI) / 180;
  const y0 = p.y || 0;   // optional look-at height (m); the camera is p.h above it
  return { target: new THREE.Vector3(t.x, y0, t.z), pos: new THREE.Vector3(t.x + Math.sin(a) * p.d, y0 + p.h, t.z + Math.cos(a) * p.d) };
}
let tour = null;
function flyTo(name) {
  const v = viewOf(typeof name === 'object' ? name : PRESETS[name]);
  tour = null;
  controls.target.copy(v.target); camera.position.copy(v.pos); controls.update();
}

// ---- look (default on; ?look=off for the plain materials)
let look = null;
const lookMode = P.get('look') || 'video';
if (lookMode !== 'off') {
  try { const { applyLook } = await import('../look/index.js'); look = applyLook(scene, renderer, { preset: lookMode === 'perf' ? 'perf' : 'video' }); }
  catch (e) { console.warn('look pack unavailable:', e.message); }
}

// ---- streaming world
const t0 = performance.now();
const stream = createStream({
  scene, camera, renderer, baseUrl: 'data/tiles/', ground: true,
  sse: +(P.get('sse') || 6), gpuBudgetMB: +(P.get('budget') || 384), fadeMs: P.has('fade') ? +P.get('fade') : 350,
  onTileMesh: look ? (m) => look.onTileMesh(m) : null,
  onTileDispose: look && look.onTileDispose ? (m) => look.onTileDispose(m) : null,   // requested from src/look
});
// ?cam=lat,lon,h,d,az[,y] (look-at point, camera height m, distance back m, azimuth deg, look-at height m) or ?view=<preset name>
if (P.get('cam')) { const [lat, lon, h, d, az, y] = P.get("cam").split(",").map(Number); flyTo({ lat, lon, h, d, az, y: y || 0 }); }
else flyTo(P.get('view') || 'Whole Bay');

// ---- GPU sim feeding the car LOD: ?sim=gpu&n=100000 (src/gpu GPUSim, WebGPU). There is no built-in driver:
// every car gets one constant action (?gas=0.3&steer=0&brake=0), as in gpu.html; feed your own with sim.setActions.
let cars = null, simFeed = null;
if (P.get('sim') === 'gpu') {
  const N = +(P.get('n') || 100000);
  cars = createCarLOD({ scene, camera, renderer, maxCars: Math.max(N, 1024), nearDistance: +(P.get('near') || 700) });
  simFeed = (async () => {
    const [{ initGPU }, { GPUSim }, { loadWorld }] = await Promise.all([import('../gpu/device.js'), import('../gpu/gpuSim.js'), import('../sim/data.js')]);
    const gpu = await initGPU();
    const { graph, stations, controls } = await loadWorld({ base: new URL('../../data/', import.meta.url) });
    const sim = await GPUSim.create(graph, { gpu, capacity: Math.max(131072, Math.ceil(N / 1024) * 1024), speed: +(P.get('speed') || 1), obs: 'none', startTime: 7.5 * 3600, controls });
    const A = new Float32Array(sim.cap * 3);
    for (let k = 0; k < sim.cap; k++) { A[3 * k] = +(P.get('steer') || 0); A[3 * k + 1] = +(P.get('gas') || 0.3); A[3 * k + 2] = +(P.get('brake') || 0); }
    sim.setActions(A);
    sim.setStations(stations); sim.scatter(N, { seed: +(P.get('seed') || 7) });
    window.sim = sim;
    let poses = null, pending = null;
    return { sim, frame(dt) {
      sim.advance(dt);
      if (!pending) pending = sim.readPoses().then((p) => { poses = p; pending = null; });   // pipelined readback, never blocks a frame
      if (poses) cars.updateFromPoses(poses);
    } };
  })().catch((e) => { console.warn('sim feed failed:', e); return null; });
}

// ---- tour: whole bay -> mid -> street and back (also used by the measurement script)
function startTour(seconds = 40) {
  const stops = ['Whole Bay', 'San Jose', 'Interchange 101/280', 'Street (SJ)', 'San Jose', 'SF', 'Street (SF)', 'Bay Bridge', 'Whole Bay'].map((k) => viewOf(PRESETS[k]));
  tour = { stops, t: 0, seconds };
}
function stepTour(dt) {
  if (!tour) return;
  tour.t += dt;
  const n = tour.stops.length - 1, u = Math.min(1, tour.t / tour.seconds) * n, i = Math.min(n - 1, Math.floor(u)), f = u - i;
  const e = f * f * (3 - 2 * f), A = tour.stops[i], B = tour.stops[i + 1];
  // interpolate heights geometrically so zooms look even
  const lerpLog = (a, b) => Math.exp(Math.log(Math.max(1, a)) + (Math.log(Math.max(1, b)) - Math.log(Math.max(1, a))) * e);
  controls.target.lerpVectors(A.target, B.target, e);
  const off = A.pos.clone().sub(A.target).lerp(B.pos.clone().sub(B.target), e);
  const len = lerpLog(A.pos.distanceTo(A.target), B.pos.distanceTo(B.target));
  camera.position.copy(controls.target).add(off.setLength(len));
  if (tour.t >= tour.seconds) tour = null;
}

// ---- UI
const bar = document.getElementById('bar'), statsEl = document.getElementById('stats'), legend = document.getElementById('legend');
const btn = (label, fn) => { const b = document.createElement('button'); b.textContent = label; b.onclick = () => fn(b); bar.appendChild(b); return b; };
KEYS.forEach((k) => btn(k, () => flyTo(k)));
btn('Tour', () => startTour());
const dbgBtn = btn('Colour by level', (b) => { const on = !stream.config.debugLevels; stream.setDebugLevels(on); b.classList.toggle('on', on); legend.style.display = on ? 'block' : 'none'; });
btn('Freeze LOD', (b) => { stream.config.frozen = !stream.config.frozen; b.classList.toggle('on', stream.config.frozen); });
const LEVEL_COLORS = ['#ff3b3b', '#ff9f1c', '#ffe14d', '#5be65b', '#2ec4ff', '#6c63ff', '#e05bff'];
legend.innerHTML = '<b>tile level</b>' + LEVEL_COLORS.map((c, i) => `<div class="row"><span class="sw" style="background:${c}"></span>L${i} · ${(102.4 / (1 << i)).toFixed(1)} km</div>`).join('');
// free fly: W/A/S/D move, Q/E down/up (speed scales with height; Shift x4)
const held = new Set();
addEventListener('keyup', (ev) => held.delete(ev.key.toLowerCase()));
function stepFly(dt) {
  if (!held.size) return;
  const f = new THREE.Vector3().subVectors(controls.target, camera.position); f.y = 0; if (f.lengthSq() < 1e-6) f.set(0, 0, -1); f.normalize();
  const r = new THREE.Vector3(-f.z, 0, f.x), m = new THREE.Vector3();
  if (held.has('w')) m.add(f); if (held.has('s')) m.sub(f); if (held.has('d')) m.add(r); if (held.has('a')) m.sub(r);
  if (held.has('e')) m.y += 1; if (held.has('q')) m.y -= 1;
  const v = Math.max(15, camera.position.y * 0.8) * (held.has('shift') ? 4 : 1) * dt;
  m.multiplyScalar(v); camera.position.add(m); controls.target.add(m);
}
addEventListener('keydown', (ev) => {
  if ('wasdqe'.includes(ev.key.toLowerCase()) || ev.key === 'Shift') held.add(ev.key.toLowerCase());
  const i = +ev.key; if (i >= 1 && i <= KEYS.length) flyTo(KEYS[i - 1]);
  if (ev.key === 't' || ev.key === 'T') startTour();
  if (ev.key === 'l' || ev.key === 'L') dbgBtn.click();
  if (ev.key === 'f' || ev.key === 'F') stream.config.frozen = !stream.config.frozen;
  if (ev.key === 'h' || ev.key === 'H') document.querySelectorAll('.panel').forEach((p) => p.classList.toggle('hidden'));
});
addEventListener('resize', () => { camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); });

// ---- loop
let frames = 0, fpsT = performance.now(), last = performance.now(), fps = 0, frameMs = 0, worst = 0;
const fpsLog = [];
renderer.setAnimationLoop(() => {
  const now = performance.now(), dt = Math.min(0.1, (now - last) / 1000);
  worst = Math.max(worst, now - last); last = now;
  stepTour(dt); stepFly(dt);
  controls.update();
  // depth range follows the camera height so markings never z-fight
  const hgt = Math.max(1, camera.position.y);
  camera.near = Math.min(50, Math.max(0.3, hgt * 0.01)); camera.far = Math.max(20000, hgt * 6 + 150000); camera.updateProjectionMatrix();
  const tu = performance.now();
  stream.update();
  if (simLive) simLive.frame(dt);
  const updMs = performance.now() - tu;
  renderer.render(scene, camera);
  frames++;
  if (now - fpsT >= 1000) {
    fps = Math.round((frames * 1000) / (now - fpsT)); frameMs = (now - fpsT) / frames; frames = 0; fpsT = now;
    fpsLog.push(fps); if (fpsLog.length > 120) fpsLog.shift();
    const s = stream.stats(), info = renderer.info;
    const lv = Object.entries(s.visibleByLevel).map(([l, c]) => `L${l}:${c}`).join(' ');
    const cs = cars ? cars.stats() : null;
    statsEl.innerHTML = `<b>${fps} fps</b>  ${frameMs.toFixed(1)} ms/frame (worst ${worst.toFixed(0)})  stream.update ${updMs.toFixed(2)} ms\n` +
      `tiles   visible ${s.tilesVisible}  loaded ${s.tilesLoaded}/${s.tilesTotal}  queue ${s.queued}+${s.inflight}\n` +
      `LOD     max L${s.maxVisibleLevel}   ${lv}\n` +
      `memory  ${s.gpuMB} MB geometry  ·  fetched ${s.fetchedMB} MB (${s.fetchedTiles} tiles)  evicted ${s.evicted}\n` +
      `draws   ${info.render.calls} calls (${s.drawCalls} tile meshes)  ${(info.render.triangles / 1000).toFixed(0)}k tris\n` +
      `camera  ${Math.round(camera.position.y)} m up · first tile ${s.firstTileMs ?? '…'} ms · look ${look ? lookMode : 'off'}` +
      (cs ? `\ncars    ${cs.near} boxes + ${cs.far} dots (${cs.drawCalls} draws)` : '');
    worst = 0;
  }
});

let simLive = null;
if (simFeed) simFeed.then((f) => { simLive = f; });
window.world = { THREE, stream, camera, controls, renderer, scene, flyTo, startTour, PRESETS, get look() { return look; }, get cars() { return cars; }, get simLive() { return simLive; }, get fps() { return fps; }, fpsLog, t0 };
await stream.ready;
window.world.readyMs = performance.now() - t0;
