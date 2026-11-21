// Live traffic-control heads for the look pack: signal heads (red / yellow / green, protected-left
// arrows), ramp meters (red / green) and STOP / YIELD plates, for the approaches near the camera.
//
// State comes from the sim's own controller: src/sim/controls.js TrafficControls.stateOf(a, move, t)
// is a pure function of (approach, movement, sim time), so the lamps are frame-exact and show
// exactly what the sim's drivers perceive. Pass the sim's controller when it has one; otherwise
// the look builds one from the same graph + data/controls pack (same numbering, same plans).
//
// Cheap: at most MAX_HEADS heads, re-selected around the camera every 0.5 s / 40 m; per frame only
// their state attribute (4 floats each) is rewritten. Five instanced draw calls in total.
import * as THREE from 'three';

const MAX_HEADS = 900, MAX_SIGNS = 500, RADIUS = 750, CELL = 250;
const T_SIGNAL = 0, T_STOP = 1, T_GIVE_WAY = 2, T_RAMP = 3, T_CROSSING = 4;
const L_GREEN = 0, L_YELLOW = 1, L_RED = 2, L_DARK = 4;
const MOVE_THROUGH = 0, MOVE_LEFT = 1;

// ---------------------------------------------------------------- geometry
function signAtlas() {
  const cv = document.createElement('canvas'); cv.width = 512; cv.height = 256;
  const g = cv.getContext('2d');
  // STOP octagon (left half)
  const oct = (r) => { g.beginPath(); for (let i = 0; i < 8; i++) { const a = Math.PI / 8 + i * Math.PI / 4; g.lineTo(128 + r * Math.cos(a), 128 + r * Math.sin(a)); } g.closePath(); };
  oct(120); g.fillStyle = '#ffffff'; g.fill();
  oct(108); g.fillStyle = '#c8102e'; g.fill();
  g.fillStyle = '#ffffff'; g.font = 'bold 74px Helvetica, Arial, sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle';
  g.fillText('STOP', 128, 132);
  // YIELD triangle, point down (right half)
  const tri = (inset) => { g.beginPath(); g.moveTo(256 + 8 + inset * 1.7, 18 + inset); g.lineTo(512 - 8 - inset * 1.7, 18 + inset); g.lineTo(384, 238 - inset * 2); g.closePath(); };
  tri(0); g.fillStyle = '#c8102e'; g.fill();
  tri(30); g.fillStyle = '#ffffff'; g.fill();
  g.fillStyle = '#c8102e'; g.font = 'bold 34px Helvetica, Arial, sans-serif'; g.fillText('YIELD', 384, 82);
  const t = new THREE.CanvasTexture(cv);
  t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8; t.generateMipmaps = true; t.minFilter = THREE.LinearMipmapLinearFilter;
  return t;
}

/** Lamp column geometry in head space (+z faces traffic, +x = the driver's right, y up). */
function lampGeometry() {
  const pos = [], lamp = [], ctr = [], idx = [];
  const disc = (cx, cy, r, id, z = 0.152) => {
    const b = pos.length / 3, seg = 16;
    pos.push(cx, cy, z); lamp.push(id); ctr.push(cx, cy, z);
    for (let i = 0; i < seg; i++) { const a = (i / seg) * Math.PI * 2; pos.push(cx + Math.cos(a) * r, cy + Math.sin(a) * r, z); lamp.push(id); ctr.push(cx, cy, z); }
    for (let i = 0; i < seg; i++) idx.push(b, b + 1 + i, b + 1 + ((i + 1) % seg));
  };
  const box = (cx, cy, w, h, d, id) => {   // housing for the left-arrow column
    const b = pos.length / 3, x0 = cx - w / 2, x1 = cx + w / 2, y0 = cy - h / 2, y1 = cy + h / 2, z0 = -d / 2, z1 = d / 2;
    const v = [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1], [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]];
    for (const p of v) { pos.push(...p); lamp.push(id); ctr.push(cx, cy, 0); }
    for (const [a, c, d2, e] of [[0, 1, 2, 3], [5, 4, 7, 6], [4, 0, 3, 7], [1, 5, 6, 2], [3, 2, 6, 7], [4, 5, 1, 0]]) idx.push(b + a, b + c, b + d2, b + a, b + d2, b + e);
  };
  disc(0, 0.36, 0.125, 0); disc(0, 0, 0.125, 1); disc(0, -0.36, 0.125, 2);
  box(-0.44, -0.36, 0.40, 0.42, 0.28, 4); disc(-0.44, -0.36, 0.125, 3);
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('lamp', new THREE.Float32BufferAttribute(lamp, 1));
  g.setAttribute('lampC', new THREE.Float32BufferAttribute(ctr, 3));
  g.setIndex(idx);
  return g;
}

function lampMaterial(U) {
  return new THREE.ShaderMaterial({
    uniforms: Object.assign(THREE.UniformsUtils.clone(THREE.UniformsLib.fog), { uViewportH: U.uViewportH, uGain: { value: 1.7 } }),
    vertexShader: /* glsl */`
#include <common>
#include <fog_pars_vertex>
#include <logdepthbuf_pars_vertex>
attribute float lamp; attribute vec3 lampC; attribute vec4 aState;
uniform float uViewportH;
varying float vLamp; varying vec4 vState; varying vec2 vLocal;
void main() {
  vLamp = lamp; vState = aState; vLocal = (position.xy - lampC.xy) / 0.125;
  float hasLeft = aState.w, type = aState.z;
  vec3 p = position;
  bool hide = (lamp > 2.5 && hasLeft < 0.5) || (abs(type - 3.0) < 0.1 && abs(lamp - 1.0) < 0.1);
  if (hide) p = lampC;                                    // collapse: no protected left / ramp meters have no amber
  // lamps never shrink below ~1.1 px so a lit signal still reads from far away
  vec4 mvC = modelViewMatrix * instanceMatrix * vec4(lampC, 1.0);
  float px = 2.0 * max(-mvC.z, 0.1) / (projectionMatrix[1][1] * uViewportH);
  // is this lamp lit? (same rule as the fragment) lit lamps grow a little and never drop below ~2.4 px
  float th = aState.x, lf = aState.y;
  bool lit = lamp < 0.5 ? abs(th - 2.0) < 0.1 : lamp < 1.5 ? abs(th - 1.0) < 0.1 : lamp < 2.5 ? abs(th) < 0.1 : (lamp < 3.5 && lf > -0.5 && lf < 3.5);
  float sc = lamp < 3.5 ? max(lit ? 1.15 : 1.0, (lit ? 2.4 : 1.0) * px / 0.125) : 1.0;
  p = lampC + (p - lampC) * vec3(sc, sc, 1.0);
  vec4 mvPosition = modelViewMatrix * instanceMatrix * vec4(p, 1.0);
  gl_Position = projectionMatrix * mvPosition;
  #include <logdepthbuf_vertex>
  #include <fog_vertex>
}`,
    fragmentShader: /* glsl */`
#include <common>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
uniform float uGain;
varying float vLamp; varying vec4 vState; varying vec2 vLocal;
vec3 lampCol(float s) { return s < 0.5 ? vec3(0.02, 1.0, 0.42) : s < 1.5 ? vec3(1.0, 0.45, 0.0) : vec3(1.0, 0.02, 0.01); }
void main() {
  #include <logdepthbuf_fragment>
  float id = floor(vLamp + 0.5), th = vState.x, lf = vState.y;
  vec3 col;
  if (id > 3.5) col = vec3(0.025);                                            // arrow housing
  else {
    float st = id < 0.5 ? 2.0 : id < 1.5 ? 1.0 : id < 2.5 ? 0.0 : lf;         // which state lights this lamp
    bool on = id < 2.5 ? abs(th - st) < 0.1 : (lf > -0.5 && lf < 3.5);
    vec3 c = lampCol(st);
    if (id > 2.5) {                                                             // left arrow glyph on the 4th lamp
      vec2 q = vLocal;
      float shaft = step(abs(q.y), 0.14) * step(-0.05, q.x) * step(q.x, 0.62);
      float head = step(q.x, 0.0) * step(-0.62, q.x) * step(abs(q.y), (q.x + 0.62) * 0.95);
      float glyph = max(shaft, head);
      on = on && glyph > 0.5;
    }
    float r = length(vLocal);
    col = on ? c * uGain * (1.15 - 0.3 * r * r) : c * 0.05 + 0.015;     // slightly hot centre
  }
  gl_FragColor = vec4(col, 1.0);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
  #include <fog_fragment>
}`,
    fog: true,
  });
}

/**
 * U: the look's shared uniforms. Returns { attach, setTime, update, info, dispose, group }.
 */
export function createSignals(U, opts = {}) {
  const group = new THREE.Group(); group.name = 'look-signals';
  const metal = new THREE.MeshStandardMaterial({ color: 0x6d7175, roughness: 0.45, metalness: 0.55 });
  const housingMat = new THREE.MeshStandardMaterial({ color: 0x1b1d1f, roughness: 0.6, metalness: 0.2 });
  const atlas = signAtlas();
  const plateMat = new THREE.MeshStandardMaterial({ map: atlas, emissiveMap: atlas, emissive: 0xffffff, emissiveIntensity: 0.22, roughness: 0.5, alphaTest: 0.5, side: THREE.FrontSide });
  plateMat.onBeforeCompile = (sh) => {
    sh.vertexShader = sh.vertexShader.replace('#include <common>', '#include <common>\nattribute float aUvOff;')
      .replace('#include <uv_vertex>', '#include <uv_vertex>\n#ifdef USE_MAP\n  vMapUv = vec2(vMapUv.x * 0.5 + aUvOff, vMapUv.y);\n#endif\n#ifdef USE_EMISSIVEMAP\n  vEmissiveMapUv = vec2(vEmissiveMapUv.x * 0.5 + aUvOff, vEmissiveMapUv.y);\n#endif');
  };
  plateMat.customProgramCacheKey = () => 'look-sign-plate';
  const backMat = new THREE.MeshStandardMaterial({ color: 0x8a8d90, roughness: 0.5, metalness: 0.5 });

  const housingGeo = new THREE.BoxGeometry(0.42, 1.12, 0.28);
  const unitPole = new THREE.BoxGeometry(1, 1, 1).translate(0, 0.5, 0);      // pole: scaled (w, h, w)
  const unitArm = new THREE.BoxGeometry(1, 1, 1).translate(0.5, 0, 0);       // arm: scaled (len, t, t), from the pole top
  const plateGeo = new THREE.PlaneGeometry(0.76, 0.76);
  const plateBackGeo = new THREE.PlaneGeometry(0.76, 0.76).rotateY(Math.PI).translate(0, 0, -0.01);

  const mk = (geo, mat, n, name) => { const m = new THREE.InstancedMesh(geo, mat, n); m.name = name; m.count = 0; m.frustumCulled = false; m.instanceMatrix.setUsage(THREE.DynamicDrawUsage); group.add(m); return m; };
  const housings = mk(housingGeo, housingMat, MAX_HEADS, 'look-signal-housings');
  const lamps = mk(lampGeometry(), lampMaterial(U), MAX_HEADS, 'look-signal-lamps');
  lamps.instanceMatrix = housings.instanceMatrix;                             // same transforms
  const stateArr = new Float32Array(MAX_HEADS * 4);
  const stateAttr = new THREE.InstancedBufferAttribute(stateArr, 4); stateAttr.setUsage(THREE.DynamicDrawUsage);
  lamps.geometry.setAttribute('aState', stateAttr);
  const poles = mk(unitPole, metal, MAX_HEADS + MAX_SIGNS, 'look-signal-poles');
  const arms = mk(unitArm, metal, MAX_HEADS, 'look-signal-arms');
  const plates = mk(plateGeo, plateMat, MAX_SIGNS, 'look-sign-plates');
  const uvOff = new THREE.InstancedBufferAttribute(new Float32Array(MAX_SIGNS), 1);
  plateGeo.setAttribute('aUvOff', uvOff);
  const backs = mk(plateBackGeo, backMat, MAX_SIGNS, 'look-sign-backs');
  backs.instanceMatrix = plates.instanceMatrix;
  for (const m of [housings, lamps, poles, arms, plates, backs]) m.visible = false;

  let tcOuter = null, apOuter = null, gridOuter = null;
  let simTime = null, lastSel = null, selAt = -1e9;
  const heads = [];   // {a, kind: 'mast'|'pole'|'ramp'}
  const st = {};

  /** Precompute the stop-line pose of every approach from the sim graph (once). */
  async function attach({ controls = null, graph = null, pack = null } = {}) {
    let c = null;
    if (controls) c = controls;
    else if (graph && pack) { const { TrafficControls } = await import('../sim/controls.js'); c = new TrafficControls(graph, pack); }
    else return false;
    const { edgePoint } = await import('../sim/roadGraph.js');
    const tc = c;                     // local until the pose table is ready (update() keys off the outer tc)
    const g = tc.graph, nA = tc.approachCount;
    let ap, grid;
    ap = { x: new Float32Array(nA), y: new Float32Array(nA), z: new Float32Array(nA), hx: new Float32Array(nA), hz: new Float32Array(nA), right: new Float32Array(nA), width: new Float32Array(nA), ok: new Uint8Array(nA) };
    grid = new Map();
    const o = {};
    for (let a = 0; a < nA; a++) {
      const ei = tc.apEdge[a]; if (ei < 0) continue;
      const e = g.edges[ei]; if (!e || !e.px) continue;
      const lanes = e.lanes || 1;
      const right = e.rightEdge ?? ((e.laneOff ? e.laneOff[0] : (lanes - 1) * 1.85) + 1.85);
      edgePoint(e, Math.max(0, Math.min(e.length, tc.apStopAt[a])), 0, o);
      ap.x[a] = o.x; ap.y[a] = o.y || 0; ap.z[a] = o.z; ap.hx[a] = o.hx; ap.hz[a] = o.hz;
      ap.right[a] = right; ap.width[a] = lanes * 3.4; ap.ok[a] = 1;
      const k = Math.floor(o.x / CELL) + ':' + Math.floor(o.z / CELL);
      let l = grid.get(k); if (!l) grid.set(k, (l = [])); l.push(a);
    }
    for (const m of [housings, lamps, poles, arms, plates, backs]) m.visible = true;
    selAt = -1e9; lastSel = null;
    apOuter = ap; gridOuter = grid; tcOuter = tc;
    return true;
  }

  const M4 = new THREE.Matrix4(), Q = new THREE.Quaternion(), P = new THREE.Vector3(), S = new THREE.Vector3(), Y = new THREE.Vector3(0, 1, 0);
  function select(cam) {
    const tc = tcOuter, ap = apOuter, grid = gridOuter;
    const cx = cam.position.x, cz = cam.position.z;
    const R = Math.max(RADIUS, Math.min(2000, cam.position.y * 3));
    const cand = [];
    const c0x = Math.floor((cx - R) / CELL), c1x = Math.floor((cx + R) / CELL), c0z = Math.floor((cz - R) / CELL), c1z = Math.floor((cz + R) / CELL);
    for (let gx = c0x; gx <= c1x; gx++) for (let gz = c0z; gz <= c1z; gz++) {
      const l = grid.get(gx + ':' + gz); if (!l) continue;
      for (const a of l) { const d = Math.hypot(ap.x[a] - cx, ap.z[a] - cz); if (d < R) cand.push([d, a]); }
    }
    cand.sort((p, q) => p[0] - q[0]);
    heads.length = 0;
    let nH = 0, nP = 0, nA = 0, nS = 0;
    const yawFacing = (a) => Math.atan2(-ap.hx[a], -ap.hz[a]);                 // +z of the head faces the driver
    const at = (a, off, along = 0) => P.set(ap.x[a] - ap.hz[a] * off + ap.hx[a] * along, 0, ap.z[a] + ap.hx[a] * off + ap.hz[a] * along);
    for (const [, a] of cand) {
      const ty = tc.type[tc.apControl[a]], y0 = ap.y[a] + 0.05, rot = Q.setFromAxisAngle(Y, yawFacing(a));
      if (ty === T_SIGNAL || ty === T_CROSSING) {
        if (nH + 2 > MAX_HEADS) continue;
        const poleOff = ap.right[a] + 1.2, mastOff = ap.right[a] - Math.min(ap.width[a] * 0.5, 5.5);
        // pole + mast arm across the approach lanes
        at(a, poleOff, 1.0); P.y = y0; poles.setMatrixAt(nP++, M4.compose(P, rot, S.set(0.22, 6.4, 0.22)));
        const armLen = poleOff - mastOff + 0.8;
        at(a, poleOff, 1.0); P.y = y0 + 6.1;
        const armRot = new THREE.Quaternion().setFromAxisAngle(Y, Math.atan2(ap.hx[a], ap.hz[a]));   // local +x -> driver's left (hz, -hx): over the road
        arms.setMatrixAt(nA++, M4.compose(P, armRot, S.set(armLen, 0.14, 0.14)));
        // over-lane head (hangs under the arm) and pole head
        at(a, mastOff, 1.0); P.y = y0 + 5.3; housings.setMatrixAt(nH, M4.compose(P, rot, S.set(1.25, 1.25, 1.25))); heads[nH++] = a;
        at(a, poleOff, 1.0); P.y = y0 + 3.4; housings.setMatrixAt(nH, M4.compose(P, rot, S.set(1, 1, 1))); heads[nH++] = a;
      } else if (ty === T_RAMP) {
        if (nH + 1 > MAX_HEADS) continue;
        at(a, ap.right[a] + 1.0, 0.5); P.y = y0; poles.setMatrixAt(nP++, M4.compose(P, rot, S.set(0.16, 2.2, 0.16)));
        P.y = y0 + 2.5; housings.setMatrixAt(nH, M4.compose(P, rot, S.set(0.85, 0.85, 0.85))); heads[nH++] = a;
      } else if (ty === T_STOP || ty === T_GIVE_WAY) {
        if (nS + 1 > MAX_SIGNS) continue;
        at(a, ap.right[a] + 1.0, 0.5); P.y = y0; poles.setMatrixAt(nP++, M4.compose(P, rot, S.set(0.07, 2.2, 0.07)));
        P.y = y0 + 2.2; plates.setMatrixAt(nS, M4.compose(P, rot, S.set(1, 1, 1)));
        uvOff.array[nS++] = ty === T_STOP ? 0 : 0.5;
      }
    }
    housings.count = lamps.count = nH; poles.count = nP; arms.count = nA; plates.count = backs.count = nS;
    for (const m of [housings, poles, arms, plates]) m.instanceMatrix.needsUpdate = true;
    uvOff.needsUpdate = true;
    lastSel = { x: cx, z: cz, y: cam.position.y };
  }

  function update(cam, fallbackTime) {
    const tc = tcOuter;
    if (!tc || !cam) return;
    const now = performance.now();
    if (!lastSel || now - selAt > 500 && Math.hypot(cam.position.x - lastSel.x, cam.position.z - lastSel.z, (cam.position.y - lastSel.y) * 0.3) > 40 || now - selAt > 3000) { select(cam); selAt = now; }
    const t = simTime ?? fallbackTime;
    for (let i = 0; i < lamps.count; i++) {
      const a = heads[i], ty = tc.type[tc.apControl[a]];
      tc.stateOf(a, MOVE_THROUGH, t, st);
      const th = st.state;
      let lf = -1;
      if (ty === T_SIGNAL && tc.apProtLeft[a]) { tc.stateOf(a, MOVE_LEFT, t, st); lf = st.state; }
      stateArr[i * 4] = th === L_DARK ? 9 : th; stateArr[i * 4 + 1] = lf; stateArr[i * 4 + 2] = ty; stateArr[i * 4 + 3] = lf >= 0 ? 1 : 0;
    }
    stateAttr.needsUpdate = true;
  }

  return {
    group, attach, update,
    /** Sim clock in seconds (film / sim pages). null = fall back to the look's time of day. */
    setTime(t) { simTime = t; },
    /** Re-pick the heads around the camera on the next frame (after a camera cut). */
    reselect() { lastSel = null; },
    get attached() { return !!tcOuter; },
    info() { return tcOuter ? { approaches: tcOuter.approachCount, heads: lamps.count, signs: plates.count } : null; },
    /** For checks: state of approach a at time t (same call the lamps use). */
    stateOf(a, t) { const o = {}; tcOuter.stateOf(a, MOVE_THROUGH, t, o); return o; },
    approachPose(a) { const ap = apOuter; return ap && ap.ok[a] ? { x: ap.x[a], y: ap.y[a], z: ap.z[a], hx: ap.hx[a], hz: ap.hz[a], right: ap.right[a] } : null; },
    get controls() { return tcOuter; },
    dispose() {
      for (const m of group.children) { m.geometry.dispose(); }
      for (const m of [metal, housingMat, plateMat, backMat, lamps.material]) m.dispose();
      atlas.dispose();
    },
  };
}
