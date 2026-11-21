// Car LOD for rendering (never imported by sim / brain / GPU compute code).
// Near cars: one InstancedMesh of low-poly clay boxes (name 'cars', userData.lookRole 'cars',
// so src/look styles it). Far cars: one THREE.Points draw with round sprites whose size is
// the car's projected length, clamped to at least `minPixelSize` px. Two draw calls total.
//
//   import { createCarLOD } from './src/stream/cars.js';
//   const cars = createCarLOD({ scene, camera, renderer, maxCars: 100000 });
//   // each frame, from the CPU sim or a GPU readback:
//   cars.update(positions /* Float32Array x,y,z per car */, headings /* rad, atan2(dz,dx) */, count, { colors });
//   cars.updateFromSim(sim)          // CPU sim (src/sim VehicleStore: sim.vs.x/y/z/heading/state/...)
//   cars.updateFromPoses(poses)      // GPU readback: await gpuSim.readPoses() -> Float32Array x,y,z,heading per slot
//                                    // (y = -1000 marks a free slot), optional { count, colors, types }
//
// Heading convention: travel direction (cos h, sin h) in the x/z plane (x east, z south),
// i.e. h = atan2(dz, dx); the car's right side is (-sin h, cos h) as in data/roads.schema.md.
import * as THREE from 'three';

/** Body + cabin, 48 vertices, unit size (x forward, y up, z right). */
export function makeCarGeometry() {
  const boxes = [[0, 0.28, 0, 1.0, 0.56, 1.0], [-0.05, 0.75, 0, 0.5, 0.38, 0.84]];
  const pos = [], nor = [], idx = [];
  const faces = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
  let vb = 0;
  for (const [cx, cy, cz, sx, sy, sz] of boxes) for (const [nx, ny, nz] of faces) {
    const u = nx ? [0, 1, 0] : [1, 0, 0];
    const v = [ny * u[2] - nz * u[1], nz * u[0] - nx * u[2], nx * u[1] - ny * u[0]];
    for (const [a, b] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
      pos.push(cx + (nx + u[0] * a + v[0] * b) * 0.5 * sx, cy + (ny + u[1] * a + v[1] * b) * 0.5 * sy, cz + (nz + u[2] * a + v[2] * b) * 0.5 * sz);
      nor.push(nx, ny, nz);
    }
    // u x v == n, so (0,1,2) is counter-clockwise seen from outside
    idx.push(vb, vb + 1, vb + 2, vb, vb + 2, vb + 3);
    vb += 4;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3));
  g.setIndex(idx);
  return g;
}

const DOT_VS = /* glsl */`
uniform float uScale; uniform float uMinPx; uniform float uSize;
attribute vec3 color;
varying vec3 vColor;
void main() {
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mv;
  gl_PointSize = max(uMinPx, uSize * uScale / max(1.0, -mv.z));
  vColor = color;
}`;
const DOT_FS = /* glsl */`
varying vec3 vColor;
void main() {
  vec2 d = gl_PointCoord - 0.5;
  if (dot(d, d) > 0.25) discard;
  gl_FragColor = vec4(vColor, 1.0);
}`;

/**
 * opts: scene, camera, renderer, maxCars=120000, nearDistance=700 (m), maxNear=30000,
 *       minPixelSize=2, length=4.6, width=1.9, height=1.5, color=[0.85,0.86,0.9] (default car colour)
 */
export function createCarLOD(opts) {
  const { scene, camera, renderer } = opts;
  const max = opts.maxCars ?? 120000, maxNear = Math.min(max, opts.maxNear ?? 30000);
  const L = opts.length ?? 4.6, W = opts.width ?? 1.9, H = opts.height ?? 1.5;
  const defColor = opts.color || [0.85, 0.86, 0.9];
  const cfg = { nearDistance: opts.nearDistance ?? 700, minPixelSize: opts.minPixelSize ?? 2 };

  const boxes = new THREE.InstancedMesh(makeCarGeometry(), new THREE.MeshLambertMaterial({ color: 0xffffff }), maxNear);
  boxes.name = 'cars'; boxes.userData.lookRole = 'cars';
  boxes.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  boxes.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(maxNear * 3), 3);
  boxes.instanceColor.setUsage(THREE.DynamicDrawUsage);
  boxes.count = 0; boxes.frustumCulled = false;

  const dotPos = new Float32Array(max * 3), dotCol = new Float32Array(max * 3);
  const dg = new THREE.BufferGeometry();
  const pa = new THREE.BufferAttribute(dotPos, 3).setUsage(THREE.DynamicDrawUsage);
  const ca = new THREE.BufferAttribute(dotCol, 3).setUsage(THREE.DynamicDrawUsage);
  dg.setAttribute('position', pa); dg.setAttribute('color', ca);
  dg.setDrawRange(0, 0);
  const dmat = new THREE.ShaderMaterial({ vertexShader: DOT_VS, fragmentShader: DOT_FS,
    uniforms: { uScale: { value: 500 }, uMinPx: { value: cfg.minPixelSize }, uSize: { value: L } } });
  const dots = new THREE.Points(dg, dmat);
  dots.name = 'carDots'; dots.userData.lookRole = 'carDots'; dots.frustumCulled = false;
  scene.add(boxes, dots);

  const camPos = new THREE.Vector3();
  let nNear = 0, nFar = 0;
  const m = boxes.instanceMatrix.array, ic = boxes.instanceColor.array;
  let cx = 0, cy = 0, cz = 0, near2 = 0;
  function begin() {
    camera.getWorldPosition(camPos);
    const vh = renderer.domElement.height || 1000;
    dmat.uniforms.uScale.value = camera.isPerspectiveCamera ? vh / (2 * Math.tan((camera.fov * Math.PI) / 360)) : vh / ((camera.top - camera.bottom) / camera.zoom);
    dmat.uniforms.uMinPx.value = cfg.minPixelSize;
    near2 = cfg.nearDistance * cfg.nearDistance;
    cx = camPos.x; cy = camPos.y; cz = camPos.z;
    nNear = 0; nFar = 0;
  }
  /** One car: near -> box (len, wid, hgt in metres), far -> dot. */
  function put(x, y, z, h, len, wid, hgt, r, g, b) {
    const dx = x - cx, dy = y - cy, dz = z - cz;
    if (dx * dx + dy * dy + dz * dz < near2 && nNear < maxNear) {
      const c = Math.cos(h), s = Math.sin(h), o = nNear * 16;
      m[o] = c * len; m[o + 1] = 0; m[o + 2] = s * len; m[o + 3] = 0;
      m[o + 4] = 0; m[o + 5] = hgt; m[o + 6] = 0; m[o + 7] = 0;
      m[o + 8] = -s * wid; m[o + 9] = 0; m[o + 10] = c * wid; m[o + 11] = 0;
      m[o + 12] = x; m[o + 13] = y; m[o + 14] = z; m[o + 15] = 1;
      ic[nNear * 3] = r; ic[nNear * 3 + 1] = g; ic[nNear * 3 + 2] = b;
      nNear++;
    } else if (nFar < max) {
      const o = nFar * 3;
      dotPos[o] = x; dotPos[o + 1] = y + 1; dotPos[o + 2] = z;
      dotCol[o] = r; dotCol[o + 1] = g; dotCol[o + 2] = b;
      nFar++;
    }
  }
  function commit() {
    boxes.count = nNear;
    boxes.instanceMatrix.clearUpdateRanges?.(); boxes.instanceMatrix.addUpdateRange?.(0, nNear * 16); boxes.instanceMatrix.needsUpdate = true;
    boxes.instanceColor.clearUpdateRanges?.(); boxes.instanceColor.addUpdateRange?.(0, nNear * 3); boxes.instanceColor.needsUpdate = true;
    pa.clearUpdateRanges?.(); pa.addUpdateRange?.(0, nFar * 3); pa.needsUpdate = true;
    ca.clearUpdateRanges?.(); ca.addUpdateRange?.(0, nFar * 3); ca.needsUpdate = true;
    dg.setDrawRange(0, nFar);
  }

  // colours as src/sim/render.js (type tint, braking amber, crashed red, shoulder dark red)
  const TYPE_COL = [[0.80, 0.80, 0.84], [0.72, 0.76, 0.86], [0.62, 0.66, 0.60], [0.90, 0.78, 0.45]];
  const BRAKE = [1.0, 0.62, 0.15], CRASH = [1.0, 0.12, 0.10], SHOULDER = [0.55, 0.10, 0.08];
  const TYPE_DIM = [[4.6, 1.85, 1.45], [4.9, 1.95, 1.75], [5.6, 2.05, 1.9], [12.0, 2.55, 3.2]];

  /** CPU sim: reads sim.vs (struct of arrays, src/sim/LAYOUT.md); free slots skipped. */
  function updateFromSim(sim, o = {}) {
    const vs = sim.vs || sim, high = vs.high ?? vs.state.length, yOff = o.yOffset ?? 0;
    begin();
    for (let i = 0; i < high; i++) {
      const st = vs.state[i];
      if (st === 0 || st === 3 || st === 4) continue;   // FREE / ARRIVED / REMOVED
      const t = vs.typeIdx ? vs.typeIdx[i] : 0;
      let c;
      if (st === 2) c = vs.flags && (vs.flags[i] & 8) ? SHOULDER : CRASH;
      else if (vs.brake && vs.brake[i] > 0.15) c = BRAKE;
      else c = TYPE_COL[t] || TYPE_COL[0];
      put(vs.x[i], vs.y[i] + yOff, vs.z[i], vs.heading[i], vs.length ? vs.length[i] : TYPE_DIM[t][0], vs.width ? vs.width[i] : TYPE_DIM[t][1], t >= 2 ? 3.2 : 1.5, c[0], c[1], c[2]);
    }
    commit();
    return nNear + nFar;
  }

  /** GPU readback (src/gpu gpuSim.readPoses()): vec4 x, y, z, heading per slot; y <= -999 = free slot. */
  function updateFromPoses(poses, o = {}) {
    const n = Math.min(o.count ?? poses.length / 4, poses.length / 4), colors = o.colors, types = o.types, yOff = o.yOffset ?? 0;
    begin();
    for (let i = 0; i < n; i++) {
      const y = poses[i * 4 + 1];
      if (y <= -999) continue;
      const t = types ? types[i] : 0, d = TYPE_DIM[t] || TYPE_DIM[0];
      const c = colors ? null : TYPE_COL[t] || TYPE_COL[0];
      put(poses[i * 4], y + yOff, poses[i * 4 + 2], poses[i * 4 + 3], d[0], d[1], d[2],
        colors ? colors[i * 3] : c[0], colors ? colors[i * 3 + 1] : c[1], colors ? colors[i * 3 + 2] : c[2]);
    }
    commit();
    return nNear + nFar;
  }

  /** Generic arrays: positions (stride 3: x,y,z or 2: x,z), headings (rad), count, { colors, stride }. */
  function update(positions, headings, count, uopts = {}) {
    const colors = uopts.colors, stride = uopts.stride ?? 3;
    count = Math.min(count ?? positions.length / stride, max);
    begin();
    for (let i = 0; i < count; i++) {
      const x = positions[i * stride], y = stride === 3 ? positions[i * stride + 1] : 0, z = positions[i * stride + stride - 1];
      put(x, y, z, headings[i], L, W, H, colors ? colors[i * 3] : defColor[0], colors ? colors[i * 3 + 1] : defColor[1], colors ? colors[i * 3 + 2] : defColor[2]);
    }
    commit();
  }

  return {
    boxes, dots, config: cfg, update, updateFromSim, updateFromPoses,
    stats: () => ({ near: nNear, far: nFar, drawCalls: (nNear ? 1 : 0) + (nFar ? 1 : 0) }),
    dispose() { scene.remove(boxes, dots); boxes.geometry.dispose(); boxes.material.dispose(); boxes.dispose?.(); dg.dispose(); dmat.dispose(); },
  };
}
