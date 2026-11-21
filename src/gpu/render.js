// Minimal WebGPU renderer that draws the GPU sim straight from its storage buffers (no
// readback): road ribbons (one static vertex buffer) + every vehicle drawn either as an
// instanced low-poly box (when it is at least `minPx` pixels long on screen) or as a
// screen-space dot of `dotPx` pixels (when it would be smaller), so traffic density reads
// even at the whole-bay zoom. Colour by speed (red -> yellow -> green) or by state.
//
//   const r = new GPURenderer(sim, canvas);
//   each frame: r.render(camera.update(aspect))     camera = OrbitCamera (below)
//   r.colorMode = 'speed' | 'state'; r.roadDim (0..1, auto from camera distance in gpu.html)
//   const slot = await r.pick(clientX, clientY, camera)   // nearest car on screen (CPU, one pose readback)
//   const f = new Follower(sim); f.slot = slot; f.poll() -> {pose, speed, state, ...}
//
// For Three.js scenes use GPUSim.readPoses() (async readback) + an InstancedMesh instead.

import { VEH_WORDS, PAR_WORDS } from './layout.js';

const ROAD_COLORS = { // by HIGHWAY_RANK
  7: [0.62, 0.55, 0.42], 6: [0.52, 0.48, 0.40], 5: [0.44, 0.44, 0.46], 4: [0.40, 0.40, 0.42], 3: [0.34, 0.34, 0.36], 2: [0.30, 0.30, 0.32], 1: [0.26, 0.26, 0.28], 0: [0.24, 0.24, 0.26],
};

// uniform: viewProj | eye | light | car (scale, policyTint, -, -) | view (w px, h px, minPx, dotPx) | mode (colorMode, roadDim, highlight slot, -)
const U_WGSL = /* wgsl */`
struct U { viewProj: mat4x4f, eye: vec4f, light: vec4f, car: vec4f, view: vec4f, mode: vec4f };
@group(0) @binding(0) var<uniform> u: U;
`;

const ROAD_WGSL = U_WGSL + /* wgsl */`
struct VO { @builtin(position) pos: vec4f, @location(0) col: vec3f };
@vertex fn vs(@location(0) p: vec3f, @location(1) c: vec3f) -> VO {
  var o: VO; o.pos = u.viewProj * vec4f(p, 1.0); o.col = c * u.mode.y; return o;
}
@fragment fn fs(i: VO) -> @location(0) vec4f { return vec4f(i.col, 1.0); }
`;

const CAR_COMMON = U_WGSL + /* wgsl */`
const VW: u32 = ${VEH_WORDS}u;
const PW: u32 = ${PAR_WORDS}u;
@group(0) @binding(1) var<storage, read> pose: array<vec4f>;
@group(0) @binding(2) var<storage, read> veh: array<u32>;   // VEH_WORDS per slot (src/gpu/layout.js)
@group(0) @binding(3) var<storage, read> par: array<u32>;   // PAR_WORDS per slot
struct VO { @builtin(position) pos: vec4f, @location(0) col: vec3f };
const TYPE_COL = array<vec3f, 4>(vec3f(0.86,0.86,0.90), vec3f(0.70,0.78,0.92), vec3f(0.62,0.70,0.58), vec3f(0.95,0.80,0.40));
fn visible(ii: u32) -> bool { let st = veh[ii * VW + 7u]; return st != 0u && st != 5u; }
// pixels the car's length spans on screen at its position
fn screenLen(p: vec4f, len: f32) -> f32 {
  let c0 = u.viewProj * vec4f(p.xyz, 1.0);
  let c1 = u.viewProj * vec4f(p.x + cos(p.w) * len, p.y, p.z + sin(p.w) * len, 1.0);
  if (c0.w <= 0.0 || c1.w <= 0.0) { return 1e9; }
  let d = (c1.xy / c1.w - c0.xy / c0.w) * 0.5 * u.view.xy;
  let dh = (u.viewProj * vec4f(p.x, p.y + len, p.z, 1.0));
  let dv = (dh.xy / dh.w - c0.xy / c0.w) * 0.5 * u.view.xy;
  return max(length(d), length(dv));
}
fn carColor(ii: u32) -> vec3f {
  let vb = ii * VW;
  let state = veh[vb + 7u]; let flags = veh[vb + 10u];
  let speed = bitcast<f32>(veh[vb + 2u]); let accel = bitcast<f32>(veh[vb + 3u]); let brake = bitcast<f32>(veh[vb + 27u]);
  if (ii == u32(u.mode.z)) { return vec3f(0.2, 0.95, 1.0); }                          // followed car
  if (state == 2u) {
    if ((flags & 64u) != 0u) { return vec3f(0.85, 0.25, 1.0); }                       // stranded
    return select(vec3f(1.0, 1.0, 1.0), vec3f(0.6, 0.6, 0.6), (flags & 8u) != 0u);   // crash scene (shoulder dim)
  }
  if (u.mode.x < 0.5) {
    // speed ramp: 0 red -> 13 m/s yellow -> 27 m/s (60 mph) green
    let t = clamp(speed / 27.0, 0.0, 1.0);
    return select(mix(vec3f(1.0, 0.85, 0.1), vec3f(0.2, 1.0, 0.35), (t - 0.5) * 2.0), mix(vec3f(1.0, 0.12, 0.08), vec3f(1.0, 0.85, 0.1), t * 2.0), t < 0.5);
  }
  var col = TYPE_COL[min(par[ii * PW + 2u], 3u)];
  if (accel < -1.5 || brake > 0.3) { col = vec3f(1.0, 0.62, 0.15); }
  if ((flags & 4u) != 0u && u.car.y > 0.5) { col = vec3f(0.45, 0.85, 1.0); }
  if ((flags & 128u) != 0u) { col = vec3f(0.2, 0.9, 0.4); }                          // heading to a gas station
  return col;
}
`;

// box pass: 72 vertices (body + cabin) per instance; collapsed when the car is under minPx on screen
const CAR_WGSL = CAR_COMMON + /* wgsl */`
const CORNERS = array<vec3f, 8>(vec3f(-0.5,0,-0.5), vec3f(0.5,0,-0.5), vec3f(0.5,0,0.5), vec3f(-0.5,0,0.5), vec3f(-0.5,1,-0.5), vec3f(0.5,1,-0.5), vec3f(0.5,1,0.5), vec3f(-0.5,1,0.5));
const IDX = array<u32, 36>(0u,2u,1u, 0u,3u,2u, 4u,5u,6u, 4u,6u,7u, 0u,1u,5u, 0u,5u,4u, 2u,3u,7u, 2u,7u,6u, 1u,2u,6u, 1u,6u,5u, 3u,0u,4u, 3u,4u,7u);
const NRM = array<vec3f, 6>(vec3f(0,-1,0), vec3f(0,1,0), vec3f(0,0,-1), vec3f(0,0,1), vec3f(1,0,0), vec3f(-1,0,0));
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VO {
  var o: VO;
  o.pos = vec4f(0.0, 0.0, -2.0, 1.0); o.col = vec3f(0.0);
  if (!visible(ii)) { return o; }
  let p = pose[ii];
  let pb = ii * PW;
  let typeIdx = par[pb + 2u]; let plen = bitcast<f32>(par[pb + 4u]); let pwid = bitcast<f32>(par[pb + 5u]);
  if (screenLen(p, plen * u.car.x) < u.view.z) { return o; }
  let box = vi / 36u;            // 0 body, 1 cabin
  let k = vi % 36u;
  let c = CORNERS[IDX[k]];
  let tall = typeIdx >= 2u;
  var size = vec3f(plen, select(0.75, 2.6, tall), pwid);
  var off = vec3f(0.0);
  if (box == 1u) {
    size = vec3f(plen * select(0.5, 0.25, tall), 0.6, pwid * 0.88);
    off = vec3f(select(-0.05 * plen, 0.37 * plen, tall), select(0.75, 2.6, tall) - 0.02, 0.0);
  }
  let lp = (c * size + off) * u.car.x;
  let ch = cos(p.w); let sh = sin(p.w);
  // car x = forward (cos h, sin h) in world xz, car z = right (-sin h, cos h)
  let wp = vec3f(p.x + lp.x * ch - lp.z * sh, p.y + 0.25 + lp.y, p.z + lp.x * sh + lp.z * ch);
  o.pos = u.viewProj * vec4f(wp, 1.0);
  let n0 = NRM[k / 6u];
  let n = vec3f(n0.x * ch - n0.z * sh, n0.y, n0.x * sh + n0.z * ch);
  let lit = 0.45 + 0.55 * max(dot(n, normalize(u.light.xyz)), 0.0);
  o.col = carColor(ii) * lit;
  return o;
}
@fragment fn fs(i: VO) -> @location(0) vec4f { return vec4f(i.col, 1.0); }
`;

// dot pass: a dotPx x dotPx screen-space square per car that is smaller than minPx on screen
const DOT_WGSL = CAR_COMMON + /* wgsl */`
const Q = array<vec2f, 6>(vec2f(-1,-1), vec2f(1,-1), vec2f(1,1), vec2f(-1,-1), vec2f(1,1), vec2f(-1,1));
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VO {
  var o: VO;
  o.pos = vec4f(0.0, 0.0, -2.0, 1.0); o.col = vec3f(0.0);
  if (!visible(ii)) { return o; }
  let p = pose[ii];
  let plen = bitcast<f32>(par[ii * PW + 4u]);
  if (screenLen(p, plen * u.car.x) >= u.view.z) { return o; }
  let c = u.viewProj * vec4f(p.x, p.y + 1.0, p.z, 1.0);
  if (c.w <= 0.0) { return o; }
  var px = u.view.w;
  if (ii == u32(u.mode.z)) { px = px * 3.0; }
  let d = Q[vi] * (0.5 * px) / (0.5 * u.view.xy) * c.w;
  o.pos = vec4f(c.xy + d, c.z * 0.999, c.w);      // pulled slightly toward the camera so dots sit on top of the roads
  o.col = carColor(ii);
  return o;
}
@fragment fn fs(i: VO) -> @location(0) vec4f { return vec4f(i.col, 1.0); }
`;

export class GPURenderer {
  constructor(sim, canvas, { graph = sim.graph, sampleCount = 4, background = [0.07, 0.09, 0.12] } = {}) {
    this.sim = sim; this.canvas = canvas; this.device = sim.device; this.sampleCount = sampleCount;
    this.ctx = canvas.getContext('webgpu');
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.ctx.configure({ device: this.device, format: this.format, alphaMode: 'opaque' });
    this.background = background;
    this.uniform = this.device.createBuffer({ size: 144, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.uData = new Float32Array(36);
    this.carScale = 1; this.policyTint = false;
    this.minPx = 3; this.dotPx = 2.5;          // device pixels (scaled by devicePixelRatio in gpu.html)
    this.colorMode = 'speed'; this.roadDim = 1; this.highlight = -1;
    this.boxes = true;                          // false skips the box pass (every car is a dot at this zoom)
    this._buildRoads(graph);
    this._pipelines();
  }

  _buildRoads(g) {
    // one ribbon per physical road (skip the twin of two-way roads), width = all lanes
    let segs = 0;
    for (const e of g.edges) { if (e.reverse >= 0 && e.reverse < e.idx) continue; segs += e.px.length - 1; }
    const verts = new Float32Array(segs * 6 * 6);
    let k = 0;
    const put = (x, y, z, c) => { verts[k++] = x; verts[k++] = y; verts[k++] = z; verts[k++] = c[0]; verts[k++] = c[1]; verts[k++] = c[2]; };
    for (const e of g.edges) {
      if (e.reverse >= 0 && e.reverse < e.idx) continue;
      const lanesTot = e.lanes + (e.oppLanes || (e.reverse >= 0 ? g.edges[e.reverse].lanes : 0));
      const w = Math.max(3, lanesTot * 3.7) * 0.5;
      const col = ROAD_COLORS[e.rank] || ROAD_COLORS[1];
      for (let i = 0; i + 1 < e.px.length; i++) {
        const x0 = e.px[i], z0 = e.pz[i], x1 = e.px[i + 1], z1 = e.pz[i + 1];
        const y0 = (e.ys ? e.ys[i] : e.y) + 0.05, y1 = (e.ys ? e.ys[i + 1] : e.y) + 0.05;
        let dx = x1 - x0, dz = z1 - z0; const L = Math.hypot(dx, dz) || 1; dx /= L; dz /= L;
        const nx = -dz * w, nz = dx * w;
        put(x0 - nx, y0, z0 - nz, col); put(x1 - nx, y1, z1 - nz, col); put(x1 + nx, y1, z1 + nz, col);
        put(x0 - nx, y0, z0 - nz, col); put(x1 + nx, y1, z1 + nz, col); put(x0 + nx, y0, z0 + nz, col);
      }
    }
    this.roadCount = k / 6;
    this.roadBuf = this.device.createBuffer({ size: Math.max(16, verts.byteLength), usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(this.roadBuf, 0, verts);
  }

  _pipelines() {
    const d = this.device, ms = { count: this.sampleCount };
    const depth = { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less-equal' };
    const roadMod = d.createShaderModule({ code: ROAD_WGSL });
    this.roadPipe = d.createRenderPipeline({
      layout: 'auto', primitive: { topology: 'triangle-list', cullMode: 'none' }, depthStencil: depth, multisample: ms,
      vertex: { module: roadMod, entryPoint: 'vs', buffers: [{ arrayStride: 24, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }, { shaderLocation: 1, offset: 12, format: 'float32x3' }] }] },
      fragment: { module: roadMod, entryPoint: 'fs', targets: [{ format: this.format }] },
    });
    this.roadBG = d.createBindGroup({ layout: this.roadPipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: this.uniform } }] });
    const mk = (code) => { const m = d.createShaderModule({ code }); return d.createRenderPipeline({ layout: 'auto', primitive: { topology: 'triangle-list', cullMode: 'none' }, depthStencil: depth, multisample: ms, vertex: { module: m, entryPoint: 'vs' }, fragment: { module: m, entryPoint: 'fs', targets: [{ format: this.format }] } }); };
    this.carPipe = mk(CAR_WGSL);
    this.dotPipe = mk(DOT_WGSL);
    this._carBG();
  }
  _carBG() {
    const b = this.sim.gpu.buffers;
    const entries = [{ binding: 0, resource: { buffer: this.uniform } }, { binding: 1, resource: { buffer: b.get('pose') } }, { binding: 2, resource: { buffer: b.get('veh') } }, { binding: 3, resource: { buffer: b.get('par') } }];
    this.carBG = this.device.createBindGroup({ layout: this.carPipe.getBindGroupLayout(0), entries });
    this.dotBG = this.device.createBindGroup({ layout: this.dotPipe.getBindGroupLayout(0), entries });
  }

  _targets() {
    const c = this.canvas, w = c.width, h = c.height;
    if (this._w === w && this._h === h) return;
    this._w = w; this._h = h;
    this.depthTex?.destroy(); this.msTex?.destroy();
    this.depthTex = this.device.createTexture({ size: [w, h], format: 'depth24plus', sampleCount: this.sampleCount, usage: GPUTextureUsage.RENDER_ATTACHMENT });
    if (this.sampleCount > 1) this.msTex = this.device.createTexture({ size: [w, h], format: this.format, sampleCount: this.sampleCount, usage: GPUTextureUsage.RENDER_ATTACHMENT });
  }

  /** Draw one frame. camera: {viewProj: Float32Array(16), eye: [x,y,z]}. */
  render(camera) {
    this._targets();
    const u = this.uData;
    u.set(camera.viewProj, 0); u[16] = camera.eye[0]; u[17] = camera.eye[1]; u[18] = camera.eye[2];
    u[20] = 0.4; u[21] = 0.8; u[22] = 0.3;
    u[24] = this.carScale; u[25] = this.policyTint ? 1 : 0;
    u[28] = this.canvas.width; u[29] = this.canvas.height; u[30] = this.minPx; u[31] = this.dotPx;
    u[32] = this.colorMode === 'state' ? 1 : 0; u[33] = this.roadDim; u[34] = this.highlight >= 0 ? this.highlight : 4294967295;
    this.device.queue.writeBuffer(this.uniform, 0, u);
    const enc = this.device.createCommandEncoder();
    const view = this.ctx.getCurrentTexture().createView();
    const [r, g, b] = this.background;
    const pass = enc.beginRenderPass({
      colorAttachments: [{ view: this.sampleCount > 1 ? this.msTex.createView() : view, resolveTarget: this.sampleCount > 1 ? view : undefined, clearValue: { r, g, b, a: 1 }, loadOp: 'clear', storeOp: this.sampleCount > 1 ? 'discard' : 'store' }],
      depthStencilAttachment: { view: this.depthTex.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'discard' },
    });
    pass.setPipeline(this.roadPipe); pass.setBindGroup(0, this.roadBG); pass.setVertexBuffer(0, this.roadBuf); pass.draw(this.roadCount);
    if (this.boxes !== false) { pass.setPipeline(this.carPipe); pass.setBindGroup(0, this.carBG); pass.draw(72, this.sim.cap); }
    pass.setPipeline(this.dotPipe); pass.setBindGroup(0, this.dotBG); pass.draw(6, this.sim.cap);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  /**
   * Nearest visible car to a screen point (CSS px), within maxPx: one pose readback (16 B per
   * slot) + a CPU projection pass. Returns the slot or -1.
   */
  async pick(clientX, clientY, camera, maxPx = 18) {
    const poses = await this.sim.readPoses();
    const rect = this.canvas.getBoundingClientRect();
    const sx = (clientX - rect.left) / rect.width * 2 - 1, sy = 1 - (clientY - rect.top) / rect.height * 2;
    const m = camera.viewProj, W = rect.width / 2, H = rect.height / 2;
    let best = -1, bestD = maxPx * maxPx;
    for (let i = 0, n = poses.length / 4; i < n; i++) {
      const y = poses[4 * i + 1]; if (y < -999) continue;
      const x = poses[4 * i], z = poses[4 * i + 2];
      const cw = m[3] * x + m[7] * y + m[11] * z + m[15]; if (cw <= 0) continue;
      const cx = (m[0] * x + m[4] * y + m[8] * z + m[12]) / cw, cy = (m[1] * x + m[5] * y + m[9] * z + m[13]) / cw;
      const dx = (cx - sx) * W, dy = (cy - sy) * H, d = dx * dx + dy * dy;
      if (d < bestD) { bestD = d; best = i; }
    }
    return best;
  }
}

/**
 * Follow one car: every poll() copies that slot's pose (16 B) and vehicle record (VEH_WORDS)
 * into a small staging buffer (ring of 2, never stalls) and returns the latest values.
 */
export class Follower {
  constructor(sim) {
    this.sim = sim; this.slot = -1; this.info = null;
    const d = sim.device, size = 16 + VEH_WORDS * 4 + PAR_WORDS * 4;
    this.size = size;
    this.bufs = [0, 1].map(() => ({ b: d.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }), busy: false }));
  }
  poll() {
    if (this.slot < 0) return null;
    const st = this.bufs.find((x) => !x.busy);
    if (st) {
      const slot = this.slot, gb = this.sim.gpu.buffers, enc = this.sim.device.createCommandEncoder();
      enc.copyBufferToBuffer(gb.get('pose'), slot * 16, st.b, 0, 16);
      enc.copyBufferToBuffer(gb.get('veh'), slot * VEH_WORDS * 4, st.b, 16, VEH_WORDS * 4);
      enc.copyBufferToBuffer(gb.get('par'), slot * PAR_WORDS * 4, st.b, 16 + VEH_WORDS * 4, PAR_WORDS * 4);
      this.sim.device.queue.submit([enc.finish()]);
      st.busy = true;
      st.b.mapAsync(GPUMapMode.READ).then(() => {
        const a = st.b.getMappedRange().slice(0); st.b.unmap(); st.busy = false;
        if (slot !== this.slot) return;
        const f = new Float32Array(a), u = new Uint32Array(a), V = 4, Pp = 4 + VEH_WORDS;
        const lanes = u[V + 14];
        this.info = {
          slot, x: f[0], y: f[1], z: f[2], heading: f[3], state: u[V + 7], flags: u[V + 10], speed: f[V + 2], accel: f[V + 3],
          edge: u[V + 0], s: f[V + 1], lat: f[V + 4], lane: lanes & 255, steer: f[V + 6], gas: f[V + 26], brake: f[V + 27],
          fuel: f[V + 16], tank: f[V + 17], refuelState: u[V + 24], routePos: u[V + 8], routeLen: u[Pp + 1], trip: u[Pp + 15], typeIdx: u[Pp + 2], free: u[V + 7] === 0,
        };
      }).catch(() => { st.busy = false; });
    }
    return this.info;
  }
}

// ---------------------------------------------------------------- camera
export class OrbitCamera {
  constructor({ target = [0, 0, 0], distance = 60000, yaw = 0, pitch = 1.2, fov = 0.8 } = {}) {
    Object.assign(this, { target: [...target], distance, yaw, pitch, fov });
    this.viewProj = new Float32Array(16); this.eye = [0, 0, 0]; this.aspect = 1;
    this.onClick = null; this.onDblClick = null; this.onUserMove = null;
  }
  /** Mouse: drag = pan, shift/right-drag = rotate, wheel = zoom; click / dblclick callbacks get (clientX, clientY). */
  attach(el) {
    let drag = null;
    el.addEventListener('contextmenu', (e) => e.preventDefault());
    el.addEventListener('pointerdown', (e) => { drag = { x: e.clientX, y: e.clientY, x0: e.clientX, y0: e.clientY, rot: e.button === 2 || e.shiftKey, moved: false }; el.setPointerCapture(e.pointerId); });
    el.addEventListener('pointerup', (e) => {
      if (drag && !drag.moved && e.button === 0 && this.onClick) this.onClick(e.clientX, e.clientY);
      drag = null;
    });
    el.addEventListener('pointermove', (e) => {
      if (!drag) return;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y; drag.x = e.clientX; drag.y = e.clientY;
      if (Math.abs(e.clientX - drag.x0) + Math.abs(e.clientY - drag.y0) > 4) { if (!drag.moved && this.onUserMove) this.onUserMove(drag.rot); drag.moved = true; }
      if (!drag.moved) return;
      if (drag.rot) { this.yaw -= dx * 0.005; this.pitch = Math.min(1.55, Math.max(0.08, this.pitch + dy * 0.005)); }
      else {
        const k = this.distance * 0.0015, c = Math.cos(this.yaw), s = Math.sin(this.yaw);
        this.target[0] -= (dx * c + dy * s) * k; this.target[2] -= (-dx * s + dy * c) * k;
      }
    });
    el.addEventListener('dblclick', (e) => { if (this.onDblClick) this.onDblClick(e.clientX, e.clientY); });
    el.addEventListener('wheel', (e) => { e.preventDefault(); this.distance = Math.min(2e5, Math.max(20, this.distance * Math.exp(e.deltaY * 0.0015))); }, { passive: false });
  }
  update(aspect) {
    this.aspect = aspect;
    const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    // yaw 0 looks north (-z); the camera sits south of the target
    const ex = this.target[0] + this.distance * cp * Math.sin(this.yaw);
    const ez = this.target[2] + this.distance * cp * Math.cos(this.yaw);
    const ey = this.target[1] + this.distance * sp;
    this.eye = [ex, ey, ez];
    const view = lookAt(this.eye, this.target, [0, 1, 0]);
    const near = Math.max(0.5, this.distance * 0.002), far = this.distance * 6 + 2000;
    mul(this.viewProj, perspective(this.fov, aspect, near, far), view);
    return this;
  }
  /** World point on the plane y = planeY under a screen point (client px on element el), or null. */
  groundAt(clientX, clientY, el, planeY = 0) {
    const r = el.getBoundingClientRect();
    const nx = (clientX - r.left) / r.width * 2 - 1, ny = 1 - (clientY - r.top) / r.height * 2;
    const inv = invert(this.viewProj); if (!inv) return null;
    const unproj = (z) => { const x = inv[0] * nx + inv[4] * ny + inv[8] * z + inv[12], y = inv[1] * nx + inv[5] * ny + inv[9] * z + inv[13], zz = inv[2] * nx + inv[6] * ny + inv[10] * z + inv[14], w = inv[3] * nx + inv[7] * ny + inv[11] * z + inv[15]; return [x / w, y / w, zz / w]; };
    const a = unproj(0), b = unproj(1);
    const dy = b[1] - a[1]; if (Math.abs(dy) < 1e-9) return null;
    const t = (planeY - a[1]) / dy; if (t < 0) return null;
    return [a[0] + (b[0] - a[0]) * t, planeY, a[2] + (b[2] - a[2]) * t];
  }
}

function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
  // WebGPU clip space z in [0, 1]
  return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, far * nf, -1, 0, 0, far * near * nf, 0]);
}
function lookAt(eye, at, up) {
  let zx = eye[0] - at[0], zy = eye[1] - at[1], zz = eye[2] - at[2];
  let l = Math.hypot(zx, zy, zz) || 1; zx /= l; zy /= l; zz /= l;
  let xx = up[1] * zz - up[2] * zy, xy = up[2] * zx - up[0] * zz, xz = up[0] * zy - up[1] * zx;
  l = Math.hypot(xx, xy, xz) || 1; xx /= l; xy /= l; xz /= l;
  const yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;
  return new Float32Array([xx, yx, zx, 0, xy, yy, zy, 0, xz, yz, zz, 0,
    -(xx * eye[0] + xy * eye[1] + xz * eye[2]), -(yx * eye[0] + yy * eye[1] + yz * eye[2]), -(zx * eye[0] + zy * eye[1] + zz * eye[2]), 1]);
}
function mul(out, a, b) {
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    let s = 0; for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
    out[c * 4 + r] = s;
  }
  return out;
}
function invert(m) {
  const a = Array.from(m), inv = new Float64Array(16);
  inv[0] = a[5] * a[10] * a[15] - a[5] * a[11] * a[14] - a[9] * a[6] * a[15] + a[9] * a[7] * a[14] + a[13] * a[6] * a[11] - a[13] * a[7] * a[10];
  inv[4] = -a[4] * a[10] * a[15] + a[4] * a[11] * a[14] + a[8] * a[6] * a[15] - a[8] * a[7] * a[14] - a[12] * a[6] * a[11] + a[12] * a[7] * a[10];
  inv[8] = a[4] * a[9] * a[15] - a[4] * a[11] * a[13] - a[8] * a[5] * a[15] + a[8] * a[7] * a[13] + a[12] * a[5] * a[11] - a[12] * a[7] * a[9];
  inv[12] = -a[4] * a[9] * a[14] + a[4] * a[10] * a[13] + a[8] * a[5] * a[14] - a[8] * a[6] * a[13] - a[12] * a[5] * a[10] + a[12] * a[6] * a[9];
  inv[1] = -a[1] * a[10] * a[15] + a[1] * a[11] * a[14] + a[9] * a[2] * a[15] - a[9] * a[3] * a[14] - a[13] * a[2] * a[11] + a[13] * a[3] * a[10];
  inv[5] = a[0] * a[10] * a[15] - a[0] * a[11] * a[14] - a[8] * a[2] * a[15] + a[8] * a[3] * a[14] + a[12] * a[2] * a[11] - a[12] * a[3] * a[10];
  inv[9] = -a[0] * a[9] * a[15] + a[0] * a[11] * a[13] + a[8] * a[1] * a[15] - a[8] * a[3] * a[13] - a[12] * a[1] * a[11] + a[12] * a[3] * a[9];
  inv[13] = a[0] * a[9] * a[14] - a[0] * a[10] * a[13] - a[8] * a[1] * a[14] + a[8] * a[2] * a[13] + a[12] * a[1] * a[10] - a[12] * a[2] * a[9];
  inv[2] = a[1] * a[6] * a[15] - a[1] * a[7] * a[14] - a[5] * a[2] * a[15] + a[5] * a[3] * a[14] + a[13] * a[2] * a[7] - a[13] * a[3] * a[6];
  inv[6] = -a[0] * a[6] * a[15] + a[0] * a[7] * a[14] + a[4] * a[2] * a[15] - a[4] * a[3] * a[14] - a[12] * a[2] * a[7] + a[12] * a[3] * a[6];
  inv[10] = a[0] * a[5] * a[15] - a[0] * a[7] * a[13] - a[4] * a[1] * a[15] + a[4] * a[3] * a[13] + a[12] * a[1] * a[7] - a[12] * a[3] * a[5];
  inv[14] = -a[0] * a[5] * a[14] + a[0] * a[6] * a[13] + a[4] * a[1] * a[14] - a[4] * a[2] * a[13] - a[12] * a[1] * a[6] + a[12] * a[2] * a[5];
  inv[3] = -a[1] * a[6] * a[11] + a[1] * a[7] * a[10] + a[5] * a[2] * a[11] - a[5] * a[3] * a[10] - a[9] * a[2] * a[7] + a[9] * a[3] * a[6];
  inv[7] = a[0] * a[6] * a[11] - a[0] * a[7] * a[10] - a[4] * a[2] * a[11] + a[4] * a[3] * a[10] + a[8] * a[2] * a[7] - a[8] * a[3] * a[6];
  inv[11] = -a[0] * a[5] * a[11] + a[0] * a[7] * a[9] + a[4] * a[1] * a[11] - a[4] * a[3] * a[9] - a[8] * a[1] * a[7] + a[8] * a[3] * a[5];
  inv[15] = a[0] * a[5] * a[10] - a[0] * a[6] * a[9] - a[4] * a[1] * a[10] + a[4] * a[2] * a[9] + a[8] * a[1] * a[6] - a[8] * a[2] * a[5];
  let det = a[0] * inv[0] + a[1] * inv[4] + a[2] * inv[8] + a[3] * inv[12];
  if (!det) return null;
  det = 1 / det;
  for (let i = 0; i < 16; i++) inv[i] *= det;
  return inv;
}
