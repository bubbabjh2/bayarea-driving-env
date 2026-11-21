// WebGPU device bootstrap + small helpers shared by the GPU sim.
//
//   const gpu = await initGPU();                 // {device, adapter, info, limits, features, timestamps}
//   const buf = gpu.storage('veh', bytes);       // tracked storage buffer (gpu.memoryBytes())
//   const k = await gpu.kernel('integrate', src, {defines}); // compute pipeline, bindings parsed from WGSL
//   const bg = gpu.bindGroup(k, {veh: buf, ...}); // bind by variable NAME (see parseBindings)
//
// Kernels declare `@group(0) @binding(N) var<storage, read_write> name: T;`. The
// pipeline layout is built explicitly from those declarations (so an unused
// binding never breaks a bind group), and bind groups are filled by variable
// name from a registry object. `// #include common` pulls in kernels/common.wgsl.

export const REQUIRED = {
  maxComputeWorkgroupsPerDimension: 65535,
  maxStorageBufferBindingSize: 128 << 20,
  maxStorageBuffersPerShaderStage: 10,
};

export async function initGPU({ powerPreference = 'high-performance', wantTimestamps = true } = {}) {
  if (typeof navigator === 'undefined' || !navigator.gpu) throw new Error('WebGPU unavailable (navigator.gpu missing). Use Chrome/Edge 113+ or Safari 26+, or launch Chrome with --enable-unsafe-webgpu.');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference });
  if (!adapter) throw new Error('WebGPU: no adapter');
  const L = adapter.limits;
  const problems = [];
  for (const [k, v] of Object.entries(REQUIRED)) if ((L[k] ?? 0) < v) problems.push(`${k}=${L[k]} < ${v}`);
  if (problems.length) throw new Error('WebGPU adapter too small: ' + problems.join(', '));
  const requiredFeatures = [];
  const timestamps = wantTimestamps && adapter.features.has('timestamp-query');
  if (timestamps) requiredFeatures.push('timestamp-query');
  if (adapter.features.has('shader-f16')) requiredFeatures.push('shader-f16');
  const requiredLimits = {
    maxStorageBufferBindingSize: L.maxStorageBufferBindingSize,
    maxBufferSize: L.maxBufferSize,
    maxStorageBuffersPerShaderStage: L.maxStorageBuffersPerShaderStage,
    maxComputeWorkgroupsPerDimension: L.maxComputeWorkgroupsPerDimension,
    maxComputeInvocationsPerWorkgroup: L.maxComputeInvocationsPerWorkgroup,
    maxComputeWorkgroupSizeX: L.maxComputeWorkgroupSizeX,
    maxComputeWorkgroupStorageSize: L.maxComputeWorkgroupStorageSize,
  };
  const device = await adapter.requestDevice({ requiredFeatures, requiredLimits });
  let info = adapter.info || {};
  if (!adapter.info && adapter.requestAdapterInfo) { try { info = await adapter.requestAdapterInfo(); } catch { /* ignore */ } }
  device.lost.then((e) => console.error('WebGPU device lost:', e.message));
  device.addEventListener?.('uncapturederror', (e) => console.error('WebGPU error:', e.error?.message || e));
  return new GPUContext(device, adapter, info, timestamps);
}

const SHADER_CACHE = new Map();
async function fetchText(url) {
  if (SHADER_CACHE.has(url)) return SHADER_CACHE.get(url);
  const r = await fetch(url); if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  const t = await r.text(); SHADER_CACHE.set(url, t); return t;
}

/** Load kernels/<name>.wgsl, resolving `// #include <file>` lines (relative to kernels/). */
export async function loadKernelSource(name, base = new URL('./kernels/', import.meta.url)) {
  const src = await fetchText(new URL(name.endsWith('.wgsl') ? name : `${name}.wgsl`, base).href);
  const parts = await Promise.all(src.split('\n').map(async (line) => {
    const m = line.match(/^\s*\/\/\s*#include\s+(\S+)/);
    return m ? loadKernelSource(m[1], base) : line;
  }));
  return parts.join('\n');
}

/** Parse `@group(g) @binding(b) var<space[, access]> name` declarations. */
export function parseBindings(code) {
  const re = /@group\((\d+)\)\s*@binding\((\d+)\)\s*var(?:<\s*(\w+)\s*(?:,\s*(\w+))?\s*>)?\s+(\w+)/g;
  const out = [];
  for (let m; (m = re.exec(code));) out.push({ group: +m[1], binding: +m[2], space: m[3] || 'handle', access: m[4] || 'read', name: m[5] });
  return out;
}

export class GPUContext {
  constructor(device, adapter, info, timestamps) {
    this.device = device; this.adapter = adapter; this.info = info; this.limits = device.limits;
    this.features = [...device.features]; this.timestamps = timestamps;
    this.buffers = new Map();
  }
  get adapterName() { const i = this.info; return [i.vendor, i.architecture, i.device, i.description].filter(Boolean).join(' ') || 'unknown'; }

  /** Create (or replace) a named buffer. usage defaults to STORAGE|COPY_DST|COPY_SRC. */
  buffer(name, size, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, data = null) {
    size = Math.max(16, Math.ceil(size / 4) * 4);
    const old = this.buffers.get(name); if (old) old.destroy();
    const b = this.device.createBuffer({ label: name, size, usage, mappedAtCreation: !!data });
    if (data) {
      const src = data instanceof ArrayBuffer ? new Uint8Array(data) : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
      new Uint8Array(b.getMappedRange()).set(src.subarray(0, Math.min(src.byteLength, size)));
      b.unmap();
    }
    this.buffers.set(name, b);
    return b;
  }
  storage(name, size, data = null, extra = 0) { return this.buffer(name, size, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | extra, data); }
  uniform(name, size) { return this.buffer(name, size, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST); }
  readback(name, size) { return this.buffer(name, size, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST); }
  /** Total bytes of all live buffers created through this context. */
  memoryBytes() { let s = 0; for (const b of this.buffers.values()) s += b.size; return s; }
  memoryReport() { return [...this.buffers.entries()].map(([k, b]) => [k, b.size]).sort((a, b) => b[1] - a[1]); }

  write(buf, data, offset = 0) {
    const u8 = data instanceof ArrayBuffer ? new Uint8Array(data) : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    this.device.queue.writeBuffer(buf, offset, u8, 0, u8.byteLength & ~3);
  }

  /** Compile a compute pipeline. defines: {NAME: value} replaced as `const NAME ... = <v>;` overrides via prefix. */
  async kernel(name, code, { entryPoint = 'main', defines = null, label = name } = {}) {
    if (defines) code = Object.entries(defines).map(([k, v]) => `const ${k} = ${v};`).join('\n') + '\n' + code;
    const module = this.device.createShaderModule({ label, code });
    const info = await module.getCompilationInfo();
    const errs = info.messages.filter((m) => m.type === 'error');
    if (errs.length) {
      const lines = code.split('\n');
      throw new Error(`WGSL ${label}: ` + errs.map((m) => `${m.lineNum}:${m.linePos} ${m.message}\n    > ${lines[m.lineNum - 1]?.trim()}`).join('\n'));
    }
    const decls = parseBindings(code);
    const groups = [];
    for (const d of decls) (groups[d.group] ||= []).push(d);
    const layouts = [];
    for (let g = 0; g < groups.length; g++) {
      const entries = (groups[g] || []).map((d) => ({
        binding: d.binding, visibility: GPUShaderStage.COMPUTE,
        buffer: { type: d.space === 'uniform' ? 'uniform' : d.access === 'read_write' ? 'storage' : 'read-only-storage' },
      }));
      layouts.push(this.device.createBindGroupLayout({ label: `${label}/g${g}`, entries }));
    }
    const pipeline = await this.device.createComputePipelineAsync({
      label, layout: this.device.createPipelineLayout({ bindGroupLayouts: layouts }), compute: { module, entryPoint },
    });
    return { name: label, pipeline, layouts, decls, groups };
  }

  /** Bind group for kernel k, group g, buffers looked up by variable name in `reg` (Map or object). */
  bindGroup(k, reg, g = 0) {
    const get = (n) => (reg instanceof Map ? reg.get(n) : reg[n]) || this.buffers.get(n);
    const entries = (k.groups[g] || []).map((d) => {
      const b = get(d.name);
      if (!b) throw new Error(`kernel ${k.name}: no buffer for binding '${d.name}'`);
      return { binding: d.binding, resource: { buffer: b } };
    });
    return this.device.createBindGroup({ label: `${k.name}/bg${g}`, layout: k.layouts[g], entries });
  }

  /** Copy a buffer region to a fresh MAP_READ buffer and return it as an ArrayBuffer (slow path, tests). */
  async read(buf, size = buf.size, offset = 0) {
    const rb = this.device.createBuffer({ size: Math.ceil(size / 4) * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(buf, offset, rb, 0, Math.ceil(size / 4) * 4);
    this.device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const out = rb.getMappedRange().slice(0);
    rb.unmap(); rb.destroy();
    return out;
  }
}

/**
 * GPU timestamp profiler: one query pair per labelled compute pass.
 *   const prof = new Profiler(gpu, 64);
 *   pass = enc.beginComputePass(prof.pass('leaders'));  ...  prof.resolve(enc); await prof.collect() -> {label: ms}
 */
export class Profiler {
  constructor(gpu, maxPasses = 64) {
    this.gpu = gpu; this.enabled = gpu.timestamps; this.max = maxPasses; this.labels = []; this.pending = false;
    if (!this.enabled) return;
    const d = gpu.device;
    this.qs = d.createQuerySet({ type: 'timestamp', count: maxPasses * 2 });
    this.res = d.createBuffer({ size: maxPasses * 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    this.rb = d.createBuffer({ size: maxPasses * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  }
  reset() { this.labels = []; }
  /** Pass descriptor with timestampWrites (or a plain descriptor when disabled / full). */
  pass(label) {
    if (!this.enabled || this.pending || this.labels.length >= this.max) return { label };
    const k = this.labels.length; this.labels.push(label);
    return { label, timestampWrites: { querySet: this.qs, beginningOfPassWriteIndex: 2 * k, endOfPassWriteIndex: 2 * k + 1 } };
  }
  resolve(enc) {
    if (!this.enabled || this.pending || !this.labels.length) return false;
    const n = this.labels.length;
    enc.resolveQuerySet(this.qs, 0, 2 * n, this.res, 0);
    enc.copyBufferToBuffer(this.res, 0, this.rb, 0, n * 16);
    this.pending = true;
    return true;
  }
  /** Returns {label: totalMs} summed over passes with the same label, or null. */
  async collect() {
    if (!this.pending) return null;
    await this.rb.mapAsync(GPUMapMode.READ);
    const t = new BigInt64Array(this.rb.getMappedRange().slice(0, this.labels.length * 16));
    this.rb.unmap();
    const out = {};
    for (let k = 0; k < this.labels.length; k++) {
      const ms = Number(t[2 * k + 1] - t[2 * k]) / 1e6;
      if (ms >= 0 && ms < 1e4) out[this.labels[k]] = (out[this.labels[k]] || 0) + ms;
    }
    this.pending = false; this.labels = [];
    return out;
  }
}
