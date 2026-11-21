// Recording mock of WebGPU for running src/gpu/gpuSim.js in node (shared by the oracle scripts; same mock as
// gpu_mock.mjs): exact buffer contents, kernels with their final sources, bind groups, command stream.
//   import { gpu, commands, setRecording, kernels, bindGroups } from './mock_webgpu.mjs';
import { readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
let kernelDir = null;
export const setKernelDir = (d) => { kernelDir = d; };
globalThis.GPUBufferUsage = { MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16, VERTEX: 32, UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512 };
globalThis.GPUShaderStage = { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 };
globalThis.GPUMapMode = { READ: 1, WRITE: 2 };
const realFetch = globalThis.fetch;
globalThis.fetch = async (u) => {
  const s = String(u);
  if (s.startsWith('file:')) { const t = readFileSync(fileURLToPath(s), 'utf8'); return { ok: true, text: async () => t, json: async () => JSON.parse(t) }; }
  return realFetch(u);
};

const { parseBindings } = await import('../../../src/gpu/device.js');

let bufId = 0;
class MockBuffer {
  constructor({ label, size, usage }) { this.label = label || `buf${bufId}`; this.id = bufId++; this.size = size; this.usage = usage; this.data = new Uint8Array(size); this.written = false; this.destroyed = false; }
  destroy() { this.destroyed = true; }
  async mapAsync() {} getMappedRange() { return this.data.buffer; } unmap() {}
}
export const commands = []; let recording = false;
export const setRecording = (r) => { recording = r; };
const rec = (c) => { if (recording) commands.push(c); };
const queue = {
  writeBuffer(b, off, src, srcOff = 0, size) { const s = new Uint8Array(src.buffer || src, (src.byteOffset || 0) + srcOff, size ?? src.byteLength - srcOff); b.data.set(s, off); b.written = true; rec({ op: 'write', buf: b.label, off, size: s.length }); },
  submit(list) { for (const cmds of list) for (const c of cmds) { if (c.op === 'clear') { c.b.data.fill(0, c.off, c.off + c.size); } else if (c.op === 'copy') { c.dst.data.set(c.src.data.subarray(c.so, c.so + c.size), c.do); } rec(c.op === 'pass' ? c : { ...c, b: c.b?.label, src: c.src?.label, dst: c.dst?.label }); } rec({ op: 'submit' }); },
  async onSubmittedWorkDone() {},
};
const device = { queue,
  limits: {}, features: new Set(),
  createBuffer: (d) => new MockBuffer(d),
  createCommandEncoder() {
    const cmds = [];
    return {
      beginComputePass() { const p = { op: 'pass', d: [] }; let pipe = null; const bgs = [];
        cmds.push(p);
        return { setPipeline(k) { pipe = k; }, setBindGroup(g, bg) { bgs[g] = bg; }, dispatchWorkgroups(x) { p.d.push([pipe.label, bgs.map((b) => b.id), x]); }, end() {} }; },
      clearBuffer(b, off = 0, size) { cmds.push({ op: 'clear', b, off, size: size ?? b.size - off }); },
      copyBufferToBuffer(src, so, dst, d, size) { cmds.push({ op: 'copy', src, so, dst, do: d, size }); },
      resolveQuerySet() {}, finish() { return cmds; },
    };
  },
  createQuerySet() { return {}; },
};
export const kernels = []; export const bindGroups = [];
export const gpu = {
  device, adapter: {}, info: {}, limits: {}, features: [], timestamps: false, buffers: new Map(), adapterName: 'mock',
  buffer(name, size, usage = 128 | 8 | 4, data = null) {
    size = Math.max(16, Math.ceil(size / 4) * 4);
    const old = this.buffers.get(name); if (old) old.destroy();
    const b = new MockBuffer({ label: name, size, usage });
    if (data) { const src = data instanceof ArrayBuffer ? new Uint8Array(data) : new Uint8Array(data.buffer, data.byteOffset, data.byteLength); b.data.set(src.subarray(0, Math.min(src.byteLength, size))); b.written = true; }
    this.buffers.set(name, b); return b;
  },
  storage(name, size, data = null, extra = 0) { return this.buffer(name, size, 128 | 8 | 4 | extra, data); },
  uniform(name, size) { return this.buffer(name, size, 64 | 8); },
  readback(name, size) { return this.buffer(name, size, 1 | 8); },
  memoryBytes() { let s = 0; for (const b of this.buffers.values()) s += b.size; return s; },
  memoryReport() { return []; },
  write(buf, data, offset = 0) { const u8 = data instanceof ArrayBuffer ? new Uint8Array(data) : new Uint8Array(data.buffer, data.byteOffset, data.byteLength); queue.writeBuffer(buf, offset, u8, 0, u8.byteLength & ~3); },
  async kernel(name, code, { entryPoint = 'main', defines = null, label = name } = {}) {
    if (defines) code = Object.entries(defines).map(([k, v]) => `const ${k} = ${v};`).join('\n') + '\n' + code;
    const decls = parseBindings(code); const groups = [];
    for (const d of decls) (groups[d.group] ||= []).push(d);
    for (let g = 0; g < groups.length; g++) groups[g] ||= [];
    const file = label.replace(/[^\w.]/g, '_');
    if (kernelDir) writeFileSync(path.join(kernelDir, file + '.wgsl'), code);
    kernels.push({ label, entryPoint, file, decls });
    return { name: label, pipeline: { label }, layouts: groups.map((_, g) => g), decls, groups };
  },
  bindGroup(k, reg, g = 0) {
    const get = (n) => (reg instanceof Map ? reg.get(n) : reg[n]) || this.buffers.get(n);
    const entries = (k.groups[g] || []).map((d) => { const b = get(d.name); if (!b) throw new Error(`no buffer ${d.name}`); return { binding: d.binding, name: d.name, buf: b.label }; });
    const bg = { id: bindGroups.length, kernel: k.name, group: g, entries }; bindGroups.push(bg); return bg;
  },
  async read(buf, size = buf.size, offset = 0) { return buf.data.slice(offset, offset + size).buffer; },
};

