// JS oracle for the Rust GPU host, no GPU needed: runs src/gpu/gpuSim.js against a recording mock of WebGPU
// in node. Captures the exact kernel sources (+ defines), the buffers each bind group uses, every buffer's
// contents after setup (world blob, params, scatter state, routes, actions) and the command stream of one
// runSteps(k) call.
//   node --max-old-space-size=14000 rust/tools/oracle/gpu_mock.mjs <dataDir> <outDir> [--n=10000] [--seed=11]
//        [--cap=131072] [--obs=v2] [--k=2] [--gas=0.3] [--controls=1]
// Out: gpu.meta.json (buffers, kernels, bind groups, commands, layout offsets, params) + gpu.bin (buffer bytes)
//      + kernels/<name>.wgsl (final source as compiled).
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const args = process.argv.slice(2);
const [dataDir, outDir] = args.filter((a) => !a.startsWith('--'));
const opt = (k, d) => { const a = args.find((s) => s.startsWith(`--${k}=`)); return a ? a.slice(k.length + 3) : d; };
mkdirSync(path.join(outDir, 'kernels'), { recursive: true });

globalThis.GPUBufferUsage = { MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16, VERTEX: 32, UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512 };
globalThis.GPUShaderStage = { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 };
globalThis.GPUMapMode = { READ: 1, WRITE: 2 };
const realFetch = globalThis.fetch;
globalThis.fetch = async (u) => {
  const s = String(u);
  if (s.startsWith('file:')) { const t = readFileSync(fileURLToPath(s), 'utf8'); return { ok: true, text: async () => t, json: async () => JSON.parse(t) }; }
  return realFetch(u);
};

const { GPUSim } = await import('../../../src/gpu/gpuSim.js');
const { parseBindings } = await import('../../../src/gpu/device.js');
const { loadWorld } = await import('../../../src/sim/data.js');

let bufId = 0;
class MockBuffer {
  constructor({ label, size, usage }) { this.label = label || `buf${bufId}`; this.id = bufId++; this.size = size; this.usage = usage; this.data = new Uint8Array(size); this.written = false; this.destroyed = false; }
  destroy() { this.destroyed = true; }
  async mapAsync() {} getMappedRange() { return this.data.buffer; } unmap() {}
}
const commands = []; let recording = false;
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
const kernels = []; const bindGroups = [];
const gpu = {
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
    writeFileSync(path.join(outDir, 'kernels', file + '.wgsl'), code);
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

const n = +opt('n', 10000), seed = +opt('seed', 11), cap = +opt('cap', 131072), obs = opt('obs', 'v2'), K = +opt('k', 2), gas = +opt('gas', 0.3);
const t0 = Date.now();
const world = await loadWorld({ base: dataDir.endsWith('/') ? dataDir : dataDir + '/' });
console.log(`world ${Date.now() - t0} ms`);
const sim = await GPUSim.create(world.graph, { gpu, capacity: Math.max(cap, n), obs, features: { accidents: true, fuel: true, collisions: true, laneChanges: true }, controls: opt('controls', '1') === '0' ? null : world.controls });
{ const A = new Float32Array(sim.cap * 3); for (let i = 0; i < sim.cap; i++) A[3 * i + 1] = gas; sim.setActions(A); }
const sc = sim.scatter(n, { seed });
console.log('scatter', JSON.stringify(sc));
// snapshot of every buffer after setup (before the first step)
const snapBufs = [...gpu.buffers.values()].filter((b) => !b.destroyed);
const parts = []; let off = 0; const bufs = [];
for (const b of snapBufs) {
  const nz = b.written || b.data.some((x) => x !== 0);
  bufs.push({ name: b.label, size: b.size, usage: b.usage, offset: nz ? off : -1 });
  if (nz) { parts.push(Buffer.from(b.data.buffer, 0, b.size)); off += b.size; }
}
writeFileSync(path.join(outDir, 'gpu.bin'), Buffer.concat(parts));
recording = true;
await sim.runSteps(K);
recording = false;
const meta = { n, seed, cap: sim.cap, obs, k: K, gas, scatter: sc, time: sim.time, step: sim.step, dt: sim.dt, layoutOff: sim.layoutOff, pairN: sim.pairN,
  world: { E: sim.world.E, N: sim.world.N, P: sim.world.P, K: sim.world.K }, jM: sim.jPack ? sim.jPack.M : 0, nCounts: sim.nCounts, T: sim.T, ioWords: sim.ioWords, IN: sim.IN,
  buffers: bufs, kernels, bindGroups, commands, slotTrip: Array.from(sim.slotTrip.subarray(0, Math.min(sim.cap, n + 16))) };
writeFileSync(path.join(outDir, 'gpu.meta.json'), JSON.stringify(meta));
console.log(`buffers ${bufs.length}, kernels ${kernels.length}, bindGroups ${bindGroups.length}, commands ${commands.length}, snapshot ${(off / 1e6).toFixed(1)} MB, ${Date.now() - t0} ms`);
