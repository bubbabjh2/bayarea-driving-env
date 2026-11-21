// Minimal binary container shared by the JS exporters and the Rust loaders (rust/bayenv-sim/src/pack.rs).
// Layout: "BAYPACK1" | u32 LE header byte length | header JSON (utf8) | zero pad to 8 | sections (each 8-aligned).
// header = { meta: {...}, sections: [{ name, dtype, offset, length }] }  (offset in bytes from the start of the data area)
import fs from 'node:fs';

const DT = new Map([[Float64Array, 'f64'], [Float32Array, 'f32'], [Int32Array, 'i32'], [Uint32Array, 'u32'], [Int16Array, 'i16'],
  [Uint16Array, 'u16'], [Int8Array, 'i8'], [Uint8Array, 'u8'], [BigInt64Array, 'i64']]);

export class PackWriter {
  constructor() { this.sections = []; this.meta = {}; }
  add(name, arr) {
    if (!ArrayBuffer.isView(arr)) throw new Error(`section ${name}: not a typed array`);
    const dtype = DT.get(arr.constructor); if (!dtype) throw new Error(`section ${name}: unsupported ${arr.constructor.name}`);
    this.sections.push({ name, dtype, arr });
    return this;
  }
  write(path) {
    let off = 0; const secs = [];
    for (const s of this.sections) { secs.push({ name: s.name, dtype: s.dtype, offset: off, length: s.arr.length }); off += Math.ceil(s.arr.byteLength / 8) * 8; }
    const hdr = Buffer.from(JSON.stringify({ meta: this.meta, sections: secs }), 'utf8');
    const head = 8 + 4 + hdr.length, pad = (8 - (head % 8)) % 8;
    const fd = fs.openSync(path, 'w');
    const pre = Buffer.alloc(12); pre.write('BAYPACK1', 0, 'latin1'); pre.writeUInt32LE(hdr.length, 8);
    fs.writeSync(fd, pre); fs.writeSync(fd, hdr); fs.writeSync(fd, Buffer.alloc(pad));
    for (const s of this.sections) {
      const b = Buffer.from(s.arr.buffer, s.arr.byteOffset, s.arr.byteLength);
      fs.writeSync(fd, b); const p = (8 - (b.length % 8)) % 8; if (p) fs.writeSync(fd, Buffer.alloc(p));
    }
    fs.closeSync(fd);
  }
}

export function readPack(path) {
  const buf = fs.readFileSync(path);
  if (buf.toString('latin1', 0, 8) !== 'BAYPACK1') throw new Error(`${path}: not a BAYPACK1 file`);
  const hl = buf.readUInt32LE(8), hdr = JSON.parse(buf.toString('utf8', 12, 12 + hl));
  const data = 12 + hl + ((8 - ((12 + hl) % 8)) % 8);
  const T = { f64: Float64Array, f32: Float32Array, i32: Int32Array, u32: Uint32Array, i16: Int16Array, u16: Uint16Array, i8: Int8Array, u8: Uint8Array, i64: BigInt64Array };
  const out = {};
  for (const s of hdr.sections) { const C = T[s.dtype]; const ab = buf.buffer.slice(buf.byteOffset + data + s.offset, buf.byteOffset + data + s.offset + s.length * C.BYTES_PER_ELEMENT); out[s.name] = new C(ab); }
  return { meta: hdr.meta, s: out };
}
