// Minimal GeoTIFF reader (render-asset build only): little/big-endian classic TIFF, tiled or stripped,
// float32/int16 single band, compression none / LZW (5) / deflate (8, 32946), predictor 1/2/3.
// Untrusted input: every offset/size is bounds-checked; nothing is executed.
import { readFileSync } from 'node:fs';
import zlib from 'node:zlib';

const TYPE_SIZE = { 1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 16: 8 };

export function readTiffHeader(buf) {
  const bo = buf.toString('latin1', 0, 2);
  if (bo !== 'II' && bo !== 'MM') throw new Error('not a TIFF');
  const le = bo === 'II';
  const u16 = (o) => { if (o + 2 > buf.length) throw new Error('oob'); return le ? buf.readUInt16LE(o) : buf.readUInt16BE(o); };
  const u32 = (o) => { if (o + 4 > buf.length) throw new Error('oob'); return le ? buf.readUInt32LE(o) : buf.readUInt32BE(o); };
  const f64 = (o) => { if (o + 8 > buf.length) throw new Error('oob'); return le ? buf.readDoubleLE(o) : buf.readDoubleBE(o); };
  if (u16(2) !== 42) throw new Error('BigTIFF / unknown TIFF version not supported');
  const ifd = u32(4);
  const n = u16(ifd);
  if (n > 500) throw new Error('bad IFD');
  const tags = {};
  for (let i = 0; i < n; i++) {
    const e = ifd + 2 + i * 12;
    const tag = u16(e), type = u16(e + 2), count = u32(e + 4);
    const sz = (TYPE_SIZE[type] || 1) * count;
    if (count > 1e7) throw new Error('bad count');
    const off = sz <= 4 ? e + 8 : u32(e + 8);
    if (off + sz > buf.length) throw new Error('tag oob');
    const vals = [];
    if (type === 2) { tags[tag] = buf.toString('latin1', off, off + count).replace(/\0+$/, ''); continue; }
    for (let k = 0; k < count; k++) {
      if (type === 3) vals.push(u16(off + k * 2));
      else if (type === 4) vals.push(u32(off + k * 4));
      else if (type === 12) vals.push(f64(off + k * 8));
      else if (type === 1 || type === 7) vals.push(buf[off + k]);
      else vals.push(null);
    }
    tags[tag] = vals;
  }
  const one = (t, d) => (tags[t] ? tags[t][0] : d);
  return {
    le, tags, width: one(256), height: one(257), bits: one(258, 16), compression: one(259, 1), predictor: one(317, 1),
    sampleFormat: one(339, 1), spp: one(277, 1), tileW: one(322), tileH: one(323), rowsPerStrip: one(278),
    offsets: tags[324] || tags[273], counts: tags[325] || tags[279],
    scale: tags[33550], tie: tags[33922], nodata: tags[42113] != null ? +tags[42113] : null,
  };
}

function lzwDecode(input, expected) {
  const out = new Uint8Array(expected);
  let op = 0, bitPos = 0, codeLen = 9;
  const dictPrefix = new Int32Array(4096), dictChar = new Uint8Array(4096), dictLen = new Int32Array(4096);
  for (let i = 0; i < 256; i++) { dictPrefix[i] = -1; dictChar[i] = i; dictLen[i] = 1; }
  let next = 258, prev = -1;
  const readCode = () => {
    const byte = bitPos >> 3; if (byte >= input.length) return 257;
    const b = (input[byte] << 16) | ((input[byte + 1] || 0) << 8) | (input[byte + 2] || 0);
    const c = (b >> (24 - (bitPos & 7) - codeLen)) & ((1 << codeLen) - 1);
    bitPos += codeLen; return c;
  };
  const emit = (code) => {
    const L = dictLen[code]; let p = op + L - 1; let c = code;
    if (op + L > expected) throw new Error('lzw overflow');
    while (c >= 0) { out[p--] = dictChar[c]; c = dictPrefix[c]; }
    op += L;
  };
  const first = (code) => { let c = code; while (dictPrefix[c] >= 0) c = dictPrefix[c]; return dictChar[c]; };
  for (;;) {
    const code = readCode();
    if (code === 257) break;
    if (code === 256) { next = 258; codeLen = 9; prev = -1; continue; }
    if (prev === -1) { emit(code); prev = code; continue; }
    if (code < next) {
      emit(code);
      if (next < 4096) { dictPrefix[next] = prev; dictChar[next] = first(code); dictLen[next] = dictLen[prev] + 1; next++; }
    } else if (code === next) {
      const fc = first(prev);
      if (next < 4096) { dictPrefix[next] = prev; dictChar[next] = fc; dictLen[next] = dictLen[prev] + 1; next++; }
      emit(code);
    } else throw new Error('lzw bad code');
    prev = code;
    if (next + 1 >= (1 << codeLen) && codeLen < 12) codeLen++;
  }
  return out;
}

/** Decode the whole single-band raster to Float32Array (row-major). */
export function readTiff(path) {
  const buf = readFileSync(path);
  const h = readTiffHeader(buf);
  if (h.spp !== 1) throw new Error('multi-band not supported');
  const bps = h.bits / 8;
  if (!((h.sampleFormat === 3 && bps === 4) || (h.sampleFormat === 2 && bps === 2) || (h.sampleFormat === 1 && bps === 2))) throw new Error(`format ${h.sampleFormat}/${h.bits} not supported`);
  if (h.width * h.height > 2e8) throw new Error('too big');
  const W = h.width, H = h.height;
  const out = new Float32Array(W * H);
  const tw = h.tileW || W, th = h.tileH || h.rowsPerStrip || H;
  const across = Math.ceil(W / tw), down = Math.ceil(H / th);
  if (h.offsets.length !== across * down) throw new Error('tile count mismatch');
  for (let ti = 0; ti < h.offsets.length; ti++) {
    const off = h.offsets[ti], cnt = h.counts[ti];
    if (off + cnt > buf.length) throw new Error('tile oob');
    const raw = buf.subarray(off, off + cnt);
    const expected = tw * th * bps;
    let data;
    if (h.compression === 1) data = new Uint8Array(raw);
    else if (h.compression === 5) data = lzwDecode(raw, expected);
    else if (h.compression === 8 || h.compression === 32946) data = new Uint8Array(zlib.inflateSync(raw, { maxOutputLength: expected * 2 }));
    else throw new Error('compression ' + h.compression + ' not supported');
    if (data.length < expected) { const d = new Uint8Array(expected); d.set(data); data = d; }
    const rowBytes = tw * bps;
    if (h.predictor === 2) {
      for (let r = 0; r < th; r++) {
        if (bps === 2) { const dv = new DataView(data.buffer, data.byteOffset); for (let c = 1; c < tw; c++) { const o = r * rowBytes + c * 2; dv.setUint16(o, (dv.getUint16(o, h.le) + dv.getUint16(o - 2, h.le)) & 0xffff, h.le); } }
        else for (let c = bps; c < rowBytes; c++) data[r * rowBytes + c] = (data[r * rowBytes + c] + data[r * rowBytes + c - bps]) & 255;
      }
    } else if (h.predictor === 3) {
      // floating-point predictor: byte-wise horizontal differencing, then bytes are stored plane-wise (MSB first)
      const tmp = new Uint8Array(rowBytes);
      for (let r = 0; r < th; r++) {
        const row = data.subarray(r * rowBytes, (r + 1) * rowBytes);
        for (let c = 1; c < rowBytes; c++) row[c] = (row[c] + row[c - 1]) & 255;
        for (let c = 0; c < tw; c++) for (let b = 0; b < bps; b++) tmp[c * bps + (h.le ? bps - 1 - b : b)] = row[b * tw + c];
        row.set(tmp);
      }
    }
    const dv = new DataView(data.buffer, data.byteOffset, data.byteLength);
    const tx = (ti % across) * tw, ty = Math.floor(ti / across) * th;
    for (let r = 0; r < th; r++) {
      const y = ty + r; if (y >= H) break;
      for (let c = 0; c < tw; c++) {
        const x = tx + c; if (x >= W) break;
        const o = (r * tw + c) * bps;
        out[y * W + x] = bps === 4 ? dv.getFloat32(o, h.le) : h.sampleFormat === 2 ? dv.getInt16(o, h.le) : dv.getUint16(o, h.le);
      }
    }
  }
  return { ...h, data: out };
}

if (process.argv[1] && process.argv[1].endsWith('tiff.mjs')) {
  const h = readTiffHeader(readFileSync(process.argv[2]));
  const { tags, offsets, counts, ...rest } = h;
  console.log(JSON.stringify({ ...rest, nTiles: offsets.length, tagIds: Object.keys(tags) }, null, 0));
}
