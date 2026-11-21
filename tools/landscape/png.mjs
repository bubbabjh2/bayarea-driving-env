// Minimal PNG writer (8-bit gray / RGB / RGBA, filter "up"/"sub" chosen per row), zlib only.
import zlib from 'node:zlib';
const CRC = new Int32Array(256).map((_, n) => { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; return c; });
function crc32(buf) { let c = -1; for (let i = 0; i < buf.length; i++) c = CRC[(c ^ buf[i]) & 255] ^ (c >>> 8); return (c ^ -1) >>> 0; }
function chunk(type, data) {
  const len = Buffer.alloc(4); len.writeUInt32BE(data.length);
  const td = Buffer.concat([Buffer.from(type, 'latin1'), data]);
  const crc = Buffer.alloc(4); crc.writeUInt32BE(crc32(td));
  return Buffer.concat([len, td, crc]);
}
/** pixels: Uint8Array (w*h*channels). channels 1, 3 or 4. */
export function encodePNG(pixels, w, h, channels = 4, level = 9) {
  const stride = w * channels;
  const raw = Buffer.alloc((stride + 1) * h);
  for (let y = 0; y < h; y++) {
    const o = y * (stride + 1), p = y * stride;
    // choose between Sub and Up by a cheap sum-of-abs heuristic
    let sSub = 0, sUp = 0;
    for (let x = 0; x < stride; x++) {
      const v = pixels[p + x];
      const sub = (v - (x >= channels ? pixels[p + x - channels] : 0)) & 255;
      const up = (v - (y > 0 ? pixels[p - stride + x] : 0)) & 255;
      sSub += sub < 128 ? sub : 256 - sub; sUp += up < 128 ? up : 256 - up;
    }
    const useUp = sUp < sSub;
    raw[o] = useUp ? 2 : 1;
    for (let x = 0; x < stride; x++) {
      const v = pixels[p + x];
      raw[o + 1 + x] = useUp ? (v - (y > 0 ? pixels[p - stride + x] : 0)) & 255 : (v - (x >= channels ? pixels[p + x - channels] : 0)) & 255;
    }
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0); ihdr.writeUInt32BE(h, 4); ihdr[8] = 8; ihdr[9] = { 1: 0, 3: 2, 4: 6 }[channels]; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
  return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', ihdr), chunk('IDAT', zlib.deflateSync(raw, { level })), chunk('IEND', Buffer.alloc(0))]);
}
