// gzip -> ArrayBuffer with the browser's DecompressionStream. Pass-through when the
// server already decoded the body (Content-Encoding: gzip) and the magic is gone.
export async function gunzipMaybe(buf) {
  const u = new Uint8Array(buf, 0, 2);
  if (u[0] !== 0x1f || u[1] !== 0x8b) return buf;
  return new Response(new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
}
