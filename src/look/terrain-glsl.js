// Shared terrain-height GLSL for the landscape layer (ground, buildings, trees, piers).
// Heights come from the shared road-conformed ground grid (data/dem/ground/ground10, via tools/landscape/build_terrain.mjs):
//   global 25 m field (R16F, uLsHeight) everywhere + 10 m detail tiles (R32F atlas, uHAtlas, page table uHPage) near the
//   camera where roads are. Both are sampled with Catmull-Rom (interpolating: passes exactly through the conformed
//   samples, so the ground meets the roads), from texelFetch (no filtering hardware precision issues).
// Detail tiles fade in through their page alpha; at a tile edge the detail weight blends toward the neighbour's state,
// so a resident tile next to a non-resident one has no step.
export const GLSL_TERRAIN = /* glsl */`
uniform sampler2D uLsHeight;
uniform vec4 uLsRoot;        // x0, z0, size, texels per side (global)
uniform float uLsExag;       // vertical scale of the real relief (1 = real; roads carry real heights)
uniform float uLsFlatY;      // ground level when flat
uniform sampler2D uHAtlas, uHPage;
uniform vec4 uHInfo;         // tiles per side, tile size (m), slot px, gutter px
uniform float uHSlots, uHOn;
vec4 ls_crw(float t) {
  float t2 = t * t, t3 = t2 * t;
  return vec4(-0.5 * t3 + t2 - 0.5 * t, 1.5 * t3 - 2.5 * t2 + 1.0, -1.5 * t3 + 2.0 * t2 + 0.5 * t, 0.5 * t3 - 0.5 * t2);
}
float ls_cr(sampler2D tex, vec2 p, ivec2 lo, ivec2 hi) {
  vec2 f = fract(p); ivec2 i0 = ivec2(floor(p)) - 1;
  vec4 wx = ls_crw(f.x), wy = ls_crw(f.y);
  float s = 0.0;
  for (int j = 0; j < 4; j++) {
    float row = 0.0;
    for (int i = 0; i < 4; i++) row += texelFetch(tex, clamp(i0 + ivec2(i, j), lo, hi), 0).r * wx[i];
    s += row * wy[j];
  }
  return s;
}
float ls_heightRaw(vec2 xz) {
  vec2 uv = (xz - uLsRoot.xy) / uLsRoot.z;
  if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) return 0.0;
  int N = int(uLsRoot.w);
  float h = ls_cr(uLsHeight, uv * uLsRoot.w - 0.5, ivec2(0), ivec2(N - 1));
  if (uHOn > 0.5) {
    vec2 tc = uv * uHInfo.x, tile = floor(tc), fr = tc - tile;
    vec4 pg = texelFetch(uHPage, ivec2(tile), 0);
    if (pg.a > 0.0) {
      int n = int(uHInfo.z);
      ivec2 so = ivec2(floor(pg.rg * 255.0 + 0.5)) * n;
      float d = ls_cr(uHAtlas, vec2(so) + fr * (uHInfo.z - 2.0 * uHInfo.w) + uHInfo.w - 0.5, so, so + ivec2(n - 1));
      float ex = min(fr.x, 1.0 - fr.x) * uHInfo.y, ez = min(fr.y, 1.0 - fr.y) * uHInfo.y;
      int g = int(uHInfo.x) - 1;
      float ax = texelFetch(uHPage, clamp(ivec2(tile) + ivec2(fr.x < 0.5 ? -1 : 1, 0), ivec2(0), ivec2(g)), 0).a;
      float az = texelFetch(uHPage, clamp(ivec2(tile) + ivec2(0, fr.y < 0.5 ? -1 : 1), ivec2(0), ivec2(g)), 0).a;
      float w = pg.a * mix(ax, 1.0, smoothstep(0.0, 60.0, ex)) * mix(az, 1.0, smoothstep(0.0, 60.0, ez));
      h = mix(h, d, w);
    }
  }
  // fade to sea level over the last 2 km of the data square (no cliff at the edge)
  vec2 e = min(uv, 1.0 - uv) * uLsRoot.z;
  return h * smoothstep(0.0, 2000.0, min(e.x, e.y));
}
`;
