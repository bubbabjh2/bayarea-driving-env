// Tile binary format shared by the tiler (tools/tiles/build_tiles.mjs, Node)
// and the browser decoder (src/stream/worker.js). No three.js, no DOM.
//
// One file per tile: data/tiles/<level>/<ix>_<iy>.bin.gz (gzip; decompressed
// in the worker with DecompressionStream). Little-endian.
//
// header (32 B)
//   u32 magic 'BTIL'  u16 version  u8 level  u8 chunkCount
//   f32 cx  f32 cz            tile centre in world metres (x east, z south)
//   f32 step                  x/z quantisation step (m); local = int16 * step
//   f32 yStep                 y (elevation) step (m);   y = int16 * yStep
//   u32 ix  u32 iy            tile index in its level
// chunk (TRI / LINES)
//   u8 role  u8 prim  u8 flags  u8 pad  u32 nVerts  u32 nIdx
//   i16 pos[3*nVerts] (pad4)  u8 cls[nVerts] (pad4)
//   [flags&HAS_LANE] u16 lane[2*nVerts]   (across m*100, along m*128 mod 480 m; 480 = lcm of the look's 1.5/3/8/32 m periods)
//   [prim TRI] u16|u32 idx[nIdx] (u32 if flags&IDX32) (pad4)
//   LINES: consecutive vertex pairs, nIdx = 0
// chunk (WATER)
//   u8 role  u8 prim=2  u8 flags(FULL=1)  u8 pad  u32 w  u32 h
//   [!FULL] bitmask ceil(w*h/8) bytes, row 0 = north (min z), bit i = x%8 (pad4)
export const MAGIC = 0x4C495442;   // 'BTIL'
export const VERSION = 1;
export const HEADER_BYTES = 32;
export const PRIM = { TRI: 0, LINES: 1, WATER: 2 };
export const CHUNK_FLAG = { HAS_LANE: 1, IDX32: 2, FULL: 1 };
export const XZ_UNITS = 32768;     // step = tileSize / XZ_UNITS: tile spans +-16384, margin half a tile each side
export const Y_STEP = 1 / 32;
export const LANE_ALONG_SCALE = 128, LANE_ALONG_WRAP = 480, LANE_ACROSS_SCALE = 100;

// Material roles. userData.role on every tile mesh; userData.lookRole is the
// src/look role it maps to today (null = look has no handler yet).
export const ROLES = [
  'main-asphalt',      // 0 main-tier ribbons (motorway..secondary), lane-level width
  'local-asphalt',     // 1 local streets incl. parking lanes
  'junction',          // 2 local intersection polygons
  'curb',              // 3 raised curb strip (top + road-side face)
  'sidewalk',          // 4 sidewalk strip
  'road-strip',        // 5 mid-level simplified strips
  'road-lines',        // 6 coarse 1-px lines (main tier)
  'local-lines',       // 7 coarse 1-px lines (local tier)
  'coast',             // 8 coastline lines
  'water',             // 9 water mask plane
  'marking-white',     // 10 solid white edge lines
  'marking-yellow',    // 11 solid yellow centre lines
  'marking-dash',      // 12 dashed white lane dividers
  'marking-stopbar',   // 13 stop bars
  'marking-crosswalk', // 14 crosswalk stripes
  'stations',          // 15 gas station markers
  'signals',           // 16 signal / stop sign posts
  'verge',             // 17 planting strip between curb and sidewalk
  'driveway',          // 18 driveway aprons across the verge
];
export const ROLE = Object.fromEntries(ROLES.map((r, i) => [r, i]));
export const LOOK_ROLE = {
  'main-asphalt': 'road', 'road-strip': 'road',
  'local-asphalt': 'local-asphalt', 'junction': 'junction', 'curb': 'curb', 'sidewalk': 'sidewalk',
  'road-lines': 'roadLines', 'local-lines': 'local', 'coast': 'shoreline', 'water': 'water',
  'marking-white': 'marking-white', 'marking-yellow': 'marking-yellow', 'marking-dash': 'marking-dash',
  'marking-stopbar': 'marking-stopbar', 'marking-crosswalk': 'marking-crosswalk',
  'stations': 'stations', 'signals': null, 'verge': 'verge', 'driveway': 'driveway',
};

export const tileKey = (level, ix, iy) => `${level}/${ix}_${iy}`;
export const tileUrl = (level, ix, iy) => `${level}/${ix}_${iy}.bin.gz`;
