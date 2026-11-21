// Shared constants for the landscape render assets (tools/landscape/*, src/look/landscape.js reads landscape.json).
import path from 'node:path';
import { fileURLToPath } from 'node:url';
export const ROOT_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
export const OUT_DIR = path.join(ROOT_DIR, 'data/landscape');
// same square as the streamed road pyramid (data/tiles/manifest.json root)
export const ROOT = { x0: -53000, z0: -57900, size: 102400 };
export const PROJ = { lat0: 37.55, lon0: -122.15, mLon: 88157.883, mLat: 111195.0802 };
export const toXZ = (lat, lon) => [(lon - PROJ.lon0) * PROJ.mLon, -(lat - PROJ.lat0) * PROJ.mLat];
export const toLL = (x, z) => [PROJ.lat0 - z / PROJ.mLat, PROJ.lon0 + x / PROJ.mLon];
export const log = (...a) => console.log(`[${new Date().toISOString().slice(11, 19)}]`, ...a);
