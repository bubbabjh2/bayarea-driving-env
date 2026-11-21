#!/usr/bin/env node
// Quadtree tile pyramid for view-dependent streaming (rendering only).
//
//   node tools/tiles/build_tiles.mjs            # data/*.bin -> data/tiles/
//   node tools/tiles/build_tiles.mjs --out data/tiles --max-level 6
//
// Inputs (all read-only, produced by the world build; re-run after every rebuild):
//   data/roads.meta.json + roads.bin             main tier (motorway..secondary)
//   data/roads-local.meta.json + roads-local.bin local tier (tertiary..living_street)
//   data/roads-local.json  (optional; only parsed if it carries sidewalk / parking tags)
//   data/stations.meta.json + stations.bin       gas stations
//   data/water.json                              bay / ocean polygons
//   data/controls.meta.json + controls.bin (opt.) signals / stop / yield / ramp meters per approach
// Output: data/tiles/manifest.json + data/tiles/<L>/<ix>_<iy>.bin.gz (format: src/stream/format.js)
//
// Lane geometry: main-tier ribbons and lane markings come from the shared module src/geom/
// (same calls as src/roads.js: sampleS / bounds / rightEdge / offsetPoint, so tapers, anchored
// right edges and fillets match the sim and the full-load viewer). Local streets use geom's
// lane geometry too (3.7 m lanes, exactly where the sim drives), plus parking, curbs and sidewalks outside it.
// Junction cut-backs follow src/geom junction mouths (sStop / sExit) and boxes (J.hull) from
// data/junctions.bin (built with crosswalk legs, = what the sim loads).
// Paint (lane lines, stop bars, crosswalks) comes from the shared line set src/geom/lines.js
// (data/lines.bin, tools/lines/build_lines.mjs): the same lines the drivers perceive.
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { RoadSet, FLAG, CLASSES_ALL } from '../../src/roadpack.js';
import { ensureLaneGeometry, buildJunctions, crosswalkLegs, JunctionSet } from '../../src/geom/index.js';
import { buildLineSet, lineSetFromBuffers, emitPaint } from '../../src/geom/lines.js';
import { ROLES, ROLE, tileKey, tileUrl, XZ_UNITS } from '../../src/stream/format.js';
import { Level, encodeTile } from './lib/emit.mjs';
import { subdivide, simplify, frames, trim, hull, length } from './lib/geo.mjs';
import { WaterRaster } from './lib/water.mjs';

const ROOT_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const args = process.argv.slice(2);
const arg = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const DATA = path.resolve(ROOT_DIR, arg('--data', 'data'));
const OUT = path.resolve(ROOT_DIR, arg('--out', 'data/tiles'));
const MAX_LEVEL = +arg('--max-level', 6);
const T0 = Date.now();
const log = (...a) => console.log(`[${((Date.now() - T0) / 1000).toFixed(1)}s]`, ...a);

// ------------------------------------------------------------------ inputs
function loadPack(metaFile) {
  const meta = JSON.parse(fs.readFileSync(path.join(DATA, metaFile), 'utf8'));
  const bin = fs.readFileSync(path.join(DATA, meta.file));
  const buf = bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength);
  const T = { Float32: Float32Array, Uint32: Uint32Array, Uint16: Uint16Array, Uint8: Uint8Array };
  const arrays = {};
  for (const [name, s] of Object.entries(meta.sections)) arrays[name === 'nodeXYZ' ? 'xyz' : name] = new T[s.type](buf, s.byteOffset, s.length);
  return new RoadSet(meta, arrays);
}
const main = loadPack('roads.meta.json');
const local = fs.existsSync(path.join(DATA, 'roads-local.meta.json')) ? loadPack('roads-local.meta.json') : null;
const water = fs.existsSync(path.join(DATA, 'water.json')) ? JSON.parse(fs.readFileSync(path.join(DATA, 'water.json'), 'utf8')) : null;
let stations = null;
if (fs.existsSync(path.join(DATA, 'stations.meta.json'))) {
  const m = JSON.parse(fs.readFileSync(path.join(DATA, 'stations.meta.json'), 'utf8'));
  const b = fs.readFileSync(path.join(DATA, m.file));
  const buf = b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
  stations = { count: m.count, xyz: new Float32Array(buf, m.sections.stationXYZ.byteOffset, m.sections.stationXYZ.length),
    info: new Uint32Array(buf, m.sections.stationInfo.byteOffset, m.sections.stationInfo.length) };
}
log(`main ${main.edgeCount} edges, local ${local ? local.edgeCount : 0} edges, stations ${stations ? stations.count : 0}, water ${water ? water.polygons.length : 0} polygons`);

// Optional per-edge OSM tags (sidewalk / parking) from the readable local JSON.
const localTags = loadLocalTags();
function loadLocalTags() {
  const f = path.join(DATA, 'roads-local.json');
  if (!local || !fs.existsSync(f)) return null;
  const fd = fs.openSync(f, 'r'); const head = Buffer.alloc(4 << 20); fs.readSync(fd, head, 0, head.length, 0); fs.closeSync(fd);
  const s = head.toString('utf8');
  if (!/"sidewalk|"parking/.test(s)) { log('local JSON has no sidewalk/parking tags: using defaults'); return null; }
  log('parsing roads-local.json for sidewalk/parking tags…');
  const doc = JSON.parse(fs.readFileSync(f, 'utf8'));
  const sw = new Uint8Array(doc.edges.length).fill(255), pk = new Uint8Array(doc.edges.length).fill(255);
  const side = (v) => v == null ? 255 : /both|yes/.test(v) ? 3 : v === 'left' ? 1 : v === 'right' ? 2 : 0;
  doc.edges.forEach((e, i) => {
    sw[i] = side(e.sidewalk ?? e.tags?.sidewalk);
    const p = e.parking ?? e['parking:lane'] ?? e.tags?.['parking:lane:both'] ?? e.tags?.parking;
    pk[i] = p == null ? 255 : /no|none/.test(p) ? 0 : side(p === 'parallel' || p === 'diagonal' || p === 'perpendicular' ? 'both' : p);
  });
  return { sw, pk };
}

// Traffic controls (data/controls.meta.json + controls.bin, format bay-controls/1):
// per control [type, flags, tierId, nodeIndex, firstApproach, approachCount], approaches as
// merged-graph edge ids (main e, local mainEdgeCount + e) + stop-line position s on that edge.
const CONTROL_TYPES = ['signal', 'stop', 'give_way', 'ramp_meter', 'crossing_signal'];
const controls = loadControls();
function loadControls() {
  const f = path.join(DATA, 'controls.meta.json');
  if (!fs.existsSync(f)) { log('no data/controls.meta.json: no stop bars / crosswalks / signal posts'); return null; }
  const meta = JSON.parse(fs.readFileSync(f, 'utf8'));
  const b = fs.readFileSync(path.join(DATA, meta.file)), buf = b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
  const v = (n, T) => new T(buf, meta.sections[n].byteOffset, meta.sections[n].length);
  const c = { meta, count: meta.count, info: v('controlInfo', Uint32Array), xyz: v('controlXYZ', Float32Array),
    approachEdge: v('approachEdge', Uint32Array), approachStopAt: v('approachStopAt', Float32Array) };
  if (meta.mainEdgeCount !== main.edgeCount) log(`WARNING controls.meta mainEdgeCount ${meta.mainEdgeCount} != roads ${main.edgeCount} (stale controls?)`);
  log(`controls: ${c.count} (${c.approachEdge.length} approaches)`);
  return c;
}

// ------------------------------------------------------------------ pyramid layout
const ROOT = (() => {
  const b = main.meta.bbox, wb = water?.meta?.bbox || b;
  const minX = Math.min(b.minX, wb.minX), maxX = Math.max(b.maxX, wb.maxX), minZ = Math.min(b.minZ, wb.minZ), maxZ = Math.max(b.maxZ, wb.maxZ);
  const leaf = 1600, size = leaf * (1 << MAX_LEVEL);
  if (Math.max(maxX - minX, maxZ - minZ) > size) throw new Error('world larger than root tile; raise --max-level');
  const cx = Math.round((minX + maxX) / 2 / 100) * 100, cz = Math.round((minZ + maxZ) / 2 / 100) * 100;
  return { x0: cx - size / 2, z0: cz - size / 2, size };
})();

// Class ids: 0 motorway 1 motorway_link 2 trunk 3 trunk_link 4 primary 5 primary_link
// 6 secondary 7 secondary_link 8 tertiary 9 tertiary_link 10 unclassified 11 residential 12 living_street
// draw-order epsilons = CLASS_STYLE[c].y in src/roads.js (the real height is the node elevation)
const YB = [0.030, 0.026, 0.028, 0.024, 0.020, 0.018, 0.016, 0.014, 0.010, 0.010, 0.006, 0.006, 0.004];
const MARK_LIFT = 0.02;   // = src/roads.js MARK_LIFT
const MOTOR = [0, 2], MAIN_NOLINK = [0, 2, 4, 6], MAIN_ALL = [0, 1, 2, 3, 4, 5, 6, 7];
const LOCAL_ALL = [8, 9, 10, 11, 12];

// What each level contains. err = geometric error (m) used by the loader's screen-space error.
const LEVELS = [
  { err: 1600, tol: 300, lines: MOTOR, coast: 300, waterRes: 2048 },
  { err: 800, tol: 120, lines: MOTOR, coast: 120, waterRes: 1024 },
  { err: 400, tol: 50, lines: MAIN_NOLINK, strips: MOTOR, coast: 50, waterRes: 512 },
  { err: 200, tol: 20, lines: MAIN_ALL, strips: MAIN_ALL, localLines: [8, 9], coast: 20, waterRes: 512 },
  { err: 100, tol: 6, lines: MAIN_ALL, ribbons: true, localLines: LOCAL_ALL, waterRes: 512 },
  { err: 40, tol: 2, lines: MAIN_ALL, ribbons: true, lane: true, farPaint: true, localLines: LOCAL_ALL, localStrips: true, stations: true, waterRes: 512 },
  { err: 0, tol: 0, ribbons: true, lane: true, markings: true, localFull: true, stations: true, controls: true, waterRes: 512 },
].slice(0, MAX_LEVEL + 1);
if (MAX_LEVEL < 6) LEVELS[LEVELS.length - 1] = { ...LEVELS[6 - (6 - MAX_LEVEL)], err: 0 };

// ------------------------------------------------------------------ helpers
function edgePoly(R, e) {
  const nodes = R.edgeNodes(e), P = { x: [], y: [], z: [] };
  for (const n of nodes) { P.x.push(R.x(n)); P.y.push(R.y(n)); P.z.push(R.z(n)); }
  return P;
}
const clsOf = (R, e) => R.classId(e) + (R.has(e, FLAG.TUNNEL) ? 16 : 0);
// Pavement surface (data/surfaces.bin, docs/SURFACES.md) in bits 5-7 of the class byte of the asphalt roles
// (main-asphalt, local-asphalt, junction): the look draws concrete, pavers, gravel, dirt, steel grate, wood or grass
// from it. 0 = asphalt (also when surfaces.bin is absent).
// surface code -> visual group: 0 asphalt, 1 concrete, 2 pavers (paving_stones, sett), 3 gravel / compacted,
// 4 dirt, 5 metal grate, 6 wood, 7 grass / sand
const SURF_VIS = [0, 1, 2, 2, 3, 3, 4, 7, 7, 5, 6, 0, 0, 0, 0, 0];
const SURF = (() => {
  const f = path.join(DATA, 'surfaces.meta.json');
  if (!fs.existsSync(f)) { console.log('no data/surfaces.bin: every road drawn as asphalt'); return null; }
  const m = JSON.parse(fs.readFileSync(f, 'utf8')), b = fs.readFileSync(path.join(DATA, m.file));
  if (m.edgeCount.main !== main.edgeCount || (local && m.edgeCount.local !== local.edgeCount)) { console.log('data/surfaces.bin does not match the road packs: ignored'); return null; }
  return b;
})();
const surfVis = (tier, e) => (SURF ? SURF_VIS[SURF[2 * (tier === local ? main.edgeCount + e : e)] & 15] : 0);
const clsSurf = (R, e) => clsOf(R, e) + 32 * surfVis(R, e);

/** Multi-column strip along P. cols = [[offset, dy, across], ...] ordered right -> left. */
function strip(level, role, P, F, cols, yBase, cls, laneAlong) {
  const n = P.x.length, k = typeof cols === 'function' ? cols(0).length : cols.length;
  const V = { x: new Array(n * k), y: new Array(n * k), z: new Array(n * k) };
  const lane = laneAlong ? new Array(n * k * 2) : null;
  for (let i = 0; i < n; i++) for (let j = 0; j < k; j++) {
    const [o, dy, ac] = typeof cols === 'function' ? cols(i)[j] : cols[j], v = i * k + j;
    V.x[v] = P.x[i] + F.rx[i] * o; V.z[v] = P.z[i] + F.rz[i] * o; V.y[v] = P.y[i] + yBase + dy;
    if (lane) { lane[v * 2] = ac ?? 0; lane[v * 2 + 1] = F.s[i]; }
  }
  const T = [];
  for (let i = 0; i < n - 1; i++) for (let j = 0; j < k - 1; j++) {
    const a = i * k + j, b = (i + 1) * k + j, c = i * k + j + 1, d = (i + 1) * k + j + 1;
    T.push(a, b, c, c, b, d);
  }
  level.tris(role, V, cls, lane, T);
}

/** Quad from a -> b of the given width (flat, y from endpoints). */
function quad(level, role, ax, ay, az, bx, by, bz, w, cls) {
  const dx = bx - ax, dz = bz - az, L = Math.hypot(dx, dz) || 1, rx = -dz / L * w / 2, rz = dx / L * w / 2;
  level.tris(role, { x: [ax + rx, bx + rx, ax - rx, bx - rx], y: [ay, by, ay, by], z: [az + rz, bz + rz, az - rz, bz - rz] }, cls, null, [0, 1, 2, 2, 1, 3]);
}
function box(level, role, cx, cy, cz, sx, sy, sz, cls) {
  const X = [cx - sx / 2, cx + sx / 2], Y = [cy - sy / 2, cy + sy / 2], Z = [cz - sz / 2, cz + sz / 2];
  const V = { x: [], y: [], z: [] };
  for (let i = 0; i < 8; i++) { V.x.push(X[i & 1]); V.y.push(Y[(i >> 1) & 1]); V.z.push(Z[(i >> 2) & 1]); }
  const T = [0, 2, 1, 1, 2, 3, 4, 5, 6, 5, 7, 6, 0, 1, 4, 1, 5, 4, 2, 6, 3, 3, 6, 7, 0, 4, 2, 2, 4, 6, 1, 3, 5, 3, 7, 5];
  level.tris(role, V, cls, null, T);
}
/** Offset polyline (each vertex moved along its mitred right vector). */
function offsetPoly(P, F, o, dy) {
  const Q = { x: [], y: [], z: [] };
  for (let i = 0; i < P.x.length; i++) { Q.x.push(P.x[i] + F.rx[i] * o); Q.y.push(P.y[i] + dy); Q.z.push(P.z[i] + F.rz[i] * o); }
  return Q;
}
/** Dashed quads along polyline offset o (dash pattern continuous across vertices). */
function dashes(level, role, P, F, o, dy, w, on, off, cls) {
  const Q = offsetPoly(P, F, o, dy), period = on + off;
  let phase = 0;
  for (let i = 0; i < Q.x.length - 1; i++) {
    const ax = Q.x[i], ay = Q.y[i], az = Q.z[i], bx = Q.x[i + 1], by = Q.y[i + 1], bz = Q.z[i + 1];
    const len = Math.hypot(bx - ax, bz - az);
    let d = 0;
    while (d < len) {
      const inP = phase % period;
      if (inP < on) {
        const d2 = Math.min(len, d + (on - inP)), t0 = d / len, t1 = d2 / len;
        if (d2 - d > 0.05) quad(level, role, ax + (bx - ax) * t0, ay + (by - ay) * t0, az + (bz - az) * t0, ax + (bx - ax) * t1, ay + (by - ay) * t1, az + (bz - az) * t1, w, cls);
        phase += d2 - d; d = d2;
      } else { const s = Math.min(len - d, period - inP); phase += s; d += s; }
    }
  }
}
function solidLine(level, role, P, F, o, dy, w, cls) {
  strip(level, role, P, F, [[o + w / 2, dy], [o - w / 2, dy]], 0, cls, false);
}

// ------------------------------------------------------------------ main tier
function mainLines(level, spec) {
  const want = new Set(spec.lines);
  for (let e = 0; e < main.edgeCount; e++) {
    if (!main.isPrimaryDir(e) || !want.has(main.classId(e))) continue;
    let P = simplify(edgePoly(main, e), spec.tol);
    P = subdivide(P, level.size / 4);
    const c = main.classId(e);
    level.lines('road-lines', { x: P.x, y: P.y.map((y) => y + YB[c]), z: P.z }, clsOf(main, e));
  }
}
function mainStrips(level, spec) {
  const want = new Set(spec.strips);
  for (let e = 0; e < main.edgeCount; e++) {
    if (!main.isPrimaryDir(e) || !want.has(main.classId(e))) continue;
    let P = subdivide(simplify(edgePoly(main, e), spec.tol), level.size / 4);
    const F = frames(P), h = main.width(e) / 2;
    strip(level, 'road-strip', P, F, [[h, 0], [-h, 0]], YB[main.classId(e)], clsOf(main, e), false);
  }
}
// ---- shared lane geometry (src/geom): merged graph, local edge e is LBASE + e
log('building lane geometry (src/geom)…');
const LG = ensureLaneGeometry(main, local);
const LBASE = main.edgeCount;
log(`lane geometry: ${LG.stats.vertices} vertices, ${LG.stats.strictHops} strict hops`);
function readSections(metaFile) {
  const meta = JSON.parse(fs.readFileSync(path.join(DATA, metaFile), 'utf8')), bufs = {};
  for (const f of new Set(Object.values(meta.sections).map((x) => x.file || meta.file))) { const b = fs.readFileSync(path.join(DATA, f)); bufs[f] = b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); }
  const T = { Float32: Float32Array, Uint32: Uint32Array, Uint8: Uint8Array }, a = {};
  for (const [k, x] of Object.entries(meta.sections)) a[k] = new T[x.type](bufs[x.file || meta.file], x.byteOffset, x.length);
  return { meta, a, bufs };
}
// junctions as the sim loads them (data/junctions.bin, crosswalk legs); rebuilt with the same legs if stale
const CTL_DOC = fs.existsSync(path.join(DATA, 'controls.json')) ? JSON.parse(fs.readFileSync(path.join(DATA, 'controls.json'), 'utf8')) : null;
const CW_DOC = fs.existsSync(path.join(DATA, 'crosswalks.json')) ? JSON.parse(fs.readFileSync(path.join(DATA, 'crosswalks.json'), 'utf8')) : null;
const LEGS = crosswalkLegs(LG, CTL_DOC ? CTL_DOC.controls : null, CW_DOC ? CW_DOC.points : null);
const JUNCTIONS = (() => {
  if (fs.existsSync(path.join(DATA, 'junctions.meta.json'))) {
    const { meta, a } = readSections('junctions.meta.json');
    if (meta.mainEdgeCount === main.edgeCount && a.mouthS.length === 2 * LG.G.edgeCount) { const J = new JunctionSet(LG.G, LG, a); J.source = 'data/junctions.bin'; return J; }
  }
  log('WARNING data/junctions.bin missing / stale: building junctions in memory with crosswalk legs');
  const J = buildJunctions(LG, LEGS); J.source = 'in memory'; return J;
})();
// lane lines (src/geom/lines.js) from data/lines.bin; built in memory if missing / stale
let LINES_SOURCE = 'data/lines.bin';
const LINES = (() => {
  if (fs.existsSync(path.join(DATA, 'lines.meta.json'))) {
    const { meta, bufs } = readSections('lines.meta.json');
    if (meta.edgeCount === LG.G.edgeCount && meta.moveCount === JUNCTIONS.moveCount) return lineSetFromBuffers(meta, bufs, LG, JUNCTIONS);
  }
  LINES_SOURCE = 'in memory';
  log('WARNING data/lines.bin missing / stale: building the line set in memory (run node tools/lines/build_lines.mjs)');
  return buildLineSet(LG, JUNCTIONS, { controls: CTL_DOC ? CTL_DOC.controls : [], legs: LEGS, points: CW_DOC ? CW_DOC.points : [] });
})();
log(`line set: ${LINES.runCount} runs, ${LINES.markCount} marks (${LINES_SOURCE})`);
const _A = {}, _B = {};
/** Pavement offsets [right, left] of the polyline at s, both directions of a two-way road (= src/roads.js pavement()). */
function pavement(g, gt, s, L) {
  const b = LG.bounds(g, s);
  if (gt < 0) return [b.right, b.left];
  return [b.right, -LG.bounds(gt, L - s).right];
}
/** geom sample positions, densified so no step exceeds maxStep metres. */
function samples(g, maxStep) {
  const ss = LG.sampleS(g), out = [ss[0]];
  for (let k = 1; k < ss.length; k++) { const d = ss[k] - ss[k - 1], n = Math.ceil(d / maxStep); for (let j = 1; j < n; j++) out.push(ss[k - 1] + (d * j) / n); out.push(ss[k]); }
  return out;
}
/** Reference-polyline points + offset vectors (the mitred normal geom offsets along) at the samples. */
function geomFrames(g, ss) {
  const P = { x: [], y: [], z: [] }, F = { rx: new Float64Array(ss.length), rz: new Float64Array(ss.length), s: Float64Array.from(ss) };
  ss.forEach((s, i) => {
    LG.offsetPoint(g, s, 0, 0, _A); LG.offsetPoint(g, s, 1, 0, _B);
    P.x.push(_A.x); P.y.push(_A.y); P.z.push(_A.z); F.rx[i] = _B.x - _A.x; F.rz[i] = _B.z - _A.z;
  });
  return { P, F };
}
/** Marking quad between s0 and s1 at offset off (half width hw), like src/roads.js buildLaneMarkings. */
function markQuad(level, role, g, s0, s1, off, hw, y, cls) {
  const V = { x: [], y: [], z: [] };
  for (const s of [s0, s1]) {
    LG.offsetPoint(g, s, off - hw, 0, _A); LG.offsetPoint(g, s, off + hw, 0, _B);
    V.x.push(_A.x, _B.x); V.y.push(_A.y + y, _B.y + y); V.z.push(_A.z, _B.z);
  }
  level.tris(role, V, cls, null, quadUp(V));
}
/** Triangles for a 4-vertex patch laid out [s0 o0, s0 o1, s1 o0, s1 o1], facing up (the look's paint is front-face only). */
function quadUp(V) {
  const o = [0, 1, 3, 2];
  let ar = 0; for (let q = 0; q < 4; q++) { const a = o[q], b = o[(q + 1) % 4]; ar += V.x[a] * V.z[b] - V.x[b] * V.z[a]; }
  return ar > 0 ? [0, 3, 1, 0, 2, 3] : [0, 1, 3, 0, 3, 2];
}
function mainRibbons(level, spec) {
  for (let e = 0; e < main.edgeCount; e++) {
    if (!main.isPrimaryDir(e)) continue;
    const c = main.classId(e), cls = clsOf(main, e), tw = main.twin(e), L = main.edgeLength[e];
    const g = e, gt = tw;
    const ss = samples(g, Math.min(400, level.size / 4));
    const { P, F } = geomFrames(g, ss);
    const pav = ss.map((s) => pavement(g, gt, s, L));
    strip(level, 'main-asphalt', P, F, (i) => [[pav[i][0], 0, pav[i][0] - pav[i][1]], [pav[i][1], 0, 0]], YB[c], clsSurf(main, e), !!spec.lane);
  }
  if (spec.farPaint) for (let e = 0; e < main.edgeCount; e++) paintEdge(level, e, { far: true });
}

// ---- paint: the shared line set (src/geom/lines.js), one call per directed edge (main e, local LBASE + e)
const ROLE_OF = { white: 'marking-white', dash: 'marking-dash', yellow: 'marking-yellow', stopbar: 'marking-stopbar', crosswalk: 'marking-crosswalk' };
const CLS_OF = { white: 0, yellow: 1, dash: 2, stopbar: 0, crosswalk: 0 };
const PAINT_STATS = {};
function paintEdge(level, g, opts = {}) {
  const c = LG.G.cls[g], key = (opts.far ? 'far ' : '') ;
  emitPaint(LINES, g, (role, V, T) => {
    const X = [], Y = [], Zs = [];
    for (let i = 0; i < V.length; i += 3) { X.push(V[i]); Y.push(V[i + 1]); Zs.push(V[i + 2]); }
    level.tris(ROLE_OF[role], { x: X, y: Y, z: Zs }, CLS_OF[role], null, T);
    PAINT_STATS[key + role] = (PAINT_STATS[key + role] || 0) + T.length / 3;
  }, { lift: YB[c] + MARK_LIFT, ...opts });
}

// ------------------------------------------------------------------ local tier
// Real-width surfaces: lanes x 3.7 m (= src/geom LANE_W = the sim's LANE_WIDTH), parking lanes, curbs, sidewalks,
// trimmed at junctions and filled with junction polygons.
const LOCAL = local ? (() => {
  const E = local.edgeCount, N = local.nodeCount;
  // urban density: local road metres per 500 m cell
  const cell = 500, dens = new Map();
  const ck = (x, z) => Math.floor(x / cell) * 100000 + Math.floor(z / cell);
  for (let e = 0; e < E; e++) {
    if (!local.isPrimaryDir(e)) continue;
    const n = local.edgeNodes(e), a = n[0], b = n[n.length - 1];
    const k = ck((local.x(a) + local.x(b)) / 2, (local.z(a) + local.z(b)) / 2);
    dens.set(k, (dens.get(k) || 0) + local.edgeLength[e]);
  }
  const densAt = (x, z) => dens.get(ck(x, z)) || 0;
  const urbanAt = (x, z) => densAt(x, z) > 2500;
  const half = new Float32Array(E * 2);    // [left, right] asphalt half widths incl. parking
  const park = new Uint8Array(E);          // bit0 parking on the left, bit1 on the right
  const side = new Uint8Array(E);          // bit0 sidewalk L, bit1 sidewalk R, bit2 curb L, bit3 curb R
  const deg = new Uint16Array(N);
  for (let e = 0; e < E; e++) {
    if (!local.isPrimaryDir(e)) continue;
    const c = local.classId(e), lanes = local.lanes(e) + local.oppLanes(e);
    const laneW = main.laneWidth;   // one lane width everywhere: src/geom and the sim use 3.7 m
    const nodes = local.edgeNodes(e), mid = nodes[nodes.length >> 1];
    const urban = urbanAt(local.x(mid), local.z(mid));
    const twoWay = local.oppLanes(e) > 0;
    let pk = 0;   // bit0 left, bit1 right
    const tagP = localTags ? localTags.pk[e] : 255;
    if (tagP !== 255) pk = tagP;
    else if (urban && twoWay && (c === 11 || c === 12)) pk = 3;
    park[e] = pk;
    let sw = 0;
    const tagS = localTags ? localTags.sw[e] : 255;
    if (tagS !== 255) sw = tagS;
    else if (urban && (c === 11 || c === 12 || c === 8)) sw = 3;
    const base = lanes * laneW / 2;
    half[e * 2] = base + (pk & 1 ? 2.0 : 0); half[e * 2 + 1] = base + (pk & 2 ? 2.0 : 0);
    let curb = sw; if (urban) curb = 3;
    side[e] = sw | (curb << 2);
    deg[nodes[0]]++; deg[nodes[nodes.length - 1]]++;
  }
  // main-tier road half width at shared nodes (local roads stop at the main road's edge)
  const mainHalf = new Float32Array(N);
  if (local.linkNodes) {
    const mh = new Float32Array(main.nodeCount);
    for (let e = 0; e < main.edgeCount; e++) {
      if (!main.isPrimaryDir(e)) continue;
      const n = main.edgeNodes(e), h = main.width(e) / 2;
      for (const v of [n[0], n[n.length - 1]]) mh[v] = Math.max(mh[v], h);
    }
    for (let i = 0; i < local.linkNodes.length; i += 2) mainHalf[local.linkNodes[i]] = mh[local.linkNodes[i + 1]];
  }
  // widest local half width at each node
  const nodeHalf = new Float32Array(N);
  for (let e = 0; e < E; e++) {
    if (!local.isPrimaryDir(e)) continue;
    const n = local.edgeNodes(e), h = Math.max(half[e * 2], half[e * 2 + 1]);
    for (const v of [n[0], n[n.length - 1]]) nodeHalf[v] = Math.max(nodeHalf[v], h);
  }
  return { half, park, side, deg, mainHalf, nodeHalf, urbanAt, densAt };
})() : null;

function localLines(level, spec) {
  const want = new Set(spec.localLines);
  for (let e = 0; e < local.edgeCount; e++) {
    if (!local.isPrimaryDir(e) || !want.has(local.classId(e))) continue;
    const P = subdivide(simplify(edgePoly(local, e), spec.tol), level.size / 4);
    const c = local.classId(e);
    level.lines('local-lines', { x: P.x, y: P.y.map((y) => y + YB[c]), z: P.z }, clsOf(local, e));
  }
}
function localStrips(level, spec) {
  for (let e = 0; e < local.edgeCount; e++) {
    if (!local.isPrimaryDir(e)) continue;
    const P = subdivide(simplify(edgePoly(local, e), spec.tol), level.size / 4);
    const F = frames(P), hl = LOCAL.half[e * 2], hr = LOCAL.half[e * 2 + 1];
    strip(level, 'local-asphalt', P, F, [[hr, 0, hl + hr], [-hl, 0, 0]], YB[local.classId(e)], clsSurf(local, e), !!spec.lane);
  }
}

function segX(a, b, c, d) {
  const o = (p, q, r) => Math.sign((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]));
  return o(a, b, c) * o(a, b, d) < 0 && o(c, d, a) * o(c, d, b) < 0;
}
/** false for bow-ties (opposite sides crossing). */
const simpleQuad = (Q) => !segX(Q[0], Q[1], Q[2], Q[3]) && !segX(Q[1], Q[2], Q[3], Q[0]);
function upQuad(Q) {
  let area = 0; for (let q = 0; q < 4; q++) { const p0 = Q[q], p1 = Q[(q + 1) % 4]; area += p0[0] * p1[1] - p1[0] * p0[1]; }
  return area > 0 ? [0, 2, 1, 0, 3, 2] : [0, 1, 2, 0, 2, 3];
}
// ------------------------------------------------------------------ L6 streets (main + local, one model)
// Every physical road (main from src/geom pavement(), local at real street widths) is cut back at
// at-grade junctions so nothing overlaps; the junction is one polygon, and each pair of angularly
// adjacent arms gets a rounded corner (curb top + face, concrete corner) between their sidewalks.
// Trims come from where the facing pavement / sidewalk edges of neighbouring arms intersect:
//   d = (h_b + h_a cos t) / sin t     (t = angle between the arms, h = facing half width)
const CURB_H = 0.15, CURB_W = 0.3, SIDEWALK_W = 1.8, VERGE_W = 1.5;
const CORNER_STEPS = 8;
let ROADS = null, ROAD_OF = null, MARK_AUDIT = null, JSET = null;
const STOP_CHECK = { n: 0, off: 0, max: 0, dataOff: 0, ex: [] };   // physical roads (built once), merged edge id -> road index

function hash01(a, b) { let h = Math.imul(a ^ 0x9e3779b9, 0x85ebca6b) ^ Math.imul(b + 0x632be5ab, 0xc2b2ae35); h ^= h >>> 15; h = Math.imul(h, 0x2c1b3c6d); h ^= h >>> 12; return (h >>> 0) / 4294967296; }

function buildStreetModel() {
  if (ROADS) return;
  const t0 = Date.now();
  const linked = new Int32Array(local ? local.nodeCount : 0).fill(-1);
  if (local && local.linkNodes) for (let i = 0; i < local.linkNodes.length; i += 2) linked[local.linkNodes[i]] = local.linkNodes[i + 1];
  const keyL = (m) => (linked[m] >= 0 ? linked[m] : main.nodeCount + m);
  ROADS = []; ROAD_OF = new Int32Array(LG.G.edgeCount).fill(-1);
  const push = (r) => { ROAD_OF[r.g] = ROADS.length; if (r.gt >= 0) ROAD_OF[r.gt] = ROADS.length; ROADS.push(r); };
  for (let e = 0; e < main.edgeCount; e++) {
    if (!main.isPrimaryDir(e)) continue;
    const c = main.classId(e), tw = main.twin(e), L = main.edgeLength[e];
    const ss = samples(e, 200), { P, F } = geomFrames(e, ss);
    const offR = [], offL = [];
    for (const s of ss) { const [r, l] = pavement(e, tw, s, L); offR.push(r); offL.push(l); }
    const nodes = main.edgeNodes(e), mid = nodes[nodes.length >> 1];
    const dens = LOCAL ? LOCAL.densAt(main.x(mid), main.z(mid)) : 0;
    const urban = dens > 2500;
    const sw = urban && (c === 4 || c === 6) ? 3 : 0, cb = urban && c >= 4 ? 3 : 0;
    push({ kind: 'main', e, g: e, gt: tw, c, cls: clsSurf(main, e), yb: YB[c], x: P.x, y: P.y, z: P.z, rx: F.rx, rz: F.rz, s: ss, cl: ss.slice(),
      offR, offL, sw, cb, verge: 0, ends: [nodes[0], nodes[nodes.length - 1]], t: [0, 0], ts: [[0, 0], [0, 0]], len: L });
  }
  if (local) for (let e = 0; e < local.edgeCount; e++) {
    if (!local.isPrimaryDir(e)) continue;
    // drivable lanes exactly as src/geom / the sim (pavement(): 3.7 m lanes, tapers, anchored edges),
    // parking lanes outside them
    const c = local.classId(e), g = LBASE + e, tw = local.twin(e), gt = tw >= 0 ? LBASE + tw : -1, Ld = local.edgeLength[e];
    const ss = samples(g, 200), { P, F } = geomFrames(g, ss), n = ss.length;
    const pR = LOCAL.park[e] & 2 ? 2.0 : 0, pL = LOCAL.park[e] & 1 ? 2.0 : 0;
    const offR = [], offL = [];
    for (const sv of ss) { const [r0, l0] = pavement(g, gt, sv, Ld); offR.push(r0 + pR); offL.push(l0 - pL); }
    const nodes = local.edgeNodes(e), mid = nodes[nodes.length >> 1];
    const dens = LOCAL.densAt(local.x(mid), local.z(mid));
    const sd = LOCAL.side[e];
    const verge = (c === 11 || c === 12) && dens > 2500 && dens < 6000 && (sd & 3) ? VERGE_W : 0;
    push({ kind: 'local', e, g, gt, c, cls: clsSurf(local, e), yb: YB[c],
      x: P.x, y: P.y, z: P.z, rx: Array.from(F.rx), rz: Array.from(F.rz), s: ss, cl: ss.slice(),
      offR, offL, sw: sd & 3, cb: (sd >> 2) & 3, verge,
      ends: [keyL(nodes[0]), keyL(nodes[nodes.length - 1])], t: [0, 0], ts: [[0, 0], [0, 0]], len: Ld });
  }
  // src/geom junctions: mouths (sStop / sExit) and boxes (hull), the same the sim drives through
  const tj = Date.now();
  JSET = JUNCTIONS;
  log(`geom junctions: ${JSET.junctionCount} (${JSET.source})`);
  // arms per node
  const arms = new Map();
  ROADS.forEach((r, ri) => {
    const n = r.x.length;
    if (n < 2) return;
    for (const end of [0, 1]) {
      const i = end ? n - 1 : 0, j = end ? n - 2 : 1;
      let ux = r.x[j] - r.x[i], uz = r.z[j] - r.z[i]; const l = Math.hypot(ux, uz) || 1; ux /= l; uz /= l;   // outward from the node
      const sR = end ? 1 : -1;   // road-right == arm-right (seen towards the node) at the end node
      const hR = sR > 0 ? r.offR[i] : -r.offL[i], hL = sR > 0 ? -r.offL[i] : r.offR[i];
      const sideBit = (armSide) => (armSide > 0) === (sR > 0) ? 2 : 1;   // road bit: 2 = right, 1 = left
      let A = arms.get(r.ends[end]); if (!A) arms.set(r.ends[end], (A = []));
      // geom mouth cut at this end: inbound edge stops at sStop, outbound resumes at sExit
      const G = LG.G, inE = end ? r.g : r.gt, outE = end ? r.gt : r.g;
      let mouthCut = 0;
      if (inE >= 0) mouthCut = Math.max(mouthCut, G.length[inE] - JSET.mouth(inE).sStop);
      if (outE >= 0) mouthCut = Math.max(mouthCut, JSET.mouth(outE).sExit);
      const nodeG = end ? G.to(r.g) : G.from(r.g);
      A.push({ ri, end, ux, uz, hR, hL, sideBit, ang: Math.atan2(uz, ux), mouthCut, nodeG });
    }
  });
  // junction classification + trims
  ROADS.junctions = [];
  for (const [key, A] of arms) {
    const deg = A.length;
    // surface arms: primary..living_street (class >= 4, either tier). A node is an at-grade junction
    // only where two or more surface arms meet (a ramp terminal on an arterial, a crossing, a T).
    // Motorway arms are never cut; trunk arms only where >= 2 surface arms cross them; ramps (links)
    // only at their surface end. A local node merely shared with a freeway node cuts nothing on the freeway.
    const surf = A.filter((a) => ROADS[a.ri].c >= 4).length;
    const hasMain = A.some((a) => ROADS[a.ri].kind === 'main');
    const atGrade = deg >= 3 ? surf >= 2 : (deg === 2 && surf === 2 && hasMain && A.some((a) => ROADS[a.ri].kind === 'local'));
    const localJoin = deg === 2 && !hasMain;
    if (!atGrade && !localJoin) continue;
    A.sort((a, b) => a.ang - b.ang);
    const J = { key, arms: A, atGrade };
    ROADS.junctions.push(J);
    if (!atGrade) continue;
    const W = (r, bit) => (r.sw & bit ? CURB_W + r.verge + SIDEWALK_W : r.cb & bit ? CURB_W : 0);
    for (let k = 0; k < deg; k++) {
      const a = A[k], ra = ROADS[a.ri];
      let t = 0; const st = [0, 0];   // per arm side: 0 = arm right, 1 = arm left
      for (const b of [A[(k + 1) % deg], A[(k + deg - 1) % deg]]) {
        if (b === a) continue;
        const rb = ROADS[b.ri];
        const cos = a.ux * b.ux + a.uz * b.uz, th = Math.acos(Math.max(-1, Math.min(1, cos))), sin = Math.sin(th);
        if (th > 2.9) continue;   // straight continuation
        // which side of a faces b: arm-right vector (seen towards the node) = (uz, -ux)
        const aRight = a.uz * b.ux - a.ux * b.uz > 0;
        const bRight = b.uz * a.ux - b.ux * a.uz > 0;
        const ha = aRight ? a.hR : a.hL, hb = bRight ? b.hR : b.hL;
        t = Math.max(t, (hb + ha * cos) / Math.max(0.15, sin) + 0.3);
        const wa = ha + W(ra, a.sideBit(aRight ? 1 : -1)), wb = hb + W(rb, b.sideBit(bRight ? 1 : -1));
        const si = aRight ? 0 : 1;
        st[si] = Math.max(st[si], (wb + wa * cos) / Math.max(0.15, sin) + 0.3);
      }
      const cap = Math.min(35, ra.len * 0.45);
      t = Math.min(Math.max(0, t), cap);
      const cuttable = ra.c >= 4 || ra.c === 1 || ra.c === 3 || (ra.c === 2 && surf >= 2);
      a.cut = cuttable;
      if (a.mouthCut > 0.01) t = Math.min(a.mouthCut, ra.len * 0.45);   // the sim's junction box starts here
      if (!cuttable) { t = 0; st[0] = st[1] = 0; }
      ra.t[a.end] = Math.max(ra.t[a.end], t);
      // road side index: 0 = road right, 1 = road left
      for (const [si, armSide] of [[0, 1], [1, -1]]) {
        const roadSide = a.sideBit(armSide) === 2 ? 0 : 1;
        ra.ts[a.end][roadSide] = Math.max(ra.ts[a.end][roadSide], Math.min(Math.max(t, st[si]), Math.min(40, ra.len * 0.48)));
      }
    }
  }
  log(`street model: ${ROADS.length} roads, ${ROADS.junctions.length} junctions in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
}

/** Road samples between cut-back distances d0 (start) and d1 (end), by the road's arc coordinate cl. */
function sliceRoad(r, d0, d1) {
  const n = r.cl.length, a = r.cl[0] + d0, b = r.cl[n - 1] - d1;
  if (b - a < 0.3) return null;
  const keys = ['x', 'y', 'z', 'rx', 'rz', 's', 'offR', 'offL'], out = { cl: [] };
  for (const k of keys) out[k] = [];
  const at = (i, t, c) => { for (const k of keys) out[k].push(r[k][i] + (r[k][i + 1] - r[k][i]) * t); out.cl.push(c); };
  for (let i = 0; i < n - 1; i++) {
    const c0 = r.cl[i], c1 = r.cl[i + 1];
    if (c1 <= a || c0 >= b) continue;
    if (!out.x.length) at(i, c1 > c0 ? Math.max(0, (a - c0) / (c1 - c0)) : 0, Math.max(a, c0));
    if (c1 < b) { for (const k of keys) out[k].push(r[k][i + 1]); out.cl.push(c1); }
    else { at(i, c1 > c0 ? (b - c0) / (c1 - c0) : 1, b); break; }
  }
  return out.x.length >= 2 ? out : null;
}

/** Point at sample i of slice S at offset o (right of the reference line). */
const ptAt = (S, i, o) => [S.x[i] + S.rx[i] * o, S.z[i] + S.rz[i] * o];

/** Band between two polylines A, B (arrays of [x, z]) at heights yA / yB, facing up. */
function band(level, role, A, B, yA, yB, cls, laneAcross) {
  const n = A.length, V = { x: [], y: [], z: [] }, lane = [], T = [];
  let along = 0;
  for (let i = 0; i < n; i++) {
    if (i) along += Math.hypot(A[i][0] - A[i - 1][0], A[i][1] - A[i - 1][1]);
    V.x.push(A[i][0], B[i][0]); V.z.push(A[i][1], B[i][1]); V.y.push(yA, yB);
    lane.push(laneAcross ? laneAcross[0] : 0, along, laneAcross ? laneAcross[1] : 0, along);
  }
  // orientation from the first quad that is not degenerate
  let flip = false;
  for (let i = 0; i < n - 1; i++) {
    const Q = [A[i], A[i + 1], B[i + 1], B[i]];
    let ar = 0; for (let q = 0; q < 4; q++) { const p0 = Q[q], p1 = Q[(q + 1) % 4]; ar += p0[0] * p1[1] - p1[0] * p0[1]; }
    if (Math.abs(ar) > 1e-6) { flip = ar > 0; break; }
  }
  for (let i = 0; i < n - 1; i++) {
    const a = 2 * i, b = 2 * i + 2, c = 2 * i + 1, d = 2 * i + 3;   // A_i, A_i+1, B_i, B_i+1
    if (flip) T.push(a, c, b, b, c, d); else T.push(a, b, c, c, b, d);
  }
  level.tris(role, V, cls, lane, T);
}

function bezierCorner(pa, ua, pb, ub) {
  // control point: where the two edge lines (along the arms) meet, if it lies inside the junction
  const den = ua[0] * ub[1] - ua[1] * ub[0];
  let X = [(pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2];
  if (Math.abs(den) > 1e-3) {
    const dx = pb[0] - pa[0], dz = pb[1] - pa[1];
    const t = (dx * ub[1] - dz * ub[0]) / den, s = (dx * ua[1] - dz * ua[0]) / den;
    if (t <= 0.5 && s <= 0.5 && t > -40 && s > -40) X = [pa[0] + ua[0] * t, pa[1] + ua[1] * t];
  }
  const out = [];
  for (let k = 0; k <= CORNER_STEPS; k++) {
    const t = k / CORNER_STEPS, a = (1 - t) * (1 - t), b = 2 * t * (1 - t), c = t * t;
    out.push([a * pa[0] + b * X[0] + c * pb[0], a * pa[1] + b * X[1] + c * pb[1]]);
  }
  return out;
}

function streetsL6(level, spec) {
  buildStreetModel();
  const ends = new Map();   // `${ri}:${end}` -> { asphalt slice, side slices }
  ROADS.forEach((r, ri) => {
    const A = sliceRoad(r, r.t[0], r.t[1]);
    if (!A) return;
    const role = r.kind === 'main' ? 'main-asphalt' : 'local-asphalt';
    strip(level, role, A, A, (i) => [[A.offR[i], 0, A.offR[i] - A.offL[i]], [A.offL[i], 0, 0]], r.yb, r.cls, true);
    const rec = { A, sides: [null, null] };
    // curbs, verges, sidewalks, driveways per road side (0 = right, 1 = left)
    for (const side of [0, 1]) {
      const bit = side === 0 ? 2 : 1;
      if (!(r.cb & bit) && !(r.sw & bit)) continue;
      const S = sliceRoad(r, r.ts[0][side], r.ts[1][side]);
      if (!S) continue;
      rec.sides[side] = S;
      const sg = side === 0 ? 1 : -1, edge = (i) => (side === 0 ? S.offR[i] : S.offL[i]);
      const o = (i, d) => edge(i) + sg * d;   // offset d metres outward from the pavement edge
      // curb top (edge -> edge + CURB_W) and face (edge, 0 -> CURB_H); cols ordered right -> left
      const top = (i) => [[o(i, CURB_W), CURB_H, CURB_W], [o(i, 0), CURB_H, 0]];
      const face = (i) => [[o(i, 0), CURB_H, 0], [o(i, 0), 0, CURB_H]];
      strip(level, 'curb', S, S, side === 0 ? top : (i) => top(i).reverse(), r.yb, r.cls, true);
      strip(level, 'curb', S, S, side === 0 ? face : (i) => face(i).reverse(), r.yb, r.cls, true);
      if (!(r.sw & bit)) continue;
      const v = r.verge, w0 = CURB_W + v, w1 = w0 + SIDEWALK_W;
      if (v) {
        const vc = (i) => [[o(i, w0), CURB_H, v], [o(i, CURB_W), CURB_H, 0]];
        strip(level, 'verge', S, S, side === 0 ? vc : (i) => vc(i).reverse(), r.yb - 0.02, r.cls, true);
        // driveways across the verge, one per ~15-24 m lot, not within 8 m of a junction cut
        const L0 = S.cl[0] + 8, L1 = S.cl[S.cl.length - 1] - 8;
        for (let d = L0 + hash01(r.e, side) * 10; d < L1 - 3.2; d += 15 + hash01(r.e * 7 + side, Math.round(d)) * 9) {
          if (hash01(r.e * 13 + side, Math.round(d * 3)) < 0.18) continue;   // some lots have no driveway here
          const D = sliceRoad(S, d - S.cl[0], S.cl[S.cl.length - 1] - d - 3.2);
          if (!D) continue;
          const dc = (i) => [[(side === 0 ? D.offR[i] : D.offL[i]) + sg * w0, CURB_H + 0.004, 3.2], [(side === 0 ? D.offR[i] : D.offL[i]) + sg * CURB_W, CURB_H + 0.004, 0]];
          strip(level, 'driveway', D, D, side === 0 ? dc : (i) => dc(i).reverse(), r.yb - 0.02, r.cls, true);
        }
      }
      const sc = (i) => [[o(i, w1), CURB_H, SIDEWALK_W], [o(i, w0), CURB_H, 0]];
      strip(level, 'sidewalk', S, S, side === 0 ? sc : (i) => sc(i).reverse(), r.yb, r.cls, true);
    }
    ends.set(ri, rec);
  });
  // paint: every directed edge of both tiers from the shared line set
  for (let g = 0; g < LG.G.edgeCount; g++) paintEdge(level, g);
  // junction polygons + rounded corners
  for (const J of ROADS.junctions) {
    const pts = [], A = J.arms;
    let y = 0, yb = 0, cls = 99, ny = 0;
    for (const a of A) {
      const rec = ends.get(a.ri); if (!rec) continue;
      const S = rec.A, i = a.end ? S.x.length - 1 : 0, r = ROADS[a.ri];
      pts.push(ptAt(S, i, S.offR[i]), ptAt(S, i, S.offL[i]));
      y += S.y[i]; ny++; yb = Math.max(yb, r.yb); cls = Math.min(cls, r.cls);
    }
    if (ny < 2) continue;
    y /= ny;
    // union with src/geom's junction box (mouth pavement corners) so the surface covers every connector
    const jg = J.atGrade ? JSET.junctionAt(A[0].nodeG) : -1;
    if (jg >= 0) for (let h = JSET.jFirstH[jg]; h < JSET.jFirstH[jg] + JSET.jCountH[jg]; h++) pts.push([JSET.hull[2 * h], JSET.hull[2 * h + 1]]);
    const H = hull(pts);
    if (H.length >= 3) {
      const T = []; for (let k = 1; k < H.length - 1; k++) T.push(0, k + 1, k);
      level.tris('junction', { x: H.map((p) => p[0]), z: H.map((p) => p[1]), y: H.map(() => y + yb + 0.003) }, cls, null, T);
    }
    if (!J.atGrade || A.length < 3) continue;
    for (let k = 0; k < A.length; k++) {
      const a = A[k], b = A[(k + 1) % A.length];
      const ra = ROADS[a.ri], rb = ROADS[b.ri];
      const recA = ends.get(a.ri), recB = ends.get(b.ri);
      if (!recA || !recB) continue;
      const aRight = a.uz * b.ux - a.ux * b.uz > 0, bRight = b.uz * a.ux - b.ux * a.uz > 0;
      const sideA = a.sideBit(aRight ? 1 : -1) === 2 ? 0 : 1, sideB = b.sideBit(bRight ? 1 : -1) === 2 ? 0 : 1;
      const SA = recA.sides[sideA], SB = recB.sides[sideB];
      if (!SA || !SB) continue;
      const cosT = a.ux * b.ux + a.uz * b.uz;
      if (cosT < -0.97) continue;   // straight-through: the sidewalks already continue
      const iA = a.end ? SA.x.length - 1 : 0, iB = b.end ? SB.x.length - 1 : 0;
      const sgA = sideA === 0 ? 1 : -1, sgB = sideB === 0 ? 1 : -1;
      const eA = sideA === 0 ? SA.offR[iA] : SA.offL[iA], eB = sideB === 0 ? SB.offR[iB] : SB.offL[iB];
      const P = (S, i, e, sg, d) => ptAt(S, i, e + sg * d);
      const ua = [a.ux, a.uz], ub = [b.ux, b.uz];
      const curb = bezierCorner(P(SA, iA, eA, sgA, 0), ua, P(SB, iB, eB, sgB, 0), ub);
      const inner = bezierCorner(P(SA, iA, eA, sgA, CURB_W), ua, P(SB, iB, eB, sgB, CURB_W), ub);
      const yy = Math.max(SA.y[iA] + ra.yb, SB.y[iB] + rb.yb);
      const swA = ra.sw & (sideA === 0 ? 2 : 1), swB = rb.sw & (sideB === 0 ? 2 : 1);
      const Q = [inner[0], curb[0], curb[CORNER_STEPS], inner[CORNER_STEPS]];
      if (!simpleQuad(Q)) continue;
      band(level, 'curb', curb, inner, yy + CURB_H, yy + CURB_H, ra.cls, [0, CURB_W]);
      band(level, 'curb', curb, curb, yy, yy + CURB_H, ra.cls, [CURB_H, 0]);
      if (swA || swB) {
        const wA = swA ? CURB_W + ra.verge + SIDEWALK_W : CURB_W + 0.01, wB = swB ? CURB_W + rb.verge + SIDEWALK_W : CURB_W + 0.01;
        const outer = bezierCorner(P(SA, iA, eA, sgA, wA), ua, P(SB, iB, eB, sgB, wB), ub);
        if (simpleQuad([inner[0], outer[0], outer[CORNER_STEPS], inner[CORNER_STEPS]])) band(level, 'sidewalk', inner, outer, yy + CURB_H, yy + CURB_H, ra.cls, [0, SIDEWALK_W]);
      }
    }
  }
}

/** Cut-back of the road carrying merged edge g at its end node (for stop lines). */
function trimAtEnd(g) {
  if (!ROADS) return 0;
  const ri = ROAD_OF[g]; if (ri < 0) return 0;
  const r = ROADS[ri];
  return r.t[g === r.g ? 1 : 0];
}

/**
 * Controls at L6, all from the shared lane geometry so they sit where the sim stops cars:
 * stop bar across the inbound lanes at the stop line (signal / stop / ramp meter), a continental
 * crosswalk just past it (signals, stops on junctions, signalised ped crossings), a dashed yield
 * line (give_way), and a post on the right edge (role 'signals', cls = control type).
 */
function controlMarkings(level) {
  if (!controls) return;
  const LW = main.laneWidth, G = LG.G;
  const span = (g, s) => {   // [inbound right, inbound left, road right, road left] offsets right of the reference line
    if (g < LBASE) {
      const R = LG.rightEdge(g, s), n = main.lanes(g), tw = main.twin(g), L = main.edgeLength[g];
      const [pr, pl] = pavement(g, tw, s, L);
      return [R, R - n * LW, pr, pl];
    }
    const e = g - LBASE, tw = local.twin(e), pe = local.isPrimaryDir(e) ? e : tw, L = local.edgeLength[e];
    const R = LG.rightEdge(g, s), n = local.lanes(e), [pr, pl] = pavement(g, tw >= 0 ? LBASE + tw : -1, s, L);
    const pk = LOCAL.park[pe], pR = (e === pe ? pk & 2 : pk & 1) ? 2 : 0, pL = (e === pe ? pk & 1 : pk & 2) ? 2 : 0;
    return [R, R - n * LW, pr + pR, pl - pL];
  };
  const yOf = (g) => (g < LBASE ? YB[main.classId(g)] : YB[local.classId(g - LBASE)]) + MARK_LIFT;
  // across-the-road quad between s0..s1 spanning offsets o0..o1
  const patch = (role, g, s0, s1, o0, o1, y, cls) => {
    const V = { x: [], y: [], z: [] };
    for (const s of [s0, s1]) for (const o of [o0, o1]) { LG.offsetPoint(g, s, o, 0, _A); V.x.push(_A.x); V.y.push(_A.y + y); V.z.push(_A.z); }
    level.tris(role, V, cls, null, quadUp(V));
  };
  for (let c = 0; c < controls.count; c++) {
    const type = controls.info[c * 6], flags = controls.info[c * 6 + 1], first = controls.info[c * 6 + 4], cnt = controls.info[c * 6 + 5];
    const onJunction = !(flags & 4);
    for (let a = first; a < first + cnt; a++) {
      const g = controls.approachEdge[a];
      if (g >= G.edgeCount) continue;
      const L = G.length[g];
      // stop bar at the sim's stop point (= src/geom sStop at junction mouths); the crosswalk lies just
      // past it, inside the junction box. Never past this road's drawn end.
      const sMax = L - trimAtEnd(g);
      // where the sim stops: src/geom sStop at a junction mouth (front bumper), the control's own
      // stop line only for mid-block controls (data/controls.* can lag a junctions rebuild)
      const sGeom = JSET.mouth(g).sStop, midBlock = !!(flags & 4);
      const sData = !midBlock && sGeom < L - 0.01 ? sGeom : controls.approachStopAt[a];
      const s = Math.max(0.5, Math.min(sData, sMax));
      if (type !== 4) {
        STOP_CHECK.n++;
        const raw = controls.approachStopAt[a], ss = JSET.mouth(g).sStop;
        if (!midBlock && ss < L - 0.01 && Math.abs(raw - ss) > 0.05) { STOP_CHECK.dataOff++; if (STOP_CHECK.ex.length < 8) STOP_CHECK.ex.push([g, type, flags, +raw.toFixed(2), +ss.toFixed(2), +L.toFixed(1)]); }
        if (!midBlock && ss < L - 0.01 && Math.abs(s - ss) > 0.05) { STOP_CHECK.off++; STOP_CHECK.max = Math.max(STOP_CHECK.max, Math.abs(s - ss)); }
      }
      // stop bars / yield lines / crosswalks: src/geom/lines.js marks (paintEdge); here only the post
      const [, , rr] = span(g, s);
      // post on the right edge at the stop line
      LG.offsetPoint(g, s, rr + 1.0, 0, _A);
      const h = type === 0 || type === 3 ? 4.5 : 2.4;
      box(level, 'signals', _A.x, _A.y + h / 2, _A.z, 0.2, h, 0.2, type);
      if (type === 0 || type === 3) box(level, 'signals', _A.x, _A.y + h - 0.5, _A.z, 0.45, 1.0, 0.45, type);   // signal head
      else box(level, 'signals', _A.x, _A.y + h - 0.35, _A.z, 0.75, 0.75, 0.08, type);                          // sign plate
    }
  }
}

function stationMarkers(level) {
  for (let i = 0; i < stations.count; i++) {
    const x = stations.xyz[i * 3], y = stations.xyz[i * 3 + 1], z = stations.xyz[i * 3 + 2], pumps = stations.info[i * 4 + 2];
    const s = 26 * (0.8 + 0.05 * pumps);   // same marker as src/stations.js
    box(level, 'stations', x, y + 3, z, s, 6, s, 0);
    box(level, 'stations', x, y + 30, z, 4, 60, 4, 1);
  }
}

function coastLines(level, tol) {
  for (const p of water.polygons) for (const r of [p.outer, ...(p.holes || [])]) {
    const ring = r.concat([r[0]]);
    let P = { x: ring.map((q) => q[0]), y: ring.map(() => 0.45), z: ring.map((q) => q[1]) };
    P = subdivide(simplify(P, tol), level.size / 4);
    level.lines('coast', P, 0);
  }
}

// ------------------------------------------------------------------ build
const raster = water ? new WaterRaster(water) : null;
const levels = LEVELS.map((spec, L) => new Level(L, ROOT, spec));
for (const lv of levels) {
  const s = lv.spec, t = Date.now();
  if (s.lines) mainLines(lv, s);
  if (s.strips) mainStrips(lv, s);
  if (s.ribbons && !s.localFull) mainRibbons(lv, s);
  if (local && s.localLines) localLines(lv, s);
  if (local && s.localStrips) localStrips(lv, s);
  if (s.localFull) streetsL6(lv, s);
  if (s.controls) { controlMarkings(lv); log(`stop bars: ${STOP_CHECK.n}; junction bars off src/geom sStop: ${STOP_CHECK.off} (max ${STOP_CHECK.max.toFixed(2)} m); data/controls stopAt != geom sStop on ${STOP_CHECK.dataOff} approaches (stale controls?) e.g. [edge,type,flags,stopAt,sStop,L] ${JSON.stringify(STOP_CHECK.ex.slice(0, 3))}`); }
  if (stations && s.stations) stationMarkers(lv);
  if (water && s.coast) coastLines(lv, s.coast);
  log(`L${lv.L}: geometry in ${((Date.now() - t) / 1000).toFixed(1)}s, ${lv.tiles.size} tiles with roads`);
}

// marking report (data/tiles/AUDIT-markings.json): paint triangles per role; the per-edge lines live in
// data/lines.bin and are checked by node tools/check_markings.mjs
MARK_AUDIT = { source: 'src/geom/lines.js via ' + LINES_SOURCE, runs: LINES.runCount, marks: LINES.markCount, triangles: PAINT_STATS, check: 'node tools/check_markings.mjs --tiles data/tiles' };
log(`paint triangles: ${JSON.stringify(PAINT_STATS)}`);

// Tile existence, top-down. A tile has children iff the next level has road content
// inside it or its water is mixed; it then gets every quadrant that has any content
// (roads or water), so refining never opens a hole.
const exists = levels.map(() => new Map());
function waterFor(lv, ix, iy) {
  if (!raster) return null;
  const b = lv.tileBounds(ix, iy);
  return raster.tile(b.minX, b.minZ, lv.size, lv.spec.waterRes);
}
{
  const root = levels[0].tile(0, 0);
  root.water = waterFor(levels[0], 0, 0);
  exists[0].set(levels[0].key(0, 0), root);
}
for (let L = 0; L < levels.length - 1; L++) {
  const lv = levels[L], nx = levels[L + 1];
  for (const t of exists[L].values()) {
    const kids = [];
    let finer = t.water && !t.water.full;
    for (let q = 0; q < 4; q++) {
      const cx = t.ix * 2 + (q & 1), cy = t.iy * 2 + (q >> 1);
      const ct = nx.tiles.get(nx.key(cx, cy));
      const hasRoads = !!(ct && ct.roles.size);
      if (hasRoads) finer = true;
      kids.push([cx, cy, ct, hasRoads]);
    }
    t.children = [];
    if (!finer) continue;
    for (const [cx, cy, ct0, hasRoads] of kids) {
      const w = t.water ? waterFor(nx, cx, cy) : null;
      if (!hasRoads && !w) continue;
      const ct = ct0 || nx.tile(cx, cy);
      ct.water = w;
      exists[L + 1].set(nx.key(cx, cy), ct);
      t.children.push(ct);
    }
  }
}

// ------------------------------------------------------------------ write
fs.rmSync(OUT, { recursive: true, force: true });
fs.mkdirSync(OUT, { recursive: true });
const manifest = {
  format: 'bay-tiles/1', generated: new Date().toISOString(),
  root: ROOT, xzUnits: XZ_UNITS, roles: ROLES,
  levels: levels.map((lv) => ({ level: lv.L, size: lv.size, err: lv.spec.err, contents: Object.keys(lv.spec).filter((k) => !['err', 'tol', 'waterRes'].includes(k)) })),
  tileFields: ['level', 'ix', 'iy', 'gzBytes', 'rawBytes', 'minX', 'minY', 'minZ', 'maxX', 'maxY', 'maxZ', 'childMask', 'verts', 'tris'],
  tiles: [],
};
const stats = levels.map(() => ({ tiles: 0, gz: 0, raw: 0, verts: 0, tris: 0, maxGz: 0, water: 0 }));
let overflowTotal = 0;
const roleStats = levels.map(() => ({}));
for (let L = 0; L < levels.length; L++) {
  fs.mkdirSync(path.join(OUT, String(L)), { recursive: true });
  for (const t of exists[L].values()) {
    const enc = encodeTile(levels[L], t);
    for (const [r, b] of t.roles) { const k = ROLES[r]; const rs = (roleStats[L][k] ||= { verts: 0, tris: 0 }); rs.verts += b.nv; rs.tris += b.idx ? b.idx.n / 3 : b.nv / 2; }
    const gz = zlib.gzipSync(enc.bytes, { level: 9 });
    fs.writeFileSync(path.join(OUT, tileUrl(L, t.ix, t.iy)), gz);
    let mask = 0;
    for (const c of t.children || []) mask |= 1 << ((c.ix - t.ix * 2) + (c.iy - t.iy * 2) * 2);
    const b = enc.bounds, r1 = (v) => Math.round(v * 10) / 10;
    manifest.tiles.push([L, t.ix, t.iy, gz.length, enc.bytes.length, r1(b.minX), r1(b.minY), r1(b.minZ), r1(b.maxX), r1(b.maxY), r1(b.maxZ), mask, enc.verts, enc.tris]);
    const s = stats[L];
    s.tiles++; s.gz += gz.length; s.raw += enc.bytes.length; s.verts += enc.verts; s.tris += enc.tris; s.maxGz = Math.max(s.maxGz, gz.length);
    if (t.water) s.water++;
    overflowTotal += enc.overflow;
  }
}
if (MARK_AUDIT) fs.writeFileSync(path.join(OUT, 'AUDIT-markings.json'), JSON.stringify(MARK_AUDIT, null, 1));
const mjson = JSON.stringify(manifest);
fs.writeFileSync(path.join(OUT, 'manifest.json'), mjson);
const mgz = zlib.gzipSync(mjson, { level: 9 });
fs.writeFileSync(path.join(OUT, 'manifest.json.gz'), mgz);

const MB = (b) => (b / 1048576).toFixed(2);
const rows = stats.map((s, L) => `  L${L}  ${String(levels[L].size / 1000).padStart(6)} km  ${String(s.tiles).padStart(5)} tiles  ${MB(s.gz).padStart(7)} MB gz  ${MB(s.raw).padStart(7)} MB raw  max ${(s.maxGz / 1024).toFixed(0).padStart(5)} KB  ${(s.verts / 1e6).toFixed(2)} M verts  ${(s.tris / 1e6).toFixed(2)} M tris  ${s.water} water`);
const tot = stats.reduce((a, s) => ({ tiles: a.tiles + s.tiles, gz: a.gz + s.gz, raw: a.raw + s.raw }), { tiles: 0, gz: 0, raw: 0 });
const report = [`tile pyramid: root ${ROOT.size / 1000} km at (${ROOT.x0}, ${ROOT.z0}), ${levels.length} levels`, ...rows,
  `  total ${tot.tiles} tiles, ${MB(tot.gz)} MB gz (${MB(tot.raw)} MB raw); manifest ${(mjson.length / 1024).toFixed(0)} KB (${(mgz.length / 1024).toFixed(0)} KB gz)`,
  `  quantisation overflow vertices: ${overflowTotal}`, `  build time ${((Date.now() - T0) / 1000).toFixed(1)} s`];
report.push('  per role (verts):', ...roleStats.map((r, L) => `  L${L} ` + Object.entries(r).sort((a, b) => b[1].verts - a[1].verts).map(([k, v]) => `${k} ${(v.verts / 1e6).toFixed(2)}M`).join(', ')));
console.log(report.join('\n'));
fs.writeFileSync(path.join(OUT, 'BUILD.txt'), report.join('\n') + '\n');
