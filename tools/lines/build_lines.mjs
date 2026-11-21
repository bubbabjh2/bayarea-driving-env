#!/usr/bin/env node
// node tools/lines/build_lines.mjs [dataDir]
// Lane lines (src/geom/lines.js) for the current data: per edge, per lane boundary, runs of line
// type / colour along s, transverse marks (stop bars, yield lines, crosswalks), and the lane-geometry
// mirror the GPU needs to evaluate line offsets / heading / curvature. Writes data/lines.{bin,meta.json} and data/lines-geom.bin (GPU mirror).
// Build order: after tools/build_junctions.mjs (uses its mouths).
import fs from 'node:fs';
import path from 'node:path';
import { loadWorld, ROOT_DIR } from './load.mjs';
import { crosswalkLegs } from '../../src/geom/index.js';
import { buildLineSet, LINE_SECTIONS, LINE_GEOM_SECTIONS, LINE_NAMES, COLOR_NAMES, MARK_NAMES, LEVEL_NAMES, PATTERNS, PAINT, OUTER_R, OUTER_L } from '../../src/geom/lines.js';

const dir = path.resolve(process.argv[2] || path.join(ROOT_DIR, 'data'));
const t0 = Date.now();
const W = loadWorld(dir, { log: console.log });
const legs = crosswalkLegs(W.LG, W.controls ? W.controls.controls : null, W.crosswalks ? W.crosswalks.points : null);
const LS = buildLineSet(W.LG, W.J, { controls: W.controls ? W.controls.controls : [], legs, points: W.crosswalks ? W.crosswalks.points : [] });
const t1 = Date.now();
const arrays = LS.arrays(), sections = {};
const TN = (a) => a instanceof Float32Array ? 'Float32' : a instanceof Uint32Array ? 'Uint32' : 'Uint8';
const bytes = {};
for (const [file, keys] of [['lines.bin', LINE_SECTIONS], ['lines-geom.bin', LINE_GEOM_SECTIONS]]) {
  const parts = []; let off = 0;
  for (const k of keys) {
    const a = arrays[k];
    const b = Buffer.from(a.buffer, a.byteOffset, a.byteLength);
    const pad = (4 - (off % 4)) % 4;
    if (pad) { parts.push(Buffer.alloc(pad)); off += pad; }
    sections[k] = { file, byteOffset: off, byteLength: b.length, type: TN(a), length: a.length };
    parts.push(b); off += b.length;
  }
  fs.writeFileSync(path.join(dir, file), Buffer.concat(parts));
  bytes[file] = off;
}
const off = bytes['lines.bin'];
const meta = {
  format: 'bay-lines/1', file: 'lines.bin', geomFile: 'lines-geom.bin', byteLength: off, geomByteLength: bytes['lines-geom.bin'], generated: new Date().toISOString(),
  edgeCount: W.G.edgeCount, mainEdgeCount: W.main.edgeCount, vertexCount: W.LG.vFirst[W.G.edgeCount], moveCount: W.J ? W.J.moveCount : 0,
  runCount: LS.runCount, markCount: LS.markCount,
  edgeIds: 'merged graph: main e, local mainEdgeCount + e (= src/sim fromRoadSet, src/geom)',
  spec: 'docs/markings.md', api: 'src/geom/lines.md',
  enums: { type: LINE_NAMES, color: COLOR_NAMES, mark: MARK_NAMES, level: LEVEL_NAMES, span: ['inbound', 'road'], outerR: OUTER_R, outerL: OUTER_L },
  patterns: PATTERNS, paint: PAINT,
  layout: {
    level: 'Uint8 per edge: marking level (0 none, 1 centre line only, 2 full)',
    runFirst: 'Uint32 per edge + 1: first run of the edge (runs sorted by boundary k, then s0)',
    runInfo: 'Uint32 per run: k (bits 0-7: lane boundary 0 = right edge .. lanes = left edge / centre; 254/255 = outer pavement edge of a run-out wedge right / left) | type << 8 (4 bits) | color << 12 (4 bits) | pattern << 16 (8 bits, index into patterns)',
    runS: 'Float32 x2 per run: [s0, s1] m along the edge (data s)',
    markFirst: 'Uint32 per edge + 1', markInfo: 'Uint32 per mark: kind | span << 8', markS: 'Float32 x2 per mark: [s0, s1]',
    lanes: 'Uint8 per edge', length: 'Float32 per edge (data length)',
    laneW: 'Float32 per edge', laneWEnd: 'Float32 x2 per edge (start, end node width)', Rd: 'Float32 per edge', Rend: 'Float32 x2 per edge', tau: 'Float32 x2 per edge',
    xR: 'Uint8 x2 per edge: run-out lane widths on the right (start, end)', xL: 'Uint8 x2 per edge (left)',
    vFirst: 'Uint32 per edge + 1: first vertex of the resampled reference polyline', 'vx/vy/vz': 'Float32 per vertex: position (world m; ~4 mm float32 precision)',
    'mx/mz': 'Float32 per vertex: mitred right normal (offset point = lerp(v) + off * lerp(m))', vs: 'Float32 per vertex: data s',
    vHead: 'Float32 per vertex: heading atan2(dz, dx) of the reference line (rad)', vCurv: 'Float32 per vertex: curvature of the reference line (1/m, + = curving left)',
    mW: 'Float32 x2 per junction movement (data/junctions.bin order): lane width at the connector entry / exit',
  },
  formulas: {
    laneWidth: 'w(e,s) = laneW + (laneWEnd[2e] - laneW) * smooth(1 - s/tau[2e]) + (laneWEnd[2e+1] - laneW) * smooth(1 - (L-s)/tau[2e+1]); smooth(u) = u*u*(3-2u) on clamp(u,0,1); a term is 0 when tau = 0',
    rightEdge: 'R(e,s) = Rd + (Rend[2e] - Rd) * smooth(1 - s/tau[2e]) + (Rend[2e+1] - Rd) * smooth(1 - (L-s)/tau[2e+1])',
    lineOffset: 'boundary k: off = R(e,s) - k * w(e,s)  (m right of the reference line); sim lat of boundary k = k * w(e,s)',
    outerEdges: 'run-out: T = min(90, L); right pavement = R + (xR[2e]*max(0,1-s/T) + xR[2e+1]*max(0,1-(L-s)/T)) * w; left = R - lanes*w - (same with xL) * w',
    curvatureAtOffset: 'kappa_off = kappa / (1 + off * kappa) (off m right of the reference line, kappa + = left)',
  },
  sections,
};
fs.writeFileSync(path.join(dir, 'lines.meta.json'), JSON.stringify(meta, null, 1));
console.log(`lines: ${LS.runCount} runs, ${LS.markCount} marks over ${W.G.edgeCount} edges in ${((t1 - t0) / 1000).toFixed(1)} s; lines.bin ${(off / 1048576).toFixed(1)} MB, lines-geom.bin ${(bytes['lines-geom.bin'] / 1048576).toFixed(1)} MB`);
