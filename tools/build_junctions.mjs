// node tools/build_junctions.mjs [dataDir]
// Precomputes src/geom/junctions.js for the current data: connector paths per (in-edge, lane) ->
// out-edge movement, the conflict matrix per junction, junction-box hulls, stop / exit points at
// every junction mouth, and ramp-merge wait (gore) points. Writes data/junctions.{bin,meta.json}.
import { readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { RoadSet } from '../src/roadpack.js';
import { existsSync } from 'node:fs';
import { graphFromRoadSets, buildLaneGeometry, buildJunctions, crosswalkLegs, JUNCTION_SECTIONS } from '../src/geom/index.js';

const dir = process.argv[2] || new URL('../data/', import.meta.url).pathname;
const TYPES = { Float32: Float32Array, Uint32: Uint32Array };
function load(metaName) {
  const meta = JSON.parse(readFileSync(path.join(dir, metaName), 'utf8'));
  const buf = readFileSync(path.join(dir, meta.file));
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  const arrays = {};
  for (const [name, s] of Object.entries(meta.sections)) arrays[name === 'nodeXYZ' ? 'xyz' : name] = new TYPES[s.type](ab, s.byteOffset, s.length);
  return new RoadSet(meta, arrays);
}
const t0 = Date.now();
const main = load('roads.meta.json'), local = load('roads-local.meta.json');
const G = graphFromRoadSets(main, local), LG = buildLaneGeometry(G);
// crosswalk legs: controls (signals, all-way stops) + painted / signalised OSM crossings
const ctlPath = path.join(dir, 'controls.json'), cwPath = path.join(dir, 'crosswalks.json');
const ctlDoc = existsSync(ctlPath) ? JSON.parse(readFileSync(ctlPath, 'utf8')) : null;
const cwDoc = existsSync(cwPath) ? JSON.parse(readFileSync(cwPath, 'utf8')) : null;
const legs = crosswalkLegs(LG, ctlDoc ? ctlDoc.controls : null, cwDoc ? cwDoc.points : null);
const J = buildJunctions(LG, legs);
const t1 = Date.now();
// controls must stop at the junction mouths: patch approachStopAt (json + bin) for junction controls
let patched = 0;
if (ctlDoc) {
  const cmeta = JSON.parse(readFileSync(path.join(dir, 'controls.meta.json'), 'utf8'));
  const cb = readFileSync(path.join(dir, cmeta.file));
  const sec = cmeta.sections.approachStopAt;
  const stops = new Float32Array(cb.buffer, cb.byteOffset + sec.byteOffset, sec.length);
  let a = 0;
  for (const c of ctlDoc.controls) {
    for (const ap of c.approaches) {
      // a junction control on a node with no junction box (only plain continuations) keeps its own stop line
      if (c.node >= 0 && c.type !== 'ramp_meter' && J.mouthS[2 * ap.edge + 1] < G.length[ap.edge] - 0.5) {
        const v = J.mouthS[2 * ap.edge + 1];
        if (Math.abs(stops[a] - v) > 0.01) patched++;
        stops[a] = v; ap.stopAt = Math.round(v * 100) / 100;
      }
      a++;
    }
  }
  cmeta.stopLinesFrom = 'data/junctions.bin mouthS (tools/build_junctions.mjs)';
  writeFileSync(path.join(dir, cmeta.file), cb);
  writeFileSync(ctlPath, JSON.stringify(ctlDoc));
  writeFileSync(path.join(dir, 'controls.meta.json'), JSON.stringify(cmeta, null, 1));
}
console.log(`crosswalk legs: ${legs.crosswalkStop.size} stop mouths, ${legs.crosswalkExit.size} exit mouths; controls stop lines patched: ${patched}`);
const arrays = J.arrays();
const sections = {};
let off = 0;
const parts = [];
for (const k of JUNCTION_SECTIONS) {
  const a = arrays[k];
  const b = Buffer.from(a.buffer, a.byteOffset, a.byteLength);
  sections[k] = { byteOffset: off, byteLength: b.length, type: a instanceof Float32Array ? 'Float32' : 'Uint32', length: a.length };
  parts.push(b); off += b.length;
}
writeFileSync(path.join(dir, 'junctions.bin'), Buffer.concat(parts));
let conflicts = 0, cross = 0, merge = 0, diverge = 0;
for (let c = 0; c < J.conf.length / 3; c++) { conflicts++; const t = J.conf[3 * c + 2]; if (t === 0) cross++; else if (t === 1) merge++; else diverge++; }
let gores = 0; for (const g of J.goreS) if (g >= 0) gores++;
const meta = {
  format: 'bay-junctions/1', file: 'junctions.bin', byteLength: off, sections,
  edgeIds: 'merged graph: main e, local mainEdgeCount + e (= src/sim fromRoadSet, src/geom)', mainEdgeCount: main.edgeCount,
  nodeIds: 'merged-graph node indices (graphFromRoadSets); they change with every rebuild, edge ids do not',
  layout: {
    mouthS: 'Float32 x2 per edge: [sExit at the start junction, sStop at the end junction] (m along the edge; 0 / length where the node is not a junction)',
    goreS: 'Float32 per edge: ramp wait point (m) where the ramp lane centre clears the mainline pavement by 1.3 m; -1 if the ramp is not a no-added-lane merge',
    jNode: 'Uint32 per junction: merged-graph node', 'jFirstM/jCountM': 'movements of the junction', 'jFirstC/jCountC': 'conflicts of the junction',
    'jFirstH/jCountH': 'junction-box hull points (x,z pairs in hull)',
    mInfo: 'Uint32 x7 per movement: [inEdge, inLane, outEdge, outLane, firstPt, ptCount, preferred]',
    mF: 'Float32 x4 per movement: [sIn (leave inEdge here), sOut (join outEdge here), connector length, turn angle rad (>0 right)]',
    pts: 'Float32 x4 per connector point: [x, y, z, metres along the connector]',
    conf: 'Uint32 x3 per conflict: [movementA, movementB, type 0 crossing / 1 merge (exits within a car width)]; movements sharing an entry lane are not listed (an ordinary queue). Entries after the last junction range (jFirstC/jCountC) are cross-junction conflicts between connectors of neighbouring junctions (crossingFix)',
    confS: 'Float32 x2 per conflict: [metres along A, metres along B] where the paths meet',
    hull: 'Float32 x2 per hull point',
  },
  counts: { junctions: J.junctionCount, movements: J.moveCount, connectorPoints: J.pts.length / 4, conflicts, crossing: cross, merge, sameEntry: diverge, rampGores: gores },
  crossingFix: J.crossingFix || null,
  buildMs: t1 - t0,
};
writeFileSync(path.join(dir, 'junctions.meta.json'), JSON.stringify(meta, null, 1));
console.log(JSON.stringify(meta.counts), JSON.stringify(meta.crossingFix), `${(off / 1e6).toFixed(1)} MB, ${t1 - t0} ms`);
