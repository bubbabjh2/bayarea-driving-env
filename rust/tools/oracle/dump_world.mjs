// JS oracle dump for the Rust port: node --max-old-space-size=14000 rust/tools/oracle/dump_world.mjs <dataDir> <outDir> [lg] [graph] [controls]
// Writes <outDir>/<name>.meta.json + <name>.bin packs (same convention as data/*.meta.json) with the arrays
// the JS reference computes, so the Rust crates can be checked element for element.
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import path from 'node:path';
import { loadWorld } from '../../../src/sim/data.js';

const [dir, out, ...what0] = process.argv.slice(2);
const what = new Set(what0.length ? what0 : ['lg', 'graph', 'controls']);
mkdirSync(out, { recursive: true });
const T = (a) => a instanceof Float64Array ? 'Float64' : a instanceof Float32Array ? 'Float32' : a instanceof Uint32Array ? 'Uint32' : a instanceof Int32Array ? 'Int32' : a instanceof Uint8Array ? 'Uint8' : a instanceof Int8Array ? 'Int8' : a instanceof Uint16Array ? 'Uint16' : 'Unknown';
function writePack(name, arrays, extra = {}) {
  const sections = {}; const parts = []; let off = 0;
  for (const [k, a] of Object.entries(arrays)) {
    const pad = (8 - (off % 8)) % 8; if (pad) { parts.push(Buffer.alloc(pad)); off += pad; }
    const b = Buffer.from(a.buffer, a.byteOffset, a.byteLength);
    sections[k] = { byteOffset: off, byteLength: b.length, type: T(a), length: a.length };
    parts.push(b); off += b.length;
  }
  writeFileSync(path.join(out, name + '.bin'), Buffer.concat(parts));
  writeFileSync(path.join(out, name + '.meta.json'), JSON.stringify({ file: name + '.bin', byteLength: off, sections, ...extra }, null, 1));
  console.log(`${name}: ${(off / 1e6).toFixed(1)} MB`);
}
const t0 = Date.now();
const W = await loadWorld({ base: dir.endsWith('/') ? dir : dir + '/' });
console.log(`loadWorld ${Date.now() - t0} ms`);
const g = W.graph, LG = g.laneGeom;
if (what.has('lg')) {
  const a = {}; for (const k of ['vFirst', 'vx', 'vy', 'vz', 'vs', 'mx', 'mz', 'Rd', 'Rend', 'tau', 'tapR', 'tapL', 'xR', 'xL', 'laneW', 'laneWEnd']) a[k] = LG[k];
  a.LW = Float64Array.of(LG.LW);
  writePack('lg', a, { stats: LG.stats });
}
if (what.has('graph')) {
  const E = g.edges.length, P = g.edges.reduce((s, e) => s + e.px.length, 0), K = g.edges.reduce((s, e) => s + e.lanes, 0);
  const a = { from: new Uint32Array(E), to: new Uint32Array(E), reverse: new Int32Array(E), lanes: new Uint8Array(E), rank: new Uint8Array(E), navRank: new Uint8Array(E),
    speedLimit: new Float64Array(E), length: new Float64Array(E), headingStart: new Float64Array(E), headingEnd: new Float64Array(E), y: new Float32Array(E), rightEdge: new Float64Array(E),
    surf: new Uint8Array(E), gndR: new Uint8Array(E), gndL: new Uint8Array(E), sStop: new Float32Array(E), sExit: new Float32Array(E), gore: new Float32Array(E),
    ptStart: new Uint32Array(E + 1), px: new Float64Array(P), pz: new Float64Array(P), cum: new Float64Array(P), ys: new Float32Array(P),
    laneStart: new Uint32Array(E + 1), laneOff: new Float32Array(K), outStart: Uint32Array.from(g.outStart), outEdges: Uint32Array.from(g.outEdges), inStart: Uint32Array.from(g.inStart), inEdges: Uint32Array.from(g.inEdges),
    nx: g.nx, nz: g.nz, nodeY: Float32Array.from(g.nodeY) };
  let p = 0, k = 0;
  for (let i = 0; i < E; i++) {
    const e = g.edges[i];
    a.from[i] = e.from; a.to[i] = e.to; a.reverse[i] = e.reverse; a.lanes[i] = e.lanes; a.rank[i] = e.rank; a.navRank[i] = e.navRank;
    a.speedLimit[i] = e.speedLimit; a.length[i] = e.length; a.headingStart[i] = e.headingStart; a.headingEnd[i] = e.headingEnd; a.y[i] = e.y; a.rightEdge[i] = e.rightEdge;
    a.surf[i] = e.surf | 0; a.gndR[i] = e.gndR | 0; a.gndL[i] = e.gndL | 0; a.sStop[i] = e.sStop; a.sExit[i] = e.sExit; a.gore[i] = e.gore;
    a.ptStart[i] = p; for (let j = 0; j < e.px.length; j++) { a.px[p] = e.px[j]; a.pz[p] = e.pz[j]; a.cum[p] = e.cum[j]; a.ys[p] = e.ys ? e.ys[j] : e.y; p++; }
    a.laneStart[i] = k; for (let l = 0; l < e.lanes; l++) a.laneOff[k++] = e.laneOff[l];
  }
  a.ptStart[E] = p; a.laneStart[E] = k;
  const st = W.stations; a.stationNode = Uint32Array.from(st, (s) => s.node);
  writePack('graph', a, { maxSpeed: g.maxSpeed, bounds: g.bounds, contactPairs: g.contactPairs ? g.contactPairs.size : 0 });
}
if (what.has('controls') && W.controls) {
  const C = W.controls, nC = C.count;
  const ph = new Float64Array(nC * 16).fill(NaN), phN = new Uint8Array(nC), phMeta = new Uint8Array(nC * 8);
  for (let c = 0; c < nC; c++) { const P = C.phases[c] || []; phN[c] = P.length; P.slice(0, 4).forEach((q, j) => { ph[16 * c + 4 * j] = q.g; ph[16 * c + 4 * j + 1] = q.y; ph[16 * c + 4 * j + 2] = q.r; phMeta[8 * c + 2 * j] = q.group; phMeta[8 * c + 2 * j + 1] = q.left ? 1 : 0; }); }
  // stateOf samples: every approach at 3 times x 3 movements
  const nA = C.approachCount, S = new Float64Array(nA * 9 * 3), Sst = new Uint8Array(nA * 9);
  const o = {}; const ts = [18000.0, 23456.7, 30001.25];
  for (let a = 0; a < nA; a++) for (let ti = 0; ti < 3; ti++) for (let m = 0; m < 3; m++) { if (C.apEdge[a] < 0) continue; C.stateOf(a, m, ts[ti], o); const q = a * 9 + ti * 3 + m; Sst[q] = o.state; S[3 * q] = o.timeIn; S[3 * q + 1] = o.timeLeft; }
  writePack('controls', { type: C.type, cycle: C.cycle, offset: C.offset, apGroup: C.apGroup, apProtLeft: C.apProtLeft, edgeApStart: C.edgeApStart, edgeApList: C.edgeApList,
    clusterOf: C.clusterOf, phases: ph, phN, phMeta, stateT: S, stateS: Sst }, { stateTimes: ts });
}
console.log(`done ${Date.now() - t0} ms`);
