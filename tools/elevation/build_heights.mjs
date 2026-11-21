// Real road heights: recompute every road node's y from the USGS 3DEP bare-earth DEM (tools/elevation/ground.mjs)
// and keep bridges / tunnels / elevated ways on their structure rules (elevation model "terrain/1", docs/ELEVATION.md).
//
//   node --max-old-space-size=12000 tools/elevation/build_heights.mjs [--in data/build/elev/flat] [--out data/build/elev/out] [--audit <dir>]
//   (normally run through tools/elevation/build_all.mjs = npm run build:elevation)
//
// In : <in>/roads.{bin,meta.json,json}, <in>/roads-local.{bin,meta.json,json}   (the "structures/2" flat-ground build;
//      its y is kept as the structure offset reference: y_old)
// Out: the same six files in <out> with real y, plus vertical-profile shape points (synthetic nodes, negative JSON ids,
//      appended after the existing nodes so every existing node index, edge id, edge length and s coordinate is
//      unchanged) and <audit>/heights.json (stats, outliers).
//
// Model (terrain/1):
//  1. ground nodes (touching any non-structure edge): DEM bilinear (2 m lidar grid in SF, 1/3" elsewhere), averaged
//     over a 4 m disk + centre; bridge abutments use the approach side; floor 0.5 m (land never under the water plane).
//  2. ground edges: profile sampled every 4 m (centre + 2.5 m either side), Gaussian sigma 4 m, pinned to the node
//     heights at the junctions; vertical shape points inserted where the piecewise-linear profile would miss the
//     smoothed DEM by > 0.12 m (Douglas-Peucker on (s, y), >= 5 m apart).
//  3. structures (bridge / tunnel / layer > 0 interior nodes): harmonic (along-road linear) interpolation between
//     their ground ends, so decks span straight between abutments; layer > 0 ways never below the DEM.
//     Structures that carried the 20 m water deck (Bay Bridge, San Mateo, Dumbarton, Richmond, ...) keep at least
//     the old deck / ramp heights (max), which were built over sea level.
//  4. clearance: every plan crossing without a shared node (both tiers) whose old build separated the roads by
//     >= 4.5 m must keep >= 6.0 m centreline separation: the upper road (bridge) is lifted, or the lower (tunnel)
//     lowered, with ramps of 5 % (freeway) / 6 % (links, arterials) / 8 % (local) spreading along the graph.
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadGround } from './ground.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const arg = (k, d) => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : d; };
// input = the flat-ground structures/2 build (committed snapshot data/src/flat, unpacked by build_all.mjs)
const IN = path.resolve(ROOT, arg('--in', 'data/build/elev/flat')), OUT = path.resolve(ROOT, arg('--out', 'data/build/elev/out'));
const AUDIT = arg('--audit', null) ? path.resolve(ROOT, arg('--audit')) : null;   // optional stats / outliers report
mkdirSync(OUT, { recursive: true }); if (AUDIT) mkdirSync(AUDIT, { recursive: true });
const log = (...a) => console.log(`[${new Date().toISOString().slice(11, 19)}]`, ...a);

const P = {
  floor: 0.5, disk: 4, step: 4, sigma: 4, sigma13: 8, sigmaFw: 12, maxFw: 0.08, maxFwLink: 0.12, lateral: 2.5, dpTol: 0.12, minGap: 5,
  clearance: 6.0, oldMin: 4.5, rampFw: 0.05, rampArt: 0.06, rampLoc: 0.08, waterDeck: 19.9,
};
const FW = new Set([0, 1, 2, 3]);                    // motorway, motorway_link, trunk, trunk_link
const LOCAL = new Set([8, 9, 10, 11, 12]);
const rampSlope = (cls) => (cls === 0 || cls === 2 ? P.rampFw : LOCAL.has(cls) ? P.rampLoc : P.rampArt);

// ------------------------------------------------------------------ load both tiers
const TYPES = { Float32: Float32Array, Uint32: Uint32Array, Int32: Int32Array };
function loadTier(name) {
  const meta = JSON.parse(readFileSync(path.join(IN, `${name}.meta.json`), 'utf8'));
  if (meta.elevation && meta.elevation.model !== 'structures/2') throw new Error(`${IN}/${name}: input must be the flat-ground structures/2 build (got ${meta.elevation.model}); use the data/src/flat snapshot`);
  const buf = readFileSync(path.join(IN, meta.file));
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  const S = {};
  for (const [k, s] of Object.entries(meta.sections)) S[k] = new TYPES[s.type](ab, s.byteOffset, s.length);
  const json = JSON.parse(readFileSync(path.join(IN, meta.json || `${name}.json`), 'utf8'));
  if (json.nodes.length !== meta.nodeCount || json.edges.length !== meta.edgeCount) throw new Error(name + ': json/bin mismatch');
  return { name, meta, S, json };
}
const T = [loadTier('roads'), loadTier('roads-local')];
const Nm = T[0].meta.nodeCount, Nl = T[1].meta.nodeCount;
// unified node ids: main n -> n; local n -> Nm + n, or the linked main node
const uni = [new Int32Array(Nm), new Int32Array(Nl)];
for (let i = 0; i < Nm; i++) uni[0][i] = i;
for (let i = 0; i < Nl; i++) uni[1][i] = Nm + i;
{ const L = T[1].S.linkNodes; for (let k = 0; k < L.length; k += 2) uni[1][L[k]] = L[k + 1]; }
const NU = Nm + Nl;
const X = new Float64Array(NU), Z = new Float64Array(NU), Yold = new Float64Array(NU);
for (let t = 0; t < 2; t++) { const xyz = T[t].S.nodeXYZ, n = T[t].meta.nodeCount;
  for (let i = 0; i < n; i++) { const u = uni[t][i]; if (t === 1 && u < Nm) continue; X[u] = xyz[3 * i]; Yold[u] = xyz[3 * i + 1]; Z[u] = xyz[3 * i + 2]; } }

// edges (merged ids: main e, local Nm_e + e)
const Em = T[0].meta.edgeCount;
const E = [];
for (let t = 0; t < 2; t++) {
  const { S, json } = T[t]; const ne = T[t].meta.edgeCount;
  for (let e = 0; e < ne; e++) {
    const f = S.edges[8 * e], n = S.edges[8 * e + 1], j = json.edges[e];
    const nodes = new Int32Array(n); for (let k = 0; k < n; k++) nodes[k] = uni[t][S.poly[f + k]];
    const tw = S.edges[8 * e + 7];
    E.push({ t, e, id: t ? Em + e : e, nodes, cls: S.edges[8 * e + 4], flags: S.edges[8 * e + 6],
      twin: tw === 0xFFFFFFFF ? -1 : (t ? Em + tw : tw), bridge: !!j.bridge, tunnel: !!j.tunnel, layer: j.layer | 0 });
  }
}
const CLASSES = T[0].meta.classesAll;
const DBG = (process.env.DBG_EDGES || '').split(',').filter(Boolean).map(Number);
const dbg = (tag) => { if (!DBG.length) return; for (const id of DBG) { const ed = E[id]; console.log('DBG', tag, id, Array.from(ed.nodes).map((u) => (typeof YY !== 'undefined' ? YY : Y)[u]?.toFixed?.(1)).join(' ')); } };
const isStruct = (ed) => ed.bridge || ed.tunnel || ed.layer > 0;
log('loaded', { Nm, Nl, NU, edges: E.length });

// node classification
const groundTouch = new Uint8Array(NU), structTouch = new Uint8Array(NU), nodeCls = new Int8Array(NU).fill(99);
for (const ed of E) for (const u of ed.nodes) { if (isStruct(ed)) structTouch[u] = 1; else groundTouch[u] = 1; if (ed.cls < nodeCls[u]) nodeCls[u] = ed.cls; }
const isGroundNode = (u) => groundTouch[u] === 1;

// undirected roads: one representative per twin pair (twins share the polyline, reversed)
let twinMismatch = 0;
const roads = [];
for (const ed of E) {
  if (ed.twin >= 0 && ed.twin < ed.id) continue;
  if (ed.twin >= 0) { const tw = E[ed.twin], n = ed.nodes.length; let same = tw.nodes.length === n;
    for (let k = 0; same && k < n; k++) if (tw.nodes[k] !== ed.nodes[n - 1 - k]) same = false;
    if (!same) { twinMismatch++; ed.twin = -1; } }
  roads.push(ed);
}
if (twinMismatch) log('WARNING twins with different polylines (processed separately):', twinMismatch);

// ------------------------------------------------------------------ DEM
const G = loadGround();
log('ground loaded');
let nanSamples = 0;
const dem = (x, z) => { const v = G.heightAt(x, z); if (!Number.isFinite(v)) { nanSamples++; return 0; } return v; };
// robust node height: median of the centre and 8 samples on a 4 m circle (ignores curbs, walls and cliffs beside the road)
const diskAt = (x, z) => { const v = [dem(x, z)]; for (let k = 0; k < 8; k++) { const a = k * Math.PI / 4; v.push(dem(x + P.disk * Math.cos(a), z + P.disk * Math.sin(a))); } v.sort((a, b) => a - b); return (v[3] + 2 * v[4] + v[5]) / 4; };
function median(vals, r) { const n = vals.length, out = new Float64Array(n), w = []; for (let i = 0; i < n; i++) { w.length = 0; for (let j = Math.max(0, i - r); j <= Math.min(n - 1, i + r); j++) w.push(vals[j]); w.sort((a, b) => a - b); out[i] = w[w.length >> 1]; } return out; }
/** two-sided slope limiter: |dy/ds| <= g, keeping the end values (best effort when the ends themselves are too steep) */
function limitSlope(y, ds, g) { const n = y.length, f = Float64Array.from(y), b = Float64Array.from(y);
  for (let i = 1; i < n; i++) f[i] = Math.min(Math.max(f[i], f[i - 1] - g * ds), f[i - 1] + g * ds);
  for (let i = n - 2; i >= 0; i--) b[i] = Math.min(Math.max(b[i], b[i + 1] - g * ds), b[i + 1] + g * ds);
  for (let i = 0; i < n; i++) { const t = n > 1 ? i / (n - 1) : 0; y[i] = (1 - t) * f[i] + t * b[i]; } }

// ------------------------------------------------------------------ 1. ground node heights
const Y = new Float64Array(NU).fill(NaN);
const adjGroundDirs = new Map(); // abutment nodes: directions into their ground edges
for (const ed of E) { if (isStruct(ed)) continue; const n = ed.nodes.length;
  for (const [k, k2] of [[0, 1], [n - 1, n - 2]]) { const u = ed.nodes[k]; if (!structTouch[u]) continue;
    const v = ed.nodes[k2], dx = X[v] - X[u], dz = Z[v] - Z[u], L = Math.hypot(dx, dz) || 1;
    if (!adjGroundDirs.has(u)) adjGroundDirs.set(u, []); adjGroundDirs.get(u).push([dx / L, dz / L, L]); } }
for (let u = 0; u < NU; u++) {
  if (!isGroundNode(u)) continue;
  if (u >= Nm && uni[1][u - Nm] !== u) continue; // (never: linked locals map to main ids)
  let y = diskAt(X[u], Z[u]);
  const dirs = adjGroundDirs.get(u);
  if (dirs) { // bridge abutment: bare-earth DEMs remove the deck; take the approach embankment 6 m back
    let best = -Infinity; for (const [dx, dz, L] of dirs) { const d = Math.min(6, L); best = Math.max(best, diskAt(X[u] + dx * d, Z[u] + dz * d)); }
    y = Math.max(y, best);
  }
  Y[u] = Math.max(P.floor, y);
}
log('ground nodes', Array.from(groundTouch).filter(Boolean).length);

// ------------------------------------------------------------------ 2. ground edge profiles + shape points
const newNodes = []; // {x, z, y}
const insertions = new Map(); // road id -> [{k (segment index), s, id}]
const nodeS = []; // scratch
function gauss(vals, step, sigma) {
  const r = Math.ceil(2.5 * sigma / step), w = []; for (let i = -r; i <= r; i++) w.push(Math.exp(-0.5 * (i * step / sigma) ** 2));
  const n = vals.length, out = new Float64Array(n);
  for (let i = 0; i < n; i++) { let s = 0, ws = 0; for (let j = -r; j <= r; j++) { const k = i + j; if (k < 0 || k >= n) continue; s += vals[k] * w[j + r]; ws += w[j + r]; } out[i] = s / ws; }
  return out;
}
function dp(s, y, a, b, tol, keep) {
  let best = -1, bd = tol;
  for (let i = a + 1; i < b; i++) { const t = (s[i] - s[a]) / (s[b] - s[a]), d = Math.abs(y[i] - (y[a] + (y[b] - y[a]) * t)); if (d > bd) { bd = d; best = i; } }
  if (best < 0) return; keep[best] = 1; dp(s, y, a, best, tol, keep); dp(s, y, best, b, tol, keep);
}
let inserted = 0;
for (const ed of roads) {
  if (isStruct(ed)) continue;
  const nd = ed.nodes, n = nd.length;
  const cum = new Float64Array(n); for (let k = 1; k < n; k++) cum[k] = cum[k - 1] + Math.hypot(X[nd[k]] - X[nd[k - 1]], Z[nd[k]] - Z[nd[k - 1]]);
  const L = cum[n - 1]; if (L < 1e-3) continue;
  const m = Math.max(2, Math.ceil(L / P.step) + 1), ss = new Float64Array(m), raw = new Float64Array(m), seg = new Int32Array(m);
  let k = 0;
  for (let i = 0; i < m; i++) {
    const s = L * i / (m - 1); while (k < n - 2 && cum[k + 1] < s) k++;
    const a = nd[k], b = nd[k + 1], sl = cum[k + 1] - cum[k] || 1, t = Math.min(1, Math.max(0, (s - cum[k]) / sl));
    const dx = (X[b] - X[a]) / sl, dz = (Z[b] - Z[a]) / sl, x = X[a] + (X[b] - X[a]) * t, z = Z[a] + (Z[b] - Z[a]) * t;
    ss[i] = s; seg[i] = k;
    raw[i] = 0.5 * dem(x, z) + 0.25 * dem(x - dz * P.lateral, z + dx * P.lateral) + 0.25 * dem(x + dz * P.lateral, z - dx * P.lateral);
  }
  const fw = FW.has(ed.cls), lidar = G.source(X[nd[0]], Z[nd[0]]) === '1m';
  const ds = L / (m - 1);
  const med = median(raw, Math.max(2, Math.round((fw ? 20 : 8) / ds)));
  const sm = gauss(med, ds, fw ? P.sigmaFw : lidar ? P.sigma : P.sigma13);
  // pin to the end-node heights (linear correction)
  const c0 = Y[nd[0]] - sm[0], c1 = Y[nd[n - 1]] - sm[m - 1];
  for (let i = 0; i < m; i++) { sm[i] = Math.max(P.floor, sm[i] + c0 + (c1 - c0) * ss[i] / L); }
  sm[0] = Y[nd[0]]; sm[m - 1] = Y[nd[n - 1]];
  if (fw) { limitSlope(sm, ds, ed.cls === 0 || ed.cls === 2 ? P.maxFw : P.maxFwLink); sm[0] = Y[nd[0]]; sm[m - 1] = Y[nd[n - 1]]; }
  // interior original nodes take the profile (they belong to this road only)
  const at = (s) => { const f = s / L * (m - 1), i = Math.min(m - 2, Math.floor(f)), t = f - i; return sm[i] + (sm[i + 1] - sm[i]) * t; };
  for (let q = 1; q < n - 1; q++) if (structTouch[nd[q]] ? false : true) Y[nd[q]] = at(cum[q]);
  // shape points: DP over (s, y) including the original nodes as fixed breakpoints
  const keep = new Uint8Array(m); keep[0] = keep[m - 1] = 1;
  // original node positions are breakpoints: split there
  const brk = [0]; { let i = 0; for (let q = 1; q < n - 1; q++) { while (i < m - 1 && ss[i + 1] <= cum[q]) i++; brk.push(i); } brk.push(m - 1); }
  // replace the sample nearest each interior node by the exact node position
  for (let q = 1; q < n - 1; q++) { const i = brk[q]; if (i > 0 && i < m - 1) { ss[i] = cum[q]; sm[i] = Y[nd[q]]; keep[i] = 2; } }
  for (let q = 0; q + 1 < brk.length; q++) if (brk[q + 1] > brk[q]) dp(ss, sm, brk[q], brk[q + 1], P.dpTol, keep);
  const ins = [];
  let lastS = 0;
  for (let i = 1; i < m - 1; i++) {
    if (keep[i] === 2) { lastS = ss[i]; continue; }
    if (keep[i] !== 1) continue;
    const s = ss[i];
    // distance to the neighbouring kept points / original nodes
    let q = 0; while (q < n - 2 && cum[q + 1] < s) q++;
    if (s - cum[q] < P.minGap * 0.6 || cum[q + 1] - s < P.minGap * 0.6 || s - lastS < P.minGap) continue;
    const a = nd[q], b = nd[q + 1], t = (s - cum[q]) / (cum[q + 1] - cum[q]);
    const id = NU + newNodes.length;
    newNodes.push({ x: X[a] + (X[b] - X[a]) * t, z: Z[a] + (Z[b] - Z[a]) * t, y: sm[i], tier: ed.t, cls: ed.cls });
    ins.push({ q, s, id }); lastS = s;
  }
  if (ins.length) { insertions.set(ed.id, ins); inserted += ins.length; }
}
log('ground profiles done; shape points inserted', inserted, 'NaN DEM samples', nanSamples);

// extended arrays (existing + new nodes) for the structure / clearance steps
const NT = NU + newNodes.length;
const XX = new Float64Array(NT), ZZ = new Float64Array(NT), YY = new Float64Array(NT), YO = new Float64Array(NT), CL = new Int8Array(NT);
XX.set(X); ZZ.set(Z); YY.set(Y); YO.set(Yold); CL.set(nodeCls);
newNodes.forEach((p, i) => { XX[NU + i] = p.x; ZZ[NU + i] = p.z; YY[NU + i] = p.y; YO[NU + i] = 0; CL[NU + i] = p.cls; });
// the final polylines (with insertions) per directed edge
function polyOf(ed) {
  const rep = ed.twin >= 0 && ed.twin < ed.id ? E[ed.twin] : ed;
  const ins = insertions.get(rep.id); if (!ins) return Array.from(ed.nodes);
  const out = []; const nd = rep.nodes; let j = 0;
  for (let q = 0; q < nd.length; q++) { out.push(nd[q]); while (j < ins.length && ins[j].q === q) out.push(ins[j++].id); }
  return rep === ed ? out : out.reverse();
}
const POLY = E.map(polyOf);

// ------------------------------------------------------------------ 3. structures
// adjacency over consecutive polyline nodes of every edge (undirected)
const adj = Array.from({ length: NT }, () => []);
for (const ed of roads) { const p = POLY[ed.id]; for (let k = 1; k < p.length; k++) { const a = p[k - 1], b = p[k], L = Math.max(0.5, Math.hypot(XX[b] - XX[a], ZZ[b] - ZZ[a])); adj[a].push([b, L, ed.cls]); adj[b].push([a, L, ed.cls]); } }
const structNodes = []; for (let u = 0; u < NU; u++) if (!isGroundNode(u)) structNodes.push(u);
const layerOnly = new Uint8Array(NT), tunnelNode = new Uint8Array(NT), bridgeNode = new Uint8Array(NT);
for (const ed of E) for (const u of ed.nodes) { if (ed.layer > 0 && !ed.bridge && !ed.tunnel) layerOnly[u] = 1; if (ed.tunnel) tunnelNode[u] = 1; if (ed.bridge) bridgeNode[u] = 1; }
// a structure node is a tunnel node when all its edges are tunnels
const allTunnel = new Uint8Array(NT).fill(1);
for (const ed of E) if (!ed.tunnel) for (const u of ed.nodes) allTunnel[u] = 0;
const Gs = new Float64Array(NT); // natural ground under / over the structure (bare earth: decks removed)
for (const u of structNodes) Gs[u] = Math.max(P.floor, diskAt(XX[u], ZZ[u]));
function sor(nodes, label) {
  let it = 0, delta = 1;
  for (; it < 50000 && delta > 1e-4; it++) {
    delta = 0;
    for (const u of nodes) { let s = 0, w = 0; for (const [v, L] of adj[u]) { s += YY[v] / L; w += 1 / L; } if (!w) continue;
      const nv = YY[u] + 1.9 * (s / w - YY[u]); delta = Math.max(delta, Math.abs(nv - YY[u])); YY[u] = nv; }
  }
  log(label, nodes.length, 'nodes, SOR iterations', it, 'last delta', delta.toExponential(2));
}
// (a) B = harmonic interpolation of the ground ends over every structure (decks span straight between abutments)
for (const u of structNodes) YY[u] = Gs[u];
sor(structNodes, 'structures: harmonic base');
dbg('a');
// (a2) structures that carry a water deck (Bay Bridge, San Mateo, Dumbarton, Richmond, Golden Gate approaches, ...):
//      interpolate the BASE under the old structure (y - y_old) instead, so stacked decks stay stacked (Bay Bridge
//      upper 7 m over lower) and the old deck profile rides on the real terrain at both ends
{
  const comp = new Int32Array(NT).fill(-1); let nc = 0, nodesW = 0; const W = [];
  for (const u0 of structNodes) {
    if (comp[u0] >= 0) continue;
    const st = [u0], mem = []; comp[u0] = nc; let water = false;
    while (st.length) { const a = st.pop(); mem.push(a); if (Yold[a] >= P.waterDeck) water = true; for (const [b] of adj[a]) if (b < NU && !isGroundNode(b) && comp[b] < 0) { comp[b] = nc; st.push(b); } }
    nc++;
    if (water) { W.push(mem); nodesW += mem.length; }
  }
  // group parallel / stacked water structures (nodes within 30 m): one base for the group
  const par = W.map((_, i) => i), find = (i) => (par[i] === i ? i : (par[i] = find(par[i])));
  const cell = new Map(), C = 30;
  W.forEach((mem, i) => { for (const a of mem) { const k = Math.floor(XX[a] / C) * 1e6 + Math.floor(ZZ[a] / C); if (!cell.has(k)) cell.set(k, new Set()); cell.get(k).add(i); } });
  for (const [k, set] of cell) { const gx = Math.floor(k / 1e6), gz = k - gx * 1e6; const all = [...set];
    for (let dx = -1; dx <= 1; dx++) for (let dz = -1; dz <= 1; dz++) { const o = cell.get((gx + dx) * 1e6 + gz + dz); if (o) for (const j of o) all.push(j); }
    for (const j of all) { const ra = find(all[0]), rb = find(j); if (ra !== rb) par[rb] = ra; } }
  const groups = new Map(); W.forEach((mem, i) => { const r = find(i); if (!groups.has(r)) groups.set(r, []); groups.get(r).push(...mem); });
  for (const mem of groups.values()) {
    // base = inverse-distance (power 1) blend of the terrain at the group's ground ends: along a bridge between two
    // ends this is exactly linear, and it is a function of (x, z) only, so stacked decks get the same base
    // (only the ends where the mainline lands when the group has any: ramps up an island hill do not lift the decks)
    let bnd = new Map(); for (const a of mem) for (const [v, , cls] of adj[a]) if ((v >= NU || isGroundNode(v)) && (cls === 0 || cls === 2)) bnd.set(v, YY[v]);
    if (!bnd.size) for (const a of mem) for (const [v] of adj[a]) if (v >= NU || isGroundNode(v)) bnd.set(v, YY[v]);
    for (const a of mem) { let sw = 0, sv = 0; for (const [v, y] of bnd) { const d = Math.max(1, Math.hypot(XX[a] - XX[v], ZZ[a] - ZZ[v])); sw += 1 / d; sv += y / d; }
      // deck = the ends' base + the stacking offset over the lowest water deck (Bay Bridge upper +7 m), never lower than
      // the old deck over the water (20 m, +7 m per extra layer)
      YY[a] = Math.max((bnd.size ? sv / sw : Gs[a]) + Math.max(0, Yold[a] - 20), P.floor + Yold[a]); }
  }
  const compsW = `${W.length} in ${groups.size} groups`;
  log('water-deck structures', compsW, 'nodes', nodesW, '(deck = max(inverse-distance base of the mainline ends + stacking offset, old water deck))');
}
dbg('a2');
// (b) bridges / elevated ways: at least their old deck height (structures/2: 7 m per layer, 20 m water decks, crossing
//     lifts) above the natural ground under them, never below the straight span
const deckNodes = structNodes.filter((u) => !allTunnel[u]);
let deckRaised = 0;
// over water (natural ground at the floor) decks keep their old water / layer deck height above the water; elsewhere
// the deck spans straight between its abutments (the bare-earth DEM already holds the approach embankments) and is
// lifted only where it must clear a road below (step 4)
for (const u of deckNodes) { if (Gs[u] > P.floor + 0.5) continue; const v = Gs[u] + Math.max(0, Yold[u]); if (v > YY[u]) { YY[u] = v; deckRaised++; } }
// (c) tunnels: interpolated between whatever they connect to (portals, bridge decks), never less than their old
//     depth below the ground over them (underpasses)
const tunnelNodes = structNodes.filter((u) => allTunnel[u]);
sor(tunnelNodes, 'tunnels: harmonic between portals / decks');
let tunnelLowered = 0;
for (const u of tunnelNodes) { const v = Gs[u] + Math.min(Yold[u], -3); if (v < YY[u]) { YY[u] = v; tunnelLowered++; } }
log('decks raised to water deck height', deckRaised, 'of', deckNodes.length, '; tunnels lowered under the ground', tunnelLowered, 'of', tunnelNodes.length);
dbg('c');
// (d) approach ramps: from every deck node, ground roads may not fall away faster than the ramp grade
//     (5 % freeway, 6 % links / arterials, 8 % local); from tunnels they may not rise faster
class Heap { constructor() { this.a = []; } push(v, u) { const a = this.a; a.push([v, u]); let i = a.length - 1; while (i > 0) { const p = (i - 1) >> 1; if (a[p][0] >= a[i][0]) break; [a[p], a[i]] = [a[i], a[p]]; i = p; } }
  pop() { const a = this.a, top = a[0], last = a.pop(); if (a.length) { a[0] = last; let i = 0; for (;;) { const l = 2 * i + 1, r = l + 1; let m = i; if (l < a.length && a[l][0] > a[m][0]) m = l; if (r < a.length && a[r][0] > a[m][0]) m = r; if (m === i) break; [a[m], a[i]] = [a[i], a[m]]; i = m; } } return top; }
  get size() { return this.a.length; } }
/** sign +1: raise y to max(y, seed - slope * dist); sign -1: lower to min(y, seed + slope * dist). Only into nodes that pass `into`. */
function spreadRamps(seeds, sign, into) {
  const best = new Float64Array(NT).fill(-Infinity), h = new Heap();
  for (const [u, v] of seeds) { const k = sign * v; if (k > best[u]) { best[u] = k; h.push(k, u); } }
  let changed = 0;
  while (h.size) { const [k, u] = h.pop(); if (k < best[u]) continue;
    for (const [w, L, cls] of adj[w0(u)]) { if (!into(w)) continue; const nk = k - rampSlope(cls) * L; if (nk > best[w] && nk > sign * YY[w]) { best[w] = nk; h.push(nk, w); } } }
  for (let u = 0; u < NT; u++) if (best[u] > -Infinity && into(u) && best[u] > sign * YY[u]) { YY[u] = sign * best[u]; changed++; }
  return changed;
}
const w0 = (u) => u;
const isGroundAny = (u) => u >= NU || isGroundNode(u);
const rampUp = spreadRamps(deckNodes.map((u) => [u, YY[u]]), 1, isGroundAny);
const rampDown = spreadRamps(tunnelNodes.map((u) => [u, YY[u]]), -1, isGroundAny);
log('approach ramps: ground nodes raised', rampUp, 'lowered', rampDown);

dbg('d');
// ------------------------------------------------------------------ 3b. plausibility limiter (DEM spikes, deck/ground joins)
// Per segment max grade: freeway mainline 8 %, freeway links 12 %, primary 20 %, secondary 25 %, tertiary 32 %,
// local 42 % (Bradford St SF ~41 %); structures 8 % (freeway) / 10 % (links) / 12 % (others). Violations are removed by
// moving both ends toward each other (iterated), so isolated spikes (retaining walls, cut edges, bridge-removal seams in
// the bare-earth DEM, abutment joins) are flattened while sustained real grades below the limit are untouched.
const segLimit = (cls, st) => st ? (cls === 0 || cls === 2 ? 0.08 : FW.has(cls) ? 0.10 : 0.12)
  : cls === 0 || cls === 2 ? P.maxFw : FW.has(cls) ? P.maxFwLink : cls <= 5 ? 0.20 : cls <= 7 ? 0.25 : cls <= 9 ? 0.32 : 0.42;
const SEG = []; for (const ed of roads) { const p = POLY[ed.id], st = isStruct(ed), g = segLimit(ed.cls, st); for (let k = 1; k < p.length; k++) SEG.push(p[k - 1], p[k], g); }
function limiter(rounds, fixed = null) {
  let viol = 0, it = 0, moved = 0;
  for (; it < rounds; it++) { viol = 0;
    for (let i = 0; i < SEG.length; i += 3) { const a = SEG[i], b = SEG[i + 1], g = SEG[i + 2];
      const L = Math.max(2, Math.hypot(XX[b] - XX[a], ZZ[b] - ZZ[a])), d = YY[b] - YY[a], ex = Math.abs(d) - g * L;
      if (ex <= 1e-3) continue; viol++;
      const fa = fixed ? fixed[a] : 0, fb = fixed ? fixed[b] : 0, wa = fa && !fb ? 0 : fb && !fa ? 1 : 0.5, sd = Math.sign(d) * ex;
      YY[a] += wa * sd; YY[b] -= (1 - wa) * sd; moved += ex; }
    if (!viol) break; }
  return { rounds: it, remaining: viol, movedM: +moved.toFixed(0) };
}
const limitStats = limiter(400);
log('limiter', limitStats);

dbg('lim');
// ------------------------------------------------------------------ 4. clearance at plan crossings
function crossingsOf() {
  // all plan crossings of segments from different roads without a shared node, both tiers
  const CELL = 200, grid = new Map(), segs = [];
  for (const ed of roads) { const p = POLY[ed.id]; for (let k = 1; k < p.length; k++) segs.push([ed.id, k - 1, p[k - 1], p[k]]); }
  for (let i = 0; i < segs.length; i++) { const [, , a, b] = segs[i];
    const x0 = Math.floor(Math.min(XX[a], XX[b]) / CELL), x1 = Math.floor(Math.max(XX[a], XX[b]) / CELL), z0 = Math.floor(Math.min(ZZ[a], ZZ[b]) / CELL), z1 = Math.floor(Math.max(ZZ[a], ZZ[b]) / CELL);
    for (let gx = x0; gx <= x1; gx++) for (let gz = z0; gz <= z1; gz++) { const key = gx * 100000 + gz; let c = grid.get(key); if (!c) grid.set(key, c = []); c.push(i); } }
  const out = [], seen = new Set();
  for (const c of grid.values()) for (let ii = 0; ii < c.length; ii++) for (let jj = ii + 1; jj < c.length; jj++) {
    const i = c[ii], j = c[jj], A = segs[i], B = segs[j];
    if (A[0] === B[0]) continue;
    const a = A[2], b = A[3], p = B[2], q = B[3];
    if (a === p || a === q || b === p || b === q) continue;
    const rx = XX[b] - XX[a], rz = ZZ[b] - ZZ[a], sx = XX[q] - XX[p], sz = ZZ[q] - ZZ[p], den = rx * sz - rz * sx;
    if (Math.abs(den) < 1e-9) continue;
    const t = ((XX[p] - XX[a]) * sz - (ZZ[p] - ZZ[a]) * sx) / den, v = ((XX[p] - XX[a]) * rz - (ZZ[p] - ZZ[a]) * rx) / den;
    if (t <= 0 || t >= 1 || v <= 0 || v >= 1) continue;
    const key = i < j ? i * 4194304 + j : j * 4194304 + i; if (seen.has(key)) continue; seen.add(key);
    out.push({ ra: A[0], rb: B[0], a, b, p, q, t, v, x: XX[a] + rx * t, z: ZZ[a] + rz * t });
  }
  return out;
}
const yAt = (Yarr, a, b, t) => Yarr[a] + (Yarr[b] - Yarr[a]) * t;
// old heights of inserted points = linear between their segment ends (they are on ground roads: 0 or ramp)
for (const [rid, ins] of insertions) { const nd = E[rid].nodes; const cum = [0]; for (let k = 1; k < nd.length; k++) cum.push(cum[k - 1] + Math.hypot(X[nd[k]] - X[nd[k - 1]], Z[nd[k]] - Z[nd[k - 1]]));
  for (const it of ins) { const a = nd[it.q], b = nd[it.q + 1], t = (it.s - cum[it.q]) / ((cum[it.q + 1] - cum[it.q]) || 1); YO[it.id] = Yold[a] + (Yold[b] - Yold[a]) * t;
  } }
const X0 = crossingsOf();
log('plan crossings without a shared node', X0.length);
const structRoad = (rid) => isStruct(E[rid]);
const crossStats = { checked: 0, exempt: 0, liftedRounds: [], inverted: 0, finalBelow: 0, finalBelowList: [] };
const lift = new Float64Array(NT); // cumulative applied lift (audit)
for (let round = 0; round < 6; round++) {
  const seeds = []; let bad = 0;
  for (const c of X0) {
    const ya0 = yAt(YO, c.a, c.b, c.t), yb0 = yAt(YO, c.p, c.q, c.v);
    if (Math.abs(ya0 - yb0) < P.oldMin) { if (round === 0) crossStats.exempt++; continue; }
    if (round === 0) crossStats.checked++;
    const ea = E[c.ra], eb = E[c.rb];
    const upA = ea.tunnel !== eb.tunnel ? !ea.tunnel : (isStruct(ea) && !ea.tunnel) !== (isStruct(eb) && !eb.tunnel) ? (isStruct(ea) && !ea.tunnel) : ya0 > yb0;
    const [ua, ub, ut, ur] = upA ? [c.a, c.b, c.t, c.ra] : [c.p, c.q, c.v, c.rb];
    const [la, lb, lt, lr] = upA ? [c.p, c.q, c.v, c.rb] : [c.a, c.b, c.t, c.ra];
    const yu = yAt(YY, ua, ub, ut), yl = yAt(YY, la, lb, lt), need = P.clearance - (yu - yl);
    if (round === 0 && yu < yl) crossStats.inverted++;
    if (round === 0 && need > 3) (crossStats.bigNeed ||= []).push({ need: +need.toFixed(1), x: +c.x.toFixed(0), z: +c.z.toFixed(0), up: CLASSES[E[ur].cls] + (E[ur].bridge ? '/bridge' : E[ur].tunnel ? '/tunnel' : '') + ' ' + ur, lo: CLASSES[E[lr].cls] + (E[lr].bridge ? '/bridge' : E[lr].tunnel ? '/tunnel' : '') + ' ' + lr, yu: +yu.toFixed(1), yl: +yl.toFixed(1), yu0: +(upA ? ya0 : yb0).toFixed(1), yl0: +(upA ? yb0 : ya0).toFixed(1) });
    if (need <= 0.01) continue;
    bad++;
    // move the structure: the upper road if it is a bridge / elevated (or neither is a tunnel), else lower the tunnel
    const lowerTunnel = E[lr].tunnel && !(E[ur].bridge || E[ur].layer > 0);
    if (lowerTunnel) { seeds.push([la, -need], [lb, -need]); }
    else { seeds.push([ua, need], [ub, need]); }
  }
  crossStats.liftedRounds.push(bad);
  if (!bad) break;
  // spread: delta(n) = max over seeds of (need - slope * graph distance), signed; Dijkstra-like relaxation
  for (const sign of [1, -1]) {
    const best = new Float64Array(NT); const heap = [];
    const push = (u, v) => { if (v <= best[u] + 1e-6) return; best[u] = v; heap.push([v, u]); };
    for (const [u, d] of seeds) if (Math.sign(d) === sign) push(u, Math.abs(d));
    heap.sort((a, b) => a[0] - b[0]);
    while (heap.length) {
      // pop max (small heaps: keep sorted by insertion)
      let bi = 0; for (let i = 1; i < heap.length; i++) if (heap[i][0] > heap[bi][0]) bi = i;
      const [v, u] = heap[bi]; heap[bi] = heap[heap.length - 1]; heap.pop();
      if (v < best[u] - 1e-9) continue;
      for (const [w, L, cls] of adj[u]) { const nv = v - rampSlope(cls) * L; if (nv > 0) push(w, nv); }
    }
    for (let u = 0; u < NT; u++) if (best[u] > 0) { YY[u] += sign * best[u]; lift[u] += sign * best[u]; }
  }
}
dbg('clear');
// the lifts add to the existing grade: limit again with the crossing segments held
{ const fixed = new Uint8Array(NT); for (const c of X0) fixed[c.a] = fixed[c.b] = fixed[c.p] = fixed[c.q] = 1;
  crossStats.limiterAfter = limiter(400, fixed); log('limiter after clearance', crossStats.limiterAfter); }
for (let u = 0; u < NT; u++) if (u >= NU || isGroundNode(u)) YY[u] = Math.max(YY[u], tunnelNode[u] ? -1e9 : P.floor - 0.3);
for (const c of X0) {
  const ya0 = yAt(YO, c.a, c.b, c.t), yb0 = yAt(YO, c.p, c.q, c.v); if (Math.abs(ya0 - yb0) < P.oldMin) continue;
  const dy = Math.abs(yAt(YY, c.a, c.b, c.t) - yAt(YY, c.p, c.q, c.v));
  if (dy < P.oldMin) { crossStats.finalBelow++; if (crossStats.finalBelowList.length < 30) crossStats.finalBelowList.push({ x: +c.x.toFixed(1), z: +c.z.toFixed(1), dy: +dy.toFixed(2), a: c.ra, b: c.rb }); }
}
{ let n = 0, mx = 0; for (let u = 0; u < NT; u++) if (lift[u] !== 0) { n++; mx = Math.max(mx, Math.abs(lift[u])); } crossStats.liftedNodes = n; crossStats.maxLift = +mx.toFixed(2); }
log('clearance', crossStats);

// ------------------------------------------------------------------ 5. audit: grades per edge segment
function gradeStats(Yarr, polys, minLen) {
  const byCls = {}; const all = { len: 0, maxG: 0, sumGL: 0, over: { 8: 0, 10: 0, 15: 0, 20: 0, 30: 0 }, edgesOver: { 10: 0, 15: 0, 20: 0, 30: 0 }, n: 0 };
  const worst = [];
  for (const ed of roads) {
    const p = polys[ed.id]; let eMax = 0; const c = CLASSES[ed.cls]; const B = byCls[c] || (byCls[c] = { len: 0, maxG: 0, sumGL: 0, over10: 0, over15: 0, over20: 0 });
    // grade over windows of >= minLen (like the physics' min segment)
    let k0 = 0;
    for (let k = 1; k < p.length; k++) {
      const L = Math.hypot(XX[p[k]] - XX[p[k - 1]], ZZ[p[k]] - ZZ[p[k - 1]]); if (L <= 0) continue;
      const g = Math.abs(Yarr[p[k]] - Yarr[p[k - 1]]) / Math.max(minLen, L);
      all.len += L; all.sumGL += g * L; B.len += L; B.sumGL += g * L;
      for (const th of [8, 10, 15, 20, 30]) if (g > th / 100) all.over[th] += L;
      if (g > 0.10) B.over10 += L; if (g > 0.15) B.over15 += L; if (g > 0.20) B.over20 += L;
      if (g > eMax) eMax = g; if (g > B.maxG) B.maxG = g; if (g > all.maxG) all.maxG = g;
      if (g > 0.30) worst.push({ g: +g.toFixed(3), cls: c, edge: ed.id, x: +XX[p[k]].toFixed(1), z: +ZZ[p[k]].toFixed(1), L: +L.toFixed(1), struct: isStruct(ed) });
      void k0;
    }
    all.n++; for (const th of [10, 15, 20, 30]) if (eMax > th / 100) all.edgesOver[th]++;
  }
  const pct = (v) => +(100 * v / all.len).toFixed(3);
  const summary = { roads: all.n, km: +(all.len / 1000).toFixed(1), maxGrade: +all.maxG.toFixed(3), meanGrade: +(all.sumGL / all.len).toFixed(4),
    pctLengthOver: Object.fromEntries(Object.entries(all.over).map(([k, v]) => [k + '%', pct(v)])),
    pctRoadsWithSegmentOver: Object.fromEntries(Object.entries(all.edgesOver).map(([k, v]) => [k + '%', +(100 * v / all.n).toFixed(2)])),
    byClass: Object.fromEntries(Object.entries(byCls).map(([k, B]) => [k, { km: +(B.len / 1000).toFixed(1), maxGrade: +B.maxG.toFixed(3), meanGrade: +(B.sumGL / B.len).toFixed(4), pctOver10: +(100 * B.over10 / B.len).toFixed(2), pctOver15: +(100 * B.over15 / B.len).toFixed(2), pctOver20: +(100 * B.over20 / B.len).toFixed(2) }])) };
  worst.sort((a, b) => b.g - a.g);
  return { summary, worst: worst.slice(0, 40) };
}
const oldPolys = E.map((ed) => Array.from(ed.nodes));
const YOorig = new Float64Array(NT); YOorig.set(Yold);
const statsOld = gradeStats(YOorig, oldPolys, 40), statsNew = gradeStats(YY, POLY, 1), statsNew10 = gradeStats(YY, POLY, 10);
// height deltas
let dsum = 0, dmax = 0, dmaxAt = null; const dh = [];
for (let u = 0; u < NU; u++) { const d = YY[u] - Yold[u]; dsum += Math.abs(d); dh.push(d); if (Math.abs(d) > dmax) { dmax = Math.abs(d); dmaxAt = { x: +X[u].toFixed(1), z: +Z[u].toFixed(1), old: +Yold[u].toFixed(2), new: +YY[u].toFixed(2) }; } }
dh.sort((a, b) => a - b);
const q = (f) => +dh[Math.floor(f * (dh.length - 1))].toFixed(2);
const audit = {
  model: 'terrain/1', params: P, generated: new Date().toISOString(),
  nodes: { existing: NU, inserted: newNodes.length, structureNodes: structNodes.length, nanDemSamples: nanSamples },
  heightDelta: { meanAbs: +(dsum / NU).toFixed(2), max: +dmax.toFixed(2), maxAt: dmaxAt, p01: q(0.01), p50: q(0.5), p99: q(0.99), newMax: +YY.reduce((m, v) => Math.max(m, v), -Infinity).toFixed(1) },
  limiter: limitStats, gradeOld_min40m: statsOld.summary, gradeNew_raw: statsNew.summary, gradeNew_min10m: statsNew10.summary,
  worstNew_raw: statsNew.worst, worstOld: statsOld.worst.slice(0, 10), clearance: crossStats,
};
if (AUDIT) writeFileSync(path.join(AUDIT, 'heights.json'), JSON.stringify(audit, null, 1));
log('audit', JSON.stringify({ heightDelta: audit.heightDelta, old: statsOld.summary.pctLengthOver, new10: statsNew10.summary.pctLengthOver, max10: statsNew10.summary.maxGrade, mean10: statsNew10.summary.meanGrade }));

// ------------------------------------------------------------------ 6. write both tiers
let nextSynthId = 0; for (const t of T) for (const nd of t.json.nodes) if (nd.id < nextSynthId) nextSynthId = nd.id;
for (let t = 0; t < 2; t++) {
  const { meta, S, json } = T[t]; const n0 = meta.nodeCount, ne = meta.edgeCount;
  // tier-local ids for new nodes of this tier
  const tierNew = []; newNodes.forEach((p, i) => { if (p.tier === t) tierNew.push(NU + i); });
  const local = new Map(); tierNew.forEach((u, i) => local.set(u, n0 + i));
  // unified -> tier index for existing nodes: build reverse map
  const rev = new Map(); for (let i = 0; i < n0; i++) rev.set(uni[t][i], i);
  const N = n0 + tierNew.length;
  const xyz = new Float32Array(3 * N); xyz.set(S.nodeXYZ);
  for (let i = 0; i < n0; i++) xyz[3 * i + 1] = YY[uni[t][i]];
  tierNew.forEach((u, i) => { xyz[3 * (n0 + i)] = XX[u]; xyz[3 * (n0 + i) + 1] = YY[u]; xyz[3 * (n0 + i) + 2] = ZZ[u]; });
  const edges = new Uint32Array(S.edges); const polyArr = [];
  // synthetic ids are negative and unique ACROSS both tiers (the JSON graph joins tiers by id)
  let minId = nextSynthId;
  for (let e = 0; e < ne; e++) {
    const p = POLY[t ? Em + e : e]; edges[8 * e] = polyArr.length; edges[8 * e + 1] = p.length;
    for (const u of p) { const i = u >= NU ? local.get(u) : rev.get(u); if (i === undefined) throw new Error('node not in tier'); polyArr.push(i); }
    json.edges[e].nodes = p.map((u) => (u >= NU ? local.get(u) : rev.get(u)));
  }
  for (let i = 0; i < n0; i++) json.nodes[i].y = +YY[uni[t][i]].toFixed(2);
  nextSynthId = minId - tierNew.length;
  tierNew.forEach((u, i) => json.nodes.push({ id: minId - 1 - i, x: +XX[u].toFixed(2), y: +YY[u].toFixed(2), z: +ZZ[u].toFixed(2) }));
  const poly = Uint32Array.from(polyArr);
  // re-pack the sections in their original order
  const arrays = { ...S, nodeXYZ: xyz, edges, poly };
  const order = Object.entries(meta.sections).sort((a, b) => a[1].byteOffset - b[1].byteOffset).map(([k]) => k);
  const parts = []; let off = 0; const sec = {};
  for (const k of order) { const a = arrays[k]; const b = Buffer.from(a.buffer, a.byteOffset, a.byteLength);
    sec[k] = { ...meta.sections[k], byteOffset: off, byteLength: b.length, length: a.length, count: a.length / meta.sections[k].stride };
    parts.push(b); off += b.length; const pad = (-b.length) & 3; if (pad) { parts.push(Buffer.alloc(pad)); off += pad; } }
  const bin = Buffer.concat(parts);
  meta.sections = sec; meta.byteLength = bin.length; meta.nodeCount = N;
  meta.elevation = { model: 'terrain/1', previous: meta.elevation, ground: 'USGS 3DEP bare earth: 1 m lidar DEM (CA_SanFrancisco_B23) in San Francisco, 1/3 arc-second seamless DEM elsewhere; NAVD88 m (data/dem/SOURCE.txt)',
    doc: 'docs/elevation.md', tool: 'tools/elevation/build_heights.mjs', params: P, shapePoints: tierNew.length, builtFrom: 'structures/2 flat-ground build (y_old = structure offsets)' };
  meta.heightsGenerated = new Date().toISOString();
  let ys = []; for (let i = 0; i < N; i++) ys.push(xyz[3 * i + 1]);
  json.meta = { ...json.meta, elevation: meta.elevation, nodeCount: N };
  writeFileSync(path.join(OUT, meta.file), bin);
  writeFileSync(path.join(OUT, `${t ? 'roads-local' : 'roads'}.meta.json`), JSON.stringify(meta, null, 1));
  writeFileSync(path.join(OUT, meta.json || `${t ? 'roads-local' : 'roads'}.json`), JSON.stringify(json));
  log('wrote', meta.file, N, 'nodes (', tierNew.length, 'new )', bin.length, 'bytes');
}
