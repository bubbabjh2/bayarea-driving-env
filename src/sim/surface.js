// Road surfaces and tyre-road friction (docs/SURFACES.md). Pure data + math: no rendering imports.
//
// Every edge carries a pavement surface code and a ground code for each side (what a wheel touches past the paved
// width): data/surfaces.bin, built by tools/surfaces/build.mjs from the OSM surface=* tags. The physics
// (src/sim/physics/dynamics.js stepDynamics, mirrored by src/gpu/kernels/physics.wgsl) scales the vehicle's tyre
// friction mu (VEHICLE_PARAMS, calibrated on dry asphalt) by the surface under its wheels:
//   peak     mu_eff  = P.mu * pk,       pk  = muPeakDry / muPeakDry(asphalt)
//   sliding  F_slide = P.mu * sk * Fz,  sk  = pk * round2(muSlideDry / muPeakDry)   (locked wheels)
//   rolling  Crr_eff = P.Crr * crrMul,  crrMul = crr / crr(asphalt)
// averaged over the right and left wheels (each side reads the pavement code while it is inside the paved width,
// else that side's ground code). Dry asphalt gives pk = 1, sk = 0.8, crrMul = 1 exactly: the physics there is
// bit-identical to the single-mu model. The wet columns are documentation only (friction does not change with weather).

import { laneW } from './roadGraph.js';

// muPeakDry, muSlideDry, muPeakWet, muSlideWet: tyre-road adhesion coefficients (passenger-car tyre);
// crr: rolling-resistance coefficient. Sources per row: docs/SURFACES.md.
export const SURFACES = [
  { name: 'asphalt',       muPeakDry: 0.85, muSlideDry: 0.68, muPeakWet: 0.60, muSlideWet: 0.50, crr: 0.013 },
  { name: 'concrete',      muPeakDry: 0.85, muSlideDry: 0.68, muPeakWet: 0.80, muSlideWet: 0.70, crr: 0.013 },
  { name: 'paving_stones', muPeakDry: 0.75, muSlideDry: 0.60, muPeakWet: 0.55, muSlideWet: 0.45, crr: 0.015 },
  { name: 'sett',          muPeakDry: 0.65, muSlideDry: 0.52, muPeakWet: 0.45, muSlideWet: 0.35, crr: 0.020 },
  { name: 'gravel',        muPeakDry: 0.60, muSlideDry: 0.55, muPeakWet: 0.55, muSlideWet: 0.50, crr: 0.025 },
  { name: 'compacted',     muPeakDry: 0.65, muSlideDry: 0.58, muPeakWet: 0.55, muSlideWet: 0.45, crr: 0.020 },
  { name: 'dirt',          muPeakDry: 0.68, muSlideDry: 0.65, muPeakWet: 0.55, muSlideWet: 0.45, crr: 0.050 },
  { name: 'grass',         muPeakDry: 0.45, muSlideDry: 0.40, muPeakWet: 0.25, muSlideWet: 0.20, crr: 0.100 },
  { name: 'sand',          muPeakDry: 0.40, muSlideDry: 0.35, muPeakWet: 0.40, muSlideWet: 0.35, crr: 0.300 },
  { name: 'metal_grate',   muPeakDry: 0.55, muSlideDry: 0.45, muPeakWet: 0.30, muSlideWet: 0.25, crr: 0.013 },
  { name: 'wood',          muPeakDry: 0.60, muSlideDry: 0.50, muPeakWet: 0.40, muSlideWet: 0.30, crr: 0.015 },
  { name: 'curb',          muPeakDry: 0.85, muSlideDry: 0.68, muPeakWet: 0.80, muSlideWet: 0.70, crr: 0.013 },  // concrete gutter / curb / sidewalk (ground class)
];
export const SURFACE_NAMES = SURFACES.map((s) => s.name);
export const SURFACE_CODES = Object.freeze(Object.fromEntries(SURFACE_NAMES.map((n, i) => [n, i])));
export const SURFACE_COUNT = 16;   // 4-bit codes; 12..15 unused
const REF = SURFACES[0];
const round2 = (x) => Math.round(x * 100) / 100;
/** Per-code physics factors (Float64Array(16) each; unused codes = asphalt). */
export const SURF_PK = new Float64Array(SURFACE_COUNT), SURF_SK = new Float64Array(SURFACE_COUNT), SURF_CRR = new Float64Array(SURFACE_COUNT);
for (let c = 0; c < SURFACE_COUNT; c++) {
  const s = SURFACES[c] || REF, pk = s.muPeakDry / REF.muPeakDry;
  SURF_PK[c] = pk; SURF_SK[c] = pk * round2(s.muSlideDry / s.muPeakDry); SURF_CRR[c] = s.crr / REF.crr;
}

/** OSM surface=* value -> code (-1: not a road surface we model; the build falls back to the class default). */
const OSM = {
  asphalt: 'asphalt', 'asphalt:lanes': 'asphalt', paved: 'asphalt', chipseal: 'asphalt', tarmac: 'asphalt',
  concrete: 'concrete', 'concrete:plates': 'concrete', 'concrete:lanes': 'concrete',
  paving_stones: 'paving_stones', brick: 'paving_stones', bricks: 'paving_stones', 'paving_stones:lanes': 'paving_stones',
  sett: 'sett', cobblestone: 'sett', unhewn_cobblestone: 'sett',
  gravel: 'gravel', fine_gravel: 'gravel', pebblestone: 'gravel',
  compacted: 'compacted',
  unpaved: 'dirt', dirt: 'dirt', ground: 'dirt', earth: 'dirt', mud: 'dirt',
  grass: 'grass', sand: 'sand', metal: 'metal_grate', 'metal_grid': 'metal_grate', steel: 'metal_grate', wood: 'wood',
};
export function osmSurfaceCode(v) { const n = OSM[v]; return n === undefined ? -1 : SURFACE_CODES[n]; }

/** Paved shoulder width (m) counted as pavement past the outer lane on freeway-class roads (sim.js freewayShoulder). */
export const PAVED_SHOULDER_FREEWAY = 3.0;
const SHOULDER_ROADS = new Set(['motorway', 'trunk', 'motorway_link', 'trunk_link']);

/**
 * Attach data/surfaces.bin (Uint8Array, 2 bytes per graph edge) to the graph: e.surf, e.gndR, e.gndL codes.
 * Returns false (graph unchanged: everything asphalt) if the record count does not match.
 */
export function attachSurfaces(graph, bytes) {
  const E = graph.edges.length;
  if (!bytes || bytes.length !== 2 * E) return false;
  for (let i = 0; i < E; i++) {
    const e = graph.edges[i];
    e.surf = bytes[2 * i] & 15; e.gndR = bytes[2 * i + 1] & 15; e.gndL = bytes[2 * i + 1] >> 4;
  }
  graph.surfaces = true;
  return true;
}

/**
 * Surface factors under a car (out.pk, out.sk, out.crr; see the header). e = current edge, s / lat = Frenet pose,
 * hw = half body width, conn = junction connector (>= 0: inside the box, all paved), taper = vs.taper.
 * Mirrored by surfFactors() in src/gpu/kernels/integrate.wgsl.
 */
export function carSurface(graph, e, s, lat, hw, conn, taper, out) {
  const cs = e.surf | 0;
  let cR = cs, cL = cs;
  const gR = e.gndR | 0, gL = e.gndL | 0;
  const J = graph.J;
  if (conn < 0 && !(J && (s < e.sExit + 1 || s > e.sStop)) && (gR !== cs || gL !== cs)) {
    const W = laneW(e, s), latR = lat - hw, latL = lat + hw, nW = e.lanes * W;
    if (latR < 0 || latL > nW) {
      let padR = 0, padL = nW;
      if (e.LG) { const bd = e.LG.bounds(e.idx, s); padR = bd.R - bd.right; padL = bd.R - bd.left; }
      else if (taper > s) { padR -= W; padL += W; }
      const sh = SHOULDER_ROADS.has(e.highway) ? PAVED_SHOULDER_FREEWAY : 0;
      const left = e.reverse >= 0 ? padL + graph.edges[e.reverse].lanes * W : padL + sh;
      if (latR < padR - sh) cR = gR;
      if (latL > left) cL = gL;
    }
  }
  out.pk = 0.5 * (SURF_PK[cR] + SURF_PK[cL]);
  out.sk = 0.5 * (SURF_SK[cR] + SURF_SK[cL]);
  out.crr = 0.5 * (SURF_CRR[cR] + SURF_CRR[cL]);
  out.codeR = cR; out.codeL = cL;
  return out;
}
