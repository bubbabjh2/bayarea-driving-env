// Synthetic road graphs in the roads.json schema, for tests and training when
// the real data/roads.json is not available.
//
// makeSyntheticGraph(): a 20 km "US 101" dual carriageway (x axis, EB at
// z=+15, WB at z=-15), an exit/on-ramp pair to "Main St" (primary, N-S at
// x=5400), a "CA 237" motorway fork at x=12000, "Oak Ave" (secondary, E-W at
// z=3000), residential "Elm St"/"Pine St", and "Bay Rd" (tertiary) linking
// CA 237's end to Pine St. Contains every maneuver type guidance detects.
// makeGridGraph(cols, rows, spacing): big two-way grid for A* perf tests.

export function makeSyntheticGraph() {
  const nodes = [], edges = [];
  const N = (id, x, z) => { nodes.push({ id, x, z }); return id; };
  // US 101 eastbound (+x) at z=+15, westbound (-x) at z=-15
  const E = [], W = [];
  for (let i = 0; i <= 20; i++) { E.push(N(`E${i}`, i * 1000, 15)); W.push(N(`W${i}`, i * 1000, -15)); }
  edges.push({ nodes: E, highway: 'motorway', ref: 'US 101', name: 'Bayshore Fwy', oneway: true, lanes: 4, maxspeed: '65 mph' });
  edges.push({ nodes: W.slice().reverse(), highway: 'motorway', ref: 'US 101', name: 'Bayshore Fwy', oneway: true, lanes: 4, maxspeed: '65 mph' });
  // Main St: primary two-way along z at x=5400
  const MS = [];
  const msZ = [-3000, -2500, -2000, -1500, -1000, -500, -200, 200, 500, 1000, 1500, 2000, 2500, 3000];
  for (const z of msZ) MS.push(N(`MS${z}`, 5400, z));
  edges.push({ nodes: MS, highway: 'primary', name: 'Main St', oneway: false, lanes: 4, maxspeed: '40 mph' });
  // EB exit at E5 -> Main St (z=200), diverging right (+z)
  N('R1', 5250, 60);
  edges.push({ nodes: ['E5', 'R1', 'MS200'], highway: 'motorway_link', name: 'Main St exit', oneway: true, lanes: 1 });
  // EB on-ramp from Main St (z=200) -> E6, merging from the right
  N('R2', 5600, 60);
  edges.push({ nodes: ['MS200', 'R2', 'E6'], highway: 'motorway_link', name: 'US 101 N on-ramp', oneway: true, lanes: 1 });
  // WB exit at W6 -> Main St (z=-200), diverging right of westbound (-z)
  N('R3', 5750, -60);
  edges.push({ nodes: ['W6', 'R3', 'MS-200'], highway: 'motorway_link', name: 'Main St exit', oneway: true, lanes: 1 });
  // WB on-ramp from Main St (z=-200) -> W5
  N('R4', 5200, -60);
  edges.push({ nodes: ['MS-200', 'R4', 'W5'], highway: 'motorway_link', name: 'US 101 S on-ramp', oneway: true, lanes: 1 });
  // CA 237 fork off EB at E12 heading south-east (right), then its own carriageway
  const C = ['E12', N('C1', 13000, 400), N('C2', 14000, 1200), N('C3', 15000, 2500), N('C4', 16000, 4000)];
  edges.push({ nodes: C, highway: 'motorway', ref: 'CA 237', oneway: true, lanes: 2, maxspeed: 65 });
  // Oak Ave: secondary two-way along x at z=3000 (shares MS3000 with Main St)
  const OA = [];
  for (const x of [0, 1000, 2000, 3000, 4000, 5000, 5400, 6000, 7000, 8000, 9000, 10000, 11000, 12000]) {
    OA.push(x === 5400 ? 'MS3000' : N(`OA${x}`, x, 3000));
  }
  edges.push({ nodes: OA, highway: 'secondary', name: 'Oak Ave', oneway: false, lanes: 2 });
  // Residential streets going south from Oak Ave
  const ELM = ['OA2000', N('ELM3500', 2000, 3500), N('ELM4000', 2000, 4000), N('ELM4500', 2000, 4500), N('ELM5000', 2000, 5000)];
  edges.push({ nodes: ELM, highway: 'residential', name: 'Elm St', oneway: false });
  const PINE = ['OA9000', N('PINE3500', 9000, 3500), N('PINE4000', 9000, 4000), N('PINE4500', 9000, 4500), N('PINE5000', 9000, 5000)];
  edges.push({ nodes: PINE, highway: 'residential', name: 'Pine St', oneway: false });
  // Bay Rd tertiary: CA 237 end -> Pine St end (two-way)
  edges.push({ nodes: ['C4', N('BR1', 12500, 4600), 'PINE5000'], highway: 'tertiary', name: 'Bay Rd', oneway: false });
  // Connector from Main St south end to Oak Ave is implicit (MS3000 shared). Also link Oak Ave west end to the WB freeway via a street.
  edges.push({ nodes: ['OA0', N('CON1', 0, 1500), 'W0'], highway: 'primary', name: 'First St', oneway: false });
  edges.push({ nodes: ['W0', 'E0'], highway: 'primary', name: 'First St', oneway: false });
  return { nodes, edges };
}

/** Big two-way grid: cols*rows nodes, spacing meters. Alternating classes. */
export function makeGridGraph(cols = 300, rows = 300, spacing = 100) {
  const nodes = [], edges = [];
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) nodes.push({ id: r * cols + c, x: c * spacing, z: r * spacing });
  for (let r = 0; r < rows; r++) {
    const ids = []; for (let c = 0; c < cols; c++) ids.push(r * cols + c);
    edges.push({ nodes: ids, highway: r % 10 === 0 ? 'primary' : 'residential', name: `Row ${r}`, oneway: false, lanes: 2 });
  }
  for (let c = 0; c < cols; c++) {
    const ids = []; for (let r = 0; r < rows; r++) ids.push(r * cols + c);
    edges.push({ nodes: ids, highway: c % 10 === 0 ? 'primary' : 'residential', name: `Col ${c}`, oneway: false, lanes: 2 });
  }
  return { nodes, edges };
}
