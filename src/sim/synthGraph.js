// Synthetic "two shores + bridges" road network in the roads.json schema.
// Used by tests, the benchmark, and sim.html when the real data is missing.
//
// Layout (meters): two rectangular grids (west = "Peninsula", east = "East Bay")
// separated by a water gap, each with a N-S motorway spine, a primary arterial,
// and residential blocks; `bridges` motorway bridges connect the shores.

export function makeSynthGraph({
  blocks = 12,          // blocks along N-S axis per shore
  cross = 6,            // blocks E-W per shore
  block = 400,          // block size, meters
  gap = 3000,           // water width
  bridges = 2,
  seed = 1,
} = {}) {
  const nodes = [], edges = [];
  const id = (shore, i, j) => `${shore}_${i}_${j}`;
  const west = -gap / 2 - cross * block, east = gap / 2;
  const shoreX = { W: west, E: east };
  for (const shore of ['W', 'E']) {
    const x0 = shoreX[shore];
    for (let i = 0; i <= cross; i++) for (let j = 0; j <= blocks; j++) {
      nodes.push({ id: id(shore, i, j), x: x0 + i * block, z: j * block });
    }
    // N-S streets
    for (let i = 0; i <= cross; i++) {
      const col = [];
      for (let j = 0; j <= blocks; j++) col.push(id(shore, i, j));
      const highway = i === (shore === 'W' ? cross : 0) ? 'motorway' : i === Math.floor(cross / 2) ? 'primary' : 'residential';
      const lanes = highway === 'motorway' ? 3 : highway === 'primary' ? 2 : 1;
      const maxspeed = highway === 'motorway' ? '65 mph' : highway === 'primary' ? '40 mph' : '25 mph';
      if (highway === 'motorway') {
        // motorway: one two-way edge, `lanes` per direction (directions are offset laterally)
        edges.push({ nodes: col, highway, oneway: false, lanes, maxspeed, name: `${shore} Fwy` });
      } else {
        edges.push({ nodes: col, highway, oneway: false, lanes, maxspeed, name: `${shore} St ${i}` });
      }
    }
    // E-W streets
    for (let j = 0; j <= blocks; j++) {
      const row = [];
      for (let i = 0; i <= cross; i++) row.push(id(shore, i, j));
      const highway = j % 4 === 0 ? 'secondary' : 'residential';
      edges.push({ nodes: row, highway, oneway: false, lanes: highway === 'secondary' ? 2 : 1,
        maxspeed: highway === 'secondary' ? '35 mph' : '25 mph', name: `${shore} Ave ${j}` });
    }
  }
  // bridges: from west motorway column (i=cross) to east motorway column (i=0)
  for (let b = 0; b < bridges; b++) {
    const j = Math.round(((b + 1) * blocks) / (bridges + 1));
    const a = id('W', cross, j), c = id('E', 0, j);
    edges.push({ nodes: [a, c], highway: 'motorway', oneway: false, lanes: 3, maxspeed: '50 mph', bridge: true, layer: 1, name: `Bridge ${b}` });
  }
  return { nodes, edges, meta: { synthetic: true, seed, gap, block, blocks, cross } };
}

/** Side classifier for the synthetic graph: east shore if x > 0. */
export const synthSideOf = (x) => (x > 0 ? 1 : 0);
