// Load a population saved by src/gpu/tools/packPopulation.mjs into the struct-of-arrays
// shape GPUSim.setPopulation expects ({routes: {flat, offset}, arrays, deltas}).
export const ARRAY_FIELDS = ['home', 'work', 'kind', 'vehicleType', 'workStart', 'depart', 'returnDepart', 'expectedTravel', 'routeLength', 'intended', 'slack', 'reactionTime', 'fuelFill', 'crossesBay'];
const TYPES = { Int32Array, Uint32Array, Float32Array, Uint8Array, Uint16Array, Float64Array };

export async function loadPopulationPack(baseUrl) {
  const meta = await (await fetch(`${baseUrl}.json`)).json();
  const buf = await (await fetch(`${baseUrl}.bin`)).arrayBuffer();
  const sec = (k) => { const s = meta.sections[k]; return s ? new TYPES[s.type](buf, s.byteOffset, s.length) : null; };
  const arrays = {};
  for (const k of ARRAY_FIELDS) { const a = sec(k); if (a) arrays[k] = a; }
  return { meta, routes: { flat: sec('routesFlat'), offset: sec('routesOffset') }, deltas: sec('deltas'), arrays, stats: meta.stats };
}
