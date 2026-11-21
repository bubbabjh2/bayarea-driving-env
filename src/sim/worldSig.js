// Graph signature for cached populations / commuter tables (src/gpu/cache/*): a 32-bit FNV-1a hash
// of what those caches depend on -- node count, edge count, every edge's from / to node index and
// its length (cm). Node positions (x, y, z) and lane geometry are not included: moving a road
// sideways or re-sampling heights keeps a cache valid; renumbering nodes or changing lengths does not.
//   graphSig(graph)  graph = src/sim/roadGraph.js SimGraph (graph.edges[i].from/.to/.length)
export function graphSig(graph) {
  let h = 0x811c9dc5 >>> 0;
  const mix = (v) => { v = v >>> 0; for (let k = 0; k < 4; k++) { h ^= (v >>> (8 * k)) & 255; h = Math.imul(h, 16777619) >>> 0; } };
  mix(graph.nodeCount); mix(graph.edges.length);
  for (const e of graph.edges) { mix(e.from); mix(e.to); mix(Math.round(e.length * 100)); }
  return 'g1-' + h.toString(16).padStart(8, '0');
}
/** True when a cache meta is known to belong to another graph (sig present and different). */
export function sigMismatch(meta, graph) { return !!(meta && meta.graphSig && meta.graphSig !== graphSig(graph)); }
