// A* routing over the SimGraph directed edges. Returns an Int32Array of edge
// indices from startNode to goalNode (or null). Cost = travel time at the speed
// limit plus a small per-junction penalty so routes prefer bigger roads.

class MinHeap {
  constructor() { this.keys = []; this.vals = []; }
  get size() { return this.keys.length; }
  push(k, v) {
    const keys = this.keys, vals = this.vals;
    let i = keys.length; keys.push(k); vals.push(v);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (keys[p] <= k) break;
      keys[i] = keys[p]; vals[i] = vals[p]; i = p;
    }
    keys[i] = k; vals[i] = v;
  }
  pop() {
    const keys = this.keys, vals = this.vals;
    const top = vals[0];
    const lk = keys.pop(), lv = vals.pop();
    const n = keys.length;
    if (n > 0) {
      let i = 0;
      while (true) {
        const l = 2 * i + 1, r = l + 1;
        let m = i, mk = lk;
        if (l < n && keys[l] < mk) { m = l; mk = keys[l]; }
        if (r < n && keys[r] < mk) { m = r; mk = keys[r]; }
        if (m === i) break;
        keys[i] = keys[m]; vals[i] = vals[m]; i = m;
      }
      keys[i] = lk; vals[i] = lv;
    }
    return top;
  }
}

/**
 * Compact routing graph: everything makeRouter needs as typed arrays (no edge objects, no node-id maps), for the
 * route Web Workers (src/gpu/streamPack.js RouteWorkers). The full world graph costs a worker ~650 MB of JS heap
 * (495k nodes); six of them plus the page overflowed the renderer's shared 4 GB V8 heap cage at the start of the
 * 1.38M full-scale run (docs/scale-crashes.md). makeRouter(pack) gives identical routes to makeRouter(graph).
 * Transfer list: routingPackBuffers(pack).
 */
export function routingPack(graph) {
  const E = graph.edges.length, eFrom = new Int32Array(E), eTo = new Int32Array(E), eTime = new Float64Array(E), eRank = new Int32Array(E), eLen = new Float64Array(E), eSpeed = new Float64Array(E);
  let maxSpeed = graph.maxSpeed || 0;
  for (let i = 0; i < E; i++) { const e = graph.edges[i]; eFrom[i] = e.from; eTo[i] = e.to; eTime[i] = e.length / e.speedLimit; eRank[i] = e.rank; eLen[i] = e.length; eSpeed[i] = e.speedLimit; if (!graph.maxSpeed && e.speedLimit > maxSpeed) maxSpeed = e.speedLimit; }
  let banned = null;
  if (graph.bannedHops && graph.bannedHops.size) {
    const keys = new Float64Array(graph.bannedHops.size), off = new Int32Array(graph.bannedHops.size + 1), vals = [];
    let k = 0; for (const [key, list] of graph.bannedHops) { keys[k] = key; off[k] = vals.length; for (const x of list) vals.push(x); k++; }
    off[k] = vals.length; banned = { keys, off, vals: Int32Array.from(vals) };
  }
  return { routingPack: 1, nodeCount: graph.nodeCount, nx: Float64Array.from(graph.nx), nz: Float64Array.from(graph.nz), outStart: Int32Array.from(graph.outStart), outEdges: Int32Array.from(graph.outEdges),
    eFrom, eTo, eTime, eRank, eLen, eSpeed, maxSpeed, banned };
}
/** A routing pack plus light edge objects {idx, from, to, length, speedLimit, rank} (~15 MB of heap), enough for
 *  src/sim/fuel.js StationNetwork / src/nav/stations.js in a worker (GPU fuel workers). */
export function liteGraph(p) {
  const E = p.eFrom.length, edges = new Array(E);
  for (let i = 0; i < E; i++) edges[i] = { idx: i, from: p.eFrom[i], to: p.eTo[i], length: p.eLen[i], speedLimit: p.eSpeed[i], rank: p.eRank[i] };
  return { ...p, edges, edgeCount: E };
}
export function routingPackBuffers(p) {
  const l = [p.nx, p.nz, p.outStart, p.outEdges, p.eFrom, p.eTo, p.eTime, p.eRank, p.eLen, p.eSpeed].map((a) => a.buffer);
  if (p.banned) l.push(p.banned.keys.buffer, p.banned.off.buffer, p.banned.vals.buffer);
  return l;
}

export function makeRouter(graph, opts = {}) {
  const { nodeCount: N, nx, nz, outStart, outEdges } = graph;
  // per-edge arrays: from a routingPack, else built from the graph's edge objects (same values)
  let eFrom = graph.eFrom, eTo = graph.eTo, eTime = graph.eTime, eRank = graph.eRank;
  if (!graph.routingPack) {
    const edges = graph.edges, E = edges.length; eFrom = new Int32Array(E); eTo = new Int32Array(E); eTime = new Float64Array(E); eRank = new Int32Array(E);
    for (let i = 0; i < E; i++) { const e = edges[i]; eFrom[i] = e.from; eTo[i] = e.to; eTime[i] = e.length / e.speedLimit; eRank[i] = e.rank; }
  }
  const gScore = new Float64Array(N);
  const cameFrom = new Int32Array(N); // edge index used to reach node
  const stamp = new Int32Array(N);     // visited generation
  const closed = new Uint8Array(N);
  let gen = 0;
  let maxSpeed = opts.maxSpeed || graph.maxSpeed || 0; if (!maxSpeed && graph.edges) { for (const e of graph.edges) if (e.speedLimit > maxSpeed) maxSpeed = e.speedLimit; } maxSpeed = Math.max(1, maxSpeed);
  const junctionPenalty = opts.junctionPenalty ?? 2; // seconds
  const weight = opts.weight ?? 1.0; // >1 = weighted A* (faster, slightly suboptimal)
  let bh = graph.bannedHops || null;
  if (graph.routingPack && graph.banned) { const b = graph.banned; bh = new Map(); for (let k = 0; k < b.keys.length; k++) bh.set(b.keys[k], Array.from(b.vals.subarray(b.off[k], b.off[k + 1]))); }
  const cache = new Map();
  const cacheMax = opts.cacheMax ?? 50000;

  /**
   * avoid (optional Set of edge ids, e.g. closed by a crash scene): never used. penalty (optional
   * Map edge -> extra seconds, live traffic like a nav app). Either bypasses the cache.
   */
  function route(start, goal, avoid = null, penalty = null) {
    if (start === goal) return new Int32Array(0);
    const key = start * N + goal;
    const hit = avoid || penalty ? undefined : cache.get(key);
    if (hit !== undefined) return hit;
    gen++;
    if (gen > 2e9) { stamp.fill(0); gen = 1; }
    const heap = new MinHeap();
    const h = (n) => weight * Math.hypot(nx[n] - nx[goal], nz[n] - nz[goal]) / maxSpeed;
    gScore[start] = 0; stamp[start] = gen; cameFrom[start] = -1; closed[start] = 0;
    heap.push(h(start), start);
    let found = false;
    while (heap.size) {
      const n = heap.pop();
      if (closed[n] === 1 && stamp[n] === gen) continue;
      closed[n] = 1;
      if (n === goal) { found = true; break; }
      const gn = gScore[n];
      // banned lane paths (data roads.meta bannedHops): in -> via -> out with via = the edge into n
      let banned = null;
      if (bh && n !== start) { const via = cameFrom[n], pre = eFrom[via], inn = pre !== start && stamp[pre] === gen ? cameFrom[pre] : -1; if (inn >= 0) banned = bh.get(inn * 4194304 + via) || null; }
      for (let i = outStart[n]; i < outStart[n + 1]; i++) {
        const ei = outEdges[i];
        if (banned && banned.includes(ei)) continue;
        if (avoid && avoid.has(ei)) continue;
        const m = eTo[ei];
        let tentative = gn + eTime[ei] + junctionPenalty / (1 + eRank[ei]);
        if (penalty) { const p = penalty.get(ei); if (p) tentative += p; }
        if (stamp[m] !== gen || tentative < gScore[m]) {
          stamp[m] = gen; closed[m] = 0;
          gScore[m] = tentative; cameFrom[m] = ei;
          heap.push(tentative + h(m), m);
        }
      }
    }
    let result = null;
    if (found) {
      const path = [];
      let n = goal;
      while (n !== start) { const ei = cameFrom[n]; path.push(ei); n = eFrom[ei]; }
      path.reverse();
      result = Int32Array.from(path);
    }
    if (avoid || penalty) return result;
    if (cache.size >= cacheMax) cache.clear();
    cache.set(key, result);
    return result;
  }
  route.cache = cache;
  return route;
}

/** Convert a node path (array of node indices) to an edge path, or null if disconnected. */
export function nodePathToEdges(graph, nodePath) {
  const out = new Int32Array(Math.max(0, nodePath.length - 1));
  for (let i = 0; i + 1 < nodePath.length; i++) {
    const ei = graph.edgeBetween(nodePath[i], nodePath[i + 1]);
    if (ei < 0) return null;
    out[i] = ei;
  }
  return out;
}

/**
 * Largest strongly connected component of the directed graph (iterative Kosaraju).
 * Returns a Uint8Array(nodeCount) with 1 for nodes inside it. Cached on the graph.
 * On the real two-tier Bay Area graph it holds ~98% of the nodes that have edges;
 * homes, workplaces and gas stations are drawn from it so every trip is routable
 * both ways (outbound and return).
 */
export function largestSCC(graph) {
  if (graph._scc) return graph._scc;
  const N = graph.nodeCount, { outStart, outEdges, inStart, inEdges, edges } = graph;
  const order = new Int32Array(N); let on = 0;
  const vis = new Uint8Array(N);
  const stackN = new Int32Array(N), stackI = new Int32Array(N);
  for (let s = 0; s < N; s++) {
    if (vis[s]) continue;
    let sp = 0; stackN[0] = s; stackI[0] = outStart[s]; vis[s] = 1;
    while (sp >= 0) {
      const n = stackN[sp];
      if (stackI[sp] < outStart[n + 1]) {
        const m = edges[outEdges[stackI[sp]++]].to;
        if (!vis[m]) { vis[m] = 1; sp++; stackN[sp] = m; stackI[sp] = outStart[m]; }
      } else { order[on++] = n; sp--; }
    }
  }
  const comp = new Int32Array(N).fill(-1);
  let nc = 0, best = -1, bestSize = 0;
  const st = new Int32Array(N);
  for (let k = on - 1; k >= 0; k--) {
    const s = order[k]; if (comp[s] >= 0) continue;
    let sp = 0, size = 0; st[0] = s; comp[s] = nc;
    while (sp >= 0) {
      const n = st[sp--]; size++;
      for (let i = inStart[n]; i < inStart[n + 1]; i++) { const m = edges[inEdges[i]].from; if (comp[m] < 0) { comp[m] = nc; st[++sp] = m; } }
    }
    if (size > bestSize) { bestSize = size; best = nc; }
    nc++;
  }
  const out = new Uint8Array(N);
  for (let i = 0; i < N; i++) if (comp[i] === best) out[i] = 1;
  graph._scc = out; graph._sccSize = bestSize;
  return out;
}
