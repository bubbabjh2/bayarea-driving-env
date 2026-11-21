// A* shortest-TIME router over a Graph (src/nav/graph.js).
//   cost(edge) = length / speed   (seconds)
//   h(node)    = euclid(node, goal) / graph.maxSpeed   (admissible)
// Uses typed-array CSR adjacency + a binary heap; a 100k-node Bay Area graph
// routes in tens of milliseconds. Also provides nearestNode(x, z) through a
// uniform spatial grid.
import { MinHeap } from './heap.js';

export class Router {
  /**
   * @param {import('./graph.js').Graph} graph
   * @param {object} [opts]  {cellSize: grid cell size in meters (default 250)}
   */
  constructor(graph, opts = {}) {
    this.graph = graph;
    const n = graph.nodes.length, m = graph.edges.length;
    this.n = n;
    this.nx = new Float64Array(n);
    this.nz = new Float64Array(n);
    for (let i = 0; i < n; i++) { this.nx[i] = graph.nodes[i].x; this.nz[i] = graph.nodes[i].z; }
    // CSR out-adjacency
    this.outStart = new Int32Array(n + 1);
    this.outEdge = new Int32Array(m);
    let k = 0;
    for (let i = 0; i < n; i++) {
      this.outStart[i] = k;
      for (const e of graph.out[i]) this.outEdge[k++] = e;
    }
    this.outStart[n] = k;
    this.eTo = new Int32Array(m);
    this.eCost = new Float64Array(m);
    for (let e = 0; e < m; e++) {
      const ed = graph.edges[e];
      this.eTo[e] = ed.to;
      this.eCost[e] = ed.length / ed.speed;
    }
    this.invMaxSpeed = 1 / Math.max(graph.maxSpeed, 1e-6);
    // search state
    this.g = new Float64Array(n);
    this.cameFrom = new Int32Array(n);
    this.closed = new Uint8Array(n);
    this.heap = new MinHeap(4096);
    this.stats = { expanded: 0, ms: 0 };
    this._buildGrid(opts.cellSize ?? 250);
  }

  _buildGrid(cell) {
    const b = this.graph.bounds;
    this.cell = cell;
    this.gx0 = isFinite(b.minX) ? b.minX : 0;
    this.gz0 = isFinite(b.minZ) ? b.minZ : 0;
    this.gw = isFinite(b.maxX) ? Math.max(1, Math.floor((b.maxX - this.gx0) / cell) + 1) : 1;
    this.gh = isFinite(b.maxZ) ? Math.max(1, Math.floor((b.maxZ - this.gz0) / cell) + 1) : 1;
    const counts = new Int32Array(this.gw * this.gh + 1);
    const cellOf = (i) => {
      const cx = Math.floor((this.nx[i] - this.gx0) / cell), cz = Math.floor((this.nz[i] - this.gz0) / cell);
      return cz * this.gw + cx;
    };
    for (let i = 0; i < this.n; i++) counts[cellOf(i) + 1]++;
    for (let c = 0; c < this.gw * this.gh; c++) counts[c + 1] += counts[c];
    this.gridStart = counts;
    this.gridNodes = new Int32Array(this.n);
    const fill = counts.slice();
    for (let i = 0; i < this.n; i++) this.gridNodes[fill[cellOf(i)]++] = i;
  }

  /**
   * Nearest graph node to (x, z). By default only nodes with an outgoing edge
   * (so the result can be a route origin). `scc: true` snaps to the largest
   * strongly connected component (every such node can reach every other one:
   * use it for trip origins/destinations so no trip targets a one-way stub).
   * Returns node index or -1.
   */
  nearestNode(x, z, { requireOut = true, requireIn = false, maxDist = Infinity, filter = null, scc = false } = {}) {
    const inMain = scc ? this.largestSCC().inLargest : null;
    const cx = Math.floor((x - this.gx0) / this.cell), cz = Math.floor((z - this.gz0) / this.cell);
    let best = -1, bestD2 = maxDist * maxDist;
    const maxRing = Math.max(this.gw, this.gh);
    for (let r = 0; r <= maxRing; r++) {
      // once the ring's inner distance exceeds best, stop
      if (best >= 0 && (r - 1) * this.cell > Math.sqrt(bestD2)) break;
      for (let dz = -r; dz <= r; dz++) {
        const zz = cz + dz; if (zz < 0 || zz >= this.gh) continue;
        for (let dx = -r; dx <= r; dx++) {
          if (Math.abs(dx) !== r && Math.abs(dz) !== r) continue; // ring only
          const xx = cx + dx; if (xx < 0 || xx >= this.gw) continue;
          const c = zz * this.gw + xx;
          for (let k = this.gridStart[c]; k < this.gridStart[c + 1]; k++) {
            const i = this.gridNodes[k];
            if (requireOut && this.outStart[i + 1] === this.outStart[i]) continue;
            if (requireIn && this.graph.in[i].length === 0) continue;
            if (inMain && !inMain[i]) continue;
            if (filter && !filter(i)) continue;
            const ex = this.nx[i] - x, ez = this.nz[i] - z;
            const d2 = ex * ex + ez * ez;
            if (d2 < bestD2) { bestD2 = d2; best = i; }
          }
        }
      }
    }
    return best;
  }

  /**
   * Shortest-time route between node indices.
   * @returns {{edges:number[], nodes:number[], length:number, time:number}|null}
   */
  route(from, to, { penalty = null } = {}) {
    const t0 = performance.now();
    // penalty: Map(edgeId -> extra seconds) for perceived blockages (wreck, queue that does not move)
    const n = this.n;
    if (from < 0 || to < 0 || from >= n || to >= n) return null;
    const g = this.g, came = this.cameFrom, closed = this.closed, heap = this.heap;
    g.fill(Infinity); came.fill(-1); closed.fill(0); heap.clear();
    const nx = this.nx, nz = this.nz, tx = nx[to], tz = nz[to], inv = this.invMaxSpeed;
    const h = (i) => { const dx = nx[i] - tx, dz = nz[i] - tz; return Math.sqrt(dx * dx + dz * dz) * inv; };
    g[from] = 0;
    heap.push(h(from), from);
    let expanded = 0, found = false;
    while (heap.size > 0) {
      const u = heap.pop();
      if (closed[u]) continue;
      if (u === to) { found = true; break; }
      closed[u] = 1;
      expanded++;
      const gu = g[u];
      for (let k = this.outStart[u], ke = this.outStart[u + 1]; k < ke; k++) {
        const e = this.outEdge[k];
        const v = this.eTo[e];
        if (closed[v]) continue;
        const ng = gu + this.eCost[e] + (penalty ? (penalty.get(e) || 0) : 0);
        if (ng < g[v]) { g[v] = ng; came[v] = e; heap.push(ng + h(v), v); }
      }
    }
    this.stats.expanded = expanded;
    this.stats.ms = performance.now() - t0;
    if (!found) return null;
    const edges = [];
    let v = to;
    while (v !== from) { const e = came[v]; edges.push(e); v = this.graph.edges[e].from; }
    edges.reverse();
    return this._result(edges, from);
  }

  _result(edges, from) {
    const nodes = [from];
    let length = 0, time = 0;
    for (const e of edges) { const ed = this.graph.edges[e]; nodes.push(ed.to); length += ed.length; time += this.eCost[e]; }
    return { edges, nodes, length, time };
  }

  /**
   * Re-route around perceived blockages, as nav apps do. route = current edge ids, pos = index of the
   * edge the car is on, blocked = edge ids to avoid (their cost is raised by `penaltySec`, so they are
   * still used when there is no alternative). The new route starts with the car's CURRENT edge (it
   * cannot leave it mid-block) and continues from that edge's end node to the old goal.
   * Returns {edges, changed, timeOld, timeNew} or null when no route exists.
   */
  rerouteAround(route, pos, blocked, { penaltySec = 900 } = {}) {
    const g = this.graph, cur = route[pos], goal = g.edges[route[route.length - 1]].to;
    // live traffic (sim v8 sim.trafficPenalty: extra seconds per slow edge, ~2 min moving average), like a nav app
    const live = typeof this.livePenalty === 'function' ? this.livePenalty() : this.livePenalty;
    const penalty = new Map(live || []); for (const e of blocked) penalty.set(e, (penalty.get(e) || 0) + penaltySec);
    const from = g.edges[cur].to;
    const r = this.route(from, goal, { penalty });
    if (!r) return null;
    let timeOld = 0; for (let k = pos + 1; k < route.length; k++) timeOld += this.eCost[route[k]] + (penalty.get(route[k]) || 0);
    let timeNew = 0; for (const e of r.edges) timeNew += this.eCost[e] + (penalty.get(e) || 0);
    const edges = [cur, ...r.edges];
    let changed = edges.length !== route.length - pos; if (!changed) for (let k = 0; k < edges.length; k++) if (edges[k] !== route[pos + k]) { changed = true; break; }
    return { edges, changed, timeOld, timeNew };
  }

  /** Route through a list of node indices in order (waypoints). */
  routeVia(nodeList) {
    const all = [];
    for (let i = 0; i + 1 < nodeList.length; i++) {
      const r = this.route(nodeList[i], nodeList[i + 1]);
      if (!r) return null;
      for (const e of r.edges) all.push(e);
    }
    return this._result(all, nodeList[0]);
  }

  /**
   * Strongly connected components (iterative Tarjan over the CSR, computed once, ~O(n+m)).
   * @returns {{comp: Int32Array, count: number, largest: number, size: number, inLargest: Uint8Array}}
   */
  largestSCC() {
    if (this._scc) return this._scc;
    const n = this.n, os = this.outStart, oe = this.outEdge, to = this.eTo;
    const index = new Int32Array(n).fill(-1), low = new Int32Array(n), comp = new Int32Array(n).fill(-1);
    const onStack = new Uint8Array(n), stack = new Int32Array(n), callV = new Int32Array(n), callK = new Int32Array(n);
    let idx = 0, sp = 0, count = 0;
    for (let root = 0; root < n; root++) {
      if (index[root] >= 0) continue;
      let cp = 0;
      callV[0] = root; callK[0] = os[root]; index[root] = low[root] = idx++; stack[sp++] = root; onStack[root] = 1;
      while (cp >= 0) {
        const v = callV[cp];
        if (callK[cp] < os[v + 1]) {
          const w = to[oe[callK[cp]++]];
          if (index[w] < 0) { cp++; callV[cp] = w; callK[cp] = os[w]; index[w] = low[w] = idx++; stack[sp++] = w; onStack[w] = 1; }
          else if (onStack[w] && index[w] < low[v]) low[v] = index[w];
        } else {
          if (low[v] === index[v]) { let w; do { w = stack[--sp]; onStack[w] = 0; comp[w] = count; } while (w !== v); count++; }
          cp--;
          if (cp >= 0) { const u = callV[cp]; if (low[v] < low[u]) low[u] = low[v]; }
        }
      }
    }
    const sizes = new Int32Array(count);
    for (let i = 0; i < n; i++) sizes[comp[i]]++;
    let largest = 0; for (let c = 1; c < count; c++) if (sizes[c] > sizes[largest]) largest = c;
    const inLargest = new Uint8Array(n);
    for (let i = 0; i < n; i++) inLargest[i] = comp[i] === largest ? 1 : 0;
    return (this._scc = { comp, count, largest, size: sizes[largest], inLargest });
  }

  /** Convenience: route between world positions via nearestNode (snapped to the largest SCC). */
  routeXZ(x0, z0, x1, z1) {
    const a = this.nearestNode(x0, z0, { requireOut: true, scc: true });
    const b = this.nearestNode(x1, z1, { requireOut: false, requireIn: true, scc: true });
    if (a < 0 || b < 0) return null;
    return this.route(a, b);
  }
}
