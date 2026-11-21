// Adapter between a brain driver (src/brain) and the simulation internals. The
// brain's eyes call raycast(origin, dir, maxDist) and carsNear(x, z, r); nothing in
// the brain touches the SoA arrays. One WorldQuery is shared by all vehicles: the
// sim calls bind(slot) before each driver.step(). Objects returned by self()/
// leader()/raycast() are reused between calls; copy them if you keep them.

function routeLen(g, r) { if (r._len == null) { let L = 0; for (let k = 0; k < r.length; k++) L += g.edges[r[k]].length; r._len = L; } return r._len; }

export class WorldQuery {
  constructor(sim) {
    this.sim = sim;
    this.slot = -1;
    this._self = {};
    this._ray = { hit: false, dist: 0, id: -1, x: 0, z: 0 };
    this._leader = { id: -1, gap: 0, speed: 0 };
  }
  bind(slot) { this.slot = slot; return this; }
  get time() { return this.sim.time; }
  get graph() { return this.sim.graph.nav || this.sim.graph; }

  /**
   * raycast(origin, dir, maxDist): origin/dir are {x,z} or [x,z] (dir need not be
   * unit). Returns {hit, dist, id, x, z} where id is the vehicle slot hit (-1 if none).
   */
  raycast(origin, dir, maxDist = 100) {
    const ox = origin.x ?? origin[0], oz = origin.z ?? origin[1];
    let dx = dir.x ?? dir[0], dz = dir.z ?? dir[1];
    const L = Math.hypot(dx, dz) || 1; dx /= L; dz /= L;
    const r = this.sim.hash.raycast(ox, oz, dx, dz, maxDist, this.slot, this._rayTmp || (this._rayTmp = { hit: false, dist: 0, slot: -1 }));
    const out = this._ray;
    out.hit = r.hit; out.dist = r.hit ? r.dist : maxDist; out.id = r.slot;
    out.x = ox + dx * out.dist; out.z = oz + dz * out.dist;
    return out;
  }

  /** Vehicles within r of (x,z): array of {id,x,z,heading,speed,length,width,crashed}. Allocates. */
  carsNear(x, z, r) {
    const vs = this.sim.vs, out = [], me = this.slot;
    this.sim.hash.forEachNear(x, z, r, (i) => {
      if (i === me) return;
      out.push({ id: i, x: vs.x[i], z: vs.z[i], heading: vs.heading[i], speed: vs.speed[i], length: vs.length[i], width: vs.width[i], crashed: vs.state[i] === 2 });
    });
    return out;
  }
  /** Non-allocating variant: fn(slot) for each vehicle within r. */
  forEachNear(x, z, r, fn) { const me = this.slot; return this.sim.hash.forEachNear(x, z, r, (i) => { if (i !== me) fn(i); }); }

  /** Pose/state of the bound vehicle. */
  self() {
    const vs = this.sim.vs, i = this.slot, g = this.sim.graph, e = g.edges[vs.edge[i]];
    const o = this._self;
    o.id = i; o.x = vs.x[i]; o.y = vs.y[i]; o.z = vs.z[i]; o.heading = vs.heading[i];
    o.speed = vs.speed[i]; o.accel = vs.accel[i]; o.lane = vs.lane[i]; o.lanes = e.lanes;
    o.edge = vs.edge[i]; o.s = vs.s[i]; o.edgeLength = e.length; o.distToNode = e.length - vs.s[i];
    o.speedLimit = e.speedLimit; o.length = vs.length[i]; o.width = vs.width[i];
    o.maxSpeed = vs.maxSpeed[i]; o.maxAccel = vs.maxAccel[i]; o.maxDecel = vs.maxDecel[i];
    const route = vs.routes[i], rp = vs.routePos[i];
    o.routePos = rp; o.routeLength = route.length;
    o.nextEdge = rp + 1 < route.length ? route[rp + 1] : -1;
    o.nextSpeedLimit = o.nextEdge >= 0 ? g.edges[o.nextEdge].speedLimit : 0;
    o.lastEdge = rp + 1 >= route.length;
    const c = this.sim.commuters[vs.commuter[i]];
    o.timeToWorkStart = c ? c.workStart - this.sim.time : 0;
    o.remainingRouteDistance = c ? Math.max(0, (vs.leg[i] === 0 ? c.L : c.routeBack ? routeLen(g, c.routeBack) : c.L) - vs.odometer[i]) : 0;
    o.reactionTime = vs.reactionTime[i];
    return o;
  }
  /** Car ahead in the current lane: {id, gap, speed} or null. */
  leader() {
    const vs = this.sim.vs, i = this.slot, L = vs.leader[i];
    if (L < 0) return null;
    const o = this._leader; o.id = L; o.gap = vs.gap[i]; o.speed = vs.speed[L]; return o;
  }
  /** Whether the junction ahead requires yielding right now. */
  mustYield() {
    const vs = this.sim.vs, i = this.slot, e = this.sim.graph.edges[vs.edge[i]];
    return this.sim.ctx.yieldRequired(i, e, e.length - vs.s[i]);
  }
  /** The route (Int32Array of directed edge indices) of the bound vehicle. */
  route() { return this.sim.vs.routes[this.slot]; }
  /** Edge metadata by index (nav edge object if available). */
  edgeInfo(idx) { const e = this.sim.graph.edges[idx]; return e.meta || e; }
}
