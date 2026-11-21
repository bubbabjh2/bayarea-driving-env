# `src/geom/`: the shared lane geometry

This module is the one definition of where every lane is. The renderer (`src/roads.js`: ribbons
and markings) uses it now. The sim's `edgePoint` must switch to it, and the GPU kernels mirror it
(the data is all flat typed arrays). If two consumers disagree about a lane position, one of them
is not calling this module.

```js
import { graphFromRoadSets, buildLaneGeometry, hopMap, mapLane, ensureLaneGeometry } from './geom/index.js';
const G  = graphFromRoadSets(main, local);   // edge ids: main e, local mainEdgeCount + e (= sim fromRoadSet)
const LG = buildLaneGeometry(G);             // ~0.8 s for the whole corridor (291k edges)
// or: const LG = ensureLaneGeometry(main, local)   (cached on the main RoadSet; roads.js uses this)

LG.latPoint(e, s, lat, out)    // sim coordinates: s = 0..edgeLength[e], lat = m from the RIGHT carriageway edge
LG.lanePoint(e, s, lane, out)  // centre of lane `lane` (0 = rightmost) = latPoint(e, s, (lane + .5) * 3.7)
LG.offsetPoint(e, s, off, 0, out) // `off` m to the right of the reference polyline
LG.rightEdge(e, s)             // offset of the right carriageway edge (lat 0)
LG.bounds(e, s)                // {right, left}: drawn pavement edges incl. lane add/drop tapers
// out = {x, y, z, hx, hz}: world position (y = deck/ground elevation) and unit heading of the lane line
hopMap(G, e, en)               // {strict, shift}: lane l of e continues as lane l - shift of en
mapLane(G, e, en, l)
```

## Lane widths (per edge, not 3.7 m everywhere)

Widths come from the data section `edgeLaneWidth` (Float32 per edge, also `laneWidth` in the
JSON): OSM `width` / lanes where tagged and plausible (2.7–4.2 m), otherwise the class default:

| class | m | class | m |
|---|---|---|---|
| motorway, motorway_link, trunk | 3.7 | trunk_link, primary | 3.6 |
| primary_link, secondary | 3.5 | secondary_link | 3.4 |
| tertiary, tertiary_link | 3.3 | unclassified, residential | 3.1 |
| living_street | 3.0 | | |

At a strict node, every edge end in the group uses one width, the anchor edge's. Each edge
blends to its own width over the same transition as its right edge, so lanes stay continuous
across a hop between roads of different widths.

**Never hardcode 3.7. Use these:**
```js
LG.laneW[e]                    // the edge's own lane width
LG.laneWidth(e, s)             // width at s (blended near strict nodes)
LG.laneCenterLat(e, s, l)      // lat of lane l's centre = (l + 0.5) * laneWidth(e, s)
LG.latToLane(e, s, lat)        // lane index containing lat
LG.dividerOffset(e, s, k)      // offset of the line between lane k-1 and k (0 = right edge, lanes = left edge)
LG.bounds(e, s).laneWidth      // same value, with the pavement / lane edges
```
`lanePoint(e, s, l)` already uses the width. A strict hop maps `lat' = lat − shift ×
laneWidth(at the node)`, and both edges have the same width at the node.

## Model

1. **Reference line.** This is the OSM polyline of the edge (`data/roads*.bin`). Vertices that turn more
   than 11.5° get a fillet: up to 12 m is cut back on each side, the corner is replaced by a
   quadratic curve, and no resampled vertex turns more than about 8.6°. Edge endpoints (graph
   nodes) never move. `s` stays the data coordinate: the resampled length is rescaled to
   `edgeLength[e]`.
2. **Normals.** Interior vertices use the unit-vector bisector with mitre length `1/cos(θ/2)`
   (audit D10). An offset point is `lerp(V_i, V_i+1, t) + off · lerp(M_i, M_i+1, t)`, so lane lines
   are continuous through every vertex (measured max jump 0.0015 m over 1.86 M samples).
3. **Lane mapping across a node** is in `hops.js` (`hopMap`). Strict hops are:
   - plain continuations;
   - every freeway-system hop (motorway / trunk / their links) at a pure merge (one way on) or
     pure diverge (one feeder), at any drawn angle: ramps are gores, never junction boxes;
   - freeway-system hops under 1.2 rad at nodes with several ins and outs (at-grade expressway
     intersections stay junctions). Lanes are counted from the right. On a plain
   continuation the left/median lanes are continuous, and lanes are added or dropped on the
   right (`shift = e.lanes - en.lanes`). At splits and merges, the edges are ordered right to left
   by geometry: the lateral offset of the point 50 m into the edge, in the frame of the edge being
   continued, with ties under 1 m broken by turn angle. Each edge takes a consecutive block of
   lanes from the right. `tools/build_roads.py compute_hops()` is the Python mirror, and the
   build's `bannedHops` are computed with this rule.
4. **Node frames.** All edge ends joined by strict hops at a node share one frame: the bisector
   of the straightest strict pair. Every lane line of every edge in that group meets at the same
   point with the same normal.
5. **Right-edge anchoring.** Each edge has a default right edge `Rd`:
   - two-way road: the divider sits on the polyline, so `Rd = lanes · 3.7`;
   - one-way carriageway: centred, so `Rd = lanes · 3.7 / 2`.

   At strict nodes the ends are solved so that `R(en, 0) = R(e, L) − shift · 3.7` for every
   strict hop. This is exactly the sim's `mapLat` rule `lat' = lat − shift · 3.7`, so a car's
   world position does not move at a hop. Away from the node the edge blends back to `Rd` with
   a smoothstep over `τ = max(60 m, 30 × |ΔR|)`, capped at half the edge. Lanes added or dropped
   at a node therefore appear on the right of a one-way carriageway (or on the outer side of a
   two-way road), and on-ramps join at the mainline's right edge (gore taper).
6. **Lane ends.**
   - Every sim lane is paved all the way to its node, so a lane centre is never off the pavement.
   - A lane that ends at a node continues as a *paved run-out* on the straightest strict
     successor. The run-out narrows to zero over `min(90 m, edge length)` and is painted with
     chevrons. A starting lane gets the mirror widening on the predecessor. The pavement edge
     is therefore continuous at every plain continuation.
   - `bounds(e, s)` returns the drawn pavement `{right, left}` plus the sim-lane edges
     `{laneRight, laneLeft}`.
   - `laneEnds(e)` returns `{endRight, endLeft, startRight, startLeft, gore}`. This is the sim
     hook: a car in an ending lane must have merged by the node. Measure `laneEndDist` to
     `s = edgeLength`, and treat lat outside `[laneLeft, laneRight]` as off the road.
7. **Ramps beside a carriageway.** A ramp drawn alongside a same-direction freeway carriageway
   within 4.5 m of height is shifted sideways until its pavement clears that carriageway by
   0.3 m (at most 4 lanes).
   - This includes the carriageway it merges into or diverges from, except within the last
     30 m (the gore).
   - The ramp's own lane-anchoring transition is at most 30 m, so a ramp stays beside the
     mainline until the gore.
   - `check.mjs` counts ramp lane centres inside the mainline pavement 30–250 m from the merge
     or diverge: 177 pairs / 6.3 km before this change, 39 pairs / 0.6 km after. The rest are
     ramps drawn across or under the mainline.
   - `node tools/check_overlaps.mjs` counts overlaps with carriageways the ramp is not joined
     to: 183 pairs / 7.6 km before, 52 / 0.8 km after.
8. **Medians.** One-way motorway and trunk carriageways are shifted right where the opposing
   carriageway's line is within 50 m to the left. The shift keeps each median edge at least
   0.3 m short of the midline, so drawn carriageways never overlap. `Rd` grows; lanes keep
   their width, and the sim drives the shifted lanes.
9. **Opposite carriageways** (2026-10-24, `docs/carriageway-fix.md`): every one-way carriageway (any class) keeps its
   inner pavement edge 0.5 m short of the midline to an opposite-direction one-way beside it (inner edges >= 1.0 m
   apart), with tapers at junction ends, group moves at strict one-way ends, and the median nose at ends pinned to a
   two-way road. Expressed in `Rd` / `Rend` / `tau` (no format change). `opts.opposing = false` / `GEOM_NO_OPPOSING=1`
   gives the old geometry.
10. **Elevation** is the polyline's `y` (deck heights, approach ramps and tunnels come from
   `tools/build_roads.py`; grades ≤ 5 % freeway, 6 % links/arterials, 8 % local).

## Checks (`node src/geom/check.mjs [dataDir]`)

Results on the current data:

| check | result |
|---|---|
| strict-hop lane continuations | 44,775 checked; 1 jumps (3.7 m) at the one remaining anchor conflict below |
| vertex continuity | 1,864,124 lane samples, max 0.0015 m |
| inverted pavement | 0 |
| tapers on edges shorter than 60 m | 1,053 (the taper is limited to the edge) |

The one remaining jump is at node 45147 (US-101 near 17997, 19164). Two coincident 4-lane
carriageways leave that node with the same heading, and an on-ramp merges at the same node, so
the clamped merge and diverge shifts cannot be satisfied by one frame
(`LG.stats.anchorConflicts`). Two-way to two-way strict hops are divider-aligned
(`shift = e.lanes − en.lanes`), which removed the other conflicts.

## GPU mirror

Everything a kernel needs is flat:
- `vFirst` (Uint32, per edge + 1), `vx/vy/vz/vs` (vertex position and data-s), `mx/mz` (scaled normals);
- `Rd`, `Rend` (2 per edge: start, end), `tau` (2 per edge);
- `tapR`, `tapL` (2 per edge, render only).

`latPoint` is: binary search `vs`, lerp position and normal, then `off = rightEdge(e, s) − lat`.

## Requests to other owners

- **Sim (`src/sim/roadGraph.js`):** make `edgePoint(e, s, laneOffset)` call `LG.latPoint` (or
  `offsetPoint` with `rightEdge`), import `hopMap`/`mapLane` from `src/geom/hops.js` (geometry
  diverge order), read `bannedHops` from the main pack, and let the lane-end observation see
  across edges (flag `LANE_DROP` marks edges whose right lane ends at their end node).
- **Film (`src/film/traffic.js`, `gpuTraffic.js`):** `CLASS_STYLE[c].y` is now a ≤ 3 cm draw
  epsilon, so `surf(e)` is effectively the true elevation. Carry the look-ahead lane through
  `mapLane` (D13). Call `ensureLaneGeometry(main, local)` before building meshes; `loadRoadPack`
  links the tiers automatically when main is loaded first.

## Junctions (`junctions.js`)

Physical intersections replace "a junction is a point". A node is a **junction** when at least
one in → out pair through it is a junction hop (not strict).

```js
import { buildJunctions, loadJunctions } from './geom/index.js';
const J = buildJunctions(LG);                                    // ~3 s, in memory (preferred), or
const J = await loadJunctions('data/junctions.meta.json', LG);   // precomputed by tools/build_junctions.mjs

J.mouth(e)            // { sStop, sExit }: m along e
J.movement(a, la, b)  // connector id for (in-edge a, in-lane la) -> out-edge b, or -1
J.describe(m)         // { inEdge, inLane, outEdge, outLane, preferred, sIn, sOut, length, turn }
J.arcPoint(m, t, out) // {x, y, z, hx, hz} at t = 0 .. length metres along the connector
J.conflictsOf(m)      // [{ other, type, s, sOther }]: type 0 crossing, 1 merge (exits within a car width)
J.movementsAt(node), J.junctionAt(node), J.hull (box polygons), J.gore(e)
J.arrays()            // flat typed arrays for the GPU (same layout as data/junctions.bin)
```

### Mouths
`J.mouth(e)` gives two points on every edge:
- `sStop`: the stop / yield point at the mouth of the junction at the end of `e`. It sits back from
  the node by half the widest other road + 1 m (the curb line), **+ 3.5 m where the leg has a
  crosswalk**, and is capped at 0.45 × length. A leg has a crosswalk when:
  - it is an approach of a signalised junction or an all-way stop (`data/controls.json`), or
  - a painted or signalised OSM crossing node (`data/crosswalks.json`) lies on it within 25 m of
    the junction.

  `crosswalkLegs(LG, controls, crosswalkPoints)` returns the sets that
  `buildJunctions(LG, sets)` takes. Building without them gives mouths without crosswalk
  setbacks, which will not match the file or the controls.
- `sExit`: where `e` resumes after the box of the junction at its start.

Both apply to every approach, controlled or not; `data/controls.*` uses the same `sStop`.
**A car's front bumper stops at `sStop`, so its centre waits at `sStop − halfLength`.**
Where the node is not a junction, `sExit = 0` and `sStop = length`.

### Connectors (one per in-lane of every junction movement)
- **Geometry.** Each connector leaves the in-edge at `sIn = sStop` on lane `inLane`, and joins the
  out-edge at `sOut = sExit` on lane `outLane`. Its end points and end tangents are the lane
  points and headings from `latPoint`.
- **Lane mapping.** `outLane = floor((inLane + 0.5) · outLanes / inLanes)`, the same as the sim's
  junction `mapLane`. `preferred` marks right turns from the rightmost lane, left turns from the
  leftmost lane, and every lane for straight movements.
- **Curve shape: minimum radius 6 m** (a sedan turns on about 5.5 m).
  - **Corner turns:** a fillet — straight, circular arc, straight. The arc radius is
    `min(tangent lengths) / tan(turn / 2)`, capped at 40 m.
  - **When the corner is too tight for 6 m:** the movement's entry point `sIn` moves back along the
    in-edge and its exit point `sOut` moves forward along the out-edge, until the arc fits. Each
    moves at most 40 m, and never past 5 % / 95 % of the edge. So **`sIn` and `sOut` are per
    movement** (from `J.describe(m)`) and can differ from the approach's `sStop` / `sExit`.
  - **Gentle S-offsets:** a cubic Bezier, stretched the same way if needed.
  - **Hairpins / no corner:** arc–straight–arc with a 6 m radius.
- **Chains on short edges.** A connector into an edge never ends after the next connector out of
  that edge begins: `sOut(previous) ≤ sIn(next)` on every edge. Where the radius extensions
  overlapped, both are capped at a split point between the mouths.
- **Tangent ends.** The first and last chords follow the lane tangents: a lead-in / lead-out of up
  to 0.5 m.
- **Sampling.** Arcs every 0.12 rad, straights every ≤ 6 m.
- **Checks** (`check.mjs`, connector radius by the circle through consecutive samples):

  | check | result |
  |---|---|
  | connectors below 6 m (most between 5.5 and 6 m) | 12,906 of 605,217 |
  | below 5.5 m (edges too short to move the entry / exit) | ~1,600 |
  | steps over 30° | 0 |
  | connector ends vs lane points | ≤ 3 mm |

- **Sim hook.** On a junction hop, a car leaves `e` at the movement's `sIn`, follows
  `arcPoint(m, t)` for `t = 0 .. length`, and enters `en` at the movement's `sOut` with
  `lat = laneCenterLat(en, sOut, outLane)`. There is no position or heading jump. A car that must
  stop for the junction stops at `min(sStop, sIn)`.

### Conflict matrix
For each pair of connectors at a junction:
- **type 1, merge:** their exits are within 3 m, which includes the same exit lane.
- **type 0, crossing:** their centre paths cross, or come within 3 m of each other anywhere
  (segment-to-segment distance; a truck is about 2.6 m wide).

`s` / `sOther` are the metres along each connector where they meet. Pairs that share an entry
lane are not listed: they are an ordinary queue. Totals:

| item | count |
|---|---|
| junctions | 85,414 |
| connectors | 605,217 |
| crossing conflicts | 619,936 |
| merge conflicts | 443,097 |

`J.hull` holds the junction box: the convex hull of all mouth pavement corners.

### Ramp gores
`J.gore(e)` applies to an on-ramp whose lanes continue on the mainline's lanes, i.e. a merge with
no added lane. Ramps with a synthesized or OSM acceleration lane already run in their own lane
and get -1.
- **Merge side.** The side the ramp joins from comes from the ramp's far end.
  - Right-side merges test ramp lane 0 against the mainline's right lane edge.
  - Left-side merges (including ramp-to-ramp merges) test the ramp's leftmost lane against the
    mainline's left lane edge.
- **Wait point.** The value is the farthest point, up to 400 m back, where that ramp lane's
  centre lies outside the mainline lane edge by half a lane + 1.3 m. **A ramp car that must
  wait waits there, outside the mainline lanes.**
- **No clear point.** 29 of 846 ramps never reach that clearance: 18 are drawn alongside the
  mainline the whole way. They use the point of largest clearance instead (`J.goreFallback`
  lists them).

Counts: 846 ramps. The wait point is back from the merge node by p10 8 m, p50 24 m, p90 88 m.

### Data file
`data/junctions.bin` + `data/junctions.meta.json` (147 MB) give the same arrays with the layout
documented in the meta. Edge ids are stable; the node indices in `jNode` are merged-graph
indices and change with every rebuild. **Build order** (enforced by `tools/verify_roads.py`, which fails on any mismatch):
1. `build_roads`
2. `build_stations`
3. `build_controls` (it decides the control types)
4. `node tools/build_junctions.mjs`
   - reads the controls and `data/crosswalks.json` for crosswalk legs;
   - writes the junctions;
   - rewrites every junction control's `approachStopAt` to `sStop`.
5. `build_crossings`
6. `npm run tiles`

## Lane lines (`lines.js`, 2026-10-24)

The painted lines are one data set, shared by the renderers (tiles, `src/roads.js`) and the
drivers' perception (`LS.lineGeom`). Every line is a lane boundary of this module:
`dividerOffset(e, s, k)`, with sim `lat = k * laneWidth(e, s)`.

- Data: `data/lines.bin` and `data/lines-geom.bin`, built by `node tools/lines/build_lines.mjs`.
  It runs after `build_junctions` and before `npm run tiles`.
- API: `src/geom/lines.md`.
- Spec and audit: `docs/markings.md`.
- Checks: `node tools/check_markings.mjs`.
