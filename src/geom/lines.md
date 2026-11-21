# `src/geom/lines.js`: lane lines (paint + perception)

One data source for the painted lines. The renderers draw it, and the drivers perceive it.
Every line is a lane boundary from `lanes.js`: `offset = rightEdge(e, s) - k * laneWidth(e, s)`,
or sim `lat = k * laneWidth(e, s)`. A car on a lane centre (`LG.laneCenterLat`) is therefore
exactly half a lane width from both of its lines, in the sim's lateral coordinate.
`node tools/check_markings.mjs` verifies this.

The marking rules are in `docs/markings.md`.

## Files

| file | what | size |
|---|---|---|
| `data/lines.bin` + `data/lines.meta.json` | runs, marks, and the per-edge offset parameters | 24 MB |
| `data/lines-geom.bin` | reference-polyline mirror for the GPU: positions, mitred normals, s, heading, curvature | 48 MB |

Build with `node tools/lines/build_lines.mjs`. Run it after `tools/build_junctions.mjs`, because
it uses the junction mouths. Run `npm run tiles` after it. The meta documents every section and
the closed-form formulas.

## Loading

```js
import { loadLineSet, buildLineSet, LINE, COLOR, MARK } from './geom/lines.js';
// browser / sim (LG = ensureLaneGeometry(main, local); J = the junction pack, needed only for look-ahead through connectors)
const LS = await loadLineSet('data/lines.meta.json', LG, J);              // add { geom: true } to also load lines-geom.bin
// node: lineSetFromBuffers(meta, { 'lines.bin': arrayBuffer }, LG, J)    (tools/lines/load.mjs loads the world)
// in memory (no data file): buildLineSet(LG, J, { controls, legs: crosswalkLegs(...), points })
```

Edge ids are the merged-graph ids: main `e`, local `mainEdgeCount + e`. These are the same ids
as `src/sim` `fromRoadSet`, `src/geom`, and `data/junctions.bin`.

## Data model

### Boundaries

Per directed edge `e` with `n = lanes`, the boundaries are `k = 0 .. n`:
- `k = 0` is the right edge.
- `k = n` is the left edge. On a two-way road this is the centre line, which both directions share.
- Two pseudo-boundaries mark the outer pavement edge of a paved run-out wedge at a lane drop or
  add: `OUTER_R = 254` and `OUTER_L = 255`.

### Runs

Each run is `{ k, s0, s1, type, color, pattern, occluded }`:
- Runs are sorted by `k`, then by `s0`.
- No run lies inside a junction mouth. Lines stop at `sStop` and resume at `sExit`.
- `type` is one of `LINE.NONE | SOLID | DASHED | DOUBLE | DOTTED`.
- `color` is one of `COLOR.WHITE | YELLOW`.
- `pattern` indexes `PATTERNS`:
  - 1: freeway, 3.66 m / 10.97 m
  - 2: street, 3.05 / 9.14
  - 3: wide dotted, 0.91 / 2.74
- `occluded` is true where the boundary lies on another carriageway's lanes (drawn carriageways
  that overlap). There `type = NONE` and nothing is painted.

### Marks

Each mark is `{ kind: MARK.STOP | YIELD | CROSSWALK, span: INBOUND | ROAD, s0, s1 }`:
- `INBOUND` spans the edge's own lanes.
- `ROAD` spans the whole pavement, both directions.
- A stop bar's `s1` equals the sim's stop point: `J.mouth(e).sStop` at a junction, or the
  control's `stopAt` mid-block.

### Edge level

`level[e]` is one of:
- `LEVEL.FULL`: every boundary is painted.
- `LEVEL.CENTRE`: only the centre line is painted.
- `LEVEL.NONE`: unmarked residential. The lane geometry still exists; the lines have `type NONE`.

## API

```js
LS.runs(e)                 // [{k, s0, s1, type, color, pattern, occluded}]
LS.marks(e)                // [{kind, span, s0, s1}]
LS.lineAt(e, k, s, out)    // {type, color, pattern, occluded} of boundary k at s (NONE outside the runs)
LS.lineOffset(e, k, s)     // m right of the reference polyline (exact boundary; OUTER_* = run-out pavement edge)
LS.paintOffset(e, k, s)    // paint centre (edge lines inset 4.5 cm so the whole line is on the pavement)
LS.level[e]                // marking level

LS.lineGeom(pos, opts, out)   // perception, see below
emitPaint(LS, e, sink, opts)  // renderers: sink(role, V, T, tag); roles white | dash | yellow | stopbar | crosswalk
```

### `lineGeom`: what a driver sees

```js
// on an edge: sim coordinates (lat = m left of the right carriageway edge)
const g = LS.lineGeom({ edge: e, s, lat }, { route, routePos, ahead: [8, 16, 32, 64] });
// on a junction connector m (data/junctions.bin movement id), t m along it,
// lat m left of the connector's right lane edge (centre = width / 2)
const g = LS.lineGeom({ conn: m, t, lat }, { route, routePos });
```

**Fields at the car's position**

| field | meaning |
|---|---|
| `lane`, `lanes`, `laneWidth`, `kind` | the car's lane, the lane count, the lane width, and `'edge'` or `'conn'` |
| `right`, `left` | `{ dist, type, color }`. `dist` is metres from the car to the centre of its right or left line, and `right.dist + left.dist = laneWidth`. |
| `lines` | `[{ k, lat, type, color }]`: every boundary across the carriageway, so a driver can see two lanes over |
| `x`, `y`, `z`, `hx`, `hz` | lane-centre point and unit heading at the car's `s` |
| `curvature` | 1/m, positive when curving left |

**Look-ahead: `ahead[i]`**

`ahead[i]` describes the car's lane centre `d` metres ahead. It contains:
- `{ d, kind, edge, s, conn, t, lane, laneWidth }`
- `{ x, y, z, hx, hz, curvature }`
- `fwd` and `side`: metres forward and metres to the left, in the frame of the current lane tangent.
- `dHeading`: radians, positive to the left.
- `right` and `left`: `{ type, color }`.
- `laneEnds`: true when this lane does not continue (lane drop) or the route ends.

How the look-ahead follows the road:
- It follows `route[routePos + 1 ...]`. With no route it takes the straightest strict
  continuation, or else the straightest movement.
- Across a strict hop it keeps the lane through `hopMap`.
- At a junction hop it leaves the edge at the movement's `sIn`, follows `J.arcPoint` along the
  connector, and joins the out-edge at `sOut` on `outLane`. These are the same paths the sim drives.
- A lane with no movement into the route's next edge uses the nearest lane that has one.

On connectors the lines are virtual lane edges with `type NONE`, because there is no paint in a
junction box. The width blends from the entry lane width to the exit lane width
(`LS.mW[2m]`, `LS.mW[2m + 1]`).

**Cost.** One call takes about 7 µs with the 4 look-ahead points (node, M4 Max). Pass
`ahead: []` for the cheap current-point query. Pass a reused `out` object to avoid
allocations. `lineAt` is a short linear scan of the edge's runs, at most about 15.

### GPU mirror (`lines.bin` + `lines-geom.bin`, flat typed arrays)

| quantity | how to compute it |
|---|---|
| line type at (e, k, s) | Scan `runInfo[runFirst[e] .. runFirst[e + 1])` for `(info & 255) == k` with `runS[2r] <= s <= runS[2r + 1]`. Type is `(info >> 8) & 15`, colour is `(info >> 12) & 15`. |
| `laneWidth(e, s)` and `rightEdge(e, s)` | Use the smoothstep formulas in `lines.meta.json.formulas`, with `laneW`, `laneWEnd`, `Rd`, `Rend`, `tau`, `length`. |
| line offset | `rightEdge(e, s) - k * laneWidth(e, s)`. Its world point is `lerp(v_i, v_i+1) + off * lerp(m_i, m_i+1)`, where `i` is found by binary search of `vs` in `[vFirst[e], vFirst[e + 1])`. |
| heading and curvature of the reference line | `vHead` and `vCurv`, interpolated. Curvature at offset `off` is `kappa / (1 + off * kappa)`. |
| connectors | Points are in `junctions.bin` `pts`; widths are `mW`. |
