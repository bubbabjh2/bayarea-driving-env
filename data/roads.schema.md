# Road data schema

Two tiers, each in two forms. All produced by `tools/build_roads.py` from
OpenStreetMap (ODbL) for the commute corridor bbox lat 37.25–37.97, lon
−122.56 to −121.80, minus the Tri‑Valley (lat > 37.60 and lon > −121.97) and
Lamorinda (lat > 37.85 and lon > −122.25). The north edge was moved from 37.85
to 37.97 (2026‑10‑19) so the Richmond–San Rafael Bridge, I‑580/I‑80 in
Richmond and the Marin end of the Golden Gate are in the graph; the format did
not change (same projection origin, sections and fields), only the counts.

| tier | classes | binary (load this) | readable copy |
|---|---|---|---|
| main | motorway, motorway_link, trunk, trunk_link, primary, primary_link, secondary, secondary_link | `roads.bin` + `roads.meta.json` | `roads.json` |
| local | tertiary, tertiary_link, unclassified, residential, living_street | `roads-local.bin` + `roads-local.meta.json` | `roads-local.json` |

Edge `i` in the binary is the same edge as `edges[i]` in the JSON of the same
tier, and node `n` is `nodes[n]`. The JSON additionally carries OSM ids,
names and refs; the binary carries everything a sim/router/GPU needs.

## Coordinates

Local metres, equirectangular about (37.55 N, 122.15 W):

```
x = (lon - lon0) * metersPerDegLon     +x = EAST
z = -(lat - lat0) * metersPerDegLat    +z = SOUTH  (north = -z)
y = elevation (up), metres NAVD88; model "terrain/1" since 2026-10-24 (docs/elevation.md): REAL terrain.
    Built by tools/elevation/build_heights.mjs FROM the structures/2 build below (kept in data/flat-backup/):
    - at-grade roads = USGS 3DEP bare earth (1 m lidar in SF, 1/3" elsewhere), robustly smoothed along the road;
      vertical shape points (synthetic nodes, negative JSON ids, appended AFTER all existing nodes) where needed, so
      existing node indices, edge ids, edge lengths (horizontal) and s coordinates are unchanged;
    - bridges / elevated ways span straight between their ground ends; water decks keep >= their old height over the
      water (Bay Bridge upper deck 7 m over the lower); tunnels interpolate between portals, >= 3 m (old depth) under
      the ground; plan crossings that had >= 4.5 m keep >= 6 m (structure lifted / tunnel lowered with 5-8 % ramps);
    - per-segment plausibility limits (DEM spikes): freeway 8 %, links 12 %, primary 20 %, secondary 25 %,
      tertiary 32 %, local 42 %; structures 8-12 %. meta.elevation = {model: 'terrain/1', previous: <structures/2>, ...}.
Previous model "structures/2" (2026-10-19, audit D2; flat ground at y = 0):
    - a structure = connected bridge / layer>0 ways (any class); its deck is level:
      7 m per layer (layer 1 -> 7 m, 2 -> 14 m, ...), >= 20 m (+7 m per extra layer) when the
      structure spans more than 150 m of water (Bay Bridge: lower deck 20 m, upper 27 m);
    - ramps are on the at-grade approach roads, not inside the bridge: y falls away from a
      deck at 5 % (motorway/trunk), 6 % (links, primary..tertiary) or 8 % (local); where two
      ramps meet the higher wins (max), so decks never dip at junctions;
    - tunnels (not part of a raised structure) sit 7 m below grade with mirror-image ramps;
    - an untagged crossing of two roads without a shared node (freeway/link involved, or an
      arterial bridge) lifts the lower-ranked road to >= 7 m over the other with the same ramps;
      the build fails below 4.5 m, except crossings of roads connected within 300 m
      (merge/diverge overlaps, toll plazas), which are reported;
    - grades never exceed 8 % on any class (build check);
    - only structures longer than 400 m that span more than 150 m of water get the 20 m deck;
      short bridges over water (High St, Park St) are 7 m;
    - sags between nearby raised sections on the freeway system (motorway, trunk, links)
      are filled: where the profile dips >= 3 m below raised points on both sides within
      600 m, it stays at the lower of the two raised levels;
    - surface junctions (>= 3 roads, none a structure, no freeway leg) are held at grade,
      unless holding them would need a grade above 8 % (then they ride the ramp);
    - arterial sags between two structures are filled only when no at-grade junction lies
      between the two raised points (otherwise the road really comes down to that junction).
```

Looking straight down with north at the top of the screen, x grows to the
right and z grows down the screen, i.e. (x, z) is an ordinary map. Exact
constants are in `meta.projection`. Right-hand traffic: for a travel direction
`(dx, dz)` the unit perpendicular pointing to the driver's **right** is
`(-dz, dx)`.

## Graph model

- Ways are **split at every junction** (node shared with another way, across
  both tiers) and consecutive pieces with identical attributes are **merged**
  across degree-2 nodes, so every edge runs junction-to-junction with a packed
  polyline. Shape points are Douglas–Peucker simplified (2 m for
  motorway/trunk, 4 m otherwise); junction nodes are never removed.
- Edges are **directed**. A two-way road is two edges that are each other's
  `twin`; a one-way road has no twin. Lane counts are per direction.
- Interior polyline nodes belong to exactly one physical road; junctions are
  the `from`/`to` nodes.
- **Synthetic nodes** have **negative `id`s** in the JSON: ramp shape points (inserted by
  distance along the edge, every <= 25 m where the profile is not flat, plus crossing lift
  points) and the split points of synthesized acceleration lanes. Shape points are interior
  only; acceleration-lane split points are degree-2 junctions. They are never in `linkNodes`
  and never station nodes. Build checks: no 180-degree fold-back in any polyline, and no edge
  longer than 1.05 x its polyline before the insertions.

## Binary pack (`roads.bin`, described by `roads.meta.json`)

One little-endian byte blob; `meta.sections[name] = {byteOffset, byteLength,
type, length, stride, count}`. Every section starts 4-byte aligned, so each
can be viewed as a typed array on the same `ArrayBuffer` and uploaded to a GPU
storage buffer as-is.

| section | type | per item | meaning |
|---|---|---|---|
| `nodeXYZ` | Float32 ×3 | node | `x, y, z` metres (y = elevation) |
| `edges` | Uint32 ×8 | edge | see fields below |
| `poly` | Uint32 | — | node indices of all edge polylines, concatenated, in travel order |
| `edgeLength` | Float32 | edge | centreline length, metres |
| `csrOffsets` | Uint32 | node+1 | CSR row offsets into `csrEdges` |
| `csrEdges` | Uint32 | edge | outgoing edge ids: edges leaving node `n` are `csrEdges[csrOffsets[n] .. csrOffsets[n+1])` |
| `edgeLaneWidth` | Float32 | edge | **new**: lane width (m) of this edge. It is OSM `width` / lanes where tagged and plausible (2.7–4.2 m), else the class default (`meta.laneWidthByClass`). Use `src/geom` `laneWidth(e, s)`, not 3.7 |
| `linkNodes` | Uint32 ×2 | shared node | **local tier only**: `(localNodeIndex, mainNodeIndex)` for every node present in both tiers (same OSM node) |
| `bannedHops` | Uint32 ×3 | banned path | **main tier only** (new, optional): `(inEdge, viaEdge, outEdge)` triples that need k lane changes inside `viaEdge` with `viaEdge` shorter than 120 m × k under the shared lane rule (`src/geom/hops.js`). Routers should not route in → via → out. Paths that are the only way through (toll plazas) are not banned. JSON: top-level `bannedHops: [[in, via, out], ...]` |

Edge record (stride 8, field order in `meta.edgeFields`):

| # | field | meaning |
|---|---|---|
| 0 | `firstNode` | offset into `poly` of this edge's first node |
| 1 | `nodeCount` | number of polyline nodes (≥ 2). `from = poly[firstNode]`, `to = poly[firstNode + nodeCount - 1]` |
| 2 | `lanesFwd` | lanes in this edge's travel direction (≥ 1) |
| 3 | `lanesBack` | lanes in the opposite direction (0 if one-way) |
| 4 | `classId` | index into `meta.classesAll` (`["motorway","motorway_link","trunk","trunk_link","primary","primary_link","secondary","secondary_link","tertiary","tertiary_link","unclassified","residential","living_street"]`) |
| 5 | `speedCmps` | speed limit in **cm/s** (÷100 → m/s). OSM maxspeed if tagged, else class default (flag `SPEED_DEFAULT`) |
| 6 | `flags` | `ONEWAY=1, BRIDGE=2, TUNNEL=4, LANES_FALLBACK=8, SPEED_DEFAULT=16, LINK=32, LANES_CLAMPED=64, LANES_SMOOTHED=128, AUX_LANE=256, LANE_DROP=512` (see Lane counts) |
| 7 | `twin` | edge id of the opposite direction of the same road, or `0xFFFFFFFF` |

Lane geometry: use **`src/geom/`** (`buildLaneGeometry`, `latPoint`, `lanePoint`), the one
shared definition (fillets, mitres, node frames, divider-anchored two-way roads, right-side lane
adds and drops, tapers; see `src/geom/README.md`). Lane width `meta.laneWidth` = 3.7 m. The old
closed form `((lanesBack - lanesFwd)/2 + k + 0.5) * 3.7` (k = 0 leftmost) is only the
straight-segment approximation for a road with lanesFwd = lanesBack. It is kept as
`laneOffsets` in the JSON for compatibility.

Routing: run Dijkstra/A* over edges using `csrOffsets`/`csrEdges` from
`to(e)`; cost `edgeLength[e] / (speedCmps[e]/100)`.

`src/roadpack.js` (`loadRoadPack(metaUrl)`) returns a `RoadSet` with these
arrays plus accessors (`edgeNodes(e)`, `outEdges(n)`, `pointAlong(e, s)`,
`laneOffsets(e)`, …). `roadSetFromJSON(doc)` builds the same object from the
JSON copy.

## JSON copy (`roads.json`, format `bay-roads/2`)

```jsonc
{
  "meta": { "format", "tier", "classes", "classIds", "projection", "laneWidth",
            "laneDefaults", "defaultSpeedKph", "elevation", "bbox", "nodeCount", "edgeCount" },
  "nodes": [ { "id": <osm node id>, "x": 0.0, "y": 0.0, "z": 0.0 }, ... ],
  "edges": [ {
      "id": 1024, "from": 312, "to": 318, "nodes": [312, 9001, 9002, 318],
      "class": "motorway", "name": "Nimitz Freeway", "ref": "I 880",
      "oneway": true, "twin": -1,
      "lanes": 4, "oppLanes": 0, "laneOffsets": [-5.55, -1.85, 1.85, 5.55], "width": 14.8,
      "lanesFallback": false, "bridge": false, "tunnel": false, "layer": 0,
      "maxspeed": 104.6,            // km/h or null (then use meta.defaultSpeedKph[class])
      "length": 812.3,              // metres
      "way": 123456                 // one of the source OSM way ids
  }, ... ]
}
```

`nodes[].id` is the OSM node id and is the join key between tiers in the JSON
form (the binary form has `linkNodes` instead).

### Lane counts (model "through lanes", 2026-10-19, audit D4/D12/D15)

`lanesFwd` / `lanesBack` are **through lanes**:
1. The raw count comes from OSM `lanes`, `lanes:forward|backward` and `turn:lanes*`.
   `lanes:both_ways` (a centre turn lane) is not a through lane.
2. On non-freeway roads, `turn:lanes` parts that are turn-only (`left`, `right`, `slight_*`,
   `sharp_*`, `merge_to_*`, `reverse`) are removed.
3. Counts are capped per class and direction: motorway 6, motorway_link 3, trunk 4,
   trunk_link 2, primary 4, primary_link 2, secondary 3, secondary_link 2, local 2,
   living_street 1. `LANES_CLAMPED` is set when the cap applied.
4. Along carriageway chains (straightest continuations with the same name/ref and road
   category), a run shorter than 400 m (freeway), 300 m (links/arterials) or 200 m (local) is
   replaced by its neighbours' count. This applies when the run is bounded by equal counts on
   both sides, or when it is a spike or dip between them. `LANES_SMOOTHED` is set on changed
   edges. Edges are merged on the smoothed counts.
5. Lane conservation on the one-way freeway system:
   - an on-ramp that joins without an added lane gets a synthesized acceleration lane (+1 for
     300 m, then the lane ends), or an auxiliary lane through to the next exit when that exit
     is within 700 m (`AUX_LANE`);
   - a lane drop with no exit moves downstream to the next exit within 2 km, else it stays and
     the edge gets `LANE_DROP`.

The build fails on any merge or drop that none of these rules explains.

Per-edge OSM `sidewalk` / `parking` (JSON only, relative to the edge's travel direction):
`"both" | "left" | "right" | "no"`, or `null` when untagged. They come from `sidewalk[:both|left|right]`
and `parking:lane:*` / `parking:*`. The raw OSM counts are
in the JSON as `lanesRaw` / `oppLanesRaw`, and the edge's lane flags as `laneFlags`.

Raw lane parsing: lane counts come from OSM `lanes`, `lanes:forward`, `lanes:backward`,
`turn:lanes[:forward|:backward]` (N `|`-separated parts = N lanes). If only a
total `lanes` exists on a two-way road it is split evenly (odd totals drop the
centre lane; min 1 per direction). With no lane tags at all the class default
applies (`meta.laneDefaults`; `lanesFallback` / `LANES_FALLBACK` is set):
one-way motorway 4, trunk 3, primary 2, secondary 2, links/local 1; two-way
per direction motorway 2, trunk 2, everything else 1. `oneway=-1` ways are
stored reversed so travel is always `from -> to`; roundabouts are one-way.

## Gas stations (`stations.json`, `stations.bin` + `stations.meta.json`)

Synthetic (not real locations), generated by `tools/build_stations.py`
(seeded, deterministic). ~250 stations on **junction nodes** of primary /
secondary / tertiary roads, never on a node touching a motorway, motorway_link,
trunk or trunk_link; extra density within 800 m of a motorway_link node
(freeway interchanges); at least 1.5 km apart. `roads.meta.json.stations`
names the meta file.

```jsonc
{ "meta": { "format": "bay-stations/1", "count": 250, "tierIds": {"main": 0, "local": 1}, ... },
  "stations": [ { "id": 0, "tier": "main", "node": 1234, "osmNode": 65432109,
                  "x": -1234.5, "z": 987.6, "y": 0.0, "pumps": 8, "nearInterchange": true }, ... ] }
```

`node` is an index into the node table of the tier's pack (`roads.bin` for
`main`, `roads-local.bin` for `local`); `x, y, z` are copied from that node so
the station can be placed without resolving it. Binary: `stationXYZ`
Float32×3 and `stationInfo` Uint32×4 `[nodeIndex, tierId, pumps, flags]`
(`flags & 1` = near interchange), same section convention as `roads.bin`.

## Water (`data/water.json`, format `bay-water/1`)

`{"meta":{bbox, projection}, "polygons":[{"outer":[[x,z],...], "holes":[[[x,z],...],...]}]}`
Same metres. Outer rings are water, holes are islands. Decoration only.

## Traffic control (`controls.json`, `controls.bin` + `controls.meta.json`)

Real OSM signals, stop signs, yield signs and ramp meters, placed on the graph by
`tools/build_controls.py` (queries `tools/queries/controls-signals.ql`,
`controls-signs.ql`). Rerun it after any rebuild of the road packs. Every record keeps OSM ids,
so the controls can be re-matched without the old indices.

**Edge ids are merged-graph ids**: a main edge `e` keeps its id, and local edge `e` becomes
`meta.mainEdgeCount + e`. This is the same numbering as `src/sim/roadGraph.js fromRoadSet()`
and `src/geom/`.

```jsonc
{ "meta": { "format": "bay-controls/1", "count", "types": ["signal","stop","give_way","ramp_meter","crossing_signal"],
            "byType": {...}, "mainEdgeCount", "build": {per-rule counts} },
  "controls": [ {
    "id": 0, "type": "signal",            // signal | stop | give_way | ramp_meter
    "tier": "main", "node": 1234,         // controlled junction (node index in that tier), -1 = mid-block / ramp meter
    "junctionOsm": 65432109,              // OSM id of that junction (rebuild-proof key), null if none
    "osmNodes": [110430737, ...],         // every OSM control node merged into this control
    "x": 0, "y": 0, "z": 0,               // junction (or control node) position
    "allWay": false,                      // all-way stop (stop=all, or every approach has a sign)
    "rampMeter": false, "midBlock": false, "crossing": false,   // crossing = mid-block pedestrian signal
    "approaches": [ { "edge": 20787,      // merged-graph edge id of the controlled approach
                      "tier": "main", "tierEdge": 20787,
                      "stopAt": 91.2,     // stop line, metres along that edge: at the junction mouth (node minus half the widest crossing road + 1.5 m)
                      "fromOsm": 1, "toOsm": 2, "class": "primary" } ] } ] }
```

Binary sections (same convention as `roads.bin`):

| section | type | per item | meaning |
|---|---|---|---|
| `controlInfo` | Uint32 ×6 | control | `[type, flags, tierId, nodeIndex, firstApproach, approachCount]`; type = index into `meta.types`; flags `ALL_WAY=1, RAMP_METER=2, MIDBLOCK=4`; tierId 0 main / 1 local / 0xFFFFFFFF none; nodeIndex 0xFFFFFFFF if none |
| `controlXYZ` | Float32 ×3 | control | position |
| `approachEdge` | Uint32 | approach | merged-graph edge id |
| `approachStopAt` | Float32 | approach | stop line, metres along the edge |

Placement rules:
- **Control on a junction node.**
  - Signals: all incoming edges of the junction (both tiers).
  - `stop=all`, or a junction whose approaches all have the same road class: all approaches, `allWay`.
  - `stop=minor` or an untagged sign: the approaches of lower class.
- **Control along a road.** It controls the approach toward the junction it faces (within 60 m
  for signals, 40 m for signs), with the stop line at the node.
  - `direction` / `traffic_signals:direction` = forward / backward is relative to the OSM way.
  - Without a direction tag, a sign faces the nearer junction and a signal faces both directions.
- **Signalised junctions.** A junction is signalised when there is a signal on the node, on
  ≥ 2 legs, or a real (non-crosswalk) signal on any leg. Every non-link incoming leg ≥ 30 m
  (and every signalled leg) gets the signal. Slip-lane links get a `give_way` record at the
  same junction. A lone crosswalk signal on one leg becomes a mid-block crossing signal.
- **All-way stops.** All-way is inferred only from `stop=all`, or from signs on ≥ 3 legs.
  - A single untagged sign on a junction of equal roads stops the legs off the through road
    (the straightest pair keeps priority).
  - A sign on the major leg facing minor roads is treated as a direction error and moved to
    the minor legs.
  - No signal or sign is put on a motorway mainline, and ramp meters only go on ramps.
- **Stop lines** sit at the junction mouth, not at the node.
- **Merging.** Signals farther than 60 m from a junction become mid-block controls (node -1).
  Pedestrian crossing signals within 40 m of a signalised junction are merged into it. A stop
  or yield sign at a signalised junction is dropped.
- **Ramp meters.** `traffic_signals=ramp_meter`, or a signal on a ramp that touches no surface
  street, control their link edge at the node.

## Grade separations (`crossings.json`, from `tools/build_crossings.py`)

`crossings` lists every place where two roads cross in plan with no shared node and at least
one of them is on the freeway system: `{a, aTwin, sa, aClass, b, bTwin, sb, bClass, x, z, ya, yb,
dy, reason}`.
- Ids are merged-graph edge ids; a twin's coordinate is `s = length − sa`.
- `dy` is the vertical separation at the crossing point.
- `reason` is set when `dy < 4.5 m`: `merge/diverge overlap`, `junction geometry` (the roads are
  joined within 300 m), `tunnel` or `at-grade (no OSM junction)`.

`overlaps` lists freeway pavements that overlap another road's pavement in plan within 4.5 m
of height, with no junction within 300 m: `{a, b, aClass, bClass, x, z, gap, dy}`. These are
frontage roads drawn too close.

Sim contact rule: vehicles on two edges that share no node can touch only if the pair is in
`crossings` with a small `dy`, or in `overlaps`. Rerun the script after any rebuild.

## Crosswalks (`crosswalks.json`, from `tools/build_crosswalks.py`)

`{"meta", "points": [[x, z, kind], ...]}` lists OSM `highway=crossing` nodes that are painted or
signalised: kind 0 = marked / zebra / uncontrolled, 1 = traffic_signals. Unmarked crossings are
left out. `tools/build_junctions.mjs` uses them, with the signal and all-way-stop legs from
`controls.json`, to set junction mouths 3.5 m further back (behind the crosswalk). It then
writes those stop lines into `controls.*` (`approachStopAt` = `junctions` `sStop`).
