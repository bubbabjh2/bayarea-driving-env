# Synthetic training worlds (`src/scenarios/`)

Small synthetic road networks (straight roads, curves, merges, junctions, grids, interchanges, ...) for
pre-training and diagnosing policies before they meet the real Bay Area. They are written in
**exactly the real world's data format** (`data/roads.schema.md`) and loaded by the **sim's own loader**,
so the CPU and GPU sims run them with no code changes. The only extra input is a data
directory (`loadWorld({ base })`) and a trip list (`scenario.json`) in place of the SPEC commute population.


```
node tools/scenarios/gen.mjs all --seed 1                    # one scenario per family -> data/scenarios/<family>-s1/
node tools/scenarios/gen.mjs 'junction:control=signal,movement=left' --seed 4
node tools/scenarios/gen.mjs pack --count 64 --seed 3 --name pack-mixed-s3          # a world pack (64 tiles)
node --test test/scenarios.test.js
```

## Modules

| file | what |
|---|---|
| `builder.js` | `World`: author nodes (x, z, y) and roads (class, lanes each way, speed, lane width, bridge / layer), controls, trips, obstacles; `finish()` splits roads at shared nodes into directed edges + twins and packs both tiers exactly like `tools/build_roads.py` output (sections `nodeXYZ, edges, poly, edgeLength, csrOffsets, csrEdges, edgeLaneWidth, linkNodes / bannedHops`, same meta keys copied from `data/roads*.meta.json`, JSON copies `bay-roads/2`), plus `stations.*`, `controls.*` (`bay-controls/1`), `crosswalks.json`, `crossings.json` (every plan crossing with the freeway system, with `dy`). Flags: `ONEWAY, BRIDGE, SPEED_DEFAULT, LINK, AUX_LANE`, and `LANE_DROP` computed on plain continuations into fewer lanes. |
| `write.js` | writes the directory, then runs the world's own **`tools/build_junctions.mjs <dir>`** (lane geometry, connectors, conflict matrix, hulls, gores, crosswalk set-backs; it moves the junction controls' stop lines to the junction mouths). |
| `families.js` | the scenario library (`FAMILIES`, `makeScenario(family, seed, overrides)`), randomised by seed. |
| `pack.js` | `packWorld(worlds)`: shelf-packs many scenarios into ONE disjoint graph (≥ 400 m apart, never connected); `mixedPack`, `parseSpec('junction:control=signal')`. |
| `load.js` | `loadScenario(dir)` = `loadWorld({ base: dir })` + `scenario.json`; `scenarioPopulation` (trips -> the sim's population SoA via `population.js assemble`: routes from the sim's A*, deltas / reaction times / vehicle types drawn like the SPEC generator, full tank, SPEC slack); `placeScenarioObstacles`, `attachScenario(env)` (re-places obstacles after `env.reset()`); `scenarioReport(sim, sw)` (per tile / per family outcome + contacts by crash type). |
| `index.js` | public API + **`scenarioLevel(spec, {seed})`** for the curriculum (below). |

Coordinates are local metres around (0, 0) (same projection meta as the real packs; x east, z south, y up).
Trips depart from 07:00 so signal plans run and ramp meters are on (06:00-10:00).

## Families

| family | what it trains / tests | parameters (random by seed; override with `family:key=value`) |
|---|---|---|
| `straight` | lane keeping, following | `lanes` 1-5 (1-2 two-way arterial / local, 3-5 one-way motorway), `length`, `rate` (veh/min), `win` (s) |
| `curve` | steering through a bend at the advisory speed (sqrt(R·2.2)) | `radius` 30-800 m (log-uniform), `sweep` 45-180°, `dir`, `lanes`, `grade` 0-6 % |
| `sbend` | reversing curves | `radius` 40-400, `sweep`, `tangent` 0-80 m |
| `hill` | crest / sag (limited sight), optional bend right behind the crest | `grade` 3-8 %, `length`, `curve` |
| `lanes` | lane drop (merge taper, `LANE_DROP`) or lane add | `kind` drop/add, `lanes` 2-4, `freeway` |
| `merge` | freeway on-ramp | `lanes` 2-4, `accel` (300 m acceleration lane as the real build synthesises, else a gore merge), `rate`, `rampRate` |
| `diverge` | freeway off-ramp | `lanes`, `exitLane` (300 m added exit lane, else an option-lane exit), `exitShare` |
| `weave` | on-ramp + off-ramp joined by an auxiliary lane | `lanes`, `weave` 300-700 m |
| `rampmeter` | metered on-ramp (mid-ramp meter, 6 s cycle, 06:00-10:00) + acceleration lane | `lanes`, `rate`, `rampRate` |
| `junction` | left / right / through at a 4-way or T | `control` signal / allway / twoway / giveway / none, `movement` left / right / through / mixed, `shape` cross / tee, `major` primary2 (signal plan gives protected lefts) / secondary1 (permissive) / residential1, `angle` (skew ±14°), `rate`, `crossRate` |
| `grid` | blocks with arterials: signals at arterial × arterial, stops for locals at arterials, all-way stops between locals | `nx`, `nz` 4-6, `block` 150-250 m, `every`, `trips` |
| `urban` | dense grid, 2-lane primaries / secondaries, signals everywhere | `nx`, `nz` 5-7, `block` 110-170 m, `trips` 150-300 |
| `interchange` | diamond: freeway under a 7 m arterial bridge, four ramps, signal or stop terminals | `lanes`, `terminal`, `rate`, `rampRate` |
| `overpass` | grade separation: a road over a freeway / arterial on a 7 m bridge (6-8 % approaches), no connection (`crossings.json` dy 7 m) | `under`, `over`, `angle` |
| `roundabout` | single-lane roundabout, give way on entry | `radius` 22-40 m, `legs` 3-4 |
| `obstacle` | stopped (disabled) cars in the lanes: steer around (multi-lane) or pass through the opposing lane (two-lane) | `lanes` 1-3, `count` 1-3 |

Each family defines its trips (origin / destination nodes, optional via nodes, counts, departure window)
and, where relevant, obstacles. Each scenario carries `view` (where the interesting part is) for the viewer.

## World packs

`packWorld([...worlds])` tiles hundreds of scenarios into one graph: one `Simulation` / one `GPUSim`
runs them all at once. `pop.tripTile[i]` maps each commuter to its tile, `scenarioReport` splits the
outcome back per scenario and family. Example: `pack-mixed-s3` = 64 tiles, 1,485 edges, 16,587 trips;
the GPU runs it at 1.37 ms/step (16.6k agents, 1.5 sim h in 74 s wall).

## Verified: no code changes needed

- **Geometry**: `node src/geom/check.mjs data/scenarios/<dir>` passes on every family (0 strict-hop lane jumps,
  0 vertex jumps, 0 inverted pavement, 0 connectors with a step > 30°); `test/scenarios.test.js` repeats the
  continuity checks for every family and checks the section layout against `data/roads.meta.json`.
- **Junctions**: built by `tools/build_junctions.mjs` unchanged; controls' `stopAt` = junction `sStop` (tested).
- **CPU sim / env**: `createTrainingEnv({ ...sw.envOpts(), trafficMode })` with an external policy (test/scenarios.test.js).

