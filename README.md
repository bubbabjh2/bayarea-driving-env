# bayarea-driving-env

A multi-agent driving environment of the San Francisco Bay Area (San Francisco ↔ San Jose, both sides of
the bay and the bridges), built from OpenStreetMap and USGS 3DEP elevation. Every car is driven by
actions you supply.

* Real road network: ~2 tiers (freeways/arterials + local streets), per-lane geometry, junction connectors,
  lane ends / merges / weaves, ramps, bridges, tunnels, stacked decks.
* Real terrain: road heights from USGS 3DEP (1 m lidar in SF, 1/3" elsewhere), grades up to ~41 % in SF.
* Traffic controls: OSM signals, stop / give-way signs and ramp meters, merged with the authoritative SFMTA (signals,
  stop signs) and San Jose (signals) inventories; every control records its source.
* Physics, not rules: dynamic bicycle model with tyres, friction circle, weight transfer, ABS/EBD, aero,
  grade, per-surface tyre friction and rolling resistance; swept-SAT rigid-body contacts; crash severity = per-vehicle delta-v; incidents block lanes and get
  cleared / towed on realistic timelines; fuel and gas stations.
* CPU reference sim (Node, plain ES modules, zero dependencies) and a WebGPU sim that mirrors it.

Licence: code MIT (LICENSE). Map data © OpenStreetMap contributors, ODbL 1.0. Elevation: U.S. Geological
Survey, 3D Elevation Program (public domain). See NOTICE.

## Setup

* Node ≥ 22. No `npm install` needed for the CPU sim.
* GPU sim: a WebGPU browser (Chrome/Edge ≥ 113). Headless tests use the installed Chrome via
  `test/gpu/run.mjs` and the `puppeteer` library (see `src/gpu/README.md`).
* Python 3 only for the OSM → road-pack build tools (`tools/build_*.py`).

```
npm test                 # CPU unit/integration tests (~3-4 min)
npm run smoke            # synthetic world, prints steps/s
npm run smoke:real       # the real Bay Area world (first load ~90 s: junctions built in memory unless data/junctions.bin exists)
node test/gpu/run.mjs test/gpu/actions.html     # GPU: external actions + observations
node test/gpu/run.mjs test/gpu/physics.html     # GPU vs CPU physics
```

## API (CPU, `src/sim/env.js`)

```js
import { createTrainingEnv } from './src/sim/env.js';
const env = await createTrainingEnv({ density: 2000, trafficMode: 'full', seed: 1 });  // real world from data/
for (;;) {
  for (const { slot, agent, obs } of env.observe()) env.act(slot, { steer, gas, brake });
  env.step();                      // dt = 0.1 s
}
```

* **Multi-agent, all cars are agents.** `env.observe()` returns every driving car this step. A car you don't
  command holds `{steer 0, gas 0, brake 0.3}` (`opts.defaultAction`). There is no background traffic model.
* **Action:** `steer` ∈ [−1, 1] = target road-wheel angle as a fraction of `maxSteer` (actuator lag + slew rate
  in the physics), `gas`, `brake` ∈ [0, 1]. Actions pass through each driver's **reaction-time delay**
  (0.6–1.8 s, log-normal, `vs.delaySteps`) before they act.
* **Observation:** a structured object per car (`src/sim/obs.js` header: lane frame, speed, neighbours,
  next traffic control and its state/time-to-change, lane ends, junction conflicts, oncoming traffic, grade,
  vehicle geometry). `env.observe({ v3: true })` adds `obs.v3`, a 358-float vector of ego-frame bodies (16 nearest
  in 8 sectors × 16 features), mirrors/blind spots, control extras and command history (`src/sim/obsLayout.js`).
  The GPU sim also emits a flat 70-float "v2" vector (`src/gpu/README.md`); the CPU has no flat v2 encoder.
  Full per-car state (3-D pose, velocities, dimensions, flags incl. crashed/disabled, applied brake) is in
  `env.sim.vs` (struct of arrays, `src/sim/LAYOUT.md`); signal states via `env.sim.controls`, wrecks /
  incidents via `env.sim.crashes` / `env.sim.activeCrashes`, placed obstacles via `env.sim.obstacles`.
* **Termination (per car):** arrival; crash with own Δv ≥ 10 km/h (disabled); running out of fuel (stranded);
  not arrived by the end of the run (default day 05:00–13:00). Waiting is never a failure. `env.outcomes()`.
* **Worlds:** `trafficMode` `'empty'` (cars don't interact), `'sparse'` (+ stopped wrecks), `'full'`;
  `obstacles`, `staticObstacles`; synthetic worlds in `data/scenarios/` (`src/scenarios/README.md`).
* **Seeding:** population + sim RNG from `seed`; runs are bit-deterministic for the same seed and actions
  (checked: two runs → identical state hash). On dry asphalt the physics is bit-identical to the original project's
  env; road surfaces (below) change the state wherever a car's wheels are on another surface.
* **Headless:** yes; the sim never loads rendering code.

Measured (Apple M-series laptop): CPU real world, 2,000-commuter population at ~06:20: ~3,000 steps/s with
~20–100 cars on the road (≈55k agent-steps/s; cost scales with active cars); synthetic 300 agents ~6,500
steps/s. GPU real world: 10k cars 1.47 ms/step, 100k cars 4.68 ms/step.

## Rust / Python (branch `rust`, `rust/`)

A native host for the GPU sim: the unchanged WGSL kernels run through wgpu (Metal / Vulkan / DX12) from Rust or
Python, with no browser. The world loader, lane geometry, junction builder, router and world packer are ported to Rust
and match the JS bit for bit. `npm run build:junctions` output is byte-identical: 6.3 s in Rust vs ~90 s in JS. The world
loads in ~60 ms. Per-car state agrees with the JS GPU sim to float tolerance (`rust/README.md`).

```python
import numpy as np, bayenv          # cd rust/crates/bayenv-py && maturin develop --release
env = bayenv.Env("data", n_cars=10000, seed=11)
env.act(np.tile(np.float32([0, 0.3, 0]), (env.n_slots, 1)))   # [steer, gas, brake] per car
env.step(20)                        # 20 steps in one GPU submit
obs, st = env.observe(), env.state()  # float32 (n, 70); dict of per-car numpy arrays
```

Measured (Apple M4 Max): 10k cars 1.12 ms/step, 100k cars 4.66 ms/step (`rust/BENCHMARKS.md`).

## Data and coordinates

`data/roads*.bin` + `*.meta.json` (schema `data/roads.schema.md`), `controls.*`, `stations.*`, `crossings.json`,
`crosswalks.json`, `water.json`, `*.json.gz` (readable copies, `npm run unpack`), `data/dem/compact/`
(20 m ground grid), `data/src/flat/` (pre-elevation road snapshot, input of the elevation build).
Build outputs, not committed: `data/junctions.bin` (junction connectors / conflict matrix; the loader builds it
in memory if absent; `npm run build:junctions`, ~90 s) and `data/lines*.bin` (lane-marking geometry for renderers;
`npm run build:lines`).
Local frame: metres, **x = east, z = south, y = up** (NAVD88 metres, bare-earth), equirectangular about
37.55 N, 122.15 W (`src/proj.js` converts to/from lat/lon). Heading = atan2(dz, dx). Right-hand traffic,
lane 0 = rightmost.

## Traffic controls

`data/controls.*` holds 39,631 controls: 13,578 signals (3,583 of them mid-block pedestrian signals), 23,218 stops,
2,313 give-ways and 522 ramp meters. They come from OSM, merged with authoritative city data by `npm run build:controls`
(`tools/controls/README.md`). Each record carries:
* `source`: `osm`, `osm+<dataset>` or `<dataset>`.
* `verified`: `true` (matched to an authoritative record), `false` (in the dataset's area but not in it; kept, because
  inventories omit Caltrans signals) or `null` (no authoritative data there).
* `timing: "synthetic"` on every signal and ramp meter.

Only openly licensed data is merged: DataSF (PDDL) and San Jose (CC BY). Five cities publish data with no licence, so
those datasets are only used to measure known gaps.

| city | dataset (role) | signals: confirmed / added / unplaced / OSM-only | stop signs |
|---|---|---|---|
| San Francisco | SFMTA signals + stop signs (merge) | 1,229 / 32 / 50 / 85 | 2,026 OSM stop controls confirmed, 1,823 added, 2,696 approaches added, 135 OSM-only removed |
| San Jose | City Owned Traffic Signals (merge) | 757 / 51 / 163 / 145 | none published |
| Oakland | TrafficSignals (verify only) | 621 / known missing 74 / 35 / 53 | none published |
| South San Francisco | Traffic_Signals_View (verify only) | 109 / known missing 12 / 14 / 27 | none published |
| Emeryville | Traffic_Signals_View (verify only) | 30 / known missing 3 / 3 / 17 | none published |
| Mountain View | Signal poles (verify only) | 116 / known missing 4 / 9 / 54 | none published |
| Cupertino | Signal heads + signs (verify only) | 68 / 0 / 6 / 7 | 171 sign records confirmed, 415 known missing |

What the columns mean:
* *unplaced*: the record has no graph junction within 30 m. Most of San Jose's are east of the map edge (−121.80°).
* *OSM-only*: an OSM signal with no record within 50 m. These are kept and flagged.
* Validation: 23 of 24 sampled intersections match. The one miss is an Oakland signal on the known-missing list
  (`tools/controls/report/validation.txt`).

No authoritative source was found for Palo Alto, Sunnyvale, Santa Clara, Redwood City, San Mateo, Menlo Park,
Burlingame, San Bruno, Millbrae, Daly City, Belmont, San Carlos, Foster City, Los Altos, Milpitas, Fremont, Newark,
Union City, Hayward, San Leandro, Alameda or Berkeley. Caltrans has no state-highway signal or ramp-meter layer. In all
of these places the controls are OSM only.

Signals are assigned per approach edge (all lanes of that edge). The fixed-time plans use two axes, protected lefts
on big arterials, ITE yellow, a 1.5 s all-red and a green wave; ramp meters run 06:00–10:00.

`src/look/signals.js` is an optional three.js helper that draws heads, poles, mast arms and STOP/YIELD plates at
every stop line, driven by the sim's signal states. The env never imports it, and its **shape is generic**: the same
pole/arm/head everywhere, not surveyed. It needs `three` (`npm i three` or an import map; see `tools/controls/README.md`).

## Road surfaces and friction

Each edge has a pavement surface code and a ground class for each side, in `data/surfaces.bin` (2 bytes per edge;
format, codes and sources in `docs/SURFACES.md`; `npm run build:surfaces`). The pavement comes from the OSM
`surface=*` tag where it is set: 50.2 % of main-tier and 34.9 % of local-tier edges. Elsewhere it defaults to
asphalt, concrete on bridges and tunnels, and steel grating on movable bridges. The ground class is what a wheel
touches past the paved width: curb / sidewalk, grass verge, gravel median, dirt or shore sand. It comes from the road
class, the `sidewalk` tags and the distance to the water polygons.

`src/sim/surface.js` holds the friction table: peak and sliding adhesion, dry and wet, and the rolling resistance per
surface, with sources. The CPU and GPU physics take the surface under the right and left wheel lines each step. They
scale the vehicle's tyre `mu` by `muPeak / muPeak(asphalt)`, scale the locked-wheel sliding friction to match, and
use the surface's rolling resistance. The codes of the last step are in `env.sim.vs.surfR` / `surfL`. Dry asphalt
is the reference, with every factor 1. Full brake from 100 km/h, sedan: asphalt / concrete 41.2 m, paving stones
45.6 m, gravel 56.7 m, steel grate 61.7 m, grass 66.4 m. The wet columns of the table are documentation only:
friction does not change with weather. Embedded rails (cable car, streetcar) are not modelled.

## Viewing the world

```
npm run build:tiles       # streamed road tiles -> data/tiles/ (~230 MB, ~30 s; builds junctions/lines first if missing)
npm run build:landscape   # terrain, land cover, buildings, trees, bridge structures -> data/landscape/ (~110 MB, ~5 min)
npm run serve             # http://127.0.0.1:8000/world.html
```

Both outputs are gitignored and rebuilt from committed data. `build:landscape` downloads OSM land cover (Overpass,
attic date pinned), DataSF building footprints and street trees on first run (`tools/landscape/raw-lock.json` holds the
sha256 of the snapshot used); it uses the 10 m ground from `npm run build:elevation` if present, else the committed 20 m
compact grid. `world.html` works without the landscape (procedural ground, `?landscape=basic`).

What renders (`src/look/README.md`, `src/stream/README.md`): roads streamed in a 7-level quadtree with LOD cross-fades;
pavement by surface (`data/surfaces.bin`: asphalt with wear and patches, concrete slabs, pavers, gravel, dirt, steel
grate, wood), elevated decks in concrete; lane lines, edge and centre lines, stop bars, yield lines and crosswalks
from the same line set the sim uses; curbs, sidewalks, verges and driveways on city streets; signal heads, ramp meters
and STOP/YIELD signs driven by the sim's controller; terrain from USGS 3DEP; OSM land cover; water with a depth
gradient; San Francisco buildings; trees; bridge piers, girders and a Golden Gate model; sky and sun for 05:30-10:00.
All textures are procedural (NOTICE). three.js r160 is vendored in `vendor/three-0.160.0/`.

Camera: drag to orbit, right-drag to pan, wheel to zoom, W/A/S/D/Q/E to fly, keys 1-9 for presets, T for a tour.
URL: `?view=Bay%20Bridge`, `?cam=lat,lon,height,distance,azimuth[,lookAtHeight]`, `?look=off|perf`,
`?sim=gpu&n=100000` (WebGPU sim cars, one constant action for every car; replace with your own via `sim.setActions`).

## Rebuilding the world

Everything committed under `data/` can be regenerated and checked against pinned hashes:

* **Elevation** (`docs/ELEVATION.md`): `npm run build:elevation` downloads the 8 pinned USGS 3DEP tiles
  (sha256-checked), builds the DEM mosaics, road heights (from `data/src/flat/`), control/station heights, the
  road-conformed 10 m ground and the compact grid; `npm run verify:elevation` hashes all outputs. Verified
  bit-identical to the original project.
* **Derived world** (`docs/WORLD-BUILD.md`): `npm run unpack && npm run build:world-derived` rebuilds crossings,
  the traffic controls (`npm run build:controls`: authoritative merge, junctions, lane markings) from the committed
  packs and snapshots; `npm run verify:world` checks every file against `tools/osm/world-manifest.json` (rebuilds are
  bit-identical; the original project's OSM-only hashes are kept as `--baseline osm-only`).
* **From OSM** (`npm run build:osm`): pinned Overpass queries with attic dates, sha256 lockfile
  (`tools/osm/osm-lock.json`). water.json and crosswalks.json rebuild bit-identically. The flat road tier
  does **not**: the original was fetched from an Overpass mirror whose database was internally inconsistent
  (no attic date reproduces it; a rebuild differs by ~±2–4 % of edges), so the committed flat snapshot
  `data/src/flat/` is the canonical input and everything downstream of it is reproducible.
* **Synthetic scenarios:** `npm run build:scenarios`.

## Limitations for full perception

What the rendered world does not contain, and what would add it:

* Signal heads, poles, mast arms and signs have one generic shape, not surveyed → survey or per-agency design standards per intersection.
* One signal head per approach, no per-lane heads; the only arrow is the protected-left lamp column → per-lane heads and arrow sections from turn-lane data.
* Signal timing is synthetic (fixed-time plans, `timing: "synthetic"`): the visible light state and phase are plausible, not real → agency timing sheets or observed phase data.
* Traffic controls are OSM-only outside San Francisco and San Jose, with known gaps (see "Traffic controls") → authoritative inventories for the other cities and Caltrans.
* Buildings exist only in San Francisco (DataSF); elsewhere buildings are a roof tint in the land-cover texture → OSM or county footprints with heights (`tools/landscape/fetch_osm.mjs buildings` supports it per area).
* Trees are low-poly clay shapes (DataSF street trees, OSM tree nodes, seeded scatter) → species-specific models.
* No street furniture (street lights, hydrants, poles other than signals, fences, barriers, guard rails, sound walls) → OSM furniture/barrier tags and Caltrans asset data.
* No parked cars, pedestrians or cyclists (parking lanes and sidewalks are geometry only) → parked-vehicle placement and pedestrian/cyclist agents.
* Sign text is only STOP and YIELD; no speed-limit, street-name, freeway guide or warning signs → sign inventory (OSM `traffic_sign`, Caltrans) and a sign renderer.
* Markings are lane lines, edge/centre lines, stop bars, yield lines and crosswalks only; no arrows, text, bike lanes, gore chevrons or parking stalls → marking data (OSM `turn:lanes`, bike-lane tags) and decal geometry.
* Cars are one two-box clay mesh in four type colours, with body tint for braking/crash; no lights, turn signals or model variety → per-type meshes with lamp emitters driven by the sim state.
* Textures are procedural and tile at 8 m / 32 m; land cover is 6.45 m/px; one asphalt style per tier → photographic or scanned material sets and decals.
* LOD changes are dithered cross-fades over 350 ms; buildings dissolve at 2.4-3.2 km, trees beyond ~1.7 km; coarse levels simplify roads to strips/lines → a fixed high-detail radius or higher tile budget for camera work.
* Lighting covers 05:30-10:00 with clear/golden/overcast skies; no shadow maps (only terrain shadows on the golden preset), no night, no headlights → a full-day sun model, shadow maps and night lighting.
* Without `npm run build:elevation`, the terrain comes from the committed 20 m compact grid (RMS 0.57 m, max 45.8 m vs the 10 m ground), so the ground can sit off the roads on steep slopes → run the full elevation build.
* Geometry residuals remain: one 3.7 m lane jump (US-101, node 45147), ~0.6 km of ramp lane centres inside a mainline and ~0.8 km of overlaps with non-joined carriageways (`src/geom/README.md`), lines hidden where drawn carriageways overlap → further carriageway fixes in `src/geom`.
* No sensor models: no lens distortion, noise, motion blur, rolling shutter or auto-exposure; rendering is a clean pinhole camera → a post-process sensor pass.

## Weather

The environment does not model weather:

* Rain, wet roads, standing water / puddles and hydroplaning: friction does not change with weather; the wet columns of the surface table (`src/sim/surface.js`, `docs/SURFACES.md`) are documented, not wired → a weather state that selects the wet columns per wheel, plus water-film depth for hydroplaning.
* Snow and ice: not modelled → snow/ice friction entries and a surface-state field.
* Fog: only a fixed height-scaled distance haze; no fog that limits visibility → a visibility parameter for render and perception range.
* Low sun and glare: the sun position is drawn, but no glare, bloom or lens flare → a glare/bloom pass tied to sun angle.
* Visually there are no wet, rain or snow materials, no rain or snow particles, no reflections on wet pavement → wet/snow material variants and particle effects.
