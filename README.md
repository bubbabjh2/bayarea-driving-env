# bayarea-driving-env

A multi-agent driving environment of the San Francisco Bay Area (San Francisco ↔ San Jose, both sides of
the bay and the bridges), built from OpenStreetMap and USGS 3DEP elevation. It is the environment only:
there is **no driving policy, no built-in/IDM driver, no neural net and no training code** in this repo.
Every car is driven by actions you supply.

* Real road network: ~2 tiers (freeways/arterials + local streets), per-lane geometry, junction connectors,
  lane ends / merges / weaves, ramps, bridges, tunnels, stacked decks.
* Real terrain: road heights from USGS 3DEP (1 m lidar in SF, 1/3" elsewhere), grades up to ~41 % in SF.
* Traffic controls: OSM signals, stop / give-way signs and ramp meters, merged with the authoritative SFMTA (signals,
  stop signs) and San Jose (signals) inventories; every control records its source. Signal timing is generated
  (synthetic), not real (see "Traffic controls").
* Physics, not rules: dynamic bicycle model with tyres, friction circle, weight transfer, ABS/EBD, aero,
  grade; swept-SAT rigid-body contacts; crash severity = per-vehicle delta-v; incidents block lanes and get
  cleared / towed on realistic timelines; fuel and gas stations.
* CPU reference sim (Node, plain ES modules, zero dependencies) and a WebGPU sim that mirrors it.

Licence: code MIT (LICENSE). Map data © OpenStreetMap contributors, ODbL 1.0. Elevation: U.S. Geological
Survey, 3D Elevation Program (public domain). See NOTICE.

## Known limitations & disclaimers

**Loading**
* **First real-map load takes ~90 s and 6–10 GB RAM** unless `data/junctions.bin` is pre-built. Build it once with
  `npm run build:junctions` (or `node --max-old-space-size=14000 tools/build_junctions.mjs data/`; see `docs/WORLD-BUILD.md`).

**Road network**
* Known geometry residuals: 3 lane-count jumps, 24 open weaves and 39 ramp overlaps (all from OSM crossings), plus 13
  stacked SoMa/YBI ramp segments over the grade limit.
* The flat road tier cannot be reproduced exactly from OSM. The original came from an inconsistent Overpass mirror,
  and re-fetching differs by about ±2–4 % of edges. The committed snapshot (`data/src/flat/`) is the canonical
  input. Elevation, junctions, lines, controls, crosswalks and water rebuild bit-identically from it.
* Rebuilding controls from OSM alone moves 17 approach stop lines at 14 controls by 0.6–7.7 m. The committed data keeps the original
  positions as the baseline.

**Traffic controls**
* Signal and stop locations are OSM merged with DataSF (signals and stop signs, PDDL) and San Jose (signals, CC BY).
  About 22 other cities and Caltrans have no usable open source, so they are OSM only. Oakland, South San
  Francisco, Emeryville, Mountain View and Cupertino data (no licence stated) were used only to verify, never merged.
  Known gaps: Oakland ~74 missing signals, South SF 12, Emeryville 3, Mountain View 4, Cupertino ~415 missing stop
  signs. Spot check: 23 of 24 intersections match.
* All signal timing is synthetic: fixed two-axis plans, protected lefts only on major roads, 1.5 s all-red, ramp
  meters 06:00–10:00, no flashing state. No city publishes timing. Records are tagged `timing: "synthetic"`.
* Controls are assigned per approach, not per lane. There are no turn arrows.
* Controls are points only. `src/look/signals.js` is an optional three.js helper that draws a generic pole, mast arm
  and head; it is not a real per-site model.

## Setup

* Node ≥ 22. No `npm install` needed for the CPU sim.
* GPU sim: a WebGPU browser (Chrome/Edge ≥ 113). Headless tests use the installed Chrome via
  `test/gpu/run.mjs` and the `puppeteer` library (see `src/gpu/README.md`).
* Python 3 only for the OSM → road-pack build tools (`tools/build_*.py`).

```
npm test                 # CPU unit/integration tests (~3-4 min)
npm run smoke            # synthetic world, trivial lane-keeping policy, prints steps/s
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
  (0.6–1.8 s, log-normal, `vs.delaySteps`) before they act — a policy must not add its own delay.
* **Observation:** a structured object per car (`src/sim/obs.js` header: lane frame, speed, neighbours,
  next traffic control and its state/time-to-change, lane ends, junction conflicts, oncoming traffic, grade,
  vehicle geometry). `env.observe({ v3: true })` adds `obs.v3`, a 358-float vector of ego-frame bodies (16 nearest
  in 8 sectors × 16 features), mirrors/blind spots, control extras and command history (`src/sim/obsLayout.js`).
  The GPU sim also emits a flat 70-float "v2" vector (`src/gpu/README.md`); the CPU has no flat v2 encoder.
  Full per-car state (3-D pose, velocities, dimensions, flags incl. crashed/disabled, applied brake) is in
  `env.sim.vs` (struct of arrays, `src/sim/LAYOUT.md`); signal states via `env.sim.controls`, wrecks /
  incidents via `env.sim.crashes` / `env.sim.activeCrashes`, placed obstacles via `env.sim.obstacles`.
* **Reward:** not returned by `step()`. `src/sim/reward.js` defines it (`reward({progress, collision | crash,
  stranded, arrived, lateBy, steerJerk, dt})`): crash −20 (or severity-scaled), stranded −20, arrival +10,
  on time +2, −0.5/min late (cap −10), annealed progress shaping, small steering-jerk term. Per-step inputs
  come from `env.sim.signalsFor(slot)`; trip-level fitness via `env.fitness()`. `examples/random_policy.mjs`
  shows the loop.
* **Termination (per car):** arrival; crash with own Δv ≥ 10 km/h (disabled); running out of fuel (stranded);
  not arrived by the end of the run (default day 05:00–13:00). Waiting is never a failure. `env.outcomes()`.
* **Curriculum:** `trafficMode` `'empty'` (cars don't interact), `'sparse'` (+ stopped wrecks), `'full'`;
  `obstacles`, `staticObstacles`; synthetic worlds in `data/scenarios/` (`src/scenarios/README.md`).
* **Seeding:** population + sim RNG from `seed`; runs are bit-deterministic for the same seed and actions
  (checked: two runs → identical state hash; also identical to the original project's env).
* **Headless:** yes; the sim never loads rendering code.

Measured (Apple M-series laptop): CPU real world, 2,000-commuter population at ~06:20: ~3,000 steps/s with
~20–100 cars on the road (≈55k agent-steps/s; cost scales with active cars); synthetic 300 agents ~6,500
steps/s. GPU real world: 10k cars 1.47 ms/step, 100k cars 4.68 ms/step.

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
