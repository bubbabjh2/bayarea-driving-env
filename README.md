# bayarea-driving-env

A multi-agent driving environment of the San Francisco Bay Area (San Francisco ↔ San Jose, both sides of
the bay and the bridges), built from OpenStreetMap and USGS 3DEP elevation. It is the environment only:
there is **no driving policy, no built-in/IDM driver, no neural net and no training code** in this repo.
Every car is driven by actions you supply.

* Real road network: ~2 tiers (freeways/arterials + local streets), per-lane geometry, junction connectors,
  lane ends / merges / weaves, ramps, bridges, tunnels, stacked decks.
* Real terrain: road heights from USGS 3DEP (1 m lidar in SF, 1/3" elsewhere), grades up to ~41 % in SF.
* Traffic controls from OSM: signals (with generated plans), stop / give-way signs, ramp meters, crossings.
* Physics, not rules: dynamic bicycle model with tyres, friction circle, weight transfer, ABS/EBD, aero,
  grade; swept-SAT rigid-body contacts; crash severity = per-vehicle delta-v; incidents block lanes and get
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

## Rebuilding the world

Everything committed under `data/` can be regenerated and checked against pinned hashes:

* **Elevation** (`docs/ELEVATION.md`): `npm run build:elevation` downloads the 8 pinned USGS 3DEP tiles
  (sha256-checked), builds the DEM mosaics, road heights (from `data/src/flat/`), control/station heights, the
  road-conformed 10 m ground and the compact grid; `npm run verify:elevation` hashes all outputs. Verified
  bit-identical to the original project.
* **Derived world** (`docs/WORLD-BUILD.md`): `npm run unpack && npm run build:world-derived` rebuilds crossings,
  junctions and lane markings from the committed packs; `npm run verify:world` checks them against the original
  project's sha256 (junctions.bin, lines.bin, lines-geom.bin, crossings.json: bit-identical).
* **From OSM** (`npm run build:osm`): pinned Overpass queries with attic dates, sha256 lockfile
  (`tools/osm/osm-lock.json`). water.json and crosswalks.json rebuild bit-identically. The flat road tier
  does **not**: the original was fetched from an Overpass mirror whose database was internally inconsistent
  (no attic date reproduces it; a rebuild differs by ~±2–4 % of edges), so the committed flat snapshot
  `data/src/flat/` is the canonical input and everything downstream of it is reproducible.
* **Synthetic scenarios:** `npm run build:scenarios`.

## Known issues (from the original project's own measurements)

* **Crash rate is far above real** (~10⁴× real freeway rates with the original learned/rule drivers) and
  high-density runs tend to **gridlock / collapse** around junction boxes and wrecks (docs in the original:
  junction-box conflicts ~45 %, unseen wrecks ~37 % of collapse causes; partly fixed in layout v8).
* Road geometry residuals: 3 lane-count jumps, 24 open weaves, 39 ramp overlaps (OSM crossings), 13 stacked
  SoMa/YBI ramp segments over the grade limit.
* Signal timing plans are **synthetic** (generated per junction), not real city timing.
* The reaction-time delay (up to 1.8 s) makes naive controllers unstable; plan for it.
* With no built-in driver, the sim's deadlock breaker / IDM-related options are inert, and every car must be
  driven by your policy (expensive at full scale on CPU: use the GPU sim).
* The CPU real-world load takes ~90 s and ~6–10 GB RAM (`--max-old-space-size=12000`) when
  `data/junctions.bin` is absent; build it once (`docs/WORLD-BUILD.md`).
