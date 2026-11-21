# GPU sim (`src/gpu/`) — WebGPU, external policy only

Every vehicle's state lives in WebGPU storage buffers and every per-vehicle stage is a compute kernel
(`kernels/*.wgsl`). It mirrors the CPU reference sim (`src/sim/sim.js`, state contract `src/sim/LAYOUT.md`):
dynamic bicycle + tyres in 10 ms sub-steps, swept-SAT rigid-body contacts and barriers, `src/geom` lane geometry
and junction connectors, traffic controls (signals / signs / ramp meters), junction yield and conflict perception,
incidents, fuel and refuelling, trip schedule / streaming population.

There is **no built-in driver and no neural net** in this repo. Cars are driven by actions the caller writes
into an `actions` buffer. The GPU also computes the observation vectors (optional) so a policy can read them back.

## Requirements

* A browser with WebGPU: Chrome / Edge ≥ 113 (macOS Metal, Windows D3D12, ChromeOS; Linux needs
  `--enable-unsafe-webgpu --enable-features=Vulkan`). The real Bay Area world at 100k vehicles uses ~0.55 GB of GPU memory.
* Headless: `node test/gpu/run.mjs <page>` launches the installed Chrome (`$CHROME_PATH`, default
  `/Applications/Google Chrome.app`) with WebGPU flags, a mock keychain, a throwaway profile and a throwaway HOME
  (parent folder `$GPU_SCRATCH`, default the OS temp dir). It needs the `puppeteer` library (found in `node_modules`,
  `$PUPPETEER_PATH`, or the npx cache: `npx --yes puppeteer@23.11.1 --version`); it never downloads its own browser.
* Pages are served from the repo root (run.mjs has its own static server; for a browser: `python3 tools/serve.py 8790 .`).
* Loading the real world in the browser takes ~75 s when `data/junctions.bin` is absent (junctions are built in memory).

## API

```js
import { GPUSim } from './src/gpu/gpuSim.js';
const sim = await GPUSim.create(graph, { capacity: 131072, dt: 0.05, obs: 'v2', controls: world.controls });
sim.setPopulation(pop, { stations });     // src/sim/population.js output or a popPack (tools/packPopulation.mjs)
// or: sim.scatter(n)                      // n cars on random routes at once (benchmarks)
const A = new Float32Array(sim.cap * 3);  // per slot [steer, gas, brake]
loop:
  sim.setActions(A);                      // uploaded; read by every DRIVING slot at each decision
  await sim.runSteps(k);                  // k sim steps in one submit (+ stats / events readback)
  const X = await sim.readObs();          // Float32Array(cap * sim.obsSize), slot order
  const st = await sim.readState();       // full vehicle state (st.veh.get(slot, field), st.pose)
  sim.getStats(); await sim.readRewardSums();   // aggregates; per-agent reward sums / outcomes
```

Options (`GPU_DEFAULTS` in `gpuSim.js`):

| option | meaning |
|---|---|
| `obs` | `'v2'` (default): 70-float vector per slot (`sense.wgsl`); `'v3'`: 358 floats (`sense` + `perceive.wgsl`: 16 nearby bodies in 8 ego sectors, mirrors, controls extras, command / leader history; layout `src/sim/obsLayout.js`); `'none'`: no observation kernels (fastest) |
| `actionInput` | `'position'` (default): `steer` = target wheel angle / maxSteer in [-1, 1]; `'rate'`: steer rate in [-1, 1] of maxSteerRate, integrated into the target |
| `decideEvery` | a slot reads its action every k steps (default 1); in between the last command is held |
| `features` | `{ laneChanges, accidents, fuel, collisions, yield, policy, zipFix, empty }`; `policy` (default true) flags every spawned car as externally driven (`laneChanges` only affected the removed built-in driver) |
| `jerkReward`, `crashSev`, `progressWeight`, `jerkWeight` | terms of the per-agent reward accumulated on the GPU (`readRewardSums`; same terms as `src/sim/reward.js`) |

Timing: the actions in the buffer at the start of a `runSteps(k)` submit are used for all k steps. Observations read
after a submit were sensed in its last step (before that step's integration), i.e. at most 1 dt old. Commands go
through the per-car reaction-time ring (`delaySteps`), exactly as on the CPU. Slots are not agent ids: map them with
the population (`sim.slotTrip[slot]` = 2 · commuter + leg, −1 free).

### Observation v2 layout (70 floats, `sense.wgsl`; offsets exported as `OBS_V2` from `gpuSim.js`)

```
 0 speed/40   1 speedLimit/40   2 laneIndex/max(1,laneCount-1)   3 laneCount/6   4 laneOffset/laneWidth (+left)
 5 headingError/(pi/4)   6 / 7 left / right lane exists
 8..19 neighbours {min(gap,120)/120, clamp(relSpeed/20)} x ownAhead, ownBehind, leftAhead, leftBehind, rightAhead, rightBehind
20 distance to next route maneuver / 2000   21..29 maneuver one-hot (continue, exit, merge, turn-left, turn-right,
   keep-left, keep-right, arrive, refuel)   30..32 lane hint (left, none, right)
33..36 driver traits (aggression, patience, headway/3, reactionTime)   37 running late   38 timeToWorkStart/3600
39 fuel fraction   40 refuel planned   41..43 road grade inputs (position mode)   44 / 45 own lane end distance / side
46..50 next control type (signal, stop, give_way, ramp_meter, crossing)   51..55 its state (green, yellow, red, sign, dark)
56 stop-line distance/250   57 time to change/30   58..60 my movement (through, left, right)   61 made a full stop
62 yield required   63 / 64 heading kink (disabled: 0 / 1)   65..69 junction conflict (distance/60, other's TTC/10,
   other stopped, I am in the box, mouth distance/60)
```

## Files

```
gpu.html                    viewer: real roads, up to 100k+ cars, constant demo actions (?n=100000&gas=0.3)
src/gpu/device.js           WebGPU init, buffer registry, kernel compiler (bindings by name), timestamp Profiler
src/gpu/gpuSim.js           GPUSim: orchestration, uploads, spawning / streaming, events, scenes, refuel, actions / obs
src/gpu/layout.js           host mirror of the WGSL structs
src/gpu/worldPack.js        SimGraph -> one u32 world blob (edges, points, CSR, geom, hops, controls, junctions)
src/gpu/render.js           raw WebGPU renderer (roads + instanced cars straight from the sim buffers)
src/gpu/popPack.js, streamPack.js, routeWorker.js, fuelWorker.js   populations, streaming routes, refuel planning
src/gpu/tools/packPopulation.mjs, packTable.mjs   node: population packs -> src/gpu/cache/ (generated, gitignored)
src/gpu/kernels/*.wgsl      one file per stage; act.wgsl = external actions -> reaction ring
test/gpu/run.mjs            headless Chrome runner
test/gpu/actions.html       compile all kernels (obs v2 / v3 / none), gas accelerates, brake stops, obs finite
test/gpu/physics.html       dynamics + contacts, CPU (src/sim/physics) vs GPU, every vehicle type
test/gpu/equivalence.html   CPU vs GPU from the same exportState snapshot with the same actions (synthetic graph)
test/gpu/bench.html         ms/step on the real world (?sizes=10000,100000&obs=v2)
test/gpu/smoke.html         WebGPU adapter check
```

Step order: `tick` → `ops` → spawn / station re-entry → contacts (hash, barrier, swept detect, resolve) →
footprint lane lists → junction lists → `leaders` (+ lane ends, controls) → junctions (perceive, courtesy, yield,
reroute scan) → `sense` (+ `perceive`) → `act` → `integrate` (ring delay, physics sub-steps, edge hops, arrival,
stranding, fuel, pose, rewards).

## Measured (Apple Silicon, Chrome headless, Metal, 2026-11-20)

| test | result |
|---|---|
| `actions.html` (2000 cars, synthetic) | PASS for obs v2 / v3 / none: 0 → 15.7 m/s on gas 0.5, 0.01 m/s after 10 s of full brake, all obs finite; ~0.75–0.9 ms/step |
| `physics.html` | PASS: max abs speed diff ≤ 2e-4 m/s, identical 0–100 km/h times and stop distances for all 4 types; contacts: 0 decision mismatches in 4000 pairs |
| `equivalence.html` (483 cars, constant per-slot actions, dt 0.05) | PASS: 100 % within 0.5 m after 1 step, 99.6 % after 10, 94.3 % after 500, 92.5 % after 1000 (divergence from chaotic contacts of the blind constant policy; float32 vs float64) |
| `bench.html`, real Bay Area world, obs v2 | 10k cars 1.47 ms/step (680 steps/s); 100k cars 4.68 ms/step (214 steps/s, 10.7x real time at dt 0.05) |
| `gpu.html?n=20000` | 60 fps |

## Not included (vs the original project)

The built-in IDM / lane-change driver (`drive.wgsl`), the brain nets and their kernels (`netFast`, `net3`, `net3list`,
`brainApply`), the RL rollout recorder (`rl.wgsl`), the training interface (`trainer.js`, `tools/trainDay.mjs`), the
telemetry sampler (`telemetry.wgsl`), and the brain-parity / DAgger test pages. The CPU-vs-GPU obs-vector parity test
was not ported: the CPU-side flat-vector encoder lived in the excluded brain code. The GPU obs kernels are unchanged
from the original, where that parity test passed. Without a built-in driver, `laneCheckInterval`, `s0`,
`speedFactor*`, `yieldLook`, `stopMargin` and the deadlock breaker (`deadlockBreak`) have no effect.
