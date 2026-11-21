# Streaming / multi-level loading (`src/stream/`, `tools/tiles/`, `data/tiles/`)

Rendering only. Nothing in `src/sim`, `src/nav` or `src/gpu` imports this. The sim still loads the full road graph;
only the *meshes* stream.

## Build the pyramid (one command, re-run after every world rebuild)

    npm run build:tiles     # ~30 s, writes data/tiles/ (+ data/tiles/BUILD.txt); ~230 MB, gitignored

It first creates the build inputs if missing: `npm run unpack` (readable JSON), `npm run build:junctions`,
`npm run build:lines`.

Inputs: `data/roads*.bin/meta`, `src/geom/` (lane geometry, same calls as the original full-load renderer),
`data/controls.*`, `data/stations.*`, `data/water.json`, `data/crosswalks.json`, `data/junctions.bin`, `data/lines.bin`,
and the OSM `sidewalk` / `parking` fields in `roads-local.json`.

| level | tile | contents |
|---|---|---|
| L0–L1 | 102 / 51 km | motorway + trunk lines (DP 300/120 m), coastline, water mask |
| L2 | 25.6 km | + primary/secondary lines, motorway/trunk strips (50 m) |
| L3 | 12.8 km | all main tier as strips (20 m) + lines, tertiary lines |
| L4 | 6.4 km | main ribbons from src/geom (full res), all local streets as lines |
| L5 | 3.2 km | + laneCoord/uv, local streets as real-width strips, stations |
| L6 | 1.6 km | everything (one street model for main + local roads, below), lane markings, stop bars, crosswalks, yield lines, signal/sign posts, stations |

L6 street model: each physical road is cut back at at-grade junctions. The cut is where the facing
pavement edges of angularly adjacent arms meet, `d = (h_b + h_a cos t) / sin t`. Each junction is then
filled with one polygon, so no roads overlap. Main roads come from src/geom `pavement()`; local streets
use 3.3 m lanes, plus 2 m parking on urban residential. Urban primary/secondary roads also get curbs
and sidewalks.

Sidewalks and curbs are cut back separately per side, using the same formula with the sidewalk width,
so the sidewalks of skewed arms never overlap. Every pair of adjacent arms gets a rounded corner: a
Bezier through the meeting point of their edge lines, with a curb top, a curb face and a concrete corner.
Suburban residential streets get a 1.5 m `verge` between the curb and the sidewalk, with seeded
`driveway` aprons every 15–24 m.

Tile format: `src/stream/format.js` (int16 tile-local positions on a grid aligned across tiles,
so there are no cracks; triangles are assigned to tiles by centroid; gzip). The manifest
(`manifest.json.gz`, 59 KB) holds bounds, level, byte sizes and child masks.

## Use

```js
import { createStream } from './src/stream/index.js';
const stream = createStream({ scene, camera, renderer, baseUrl: 'data/tiles/',
  onTileMesh: (m) => look ? look.onTileMesh(m) : m,       // src/look styles tiles before their first frame
  onTileDispose: (m) => look?.onTileDispose?.(m) });
await stream.ready;                                      // root tile is on screen
// every frame, before render:
stream.update();
stream.stats();   // {tilesLoaded, tilesVisible, visibleByLevel, maxVisibleLevel, gpuMB, fetchedMB, drawCalls, ...}
await stream.settled();   // nothing queued, loading or building (frame-exact capture: call after camera cuts)
stream.setDebugLevels(true);   // colour tiles by level
```

Options: `sse` (px, default 6), `hysteresis` (0.75), `gpuBudgetMB` (384), `maxConcurrent` (4),
`workers` (2–4), `buildMsPerFrame` (6), `maxLevel`, `ground` (adds a land plane), `roles` (a Set of
roles to build).

How it works:
- Selection descends from the root and refines a tile when `err * viewportH / (2 tan(fov/2)) / distance > sse`.
  It stays refined until the error drops below `sse * hysteresis`. Tiles outside the view frustum are skipped.
- No holes: a parent is replaced only when every visible child is decoded and built. The swap is instant.
- Fetch, gunzip (DecompressionStream) and decode run in module workers. Requests are prioritised
  (coarse levels first, then larger error), stale ones are dropped, and transient errors retry with backoff.
- Meshes are merged per tile per role, and every object carries `userData.role` (format.js `ROLES`) and
  `userData.lookRole`. `object.position` is in world units (the tile centre).
- Cross-fade: when the LOD changes, the outgoing and incoming tiles dissolve over `fadeMs` (350 ms) with
  complementary 4×4 Bayer screen-door patterns, so there is no blending or sorting and no pixel is drawn
  twice. The dither is injected once into every tile material, including the look's ShaderMaterials
  (`injectDither`). A fading tile draws with a shared clone of its material (same program, so no compile).
  Pass `now: () => clockMs` for frame-exact capture, or `fadeMs: 0` for an instant swap.
- LRU eviction runs above the budget. A tile that was touched this frame is never evicted. When the working
  set is larger than the budget, refinement stops (coarser tiles stay on screen) instead of thrashing.
- Elevation comes from the node `y` values (bridges, ramps), plus the few-cm class epsilons of `src/stream/classStyle.js`.

## Cars (`cars.js`)

```js
import { createCarLOD } from './src/stream/cars.js';
const cars = createCarLOD({ scene, camera, renderer, maxCars: 100000, nearDistance: 700, minPixelSize: 2 });
cars.update(positions /* xyz per car */, headings /* atan2(dz,dx) */, count, { colors });
cars.updateFromSim(env.sim);                  // CPU: src/sim Simulation (sim.vs: x, y, z, heading, state, typeIdx, length, width, brake, flags)
cars.updateFromPoses(await gpuSim.readPoses());   // GPU: vec4 x, y, z, heading per slot (y = -1000 free); pipeline it
```

`world.html?sim=gpu&n=100000` shows the WebGPU sim.

Cars within `nearDistance` are drawn as clay boxes in one InstancedMesh (`name 'cars'`, so the look
styles it). The other cars are drawn as one `THREE.Points` draw, where each dot is the car's projected
length, never smaller than `minPixelSize`. That is two draw calls in total.

## Viewer

`world.html` (`src/viewer/world.js`): fly over the whole bay. Presets 1–9, T tour, L colours tiles by level,
F freezes the selection, WASD/QE free fly. URL: `?view=<preset>`, `?cam=lat,lon,h,d,az`, `?look=off|perf`,
`?sse=`, `?budget=`, `?sim=gpu&n=`.
