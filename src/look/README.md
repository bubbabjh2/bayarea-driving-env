# Look pack (`src/look/`)

Materials, lighting, sky, landscape and traffic-control heads for the viewer (`world.html`). Render only: nothing in
`src/sim`, `src/gpu`, `src/nav` or `src/geom` imports this folder (`test/render-isolation.test.js`).

All road, paint, concrete, grass and water textures are **procedural**: generated on canvases or in GLSL at
`applyLook()` time (`textures.js`, `materials.js`), about 10 MB of GPU textures. No image files are downloaded. The
landscape textures (`data/landscape/*.png`) are rasterised by `tools/landscape/` from OSM land cover, DataSF
footprints and USGS 3DEP elevation (see NOTICE).

```js
import { applyLook } from './src/look/index.js';
const look = applyLook(scene, renderer, { preset: 'video' });   // or 'perf'
const stream = createStream({ ..., onTileMesh: (m) => look.onTileMesh(m), onTileDispose: (m) => look.onTileDispose(m) });
```

## Handle

| member | use |
|---|---|
| `setTimeOfDay(h)` | 5.5 (dawn) … 10. Sun direction and colour, sky, fog. Default 7.6. |
| `setClock(sec)` | Water animation clock (`null` = wall time). |
| `setSimTime(sec)` | Sim clock for the signal lamps (default: time of day × 3600). |
| `attachSignals({ controls })` | Drive the heads from a sim's `TrafficControls` (else one is built from `data/`). |
| `onTileMesh(mesh)` / `onTileDispose(obj)` | Style a streamed tile before its first frame / release it. |
| `refresh()` | Rescan the scene (also every 30 frames). |
| `terrainHeight(x, z)` | Ground height (m), the same function the shaders use. |
| `setSky(name)` | `clear` / `golden` / `overcast`. |
| `landscape.settled()` | Resolves when no landscape tile is loading or fading. |
| `info()` / `remove()` | What was matched / restore the original materials. |

URL toggles (any page that calls `applyLook`): `landscape=basic|full`, `terrain=auto|0|1`, `buildings=0|1`,
`trees=0|1`, `structures=0|1`, `sky=clear|golden|overcast`.

## Material roles

Objects are matched by `userData.lookRole`, then by `name` (the stream tiles set both, `src/stream/format.js`):

| role | look |
|---|---|
| `main-asphalt`, `road-strip`, `road-lines` | Asphalt shader: canvas texture in world xz (8 m + 32 m tiles), macro wear, repair patches, low-sun sheen; class tone (freeway darker); elevated decks blend to concrete. Far roads become anti-aliased line quads. |
| `local-asphalt`, `junction` | Residential asphalt: lighter, per-block colour drift, utility patches, crack-seal lines, manhole/valve covers. |
| `sidewalk`, `driveway` | Concrete slabs (1.5 m / 3 m) with saw-cut joints, per-slab tone and stains. |
| `curb` | Light concrete, bright top, darker face. |
| `verge`, `planting` | Irrigated lawn with mowing patches and dry spots. |
| `marking-*` (`white`, `yellow`, `dash`, `crosswalk`, `stopbar`) | Lit paint (`yellow`/`center` names are yellow). |
| `signals` | Placeholder posts by control type; replaced by live heads (`signals.js`) near the camera. |
| `stations` | Light canopy with a teal glow. |
| `water` | Look water shader (in full landscape mode the landscape surface draws the water). |
| `cars` (InstancedMesh) | Clay material, darker underside, brake/crash colours, blob contact shadows. |
| page lights, background, fog | Replaced by `look-lights`, the `look-sky` dome and height-scaled haze. |

## Landscape (`landscape.js`, `buildings.js`, `structures.js`, `terrain-glsl.js`, `landscape-worker.js`)

Built by `npm run build:landscape` into `data/landscape/` (gitignored, ~112 MB):

* **Ground:** one CDLOD terrain mesh (75 m to 307 km) on the road-conformed ground (`data/dem/ground/ground10.f32`
  from `npm run build:elevation`, else the committed 20 m compact grid), sampled with Catmull-Rom so the ground passes
  through the samples the roads were conformed to. Water is part of the same surface (shallow-to-deep gradient,
  anti-aliased shoreline, sky reflection, sun glint).
* **Land cover:** ~157k OSM polygons in 27 classes (parks, forest, scrub, golf, farmland, residential, industrial,
  airports, salt ponds, wetlands, …) rasterised into 3.2 km tiles at 6.45 m/px plus a 25 m global texture; open land is
  tinted by elevation and aspect; class edges are reconstructed crisp when magnified.
* **Buildings:** all of San Francisco from DataSF footprints with lidar heights, pushed out of the road clearance band;
  drawn within ~3.5 km. Other areas: roof tint in the land-cover texture only.
* **Trees:** DataSF street trees in SF, OSM tree nodes and a seeded scatter by land cover; never on roads, roofs or
  water; drawn within ~1.7 km.
* **Structures:** deck girders under elevated segments, pier bents every ~38 m, and a Golden Gate Bridge model aligned
  to the road deck.

## Traffic controls (`signals.js`)

Signal heads (with a protected-left arrow column where the plan has one), ramp meters, STOP/YIELD plates on poles with
mast arms, within ~750 m of the camera, driven by `TrafficControls.stateOf` from `src/sim/controls.js` (the sim's own
controller, a pure function of the sim clock). Shapes are generic, not surveyed.

## Files

`index.js` (applyLook, presets, role matching, time of day), `materials.js` (GLSL and material factories),
`textures.js` (canvas generators: asphalt 1024²/512², grass 512², coast masks, town-density mask), `landscape.js`,
`buildings.js`, `structures.js`, `terrain-glsl.js`, `landscape-worker.js`, `signals.js`.
