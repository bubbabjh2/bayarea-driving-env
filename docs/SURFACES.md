# Road surfaces and tyre-road friction

Every edge of the road graph has a **pavement surface** and, for each side, a **ground class**: what a wheel
touches once it is past the paved width. The physics scales tyre friction and rolling resistance by the surface
under the wheels. The CPU (`src/sim/physics/dynamics.js`) and the GPU (`src/gpu/kernels/physics.wgsl`,
`integrate.wgsl surfFactors`) use the same table. The same codes are meant for renderers, so a concrete
road can be drawn as concrete and a verge as grass.

## Data: `data/surfaces.bin` + `data/surfaces.meta.json`

Built by `npm run build:surfaces` (`tools/surfaces/build.mjs`, deterministic; sha256 in
`tools/osm/world-manifest.json`, checked by `npm run verify:world`). `build:world-derived` runs it whenever the raw
snapshots are present. Inputs:
* the pinned raw Overpass snapshots `data/osm/raw/*.json` (`node tools/osm/fetch.mjs`, locked in
  `tools/osm/osm-lock.json`): each way's `surface=*` and `bridge=movable` tags, matched to edges by the OSM `way` id;
* `data/roads.json`, `data/roads-local.json` (`npm run unpack`): class, bridge / tunnel, the `sidewalk` side tags;
* `data/water.json` (the shoreline).

Layout: **2 bytes per edge**. Main-tier edges come first, then local-tier edges. This is the SimGraph edge order, so
local edge `k` is record `main.edgeCount + k`.

| byte | bits | meaning |
|---|---|---|
| 0 | 0-3 | pavement surface code |
| 0 | 4 | 1 = from the OSM `surface` tag, 0 = default |
| 0 | 5 | 1 = the edge's way is not in the raw snapshot (the flat road snapshot came from a different Overpass mirror, README "Rebuilding the world"), so the default was used |
| 1 | 0-3 | ground code right of the edge (driver's right) |
| 1 | 4-7 | ground code left of the edge (one-way: the median side; two-way: beyond the opposing lanes) |

Reading it: `src/sim/surface.js attachSurfaces(graph, bytes)` sets `e.surf`, `e.gndR` and `e.gndL`.
`loadWorld` does this automatically when the file matches the packs. A renderer can index the file directly with
`(tier === 'main' ? i : mainEdgeCount + i)`.

### Codes

| code | name | OSM `surface=` values | as ground class |
|---|---|---|---|
| 0 | asphalt | asphalt, asphalt:lanes, paved, chipseal, tarmac | |
| 1 | concrete | concrete, concrete:plates, concrete:lanes | |
| 2 | paving_stones | paving_stones, brick, bricks | |
| 3 | sett | sett, cobblestone, unhewn_cobblestone | |
| 4 | gravel | gravel, fine_gravel, pebblestone | freeway median (one-way carriageway, left) |
| 5 | compacted | compacted | |
| 6 | dirt | unpaved, dirt, ground, earth, mud | rural lanes (`unclassified`), unpaved roads |
| 7 | grass | grass | freeway verge (right), streets with `sidewalk=no` |
| 8 | sand | sand | at-grade edges within 25 m of the water polygons (shore / rip-rap) |
| 9 | metal_grate | metal, metal_grid, steel | |
| 10 | wood | wood | |
| 11 | curb | (ground only) | concrete gutter, curb and sidewalk: sidewalk tagged on that side, or untagged urban streets |

Codes 12 to 15 are unused.

Defaults when a way has no surface tag: asphalt; `bridge` / `tunnel` concrete; `bridge=movable` metal_grate (the
Bay Area's movable bridges in the data, 3rd St, Illinois St and Park St, are tagged `metal`, which is their open
steel-grid deck). Ground on bridges and tunnels is the deck surface itself.

Coverage (`surfaces.meta.json coverage`). These are directed edges, so a two-way road counts twice.

| tier | edges | OSM surface tag | defaulted | of which the way is not in the snapshot |
|---|---|---|---|---|
| main | 51,011 | 25,623 (50.2 %) | 25,388 (49.8 %) | 194 |
| local | 240,144 | 83,879 (34.9 %) | 156,265 (65.1 %) | 358 |

Result: main is 48,240 asphalt, 2,764 concrete (913 tagged, the rest bridges / tunnels), 4 metal_grate and 3
paving_stones. Local is 238,166 asphalt, 1,601 concrete, 154 paving_stones, 142 dirt, 60 gravel, 13 sett, 6 wood and
2 compacted. `paved` (777 main, 2,997 local edges) is mapped to asphalt.

**Not modelled.** These need OSM layers that the pinned queries do not fetch (they only fetch `highway=*` ways):
embedded rails (SF cable car lines on Powell / Hyde / Mason / California St, Muni streetcar tracks), lane-level
surfaces (crosswalk paint, steel plates) and `smoothness`.

## Friction table (`src/sim/surface.js SURFACES`)

The coefficients below are tyre-road adhesion values for a passenger-car tyre. The vehicle types' own `mu`
(`params.js`: 1.0 for sedan / SUV, 0.8 for truck / bus, calibrated on dry asphalt) is scaled by the ratio to
dry asphalt:

* peak friction (friction circle, ABS / traction limit): `mu_eff = P.mu · pk`, where `pk = muPeakDry / 0.85`
* sliding friction (locked wheels of a disabled car): `P.mu · sk · Fz`, where `sk = pk · round2(muSlideDry / muPeakDry)`.
  On asphalt `sk = 0.8`, the model's existing kinetic ratio.
* rolling resistance: `Crr_eff = P.Crr · crr / 0.013`

| surface | muPeakDry | muSlideDry | muPeakWet | muSlideWet | crr | pk | sk | crr × | source |
|---|---|---|---|---|---|---|---|---|---|
| asphalt | 0.85 | 0.68 | 0.60 | 0.50 | 0.013 | 1.000 | 0.800 | 1.00 | [W] T1.3: dry 0.8–0.9 / 0.75, wet 0.5–0.7 / 0.45–0.6; [W] T1.1 crr 0.013 |
| concrete | 0.85 | 0.68 | 0.80 | 0.70 | 0.013 | 1.000 | 0.800 | 1.00 | [W] T1.3: dry 0.8–0.9 / 0.75, wet 0.8 / 0.7; [W] T1.1 |
| paving_stones | 0.75 | 0.60 | 0.55 | 0.45 | 0.015 | 0.882 | 0.706 | 1.15 | estimate: below asphalt (smooth units, joints) |
| sett | 0.65 | 0.52 | 0.45 | 0.35 | 0.020 | 0.765 | 0.612 | 1.54 | estimate: polished stone setts, uneven |
| gravel | 0.60 | 0.55 | 0.55 | 0.50 | 0.025 | 0.706 | 0.649 | 1.92 | [W] T1.3 gravel 0.6 / 0.55; [W] T1.1 crr between rolled gravel 0.02 and tarmacadam 0.025 |
| compacted | 0.65 | 0.58 | 0.55 | 0.45 | 0.020 | 0.765 | 0.681 | 1.54 | estimate between gravel and earth road; [W] T1.1 rolled gravel 0.02 |
| dirt | 0.68 | 0.65 | 0.55 | 0.45 | 0.050 | 0.800 | 0.768 | 3.85 | [W] T1.3 earth road dry 0.68 / 0.65, wet 0.55 / 0.4–0.5; [W] T1.1 unpaved road 0.05 |
| grass | 0.45 | 0.40 | 0.25 | 0.20 | 0.100 | 0.529 | 0.471 | 7.69 | estimate (vegetation on soil); [W] T1.1 field 0.1–0.35 |
| sand | 0.40 | 0.35 | 0.40 | 0.35 | 0.300 | 0.471 | 0.409 | 23.08 | estimate; [W] T1.1 field / loose ground up to 0.35 |
| metal_grate | 0.55 | 0.45 | 0.30 | 0.25 | 0.013 | 0.647 | 0.531 | 1.00 | estimate: open steel-grid decks have markedly lower skid resistance than pavement, especially wet |
| wood | 0.60 | 0.50 | 0.40 | 0.30 | 0.015 | 0.706 | 0.586 | 1.15 | estimate |
| curb | 0.85 | 0.68 | 0.80 | 0.70 | 0.013 | 1.000 | 0.800 | 1.00 | = concrete |

[W] J. Y. Wong, *Theory of Ground Vehicles*, 4th ed., Wiley, 2008: Table 1.1 (coefficient of rolling resistance)
and Table 1.3 (average peak and sliding coefficients of road adhesion). Rows marked "estimate" have no single
standard value. They were chosen relative to Wong's values for the nearest surface.

**The wet columns are documentation only.** Friction does not change with weather: the sim has no weather or wet
state, and the physics always uses the dry columns.

## Physics: which surface is under the car

`carSurface(graph, e, s, lat, hw, conn, taper)` (CPU) and `surfFactors` (GPU) look at the right and left wheel
lines (`lat ∓ hw`, the half body width). A wheel line reads the pavement code while it is inside the paved width,
and its side's ground code once it is past it. The paved width is:
* right: the pavement edge (`LaneGeometry.bounds`, including lane add / drop run-outs), plus the 3 m paved shoulder
  on freeway-class roads;
* left: two-way roads, the pavement edge plus the opposing lanes; one-way roads, the pavement edge plus the 3 m
  shoulder on freeway-class roads.

Inside a junction box (on a connector, or before `sExit + 1` / past `sStop` of an edge) everything is pavement.
The factors are averaged over the two sides, so a car with its right wheels on the verge has half-grass friction.
Both axles use the same factors. `vs.surfR` / `vs.surfL` record the codes of the last step.
The road-edge barriers (README "API") still stop a car about 1 m (surface streets) or 3 m (freeways) past the
lanes, so the ground class acts in that strip and wherever the barriers are off (`trafficMode: 'empty'`).

On dry asphalt every factor is exactly 1 (`sk` 0.8), so the physics there is bit-identical to the single-mu model.
A graph without `surfaces.bin` (synthetic scenarios, or a pack mismatch) is all asphalt.

## Results (`test/sim/surface.test.js`, `test/gpu/physics.html`)

Sedan, full brake from 100 km/h, level road (ABS-limited, aero and rolling included):

| surface | stop (m) | / asphalt | 1 / pk |
|---|---|---|---|
| asphalt, concrete, curb | 41.2 | 1.00 | 1.00 |
| paving_stones | 45.6 | 1.11 | 1.13 |
| dirt | 48.2 | 1.17 | 1.25 |
| sett, compacted | 52.0 | 1.26 | 1.31 |
| sand | 55.1 | 1.34 | 2.13 (rolling drag ~0.3 g also brakes) |
| gravel | 56.7 | 1.38 | 1.42 |
| wood | 57.2 | 1.39 | 1.42 |
| metal_grate | 61.7 | 1.50 | 1.55 |
| grass | 66.4 | 1.61 | 1.89 |

The ratio stays a little under `1/pk` because aero and rolling drag do not scale with mu. The GPU matches the CPU on
braking, lane changes and locked slides on gravel, grass, sand and metal_grate for every vehicle type (worst relative
difference 1.5e-4). `test/gpu/equivalence.html?surf=mixed` puts a mixed surface map on the synthetic graph and
passes: 97 % of cars are within 0.5 m after 1000 steps.
