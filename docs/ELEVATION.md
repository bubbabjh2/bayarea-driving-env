# Terrain elevation

Every road, signal, station and ground cell has a real height from the USGS 3D Elevation Program (3DEP) bare-earth
DEMs. The sim reads road heights from the road packs (`data/roads*.bin`). Road grade feeds the vehicle physics:
gravity, axle-load shift and hill-hold. The ground grids are for rendering and for anything else that needs terrain
away from the roads.

Elevation data: **U.S. Geological Survey, 3D Elevation Program**, public domain (U.S. Government work).
Road geometry: © OpenStreetMap contributors, ODbL 1.0.

## Coordinate frame

- **Project metres**, an equirectangular projection about lat0 = 37.55 N, lon0 = −122.15 (`src/proj.js`, which is the
  same code as `tools/build_roads.py`). It uses Earth radius 6,371,008.8 m:
  `x = (lon − lon0)·kx` (+x = east), `z = −(lat − lat0)·kz` (+z = south), `kz = R·π/180`, `kx = kz·cos(lat0)`.
- **y is up**, in metres **NAVD88** (orthometric, bare earth). Sea level / the bay water plane is y = 0. Land is never
  below 0.5 m on roads and 0.4 m on the ground grid.
- Heading in the sim is `atan2(dz, dx)`; `src/sim/LAYOUT.md` has the details.

## Sources (pinned in `tools/elevation/dem-sources.json`)

| product | tiles | resolution | used for |
|---|---|---|---|
| 3DEP 1/3 arc-second seamless DEM | `USGS_13_n38w122`, `n38w123`, `n39w122`, `n39w123` (NAD83 geographic) | ~10 m | everywhere |
| 3DEP 1 m lidar DEM, project `CA_SanFrancisco_B23` | `x54y418`, `x54y419`, `x55y418`, `x55y419` (UTM 10N) | 1 m, cached at 2 m | San Francisco |

The manifest pins the URL, byte size and sha256 of each file, plus its `Last-Modified` time and the S3 md5 ETag for
the 1 m tiles. The files were downloaded on 2026-10-24. USGS serves the 1/3" tiles under a moving `current/` path, so
the sha256 is the real pin. If USGS ever replaces a tile, `fetch_dem.mjs` stops with a checksum mismatch instead of
silently building different terrain.

Check on 2026-11-20:
- All 8 URLs return HTTP 200 with the pinned Content-Length.
- Every Last-Modified date is before 2026-10-24.
- The four 1 m ETags equal the md5 of the pinned files.
- `USGS_1M_10_x55y419_CA_SanFrancisco_B23.tif` was downloaded in full and its sha256 matches.

The other Bay Area 1 m projects (Alameda 2021, Contra Costa 2022, Santa Clara 2020, CaliforniaGaps 2023) are not used.
In those areas the 1/3" product is already derived from lidar.

## Rebuild

```
npm run build:elevation                 # = node tools/elevation/build_all.mjs   (~2.1 GB download, ~12 GB RAM)
npm run build:elevation -- --from DIR   # copy the tiles from a local cache (DIR/13/*.tif, DIR/1m/*.tif) instead
npm run build:elevation -- --install    # also install the rebuilt roads / stations / controls into data/
npm run verify:elevation                # sha256 of every output vs tools/elevation/expected.sha256
```

Steps (`tools/elevation/build_all.mjs`):

1. `fetch_dem.mjs` downloads to `data/dem/raw/` (gitignored). It resumes partial downloads and checks every sha256.
2. `build_dem.mjs` writes `data/dem/ground/g13.f32`, a 1/3" mosaic (13500 × 10800 Float32). It also writes
   `g1m.f32`, the SF lidar averaged into 2 m cells and cross-faded into g13 over the last 150 m before its no-data edge,
   and `ground.json`. The sampler is `tools/elevation/ground.mjs` (`loadGround().heightAt(x, z)`).
3. The committed flat-ground road snapshot `data/src/flat/` is unpacked to `data/build/elev/flat/`. This snapshot is
   the OSM build `tools/build_roads.py` before terrain (elevation model `structures/2`: ground roads at y = 0, with
   bridge, tunnel and ramp offsets).
4. `build_heights.mjs` produces road heights model `terrain/1`, written to `data/build/elev/out/roads*`:
   - Ground nodes take a robust DEM median over a 4 m disk.
   - Edge profiles are sampled every 4 m, then median and Gaussian filtered.
   - Vertical shape points are inserted by Douglas–Peucker (219,236 synthetic nodes). Node and edge ids, lengths and s
     are unchanged.
   - Bridge and tunnel decks span between their abutments.
   - Grades are capped per class: 8 % motorway up to 42 % local.
   - Crossings keep ≥ 6 m clearance.
   - `--audit DIR` (optional) also writes `heights.json` with grade statistics and outliers.
5. `patch_points.mjs` sets each traffic control's and gas station's y to its road node's height.
6. `build_conform.mjs` writes `data/dem/ground/ground10.{f32,mask,json}`, a 10 m render ground on the 102.4 km root
   square (x0 −53000, z0 −57900):
   - At-grade road corridors are set to the road surface − 0.15 m, feathered over 12 m.
   - Bridges and tunnels are not conformed.
   - Water (`data/water.json`) is exactly 0.
   - The mask bits are road 1, feather 2, bridge deck 4, tunnel 8 and water 16.
7. `build_compact.mjs` writes the small committed copy (below).
8. `verify.mjs` checks the outputs against the pinned hashes.

`data/controls.*` in the repo is a later stage. `tools/build_junctions.mjs` recomputes stop lines from the junction
build, taking the patched controls from step 5 as its input. Run it after `--install` to get the committed controls.
`data/crossings.json`, `data/junctions.bin` and `data/lines*.bin` were also regenerated from the terrain roads (see the
road-data pipeline docs).

## Committed compact ground (`data/dem/compact/`)

This copy lets a fresh clone have terrain without the 2 GB download:
- `ground20.dh16.gz`: 5120 × 5120 cells of 20 m, each the mean of a 2 × 2 block of ground10. Stored as Int16
  decimetres, row-delta coded and gzipped.
- `ground20.mask.gz`: the OR of the four ground10 mask bits.
- `ground20.json`: the grid definition and the measured error against ground10.

Read it with `tools/elevation/compact.mjs`: `loadCompactGround().heightAt(x, z)` and `maskAt(x, z)`. Use the full
`ground10` or `g13` / `g1m` grids (rebuild above) when you need more than 20 m detail. The road packs carry the
full-resolution road heights either way.

## Reproducibility check

Run on 2026-11-20 in this repo (`npm run build:elevation`, tiles via `--from` a cache, then a separate live download
of all 8 tiles into a scratch folder): **29 of 29 outputs match** the original project's 2026-10-24 build bit for bit
(`tools/elevation/expected.sha256`). For JSON metadata, only the `generated` / `heightsGenerated` timestamps are
excluded, plus `roads` in ground10.json, which is the input path.

- DEM caches: `g13.f32`, `g1m.f32` and `ground.json` match.
- Ground: `ground10.f32`, `ground10.mask` and `ground10.json` match.
- Roads: `roads.bin`, `roads-local.bin`, `roads*.json` and `roads*.meta.json` match. The committed `data/roads*` packs
  equal the rebuild.
- `stations.*` match the committed files.
- `controls.*` from `patch_points.mjs` match the original intermediate (pre-junction-fix) controls. The committed
  `data/controls.*` is the later `tools/build_junctions.mjs` output.
- Live download: all 8 USGS URLs re-downloaded on 2026-11-20 reproduce the pinned sha256.
- Compact grid: 20.4 MB heights + 1.0 MB mask. Error against ground10 (bilinear at every 10 m centre) is
  RMS 0.57 m overall and 0.67 m on land, max 45.8 m at cliffs and steep cuts.

The build is deterministic: there is no randomness, and everything is processed in a fixed order. Peak memory is about
12 GB (`ELEV_HEAP_MB` sets the node heap). It takes about 3 minutes on an M-series Mac with the tiles on disk.
