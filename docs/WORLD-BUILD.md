# Rebuilding the world data

Map data © OpenStreetMap contributors, ODbL 1.0 (https://www.openstreetmap.org/copyright). Everything under `data/`
derived from OSM (roads, controls, crosswalks, crossings, water, junctions, lane lines) is a Derivative Database under
the ODbL. Elevation: U.S. Geological Survey, 3D Elevation Program (public domain), see the elevation section of the README.

## What is committed and what is built

| file | committed | produced by | needs |
|---|---|---|---|
| `data/water.json` | yes | `tools/build_water.py` | OSM `coast` |
| `data/roads{,-local}.{bin,meta.json}`, `.json.gz` | yes | `tools/build_roads.py` (flat) → elevation step (`build_heights`) | OSM roads + local tiles, `water.json`, DEM |
| `data/stations.*` | yes | `tools/build_stations.py` (synthetic, seed 7) → elevation `patch_points` | roads |
| `data/controls.{bin,meta.json}`, `.json.gz` | yes | `tools/build_controls.py` → `patch_points` (y) → `build_junctions.mjs` (stop lines) | OSM controls + road inputs |
| `data/crosswalks.json` | yes | `tools/build_crosswalks.py` | OSM `crossings` |
| `data/crossings.json` | yes | `tools/build_crossings.py` | terrain roads JSON |
| `data/junctions.{bin,meta.json}` | **no** (198 MB) | `tools/build_junctions.mjs` (~90 s, ~10 GB heap) | roads, controls, crosswalks |
| `data/lines.{bin,meta.json}`, `data/lines-geom.bin` | **no** (24 + 60 MB) | `tools/lines/build_lines.mjs` (~7 s) | roads, junctions, controls, crosswalks |

The sim builds the junctions in memory when `data/junctions.bin` is missing (slow on every load), so run the
derive step once after cloning:

```sh
node tools/osm/unpack.mjs                         # data/*.json.gz -> data/*.json (the tools read the JSON copies)
python3 -I tools/build_crossings.py data          # optional: already committed
node --max-old-space-size=14000 tools/build_junctions.mjs data/
node --max-old-space-size=14000 tools/lines/build_lines.mjs data
node tools/osm/verify.mjs                         # sha256 against the original project's files
```

## Full rebuild from OSM (pinned snapshots)

The original raw Overpass responses were not kept. The inputs are pinned instead as Overpass **attic** queries: each
`tools/queries/<name>.ql` is sent with `[date:"…"]` from `tools/osm/sources.json` (just after the original downloads:
road, local and coast queries 2026-10-20T04:00Z, controls 06:30Z, crosswalks 14:00Z). `tools/osm/fetch.mjs` saves them to
`data/osm/raw/` (gitignored), records the sha256 of each response's element list in `tools/osm/osm-lock.json` and
checks it on later fetches. It runs the queries one at a time with a pause, and backs off when the server is busy.
Override the server with `OVERPASS_URL` (it must support attic queries; overpass-api.de does).

```sh
bash tools/osm/build_world.sh all     # = osm, npm run build:elevation, derive
```

Steps in the original order:

1. `node tools/osm/fetch.mjs`
2. `python3 -I tools/build_water.py --bbox 37.20,-122.62,38.02,-121.72 data/osm/raw/coast.json data/water.json`
3. `python3 -I tools/build_roads.py --out-dir data/osm/flat --water data/water.json <roads> <roads-north> <local-tile1 … local-tile7>`
   (default corridor bbox 37.25,-122.56,37.97,-121.80 minus the Tri-Valley and Lamorinda boxes). Output: the
   flat-ground `structures/2` build.
4. `python3 -I tools/build_stations.py --out-dir data/osm/flat` (synthetic stations, seed 7; adds `stations` to roads.meta.json)
5. `python3 -I tools/build_controls.py --data data/osm/flat <controls-signals> <controls-signs> -- <same road inputs>`
6. `python3 -I tools/build_crosswalks.py data/osm/raw/crossings.json data`
7. elevation (README, `npm run build:elevation`): `build_heights` (flat → terrain roads in `data/`), `patch_points`
   (controls + stations y)
8. `python3 -I tools/build_crossings.py data`
9. `node tools/build_junctions.mjs data/` (also writes the junction-mouth stop lines into `data/controls.*`)
10. `node tools/lines/build_lines.mjs data`

`node tools/osm/verify.mjs --flat data/osm/flat` compares every output with the original project's hashes
(`tools/osm/world-manifest.json`; `flat/*` = the flat build of step 3–5).

## Verification against the original project (2026-11-20)

Deterministic steps, rebuilt in a scratch copy of this repo's `data/` and hashed against the original:

| output | result |
|---|---|
| `junctions.bin` (198 MB) | **bit-exact** (`build_junctions.mjs`, 84 s; controls stop lines patched 0, controls unchanged) |
| `lines.bin`, `lines-geom.bin` | **bit-exact** (`build_lines.mjs`, 6 s) |
| `crossings.json` | **bit-exact** (`build_crossings.py`, 7 s) |
| `junctions.meta.json`, `lines.meta.json` | equal except `generated` / `buildMs` |

From the pinned OSM attic snapshots (overpass-api.de, fetched 2026-11-20):

| output | result |
|---|---|
| `water.json` (coast @ 2026-10-20T04:00Z) | **bit-exact** |
| `crosswalks.json` (crossings @ 2026-10-20T14:00Z) | **bit-exact** (66,982 points) |
| flat `roads*` (structures/2) | **not exact**: main tier 50,287 vs 51,011 edges, local 240,962 vs 240,144; km per class within ±4 % (motorway 1,137 vs 1,182 km) |
| flat `controls*` | not exact (follows the roads): 38,158 vs 37,865 controls; every original control OSM node is in the snapshot |
| flat `stations*` | not exact: synthetic placement (seed 7) over a different candidate list |

Why the road tier does not reproduce: the original download (2026-10-19 evening, README of the original project names
`overpass.private.coffee`) came from a mirror whose database was internally inconsistent. In `roads.ql`'s area the
original keeps node positions that OSM changed in June–July 2026 (e.g. a node at its 2013 position although v8 moved it
17 m on 2026-07-13) and also contains ways first created on 2026-07-22/23 and a motorway way deleted on 2026-08-04.
Attic probes of `roads.ql` (original nodes missing / moved in the snapshot, out of 56,544): 06-15 117/430,
07-10 117/624, 07-28 86/1,196, 08-02 86/1,204, 10-20 297/2,109. No single date matches. `local-tile3a/3b/7` match the
original node-for-node at 2026-10-20T04:00Z, `roads-north` and `local-tile4a/4b` within a few dozen nodes;
`local-tile1/2a/2b/5/6` differ by tens to hundreds of nodes (same mirror effect).

Consequence: **the canonical road input is the committed flat-ground snapshot** (`data/src/flat/`, the original
`structures/2` build). The elevation step and everything after it reproduce the original bit-exactly from it.
The OSM path above rebuilds a current-schema world from pinned public snapshots, close to but not identical with the
original; use `node tools/osm/verify.mjs --flat data/osm/flat` to see which files differ.
