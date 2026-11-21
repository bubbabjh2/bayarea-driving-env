#!/usr/bin/env bash
# Full rebuild of the world data from the pinned OSM snapshots (docs/WORLD-BUILD.md).
#   bash tools/osm/build_world.sh osm       # fetch + flat-ground OSM build -> data/osm/flat, data/water.json, data/crosswalks.json
#   bash tools/osm/build_world.sh derive    # after the elevation step: crossings, controls merge (tools/controls/), junctions, lane lines
#   bash tools/osm/build_world.sh all       # osm, then the elevation step (npm run build:elevation), then derive
# FLAT (default data/osm/flat) = where the flat-ground (structures/2) build goes; the elevation step reads it.
# Map data (c) OpenStreetMap contributors, ODbL 1.0.
set -euo pipefail
cd "$(dirname "$0")/../.."
RAW=data/osm/raw
FLAT=${FLAT:-data/osm/flat}
HEAP=${HEAP:-14000}
ROADS_IN="$RAW/roads.json $RAW/roads-north.json $RAW/local-tile1.json $RAW/local-tile2a.json $RAW/local-tile2b.json $RAW/local-tile3a.json $RAW/local-tile3b.json $RAW/local-tile4a.json $RAW/local-tile4b.json $RAW/local-tile5.json $RAW/local-tile6.json $RAW/local-tile7.json"

osm() {
  node tools/osm/fetch.mjs
  mkdir -p "$FLAT"
  python3 -I tools/build_water.py --bbox 37.20,-122.62,38.02,-121.72 "$RAW/coast.json" data/water.json
  python3 -I tools/build_roads.py --out-dir "$FLAT" --water data/water.json $ROADS_IN
  python3 -I tools/build_stations.py --out-dir "$FLAT"
  python3 -I tools/build_controls.py --data "$FLAT" "$RAW/controls-signals.json" "$RAW/controls-signs.json" -- $ROADS_IN
  python3 -I tools/build_crosswalks.py "$RAW/crossings.json" data
  node tools/osm/verify.mjs --flat "$FLAT"
}
derive() {
  python3 -I tools/build_crossings.py data
  # traffic controls: OSM (data/src/flat/controls.*) + authoritative merge -> patch_points -> junctions -> lines
  HEAP_MB=$HEAP node tools/controls/build.mjs --no-verify
  node tools/osm/verify.mjs
}
case "${1:-all}" in
  osm) osm ;;
  derive) derive ;;
  all) osm; npm run build:elevation; derive ;;
  *) echo "usage: $0 osm|derive|all"; exit 2 ;;
esac
