#!/usr/bin/env bash
# Fetch the spot-check intersections (*.ql next to this script) from the public Overpass API, gently:
# one request at a time, 5 s apart, exponential back-off when the server is busy. Outputs data/build/spotcheck/<name>.json.
#   bash tools/controls/spotcheck/fetch.sh && python3 -I tools/controls/spotcheck/compare.py
set -uo pipefail
cd "$(dirname "$0")/../../.."
OUT=data/build/spotcheck; mkdir -p "$OUT"
URL=${OVERPASS_URL:-https://overpass-api.de/api/interpreter}
for q in tools/controls/spotcheck/*.ql; do
  n=$(basename "$q" .ql); f="$OUT/$n.json"
  python3 -c "import json,sys; assert json.load(open(sys.argv[1]))['elements']" "$f" 2>/dev/null && continue
  for t in 15 30 60 120; do
    curl -s --max-time 90 -A "bayarea-driving-env-spotcheck/1.0" --data-urlencode "data@$q" "$URL" -o "$f"
    python3 -c "import json,sys; assert json.load(open(sys.argv[1]))['elements']" "$f" 2>/dev/null && break
    sleep $t
  done
  echo "$n $(wc -c < "$f") bytes"; sleep 5
done
