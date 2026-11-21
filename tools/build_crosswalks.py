#!/usr/bin/env python3
"""
build_crosswalks.py - OSM marked pedestrian crossings -> data/crosswalks.json (projected points).

    python3 -I tools/build_crosswalks.py crossings.json [data]

Input: Overpass JSON of node[highway=crossing] (tools/queries/crossings.ql). Kept: crossing =
marked | zebra | traffic_signals | uncontrolled (painted). Unmarked crossings are skipped.
Output {"meta": {...}, "points": [[x, z, kind], ...]}, kind 0 marked/zebra/uncontrolled,
1 signalised. tools/build_junctions.mjs uses them to set junction mouths back behind crosswalks.
"""
import json
import math
import os
import sys

LAT0, LON0 = 37.55, -122.15
R = 6371008.8
KX = math.cos(math.radians(LAT0)) * R * math.pi / 180.0
KZ = R * math.pi / 180.0
KEEP = {"marked": 0, "zebra": 0, "uncontrolled": 0, "uncontrolled;marked": 0, "traffic_signals": 1}


def main(argv):
    src, data = argv[0], argv[1] if len(argv) > 1 else "data"
    d = json.load(open(src))
    pts = []
    for e in d.get("elements", []):
        if e.get("type") != "node":
            continue
        k = KEEP.get(str((e.get("tags") or {}).get("crossing", "")).lower())
        if k is None:
            continue
        pts.append([round((e["lon"] - LON0) * KX, 2), round(-(e["lat"] - LAT0) * KZ, 2), k])
    json.dump({"meta": {"format": "bay-crosswalks/1", "count": len(pts), "kinds": ["painted", "signalised"],
                        "source": "OpenStreetMap highway=crossing (ODbL)"}, "points": pts},
              open(os.path.join(data, "crosswalks.json"), "w"), separators=(",", ":"))
    print(f"crosswalks: {len(pts)} painted / signalised crossing nodes -> {data}/crosswalks.json")


if __name__ == "__main__":
    main(sys.argv[1:])
