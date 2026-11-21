#!/usr/bin/env python3
"""
build_stations.py - place synthetic gas stations on the road graph.

    python3 -I tools/build_stations.py [--count 250] [--seed 7] [--out-dir data]

Reads data/roads.meta.json (+ roads.bin) and, if present, data/roads-local.*.
Stations sit on JUNCTION nodes (nodes with outgoing edges, so a car can route
to them) of primary / secondary / tertiary roads, never on a node that touches
a motorway, motorway_link, trunk or trunk_link.  Density is boosted within
800 m of a motorway_link node (freeway interchanges) and stations are at
least 1.5 km apart.  Deterministic for a given seed.

Writes:
    data/stations.json        {meta, stations:[{id, tier, node, osmNode, x, z, y, pumps, nearInterchange}]}
    data/stations.bin         Float32 stationXYZ x3 | Uint32 stationInfo x4 [nodeIndex, tierId, pumps, flags]
    data/stations.meta.json   section offsets (same convention as roads.meta.json)
and adds "stations": "stations.meta.json" to data/roads.meta.json.
"""
import json
import math
import os
import random
import sys
from array import array

STATION_CLASSES = {"primary", "secondary", "tertiary"}
FORBIDDEN = {"motorway", "motorway_link", "trunk", "trunk_link"}
INTERCHANGE_R = 800.0
MIN_SPACING = 1500.0
TIER_ID = {"main": 0, "local": 1}
FLAG_NEAR_INTERCHANGE = 1


def load_pack(meta_path):
    meta = json.load(open(meta_path))
    blob = open(os.path.join(os.path.dirname(meta_path), meta["file"]), "rb").read()
    S = {}
    for name, s in meta["sections"].items():
        a = array({"Float32": "f", "Uint32": "I"}[s["type"]])
        a.frombytes(blob[s["byteOffset"]:s["byteOffset"] + s["byteLength"]])
        if sys.byteorder != "little":
            a.byteswap()
        S[name] = a
    return meta, S


class Grid:
    def __init__(self, cell):
        self.cell = cell
        self.cells = {}

    def key(self, x, z):
        return (int(math.floor(x / self.cell)), int(math.floor(z / self.cell)))

    def add(self, x, z, payload=None):
        self.cells.setdefault(self.key(x, z), []).append((x, z, payload))

    def near(self, x, z, r):
        kx, kz = self.key(x, z)
        n = int(math.ceil(r / self.cell))
        r2 = r * r
        for i in range(kx - n, kx + n + 1):
            for j in range(kz - n, kz + n + 1):
                for px, pz, p in self.cells.get((i, j), ()):
                    if (px - x) ** 2 + (pz - z) ** 2 <= r2:
                        yield p


def candidates_from_tier(tier, meta, S, osm_ids):
    """Yield (tier, nodeIndex, osmId, x, z, y) for eligible junction nodes."""
    classes = meta["classesAll"]
    stride = meta["edgeStride"]
    F = {n: i for i, n in enumerate(meta["edgeFields"])}
    xyz, edges, poly = S["nodeXYZ"], S["edges"], S["poly"]
    N, E = meta["nodeCount"], meta["edgeCount"]
    eligible = [False] * N
    forbidden = [False] * N
    link_nodes = []
    for e in range(E):
        o = e * stride
        cls = classes[edges[o + F["classId"]]]
        first, cnt = edges[o + F["firstNode"]], edges[o + F["nodeCount"]]
        a, b = poly[first], poly[first + cnt - 1]
        if cls in STATION_CLASSES:
            eligible[a] = eligible[b] = True
        if cls in FORBIDDEN:
            forbidden[a] = forbidden[b] = True
            for k in range(first, first + cnt):
                forbidden[poly[k]] = True
        if cls == "motorway_link":
            for k in range(first, first + cnt):
                n = poly[k]
                link_nodes.append((xyz[3 * n], xyz[3 * n + 2]))
    out = []
    for n in range(N):
        if eligible[n] and not forbidden[n]:
            out.append((tier, n, osm_ids[n] if osm_ids else None, xyz[3 * n], xyz[3 * n + 2], xyz[3 * n + 1]))
    return out, link_nodes


def main(argv):
    count, seed, out_dir = 250, 7, "data"
    i = 0
    while i < len(argv):
        if argv[i] == "--count":
            count = int(argv[i + 1]); i += 2
        elif argv[i] == "--seed":
            seed = int(argv[i + 1]); i += 2
        elif argv[i] == "--out-dir":
            out_dir = argv[i + 1]; i += 2
        else:
            print(__doc__); return 2
    rng = random.Random(seed)

    cands, links = [], []
    tiers_loaded = []
    for tier, meta_name in (("main", "roads.meta.json"), ("local", "roads-local.meta.json")):
        mp = os.path.join(out_dir, meta_name)
        if not os.path.exists(mp):
            continue
        meta, S = load_pack(mp)
        osm_ids = None
        jp = os.path.join(out_dir, meta.get("json", ""))
        if os.path.exists(jp):
            osm_ids = [n["id"] for n in json.load(open(jp))["nodes"]]
        c, l = candidates_from_tier(tier, meta, S, osm_ids)
        cands.extend(c); links.extend(l)
        tiers_loaded.append(tier)
        print(f"   {tier}: {len(c)} candidate junction nodes, {len(l)} motorway_link nodes", file=sys.stderr)

    link_grid = Grid(INTERCHANGE_R)
    for x, z in links:
        link_grid.add(x, z, True)
    near = []
    far = []
    for c in cands:
        (near if any(link_grid.near(c[3], c[4], INTERCHANGE_R)) else far).append(c)
    print(f"   {len(near)} candidates near interchanges, {len(far)} elsewhere", file=sys.stderr)

    rng.shuffle(near); rng.shuffle(far)
    placed = []
    placed_grid = Grid(MIN_SPACING)

    def try_place(c, is_near):
        if any(placed_grid.near(c[3], c[4], MIN_SPACING)):
            return False
        pumps = rng.randint(6, 12) if is_near else rng.randint(4, 10)
        placed.append({"id": len(placed), "tier": c[0], "node": c[1], "osmNode": c[2],
                       "x": round(c[3], 1), "z": round(c[4], 1), "y": round(c[5], 1),
                       "pumps": pumps, "nearInterchange": is_near})
        placed_grid.add(c[3], c[4], True)
        return True

    # ~45% of stations near interchanges first, then fill from everywhere
    target_near = int(count * 0.45)
    ni = 0
    for c in near:
        if ni >= target_near or len(placed) >= count:
            break
        if try_place(c, True):
            ni += 1
    pool = near[ni:] + far
    rng.shuffle(pool)
    near_set = {(c[0], c[1]) for c in near}
    for c in pool:
        if len(placed) >= count:
            break
        try_place(c, (c[0], c[1]) in near_set)
    n_near = sum(1 for s in placed if s["nearInterchange"])
    print(f"== placed {len(placed)} stations ({n_near} near interchanges, "
          f"{sum(s['pumps'] for s in placed)} pumps total)", file=sys.stderr)

    # ---- JSON
    doc = {"meta": {"format": "bay-stations/1", "count": len(placed), "seed": seed,
                    "rules": {"classes": sorted(STATION_CLASSES), "neverTouching": sorted(FORBIDDEN),
                              "interchangeRadiusM": INTERCHANGE_R, "minSpacingM": MIN_SPACING},
                    "tiers": tiers_loaded, "tierIds": TIER_ID,
                    "nodeDoc": "node is an index into the node table of the file named by tier (roads.bin / roads-local.bin)"},
           "stations": placed}
    jp = os.path.join(out_dir, "stations.json")
    json.dump(doc, open(jp, "w"), separators=(",", ":"))

    # ---- binary
    xyz = array("f"); info = array("I")
    for s in placed:
        xyz.extend((s["x"], s["y"], s["z"]))
        info.extend((s["node"], TIER_ID[s["tier"]], s["pumps"], FLAG_NEAR_INTERCHANGE if s["nearInterchange"] else 0))
    sections = [("stationXYZ", xyz, "Float32", 3), ("stationInfo", info, "Uint32", 4)]
    bp = os.path.join(out_dir, "stations.bin")
    sec_meta, off = {}, 0
    with open(bp, "wb") as f:
        for name, arr, typ, stride in sections:
            if sys.byteorder != "little":
                arr = array(arr.typecode, arr); arr.byteswap()
            b = arr.tobytes()
            sec_meta[name] = {"byteOffset": off, "byteLength": len(b), "type": typ,
                              "length": len(arr), "stride": stride, "count": len(arr) // stride}
            f.write(b); off += len(b)
    meta = {"format": "bay-stations-bin/1", "file": "stations.bin", "byteLength": off, "endianness": "little",
            "count": len(placed), "sections": sec_meta,
            "stationInfoFields": ["nodeIndex", "tierId", "pumps", "flags"],
            "tierIds": TIER_ID, "flagBits": {"NEAR_INTERCHANGE": FLAG_NEAR_INTERCHANGE},
            "json": "stations.json"}
    mp = os.path.join(out_dir, "stations.meta.json")
    json.dump(meta, open(mp, "w"), indent=1)

    # link from roads.meta.json
    rmp = os.path.join(out_dir, "roads.meta.json")
    if os.path.exists(rmp):
        rm = json.load(open(rmp))
        rm["stations"] = "stations.meta.json"
        json.dump(rm, open(rmp, "w"), indent=1)
    print(f"== wrote {jp}, {bp} ({off} bytes), {mp}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
