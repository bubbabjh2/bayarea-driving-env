#!/usr/bin/env python3
"""
merge.py - merge authoritative traffic-control datasets into the OSM-derived controls.

    python3 -I tools/controls/merge.py [--in data/src/flat] [--data data] [--src data/controls-src]
                                       [--report tools/controls/report/merge-report.json] [--osm-only]

Pipeline (tools/controls/build.mjs, `npm run build:controls`):
  tools/build_controls.py (OSM, on the flat-ground build: data/src/flat/controls.*)
  -> THIS (writes data/controls.{bin,json,meta.json})
  -> tools/elevation/patch_points.mjs data (control y = road node y)
  -> tools/build_junctions.mjs data/ (junction stop lines -> junction mouths, crosswalk legs)
  -> tools/lines/build_lines.mjs data
--in   the OSM controls (controls.json or controls.json.gz, controls.meta.json); never modified.
--data the road packs (roads.json, roads-local.json, roads.meta.json: x / z and edge ids are the same in the
       flat and the terrain build) and the output directory.
--osm-only  copy the OSM controls unchanged (reproduces the original OSM-only data/controls.*).

Datasets: tools/controls/sources.json. role 'merge' (licence allows redistribution: DataSF PDDL, San Jose CC BY)
are written into the data; role 'verify' (no licence stated) are only compared with, and their gaps are
reported as known missing in --report, never added.

Rules for 'merge' datasets:
  signals   An authoritative signal within MATCH_M of an OSM junction signal confirms it (source "osm+<id>").
            Unmatched: snapped to the nearest graph junction (>= 3 legs) within SNAP_M; a stop / give_way control
            on that junction is replaced (a signal wins); every incoming non-motorway leg gets an approach with
            its stop line at the junction mouth (source "<id>"). Farther than SNAP_M: not placed (reported).
            OSM signals with no authoritative signal within ORPHAN_M inside the dataset's area are kept and
            marked verified = false (city inventories omit Caltrans / neighbouring-city signals).
  stop signs (SF: one record per sign with the direction of travel it faces) A record facing direction d at
            junction J adds a stop approach on J's incoming leg whose end heading is closest to d (<= 50 deg).
            A junction whose every incoming leg (>= 3) has a stop becomes all-way. OSM stop controls with no
            record within ORPHAN_M are removed when the inventory is complete (inventoryComplete in
            sources.json: SFMTA's legislated list) and none of their legs is a numbered (state) route.
Every control gets `source` (osm | osm+<id> | <id>), `verified` (true / false / null = no merged authoritative
data covers it) and `authIds`; signals and ramp meters get `timing: "synthetic"` (no city publishes timing plans).
The binary layout (controls.bin) is unchanged.
"""
import gzip
import json
import math
import os
import shutil
import sys
from array import array
from collections import Counter, defaultdict

MATCH_M, SNAP_M, ORPHAN_M, AREA_M = 40.0, 30.0, 50.0, 300.0
TYPES = ["signal", "stop", "give_way", "ramp_meter", "crossing_signal"]
FLAG_ALL_WAY, FLAG_RAMP_METER, FLAG_MIDBLOCK = 1, 2, 4
NO_NODE = 0xFFFFFFFF
DIRS = {"NB": -math.pi / 2, "SB": math.pi / 2, "EB": 0.0, "WB": math.pi, "NE": -math.pi / 4, "NW": -3 * math.pi / 4,
        "SE": math.pi / 4, "SW": 3 * math.pi / 4}   # direction of travel -> heading atan2(dz, dx), x east, z south
KEEP = {   # which records of a dataset are controls (active signals, stop plates)
    "sf-signals": lambda p: p.get("type") in ("SIGNAL", "CALTRANS", "CALTRANS - HAWK"),
    "sj-signals": lambda p: (p.get("SHOPTYPE") or "").startswith("Traffic Signal") and (p.get("STATUS") or "").startswith("Existing"),
    "cupertino-signal-heads": lambda p: p.get("Status") == "Active",
    "cupertino-signs": lambda p: (p.get("MUTCDRegulatory") or "").startswith("R1-1") and p.get("Status") in (None, "Active"),
}


def arg(k, d):
    return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d


def load_json(p):
    if os.path.exists(p):
        return json.load(open(p))
    return json.load(gzip.open(p + ".gz"))


class Grid:
    def __init__(self, cell=50.0):
        self.c, self.g = cell, defaultdict(list)

    def add(self, x, z, v):
        self.g[(int(x // self.c), int(z // self.c))].append((x, z, v))

    def near(self, x, z, r):
        out, c = [], self.c
        for i in range(int((x - r) // c), int((x + r) // c) + 1):
            for j in range(int((z - r) // c), int((z + r) // c) + 1):
                for (a, b, v) in self.g.get((i, j), ()):
                    d = math.hypot(a - x, b - z)
                    if d <= r:
                        out.append((d, v))
        out.sort(key=lambda t: t[0])
        return out


def main():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    src_in = arg("--in", os.path.join(root, "data/src/flat"))
    data = arg("--data", os.path.join(root, "data"))
    src = arg("--src", os.path.join(root, "data/controls-src"))
    report_path = arg("--report", os.path.join(root, "tools/controls/report/merge-report.json"))
    if "--osm-only" in sys.argv:
        for f in ("controls.bin", "controls.meta.json"):
            shutil.copyfile(os.path.join(src_in, f), os.path.join(data, f))
        doc = load_json(os.path.join(src_in, "controls.json"))
        json.dump(doc, open(os.path.join(data, "controls.json"), "w"), separators=(",", ":"))
        print("osm-only: copied the OSM controls", file=sys.stderr)
        return
    sources = json.load(open(os.path.join(root, "tools/controls/sources.json")))["sources"]
    lock = json.load(open(os.path.join(root, "tools/controls/sources.lock.json")))
    doc = load_json(os.path.join(src_in, "controls.json"))
    meta_in, controls = doc["meta"], doc["controls"]
    n_osm = len(controls)
    Em = meta_in["mainEdgeCount"]

    # ------------------------------------------------------------ graph (both tiers, merged edge ids)
    rm = json.load(open(os.path.join(data, "roads.meta.json")))["projection"]
    kx, kz, la0, lo0 = rm["metersPerDegLon"], rm["metersPerDegLat"], rm["lat0"], rm["lon0"]
    proj = lambda lo, la: ((lo - lo0) * kx, -(la - la0) * kz)
    E = {}                      # merged edge id -> (tier, edge fields, end heading)
    into = defaultdict(list)    # junction key (end position, 0.1 m) -> merged edge ids ending there
    deg = Counter()
    for t, fn in (("main", "roads.json"), ("local", "roads-local.json")):
        r = load_json(os.path.join(data, fn))
        N = r["nodes"]
        for e in r["edges"]:
            g = e["id"] if t == "main" else Em + e["id"]
            a, b, f0 = N[e["nodes"][-2]], N[e["nodes"][-1]], N[e["nodes"][0]]
            E[g] = (t, {k: e[k] for k in ("id", "from", "to", "class", "length", "width", "name", "ref")},
                    math.atan2(b["z"] - a["z"], b["x"] - a["x"]))
            key = (round(b["x"], 1), round(b["z"], 1))
            into[key].append(g)
            deg[key] += 1
            deg[(round(f0["x"], 1), round(f0["z"], 1))] += 1
        del r
    junctions = Grid(40.0)
    for key, d in sorted(deg.items()):
        if d >= 3 and into.get(key):
            junctions.add(key[0], key[1], key)

    def mouth_stop(key, g):
        widest = max((E[o][1]["width"] for o in into[key] if o != g), default=E[g][1]["width"])
        return round(max(0.0, E[g][1]["length"] - (widest / 2 + 1.5)), 2)

    def legs(key):
        return [g for g in into[key] if E[g][1]["class"] != "motorway"]

    # ------------------------------------------------------------ authoritative points
    def feats(s):
        p = os.path.join(src, s["id"] + ".geojson.gz") if s["role"] == "merge" else os.path.join(src, "verify", s["id"] + ".geojson.gz")
        if not os.path.exists(p):
            print(f"  {s['id']}: no snapshot ({os.path.relpath(p, root)}), skipped (node tools/controls/fetch.mjs)", file=sys.stderr)
            return None
        return json.load(gzip.open(p))["features"]

    sig_pts, stop_pts, have = defaultdict(list), defaultdict(list), set()
    for s in sources:
        fs = feats(s)
        if fs is None:
            continue
        have.add(s["id"])
        k = KEEP.get(s["id"], lambda p: True)
        for f in fs:
            gm, p = f.get("geometry"), f.get("properties") or {}
            if not gm or gm.get("type") != "Point" or not k(p):
                continue
            x, z = proj(*gm["coordinates"][:2])
            fid = p.get("objectid") or p.get("OBJECTID") or f.get("id")
            if s["kind"] in ("signal", "signal-pole", "signal-head"):
                sig_pts[s["id"]].append((x, z, fid))
            else:
                stop_pts[s["id"]].append((x, z, fid, p.get("st_facing") or p.get("TravelDirection")))
    # pole / head datasets: one record per pole or head -> one per intersection (40 m)
    for s in sources:
        if s["kind"] in ("signal-pole", "signal-head") and sig_pts.get(s["id"]):
            cl = []
            for (x, z, fid) in sig_pts[s["id"]]:
                hit = next((c for c in cl if math.hypot(c[0] / c[2] - x, c[1] / c[2] - z) < 40), None)
                if hit is None:
                    hit = [0.0, 0.0, 0, fid]; cl.append(hit)
                hit[0] += x; hit[1] += z; hit[2] += 1
            sig_pts[s["id"]] = [(a / n, b / n, fid) for a, b, n, fid in cl]

    # ------------------------------------------------------------ index the OSM controls
    for c in controls:
        c["source"], c["verified"], c["authIds"] = "osm", None, []
    sig_grid, by_key = Grid(), {}
    for c in controls:
        if c["type"] == "signal" and not c["midBlock"]:
            sig_grid.add(c["x"], c["z"], c)
        if c["node"] != -1:
            by_key[(round(c["x"], 1), round(c["z"], 1), c["type"])] = c

    rep = {"doc": "tools/controls/merge.py: per-city coverage (merge = written into data/controls.*, verify = compared only)",
           "cities": {}, "unplaced": [], "knownMissing": [], "removed": []}
    removed = set()
    role = {s["id"]: s["role"] for s in sources}
    complete = {s["id"]: bool(s.get("inventoryComplete")) for s in sources}
    city_of = {s["id"]: s["city"] for s in sources}

    def row(sid):
        return rep["cities"].setdefault(city_of[sid], {}).setdefault(sid, Counter())

    def new_control(typ, key, apps, sid, fid):
        g0 = apps[0]
        c = {"type": typ, "tier": E[g0][0], "node": E[g0][1]["to"], "junctionOsm": None, "osmNodes": [],
             "x": key[0], "y": 0.0, "z": key[1], "allWay": False, "rampMeter": False, "midBlock": False, "crossing": False,
             "approaches": [{"edge": g, "stopAt": mouth_stop(key, g)} for g in sorted(apps)],
             "source": sid, "verified": True, "authIds": [fid]}
        controls.append(c)
        return c

    def lonlat(x, z):
        return [round(lo0 + x / kx, 6), round(la0 - z / kz, 6)]

    # ------------------------------------------------------------ signals
    for sid, pts in sig_pts.items():
        R, area, merge = row(sid), Grid(100.0), role[sid] == "merge"
        seen = set()
        for (x, z, fid) in pts:
            area.add(x, z, fid)
            R["authoritative"] += 1
            hit = sig_grid.near(x, z, MATCH_M)
            if hit:
                c = hit[0][1]
                R["confirmed"] += 1
                if merge:
                    if c["source"] == "osm":
                        c["source"] = "osm+" + sid
                    c["verified"] = True
                    c["authIds"].append(fid)
                continue
            j = junctions.near(x, z, SNAP_M)
            if not j or not legs(j[0][1]):
                R["unplaced"] += 1
                rep["unplaced"].append({"source": sid, "id": fid, "lonlat": lonlat(x, z)})
                continue
            key = j[0][1]
            if key in seen:
                continue
            seen.add(key)
            if not merge:
                R["knownMissing"] += 1
                rep["knownMissing"].append({"source": sid, "type": "signal", "id": fid})   # verify-only source: no coordinates (unlicensed data)
                continue
            for t in ("stop", "give_way"):
                old = by_key.get((key[0], key[1], t))
                if old is not None and id(old) not in removed:
                    removed.add(id(old)); R["signsReplacedBySignal"] += 1
            c = new_control("signal", key, legs(key), sid, fid)
            sig_grid.add(key[0], key[1], c)
            R["added"] += 1
        for c in controls:
            if c["type"] != "signal" or c["midBlock"] or id(c) in removed or c["authIds"] or c["source"] != "osm":
                continue
            if area.near(c["x"], c["z"], AREA_M) and not area.near(c["x"], c["z"], ORPHAN_M):
                R["osmNotInDataset"] += 1
                if merge:
                    c["verified"] = False

    # ------------------------------------------------------------ stop signs (per approach)
    for sid, pts in stop_pts.items():
        R, area, merge = row(sid), Grid(100.0), role[sid] == "merge"
        for (x, z, fid, facing) in pts:
            area.add(x, z, fid)
            R["authoritative"] += 1
            j = junctions.near(x, z, SNAP_M)
            if not j:
                R["unplaced"] += 1
                continue
            key = j[0][1]
            if any(cc["type"] == "signal" and id(cc) not in removed for _, cc in sig_grid.near(key[0], key[1], 25)):
                R["atSignals"] += 1
                continue
            ap = legs(key)
            if facing in DIRS:
                want = DIRS[facing]
                best = min(ap, key=lambda g: abs(math.remainder(E[g][2] - want, 2 * math.pi)), default=None)
                if best is None or abs(math.remainder(E[best][2] - want, 2 * math.pi)) > math.radians(50):
                    R["noMatchingLeg"] += 1
                    continue
                ap = [best]
            c = by_key.get((key[0], key[1], "stop"))
            if c is not None and id(c) in removed:
                c = None
            if not merge:
                if c is not None and all(g in {a["edge"] for a in c["approaches"]} for g in ap):
                    R["confirmed"] += 1
                else:
                    R["knownMissing"] += 1
                    rep["knownMissing"].append({"source": sid, "type": "stop", "id": fid, "facing": facing})   # no coordinates (unlicensed data)
                continue
            if c is None:
                c = new_control("stop", key, ap, sid, fid)
                by_key[(key[0], key[1], "stop")] = c
                R["controlsAdded"] += 1
            else:
                have_e = {a["edge"] for a in c["approaches"]}
                new = [g for g in ap if g not in have_e]
                c["approaches"] += [{"edge": g, "stopAt": mouth_stop(key, g)} for g in new]
                R["approachesAdded"] += len(new)
                if c["source"] == "osm":
                    R["controlsConfirmed"] += 1
                    c["source"] = "osm+" + sid
                c["verified"] = True
                c["authIds"].append(fid)
            if len({a["edge"] for a in c["approaches"]}) >= len(legs(key)) >= 3:
                c["allWay"] = True
        for c in controls:
            if c["type"] != "stop" or c["source"] != "osm" or id(c) in removed or c["midBlock"]:
                continue
            if area.near(c["x"], c["z"], AREA_M) and not area.near(c["x"], c["z"], ORPHAN_M):
                R["osmNotInDataset"] += 1
                if merge and complete[sid] and not any(E[a["edge"]][1]["ref"] for a in c["approaches"]):
                    removed.add(id(c)); R["osmRemoved"] += 1
                elif merge:
                    c["verified"] = False

    rep["cities"] = {city: {sid: {"role": role[sid], **dict(R)} for sid, R in v.items()} for city, v in rep["cities"].items()}
    rep["removed"] = [{"type": c["type"], "osmNodes": c["osmNodes"], "lonlat": lonlat(c["x"], c["z"]),
                       "names": sorted({E[a["edge"]][1]["name"] or "" for a in c["approaches"]})} for c in controls if id(c) in removed]

    # ------------------------------------------------------------ write (same layout as tools/build_controls.py)
    out_c = [c for c in controls if id(c) not in removed and c["approaches"]]
    out_c.sort(key=lambda c: (TYPES.index(c["type"]), c["x"], c["z"]))
    tier_id = {"main": 0, "local": 1}
    info, xyz, a_edge, a_stop = array("I"), array("f"), array("I"), array("f")
    out = []
    for cid, c in enumerate(out_c):
        apps = []
        for a in sorted(c["approaches"], key=lambda a: a["edge"]):
            t, e = E[a["edge"]][0], E[a["edge"]][1]
            apps.append({"edge": a["edge"], "tier": t, "tierEdge": e["id"], "stopAt": a["stopAt"],
                         "fromOsm": a.get("fromOsm"), "toOsm": a.get("toOsm"), "class": e["class"]})
        flags = (FLAG_ALL_WAY if c["allWay"] else 0) | (FLAG_RAMP_METER if c["type"] == "ramp_meter" else 0) | \
                (FLAG_MIDBLOCK if c.get("midBlock") else 0)
        rec = {"id": cid}
        rec.update({k: c.get(k) for k in ("type", "tier", "node", "junctionOsm", "osmNodes", "x", "y", "z", "allWay",
                                          "rampMeter", "midBlock", "crossing")})
        rec.update({"approaches": apps, "source": c["source"], "verified": c["verified"], "authIds": c["authIds"]})
        if c["type"] in ("signal", "ramp_meter"):
            rec["timing"] = "synthetic"
        out.append(rec)
        node = c["node"]
        info.extend((TYPES.index(c["type"]), flags, tier_id.get(c["tier"], NO_NODE) if node != -1 else NO_NODE,
                     node if node >= 0 else NO_NODE, len(a_edge), len(apps)))
        xyz.extend((c["x"], c["y"], c["z"]))
        for a in apps:
            a_edge.append(a["edge"]); a_stop.append(a["stopAt"])
    merged_ids = [s["id"] for s in sources if s["role"] == "merge" and s["id"] in have]
    meta = dict(meta_in)
    meta.update({
        "count": len(out), "byType": dict(Counter(c["type"] for c in out)),
        "allWay": sum(1 for c in out if c["allWay"]), "midBlock": sum(1 for c in out if c["midBlock"]),
        "source": "OpenStreetMap via Overpass API (ODbL) + " + ", ".join(merged_ids) + " (tools/controls/sources.json)",
        "merge": {"tool": "tools/controls/merge.py", "bySource": dict(sorted(Counter(c["source"] for c in out).items())),
                  "sources": {i: {**lock.get(i, {}), "city": city_of[i]} for i in merged_ids},
                  "verifyOnly": [s["id"] for s in sources if s["role"] == "verify"],
                  "report": "tools/controls/report/merge-report.json"},
        "timing": "synthetic: every signal / ramp-meter plan is generated by src/sim/controls.js (fixed time, two axes, "
                  "protected lefts on big arterials, ITE yellow, 1.5 s all-red, green wave); no city publishes timing plans",
        "recordFields": "source (osm | osm+<dataset> | <dataset>), verified (true / false / null), authIds, timing",
    })
    json.dump({"meta": meta, "controls": out}, open(os.path.join(data, "controls.json"), "w"), separators=(",", ":"))
    sections = [("controlInfo", info, "Uint32", 6), ("controlXYZ", xyz, "Float32", 3),
                ("approachEdge", a_edge, "Uint32", 1), ("approachStopAt", a_stop, "Float32", 1)]
    sec, off = {}, 0
    with open(os.path.join(data, "controls.bin"), "wb") as f:
        for name, arr, typ, stride in sections:
            if sys.byteorder != "little":
                arr = array(arr.typecode, arr); arr.byteswap()
            b = arr.tobytes()
            sec[name] = {"byteOffset": off, "byteLength": len(b), "type": typ, "length": len(arr), "stride": stride, "count": len(arr) // stride}
            f.write(b); off += len(b)
    mb = json.load(open(os.path.join(src_in, "controls.meta.json")))
    mb.update(meta); mb.update({"byteLength": off, "sections": sec})
    json.dump(mb, open(os.path.join(data, "controls.meta.json"), "w"), indent=1)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    json.dump(rep, open(report_path, "w"), indent=1)
    for city, v in rep["cities"].items():
        for sid, R in v.items():
            print(f"  {city:20s} {sid:24s} {R}", file=sys.stderr)
    print(f"controls {n_osm} OSM -> {len(out)} ({dict(Counter(c['type'] for c in out))})", file=sys.stderr)


if __name__ == "__main__":
    main()
