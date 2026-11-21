#!/usr/bin/env python3
"""
build_controls.py - real traffic control from OSM onto the road graph -> data/controls.*

    python3 -I tools/build_controls.py [--data data] controls-signals.json controls-signs.json \
        -- road-input1.json road-input2.json ...

Control inputs: Overpass JSON of nodes with highway=traffic_signals|stop|give_way,
crossing=traffic_signals, traffic_signals=ramp_meter (tools/queries/controls-*.ql).
Road inputs: the same OSM files tools/build_roads.py was built from. Their way node lists give
each control's position along its road and the direction of the OSM way, which the
direction / traffic_signals:direction tags refer to.

Rules (all documented in data/roads.schema.md, "Traffic control"):
- A control node on a graph junction (an edge endpoint) controls that junction. Its approaches
  are the junction's incoming edges (both tiers): all of them for signals and for stop=all,
  the minor ones (lower road class) for stop=minor or an untagged stop/give_way, and all of
  them when every approach has the same class (allWay). The stop line is at the junction
  mouth: back from the node by half the widest crossing road's width + 1.5 m.
- A control node part-way along a road controls the junction it faces (the next junction
  within 60 m for signals, 40 m for signs). The direction tag picks the travel direction;
  without one, a sign faces the nearer junction and a signal faces both. The stop line is at
  the node (stopAt = metres along the approach edge).
- Signals farther than 60 m from any junction (mid-block pedestrian signals) have node -1
  and stop lines on both directions.
- Ramp meters (traffic_signals=ramp_meter, or a signal on a motorway_link that does not touch
  a non-link road) control their link edge.
- Several OSM nodes for one junction (per-approach signals, crosswalk signals within 40 m)
  are merged into one control, and their OSM ids are kept.
Output keys survive a rebuild of data/: every control stores its OSM node ids and the
junction's OSM id, and every approach stores the OSM ids of its edge's from/to nodes. Rerun
this script after tools/build_roads.py.
"""
import json
import math
import os
import sys
from array import array
from collections import Counter, defaultdict

CLASSES_ALL = ["motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
               "secondary", "secondary_link", "tertiary", "tertiary_link", "unclassified",
               "residential", "living_street"]
TYPES = ["signal", "stop", "give_way", "ramp_meter", "crossing_signal"]
FLAG_ALL_WAY, FLAG_RAMP_METER, FLAG_MIDBLOCK = 1, 2, 4
NO_NODE = 0xFFFFFFFF
SNAP_M = 15.0
NEAR_SIGNAL_M, NEAR_SIGN_M, CROSSWALK_MERGE_M = 60.0, 40.0, 40.0


def load_json(p):
    with open(p, "rb") as f:
        raw = f.read()
    if not raw.lstrip().startswith(b"{"):
        print(f"!! {p}: not JSON, skipped", file=sys.stderr)
        return {"elements": []}
    return json.loads(raw)


def main(argv):
    data = "data"
    if "--data" in argv:
        i = argv.index("--data")
        data = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    if "--" not in argv:
        print(__doc__)
        return 2
    k = argv.index("--")
    ctl_files, road_files = argv[:k], argv[k + 1:]

    # ---------------------------------------------------------------- graph (both tiers)
    tiers = {}
    for t, fn in (("main", "roads.json"), ("local", "roads-local.json")):
        tiers[t] = json.load(open(os.path.join(data, fn)))
    meta = tiers["main"]["meta"]
    kx, kz = meta["projection"]["metersPerDegLon"], meta["projection"]["metersPerDegLat"]
    lat0, lon0 = meta["projection"]["lat0"], meta["projection"]["lon0"]
    proj = lambda la, lo: ((lo - lon0) * kx, -(la - lat0) * kz)
    Em = len(tiers["main"]["edges"])
    base = {"main": 0, "local": Em}
    # merged edges: (tier, e, json edge)
    E = []
    for t in ("main", "local"):
        for e in tiers[t]["edges"]:
            E.append((t, e))
    osm_of = lambda t, i: tiers[t]["nodes"][i]["id"]
    endpoint = set()
    incoming = defaultdict(list)       # junction OSM id -> merged edge ids
    touching = defaultdict(list)       # junction OSM id -> merged edge ids (in and out)
    for g, (t, e) in enumerate(E):
        endpoint.add(osm_of(t, e["from"]))
        endpoint.add(osm_of(t, e["to"]))
        incoming[osm_of(t, e["to"])].append(g)
        touching[osm_of(t, e["to"])].append(g)
        touching[osm_of(t, e["from"])].append(g)

    # stop lines from the shared junction geometry (tools/build_junctions.mjs -> data/junctions.*),
    # so the controls, the sim's uncontrolled yield points and the drawn junction boxes agree
    geom_mouth = None
    jm = os.path.join(data, "junctions.meta.json")
    if os.path.exists(jm):
        jmeta = json.load(open(jm))
        sec = jmeta["sections"]["mouthS"]
        with open(os.path.join(data, jmeta["file"]), "rb") as f:
            f.seek(sec["byteOffset"])
            geom_mouth = array("f")
            geom_mouth.frombytes(f.read(sec["byteLength"]))
        if len(geom_mouth) != 2 * len(E):
            geom_mouth = None
        print(f"   stop lines from data/junctions.bin ({'ok' if geom_mouth else 'size mismatch, ignored'})", file=sys.stderr)

    def mouth(j, g):
        if geom_mouth is not None and geom_mouth[2 * g + 1] < E[g][1]["length"] - 0.5:
            return float(geom_mouth[2 * g + 1])
        """Stop line for approach g at junction j: back from the node centre by half the width of
        the widest crossing road + 1.5 m (crosswalk), so cars stop at the junction mouth."""
        t, e = E[g]
        tw = base[t] + e["twin"] if e["twin"] >= 0 else -1
        hw = 0.0
        for h in touching[j]:
            if h == g or h == tw:
                continue
            eh = E[h][1]
            hw = max(hw, (eh["lanes"] + eh["oppLanes"]) * 3.7 / 2)
        return max(0.0, e["length"] - min(hw + 1.5, 0.5 * e["length"]))
    node_tier_index = {}
    for t in ("main", "local"):
        for i, n in enumerate(tiers[t]["nodes"]):
            node_tier_index.setdefault(n["id"], (t, i))

    # spatial index of directed edges' pieces
    cell = 50.0
    grid = defaultdict(list)
    for g, (t, e) in enumerate(E):
        N = tiers[t]["nodes"]
        nd = e["nodes"]
        for q in range(len(nd) - 1):
            A, B = N[nd[q]], N[nd[q + 1]]
            for gx in range(int(min(A["x"], B["x"]) // cell), int(max(A["x"], B["x"]) // cell) + 1):
                for gz in range(int(min(A["z"], B["z"]) // cell), int(max(A["z"], B["z"]) // cell) + 1):
                    grid[(gx, gz)].append((g, q))

    def project_on(g, x, z):
        """(dist, s, heading) of point on merged edge g."""
        t, e = E[g]
        N = tiers[t]["nodes"]
        nd = e["nodes"]
        best = (1e18, 0.0, (1.0, 0.0))
        acc = 0.0
        for q in range(len(nd) - 1):
            A, B = N[nd[q]], N[nd[q + 1]]
            dx, dz = B["x"] - A["x"], B["z"] - A["z"]
            L = math.hypot(dx, dz)
            if L > 0:
                tt = max(0.0, min(1.0, ((x - A["x"]) * dx + (z - A["z"]) * dz) / (L * L)))
                d = math.hypot(A["x"] + dx * tt - x, A["z"] + dz * tt - z)
                if d < best[0]:
                    best = (d, acc + L * tt, (dx / L, dz / L))
            acc += L
        s = best[1] * (e["length"] / acc) if acc > 0 else 0.0
        return best[0], s, best[2]

    def nearest_edges(x, z, classes):
        cand = set()
        for gx in range(int((x - SNAP_M) // cell), int((x + SNAP_M) // cell) + 1):
            for gz in range(int((z - SNAP_M) // cell), int((z + SNAP_M) // cell) + 1):
                for (g, q) in grid.get((gx, gz), ()):
                    cand.add(g)
        res = []
        for g in cand:
            if classes and E[g][1]["class"] not in classes:
                continue
            d, s, h = project_on(g, x, z)
            if d <= SNAP_M:
                res.append((d, g, s, h))
        res.sort()
        return res

    # ---------------------------------------------------------------- raw inputs
    ctl = {}
    for f in ctl_files:
        for el in load_json(f).get("elements", []):
            if el.get("type") == "node":
                ctl[el["id"]] = el
    need = set(ctl)
    way_of = defaultdict(list)       # control id -> [(way nodes, index, class)]
    coords = {}
    road_ways = []
    for f in road_files:
        doc = load_json(f)
        for el in doc.get("elements", []):
            if el.get("type") == "node":
                coords[el["id"]] = (el["lat"], el["lon"])
            elif el.get("type") == "way":
                tags = el.get("tags") or {}
                if tags.get("highway") in CLASSES_ALL:
                    road_ways.append(el)
        del doc
    for w in road_ways:
        nl = w["nodes"]
        cls = w["tags"]["highway"]
        ow = str(w["tags"].get("oneway", "")).lower()
        for i, n in enumerate(nl):
            if n in need:
                way_of[n].append((nl, i, cls, ow))
    print(f"   {len(ctl)} control nodes; {len(way_of)} lie on a road way of the graph", file=sys.stderr)

    # ---------------------------------------------------------------- per control node -> claims
    claims = []    # dicts: type, osm, junction OSM id or None, approaches [(g, stopAt)], allWay, ramp, x, z
    stats = Counter()

    def way_fwd(n):
        """Unit vector of the OSM way's forward direction at node n (first parent way)."""
        nl, i, cls, ow = way_of[n][0]
        a = nl[i - 1] if i > 0 else nl[i]
        b = nl[i + 1] if i + 1 < len(nl) else nl[i]
        if a not in coords or b not in coords or a == b:
            return None
        ax, az = proj(*coords[a])
        bx, bz = proj(*coords[b])
        L = math.hypot(bx - ax, bz - az) or 1.0
        v = ((bx - ax) / L, (bz - az) / L)
        return (-v[0], -v[1]) if ow == "-1" else v

    def classify(tags):
        hw = tags.get("highway")
        ts = str(tags.get("traffic_signals", "")).lower()
        if ts.startswith("ramp") or hw == "ramp_meter" or "ramp_meter" in tags:
            return "ramp_meter"
        if hw == "traffic_signals":
            return "signal"
        if hw == "stop":
            return "stop"
        if hw == "give_way":
            return "give_way"
        if tags.get("crossing") == "traffic_signals":
            return "crossing_signal"
        return None

    # road importance (smaller = more important): mainlines before ramps, then arterials, locals
    IMPORTANCE = {"motorway": 0, "trunk": 1, "motorway_link": 2, "trunk_link": 3, "primary": 4, "primary_link": 5,
                  "secondary": 6, "secondary_link": 6, "tertiary": 7, "tertiary_link": 7, "unclassified": 8,
                  "residential": 8, "living_street": 9}

    def rank(g):
        return IMPORTANCE[E[g][1]["class"]]

    def away(g):
        """Unit vector from the junction back along incoming edge g."""
        t, e = E[g]
        N = tiers[t]["nodes"]
        a, b = N[e["nodes"][-1]], N[e["nodes"][-2]]
        dx, dz = b["x"] - a["x"], b["z"] - a["z"]
        L = math.hypot(dx, dz) or 1.0
        return dx / L, dz / L

    def through_pair(app):
        best, bd = set(), 2.0
        for i in range(len(app)):
            for k in range(i + 1, len(app)):
                u, v = away(app[i]), away(app[k])
                d = u[0] * v[0] + u[1] * v[1]           # -1 = straight through
                if d < -0.7 and d + 1 < bd:
                    bd, best = d + 1, {app[i], app[k]}
        return best

    for n, el in ctl.items():
        tags = el.get("tags") or {}
        typ = classify(tags)
        if typ is None:
            continue
        if n not in way_of:
            stats[f"{typ}: not on a graph road (footway / outside)"] += 1
            continue
        if n not in coords:
            coords[n] = (el["lat"], el["lon"])
        x, z = proj(el["lat"], el["lon"])
        classes = {c for (_, _, c, _) in way_of[n]}
        if typ == "signal" and classes <= {"motorway_link", "trunk_link"}:
            typ = "ramp_meter"      # a signal on a freeway ramp that touches no surface street
        direction = str(tags.get("traffic_signals:direction") or tags.get("direction") or "").lower()
        stopv = str(tags.get("stop", "")).lower()
        # --- on a junction node
        if n in endpoint:
            app = incoming[n]
            if not app:
                stats[f"{typ}: junction without approaches"] += 1
                continue
            if typ in ("signal", "crossing_signal", "ramp_meter"):
                sel = app
                allway = False
            else:
                top = min(rank(g) for g in app)
                same = all(rank(g) == top for g in app)
                if stopv == "all":
                    sel, allway = app, True
                elif same:
                    # one untagged sign on a junction of equal roads: the through road (the
                    # straightest pair of legs) has priority, the other legs stop
                    thr = through_pair(app)
                    sel, allway = [g for g in app if g not in thr] or app, False
                else:
                    sel, allway = [g for g in app if rank(g) > top] or app, False
            claims.append({"type": typ, "osm": n, "junction": n, "app": [(g, mouth(n, g)) for g in sel],
                           "allWay": allway, "x": x, "z": z, "stopTag": stopv})
            stats[f"{typ}: on junction"] += 1
            continue
        # --- part-way along a road: snap to the directed edges through the node
        cand = nearest_edges(x, z, classes)
        if not cand:
            stats[f"{typ}: could not snap (> {SNAP_M} m)"] += 1
            continue
        d0, g0, s0, h0 = cand[0]
        t0, e0 = E[g0]
        dirs = [(g0, s0, h0)]
        if e0["twin"] >= 0:
            gt = base[t0] + e0["twin"]
            L = e0["length"]
            dirs.append((gt, L - s0, (-h0[0], -h0[1])))
        wf = way_fwd(n)
        if direction in ("forward", "backward") and wf:
            sgn = 1 if direction == "forward" else -1
            dirs = [dd for dd in dirs if (dd[2][0] * wf[0] + dd[2][1] * wf[1]) * sgn > 0] or dirs[:1]
        elif direction in ("both", "") and typ in ("stop", "give_way") and len(dirs) == 2:
            if direction == "":
                # a sign faces the nearer junction
                dirs = [min(dirs, key=lambda dd: E[dd[0]][1]["length"] - dd[1])]
        near = NEAR_SIGNAL_M if typ in ("signal", "crossing_signal") else NEAR_SIGN_M
        for (g, s, h) in dirs:
            t, e = E[g]
            dist_to_end = e["length"] - s
            junc = osm_of(t, e["to"]) if dist_to_end <= near else None
            if typ == "ramp_meter":
                junc = None
            stop_s = max(0.0, min(s, mouth(junc, g))) if junc is not None else max(0.0, s)
            claims.append({"type": typ, "osm": n, "junction": junc, "app": [(g, stop_s)],
                           "allWay": stopv == "all", "x": x, "z": z, "stopTag": stopv})
        stats[f"{typ}: along a road ({'near junction' if any(E[g][1]['length'] - s <= near for g, s, h in dirs) else 'mid-block'})"] += 1

    # ---------------------------------------------------------------- merge claims per junction
    by_j = defaultdict(list)
    loose = []
    for c in claims:
        (by_j[(c["junction"], "sig" if c["type"] in ("signal", "crossing_signal") else c["type"])] if c["junction"] is not None else loose).append(c)
    controls = []

    def jpos(j):
        t, i = node_tier_index[j]
        nd = tiers[t]["nodes"][i]
        return t, i, nd["x"], nd.get("y", 0.0), nd["z"]

    for (j, kind), lst in by_j.items():
        types = Counter(c["type"] for c in lst)
        if kind == "sig":
            typ = "signal" if types.get("signal") else "crossing_signal"
            if typ == "crossing_signal":
                typ = "signal"     # crosswalk signals at a junction are its pedestrian heads
        else:
            typ = kind
        app = {}
        for c in lst:
            for g, s in c["app"]:
                app[g] = min(app.get(g, 1e18), s)
        t, i, x, y, z = jpos(j)
        osm_ids = sorted({c["osm"] for c in lst})
        if typ == "signal":
            legs = {g for c in lst for g, _ in c["app"]}
            real = any(c["type"] == "signal" for c in lst)
            if any(c["osm"] == j for c in lst) or len(legs) >= 2 or real:
                # a signalised junction: every non-link incoming leg gets the signal; slip-lane
                # links get a give_way record at the same junction
                links = [g for g in incoming[j] if E[g][1]["class"].endswith("_link")]
                for g in incoming[j]:
                    # every real leg (>= 30 m; short internal links of a dual-carriageway
                    # junction stay free so cars do not stop twice)
                    if g not in links and (E[g][1]["length"] >= 30 or g in legs):
                        app.setdefault(g, mouth(j, g))
                for g in links:
                    app.pop(g, None)
                stats["signalised junctions (legs filled)"] += 1
                if links:
                    controls.append({"type": "give_way", "osmNodes": osm_ids, "junctionOsm": j, "tier": t, "node": i,
                                     "x": x, "y": y, "z": z, "allWay": False, "rampMeter": False,
                                     "app": {g: mouth(j, g) for g in links}, "derived": "slip lane at a signal"})
            else:
                # a crosswalk signal on one leg of an otherwise unsignalised junction: a mid-block signal
                stats["single-leg signals demoted to crosswalk signals"] += 1
                c0 = lst[0]
                controls.append({"type": "signal", "osmNodes": osm_ids, "junctionOsm": None, "tier": None, "node": -1,
                                 "x": c0["x"], "y": 0.0, "z": c0["z"], "allWay": False, "rampMeter": False,
                                 "app": app, "midBlock": True, "crossing": True})
                continue
            allway = False
        else:
            signed_legs = {g for c in lst for g, _ in c["app"]}
            allway = any(c["allWay"] for c in lst) or (len(osm_ids) >= 3 and len(signed_legs) >= 3)
            if not allway:
                # a sign on a leg of higher class than every uncontrolled leg is a direction error
                # (OSM direction=forward/backward mixed up): the minor legs get it instead
                others = [g for g in incoming[j] if g not in app]
                if others:
                    top_other = min(rank(o) for o in others)          # best (major) class among the rest
                    wrong = [g for g in app if rank(g) < top_other]
                    if wrong:
                        for g in wrong:
                            del app[g]
                        minor = [o for o in others if rank(o) == max(rank(x) for x in others)]
                        for o in minor:
                            app.setdefault(o, mouth(j, o))
                        stats["signs moved from the major to the minor legs"] += 1
        controls.append({"type": typ, "osmNodes": osm_ids, "junctionOsm": j,
                         "tier": t, "node": i, "x": x, "y": y, "z": z, "allWay": bool(allway),
                         "rampMeter": False, "app": app})
    # stops already covered by a signal at the same junction are dropped (signal wins)
    sig_j = {c["junctionOsm"] for c in controls if c["type"] == "signal"}
    before = len(controls)
    controls = [c for c in controls if not (c["type"] in ("stop", "give_way") and c["junctionOsm"] in sig_j and not c.get("derived"))]
    stats["signs dropped (junction is signalised)"] += before - len(controls)
    # loose (mid-block) claims: crosswalk signals within 40 m of a signalised junction merge into it
    sig_pos = [(c["x"], c["z"], c) for c in controls if c["type"] == "signal"]
    cell2 = 100.0
    sgrid = defaultdict(list)
    for (sx, sz, c) in sig_pos:
        sgrid[(int(sx // cell2), int(sz // cell2))].append((sx, sz, c))
    mid = defaultdict(list)
    for c in loose:
        merged = False
        if c["type"] in ("signal", "crossing_signal"):
            gx, gz = int(c["x"] // cell2), int(c["z"] // cell2)
            for dx in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for (sx, sz, sc) in sgrid.get((gx + dx, gz + dz), ()):
                        if not merged and math.hypot(sx - c["x"], sz - c["z"]) <= CROSSWALK_MERGE_M:
                            sc["osmNodes"] = sorted(set(sc["osmNodes"]) | {c["osm"]})
                            merged = True
            if merged:
                stats["crosswalk signals merged into a junction signal"] += 1
                continue
        mid[(c["osm"], c["type"])].append(c)
    for (osm, typ), lst in mid.items():
        app = {}
        for c in lst:
            for g, s in c["app"]:
                app[g] = min(app.get(g, 1e18), s)
        c0 = lst[0]
        controls.append({"type": "ramp_meter" if typ == "ramp_meter" else ("signal" if typ in ("signal", "crossing_signal") else typ),
                         "osmNodes": [osm], "junctionOsm": None, "tier": None, "node": -1,
                         "x": c0["x"], "y": 0.0, "z": c0["z"], "allWay": False,
                         "rampMeter": typ == "ramp_meter", "app": app, "midBlock": typ != "ramp_meter",
                         "crossing": typ == "crossing_signal"})
    # y for mid-block controls from the approach edge
    for c in controls:
        if c["node"] == -1 and c["app"]:
            g = next(iter(c["app"]))
            t, e = E[g]
            N = tiers[t]["nodes"]
            c["y"] = max(N[e["nodes"][0]]["y"], N[e["nodes"][-1]]["y"]) if abs(N[e["nodes"][0]]["y"] - N[e["nodes"][-1]]["y"]) < 0.5 else N[e["nodes"][0]]["y"]

    # sanity: no signal/sign on a motorway mainline; ramp meters only on ramps
    for c in controls:
        for g in list(c["app"]):
            cls = E[g][1]["class"]
            if (c["type"] != "ramp_meter" and cls == "motorway") or (c["type"] == "ramp_meter" and not cls.endswith("_link")):
                del c["app"][g]
                stats["approaches dropped (sign on motorway mainline / meter off a ramp)"] += 1
    controls = [c for c in controls if c["app"]]
    controls.sort(key=lambda c: (TYPES.index(c["type"]), c["x"], c["z"]))
    # ---------------------------------------------------------------- write
    out = []
    info, xyz, a_edge, a_stop = array("I"), array("f"), array("I"), array("f")
    for cid, c in enumerate(controls):
        apps = []
        for g, s in sorted(c["app"].items()):
            t, e = E[g]
            apps.append({"edge": g, "tier": t, "tierEdge": e["id"], "stopAt": round(min(s, e["length"]), 2),
                         "fromOsm": osm_of(t, e["from"]), "toOsm": osm_of(t, e["to"]), "class": e["class"]})
        flags = (FLAG_ALL_WAY if c["allWay"] else 0) | (FLAG_RAMP_METER if c["type"] == "ramp_meter" else 0) | \
                (FLAG_MIDBLOCK if c.get("midBlock") else 0)
        out.append({"id": cid, "type": c["type"], "tier": c["tier"], "node": c["node"],
                    "junctionOsm": c["junctionOsm"], "osmNodes": c["osmNodes"],
                    "x": round(c["x"], 2), "y": round(c["y"], 2), "z": round(c["z"], 2),
                    "allWay": c["allWay"], "rampMeter": c["type"] == "ramp_meter",
                    "midBlock": bool(c.get("midBlock")), "crossing": bool(c.get("crossing")),
                    "approaches": apps})
        info.extend((TYPES.index(c["type"]), flags, {"main": 0, "local": 1, None: 0xFFFFFFFF}[c["tier"]],
                     c["node"] if c["node"] >= 0 else NO_NODE, len(a_edge), len(apps)))
        xyz.extend((c["x"], c["y"], c["z"]))
        for a in apps:
            a_edge.append(a["edge"])
            a_stop.append(a["stopAt"])
    cstats = Counter(c["type"] for c in out)
    allway = sum(1 for c in out if c["allWay"])
    mid_n = sum(1 for c in out if c["midBlock"])
    meta_out = {"format": "bay-controls/1", "count": len(out), "types": TYPES,
                "flagBits": {"ALL_WAY": FLAG_ALL_WAY, "RAMP_METER": FLAG_RAMP_METER, "MIDBLOCK": FLAG_MIDBLOCK},
                "edgeIds": "merged graph: main e, local mainEdgeCount + e (= src/sim fromRoadSet, src/geom)",
                "mainEdgeCount": Em, "byType": dict(cstats), "allWay": allway, "midBlock": mid_n,
                "source": "OpenStreetMap via Overpass API (ODbL)", "build": dict(stats)}
    jp = os.path.join(data, "controls.json")
    with open(jp, "w") as f:
        json.dump({"meta": meta_out, "controls": out}, f, separators=(",", ":"))
    sections = [("controlInfo", info, "Uint32", 6), ("controlXYZ", xyz, "Float32", 3),
                ("approachEdge", a_edge, "Uint32", 1), ("approachStopAt", a_stop, "Float32", 1)]
    bp = os.path.join(data, "controls.bin")
    sec, off = {}, 0
    with open(bp, "wb") as f:
        for name, arr, typ, stride in sections:
            if sys.byteorder != "little":
                arr = array(arr.typecode, arr)
                arr.byteswap()
            b = arr.tobytes()
            sec[name] = {"byteOffset": off, "byteLength": len(b), "type": typ, "length": len(arr),
                         "stride": stride, "count": len(arr) // stride}
            f.write(b)
            off += len(b)
    mb = dict(meta_out)
    mb.update({"file": "controls.bin", "json": "controls.json", "byteLength": off, "sections": sec,
               "controlInfoFields": ["type", "flags", "tierId", "nodeIndex", "firstApproach", "approachCount"]})
    json.dump(mb, open(os.path.join(data, "controls.meta.json"), "w"), indent=1)
    print("== controls by type:", dict(cstats), f"all-way stops {allway}, mid-block {mid_n}, "
          f"approaches {len(a_edge)}", file=sys.stderr)
    for k in sorted(stats):
        print(f"   {k:60} {stats[k]}", file=sys.stderr)
    print(f"== wrote {jp}, {bp} ({off} bytes), controls.meta.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
