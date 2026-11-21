#!/usr/bin/env python3
"""
build_roads.py - OSM (Overpass JSON) -> compact drivable road graph for the
Bay Area traffic sim (JSON + GPU-ready binary pack).

Usage:
    python3 -I tools/build_roads.py [options] input1.json [input2.json ...]

Options:
    --out-dir data                 where roads*.json / roads*.bin / *.meta.json go
    --bbox S,W,N,E                 keep only nodes inside (default: commute corridor
                                   37.25,-122.56,37.97,-121.80)
    --exclude S,W,N,E              drop nodes inside this box (repeatable; defaults
                                   drop the Tri-Valley 37.60,-121.97,38.5,-121.0 and
                                   Lamorinda 37.85,-122.25,38.5,-121.0)
    --no-simplify                  skip Douglas-Peucker / degree-2 merging

Pipeline: load+merge inputs -> clip -> split ways at junctions -> merge
degree-2 nodes with identical attributes -> Douglas-Peucker (2 m motorway/trunk,
4 m others) -> bake elevation (bridges/layers) -> write per tier:
    roads.json / roads-local.json           readable copy (schema: data/roads.schema.md)
    roads.bin  / roads-local.bin            little-endian typed-array pack
    roads.meta.json / roads-local.meta.json section offsets + enums for the pack

Coordinate system (also stored in meta.projection):
    equirectangular around lat0=37.55, lon0=-122.15
    x = (lon - lon0) * cos(lat0) * R * pi/180   (metres, +x = EAST)
    z = -(lat - lat0) * R * pi/180              (metres, +z = SOUTH, north = -z)
    y = elevation (metres, up): 0 at grade, bridges 15 m (+10 m per extra layer),
        non-bridge layer>0 8 m/layer, tunnels 0; ramps over the last 140 m of an
        elevated edge down to the lowest elevation at its end node.
Right-hand traffic: the perpendicular to the RIGHT of travel (dx,dz) is (-dz, dx).
"""
import json
import math
import os
import re
import sys
import time
from array import array
from collections import Counter, defaultdict, deque

LAT0, LON0 = 37.55, -122.15
R_EARTH = 6371008.8
LANE_W = 3.7
RAMP_M = 140.0
BRIDGE_Y, LAYER_BRIDGE_Y, LAYER_Y = 15.0, 10.0, 8.0

# Corridor: SJ (south) to SF + Marin end of the Golden Gate and the Richmond-San
# Rafael Bridge / I-580 / I-80 in Richmond (north).  Excluded: the Tri-Valley
# and the hills/Lamorinda east of Berkeley north of 37.85.
DEFAULT_BBOX = (37.25, -122.56, 37.97, -121.80)
DEFAULT_EXCLUDE = [(37.60, -121.97, 38.5, -121.0),    # Tri-Valley (Dublin/Pleasanton/Livermore)
                   (37.85, -122.25, 38.5, -121.0)]    # Orinda/Lafayette/Walnut Creek

MAIN_CLASSES = ["motorway", "motorway_link", "trunk", "trunk_link",
                "primary", "primary_link", "secondary", "secondary_link"]
LOCAL_CLASSES = ["tertiary", "tertiary_link", "unclassified", "residential",
                 "living_street"]
ALL_CLASSES = MAIN_CLASSES + LOCAL_CLASSES          # classId = index in this list
FAST_CLASSES = {"motorway", "motorway_link", "trunk", "trunk_link"}
# per-class lane width (m) when OSM has no usable width tag
LANE_WIDTH_CLASS = {"motorway": 3.7, "motorway_link": 3.7, "trunk": 3.7, "trunk_link": 3.6,
                    "primary": 3.6, "primary_link": 3.5, "secondary": 3.5, "secondary_link": 3.4,
                    "tertiary": 3.3, "tertiary_link": 3.3, "unclassified": 3.1, "residential": 3.1,
                    "living_street": 3.0}

ONEWAY_DEFAULT = {"motorway": 4, "motorway_link": 1, "trunk": 3, "trunk_link": 1,
                  "primary": 2, "primary_link": 1, "secondary": 2, "secondary_link": 1,
                  "tertiary": 1, "tertiary_link": 1, "unclassified": 1,
                  "residential": 1, "living_street": 1}
TWOWAY_DEFAULT_PER_DIR = {"motorway": 2, "motorway_link": 1, "trunk": 2, "trunk_link": 1,
                          "primary": 1, "primary_link": 1, "secondary": 1, "secondary_link": 1,
                          "tertiary": 1, "tertiary_link": 1, "unclassified": 1,
                          "residential": 1, "living_street": 1}
DEFAULT_SPEED_KPH = {"motorway": 105, "motorway_link": 65, "trunk": 90, "trunk_link": 55,
                     "primary": 65, "primary_link": 45, "secondary": 55, "secondary_link": 40,
                     "tertiary": 45, "tertiary_link": 35, "unclassified": 40,
                     "residential": 40, "living_street": 15}

FLAG_ONEWAY, FLAG_BRIDGE, FLAG_TUNNEL, FLAG_LANES_FALLBACK, FLAG_SPEED_DEFAULT, FLAG_LINK = 1, 2, 4, 8, 16, 32
EDGE_STRIDE = 8
EDGE_FIELDS = ["firstNode", "nodeCount", "lanesFwd", "lanesBack", "classId", "speedCmps", "flags", "twin"]
NO_TWIN = 0xFFFFFFFF

KX = math.cos(math.radians(LAT0)) * R_EARTH * math.pi / 180.0
KZ = R_EARTH * math.pi / 180.0


def project(lat, lon):
    return (lon - LON0) * KX, -(lat - LAT0) * KZ


# ----------------------------------------------------------------------------- tag parsing
_int_re = re.compile(r"-?\d+")


def parse_int(s):
    if s is None:
        return None
    m = _int_re.search(str(s))
    if not m:
        return None
    v = int(m.group(0))
    return v if v > 0 else None


def parse_int_signed(s):
    if s is None:
        return None
    m = _int_re.search(str(s))
    return int(m.group(0)) if m else None


def parse_lanes_from_turn(s):
    return len(str(s).split("|")) if s else None


def parse_maxspeed(s):
    if not s:
        return None
    s = str(s).strip().lower()
    m = re.search(r"(\d+(\.\d+)?)", s)
    if not m:
        return None
    v = float(m.group(1))
    if "mph" in s:
        v *= 1.609344
    return round(v, 1)


def truthy(v):
    return v is not None and str(v).lower() not in ("no", "false", "0", "")


def lane_counts(tags, cls, oneway):
    lanes = parse_int(tags.get("lanes")) or parse_lanes_from_turn(tags.get("turn:lanes"))
    lf = parse_int(tags.get("lanes:forward")) or parse_lanes_from_turn(tags.get("turn:lanes:forward"))
    lb = parse_int(tags.get("lanes:backward")) or parse_lanes_from_turn(tags.get("turn:lanes:backward"))
    if oneway:
        if lanes is not None:
            return lanes, 0, False
        if lf is not None:
            return lf, 0, False
        return ONEWAY_DEFAULT[cls], 0, True
    if lf is not None and lb is not None:
        return lf, lb, False
    if lanes is not None:
        bw = parse_int(tags.get("lanes:both_ways")) or 0   # centre two-way turn lane
        if bw and lanes - bw >= 2:
            lanes -= bw
        if lf is not None:
            return lf, max(1, lanes - lf), False
        if lb is not None:
            return max(1, lanes - lb), lb, False
        per = max(1, lanes // 2)
        return per, per, False
    d = TWOWAY_DEFAULT_PER_DIR[cls]
    if lf is not None:
        return lf, d, True
    if lb is not None:
        return d, lb, True
    return d, d, True


def side_set(tags, key):
    """OSM sidewalk / parking tags -> (left, right) booleans relative to the way, or None if untagged.
    key 'sidewalk' reads sidewalk[=both|left|right|no|separate], sidewalk:both|left|right;
    key 'parking' reads parking:lane:*, parking:* (no / no_parking / no_stopping = none)."""
    yes = lambda v: v is not None and str(v).lower() not in ("no", "none", "no_parking", "no_stopping", "separate", "fire_lane")
    if key == "sidewalk":
        v = tags.get("sidewalk")
        if v is not None:
            v = str(v).lower()
            return {"both": (True, True), "left": (True, False), "right": (False, True)}.get(v, (False, False))
        b, l, r = tags.get("sidewalk:both"), tags.get("sidewalk:left"), tags.get("sidewalk:right")
    else:
        b = tags.get("parking:lane:both") or tags.get("parking:both")
        l = tags.get("parking:lane:left") or tags.get("parking:left")
        r = tags.get("parking:lane:right") or tags.get("parking:right")
    if b is None and l is None and r is None:
        return None
    return (yes(l) if l is not None else yes(b), yes(r) if r is not None else yes(b))


def side_word(lr, rev):
    if lr is None:
        return None
    l, r = (lr[1], lr[0]) if rev else lr
    return "both" if l and r else "left" if l else "right" if r else "no"


def parse_width(v):
    """OSM width -> metres (handles '12', '12 m', "40'", '40 ft'); None if absent / unparsable."""
    if v is None:
        return None
    t = str(v).strip().lower()
    m = re.search(r"(\d+(\.\d+)?)", t)
    if not m:
        return None
    w = float(m.group(1))
    if "'" in t or "ft" in t:
        w *= 0.3048
    return w if 2.5 <= w <= 80 else None


def lane_width_of(a, fwd, back):
    """Per-lane width (m): OSM carriageway width / lanes when plausible (2.7-4.2 m), else class default."""
    w = a.get("width_tag")
    if w:
        per = w / max(1, fwd + back)
        if 2.7 <= per <= 4.2:
            return round(per, 2)
    return LANE_WIDTH_CLASS[a["class"]]


def lane_offsets(lanes, opp):
    base = (opp - lanes) / 2.0
    return [round((base + k + 0.5) * LANE_W, 2) for k in range(lanes)]


# ----------------------------------------------------------------------------- loading
def load_inputs(paths):
    nodes, ways = {}, {}
    for p in paths:
        t0 = time.time()
        with open(p, "rb") as f:
            raw = f.read()
        if not raw.lstrip().startswith(b"{"):
            print(f"!! {p}: not JSON (starts {raw[:40]!r}); skipping", file=sys.stderr)
            continue
        try:
            doc = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"!! {p}: invalid JSON ({e}); skipping", file=sys.stderr)
            continue
        del raw
        nn = nw = 0
        for el in doc.get("elements", []):
            t = el.get("type")
            if t == "node":
                nodes[el["id"]] = (el["lat"], el["lon"])
                nn += 1
            elif t == "way":
                tags = el.get("tags") or {}
                if tags.get("highway") in ALL_CLASSES and len(el.get("nodes", [])) >= 2:
                    ways[el["id"]] = el
                    nw += 1
        print(f"   loaded {os.path.basename(p)}: {nn} nodes, {nw} road ways ({time.time() - t0:.1f}s)",
              file=sys.stderr)
        del doc
    return nodes, ways


def in_box(lat, lon, b):
    return b[0] <= lat <= b[2] and b[1] <= lon <= b[3]


# ----------------------------------------------------------------------------- geometry
def dp_simplify(pts, tol):
    """Douglas-Peucker on list of (x,z); returns list of kept indices (endpoints always)."""
    n = len(pts)
    if n < 3:
        return list(range(n))
    keep = [False] * n
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    tol2 = tol * tol
    while stack:
        a, b = stack.pop()
        ax, az = pts[a]; bx, bz = pts[b]
        dx, dz = bx - ax, bz - az
        L = dx * dx + dz * dz
        best, bi = 0.0, -1
        for i in range(a + 1, b):
            px, pz = pts[i]
            if L == 0:
                d2 = (px - ax) ** 2 + (pz - az) ** 2
            else:
                t = ((px - ax) * dx + (pz - az) * dz) / L
                t = 0.0 if t < 0 else 1.0 if t > 1 else t
                d2 = (px - (ax + t * dx)) ** 2 + (pz - (az + t * dz)) ** 2
            if d2 > best:
                best, bi = d2, i
        if best > tol2 and bi > 0:
            keep[bi] = True
            stack.append((a, bi)); stack.append((bi, b))
    return [i for i in range(n) if keep[i]]


def seg_elev(a):
    if a["tunnel"]:
        return 0.0
    if a["bridge"]:
        return BRIDGE_Y + max(0, a["layer"] - 1) * LAYER_BRIDGE_Y
    if a["layer"] > 0:
        return a["layer"] * LAYER_Y
    return 0.0


# ----------------------------------------------------------------------------- lanes (audit D4/D12/D15)
TURN_ONLY = {"left", "right", "slight_left", "slight_right", "sharp_left", "sharp_right",
             "merge_to_left", "merge_to_right", "reverse"}
# per-direction caps (D12); LANES_CLAMPED is set when a tag exceeded it
CLASS_MAX_PER_DIR = {"motorway": 6, "motorway_link": 3, "trunk": 4, "trunk_link": 2,
                     "primary": 4, "primary_link": 2, "secondary": 3, "secondary_link": 2,
                     "tertiary": 2, "tertiary_link": 2, "unclassified": 2, "residential": 2,
                     "living_street": 1}
MIN_RUN = {"fw": 400.0, "link": 300.0, "art": 300.0, "loc": 200.0}
FLAG_LANES_CLAMPED, FLAG_LANES_SMOOTHED, FLAG_AUX_LANE, FLAG_LANE_DROP = 64, 128, 256, 512
ACCEL_M, TAPER_M, AUX_TO_EXIT_M, DROP_TO_EXIT_M = 300.0, 90.0, 700.0, 2000.0

# elevation (audit D2)
LAYER_CLEAR = 7.0          # deck height per layer for overpasses (layer 1 -> 7 m)
WATER_DECK = 20.0          # minimum deck height of a structure that spans water
CROSS_CLEAR = 7.0          # height given to the upper road at an untagged crossing
MIN_CLEAR = 4.5            # build fails below this at crossings without a shared node
GRADE = {"fw": 0.05, "link": 0.06, "art": 0.06, "loc": 0.08}


def cat(cls):
    if cls in ("motorway", "trunk"):
        return "fw"
    if cls.endswith("_link"):
        return "link"
    if cls in ("primary", "secondary", "tertiary"):
        return "art"
    return "loc"


def turn_only_count(s):
    if not s:
        return 0
    n = 0
    for part in str(s).split("|"):
        toks = {t.strip() for t in part.split(";") if t.strip()}
        if toks and toks <= TURN_ONLY:
            n += 1
    return n


def through_lanes(tags, cls, oneway, fwd, back, fallback):
    """Through lanes per direction: drop turn-only pockets on non-freeway roads, clamp per class."""
    tf, tb = fwd, back
    if not fallback and cat(cls) not in ("fw",) and not cls.endswith("_link"):
        if oneway:
            tf = fwd - turn_only_count(tags.get("turn:lanes") or tags.get("turn:lanes:forward"))
        else:
            tf = fwd - turn_only_count(tags.get("turn:lanes:forward"))
            tb = back - turn_only_count(tags.get("turn:lanes:backward"))
    tf = max(1, tf)
    tb = max(1, tb) if not oneway else 0
    mx = CLASS_MAX_PER_DIR[cls]
    clamped = tf > mx or tb > mx
    return min(tf, mx), (min(tb, mx) if not oneway else 0), clamped


SYN = [0]


def new_node(xz, osm_nodes, x, z):
    SYN[0] -= 1
    nid = SYN[0]
    xz[nid] = (x, z)
    osm_nodes[nid] = (LAT0 - z / KZ, LON0 + x / KX)
    return nid


def cumlen(nl, xz):
    cum = [0.0]
    for a, b in zip(nl, nl[1:]):
        cum.append(cum[-1] + math.hypot(xz[b][0] - xz[a][0], xz[b][1] - xz[a][1]))
    return cum


def insert_points(seg, dists, xz, osm_nodes, snap=1.0):
    """Insert nodes at the given distances along seg (sorted by distance, never by id).
    Returns the node id used for each requested distance (an existing vertex within `snap` m is reused)."""
    nl = seg["nodes"]
    cum = cumlen(nl, xz)
    L = cum[-1]
    res = {}
    todo = []
    for d in dists:
        d = min(max(d, 0.0), L)
        k = min(range(len(cum)), key=lambda i: abs(cum[i] - d))
        if abs(cum[k] - d) <= snap:
            res[d] = nl[k]
        else:
            todo.append(d)
    if todo:
        todo = sorted(set(todo))
        out, j = [], 0
        for i in range(len(nl)):
            if i > 0:
                while j < len(todo) and todo[j] < cum[i]:
                    d = todo[j]
                    t = (d - cum[i - 1]) / max(1e-9, cum[i] - cum[i - 1])
                    (x0, z0), (x1, z1) = xz[nl[i - 1]], xz[nl[i]]
                    nid = new_node(xz, osm_nodes, x0 + (x1 - x0) * t, z0 + (z1 - z0) * t)
                    out.append(nid)
                    res[d] = nid
                    j += 1
            out.append(nl[i])
        seg["nodes"] = out
    return [res[min(max(d, 0.0), L)] for d in dists]


def seg_dir_vec(s, xz, at_start):
    nl = s["nodes"]
    a, b = (xz[nl[0]], xz[nl[1]]) if at_start else (xz[nl[-1]], xz[nl[-2]])
    dx, dz = b[0] - a[0], b[1] - a[1]
    L = math.hypot(dx, dz) or 1.0
    return dx / L, dz / L


def continuation_chains(segs, xz, max_turn_cos=-0.82):
    """Carriageway chains: mutual straightest continuations (same category, same oneway,
    compatible name/ref, |turn| < ~35 deg). Returns [[(seg idx, forward?), ...], ...]."""
    inc = defaultdict(list)
    for i, s in enumerate(segs):
        if s["nodes"][0] == s["nodes"][-1]:
            continue
        inc[s["nodes"][0]].append((i, True))
        inc[s["nodes"][-1]].append((i, False))

    def compatible(a, b):
        if cat(a["class"]) != cat(b["class"]) or a["oneway"] != b["oneway"]:
            return False
        for k in ("ref", "name"):
            if a[k] and b[k] and a[k] != b[k]:
                return False
        return True

    best = {}
    for n, lst in inc.items():
        if len(lst) < 2:
            continue
        for (i, ai) in lst:
            si = segs[i]
            vi = seg_dir_vec(si, xz, ai)
            cand, bc = None, max_turn_cos
            for (j, aj) in lst:
                if j == i:
                    continue
                sj = segs[j]
                if not compatible(si["attrs"], sj["attrs"]):
                    continue
                if si["attrs"]["oneway"] and ai == aj:
                    continue
                vj = seg_dir_vec(sj, xz, aj)
                d = vi[0] * vj[0] + vi[1] * vj[1]
                if d < bc:
                    bc, cand = d, (j, aj)
            if cand:
                best[(i, ai)] = cand
    link = {k: v for k, v in best.items() if best.get(v) == k}
    seen, chains = set(), []
    for i0 in range(len(segs)):
        if i0 in seen or segs[i0]["nodes"][0] == segs[i0]["nodes"][-1]:
            continue
        cur, guard = (i0, True), {i0}
        while cur in link:
            j, aj = link[cur]
            if j in guard:
                break
            guard.add(j)
            cur = (j, not aj)
        chain, k, vis = [], cur, set()
        while True:
            j, aj = k
            if j in vis or j in seen:
                break
            vis.add(j)
            chain.append((j, aj))
            nxt = (j, not aj)
            if nxt not in link:
                break
            k = link[nxt]
        seen |= vis
        if chain:
            chains.append(chain)
    return chains


def smooth_runs(vals, lens, minrun, protect=None):
    """Replace short runs bounded by equal counts (or spikes/dips) with the neighbour count.
    Runs containing a protected element (synthesized acceleration/aux lanes) are kept."""
    changed = 0
    for _ in range(100000):
        runs = []
        for idx, (v, L) in enumerate(zip(vals, lens)):
            if runs and runs[-1][2] == v:
                runs[-1][1] = idx + 1
                runs[-1][3] += L
            else:
                runs.append([idx, idx + 1, v, L])
        best = None
        for r in range(1, len(runs) - 1):
            if runs[r][3] >= minrun:
                continue
            if protect and any(protect[q] for q in range(runs[r][0], runs[r][1])):
                continue
            a, b, c = runs[r - 1][2], runs[r][2], runs[r + 1][2]
            if a == c:
                nv = a
            elif b > max(a, c):
                nv = max(a, c)
            elif b < min(a, c):
                nv = min(a, c)
            else:
                continue
            if best is None or runs[r][3] < best[0]:
                best = (runs[r][3], r, nv)
        if best is None:
            break
        _, r, nv = best
        for idx in range(runs[r][0], runs[r][1]):
            vals[idx] = nv
        changed += 1
    return changed


def smooth_lanes(segs, xz, protect_aux=False):
    chains = continuation_chains(segs, xz)
    changed = 0
    for ch in chains:
        if len(ch) < 3:
            continue
        lens = [cumlen(segs[j]["nodes"], xz)[-1] for j, _ in ch]
        vf = [segs[j]["fwd"] if fw else segs[j]["back"] for j, fw in ch]
        vb = [segs[j]["back"] if fw else segs[j]["fwd"] for j, fw in ch]
        mr = MIN_RUN[cat(segs[ch[0][0]]["attrs"]["class"])]
        prot = [bool(segs[j].get("flags", 0) & FLAG_AUX_LANE) for j, _ in ch] if protect_aux else None
        c1 = smooth_runs(vf, lens, mr, prot)
        c2 = smooth_runs(vb, lens, mr, prot) if not segs[ch[0][0]]["attrs"]["oneway"] else 0
        if c1 or c2:
            changed += c1 + c2
            for (j, fw), a, b in zip(ch, vf, vb):
                s = segs[j]
                nf, nb = (a, b) if fw else (b, a)
                if (nf, nb) != (s["fwd"], s["back"]):
                    if protect_aux and (nf > s["fwd"] or nb > s["back"]):
                        s["flags"] = s.get("flags", 0) | FLAG_AUX_LANE   # carries an extended lane
                    s["fwd"], s["back"] = nf, nb
                    s["flags"] = s.get("flags", 0) | FLAG_LANES_SMOOTHED
    return len(chains), changed


# ----------------------------------------------------------------------------- lane conservation (D15)
def split_seg(segs, idx, dist, xz, osm_nodes):
    """Split segs[idx] at `dist` m; the second half is appended. Returns (idx, new idx)."""
    s = segs[idx]
    nid = insert_points(s, [dist], xz, osm_nodes, snap=3.0)[0]
    nl = s["nodes"]
    k = nl.index(nid)
    if k == 0 or k == len(nl) - 1:
        return idx, None
    b = dict(s)
    b["nodes"] = nl[k:]
    s["nodes"] = nl[:k + 1]
    segs.append(b)
    return idx, len(segs) - 1


def conserve_lanes(segs, xz, osm_nodes):
    """Lane conservation at nodes of the one-way freeway system (motorway/trunk + links)."""
    stats = Counter()
    seglen = lambda i: cumlen(segs[i]["nodes"], xz)[-1]

    def incidence():
        ins, outs, other = defaultdict(list), defaultdict(list), set()
        for i, s in enumerate(segs):
            a = s["attrs"]
            n0, n1 = s["nodes"][0], s["nodes"][-1]
            if a["oneway"] and cat(a["class"]) in ("fw", "link"):
                outs[n0].append(i)
                ins[n1].append(i)
            else:
                other.add(n0)
                other.add(n1)
        return ins, outs, other

    def walk(j, maxlen, ins, outs, other):
        path, acc = [j], seglen(j)
        while acc < maxlen:
            n = segs[path[-1]]["nodes"][-1]
            if n in other:
                return path, "surface", acc
            o, i = outs.get(n, []), ins.get(n, [])
            if len(o) >= 2 and len(i) == 1:
                return path, "diverge", acc
            if len(o) != 1 or len(i) != 1 or o[0] in path:
                return path, "node", acc
            path.append(o[0])
            acc += seglen(o[0])
        return path, "far", acc

    def mark(i, flag):
        segs[i]["flags"] = segs[i].get("flags", 0) | flag

    # 1) merges: an on-ramp (or freeway) joining without an added lane gets an acceleration
    #    lane (+1 for 250 m, then the drop; the renderer tapers it over 90 m) or, if the next
    #    diverge is close, an auxiliary lane that becomes the exit lane.
    ins, outs, other = incidence()
    merges = [n for n in list(outs) if len(ins.get(n, [])) >= 2 and len(outs[n]) == 1 and n not in other]
    for n in merges:
        if len(ins.get(n, [])) < 2 or len(outs.get(n, [])) != 1:
            continue
        o = outs[n][0]
        d = sum(segs[i]["fwd"] for i in ins[n]) - segs[o]["fwd"]
        if d <= 0:
            stats["merge: balanced or lane added"] += 1
            continue
        if cat(segs[o]["attrs"]["class"]) == "link":
            stats["merge: ramp-to-ramp, lanes merge at node"] += 1
            continue
        if segs[o]["fwd"] >= CLASS_MAX_PER_DIR[segs[o]["attrs"]["class"]]:
            stats["merge: mainline at the lane cap, lanes merge at node"] += 1
            continue
        path, kind, acc = walk(o, AUX_TO_EXIT_M, ins, outs, other)
        if kind == "diverge":
            for j in path:
                segs[j]["fwd"] += 1
                mark(j, FLAG_AUX_LANE)
            stats["merge: auxiliary lane added through to next exit"] += 1
        else:
            acc2 = 0.0
            for j in path:
                L = seglen(j)
                if acc2 + L <= ACCEL_M + 1.0:
                    segs[j]["fwd"] += 1
                    mark(j, FLAG_AUX_LANE)
                    acc2 += L
                    continue
                end_old = segs[j]["nodes"][-1]
                a, b = split_seg(segs, j, ACCEL_M - acc2, xz, osm_nodes)
                if b is not None:
                    mid = segs[a]["nodes"][-1]
                    ins[end_old] = [b if x == a else x for x in ins[end_old]]
                    ins[mid] = [a]
                    outs[mid] = [b]
                    segs[b]["flags"] = segs[b].get("flags", 0) & ~(FLAG_AUX_LANE | FLAG_LANE_DROP)
                segs[a]["fwd"] += 1
                mark(a, FLAG_AUX_LANE)
                mark(a, FLAG_LANE_DROP)
                break
            else:
                mark(path[-1], FLAG_LANE_DROP)
            stats["merge: acceleration lane synthesized"] += 1
        if d > 1:
            stats["merge: extra in-lanes merge at node (multi-lane ramp)"] += 1

    # 2) lane drops on a simple continuation: move to the next exit within 2 km, else keep it
    #    flagged (the sim's lane-end observation gives the warning; the renderer tapers it)
    ins, outs, other = incidence()
    for n in list(ins):
        if n in other or len(ins[n]) != 1 or len(outs.get(n, [])) != 1:
            continue
        i, o = ins[n][0], outs[n][0]
        if segs[i]["fwd"] <= segs[o]["fwd"] or i == o:
            continue
        path, kind, acc = walk(o, DROP_TO_EXIT_M, ins, outs, other)
        if kind == "diverge":
            target = segs[i]["fwd"]
            for j in path:
                if segs[j]["fwd"] < target:
                    segs[j]["fwd"] = target
                    mark(j, FLAG_AUX_LANE)
            stats["drop: moved to the next exit"] += 1
        else:
            mark(i, FLAG_LANE_DROP)
            stats["drop: kept, flagged (warning + taper)"] += 1

    return stats


def remove_flicker(segs, xz, window=300.0):
    """Last-resort flicker pass on one-way motorway/trunk carriageways, with the audit's own
    successor rule (same class, straightest continuation within 20 deg): a lane-count change
    that reverts within `window` m is removed, unless the run carries a synthesized lane."""
    outs = defaultdict(list)
    for i, s in enumerate(segs):
        a = s["attrs"]
        if a["oneway"] and a["class"] in ("motorway", "trunk"):
            outs[s["nodes"][0]].append(i)

    def head(i, end):
        nl = segs[i]["nodes"]
        a, b = (xz[nl[-2]], xz[nl[-1]]) if end else (xz[nl[0]], xz[nl[1]])
        return math.atan2(b[1] - a[1], b[0] - a[0])

    def succ(i):
        best, bd = None, 0.35
        for o in outs.get(segs[i]["nodes"][-1], ()):
            if segs[o]["attrs"]["class"] != segs[i]["attrs"]["class"] or o == i:
                continue
            dd = abs((head(o, False) - head(i, True) + math.pi) % (2 * math.pi) - math.pi)
            if dd < bd:
                bd, best = dd, o
        return best
    fixed = 0
    for i, s in enumerate(segs):
        a = s["attrs"]
        if not (a["oneway"] and a["class"] in ("motorway", "trunk")):
            continue
        j = succ(i)
        if j is None or segs[j]["fwd"] == s["fwd"]:
            continue
        run, acc, t = [j], cumlen(segs[j]["nodes"], xz)[-1], j
        for _ in range(20):
            if acc >= window:
                break
            t = succ(t)
            if t is None or t in run:
                break
            if segs[t]["fwd"] == s["fwd"]:
                if not any(segs[r].get("flags", 0) & FLAG_AUX_LANE for r in run + [i]):
                    for r in run:
                        segs[r]["fwd"] = s["fwd"]
                        segs[r]["flags"] = segs[r].get("flags", 0) | FLAG_LANES_SMOOTHED
                    fixed += 1
                break
            run.append(t)
            acc += cumlen(segs[t]["nodes"], xz)[-1]
    return fixed


def lane_balance(segs, xz):
    """Final lane balance at freeway-system nodes; refreshes LANE_DROP flags."""
    stats = Counter()
    ins, outs, other = defaultdict(list), defaultdict(list), set()
    for i, s in enumerate(segs):
        a = s["attrs"]
        n0, n1 = s["nodes"][0], s["nodes"][-1]
        if a["oneway"] and cat(a["class"]) in ("fw", "link"):
            outs[n0].append(i)
            ins[n1].append(i)
        else:
            other.add(n0)
            other.add(n1)
    for s in segs:
        s["flags"] = s.get("flags", 0) & ~FLAG_LANE_DROP
    for n in ins:
        if n not in other and len(ins[n]) == 1 and len(outs.get(n, [])) == 1:
            i, o = ins[n][0], outs[n][0]
            if segs[i]["fwd"] > segs[o]["fwd"]:
                segs[i]["flags"] |= FLAG_LANE_DROP
    unexplained = []
    for n in set(ins) | set(outs):
        if n in other:
            continue
        I, O = ins.get(n, []), outs.get(n, [])
        if not I or not O:
            continue
        si, so = sum(segs[i]["fwd"] for i in I), sum(segs[o]["fwd"] for o in O)
        if si == so:
            stats["final: balanced"] += 1
        elif len(I) == 1 and len(O) == 1:
            stats["final: simple add" if so > si else "final: simple drop (flagged)"] += 1
            if so < si and not (segs[I[0]].get("flags", 0) & FLAG_LANE_DROP):
                unexplained.append((n, si, so))
        elif len(I) == 1:
            stats["final: diverge, option lanes" if so > si else "final: diverge, lane ends at gore"] += 1
        elif len(O) == 1:
            if so > si:
                stats["final: merge, lane added"] += 1
            elif cat(segs[O[0]]["attrs"]["class"]) == "link":
                stats["final: ramp merge"] += 1
            elif any(segs[i].get("flags", 0) & FLAG_AUX_LANE for i in I) or \
                    (segs[O[0]].get("flags", 0) & FLAG_AUX_LANE):
                stats["final: merge with synthesized lane"] += 1
            elif segs[O[0]]["fwd"] >= CLASS_MAX_PER_DIR[segs[O[0]]["attrs"]["class"]]:
                stats["final: merge at the lane cap, lanes merge at node"] += 1
            else:
                stats["final: merge, lanes merge at node"] += 1
                unexplained.append((n, si, so))
        else:
            stats["final: weave node (n in, m out)"] += 1
    return stats, unexplained


# ----------------------------------------------------------------------------- elevation (D2)
def water_mask(water_path, cell=100.0):
    """Raster of water cells (even-odd fill of data/water.json) -> (fn(x,z)->bool)."""
    if not water_path or not os.path.exists(water_path):
        return lambda x, z: False
    w = json.load(open(water_path))
    rings = []
    for p in w["polygons"]:
        rings.append(p["outer"])
        rings.extend(p["holes"])
    zs = [pt[1] for r in rings for pt in r]
    xs = [pt[0] for r in rings for pt in r]
    z0, z1, x0 = min(zs), max(zs), min(xs)
    rows = {}
    edges = []
    for r in rings:
        for a, b in zip(r, r[1:] + r[:1]):
            if a[1] != b[1]:
                edges.append((a[0], a[1], b[0], b[1]))
    by_row = defaultdict(list)
    for (ax, az, bx, bz) in edges:
        lo, hi = min(az, bz), max(az, bz)
        for k in range(int((lo - z0) // cell), int((hi - z0) // cell) + 1):
            by_row[k].append((ax, az, bx, bz))
    for k, lst in by_row.items():
        zc = z0 + (k + 0.5) * cell
        xsx = sorted(ax + (zc - az) * (bx - ax) / (bz - az) for (ax, az, bx, bz) in lst
                     if (az <= zc < bz) or (bz <= zc < az))
        rows[k] = xsx
    import bisect

    def inside(x, z):
        k = int((z - z0) // cell)
        xsx = rows.get(k)
        if not xsx:
            return False
        return bisect.bisect_left(xsx, x) % 2 == 1
    return inside


def seg_layer(a):
    return a["layer"] if a["layer"] else 0


def is_elevated(a):
    return (a["bridge"] or a["layer"] > 0) and not a["tunnel"]


def solve_elevation(segs, xz, osm_nodes, water_path):
    in_water = water_mask(water_path)
    # ---- structures: connected elevated segments (any class) -> deck height
    parent = {}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    elev_idx = [i for i, s in enumerate(segs) if is_elevated(s["attrs"])]
    by_node = defaultdict(list)
    for i in elev_idx:
        parent[i] = i
        by_node[segs[i]["nodes"][0]].append(i)
        by_node[segs[i]["nodes"][-1]].append(i)
    for lst in by_node.values():
        for j in lst[1:]:
            ra, rb = find(lst[0]), find(j)
            if ra != rb:
                parent[rb] = ra
    water_len = defaultdict(float)
    for i in elev_idx:
        nl = segs[i]["nodes"]
        for a, b in zip(nl, nl[1:]):
            L = math.hypot(xz[b][0] - xz[a][0], xz[b][1] - xz[a][1])
            k = max(1, int(L // 50))
            for t in range(k):
                f = (t + 0.5) / k
                x = xz[a][0] + (xz[b][0] - xz[a][0]) * f
                z = xz[a][1] + (xz[b][1] - xz[a][1]) * f
                if in_water(x, z):
                    water_len[find(i)] += L / k
    struct_len = defaultdict(float)
    for i in elev_idx:
        struct_len[find(i)] += cumlen(segs[i]["nodes"], xz)[-1]
    deck = {}
    nstruct = len({find(i) for i in elev_idx})
    nwater = sum(1 for r in {find(i) for i in elev_idx} if water_len[r] > 150 and struct_len[r] > 400)
    for i in elev_idx:
        a = segs[i]["attrs"]
        L = max(1, a["layer"])
        h = LAYER_CLEAR * L
        if water_len[find(i)] > 150 and struct_len[find(i)] > 400:
            h = max(h, WATER_DECK + LAYER_CLEAR * (L - 1))
        deck[i] = h

    def build_graph():
        adj = defaultdict(list)
        for i, s in enumerate(segs):
            g = GRADE[cat(s["attrs"]["class"])]
            nl = s["nodes"]
            for a, b in zip(nl, nl[1:]):
                L = math.hypot(xz[b][0] - xz[a][0], xz[b][1] - xz[a][1])
                adj[a].append((b, L, g))
                adj[b].append((a, L, g))
        return adj

    import heapq

    NEG = -1e9

    def propagate(src, caps=None):
        """Max-plus: lo(v) = max over sources (h - g * dist); NEG where unreachable.
        caps: node -> max value (surface junctions pinned to grade)."""
        adj = build_graph()
        caps = caps or {}
        y = defaultdict(lambda: NEG)
        for v, h in src.items():
            h = min(h, caps.get(v, 1e18))
            if h > y[v]:
                y[v] = h
        heap = [(-h, v) for v, h in list(y.items())]
        heapq.heapify(heap)
        while heap:
            nh, v = heapq.heappop(heap)
            h = -nh
            if h < y[v] - 1e-9:
                continue
            for (w, L, g) in adj[v]:
                c = min(h - g * L, caps.get(w, 1e18))
                if c > y[w] + 1e-6 and c > -60.0:
                    y[w] = c
                    heapq.heappush(heap, (-c, w))
        return y

    src = {}
    for i in elev_idx:
        for n in segs[i]["nodes"]:
            src[n] = max(src.get(n, 0.0), deck[i])
    # surface junctions (>= 3 roads, none a bridge / layer / tunnel) stay at grade (audit r7);
    # a pin is released only where it would force a grade above the class limit
    deg = defaultdict(set)
    struct_node = set()
    for i, sg in enumerate(segs):
        for n in (sg["nodes"][0], sg["nodes"][-1]):
            deg[n].add(i)
            a = sg["attrs"]
            if a["bridge"] or a["layer"] != 0 or a["tunnel"] or cat(a["class"]) in ("fw", "link"):
                struct_node.add(n)      # structures and the freeway system are not pinned
    pinned = {n for n, ss in deg.items() if len(ss) >= 3 and n not in struct_node}
    n_pinned0 = len(pinned)
    LH = [None, None]
    sunk = {}      # tunnel segment -> depth (the whole tunnel interior sits at this depth)

    def propagate_all(src):
        """y = max(lo, min(hi, 0)): lo = ramps down from decks / crossing lifts (max-plus),
        hi = ramps up from tunnels sunk 7 m (min-plus). Both are g-Lipschitz along the road,
        so y is too (grade <= class limit); where a deck and a tunnel compete the deck wins."""
        lo = propagate(src, {n: 0.0 for n in pinned})
        sinks = {}
        sunk.clear()
        for i, sg in enumerate(segs):
            if sg["attrs"]["tunnel"]:
                d = -LAYER_CLEAR * max(1, abs(sg["attrs"]["layer"]))
                sunk[i] = d
                for n in sg["nodes"]:
                    sinks[n] = min(sinks.get(n, 0.0), d)
        hi_neg = propagate({k: -v for k, v in sinks.items()})
        hi = defaultdict(lambda: -NEG)
        for k, v in hi_neg.items():
            if v > NEG:
                hi[k] = -v
        LH[0], LH[1] = lo, hi
        y = defaultdict(float)
        for k in set(lo) | set(hi):
            y[k] = max(lo[k], min(hi[k], 0.0))
        return y
    y = propagate_all(src)

    def release_pins():
        bad = set()
        for i, sg in enumerate(segs):
            lim = 0.08                       # schema limit for every class
            nl = sg["nodes"]
            for a_, b_ in zip(nl, nl[1:]):
                L = math.hypot(xz[b_][0] - xz[a_][0], xz[b_][1] - xz[a_][1])
                if L > 0.5 and abs(y[b_] - y[a_]) > lim * L + 1e-6:
                    bad.update(n for n in (a_, b_) if n in pinned)
        return bad
    for _ in range(6):
        rel = release_pins()
        if not rel:
            break
        pinned.difference_update(rel)
        y = propagate_all(src)

    # ---- fill sags between nearby raised sections on one carriageway (audit r4): where the
    # profile dips >= 3 m below raised points on both sides within 600 m, keep it at the lower
    # of the two raised levels (a continuous viaduct instead of a roller coaster)
    sag_chains = [ch for ch in continuation_chains(segs, xz) if cat(segs[ch[0][0]]["attrs"]["class"]) in ("fw", "link", "art")]

    def fill_sags(y):
        filled = 0
        for rnd in range(12):
            want = defaultdict(list)
            for ch in sag_chains:
                prof = []          # (s along chain, y, seg, dist along seg)
                acc = 0.0
                for (j, fwd) in ch:
                    nl = segs[j]["nodes"] if fwd else segs[j]["nodes"][::-1]
                    cum = cumlen(nl, xz)
                    L = cum[-1]
                    g = GRADE[cat(segs[j]["attrs"]["class"])]
                    base = deck.get(j, NEG)
                    for k in range(len(nl)):
                        prof.append((acc + cum[k], y[nl[k]], j, cum[k] if fwd else L - cum[k], nl[k]))
                        if k < len(nl) - 1:      # the dip inside a piece (max-plus profile)
                            ya, yb, Lp = y[nl[k]], y[nl[k + 1]], cum[k + 1] - cum[k]
                            t = (ya - yb + g * Lp) / (2 * g)
                            if 1.0 < t < Lp - 1.0:
                                yt = max(base, 0.0 if base == NEG else base, ya - g * t, yb - g * (Lp - t)) if base != NEG else max(0.0, ya - g * t, yb - g * (Lp - t))
                                d = cum[k] + t
                                prof.append((acc + d, yt, j, d if fwd else L - d, None))
                    acc += L
                if len(prof) < 3:
                    continue
                prof.sort(key=lambda p_: p_[0])
                ss = [p_[0] for p_ in prof]
                prof_nodes = [p_[4] for p_ in prof]
                import bisect
                for i, (s0, y0, j, dj, _n) in enumerate(prof):
                    a, b = bisect.bisect_left(ss, s0 - 600), bisect.bisect_right(ss, s0 + 600)
                    left = max((prof[q][1] for q in range(a, i)), default=0.0)
                    right = max((prof[q][1] for q in range(i + 1, b)), default=0.0)
                    tgt = min(left, right)
                    if cat(segs[j]["attrs"]["class"]) == "art" and tgt >= 3.0 and tgt - y0 >= 3.0 - 1e-6:
                        # an arterial dipping between two structures is filled only if no at-grade
                        # (pinned) junction lies between the two raised points; otherwise the road
                        # really comes down to that junction
                        li = max(range(a, i), key=lambda q: prof[q][1], default=None)
                        ri = max(range(i + 1, b), key=lambda q: prof[q][1], default=None)
                        if li is None or ri is None or any(prof_nodes[q] in pinned for q in range(li, ri + 1) if prof_nodes[q] is not None):
                            continue
                    if tgt >= 3.0 and tgt - y0 >= 3.0 - 1e-6 and not segs[j]["attrs"]["tunnel"]:
                        want[j].append((dj, tgt))
            if not want:
                break
            for j, lst in want.items():
                ids = insert_points(segs[j], [d for d, _ in lst], xz, osm_nodes, snap=5.0)
                for nid, (_, h) in zip(ids, lst):
                    if h > src.get(nid, 0.0):
                        src[nid] = h
                        filled += 1
            y = propagate_all(src)

        return y, filled

    y, filled = fill_sags(y)
    print(f"   sag filling: {filled} raised points between nearby structures", file=sys.stderr)

    # ---- crossings without a shared node: lift the upper road to >= under + 7 m
    def checked(a):
        return cat(a["class"]) in ("fw", "link")

    def crossings(y):
        cell = 200.0
        grid = defaultdict(list)
        pieces = []
        for i, s in enumerate(segs):
            a = s["attrs"]
            if not (checked(a) or is_elevated(a) or a["tunnel"]):
                continue
            nl = s["nodes"]
            for k in range(len(nl) - 1):
                p, q = nl[k], nl[k + 1]
                pid = len(pieces)
                pieces.append((i, k, p, q))
                (px, pz), (qx, qz) = xz[p], xz[q]
                for gx in range(int(min(px, qx) // cell), int(max(px, qx) // cell) + 1):
                    for gz in range(int(min(pz, qz) // cell), int(max(pz, qz) // cell) + 1):
                        grid[(gx, gz)].append(pid)
        res = []
        seen = set()
        for i, s in enumerate(segs):
            a = s["attrs"]
            nl = s["nodes"]
            for k in range(len(nl) - 1):
                p, q = nl[k], nl[k + 1]
                (px, pz), (qx, qz) = xz[p], xz[q]
                cand = set()
                for gx in range(int(min(px, qx) // cell), int(max(px, qx) // cell) + 1):
                    for gz in range(int(min(pz, qz) // cell), int(max(pz, qz) // cell) + 1):
                        cand.update(grid.get((gx, gz), ()))
                for pid in cand:
                    j, kk, r, t = pieces[pid]
                    if j == i or len({p, q, r, t}) < 4:
                        continue
                    b = segs[j]["attrs"]
                    if not (checked(a) or checked(b)):
                        if cat(a["class"]) == "loc" and cat(b["class"]) == "loc":
                            continue
                    key = (min((i, k), (j, kk)), max((i, k), (j, kk)))
                    if key in seen:
                        continue
                    seen.add(key)
                    (rx, rz), (tx, tz) = xz[r], xz[t]
                    d1x, d1z, d2x, d2z = qx - px, qz - pz, tx - rx, tz - rz
                    den = d1x * d2z - d1z * d2x
                    hit = False
                    if abs(den) >= 1e-9:
                        u = ((rx - px) * d2z - (rz - pz) * d2x) / den
                        v = ((rx - px) * d1z - (rz - pz) * d1x) / den
                        hit = 0 < u < 1 and 0 < v < 1
                    if not hit and (a["tunnel"] or b["tunnel"]):
                        # a road running over / along a tunnel without crossing it in plan: an
                        # overlap of the pavements counts like a crossing (the road must pass over)
                        L1, L2 = math.hypot(d1x, d1z), math.hypot(d2x, d2z)
                        if L1 > 1 and L2 > 1:
                            mx_, mz_ = (px + qx) / 2, (pz + qz) / 2
                            v = max(0.0, min(1.0, ((mx_ - rx) * d2x + (mz_ - rz) * d2z) / (L2 * L2)))
                            dd = math.hypot(rx + d2x * v - mx_, rz + d2z * v - mz_)
                            hwa = (segs[i]["fwd"] + segs[i]["back"]) * LANE_WIDTH_CLASS[a["class"]] / 2
                            hwb = (segs[j]["fwd"] + segs[j]["back"]) * LANE_WIDTH_CLASS[b["class"]] / 2
                            if dd < hwa + hwb - 0.5 and 0.0 < v < 1.0:
                                u, hit = 0.5, True
                    if not hit:
                        continue
                    ya = y[p] + (y[q] - y[p]) * u
                    yb = y[r] + (y[t] - y[r]) * v
                    res.append((i, k, u, ya, j, kk, v, yb))
        return res

    def upper(i, ya, j, yb):
        a, b = segs[i]["attrs"], segs[j]["attrs"]
        if a["tunnel"] != b["tunnel"]:
            return j if a["tunnel"] else i
        if a["layer"] != b["layer"]:
            return i if a["layer"] > b["layer"] else j
        if a["bridge"] != b["bridge"]:
            return i if a["bridge"] else j
        if abs(ya - yb) > 0.5:
            return i if ya > yb else j
        ra, rb = ALL_CLASSES.index(a["class"]), ALL_CLASSES.index(b["class"])
        return i if ra > rb else j

    def near_connected(src_nodes, targets, limit=300.0):
        """True if a target vertex is reachable from src_nodes within `limit` m of road."""
        adj = graph_cache[0]
        dist = {v: 0.0 for v in src_nodes}
        heap = [(0.0, v) for v in src_nodes]
        while heap:
            d, v = heapq.heappop(heap)
            if v in targets:
                return True
            if d > dist.get(v, 1e18):
                continue
            for (w, L, g) in adj[v]:
                nd = d + L
                if nd <= limit and nd < dist.get(w, 1e18):
                    dist[w] = nd
                    heapq.heappush(heap, (nd, w))
        return False

    tight = []
    gore = set()   # near-parallel crossings of roads connected within 300 m: merge/diverge overlaps

    def low(cs):
        out = []
        for c in cs:
            if abs(c[3] - c[7]) >= MIN_CLEAR:
                continue
            i, k, u, ya, j, kk, v, yb = c
            key = (segs[i]["nodes"][k], segs[i]["nodes"][k + 1], segs[j]["nodes"][kk], segs[j]["nodes"][kk + 1])
            if key in gore:
                continue
            a_nodes = {segs[i]["nodes"][k], segs[i]["nodes"][k + 1]}
            b_nodes = {segs[j]["nodes"][kk], segs[j]["nodes"][kk + 1]}
            (p0x, p0z), (p1x, p1z) = xz[segs[i]["nodes"][k]], xz[segs[i]["nodes"][k + 1]]
            (q0x, q0z), (q1x, q1z) = xz[segs[j]["nodes"][kk]], xz[segs[j]["nodes"][kk + 1]]
            la, lb = math.hypot(p1x - p0x, p1z - p0z), math.hypot(q1x - q0x, q1z - q0z)
            cosang = abs((p1x - p0x) * (q1x - q0x) + (p1z - p0z) * (q1z - q0z)) / max(1e-9, la * lb)
            if cosang > 0.9 and near_connected(a_nodes, b_nodes):
                gore.add(key)      # merge/diverge overlap of connected roads, not a grade separation
                continue
            if near_connected(a_nodes, b_nodes, 160.0):
                # connected within 160 m: a 7 m separation would need > 5 % (OSM junction
                # geometry / missing shared node); left at grade and reported
                gore.add(key)
                tight.append((segs[i]["attrs"]["class"], segs[i]["attrs"]["name"], segs[j]["attrs"]["class"],
                              segs[j]["attrs"]["name"], round(xz[segs[i]["nodes"][k]][0]), round(xz[segs[i]["nodes"][k]][1])))
                continue
            out.append(c)
        return out

    graph_cache = [build_graph()]
    lifted = 0

    def crossing_rounds(y):
        nonlocal lifted
        for rnd in range(10):
            graph_cache[0] = build_graph()
            bad = low(crossings(y))
            print(f"   elevation round {rnd}: {len(bad)} crossings without a shared node below {MIN_CLEAR} m",
                  file=sys.stderr)
            if not bad:
                break
            want = defaultdict(list)   # seg -> [(dist, height)]
            for (i, k, u, ya, j, kk, v, yb) in bad:
                up = upper(i, ya, j, yb)
                if up == i:
                    seg, kp, f, need = i, k, u, yb + CROSS_CLEAR
                else:
                    seg, kp, f, need = j, kk, v, ya + CROSS_CLEAR
                cum = cumlen(segs[seg]["nodes"], xz)
                want[seg].append((cum[kp] + (cum[kp + 1] - cum[kp]) * f, need))
            for seg, lst in want.items():
                ids = insert_points(segs[seg], [d for d, _ in lst], xz, osm_nodes, snap=2.0)
                for nid, (_, h) in zip(ids, lst):
                    src[nid] = max(src.get(nid, 0.0), h)
                    pinned.discard(nid)          # a crossing lift point is never held at grade
                    lifted += 1
            y = propagate_all(src)
            for _ in range(6):
                rel = release_pins()
                if not rel:
                    break
                pinned.difference_update(rel)
                y = propagate_all(src)

        return y

    y = crossing_rounds(y)
    y, filled2 = fill_sags(y)          # lifts can open new sags between structures
    filled += filled2
    y = crossing_rounds(y)
    for _ in range(6):
        rel = release_pins()
        if not rel:
            break
        pinned.difference_update(rel)
        y = propagate_all(src)
    graph_cache[0] = build_graph()
    remaining = low(crossings(y))

    # ---- shape points on ramps: sample the exact profile every <= 25 m wherever y != 0
    # (the profile is g-Lipschitz, so straight lines between samples keep the grade limit)
    lo, hi = LH
    kinks = 0
    for si, s in enumerate(segs):
        g = GRADE[cat(s["attrs"]["class"])]
        base_lo = deck.get(si, NEG)
        nl = s["nodes"]
        cum = cumlen(nl, xz)
        ins = []
        for k in range(len(nl) - 1):
            p, q, L = nl[k], nl[k + 1], cum[k + 1] - cum[k]
            if L < 30.0 or (abs(y[p]) <= 1e-6 and abs(y[q]) <= 1e-6):
                continue
            if abs(y[p] - y[q]) >= g * L - 1e-6:    # a steep (pinned) piece is a straight ramp
                continue
            n_in = int(L // 25.0)
            for t_i in range(1, n_in + 1):
                t = L * t_i / (n_in + 1)
                lo_t = max(base_lo, lo[p] - g * t, lo[q] - g * (L - t))
                hi_t = min(hi[p] + g * t, hi[q] + g * (L - t))
                if si in sunk:
                    hi_t = min(hi_t, sunk[si])     # the tunnel interior stays at its depth
                h = max(lo_t, min(hi_t, 0.0))
                if abs(h - (y[p] + (y[q] - y[p]) * t / L)) > 0.02:
                    ins.append((cum[k] + t, h))
        if ins:
            ids = insert_points(s, [d for d, _ in ins], xz, osm_nodes, snap=0.5)
            for nid, (_, h) in zip(ids, ins):
                if nid not in y or (y[nid] == 0.0 and nid < 0):
                    y[nid] = h
            kinks += len(ins)
    stats = {"structures": nstruct, "waterStructures": nwater, "elevatedSegments": len(elev_idx),
             "crossingLifts": lifted, "kinkNodes": kinks, "remainingLowCrossings": len(remaining),
             "goreOverlapsConnectedWithin300m": len(gore) - len(tight), "tightConnectedCrossings": len(tight),
             "surfaceJunctionsPinned": len(pinned), "pinsReleased": n_pinned0 - len(pinned)}
    if tight:
        print(f"   tight crossings (connected within 160 m, left at grade): {tight[:12]}", file=sys.stderr)
    return y, stats, remaining


def check_geometry(segs, xz, len0):
    """D1: no 180-degree fold-backs; edge length <= 1.05 x its pre-insertion length."""
    folds, longer = [], []
    for i, s in enumerate(segs):
        nl = s["nodes"]
        for k in range(1, len(nl) - 1):
            a, b, c = xz[nl[k - 1]], xz[nl[k]], xz[nl[k + 1]]
            ux, uz, vx, vz = b[0] - a[0], b[1] - a[1], c[0] - b[0], c[1] - b[1]
            lu, lv = math.hypot(ux, uz), math.hypot(vx, vz)
            if lu < 1 or lv < 1:
                continue
            if (ux * vx + uz * vz) / (lu * lv) < -0.95:
                folds.append((i, s["attrs"]["name"], nl[k]))
                break
        L = cumlen(nl, xz)[-1]
        L0 = len0.get(id(s))
        if L0 and L > 1.05 * L0 + 0.5:
            longer.append((i, s["attrs"]["name"], L, L0))
    return folds, longer


# ----------------------------------------------------------------------------- build
def build(paths, bbox, excludes, simplify=True, water_path=None):
    print("== loading", file=sys.stderr)
    osm_nodes, ways = load_inputs(paths)
    print(f"   merged: {len(osm_nodes)} nodes, {len(ways)} ways", file=sys.stderr)

    # ---- clip: keep nodes in bbox and outside exclusion boxes
    keep_node = {}
    for nid, (lat, lon) in osm_nodes.items():
        ok = in_box(lat, lon, bbox) and not any(in_box(lat, lon, ex) for ex in excludes)
        keep_node[nid] = ok
    xz = {nid: project(*ll) for nid, ll in osm_nodes.items() if keep_node[nid]}

    # ---- way attributes + runs of kept nodes
    runs = []   # (attrs, [node ids])
    stats_ways = defaultdict(lambda: {"ways": 0, "fallback": 0, "clamped": 0, "turnpockets": 0})
    for wid in sorted(ways):
        w = ways[wid]
        tags = w.get("tags") or {}
        cls = tags["highway"]
        ow = str(tags.get("oneway", "")).lower()
        junction = str(tags.get("junction", "")).lower()
        oneway = ow in ("yes", "true", "1", "-1") or junction in ("roundabout", "circular")
        if ow in ("no", "false", "0"):
            oneway = False
        if len([n for n in w["nodes"] if keep_node.get(n)]) < 2:
            continue
        fwd_raw, back_raw, fallback = lane_counts(tags, cls, oneway)
        fwd, back, clamped = through_lanes(tags, cls, oneway, fwd_raw, back_raw, fallback)
        attrs = {
            "class": cls, "oneway": oneway, "fwd": fwd, "back": back, "fallback": fallback,
            "fwd_raw": fwd_raw, "back_raw": back_raw, "clamped": clamped,
            "bridge": truthy(tags.get("bridge")),
            "tunnel": truthy(tags.get("tunnel")) or str(tags.get("covered", "")).lower() == "yes",
            "layer": parse_int_signed(tags.get("layer")) or 0,
            "maxspeed": parse_maxspeed(tags.get("maxspeed")),
            "name": tags.get("name"), "ref": tags.get("ref"),
            "bridge_name": tags.get("bridge:name") or tags.get("name"),
            "sidewalk": side_set(tags, "sidewalk"), "parking": side_set(tags, "parking"),
            "width_tag": parse_width(tags.get("width") or tags.get("est_width")),
            "way": wid,
        }
        stats_ways[cls]["ways"] += 1
        if fallback:
            stats_ways[cls]["fallback"] += 1
        if clamped:
            stats_ways[cls]["clamped"] += 1
        if (fwd, back) != (min(fwd_raw, CLASS_MAX_PER_DIR[cls]), min(back_raw, CLASS_MAX_PER_DIR[cls]) if not oneway else 0):
            stats_ways[cls]["turnpockets"] += 1
        wn = w["nodes"] if ow != "-1" else w["nodes"][::-1]
        cur = []
        for n in wn:
            if keep_node.get(n, False):
                cur.append(n)
            else:
                if len(cur) >= 2:
                    runs.append((attrs, cur))
                cur = []
        if len(cur) >= 2:
            runs.append((attrs, cur))

    # ---- junction split
    use = defaultdict(int)
    for _, nl in runs:
        for n in nl:
            use[n] += 1
    segs = []
    for attrs, nl in runs:
        seen = defaultdict(int)
        for n in nl:
            seen[n] += 1
        cut = [0] + [i for i in range(1, len(nl) - 1) if use[nl[i]] > 1 or seen[nl[i]] > 1] + [len(nl) - 1]
        for a, b in zip(cut, cut[1:]):
            s = nl[a:b + 1]
            if len(s) >= 2 and not (len(s) == 2 and s[0] == s[1]):
                segs.append({"attrs": attrs, "nodes": s, "fwd": attrs["fwd"], "back": attrs["back"],
                             "flags": FLAG_LANES_CLAMPED if attrs["clamped"] else 0})
    n_nodes_before = len({n for s in segs for n in s["nodes"]})
    n_segs_before = len(segs)
    print(f"   after clip+split: {n_segs_before} segments, {n_nodes_before} nodes", file=sys.stderr)

    # ---- lane-count smoothing along carriageway chains (D4)
    nch, nchg = smooth_lanes(segs, xz)
    print(f"   lane smoothing: {nch} carriageway chains, {nchg} short runs replaced "
          f"(min run 400 m freeway / 200 m other)", file=sys.stderr)

    # ---- merge degree-2 nodes with identical attributes (smoothed lane counts)
    if simplify:
        def sig(s):
            a = s["attrs"]
            return (a["class"], a["oneway"], s["fwd"], s["back"], a["bridge"], a["tunnel"],
                    a["layer"], a["maxspeed"], a["name"], a["ref"])
        node_segs = defaultdict(list)
        for i, s in enumerate(segs):
            node_segs[s["nodes"][0]].append(i)
            node_segs[s["nodes"][-1]].append(i)
        alive = [True] * len(segs)
        merged = []

        def can_merge(i, j):
            si, sj = segs[i], segs[j]
            if i == j or sig(si) != sig(sj):
                return False
            if si["nodes"][0] == si["nodes"][-1] or sj["nodes"][0] == sj["nodes"][-1]:
                return False
            return True

        for i in range(len(segs)):
            if not alive[i]:
                continue
            s = segs[i]
            nodes_ = list(s["nodes"])
            oneway = s["attrs"]["oneway"]
            flags = s.get("flags", 0)
            alive[i] = False
            chain_ids = {i}
            for direction in ("fwd", "back"):
                while True:
                    end = nodes_[-1] if direction == "fwd" else nodes_[0]
                    cand = [j for j in node_segs[end] if alive[j]]
                    others = [j for j in node_segs[end] if j not in chain_ids]
                    if len(node_segs[end]) != 2 or len(cand) != 1 or len(others) != 1:
                        break
                    j = cand[0]
                    if not can_merge(i, j):
                        break
                    t = segs[j]["nodes"]
                    if direction == "fwd":
                        if t[0] == end:
                            ext = t[1:]
                        elif not oneway and t[-1] == end:
                            ext = t[-2::-1]
                        else:
                            break
                        if ext and ext[-1] == nodes_[0]:
                            break
                        nodes_.extend(ext)
                    else:
                        if t[-1] == end:
                            ext = t[:-1]
                        elif not oneway and t[0] == end:
                            ext = t[:0:-1]
                        else:
                            break
                        if ext and ext[0] == nodes_[-1]:
                            break
                        nodes_ = ext + nodes_
                    alive[j] = False
                    flags |= segs[j].get("flags", 0)
                    chain_ids.add(j)
            merged.append({"attrs": s["attrs"], "nodes": nodes_, "fwd": s["fwd"], "back": s["back"],
                           "flags": flags})
        segs = merged
        print(f"   after degree-2 merge: {len(segs)} segments "
              f"(-{n_segs_before - len(segs)})", file=sys.stderr)

        # ---- Douglas-Peucker per segment (endpoints + junction nodes kept)
        removed = 0
        for s in segs:
            nl = s["nodes"]
            if len(nl) < 3:
                continue
            tol = 2.0 if s["attrs"]["class"] in FAST_CLASSES else 4.0
            pts = [xz[n] for n in nl]
            keep_idx = dp_simplify(pts, tol)
            new = [nl[i] for i in keep_idx]
            removed += len(nl) - len(new)
            s["nodes"] = new
        spikes = 0
        for s in segs:   # drop interior fold-back spikes (OSM digitising errors)
            nl = s["nodes"]
            changed = True
            while changed and len(nl) > 2:
                changed = False
                for k in range(1, len(nl) - 1):
                    a, b, c = xz[nl[k - 1]], xz[nl[k]], xz[nl[k + 1]]
                    ux, uz, vx, vz = b[0] - a[0], b[1] - a[1], c[0] - b[0], c[1] - b[1]
                    lu, lv = math.hypot(ux, uz), math.hypot(vx, vz)
                    if lu > 1e-6 and lv > 1e-6 and (ux * vx + uz * vz) / (lu * lv) < -0.95:
                        del nl[k]
                        spikes += 1
                        changed = True
                        break
        print(f"   removed {spikes} fold-back spike vertices", file=sys.stderr)
        n_nodes_after = len({n for s in segs for n in s["nodes"]})
        print(f"   after Douglas-Peucker: {n_nodes_after} nodes (-{n_nodes_before - n_nodes_after}, "
              f"{removed} shape points dropped)", file=sys.stderr)

    # ---- lane conservation on the freeway system (D15)
    cstats = conserve_lanes(segs, xz, osm_nodes)
    nch2, nchg2 = smooth_lanes(segs, xz, protect_aux=True)
    print(f"   post-conservation smoothing: {nchg2} short runs replaced (synthesized lanes kept)", file=sys.stderr)
    nfl = remove_flicker(segs, xz)
    print(f"   flicker pass (audit successor rule): {nfl} short reverting runs removed", file=sys.stderr)
    c2 = conserve_lanes(segs, xz, osm_nodes)     # second pass for merges changed by the smoothing
    for k, v in c2.items():
        cstats["pass 2 " + k] += v
    capped = 0
    for sg in segs:   # final cap: class max (+1 where a synthesized lane runs)
        cap = CLASS_MAX_PER_DIR[sg["attrs"]["class"]] + (1 if sg.get("flags", 0) & FLAG_AUX_LANE else 0)
        if sg["fwd"] > cap or sg["back"] > cap:
            sg["fwd"], sg["back"] = min(sg["fwd"], cap), min(sg["back"], cap)
            sg["flags"] = sg.get("flags", 0) | FLAG_LANES_CLAMPED
            capped += 1
    print(f"   final lane cap applied to {capped} segments", file=sys.stderr)
    bstats, unexplained = lane_balance(segs, xz)
    cstats.update(bstats)
    print("   lane conservation at freeway nodes:", file=sys.stderr)
    for k in sorted(cstats):
        print(f"      {k:58} {cstats[k]}", file=sys.stderr)
    print(f"      unexplained imbalances: {len(unexplained)}", file=sys.stderr)

    # ---- elevation: structures, ramps on approaches, crossing clearance (D2)
    len0 = {id(s): cumlen(s["nodes"], xz)[-1] for s in segs}
    node_y, estats, low = solve_elevation(segs, xz, osm_nodes, water_path)
    print(f"   elevation: {estats}", file=sys.stderr)

    # ---- lengths
    for s in segs:
        s["cum"] = cumlen(s["nodes"], xz)
        s["length"] = s["cum"][-1]
    segs = [s for s in segs if s["length"] > 0]

    # ---- build checks (fail the build)
    folds, longer = check_geometry(segs, xz, len0)
    bad_grade = []
    for i, s in enumerate(segs):
        lim = 0.08 + 1e-3
        nl = s["nodes"]
        for a, b in zip(nl, nl[1:]):
            L = math.hypot(xz[b][0] - xz[a][0], xz[b][1] - xz[a][1])
            if L > 0.5 and abs(node_y.get(b, 0) - node_y.get(a, 0)) / L > lim:
                bad_grade.append((i, s["attrs"]["name"], round(abs(node_y.get(b, 0) - node_y.get(a, 0)) / L, 3)))
                break
    low = [(segs[i]["attrs"]["class"], segs[i]["attrs"]["name"], segs[j]["attrs"]["class"], segs[j]["attrs"]["name"],
            round(ya, 1), round(yb, 1), [round(v) for v in xz[segs[i]["nodes"][k]]]) for (i, k, u, ya, j, kk, v, yb) in low]
    checks = {"folds": folds, "tooLong": longer, "grade>8%": bad_grade, "lowCrossings": low,
              "unexplainedLaneImbalance": unexplained}
    print("   build checks:", {k: len(v) for k, v in checks.items()}, file=sys.stderr)
    node_y = {n: node_y.get(n, 0.0) for s in segs for n in s["nodes"]}
    return osm_nodes, xz, node_y, segs, stats_ways, checks


# ----------------------------------------------------------------------------- output
def directed_edges(segs):
    """Expand segments to directed edges (twins for two-way)."""
    edges = []
    for s in segs:
        a = s["attrs"]
        base = {"seg": s, "oneway": a["oneway"]}
        f = dict(base, nodes=s["nodes"], lanes=s["fwd"], opp=s["back"],
                 raw=(a["fwd_raw"], a["back_raw"]), rev=False)
        edges.append(f)
        if not a["oneway"]:
            b = dict(base, nodes=s["nodes"][::-1], lanes=s["back"], opp=s["fwd"],
                     raw=(a["back_raw"], a["fwd_raw"]), rev=True)
            edges.append(b)
            f["twin"] = len(edges) - 1
            b["twin"] = len(edges) - 2
        else:
            f["twin"] = -1
    return edges



# ----------------------------------------------------------------------------- lane hops (mirrors src/geom/hops.js)
FW_CLASSES = {"motorway", "motorway_link", "trunk", "trunk_link"}


def compute_hops(edges, xz, order="geometry"):
    """Python mirror of the shared hop rule (src/geom/hops.js): returns hop(i, j) -> (strict, shift).
    order="geometry": diverging/merging edges are ordered right-to-left by the lateral offset of
    the point 50 m into (out) / before (in) the edge, in the frame of the edge being continued;
    ties under 1 m fall back to the turn angle. order="turn": the sim's legacy turn-angle order."""
    import functools
    out_, in_ = defaultdict(list), defaultdict(list)
    for i, e in enumerate(edges):
        out_[e["nodes"][0]].append(i)
        in_[e["nodes"][-1]].append(i)

    def head(i, end):
        nl = edges[i]["nodes"]
        a, b = (xz[nl[-2]], xz[nl[-1]]) if end else (xz[nl[0]], xz[nl[1]])
        return math.atan2(b[1] - a[1], b[0] - a[0])

    def turn(i, j):
        x = head(j, False) - head(i, True)
        while x > math.pi: x -= 2 * math.pi
        while x < -math.pi: x += 2 * math.pi
        return x

    def point(i, dist, from_end):
        nl = edges[i]["nodes"]
        if from_end:
            nl = nl[::-1]
        acc = 0.0
        for a, b in zip(nl, nl[1:]):
            L = math.hypot(xz[b][0] - xz[a][0], xz[b][1] - xz[a][1])
            if acc + L >= dist:
                t = (dist - acc) / (L or 1)
                return xz[a][0] + (xz[b][0] - xz[a][0]) * t, xz[a][1] + (xz[b][1] - xz[a][1]) * t
            acc += L
        return xz[nl[-1]]

    def side(ref_heading, node, p):
        dx, dz = math.cos(ref_heading), math.sin(ref_heading)
        return (p[0] - xz[node][0]) * -dz + (p[1] - xz[node][1]) * dx

    wcache = {}

    def weave_shift(node, i, j):
        if node not in wcache:
            ins_, outs_ = list(in_[node]), list(out_[node])
            wid = lambda l: min(l, key=lambda a: (-edges[a]["lanes"], a))
            h_in, h_out = head(wid(ins_), True), head(wid(outs_), False)
            so = {o: side(h_in, node, point(o, 50.0, False)) for o in outs_}
            si = {x: side(h_out, node, point(x, 50.0, True)) for x in ins_}
            c1 = lambda a, b: (-1 if so[a] > so[b] else 1) if abs(so[a] - so[b]) >= 1.0 else (a > b) - (a < b)
            c2 = lambda a, b: (-1 if si[a] > si[b] else 1) if abs(si[a] - si[b]) >= 1.0 else (a > b) - (a < b)
            outs_.sort(key=functools.cmp_to_key(c1))
            ins_.sort(key=functools.cmp_to_key(c2))
            oo, oi, acc = {}, {}, 0
            for o in reversed(outs_):          # offsets from the left (median) side
                oo[o] = acc; acc += edges[o]["lanes"]
            acc = 0
            for x in reversed(ins_):
                oi[x] = acc; acc += edges[x]["lanes"]
            wcache[node] = (oo, oi)
        oo, oi = wcache[node]
        return edges[i]["lanes"] - edges[j]["lanes"] + oi[i] - oo[j]

    cache = {}

    def hop(i, j):
        k = (i, j)
        if k in cache:
            return cache[k]
        e, en = edges[i], edges[j]
        outs = [o for o in out_[e["nodes"][-1]] if o != e.get("twin", -1)]
        ins = [o for o in in_[en["nodes"][0]] if o != en.get("twin", -1)]
        simple = len(outs) <= 1 and len(ins) <= 1
        ce, cn = e["seg"]["attrs"]["class"], en["seg"]["attrs"]["class"]
        strict = simple or (ce in FW_CLASSES and cn in FW_CLASSES and
                            (len(outs) <= 1 or len(ins) <= 1 or abs(turn(i, j)) < 1.2))
        shift = 0
        if strict and not simple and e["opp"] > 0 and en["opp"] > 0:
            shift = e["lanes"] - en["lanes"]
        elif strict and not simple:
            div = mer = 0
            node = e["nodes"][-1]
            if len(outs) > 1:
                if order == "geometry":
                    h = head(i, True)
                    sd = {o: side(h, node, point(o, 50.0, False)) for o in outs}
                    cmp = lambda a, b: (-1 if sd[a] > sd[b] else 1) if abs(sd[a] - sd[b]) >= 1.0 else \
                        (-1 if turn(i, a) > turn(i, b) else 1 if turn(i, a) < turn(i, b) else 0)
                    outs.sort(key=functools.cmp_to_key(cmp))
                else:
                    outs.sort(key=lambda o: -turn(i, o))
                for o in outs:
                    if o == j: break
                    div += edges[o]["lanes"]
                div = max(0, min(div, e["lanes"] - en["lanes"]))
            if len(ins) > 1:
                if order == "geometry":
                    h = head(j, False)
                    sd = {o: side(h, node, point(o, 50.0, True)) for o in ins}
                    cmp = lambda a, b: (-1 if sd[a] > sd[b] else 1) if abs(sd[a] - sd[b]) >= 1.0 else \
                        (-1 if turn(a, j) > turn(b, j) else 1 if turn(a, j) < turn(b, j) else 0)
                    ins.sort(key=functools.cmp_to_key(cmp))
                else:
                    ins.sort(key=lambda o: -turn(o, j))
                for o in ins:
                    if o == i: break
                    mer += edges[o]["lanes"]
                mer = max(0, min(mer, en["lanes"] - e["lanes"]))
            shift = div - mer
        elif strict:
            shift = e["lanes"] - en["lanes"]
        cache[k] = (strict, shift)
        return cache[k]
    return hop, out_, in_


def impossible_weaves(edges, xz, order="geometry"):
    """in -> e -> out paths on the freeway system that need k lane changes inside e with
    e shorter than 120 m * k (audit a16 logic). Returns [(p, e, o, k, length)]."""
    hop, out_, in_ = compute_hops(edges, xz, order)
    bad = []
    for ei, e in enumerate(edges):
        if e["seg"]["attrs"]["class"] not in FW_CLASSES:
            continue
        L = e["seg"]["length"]
        for p in in_[e["nodes"][0]]:
            if p == e.get("twin", -1):
                continue
            sp, shp = hop(p, ei)
            if not sp:
                continue
            lo, hi = max(0, -shp), min(e["lanes"] - 1, edges[p]["lanes"] - 1 - shp)
            if lo > hi:
                continue
            for o in out_[e["nodes"][-1]]:
                if o == e.get("twin", -1):
                    continue
                so, sho = hop(ei, o)
                if not so:
                    continue
                lo2, hi2 = max(0, sho), min(e["lanes"] - 1, edges[o]["lanes"] - 1 + sho)
                if lo2 > hi2: k = 99
                elif hi < lo2: k = lo2 - hi
                elif hi2 < lo: k = lo - hi2
                else: k = 0
                if k > 0 and L < 120 * k:
                    bad.append((p, ei, o, k, L))
    # never ban the only way on: if every continuation of p through e (and p has no other
    # out-edge), or every way into o through e (and e is o's only feeder) would be banned,
    # keep the easiest (smallest k) of them
    banned = {(p, e, o): (k, L) for (p, e, o, k, L) in bad}
    kept = set()
    by_p, by_o = defaultdict(list), defaultdict(list)
    for (p, e, o) in banned:
        by_p[(p, e)].append(o)
        by_o[(e, o)].append(p)
    for (p, e), os_ in by_p.items():
        allo = [o for o in out_[edges[e]["nodes"][-1]] if o != edges[e].get("twin", -1)]
        othr = [x for x in out_[edges[p]["nodes"][-1]] if x != e and x != edges[p].get("twin", -1)]
        if not othr and len(os_) == len(allo):
            kept.add(min(((p, e, o) for o in os_), key=lambda t: banned[t][0]))
    for (e, o), ps in by_o.items():
        alli = [x for x in in_[edges[e]["nodes"][0]] if x != edges[e].get("twin", -1)]
        feeders = [x for x in in_[edges[o]["nodes"][0]] if x != e and x != edges[o].get("twin", -1)]
        if not feeders and len(ps) == len(alli):
            kept.add(min(((p, e, o) for p in ps), key=lambda t: banned[t][0]))
    impossible_weaves.kept = sorted(kept)
    return [(p, e, o, k, L) for (p, e, o), (k, L) in banned.items() if (p, e, o) not in kept]


def write_tier(out_dir, tier, classes, xz, node_y, segs, link_index=None):
    tier_segs = [s for s in segs if s["attrs"]["class"] in classes]
    edges = directed_edges(tier_segs)
    # weaves are evaluated on the merged graph (main ids, then local ids), like src/geom and the sim
    local_edges = directed_edges([s for s in segs if s["attrs"]["class"] in LOCAL_CLASSES]) if tier == "main" else []
    for le in local_edges:
        if le["twin"] >= 0:
            le["twin"] += len(edges)
    weave_edges = lambda: edges + local_edges
    banned = impossible_weaves(weave_edges(), xz, "geometry") if tier == "main" else []
    if tier == "main":
        # impossible weaves that are the only way through: give the short via edge the lane
        # count of its neighbours (a gore piece), so no lane change is needed inside it
        fixed_vias = 0
        for _ in range(4):
            kept = impossible_weaves.kept
            if not kept:
                break
            for (p, ei, o) in kept:
                e = edges[ei]
                if e["seg"]["length"] >= 150:
                    continue
                v = max(edges[p]["lanes"], edges[o]["lanes"])
                sg = e["seg"]
                if e["rev"]:
                    sg["back"] = v
                else:
                    sg["fwd"] = v
                sg["flags"] = sg.get("flags", 0) | FLAG_LANES_SMOOTHED
                fixed_vias += 1
            edges = directed_edges(tier_segs)
            banned = impossible_weaves(weave_edges(), xz, "geometry")
        print(f"   gore via edges re-laned so the only path through needs no lane change: {fixed_vias}", file=sys.stderr)
    if tier == "main":
        legacy = impossible_weaves(weave_edges(), xz, "turn")
        bset = {(p, e, o) for (p, e, o, k, L) in banned}
        kept = impossible_weaves.kept if hasattr(impossible_weaves, "kept") else []
        banned = impossible_weaves(weave_edges(), xz, "geometry")
        kept = impossible_weaves.kept
        bset = {(p, e, o) for (p, e, o, k, L) in banned}
        print(f"   impossible weaves (k lane changes in < 120 m x k), geometry order: {len(banned)} emitted "
              f"as bannedHops, {len(kept)} kept open because they are the only way through "
              f"(e.g. toll plazas): {kept[:6]}", file=sys.stderr)
        print(f"   with the sim's legacy turn-angle order: {len(legacy)} paths, not banned: "
              f"{sum(1 for (p, e, o, k, L) in legacy if (p, e, o) not in bset)}", file=sys.stderr)
    node_ids = sorted({n for s in tier_segs for n in s["nodes"]})
    index = {nid: i for i, nid in enumerate(node_ids)}
    N, E = len(node_ids), len(edges)

    # ---------------- JSON (readable copy)
    nodes_out = [{"id": nid, "x": round(xz[nid][0], 2), "y": round(node_y[nid], 2),
                  "z": round(xz[nid][1], 2)} for nid in node_ids]
    edges_out = []
    for i, e in enumerate(edges):
        a = e["seg"]["attrs"]
        edges_out.append({
            "id": i, "from": index[e["nodes"][0]], "to": index[e["nodes"][-1]],
            "nodes": [index[n] for n in e["nodes"]],
            "class": a["class"], "name": a["name"], "ref": a["ref"],
            "oneway": a["oneway"], "twin": e["twin"],
            "lanes": e["lanes"], "oppLanes": e["opp"],
            "laneOffsets": lane_offsets(e["lanes"], e["opp"]),
            "width": round((e["lanes"] + e["opp"]) * LANE_W, 2),
            "lanesFallback": a["fallback"],
            "lanesRaw": e["raw"][0], "oppLanesRaw": e["raw"][1],
            "laneWidth": lane_width_of(a, e["lanes"], e["opp"]),
            "sidewalk": side_word(a["sidewalk"], e["rev"]), "parking": side_word(a["parking"], e["rev"]),
            "laneFlags": e["seg"].get("flags", 0) | (FLAG_LANES_CLAMPED if a["clamped"] else 0),
            "bridge": a["bridge"], "tunnel": a["tunnel"], "layer": a["layer"],
            "maxspeed": a["maxspeed"], "length": round(e["seg"]["length"], 1),
            "way": a["way"],
        })
    xs = [n["x"] for n in nodes_out] or [0]
    zs = [n["z"] for n in nodes_out] or [0]
    bbox_m = {"minX": min(xs), "maxX": max(xs), "minZ": min(zs), "maxZ": max(zs)}
    meta_common = {
        "format": "bay-roads/2", "tier": tier, "classes": classes,
        "classIds": {c: i for i, c in enumerate(ALL_CLASSES)},
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "OpenStreetMap via Overpass API (ODbL)",
        "projection": {
            "type": "equirectangular", "lat0": LAT0, "lon0": LON0, "earthRadius": R_EARTH,
            "metersPerDegLon": round(KX, 4), "metersPerDegLat": round(KZ, 4),
            "axes": "x = east (m), y = up (m, elevation baked in), z = south (m), north = -z",
            "rightOfTravel": "for direction (dx,dz) the right-hand perpendicular is (-dz, dx)",
        },
        "laneWidth": LANE_W,
        "laneDefaults": {"oneway": {c: ONEWAY_DEFAULT[c] for c in classes},
                         "twowayPerDirection": {c: TWOWAY_DEFAULT_PER_DIR[c] for c in classes}},
        "defaultSpeedKph": {c: DEFAULT_SPEED_KPH[c] for c in classes},
        "elevation": {"model": "structures/2", "deckPerLayer": LAYER_CLEAR, "waterDeckMin": WATER_DECK,
                      "crossingClearance": CROSS_CLEAR, "minClearance": MIN_CLEAR, "maxGrade": GRADE},
        "laneWidthByClass": {c: LANE_WIDTH_CLASS[c] for c in classes},
        "laneModel": {"throughLanes": True, "classMaxPerDirection": {c: CLASS_MAX_PER_DIR[c] for c in classes},
                      "minRunMeters": MIN_RUN, "accelLaneMeters": ACCEL_M, "taperMeters": TAPER_M},
        "bbox": bbox_m, "nodeCount": N, "edgeCount": E,
    }
    jpath = os.path.join(out_dir, "roads.json" if tier == "main" else "roads-local.json")
    with open(jpath, "w") as f:
        doc = {"meta": meta_common, "nodes": nodes_out, "edges": edges_out}
        if tier == "main":
            doc["bannedHops"] = [[p, ei, o] for (p, ei, o, k, L) in banned]
        json.dump(doc, f, separators=(",", ":"))
    print(f"== wrote {jpath}: {N} nodes, {E} directed edges, {os.path.getsize(jpath) / 1e6:.1f} MB",
          file=sys.stderr)

    # ---------------- binary pack
    xyz = array("f")
    for nid in node_ids:
        xyz.extend((xz[nid][0], node_y[nid], xz[nid][1]))
    etab = array("I")
    poly = array("I")
    elen = array("f")
    out_adj = defaultdict(list)
    for i, e in enumerate(edges):
        a = e["seg"]["attrs"]
        first = len(poly)
        poly.extend(index[n] for n in e["nodes"])
        flags = 0
        if a["oneway"]: flags |= FLAG_ONEWAY
        if a["bridge"]: flags |= FLAG_BRIDGE
        if a["tunnel"]: flags |= FLAG_TUNNEL
        if a["fallback"]: flags |= FLAG_LANES_FALLBACK
        if a["class"].endswith("_link"): flags |= FLAG_LINK
        flags |= e["seg"].get("flags", 0) & (FLAG_LANES_CLAMPED | FLAG_LANES_SMOOTHED)
        if not e["rev"]:   # aux / drop flags describe the travel direction of one-way segments
            flags |= e["seg"].get("flags", 0) & (FLAG_AUX_LANE | FLAG_LANE_DROP)
        if a["maxspeed"] is None:
            flags |= FLAG_SPEED_DEFAULT
            kph = DEFAULT_SPEED_KPH[a["class"]]
        else:
            kph = a["maxspeed"]
        speed_cmps = int(round(kph / 3.6 * 100))
        etab.extend((first, len(e["nodes"]), e["lanes"], e["opp"], ALL_CLASSES.index(a["class"]),
                     speed_cmps, flags, e["twin"] if e["twin"] >= 0 else NO_TWIN))
        elen.append(e["seg"]["length"])
        out_adj[index[e["nodes"][0]]].append(i)
    csr_off = array("I", [0] * (N + 1))
    csr_edges = array("I")
    for n in range(N):
        csr_off[n] = len(csr_edges)
        csr_edges.extend(out_adj.get(n, ()))
    csr_off[N] = len(csr_edges)

    lwidth = array("f", (lane_width_of(e["seg"]["attrs"], e["lanes"], e["opp"]) for e in edges))
    sections = [("nodeXYZ", xyz, "Float32", 3), ("edges", etab, "Uint32", EDGE_STRIDE),
                ("poly", poly, "Uint32", 1), ("edgeLength", elen, "Float32", 1),
                ("csrOffsets", csr_off, "Uint32", 1), ("csrEdges", csr_edges, "Uint32", 1),
                ("edgeLaneWidth", lwidth, "Float32", 1)]
    if tier == "main":
        banned_arr = array("I")
        for (p, ei, o, k, L) in banned:
            banned_arr.extend((p, ei, o))
        sections.append(("bannedHops", banned_arr, "Uint32", 3))
    if link_index is not None:
        links = array("I")
        for nid in node_ids:
            if nid in link_index:
                links.extend((index[nid], link_index[nid]))
        sections.append(("linkNodes", links, "Uint32", 2))

    bpath = os.path.join(out_dir, "roads.bin" if tier == "main" else "roads-local.bin")
    mpath = os.path.join(out_dir, "roads.meta.json" if tier == "main" else "roads-local.meta.json")
    sec_meta = {}
    off = 0
    with open(bpath, "wb") as f:
        for name, arr, typ, stride in sections:
            if sys.byteorder != "little":
                arr = array(arr.typecode, arr); arr.byteswap()
            b = arr.tobytes()
            sec_meta[name] = {"byteOffset": off, "byteLength": len(b), "type": typ,
                              "length": len(arr), "stride": stride, "count": len(arr) // stride}
            f.write(b)
            off += len(b)
            pad = (-len(b)) % 4
            if pad:
                f.write(b"\0" * pad); off += pad
    meta_bin = dict(meta_common)
    meta_bin.update({
        "format": "bay-roads-bin/1", "file": os.path.basename(bpath), "byteLength": off,
        "endianness": "little",
        "sections": sec_meta,
        "edgeStride": EDGE_STRIDE, "edgeFields": EDGE_FIELDS,
        "edgeFieldDoc": {
            "firstNode": "offset into poly[] of this edge's first node index (travel order)",
            "nodeCount": "number of node indices in poly[] for this edge (>=2); from=poly[firstNode], to=poly[firstNode+nodeCount-1]",
            "lanesFwd": "lanes in this edge's travel direction",
            "lanesBack": "lanes in the opposite direction (0 if oneway)",
            "classId": "index into meta.classIds / classesAll",
            "speedCmps": "speed limit in cm/s (divide by 100 for m/s); class default if FLAG_SPEED_DEFAULT",
            "flags": "bit set, see flagBits",
            "twin": "edge id of the opposite direction of the same road, or 0xFFFFFFFF",
        },
        "flagBits": {"ONEWAY": FLAG_ONEWAY, "BRIDGE": FLAG_BRIDGE, "TUNNEL": FLAG_TUNNEL,
                     "LANES_FALLBACK": FLAG_LANES_FALLBACK, "SPEED_DEFAULT": FLAG_SPEED_DEFAULT,
                     "LINK": FLAG_LINK, "LANES_CLAMPED": FLAG_LANES_CLAMPED,
                     "LANES_SMOOTHED": FLAG_LANES_SMOOTHED, "AUX_LANE": FLAG_AUX_LANE,
                     "LANE_DROP": FLAG_LANE_DROP},
        "classesAll": ALL_CLASSES,
        "noTwin": NO_TWIN,
        "laneOffsetFormula": "offset_k = ((lanesBack - lanesFwd)/2 + k + 0.5) * laneWidth, metres to the RIGHT of travel, k=0 is the leftmost lane",
        "csrDoc": "outgoing edges of node n are csrEdges[csrOffsets[n] .. csrOffsets[n+1]); each is an edge id whose from-node is n",
        "linkNodesDoc": "pairs (thisTierNodeIndex, mainTierNodeIndex) for nodes shared with roads.bin" if link_index is not None else None,
        "json": os.path.basename(jpath),
    })
    with open(mpath, "w") as f:
        json.dump(meta_bin, f, indent=1)
    print(f"== wrote {bpath}: {off / 1e6:.2f} MB  (+ {mpath})", file=sys.stderr)
    for name, arr, typ, stride in sections:
        print(f"     {name:12} {typ:8} x{len(arr):9d}  {len(arr) * 4 / 1e6:6.2f} MB", file=sys.stderr)
    return tier_segs, edges, node_ids, index


# ----------------------------------------------------------------------------- stats
def components(segs, classes=None):
    adj = defaultdict(list)
    for s in segs:
        if classes and s["attrs"]["class"] not in classes:
            continue
        a, b = s["nodes"][0], s["nodes"][-1]
        adj[a].append(b); adj[b].append(a)
    seen, comps = set(), []
    for s0 in adj:
        if s0 in seen:
            continue
        q = deque([s0]); seen.add(s0); members = 0
        while q:
            u = q.popleft(); members += 1
            for v in adj[u]:
                if v not in seen:
                    seen.add(v); q.append(v)
        comps.append(members)
    comps.sort(reverse=True)
    return comps


def comp_lookup(segs, classes):
    adj = defaultdict(list)
    for s in segs:
        if s["attrs"]["class"] not in classes:
            continue
        a, b = s["nodes"][0], s["nodes"][-1]
        adj[a].append(b); adj[b].append(a)
    comp = {}
    cid = 0
    for s0 in adj:
        if s0 in comp:
            continue
        q = deque([s0]); comp[s0] = cid
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in comp:
                    comp[v] = cid; q.append(v)
        cid += 1
    return comp


def strong_components(segs, classes=None):
    """Sizes of strongly connected components of the directed junction graph."""
    adj, radj = defaultdict(list), defaultdict(list)
    nodes = set()
    for s in segs:
        if classes and s["attrs"]["class"] not in classes:
            continue
        a, b = s["nodes"][0], s["nodes"][-1]
        nodes.add(a); nodes.add(b)
        adj[a].append(b); radj[b].append(a)
        if not s["attrs"]["oneway"]:
            adj[b].append(a); radj[a].append(b)
    order, seen = [], set()
    for s0 in nodes:                      # iterative DFS finish order
        if s0 in seen:
            continue
        seen.add(s0)
        stack = [(s0, iter(adj[s0]))]
        while stack:
            u, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                stack.pop(); order.append(u)
            elif nxt not in seen:
                seen.add(nxt); stack.append((nxt, iter(adj[nxt])))
    comp, sizes = {}, []
    for s0 in reversed(order):
        if s0 in comp:
            continue
        cid = len(sizes); comp[s0] = cid; q = [s0]; n = 0
        while q:
            u = q.pop(); n += 1
            for v in radj[u]:
                if v not in comp:
                    comp[v] = cid; q.append(v)
        sizes.append(n)
    sizes.sort(reverse=True)
    return sizes


def main(argv):
    out_dir = "data"
    bbox = DEFAULT_BBOX
    excludes = list(DEFAULT_EXCLUDE)
    simplify = True
    water = os.path.join("data", "water.json")
    nofail = False
    inputs = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--out-dir":
            out_dir = argv[i + 1]; i += 2
        elif a == "--bbox":
            bbox = tuple(float(v) for v in argv[i + 1].split(",")); i += 2
        elif a == "--exclude":
            excludes.append(tuple(float(v) for v in argv[i + 1].split(","))); i += 2
        elif a == "--no-exclude":
            excludes = []; i += 1
        elif a == "--no-simplify":
            simplify = False; i += 1
        elif a == "--water":
            water = argv[i + 1]; i += 2
        elif a == "--no-fail":
            nofail = True; i += 1
        else:
            inputs.append(a); i += 1
    if not inputs:
        print(__doc__); return 2
    os.makedirs(out_dir, exist_ok=True)

    SYN[0] = 0
    osm_nodes, xz, node_y, segs, stats_ways, checks = build(inputs, bbox, excludes, simplify, water)

    main_segs, main_edges, main_node_ids, main_index = write_tier(out_dir, "main", MAIN_CLASSES, xz, node_y, segs)
    has_local = any(s["attrs"]["class"] in LOCAL_CLASSES for s in segs)
    if has_local:
        local_segs, local_edges, local_node_ids, _ = write_tier(
            out_dir, "local", LOCAL_CLASSES, xz, node_y, segs, link_index=main_index)

    # ---- per-class stats
    st = defaultdict(lambda: {"segs": 0, "dir": 0, "km": 0.0, "bridge_km": 0.0, "tunnel_km": 0.0})
    bridges_named = defaultdict(float)
    for s in segs:
        a = s["attrs"]; c = a["class"]; km = s["length"] / 1000
        st[c]["segs"] += 1; st[c]["dir"] += 1 if a["oneway"] else 2; st[c]["km"] += km
        if a["bridge"]:
            st[c]["bridge_km"] += km
            if a["bridge_name"]:
                bridges_named[a["bridge_name"]] += km
        if a["tunnel"]:
            st[c]["tunnel_km"] += km
    print("\n== per-class stats (centreline km, once per road segment)", file=sys.stderr)
    print(f"   {'class':16}{'osm ways':>9}{'segments':>10}{'dir.edges':>11}{'km':>9}{'bridge km':>11}"
          f"{'tunnel km':>11}{'lanes-fallback':>16}", file=sys.stderr)
    tot = 0.0
    for c in ALL_CLASSES:
        if c not in st:
            continue
        s = st[c]; w = stats_ways[c]; tot += s["km"]
        print(f"   {c:16}{w['ways']:9d}{s['segs']:10d}{s['dir']:11d}{s['km']:9.1f}{s['bridge_km']:11.1f}"
              f"{s['tunnel_km']:11.1f}{w['fallback']:8d}/{w['ways']:<7d}", file=sys.stderr)
    print(f"   total centreline km: {tot:.1f}", file=sys.stderr)

    print("\n== named bridges (km of bridge=yes centreline, by name or bridge:name)", file=sys.stderr)
    for w in ["Bay Bridge", "San Mateo", "Dumbarton", "Richmond", "Golden Gate"]:
        hits = sorted(((k, v) for k, v in bridges_named.items() if w.lower() in k.lower()), key=lambda kv: -kv[1])
        if hits:
            print(f"   [ok] {w:12} -> {hits[0][0]!r}: {hits[0][1]:.2f} km", file=sys.stderr)
        else:
            print(f"   [--] {w:12} -> not found", file=sys.stderr)

    print("\n== connectivity", file=sys.stderr)
    fast = list(FAST_CLASSES)
    comps = components(segs, fast)
    print(f"   motorway+trunk subgraph: {len(comps)} weakly connected components; largest {comps[0] if comps else 0} "
          f"nodes, next {comps[1:6]}", file=sys.stderr)
    comps_main = components(main_segs)
    print(f"   main tier (all classes): {len(comps_main)} components; largest {comps_main[0] if comps_main else 0} "
          f"of {sum(comps_main)} junction nodes ({100.0 * comps_main[0] / max(1, sum(comps_main)):.2f}%)",
          file=sys.stderr)
    probes = {"SF (US101/I-80)": (37.772, -122.405), "Oakland (I-880/I-980)": (37.800, -122.280),
              "San Jose (I-280/US101)": (37.340, -121.860), "Fremont (I-880/CA84)": (37.52, -121.98),
              "San Mateo (US101)": (37.565, -122.300), "Marin side of GG (US101)": (37.835, -122.48)}
    cl = comp_lookup(segs, fast)
    if cl:
        for label, (la, lo) in probes.items():
            tx, tz = project(la, lo)
            best = min(cl, key=lambda n: (xz[n][0] - tx) ** 2 + (xz[n][1] - tz) ** 2)
            d = math.hypot(xz[best][0] - tx, xz[best][1] - tz)
            print(f"   probe {label:26} -> node {best} ({d:.0f} m away) in component #{cl[best]}", file=sys.stderr)

    sc = strong_components(main_segs)
    print(f"   main tier strongly connected: largest SCC {sc[0] if sc else 0} of {sum(sc)} junction nodes "
          f"({100.0 * (sc[0] if sc else 0) / max(1, sum(sc)):.2f}%), {len(sc)} SCCs", file=sys.stderr)

    print("\n== lane counts (main tier, directed edges / km)", file=sys.stderr)
    fb_e = fb_km = all_e = all_km = 0.0
    for s in main_segs:
        a = s["attrs"]; nd = 1 if a["oneway"] else 2; km = s["length"] / 1000 * nd
        all_e += nd; all_km += km
        if a["fallback"]:
            fb_e += nd; fb_km += km
    print(f"   lanes from class fallback: {100 * fb_e / max(1, all_e):.1f}% of directed edges, "
          f"{100 * fb_km / max(1e-9, all_km):.1f}% of directed km", file=sys.stderr)
    for c in ("motorway", "trunk", "primary", "secondary"):
        ce = [s for s in main_segs if s["attrs"]["class"] == c]
        km = sum(s["length"] for s in ce) or 1e-9
        fkm = sum(s["length"] for s in ce if s["attrs"]["fallback"])
        print(f"   {c:10} fallback {100 * fkm / km:5.1f}% of km", file=sys.stderr)

    if has_local:
        shared = set(main_node_ids) & set(local_node_ids)
        comps_all = components(segs)
        print(f"\n== local tier: {len(local_node_ids)} nodes, {len(local_edges)} directed edges; "
              f"{len(shared)} nodes shared with main tier", file=sys.stderr)
        print(f"   merged main+local graph: {len(comps_all)} components; largest {comps_all[0]} of "
              f"{sum(comps_all)} junction nodes ({100.0 * comps_all[0] / max(1, sum(comps_all)):.2f}%)",
              file=sys.stderr)
        sca = strong_components(segs)
        print(f"   merged main+local strongly connected: largest SCC {sca[0]} of {sum(sca)} junction nodes "
              f"({100.0 * sca[0] / max(1, sum(sca)):.2f}%)", file=sys.stderr)
    failed = {k: v for k, v in checks.items() if v}
    if failed:
        print("\n== BUILD CHECKS FAILED", file=sys.stderr)
        for k, v in failed.items():
            print(f"   {k}: {len(v)}  e.g. {v[:5]}", file=sys.stderr)
        return 0 if nofail else 1
    print("\n== build checks passed (no folds, no over-long edges, grades <= 8%, "
          "crossing clearance >= 4.5 m, no unexplained lane imbalance)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
