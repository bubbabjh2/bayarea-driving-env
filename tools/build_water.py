#!/usr/bin/env python3
"""
build_water.py - OSM natural=coastline ways (Overpass JSON) -> filled water
polygons for the viewer, in the same projected metres as roads.json.

Usage:
    python3 -I tools/build_water.py [--bbox S,W,N,E] coast.json data/water.json

The bbox should be the one the coastline was *queried* with; make it larger
than the road bbox so the water does not end in a hard edge near the roads.

Method: chain coastline ways end-to-end, clip them to the bbox, then close the
open chains along the bbox perimeter walking clockwise (OSM coastline has land
on the LEFT and water on the RIGHT, so walking the rectangle clockwise keeps
water on the right too).  Closed rings fully inside the bbox are islands
(holes) when counter-clockwise, or lakes (extra water) when clockwise.

Output: {"meta":{...}, "polygons":[{"outer":[[x,z],...],"holes":[[[x,z],...],...]}]}
"""
import json
import math
import sys
import os
from collections import defaultdict

LAT0, LON0 = 37.55, -122.15
R_EARTH = 6371008.8
KX = math.cos(math.radians(LAT0)) * R_EARTH * math.pi / 180.0
KZ = R_EARTH * math.pi / 180.0
BBOX = (37.20, -122.62, 38.02, -121.72)  # S, W, N, E (= tools/queries/coast.ql)
SIMPLIFY_M = 12.0


def project(lat, lon):
    return (lon - LON0) * KX, -(lat - LAT0) * KZ


def chain_ways(ways):
    """ways: list of node-id lists. Returns list of node-id chains."""
    by_first = defaultdict(list)
    for i, w in enumerate(ways):
        by_first[w[0]].append(i)
    last_nodes = {w[-1] for w in ways}
    used = [False] * len(ways)
    chains = []

    def follow(i):
        chain = list(ways[i])
        used[i] = True
        while True:
            nxt = [j for j in by_first.get(chain[-1], []) if not used[j]]
            if not nxt:
                break
            j = nxt[0]
            used[j] = True
            chain.extend(ways[j][1:])
            if chain[-1] == chain[0]:
                break
        return chain

    # open chains first: starts that are not the end of another way
    for i, w in enumerate(ways):
        if not used[i] and w[0] not in last_nodes:
            chains.append(follow(i))
    # everything left is part of a ring
    for i in range(len(ways)):
        if not used[i]:
            chains.append(follow(i))
    return chains


def clip_polyline(pts, minx, minz, maxx, maxz):
    """Clip polyline to rect; returns list of pieces (each list of (x,z))."""
    def inside(p):
        return minx <= p[0] <= maxx and minz <= p[1] <= maxz

    def clip_seg(a, b):
        # Liang-Barsky: returns (t0, t1) or None
        dx, dz = b[0] - a[0], b[1] - a[1]
        t0, t1 = 0.0, 1.0
        for p, q in ((-dx, a[0] - minx), (dx, maxx - a[0]), (-dz, a[1] - minz), (dz, maxz - a[1])):
            if p == 0:
                if q < 0:
                    return None
            else:
                t = q / p
                if p < 0:
                    if t > t1:
                        return None
                    if t > t0:
                        t0 = t
                else:
                    if t < t0:
                        return None
                    if t < t1:
                        t1 = t
        return t0, t1

    pieces = []
    cur = []
    for a, b in zip(pts, pts[1:]):
        r = clip_seg(a, b)
        if r is None:
            if cur:
                pieces.append(cur); cur = []
            continue
        t0, t1 = r
        pa = (a[0] + (b[0] - a[0]) * t0, a[1] + (b[1] - a[1]) * t0)
        pb = (a[0] + (b[0] - a[0]) * t1, a[1] + (b[1] - a[1]) * t1)
        if t0 > 0 or not cur:
            if cur:
                pieces.append(cur)
            cur = [pa]
        cur.append(pb)
        if t1 < 1:
            pieces.append(cur); cur = []
    if cur:
        pieces.append(cur)
    return [p for p in pieces if len(p) >= 2]


def snap(p, minx, minz, maxx, maxz, eps=0.5):
    x, z = p
    if abs(x - minx) < eps: x = minx
    if abs(x - maxx) < eps: x = maxx
    if abs(z - minz) < eps: z = minz
    if abs(z - maxz) < eps: z = maxz
    return (x, z)


def on_boundary(p, minx, minz, maxx, maxz):
    return p[0] in (minx, maxx) or p[1] in (minz, maxz)


def perim_param(p, minx, minz, maxx, maxz):
    """Clockwise (on a north-up map: x right, z down) distance along the rect."""
    W, H = maxx - minx, maxz - minz
    x, z = p
    if z == minz:           # top edge, left -> right
        return x - minx
    if x == maxx:           # right edge, top -> bottom
        return W + (z - minz)
    if z == maxz:           # bottom edge, right -> left
        return W + H + (maxx - x)
    return 2 * W + H + (maxz - z)  # left edge, bottom -> top


def corners_between(p0, p1, minx, minz, maxx, maxz):
    """Corner points passed when walking clockwise from perimeter param p0 to p1."""
    W, H = maxx - minx, maxz - minz
    P = 2 * (W + H)
    corners = [(W, (maxx, minz)), (W + H, (maxx, maxz)), (2 * W + H, (minx, maxz)), (P, (minx, minz))]
    out = []
    d = (p1 - p0) % P
    if d == 0:
        d = P
    for cp, c in corners:
        off = (cp - p0) % P
        if 0 < off < d:
            out.append((off, c))
    out.sort()
    return [c for _, c in out]


def signed_area(ring):
    s = 0.0
    for (x0, z0), (x1, z1) in zip(ring, ring[1:] + ring[:1]):
        s += x0 * z1 - x1 * z0
    return s / 2.0   # positive = clockwise on a north-up (z down) map


def point_in_ring(pt, ring):
    x, z = pt
    inside = False
    n = len(ring)
    for i in range(n):
        x0, z0 = ring[i]
        x1, z1 = ring[(i + 1) % n]
        if (z0 > z) != (z1 > z):
            xi = x0 + (z - z0) * (x1 - x0) / (z1 - z0)
            if xi > x:
                inside = not inside
    return inside


def simplify(pts, tol):
    """Iterative Douglas-Peucker."""
    if len(pts) < 3:
        return pts
    keep = [False] * len(pts)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        ax, az = pts[a]; bx, bz = pts[b]
        dx, dz = bx - ax, bz - az
        L = dx * dx + dz * dz
        best, bi = 0.0, -1
        for i in range(a + 1, b):
            px, pz = pts[i]
            if L == 0:
                d = math.hypot(px - ax, pz - az)
            else:
                t = max(0.0, min(1.0, ((px - ax) * dx + (pz - az) * dz) / L))
                d = math.hypot(px - (ax + t * dx), pz - (az + t * dz))
            if d > best:
                best, bi = d, i
        if best > tol and bi > 0:
            keep[bi] = True
            stack.append((a, bi)); stack.append((bi, b))
    return [p for p, k in zip(pts, keep) if k]


def main(argv):
    global BBOX
    args = []
    i = 0
    while i < len(argv):
        if argv[i] == "--bbox":
            BBOX = tuple(float(v) for v in argv[i + 1].split(",")); i += 2
        else:
            args.append(argv[i]); i += 1
    if len(args) < 2:
        print(__doc__); return 2
    src, dst = args[0], args[1]
    with open(src, "rb") as f:
        raw = f.read()
    if not raw.lstrip().startswith(b"{"):
        print("input is not JSON", file=sys.stderr); return 1
    doc = json.loads(raw)
    nodes, ways = {}, []
    raw_ways = []
    for el in doc.get("elements", []):
        if el.get("type") == "node":
            nodes[el["id"]] = project(el["lat"], el["lon"])
        elif el.get("type") == "way" and (el.get("tags") or {}).get("natural") == "coastline":
            raw_ways.append(el["nodes"])
    for wn in raw_ways:  # ways precede nodes in Overpass output: resolve after
        ns = [n for n in wn if n in nodes]
        if len(ns) >= 2:
            ways.append(ns)
    print(f"coastline: {len(ways)} ways, {len(nodes)} nodes", file=sys.stderr)

    minx, maxz = project(BBOX[0], BBOX[1])   # south-west -> min x, max z
    maxx, minz = project(BBOX[2], BBOX[3])   # north-east -> max x, min z

    chains = chain_ways(ways)
    open_pieces, rings = [], []
    for ch in chains:
        pts = [nodes[n] for n in ch]
        closed = ch[0] == ch[-1]
        if closed and all(minx <= x <= maxx and minz <= z <= maxz for x, z in pts):
            rings.append(pts[:-1])
            continue
        for piece in clip_polyline(pts, minx, minz, maxx, maxz):
            piece = [snap(p, minx, minz, maxx, maxz) for p in piece]
            if closed and piece[0] == piece[-1] and len(piece) > 3:
                rings.append(piece[:-1])
            elif on_boundary(piece[0], minx, minz, maxx, maxz) and on_boundary(piece[-1], minx, minz, maxx, maxz):
                open_pieces.append(piece)
            else:
                # dangling coastline (data gap): cannot close, skip
                pass
    print(f"chains: {len(chains)} -> {len(open_pieces)} boundary-to-boundary pieces, "
          f"{len(rings)} closed rings", file=sys.stderr)

    # close open pieces along the perimeter (clockwise)
    P = 2 * ((maxx - minx) + (maxz - minz))
    starts = []
    for i, pc in enumerate(open_pieces):
        starts.append((perim_param(pc[0], minx, minz, maxx, maxz), i))
    starts.sort()
    used = [False] * len(open_pieces)
    water = []
    for i0 in range(len(open_pieces)):
        if used[i0]:
            continue
        ring = []
        i = i0
        guard = 0
        while True:
            used[i] = True
            ring.extend(open_pieces[i])
            pe = perim_param(open_pieces[i][-1], minx, minz, maxx, maxz)
            # next unused start clockwise from pe (cyclic); the origin piece is
            # allowed again to close the loop
            best, bj = None, None
            for ps, j in starts:
                if used[j] and j != i0:
                    continue
                off = (ps - pe) % P
                if best is None or off < best:
                    best, bj = off, j
            ring.extend(corners_between(pe, perim_param(open_pieces[bj][0], minx, minz, maxx, maxz),
                                        minx, minz, maxx, maxz))
            if bj == i0:
                break
            i = bj
            guard += 1
            if guard > 100000:
                print("!! runaway closure loop", file=sys.stderr); break
        water.append(ring)

    # closed rings: clockwise = water, counter-clockwise = island (hole)
    islands, lakes = [], []
    for r in rings:
        a = signed_area(r)
        if abs(a) < 50:  # < 50 m^2, noise
            continue
        (lakes if a > 0 else islands).append(r)
    print(f"water polygons from closure: {len(water)}; lakes: {len(lakes)}; islands: {len(islands)}",
          file=sys.stderr)

    polys = []
    for w in water + lakes:
        outer = simplify(w, SIMPLIFY_M)
        if len(outer) < 3:
            continue
        polys.append({"outer": outer, "holes": []})
    for isl in islands:
        isl_s = simplify(isl, SIMPLIFY_M)
        if len(isl_s) < 3:
            continue
        for poly in polys:
            if point_in_ring(isl_s[0], poly["outer"]):
                poly["holes"].append(isl_s)
                break

    # sanity probes
    probes = {"mid-bay (water)": (37.62, -122.25, True), "SF downtown (land)": (37.79, -122.41, False),
              "Oakland (land)": (37.80, -122.27, False), "San Jose (land)": (37.33, -121.89, False),
              "Pacific off Pacifica (water)": (37.62, -122.515, True), "Alameda (island, land)": (37.76, -122.25, False)}
    ok = True
    for label, (la, lo, want) in probes.items():
        p = project(la, lo)
        inw = False
        for poly in polys:
            if point_in_ring(p, poly["outer"]) and not any(point_in_ring(p, h) for h in poly["holes"]):
                inw = True; break
        flag = "ok" if inw == want else "MISMATCH"
        if inw != want:
            ok = False
        print(f"   probe {label:30} water={inw} [{flag}]", file=sys.stderr)

    total_v = sum(len(p["outer"]) + sum(len(h) for h in p["holes"]) for p in polys)
    out = {"meta": {"format": "bay-water/1", "projection": {"lat0": LAT0, "lon0": LON0},
                    "bbox": {"minX": minx, "maxX": maxx, "minZ": minz, "maxZ": maxz},
                    "probesOk": ok},
           "polygons": [{"outer": [[round(x, 1), round(z, 1)] for x, z in p["outer"]],
                         "holes": [[[round(x, 1), round(z, 1)] for x, z in h] for h in p["holes"]]}
                        for p in polys]}
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    with open(dst, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"wrote {dst}: {len(polys)} polygons, {total_v} vertices, "
          f"{os.path.getsize(dst) / 1e3:.0f} KB, probes {'OK' if ok else 'FAILED'}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
