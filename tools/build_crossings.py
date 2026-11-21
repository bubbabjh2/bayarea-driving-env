#!/usr/bin/env python3
"""
build_crossings.py - every grade-separated crossing (two roads cross in plan with no shared node)
that involves the freeway system (motorway / trunk / their links), with the vertical separation.

    python3 -I tools/build_crossings.py [data]

Writes data/crossings.json for the sim's contact decisions: two vehicles on a crossing pair can
only touch if the vertical separation at the crossing point is small.
{"meta": {...}, "crossings": [{"a": edgeId, "aTwin": id|-1, "sa": m, "b": edgeId, "bTwin": id|-1,
  "sb": m, "x", "z", "ya", "yb", "dy", "reason": null|"..."}]}
Edge ids are merged-graph ids (main e, local mainEdgeCount + e). sa / sb are metres along a / b
(the twins run the other way: s = length - sa). reason is set when dy < 4.5 m:
"merge/diverge overlap" (near-parallel roads joined within 300 m), "junction geometry" (crossing
roads joined within 300 m), "tunnel", or "at-grade (no OSM junction)".
"""
import heapq
import json
import math
import os
import sys
from collections import defaultdict

FW = {"motorway", "motorway_link", "trunk", "trunk_link"}


def main(argv):
    D = argv[0] if argv else "data"
    m = json.load(open(os.path.join(D, "roads.json")))
    l = json.load(open(os.path.join(D, "roads-local.json")))
    Em = len(m["edges"])
    segs = []          # (merged id, twin merged id, edge, nodes(list of dict))
    for base, doc in ((0, m), (Em, l)):
        N = doc["nodes"]
        for e in doc["edges"]:
            if e["twin"] >= 0 and e["twin"] < e["id"]:
                continue
            segs.append((base + e["id"], base + e["twin"] if e["twin"] >= 0 else -1, e, N))
    canon = lambda N, i: N[i]["id"]
    adj = defaultdict(list)
    for g, t, e, N in segs:
        nd = e["nodes"]
        for p, q in zip(nd, nd[1:]):
            L = math.hypot(N[q]["x"] - N[p]["x"], N[q]["z"] - N[p]["z"])
            adj[canon(N, p)].append((canon(N, q), L))
            adj[canon(N, q)].append((canon(N, p), L))

    def connected(a, b, limit=300.0):
        dist = {v: 0.0 for v in a}
        hp = [(0.0, v) for v in a]
        while hp:
            d, v = heapq.heappop(hp)
            if v in b:
                return True
            if d > dist.get(v, 1e18):
                continue
            for w, L in adj[v]:
                nd_ = d + L
                if nd_ <= limit and nd_ < dist.get(w, 1e18):
                    dist[w] = nd_
                    heapq.heappush(hp, (nd_, w))
        return False

    cell = 200.0
    grid = defaultdict(list)
    pieces = []
    for si, (g, t, e, N) in enumerate(segs):
        nd = e["nodes"]
        acc = 0.0
        for k in range(len(nd) - 1):
            A, B = N[nd[k]], N[nd[k + 1]]
            L = math.hypot(B["x"] - A["x"], B["z"] - A["z"])
            pieces.append((si, A, B, acc, L))
            acc += L
            if e["class"] in FW:
                pid = len(pieces) - 1
                for gx in range(int(min(A["x"], B["x"]) // cell), int(max(A["x"], B["x"]) // cell) + 1):
                    for gz in range(int(min(A["z"], B["z"]) // cell), int(max(A["z"], B["z"]) // cell) + 1):
                        grid[(gx, gz)].append(pid)
    out, seen = [], set()
    for qi, (sj, P, Q, accq, Lq) in enumerate(pieces):
        cand = set()
        for gx in range(int(min(P["x"], Q["x"]) // cell), int(max(P["x"], Q["x"]) // cell) + 1):
            for gz in range(int(min(P["z"], Q["z"]) // cell), int(max(P["z"], Q["z"]) // cell) + 1):
                cand.update(grid.get((gx, gz), ()))
        for pi in cand:
            si, A, B, acca, La = pieces[pi]
            if si == sj:
                continue
            key = (min(pi, qi), max(pi, qi))
            if key in seen:
                continue
            seen.add(key)
            if len({A["id"], B["id"], P["id"], Q["id"]}) < 4:
                continue
            d1x, d1z, d2x, d2z = B["x"] - A["x"], B["z"] - A["z"], Q["x"] - P["x"], Q["z"] - P["z"]
            den = d1x * d2z - d1z * d2x
            if abs(den) < 1e-9:
                continue
            u = ((P["x"] - A["x"]) * d2z - (P["z"] - A["z"]) * d2x) / den
            v = ((P["x"] - A["x"]) * d1z - (P["z"] - A["z"]) * d1x) / den
            if not (1e-6 < u < 1 - 1e-6 and 1e-6 < v < 1 - 1e-6):
                continue
            ga, ta, ea, Na = segs[si]
            gb, tb, eb, Nb = segs[sj]
            ya = A["y"] + (B["y"] - A["y"]) * u
            yb = P["y"] + (Q["y"] - P["y"]) * v
            dy = abs(ya - yb)
            reason = None
            if dy < 4.5:
                cos = abs(d1x * d2x + d1z * d2z) / max(1e-9, math.hypot(d1x, d1z) * math.hypot(d2x, d2z))
                if ea["tunnel"] or eb["tunnel"]:
                    reason = "tunnel"
                elif connected({A["id"], B["id"]}, {P["id"], Q["id"]}):
                    reason = "merge/diverge overlap" if cos > 0.9 else "junction geometry"
                else:
                    reason = "at-grade (no OSM junction)"
            sa = (acca + La * u) * ea["length"] / max(1e-9, sum(math.hypot(Na[q]["x"] - Na[p]["x"], Na[q]["z"] - Na[p]["z"]) for p, q in zip(ea["nodes"], ea["nodes"][1:])))
            sb = (accq + Lq * v) * eb["length"] / max(1e-9, sum(math.hypot(Nb[q]["x"] - Nb[p]["x"], Nb[q]["z"] - Nb[p]["z"]) for p, q in zip(eb["nodes"], eb["nodes"][1:])))
            out.append({"a": ga, "aTwin": ta, "sa": round(sa, 2), "aClass": ea["class"],
                        "b": gb, "bTwin": tb, "sb": round(sb, 2), "bClass": eb["class"],
                        "x": round(A["x"] + d1x * u, 1), "z": round(A["z"] + d1z * u, 1),
                        "ya": round(ya, 2), "yb": round(yb, 2), "dy": round(dy, 2), "reason": reason})
    # pavement overlaps without a crossing: a non-freeway road whose drawn pavement overlaps a
    # freeway's (lateral gap < sum of half widths) at |dy| < 4.5 m, not joined within 300 m
    hw = lambda e: (e["lanes"] + e["oppLanes"]) * e.get("laneWidth", 3.7) / 2
    ngrid = defaultdict(list)
    for pid, (si, A, B, acc, L) in enumerate(pieces):
        if segs[si][2]["class"] in FW:
            continue
        for gx in range(int(min(A["x"], B["x"]) // cell), int(max(A["x"], B["x"]) // cell) + 1):
            for gz in range(int(min(A["z"], B["z"]) // cell), int(max(A["z"], B["z"]) // cell) + 1):
                ngrid[(gx, gz)].append(pid)
    overlaps = {}
    for pid, (si, A, B, acc, L) in enumerate(pieces):
        ea = segs[si][2]
        if ea["class"] not in FW or L < 1:
            continue
        for k in range(max(1, int(L // 10))):
            t = (k + 0.5) / max(1, int(L // 10))
            x, z, y = A["x"] + (B["x"] - A["x"]) * t, A["z"] + (B["z"] - A["z"]) * t, A["y"] + (B["y"] - A["y"]) * t
            for qid in ngrid.get((int(x // cell), int(z // cell)), ()):
                sj, P, Q, accq, Lq = pieces[qid]
                eb = segs[sj][2]
                dx, dz = Q["x"] - P["x"], Q["z"] - P["z"]
                LL = dx * dx + dz * dz
                if LL < 1e-6:
                    continue
                v = max(0.0, min(1.0, ((x - P["x"]) * dx + (z - P["z"]) * dz) / LL))
                d = math.hypot(P["x"] + dx * v - x, P["z"] + dz * v - z)
                yb = P["y"] + (Q["y"] - P["y"]) * v
                if d < hw(ea) + hw(eb) - 0.5 and abs(y - yb) < 4.5:
                    key = (segs[si][0], segs[sj][0])
                    if key in overlaps:
                        continue
                    if connected({A["id"], B["id"]}, {P["id"], Q["id"]}):
                        continue
                    overlaps[key] = {"a": segs[si][0], "b": segs[sj][0], "aClass": ea["class"], "bClass": eb["class"],
                                     "x": round(x, 1), "z": round(z, 1), "gap": round(d, 2), "dy": round(abs(y - yb), 2)}
    low = [c for c in out if c["dy"] < 4.5]
    fwnf = [c for c in low if (c["aClass"] in FW) != (c["bClass"] in FW)]
    from collections import Counter
    meta = {"format": "bay-crossings/1", "count": len(out), "below4_5m": len(low),
            "below4_5m_freeway_x_nonfreeway": len(fwnf),
            "byReason": dict(Counter(c["reason"] for c in low)),
            "edgeIds": "merged graph: main e, local mainEdgeCount + e", "mainEdgeCount": Em,
            "note": "all crossings in plan with no shared node that involve the freeway system; dy = vertical separation at the crossing point"}
    meta["pavementOverlapsFreewayXOther"] = len(overlaps)
    json.dump({"meta": meta, "crossings": out, "overlaps": list(overlaps.values())}, open(os.path.join(D, "crossings.json"), "w"), separators=(",", ":"))
    print(json.dumps(meta, indent=1))
    for c in fwnf:
        print("  ", c["aClass"], c["bClass"], c["x"], c["z"], c["dy"], c["reason"])
    from collections import Counter as C2
    print("pavement overlaps by class pair:", C2((o["aClass"], o["bClass"]) for o in overlaps.values()).most_common(8))
    for o in list(overlaps.values())[:8]:
        print("  ", o)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
