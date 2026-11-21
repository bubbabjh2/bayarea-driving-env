#!/usr/bin/env python3
"""
validate_pack.py - sanity-check a roads binary pack against its JSON copy.

    python3 -I tools/validate_pack.py data/roads.meta.json [data/roads.meta.json for main tier when checking local]

Checks: section alignment, node/edge counts match JSON, every polyline index
in range, from/to consistency, twin symmetry (twin's twin is self, reversed
polyline, swapped lane counts), CSR adjacency == edges grouped by from-node,
lengths within 1% of JSON, linkNodes map to the main tier at the same place.
"""
import json
import os
import sys
from array import array


def load_sections(meta_path):
    meta = json.load(open(meta_path))
    bin_path = os.path.join(os.path.dirname(meta_path), meta["file"])
    blob = open(bin_path, "rb").read()
    assert len(blob) == meta["byteLength"], "byteLength mismatch"
    out = {}
    for name, s in meta["sections"].items():
        assert s["byteOffset"] % 4 == 0, f"{name} not 4-byte aligned"
        code = {"Float32": "f", "Uint32": "I"}[s["type"]]
        a = array(code)
        a.frombytes(blob[s["byteOffset"]:s["byteOffset"] + s["byteLength"]])
        if sys.byteorder != "little":
            a.byteswap()
        assert len(a) == s["length"], f"{name} length mismatch"
        out[name] = a
    return meta, out


def main(argv):
    if not argv:
        print(__doc__); return 2
    meta, S = load_sections(argv[0])
    json_path = os.path.join(os.path.dirname(argv[0]), meta["json"])
    doc = json.load(open(json_path))
    N, E = meta["nodeCount"], meta["edgeCount"]
    xyz, edges, poly = S["nodeXYZ"], S["edges"], S["poly"]
    stride = meta["edgeStride"]
    F = {n: i for i, n in enumerate(meta["edgeFields"])}
    NO_TWIN = meta["noTwin"]
    errors = 0

    def err(msg):
        nonlocal errors
        errors += 1
        if errors <= 20:
            print("   !!", msg)

    assert len(xyz) == 3 * N and len(doc["nodes"]) == N, "node count mismatch"
    assert len(edges) == stride * E and len(doc["edges"]) == E, "edge count mismatch"
    for i in range(N):
        n = doc["nodes"][i]
        if abs(xyz[3 * i] - n["x"]) > 0.1 or abs(xyz[3 * i + 2] - n["z"]) > 0.1 or abs(xyz[3 * i + 1] - n["y"]) > 0.1:
            err(f"node {i} xyz differs from JSON")
    out_adj = [[] for _ in range(N)]
    for e in range(E):
        o = e * stride
        first, cnt = edges[o + F["firstNode"]], edges[o + F["nodeCount"]]
        je = doc["edges"][e]
        if cnt < 2 or first + cnt > len(poly):
            err(f"edge {e}: bad polyline range"); continue
        pl = poly[first:first + cnt]
        if list(pl) != je["nodes"]:
            err(f"edge {e}: polyline differs from JSON")
        if pl[0] != je["from"] or pl[-1] != je["to"]:
            err(f"edge {e}: from/to mismatch")
        if any(n >= N for n in pl):
            err(f"edge {e}: node index out of range")
        if edges[o + F["lanesFwd"]] != je["lanes"] or edges[o + F["lanesBack"]] != je["oppLanes"]:
            err(f"edge {e}: lane counts differ from JSON")
        if meta["classesAll"][edges[o + F["classId"]]] != je["class"]:
            err(f"edge {e}: class differs")
        if edges[o + F["lanesFwd"]] < 1:
            err(f"edge {e}: zero forward lanes")
        t = edges[o + F["twin"]]
        if t == NO_TWIN:
            if not (edges[o + F["flags"]] & 1):
                err(f"edge {e}: no twin but not flagged oneway")
            if je["twin"] != -1:
                err(f"edge {e}: JSON has twin, binary does not")
        else:
            if t >= E:
                err(f"edge {e}: twin out of range"); continue
            ot = t * stride
            if edges[ot + F["twin"]] != e:
                err(f"edge {e}: twin {t} does not point back")
            tf, tc = edges[ot + F["firstNode"]], edges[ot + F["nodeCount"]]
            if list(poly[tf:tf + tc]) != list(pl)[::-1]:
                err(f"edge {e}: twin polyline is not the reverse")
            if edges[ot + F["lanesFwd"]] != edges[o + F["lanesBack"]] or edges[ot + F["lanesBack"]] != edges[o + F["lanesFwd"]]:
                err(f"edge {e}: twin lane counts not swapped")
        L = S["edgeLength"][e]
        if abs(L - je["length"]) > max(0.2, 0.01 * je["length"]):
            err(f"edge {e}: length {L} vs JSON {je['length']}")
        if edges[o + F["speedCmps"]] <= 0:
            err(f"edge {e}: non-positive speed")
        out_adj[pl[0]].append(e)
    off, ce = S["csrOffsets"], S["csrEdges"]
    assert len(off) == N + 1 and off[0] == 0 and off[N] == E == len(ce), "CSR sizes wrong"
    for n in range(N):
        if sorted(ce[off[n]:off[n + 1]]) != sorted(out_adj[n]):
            err(f"node {n}: CSR row differs from edges grouped by from-node")
    if "linkNodes" in S:
        if len(argv) < 2:
            print("   (pass the main tier meta as 2nd arg to verify linkNodes positions)")
        else:
            mmeta, MS = load_sections(argv[1])
            mx = MS["nodeXYZ"]
            ln = S["linkNodes"]
            for k in range(0, len(ln), 2):
                a, b = ln[k], ln[k + 1]
                if a >= N or b >= mmeta["nodeCount"]:
                    err(f"linkNodes pair {k // 2} out of range"); continue
                if abs(xyz[3 * a] - mx[3 * b]) > 0.2 or abs(xyz[3 * a + 2] - mx[3 * b + 2]) > 0.2:
                    err(f"linkNodes pair {k // 2}: positions differ")
            print(f"   linkNodes: {len(ln) // 2} shared nodes verified against main tier")
    print(f"{argv[0]}: {N} nodes, {E} edges, {len(poly)} polyline indices -> "
          f"{'OK' if errors == 0 else str(errors) + ' ERRORS'}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
