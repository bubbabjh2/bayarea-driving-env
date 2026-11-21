#!/usr/bin/env python3
"""
verify_roads.py - acceptance checks for the road data (AUDIT-roads.md, D1/D2/D4/D12/D15),
using the current elevation and lane model.

    python3 -I tools/verify_roads.py [data_dir] [brain_failures.json]

Prints: folds and over-long edges, Bay Bridge deck length, grades per class, crossings
without a shared node by clearance (with the reason for any < 4.5 m: tunnel, merge/diverge
overlap or tight junction geometry of connected roads), elevated edges that never leave grade,
deck dips, lane flicker (split into synthesized acceleration/auxiliary lanes vs other),
lanes above the class cap, bannedHops, and, if given the brain agent's failure file, the
lane-count context at every failure hot spot on the new data.
"""
import heapq
import json
import math
import os
import sys
from collections import Counter, defaultdict

FW = {"motorway", "trunk", "motorway_link", "trunk_link"}
CAP = {"motorway": 6, "motorway_link": 3, "trunk": 4, "trunk_link": 2, "primary": 4, "primary_link": 2,
       "secondary": 3, "secondary_link": 2, "tertiary": 2, "tertiary_link": 2, "unclassified": 2,
       "residential": 2, "living_street": 1}
AUX, DROP = 256, 512


def main(argv):
    argv = [a for a in argv if a != "-v"]
    D = argv[0] if argv else "data"
    m = json.load(open(os.path.join(D, "roads.json")))
    l = json.load(open(os.path.join(D, "roads-local.json")))
    tiers = {"main": m, "local": l}

    print("== D1 folds / length")
    for t, d in tiers.items():
        N, E = d["nodes"], d["edges"]
        folds = longer = 0
        for e in E:
            nd = e["nodes"]
            for i in range(1, len(nd) - 1):
                a, b, c = N[nd[i - 1]], N[nd[i]], N[nd[i + 1]]
                ux, uz, vx, vz = b["x"] - a["x"], b["z"] - a["z"], c["x"] - b["x"], c["z"] - b["z"]
                lu, lv = math.hypot(ux, uz), math.hypot(vx, vz)
                if lu >= 1 and lv >= 1 and (ux * vx + uz * vz) / (lu * lv) < -0.95:
                    folds += 1
                    break
            chord = sum(math.hypot(N[q]["x"] - N[p]["x"], N[q]["z"] - N[p]["z"]) for p, q in zip(nd, nd[1:]))
            if e["length"] > 1.05 * chord + 0.5:
                longer += 1
        print(f"   {t}: fold-backs {folds}, length > 1.05 x chord-sum {longer}")
    N, E = m["nodes"], m["edges"]
    bb = [e for e in E if e["bridge"] and (e["name"] or "").startswith("Dwight D. Eisenhower") and e["length"] > 1500]
    for e in sorted(bb, key=lambda e: -e["length"])[:6]:
        a, b = N[e["nodes"][0]], N[e["nodes"][-1]]
        ys = [N[n]["y"] for n in e["nodes"][1:-1]] or [N[e["nodes"][0]]["y"]]
        print(f"   Bay Bridge span edge {e['id']} layer {e['layer']}: length {e['length']:.0f} m, endpoints "
              f"{math.hypot(b['x'] - a['x'], b['z'] - a['z']):.0f} m apart, deck y {min(ys):.1f}-{max(ys):.1f} m")

    print("== D2 grades (pieces >= 1 m)")
    for t, d in tiers.items():
        N, E = d["nodes"], d["edges"]
        worst = Counter()
        over = Counter()
        for e in E:
            fw = e["class"] in FW
            lim = 0.08 + 0.003      # schema limit 8 % for every class (+ rounding of y to 1 cm)
            nd = e["nodes"]
            for p, q in zip(nd, nd[1:]):
                h = math.hypot(N[q]["x"] - N[p]["x"], N[q]["z"] - N[p]["z"])
                if h < 1:
                    continue
                g = abs(N[q]["y"] - N[p]["y"]) / h
                k = "freeway" if fw else "other"
                worst[k] = max(worst[k], g)
                if g > lim + 0.002:
                    over[k] += 1
        print(f"   {t}: max grade freeway {100 * worst['freeway']:.1f} %, other {100 * worst['other']:.1f} %; "
              f"pieces over the 8 % limit: {dict(over) or 0}")

    print("== D2 crossings without a shared node")
    allN, segs = [], []
    for t, d in tiers.items():
        N = d["nodes"]
        off = len(allN)
        allN.extend(N)
        for e in d["edges"]:
            if e["twin"] >= 0 and e["twin"] < e["id"]:
                continue
            nd = [n + off for n in e["nodes"]]
            segs.append((t, e, nd))
    ids = [n["id"] for n in allN]
    # unify shared nodes (same OSM id) for connectivity
    canon = {}
    for i, nid in enumerate(ids):
        canon.setdefault(nid, i)
    cn = [canon[nid] for nid in ids]
    adj = defaultdict(list)
    for t, e, nd in segs:
        for p, q in zip(nd, nd[1:]):
            L = math.hypot(allN[q]["x"] - allN[p]["x"], allN[q]["z"] - allN[p]["z"])
            adj[cn[p]].append((cn[q], L))
            adj[cn[q]].append((cn[p], L))

    def connected(a, b, limit=300.0):
        dist = {v: 0.0 for v in a}
        hp = [(0.0, v) for v in a]
        while hp:
            dd, v = heapq.heappop(hp)
            if v in b:
                return True
            if dd > dist.get(v, 1e18):
                continue
            for w, L in adj[v]:
                nd_ = dd + L
                if nd_ <= limit and nd_ < dist.get(w, 1e18):
                    dist[w] = nd_
                    heapq.heappush(hp, (nd_, w))
        return False

    cell = 200.0
    grid = defaultdict(list)
    pieces = []
    for si, (t, e, nd) in enumerate(segs):
        for p, q in zip(nd, nd[1:]):
            k = len(pieces)
            pieces.append((si, p, q))
            A, B = allN[p], allN[q]
            for gx in range(int(min(A["x"], B["x"]) // cell), int(max(A["x"], B["x"]) // cell) + 1):
                for gz in range(int(min(A["z"], B["z"]) // cell), int(max(A["z"], B["z"]) // cell) + 1):
                    grid[(gx, gz)].append(k)
    res = Counter()
    why = Counter()
    lows = []
    seen = set()
    for cellk, lst in grid.items():
        for x in range(len(lst)):
            for y_ in range(x + 1, len(lst)):
                a, b = lst[x], lst[y_]
                if (a, b) in seen:
                    continue
                seen.add((a, b))
                sa, p, q = pieces[a]
                sb, r, s = pieces[b]
                if sa == sb or len({cn[p], cn[q], cn[r], cn[s]}) < 4:
                    continue
                ta, ea, _ = segs[sa]
                tb, eb, _ = segs[sb]
                fa, fb = ea["class"] in FW, eb["class"] in FW
                if ta == "local" and tb == "local":
                    continue
                if not (fa or fb):
                    if ta == "local" or tb == "local":
                        continue
                    if not (ea["bridge"] or eb["bridge"] or ea["layer"] or eb["layer"]):
                        continue
                P, Q, R, S = allN[p], allN[q], allN[r], allN[s]
                d1x, d1z, d2x, d2z = Q["x"] - P["x"], Q["z"] - P["z"], S["x"] - R["x"], S["z"] - R["z"]
                den = d1x * d2z - d1z * d2x
                if abs(den) < 1e-9:
                    continue
                t_ = ((R["x"] - P["x"]) * d2z - (R["z"] - P["z"]) * d2x) / den
                u = ((R["x"] - P["x"]) * d1z - (R["z"] - P["z"]) * d1x) / den
                if not (1e-6 < t_ < 1 - 1e-6 and 1e-6 < u < 1 - 1e-6):
                    continue
                dy = abs(P["y"] + (Q["y"] - P["y"]) * t_ - R["y"] - (S["y"] - R["y"]) * u)
                kind = "freeway involved" if (fa or fb) else "arterial bridges"
                tier = "local x freeway" if (ta == "local" or tb == "local") else "main"
                ok = dy >= 4.5
                res[(tier, kind, "ok" if ok else "<4.5 m")] += 1
                if not ok:
                    la_, lb_ = math.hypot(d1x, d1z), math.hypot(d2x, d2z)
                    ang = math.degrees(math.acos(min(1.0, abs(d1x * d2x + d1z * d2z) / max(1e-9, la_ * lb_))))
                    if ea["tunnel"] or eb["tunnel"]:
                        w = "tunnel (road passes over a tunnel)"
                    elif connected({cn[p], cn[q]}, {cn[r], cn[s]}):
                        w = "connected within 300 m, " + ("near-parallel overlap (< 25 deg)" if ang < 25 else "crossing angle >= 25 deg")
                    else:
                        w = "UNRESOLVED"
                    why[w] += 1
                    lows.append((kind, w, round(ang), ea["class"], ea["name"], ea["id"], eb["class"], eb["name"], eb["id"],
                                 round(P["x"]), round(P["z"]), round(dy, 1)))
    for k in sorted(res):
        print("  ", k, res[k])
    print("   below 4.5 m, by reason:", dict(why) or 0)
    if "-v" in sys.argv:
        for x in sorted(lows):
            print("     ", x)

    print("== D2 structures")
    for t, d in tiers.items():
        N, E = d["nodes"], d["edges"]
        el = [e for e in E if (e["bridge"] or e["layer"] > 0) and not e["tunnel"] and (e["twin"] < 0 or e["id"] < e["twin"])]
        never = sum(1 for e in el if max(N[n]["y"] for n in e["nodes"]) < 0.5)
        deck = {}
        for e in el:
            inner = [N[n]["y"] for n in e["nodes"][1:-1]] or [max(N[e["nodes"][0]]["y"], N[e["nodes"][-1]]["y"])]
            deck[e["id"]] = max(inner)
        at = defaultdict(list)
        for e in el:
            at[e["nodes"][0]].append(e)
            at[e["nodes"][-1]].append(e)
        dips = sum(1 for n, lst in at.items() if len(lst) >= 2 and N[n]["y"] < min(deck[e["id"]] for e in lst) - 0.5)
        mx = max((N[n]["y"] for n in range(len(N))), default=0)
        print(f"   {t}: elevated edges {len(el)}, never leave grade {never}, deck dips at junctions {dips}, max y {mx:.1f} m")

    print("== D4 lane flicker (change at a continuation reverting within 300 m), main tier")
    N, E = m["nodes"], m["edges"]
    out = defaultdict(list)
    for e in E:
        out[e["from"]].append(e)

    def hd(e, end):
        nd = e["nodes"]
        a, b = (N[nd[-2]], N[nd[-1]]) if end else (N[nd[0]], N[nd[1]])
        return math.atan2(b["z"] - a["z"], b["x"] - a["x"])

    def succ(e):
        best, bd = None, 0.35
        for o in out[e["to"]]:
            if o["id"] == e["twin"] or o["class"] != e["class"]:
                continue
            dd = abs((hd(o, 0) - hd(e, 1) + math.pi) % (2 * math.pi) - math.pi)
            if dd < bd:
                bd, best = dd, o
        return best
    for cls in ("motorway", "trunk", "primary"):
        fl = Counter()
        for e in E:
            if e["class"] != cls:
                continue
            s = succ(e)
            if not s or s["lanes"] == e["lanes"]:
                continue
            acc, t, k, seq = s["length"], s, 0, [s]
            while acc < 300 and k < 20:
                t = succ(t)
                k += 1
                if not t:
                    break
                if t["lanes"] == e["lanes"]:
                    synth = any(x.get("laneFlags", 0) & AUX for x in seq + [e])
                    fl["synthesized accel/aux lane" if synth else "other"] += 1
                    break
                seq.append(t)
                acc += t["length"]
        print(f"   {cls}: {dict(fl) or 0}")
    caps = sum(1 for d in tiers.values() for e in d["edges"] if e["lanes"] > CAP[e["class"]] + (1 if e.get("laneFlags", 0) & AUX else 0))
    print(f"   edges above the class cap per direction (+1 allowed for a synthesized lane): {caps}")
    print(f"   bannedHops (impossible weaves): {len(m.get('bannedHops', []))}")

    print("== build order: controls stop lines == junction mouths (tools/build_junctions.mjs after build_controls.py)")
    jm_path, cm_path = os.path.join(D, "junctions.meta.json"), os.path.join(D, "controls.meta.json")
    order_ok = True
    if os.path.exists(jm_path) and os.path.exists(cm_path):
        from array import array as _arr
        jm, cm = json.load(open(jm_path)), json.load(open(cm_path))
        def sec(meta, name):
            sct = meta["sections"][name]
            a = _arr("f" if sct["type"] == "Float32" else "I")
            with open(os.path.join(D, meta["file"]), "rb") as f:
                f.seek(sct["byteOffset"]); a.frombytes(f.read(sct["byteLength"]))
            return a
        mouth = sec(jm, "mouthS"); info = sec(cm, "controlInfo"); aedge = sec(cm, "approachEdge"); astop = sec(cm, "approachStopAt")
        bad = 0; ex = None
        for c in range(len(info) // 6):
            typ, node, first, cnt = info[6 * c], info[6 * c + 3], info[6 * c + 4], info[6 * c + 5]
            if node == 0xFFFFFFFF or typ == 3:
                continue
            for a in range(first, first + cnt):
                g = aedge[a]
                L = (m["edges"][g] if g < len(m["edges"]) else l["edges"][g - len(m["edges"])])["length"]
                if mouth[2 * g + 1] >= L - 0.5:
                    continue          # no junction box there: the control keeps its own stop line
                if abs(astop[a] - mouth[2 * aedge[a] + 1]) > 0.05:
                    bad += 1
                    ex = ex or (aedge[a], round(astop[a], 2), round(mouth[2 * aedge[a] + 1], 2))
        print(f"   junction-control approaches whose stop line differs from sStop: {bad}" + (f"  e.g. edge/stopAt/sStop {ex}" if ex else ""))
        order_ok = bad == 0
    else:
        print("   junctions.* or controls.* missing")

    if len(argv) <= 1:
        return 0 if order_ok else 1
    if len(argv) > 1:
        print("== D15 brain-run failure spots on the new data (hot edges, main tier)")
        fz = json.load(open(argv[1]))
        hot = [h for h in fz["hotEdges"] if h["tier"] == "main"]
        banned_edges = {x for h in m.get("bannedHops", []) for x in h}
        cls_of = Counter()
        rows = []
        for h in hot:
            best, bd = None, 30.0
            for e in E:
                if e["class"] != h["highway"]:
                    continue
                for p, q in zip(e["nodes"], e["nodes"][1:]):
                    A, B = N[p], N[q]
                    dx, dz = B["x"] - A["x"], B["z"] - A["z"]
                    L2 = dx * dx + dz * dz or 1
                    tt = max(0, min(1, ((h["x"] - A["x"]) * dx + (h["z"] - A["z"]) * dz) / L2))
                    dd = math.hypot(A["x"] + dx * tt - h["x"], A["z"] + dz * tt - h["z"])
                    if dd < bd:
                        bd, best = dd, e
            if not best:
                cls_of["not found within 30 m"] += 1
                continue
            # lane profile 400 m up/downstream along the straightest same-class continuation
            prev = [x for x in E if x["to"] == best["from"] and x["id"] != best["twin"] and x["class"] in FW | {best["class"]}]
            nxt = [x for x in out[best["to"]] if x["id"] != best["twin"]]
            seqd, t, acc = [best["lanes"]], best, 0.0
            while acc < 400:
                t = succ(t)
                if not t:
                    break
                seqd.append(t["lanes"])
                acc += t["length"]
            changes = sum(1 for a, b in zip(seqd, seqd[1:]) if a != b)
            flags = best.get("laneFlags", 0)
            if best["id"] in banned_edges:
                c = "impossible weave -> bannedHops"
            elif changes == 0 and len({p["lanes"] for p in prev} | {best["lanes"]}) <= 1:
                c = "clean (no lane-count change within 400 m)"
            elif flags & DROP:
                c = "flagged lane drop (warning + taper)"
            elif flags & AUX:
                c = "synthesized accel/aux lane"
            else:
                c = "lane-count change (merge/diverge/add)"
            cls_of[c] += 1
            rows.append((h["srcEdgeId"], h["lanesSeq"], h["count"], best["id"], best["lanes"], seqd[:6], c))
        for k, v in cls_of.most_common():
            print(f"   {v:4d}  {k}")
        rows.sort(key=lambda r: -r[2])
        for r in rows[:12]:
            print(f"   old edge {r[0]} ({r[1]}, {r[2]} failures) -> new edge {r[3]} lanes {r[4]} downstream {r[5]}: {r[6]}")
    return 0 if order_ok else 1


def _unused():
    pass


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
