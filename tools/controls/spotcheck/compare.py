import json, math, os, glob, sys
# Compare named intersections with live OSM (responses fetched by tools/controls/spotcheck/fetch.sh).
#   python3 -I tools/controls/spotcheck/compare.py [data] [data/build/spotcheck]
D = sys.argv[1] if len(sys.argv) > 1 else 'data'
RESP = sys.argv[2] if len(sys.argv) > 2 else os.path.join(D, 'build', 'spotcheck')
m = json.load(open(os.path.join(D, 'roads.meta.json')))['projection']
kx, kz, la0, lo0 = m['metersPerDegLon'], m['metersPerDegLat'], m['lat0'], m['lon0']
P = lambda la, lo: ((lo - lo0) * kx, -(la - la0) * kz)
ctl = (json.load(open(os.path.join(D, 'controls.json'))) if os.path.exists(os.path.join(D, 'controls.json')) else json.load(__import__('gzip').open(os.path.join(D, 'controls.json.gz'))))['controls']
tiers = {t: json.load(open(os.path.join(D, f))) if os.path.exists(os.path.join(D, f)) else json.load(__import__('gzip').open(os.path.join(D, f + '.gz'))) for t, f in (('main', 'roads.json'), ('local', 'roads-local.json'))}
def edge_pt(tier, te, s):
    T = tiers[tier]; e = T['edges'][te]; ns = [T['nodes'][k] for k in e['nodes']]
    acc = 0
    for a, b in zip(ns, ns[1:]):
        d = math.hypot(b['x'] - a['x'], b['z'] - a['z'])
        if acc + d >= s or b is ns[-1]:
            f = 0 if d == 0 else min(1, max(0, (s - acc) / d)); return (a['x'] + f * (b['x'] - a['x']), a['z'] + f * (b['z'] - a['z'])), e
        acc += d
    return (ns[-1]['x'], ns[-1]['z']), e
for f in sorted(glob.glob(os.path.join(RESP, '*.json'))):
    try: j = json.load(open(f))
    except Exception: print(os.path.basename(f), "fetch failed"); continue
    el = j.get("elements", [])
    if not el: print(os.path.basename(f), 'NO DATA', j.get('remark', '')[:80]); continue
    xs = [e for e in el if not e.get('tags')] or el[:1]
    jx = [P(e['lat'], e['lon']) for e in el if not set((e.get('tags') or {}).values()) & {'traffic_signals', 'stop', 'give_way'}] or [P(xs[0]['lat'], xs[0]['lon'])]
    cx = sum(p[0] for p in jx) / len(jx); cz = sum(p[1] for p in jx) / len(jx)
    osm = [e for e in el if set((e.get('tags') or {}).values()) & {'traffic_signals', 'stop', 'give_way'}]
    otypes = sorted(set(v for e in osm for k, v in e['tags'].items() if k in ('highway', 'crossing') and v in ('traffic_signals', 'stop', 'give_way')))
    near = [c for c in ctl if math.hypot(c['x'] - cx, c['z'] - cz) < 80]
    ids = set(n for c in near for n in c['osmNodes'])
    miss = [e['id'] for e in osm if e['id'] not in ids and e['tags'].get('highway') == 'traffic_signals']
    # stop-line distances from the junction centre, per approach
    rows = []
    for c in near:
        for a in c['approaches']:
            (x, z), e = edge_pt(a['tier'], a['tierEdge'], a['stopAt'])
            rows.append((c['type'], round(math.hypot(x - cx, z - cz), 1), e.get('name'), a['class']))
    sig = [r for r in rows if r[0] == 'signal']
    print(f"{os.path.basename(f)[:-5]:28s} osm: {len(osm)} ctl nodes {otypes}; data: {len(near)} controls {sorted(set(c['type'] for c in near))}, {len(rows)} approaches; signal stop lines {sorted(r[1] for r in sig)} m from centre; OSM signal nodes not in data: {len(miss)}")

# accuracy: OSM per-approach signal / stop nodes (mapped away from the junction node, usually at the stop line)
print('\nper-approach OSM control node -> nearest data stop line (m):')
allD = []
for f in sorted(glob.glob(os.path.join(RESP, '*.json'))):
    try: el = json.load(open(f)).get('elements', [])
    except Exception: continue
    if not el: continue
    xn = [P(e['lat'], e['lon']) for e in el if not e.get('tags')]
    osm = [e for e in el if (e.get('tags') or {}).get('highway') in ('traffic_signals', 'stop')]
    lines = []
    for c in ctl:
        if xn and min(math.hypot(c['x'] - x, c['z'] - z) for x, z in xn) < 120:
            for a in c['approaches']:
                lines.append(edge_pt(a['tier'], a['tierEdge'], a['stopAt'])[0])
    ds = []
    for e in osm:
        x, z = P(e['lat'], e['lon'])
        if xn and min(math.hypot(x - a, z - b) for a, b in xn) < 6: continue   # on the junction node itself
        if lines: ds.append(round(min(math.hypot(x - a, z - b) for a, b in lines), 1))
    allD += ds
    print(f"  {os.path.basename(f)[:-5]:28s} {sorted(ds)}")
allD.sort(); n = len(allD)
if n: print(f"  all: n={n} median {allD[n // 2]} m, p80 {allD[int(n * 0.8)]} m, max {allD[-1]} m")
