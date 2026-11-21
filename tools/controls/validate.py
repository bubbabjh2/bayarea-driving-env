# Validate data/controls.json against the authoritative records at 24 sampled intersections (seeded, per city).
#   python3 -I tools/controls/validate.py [data] [data/controls-src] > tools/controls/report/validation.txt
# Merged datasets (SF, San Jose) and verify-only ones (Oakland, South SF, Mountain View; data/controls-src/verify/,
# fetched by tools/controls/fetch.mjs) are sampled; a missing verify-only snapshot is skipped.
import json, gzip, math, os, random, sys
from collections import defaultdict
D = sys.argv[1] if len(sys.argv) > 1 else 'data'
S = sys.argv[2] if len(sys.argv) > 2 else os.path.join(D, 'controls-src')
pr = json.load(open(os.path.join(D, 'roads.meta.json')))['projection']
P = lambda lo, la: ((lo - pr['lon0']) * pr['metersPerDegLon'], -(la - pr['lat0']) * pr['metersPerDegLat'])
ctl = json.load(open(os.path.join(D, 'controls.json')))['controls']
tiers = {t: json.load(open(os.path.join(D, f))) for t, f in (('main', 'roads.json'), ('local', 'roads-local.json'))}
def stop_pt(a):
    T = tiers[a['tier']]; e = T['edges'][a['tierEdge']]; ns = [T['nodes'][k] for k in e['nodes']]; acc = 0
    for p, q in zip(ns, ns[1:]):
        d = math.hypot(q['x'] - p['x'], q['z'] - p['z'])
        if acc + d >= a['stopAt']: f = (a['stopAt'] - acc) / d if d else 0; return p['x'] + f * (q['x'] - p['x']), p['z'] + f * (q['z'] - p['z'])
        acc += d
    return ns[-1]['x'], ns[-1]['z']
def F(i):
    for p in (os.path.join(S, i + '.geojson.gz'), os.path.join(S, 'verify', i + '.geojson.gz')):
        if os.path.exists(p): return json.load(gzip.open(p))['features']
    return []
name = {
 'sf-signals': lambda p: f"{p.get('street1')} & {p.get('street2')}",
 'sj-signals': lambda p: f"{p.get('INTERSECTIONNAME1')} & {p.get('INTERSECTIONNAME2')}",
 'oakland-signals': lambda p: p.get('LOCATION'),
 'ssf-signals': lambda p: p.get('Location'),
 'mv-signals': lambda p: f"pole {p.get('FACILITYID')} ({p.get('MOUNTTYPE')})",
}
keep = {'sf-signals': lambda p: p.get('type') == 'SIGNAL', 'sj-signals': lambda p: p['SHOPTYPE'] == 'Traffic Signal' and p['STATUS'] == 'Existing'}
R = random.Random(7); rows = []
for sid, n in (('sf-signals', 6), ('sj-signals', 6), ('oakland-signals', 4), ('ssf-signals', 2), ('mv-signals', 2)):
    fs = [f for f in F(sid) if f.get('geometry') and keep.get(sid, lambda p: True)(f['properties'])]
    if not fs: print(f'{sid}: no snapshot, skipped'); continue
    for f in R.sample(fs, n):
        x, z = P(*f['geometry']['coordinates'][:2])
        near = [c for c in ctl if c['type'] == 'signal' and math.hypot(c['x'] - x, c['z'] - z) < 60]
        aps = [a for c in near for a in c['approaches']]
        dl = sorted(round(math.hypot(px - x, pz - z), 1) for px, pz in map(stop_pt, aps))
        src = sorted({c['source'] for c in near})
        rows.append((sid, name[sid](f['properties']), len(near) > 0, len(aps), dl, src))
# SF stop intersections: approaches in the authoritative list vs in the data
st = defaultdict(list)
for f in F('sf-stop-signs'):
    if f.get('geometry'): st[f['properties'].get('cnn')].append(f)
for cnn in R.sample(sorted(k for k in st if k), 4):
    fs = st[cnn]; x, z = P(*fs[0]['geometry']['coordinates'][:2])
    near = [c for c in ctl if c['type'] in ('stop',) and math.hypot(c['x'] - x, c['z'] - z) < 40]
    aps = [a for c in near for a in c['approaches']]
    p = fs[0]['properties']
    rows.append(('sf-stop-signs', f"{p.get('street')} & {p.get('x_street')} ({len(fs)} signs: {','.join(sorted(str(g['properties'].get('st_facing')) for g in fs))})", len(near) > 0, len(aps), sorted(round(math.hypot(px - x, pz - z), 1) for px, pz in map(stop_pt, aps)), sorted({c['source'] for c in near}) + (['allWay'] if any(c['allWay'] for c in near) else [])))
ok = 0
for r in rows:
    ok += r[2]; print(f"{r[0]:15s} {str(r[1])[:58]:58s} present={'Y' if r[2] else 'N'} approaches={r[3]:2d} stopLine m from record {r[4]} {r[5]}")
print(f"present {ok}/{len(rows)}")
