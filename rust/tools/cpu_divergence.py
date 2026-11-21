"""CPU half of the CPU-vs-GPU crash divergence dumps (rust/docs/cpu-vs-gpu.md).

    python rust/tools/cpu_divergence.py data [n=5000] [steps=3000] [start=25200] [seed=42] [--agents a,b,c] [--k 3] [--window 50]

Runs the compare_backends.py setup (k = 1, every driving car [0, 0.3, 0]; GPU with cpu_connector_join and the
spawn-default rows) on both backends, picks the first k agents that are CRASHED (kind 2) on the CPU only (or the
given --agents), reruns the CPU and writes, for each, the CPU state of that agent's slot for the `window` steps
before its crash step and the crash step itself to rust/oracle/divergence/cpu-<agent>.jsonl (one JSON object per
step, fields in the doc).
"""
import sys, os, json, collections
import numpy as np
import bayenv

args = [a for a in sys.argv[1:]]
def opt(name, d):
    if name in args: i = args.index(name); v = args[i + 1]; del args[i:i + 2]; return v
    return d
pick = opt("--agents", None); k_pick = int(opt("--k", "3")); window = int(opt("--window", "50"))
data = args[0] if len(args) > 0 else "data"
n = int(args[1]) if len(args) > 1 else 5000
steps = int(args[2]) if len(args) > 2 else 3000
start = float(args[3]) if len(args) > 3 else 7 * 3600.0
seed = int(args[4]) if len(args) > 4 else 42
FIELDS = ["edge", "s", "lane", "lane_lo", "lane_hi", "lat", "heading_rel", "yaw_rate", "speed", "v_lat", "x", "y", "z", "heading",
          "leader", "gap", "ctl_type", "ctl_state", "ctl_dist", "flags", "state", "conn", "route_pos", "steer", "gas", "brake"]

def make(backend):
    if backend == "cpu": return bayenv.Env(data, n_cars=n, seed=seed, backend="cpu", start_time=start, obs="none")
    return bayenv.Env(data, n_cars=n, seed=seed, population=n, pop_seed=float(seed), start_time=start, obs="none", cpu_connector_join=True)

def act(env, backend):
    m = env.n_slots if backend == "cpu" else env.capacity
    A = np.zeros((m, 3), np.float32); A[:, 1] = 0.3
    if backend == "gpu": A[env.state()["state"][:m] != 1] = (0.0, 0.0, 0.3)
    env.act(A)

def crashes(backend):
    env = make(backend); out = {}
    for t in range(steps):
        act(env, backend); env.step(1)
        e = env.events()
        for kd, a, s in zip(e["kind"], e["agent"], e["step"]):
            if kd == 2 and int(a) not in out: out[int(a)] = int(s)
    return out

if pick: agents = {int(a): None for a in pick.split(",")}
else:
    c = crashes("cpu"); g = crashes("gpu")
    only = sorted((s, a) for a, s in c.items() if a not in g)
    agents = {a: s for s, a in only[:k_pick]}
    print(f"cpu crashed {len(c)}, gpu {len(g)}, cpu only {len(only)}; picked {agents}")

os.makedirs("rust/oracle/divergence", exist_ok=True)
env = make("cpu")
hist = {a: collections.deque(maxlen=window + 1) for a in agents}
done = set()
for t in range(steps):
    act(env, "cpu"); env.step(1)
    st = env.state(); e = env.events()
    step = int(env.steps)
    agent_of = 2 * st["commuter"].astype(np.int64) + st["leg"].astype(np.int64)
    for a in agents:
        if a in done: continue
        idx = np.nonzero((agent_of == a) & (st["commuter"] >= 0) & (st["state"] != 0))[0]
        if len(idx):
            i = int(idx[0])
            rec = {"step": step, "time": float(env.time), "agent": a, "slot": i}
            for f in FIELDS:
                v = st[f][i]; rec[f] = None if (isinstance(v, (float, np.floating)) and not np.isfinite(v)) else (float(v) if isinstance(v, (float, np.floating)) else int(v))
            hist[a].append(rec)
        crashed_now = any(int(kd) == 2 and int(x) == a for kd, x in zip(e["kind"], e["agent"]))
        if crashed_now:
            with open(f"rust/oracle/divergence/cpu-{a}.jsonl", "w") as fh:
                for r in hist[a]: fh.write(json.dumps(r) + "\n")
            done.add(a); print(f"agent {a}: crash at step {step}, wrote {len(hist[a])} rows")
    if len(done) == len(agents): break
