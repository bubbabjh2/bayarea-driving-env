"""GPU half (and any-backend tracer) of the CPU-vs-GPU crash divergence dumps (rust/docs/crash-divergence.md).

    python rust/tools/gpu_divergence.py data --agents a,b,c [--backend gpu|cpu] [--from S] [--to T] [--prefix gpu]
                                         [--n 5000] [--start 25200] [--seed 42] [--cpu-dir DIR] [--no-contact-tangent]

Same setup as cpu_divergence.py / compare_backends.py --cpu-connector-join --spawn-default (k = 1, every driving car
[0, 0.3, 0], non-driving GPU rows [0, 0, 0.3], GPU with cpu_connector_join). Writes, per agent, one JSON object per
step in [from, to] (state at the END of the step, the cpu_divergence.py keys) to
rust/oracle/divergence/<prefix>-<agent>.jsonl. Default steps: those of rust/oracle/divergence/cpu-<agent>.jsonl
(or --cpu-dir). Agents are mapped to slots by this backend's own spawn rows (events kind 6). Prints every event row
of the traced agents in the window, with the partner slot mapped back to its agent. GPU: the default
model (cpu_connector_join, contact_tangent); --no-contact-tangent: contact_tangent=False (the JS kernels' tangent; the
gpu-<agent> dumps of rust/docs/crash-divergence.md).
"""
import sys, os, json
import numpy as np
import bayenv

args = sys.argv[1:]
ct = "--no-contact-tangent" not in args; args = [a for a in args if a != "--no-contact-tangent"]
def opt(name, d):
    if name in args: i = args.index(name); v = args[i + 1]; del args[i:i + 2]; return v
    return d
backend = opt("--backend", "gpu"); prefix = opt("--prefix", backend)
agents = [int(a) for a in opt("--agents", "").split(",") if a]
f0 = opt("--from", None); t0 = opt("--to", None)
n = int(opt("--n", "5000")); start = float(opt("--start", "25200")); seed = int(opt("--seed", "42"))
out_dir = "rust/oracle/divergence"; cpu_dir = opt("--cpu-dir", out_dir)
data = args[0] if args else "data"
FIELDS = ["edge", "s", "lane", "lane_lo", "lane_hi", "lat", "heading_rel", "yaw_rate", "speed", "v_lat", "x", "y", "z", "heading",
          "leader", "gap", "ctl_type", "ctl_state", "ctl_dist", "flags", "state", "conn", "route_pos", "steer", "gas", "brake"]

win = {}
for a in agents:
    if f0 is not None: win[a] = (int(f0), int(t0))
    else:
        st = [json.loads(l)["step"] for l in open(f"{cpu_dir}/cpu-{a}.jsonl")]
        win[a] = (min(st), max(st) if t0 is None else int(t0))
last = max(w[1] for w in win.values())

if backend == "cpu": env = bayenv.Env(data, n_cars=n, seed=seed, backend="cpu", start_time=start, obs="none")
else: env = bayenv.Env(data, n_cars=n, seed=seed, population=n, pop_seed=float(seed), start_time=start, obs="none", cpu_connector_join=True, contact_tangent=ct)
slot_of = {}; agent_of_slot = {}; rows = {a: [] for a in agents}; evs = []
KIND = {1: "arrived", 2: "crashed", 3: "contact", 4: "stranded", 5: "cleared", 6: "spawned", 7: "missed_exit", 8: "violation"}
for t in range(last):
    m = env.n_slots if backend == "cpu" else env.capacity
    A = np.zeros((m, 3), np.float32); A[:, 1] = 0.3
    if backend == "gpu": A[env.state()["state"][:m] != 1] = (0.0, 0.0, 0.3)
    env.act(A); env.step(1)
    step = int(env.steps); e = env.events()
    for k, sl, ag in zip(e["kind"], e["slot"], e["agent"]):
        if k == 6: slot_of[int(ag)] = int(sl); agent_of_slot[int(sl)] = int(ag)
    for i in range(len(e["kind"])):
        a = int(e["agent"][i])
        if a in win and win[a][0] <= step <= win[a][1] or (a in win and int(e["kind"][i]) in (2, 3)):
            o = int(e["other"][i])
            evs.append((step, a, KIND.get(int(e["kind"][i])), int(e["slot"][i]), o, agent_of_slot.get(o, -1) if o >= 0 else -1))
    want = [a for a in agents if win[a][0] <= step <= win[a][1] and a in slot_of]
    if not want: continue
    st = env.state()
    for a in want:
        i = slot_of[a]
        if st["state"][i] == 0: continue
        rec = {"step": step, "time": float(env.time), "agent": a, "slot": i}
        for f in FIELDS:
            v = st[f][i]
            rec[f] = (None if not np.isfinite(v) else float(v)) if isinstance(v, (float, np.floating)) else int(v)
        rows[a].append(rec)
os.makedirs(out_dir, exist_ok=True)
for a in agents:
    with open(f"{out_dir}/{prefix}-{a}.jsonl", "w") as fh:
        for r in rows[a]: fh.write(json.dumps(r) + "\n")
    print(f"agent {a}: slot {slot_of.get(a)}, {len(rows[a])} rows -> {out_dir}/{prefix}-{a}.jsonl")
for ev in evs: print("event step %d agent %d %s slot %d other_slot %d other_agent %d" % ev)
