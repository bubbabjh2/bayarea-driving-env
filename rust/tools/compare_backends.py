"""CPU vs GPU events on the same scenario, k = 1 (rust/README.md "CPU vs GPU events").

    python rust/tools/compare_backends.py data [n_commuters=5000] [steps=3000] [start=25200] [seed=42] [--legacy-gpu] [--spawn-default] [--missed-out-edge] [--no-contact-tangent]

--legacy-gpu: GPU with cpu_connector_join=False and contact_tangent=False, the JS GPU kernels bit-for-bit including their
junction connector-entry bug (rust/docs/kerb-investigation.md) and their f32 contact tangent (rust/docs/crash-divergence.md).
--missed-out-edge: GPU missed exits through a junction connector are reported on the taken out-edge (CPU timing).
--no-contact-tangent: GPU with contact_tangent=False only (the JS kernels' f32 friction tangent in the contact impulse,
rust/docs/crash-divergence.md).
--spawn-default: GPU rows of slots that are not driving get the CPU sim's default action [0, 0, 0.3]. The CPU applies
an action only to cars driving when act() is called, so a car spawning during a step brakes (default action) on its
first decision; the GPU holds the slot's row (gas 0.3). This shifts the CPU trajectories by ~2 steps (harness only).

Same population (population.js, `seed`), same start time and dt (0.05), every driving car commanded [0, 0.3, 0]
every step on both backends. Events are matched by (agent, kind) and by step (exact and within ±20 steps); arrivals
also compare late_by. Slots differ between backends, so agents (trip ids) are the key.
"""
import sys, collections
import numpy as np
import bayenv

cj = "--legacy-gpu" not in sys.argv; sd = "--spawn-default" in sys.argv; mo = "--missed-out-edge" in sys.argv; ct = cj and "--no-contact-tangent" not in sys.argv
sys.argv = [a for a in sys.argv if a not in ("--legacy-gpu", "--spawn-default", "--missed-out-edge", "--no-contact-tangent")]
data = sys.argv[1] if len(sys.argv) > 1 else "data"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
start = float(sys.argv[4]) if len(sys.argv) > 4 else 7 * 3600.0
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 42
KIND = {1: "arrived", 2: "crashed", 3: "contact", 4: "stranded", 5: "cleared", 6: "spawned", 7: "missed_exit", 8: "violation"}

def run(backend):
    if backend == "cpu": env = bayenv.Env(data, n_cars=n, seed=seed, backend="cpu", start_time=start, obs="none")
    else: env = bayenv.Env(data, n_cars=n, seed=seed, population=n, pop_seed=float(seed), start_time=start, obs="none", cpu_connector_join=cj, missed_at_out_edge=mo, contact_tangent=ct)
    rows = collections.defaultdict(list); late = {}
    for t in range(steps):
        m = env.n_slots if backend == "cpu" else env.capacity
        A = np.zeros((m, 3), np.float32); A[:, 1] = 0.3
        if sd and backend == "gpu": A[env.state()["state"][:m] != 1] = (0.0, 0.0, 0.3)
        env.act(A); env.step(1)
        e = env.events()
        for k, a, s, lb in zip(e["kind"], e["agent"], e["step"], e["late_by"]):
            rows[(int(a), int(k))].append(int(s))
            if k == 1: late[int(a)] = float(lb)
    return rows, late, env.stats()

cpu, cl, cs = run("cpu"); gpu, gl, gs = run("gpu")
print(f"{n} commuters, {steps} steps from t={start:.0f}, k=1, gpu cpu_connector_join={cj}, contact_tangent={ct}, spawn_default={sd}")
for k, name in KIND.items():
    c = {key: v for key, v in cpu.items() if key[1] == k}; g = {key: v for key, v in gpu.items() if key[1] == k}
    if not c and not g: continue
    both = set(c) & set(g)
    exact = sum(1 for key in both if c[key][0] == g[key][0]); near = sum(1 for key in both if abs(c[key][0] - g[key][0]) <= 20)
    print(f"{name:12} cpu {sum(map(len, c.values())):6} gpu {sum(map(len, g.values())):6}  agents both {len(both):5} / cpu {len(c):5} / gpu {len(g):5}"
          f"  first step equal {exact:5}, within 20 {near:5}")
common = set(cl) & set(gl)
if common: d = np.array([abs(cl[a] - gl[a]) for a in common]); print(f"late_by |cpu - gpu| over {len(common)} arrivals: median {np.median(d):.2f} s, max {d.max():.2f} s")
print("stats cpu", {k: cs.get(k) for k in ["driving", "arrived", "crashed_vehicles", "minor_contacts", "spawned"]})
print("stats gpu", {k: gs.get(k) for k in ["driving", "arrived", "crashed_vehicles", "minor_contacts", "spawned"]})
