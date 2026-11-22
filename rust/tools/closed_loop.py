# CPU vs GPU, the eval pattern: same population, a fixed simple policy on obs v2 (speed keeping, leader gap, stops for
# red lights / stop signs and holds a light brake there), with static_hold + idle_creep on / off.
# usage: closed_loop.py data n steps backend on|off steer_delay [hold_brake]
# The steering is a lane-keeping controller on the delay-compensated obs 4 (lane offset) / 5 (heading error), with
# per-trip noise swings, so a wrong delay compensation shows up as weaving and kerb strikes.
import sys, numpy as np, bayenv
data, n, steps, backend, flag = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5] == "on"
sd = float(sys.argv[6]) if len(sys.argv) > 6 else 0.15
hold_b = float(sys.argv[7]) if len(sys.argv) > 7 else 0.03
import os
kw = dict(static_hold=flag, idle_creep=flag, steer_delay=sd)
if backend == "gpu" and os.environ.get("OAS") == "0": kw["obs_after_step"] = False
start = 7 * 3600.0
if backend == "cpu": env = bayenv.Env(data, n_cars=n, seed=777, backend="cpu", start_time=start, obs="v2", **kw)
else: env = bayenv.Env(data, n_cars=n, seed=777, population=n, pop_seed=777.0, start_time=start, obs="v2", **kw)
dt = env.dt
stopped = {}   # slot -> seconds stopped
arr = 0; spawned = 0; bar_n = [0]
def policy(o):
    v = o[:, 0] * 40; vl = np.maximum(o[:, 1] * 40, 5.0)
    gap = o[:, 8] * 120
    a = 0.8 * (0.9 * vl - v)
    a = np.where(gap < 2.0 * v + 6.0, np.minimum(a, -1.5 - (2.0 * v + 6.0 - gap) * 0.5), a)
    a = np.where(gap < 4.0, -4.0, a)
    stopc = (o[:, 53] > 0.5) | (o[:, 52] > 0.5) | (o[:, 47] > 0.5) & (o[:, 61] < 0.5)
    d = o[:, 56] * 250
    need = stopc & (d < v * v / 3.0 + 8.0)
    a = np.where(need, np.minimum(a, -(v * v) / (2 * np.maximum(d - 1.5, 0.5)) - 0.3), a)
    A = np.zeros((len(o), 3), np.float32)
    A[:, 0] = np.clip(-2.5 * o[:, 4] + 2.0 * o[:, 5] + noise, -1, 1)
    A[:, 1] = np.clip(a / 3.0, 0, 1); A[:, 2] = np.clip(-a / 6.5, 0, 1)
    hold = (v < 0.5) & ((need) | (gap < 6.0))
    A[hold, 1] = 0.0; A[hold, 2] = hold_b
    return A, v
rng = np.random.default_rng(1)
noise = 0.0
for t in range(steps):
    o = np.array(env.observe())
    if t % 6 == 0: noise = 0.08 * np.sin(np.arange(len(o)) * 0.37 + t * 0.013) + 0.05 * rng.standard_normal(len(o)) * 0
    noise = 0.08 * np.sin(np.arange(len(o)) * 0.37 + t * 0.013)
    A, v = policy(o)
    m = env.n_slots if backend == "cpu" else env.capacity
    if A.shape[0] < m: A = np.vstack([A, np.zeros((m - A.shape[0], 3), np.float32)])
    env.act(A[:m]); env.step(1)
    e = env.events()
    for k, s, oth in zip(e["kind"], e["slot"], e["other"]):
        if k == 3 and oth == -1: bar_n[0] += 1
        if k == 1: arr += 1
        if k == 6: spawned += 1; stopped[int(s)] = 0.0
        if k in (1, 2, 4): stopped.pop(int(s), None)
    st = env.state()
    sp = st["speed"]; state = st["state"]
    for s in list(stopped):
        if s < len(state) and state[s] == 1: stopped[s] = stopped[s] + dt if sp[s] < 0.2 else 0.0
        else: stopped.pop(s, None)
long = sum(1 for x in stopped.values() if x > 300)
stats = env.stats()
print(f"{backend} steer_delay {sd} hold+creep {'on' if flag else 'off'}: spawned {spawned} arrived {arr} driving {int(stats.get('driving', 0))} crashed {int(stats.get('crashed_vehicles', 0))} barrier_rows {bar_n[0]} stopped>5min {long}")
