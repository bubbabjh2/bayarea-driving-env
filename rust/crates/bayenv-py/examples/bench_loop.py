"""Per-step RL loop overhead and multi-env throughput.
python rust/crates/bayenv-py/examples/bench_loop.py data"""
import sys, time
import numpy as np
import bayenv

data = sys.argv[1] if len(sys.argv) > 1 else "data"
def timeit(f, n):
    for _ in range(5): f()
    t = time.perf_counter()
    for _ in range(n): f()
    return (time.perf_counter() - t) / n * 1e3

for n in [10000, 100000]:
    env = bayenv.Env(data, n_cars=n, seed=11, accidents=True)
    A = np.zeros((env.n_slots, 3), np.float32); A[:, 1] = 0.3; env.act(A)
    obs = np.empty((env.n_slots, env.obs_size), np.float32)
    a = timeit(lambda: env.step(20), 20) / 20
    b = timeit(lambda: (env.act(A), env.step(1), env.observe(out=obs)), 100)
    c = timeit(lambda: env.step_observe(A, 1, out=obs), 100)
    print(f"Env n={n}: step(20) {a:.3f} ms/step | act+step(1)+observe {b:.3f} ms | step_observe(A, 1, out=) {c:.3f} ms")
    # observation variants, round robin (blocks of 10 calls in turn, so every variant sees the same sim state; the
    # step cost drifts as traffic evolves), min over 30 blocks
    h16 = np.empty((env.n_slots, env.obs_size), np.float16)
    rows = {
        "act+step(1)+sync (no obs)": lambda: (env.act(A), env.step(1), env.sync()),
        "observe(out=)": lambda: env.observe(out=obs),
        "observe(copy=False)": lambda: env.observe(copy=False),
        "step_observe(A, 1, out=)": lambda: env.step_observe(A, 1, out=obs),
        "step_observe(A, 1, copy=False)": lambda: env.step_observe(A, 1, copy=False),
        "step_observe(A, 1, out=, lag=1)": lambda: env.step_observe(A, 1, out=obs, lag=1),
        "step_observe(A, 1, copy=False, lag=1)": lambda: env.step_observe(A, 1, copy=False, lag=1),
        "step_async(A, 1) + observe_wait(copy=False)": lambda: (env.step_async(A, 1), env.observe_wait(copy=False)),
        "step_observe(A, 1, out=f16, dtype=float16)": lambda: env.step_observe(A, 1, out=h16, dtype="float16"),
        "step_observe(A, 1, copy=False, dtype=float16)": lambda: env.step_observe(A, 1, copy=False, dtype="float16"),
        # a policy stand-in reading the whole observation (obs.sum(), ~0.7 ms at 100k): hidden behind the GPU with lag=1
        "policy + step_observe(A, 1, out=)": lambda: env.step_observe(A, 1, out=obs).sum(),
        "policy + step_observe(A, 1, copy=False)": lambda: env.step_observe(A, 1, copy=False).sum(),
        "policy + step_observe(A, 1, copy=False, lag=1)": lambda: env.step_observe(A, 1, copy=False, lag=1).sum(),
    }
    best = {k: 1e9 for k in rows}
    for f in rows.values(): f()
    for _ in range(30):
        for name, f in rows.items():
            f(); t = time.perf_counter()
            for _ in range(10): f()
            best[name] = min(best[name], (time.perf_counter() - t) / 10 * 1e3)
    for name in rows: print(f"    {name}: {best[name]:.3f} ms")
    del env
for E, n in [(1, 10000), (4, 10000), (8, 10000), (16, 2500), (4, 25000)]:
    v = bayenv.VecEnv(data, n_envs=E, n_cars=n, seed=11, accidents=True)
    A = np.zeros((E, v.n_slots, 3), np.float32); A[..., 1] = 0.3
    out = np.empty((E, v.n_slots, v.obs_size), np.float32)
    c = timeit(lambda: v.step_observe(A, 1, out=out), 50)
    print(f"VecEnv {E} x {n}: step_observe(A, 1) {c:.3f} ms per call -> {E * n / c * 1e3 / 1e6:.2f} M car-steps/s ({E / c * 1e3:.0f} env-steps/s)")
    for name, f in {"copy=False": lambda: v.step_observe(A, 1, copy=False), "copy=False, lag=1": lambda: v.step_observe(A, 1, copy=False, lag=1),
                    "out=, lag=1": lambda: v.step_observe(A, 1, out=out, lag=1)}.items():
        c = timeit(f, 50); print(f"    step_observe(A, 1, {name}) {c:.3f} ms -> {E / c * 1e3:.0f} env-steps/s")
    del v
