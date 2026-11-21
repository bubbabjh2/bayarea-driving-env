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
    print(f"Env n={n}: step(20) {a:.3f} ms/step | act+step(1)+observe {b:.3f} ms | step_observe(A, 1) {c:.3f} ms")
    del env
for E, n in [(1, 10000), (4, 10000), (8, 10000), (16, 2500), (4, 25000)]:
    v = bayenv.VecEnv(data, n_envs=E, n_cars=n, seed=11, accidents=True)
    A = np.zeros((E, v.n_slots, 3), np.float32); A[..., 1] = 0.3
    out = np.empty((E, v.n_slots, v.obs_size), np.float32)
    c = timeit(lambda: v.step_observe(A, 1, out=out), 50)
    print(f"VecEnv {E} x {n}: step_observe(A, 1) {c:.3f} ms per call -> {E * n / c * 1e3 / 1e6:.2f} M car-steps/s ({E / c * 1e3:.0f} env-steps/s)")
    del v
