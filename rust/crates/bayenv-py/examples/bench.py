"""Python-side steps/s: python rust/crates/bayenv-py/examples/bench.py data [10000,100000]"""
import sys, time
import numpy as np
import bayenv

data = sys.argv[1] if len(sys.argv) > 1 else "data"
sizes = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "10000,100000").split(",")]
for n in sizes:
    t = time.perf_counter(); env = bayenv.Env(data, n_cars=n, seed=11, accidents=True); setup = time.perf_counter() - t
    A = np.zeros((env.n_slots, 3), np.float32); A[:, 1] = 0.3; env.act(A)
    for _ in range(5): env.step(20)
    t = time.perf_counter()
    for _ in range(20): env.step(20)                       # 400 steps, batched (one call per 20 steps)
    batched = (time.perf_counter() - t) / 400 * 1e3
    obs = np.empty((env.n_slots, env.obs_size), np.float32)
    t = time.perf_counter()
    for _ in range(100):                                   # RL loop: act -> 1 step -> observe, every step
        env.act(A); env.step(1); env.observe(out=obs)
    loop = (time.perf_counter() - t) / 100 * 1e3
    print(f"n {n}: setup {setup:.2f} s, step(20) {batched:.3f} ms/step ({1000 / batched:.0f} steps/s), "
          f"act+step(1)+observe {loop:.3f} ms/step ({1000 / loop:.0f} steps/s)")
