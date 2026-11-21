"""Observation readback variants vs the default path: copy=False views, out=, step_async/observe_wait, lag=1 and VecEnv
must give bit-identical float32 observations; float16 error per field. Optional argv[2]: .npy of the default
sequence from an older build (python obs_check.py data [ref.npy] [vecref.npy]).
python rust/crates/bayenv-py/examples/obs_check.py data"""
import sys
import numpy as np
import bayenv

data = sys.argv[1] if len(sys.argv) > 1 else "data"
T = 40

def run(mode, n=10000):
    """The reference sequence: observe(), then step_observe with varying actions / k, a plain step(5) every 10."""
    env = bayenv.Env(data, n_cars=n, seed=11, accidents=True)
    A = np.zeros((env.n_slots, 3), np.float32); A[:, 1] = 0.3
    rng = np.random.default_rng(0)
    res = [np.array(env.observe(copy=(mode != "view")))]
    out = np.empty((env.n_slots, env.obs_size), np.float32)
    lagged = []
    for t in range(T):
        A[:, 0] = rng.uniform(-0.2, 0.2, env.n_slots).astype(np.float32)
        k = 1 + t % 3
        if mode == "default": res.append(env.step_observe(A, k))
        elif mode == "out": env.step_observe(A, k, out=out); res.append(out.copy())
        elif mode == "view":
            v = env.step_observe(A, k, copy=False); assert not v.flags.writeable; res.append(np.array(v))
        elif mode == "async": env.step_async(A, k); res.append(env.observe_wait())
        elif mode == "lag":   # returns the obs before this call's steps
            lagged.append(np.array(env.step_observe(A, k, lag=1, copy=False)))
        if t % 10 == 9:
            env.step(5); res.append(env.observe())
    if mode == "lag": return lagged, env
    return res

ref = run("default")
ok = True
for m in ["out", "view", "async"]:
    r = run(m); same = all(np.array_equal(a.view(np.uint32), b.view(np.uint32)) for a, b in zip(ref, r))
    print(f"{m:8s} bit-identical: {same}"); ok &= same
# lag=1: call t returns the obs as of its start
lagged, env = run("lag")
exp = []   # expected lagged[t]: obs at the start of call t in the default sequence
i = 0
for t in range(T):
    exp.append(ref[i]); i += 1
    if t % 10 == 9: i += 1   # step(5) + observe entry
same = all(np.array_equal(a.view(np.uint32), b.view(np.uint32)) for a, b in zip(exp, lagged))
same &= np.array_equal(env.observe_wait().view(np.uint32), ref[-1].view(np.uint32))
print(f"lag=1    bit-identical (shifted by one call): {same}"); ok &= same
if len(sys.argv) > 2:
    old = np.load(sys.argv[2]); same = np.array_equal(old.view(np.uint32), np.stack(ref).view(np.uint32))
    print(f"default  bit-identical with {sys.argv[2]}: {same}"); ok &= same
# VecEnv: default / view / lag / async
def vrun(mode):
    v = bayenv.VecEnv(data, n_envs=3, n_cars=3000, seed=11, accidents=True)
    A = np.zeros((3, v.n_slots, 3), np.float32); A[..., 1] = 0.3
    rng = np.random.default_rng(1); res = []
    for t in range(20):
        A[..., 0] = rng.uniform(-0.2, 0.2, A.shape[:2]).astype(np.float32)
        if mode == "default": res.append(np.array(v.step_observe(A, 1 + t % 2)))
        elif mode == "view": res.append(np.array(v.step_observe(A, 1 + t % 2, copy=False)))
        elif mode == "async": v.step_async(A, 1 + t % 2); res.append(v.observe_wait())
        elif mode == "lag": res.append(np.array(v.step_observe(A, 1 + t % 2, lag=1)))
        if t % 7 == 6:
            if mode == "lag": res.append(v.observe_wait())
            v.step(3)
    if mode == "lag": res.append(v.observe_wait())
    return np.stack(res)
vref = vrun("default")
for m in ["view", "async"]:
    same = np.array_equal(vrun(m).view(np.uint32), vref.view(np.uint32)); print(f"VecEnv {m:6s} bit-identical: {same}"); ok &= same
lg = vrun("lag")
# lg: per call t the obs at its start, plus observe_wait() (= end of call t) before every step(3) and at the end.
# The start of call t (t > 0, no step(3) right before) is the end of call t - 1.
ends, j = [], 0
for t in range(20):
    j += 1
    if t % 7 == 6: ends.append((t, j)); j += 1
same = all(np.array_equal(lg[j].view(np.uint32), vref[t].view(np.uint32)) for t, j in ends)
same &= np.array_equal(lg[-1].view(np.uint32), vref[-1].view(np.uint32))
same &= all(np.array_equal(lg[t + 1 + t // 7].view(np.uint32), vref[t].view(np.uint32)) for t in range(19) if t % 7 != 6)
print(f"VecEnv lag=1  bit-identical (shifted): {same}"); ok &= same
if len(sys.argv) > 3:
    old = np.load(sys.argv[3]); same = np.array_equal(old.view(np.uint32), vref.view(np.uint32))
    print(f"VecEnv default bit-identical with {sys.argv[3]}: {same}"); ok &= same

# float16: error per field over a real run (100 x 10 steps of 10k cars, gas 0.3, small random steering)
env = bayenv.Env(data, n_cars=10000, seed=11, accidents=True)
A = np.zeros((env.n_slots, 3), np.float32); A[:, 1] = 0.3
rng = np.random.default_rng(2)
mx = np.zeros(env.obs_size); sm = np.zeros(env.obs_size); rel = np.zeros(env.obs_size); cnt = 0; amax = np.zeros(env.obs_size)
for t in range(100):
    A[:, 0] = rng.uniform(-0.2, 0.2, env.n_slots).astype(np.float32)
    env.step_observe(A, 10)
    f = env.observe(); h = env.observe(dtype="float16"); assert h.dtype == np.float16
    e = np.abs(h.astype(np.float32) - f)
    mx = np.maximum(mx, e.max(0)); sm += e.sum(0); cnt += len(f); amax = np.maximum(amax, np.abs(f).max(0))
    rel = np.maximum(rel, (e / np.maximum(np.abs(f), 1e-30)).max(0))
    exp16 = f.clip(-65504, 65504).astype(np.float16)
    assert np.array_equal(exp16.view(np.uint16), h.view(np.uint16)), "GPU f16 != numpy round-to-nearest"
print("float16 == numpy float32->float16 rounding: True")
print("field  max|x|      max abs err  mean abs err  max rel err")
for i in range(env.obs_size): print(f"{i:5d}  {amax[i]:10.4g}  {mx[i]:11.3g}  {sm[i] / cnt:12.3g}  {rel[i]:11.3g}")
print(f"all fields: max abs err {mx.max():.3g} (field {mx.argmax()}), mean abs err {sm.sum() / cnt / env.obs_size:.3g}, max rel err {rel.max():.3g}")
print("OK" if ok else "MISMATCH"); sys.exit(0 if ok else 1)
