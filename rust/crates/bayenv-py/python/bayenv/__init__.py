"""Bay Area driving environment (native GPU sim via wgpu; CPU sim when bayenv-sim is merged).

    import bayenv
    env = bayenv.Env("data", n_cars=10000, seed=11)
    env.act(actions)       # float32 (N, 3): steer, gas, brake
    env.step(20)           # 20 steps, one GPU submit
    obs = env.observe()    # float32 (n_slots, 70)
"""
from ._bayenv import Env, VecEnv, world_info, OBS_V2_SIZE

__all__ = ["Env", "VecEnv", "world_info", "OBS_V2_SIZE"]
