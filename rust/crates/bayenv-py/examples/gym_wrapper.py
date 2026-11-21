"""Thin Gymnasium-style wrapper around bayenv.Env (all cars at once: one observation row and one action row per car).

    env = BayEnvGym("data", n_cars=1000, seed=1, k=5)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

The environment defines no reward: `reward` is all zeros (compute your own from `info["state"]`).
`terminated[i]` is True for a car that crashed (stopped wreck or disabled) or left the road (state != driving).
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import bayenv


class BayEnvGym(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, data_dir="data", n_cars=1000, seed=1, k=1, max_steps=12000, **env_kwargs):
        self.data_dir, self.n_cars, self.seed0, self.k, self.max_steps, self.kw = data_dir, n_cars, seed, k, max_steps, env_kwargs
        self.env = None
        self._make(seed)
        n, d = self.env.n_slots, self.env.obs_size
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n, d), dtype=np.float32)
        self.action_space = spaces.Box(np.array([[-1, 0, 0]] * n, np.float32), np.array([[1, 1, 1]] * n, np.float32), dtype=np.float32)
        self._obs = np.empty((n, d), np.float32)

    def _make(self, seed):
        self.env = bayenv.Env(self.data_dir, n_cars=self.n_cars, seed=seed, **self.kw)
        self.t = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._make(self.seed0 if seed is None else seed)
        self.env.step(1)                       # one step so the observation vectors are sensed
        return self.env.observe(out=self._obs).copy(), {"time": self.env.time}

    def step(self, action):
        self.env.act(np.ascontiguousarray(action, dtype=np.float32))
        self.env.step(self.k)                  # k sim steps per call, no Python per step
        self.t += self.k
        obs = self.env.observe(out=self._obs).copy()
        st = self.env.state()
        terminated = st["crashed"] | st["disabled"] | (st["state"] != 1)
        truncated = np.full(terminated.shape, self.t >= self.max_steps)
        return obs, np.zeros(len(obs), np.float32), terminated, truncated, {"state": st, "time": self.env.time}


if __name__ == "__main__":
    import sys
    e = BayEnvGym(sys.argv[1] if len(sys.argv) > 1 else "data", n_cars=1000, seed=1, k=5)
    obs, info = e.reset()
    for _ in range(20):
        a = np.zeros(e.action_space.shape, np.float32); a[:, 1] = 0.3
        obs, r, term, trunc, info = e.step(a)
    print(obs.shape, "mean speed", float(info["state"]["speed"].mean()), "terminated", int(term.sum()), "t", info["time"])
