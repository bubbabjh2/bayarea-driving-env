# CPU vs GPU backend: expected differences

The two backends implement the same model twice. The CPU sim is `src/sim` in JS and `bayenv-sim` in Rust, bit-exact
with each other. The GPU sim is `src/gpu` in JS (WGSL) and `bayenv-gpu` in Rust (the same WGSL through wgpu), equal
within float tolerance. The CPU and GPU models are not equal to each other. The JS project's own CPU-vs-GPU check
(`test/gpu/equivalence.html`, synthetic graph) reaches 92.5 % of cars within 0.5 m after 1,000 steps. On the real
world with blind constant actions the differences grow faster, because crashes and kerb strikes amplify them.

Measured with `rust/tools/compare_backends.py data 5000 1200` (5,000 commuters from population.js, seed 42, from
07:00, dt 0.05, k = 1, every driving car commanded [steer 0, gas 0.3, brake 0], rows matched by agent):

| event | CPU | GPU | agree |
|---|---|---|---|
| spawned | 834 | 834 | 833 agents on the identical step, 834 within 20 steps |
| arrived | 4 | 4 | 3 common agents; late_by within 1.6 s |
| crashed (car disabled) | 28 | 41 | 13 common agents |
| contact rows | 17 | 1,394 | 1,388 of the GPU rows are barrier (kerb) strikes, vs 12 on the CPU |
| missed exit | 120 | 86 | 74 common agents |
| signal / stop violation | 1,627 | 1,592 | 655 common agents |

What this means:
* **Spawning and the trip schedule are identical:** same population, same departure step.
* **Kerb strikes** are the largest difference. Both report a strike once, when it starts. On the GPU, cars driving
  blind (steer 0) leave the lane, hit the kerb and bounce back far more often: about 5 separate strikes per car over
  the run, at a median Δv of 2 km/h. The CPU does this rarely. This is a dynamics difference of the GPU model
  (f32, 10 ms physics sub-steps, its own lateral integration in `integrate.wgsl` / `physics.wgsl`), not a repeated
  report.
* **Crashes, missed exits and violations** depend on where the cars drift, so their counts and timing diverge with
  the trajectories (same order of magnitude, different agents).
* For training, treat the backends as two related environments. Compare policies within one backend. Use the
  harness when you need to check that a reward or metric behaves the same on both.
