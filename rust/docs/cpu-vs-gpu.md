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
  the run, at a median Δv of 2 km/h. The CPU does this rarely. **Root cause: a JS-GPU bug** in the junction connector
  entry (`integrate.wgsl:228` tests the end-of-step `s`, `vehicle.js:368` the start-of-step `sArrive`), which gives
  almost every car entering a junction box a constant heading offset of up to 0.09 rad. See `kerb-investigation.md`.
  `GpuOptions.cpu_connector_join` / `Env(..., cpu_connector_join=True)` (default off) fixes it. With the fix the
  numbers are: contacts 17 / 10, crashed 28 / 31 (27 common), missed exits 120 / 123 (116 common), violations
  1,627 / 1,625. With `compare_backends.py --spawn-default` as well, crashed is 28 / 27 (26 on the same step) and
  violations 1,627 / 1,627.
* **Crashes, missed exits and violations** depend on where the cars drift, so their counts and timing diverge with
  the trajectories (same order of magnitude, different agents).
* For training, treat the backends as two related environments. Compare policies within one backend. Use the
  harness when you need to check that a reward or metric behaves the same on both.

## Crash divergence dumps (CPU half: `rust/tools/cpu_divergence.py`)

`python rust/tools/cpu_divergence.py data [n=5000] [steps=3000] [start=25200] [seed=42] [--agents a,b,c] [--k 3] [--window 50]`
runs the `compare_backends.py --cpu-connector-join --spawn-default` setup on both backends, picks the first k agents
that are CRASHED (event kind 2) on the CPU only, reruns the CPU and writes `rust/oracle/divergence/cpu-<agent>.jsonl`:
one JSON object per step for the `window` steps before the agent's crash step and the crash step itself, read from
`env.state()` after `env.step(1)` (the state at the END of that step), for the slot holding that agent:

| key | meaning |
|---|---|
| step | 1-based global step index (= events `step`) |
| time | sim time (s) at the end of the step |
| agent, slot | 2 * commuter + leg; slot index (slots differ between backends: match by agent) |
| edge, s, route_pos | current directed edge, m along it, index in the route |
| lane, lane_lo, lane_hi | lane under the centre; footprint lanes (lane_hi < lane_lo = off the lanes) |
| lat, heading_rel | m from the right carriageway edge (+ left); rad heading minus lane direction (+ left) |
| x, y, z, heading | world pose (heading = atan2(dz, dx)) |
| speed, v_lat, yaw_rate | body u (m/s), body v (+ left), r (rad/s, + left) |
| steer, gas, brake | road-wheel angle (rad), applied (reaction-delayed) pedals |
| leader, gap | leader slot (-1 none, -2 lane end, -3 junction conflict point) and bumper gap (m); a slot number is a slot of the same backend |
| ctl_type, ctl_state, ctl_dist | next traffic control: -1 none / 0 signal / 1 stop / 2 give way / 3 ramp meter / 4 crossing; 0 green / 1 yellow / 2 red / 3 sign / 4 dark; m to its stop line |
| flags, state, conn | vehicle flags (bit5 DISABLED, bit1 COLLIDED, ...), 1 driving / 2 crashed, junction connector (-1 none) |

Non-finite floats are written as null. The GPU half should write `gpu-<agent>.jsonl` with the same keys and steps.
