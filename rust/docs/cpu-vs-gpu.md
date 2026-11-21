# CPU vs GPU backend: expected differences

The two backends implement the same model twice. The CPU sim is `src/sim` in JS and `bayenv-sim` in Rust, bit-exact
with each other. The GPU sim is `src/gpu` in JS (WGSL) and `bayenv-gpu` in Rust (the same WGSL through wgpu).

The Rust GPU defaults to `GpuOptions.cpu_connector_join = true`. This fixes a bug in the JS GPU kernel's junction
connector entry, described in `kerb-investigation.md`. With `false` (legacy mode), the Rust GPU runs the JS kernels
and matches the JS GPU within float tolerance, including the bug.

**Legacy GPU runs and every GPU checkpoint trained before this default were trained with the bug.** In those runs,
almost every car entering a junction box got a constant heading offset of up to 0.09 rad. A car that does not steer
against the offset drifts into the kerb. Those policies learned to correct a drift that the CPU model, and now the
default GPU model, does not have.

Measured with `rust/tools/compare_backends.py data 5000 1200 25200 42 [--legacy-gpu] [--spawn-default]`. The setup is
5,000 commuters from population.js, seed 42, from 07:00, dt 0.05, k = 1, with every driving car commanded
[steer 0, gas 0.3, brake 0]. Rows are matched by agent.

| event (CPU / GPU) | default GPU, `--spawn-default` | default GPU | legacy GPU (`--legacy-gpu`) |
|---|---|---|---|
| spawned | 834 / 834 | 834 / 834 | 834 / 834 |
| arrived | 4 / 4, late_by identical | 4 / 4, max 0.15 s | 4 / 4, max 1.6 s |
| crashed (car disabled) | 28 / 27, 27 common, 26 on the same step | 28 / 31, 27 common | 28 / 41, 13 common |
| contact rows | 17 / 12 | 17 / 10 | 17 / 1,394 (1,388 barrier strikes) |
| missed exit | 120 / 123, 116 common | 120 / 123, 116 common | 120 / 86, 74 common |
| signal / stop violation | 1,627 / 1,627, 657 on the same step | 1,627 / 1,625 | 1,627 / 1,592 |

`crates/bayenv-gpu/tests/cpu_vs_gpu.rs` runs the first column as a test, so it is part of `check.sh`. It asserts
aggregate agreement within tolerances around these values.

What this means:
* **Spawning and the trip schedule are identical:** same population, same departure step.
* **`--spawn-default`** only changes the harness. The CPU applies `act()` rows only to cars that are driving when it
  is called, so a car that spawns during a step brakes (the default action) on its first decision. The GPU uses the
  slot's held row. This shifts trajectories by about 2 steps, but not the aggregates.
* **Remaining differences** are f32 vs f64 arithmetic and the order of contact resolution. Missed exits are also
  reported at a different point: the GPU at the connector join, the CPU in `_missedExits`. Contacts amplify these.
  With the default GPU model, a single car matches the CPU to 1e-3 m over 3,000 steps.
* **For training,** the default GPU model is the CPU model. Compare policies trained on legacy GPU runs only with
  `cpu_connector_join=False`, or retrain or fine-tune them on the default model.

## Crash divergence dumps (CPU half: `rust/tools/cpu_divergence.py`)

`python rust/tools/cpu_divergence.py data [n=5000] [steps=3000] [start=25200] [seed=42] [--agents a,b,c] [--k 3] [--window 50]`
runs the `compare_backends.py --spawn-default` setup (default GPU model, `cpu_connector_join=True`) on both backends, picks the first k agents
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
