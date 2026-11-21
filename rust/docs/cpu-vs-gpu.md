# CPU vs GPU backend: expected differences

The two backends implement the same model twice. The CPU sim is `src/sim` in JS and `bayenv-sim` in Rust, bit-exact
with each other. The GPU sim is `src/gpu` in JS (WGSL) and `bayenv-gpu` in Rust (the same WGSL through wgpu).

The Rust GPU defaults to `GpuOptions.cpu_connector_join = true` and `contact_tangent = true`. These fix two bugs in
the JS GPU kernels: the junction connector entry (`kerb-investigation.md`) and the f32 friction tangent of the contact
impulse (`crash-divergence.md`). With both `false` (legacy mode), the Rust GPU runs the JS kernels and matches the JS
GPU within float tolerance, including the bugs.

**Legacy GPU runs and every GPU checkpoint trained before this default were trained with the bug.** In those runs,
almost every car entering a junction box got a constant heading offset of up to 0.09 rad. A car that does not steer
against the offset drifts into the kerb. Those policies learned to correct a drift that the CPU model, and now the
default GPU model, does not have. The contact-tangent bug affects only crash outcomes: aligned rear-end wrecks spin
to the kerb instead of stopping in their lane.

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
| signal / stop violation | 1,627 / 1,626, 659 on the same step | 1,627 / 1,625 | 1,627 / 1,592 |

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
* **Crashes on one backend only.** Over 3,000 steps with `--spawn-default`, 9 agents crash on the CPU only and 2 on
  the GPU only. There are two deterministic causes, described in `crash-divergence.md`.
  (A) After a missed turn, the CPU re-plans one step later than the GPU. It then joins the next junction box
  re-projected, with a heading error of about 0.2 rad. This is a CPU artefact, and no GPU flag is provided.
  `missed_at_out_edge` changes only when the GPU reports the missed exit, not its route.
  (B) A **JS-GPU bug:** in an aligned rear-end, the f32 friction tangent of the contact impulse is rounding noise
  (`collidefns.wgsl:60`), and the wrecks spin to the kerb. `GpuOptions.contact_tangent` (default on;
  `Env(..., contact_tangent=False)` is the legacy JS model) fixes it. Crashed agents in common go from 67 to 70
  (CPU 76, GPU 71).
* **For training,** the default GPU model is the CPU model. Compare policies trained on legacy GPU runs only with
  `cpu_connector_join=False, contact_tangent=False`, or retrain or fine-tune them on the default model.

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

Non-finite floats are written as null. The GPU half, `rust/tools/gpu_divergence.py data --agents a,b,c`, writes
`gpu-<agent>.jsonl` with the same keys and steps. The GPU `state()` has the same columns. With `--backend cpu`,
`--from` / `--to` and `--prefix`, the tool traces any agent on either backend. Results are in
`crash-divergence.md`.

## Missed-exit timing

The JS GPU kernel (`integrate.wgsl`, the `EV_MISSED` emit inside the junction-connector choice) reports a missed exit
through a junction when the car commits to the wrong connector. The JS CPU sim (`sim.js _missedExits`) reports it once
the car is on the out-edge it took, after the junction box. The Rust GPU host reproduces the JS GPU by default.
`GpuOptions.missed_at_out_edge` (Python `Env(missed_at_out_edge=True)`) delays these rows until the car is on the taken
edge, matching the CPU timing. The row's `step` is then the end of the step batch: exact with k = 1.

compare_backends `--spawn-default --missed-out-edge` (5,000 commuters, 3,000 steps, 07:00, seed 42):
missed_exit 603 vs 603 rows. Of the 386 agents both backends report, 383 are on the same step (142 without the option).
