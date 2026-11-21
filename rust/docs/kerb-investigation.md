# Kerb strikes: CPU vs GPU (investigation)

> **Status:** the fix is the default on `rust` (`GpuOptions.cpu_connector_join = true`, Python
> `cpu_connector_join=True`). `false` is the legacy mode that reproduces the JS GPU bit-for-bit, and every JS-oracle
> check runs in it. **Legacy GPU runs and all earlier GPU checkpoints were trained with the bug described below.**
> `crates/bayenv-gpu/tests/cpu_vs_gpu.rs` (in `check.sh`) asserts CPU / GPU aggregate agreement on the default model.

**Problem.** On the `compare_backends.py` scenario (5,000 commuters, seed 42, from 07:00, dt 0.05, k = 1, every
driving car commanded [steer 0, gas 0.3, brake 0], position-mode steering) the GPU logged 1,394 contact rows over
1,200 steps, 1,388 of them barrier (kerb) strikes. The CPU logged 17, 12 of them barrier strikes. The CPU model is
the reference.

**Root cause.** A JS-GPU porting bug in the junction connector entry in `integrate.wgsl`. It is not an intended
approximation. When a car enters the connector of its movement, `src/sim/vehicle.js` re-projects it onto the
connector, joining at the point nearest its body and adding the heading difference to `headingRel`. It does this only
when the car was already past the entry point *before* the step: `sArrive > sIn + 0.3`, where `sArrive` is the
start-of-step `s` or the point where the car arrived on the edge. This is the case of back-to-back junction boxes on
a short edge. The kernel tests the end-of-step `s` instead (`s > sIn + 0.3`). Above 6 m/s a car moves more than
0.3 m per step, so on the GPU almost every ordinary box entry takes the re-projection branch.

The branch keeps the car's world heading and sets `headingRel += dh`, with `|dh|` up to about 0.09 rad. Here `dh` is
the angle between the connector tangent at the join point and the edge direction. The yaw rate stays 0 and the
steering stays 0, so nothing in the dynamics ever removes the offset. The road frame bends with the connector and the
out-edge, and `headingRel` is relative to that frame. The car therefore keeps a constant heading error and drifts
sideways at `u sin(dh)`, up to 2 m/s at 24 m/s, until it hits the kerb. The barrier then bounces it back, which
explains about 5 strikes per car at a low Δv.

| | JS CPU (`src/sim/vehicle.js`) | JS GPU (`src/gpu/kernels/integrate.wgsl`) | Rust CPU (`bayenv-sim/src/vehicle.rs`) |
|---|---|---|---|
| start-of-step `s` | 318 `let sArrive = s0` | 157 `let s0 = v.s` (never used for this test) | 393 |
| connector exit → out-edge | 336 `sArrive = J.mF[4 * m + 1]` | 188 (nothing) | 411 |
| the test | 368 `if (sArrive > sIn + 0.3 && e.LG)` | **228 `if (s > sIn + 0.3 && P.hasLG != 0u)`** | 447 |
| heading update | 385 `this.headingRel[i] += dh` | 244 `v.headingRel = v.headingRel + dh` | |
| edge hop | 438 `sArrive = 0` | 288 (nothing) | 521 |

The kernel's own comment states the CPU intent ("already past the entry point (back-to-back boxes on a short edge)"),
so the condition is a bug, not a deliberate GPU simplification.

## Ruled out

* **Vehicle dynamics**: `physics.wgsl` is line for line `dynamics.js`. This covers the dynamic bicycle model,
  10 ms sub-steps, the steering actuator lag and slew, the friction circle and the kinematic blend.
* **Reaction ring**: `act.wgsl` and `integrate.wgsl (d)` follow `vehicle.js integrate`.
* **Strict-hop `mapLat`**, the barrier geometry (`contacts.wgsl barrier` vs `sim.js _barrier`) and the strike
  reporting (F_WALL / F_WALL_PREV vs `_barrierTouch`).
* **Road-frame integration**: on both backends `headingRel` is relative to the lane direction, and the frame absorbs
  the road curvature on straight edges, curved edges and connectors alike. (In the JS models. The later `curvature_dynamics` model fix, off by default for now, stops
  the frame turning the car: `docs/physics-audit.md`.)

With the patched connector entry, a single car on the GPU matches the CPU to f32 noise over 19 edges and 3,000 steps
(next section), so none of the terms above differ.

## Evidence

### Single car (Rust CPU, bit-exact with the JS CPU, vs Rust GPU)

The setup is `bayenv.Env(n_cars=1, seed=42, backend="cpu")` against `population=1, pop_seed=42` on the GPU, with the
same start time (30,000 s). The car spawns at step 1,667 and is commanded [0, 0.3, 0]. The GPU rows of non-driving
slots get the CPU default action [0, 0, 0.3]; see the harness note below. The trace columns are step, edge, s, lat,
headingRel, yawRate, vLat, steerAngle, speed and the world heading.

```
C 2057 18147 150.611 1.75 0           0 0 0 24.4573 1.71705
G 2057 18147 150.612 1.75 0           0 0 0 24.4573 1.71705
C 2058 18147 151.835 1.75 0           0 0 0 24.5015 1.71247   <- CPU: enters the connector, frame turns with it
G 2058 18147 151.825 1.75124 -0.00450289 0 0 0 24.5015 1.71705 <- GPU: re-projected, keeps the old world heading
C 2059 18147 153.061 1.75 0           0 0 0 24.5288 1.70203
G 2059 18147 153.051 1.74572 -0.00450289 0 0 0 24.5288 1.70663 <- constant offset, lat drifts
C 2060 18147 154.288 1.75 0           0 0 0 24.556  1.68948
G 2060 18147 154.279 1.7402  -0.00450289 0 0 0 24.5561 1.69408
```

The first divergence is at step 2,058, at the first junction box entry. Up to that step the two traces agree to
1e-3 m. They cover edge 18149, whose world heading changes by 0.22 rad, and an edge hop. After the divergence the GPU
car strikes the kerb at steps 2,473, 2,577 and 2,865 (Δv 1.5, 8.9 and 2.6 km/h) and is disabled at step 2,886
(10.6 km/h). The CPU car has no contact.

With `cpu_connector_join` on, the traces over 3,000 steps and 19 edges match as follows. The edges include the
straight 4727 (0.002 rad over 182 m) and the curved 18145 (1.03 rad over 345 m), plus box connectors.

* `lat`, `headingRel`, `yawRate`, `vLat` and `steerAngle` are identical.
* `|Δs| ≤ 9.6e-4 m` and `|Δspeed| ≤ 8.4e-5 m/s`.
* The events are identical: spawn, and signal violations at steps 2,058, 2,293, 2,975 and 2,996.

### JS GPU in Chrome = Rust GPU (flag off)

The JS GPU was run with `rust/tools/oracle/gpu_kerb_trace.html` in Chrome / Metal. The scenario is the
`compare_parity.sh` pop pack: 20,000 commuters, seed 42, from 07:00, capacity 32,768, gas 0.3, 1,200 steps. A frozen
heading offset is `|headingRel| > 1e-3` at yaw rate 0 and steer 0. The JS CPU column comes from
`tools/rust/trace.mjs buildEnv` with world `real`, n 20,000, seed 42 and the `const` policy.

| after step | JS-GPU frozen / at kerb | Rust GPU off | Rust GPU `cpu_connector_join` | JS CPU frozen / at kerb |
|---|---|---|---|---|
| 300 | 614 / 65 | 613 / 64 | 3 / 5 | 5 / 7 |
| 600 | 945 / 281 | 945 / 282 | 3 / 6 | 16 / 18 |
| 900 | 472 / 283 | 472 / 273 | 5 / 4 | 19 / 21 |
| 1200 | 231 / 243 | 231 / 239 | 2 / 7 | 31 / 22 |

Excerpt from the same run: the same car (slot 1) in the JS GPU, the Rust GPU and the Rust GPU with the patch, at a box
entry.

```
JS-GPU      577 255965 366.695 1.49527 -0.0851789 0 0 0 24.1088
RS-GPU off  577 255965 366.695 1.49612 -0.0851789 0 0 0 24.1088
RS-GPU cj   577 255965 367.076 1.55     0         0 0 0 24.1088
JS-GPU      578 255965 367.897 1.39264 -0.0851789 0 0 0 24.1371   (lat -0.1 m per step = 2 m/s)
```

### compare_backends.py (5,000 commuters, 1,200 steps, from 07:00)

| event (CPU / GPU) | legacy GPU (`--legacy-gpu`) | legacy, `--spawn-default` | fixed GPU (now the default) | fixed, `--spawn-default` |
|---|---|---|---|---|
| contact rows | 17 / 1,394 | 17 / 1,336 | 17 / 10 | 17 / 12 |
| crashed (car disabled) | 28 / 41 (13 common) | 28 / 39 | 28 / 31 (27 common) | 28 / 27 (27 common, 26 on the same step) |
| missed exit | 120 / 86 (74 common) | 120 / 91 | 120 / 123 (116 common) | 120 / 123 (116 common, 100 within 20 steps) |
| signal / stop violation | 1,627 / 1,592 | 1,627 / 1,596 | 1,627 / 1,625 | 1,627 / 1,627 (657 on the same step) |
| arrived, late_by | 4 / 4, max 1.6 s | | 4 / 4, max 0.15 s | 4 / 4, identical |

20,000 commuters, 1,200 steps (`parity_run`, pop pack plus 40 obstacles): crash events go from 5,693 to 320 and
crash scenes from 317 to 229.

**Harness note (`--spawn-default`).** The CPU applies `act()` rows only to cars that are driving when `act()` is
called. A car that spawns during the step therefore takes the sim's default action [0, 0, 0.3] on its first decision.
The GPU holds the slot's row, gas 0.3, from the first step. The CPU trajectories are shifted by about 2 steps, which
is why "first step equal" was near 0 for crashes and missed exits. This is a difference in the env API, not in the
physics. `--spawn-default` gives non-driving GPU slots the CPU default action.

The missed exits still differ in their reported step: 15 of 116 are on the same step, 100 within 20 steps. The GPU
emits a missed exit at the connector join, while the CPU emits it in `_missedExits` after the integration. This was
not investigated further.

## Fix (originally flagged, default off; now the default)

`GpuOptions.cpu_connector_join` applies the text patch `crates/bayenv-gpu/kernels/cpu_connector_join.wgsl` to the
integrate kernel. It is on by default. Legacy mode is `GpuOptions { cpu_connector_join: false, .. }`, Python
`bayenv.Env(..., cpu_connector_join=False)` / `VecEnv(..., cpu_connector_join=False)`, or
`BAYENV_GPU_OPTS=cpu_connector_join=0` in the examples (the JS-oracle examples default to legacy). The patch adds `var sArr = s0`, sets `sArr = movSOut` at a
connector exit and `sArr = 0` after an edge hop, and tests `sArr > sIn + 0.3`. This is the CPU's `sArrive`.
`kernels::patch_text` asserts that every original snippet occurs exactly once, and a unit test in `kernels.rs`
checks the same.

With the flag off (legacy mode), the kernels are the JS kernels. A Python state dump (veh + pose after 400 steps, scatter of 10,000,
seed 11) is byte-identical between the pre-change build and this build, and the run is deterministic. The 20k
population run is not deterministic run to run (contact atomics), so it is not used for this check.

Recommendation: fix the JS kernel upstream with the same four lines in `src/gpu/kernels/integrate.wgsl`. The Rust
GPU now uses the fix by default, so new training sees the CPU's connector entry. Policies trained on legacy runs saw the
drift and should be fine-tuned on the default model.
