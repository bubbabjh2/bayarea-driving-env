# Crashes on the CPU only: CPU vs GPU (investigation)

**Problem.** The scenario is `compare_backends.py --spawn-default`: 5,000 commuters, seed 42, from 07:00, dt 0.05,
k = 1, 3,000 steps, with every driving car commanded [steer 0, gas 0.3, brake 0]. The GPU runs with
`cpu_connector_join`, which is now the default. In this scenario, 9 agents crash (event kind 2) on the CPU but not on
the GPU. The first three, agents 4114, 8318 and 8956 (CPU crash at steps
1,162, 1,257 and 1,353), were traced. The CPU model is the reference.

**Result.** There are two causes. Neither is chaos from earlier float differences: up to the first divergence each
car matches to f32 noise, and each cause is a deterministic mechanism that fires again wherever its trigger occurs.

| agent | first divergence (step, field: CPU / GPU) | cause |
|---|---|---|
| 4114 | 1,142 `conn` −1 / 114059: the GPU enters the next box in the step it leaves box 95393, the CPU does not; 1,143 `heading_rel` −0.239 / 0, `s` 8.287 / 9.571, `lat` 4.919 / 5.400 | **A**, missed-turn re-plan timing (CPU artefact) |
| 8318 | 1,207 `ctl_dist` 265.8 / 385.8 (the CPU still follows its stale route); 1,218 `conn` −1 / 187642; 1,219 `heading_rel` −0.255 / 0 | **A** |
| 8956 | 1,303 `leader` slot 633 / 76, `gap` 80.5 / 30.2. The cars ahead are the wrecks of agents 6748 and 9682: in lane 1 on the CPU, against the right kerb on the GPU. Their first divergence is step 783 (the crash), `yaw_rate` 0.000 / −0.985, `v_lat` 0.000 / 0.921, Δv 17.3 / 19.6 km/h | **B**, f32 friction tangent in the contact impulse (GPU bug) |

Of the other CPU-only crashes, 4060 (dh −0.199 rad at step 1,702) and 656 (dh −0.211 rad at step 1,049, kerb
strikes from step 1,079, crash at step 1,972) are also cause A. The pair 4762 / 2920 was not traced.

## A. Missed turn: the CPU re-plans one step late, the late box entry is re-projected (CPU artefact, no GPU fix)

At a junction box, a car that is in the wrong lane for its turn takes the movement its lane leads to (a missed turn).
Both backends patch the next route entry to the out-edge `took`. They differ in when the rest of the route is
re-planned:

* **CPU:** `_missedExits` re-plans only when the car is on `took`, after `_integrate`. In the step in which the car
  leaves the connector onto `took`, its route after `took` is still the old plan. `J.movement(took, lane,
  route[next])` therefore finds no movement, and the car does not enter the next box, even though it is already
  past that box's entry point `sIn`. The re-plan follows at the end of the step. In the next step the car joins the
  box with `sArrive` = the start-of-step `s`, which is more than `sIn + 0.3`. This takes the "already past the entry
  point" branch: the car joins at the point nearest its body and keeps its world heading, so
  `headingRel += dh`. Here `|dh|` is 0.20–0.26 rad, the turn of the connector over the distance the car has already
  covered. With steer 0 and yaw rate 0, nothing removes the offset. The road frame turns with the connector, so the
  car keeps a constant heading error and drifts sideways at `u sin(dh)`, about 8 m/s at 34 m/s. It reaches the kerb
  about 20 steps later (Δv ≥ 10 km/h, disabled).
* **GPU:** the host re-plans as soon as the kernel emits `EV_MISSED` at the connector entry. The new route starts
  with the in-edge (`prefix`), so it is in place one step later (4114: missed turn at step 1,132, `route_pos` 0 on
  the in-edge at step 1,133). At the connector exit the integrate loop continues straight into the next box in the
  same step, with `sArr` = `movSOut` ≤ `sIn + 0.3`. There is no re-projection, and `heading_rel` stays 0.

The trace for 4114 is below. The columns are step, edge, s, lat, headingRel, route_pos and conn.

```
C 1141 24835 105.047 5.400  0      14 95393   (both in box 95393, route_pos differs: the GPU route was re-planned at 1133)
G 1141 24835 105.047 5.400  0       0 95393
C 1142 40823   7.848 5.400  0       0    -1   <- CPU: onto took, no next movement (stale route); re-plan after integrate
G 1142 40823   7.849 5.400  0       1 114059  <- GPU: chains into the next box in the same step
C 1143 40823   8.287 4.919 -0.2390  0 114059  <- CPU: joins one step late, re-projected: s back 1.3 m, dh = -0.239
G 1143 40823   9.571 5.400  0       1 114059
C 1162  7059  10.133 0.160 -0.120 crashed (kerb)          G 1162 40873 14.401 5.400 0 driving
```

| | JS CPU | JS GPU | Rust |
|---|---|---|---|
| missed turn: patch `route[next] = took` | `src/sim/vehicle.js:360` (+ `this.missed`) | `src/gpu/kernels/integrate.wgsl:220`, `emit(EV_MISSED)` 221 | `bayenv-sim/src/vehicle.rs:546` |
| re-plan | `src/sim/sim.js:1020` (only on `took`), `:1029`, called after `_integrate` at `:262-263` | `src/gpu/gpuSim.js:567-570` (queued at the event, `prefix` = in-edge), applied in `_applyReroute` `:1492` | `bayenv-sim/src/extra.rs:165`; `bayenv-gpu/src/sim.rs:913` |
| next box lookup on the out-edge | `vehicle.js:343` `J.movement(e.idx, la, route[next])` | `integrate.wgsl:203` | |
| connector exit sets the arrival point | `vehicle.js:336` `sArrive = J.mF[4 * m + 1]` | `integrate.wgsl:188` (+ `cpu_connector_join` `sArr`) | |
| re-projection test, heading | `vehicle.js:368` `sArrive > sIn + 0.3`, `:385` `headingRel += dh` | `integrate.wgsl:228`, `:244` | `vehicle.rs:556` |

This is a model difference between the backends, not a GPU bug. The GPU does what the CPU would do if its route were
current. The CPU behaviour is an artefact: the re-projection branch is meant for back-to-back boxes on a short edge,
and here it fires only because the route is one step stale. The same timing difference explains why missed exits
agree on the step in only 142 of 386 common agents: the GPU reports a missed exit at the connector entry, the CPU when
the car reaches `took`. `GpuOptions.missed_at_out_edge` aligns only the reported step, not the route re-plan, so it does not
change cause A. A GPU flag that copies the CPU's late re-plan would reproduce an artefact that causes crashes,
so none is provided. The suggested fix belongs in the CPU, in `src/sim` and `bayenv-sim`, which are out of scope
here. Either re-plan at the connector exit, inside `integrate`, before the next-movement lookup, or keep the
connector's `sOut` as `sArrive` for the first box entry after a re-plan.

## B. Aligned rear-end: an f32 rounding "tangent" adds a full friction impulse (GPU bug, flagged fix)

Agents 6748 (rear) and 9682 (front) drive identically on both backends up to step 782 (`|Δs|` ≤ 6e-4 m). At step
783 they collide at 38.4 and 30.4 m/s, both at the lane centre (`lat` 5.4, `heading_rel` 0, yaw 0). This is a
perfectly collinear rear-end. The difference is in `resolveContact`:

* **collide.js** (f64): the tangent `t = rv − vn·n` has `|t|` ≈ 1e-15, below 1e-6, so there is no friction impulse.
  Δv is 17.3 km/h for each car, which is `(1 + e)/2 · 8.03 m/s`, and the yaw stays 0. Both wrecks slide straight on
  locked wheels and stop in lane 1 at `s` 245.9 / 253.4.
* **collidefns.wgsl** (f32): with `|rv|` ≈ 8 m/s and velocities of about 38 m/s (ulp ≈ 4e-6), the subtraction
  cancels to rounding noise of order 1e-6. This can pass the `tl > 1e-6` test. The normalised "tangent" then points
  in a noise direction with a large component along `n`. A replay of the step-782 state in numpy f32 with the
  threshold just below the noise gives `t·n` = −0.31. Then `rv·t` is a sizeable fraction of `vn`, `jt` saturates at
  `μ·jn`, and both cars get a sideways impulse and a same-sign yaw. The kernel gives Δv 19.6 km/h and yaw −0.98 rad/s
  for both cars. The wrecks spin on locked wheels to the right kerb (`lat` 0.03 / 0.65, headings 3.05 / −1.07 rad)
  and leave lane 1 clear. On the GPU, agent 8956 passes them. On the CPU it hits the wreck of 6748 at step 1,353.

| | JS CPU `src/sim/physics/collide.js` | JS GPU `src/gpu/kernels/collidefns.wgsl` |
|---|---|---|
| tangent | 87 `let tx = rvx - vn * c.nx, ...` (f64) | 57 `var tx = rvx - vn * c.nx; ...` (f32) |
| test | 90 `if (tl > 1e-6)` | **60 `if (tl > 1e-6)`**: an absolute threshold below the f32 noise of `rv` |
| friction clamp | 95 `jt` to `±mu * jn` | 64 |

The kernel comment says it is "1:1 with collide.js", but an absolute 1e-6 threshold does not carry over to f32. This
is a port bug, deterministic for every aligned contact at speed, not chaos.

### Fix (flagged, default off): `GpuOptions.contact_tangent`

`crates/bayenv-gpu/kernels/contact_tangent.wgsl`, applied by `kernels::model_patches` to the `contacts` kernels and to
`kernels/resolve_fused.wgsl`, makes two changes. It removes the normal component of `t` again, so `t` is exactly
orthogonal to `n`. And it treats a slip below `max(1e-6, 1e-5·|rv|)` as no slip. In f64 the CPU's friction impulse
for such a slip is at most `|t|/kT`, which is negligible, so the patch agrees with collide.js. Ways to enable it:
Python `Env(..., contact_tangent=True)`, `VecEnv(..., contact_tangent=True)`, `BAYENV_GPU_OPTS=contact_tangent=1`,
or `compare_backends.py --contact-tangent`.

With the patch, the step-783 crash of 6748 / 9682 gives yaw 0.003 rad/s and speeds after the impulse equal to the
CPU's. The wrecks stop in lane 1 at `s` 245.859 / 253.425 (CPU: 245.857 / 253.425). Agent 8956 hits the 6748 wreck
at step 1,352 (CPU: 1,353).

`compare_backends.py data 5000 3000 25200 42 --spawn-default` (default GPU model), without and with
`--contact-tangent`:

| event (CPU / GPU) | off | `contact_tangent` |
|---|---|---|
| crashed | 76 / 69, 67 common, 64 on the same step | 76 / 71, **70 common**, 65 on the same step |
| contact rows | 129 / 116, 49 common agents | 129 / 117, **53 common agents** |
| missed exit | 603 / 605 | 603 / 604 |
| violations | 3,742 / 3,745 | 3,742 / 3,741 |
| arrived | 128 / 129, 127 common | unchanged |

CPU-only crashes go from 9 to 6, and GPU-only crashes from 2 to 1. Four of the remaining six are cause A (4114,
8318, 4060, 656).

With the flag off, the kernel sources are the JS kernels unchanged; the unit test `kernels::tests` asserts this. The
dumps `gpu-{4114,8318,8956}` and `gpu-q-{6748,9682}` from a build without the patch are byte-identical to those from
this build.

## Dumps (`rust/oracle/divergence/`)

`rust/tools/gpu_divergence.py` writes these files: the GPU half, and with `--backend cpu` any CPU trace. The keys are
those of `cpu_divergence.py`. Agents are mapped to slots by each backend's own spawn rows. The GPU columns lane_lo /
lane_hi, leader, gap, conn, ctl_type / ctl_state / ctl_dist, commuter / leg and agent come from `Env.state()` on the
GPU branch. They are read from the veh record (`lanes` bits, `leader`, `gap`, `taper` − 1 = connector) and from the
ctl record (words 1, 2 and 5).

| files | content |
|---|---|
| `cpu-` / `gpu-{4114,8318,8956}` | the 50 steps before the CPU crash and the crash step. The CPU files are byte-identical to the CPU agent's dumps |
| `cpu-` / `gpu-q-{6748,9682}` | the wreck pair ahead of 8956, steps 1–800 |
| `cpu-` / `gpu-p-{6748,9682,8956}` | steps 900–1,353 |
| `gpuct-{6748,9682,8956}` | GPU with `contact_tangent`, steps 780–1,353 |
| `cpu-r` / `gpuct-r-{4060,656}`, `cpu-s` / `gpuct-s-656` | cause A on two more agents (GPU with `contact_tangent`) |

`python rust/tools/gpu_divergence.py data --agents 4114,8318,8956 [--cpu-dir DIR]` reproduces the GPU half, and runs
in about 15 s.
