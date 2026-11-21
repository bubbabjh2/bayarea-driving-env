# `src/sim/physics/` — the single vehicle physics module

Owner: sim agent. Used by the sim (`src/sim/vehicle.js`), meant to be imported by the brain's
training sim (`src/brain/routeSim.js`, replacing `src/brain/dynamics.js`) and mirrored 1:1 by the
GPU kernels. No other module may implement vehicle motion (SPEC "Vehicles and dynamics: full physics").

```js
import { VEHICLE_PARAMS, createState, stepDynamics, pedalsFor, maxDriveAccel, resistance, fuelFlow,
         obbContact, sweptContact, resolveContact, separate, boxInertia, PHYS } from './src/sim/physics/index.js';
```

## Parameters (`params.js`)

`VEHICLE_PARAMS[typeIdx]` (0 sedan, 1 SUV, 2 truck, 3 bus): `length, width, height, L (wheelbase), a, b
(CG to front/rear axle), hcg, m, Iz, CdA, Crr, Pmax (W at the wheels), F0 (N, low-speed traction limit),
drive ('front'|'rear'|'all'), vmax, brakeDecel (m/s² at brake = 1 before the tyre limit), brakeFront,
Cf, Cr (cornering stiffness, N/rad), mu (peak road friction on dry asphalt; scaled per surface), maxSteer (rad), maxSteerRate (rad/s),
fuel ('gasoline'|'diesel'), bsfcMin (g/kWh), idleLph, accessoryW, tankL`. Constants `RHO_AIR = 1.225`,
`G = 9.81`, `FUEL_DENSITY` (g/L).

## Dynamics (`dynamics.js`)

State (`createState({u, v, r, steer, ax, hr, lat, s})`):

| field | meaning |
|---|---|
| `u` | forward body velocity of the CG (m/s, ≥ 0: no reverse gear) |
| `v` | lateral body velocity (m/s, + = left) |
| `r` | yaw rate (rad/s, + = turning left; world heading `atan2(dz,dx)` changes by −r) |
| `steer` | front-wheel angle (rad, + = left) |
| `ax` | longitudinal acceleration of the last sub-step (weight transfer) |
| `hr, lat, s` | pose in the road (Frenet) frame: heading relative to the lane (+ = left), m from the carriageway's right boundary, m along the edge |
| `fuelL, power` | litres burnt during the last call, engine output (W) of the last sub-step |

`stepDynamics(st, steerRate, gas, brake, P, dt, grade = 0, locked = false)` advances `dt` seconds in
sub-steps of `PHYS_DT = 10 ms` (semi-implicit Euler). Controls are the APPLIED (reaction-delayed)
values: `steerRate ∈ [-1, 1]` × `maxSteerRate`, `gas, brake ∈ [0, 1]`. `grade` = dy/ds of the road (real terrain since 2026-10-24, up to ±35 %; docs/elevation.md). On a slope the
axle loads shift by m g sinθ hcg / L, gravity enters the road load, a stopped car's brakes hold against gravity on a
downhill, and `s` advances by the horizontal component u cosθ. No reverse gear: on an uphill a car never rolls back.
`locked` = disabled car: no traction, wheels locked, sliding friction until rest.
`surf` (11th argument, optional) = road-surface factors under the wheels `{pk, sk, crr}` from `src/sim/surface.js`
`carSurface`: peak friction `mu·pk`, locked-wheel sliding friction `mu·sk`, rolling resistance `Crr·crr`. Omitted or
`{1, 0.8, 1}` = dry asphalt, exactly the single-mu model. Table and sources: `docs/SURFACES.md`.

Per sub-step: steer integrates and clamps; normal loads with longitudinal weight transfer; traction
`gas·min(F0, Pmax/u)` split by drive layout; brakes `brake·m·brakeDecel` split front/rear; each axle's
longitudinal force limited to 0.98 μ Fz (ABS), the lateral capacity is what the friction circle leaves;
slip angles `αf = δ − atan((v + a r)/u)`, `αr = −atan((v − b r)/u)`; tyre force `Fy = Fymax·tanh(C α / Fymax)`;
road load `½ρ CdA u|u| + Crr m g + m g sinθ`; body equations
`u̇ = (Fxf cosδ − Fyf sinδ + Fxr − R)/m + v r`, `v̇ = (Fxf sinδ + Fyf cosδ + Fyr)/m − u r`,
`ṙ = (a(Fxf sinδ + Fyf cosδ) − b Fyr)/Iz`; below 5 km/h the lateral state blends to the kinematic bicycle;
pose `hr += r h`, `lat += (u sin hr + v cos hr) h`, `s += (u cos hr − v sin hr) h`.
Road curvature is absorbed by the Frenet frame: the car turns with the road without any tyre force. This was meant
as a simplification (the edge polylines are not smooth enough to steer along), but it lets a car take a freeway-speed
right angle with the wheel straight (median peak 14.6 g in `rust/docs/physics-audit.md`). The Rust host has a fix
(`curvature_dynamics`: world heading changes by yaw only). It is off by default until the lane polylines are smoothed
(`rust/DECISIONS.md`, entry 23).

Fuel: engine power = traction power / 0.88 + accessories; flow = max(idle, P·BSFC(load)),
`BSFC = bsfcMin·(1 + 0.9 (1 − load)³)` g/kWh.

Helpers: `pedalsFor(P, accel, u, grade, out)` → `{gas, brake}` for a desired acceleration;
`maxDriveAccel(P, u)`; `resistance(P, u, grade)`; `fuelFlow(P, watts)` (L/s); `bsfc(P, load)`.
For a delay predictor, the steady-state yaw gain is `r = u δ / (L + Kus u²)`, `Kus = m/L (b/Cf − a/Cr)`.

Validation (`test/sim/physics.test.js`): sedan 0–100 km/h 8.7 s, truck ~29 s; 100–0 km/h braking 39.9 m;
coastdown = (aero + rolling)/m within 3 %; slides at ~1.0 g (μ = 1.0 dry asphalt peak); understeer gradient
1–4 deg/g; delayed lane change at 100 km/h settles with < 0.6 m overshoot; full-lock swerve at 120 km/h
loses grip; ~11 km/L sedan at 100 km/h.

## Collisions (`collide.js`)

Body = `{x, z, yaw, vx, vz, w, hl, hw, invMass, invInertia}` (w = d yaw/dt; static: invMass = invInertia = 0).
`obbContact(A, B, out)` → `{depth, nx, nz, px, pz}` or null (SAT; n from A to B; contact point = deepest
incident vertex / edge midpoint). `sweptContact(A0, A, B0, B, out, maxStep = 0.5)` → first overlapping
fraction t ∈ (0, 1] along the step (relative sub-steps ≤ 0.5 m, no tunnelling) or −1.
`resolveContact(A, B, c, e = 0.2, μ = 0.5, out)` → impulse with restitution, Coulomb friction and angular
terms; returns `{jn, jt, dvA, dvB, closing}` (delta-v in m/s = crash severity). `separate(A, B, c)`
positional correction. `boxInertia(m, length, width)`.
