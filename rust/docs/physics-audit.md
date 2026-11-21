# Physics audit: where the simulator departs from reality

**Status:** audit only. No sim code was changed. The scenario code is in
`crates/bayenv-sim/examples/physics_audit/` (`main.rs`: vehicle dynamics and contacts with no world;
`world_audit.rs`: a geometry scan of the real world, targeted scenarios on real edges, and a rollout anomaly scan).
Every number below comes from that code on the `rust` branch (bayenv-sim, the default model, which is bit-exact with
the JS CPU sim apart from the model-fix flags) and the real Bay Area world in `data/`.

```
cargo run --release -p bayenv-sim --example physics_audit -- dyn            # A: vehicle dynamics
cargo run --release -p bayenv-sim --example physics_audit -- contact        # B: rigid-body contacts
cargo run --release -p bayenv-sim --example physics_audit -- world ../data  # W: world geometry
cargo run --release -p bayenv-sim --example physics_audit -- scen ../data   # S: targeted scenarios
cargo run --release -p bayenv-sim --example physics_audit -- sim ../data 5000 2400 keep|const|wobble|fast|pursuit  # R: rollout scan
# AUDIT_JS=1: the JS model; AUDIT_CURV=1: curvature_dynamics on; AUDIT_SMOOTH=1: also smooth_geometry (DECISIONS.md 33);
# AUDIT_CRASHES=1: crash sites by kind; AUDIT_TRACE=<slot>: per-step trace of one car
```

The GPU kernels (`src/gpu/kernels/physics.wgsl`, `integrate.wgsl`, `contacts.wgsl`, `collidefns.wgsl`) were read
line by line against the CPU code. They mirror it (see `kerb-investigation.md` and `cpu-vs-gpu.md`), so every model
finding below applies to both backends unless it says otherwise. The GPU was not run for this audit, because the
machine was busy with a training run.

Earlier investigations, not repeated here: `kerb-investigation.md` (GPU connector join),
`crash-divergence.md` (late missed-turn re-plan, and the f32 friction tangent), `determinism.md` and
`cpu-vs-gpu.md`. The fixes `cpu_connector_join`, `contact_tangent`, `replan_at_missed_event` and
`lane_index_snapshot` are already the default.

## Ranked findings

| # | finding | severity | training / video | kind | fix as a model-fix flag? |
|---|---|---|---|---|---|
| 1 | **Road and junction curvature costs no tyre force.** Cars take any curve at any speed | critical | both | deliberate simplification, with consequences far beyond the intent | yes: `curvature_dynamics` |
| 2 | **Stopped wrecks and static obstacles have infinite mass**, so the Δv of hitting one is about 2× too high and pileups never push them | high | training (crash penalty), video (bus stops dead against a sedan) | bug | yes: `wreck_mass` |
| 3 | **No crash event while already in contact** (barrier and pairs). A strike's Δv is split over steps and only the first slice counts | high | training (cheap wall-riding, under-penalised impacts) | bug | yes: `contact_event_per_impulse` |
| 4 | **No lateral containment inside junction boxes.** At the exit `lat` is snapped back, a teleport of up to 16 m; the world speed of an offset car is amplified up to 2.6× | high | both | simplification, with a bug-like consequence | yes: `connector_barrier` |
| 5 | **Barrier positional correction removes the full depth in one step** (lateral teleport of up to 1.7 m) | medium | video, slightly training | simplification | yes (with #4) |
| 6 | **A wreck sliding into the end of an edge freezes in place but keeps its speed** | medium | both (phantom-velocity wreck, wrong wreck positions) | bug | yes: `wreck_edge_hop` |
| 7 | **`write_back` clamps backward speed to 0**: a car pushed backwards (head-on with something heavier) loses its momentum | medium | crash outcomes | simplification (no reverse gear) | yes |
| 8 | **Heading jumps at strict hops without a junction box** (2,973 hops kink more than 1 rad; 202–3,582 jumps above 0.3 rad per rollout) | medium | video, contacts | geometry/frame | partly fixed by #1 |
| 9 | **Brakes do nothing below 0.02 m/s and rolling resistance nothing below 0.05 m/s.** 20–65 % of stops creep up to 1.2 m per minute under a held brake | low–medium | stop lines, long queues | bug | yes: `standstill_hold` |
| 10 | **Kerbs are rigid walls everywhere**: no mounting, no run-off onto grass beyond the 1 m / 3 m shoulder. A 3 m/s lateral kerb touch disables a car | medium | training (kerb strikes = crashes) | deliberate simplification | design decision |
| 11 | **No lateral load transfer or rollover.** Trucks and buses corner at 0.55–0.64 g; real rollover thresholds are about 0.35–0.5 g | low now (masked by #1), medium after #1 | both | simplification | yes: `rollover_limit` |
| 12 | **No rollback on uphill and no idle creep** (no reverse gear) | low | stuck cases; realism of hill starts | deliberate | no (document) |
| 13 | **Grade is 0 inside junction boxes**; steep SF intersections are flat to the physics | low | SF hill starts | simplification | yes |
| 14 | Numeric edge cases: dt = 0 gives `ax = NaN`, which poisons the state; a negative dt runs backwards; a NaN state is sticky. **GPU `clamp(NaN)`** on an action is indeterminate in WGSL (Metal returns the lower bound, so a NaN steer becomes full left lock; the CPU maps NaN to 0) | low (unreachable with sane inputs) | CPU/GPU disagreement on a bad policy output | bug | host-side sanitising, no flag needed |
| 15 | Disable threshold 10 km/h own Δv, restitution 0.2 and friction 0.5, truck brakes 0.66 g | low | crash rates | calibration | tune, no flag |

The rest of this document gives the inventory, the measurements against real-world values, and the details of each
finding.

## 1. Inventory of physical models

| model | where (CPU / GPU) | summary |
|---|---|---|
| vehicle dynamics | `physics.rs step_dynamics` = `dynamics.js` / `physics.wgsl` | planar dynamic bicycle, 10 ms semi-implicit Euler sub-steps; longitudinal load transfer (and grade load shift); tanh "brush-like" lateral tyre (no fall-off past the peak); friction circle per axle; ABS 0.95 μFz, traction control 0.98 μFz; EBD and a rear-axle ESC limit; aero `½ρCdA u²`, rolling `Crr m g` (only above 0.05 m/s), grade `m g sinθ`; power-limited CVT drive `min(F0, Pmax/u)`; vmax governor (drive only); kinematic bicycle blend below 1.4 m/s; no reverse (u ≥ 0) |
| steering | same | position mode: target × maxSteer, first-order lag (0.10 / 0.12 s), slew 0.80 / 0.53 rad/s at the wheel; max 0.60 rad (cars), 0.50 (heavy) |
| vehicle classes | `VEHICLE_PARAMS` | sedan 1.5 t / 100 kW FWD; SUV 2.1 t / 150 kW AWD; box truck 15 t / 260 kW; bus 13 t / 200 kW; μ 1.0 cars, 0.8 heavy |
| surfaces | `surface.rs` = `surface.js` / `integrate.wgsl surfFactors` | peak and slide μ and Crr per OSM surface (asphalt μ_eff 1.0, grass 0.53, gravel 0.71 × the vehicle's μ), averaged over the left and right wheels; dry only (wet is documentation only) |
| pose / road frame | `vehicle.rs integrate_post`, `update_pose` = `vehicle.js` / `integrate.wgsl` | Frenet state (s, lat, heading_rel) on the edge or junction connector; the world pose is derived each step; edge hops map lat (strict hops snap near the lane edge); connector entry/exit; s advances by `u cosθ` |
| grade | `World::edge_grade` | ±6 m average of the polyline y, clamped to ±35 %; 0 inside connectors |
| contacts | `contacts.rs`, `physics.rs` = `sim.js` + `collide.js` / `contacts.wgsl`, `collidefns.wgsl` | OBB SAT with swept sub-steps (≤ 0.5 m), sequential impulse with restitution 0.2, Coulomb friction 0.5, angular terms, positional correction 0.8 × depth; levels separated by `layer_gap` 2.5 m; freeway contacts only between node-adjacent or listed edge pairs |
| barriers | `contacts.rs barrier` / `contacts.wgsl barrier` | a rigid wall at the paved edge + shoulder (1 m city, 3 m freeway) and past the opposite lanes; e 0.15, μ 0.4; inactive in junction boxes and within 1 m of the edge ends |
| crash model | `contacts.rs crash_event` | severity = larger own Δv; disabled at own Δv ≥ 10 km/h; locked-wheel slide afterwards; scene timeline |
| integration | `Sim::step` | dt 0.05 s (create_real 0.1 s), physics sub-steps of 10 ms, f64 compute with f32 storage of the state (CPU and GPU) |

## 2. Vehicle dynamics against reality (A)

### Acceleration

| type | 0–50 km/h | 0–100 km/h | sim vmax | real reference |
|---|---|---|---|---|
| sedan | 3.8 s | 8.7 s | 198 km/h | Camry 2.5 / Accord 1.5T: 0–100 in 7.5–8.5 s (Car and Driver tests) |
| SUV | 3.5 s | 8.1 s | 180 | RAV4 / CR-V: 8–9 s |
| truck (15 t) | 6.5 s | 28.7 s | 112 | loaded class 6–7 box truck: about 30–45 s to 100 km/h |
| bus (13 t) | 7.3 s | 37.6 s | 108 | 40-ft transit bus: 0–50 km/h in about 10–14 s (APTA / Altoona test data); slightly quick |

### Braking (brake = 1, no reaction time)

| type, surface | 30 km/h | 60 km/h | 100 km/h | real reference |
|---|---|---|---|---|
| sedan ABS, dry asphalt | 3.7 m | 14.9 m | 41.2 m (0.95 g) | modern sedans 100–0 in 36–40 m (about 1.0 g) |
| sedan locked | 4.3 | 17.3 | 47.7 (0.82 g) | locked-wheel dry asphalt μ 0.7–0.8 (accident-reconstruction tables): 49–56 m |
| sedan ABS, grass | 6.0 | 23.9 | 65.6 | dry grass μ 0.35–0.5: 79–112 m (the sim's grass is slightly grippy) |
| truck ABS | 5.2 | 21.0 | 57.8 (0.68 g) | FMVSS 121 loaded trucks 60–0 mph within 76–94 m (0.4–0.5 g): about 80 m from 100 km/h. **The sim's trucks brake about 30 % too well** |
| bus ABS | 5.2 | 20.9 | 57.4 | transit bus about 0.5 g: about 75 m |

Downhill 60→0 km/h: sedan 17.7 m on −15 % and 23.3 m on −35 %; truck 26.6 m and 39 m. These are physical
(g sinθ subtracts).

**Wheel lockup without ABS is not reachable by a driver.** `locked` is only used for disabled cars, and brake = 1
always uses ABS. This is a deliberate choice (ABS is mandatory in the US since MY2012), so nothing needs fixing.

### Cornering (skid pad, position steer held, speed held)

| type | max steady a_y at 15 / 25 m/s | static stability factor t/2h | real |
|---|---|---|---|
| sedan | 0.86 / 0.78 g | 1.44 | 0.85–0.95 g on dry asphalt; never rolls |
| SUV | 0.83 / 0.74 g | 1.18 | 0.75–0.85 g; NHTSA rollover ratings |
| truck | 0.60 / 0.55 g | 0.80 | **rollover at about 0.35–0.5 g loaded** (UMTRI, Winkler) |
| bus | 0.64 / 0.50 g | 0.91 | about 0.4–0.5 g |

The tyre curve saturates and holds at its peak: Fy/Fymax is 0.34, 0.71, 0.95, 1.00 and 1.00 at 2°, 5°, 10°, 20°
and 45° slip. Real tyres lose 10–30 % past the peak. So sliding is easier to recover than in reality. This is
minor.

High-speed stability is fine. A 2 % lock step at 194 km/h decays without oscillation, and a full-lock swerve
spins, as it would in reality. Braking at the limit in a 25 m/s curve spins the AWD SUV but not the others.
Real ESC would prevent this, so it is minor.

Minimum turning circle (outer front corner): sedan 12.3 m, Camry 11.6–12.2 m. SUV 13.3 m. Truck 26.9 m and bus
30.4 m, against about 15–17 m for a box truck and about 26 m wall to wall for a 40-ft bus. The heavy vehicles turn
a little wide.

### Grades

The world has 1.97 % of its road length above 15 % grade and 0.07 % above 25 %. The steepest raw segment longer than
5 m is 54 % (a 5 m edge; the physics' ±6 m average and its 35 % clamp see 0 km above 35 %).

| type | 0–30 km/h on 15 % | speed after 60 s on 6 % / 15 % / 25 % |
|---|---|---|
| sedan | 3.8 s | 173 / 126 / 90 km/h |
| truck | 5.7 s | 78 / 40 / 25 km/h; never reaches 30 km/h on 25 % |
| bus | 7.5 s | 69 / 35 / 22 km/h |

A loaded truck on I-280's roughly 6 % climbs holds about 60–80 km/h in reality, so this matches. A sedan starts on
SF's 31 % blocks (Filbert, 22nd St) at 0.5 m/s², which is plausible.

**No rollback (finding 12):** a car stopped on a 35 % uphill with brake 0 stays at u = 0, s = 0. A real car rolls back
at about 3 m/s². On a downhill the car does roll (17 km/h after 10 s on 6 %). There is no idle creep: an automatic in
Drive creeps at 5–8 km/h in reality, the sim stays at 0.

### Very low speed (finding 9)

```
sedan  169/400 stops end with u > 0 under the held brake; worst 1.19 m crept in 60 s at u = 0.0199 m/s
truck  261/400 ...; bus 255/400; suv 183/400
coast from 1 m/s with no pedals: never stops (rolling resistance switches off below 0.05 m/s)
gas 0.01 from rest: settles at 0.05 m/s (the drive force is below the rolling resistance, which is off at rest)
```

The cause is in `step_dynamics`. Below `moving = u > 0.02` the brakes only *hold* against drive and gravity,
`hold = min(Fb, max(0, Fd − Wg))`, which is 0 on the level with gas 0. Rolling resistance is gated at `u > 0.05`. A
10 ms sub-step that lands u in (0, 0.02] therefore stays there for ever. This is the same in the GPU code.

**Fix:** below the threshold, let the brake (and the rolling resistance) bring u to exactly 0:
`if !moving && Fb + roll > |net force| * m { un = 0 }`.

### Numerics

| input | result |
|---|---|
| dt = 0 | `ax = 0/0 = NaN` stored, so the next call's normal loads are NaN and the state is poisoned. The sim's dt is fixed > 0, so this is only a guard issue |
| dt < 0 | integrates backwards (s −1.0 m), no guard |
| dt = 10 s | fine (1,000 sub-steps) |
| NaN / inf controls | CPU: NaN → 0 (`jm::or`), inf → clamped. **GPU:** `clamp(NaN, lo, hi)` is an indeterminate value in WGSL; Metal's `fmin`/`fmax` return the lower bound, so a NaN steer would be −1 (full lock). The CPU and GPU disagree on a NaN policy output. Sanitise actions on the host (`bayenv-gpu set_actions`) |
| NaN state | sticky (never recovers) |
| f32 storage | the longest edge is 10,968 m, where the f32 ulp is 1.3 mm against a 5 mm step at 0.1 m/s, so low-speed creep is resolved. x/z are re-derived from s every step, so no drift accumulates |

## 3. Contacts against reality (B, S2–S6)

### Canonical crashes (`resolve_contact`, then the `write_back` clamp)

| scenario | sim Δv (striker / struck) | physics expectation (e ≈ 0.1–0.2) | note |
|---|---|---|---|
| sedan 50 km/h rear-ends a stopped *driving* sedan | 30 / 30 km/h | 27–30 / 27–30 | correct |
| sedan 50 into a stopped **wreck / parked car** | **60 / 0** | 27–30 / 27–30 | finding 2: the wreck is a rigid wall |
| sedan 69 into a parked sedan (S2, real edge) | **82.5 / 0** | about 41 / 41 | finding 2, end-to-end on the map |
| bus 50 into a stopped sedan wreck | **60 / 0, the bus stops dead** | bus about 10, sedan about 50 | finding 2 |
| sedan 100 into a wreck | **120** | about 60 | finding 2 |
| head-on sedan 50 vs sedan 50, aligned | 60 / 60 | 55–60 | correct |
| head-on truck 50 vs sedan 50 | 10.9 / 109 | 11 / 109; **then the sedan's backward 16.4 m/s is clamped to 0** (−200 kJ, momentum not conserved) | finding 7 |
| T-bone at the centre | 30 / 30 | correct | |
| T-bone at the rear quarter | 17.6 / 17.6, struck car spins at 4.6 rad/s | plausible | |
| lane-change sideswipe at 108 km/h, 2.4 m/s lateral | 2.3 / 2.3 | plausible | |

Energy: 100,000 random shallow contacts (depth < 5 cm, the depth the swept test finds) with the sim's μ 0.5 and with
μ 0: **no kinetic-energy gain in any of them.** The impulse solver is dissipative. (An earlier version of this test
gave fixed bodies a velocity, which the sim never does, and showed a spurious gain.) Rollouts also showed no pair
contact whose two cars gained more than 2 % KE over the step.

### Finding 2: stopped wrecks are immovable

`contacts.rs body()`: `fixed = !moving && state == STATE_CRASHED` gives `inv_mass = inv_inertia = 0`
(`contacts.wgsl:44-47` does the same). This covers every wreck at rest, every `place_static_obstacles` /
scenario obstacle, and `place_obstacles` props once stopped. It is meant to stop two resting wrecks from jittering
each other, but it makes every secondary crash a crash into a wall: twice the Δv, and nothing is shoved.
Secondary crashes into wrecks are common whenever traffic jams behind an incident.

**Fix (`wreck_mass`, default on):** use the real mass in pair contacts. If resting wrecks must not drift, apply the
locked-tyre static friction to them (the dynamics already slide locked wheels at 0.8 μ).

### Finding 3: no event while already in contact

`barrier()` emits only when `!was` (it was not touching in the previous step). Pair contacts emit only when the key is
not in `self.touch` (touching in the previous step). The GPU does the same with `F_WALL_PREV`. The impulse is applied
on every step either way. S6 on a real residential edge:

```
sustained (leaning on the kerb at 0.3 m/s), then a 6 m/s lateral strike:
  [40 vlat -6.0->-4.7 dv 4.8 km/h  no event] [41 vlat -4.7->-1.3 dv 12.0 km/h  no event] ... state 1 (driving)
fresh strike at 3.7 m/s lateral:
  [46 dv 4.0 km/h  CONTACT 4.7 km/h] [47 dv 6.6 km/h  no event]   (real: about 15 km/h, one event)
```

A car that keeps touching the wall is immune to wall crashes, and every wall strike reports only the Δv of its
first step. Barrier severity is therefore under-reported, and the crash penalty can be dodged by grazing.

**Fix (`contact_event_per_impulse`):** accumulate the impulse Δv over a contact episode. Emit or update the event
(and disable at ≥ 10 km/h) once the accumulated Δv crosses a tier, instead of only on the first step.

### Finding 7: backward speed clamped in `write_back`

`write_back` sets `u = max(0, u_body)`. The pure test shows a sedan hit head-on by a truck should leave at 16.4 m/s
backwards. The sim zeroes this, so 200 kJ and the momentum vanish, and the truck re-penetrates on the next step. This
follows from "no reverse gear", but a body pushed backwards is not driving. **Fix:** allow u < 0 for CRASHED cars
(locked-wheel slide), or store the backward motion as `heading_rel += π`.

### Parked and stopped cars

A stopped *driving* car (queue at a light) is a normal movable body, so queue rear-ends are correct. Only wrecks
and obstacles are walls (finding 2).

### S4 / finding 6: a wreck sliding into an edge end

```
(s − length / speed m/s) per step: -1.42/20.3 -0.41/19.9 -0.00/19.5 -0.00/19.1 ... -0.00/9.2   (x frozen at -5795)
```

`integrate_post`: `if locked { s = end; break; }` for both the edge end and the connector end. The wreck stops at the
node but keeps 19.5 m/s, decaying by tyre friction, while it sits still. Contacts read that speed, so the wreck hits
cars at the node with phantom momentum. Real wrecks slide through junctions. **Fix (`wreck_edge_hop`):** let a
locked car continue onto the route's next edge or connector (as a driving car does), or zero its speed when it is
clamped.

### Reversing and boxed-in cars (S5)

There is no reverse gear (the action space is steer / gas / brake; u ≥ 0). A sedan stopped 1 m behind a parked car on
a one-lane edge escaped with full lock and gas 0.3: it moved 9 m, heading_rel −1.25, with 3 contact touches. In this
case it is not stuck, but it can only escape by scraping, and a car nose-in against a wall with no lateral room can
never leave. The rollouts below had no car stuck for more than 300 s. Deliberate (finding 12); document it.

## 4. Geometry the physics touches (W, R)

| check | result |
|---|---|
| edges | 291,155 directed edges, 562,530 edge-to-edge hops; min length 0.29 m; 20 edges < 1 m; 649 < 4.5 m; 10,126 < 12 m. The integrate loop handles several hops per step |
| zero-length polyline segments | 3 of 943,271 (guarded by `jm::or(len, 1)`) |
| lane width | 2.74–3.70 m |
| elevation at hops | 0 hops with \|Δy\| > 0.25 m; connectors' end points within 0.4 m of the edges. **No elevation discontinuities** |
| junction turns | 349,559 movements turning more than 69°; 254,402 have a mean radius < 8 m (1 g there = 32 km/h; 0.3 g comfort = 17 km/h) |
| strict hops without a box | 12,844; frame heading jumps > 0.2 rad: 5,636; > 1 rad: 2,973 (finding 8) |

### Rollout scan (R): 5,000 commuters from 07:00, 2,400 steps (dt 0.05), about 870 cars

`keep` is a lane-keeping, speed-limit, leader-and-red-light controller. `const` is [steer 0, gas 0.3, brake 0], the
`compare_backends.py` policy.

| anomaly | keep | const |
|---|---|---|
| non-finite pose or speed | 0 | 0 |
| teleport (moved > 1.5 × speed × dt + 0.5 m) | 206 (0.010 % of car-steps) | 488 (0.025 %) |
| worst teleport | 16.3 m: a car at lat −14.9 in a connector snapped to lat 1.65 on the exit edge | 42.5 m at an edge hop |
| heading jump > 0.3 rad in one step, no contact | 202 | 3,582 |
| y jump (level change) | 0 | 0 |
| \|accel\| > 12 m/s² without a pair contact | 244 (max 107) | 16; these are silent barrier impulses (finding 3) |
| speed > vmax | 1,113 car-steps | 3,309: heavy vehicles coasting down 17 % grades (the governor cuts only the drive); physical |
| **world lateral accel > μg** at > 8 m/s | **3.9 %** of car-steps, 93 % with < 0.3 μg from the tyre model | **5.4 %**, 99.9 % |
| spawn overlapping another body | 0 / 871 | 0 / 871 |
| bodies overlapping after a step | 0 | 6 pairs (5 on the same edge, 1 adjacent) + 44 wreck–wreck |
| xz-overlap across levels | 0 | 0 |
| contacts between cars > 1 m apart in y | 0 | 0 |
| cars stopped > 300 s | 0 | 0 |

Teleport examples inside a connector (`const`, finding 4). A car with a heading error drifts at −0.84 m per step in
lat, and its world displacement is 3.75 m per step at 28.8 m/s (1.44 m expected), because an offset point on a tight
connector moves at u (1 + off/R). Then comes the snap at the exit. Barrier example (finding 5): lat −1.75 → −0.02
in one step.

Stacked roads, tunnels and overpasses: in these short runs no pair overlapped in xz at a vertical separation
between 2.5 and 4.5 m, and no contact was resolved across levels. `layer_gap` = 2.5 m is below a truck's 3.6 m
height, so a truck under a real low structure (< 4.5 m clearance) would pass through. Hops show no Δy jumps. Longer
or full-population runs should repeat the "xz-overlap across levels" counter before this is called clean.

## 5. Details of the top finding

### Finding 1: curvature is free

`step_dynamics` integrates `hr += r·h` in the road frame, and the frame bends with the edge or connector polyline.
`update_pose` takes the world heading as the frame heading minus `heading_rel`. There is no `−κ(s)·ds` term
anywhere (`dynamics.js`, `physics.rs`, `physics.wgsl`, `vehicle.js`, `vehicle.rs`, `integrate.wgsl`). A car with
steer 0 and yaw rate 0 follows every curve and every junction turn exactly, and the tyres produce no side force.
The JS README says this was deliberate ("the edge polylines are not smooth enough to steer along").

S1 (real map, empty roads): 948 cars were held at 108 km/h with steer 0 for 60 s. They took 2,364 junction turns of
more than 69° at over 90 km/h, with **0 barrier or contact events**, and 933 were still driving at the end. The
median per-car peak world lateral acceleration was 14.6 g, the maximum 28.9 g. The rollouts show the same for an
ordinary policy (3.9–5.4 % of moving car-steps above μg).

Consequences:
* The policy never has to slow for a curve or a turn. The speed–curvature trade-off, run-off-road crashes and
  turning crashes are absent from training.
* Steering only corrects lane offset and the connector heading offsets, so steering skill learned in the sim does not
  transfer.
* On camera, cars turn 90° at highway speed.
* The tyre-limit work (friction circle, ESC, ABS in curves, the surface μ) only matters for lane changes.

**Fix (`curvature_dynamics`, default on, legacy in `SimOpts::js()` and as a GPU option):** after each sub-step,
subtract the frame's heading change over the arc travelled: `hr -= Δψ_frame(s → s + ds)`. Use the smoothed
`connector_point` heading inside boxes and `edge_point_dir` (or the lane graph's `offset_point`) on edges. The car then
has to yaw to follow the road, and the existing tyre model limits it to about 0.8 g. Polyline kinks (finding 8) need
the heading to be smoothed over a few metres, or the kink turns into a heading impulse the driver must steer
against. Expect a large drop in arrival rate until policies learn to brake for turns. **Retrain after enabling.**

## 6. Classification summary

* **Bugs (fix behind flags, default on):** 2 wreck mass, 3 contact event per impulse, 6 wreck edge hop, 9
  standstill hold, 14 GPU NaN actions (host sanitising).
* **Simplifications with large effects (model upgrade, flag):** 1 curvature, 4 connector containment, 5 barrier
  correction, 7 backward clamp, 11 rollover, 13 connector grade.
* **Deliberate, document only:** no reverse, no rollback, no idle creep, ABS always on, rigid kerbs (finding 10:
  real 15 cm kerbs are mountable, so consider a soft kerb with a Δv limit before it counts as a crash), dry roads.
* **Calibration:** truck and bus braking about 0.66 g, against about 0.45–0.5 g real; disable at 10 km/h own Δv
  (real tow-away crashes cluster around 15–25 km/h Δv in NASS-CDS); grass μ 0.53 is slightly high.

## 7. DECISIONS.md-style entries

### Physics audit: curvature is free

- **Date:** 2026-11-21
- **Finding:** cars follow road and junction curves without tyre force, because the road frame absorbs the
  curvature. 948 cars at 108 km/h with steer 0 took 2,364 sharp junction turns with no crash; median peak world
  lateral acceleration 14.6 g.
- **Why it matters:** the agents never learn to slow for corners. Cornering crashes cannot happen, and turns look
  impossible on camera.
- **On camera:** "Our cars could take a 90-degree turn at highway speed, because the road quietly turned them. The
  tyres never felt the corner."

### Physics audit: wrecks were made of concrete

- **Date:** 2026-11-21
- **Finding:** a stopped wreck or parked car has infinite mass in contacts. A bus at 50 km/h stops dead against a
  parked sedan; a 69 km/h rear-end reports Δv 82.5 km/h instead of about 41.
- **Why it matters:** secondary crashes are reported at double severity, and pileups never push wrecks.
- **On camera:** "Every parked car in our city was secretly an immovable wall. A bus would bounce off a sedan."

### Physics audit: leaning on the wall made you invincible

- **Date:** 2026-11-21
- **Finding:** crash events fire only on the first step of a contact. A car already touching the kerb took a 12 km/h
  lateral impact with no event and kept driving.
- **Why it matters:** wall strikes are under-penalised, and an RL agent can exploit this.
- **On camera:** "We found a loophole: if you were already touching the wall, hitting it harder didn't count."

### Physics audit: junction boxes had no walls

- **Date:** 2026-11-21
- **Finding:** inside a junction there is no lateral limit. Cars drifted up to 15 m off the connector and were snapped
  back at the exit, a teleport of up to 16 m.
- **On camera:** "Inside an intersection a car could wander anywhere, then teleport back onto the road."

### Physics audit: what held up

- **Date:** 2026-11-21
- **Finding:** braking (sedan 100–0 in 41 m), acceleration (0–100 in 8.7 s), skid-pad grip (0.86 g), grades, energy
  conservation of the impulse solver (no gain in 100,000 random contacts), spawns (0 overlaps), elevation
  continuity at 562k hops, and no level-crossing contacts.
- **On camera:** "Most of the physics checked out against real cars, which is exactly why the few that didn't matter."
