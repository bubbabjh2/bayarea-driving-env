# Sim state layout (CPU reference ↔ GPU contract)

> Env-only build: this file is kept from the original project. The original built-in driver (`driver.js`:
> IDM / pure pursuit), `src/brain/*` and `src/transformer/*` are NOT part of this repository; mentions of them
> below (e.g. "built-in driver" fields such as `targetLane`, `lastDecision`, `mergeCap`, `maxAccel`, `maxDecel`)
> describe state the sim still allocates but that only a policy would use. `src/brain/dynamics.js` = the
> vehicle parameters now in `src/sim/physics/params.js`.

Owner: sim agent (`src/sim/`). The GPU agent (`src/gpu/`) mirrors this. Layout version:
`LAYOUT_VERSION` in `src/sim/state.js` (currently **7**). Every change gets a changelog line at the
bottom. Snapshot / restore: `exportState(sim)` / `importState(sim, snap)` (`src/sim/state.js`);
`STATE_FIELDS` lists every per-vehicle array with its type and stride.

## Conventions

* World: x = east, z = south, y = up (metres), as `data/roads.schema.md`. Heading = `atan2(dz, dx)`.
* Physics: `src/sim/physics/` (README there) is the only motion model: `params.js VEHICLE_PARAMS[typeIdx]`,
  `dynamics.js stepDynamics`, `collide.js`. Lane geometry: `src/geom` (world-owned) via `graph.laneGeom`
  (`latPoint`, `bounds`, `hopMap`) when the real packs are loaded; synthetic graphs fall back to `edgePoint`.
* Graph: `SimGraph` (`roadGraph.js`). Merged two-tier graph from `data/roads.bin` + `data/roads-local.bin`
  (`data.js loadWorld` → `fromRoadSet`): main-tier node/edge indices kept, local-tier nodes not shared
  with the main tier appended (`graph.tierNodeMap[tier][packNode]`), local edges appended
  (`edge = mainEdgeCount + localEdge`). Edge fields used by the step: `length`, `lanes` (per direction),
  `speedLimit` (m/s), `from`, `to`, `reverse` (twin or -1), `rank`, `isLink`, `headingStart/End`,
  polyline `px, pz, cum`, `ys` / `y`, `laneOff` (lane-centre offsets right of the polyline, lane 0 =
  rightmost), `rightEdge = laneOff[0] + 1.85` (carriageway right boundary, m right of the polyline).
  CSR: `outStart/outEdges`, `inStart/inEdges` (Int32Array).
* Lanes: width `LANE_WIDTH = 3.7`. Lane `k` (0 = rightmost) spans lat ∈ [3.7k, 3.7(k+1)), centre 3.7(k+0.5).
* Time step `dt` (headless / training default 0.1 s; `SIM_DEFAULTS.dt` 0.05 s for the viewer).

## Per-vehicle state (`VehicleStore`, `vehicle.js`) — struct of arrays indexed by slot

Slots `0 .. vs.high-1`; free slots have `state = 0` and are recycled (LIFO free list `vs.free`).
`vs.capacity` grows by doubling (every array is reallocated; re-read the views after `grow()`).

| array | type | per | units / meaning |
|---|---|---|---|
| `state` | Uint8 | 1 | 0 FREE, 1 DRIVING, 2 CRASHED (any lane-blocking stopped car: crash, stranded, wreck), 3 ARRIVED (unused), 4 REMOVED (unused), 5 REFUELING (in a station lot; not in lane lists / hash) |
| `typeIdx` | Uint8 | 1 | 0 sedan, 1 SUV, 2 truck, 3 bus (`vehicleTypes.js`) |
| `flags` | Uint8 | 1 | bit0 LEARNER, bit1 COLLIDED (this step, for the driver), bit2 BRAIN (has a Driver), bit3 SHOULDER (scene moved off the lane / squeezable: not an obstacle), bit4 SPEEDING (>1.1 x limit), bit5 DISABLED (crash delta-v ≥ 10 km/h), bit6 STRANDED, bit7 REFUEL (detour planned) |
| `edge` | Int32 | 1 | current directed edge |
| `s` | Float32 | 1 | m along the edge centreline from `from` |
| `lat` | Float32 | 1 | m from the carriageway's RIGHT boundary (+ = left); may be < 0 or > lanes·3.7 (shoulder / taper / off-road) |
| `headingRel` | Float32 | 1 | rad, car heading minus lane direction, + = toward the LEFT (= −`he` of `src/brain/dynamics.js`) |
| `steerAngle` | Float32 | 1 | rad at the front wheels, + = left, clamped ±`maxSteer` |
| `lane` | Int8 | 1 | DERIVED: lane under the centre, `floor(lat/3.7)` clamped to [0, lanes−1] |
| `laneLo`, `laneHi` | Int8 | 1 | DERIVED: footprint lanes. half-extent `he = ½(width·|cos hr| + length·|sin hr|)`; lo = floor((lat−he)/3.7), hi = floor((lat+he−1e-4)/3.7), clamped to [0, lanes−1], at most `MAX_OCC = 4` lanes |
| `seg` | Int32 | 1 | polyline segment hint for `edgePoint` |
| `speed` | Float32 | 1 | u: forward body velocity of the CG, m/s ≥ 0 (no reverse gear) |
| `vLat` | Float32 | 1 | v: lateral body velocity, m/s, + = left (tyre slip / sliding) |
| `yawRate` | Float32 | 1 | r: rad/s, + = turning left (world heading changes by −r) |
| `axPrev` | Float32 | 1 | longitudinal acceleration of the last physics sub-step (weight transfer) |
| `accel` | Float32 | 1 | m/s², (u_end − u_start)/dt of the last control step |
| `mass, inertia` | Float32 | 1 | kg, kg m² (`VEHICLE_PARAMS[type].m / .Iz`) |
| `hl, hw` | Float32 | 1 | hitbox half length / half width (m) |
| `px, pz, pyaw` | Float32 | 1 | world pose at the start of the last integration (swept collisions) |
| `power` | Float32 | 1 | engine output (W) of the last sub-step (fuel) |
| `x, y, z, heading` | Float32 | 1 | DERIVED world pose: `edgePoint(e, s, e.rightEdge − lat)`, heading = road heading − headingRel |
| `length, width` | Float32 | 1 | m (type) |
| `wheelbase, maxSteer, maxSteerRate` | Float32 | 1 | m, rad, rad/s (`src/brain/dynamics.js VEHICLE_DYNAMICS[type]`) |
| `maxSpeed` | Float32 | 1 | m/s (type) |
| `maxAccel` | Float32 | 1 | DRIVER comfort acceleration (built-in driver only): min(delta.maxAccel × min(1, 2400/m), 0.9 F0/m) |
| `maxDecel` | Float32 | 1 | DRIVER hardest braking: min(delta.maxDecel × emergencyDecel/8 × 1.6, emergencyDecel = min(brakeDecel, μ g)) |
| `reactionTime, headwayPref, aggression, patience` | Float32 | 1 | driver traits (delta[2], [3], [0], [1]) |
| `steerRate, gas, brake` | Float32 | 1 | controls APPLIED this step (after the delay) |
| `ctrlRing` | Float32 | 40×3 | reaction-time ring: record for step k at `((slot·40 + k mod 40)·3 + {0 steerRate, 1 gas, 2 brake})` |
| `delaySteps` | Uint32 | 1 | round(reactionTime / dt), ≤ 39 |
| `cmdStep` | Int32 | 1 | step index of the last command written |
| `lastDecision` | Float32 | 1 | built-in driver: time of the last lane-target decision |
| `targetLane` | Int8 | 1 | built-in driver: lane it steers to (−1 none); remapped at edge hops |
| `laneEndDist` | Float32 | 1 | m to the node where my lane ends (lanes are paved to the node, src/geom run-out beyond; 1e9 none) — observation |
| `mergeCap` | Float32 | 1 | m/s: speed that still allows the lane changes a lane end on a later short edge requires (1e9 none; built-in driver) |
| `ctlApproach` | Int32 | 1 | next traffic-control approach on my route within 250 m (controls.js index, −1 none) |
| `ctlType` | Int8 | 1 | −1 none, 0 signal, 1 stop, 2 give_way, 3 ramp_meter, 4 crossing |
| `ctlState` | Int8 | 1 | 0 green, 1 yellow, 2 red, 3 sign, 4 dark (for my movement there) |
| `ctlDist` | Float32 | 1 | m from my front bumper to the stop line |
| `ctlTimeIn, ctlTimeLeft` | Float32 | 1 | s in the current light state / until it changes |
| `ctlMove` | Int8 | 1 | my movement at that control: 0 through, 1 left, 2 right |
| `ctlStopped` | Uint8 | 1 | 1 once I came to a full stop within 6 m of that stop line |
| `ctlInside, ctlInsideT` | Int32 / Float32 | 1 | signal cluster I entered (its inner stop lines belong to my movement; −1 none) and when (forgotten after 15 s) |
| `laneEndSide` | Int8 | 1 | +1 surviving lanes on my left, −1 right, 0 none |
| `conn` | Int32 | 1 | junction connector (src/geom `J` movement id) I am driving through, −1 none. While ≥ 0: `edge` is still the in-edge, `s` keeps counting in the in-edge frame (t = s − sIn along the connector, up to sIn + J.mLen), `lat` keeps its offset from the in-lane centre, pose = `J.arcPoint(conn, t)` offset left by `lat − (inLane+0.5)·3.7`; no edge barriers, no lane changes |
| `passWreck` | Int32 | 1 | slot of the wreck I am passing through the opposing lane of a two-lane road (−1 none); while set I steer to `lat = 1.5·3.7` (the oncoming lane), at most 5 m/s, and oncoming cars in their inner lane see me as a stationary obstacle (`leader −3`) |
| `taper` | Float32 | 1 | synthetic graphs only (no src/geom): s up to which an ended lane still exists past its node (0 none) |
| `routePos` | Int32 | 1 | index into my route of `edge` |
| `routeOff, routeLen` | Int32 | 1 | my route = `RouteStore.flat[routeOff .. routeOff+routeLen)` (−1 = not in the store, e.g. wrecks) |
| `routes[]` | JS array | 1 | `Int32Array` view of the same route (CPU convenience) |
| `routeDist, routeTotal` | Float32 | 1 | m driven along / total length of the current route |
| `commuter` | Int32 | 1 | commuter id (−1 wreck) |
| `leg` | Uint8 | 1 | 0 home→work, 1 work→home |
| `spawnTime, odometer` | Float32 | 1 | s, m |
| `stoppedSince` | Float32 | 1 | time speed fell below 0.2 m/s, −1 while moving |
| `crashedUntil` | Float32 | 1 | scene clear time |
| `leader` | Int32 | 1 | slot ahead (−1 none, −2 lane-end obstacle, −3 junction conflict point: stationary obstacle), refreshed each step |
| `gap` | Float32 | 1 | bumper gap to `leader` (m), 1e9 none |
| `leader2, gap2` | Int32 / Float32 | 1 | the car in front of my leader in its lane (same edge; −1 / 1e9 none) |
| `pullUntil` | Float32 | 1 | sim time until which a driver in a minor freeway crash waits on the shoulder (0 none) |
| `fuel, tank` | Float32 | 1 | litres |
| `refuelState` | Uint8 | 1 | 0 none, 1 driving to station, 2 queued, 3 pumping, 4 filled (re-entering) |
| `station` | Int32 | 1 | station index (`sim.stations.list`), −1 |
| `stopPos` | Int32 | 1 | route index whose edge ends at the station, −1 |
| `pumpUntil` | Float32 | 1 | queue-entry time (state 2) / pump end time (3, 4) |
| `lastFuelCheck` | Float32 | 1 | time of the last refuel check |

Removed in layout 3 / 4 (do not read): `lane` as an authoritative lane, `laneFrom`, `laneT`,
`laneChangeCooldown`, `throttle`, `emergencyDecel`, `dragK` (v4).

## Route packing (`routeStore.js`)

`sim.routeStore.flat: Int32Array` of directed edge ids. `[0, baseLen)` = the population pack:
the route of trip `k = 2·commuter + leg` is `flat[popOffset[k] .. popOffset[k+1])`
(`popOffset: Uint32Array(2n+1)`, the same as `population.routes.offset`). Routes made during the run
(refuel detours) are bump-allocated at `used`; on overflow the dynamic region is compacted
(live routes relocated, `routeOff` updated, `flat` replaced — re-upload after a compaction,
`routeStore.compactions` counts them).

## Lane lists (`spatial.js LaneIndex`) — rebuilt every step after spawning

Key `k = edge·8 + lane`. Every DRIVING / CRASHED car is entered once per lane in `laneLo..laneHi`.
`order[lstart[k] .. lend[k])` = slots in that (edge, lane), sorted by `s` ascending, ties by slot.
`entPos[slot·4 + (lane − laneLo[slot])]` = index of that car's entry in `order`. Untouched keys have
`lstart = lend = 0` (reset each rebuild). `ahead(i, lane)` = `order[entPos+1]` if < `lend`,
`first(edge, lane)` = `order[lstart]`, `last` = `order[lend−1]`, `neighborsAt(edge, lane, s)` = binary search.
Spatial hash (`SpatialHash`, 25 m cells, 2^18 table, counting sort) holds DRIVING/CRASHED cars; it is
only used by `carsNear` / `obb` collision diagnostics, not by the default step.

## Junction lane mapping (`roadGraph.js hopMap / mapLane / mapLat`)

For a hop `e → en`: **strict** (both on the freeway system and |turn| < 0.5 rad, or a plain
1-in/1-out continuation): lane l → l − shift, `lat' = lat − 3.7·shift`; shift from diverge/merge
lane blocks (out-edges / in-edges sorted right-to-left by turn angle; see the code comment) or
`e.lanes − en.lanes` for a plain continuation. **junction** (intersection): `lat' = lat·en.lanes/e.lanes`,
lane l → floor((l+0.5)·en.lanes/e.lanes). A lane mapping outside en ENDS at the node: it tapers out over
`TAPER = 40` m of en (`vs.taper`).

## Per-step order (`Simulation.step`, layout 4)

1. `time += dt; stepCount++`
2. spawn: station re-entries (`_leaveStations`), pending + due departures (`_trySpawn`); new cars get
   `savePose` (start pose = current pose)
3. `hash.rebuild` (DRIVING + CRASHED, positions from the previous integration)
4. **contacts** (`_contacts`): for each DRIVING / CRASHED body, hash neighbours j > i within
   `hd_i + 6.5 + sweep_i + 5` m, `|y_i − y_j| ≤ 2.5`, not both static; skipped: same edge with
   `|s_i − s_j| > hl_i + hl_j + 1` (offset-fold artefact) and freeway-vs-non-adjacent edges (grade separation).
   `sweptContact(start pose → current pose, ≤ 0.5 m relative sub-steps)`; at the first overlapping
   fraction the bodies are rewound there, `resolveContact` (e = 0.2, μ = 0.5, angular) and `separate`
   (80 % of depth − 1 cm). Body velocity from slot: `v = u·f + vLat·l`, `f = (cos ψ, sin ψ)`, `l = (sin ψ, −cos ψ)`,
   `ω = −yawRate`; back: `u = max(0, v·f)`, `vLat = v·l`, `yawRate = −ω`; position change → `s += Δ·t_road`,
   `lat += Δ·left_road`. Disabled cars at rest are static (invMass 0). Barriers (`_barrier`, DRIVING only):
   pavement bounds from `laneGeom.bounds(e, s)` (incl. tapers), right barrier = right pavement edge − 1 m
   shoulder; left = left pavement edge + 1 m (one-way) or + opposing lanes + 0.5 m (two-way); the
   footprint's lateral extent `|sin hr|·hl + |cos hr|·hw` vs the barrier → static-wall contact at the
   outermost corner (e = 0.15, μ = 0.4), lat pushed back by the depth. A NEW contact (pair not touching the
   previous step, 30 s per-pair cooldown, at least one DRIVING car, max Δv ≥ 1 km/h) is a crash event;
   cars with own Δv ≥ 10 km/h become CRASHED + DISABLED
5. `lanes.rebuild` (footprint lists)
6. leaders (`_findLeaders`): footprint lanes, cross-edge (mapLane), zipper merges, wrecks within 15 m of
   the merge node on other in-edges, lane ends (`_laneEnd`, ≤ 450 m ahead: obstacle at the taper start
   − 15 m, or at the node once merging), scene records, + 15 m merge room before stopped obstacles
6b. controls (`_controlObs`, when `opts.controls`): next control per car, light state via the signal plan
   (controls.js `stateOf`), violation events when a stop line is crossed on red (> 0.5 s) / without stopping at a sign
7. drive: brain Driver (`{steerRate, gas, brake}`) or built-in `idmControl` → `vs.command(i, sr, gas, brake, step)`
8. integrate: `savePose`; delayed controls (out of fuel: gas 0, brake ≥ 0.3);
   `physics.stepDynamics(state, sr, gas, brake, VEHICLE_PARAMS[type], dt, grade(edge, s))` (10 ms sub-steps)
   → `speed, vLat, yawRate, steerAngle, axPrev, headingRel, lat, s`, fuel −= litres; edge hops (`mapLat`:
   strict = lat − 3.7·shift, junction = world re-projection kept on the pavement); `updateLanes`.
   CRASHED cars that still move integrate with `locked = true` (sliding to rest). Then arrival / refuel
   stop / stranding / stuck (30 min) / refuel planning; `updatePose` (`laneGeom.latPoint`)
9. scenes: pumps, scene timers, lane masks of sliding wrecks refreshed; freeway density sample every 60 s

## Crash / fuel / stations

Scene record (`sim.crashes`, `sim.activeCrashes`, `sim.edgeCrashes: Map(edge → records)`,
`sim.edgeHasCrash: Uint8Array(edges)`): `{edge, s, laneMask (bit per lane), responseState 'active'|'shoulder'|'cleared',
squeeze, blockStart, blockEnd, clearEnd, squeezeAt, vehicles[], severity, impactSpeed, lanesBlocked, isCrash}`.
Stations: `sim.stations` (`fuel.js StationNetwork`): `list[k] = {node, pumps, x, z}`, `busy: Int32Array`,
`queue[k]: slot[]` FIFO.

## Input version 3 perception (docs/perception-design.md)
CPU `src/sim/perceive.js` reads the state at the decision: bodies from `sim.hash` (positions of the last
integration, the same state as the GPU's contacts hash), `vs.brake` (applied pedal) for brake lamps, route + junction
movement (`movementAhead`) for indicators / conflicts, `vs.ctrlRing` at `stepCount - delaySteps` and `stepCount - 1`
for the own-command inputs. GPU: `perceive.wgsl` + `sense.wgsl senseV3`; leader-history records after the input rows
of the `inputs` buffer (`LH_OFF`, 48 floats per slot).

## Changelog

* v1 (Oct 19): first publication was skipped; lane/laneFrom/laneT model (pre steering).
* v2 (Oct 19): reaction ring stored a single throttle per step.
* v3 (Oct 19): steering only. Added `lat, headingRel, steerAngle, laneLo, laneHi, wheelbase, maxSteer,
  maxSteerRate, steerRate, gas, brake, targetLane, laneEndDist, laneEndSide, taper, routeOff, routeLen,
  routeDist, routeTotal, fuel, tank, refuelState, station, stopPos, pumpUntil, lastFuelCheck`;
  `ctrlRing` stride 3 `[steerRate, gas, brake]`; `lane` is derived; lane lists are footprint-based
  (`entPos` per occupied lane); `leader = −2` lane-end obstacle; dynamics = `src/brain/dynamics.js`.
  Removed `laneFrom, laneT, laneChangeCooldown, throttle, emergencyDecel, dragK`.
* **v4 (Oct 20)**: full physics (`src/sim/physics/`). Added `vLat, yawRate, axPrev, mass, inertia, hl, hw,
  px, pz, pyaw, power`, flag bit5 DISABLED. `speed` is now the body-frame forward velocity u; motion =
  dynamic bicycle (slip angles, saturating tyres, friction circle, weight transfer, power-limited traction,
  aero / rolling / grade) in 10 ms sub-steps; fuel from engine power via BSFC. Collisions are rigid-body
  contacts only (swept SAT, impulses, barriers); removed the lane-gap collision check, contactSpeed bumps,
  the stochastic accident model and the rule-based severity (severity = per-vehicle Δv; minor < 10 km/h
  drivable, moderate 10–25, severe 25–50, very severe > 50; clearance 15–30 min / 30–60 / 1–3 h / 2–5 h).
  `maxAccel / maxDecel` are now driver limits only. World pose from `src/geom` (`laneGeom.latPoint`) and hop
  lane mapping from `src/geom/hops.js`. Lane ends: the taper is BEFORE the node (geometry), so
  `laneEndDist` is the distance to the taper start. Removed `dragK`. VEHICLE_TYPES keeps the legacy
  v3 fields (maxAccel, maxDecel, fuelBase, fuelAero, idleLps) for compatibility only.
* **v5 (Oct 20)**: traffic controls (`src/sim/controls.js`, data/controls.*): added `ctlApproach, ctlType, ctlState,
  ctlDist, ctlTimeIn, ctlTimeLeft, ctlMove, ctlStopped` (observation; never enforced) and `mergeCap`. `laneEndDist`
  now measures to the node (src/geom paves lanes to the node; run-outs beyond). Built-in driver: IDM judged from
  the state the queued ring commands lead to (`driver.js predictLongitudinal`): `gapP = gap − d_pred + v_L·τ`,
  `v = v_pred` in the IDM interaction and the emergency brake (replaces the 0.6 anticipation factor).
  Road grade = clamp(Δy / max(40 m, segment), ±10 %) (superseded 2026-10-24: 12 m window, ±35 %, see the end). Tyre side forces fade out below 5 km/h (dynamics.js).
* **v6 (Oct 20)**: incident realism (docs/realism.md §7, `crashModel.js incidentTimeline`): CHP notified 3.4 min,
  on scene +6.0 min; clearance moderate (PDO) 10–30 min median 20, severe (injury) 45–75 min median 50, very severe
  (major) 2–4 h median 150; lanes stay blocked until the wrecks are moved (PDO: responders + ~5 min; injury 80 %,
  major 90 % of the clearance), then the wrecks are towed onto the right shoulder (`lat = −(hw + 0.3)`, off every lane
  list). A car whose body is entirely outside the travel lanes has `laneHi = laneLo − 1` (in no lane list).
  Freeway shoulders 3 m (`freewayShoulder`), surface roads 1 m. Minor freeway crashes (Δv 3–10 km/h): drivers pull
  onto the shoulder for 5–15 min (`pullUntil`), then continue. Added `leader2, gap2, pullUntil`; the built-in IDM
  also keeps a gap to `leader2`. Contacts between a freeway-system edge and a non-adjacent edge only for pairs in
  data/crossings.json with dy < 2.5 m or in its `overlaps`.
* **v7 (Oct 20)**: traffic controls: signals within 60 m are one controller (`controls.clusterOf`, shared plan);
  every stop line per edge is perceived (`edgeApStart/edgeApList`); a car that crossed a signal's stop line skips the
  inner stop lines of that cluster (`ctlInside, ctlInsideT`); violations count the front-bumper crossing (prevDist ≥ 0
  → < 0) once per approach. Built-in driver: stop-line control brakes to the line from the delay-predicted state and
  holds the brake when stopped within 8 m (no IDM creep); steering predictor forward-simulates the queued ring
  commands through `physics.stepDynamics` (50 ms prediction sub-steps); pull-overs only on motorway / trunk mainline,
  crawling until the body is off the lanes. Crash records carry `type` (barrier | rear-end | sideswipe | angle |
  head-on, `-wreck` when a disabled car is involved), `maneuver` (lane-change | merge | junction | queue-approach |
  cruise | wreck) and `roadClass`. `stats.stuckBy` classifies the car holding each stuck queue.
* **v8 (Oct 20)**: physical junctions (src/geom `junctions.js`, `graph.J`, per edge `sStop / sExit / gore`).
  Wreck perception is body-based: `_blocking` / `blockedAhead` skip a car only when its body is off the lanes
  (`laneHi < laneLo`), never because of `FLAG_SHOULDER`. Squeeze / tow move wrecks onto the paved shoulder
  (freeways, ≥ 15 m from either node) or haul them away where the shoulder is too narrow. New field `conn`:
  a car passing `e.sStop` with a junction movement for (edge, lane, next edge) follows the connector through the
  box and enters the out-edge at `s = sOut` with `lat = (outLane + 0.5)·3.7 + offset` (no pose jump). Cars spawn at
  `sExit`. Waiting / yielding cars stop with the front bumper at `sStop`. Perception near junctions: any stationary
  or crawling body touching my swept connector path, cars on conflicting movements (merge = follow the one nearer
  the merge point; crossing = stop short while it is on the point, or when it arrives first), `leader = −3` for a
  crossing point. Entry yield (`_yieldJ`) uses the conflict matrix with no time-outs.
  Drivers who see a full closure on their route within 300 m re-plan from the end of their edge around every
  closed edge (`opts.reroute`, `stats.reroutes`). On two-lane roads a car stopped behind a wreck that closes its
  lane passes it through the opposing lane when that lane is clear for the whole pass (`passWreck`). Crash counting:
  one crash per pair of cars (a repeat contact of the same pair counts only if it disables a car;
  `stats.repeatContacts`), `stats.disablingCrashes` separately; the `lane-change` maneuver label needs the striking
  body to straddle a lane line.
  Strict diverges are physical: a car whose lane does not continue onto the planned out-edge but onto another
  one takes that edge (`missedExit` event, `stats.missedExits`), and the sim re-plans from it (`reroute` event,
  reason `missedExit`; brain drivers get `wrapper.setRoute`). Brain `driver.rerouteRequest` (edge list from the
  current edge) is applied by the sim (`reroute`, reason `brain`). Ramps that merge into the mainline's lanes
  (`e.gore >= 0`) wait at the gore point when the mainline car going first leaves no room. Brain obs (v8):
  `oncoming {gap, closing}`, `passingWreck`, `junction {mouthDistance, inBox, distance, timeToConflict, stopped}`.
  Minor-contact crash records carry `commuters, x, y, z` like disabling ones.
  Waiting is normal traffic: a car stopped > `stuckTimeout` is only labelled (`stats.stuck`, `stuckBy`, a `stuck`
  event); trips fail only by crash, stranding or not arriving by the end of the run (headless default 05:00–13:00).
  Opposing-lane passing is off by default (`opts.passWrecks`). Live traffic: `sim.trafficPenalty` (Map edge →
  extra seconds, from a ~2-minute moving average of observed edge speeds, refreshed every 60 s) feeds re-routing
  (`router(from, goal, avoid, penalty)`); cars stopped > 90 s re-plan when the new way is > 60 s faster.
  Per-edge lane widths: every lane centre / lane↔lat mapping uses `roadGraph.laneW(e, s)` (src/geom
  `LG.laneWidth`), 3.7 only on synthetic graphs. Junctions load from data/junctions.bin (built in memory if absent).
  Training obstacles: `Simulation.placeObstacles({count, kinds, seed})` (env option `obstacles`); scenes with
  `blockDuration >= 1e8` are permanent (never squeezed or towed).
* **v9 (Oct 20)**: action space = target wheel POSITION. Ring record `[steer, gas, brake]`: in `actionMode`
  'position' (default; `Simulation` opt `actionMode`, store field `vs.actionMode`) steer ∈ [−1, 1] is the target
  wheel angle as a fraction of `maxSteer`; 'rate' keeps the legacy steerRate meaning (transition flag for old
  checkpoints; a `{steerRate}` driver in position mode is integrated: target += steerRate·maxSteerRate·dt/maxSteer,
  `sim.steerCommand`). Actuator (physics/dynamics.js, every 10 ms sub-step h, after the reaction-delay ring):
  `sl += (steer·maxSteer − sl)·min(1, h/steerLag)` (first-order lag: arms), then
  `steerAngle += clamp(sl − steerAngle, ±steerSlew·h)` (road-wheel rate limit), clamped to ±maxSteer.
  Params (physics/params.js): sedan steerSlew 0.80 rad/s, suv 0.73, truck / bus 0.53; steerLag 0.10 s
  (0.12 heavy): full lock → centre ≈ 0.75 s (cars) / 0.95 s (heavy), lock → opposite lock ≈ 1.5 s; ≈ 730°/s at
  a 16:1 hand-wheel (cars), ≈ 600°/s at 20:1 (heavy). New per-vehicle fields: `steerSl` (rad, lagged target)
  and `steerTarget` (applied command after the delay). Built-in driver outputs `{steer, gas, brake}` (pure
  pursuit desired angle / maxSteer); brain obs add `steerTarget` and `actionMode` (obs.steerAngle = actual).
  Also: junction movements leave the in-edge at the movement's own sIn (J.describe; may differ from sStop),
  waiting stops at min(sStop, sIn), a car already past sIn on a short edge joins the connector at the point
  nearest in position and heading; no edge barriers inside the box parts of edges (s < sExit + 1 or s > sStop);
  wrong-lane turns on edges ≥ 80 m become missed turns (at most once per 5 min per car).
* **v9 + 2026-10-23 (barrier-crash fixes, docs/barrier-crashes.md; layout unchanged)**: brake system in
  physics/dynamics.js (mirrored in physics.wgsl): EBD front share = max(brakeFront, Fzf/(Fzf+Fzr) + 0.05) ≤ 0.95;
  ABS holds braking at 0.95 μFz (traction still 0.98); while braking the rear brakes only take what the friction circle
  leaves after the rear side force at the current slip angle (cornering brake control). Brain observation on a
  junction connector (obs.js fillObs, GPU sense.wgsl): laneIndex 0, laneCount 1, laneOffset = lat −
  (inLane + 0.5)·W(in-edge, sIn) (the movement's path), no side lanes, no lane-end report (`obs.onConnector`).
  GPU: brain inputs 65–69 (junction obs) are now written (jyield.wgsl junctionInputs); they were always 0 before.

* **v9 + 2026-10-24 (real elevation, docs/elevation.md; layout unchanged)**: road heights are real (data elevation model
  `terrain/1`: USGS 3DEP bare earth, SF up to ~31-41 % grades). Road grade = rise over a 12 m window centred on s,
  clipped to the edge (`vehicle.js edgeGrade`, GPU `common.wgsl edgeGrade`), clamped to ±35 % (was Δy / max(40 m,
  segment) clamped to ±10 %). physics/dynamics.js (mirrored in physics.wgsl): the slope shifts axle load
  (`Fzf -= m g sinθ hcg / L`); at standstill the brakes (and a locked wreck's tyres) hold against gravity on a
  downhill (before: only against the drive force, so stopped cars crept downhill); `s` advances by the horizontal
  component `u cosθ` (s is horizontal arc length). Still no reverse gear: on an uphill a car with too little drive
  stays put (hill-hold), it never rolls back. Junction connectors keep grade 0.
