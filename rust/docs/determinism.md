# Same-seed determinism of the GPU sim (investigation)

**Problem.** The prefetch agent ran the same `bayenv-gpu` binary twice on the streaming full population (1.38 M
commuters, from 07:30, about 100 k cars on the road, `refuel_budget_ms = inf`). The two runs gave different `events()`
after about 1 sim minute. Same-seed runs must be bit-identical, because training reproducibility depends on it.

**Tool.** `crates/bayenv-gpu/examples/determinism.rs` steps two sims with the same options in lockstep:

```
determinism <data> scatter|pop|stream [n] [start_h] [calls] [steps]
```

After every call it compares the full GPU state (veh, par, pose, ctl), the host tables (slot trips, slot routes,
route mirror) and the event records. At the first difference it prints the call, the buffer, and for veh the first
differing slots and fields; for a different leader it also prints the candidate cars. `DET_OPTS` and
`BAYENV_GPU_OPTS` switch single fixes off. Two separate processes give the same per-call hashes, so the process
(HashMap seeds, thread scheduling) does not matter once the sim is deterministic.

## Root causes

With every determinism fix off (the JS kernels and host order), all scales diverge:

| scale | first difference |
|---|---|
| 10 k scatter, 07:30 | step 333: one car's `speed`, last ulp (contact resolution order) |
| 10 k scatter, collisions off | deterministic for 2,400 steps (only the junction list links in ctl differ) |
| 20 k population | step 1..20: different cars spawned (lane claims) |
| 20 k population, default `refuel_budget_ms = 4` (old Rust default) | call 12: a refuel plan applied in one run only |
| 1.38 M streaming, 07:30 | step 2..3: different leaders at merges and junctions |

Every source is a race between GPU threads, or an order that comes out of atomics and feeds back into the state.
There are also two host-side sources.

1. **Contact resolution order.** `src/gpu/kernels/contacts.wgsl:348` `prio(k)` takes random bits of the contact's
   list index `k`. Detect assigns `k` with `atomicAdd` (`contacts.wgsl:254`), so the order differs from run to run. A
   contact resolves in a parallel round (`resolveRound` / `kernels/resolve_fused.wgsl`) when it holds the minimum
   priority on both of its cars. Two contacts that share a car (a queue pressed together, a car between a wreck and
   the kerb) are therefore resolved in a run-dependent order, and the f32 impulses do not commute.
2. **Spawn and station-exit lane claims.** `src/gpu/kernels/spawn.wgsl:72` claims a lane with `atomicExchange`. When
   two spawns, or two cars leaving a station, want one lane in the same step, the first thread wins.
3. **Event order on the host.** The kernels `emit()` events with `atomicAdd` (`common.wgsl:486`), so the order is
   run-dependent. The host handles them in that order (`sim.rs` `ingest`), and the order feeds the spawn-queue free
   list, the slot free list, the reroute and refuel queues, and the route arena. In the next steps it changes which
   slot and which queue entry a departure gets.
4. **Perception read / write races and list-order ties:**
   * `leaders.wgsl`: thread i reads another car's control record (`signedL` / `mustStopL` in the zipper merge, and
     `convergeBodies`, ctl words 1, 2, 5, 6, 8; `leaders.wgsl:265`, `:289-296`). That car's thread rewrites the record
     in the same pass: the new-occupant reset at `:27-32` and `controlObs` at `:240`. This is the 1.38 M first
     difference, at step 2.
   * `courtesy.wgsl:40-45`: `apply` recomputes the proposal from the leader and gap that other `apply` threads are
     writing. Two equal minimum gaps both write b's leader.
   * `junction.wgsl` `jperceive` (`:64`, `:150`, `:192`, `:255`) and `jyield.wgsl` `prioJ` (`:42`) read another car's
     leader, gap and conflict partner while that car's thread writes its own.
   * Ties in iteration order: `jlink` builds the per-movement, wreck-passer and off-lane lists with
     `atomicExchange` (`junction.wgsl:38-47`), and `hcount` ranks the cars of a hash cell with `atomicAdd`
     (`hash.wgsl:24`). Perception then keeps the first of equal gaps (`junction.wgsl:74` and others) or the first 32
     candidates. Two cars that leave a station onto one lane both sit at s = 0, so an equal gap is common.
5. **Wall-clock refuel budget.** `GpuOptions.refuel_budget_ms` defaulted to 4 ms (the JS `refuelBudgetMs`): the
   number of refuel plans made per frame depended on wall-clock time. Python `Env` already used `inf`; `VecEnv` and
   Rust users got 4 ms.

Things that were checked and are deterministic: the route prefetch workers (`RouteProvider`, results per OD), the
batched reroute and refuel planning (`0f6b9a5`), the HashMap uses on the host (lookups only; iteration goes through
order vectors), `build.sortLanes` (ties broken by slot), and the scans.

## Fix: `GpuOptions` flags, default on; `false` = the JS kernels / host order

| flag | change |
|---|---|
| `deterministic_contacts` | `kernels/contact_order.wgsl`: the priority is a hash of the pair (i, j) in the high bits and i + j in the low 18 bits, which is unique among the contacts that share a car. Needs capacity ≤ 131,072 (checked). |
| `deterministic_spawn` | `kernels/spawn_order.wgsl`: `spawnClaim` / `spawnCommit` and `leaveClaim` / `leaveCommit` passes (`kernels::DET_ROUNDS` = 1). Each entry picks its lane as `entryLane` does and bids `atomicMax(0xffffffff - id)`. The smallest queue entry or slot wins, and the loser tries again in the next step (the JS: another lane in the same pass). `laneClaim` grows by K + cap + spawn_queue words. |
| `deterministic_events` | host `canonical_event_order`: within each step (stepMark segments), a stable sort of the events by slot. A slot's own sequence is kept: CRASHX_* before CRASH, TRIPX before ARRIVE. |
| `deterministic_leaders` | `leaders_order.wgsl`: other cars' control records come from a snapshot taken before the pass (`ctlSnap`, or `jsortB` with junctions; ctl 25, 26, 27, 15). `courtesy_order.wgsl`: proposals kept from `propose`, ties resolved by `atomicMin(k)`, leader written by `commit`. `junction_order.wgsl` / `jyield_order.wgsl`: other cars' leader / gap / partner come from snapshots (ctl 12 / 13, written by the leaders pass and by courtesy `commit`). `jsortA` / `jsortB` relink the three junction lists in slot order. `hash_order.wgsl`: `hrank` plus a second `hscatter` order every hash cell by slot. |
| `refuel_budget_ms` | default `inf`: every queued plan each frame, as one parallel batch. A finite value is wall-clock dependent and documented as such. |

All of the patches are text patches on the resolved JS sources (`kernels::order_patches`; every original snippet
must occur exactly once). With every flag off, the kernels are the JS kernels: `kernels::tests::order_patches_apply`
checks this. The JS-oracle checks (`host_oracle`, `oracle_run`, `parity_run`, `parity_check`, `opt_check`) set the
four flags to `false`.

## Results

| scale | flags on |
|---|---|
| 10 k scatter, 07:30 | identical for 2,400 steps |
| 20 k population, 07:30 | identical for 2,200 steps; the hash at step 800 is the same in two separate processes |
| 20 k streaming, 07:30 | identical for 1,200 steps |
| 1.38 M streaming, 07:30, 87-100 k cars | identical for 3,600 steps (3 sim minutes); veh, par, pose, ctl, tables and records all equal |

Each flag is needed on its own. With only one of them off, the 20 k population diverges:

| flag off | first difference |
|---|---|
| `deterministic_contacts` | call 32 |
| `deterministic_spawn` | call 1 |
| `deterministic_events` | call 2 |
| `deterministic_leaders` | 1.38 M streaming, step 2 |

**Cost.** `stream_prefetch prof` with 1.38 M commuters from 07:30, about 100 k cars, 2,400 steps, 20 steps per
call, under the bench lock, two alternating pairs of runs. The GPU wait per step goes from 3.37 to 3.65 ms (+0.28 ms, +8 %), and the
wall time per step from 5.02 to 5.33 ms (+6 %). Per-kernel times come from `stream_prefetch prof` with
`BAYENV_GPU_OPTS=profile=1`. The largest additions are `junction.jsortB` (0.11 ms, which includes the control
snapshot), `jsortA` (0.06), the spawn and leave claim / commit passes (0.1 to 0.2), `hrank` plus the second
`hscatter` (0.03) and `courtesy.commit` (0.02).

**Check.** `crates/bayenv-gpu/tests/determinism.rs` is part of `check.sh`. It runs two 20,000-commuter populations
from 07:30 for 800 steps and requires identical state, table and record hashes after every 20 steps. It fails with
`deterministic_spawn`, `deterministic_events` or `deterministic_contacts` off. The leaders races need the
full-population scale; use `examples/determinism.rs` for those.
