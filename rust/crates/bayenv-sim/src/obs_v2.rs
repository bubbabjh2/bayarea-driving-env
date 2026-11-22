//! Input-version-2 brain observation: the flat 70-float vector of the training repo's CPU driver
//! (bayarea-driving-sim `src/brain/driver.js` `Driver.sense` for a position-mode, `inputVersion: 2` driver), fed the
//! way `src/sim/brainAdapter.js` feeds it from the sim. The same 70 floats the GPU sim emits (`src/gpu/kernels/
//! sense.wgsl`, `src/gpu/README.md` "Observation v2"), but the reference here is the CPU JS code, not the GPU kernel.
//! Bit-exact against `tools/rust/v2_oracle.mjs` (test `tests/obs_v2_trace.rs`, bin `v2_check`).
//!
//! # What is ported (where the line is drawn)
//! `Driver.sense` = `Eyes.look` (eyes.js) on the driver's obs copy + guidance + traits + lateness + fuel, then
//! `Driver._adjust` -> `adjustInputs` (adjust.js: delay compensation of 4 / 5 / 41 through the driver's forward model
//! `dynamics.js stepLateral`, lane ends as stopped obstacles, junction conflict obstacle, inputs 44 / 45), then the
//! position-mode overwrite of 41..43 with the road grade inputs (`clampG`). Its inputs are the env obs (`obs.rs
//! fill_obs` + signals), the agent's 64-float profile (traits = delta[0..4]), `VEHICLE_PARAMS` and the per-trip state
//! below. NOT ported (brain-only, after / beside the 70 raw floats): the network (net.js / net3.js, its feature /
//! normalisation layers), the reflex (adjust.js `reflex`), the heuristic teacher (but its input, `eyes.near`, IS
//! exported: [`Near`], via [`V2Encoder::encode_v2_near`] / [`V2Encoder::encode_batch_near`]), rewards, the v3 extension (inputs 70..357, already in `obs.rs fill_obs3`), the in-driver reaction delay
//! (`applyReactionDelay`, off), re-route requests (`_checkReroute`, needs a router + the driver's own policy).
//! Junction heading kinks (63 / 64) are disabled in guidance.js (`KINKS_ENABLED = false`): always 0 / 1.
//!
//! # Per-slot state (one [`TripState`] per trip, kept by [`V2Encoder`])
//! * the route guidance (guidance.js `Guidance`: maneuver list + refuel stop `done` flag), built when the encoder first
//!   sees the trip (the sim's route at that moment, NO refuel stop: brainAdapter's factory ignores `extra.stop`), and
//!   rebuilt whenever the sim's route for the slot changes (refuel detour / re-route: sim.js calls
//!   `wrapper.setRoute(edges, stopInfo)`; with a planned refuel the new guidance carries the 'refuel' stop);
//! * the pending-steer ring (driver.js `_pend` / `_pendN`, `predictDelay`): the last `round(reactionTime / dt)` (max 39)
//!   steering commands the car executed, oldest first. Driver.act pushes its own command; here
//!   [`V2Encoder::record_actions`] pushes the command the env will apply (`sim.actions[slot]`: steer, else steerRate,
//!   else the default action's steer).
//! A trip is identified by (commuter, leg, spawn time); a new trip in a slot starts a fresh state.
//!
//! # Call protocol (as the JS driver loop: act -> step -> sense)
//! every step: `env.act(..)` for the cars, then `enc.record_actions(&env.sim)`, then `env.step(1)`, then
//! `enc.encode_v2(&mut env.sim, slot, row)` for EVERY driving slot (the refuel stop is completed by the first sense
//! that sees the car stopped at the station with no refuel pending, so senses must not be skipped). Encoding calls
//! `obs::observe_one` (fill_obs: like env.observe it refreshes the junction-yield cache, see obs.rs; with the default
//! `deadlockBreak 0` this does not change the trajectory, checked by `tests/obs_v2_trace.rs`).
//!
//! # Layout (`OBS2` = 70 f32; eyes.js INPUT VECTOR LAYOUT, as modified by adjust.js / driver.js)
//! ```text
//!  0 speed / 40                1 speedLimit / 40            2 laneIndex / max(1, laneCount-1)   3 laneCount / 6
//!  4 clamp(predicted laneOffset / laneWidth, -1, 1)   (+ left; delay-compensated, adjust.js)
//!  5 clamp(predicted headingError / (pi/4), -1, 1)    (headingError = heading - laneHeading, + right; delay-compensated)
//!  6 left lane exists (0/1)    7 right lane exists (0/1)
//!  8..19 neighbours, 6 slots x {gap, rel}: ownAhead, ownBehind, leftAhead, leftBehind, rightAhead, rightBehind
//!        gap = min(gap, 120) / 120 (1 = none; 0 / 0 when that lane does not exist), rel = clamp(rel / 20, -1, 1);
//!        own ahead: min with the junction-conflict obstacle (adjust.js), left / right ahead: min with that lane's end
//!        (guidance lane-end lookahead 400 m) as a stopped obstacle (rel = clamp(-speed / 20))
//! 20 min(distance to the next maneuver / 2000, 1)
//! 21..29 next maneuver one-hot: continue, exit, merge, turn-left, turn-right, keep-left, keep-right, arrive, refuel
//! 30..32 lane hint one-hot: left, none, right (side of an exit / turn / keep / refuel within 2000 m)
//! 33 aggression  34 patience  35 preferredHeadway / 3  36 reactionTime   (agent profile delta[0], [1], [3] / 3, [2])
//! 37 running late = clamp((remainingRouteDistance / max(timeToWorkStart, 60) - speedLimit) / speedLimit, -1, 1)
//! 38 clamp(timeToWorkStart / 3600, -1, 1)
//! 39 fuel fraction (0..1)     40 refuel planned (0/1)
//! 41..43 road grade inputs clamp(grade[k] / 0.35, -1, 1) (gradeObs.js, + uphill; position mode, input version 2)
//! 44 own lane end min(distance / 300, 1) (1 = none)   45 its side (+1 surviving lanes left, -1 right, 0 none)
//! 46..50 next traffic control type one-hot (signal, stop, give_way, ramp_meter, crossing; all 0 = none within 250 m)
//! 51..55 its state one-hot (green, yellow, red, sign, dark)   56 min(stop-line distance / 250, 1) (1 = none)
//! 57 min(timeToChange / 30, 1) (0 = none)   58..60 my movement one-hot (through, left, right)   61 made a full stop
//! 62 yield required (0/1)     63 heading kink angle (always 0)   64 kink distance (always 1)
//! 65 min(junction conflict distance / 60, 1) (1 = none)   66 min(other's timeToConflict / 10, 1) (1 = none)
//! 67 that body is stopped (0/1)   68 I am inside the junction box (0/1)   69 min(mouthDistance / 60, 1) (1 = none)
//! ```
use crate::jm;
use crate::leaders::lane_end_ahead;
use crate::obs::{observe_one, observe_one_ro, Obs};
use crate::physics::VEHICLE_PARAMS;
use crate::sim::Sim;
use crate::vehicle::STATE_DRIVING;
use crate::world::World;

pub const OBS2: usize = 70;
pub const NEIGHBOR_OFFSET: usize = 8;
pub const MANEUVER_OFFSET: usize = 20;
pub const HINT_OFFSET: usize = 30;
pub const TRAIT_OFFSET: usize = 33;
pub const LATE_OFFSET: usize = 37;
pub const FUEL_OFFSET: usize = 39;
pub const WHEEL_OFFSET: usize = 41;
pub const LANE_END_OFFSET: usize = 44;
pub const CONTROL_OFFSET: usize = 46;
pub const KINK_OFFSET: usize = 63;
pub const JUNCTION_OFFSET: usize = 65;
pub const RANGE: f64 = 120.0;
pub const SPEED_NORM: f64 = 40.0;
pub const REL_SPEED_NORM: f64 = 20.0;
pub const MANEUVER_DIST_NORM: f64 = 2000.0;
/// guidance.js laneHintHorizon default (brainAdapter passes `brain.laneHintHorizon`, undefined)
pub const LANE_HINT_HORIZON: f64 = 2000.0;
/// Driver._adjust planning lookahead for the side lanes ({maxHops: 8, maxDist: 400}; only maxDist reaches laneEnds.js)
pub const SIDE_LANE_END_LOOKAHEAD: f64 = 400.0;
/// driver.js `_pend` capacity - 1
pub const MAX_PENDING: usize = 39;

/// guidance.js MANEUVER_TYPES indices
pub mod man {
    pub const CONTINUE: u8 = 0; pub const EXIT: u8 = 1; pub const MERGE: u8 = 2; pub const TURN_LEFT: u8 = 3; pub const TURN_RIGHT: u8 = 4;
    pub const KEEP_LEFT: u8 = 5; pub const KEEP_RIGHT: u8 = 6; pub const ARRIVE: u8 = 7; pub const REFUEL: u8 = 8;
}
/// maneuver side (null / 'left' / 'right'); LANE_HINTS index = [left, none, right]
pub const SIDE_NONE: u8 = 0;
pub const SIDE_LEFT: u8 = 1;
pub const SIDE_RIGHT: u8 = 2;

const TAU: f64 = jm::PI * 2.0;
#[inline]
fn wrap_angle(a: f64) -> f64 { let a = a % TAU; if a > jm::PI { a - TAU } else if a < -jm::PI { a + TAU } else { a } }
#[inline]
fn clamp(v: f64, lo: f64, hi: f64) -> f64 { if v < lo { lo } else if v > hi { hi } else { v } }
/// geom.js turnSide (threshold 0): None = 'straight'
#[inline]
fn turn_side(d: f64) -> Option<u8> { if d < 0.0 { Some(SIDE_LEFT) } else if d > 0.0 { Some(SIDE_RIGHT) } else { None } }
#[inline]
fn b(x: bool) -> f32 { if x { 1.0 } else { 0.0 } }

/// Per-edge facts guidance.js reads from the SimGraph edges.
#[derive(Clone, Debug)]
pub struct NavEdges {
    /// `e.navRank ?? e.rank` = nav/graph.js CLASS_RANK[highway] ?? 1
    pub rank: Vec<i32>,
    /// motorway | trunk
    pub major: Vec<bool>,
    /// motorway | trunk | motorway_link | trunk_link
    pub major_or_link: Vec<bool>,
    /// interned `e.ref` / `e.name` (0 = ''). The world packs carry no names: all 0 unless set with
    /// [`V2Encoder::with_names`] (the real world and the scenario graphs have none; the synthetic graph does).
    pub ref_id: Vec<i32>,
    pub name_id: Vec<i32>,
}
pub fn class_rank(highway: &str) -> i32 {
    match highway {
        "motorway" => 9, "trunk" => 8, "motorway_link" => 7, "trunk_link" => 6, "primary" => 5, "primary_link" => 4,
        "secondary" | "secondary_link" => 3, "tertiary" | "tertiary_link" => 2, "unclassified" | "residential" => 1,
        "living_street" | "service" => 0, _ => 1,
    }
}
impl NavEdges {
    pub fn new(g: &World) -> NavEdges {
        let n = g.edge_count;
        let hw: Vec<&str> = (0..n).map(|e| g.highway(e)).collect();
        NavEdges {
            rank: hw.iter().map(|h| class_rank(h)).collect(),
            major: hw.iter().map(|h| *h == "motorway" || *h == "trunk").collect(),
            major_or_link: hw.iter().map(|h| matches!(*h, "motorway" | "trunk" | "motorway_link" | "trunk_link")).collect(),
            ref_id: vec![0; n], name_id: vec![0; n],
        }
    }
}

/// One guidance.js maneuver (text fields dropped; `to_ref` / `to_name` kept for the keep coalescing).
#[derive(Clone, Copy, Debug)]
pub struct Maneuver { pub ty: u8, pub side: u8, pub at: f64, pub edge_pos: usize, pub stop: bool, pub done: bool, pub to_ref: i32, pub to_name: i32 }

/// guidance.js Guidance over a SimGraph (the parts Driver.sense uses).
#[derive(Clone, Debug, Default)]
pub struct Guidance {
    pub edge_ids: Vec<i32>,
    pub start_dist: Vec<f64>,
    pub total_length: f64,
    pub maneuvers: Vec<Maneuver>,
}

/// Guidance.at(d) result (the fields eyes.js reads)
#[derive(Clone, Copy, Debug)]
pub struct GuideAt { pub next: usize, pub distance_to_next: f64, pub lane_hint: u8,
    /// index of `following` (the maneuver after next, skipping completed ones), None = null
    pub following: Option<usize> }

impl Guidance {
    /// `new Guidance(graph, edges, {stops: stop_node.map(|n| [{node: n, type: 'refuel'}])})`
    pub fn new(g: &World, nav: &NavEdges, edges: &[i32], stop_node: Option<i32>) -> Guidance {
        let n = edges.len();
        let mut sd = vec![0.0f64; n + 1];
        for i in 0..n { sd[i + 1] = sd[i] + g.e_length[edges[i] as usize]; }
        let mut gd = Guidance { edge_ids: edges.to_vec(), total_length: sd[n], start_dist: sd, maneuvers: Vec::new() };
        gd.build(g, nav, stop_node);
        gd
    }

    fn build(&mut self, g: &World, nav: &NavEdges, stop_node: Option<i32>) {
        let edges = &self.edge_ids;
        let n = edges.len();
        let turn_threshold = (30.0 * jm::PI) / 180.0;
        let fork_cone = (45.0 * jm::PI) / 180.0;
        let dest_after_links = |i: usize| -> Option<usize> { (i..n).map(|k| edges[k] as usize).find(|&e| !g.e_is_link[e]) };
        let mut list: Vec<Maneuver> = Vec::new();
        let mk = |ty: u8, side: u8, at: f64, edge_pos: usize, to: Option<usize>| Maneuver {
            ty, side, at, edge_pos, stop: false, done: false,
            to_ref: to.map(|e| nav.ref_id[e]).unwrap_or(0), to_name: to.map(|e| nav.name_id[e]).unwrap_or(0) };
        let mut i = 0;
        while i + 1 < n {
            let (e1, e2) = (edges[i] as usize, edges[i + 1] as usize);
            let node = g.e_to[e1] as usize;
            let from1 = g.e_from[e1];
            let delta = wrap_angle(g.e_head_start[e2] - g.e_head_end[e1]);
            let outs = &g.out_edges[g.out_start[node] as usize..g.out_start[node + 1] as usize];
            let others = || outs.iter().map(|&o| o as usize).filter(move |&o| o != e2 && g.e_to[o] != from1);
            // `others.length === 0 && e1.srcId === e2.srcId`: SimGraph edges carry no srcId (undefined === undefined)
            if others().next().is_none() { i += 1; continue; }
            let (l1, l2) = (g.e_is_link[e1], g.e_is_link[e2]);
            let mut found: Option<(u8, u8, Option<usize>)> = None;
            if !l1 && l2 && nav.major[e1] {
                let main = others().find(|&o| !g.e_is_link[o]);
                let side = match main { Some(m) => turn_side(wrap_angle(g.e_head_start[e2] - g.e_head_start[m])), None => turn_side(delta) };
                found = Some((man::EXIT, side.unwrap_or(SIDE_RIGHT), dest_after_links(i + 1)));
            } else if l1 && !l2 && (nav.major[e2] || nav.rank[e2] >= 5) {
                let ins = &g.in_edges[g.in_start[node] as usize..g.in_start[node + 1] as usize];
                let main_in = ins.iter().map(|&o| o as usize).find(|&o| o != e1 && !g.e_is_link[o]);
                let side = match main_in { Some(m) => if turn_side(wrap_angle(g.e_head_end[e1] - g.e_head_end[m])) == Some(SIDE_LEFT) { SIDE_RIGHT } else { SIDE_LEFT }, None => SIDE_RIGHT };
                found = Some((man::MERGE, side, Some(e2)));
            } else if jm::abs(delta) > turn_threshold && !(nav.major_or_link[e1] && nav.major_or_link[e2]) {
                let side = turn_side(delta).unwrap_or(SIDE_NONE);
                found = Some((if side == SIDE_LEFT { man::TURN_LEFT } else { man::TURN_RIGHT }, side, Some(e2)));
            } else {
                let r2 = nav.rank[e2];
                if let Some(fk) = others().find(|&o| nav.rank[o] >= r2 && jm::abs(wrap_angle(g.e_head_start[o] - g.e_head_end[e1])) < fork_cone) {
                    let side = turn_side(wrap_angle(g.e_head_start[e2] - g.e_head_start[fk])).unwrap_or(if turn_side(delta) == Some(SIDE_LEFT) { SIDE_LEFT } else { SIDE_RIGHT });
                    let to = if l2 { dest_after_links(i + 1).or(Some(e2)) } else { Some(e2) };
                    found = Some((if side == SIDE_LEFT { man::KEEP_LEFT } else { man::KEEP_RIGHT }, side, to));
                } else {
                    let (r1, n1, rr2, n2) = (nav.ref_id[e1], nav.name_id[e1], nav.ref_id[e2], nav.name_id[e2]);
                    if l1 == l2 && (rr2 != 0 || n2 != 0) && (rr2 != r1 || (rr2 == 0 && n2 != n1)) { found = Some((man::CONTINUE, SIDE_NONE, Some(e2))); }
                }
            }
            if let Some((ty, side, to)) = found {
                let at = self.start_dist[i + 1];
                let skip = ty == man::CONTINUE && list.last().map_or(false, |p| at - p.at < 200.0);
                if !skip { list.push(mk(ty, side, at, i + 1, to)); }
            }
            i += 1;
        }
        // coalesce keep-left / keep-right steps
        let is_keep = |m: &Maneuver| m.ty == man::KEEP_LEFT || m.ty == man::KEEP_RIGHT;
        let mut i = 1usize;
        while i < list.len() {
            let (a, bb) = (list[i - 1], list[i]);
            if !is_keep(&a) || !is_keep(&bb) { i += 1; continue; }
            let gap = bb.at - a.at;
            if bb.ty == a.ty && gap < 400.0 {
                if bb.to_ref != 0 || bb.to_name != 0 { list[i - 1].to_ref = bb.to_ref; list[i - 1].to_name = bb.to_name; }
                list.remove(i);
            } else if bb.ty != a.ty && gap < 300.0 && (bb.to_ref != 0 || bb.to_name != 0) && bb.to_ref == a.to_ref && bb.to_name == a.to_name {
                list.remove(i);
            } else { i += 1; }
        }
        // mandatory refuel stop where the route first reaches the stop node
        if let Some(node) = stop_node {
            if let Some(pos) = (0..n).find(|&k| g.e_to[edges[k] as usize] == node) {
                let mut m = mk(man::REFUEL, SIDE_RIGHT, self.start_dist[pos + 1], pos + 1, None);
                m.stop = true;
                let mut k = 0; while k < list.len() && list[k].at < m.at { k += 1; }
                list.insert(k, m);
            }
        }
        list.push(mk(man::ARRIVE, SIDE_NONE, self.total_length, n.saturating_sub(1), None));
        self.maneuvers = list;
    }

    /// Guidance.edgePosAt(d)
    pub fn edge_pos_at(&self, d: f64) -> usize {
        let sd = &self.start_dist;
        let (mut lo, mut hi) = (0usize, self.edge_ids.len().saturating_sub(1));
        if d <= 0.0 { return 0; }
        if d >= self.total_length { return hi; }
        while lo < hi { let m = (lo + hi + 1) >> 1; if sd[m] <= d { lo = m; } else { hi = m - 1; } }
        lo
    }

    /// Guidance.at(d): next maneuver, distance to it, lane hint
    pub fn at(&self, d: f64) -> GuideAt {
        let ms = &self.maneuvers;
        let (mut lo, mut hi) = (0usize, ms.len() - 1);
        while lo < hi { let m = (lo + hi) >> 1; if ms[m].at < d - 1.0 { lo = m + 1; } else { hi = m; } }
        while lo < ms.len() - 1 && ms[lo].done { lo += 1; }
        let mut k = lo as isize - 1;
        while k >= 0 && d - ms[k as usize].at < 15.0 { if ms[k as usize].stop && !ms[k as usize].done { lo = k as usize; break; } k -= 1; }
        let nx = &ms[lo];
        let mut fi = lo + 1; while fi < ms.len() - 1 && ms[fi].done { fi += 1; }
        let following = if fi < ms.len() { Some(fi) } else { None };
        let distance_to_next = jm::max(0.0, nx.at - d);
        let lane_hint = if nx.side != SIDE_NONE && nx.ty != man::MERGE && distance_to_next <= LANE_HINT_HORIZON { nx.side } else { SIDE_NONE };
        GuideAt { next: lo, distance_to_next, lane_hint, following }
    }

    /// Guidance.laneEndAhead(d, lane, {maxDist}): distance to the end of `lane` (sim lane mapping), or None
    pub fn lane_end_ahead(&self, g: &World, d: f64, lane: i32, max_dist: f64) -> Option<f64> {
        self.lane_end_ahead_side(g, d, lane, max_dist).map(|x| x.0)
    }
    /// Guidance.laneEndAhead(d, lane, {maxDist}) as (distance, side), or None
    pub fn lane_end_ahead_side(&self, g: &World, d: f64, lane: i32, max_dist: f64) -> Option<(f64, i32)> {
        let i = self.edge_pos_at(d);
        let stop_at = self.maneuvers.iter().find(|m| m.stop && !m.done).map_or(-1, |m| m.edge_pos as i32 - 1);
        if stop_at == i as i32 { return None; }
        let e = self.edge_ids[i] as usize;
        if lane < 0 || lane >= g.e_lanes[e] { return None; }
        let o = lane_end_ahead(g, &self.edge_ids, i, d - self.start_dist[i], lane, max_dist, stop_at);
        if o.dist < 1e8 { Some((o.dist, o.side)) } else { None }
    }
}

/// driver.js `this.eyes.near` after Driver.sense: the raw neighbour summary (metres, m/s; Infinity = none) eyes.js
/// `look` fills from obs.neighbors, then Driver._adjust mirrors the adjust.js results into it (predicted heading
/// error; lane-end / junction-conflict obstacles as stopped leaders / side-lane leaders; lane-end distances).
/// Field names = the JS keys (camelCase in [`NEAR_FIELDS`]). Values are the JS f64 numbers (gaps / speeds from the
/// f32 obs.neighbors widened, `v + nb[k]` in f64), bit-exact.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Near {
    /// own-lane leader gap (m; min with the own lane end (when not sim-injected) and the junction-conflict obstacle)
    pub leader_gap: f64, pub leader_speed: f64,
    /// leader speed - my speed (nb[1]; -speed when an obstacle is the leader)
    pub leader_rel: f64,
    pub behind_gap: f64, pub behind_speed: f64,
    /// left-lane ahead gap (min with the left lane's end as a stopped obstacle), behind gap, speeds
    pub left_ahead: f64, pub left_behind: f64, pub left_ahead_speed: f64, pub left_behind_speed: f64,
    pub right_ahead: f64, pub right_behind: f64, pub right_ahead_speed: f64, pub right_behind_speed: f64,
    pub left_exists: bool, pub right_exists: bool,
    /// delay-compensated heading error (adjust.js r.he; + right, rad)
    pub heading_error: f64,
    /// adjust.js r.ownGap / r.leftGap / r.rightGap: lane-end obstacle gaps (Infinity = none; ownGap is Infinity when
    /// the sim injects its own lane-end obstacle, i.e. always with the sim's obs)
    pub lane_end_gap: f64, pub left_end_gap: f64, pub right_end_gap: f64,
    /// own lane end: obs.laneEndDistance / laneEndSide when finite, else the 400 m guidance plan (Infinity / 0 = none)
    pub end_dist: f64, pub end_side: f64,
    /// left / right lane end distances (guidance, 400 m lookahead; Infinity = none)
    pub left_end_dist: f64, pub right_end_dist: f64,
}
/// JS keys of `eyes.near`, in [`Near::to_array`] order (booleans as 0 / 1)
pub const NEAR_FIELDS: [&str; 23] = ["leaderGap", "leaderSpeed", "leaderRel", "behindGap", "behindSpeed",
    "leftAhead", "leftBehind", "leftAheadSpeed", "leftBehindSpeed", "rightAhead", "rightBehind", "rightAheadSpeed", "rightBehindSpeed",
    "leftExists", "rightExists", "headingError", "laneEndGap", "leftEndGap", "rightEndGap", "endDist", "endSide", "leftEndDist", "rightEndDist"];
pub const NEAR_N: usize = NEAR_FIELDS.len();
impl Default for Near {
    fn default() -> Self {
        let inf = f64::INFINITY;
        Near { leader_gap: inf, leader_speed: 0.0, leader_rel: 0.0, behind_gap: inf, behind_speed: 0.0, left_ahead: inf, left_behind: inf,
            left_ahead_speed: 0.0, left_behind_speed: 0.0, right_ahead: inf, right_behind: inf, right_ahead_speed: 0.0, right_behind_speed: 0.0,
            left_exists: false, right_exists: false, heading_error: 0.0, lane_end_gap: inf, left_end_gap: inf, right_end_gap: inf,
            end_dist: inf, end_side: 0.0, left_end_dist: inf, right_end_dist: inf }
    }
}
impl Near {
    pub fn to_array(&self) -> [f64; NEAR_N] {
        let bf = |x: bool| if x { 1.0 } else { 0.0 };
        [self.leader_gap, self.leader_speed, self.leader_rel, self.behind_gap, self.behind_speed,
         self.left_ahead, self.left_behind, self.left_ahead_speed, self.left_behind_speed, self.right_ahead, self.right_behind, self.right_ahead_speed, self.right_behind_speed,
         bf(self.left_exists), bf(self.right_exists), self.heading_error, self.lane_end_gap, self.left_end_gap, self.right_end_gap,
         self.end_dist, self.end_side, self.left_end_dist, self.right_end_dist]
    }
}

/// The rest of what the heuristic teacher (heuristicDriver.js act) reads after Driver.sense, besides `eyes.near`:
/// the Driver's private obs copy `o` AFTER Driver._adjust, `this.guide` (Guidance.at(routeDistance), after a refuel
/// stop completion) and `this.late`. All f64, unclamped (the 70 inputs hold some of these clamped / f32-rounded).
/// JS null / undefined = `None`. With the sim's obs `control`, `control.yieldRequired`, `grade` and `guide` are
/// always present and `kink` is always null (KINKS_ENABLED = false); the Options mirror the JS types.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TeacherObs {
    /// o.speed = obs.speed
    pub speed: f64,
    /// o.heading = o.laneHeading + r.he (delay-compensated)
    pub heading: f64,
    /// o.laneHeading = obs.laneHeading
    pub lane_heading: f64,
    pub lane_width: f64, pub speed_limit: f64,
    /// obs.laneIndex / obs.laneCount (raw, not clamped)
    pub lane_index: f64, pub lane_count: f64,
    /// o.laneOffset = r.off, o.steerAngle = r.steer (delay-compensated)
    pub lane_offset: f64, pub steer_angle: f64,
    pub control: Option<TeacherControl>,
    pub kink: Option<TeacherKink>,
    /// o.grade [g0, g1, g2] raw (gradeObs.js, + uphill)
    pub grade: Option<[f64; 3]>,
    pub guide: Option<TeacherGuide>,
    /// this.late.lateFeature (eyes.js lateness: clamp((required - speedLimit) / speedLimit, -1, 1), 0 when undefined)
    pub late_feature: f64,
}
/// obs.control (sim obs.js): type -1 none / 0 signal / 1 stop / 2 give_way / 3 ramp_meter / 4 crossing; state 0 green,
/// 1 yellow, 2 red, 3 sign, 4 dark; distance (Infinity when type -1); movement 0 through / 1 left / 2 right
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TeacherControl { pub ty: f64, pub state: f64, pub distance: f64, pub time_in_state: f64, pub time_to_change: f64, pub movement: f64, pub stopped: bool, pub yield_required: Option<bool> }
/// Guidance.nextKink result {distance, angle} (never produced: KINKS_ENABLED = false)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TeacherKink { pub distance: f64, pub angle: f64 }
/// a guidance maneuver: `ty` = guidance.js MANEUVER_TYPES index ([`man`]), at_distance = m.atDistance
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TeacherManeuver { pub ty: u8, pub at_distance: f64 }
/// Guidance.at(routeDistance): laneHint ([`SIDE_NONE`] / [`SIDE_LEFT`] / [`SIDE_RIGHT`] for 'none' / 'left' / 'right'),
/// distanceToNext, nextManeuver, following (null = None)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TeacherGuide { pub lane_hint: u8, pub distance_to_next: f64, pub next: TeacherManeuver, pub following: Option<TeacherManeuver> }

/// Column names of [`TeacherObs::to_array`] (= the oracle's v2.teacher rows). `*.present` / booleans are 0 / 1;
/// absent values (None) are NaN; laneHint is 0 none / 1 left / 2 right (SIDE_*), maneuver types MANEUVER_TYPES indices.
pub const TEACHER_FIELDS: [&str; 35] = ["speed", "heading", "laneHeading", "laneWidth", "speedLimit", "laneIndex", "laneCount", "laneOffset", "steerAngle",
    "control.present", "control.type", "control.state", "control.distance", "control.timeInState", "control.timeToChange", "control.movement", "control.stopped",
    "control.yieldRequired.present", "control.yieldRequired",
    "kink.present", "kink.distance", "kink.angle", "grade.present", "grade.0", "grade.1", "grade.2",
    "guide.present", "guide.laneHint", "guide.distanceToNext", "guide.next.type", "guide.next.atDistance",
    "guide.following.present", "guide.following.type", "guide.following.atDistance", "lateFeature"];
pub const TEACHER_N: usize = TEACHER_FIELDS.len();
impl Default for TeacherObs {
    fn default() -> Self {
        TeacherObs { speed: 0.0, heading: 0.0, lane_heading: 0.0, lane_width: 0.0, speed_limit: 0.0, lane_index: 0.0, lane_count: 0.0, lane_offset: 0.0, steer_angle: 0.0,
            control: None, kink: None, grade: None, guide: None, late_feature: 0.0 }
    }
}
impl TeacherObs {
    pub fn to_array(&self) -> [f64; TEACHER_N] {
        let bf = |x: bool| if x { 1.0 } else { 0.0 };
        let nan = f64::NAN;
        let c = self.control;
        let cf = |f: fn(&TeacherControl) -> f64| c.as_ref().map_or(nan, f);
        let yr = c.and_then(|c| c.yield_required);
        let g = self.guide;
        let fo = g.and_then(|g| g.following);
        [self.speed, self.heading, self.lane_heading, self.lane_width, self.speed_limit, self.lane_index, self.lane_count, self.lane_offset, self.steer_angle,
         bf(c.is_some()), cf(|c| c.ty), cf(|c| c.state), cf(|c| c.distance), cf(|c| c.time_in_state), cf(|c| c.time_to_change), cf(|c| c.movement), cf(|c| if c.stopped { 1.0 } else { 0.0 }),
         bf(yr.is_some()), yr.map_or(nan, bf),
         bf(self.kink.is_some()), self.kink.map_or(nan, |k| k.distance), self.kink.map_or(nan, |k| k.angle),
         bf(self.grade.is_some()), self.grade.map_or(nan, |x| x[0]), self.grade.map_or(nan, |x| x[1]), self.grade.map_or(nan, |x| x[2]),
         bf(g.is_some()), g.map_or(nan, |g| g.lane_hint as f64), g.map_or(nan, |g| g.distance_to_next), g.map_or(nan, |g| g.next.ty as f64), g.map_or(nan, |g| g.next.at_distance),
         bf(fo.is_some()), fo.map_or(nan, |m| m.ty as f64), fo.map_or(nan, |m| m.at_distance), self.late_feature]
    }
}

/// Per-trip driver state (driver.js Driver: guidance + pending-steer ring).
#[derive(Clone, Debug)]
pub struct TripState {
    /// (commuter, leg, spawn_time bits)
    pub key: (i32, u8, u32),
    pub guide: Guidance,
    pub pend: [f32; MAX_PENDING + 1],
    pub pend_n: usize,
    /// vs.routeOff / RouteStore.compactions at the last sync (setRoute detection)
    pub route_off: i32,
    pub compactions: u64,
}

/// v2 encoder: per-slot trip state + the per-edge guidance facts.
pub struct V2Encoder {
    pub nav: NavEdges,
    pub slots: Vec<Option<TripState>>,
    obs: Obs,
}

#[inline]
fn trip_key(sim: &Sim, i: usize) -> (i32, u8, u32) { (sim.vs.commuter[i], sim.vs.leg[i], sim.vs.spawn_time[i].to_bits()) }

impl V2Encoder {
    pub fn new(g: &World) -> V2Encoder { V2Encoder { nav: NavEdges::new(g), slots: Vec::new(), obs: Obs::default() } }
    /// Road ref / name ids per edge (interned, 0 = ''), for guidance 'continue' maneuvers and keep coalescing.
    pub fn with_names(mut self, ref_id: Vec<i32>, name_id: Vec<i32>) -> V2Encoder {
        assert_eq!(ref_id.len(), self.nav.rank.len()); assert_eq!(name_id.len(), self.nav.rank.len());
        self.nav.ref_id = ref_id; self.nav.name_id = name_id; self
    }
    /// Forget every trip (call after env.reset()).
    pub fn reset(&mut self) { self.slots.clear(); }

    /// The trip state of `slot`, (re)created for a new trip, guidance rebuilt when the sim's route changed.
    fn sync(&mut self, sim: &Sim, i: usize) -> &mut TripState {
        if self.slots.len() <= i { self.slots.resize_with(i + 1, || None); }
        let key = trip_key(sim, i);
        let route = &sim.vs.routes[i];
        let g = &sim.g;
        let st = &mut self.slots[i];
        let (off, comp) = (sim.vs.route_off[i], sim.routes.compactions);
        match st {
            Some(s) if s.key == key => {
                // A sim setRoute (refuel detour / reroute / missed-exit re-plan) always stores the route at a fresh
                // route-store offset; a missed exit's in-place edit (vehicle.js r2[next] = took) sets routeOff -1 and
                // is NOT a setRoute (the JS driver keeps its copy); a store compaction relocates offsets without
                // changing content (then the content decides).
                if off != s.route_off {
                    let set = off >= 0 && (s.route_off < 0 || comp == s.compactions || s.guide.edge_ids != *route);
                    if set {
                        // brainAdapter wrapper.setRoute(edges, stopInfo): the refuel stop when one is planned
                        let stop = if sim.vs.refuel_state[i] == 1 { sim.stations.as_ref().map(|n| n.node[sim.vs.station[i] as usize]) } else { None };
                        s.guide = Guidance::new(g, &self.nav, route, stop);
                    }
                    s.route_off = off;
                }
                s.compactions = comp;
            }
            _ => { *st = Some(TripState { key, guide: Guidance::new(g, &self.nav, route, None), pend: [0.0; MAX_PENDING + 1], pend_n: 0, route_off: off, compactions: comp }); }
        }
        st.as_mut().unwrap()
    }

    /// Driver.act pending-ring bookkeeping (`predictDelay`) for `slot` with its executed steering command.
    pub fn push_command(&mut self, sim: &Sim, i: usize, cmd: f64) {
        let rt = sim.pop.deltas[sim.vs.commuter[i] as usize * 64 + 2] as f64;
        let keep = jm::min(MAX_PENDING as f64, jm::max(0.0, jm::round(rt / sim.dt))) as usize;
        // steer_delay (SimOpts): steering lands after its own, shorter delay, so only that many commands are pending
        // (the JS: the reaction delay)
        let sds = sim.vs.model.steer_delay_steps;
        let keep = if sds >= 0 { keep.min(sds as usize) } else { keep };
        let s = self.sync(sim, i);
        if s.pend_n >= keep {
            if keep > 0 { s.pend.copy_within(s.pend_n - keep + 1..s.pend_n, 0); s.pend_n = keep - 1; } else { s.pend_n = 0; }
        }
        if keep > 0 { s.pend[s.pend_n] = cmd as f32; s.pend_n += 1; }
    }

    /// Per-step hook: after the actions are set (env.act) and before env.step, push every driving car's command
    /// (steer, else steerRate, else the default action's steer) into its pending ring.
    pub fn record_actions(&mut self, sim: &Sim) {
        for i in 0..sim.vs.high {
            if sim.vs.state[i] != STATE_DRIVING { continue; }
            let a = sim.actions.get(i).copied().flatten();
            let cmd = match a { Some(a) => a.steer.or(a.steer_rate).unwrap_or(0.0), None => sim.opts.default_action.steer.unwrap_or(0.0) };
            self.push_command(sim, i, cmd);
        }
    }

    /// Driver.sense for `slot` (a driving car): writes the 70 floats into out[..70]. Call once per step per driving slot.
    pub fn encode_v2(&mut self, sim: &mut Sim, slot: usize, out: &mut [f32]) { self.encode_v2_impl(sim, slot, out, None, None) }
    /// encode_v2 that also writes the driver's `eyes.near` after the sense into `near`.
    pub fn encode_v2_near(&mut self, sim: &mut Sim, slot: usize, out: &mut [f32], near: &mut Near) { self.encode_v2_impl(sim, slot, out, Some(near), None) }
    /// encode_v2 that also writes `eyes.near` and the teacher's view ([`TeacherObs`]: adjusted obs, guide, lateness).
    pub fn encode_v2_teacher(&mut self, sim: &mut Sim, slot: usize, out: &mut [f32], near: &mut Near, teacher: &mut TeacherObs) { self.encode_v2_impl(sim, slot, out, Some(near), Some(teacher)) }
    fn encode_v2_impl(&mut self, sim: &mut Sim, slot: usize, out: &mut [f32], near: Option<&mut Near>, teacher: Option<&mut TeacherObs>) {
        let i = slot;
        let mut o = std::mem::take(&mut self.obs);
        observe_one(sim, i, &mut o, None);
        let sim: &Sim = sim;
        let st = self.sync(sim, i);
        encode_core(st, sim, i, &o, out, near, teacher);
        self.obs = o;
    }

    /// encode_v2 for many driving slots (ascending, distinct) into rows of OBS2 floats: trip states synced in slot
    /// order, then observation + encoding per car in parallel (opts.parallel; each car reads only shared state and
    /// its own trip state), then the observations' state effects applied. Same result as encode_v2 slot by slot.
    pub fn encode_batch(&mut self, sim: &mut Sim, slots: &[usize], out: &mut [f32]) { self.encode_batch_teacher(sim, slots, out, None, None) }
    /// encode_batch that also writes each slot's `eyes.near` into `near[k]` (k = index into `slots`) when given.
    pub fn encode_batch_near(&mut self, sim: &mut Sim, slots: &[usize], out: &mut [f32], near: Option<&mut [Near]>) { self.encode_batch_teacher(sim, slots, out, near, None) }
    /// encode_batch that also writes `near[k]` and / or `teacher[k]` ([`TeacherObs`]) per slot when given.
    pub fn encode_batch_teacher(&mut self, sim: &mut Sim, slots: &[usize], out: &mut [f32], near: Option<&mut [Near]>, teacher: Option<&mut [TeacherObs]>) {
        assert!(out.len() >= slots.len() * OBS2);
        let mut scratch: Vec<Near> = Vec::new();
        let (near, want): (&mut [Near], bool) = match near {
            Some(n) => { assert!(n.len() >= slots.len(), "encode_batch_near: near too short"); (&mut n[..slots.len()], true) }
            None => { scratch.resize(slots.len(), Near::default()); (&mut scratch[..], false) }
        };
        let mut tscratch: Vec<TeacherObs> = Vec::new();
        let (teacher, want_t): (&mut [TeacherObs], bool) = match teacher {
            Some(n) => { assert!(n.len() >= slots.len(), "encode_batch_teacher: teacher too short"); (&mut n[..slots.len()], true) }
            None => { tscratch.resize(slots.len(), TeacherObs::default()); (&mut tscratch[..], false) }
        };
        for &i in slots { self.sync(sim, i); }
        let me: &Sim = sim;
        let mut states: Vec<&mut TripState> = Vec::with_capacity(slots.len());
        {
            let mut it = self.slots.iter_mut().enumerate();
            for &i in slots { loop { let (k, s) = it.next().expect("encode_batch: slots must be ascending"); if k == i { states.push(s.as_mut().unwrap()); break; } } }
        }
        let job = |st: &mut TripState, i: usize, row: &mut [f32], nr: &mut Near, tr: &mut TeacherObs| -> Option<i32> {
            let mut o = Obs::default();
            observe_one_ro(me, i, &mut o, None);
            encode_core(st, me, i, &o, row, if want { Some(nr) } else { None }, if want_t { Some(tr) } else { None });
            o.yield_write
        };
        let writes: Vec<Option<i32>> = if me.opts.parallel && slots.len() >= 128 {
            use rayon::prelude::*;
            states.into_par_iter().zip(slots.par_iter()).zip(out[..slots.len() * OBS2].par_chunks_mut(OBS2)).zip(near.par_iter_mut()).zip(teacher.par_iter_mut()).with_min_len(16)
                .map(|((((st, &i), row), nr), tr)| job(st, i, row, nr, tr)).collect()
        } else {
            states.into_iter().zip(slots.iter()).zip(out[..slots.len() * OBS2].chunks_mut(OBS2)).zip(near.iter_mut()).zip(teacher.iter_mut())
                .map(|((((st, &i), row), nr), tr)| job(st, i, row, nr, tr)).collect()
        };
        for (k, w) in writes.into_iter().enumerate() { if let Some(v) = w { sim.set_yield_by(slots[k], v); } }
    }
}

/// Driver.sense on an observation already filled for slot i (the 70 floats into out[..70]).
fn encode_core(st: &mut TripState, sim: &Sim, i: usize, o: &Obs, out: &mut [f32], near: Option<&mut Near>, teacher: Option<&mut TeacherObs>) {
        let g = sim.gw();
        let delta = &sim.pop.deltas[sim.vs.commuter[i] as usize * 64..sim.vs.commuter[i] as usize * 64 + 4];
        let (aggression, patience, reaction_time, headway) = (delta[0] as f64, delta[1] as f64, delta[2] as f64, delta[3] as f64);
        let dt = sim.dt;
        let rd = o.route_distance;
        // guidance + refuel stop completion (Driver.sense; stopDwell 0)
        let mut ga = st.guide.at(rd);
        {
            let nm = st.guide.maneuvers[ga.next];
            if nm.stop && !nm.done && jm::abs(nm.at - rd) < 10.0 && o.speed < 0.3 && !o.signals.refuel_planned {
                st.guide.maneuvers[ga.next].done = true;
                ga = st.guide.at(rd);
            }
        }
        let inp = &mut out[..OBS2];
        // ---------------- eyes.js look
        let lane_width = jm::or(o.lane_width, 3.7);
        let lane_count = jm::max(1.0, o.lane_count as f64);
        let lane_index = clamp(o.lane_index as f64, 0.0, lane_count - 1.0);
        let heading_error = wrap_angle(o.heading - o.lane_heading);
        let left_exists = lane_index < lane_count - 1.0;
        let right_exists = lane_index > 0.0;
        let v = o.speed;
        let nb = &o.neighbors;
        let gp = |k: usize| -> f64 { let x = nb[2 * k] as f64; if x >= RANGE || !(x < f64::INFINITY) { f64::INFINITY } else { x } };
        inp[0] = (v / SPEED_NORM) as f32;
        inp[1] = (o.speed_limit / SPEED_NORM) as f32;
        inp[2] = (lane_index / jm::max(1.0, lane_count - 1.0)) as f32;
        inp[3] = (lane_count / 6.0) as f32;
        inp[4] = clamp(jm::or(o.lane_offset, 0.0) / lane_width, -1.0, 1.0) as f32;
        inp[5] = clamp(heading_error / (jm::PI / 4.0), -1.0, 1.0) as f32;
        inp[6] = b(left_exists); inp[7] = b(right_exists);
        for k in 0..6 {
            let at = NEIGHBOR_OFFSET + 2 * k;
            let exists = match k { 2 | 3 => left_exists, 4 | 5 => right_exists, _ => true };
            let (gap, other) = (gp(k), v + nb[2 * k + 1] as f64);
            if !exists { inp[at] = 0.0; inp[at + 1] = 0.0; }
            else if gap == f64::INFINITY || gap >= RANGE { inp[at] = 1.0; inp[at + 1] = 0.0; }
            else { inp[at] = (jm::max(0.0, gap) / RANGE) as f32; inp[at + 1] = clamp((other - v) / REL_SPEED_NORM, -1.0, 1.0) as f32; }
        }
        for x in inp[MANEUVER_OFFSET..TRAIT_OFFSET].iter_mut() { *x = 0.0; }
        let nm = st.guide.maneuvers[ga.next];
        inp[MANEUVER_OFFSET] = jm::min(ga.distance_to_next / MANEUVER_DIST_NORM, 1.0) as f32;
        inp[MANEUVER_OFFSET + 1 + nm.ty as usize] = 1.0;
        inp[HINT_OFFSET + match ga.lane_hint { SIDE_LEFT => 0, SIDE_RIGHT => 2, _ => 1 }] = 1.0;
        inp[TRAIT_OFFSET] = aggression as f32;
        inp[TRAIT_OFFSET + 1] = patience as f32;
        inp[TRAIT_OFFSET + 2] = (headway / 3.0) as f32;
        inp[TRAIT_OFFSET + 3] = reaction_time as f32;
        // lateness(signals.timeToWorkStart, signals.remainingRouteDistance, speedLimit)
        let sig = &o.signals;
        let (late, tws) = if !(o.speed_limit > 0.0) { (0.0, 0.0) } else {
            let required = sig.remaining_route_distance / jm::max(sig.time_to_work_start, 60.0);
            (clamp((required - o.speed_limit) / o.speed_limit, -1.0, 1.0), sig.time_to_work_start)
        };
        inp[LATE_OFFSET] = late as f32;
        inp[LATE_OFFSET + 1] = clamp(tws / 3600.0, -1.0, 1.0) as f32;
        inp[FUEL_OFFSET] = clamp(sig.fuel_fraction, 0.0, 1.0) as f32;
        inp[FUEL_OFFSET + 1] = b(sig.refuel_planned);
        let max_steer = VEHICLE_PARAMS[o.vehicle_type].max_steer;   // o.maxSteer = this.vehicle.maxSteer
        inp[WHEEL_OFFSET] = clamp(jm::or(o.steer_angle, 0.0) / jm::or(max_steer, 0.6), -1.0, 1.0) as f32;
        inp[WHEEL_OFFSET + 1] = clamp(jm::or(o.gas, 0.0), 0.0, 1.0) as f32;
        inp[WHEEL_OFFSET + 2] = clamp(jm::or(o.brake, 0.0), 0.0, 1.0) as f32;
        inp[LANE_END_OFFSET] = 1.0; inp[LANE_END_OFFSET + 1] = 0.0;
        for x in inp[CONTROL_OFFSET..CONTROL_OFFSET + 17].iter_mut() { *x = 0.0; }
        let ct = &o.control;
        if ct.ty >= 0 && ct.ty <= 4 && ct.distance < 250.0 {
            inp[CONTROL_OFFSET + ct.ty as usize] = 1.0;
            if ct.state >= 0 && ct.state <= 4 { inp[CONTROL_OFFSET + 5 + ct.state as usize] = 1.0; }
            inp[CONTROL_OFFSET + 10] = clamp(ct.distance / 250.0, 0.0, 1.0) as f32;
            inp[CONTROL_OFFSET + 11] = if ct.time_to_change > 0.0 && ct.time_to_change < f64::INFINITY { clamp(ct.time_to_change / 30.0, 0.0, 1.0) as f32 } else { 0.0 };
            if ct.movement >= 0 && ct.movement <= 2 { inp[CONTROL_OFFSET + 12 + ct.movement as usize] = 1.0; }
            inp[CONTROL_OFFSET + 15] = b(ct.stopped);
            inp[CONTROL_OFFSET + 16] = b(ct.yield_required);
        } else { inp[CONTROL_OFFSET + 10] = 1.0; }
        inp[KINK_OFFSET] = 0.0; inp[KINK_OFFSET + 1] = 1.0;   // Guidance.nextKink: KINKS_ENABLED = false
        let jn = &o.junction;
        let jd = jn.distance < f64::INFINITY;
        inp[JUNCTION_OFFSET] = if jd { clamp(jn.distance / 60.0, 0.0, 1.0) as f32 } else { 1.0 };
        inp[JUNCTION_OFFSET + 1] = if jd && jn.time_to_conflict < f64::INFINITY { clamp(jn.time_to_conflict / 10.0, 0.0, 1.0) as f32 } else { 1.0 };
        inp[JUNCTION_OFFSET + 2] = b(jd && jn.stopped);
        inp[JUNCTION_OFFSET + 3] = b(jn.in_box);
        inp[JUNCTION_OFFSET + 4] = if jn.mouth_distance < f64::INFINITY { clamp(jn.mouth_distance / 60.0, 0.0, 1.0) as f32 } else { 1.0 };

        // ---------------- Driver._adjust -> adjust.js adjustInputs
        let p = &VEHICLE_PARAMS[o.vehicle_type];
        let wheelbase = jm::or(o.wheelbase, p.l);
        let p_max_steer = jm::or(o.max_steer, p.max_steer);
        let max_steer_rate = jm::or(o.max_steer_rate, p.max_steer_rate);
        let kus = (p.m / p.l) * (p.b / p.cf - p.a / p.cr);
        let steer_lag = jm::or(jm::or(p.steer_lag, 0.1), 0.1);
        let steer_slew = jm::or(jm::or(p.steer_slew, p.max_steer_rate), max_steer_rate);
        let length = jm::or(o.length, 4.5);
        // a) delay compensation: stepLateral over the pending commands (position mode)
        let (mut s_steer, mut s_sl, mut s_he, mut s_lat) = (jm::or(o.steer_angle, 0.0), jm::or(o.steer_angle, 0.0), heading_error, jm::or(o.lane_offset, 0.0));
        let lag = jm::or(steer_lag, 0.1);
        let slew = jm::or(steer_slew, max_steer_rate) * dt;
        for k in 0..st.pend_n {
            let cmd = st.pend[k] as f64;
            let r0 = if cmd < -1.0 { -1.0 } else if cmd > 1.0 { 1.0 } else { cmd };
            s_sl += (r0 * p_max_steer - s_sl) * jm::min(1.0, dt / lag);
            let d = s_sl - s_steer;
            let mut s = s_steer + if d > slew { slew } else if d < -slew { -slew } else { d };
            if s > p_max_steer { s = p_max_steer; } else if s < -p_max_steer { s = -p_max_steer; }
            s_steer = s;
            s_he -= ((v * s) / (wheelbase + kus * v * v)) * dt;
            s_lat -= v * jm::sin(s_he) * dt;
        }
        inp[4] = clamp(s_lat / o.lane_width, -1.0, 1.0) as f32;
        inp[5] = clamp(s_he / (jm::PI / 4.0), -1.0, 1.0) as f32;
        inp[41] = clamp(s_steer / p_max_steer, -1.0, 1.0) as f32;
        // b) lane ends / junction conflict as stopped obstacles
        let (mut own_end, mut own_side) = (f64::INFINITY, 0.0f64);
        let (mut left_end, mut right_end) = (f64::INFINITY, f64::INFINITY);
        let mut plan: Option<(f64, i32)> = None;   // Driver._adjust `plan` (own lane, 400 m): only feeds near.endDist
        if !o.on_connector {
            own_end = o.lane_end_distance; own_side = o.lane_end_side as f64;   // obs.laneEndDistance: the sim's (injected)
            let lane = o.lane_index;
            if near.is_some() { plan = st.guide.lane_end_ahead_side(&g, rd, lane, SIDE_LANE_END_LOOKAHEAD); }
            if left_exists { if let Some(e) = st.guide.lane_end_ahead(&g, rd, lane + 1, SIDE_LANE_END_LOOKAHEAD) { left_end = e; } }
            if right_exists { if let Some(e) = st.guide.lane_end_ahead(&g, rd, lane - 1, SIDE_LANE_END_LOOKAHEAD) { right_end = e; } }
        }
        let rel = clamp(-v / REL_SPEED_NORM, -1.0, 1.0);
        let half = 0.5 * length + 1.0;
        let put = |inp: &mut [f32], k: usize, gap: f64| {
            let at = NEIGHBOR_OFFSET + 2 * k;
            if gap < RANGE && gap / RANGE < inp[at] as f64 { inp[at] = (gap / RANGE) as f32; inp[at + 1] = rel as f32; }
        };
        inp[LANE_END_OFFSET] = if own_end < f64::INFINITY { clamp(own_end / 300.0, 0.0, 1.0) as f32 } else { 1.0 };
        inp[LANE_END_OFFSET + 1] = if own_end < f64::INFINITY { own_side as f32 } else { 0.0 };
        let (mut junc_gap, mut left_gap, mut right_gap) = (f64::INFINITY, f64::INFINITY, f64::INFINITY);
        if jd {
            let my_t = (jn.distance + length) / jm::max(v, 1.0);
            if jn.stopped || jn.time_to_conflict < my_t + 2.0 { junc_gap = jm::max(0.0, jn.distance - 2.0); put(inp, 0, junc_gap); }
        }
        if left_exists && left_end < f64::INFINITY { left_gap = jm::max(0.0, left_end - half); put(inp, 2, left_gap); }
        if right_exists && right_end < f64::INFINITY { right_gap = jm::max(0.0, right_end - half); put(inp, 4, right_gap); }
        if let Some(nr) = near {
            // eyes.js look (obs.neighbors branch), then Driver._adjust's mirror of the adjust.js result. The own-lane
            // end is always sim-injected here (obs.laneEndDistance is defined), so r.ownGap stays Infinity.
            let own_gap = f64::INFINITY;
            let nbv = |k: usize| v + nb[k] as f64;
            *nr = Near {
                leader_gap: gp(0), leader_speed: nbv(1), leader_rel: nb[1] as f64, behind_gap: gp(1), behind_speed: nbv(3),
                left_ahead: gp(2), left_ahead_speed: nbv(5), left_behind: gp(3), left_behind_speed: nbv(7),
                right_ahead: gp(4), right_ahead_speed: nbv(9), right_behind: gp(5), right_behind_speed: nbv(11),
                left_exists, right_exists, heading_error: s_he,
                lane_end_gap: own_gap, left_end_gap: left_gap, right_end_gap: right_gap,
                end_dist: f64::INFINITY, end_side: 0.0, left_end_dist: left_end, right_end_dist: right_end,
            };
            if own_gap < nr.leader_gap { nr.leader_gap = own_gap; nr.leader_speed = 0.0; nr.leader_rel = -v; }
            if junc_gap < nr.leader_gap { nr.leader_gap = junc_gap; nr.leader_speed = 0.0; nr.leader_rel = -v; }
            if left_gap < nr.left_ahead { nr.left_ahead = left_gap; nr.left_ahead_speed = 0.0; }
            if right_gap < nr.right_ahead { nr.right_ahead = right_gap; nr.right_ahead_speed = 0.0; }
            if own_end < f64::INFINITY { nr.end_dist = own_end; nr.end_side = own_side; }
            else if let Some((d, s)) = plan { nr.end_dist = d; nr.end_side = s as f64; }
        }
        if let Some(t) = teacher {
            // Driver.sense's private obs copy after _adjust (o.laneOffset = r.off, o.heading = laneHeading + r.he,
            // o.steerAngle = r.steer), this.guide, this.late
            let ms = &st.guide.maneuvers;
            let mv = |k: usize| TeacherManeuver { ty: ms[k].ty, at_distance: ms[k].at };
            *t = TeacherObs {
                speed: v, heading: o.lane_heading + s_he, lane_heading: o.lane_heading, lane_width: o.lane_width, speed_limit: o.speed_limit,
                lane_index: o.lane_index as f64, lane_count: o.lane_count as f64, lane_offset: s_lat, steer_angle: s_steer,
                control: Some(TeacherControl { ty: ct.ty as f64, state: ct.state as f64, distance: ct.distance, time_in_state: ct.time_in_state,
                    time_to_change: ct.time_to_change, movement: ct.movement as f64, stopped: ct.stopped, yield_required: Some(ct.yield_required) }),
                kink: None, grade: Some(o.grade),
                guide: Some(TeacherGuide { lane_hint: ga.lane_hint, distance_to_next: ga.distance_to_next, next: mv(ga.next), following: ga.following.map(mv) }),
                late_feature: late,
            };
        }
        // position mode, input version 2: road grade in 41..43
        for k in 0..3 { let x = o.grade[k] / 0.35; inp[WHEEL_OFFSET + k] = (if x > 1.0 { 1.0 } else if x < -1.0 { -1.0 } else { x }) as f32; }
}

// ---------------------------------------------------------------- differential check (tools/rust/v2_oracle.mjs)
/// One v2 mismatch: recorded step, slot, input index, JS / Rust value.
#[derive(Debug)]
pub struct V2Mismatch { pub step: usize, pub slot: usize, pub index: usize, pub js: f32, pub rust: f32 }

/// Result of [`check_v2`]
#[derive(Debug, Default)]
pub struct V2Check { pub steps: usize, pub rows: usize, pub mismatches: Vec<V2Mismatch>, pub encode_ns: f64, pub encodes: usize,
    /// obs::observe_one alone (fill_obs + signals), timed on the recorded rows: the part of encode_ns that is the env obs
    pub observe_ns: f64, pub state_unchanged: Option<bool>,
    /// rows whose `eyes.near` was compared (v2.pack section v2.near present), and the near mismatches
    pub near_rows: usize, pub near_mismatches: Vec<NearMismatch>,
    /// rows whose TeacherObs was compared (section v2.teacher), its mismatches (field = TEACHER_FIELDS name), coverage
    pub teacher_rows: usize, pub teacher_mismatches: Vec<NearMismatch>, pub teacher_coverage: TeacherCoverage }

/// How many compared rows exercised each optional / variable TeacherObs part (counted on the Rust values).
#[derive(Debug, Default, Clone, Copy)]
pub struct TeacherCoverage {
    /// control object present / with a real control (type >= 0)
    pub control_present: usize, pub control_real: usize,
    /// control.yieldRequired defined / true
    pub yield_defined: usize, pub yield_true: usize,
    pub kink_present: usize, pub guide_present: usize,
    /// guide.following non-null; laneHint != none
    pub following: usize, pub lane_hint: usize,
    /// lateFeature != 0
    pub late_nonzero: usize,
}

/// One `eyes.near` mismatch: recorded step, slot, field (NEAR_FIELDS name), JS / Rust value.
#[derive(Debug)]
pub struct NearMismatch { pub step: usize, pub slot: usize, pub field: &'static str, pub js: f64, pub rust: f64 }

/// Replay the run in `dir` (v2_oracle.mjs output: world.pack, pop.pack, run.json, v2.pack) with the harness policy,
/// encoding every driving car on every step, and compare the recorded rows bit for bit. With `state_check`, also
/// runs the same env without any observing and compares the whole vehicle store at the end (observing must not
/// change the trajectory).
pub fn check_v2(dir: &str, max: usize, state_check: bool) -> V2Check {
    use crate::harness::{env_from_dir, Policy};
    let (mut env, run) = env_from_dir(dir);
    let pk = crate::pack::Pack::read(&format!("{dir}/v2.pack")).expect("v2.pack");
    let every = pk.meta["every"].as_u64().unwrap_or(1) as usize;
    let steps = pk.meta["steps"].as_u64().unwrap() as usize;
    let (vn, vslot, vx) = (pk.u32s("v2.n"), pk.i32s("v2.slot"), pk.f32s("v2.x"));
    // optional section (newer oracles): v2.near = f64 rows of NEAR_N, field order meta.nearFields
    let vnear = if pk.has("v2.near") {
        let names: Vec<String> = pk.meta["nearFields"].as_array().expect("meta.nearFields").iter().map(|x| x.as_str().unwrap().to_string()).collect();
        assert_eq!(names, NEAR_FIELDS.iter().map(|s| s.to_string()).collect::<Vec<_>>(), "v2.near field order");
        Some(pk.f64s("v2.near"))
    } else { None };
    let vteach = if pk.has("v2.teacher") {
        let names: Vec<String> = pk.meta["teacherFields"].as_array().expect("meta.teacherFields").iter().map(|x| x.as_str().unwrap().to_string()).collect();
        assert_eq!(names, TEACHER_FIELDS.iter().map(|s| s.to_string()).collect::<Vec<_>>(), "v2.teacher field order");
        Some(pk.f64s("v2.teacher"))
    } else { None };
    let aux = vnear.is_some() || vteach.is_some();
    let mut near_row = Near::default();
    let mut teach_row = TeacherObs::default();
    let mut batch_near: Vec<Near> = Vec::new();
    let mut batch_teach: Vec<TeacherObs> = Vec::new();
    let mut enc = V2Encoder::new(&env.world).with_names(pk.i32s("edge.ref"), pk.i32s("edge.name"));
    let mut pol = Policy::new(run["policy"].as_str().unwrap(), run["seed"].as_f64().unwrap());
    let mut res = V2Check::default();
    let (mut rec, mut off) = (0usize, 0usize);
    let mut row = [0f32; OBS2];
    let (mut ns, mut ons) = (0u128, 0u128);
    let mut tmp = Obs::default();
    let batch = std::env::var("BAYENV_V2_BATCH").map(|v| !v.is_empty()).unwrap_or(false);
    let mut batch_rows: Vec<f32> = Vec::new();
    'outer: for k in 1..=steps {
        pol.act(&mut env);
        enc.record_actions(&env.sim);
        env.step(1);
        let slots = crate::obs::driving_slots(&env.sim);
        if batch { batch_rows.resize(slots.len() * OBS2, 0.0); batch_near.resize(slots.len(), Near::default()); batch_teach.resize(slots.len(), TeacherObs::default()); }
        let recorded = k % every == 0;
        if recorded && vn[rec] as usize != slots.len() {
            res.mismatches.push(V2Mismatch { step: k, slot: usize::MAX, index: usize::MAX, js: vn[rec] as f32, rust: slots.len() as f32 });
            res.steps = k; break;
        }
        if batch {
            let t = std::time::Instant::now();
            if aux { enc.encode_batch_teacher(&mut env.sim, &slots, &mut batch_rows[..], Some(&mut batch_near[..]), Some(&mut batch_teach[..])); }
            else { enc.encode_batch(&mut env.sim, &slots, &mut batch_rows[..]); }
            ns += t.elapsed().as_nanos(); res.encodes += slots.len();
        }
        for (r, &i) in slots.iter().enumerate() {
            if batch { row.copy_from_slice(&batch_rows[r * OBS2..(r + 1) * OBS2]); near_row = batch_near.get(r).copied().unwrap_or_default(); teach_row = batch_teach.get(r).copied().unwrap_or_default(); if !recorded { continue; } } else {
            let t = std::time::Instant::now();
            if aux { enc.encode_v2_teacher(&mut env.sim, i, &mut row, &mut near_row, &mut teach_row); } else { enc.encode_v2(&mut env.sim, i, &mut row); }
            ns += t.elapsed().as_nanos(); res.encodes += 1;
            if !recorded { continue; }
            let t = std::time::Instant::now();
            observe_one(&mut env.sim, i, &mut tmp, None);
            ons += t.elapsed().as_nanos();
            }
            let q = off + r;
            if vslot[q] as usize != i { res.mismatches.push(V2Mismatch { step: k, slot: i, index: usize::MAX, js: vslot[q] as f32, rust: i as f32 }); res.steps = k; break 'outer; }
            for d in 0..OBS2 {
                let (x, y) = (vx[q * OBS2 + d], row[d]);
                if x.to_bits() != y.to_bits() && !(x.is_nan() && y.is_nan()) {
                    res.mismatches.push(V2Mismatch { step: k, slot: i, index: d, js: x, rust: y });
                    if res.mismatches.len() >= max { res.steps = k; break 'outer; }
                }
            }
            if let Some(vn2) = &vnear {
                let a = near_row.to_array();
                for (f, &y) in a.iter().enumerate() {
                    let x = vn2[q * NEAR_N + f];
                    if x.to_bits() != y.to_bits() && !(x.is_nan() && y.is_nan()) {
                        res.near_mismatches.push(NearMismatch { step: k, slot: i, field: NEAR_FIELDS[f], js: x, rust: y });
                        if res.near_mismatches.len() >= max { res.steps = k; break 'outer; }
                    }
                }
                res.near_rows += 1;
            }
            if let Some(vt) = &vteach {
                let a = teach_row.to_array();
                for (f, &y) in a.iter().enumerate() {
                    let x = vt[q * TEACHER_N + f];
                    if x.to_bits() != y.to_bits() && !(x.is_nan() && y.is_nan()) {
                        res.teacher_mismatches.push(NearMismatch { step: k, slot: i, field: TEACHER_FIELDS[f], js: x, rust: y });
                        if res.teacher_mismatches.len() >= max { res.steps = k; break 'outer; }
                    }
                }
                let (t, c) = (&teach_row, &mut res.teacher_coverage);
                if let Some(ct) = t.control { c.control_present += 1; if ct.ty >= 0.0 { c.control_real += 1; } if let Some(y) = ct.yield_required { c.yield_defined += 1; if y { c.yield_true += 1; } } }
                if t.kink.is_some() { c.kink_present += 1; }
                if let Some(g) = t.guide { c.guide_present += 1; if g.following.is_some() { c.following += 1; } if g.lane_hint != SIDE_NONE { c.lane_hint += 1; } }
                if t.late_feature != 0.0 { c.late_nonzero += 1; }
                res.teacher_rows += 1;
            }
            res.rows += 1;
        }
        if recorded { off += slots.len(); rec += 1; }
        res.steps = k;
    }
    res.encode_ns = ns as f64 / res.encodes.max(1) as f64;
    res.observe_ns = ons as f64 / res.rows.max(1) as f64;
    if state_check && res.mismatches.is_empty() {
        let (mut env2, _) = env_from_dir(dir);
        let mut pol2 = Policy::new(run["policy"].as_str().unwrap(), run["seed"].as_f64().unwrap());
        for _ in 1..=steps { pol2.act(&mut env2); env2.step(1); }
        let a = env.sim.vs.fields(); let b2 = env2.sim.vs.fields();
        res.state_unchanged = Some(env.sim.vs.high == env2.sim.vs.high && a.len() == b2.len() && a.iter().zip(b2.iter()).all(|(x, y)| x.0 == y.0 && x.3 == y.3));
    }
    res
}
