//! Input-version-3 brain observation on the CPU: the 358-float row the GPU emits with `GpuOptions::obs =
//! ObsMode::V3` (bayenv-gpu: `src/gpu/kernels/sense.wgsl` + `senseV3`, `perceive.wgsl`, `kernels/model_sense.wgsl`), so
//! v3 nets can be trained on the CPU (teacher / DAgger) and evaluated on either backend. The GPU row is the definition.
//!
//! # Layout (`obs::OBS3` = 358 f32; docs/perception-design.md §3, private trainer `brain/src/net3.rs`)
//! ```text
//!   0..69   the v2 vector as V2Encoder writes it (grade in 41..43 / 0.35 clamped, delay-compensated 4 / 5 / 41 with
//!           min(reaction, steer_delay) pending commands), with 8..19 (gpu_neighbours) and 20..32 (gpu_guidance)
//!           rewritten where the GPU's sense.wgsl defines them differently from the JS driver (DECISIONS.md 42)
//!  70..78   mirror inputs (obs.rs sense_bodies)
//!  79..86   controls extras (obs.rs sense_controls3)
//!  87..93   own commands (obs.rs sense_commands: 87 = steer at the steer delay, 88 / 89 pedals at the reaction delay,
//!           90..92 the newest command, 93 = 1)
//!  94..100  leader history (obs.rs LeaderHistory, per trip, stepped once per encode with the final inputs 8 / 9)
//! 101       0
//! 102..357  16 bodies x 16 features (obs.rs sense_bodies / body_features; empty body = 16 zeros, validity at 15)
//! ```
//! `obs.rs fill_obs3` (the JS env.observe({v3: true}) vector) is the same 70..93 / 102..357 with 0..69 and 94..101 left
//! at zero (the JS leaves those to the policy-side encoder); this module fills them as the GPU does.
//! Known GPU differences kept on purpose (GPU bugs, DECISIONS.md 42): the GPU writes 62 / 65..69 before integrate (a
//! step older than the rest of its row); the CPU writes the state after the step.
//!
//! # Call protocol
//! As V2Encoder (act -> `record_actions` -> `Sim::step` -> encode every driving slot): the leader history advances
//! once per encode, like the GPU's sense (once per step for every policy car with decide_every 1), so encode each
//! driving car once after every step. Per-trip state (guidance, pending steer ring, leader history) is kept in the
//! wrapped V2Encoder's TripState.
use crate::obs::{sense_bodies, sense_commands, sense_controls3, Obs, OBS3};
use crate::leaders::lane_end_ahead;
use crate::obs_v2::{man, NEIGHBOR_OFFSET, RANGE, REL_SPEED_NORM, SIDE_LANE_END_LOOKAHEAD, Guidance, NavEdges, Near, TeacherObs, TripState, V2Encoder, MANEUVER_OFFSET, OBS2, SIDE_LEFT, SIDE_RIGHT, TRAIT_OFFSET};
use crate::sim::Sim;
use crate::world::World;

/// Width of a v3 row.
pub const OBS_V3: usize = OBS3;

/// The v3 part of the row (70..357) for slot i, after the v2 part (0..69, final) is in `row`; also rewrites the
/// guidance inputs 20..32 the GPU way (gpu_guidance).
pub(crate) fn v3_tail(st: &mut TripState, nav: &NavEdges, sim: &Sim, i: usize, o: &Obs, row: &mut [f32]) {
    let row = &mut row[..OBS3];
    {
        let route = &sim.vs.routes[i];
        let gd = if st.guide.edge_ids == *route { &st.guide } else {
            if st.v3_guide.as_ref().map_or(true, |g| g.edge_ids != *route) { st.v3_guide = Some(Guidance::new(sim.gw(), nav, route, None)); }
            st.v3_guide.as_ref().unwrap()
        };
        gpu_guidance(gd, sim, i, row);
    }
    gpu_neighbours(sim, i, o, row);
    row[OBS2..].fill(0.0);
    sense_bodies(sim, i, row, o.lane_width, o.lane_offset);
    sense_controls3(sim, i, row);
    sense_commands(sim, i, row);
    st.lhist.step(row, o.leader_id);
}

/// Guidance inputs 20..32 as the GPU's sense.wgsl computes them from the route's maneuver codes (bayenv-world
/// nav::pack_route_words = the same transitions and keep coalescing as `gd`'s maneuver list, without 'continue'
/// (no road names) and without the refuel stop, which the GPU takes from the car's stop position). Differs from the
/// v2 (JS driver) inputs: the distance is measured from the car's position on its edge (the JS driver: from the
/// route distance driven, which starts at 0 at the spawn point, so it was long by the spawn offset), maneuvers are
/// looked up by route position, and with nothing within 2 km the type is the next one on the route (distance 1).
pub fn gpu_guidance(gd: &Guidance, sim: &Sim, i: usize, row: &mut [f32]) {
    let (vs, g) = (&sim.vs, sim.gw());
    let route = &vs.routes[i]; let n = route.len();
    let rp = vs.route_pos[i].max(0) as usize;
    // the codes as (route entry, type, side): a maneuver at edge_pos p sits on the hop out of entry p - 1, arrive on
    // the last entry
    let codes = gd.maneuvers.iter().filter(|m| m.ty != man::CONTINUE && m.ty != man::REFUEL)
        .map(|m| (if m.ty == man::ARRIVE { n.saturating_sub(1) } else { m.edge_pos.saturating_sub(1) }, m.ty, m.side));
    let mut it = codes.filter(|c| c.0 >= rp).peekable();
    let next_ty = it.peek().map_or(man::ARRIVE, |c| c.1);
    let e = vs.edge[i] as usize;
    let mut d = (g.e_length[e] as f32 - vs.s[i]) as f64;
    let gib = sim.opts.v3_guide_in_box;
    let (mut k, mut typ, mut sd, mut found) = (rp, 0u8, 0u8, false);
    for _ in 0..256 {
        if vs.refuel_state[i] == 1 && vs.stop_pos[i] == k as i32 { typ = man::REFUEL; sd = SIDE_RIGHT; found = true; break; }
        while it.peek().map_or(false, |c| c.0 < k) { it.next(); }
        // v3_guide_in_box (GpuOptions::guide_in_box, model_sense.wgsl MF_GUIDE_BOX): > 1 m past the in-edge's end
        // (inside the box) the maneuver being crossed counts as passed
        if let Some(&(_, t, s)) = it.peek().filter(|c| c.0 == k && !(gib && d < -1.0)) { typ = t; sd = s; found = true; break; }
        if k + 1 >= n || d > 2000.0 { break; }
        k += 1;
        d += g.e_length[route[k] as usize];
    }
    if !found { typ = next_ty; sd = 0; d = 1e9; }
    if gib { d = d.max(0.0); }
    for x in row[MANEUVER_OFFSET..TRAIT_OFFSET].iter_mut() { *x = 0.0; }
    row[MANEUVER_OFFSET] = (d / 2000.0).min(1.0) as f32;
    row[MANEUVER_OFFSET + 1 + (typ as usize).min(8)] = 1.0;
    let near = d <= 2000.0 && typ != man::MERGE;
    let hint = if near && sd == SIDE_LEFT { 0 } else if near && sd == SIDE_RIGHT { 2 } else { 1 };
    row[MANEUVER_OFFSET + 10 + hint] = 1.0;
}

/// Neighbour inputs 8..19 as the GPU's sense.wgsl writes them: the six lane slots from the footprint lane lists
/// (eyes.js, as v2), then only the side lanes' ends as stopped obstacles (adjust.js putEnd), the lane end looked up
/// from the car's route position and its position on the edge. Differs from the v2 (JS driver) inputs: the junction-
/// conflict obstacle in the own-ahead slot only with opts.v3_junction_obstacle (the JS GPU had none), and the side lane ends are not found through the
/// route distance (which misses the spawn offset).
pub fn gpu_neighbours(sim: &Sim, i: usize, o: &Obs, row: &mut [f32]) {
    let (vs, g) = (&sim.vs, sim.gw());
    let lane_count = (o.lane_count as f64).max(1.0);
    let lane_index = (o.lane_index as f64).max(0.0).min(lane_count - 1.0);
    let (left_ex, right_ex) = (lane_index < lane_count - 1.0, lane_index > 0.0);
    let v = o.speed; let nb = &o.neighbors;
    for k in 0..6 {
        let at = NEIGHBOR_OFFSET + 2 * k;
        let exists = match k { 2 | 3 => left_ex, 4 | 5 => right_ex, _ => true };
        let x = nb[2 * k] as f64;
        let gap = if x >= RANGE || !(x < f64::INFINITY) { f64::INFINITY } else { x };
        if !exists { row[at] = 0.0; row[at + 1] = 0.0; }
        else if gap == f64::INFINITY { row[at] = 1.0; row[at + 1] = 0.0; }
        else { row[at] = (gap.max(0.0) / RANGE) as f32; row[at + 1] = ((nb[2 * k + 1] as f64) / REL_SPEED_NORM).max(-1.0).min(1.0) as f32; }
    }
    let rel = (-v / REL_SPEED_NORM).max(-1.0).min(1.0);
    // v3_junction_obstacle (GpuOptions::junction_obstacle, model_jyield.wgsl / model_sense.wgsl MF_JUNC_OBST): the
    // junction-conflict obstacle into the own-ahead slot, as obs v2 (obs_v2.rs encode_core / adjust.js junctionObs)
    if sim.opts.v3_junction_obstacle {
        let jn = &o.junction;
        if jn.distance < 1e8 {
            let my_t = (jn.distance + vs.length[i] as f64) / v.max(1.0);
            if jn.stopped || jn.time_to_conflict < my_t + 2.0 {
                let gap = (jn.distance - 2.0).max(0.0);
                if gap < RANGE && gap / RANGE < row[NEIGHBOR_OFFSET] as f64 { row[NEIGHBOR_OFFSET] = (gap / RANGE) as f32; row[NEIGHBOR_OFFSET + 1] = rel as f32; }
            }
        }
    }
    let half = 0.5 * vs.length[i] as f64 + 1.0;
    let lane = vs.lane[i] as i32;
    let (route, rp) = (&vs.routes[i], vs.route_pos[i]);
    if rp < 0 || rp as usize >= route.len() || route[rp as usize] != vs.edge[i] { return; }
    let stop = if vs.refuel_state[i] == 1 { vs.stop_pos[i] } else { -1 };
    let n = g.e_lanes[vs.edge[i] as usize];
    for (ex, l, k) in [(left_ex, lane + 1, 2usize), (right_ex, lane - 1, 4usize)] {
        if !ex || l < 0 || l >= n { continue; }
        let e = lane_end_ahead(g, route, rp as usize, vs.s[i] as f64, l, SIDE_LANE_END_LOOKAHEAD, stop);
        if e.dist < 1e8 {
            let gap = (e.dist - half).max(0.0);
            let at = NEIGHBOR_OFFSET + 2 * k;
            if gap < RANGE && gap / RANGE < row[at] as f64 { row[at] = (gap / RANGE) as f32; row[at + 1] = rel as f32; }
        }
    }
}

/// v3 encoder: V2Encoder (0..69 + the per-trip state) plus the v3 inputs.
pub struct V3Encoder { pub v2: V2Encoder }

impl V3Encoder {
    pub fn new(g: &World) -> V3Encoder { V3Encoder { v2: V2Encoder::new(g) } }
    /// Road ref / name ids per edge (V2Encoder::with_names).
    pub fn with_names(self, ref_id: Vec<i32>, name_id: Vec<i32>) -> V3Encoder { V3Encoder { v2: self.v2.with_names(ref_id, name_id) } }
    /// Forget every trip (call after env.reset()).
    pub fn reset(&mut self) { self.v2.reset(); }
    /// V2Encoder::push_command.
    pub fn push_command(&mut self, sim: &Sim, i: usize, cmd: f64) { self.v2.push_command(sim, i, cmd); }
    /// V2Encoder::record_actions: call after the actions are set, before Sim::step.
    pub fn record_actions(&mut self, sim: &Sim) { self.v2.record_actions(sim); }
    /// One driving slot into out[..OBS3].
    pub fn encode(&mut self, sim: &mut Sim, slot: usize, out: &mut [f32]) {
        assert!(out.len() >= OBS3);
        self.v2.encode_v2_impl(sim, slot, out, None, None, true)
    }
    /// One driving slot into out[..OBS3], with the v2 `eyes.near` and the teacher's view.
    pub fn encode_teacher(&mut self, sim: &mut Sim, slot: usize, out: &mut [f32], near: &mut Near, teacher: &mut TeacherObs) {
        assert!(out.len() >= OBS3);
        self.v2.encode_v2_impl(sim, slot, out, Some(near), Some(teacher), true)
    }
    /// Driving slots (ascending, distinct) into rows of OBS3 floats (parallel as V2Encoder::encode_batch).
    pub fn encode_batch(&mut self, sim: &mut Sim, slots: &[usize], out: &mut [f32]) { self.v2.encode_rows(sim, slots, out, OBS3, None, None) }
    /// encode_batch that also writes `near[k]` and / or `teacher[k]` (V2Encoder::encode_batch_teacher) per slot.
    pub fn encode_batch_teacher(&mut self, sim: &mut Sim, slots: &[usize], out: &mut [f32], near: Option<&mut [Near]>, teacher: Option<&mut [TeacherObs]>) {
        self.v2.encode_rows(sim, slots, out, OBS3, near, teacher)
    }
}
