//! Per-slot event records for training, shared by the GPU host and the CPU sim (identical shape): one row per event, struct of arrays.
//! kinds: 1 arrived, 2 crashed (this car disabled), 3 contact (not disabled), 4 stranded, 5 cleared (wreck towed /
//! removed, slot freed), 6 spawned, 7 missed exit, 8 signal / stop violation.
pub mod kind { pub const ARRIVED: u8 = 1; pub const CRASHED: u8 = 2; pub const CONTACT: u8 = 3; pub const STRANDED: u8 = 4; pub const CLEARED: u8 = 5; pub const SPAWNED: u8 = 6; pub const MISSED_EXIT: u8 = 7; pub const VIOLATION: u8 = 8; }

#[derive(Clone, Default, Debug)]
pub struct EventLog {
    pub time: Vec<f64>, pub kind: Vec<u8>, pub slot: Vec<i32>, pub agent: Vec<i64>, pub other: Vec<i32>,
    pub dv_kmh: Vec<f32>, pub other_dv_kmh: Vec<f32>, pub share: Vec<f32>, pub late_by: Vec<f32>, pub progress_m: Vec<f32>,
    pub trip_m: Vec<f32>, pub trip_s: Vec<f32>,
}
impl EventLog {
    pub fn len(&self) -> usize { self.kind.len() }
    pub fn is_empty(&self) -> bool { self.kind.is_empty() }
    #[allow(clippy::too_many_arguments)]
    pub fn push(&mut self, time: f64, kind: u8, slot: u32, agent: i64, other: i64, dv: f32, odv: f32, share: f32) {
        self.time.push(time); self.kind.push(kind); self.slot.push(slot as i32); self.agent.push(agent); self.other.push(other as i32);
        self.dv_kmh.push(dv); self.other_dv_kmh.push(odv); self.share.push(share);
        self.late_by.push(f32::NAN); self.progress_m.push(f32::NAN); self.trip_m.push(f32::NAN); self.trip_s.push(f32::NAN);
    }
    /// Take everything recorded so far.
    pub fn drain(&mut self) -> EventLog { std::mem::take(self) }
}
