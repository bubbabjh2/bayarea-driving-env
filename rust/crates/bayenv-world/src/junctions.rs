//! Physical junctions (`src/geom/junctions.js` JunctionSet; file `data/junctions.bin`, layout in
//! `data/junctions.meta.json`): connectors per (in-edge, lane) -> out-edge movement, conflict matrix,
//! junction-box hulls, stop / exit mouths and ramp gores.
use crate::blob::{Pack, Slice};
use crate::jsmath;
use std::{io, path::Path};

/// Section order of `data/junctions.bin`.
pub const JUNCTION_SECTIONS: [&str; 15] = ["mouthS", "goreS", "jNode", "jFirstM", "jCountM", "jFirstC", "jCountC", "jFirstH", "jCountH",
    "mInfo", "mF", "pts", "conf", "confS", "hull"];

#[derive(Clone, Default)]
pub struct JunctionSet {
    /// 2 per edge: [sExit at the start junction, sStop at the end junction]
    pub mouth_s: Slice<f32>,
    /// per edge: ramp wait point (m), -1 none
    pub gore_s: Slice<f32>,
    pub j_node: Slice<u32>, pub j_first_m: Slice<u32>, pub j_count_m: Slice<u32>, pub j_first_c: Slice<u32>,
    pub j_count_c: Slice<u32>, pub j_first_h: Slice<u32>, pub j_count_h: Slice<u32>,
    /// 7 per movement: inEdge, inLane, outEdge, outLane, firstPt, ptCount, preferred
    pub m_info: Slice<u32>,
    /// 4 per movement: sIn, sOut, connector length, turn angle (rad, > 0 right)
    pub m_f: Slice<f32>,
    /// 4 per connector point: x, y, z, metres along
    pub pts: Slice<f32>,
    /// 3 per conflict: movementA, movementB, type (0 crossing, 1 merge)
    pub conf: Slice<u32>,
    /// 2 per conflict: s along A, s along B
    pub conf_s: Slice<f32>,
    /// 2 per hull point
    pub hull: Slice<f32>,
    /// build statistics written to junctions.meta.json (`crossingFix`)
    pub crossing_fix: Option<serde_json::Value>,
}

/// Owned arrays produced by the builder (convert with [`JunctionArrays::into_set`]).
#[derive(Clone, Default, Debug)]
pub struct JunctionArrays {
    pub mouth_s: Vec<f32>, pub gore_s: Vec<f32>,
    pub j_node: Vec<u32>, pub j_first_m: Vec<u32>, pub j_count_m: Vec<u32>, pub j_first_c: Vec<u32>, pub j_count_c: Vec<u32>,
    pub j_first_h: Vec<u32>, pub j_count_h: Vec<u32>,
    pub m_info: Vec<u32>, pub m_f: Vec<f32>, pub pts: Vec<f32>, pub conf: Vec<u32>, pub conf_s: Vec<f32>, pub hull: Vec<f32>,
    pub crossing_fix: Option<serde_json::Value>,
}

impl JunctionArrays {
    pub fn into_set(self) -> JunctionSet {
        JunctionSet {
            mouth_s: self.mouth_s.into(), gore_s: self.gore_s.into(), j_node: self.j_node.into(), j_first_m: self.j_first_m.into(),
            j_count_m: self.j_count_m.into(), j_first_c: self.j_first_c.into(), j_count_c: self.j_count_c.into(),
            j_first_h: self.j_first_h.into(), j_count_h: self.j_count_h.into(), m_info: self.m_info.into(), m_f: self.m_f.into(),
            pts: self.pts.into(), conf: self.conf.into(), conf_s: self.conf_s.into(), hull: self.hull.into(), crossing_fix: self.crossing_fix,
        }
    }
    /// The `data/junctions.bin` byte image (sections concatenated in [`JUNCTION_SECTIONS`] order).
    pub fn to_bin(&self) -> Vec<u8> {
        let mut out = Vec::new();
        macro_rules! put { ($($f:ident),*) => { $( out.extend_from_slice(bytemuck::cast_slice(&self.$f)); )* } }
        put!(mouth_s, gore_s, j_node, j_first_m, j_count_m, j_first_c, j_count_c, j_first_h, j_count_h, m_info, m_f, pts, conf, conf_s, hull);
        out
    }
}

impl JunctionSet {
    /// Load `junctions.meta.json` + `junctions.bin` (mmap'd).
    pub fn load(dir: &Path) -> io::Result<JunctionSet> {
        let p = Pack::open(dir, "junctions.meta.json")?;
        Ok(JunctionSet {
            mouth_s: p.req("mouthS")?, gore_s: p.req("goreS")?, j_node: p.req("jNode")?, j_first_m: p.req("jFirstM")?,
            j_count_m: p.req("jCountM")?, j_first_c: p.req("jFirstC")?, j_count_c: p.req("jCountC")?, j_first_h: p.req("jFirstH")?,
            j_count_h: p.req("jCountH")?, m_info: p.req("mInfo")?, m_f: p.req("mF")?, pts: p.req("pts")?, conf: p.req("conf")?,
            conf_s: p.req("confS")?, hull: p.req("hull")?, crossing_fix: p.meta.get("crossingFix").cloned(),
        })
    }
    #[inline] pub fn move_count(&self) -> usize { self.m_info.len() / 7 }
    #[inline] pub fn junction_count(&self) -> usize { self.j_node.len() }
    #[inline] pub fn conflict_count(&self) -> usize { self.conf.len() / 3 }
    pub fn describe(&self, m: usize) -> Movement {
        let (i, f) = (&self.m_info[7 * m..7 * m + 7], &self.m_f[4 * m..4 * m + 4]);
        Movement { in_edge: i[0], in_lane: i[1], out_edge: i[2], out_lane: i[3], first_pt: i[4], pt_count: i[5], preferred: i[6] != 0,
            s_in: f[0], s_out: f[1], length: f[2], turn: f[3] }
    }
    /// `arcPoint(m, t)`: (x, y, z, hx, hz) t metres along connector m.
    pub fn arc_point(&self, m: usize, t: f64) -> crate::lanes::Pt {
        let first = self.m_info[7 * m + 4] as usize; let n = self.m_info[7 * m + 5] as usize; let p = &self.pts;
        let mut k = 0;
        while k + 2 < n && (p[4 * (first + k + 1) + 3] as f64) < t { k += 1; }
        let (a, b) = (4 * (first + k), 4 * (first + k + 1));
        let mut d = p[b + 3] as f64 - p[a + 3] as f64; if d == 0.0 || d.is_nan() { d = 1.0; }
        let u = jsmath::max(0.0, jsmath::min(1.0, (t - p[a + 3] as f64) / d));
        let lerp = |o: usize| p[a + o] as f64 + (p[b + o] as f64 - p[a + o] as f64) * u;
        let (hx, hz) = (p[b] as f64 - p[a] as f64, p[b + 2] as f64 - p[a + 2] as f64);
        let mut l = jsmath::hypot(hx, hz); if l == 0.0 || l.is_nan() { l = 1.0; }
        crate::lanes::Pt { x: lerp(0), y: lerp(1), z: lerp(2), hx: hx / l, hz: hz / l }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Movement { pub in_edge: u32, pub in_lane: u32, pub out_edge: u32, pub out_lane: u32, pub first_pt: u32, pub pt_count: u32,
    pub preferred: bool, pub s_in: f32, pub s_out: f32, pub length: f32, pub turn: f32 }

/// Crosswalk set-backs for the junction builder (`crosswalkLegs`): edge ends that stop / exit before a crosswalk.
#[derive(Clone, Default, Debug)]
pub struct CrosswalkLegs { pub crosswalk_stop: std::collections::HashSet<u32>, pub crosswalk_exit: std::collections::HashSet<u32> }

/// `buildJunctions(LG, legs)`. OWNER: geom port agent (src/junctions_build.rs).
pub fn build_junctions(lg: &crate::lanes::LaneGeometry, g: &crate::lanegraph::LaneGraph, legs: Option<&CrosswalkLegs>) -> JunctionArrays {
    crate::junctions_build::build_junctions(lg, g, legs)
}
