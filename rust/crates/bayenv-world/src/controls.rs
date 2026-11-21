//! Traffic controls (`data/controls.*`) and their fixed-time signal plans: port of `src/sim/controls.js`
//! `TrafficControls` (construction + `approachAhead` + `stateOf`, all pure functions of the world and time).
use crate::blob::{Pack, Slice};
use crate::graph::SimGraph;
use crate::jsmath;
use std::collections::HashMap;
use std::{io, path::Path};

pub mod ctrl { pub const SIGNAL: i8 = 0; pub const STOP: i8 = 1; pub const GIVE_WAY: i8 = 2; pub const RAMP_METER: i8 = 3; pub const CROSSING: i8 = 4; }
pub mod light { pub const GREEN: u8 = 0; pub const YELLOW: u8 = 1; pub const RED: u8 = 2; pub const SIGN: u8 = 3; pub const DARK: u8 = 4; }
pub mod movement { pub const THROUGH: u8 = 0; pub const LEFT: u8 = 1; pub const RIGHT: u8 = 2; }

/// Raw `data/controls.*` arrays.
pub struct ControlsPack {
    pub meta: serde_json::Value,
    /// 6 per control: type, flags, tierId, nodeIndex, firstApproach, approachCount
    pub info: Slice<u32>,
    pub xyz: Slice<f32>,
    pub approach_edge: Slice<u32>,
    pub approach_stop_at: Slice<f32>,
}
impl ControlsPack {
    pub fn load(dir: &Path) -> io::Result<ControlsPack> {
        let p = Pack::open(dir, "controls.meta.json")?;
        Ok(ControlsPack { info: p.req("controlInfo")?, xyz: p.req("controlXYZ")?, approach_edge: p.req("approachEdge")?, approach_stop_at: p.req("approachStopAt")?, meta: p.meta })
    }
    pub fn main_edge_count(&self) -> Option<u64> { self.meta["mainEdgeCount"].as_u64() }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Phase { pub group: u8, pub left: bool, pub g: f64, pub y: f64, pub r: f64 }

/// Output of [`TrafficControls::state_of`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CtlState { pub ty: i8, pub state: u8, pub time_in: f64, pub time_left: f64 }

/// `movementOf(e, en)`
pub fn movement_of(g: &SimGraph, e: usize, en: usize) -> u8 {
    let mut d = g.heading_start[en] - g.heading_end[e];
    while d > std::f64::consts::PI { d -= 2.0 * std::f64::consts::PI; }
    while d < -std::f64::consts::PI { d += 2.0 * std::f64::consts::PI; }
    if d < -0.5 { movement::LEFT } else if d > 0.5 { movement::RIGHT } else { movement::THROUGH }
}

fn class_w(c: u8) -> f64 { [4.0, 2.0, 4.0, 2.0, 3.0, 2.0, 2.5, 1.5, 1.5, 1.0, 1.0, 1.0, 1.0][c as usize] }
fn axis_of(h: f64) -> f64 { let mut a = h % std::f64::consts::PI; if a < 0.0 { a += std::f64::consts::PI; } a }

#[derive(Clone, Default)]
pub struct TrafficControls {
    pub count: usize, pub approach_count: usize,
    pub ty: Vec<i8>, pub flags: Vec<u8>,
    /// Float32Array in JS
    pub cycle: Vec<f32>, pub offset: Vec<f32>,
    /// signals: phase ring (shared by every member of a cluster); empty otherwise
    pub phases: Vec<Vec<Phase>>,
    pub ap_control: Vec<i32>, pub ap_stop_at: Vec<f32>, pub ap_edge: Vec<i32>, pub ap_group: Vec<i8>, pub ap_prot_left: Vec<u8>,
    pub edge_approach: Vec<i32>,
    /// every stop line per edge sorted by position: CSR (E + 1) + list
    pub edge_ap_start: Vec<i32>, pub edge_ap_list: Vec<i32>,
    pub cluster_of: Vec<i32>,
    pub ramp_start: f64, pub ramp_end: f64,
    pub signal_clusters: usize, pub skipped_approaches: usize,
}

impl TrafficControls {
    /// `new TrafficControls(graph, pack)` with the default options (clusterDist 60 m, ramp meters 06:00-10:00).
    pub fn new(g: &SimGraph, pack: &ControlsPack) -> TrafficControls {
        let n_c = pack.info.len() / 6; let n_a = pack.approach_edge.len(); let e_n = g.edge_count;
        let mut t = TrafficControls { count: n_c, approach_count: n_a, ty: vec![0; n_c], flags: vec![0; n_c], cycle: vec![0.0; n_c], offset: vec![0.0; n_c],
            phases: vec![Vec::new(); n_c], ap_control: vec![0; n_a], ap_stop_at: vec![0.0; n_a], ap_edge: vec![0; n_a], ap_group: vec![0; n_a],
            ap_prot_left: vec![0; n_a], edge_approach: vec![-1; e_n], ramp_start: 6.0 * 3600.0, ramp_end: 10.0 * 3600.0, ..Default::default() };
        let mut skipped = 0; let mut sig_aps: Vec<(usize, Vec<usize>)> = Vec::new();
        for c in 0..n_c {
            let ty = pack.info[c * 6] as i32 as i8; let fl = pack.info[c * 6 + 1]; let first = pack.info[c * 6 + 4] as usize; let cnt = pack.info[c * 6 + 5] as usize;
            t.ty[c] = ty; t.flags[c] = fl as u8;
            let mut aps = Vec::new();
            for a in first..first + cnt {
                let e = pack.approach_edge[a] as usize;
                t.ap_control[a] = c as i32; t.ap_edge[a] = if e < e_n { e as i32 } else { -1 }; t.ap_stop_at[a] = pack.approach_stop_at[a];
                if e >= e_n { skipped += 1; continue; }
                if t.edge_approach[e] < 0 || pack.approach_stop_at[a] < t.ap_stop_at[t.edge_approach[e] as usize] { t.edge_approach[e] = a as i32; }
                aps.push(a);
            }
            if ty == ctrl::SIGNAL { sig_aps.push((c, aps)); }
            else if ty == ctrl::CROSSING { t.cycle[c] = 90.0; t.offset[c] = ((c as f64 * 37.0) % 90.0) as f32; }
            else if ty == ctrl::RAMP_METER { t.cycle[c] = 6.0; t.offset[c] = ((c as f64 * 1.7) % 6.0) as f32; }
        }
        // cluster signals closer than 60 m (union-find with path halving, JS Map insertion order)
        let cl = 60.0f64;
        let mut parent: HashMap<usize, usize> = sig_aps.iter().map(|(c, _)| (*c, *c)).collect();
        fn find(p: &mut HashMap<usize, usize>, mut c: usize) -> usize {
            while p[&c] != c { let gp = p[&p[&c]]; p.insert(c, gp); c = gp; } c
        }
        let mut cell: HashMap<i64, Vec<usize>> = HashMap::new();
        for (c, _) in &sig_aps {
            let c = *c; let (x, z) = (pack.xyz[c * 3] as f64, pack.xyz[c * 3 + 2] as f64);
            let (cx, cz) = ((x / cl).floor() as i64, (z / cl).floor() as i64);
            for dx in -1..=1 { for dz in -1..=1 {
                if let Some(l) = cell.get(&((cx + dx) * 100003 + cz + dz)) { for &o in l.clone().iter() {
                    if jsmath::hypot(pack.xyz[o * 3] as f64 - x, pack.xyz[o * 3 + 2] as f64 - z) < cl {
                        let (ro, rc) = (find(&mut parent, o), find(&mut parent, c)); parent.insert(ro, rc);
                    } } }
            } }
            cell.entry(cx * 100003 + cz).or_default().push(c);
        }
        let mut order: Vec<usize> = Vec::new(); let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
        for (c, _) in &sig_aps { let r = find(&mut parent, *c); clusters.entry(r).or_insert_with(|| { order.push(r); Vec::new() }).push(*c); }
        t.cluster_of = (0..n_c as i32).collect();
        for r in &order { for &c in &clusters[r] { t.cluster_of[c] = *r as i32; } }
        let aps_of: HashMap<usize, &Vec<usize>> = sig_aps.iter().map(|(c, a)| (*c, a)).collect();
        for r in &order {
            let members = &clusters[r];
            let aps: Vec<usize> = members.iter().flat_map(|c| aps_of[c].iter().cloned()).collect();
            t.plan_signal(g, members[0], &aps, &pack.xyz);
            let (cy, of, ph) = (t.cycle[members[0]], t.offset[members[0]], t.phases[members[0]].clone());
            for &c in members { t.cycle[c] = cy; t.offset[c] = of; t.phases[c] = ph.clone(); }
        }
        t.signal_clusters = order.len(); t.skipped_approaches = skipped;
        let mut per = vec![0i32; e_n + 1];
        for a in 0..n_a { if t.ap_edge[a] >= 0 { per[t.ap_edge[a] as usize + 1] += 1; } }
        for e in 0..e_n { per[e + 1] += per[e]; }
        let mut fill = per[..e_n].to_vec(); let mut list = vec![0i32; per[e_n] as usize];
        for a in 0..n_a { if t.ap_edge[a] >= 0 { let e = t.ap_edge[a] as usize; list[fill[e] as usize] = a as i32; fill[e] += 1; } }
        for e in 0..e_n { let sub = &mut list[per[e] as usize..per[e + 1] as usize];
            if sub.len() > 1 { let st = &t.ap_stop_at; sub.sort_by(|&x, &y| (st[x as usize] as f64 - st[y as usize] as f64).partial_cmp(&0.0).unwrap_or(std::cmp::Ordering::Equal)); } }
        t.edge_ap_start = per; t.edge_ap_list = list;
        t
    }

    fn plan_signal(&mut self, g: &SimGraph, c: usize, aps: &[usize], xyz: &[f32]) {
        if aps.is_empty() { self.cycle[c] = 60.0; self.phases[c] = vec![Phase { group: 0, left: false, g: 54.0, y: 4.0, r: 2.0 }]; return; }
        use std::f64::consts::PI;
        let ax0 = axis_of(g.heading_end[self.ap_edge[aps[0]] as usize]);
        let (mut w, mut spd, mut big) = ([0.0f64; 2], [0.0f64; 2], [false; 2]);
        for &a in aps {
            let e = self.ap_edge[a] as usize;
            let mut d = (axis_of(g.heading_end[e]) - ax0).abs(); if d > PI / 2.0 { d = PI - d; }
            let grp = if d < PI / 4.0 { 0 } else { 1 };
            self.ap_group[a] = grp as i8;
            w[grp] = jsmath::max(w[grp], class_w(g.class[e]));
            spd[grp] = jsmath::max(spd[grp], g.speed_limit[e]);
            let cls = g.class[e];
            if (cls == 4 || cls == 2 || cls == 6) && g.lanes[e] >= 2 && aps.len() >= 3 { big[grp] = true; }
        }
        let groups: Vec<usize> = if w[1] > 0.0 { vec![0, 1] } else { vec![0] };
        let yel = |v: f64| jsmath::min(5.0, jsmath::max(3.0, 1.0 + v / (2.0 * 3.05))); let all_red = 1.5;
        let mut phases = Vec::new();
        for &k in &groups {
            if big[k] { phases.push(Phase { group: k as u8, left: true, g: 0.0, y: 3.0, r: all_red }); }
            phases.push(Phase { group: k as u8, left: false, g: 0.0, y: yel(spd[k]), r: all_red });
        }
        let any_big = big[0] || big[1]; let arterial = jsmath::max(w[0], w[1]) >= 2.5;
        let cyc = if any_big { 120.0 } else if arterial { 90.0 } else { 60.0 };
        let mut lost = 0.0; for p in &phases { lost += p.y + p.r; }
        let avail = jsmath::max(10.0 * phases.len() as f64, cyc - lost);
        let mut left_share = 0.0; for p in &phases { if p.left { left_share += 0.12; } }
        let through_avail = avail * (1.0 - left_share);
        let wsum = groups.iter().fold(0.0, |s, &k| s + w[k]);
        for p in phases.iter_mut() { p.g = if p.left { jsmath::max(7.0, avail * 0.12) } else { jsmath::max(10.0, through_avail * w[p.group as usize] / wsum) }; }
        let mut tot = 0.0; for p in &phases { tot += p.g + p.y + p.r; }
        self.cycle[c] = tot as f32;
        let major = if w[1] > w[0] { 1 } else { 0 };
        let maj = *aps.iter().find(|&&a| self.ap_group[a] as usize == major).unwrap();
        let h = g.heading_end[self.ap_edge[maj] as usize]; let (x, z) = (xyz[c * 3] as f64, xyz[c * 3 + 2] as f64);
        let along = x * jsmath::cos(h) + z * jsmath::sin(h); let v = jsmath::max(8.0, spd[major] * 0.9);
        self.offset[c] = (((along / v) % tot + tot) % tot) as f32;
        for &a in aps { if big[self.ap_group[a] as usize] { self.ap_prot_left[a] = 1; } }
        self.phases[c] = phases;
    }

    /// First stop line on edge e at or ahead of s (front bumper), -1 none.
    pub fn approach_ahead(&self, e: usize, s: f64) -> i32 {
        for k in self.edge_ap_start[e]..self.edge_ap_start[e + 1] { let a = self.edge_ap_list[k as usize]; if self.ap_stop_at[a as usize] as f64 - s > -1.0 { return a; } }
        -1
    }

    /// `stateOf(a, move, t)`.
    pub fn state_of(&self, a: usize, mv: u8, t: f64) -> CtlState {
        let c = self.ap_control[a] as usize; let ty = self.ty[c];
        let mut out = CtlState { ty, state: 0, time_in: 0.0, time_left: 1e9 };
        if ty == ctrl::STOP || ty == ctrl::GIVE_WAY { out.state = light::SIGN; return out; }
        let cy = self.cycle[c] as f64;
        let tc = ((t + self.offset[c] as f64) % cy + cy) % cy;
        if ty == ctrl::RAMP_METER {
            let tod = t % 86400.0;
            if tod < self.ramp_start || tod > self.ramp_end { out.state = light::DARK; return out; }
            if tc < 2.0 { out.state = light::GREEN; out.time_in = tc; out.time_left = 2.0 - tc; } else { out.state = light::RED; out.time_in = tc - 2.0; out.time_left = cy - tc; }
            return out;
        }
        if ty == ctrl::CROSSING {
            if tc < 75.0 { out.state = light::GREEN; out.time_in = tc; out.time_left = 75.0 - tc; }
            else if tc < 78.0 { out.state = light::YELLOW; out.time_in = tc - 75.0; out.time_left = 78.0 - tc; }
            else { out.state = light::RED; out.time_in = tc - 78.0; out.time_left = cy - tc; }
            return out;
        }
        let ph = &self.phases[c]; let grp = self.ap_group[a] as u8; let prot = self.ap_prot_left[a] == 1;
        let mine = |p: &Phase| p.group == grp && (if mv == movement::LEFT && prot { p.left } else { !p.left });
        let n = ph.len();
        let (mut t0, mut cur, mut in_phase) = (0.0f64, usize::MAX, 0.0f64);
        for (k, p) in ph.iter().enumerate() { let len = p.g + p.y + p.r; if tc < t0 + len { cur = k; in_phase = tc - t0; break; } t0 += len; }
        if cur == usize::MAX { cur = n - 1; in_phase = 0.0; }
        let p = &ph[cur];
        if mine(p) && in_phase < p.g { out.state = light::GREEN; out.time_in = in_phase; out.time_left = p.g - in_phase; return out; }
        if mine(p) && in_phase < p.g + p.y { out.state = light::YELLOW; out.time_in = in_phase - p.g; out.time_left = p.g + p.y - in_phase; return out; }
        out.state = light::RED;
        let mut until = (p.g + p.y + p.r) - in_phase; let mut k = (cur + 1) % n; let mut guard = 0;
        while !mine(&ph[k]) && { let g0 = guard; guard += 1; g0 < n } { until += ph[k].g + ph[k].y + ph[k].r; k = (k + 1) % n; }
        out.time_left = until;
        let mut since = if mine(p) { in_phase - p.g - p.y } else { in_phase }; k = (cur + n - 1) % n; guard = 0;
        while !mine(&ph[k]) && { let g0 = guard; guard += 1; g0 < n } { since += ph[k].g + ph[k].y + ph[k].r; k = (k + n - 1) % n; }
        if mine(&ph[k]) && k != cur { since += ph[k].r; }
        out.time_in = jsmath::max(0.0, since);
        out
    }
}
