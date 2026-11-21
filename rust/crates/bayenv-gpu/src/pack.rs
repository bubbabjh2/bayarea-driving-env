//! World pack for the GPU kernels: port of `src/gpu/worldPack.js` (packWorld, computeStopBack, packRoutes /
//! packRoute, packControls, extraConflicts, packJunctions, connectorSpeed) and of the world-blob assembly in
//! `GPUSim._init` (section offsets `layoutOff`, grade-separation pairs, junction mouth / gore interleave,
//! movements-into CSR). Byte-identical to the JS blob (checked against `rust/oracle/gpu10k`).
//!
//! Blob layout (u32 words, in order): edges (E x 16) | pts (P x 4 f32) | laneOff (K f32) | adj
//! [outStart N+1 | inStart N+1 | outEdges E | inEdges E] | edge heads (E, zero) | crash blocks (maxBlocks x 8, zero)
//! | geo (E x 12) | lgVerts (V x 6 f32: x, z, y, s, mx, mz) | hops | ctl edgeAp | ctl approaches | ctl controls | pairs (2 per pair)
//! | junction mouth/gore (3E f32) | movements (9 per) | edgeMov CSR | connector pts | conflict CSR | movements-into CSR.
use bayenv_world::hops::HopTable;
use bayenv_world::jsmath;
use bayenv_world::junctions::JunctionSet;
use bayenv_world::nav::{self, ManeuverTable, EDGE_ID_MASK};
use bayenv_world::{SimGraph, TrafficControls, World};

pub const EDGE_WORDS: usize = 16;
pub const GEO_WORDS: usize = 12;
/// layout.js BLOCK_WORDS
pub const BLOCK_WORDS: usize = 8;
/// GPU_DEFAULTS.maxBlocks
pub const MAX_BLOCKS: usize = 4096;
const LANE_WIDTH: f64 = 3.7;

#[inline] fn fb(x: f64) -> u32 { (x as f32).to_bits() }

/// `GPUSim.layoutOff`: word offsets of the blob sections.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LayoutOff {
    pub pts_off: usize, pub lane_off_off: usize, pub adj_off: usize, pub eb_off: usize, pub blk_off: usize,
    pub geo_off: usize, pub lgv_off: usize, pub hop_off: usize, pub ctl_edge_off: usize, pub ctl_ap_off: usize,
    pub ctl_c_off: usize, pub pair_off: usize, pub j_mouth_off: usize, pub j_mov_off: usize, pub j_edge_mov_off: usize,
    pub j_pts_off: usize, pub j_conf_off: usize, pub j_into_off: usize,
}
impl LayoutOff {
    /// (JS name, value) in `layoutOff` key order.
    pub fn entries(&self) -> [(&'static str, usize); 18] {
        [("ptsOff", self.pts_off), ("laneOffOff", self.lane_off_off), ("adjOff", self.adj_off), ("ebOff", self.eb_off), ("blkOff", self.blk_off),
         ("geoOff", self.geo_off), ("lgvOff", self.lgv_off), ("hopOff", self.hop_off), ("ctlEdgeOff", self.ctl_edge_off), ("ctlApOff", self.ctl_ap_off),
         ("ctlCOff", self.ctl_c_off), ("pairOff", self.pair_off), ("jMouthOff", self.j_mouth_off), ("jMovOff", self.j_mov_off),
         ("jEdgeMovOff", self.j_edge_mov_off), ("jPtsOff", self.j_pts_off), ("jConfOff", self.j_conf_off), ("jIntoOff", self.j_into_off)]
    }
}

/// `packControls` metadata (`GPUSim.ctlPack` minus the arrays, which are in the blob).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CtlMeta { pub ramp_start: f64, pub ramp_end: f64, pub list_off: usize, pub n_a: usize, pub n_c: usize }

/// `packJunctions` counts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct JunctionMeta { pub m: usize, pub p: usize, pub c: usize, pub extra: usize }

pub struct WorldBlob {
    /// the `world` storage buffer (byte size = `max(16, 4 * words.len())`, see [`WorldBlob::bytes`])
    pub words: Vec<u32>,
    pub off: LayoutOff,
    pub e: usize, pub n: usize, pub p: usize, pub k: usize,
    pub has_lg: bool,
    pub pair_n: usize,
    /// `jPack.M` (None = no junctions packed)
    pub j_m: Option<usize>,
    pub junctions: Option<JunctionMeta>,
    pub ctl: Option<CtlMeta>,
}
impl WorldBlob {
    /// Buffer image as `gpu.buffer()` sizes it (>= 16 bytes, multiple of 4).
    pub fn bytes(&self) -> Vec<u8> {
        let mut b: Vec<u8> = bytemuck::cast_slice(&self.words).to_vec();
        if b.len() < 16 { b.resize(16, 0); }
        b
    }
    /// `laneBase(e)`
    pub fn lane_base(&self, e: usize) -> u32 { self.words[e * EDGE_WORDS + 3] }
}

/// `computeStopBack(g)` (sim.js yieldRequired's e.stopBack).
pub fn compute_stop_back(g: &SimGraph) -> Vec<f32> {
    let lw = LANE_WIDTH;
    (0..g.edge_count).map(|i| {
        let t = g.to[i] as usize; let rev = g.reverse[i];
        let (mut inc, mut hw) = (0, 0.0f64);
        let w = |o: usize| (if g.reverse[o] >= 0 { 2.0 } else { 1.0 }) * g.lanes[o] as f64 * lw * 0.5;
        for &ei in g.in_of(t) { if ei as usize == i || ei as i32 == rev { continue; } inc += 1; hw = jsmath::max(hw, w(ei as usize)); }
        for &ei in g.out_of(t) { if ei as i32 == rev { continue; } hw = jsmath::max(hw, w(ei as usize)); }
        (if inc > 0 { jsmath::min(12.0, jsmath::max(1.5, hw + 1.0)) } else { 1.5 }) as f32
    }).collect()
}

/// Section sizes of `packWorld` (words).
struct WorldSizes { e: usize, n: usize, p: usize, k: usize, v: usize, hop_n: usize, has_lg: bool }
fn world_sizes(w: &World) -> WorldSizes {
    let g = &w.graph; let e = g.edge_count;
    let has_lg = w.lane_graph.edge_count == e && w.lanes.v_first.len() == e + 1;
    let hop_n = (0..e).map(|i| g.out_degree(g.to[i] as usize)).sum();
    WorldSizes { e, n: g.node_count, p: g.px.len(), k: g.lane_off.len(), v: if has_lg { w.lanes.v_first[e] as usize } else { 0 }, hop_n, has_lg }
}

/// `packWorld(graph)` written into the blob: edges, pts, laneOff, adj at `[0, eb_off)` and geo, lgVerts, hops
/// at `[geo_off, ctl_edge_off)`.
fn pack_world_into(w: &World, s: &WorldSizes, off: &LayoutOff, blob: &mut [u32]) {
    let g = &w.graph; let e_n = s.e;
    let stop_back = compute_stop_back(g);
    // edges
    for i in 0..e_n {
        let o = i * EDGE_WORDS; let ed = &mut blob[o..o + EDGE_WORDS];
        let c = g.class[i];
        ed[0] = g.pt_start[i]; ed[1] = g.pt_start[i + 1] - g.pt_start[i]; ed[2] = g.lanes[i] as u32; ed[3] = g.lane_start[i];
        ed[4] = fb(g.length[i]); ed[5] = fb(g.speed_limit[i]); ed[6] = g.from[i]; ed[7] = g.to[i];
        ed[8] = g.reverse[i] as u32; ed[9] = g.rank[i] as u32;
        let fw = c == 0 || c == 2; let sys = c <= 3;
        ed[10] = (g.is_link(i) as u32) | if g.has(i, bayenv_world::graph::ef::BRIDGE) { 2 } else { 0 } | if g.has(i, bayenv_world::graph::ef::TUNNEL) { 4 } else { 0 }
            | if fw { 16 } else { 0 } | if sys { 32 | 64 } else { 0 }
            | ((g.surf[i] as u32) & 15) << 16 | ((g.gnd_r[i] as u32) & 15) << 20 | ((g.gnd_l[i] as u32) & 15) << 24;
        ed[11] = stop_back[i].to_bits();
        ed[12] = fb(g.heading_start[i]); ed[13] = fb(g.heading_end[i]); ed[14] = g.y[i].to_bits(); ed[15] = g.nav_rank[i] as u32;
    }
    // pts (x, z, y, cum)
    let pts = &mut blob[off.pts_off..off.pts_off + 4 * s.p];
    for j in 0..s.p { let q = &mut pts[4 * j..4 * j + 4]; q[0] = fb(g.px[j]); q[1] = fb(g.pz[j]); q[2] = g.ys[j].to_bits(); q[3] = fb(g.cum[j]); }
    for (d, &v) in blob[off.lane_off_off..].iter_mut().zip(g.lane_off.iter()) { *d = v.to_bits(); }
    let a = off.adj_off; let n1 = s.n + 1;
    blob[a..a + n1].copy_from_slice(&g.out_start); blob[a + n1..a + 2 * n1].copy_from_slice(&g.in_start);
    blob[a + 2 * n1..a + 2 * n1 + e_n].copy_from_slice(&g.out_edges); blob[a + 2 * n1 + e_n..a + 2 * n1 + 2 * e_n].copy_from_slice(&g.in_edges);
    // geo + lgVerts
    let lg = &w.lanes;
    {
        let geo = &mut blob[off.geo_off..off.geo_off + e_n * GEO_WORDS];
        if s.has_lg {
            let (has_x, has_w, has_we) = (lg.x_r.len() >= 2 * e_n && lg.x_l.len() >= 2 * e_n, lg.lane_w.len() >= e_n, lg.lane_w_end.len() >= 2 * e_n);
            for e in 0..e_n {
                let o = &mut geo[e * GEO_WORDS..(e + 1) * GEO_WORDS];
                o[0] = lg.v_first[e]; o[1] = lg.rd[e].to_bits(); o[2] = lg.rend[2 * e].to_bits(); o[3] = lg.rend[2 * e + 1].to_bits();
                o[4] = lg.tau[2 * e].to_bits(); o[5] = lg.tau[2 * e + 1].to_bits();
                o[6] = lg.tap_r[2 * e] as u32 | (lg.tap_r[2 * e + 1] as u32) << 8 | (lg.tap_l[2 * e] as u32) << 16 | (lg.tap_l[2 * e + 1] as u32) << 24;
                if has_x { o[8] = lg.x_r[2 * e] as u32 | (lg.x_r[2 * e + 1] as u32) << 8 | (lg.x_l[2 * e] as u32) << 16 | (lg.x_l[2 * e + 1] as u32) << 24; }
                let wd = if has_w { lg.lane_w[e] } else { 3.7f32 };
                o[9] = wd.to_bits();
                o[10] = if has_we { lg.lane_w_end[2 * e] } else { wd }.to_bits();
                o[11] = if has_we { lg.lane_w_end[2 * e + 1] } else { wd }.to_bits();
            }
            let lv = &mut blob[off.lgv_off..off.lgv_off + 6 * s.v];
            for v in 0..s.v {
                let q = &mut lv[6 * v..6 * v + 6];
                q[0] = fb(lg.vx[v]); q[1] = fb(lg.vz[v]); q[2] = lg.vy[v].to_bits(); q[3] = fb(lg.vs[v]); q[4] = fb(lg.mx[v]); q[5] = fb(lg.mz[v]);
            }
        }
    }
    // hop table: geo[e * 12 + 7] = base; strict | (clamp(shift, -64, 63) + 64) << 1 per out-edge of e.to
    let mut base = 0u32;
    for e in 0..e_n { blob[off.geo_off + e * GEO_WORDS + 7] = base; base += g.out_degree(g.to[e] as usize) as u32; }
    if s.has_lg {
        let ht = HopTable::new(&w.lane_graph);
        let mut q = off.hop_off;
        for e in 0..e_n { for &en in g.out_of(g.to[e] as usize) {
            let h = ht.get(&w.lane_graph, e, en as usize);
            blob[q] = (h.strict as u32) | (((h.shift.clamp(-64, 63) + 64) as u32) << 1); q += 1;
        } }
    } else {
        // roadGraph.js hopMap without lane geometry is not ported (every real world has lane geometry)
        unimplemented!("packWorld without lane geometry");
    }
}

/// `packControls(C, E)`: (edgeAp, ap, ctl, meta).
pub fn pack_controls(c: &TrafficControls, e_n: usize) -> (Vec<u32>, Vec<u32>, Vec<u32>, CtlMeta) {
    let (start, list): (Vec<u32>, Vec<u32>) = if !c.edge_ap_start.is_empty() {
        (c.edge_ap_start.iter().map(|&x| x as u32).collect(), c.edge_ap_list.iter().map(|&x| x as u32).collect())
    } else {
        let mut s = vec![0u32; e_n + 1]; let mut l = Vec::new();
        for e in 0..e_n { let a = c.edge_approach.get(e).copied().unwrap_or(-1); if a >= 0 { l.push(a as u32); } s[e + 1] = l.len() as u32; }
        (s, l)
    };
    let (n_a, n_c) = (c.approach_count, c.count);
    let mut ap = vec![0u32; n_a.max(1) * 4];
    for a in 0..n_a {
        ap[4 * a] = c.ap_control[a] as u32; ap[4 * a + 1] = c.ap_stop_at[a].to_bits();
        ap[4 * a + 2] = (c.ap_group[a] as i32 as u32 & 255) | if c.ap_prot_left[a] != 0 { 256 } else { 0 };
        ap[4 * a + 3] = c.ap_edge[a].max(0) as u32;
    }
    const R: usize = 24;
    let mut ctl = vec![0u32; n_c.max(1) * R];
    for k in 0..n_c {
        let o = R * k; let ph = &c.phases[k];
        let cy = c.cycle[k]; let of = c.offset[k];
        ctl[o] = c.ty[k] as i32 as u32;
        ctl[o + 1] = if cy != 0.0 && !cy.is_nan() { cy } else { 60.0 }.to_bits();
        ctl[o + 2] = if of != 0.0 && !of.is_nan() { of } else { 0.0 }.to_bits();
        let np = ph.len().min(4); ctl[o + 3] = np as u32;
        for (j, p) in ph.iter().take(4).enumerate() {
            let q = o + 4 + 4 * j;
            ctl[q] = (p.group as u32 & 255) | if p.left { 256 } else { 0 }; ctl[q + 1] = fb(p.g); ctl[q + 2] = fb(p.y); ctl[q + 3] = fb(p.r);
        }
        ctl[o + 20] = if c.cluster_of.len() == n_c { c.cluster_of[k] as u32 } else { k as u32 };
    }
    let list_off = start.len();
    let mut edge_ap = start; edge_ap.extend_from_slice(&list);
    (edge_ap, ap, ctl, CtlMeta { ramp_start: c.ramp_start, ramp_end: c.ramp_end, list_off, n_a, n_c })
}

/// `extraConflicts(J, edgeTo)`: [A, B, sA, sB] per extra conflict (A < B), as the JS Float32Array.
pub fn extra_conflicts(j: &JunctionSet, edge_to: &[u32]) -> Vec<f32> {
    let mi = &j.m_info; let pts = &j.pts; let m_n = mi.len() / 7; const R: f64 = 4.2;
    // byNode (Map insertion order = first movement of each node)
    let mut node_slot: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    let mut groups: Vec<Vec<u32>> = Vec::new();
    for m in 0..m_n {
        let nd = edge_to[mi[7 * m] as usize];
        let s = *node_slot.entry(nd).or_insert_with(|| { groups.push(Vec::new()); groups.len() - 1 });
        groups[s].push(m as u32);
    }
    // known conflicts: CSR of partners per movement, sorted
    let c0 = j.conf.len() / 3;
    let mut ks = vec![0u32; m_n + 1];
    for c in 0..c0 { ks[j.conf[3 * c] as usize + 1] += 1; ks[j.conf[3 * c + 1] as usize + 1] += 1; }
    for m in 0..m_n { ks[m + 1] += ks[m]; }
    let mut fill = ks[..m_n].to_vec(); let mut kl = vec![0u32; ks[m_n] as usize];
    for c in 0..c0 { let (a, b) = (j.conf[3 * c] as usize, j.conf[3 * c + 1] as usize); kl[fill[a] as usize] = b as u32; fill[a] += 1; kl[fill[b] as usize] = a as u32; fill[b] += 1; }
    for m in 0..m_n { kl[ks[m] as usize..ks[m + 1] as usize].sort_unstable(); }
    let known = |a: usize, b: u32| kl[ks[a] as usize..ks[a + 1] as usize].binary_search(&b).is_ok();
    // bounding boxes (Float32Array)
    let mut bb = vec![0f32; 4 * m_n];
    for m in 0..m_n {
        let (f, n) = (mi[7 * m + 4] as usize, mi[7 * m + 5] as usize);
        let (mut x0, mut x1, mut z0, mut z1) = (1e30f64, -1e30f64, 1e30f64, -1e30f64);
        for k in 0..n { let (x, z) = (pts[4 * (f + k)] as f64, pts[4 * (f + k) + 2] as f64); if x < x0 { x0 = x; } if x > x1 { x1 = x; } if z < z0 { z0 = z; } if z > z1 { z1 = z; } }
        bb[4 * m] = x0 as f32; bb[4 * m + 1] = x1 as f32; bb[4 * m + 2] = z0 as f32; bb[4 * m + 3] = z1 as f32;
    }
    let run = |list: &[u32], out: &mut Vec<f32>| {
        for p in 0..list.len() {
            let m = list[p] as usize; let (f1, n1) = (mi[7 * m + 4] as usize, mi[7 * m + 5] as usize);
            for &o in &list[p + 1..] {
                let o = o as usize;
                if mi[7 * o] == mi[7 * m] && mi[7 * o + 1] == mi[7 * m + 1] { continue; }
                if known(m, o as u32) { continue; }
                let b = |k: usize| bb[k] as f64;
                if b(4 * o) > b(4 * m + 1) + R || b(4 * m) > b(4 * o + 1) + R || b(4 * o + 2) > b(4 * m + 3) + R || b(4 * m + 2) > b(4 * o + 3) + R { continue; }
                let (f2, n2) = (mi[7 * o + 4] as usize, mi[7 * o + 5] as usize);
                let (mut best, mut sa, mut sb) = (1e18f64, 0f32, 0f32);
                for i in 0..n1 {
                    let (xi, zi) = (pts[4 * (f1 + i)] as f64, pts[4 * (f1 + i) + 2] as f64);
                    for k in 0..n2 {
                        let (dx, dz) = (xi - pts[4 * (f2 + k)] as f64, zi - pts[4 * (f2 + k) + 2] as f64); let d = dx * dx + dz * dz;
                        if d < best { best = d; sa = pts[4 * (f1 + i) + 3]; sb = pts[4 * (f2 + k) + 3]; }
                    }
                }
                if best < R * R { out.extend_from_slice(&[m as f32, o as f32, sa, sb]); }
            }
        }
    };
    // parallel over contiguous group ranges, concatenated in order
    let nt = std::thread::available_parallelism().map_or(4, |n| n.get()).min(16);
    let per = groups.len().div_ceil(nt * 8).max(1);
    let chunks: Vec<&[Vec<u32>]> = groups.chunks(per).collect();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut results: Vec<Vec<f32>> = vec![Vec::new(); chunks.len()];
    let res_ptr: Vec<std::sync::Mutex<Vec<f32>>> = (0..chunks.len()).map(|_| std::sync::Mutex::new(Vec::new())).collect();
    std::thread::scope(|sc| {
        for _ in 0..nt { sc.spawn(|| loop {
            let c = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed); if c >= chunks.len() { break; }
            let mut out = Vec::new(); for l in chunks[c] { run(l, &mut out); }
            *res_ptr[c].lock().unwrap() = out;
        }); }
    });
    for (c, r) in res_ptr.into_iter().enumerate() { results[c] = r.into_inner().unwrap(); }
    results.concat()
}

/// `connectorSpeed(J, m)`: comfortable speed (m/s) through connector m, as the JS Float32Array cache holds it.
pub fn connector_speed(j: &JunctionSet, m: usize) -> f32 {
    let first = j.m_info[7 * m + 4] as usize; let n = j.m_info[7 * m + 5] as usize; let p = &j.pts;
    let pf = |i: usize| p[i] as f64;
    let dir = |k: usize| { let a = 4 * (first + k); jsmath::atan2(pf(a + 6) - pf(a + 2), pf(a + 4) - pf(a)) };
    let mut kap = 0.0f64;
    if n >= 1 {
        for k in 0..n - 1 {
            let mut q = k + 1;
            while q < n - 1 && pf(4 * (first + q) + 3) - pf(4 * (first + k) + 3) < 8.0 { q += 1; }
            let l = pf(4 * (first + q) + 3) - pf(4 * (first + k) + 3); if l < 1.0 { continue; }
            let mut d = dir(q.min(n - 2)) - dir(k); d = jsmath::atan2(jsmath::sin(d), jsmath::cos(d)).abs();
            kap = jsmath::max(kap, d / jsmath::max(l, 8.0));
            if q >= n - 1 { break; }
        }
    }
    if n <= 2 { kap = (j.m_f[4 * m + 3] as f64).abs() / jsmath::max(8.0, j.m_f[4 * m + 2] as f64); }
    (if kap > 1e-4 { jsmath::max(2.5, jsmath::sqrt(3.0 / kap)) } else { 1e3 }) as f32
}

/// `packJunctions(J, E, connectorSpeed, extra)` written into the blob at the junction offsets (mouth/gore
/// interleaved as GPUSim._init stores them), plus the movements-into CSR.
fn pack_junctions_into(j: &JunctionSet, e_n: usize, extra: &[f32], off: &LayoutOff, blob: &mut [u32]) -> JunctionMeta {
    let mi = &j.m_info; let m_n = mi.len() / 7;
    // mouth / gore: [sExit, sStop, gore] per edge
    for e in 0..e_n { let o = off.j_mouth_off + 3 * e; blob[o] = j.mouth_s[2 * e].to_bits(); blob[o + 1] = j.mouth_s[2 * e + 1].to_bits(); blob[o + 2] = j.gore_s[e].to_bits(); }
    // movements (9 words)
    {
        let mov = &mut blob[off.j_mov_off..off.j_mov_off + 9 * m_n];
        let nt = std::thread::available_parallelism().map_or(4, |n| n.get()).min(16);
        let per = m_n.div_ceil(nt).max(1);
        std::thread::scope(|sc| {
            for (ci, ch) in mov.chunks_mut(9 * per).enumerate() { sc.spawn(move || {
                for (r, o) in ch.chunks_mut(9).enumerate() {
                    let m = ci * per + r; let i = 7 * m;
                    o[0] = (mi[i] & EDGE_ID_MASK) | ((mi[i + 1] & 255) << 20) | ((mi[i + 6] != 0) as u32) << 28;
                    o[1] = (mi[i + 2] & EDGE_ID_MASK) | ((mi[i + 3] & 255) << 20);
                    o[2] = mi[i + 4]; o[3] = mi[i + 5];
                    for q in 0..4 { o[4 + q] = j.m_f[4 * m + q].to_bits(); }
                    o[8] = connector_speed(j, m).to_bits();
                }
            }); }
        });
    }
    // edgeMov CSR (sorted by (inLane, outEdge), stable)
    let csr = |key: usize, dst: &mut [u32], sort: bool| {
        let (cnt, list) = dst.split_at_mut(e_n + 1);
        cnt.fill(0);
        for m in 0..m_n { cnt[mi[7 * m + key] as usize + 1] += 1; }
        for e in 0..e_n { cnt[e + 1] += cnt[e]; }
        let mut fill = cnt[..e_n].to_vec();
        for m in 0..m_n { let e = mi[7 * m + key] as usize; list[fill[e] as usize] = m as u32; fill[e] += 1; }
        if sort { for e in 0..e_n { let sub = &mut list[cnt[e] as usize..cnt[e + 1] as usize];
            if sub.len() > 1 { sub.sort_by(|&x, &y| (mi[7 * x as usize + 1], mi[7 * x as usize + 2]).cmp(&(mi[7 * y as usize + 1], mi[7 * y as usize + 2]))); } } }
    };
    csr(0, &mut blob[off.j_edge_mov_off..off.j_edge_mov_off + e_n + 1 + m_n], true);
    // connector points [x, z, y, s]
    let p_n = j.pts.len() / 4;
    for p in 0..p_n { let o = off.j_pts_off + 4 * p; let a = &j.pts[4 * p..4 * p + 4]; blob[o] = a[0].to_bits(); blob[o + 1] = a[2].to_bits(); blob[o + 2] = a[1].to_bits(); blob[o + 3] = a[3].to_bits(); }
    // conflicts CSR: M + 1 starts, then [other, type, sMine, sOther] x 2C
    let c0 = j.conf.len() / 3; let x_n = extra.len() / 4; let c_n = c0 + x_n;
    let ca = |c: usize| if c < c0 { j.conf[3 * c] as usize } else { extra[4 * (c - c0)] as usize };
    let cb = |c: usize| if c < c0 { j.conf[3 * c + 1] as usize } else { extra[4 * (c - c0) + 1] as usize };
    {
        let (cc, ce) = blob[off.j_conf_off..off.j_conf_off + m_n + 1 + 8 * c_n].split_at_mut(m_n + 1);
        for c in 0..c_n { cc[ca(c) + 1] += 1; cc[cb(c) + 1] += 1; }
        for m in 0..m_n { cc[m + 1] += cc[m]; }
        let mut cf = cc[..m_n].to_vec();
        for c in 0..c_n {
            let (a, b) = (ca(c), cb(c));
            let (t, sa, sb) = if c < c0 { (j.conf[3 * c + 2], j.conf_s[2 * c], j.conf_s[2 * c + 1]) } else { (0, extra[4 * (c - c0) + 2], extra[4 * (c - c0) + 3]) };
            let k = cf[a] as usize; cf[a] += 1; ce[4 * k] = b as u32; ce[4 * k + 1] = t; ce[4 * k + 2] = sa.to_bits(); ce[4 * k + 3] = sb.to_bits();
            let k = cf[b] as usize; cf[b] += 1; ce[4 * k] = a as u32; ce[4 * k + 1] = t; ce[4 * k + 2] = sb.to_bits(); ce[4 * k + 3] = sa.to_bits();
        }
    }
    // movements into each edge (sim.js _movesInto)
    csr(2, &mut blob[off.j_into_off..off.j_into_off + e_n + 1 + m_n], false);
    JunctionMeta { m: m_n, p: p_n, c: c_n, extra: x_n }
}

/// The `world` storage buffer of `GPUSim._init` (with `opts.controls = w.controls`, junctions on).
pub fn build_world_blob(w: &World, max_blocks: usize) -> WorldBlob {
    let g = &w.graph; let s = world_sizes(w); let e_n = s.e;
    assert!(e_n <= EDGE_ID_MASK as usize + 1, "packWorld: {e_n} edges > 2^20");
    let mut o = LayoutOff::default();
    o.pts_off = e_n * EDGE_WORDS; o.lane_off_off = o.pts_off + 4 * s.p; o.adj_off = o.lane_off_off + s.k.max(1);
    o.eb_off = o.adj_off + 2 * (s.n + 1) + 2 * e_n; o.blk_off = o.eb_off + e_n;
    o.geo_off = o.blk_off + max_blocks * BLOCK_WORDS; o.lgv_off = o.geo_off + e_n * GEO_WORDS;
    o.hop_off = o.lgv_off + if s.has_lg { 6 * s.v } else { 6 };
    let ctl = w.controls.as_ref().map(|c| pack_controls(c, e_n));
    o.ctl_edge_off = o.hop_off + s.hop_n.max(1);
    o.ctl_ap_off = o.ctl_edge_off + ctl.as_ref().map_or(0, |c| c.0.len());
    o.ctl_c_off = o.ctl_ap_off + ctl.as_ref().map_or(0, |c| c.1.len());
    o.pair_off = o.ctl_c_off + ctl.as_ref().map_or(0, |c| c.2.len());
    let pairs: &[(u32, u32)] = w.contact_pairs.as_ref().map_or(&[], |p| &p.sorted);
    let jn = w.junctions.as_ref().filter(|j| j.mouth_s.len() == 2 * e_n);
    let extra = jn.map(|j| extra_conflicts(j, &g.to)).unwrap_or_default();
    let (m_n, jp_n, c_n) = jn.map_or((0, 0, 0), |j| (j.move_count(), j.pts.len() / 4, j.conflict_count() + extra.len() / 4));
    o.j_mouth_off = o.pair_off + 2 * pairs.len();
    o.j_mov_off = o.j_mouth_off + if jn.is_some() { 3 * e_n } else { 0 };
    o.j_edge_mov_off = o.j_mov_off + 9 * m_n;
    o.j_pts_off = o.j_edge_mov_off + if jn.is_some() { e_n + 1 + m_n } else { 0 };
    o.j_conf_off = o.j_pts_off + 4 * jp_n;
    o.j_into_off = o.j_conf_off + if jn.is_some() { m_n + 1 + 8 * c_n } else { 0 };
    let total = o.j_into_off + if jn.is_some() { e_n + 1 + m_n } else { 0 };
    let mut blob = vec![0u32; total];
    pack_world_into(w, &s, &o, &mut blob);
    let ctl_meta = ctl.map(|(ea, ap, ct, meta)| {
        blob[o.ctl_edge_off..o.ctl_edge_off + ea.len()].copy_from_slice(&ea);
        blob[o.ctl_ap_off..o.ctl_ap_off + ap.len()].copy_from_slice(&ap);
        blob[o.ctl_c_off..o.ctl_c_off + ct.len()].copy_from_slice(&ct);
        meta
    });
    for (k, &(a, b)) in pairs.iter().enumerate() { blob[o.pair_off + 2 * k] = a; blob[o.pair_off + 2 * k + 1] = b; }
    let jm = jn.map(|j| pack_junctions_into(j, e_n, &extra, &o, &mut blob));
    WorldBlob { words: blob, off: o, e: e_n, n: s.n, p: s.p, k: s.k, has_lg: s.has_lg, pair_n: pairs.len(), j_m: jm.map(|j| j.m), junctions: jm, ctl: ctl_meta }
}

/// Route packer (`packRoutes` / `packRoute`) with the per-hop maneuver codes precomputed once per world
/// (~20 ms); reuse it for every spawn / reroute.
pub struct RoutePacker { table: ManeuverTable }
impl RoutePacker {
    pub fn new(g: &SimGraph) -> RoutePacker { RoutePacker { table: ManeuverTable::new(g) } }
    pub fn table(&self) -> &ManeuverTable { &self.table }
    /// `packRoutes(g, flat, offset)`
    pub fn pack_routes(&self, g: &SimGraph, flat: &[i32], offset: &[u32]) -> Vec<u32> { nav::pack_routes(g, &self.table, flat, offset) }
    /// `packRoute(g, edges)`, into a caller buffer (`out.len() == route.len()`).
    pub fn pack_route_into(&self, g: &SimGraph, route: &[i32], out: &mut [u32]) { nav::pack_route_words(g, &self.table, route, out) }
    /// `packRoute(g, edges)` (`max(1, len)` words).
    pub fn pack_route(&self, g: &SimGraph, route: &[i32]) -> Vec<u32> {
        let mut out = vec![0u32; route.len().max(1)]; let n = route.len(); nav::pack_route_words(g, &self.table, route, &mut out[..n]); out
    }
}

/// One-off `packRoutes(w.graph, flat, offset)` (builds the maneuver table: use [`RoutePacker`] on hot paths).
pub fn pack_routes(w: &World, flat: &[i32], offset: &[u32]) -> Vec<u32> { RoutePacker::new(&w.graph).pack_routes(&w.graph, flat, offset) }

impl From<LayoutOff> for crate::sim::LayoutOff {
    fn from(o: LayoutOff) -> Self {
        let u = |x: usize| x as u32;
        crate::sim::LayoutOff { pts: u(o.pts_off), lane_off: u(o.lane_off_off), adj: u(o.adj_off), eb: u(o.eb_off), blk: u(o.blk_off),
            geo: u(o.geo_off), lgv: u(o.lgv_off), hop: u(o.hop_off), ctl_edge: u(o.ctl_edge_off), ctl_ap: u(o.ctl_ap_off), ctl_c: u(o.ctl_c_off),
            pair: u(o.pair_off), j_mouth: u(o.j_mouth_off), j_mov: u(o.j_mov_off), j_edge_mov: u(o.j_edge_mov_off), j_pts: u(o.j_pts_off),
            j_conf: u(o.j_conf_off), j_into: u(o.j_into_off) }
    }
}
/// For `GpuSim::with_blob`.
impl From<WorldBlob> for crate::sim::WorldBlob {
    fn from(b: WorldBlob) -> Self {
        crate::sim::WorldBlob { words: b.words, off: b.off.into(), e: b.e, n: b.n, p: b.p, k: b.k, has_lg: b.has_lg, pair_n: b.pair_n, j_m: b.j_m,
            ctl: b.ctl.map(|c| (c.list_off as u32, c.ramp_start, c.ramp_end)) }
    }
}
