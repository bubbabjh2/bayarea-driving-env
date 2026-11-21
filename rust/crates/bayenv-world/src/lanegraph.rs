//! `src/geom/graph.js`: the minimal merged directed graph the lane-geometry / junction builders work on.
use crate::roadpack::RoadSet;

pub const LANE_W: f64 = 3.7;
/// Default lane width per class (CLASSES_ALL order) when the pack has no edgeLaneWidth.
pub const LANE_WIDTH_CLASS: [f32; 13] = [3.7, 3.7, 3.7, 3.6, 3.6, 3.5, 3.5, 3.4, 3.3, 3.3, 3.1, 3.1, 3.0];

/// LaneGraph over main (+ local) tiers, numbered exactly like the sim (`graphFromRoadSets`).
#[derive(Clone, Default)]
pub struct LaneGraph {
    /// 3 per node (f64, from the f32 packs)
    pub xyz: Vec<f64>,
    pub first: Vec<u32>, pub count: Vec<u32>, pub poly: Vec<u32>,
    pub lanes: Vec<u8>, pub opp: Vec<u8>, pub twin: Vec<i32>, pub cls: Vec<u8>,
    pub length: Vec<f32>, pub flags: Vec<u32>, pub lane_w: Vec<f32>,
    pub node_count: usize, pub edge_count: usize,
    pub out_start: Vec<u32>, pub in_start: Vec<u32>, pub out_list: Vec<u32>, pub in_list: Vec<u32>,
    pub main_edge_count: usize,
    /// per tier: tier node -> merged node
    pub tier_node_map: Vec<Vec<i32>>,
}

impl LaneGraph {
    #[inline] pub fn from(&self, e: usize) -> usize { self.poly[self.first[e] as usize] as usize }
    #[inline] pub fn to(&self, e: usize) -> usize { self.poly[(self.first[e] + self.count[e] - 1) as usize] as usize }
    #[inline] pub fn nodes_of(&self, e: usize) -> &[u32] { &self.poly[self.first[e] as usize..(self.first[e] + self.count[e]) as usize] }
    #[inline] pub fn out_edges(&self, n: usize) -> &[u32] { &self.out_list[self.out_start[n] as usize..self.out_start[n + 1] as usize] }
    #[inline] pub fn in_edges(&self, n: usize) -> &[u32] { &self.in_list[self.in_start[n] as usize..self.in_start[n + 1] as usize] }
    #[inline] pub fn is_freeway(&self, e: usize) -> bool { self.cls[e] <= 3 }
    #[inline] pub fn x(&self, n: usize) -> f64 { self.xyz[3 * n] }
    #[inline] pub fn y(&self, n: usize) -> f64 { self.xyz[3 * n + 1] }
    #[inline] pub fn z(&self, n: usize) -> f64 { self.xyz[3 * n + 2] }

    fn finish(&mut self) {
        let (n, e) = (self.xyz.len() / 3, self.first.len());
        self.node_count = n; self.edge_count = e;
        let mut oc = vec![0u32; n + 1]; let mut ic = vec![0u32; n + 1];
        for k in 0..e { oc[self.from(k) + 1] += 1; ic[self.to(k) + 1] += 1; }
        for i in 0..n { oc[i + 1] += oc[i]; ic[i + 1] += ic[i]; }
        let (mut of, mut inf) = (oc[..n].to_vec(), ic[..n].to_vec());
        let (mut ol, mut il) = (vec![0u32; e], vec![0u32; e]);
        for k in 0..e {
            let (f, t) = (self.from(k), self.to(k));
            ol[of[f] as usize] = k as u32; of[f] += 1;
            il[inf[t] as usize] = k as u32; inf[t] += 1;
        }
        self.out_start = oc; self.in_start = ic; self.out_list = ol; self.in_list = il;
    }
}

/// Merged node numbering of `fromRoadSet` / `graphFromRoadSets`: main nodes keep their ids, local nodes in
/// linkNodes reuse the main index, other local nodes are appended in order. Returns (N, per-tier maps).
pub fn merged_node_maps(sets: &[&RoadSet]) -> (usize, Vec<Vec<i32>>) {
    let mut n = 0usize; let mut maps = Vec::new();
    for (si, rs) in sets.iter().enumerate() {
        let mut map = vec![-1i32; rs.node_count];
        if si > 0 { if let Some(l) = &rs.link_nodes { for k in (0..l.len()).step_by(2) { map[l[k] as usize] = l[k + 1] as i32; } } }
        for m in map.iter_mut() { if *m < 0 { *m = n as i32; n += 1; } }
        maps.push(map);
    }
    (n, maps)
}

/// `graphFromRoadSets(main, local)`.
pub fn graph_from_road_sets(main: &RoadSet, local: Option<&RoadSet>) -> LaneGraph {
    let sets: Vec<&RoadSet> = std::iter::once(main).chain(local).collect();
    let (n, maps) = merged_node_maps(&sets);
    let mut g = LaneGraph { xyz: vec![0.0; 3 * n], ..Default::default() };
    let (mut e_tot, mut p_tot) = (0, 0);
    for (si, rs) in sets.iter().enumerate() {
        for i in 0..rs.node_count { let k = maps[si][i] as usize; g.xyz[3 * k] = rs.x(i); g.xyz[3 * k + 1] = rs.y(i); g.xyz[3 * k + 2] = rs.z(i); }
        e_tot += rs.edge_count; p_tot += rs.poly.len();
    }
    g.first.reserve(e_tot); g.poly.reserve(p_tot);
    let mut base = 0i64;
    for (si, rs) in sets.iter().enumerate() {
        for e in 0..rs.edge_count {
            g.first.push(g.poly.len() as u32);
            let nodes = rs.edge_nodes(e);
            g.count.push(nodes.len() as u32);
            for &nd in nodes { g.poly.push(maps[si][nd as usize] as u32); }
            g.lanes.push(rs.lanes(e) as u8); g.opp.push(rs.opp_lanes(e) as u8);
            let t = rs.twin(e); g.twin.push(if t >= 0 { (base + t) as i32 } else { -1 });
            let c = rs.class_id(e) as u8; g.cls.push(c);
            g.length.push(rs.edge_length[e]); g.flags.push(rs.flags(e));
            g.lane_w.push(match &rs.edge_lane_width { Some(w) => w[e], None => LANE_WIDTH_CLASS[c as usize] });
        }
        base += rs.edge_count as i64;
    }
    g.main_edge_count = main.edge_count;
    g.tier_node_map = maps;
    g.finish();
    g
}
