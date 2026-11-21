//! One road tier (`data/roads.bin` / `data/roads-local.bin`), port of `src/roadpack.js` RoadSet.
use crate::blob::{Pack, Slice};
use std::{io, path::Path};

pub const EDGE_STRIDE: usize = 8;
pub const NO_TWIN: u32 = 0xFFFF_FFFF;
pub mod flag {
    pub const ONEWAY: u32 = 1; pub const BRIDGE: u32 = 2; pub const TUNNEL: u32 = 4; pub const LANES_FALLBACK: u32 = 8;
    pub const SPEED_DEFAULT: u32 = 16; pub const LINK: u32 = 32; pub const LANES_CLAMPED: u32 = 64; pub const LANES_SMOOTHED: u32 = 128;
    pub const AUX_LANE: u32 = 256; pub const LANE_DROP: u32 = 512;
}
/// Road classes in `classId` order (`CLASSES_ALL`).
pub const CLASSES_ALL: [&str; 13] = ["motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
    "secondary", "secondary_link", "tertiary", "tertiary_link", "unclassified", "residential", "living_street"];

/// Typed zero-copy view of one tier (all little-endian, mmap'd).
pub struct RoadSet {
    pub meta: serde_json::Value,
    pub lane_width: f64,
    /// 3 per node: x, y (elevation), z
    pub xyz: Slice<f32>,
    /// 8 per edge: firstNode, nodeCount, lanesFwd, lanesBack, classId, speedCmps, flags, twin
    pub edges: Slice<u32>,
    pub poly: Slice<u32>,
    pub edge_length: Slice<f32>,
    pub csr_offsets: Slice<u32>,
    pub csr_edges: Slice<u32>,
    /// (thisTierNode, mainTierNode) pairs, local tier only
    pub link_nodes: Option<Slice<u32>>,
    /// (inEdge, viaEdge, outEdge) triples, main tier only
    pub banned_hops: Option<Slice<u32>>,
    pub edge_lane_width: Option<Slice<f32>>,
    pub node_count: usize,
    pub edge_count: usize,
}

impl RoadSet {
    pub fn load(dir: &Path, meta_file: &str) -> io::Result<RoadSet> {
        let p = Pack::open(dir, meta_file)?;
        let xyz: Slice<f32> = p.req("nodeXYZ")?;
        let edges: Slice<u32> = p.req("edges")?;
        let (node_count, edge_count) = (xyz.len() / 3, edges.len() / EDGE_STRIDE);
        Ok(RoadSet {
            lane_width: p.meta["laneWidth"].as_f64().unwrap_or(3.7),
            poly: p.req("poly")?, edge_length: p.req("edgeLength")?, csr_offsets: p.req("csrOffsets")?, csr_edges: p.req("csrEdges")?,
            link_nodes: p.get("linkNodes"), banned_hops: p.get("bannedHops"), edge_lane_width: p.get("edgeLaneWidth"),
            xyz, edges, node_count, edge_count, meta: p.meta,
        })
    }
    #[inline] fn f(&self, e: usize, k: usize) -> u32 { self.edges[e * EDGE_STRIDE + k] }
    #[inline] pub fn first_node(&self, e: usize) -> usize { self.f(e, 0) as usize }
    #[inline] pub fn node_count_of(&self, e: usize) -> usize { self.f(e, 1) as usize }
    #[inline] pub fn lanes(&self, e: usize) -> u32 { self.f(e, 2) }
    #[inline] pub fn opp_lanes(&self, e: usize) -> u32 { self.f(e, 3) }
    #[inline] pub fn class_id(&self, e: usize) -> u32 { self.f(e, 4) }
    #[inline] pub fn speed_mps(&self, e: usize) -> f64 { self.f(e, 5) as f64 / 100.0 }
    #[inline] pub fn flags(&self, e: usize) -> u32 { self.f(e, 6) }
    #[inline] pub fn twin(&self, e: usize) -> i64 { let t = self.f(e, 7); if t == NO_TWIN { -1 } else { t as i64 } }
    #[inline] pub fn edge_nodes(&self, e: usize) -> &[u32] { let f = self.first_node(e); &self.poly[f..f + self.node_count_of(e)] }
    #[inline] pub fn from(&self, e: usize) -> u32 { self.poly[self.first_node(e)] }
    #[inline] pub fn to(&self, e: usize) -> u32 { self.poly[self.first_node(e) + self.node_count_of(e) - 1] }
    #[inline] pub fn x(&self, n: usize) -> f64 { self.xyz[3 * n] as f64 }
    #[inline] pub fn y(&self, n: usize) -> f64 { self.xyz[3 * n + 1] as f64 }
    #[inline] pub fn z(&self, n: usize) -> f64 { self.xyz[3 * n + 2] as f64 }
    /// `laneOffsets(e)` (Float32, k = 0 leftmost): ((opp - n)/2 + k + 0.5) * laneWidth
    pub fn lane_offsets(&self, e: usize) -> Vec<f32> {
        let (n, o) = (self.lanes(e) as f64, self.opp_lanes(e) as f64);
        (0..self.lanes(e)).map(|k| (((o - n) / 2.0 + k as f64 + 0.5) * self.lane_width) as f32).collect()
    }
}
