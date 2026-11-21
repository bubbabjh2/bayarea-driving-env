//! Small world layers: surfaces, grade separations (crossings.json), gas stations, the 20 m DEM grid.
use crate::blob::{Pack, Slice};
use crate::graph::SimGraph;
use crate::jsmath;
use std::{collections::HashSet, io, io::Read, path::Path};

/// `data/surfaces.bin`: 2 bytes per edge (pavement code, ground right | ground left << 4). Attaches to the graph.
pub fn attach_surfaces(dir: &Path, g: &mut SimGraph, main_edges: usize, local_edges: Option<usize>) -> io::Result<bool> {
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("surfaces.meta.json"))?)?;
    let ec = &meta["edgeCount"];
    if ec["main"].as_u64() != Some(main_edges as u64) { return Ok(false); }
    if let Some(l) = local_edges { if ec["local"].as_u64() != Some(l as u64) { return Ok(false); } }
    let bytes = std::fs::read(dir.join(meta["file"].as_str().unwrap_or("surfaces.bin")))?;
    let e_n = g.edge_count;
    if bytes.len() < 2 * e_n { return Ok(false); }
    for i in 0..e_n { g.surf[i] = bytes[2 * i] & 15; g.gnd_r[i] = bytes[2 * i + 1] & 15; g.gnd_l[i] = bytes[2 * i + 1] >> 4; }
    Ok(true)
}

/// Edge pairs allowed to touch across a grade separation (data.js: crossings with dy < 2.5 m + pavement overlaps).
#[derive(Clone, Default)]
pub struct ContactPairs {
    /// key `min * 4194304 + max` (as `graph.contactPairs`)
    pub set: HashSet<u64>,
    /// sorted (lo, hi) pairs (GPU upload order)
    pub sorted: Vec<(u32, u32)>,
    pub crossings_loaded: usize,
}
impl ContactPairs {
    #[inline] pub fn key(a: u32, b: u32) -> u64 { if a < b { a as u64 * 4194304 + b as u64 } else { b as u64 * 4194304 + a as u64 } }
    pub fn contains(&self, a: u32, b: u32) -> bool { self.set.contains(&Self::key(a, b)) }
    pub fn load(dir: &Path, main_edges: usize) -> io::Result<Option<ContactPairs>> {
        let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("crossings.json"))?)?;
        if v["meta"]["mainEdgeCount"].as_u64() != Some(main_edges as u64) { return Ok(None); }
        let mut set = HashSet::new();
        let id = |x: &serde_json::Value| x.as_i64().unwrap_or(-1);
        let cr = v["crossings"].as_array().cloned().unwrap_or_default();
        for c in &cr {
            if !(c["dy"].as_f64().unwrap_or(f64::NAN) < 2.5) { continue; }
            for a in [id(&c["a"]), id(&c["aTwin"])] { for b in [id(&c["b"]), id(&c["bTwin"])] {
                if a >= 0 && b >= 0 { set.insert(Self::key(a as u32, b as u32)); } } }
        }
        for o in v["overlaps"].as_array().cloned().unwrap_or_default() {
            let (a, b) = (id(&o["a"]), id(&o["b"])); if a >= 0 && b >= 0 { set.insert(Self::key(a as u32, b as u32)); }
        }
        let mut sorted: Vec<(u32, u32)> = set.iter().map(|&k| ((k / 4194304) as u32, (k % 4194304) as u32)).collect();
        sorted.sort();
        Ok(Some(ContactPairs { set, sorted, crossings_loaded: cr.len() }))
    }
}

#[derive(Clone, Debug)]
pub struct Station { pub id: usize, pub node: usize, pub x: f64, pub y: f64, pub z: f64, pub pumps: u32, pub near_interchange: bool, pub tier: u8 }

/// `stationsOnGraph(loadStationPack())`.
pub fn load_stations(dir: &Path, g: &SimGraph, grid: &NodeGrid) -> io::Result<Vec<Station>> {
    let p = Pack::open(dir, "stations.meta.json")?;
    let xyz: Slice<f32> = p.req("stationXYZ")?; let info: Slice<u32> = p.req("stationInfo")?;
    let n = p.meta["count"].as_u64().unwrap_or(0) as usize; let mut out = Vec::new();
    for i in 0..n {
        let (x, y, z) = (xyz[i * 3] as f64, xyz[i * 3 + 1] as f64, xyz[i * 3 + 2] as f64);
        let (tn, tier, pumps, flags) = (info[i * 4] as usize, info[i * 4 + 1] as usize, info[i * 4 + 2], info[i * 4 + 3]);
        let mut node: i64 = g.tier_node_map.get(tier).and_then(|m| m.get(tn)).map(|&v| v as i64).unwrap_or(-1);
        if !(node >= 0) || jsmath::hypot(g.nx[node as usize] - x, g.nz[node as usize] - z) > 5.0 {
            node = grid.nearest(g, x, z, |k| g.out_degree(k) > 0 && g.in_degree(k) > 0);
        }
        if node < 0 { continue; }
        out.push(Station { id: i, node: node as usize, x, y, z, pumps: pumps.max(1), near_interchange: flags & 1 != 0, tier: tier as u8 });
    }
    Ok(out)
}

/// `makeNearestNode`: 250 m bucket grid over the graph nodes.
#[derive(Clone, Default)]
pub struct NodeGrid { cell: f64, cols: i64, rows: i64, min_x: f64, min_z: f64, start: Vec<u32>, items: Vec<u32> }
impl NodeGrid {
    pub fn new(g: &SimGraph) -> NodeGrid {
        let b = g.bounds; let cell = 250.0;
        let cols = (((b.max_x - b.min_x) / cell).ceil() as i64 + 1).max(1); let rows = (((b.max_z - b.min_z) / cell).ceil() as i64 + 1).max(1);
        let mut s = NodeGrid { cell, cols, rows, min_x: b.min_x, min_z: b.min_z, start: vec![0; (cols * rows + 1) as usize], items: vec![0; g.node_count] };
        let cells: Vec<usize> = (0..g.node_count).map(|i| s.cell_of(g.nx[i], g.nz[i])).collect();
        for &c in &cells { s.start[c + 1] += 1; }
        for i in 0..(cols * rows) as usize { s.start[i + 1] += s.start[i]; }
        let mut fill = s.start[..(cols * rows) as usize].to_vec();
        for (i, &c) in cells.iter().enumerate() { s.items[fill[c] as usize] = i as u32; fill[c] += 1; }
        s
    }
    fn cell_of(&self, x: f64, z: f64) -> usize {
        let cx = ((x - self.min_x) / self.cell).floor().max(0.0).min((self.cols - 1) as f64) as i64;
        let cz = ((z - self.min_z) / self.cell).floor().max(0.0).min((self.rows - 1) as f64) as i64;
        (cz * self.cols + cx) as usize
    }
    /// `graph.nearestNode(x, z, filter)`, -1 none.
    pub fn nearest(&self, g: &SimGraph, x: f64, z: f64, filter: impl Fn(usize) -> bool) -> i64 {
        if g.node_count == 0 { return -1; }
        let cx0 = ((x - self.min_x) / self.cell).floor() as i64; let cz0 = ((z - self.min_z) / self.cell).floor() as i64;
        let (mut best, mut best_d) = (-1i64, f64::INFINITY);
        let max_r = self.cols.max(self.rows) + 1;
        for r in 0..=max_r {
            if best >= 0 && best_d.sqrt() <= (r - 1) as f64 * self.cell { break; }
            for cz in cz0 - r..=cz0 + r { if cz < 0 || cz >= self.rows { continue; }
                for cx in cx0 - r..=cx0 + r { if cx < 0 || cx >= self.cols { continue; }
                    if (cx - cx0).abs() != r && (cz - cz0).abs() != r { continue; }
                    let c = (cz * self.cols + cx) as usize;
                    for &k in &self.items[self.start[c] as usize..self.start[c + 1] as usize] {
                        let k = k as usize; if !filter(k) { continue; }
                        let d = (g.nx[k] - x).powi(2) + (g.nz[k] - z).powi(2);
                        if d < best_d { best_d = d; best = k as i64; }
                    } } }
        }
        best
    }
}

/// Committed 20 m ground grid (`data/dem/compact/ground20.*`, `tools/elevation/compact.mjs`).
#[derive(Clone, Default)]
pub struct Dem { pub x0: f64, pub z0: f64, pub cell: f64, pub n: usize, pub heights: Vec<f32>, pub mask: Vec<u8> }
impl Dem {
    pub fn load(dir: &Path) -> io::Result<Dem> {
        let d = dir.join("dem/compact");
        let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(d.join("ground20.json"))?)?;
        let n = meta["n"].as_u64().unwrap_or(0) as usize;
        let gunzip = |f: &str| -> io::Result<Vec<u8>> { let mut v = Vec::new(); flate2::read::GzDecoder::new(std::fs::File::open(d.join(f))?).read_to_end(&mut v)?; Ok(v) };
        let raw = gunzip("ground20.dh16.gz")?; let dh: &[i16] = bytemuck::cast_slice(&raw[..n * n * 2]);
        let mut h = vec![0f32; n * n];
        for j in 0..n { let mut v = 0f64; for i in 0..n { v += dh[j * n + i] as f64; h[j * n + i] = (v / 10.0) as f32; } }
        Ok(Dem { x0: meta["root"]["x0"].as_f64().unwrap_or(0.0), z0: meta["root"]["z0"].as_f64().unwrap_or(0.0), cell: meta["cell"].as_f64().unwrap_or(20.0), n, heights: h, mask: gunzip("ground20.mask.gz")? })
    }
    /// Bilinear height (m NAVD88), NaN outside the outer cell centres.
    pub fn height_at(&self, x: f64, z: f64) -> f64 {
        let n = self.n; let c = (x - self.x0) / self.cell - 0.5; let r = (z - self.z0) / self.cell - 0.5;
        if !(c >= 0.0 && r >= 0.0 && c <= (n - 1) as f64 && r <= (n - 1) as f64) { return f64::NAN; }
        let c0 = jsmath::min((n - 2) as f64, c.floor()); let r0 = jsmath::min((n - 2) as f64, r.floor());
        let (fc, fr) = (c - c0, r - r0); let k = r0 as usize * n + c0 as usize; let hh = |i: usize| self.heights[i] as f64;
        (hh(k) * (1.0 - fc) + hh(k + 1) * fc) * (1.0 - fr) + (hh(k + n) * (1.0 - fc) + hh(k + n + 1) * fc) * fr
    }
    pub fn mask_at(&self, x: f64, z: f64) -> u8 {
        let i = ((x - self.x0) / self.cell).floor(); let j = ((z - self.z0) / self.cell).floor();
        if i >= 0.0 && j >= 0.0 && (i as usize) < self.n && (j as usize) < self.n { self.mask[j as usize * self.n + i as usize] } else { 0 }
    }
}
