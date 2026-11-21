//! Helpers for comparing with the JS oracle dumps (`rust/tools/oracle/gpu_mock.mjs`).
use crate::sim::{LayoutOff, WorldBlob};
use std::path::Path;

pub struct OracleDump { pub meta: serde_json::Value, pub bin: Vec<u8> }
impl OracleDump {
    pub fn open(dir: &Path) -> Result<OracleDump, String> {
        let meta = serde_json::from_str(&std::fs::read_to_string(dir.join("gpu.meta.json")).map_err(|e| format!("{}: {e}", dir.display()))?).map_err(|e| e.to_string())?;
        let bin = std::fs::read(dir.join("gpu.bin")).map_err(|e| e.to_string())?;
        Ok(OracleDump { meta, bin })
    }
    /// Bytes of a buffer after the JS setup (None: all zero / absent).
    pub fn buffer(&self, name: &str) -> Option<&[u8]> {
        let b = self.meta["buffers"].as_array()?.iter().find(|b| b["name"] == name)?;
        let off = b["offset"].as_i64()?; if off < 0 { return None; }
        Some(&self.bin[off as usize..off as usize + b["size"].as_u64()? as usize])
    }
    pub fn layout_off(&self) -> LayoutOff {
        let l = &self.meta["layoutOff"]; let g = |k: &str| l[k].as_u64().unwrap_or(0) as u32;
        LayoutOff { pts: g("ptsOff"), lane_off: g("laneOffOff"), adj: g("adjOff"), eb: g("ebOff"), blk: g("blkOff"), geo: g("geoOff"), lgv: g("lgvOff"), hop: g("hopOff"),
            ctl_edge: g("ctlEdgeOff"), ctl_ap: g("ctlApOff"), ctl_c: g("ctlCOff"), pair: g("pairOff"), j_mouth: g("jMouthOff"), j_mov: g("jMovOff"), j_edge_mov: g("jEdgeMovOff"),
            j_pts: g("jPtsOff"), j_conf: g("jConfOff"), j_into: g("jIntoOff") }
    }
    /// The world blob of the JS run.
    pub fn world_blob(&self) -> WorldBlob {
        let w = &self.meta["world"]; let u = |k: &str| w[k].as_u64().unwrap() as usize;
        let bytes = self.buffer("world").expect("world buffer");
        let e = u("E");
        WorldBlob { words: bytemuck::cast_slice(bytes).to_vec(), off: self.layout_off(), e, n: u("N"), p: u("P"), k: u("K"), has_lg: true,
            pair_n: self.meta["pairN"].as_u64().unwrap_or(0) as usize, j_m: self.meta["jM"].as_u64().filter(|&m| m > 0).map(|m| m as usize),
            ctl: Some(((e + 1) as u32, 6.0 * 3600.0, 10.0 * 3600.0)) }
    }
    /// Dispatch lists of the recorded runSteps passes: Vec per pass of (kernel label, workgroups).
    pub fn passes(&self) -> Vec<Vec<(String, u32)>> {
        self.meta["commands"].as_array().unwrap().iter().filter(|c| c["op"] == "pass")
            .map(|c| c["d"].as_array().unwrap().iter().map(|d| (d[0].as_str().unwrap().to_string(), d[2].as_u64().unwrap() as u32)).collect()).collect()
    }
}
