//! `buildJunctions` / `crosswalkLegs` port (src/geom/junctions.js). OWNER: geom port agent. Output must equal
//! `npm run build:junctions` (data/junctions.bin) byte for byte.
use crate::junctions::{CrosswalkLegs, JunctionArrays};
use crate::lanegraph::LaneGraph;
use crate::lanes::LaneGeometry;

pub fn build_junctions(_lg: &LaneGeometry, _g: &LaneGraph, _legs: Option<&CrosswalkLegs>) -> JunctionArrays {
    unimplemented!("buildJunctions port in progress")
}
