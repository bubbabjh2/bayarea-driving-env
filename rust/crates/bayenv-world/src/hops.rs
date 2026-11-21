//! `src/geom/hops.js`: strict / junction hop classification and lane mapping. OWNER: geom port agent.
use crate::lanegraph::LaneGraph;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Hop { pub strict: bool, pub shift: i32 }

/// `hopMap(G, e, en)`.
pub fn hop_map(_g: &LaneGraph, _e: usize, _en: usize) -> Hop { unimplemented!("hops.js port in progress") }
