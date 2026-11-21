//! Route guidance (`src/nav/guidance.js`): maneuvers along a route, used by the GPU route words
//! (`worldPack.js packRoutes`) and by the CPU observation. OWNER: world agent (lead).
use crate::graph::SimGraph;

/// `MANEUVER_TYPES` order.
pub const MANEUVER_TYPES: [&str; 9] = ["continue", "exit", "merge", "turn-left", "turn-right", "keep-left", "keep-right", "arrive", "refuel"];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Maneuver { pub ty: u8, pub side: u8, pub edge_pos: usize, pub dist: f64 }

/// `new Guidance(graph, route).maneuvers`.
pub fn maneuvers(_g: &SimGraph, _route: &[u32]) -> Option<Vec<Maneuver>> { unimplemented!("guidance.js port in progress") }
