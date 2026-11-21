//! Navigation: route guidance (`src/nav/guidance.js`) and the A* router (`src/sim/route.js makeRouter`).
//!
//! Guidance runs on the graph the GPU packer hands it (`g.nav || g`): a `loadWorld` graph has no `.nav`, so it is the
//! SimGraph itself. Its edges carry no `srcId`, `ref` or `name` (`fromRoadSet` leaves them `undefined` / `''`), which
//! fixes three guidance.js rules for this graph:
//! * `others.length === 0 && e1.srcId === e2.srcId` (undefined === undefined): a node with no alternative is never a maneuver;
//! * `continue` (needs `e2.ref || e2.name`) never fires;
//! * keep coalescing: only the same-side rule (same type within 400 m) applies, the opposite-side rule needs a road name.
//!
//! So the maneuver at a node depends on the (in-edge, out-edge) pair only, plus the same-side keep coalescing along
//! the route: [`ManeuverTable`] precomputes the code of every hop once and [`pack_route_words`] packs a route in O(n).
pub mod router;

use crate::graph::SimGraph;
use std::f64::consts::PI;

pub use router::{RouteQuery, Router, RouterOpts};

/// `MANEUVER_TYPES` order.
pub const MANEUVER_TYPES: [&str; 9] = ["continue", "exit", "merge", "turn-left", "turn-right", "keep-left", "keep-right", "arrive", "refuel"];
pub mod man {
    pub const CONTINUE: u8 = 0; pub const EXIT: u8 = 1; pub const MERGE: u8 = 2; pub const TURN_LEFT: u8 = 3; pub const TURN_RIGHT: u8 = 4;
    pub const KEEP_LEFT: u8 = 5; pub const KEEP_RIGHT: u8 = 6; pub const ARRIVE: u8 = 7; pub const REFUEL: u8 = 8;
}
/// side: 0 none, 1 left, 2 right (worldPack.js SIDE_*)
pub const SIDE_NONE: u8 = 0;
pub const SIDE_LEFT: u8 = 1;
pub const SIDE_RIGHT: u8 = 2;
/// worldPack.js EDGE_ID_MASK
pub const EDGE_ID_MASK: u32 = 0xFFFFF;

/// One guidance.js maneuver: `{type, side, edgePos, atDistance}` (text fields dropped).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Maneuver { pub ty: u8, pub side: u8, pub edge_pos: usize, pub dist: f64 }

const TAU: f64 = PI * 2.0;
#[inline]
fn wrap_angle(a: f64) -> f64 {
    let a = a % TAU;
    if a > PI { a - TAU } else if a < -PI { a + TAU } else { a }
}
/// turnSide: 1 left, 2 right, 0 straight
#[inline]
fn turn_side(d: f64) -> u8 { if d < 0.0 { SIDE_LEFT } else if d > 0.0 { SIDE_RIGHT } else { SIDE_NONE } }
#[inline] fn major(g: &SimGraph, e: usize) -> bool { let c = g.class[e]; c == 0 || c == 2 }
#[inline] fn major_or_link(g: &SimGraph, e: usize) -> bool { g.class[e] <= 3 }
/// nav graph CLASS_RANK.primary
const RANK_PRIMARY: u8 = 5;

/// guidance.js `_buildManeuvers` loop body for the hop e1 -> e2 (before coalescing): (type, side), or None.
pub fn transition(g: &SimGraph, e1: usize, e2: usize) -> Option<(u8, u8)> {
    let turn_threshold = 30.0 * PI / 180.0; let fork_cone = 45.0 * PI / 180.0;
    let node = g.to[e1] as usize; let from1 = g.from[e1];
    let delta = wrap_angle(g.heading_start[e2] - g.heading_end[e1]);
    let others = || g.out_of(node).iter().map(|&o| o as usize).filter(move |&o| o != e2 && g.to[o] != from1);
    if others().next().is_none() { return None; }
    let (l1, l2) = (g.is_link(e1), g.is_link(e2));
    if !l1 && l2 && major(g, e1) {
        let main = others().find(|&o| !g.is_link(o));
        let mut side = match main { Some(m) => turn_side(wrap_angle(g.heading_start[e2] - g.heading_start[m])), None => turn_side(delta) };
        if side == SIDE_NONE { side = SIDE_RIGHT; }
        return Some((man::EXIT, side));
    }
    if l1 && !l2 && (major(g, e2) || g.nav_rank[e2] >= RANK_PRIMARY) {
        let main_in = g.in_of(node).iter().map(|&o| o as usize).find(|&o| o != e1 && !g.is_link(o));
        let side = match main_in { Some(m) => if turn_side(wrap_angle(g.heading_end[e1] - g.heading_end[m])) == SIDE_LEFT { SIDE_RIGHT } else { SIDE_LEFT }, None => SIDE_RIGHT };
        return Some((man::MERGE, side));
    }
    if delta.abs() > turn_threshold && !(major_or_link(g, e1) && major_or_link(g, e2)) {
        let side = turn_side(delta);
        return Some((if side == SIDE_LEFT { man::TURN_LEFT } else { man::TURN_RIGHT }, side));
    }
    let r2 = g.nav_rank[e2];
    if let Some(f) = others().find(|&o| g.nav_rank[o] >= r2 && wrap_angle(g.heading_start[o] - g.heading_end[e1]).abs() < fork_cone) {
        let mut side = turn_side(wrap_angle(g.heading_start[e2] - g.heading_start[f]));
        if side == SIDE_NONE { side = if turn_side(delta) == SIDE_LEFT { SIDE_LEFT } else { SIDE_RIGHT }; }
        return Some((if side == SIDE_LEFT { man::KEEP_LEFT } else { man::KEEP_RIGHT }, side));
    }
    None
}

#[inline] fn is_keep(t: u8) -> bool { t == man::KEEP_LEFT || t == man::KEEP_RIGHT }

/// `new Guidance(graph, route).maneuvers` (no stops) as (type, side, edgePos, atDistance). None where the JS
/// constructor throws (an edge id out of range).
pub fn maneuvers(g: &SimGraph, route: &[u32]) -> Option<Vec<Maneuver>> {
    if route.iter().any(|&e| e as usize >= g.edge_count) { return None; }
    let n = route.len();
    let mut list: Vec<Maneuver> = Vec::new();
    let mut d = 0.0f64;
    for i in 0..n {
        d += g.length[route[i] as usize];
        if i + 1 >= n { break; }
        if let Some((ty, side)) = transition(g, route[i] as usize, route[i + 1] as usize) {
            list.push(Maneuver { ty, side, edge_pos: i + 1, dist: d });
        }
    }
    // coalesce same-side keeps within 400 m (the opposite-side rule needs road names: none on a SimGraph)
    let mut out: Vec<Maneuver> = Vec::with_capacity(list.len() + 1);
    for m in list {
        if let Some(a) = out.last() { if is_keep(a.ty) && m.ty == a.ty && m.dist - a.dist < 400.0 { continue; } }
        out.push(m);
    }
    out.push(Maneuver { ty: man::ARRIVE, side: SIDE_NONE, edge_pos: n.saturating_sub(1), dist: d });
    Some(out)
}

/// Code of every hop (e -> k-th out-edge of e.to): 0 none, else `1 | type << 1 | side << 5` (packRoutes `code`).
#[derive(Clone, Default)]
pub struct ManeuverTable {
    /// E + 1: first hop of edge e (= worldPack hop table base)
    pub hop_base: Vec<u32>,
    pub code: Vec<u8>,
}

#[inline] fn code_of(t: Option<(u8, u8)>) -> u8 { match t { Some((ty, side)) => 1 | (ty << 1) | (side << 5), None => 0 } }

impl ManeuverTable {
    pub fn new(g: &SimGraph) -> ManeuverTable {
        use rayon::prelude::*;
        let e_n = g.edge_count;
        let mut hop_base = vec![0u32; e_n + 1];
        for e in 0..e_n { hop_base[e + 1] = hop_base[e] + g.out_degree(g.to[e] as usize) as u32; }
        let mut code = vec![0u8; hop_base[e_n] as usize];
        let chunks: Vec<(usize, usize)> = (0..e_n).step_by(4096).map(|a| (a, (a + 4096).min(e_n))).collect();
        let parts: Vec<Vec<u8>> = chunks.par_iter().map(|&(a, b)| {
            let mut v = Vec::with_capacity((hop_base[b] - hop_base[a]) as usize);
            for e in a..b { for &en in g.out_of(g.to[e] as usize) { v.push(code_of(transition(g, e, en as usize))); } }
            v
        }).collect();
        let mut o = 0; for p in parts { code[o..o + p.len()].copy_from_slice(&p); o += p.len(); }
        ManeuverTable { hop_base, code }
    }
    /// Code of the hop a -> b (computed directly when b does not leave a.to).
    #[inline]
    pub fn get(&self, g: &SimGraph, a: usize, b: usize) -> u8 {
        let outs = g.out_of(g.to[a] as usize);
        for (k, &o) in outs.iter().enumerate() { if o as usize == b { return self.code[self.hop_base[a] as usize + k]; } }
        code_of(transition(g, a, b))
    }
}

/// worldPack.js packRoutes for one route: words `edge (20 bits) | present << 20 | type << 21 | side << 25 | nextType << 27`.
/// `out.len() == route.len()`. Ids are `i32` as in the JS Int32Array (negative / out-of-range ids: guidance throws,
/// the JS then stores only the arrive code).
pub fn pack_route_words(g: &SimGraph, t: &ManeuverTable, route: &[i32], out: &mut [u32]) {
    let n = route.len(); assert_eq!(out.len(), n);
    if n == 0 { return; }
    let e_n = g.edge_count;
    let valid = route.iter().all(|&e| e >= 0 && (e as usize) < e_n);
    // forward: code per entry (low 7 bits), with the same-side keep coalescing (400 m)
    if valid {
        let mut d = 0.0f64; let (mut last_ty, mut last_d) = (u8::MAX, 0.0f64);
        for i in 0..n - 1 {
            let a = route[i] as usize; d += g.length[a];
            let mut c = t.get(g, a, route[i + 1] as usize);
            if c != 0 {
                let ty = (c >> 1) & 15;
                if is_keep(ty) && ty == last_ty && d - last_d < 400.0 { c = 0; } else { last_ty = ty; last_d = d; }
            }
            out[i] = c as u32;
        }
    } else { for o in out[..n - 1].iter_mut() { *o = 0; } }
    out[n - 1] = 1 | ((man::ARRIVE as u32) << 1);
    let mut next = man::ARRIVE as u32;
    for k in (0..n).rev() {
        let c = out[k];
        if c & 1 != 0 { next = (c >> 1) & 15; }
        out[k] = (route[k] as u32 & EDGE_ID_MASK) | (c << 20) | (next << 27);
    }
}

/// `packRoutes(g, flat, offset)`: a route pool (routes delimited by `offset`) -> route words (`max(1, flat.len())`).
pub fn pack_routes(g: &SimGraph, t: &ManeuverTable, flat: &[i32], offset: &[u32]) -> Vec<u32> {
    let mut out = vec![0u32; flat.len().max(1)];
    for r in 0..offset.len().saturating_sub(1) {
        let (a, b) = (offset[r] as usize, offset[r + 1] as usize);
        if b <= a { continue; }
        pack_route_words(g, t, &flat[a..b], &mut out[a..b]);
    }
    out
}
