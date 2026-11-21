//! `src/geom/hops.js`: strict / junction hop classification and lane mapping across a node.
//! OWNER: geom port agent. Bit-exact with the JS (V8 sort order, `jsmath` trig).
use crate::jsmath;
use crate::lanegraph::{LaneGraph, LANE_W};
use std::f64::consts::PI;

const TURN_STRICT: f64 = 1.2;

/// `hopMap(G, e, en)`: strict = continuous carriageway (lane `l` of e continues as lane `l - shift` of en).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Hop { pub strict: bool, pub shift: i32 }

#[inline]
fn is_freeway_class(c: u8) -> bool { c <= 3 }

/// Heading (rad) of the first (end = false) or last (end = true) data segment of e.
pub fn heading(g: &LaneGraph, e: usize, end: bool) -> f64 {
    let (f, c) = (g.first[e] as usize, g.count[e] as usize);
    let (a, b) = if end { (g.poly[f + c - 2], g.poly[f + c - 1]) } else { (g.poly[f], g.poly[f + 1]) };
    let (a, b) = (a as usize, b as usize);
    jsmath::atan2(g.z(b) - g.z(a), g.x(b) - g.x(a))
}

/// `turnAngle(G, e, en)`: > 0 = right turn (x east, z south).
pub fn turn_angle(g: &LaneGraph, e: usize, en: usize) -> f64 {
    let mut d = heading(g, en, false) - heading(g, e, true);
    while d > PI { d -= 2.0 * PI; }
    while d < -PI { d += 2.0 * PI; }
    d
}

/// `pointInto(G, e, dist, fromEnd)`: point `dist` m into the edge from its start, or back from its end.
pub fn point_into(g: &LaneGraph, e: usize, dist: f64, from_end: bool) -> [f64; 2] {
    let (f, c) = (g.first[e] as usize, g.count[e] as usize);
    let mut acc = 0.0;
    for k in 0..c.saturating_sub(1) {
        let (a, b) = if from_end { (g.poly[f + c - 1 - k], g.poly[f + c - 2 - k]) } else { (g.poly[f + k], g.poly[f + k + 1]) };
        let (a, b) = (a as usize, b as usize);
        let l = jsmath::hypot(g.x(b) - g.x(a), g.z(b) - g.z(a));
        if acc + l >= dist {
            let t = if l > 0.0 { (dist - acc) / l } else { 0.0 };
            return [g.x(a) + (g.x(b) - g.x(a)) * t, g.z(a) + (g.z(b) - g.z(a)) * t];
        }
        acc += l;
    }
    let n = g.poly[if from_end { f } else { f + c - 1 }] as usize;
    [g.x(n), g.z(n)]
}

fn side(g: &LaneGraph, h: f64, node: usize, p: [f64; 2]) -> f64 {
    let (dx, dz) = (jsmath::cos(h), jsmath::sin(h));
    (p[0] - g.x(node)) * -dz + (p[1] - g.z(node)) * dx
}

/// `hopMap(G, e, en)` (uncached; see [`HopTable`] for the builders).
pub fn hop_map(g: &LaneGraph, e: usize, en: usize) -> Hop {
    let node = g.to(e);
    let mut outs: Vec<u32> = g.out_edges(node).iter().copied().filter(|&o| o as i32 != g.twin[e]).collect();
    let mut ins: Vec<u32> = g.in_edges(g.from(en)).iter().copied().filter(|&i| i as i32 != g.twin[en]).collect();
    let simple = outs.len() <= 1 && ins.len() <= 1;
    let strict = simple || (is_freeway_class(g.cls[e]) && is_freeway_class(g.cls[en])
        && (outs.len() <= 1 || ins.len() <= 1 || turn_angle(g, e, en).abs() < TURN_STRICT));
    let (le, len) = (g.lanes[e] as i32, g.lanes[en] as i32);
    let mut shift = 0;
    if strict && simple { shift = le - len; }
    else if strict && g.opp[e] > 0 && g.opp[en] > 0 { shift = le - len; }
    else if strict {
        let (mut div, mut mer) = (0i32, 0i32);
        if outs.len() > 1 {
            let he = heading(g, e, true);
            let sd: Vec<(u32, f64)> = outs.iter().map(|&o| (o, side(g, he, node, point_into(g, o as usize, 50.0, false)))).collect();
            let sdo = |o: u32| sd.iter().find(|p| p.0 == o).unwrap().1;
            v8_sort(&mut outs, |&a, &b| {
                let (sa, sb) = (sdo(a), sdo(b));
                if (sa - sb).abs() >= 1.0 { sb - sa } else { turn_angle(g, e, b as usize) - turn_angle(g, e, a as usize) }
            });
            for &o in &outs { if o as usize == en { break; } div += g.lanes[o as usize] as i32; }
            div = div.min(le - len).max(0);
        }
        if ins.len() > 1 {
            let hn = heading(g, en, false);
            let sd: Vec<(u32, f64)> = ins.iter().map(|&i| (i, side(g, hn, node, point_into(g, i as usize, 50.0, true)))).collect();
            let sdo = |o: u32| sd.iter().find(|p| p.0 == o).unwrap().1;
            v8_sort(&mut ins, |&a, &b| {
                let (sa, sb) = (sdo(a), sdo(b));
                if (sa - sb).abs() >= 1.0 { sb - sa } else { turn_angle(g, b as usize, en) - turn_angle(g, a as usize, en) }
            });
            for &i in &ins { if i as usize == e { break; } mer += g.lanes[i as usize] as i32; }
            mer = mer.min(len - le).max(0);
        }
        shift = div - mer;
    }
    Hop { strict, shift }
}

/// `mapLane(G, e, en, l)`: lane of en that lane l of e continues as (may be out of range at a strict lane end).
pub fn map_lane(g: &LaneGraph, e: usize, en: usize, l: i32) -> i32 {
    let h = hop_map(g, e, en);
    if h.strict { l - h.shift } else {
        let scale = g.lanes[en] as f64 / g.lanes[e] as f64;
        jsmath::min(g.lanes[en] as f64 - 1.0, ((l as f64 + 0.5) * scale).floor()) as i32
    }
}

/// `mapLatStrict(G, e, en, lat)`: lat on en for a strict hop, None for junction hops.
pub fn map_lat_strict(g: &LaneGraph, e: usize, en: usize, lat: f64) -> Option<f64> {
    let h = hop_map(g, e, en);
    if h.strict { Some(lat - h.shift as f64 * LANE_W) } else { None }
}

/// Every hop (in-edge a -> out-edge b at `to(a)`), precomputed in parallel: `get(a, b)` = `hop_map(g, a, b)`.
pub struct HopTable { start: Vec<u32>, hops: Vec<Hop> }
impl HopTable {
    pub fn new(g: &LaneGraph) -> HopTable {
        use rayon::prelude::*;
        let e_n = g.edge_count;
        let mut start = vec![0u32; e_n + 1];
        for a in 0..e_n { start[a + 1] = start[a] + g.out_edges(g.to(a)).len() as u32; }
        let per: Vec<Vec<Hop>> = (0..e_n).into_par_iter().map(|a| g.out_edges(g.to(a)).iter().map(|&b| hop_map(g, a, b as usize)).collect()).collect();
        HopTable { start, hops: per.into_iter().flatten().collect() }
    }
    /// b must be an out-edge of `to(a)`.
    #[inline]
    pub fn get(&self, g: &LaneGraph, a: usize, b: usize) -> Hop {
        let outs = g.out_edges(g.to(a));
        let k = outs.iter().position(|&o| o as usize == b).expect("HopTable::get: b is not an out-edge of to(a)");
        self.hops[self.start[a] as usize + k]
    }
}

/// V8 `Array.prototype.sort` (TimSort, third_party/v8/builtins/array-sort.tq) for arrays shorter than 64
/// (one forced run: CountAndMakeRun + BinaryInsertionSort). The comparator returns the JS number; the `< 0`
/// and `>= 0` tests follow V8 exactly (NaN is neither), so inconsistent comparators give the JS order.
pub fn v8_sort<T: Copy>(v: &mut [T], mut cmp: impl FnMut(&T, &T) -> f64) {
    let n = v.len();
    if n < 2 { return; }
    assert!(n < 64, "v8_sort: only the single-run (n < 64) path is implemented, got {n}");
    // CountAndMakeRun(0, n)
    let mut run = 2usize;
    let desc = cmp(&v[1], &v[0]) < 0.0;
    let mut prev = v[1];
    for idx in 2..n {
        let cur = v[idx];
        let o = cmp(&cur, &prev);
        if desc { if o >= 0.0 { break; } } else if o < 0.0 { break; }
        prev = cur; run += 1;
    }
    if desc { v[..run].reverse(); }
    // BinaryInsertionSort(0, run, n)
    for start in run..n {
        let (mut left, mut right) = (0usize, start);
        let pivot = v[start];
        while left < right {
            let mid = left + ((right - left) >> 1);
            if cmp(&pivot, &v[mid]) < 0.0 { right = mid; } else { left = mid + 1; }
        }
        let mut p = start;
        while p > left { v[p] = v[p - 1]; p -= 1; }
        v[left] = pivot;
    }
}

#[cfg(test)]
mod tests {
    use super::v8_sort;
    #[test]
    fn sort_stable_numeric() {
        let mut v = vec![(3, 0), (1, 1), (3, 2), (2, 3), (1, 4)];
        v8_sort(&mut v, |a, b| (a.0 - b.0) as f64);
        assert_eq!(v, vec![(1, 1), (1, 4), (2, 3), (3, 0), (3, 2)]);
        let mut d = vec![5, 4, 3, 2, 1, 0];
        v8_sort(&mut d, |a, b| (*a - *b) as f64);
        assert_eq!(d, vec![0, 1, 2, 3, 4, 5]);
    }
}
