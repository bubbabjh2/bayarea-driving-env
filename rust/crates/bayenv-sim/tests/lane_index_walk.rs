//! ENV-REQUESTS #15 regression: a lane-index walk after integrate on a car that crossed onto another edge.
//! State from the dumped hang (bayarea-driving-sim cpu_vec 28 2000 600 --fast-learn, env 17, step 460): slot 361
//! indexed on edge 47450 lanes 1..2 (entries 318, 319), then on edge 20163 (never indexed: its keys' range start is
//! 0) with the same lanes. The JS `behind(361, 2)` reads key (20163, 2) and entPos[361][1] - 1 = 318 >= 0, so it
//! returns order[318] = 361 itself, and `while conn[j] >= 0: j = behind(j)` (sim._yieldJ) never ends.
use bayenv_sim::spatial::LaneIndex;
use bayenv_sim::vehicle::{VehicleStore, STATE_DRIVING};

fn store() -> VehicleStore {
    let mut vs = VehicleStore::new(8);
    vs.high = 2;
    // slot 0: the car behind on the same edge / lane 1 (keeps a real entry before slot 1's lane-1 entry)
    for (i, (e, lo, hi, s)) in [(10, 1i8, 1i8, 5.0f32), (10, 1, 2, 9.0)].into_iter().enumerate() {
        vs.state[i] = STATE_DRIVING; vs.edge[i] = e; vs.lane_lo[i] = lo; vs.lane_hi[i] = hi; vs.lane[i] = hi; vs.s[i] = s;
    }
    vs
}

#[test]
fn behind_after_an_edge_change_js_vs_snapshot() {
    let mut vs = store();
    let mut li = LaneIndex::new(64, 8);
    li.rebuild(&vs);
    assert_eq!(li.behind(&vs, 1, 1), 0, "consistent index: slot 0 is behind slot 1 in lane 1");
    assert_eq!(li.behind(&vs, 1, 2), -1, "nobody behind slot 1 in lane 2");
    // slot 1 crosses onto edge 20 (never indexed) with the same lanes, before the index is rebuilt
    vs.edge[1] = 20;
    // the JS (snapshot off): key (20, 2) has range start 0, entPos[1][1] - 1 is slot 1's own lane-1 entry -> itself
    li.snapshot = false;
    assert_eq!(li.behind(&vs, 1, 2), 1, "JS lookup returns the car itself: a walk `j = behind(j)` never ends");
    // the fix: walks read the slot as indexed (edge 10), so positions move strictly within one range
    li.snapshot = true;
    assert_eq!(li.behind(&vs, 1, 2), -1);
    assert_eq!(li.behind(&vs, 1, 1), 0);
    let mut j = 1i32; let mut steps = 0;
    while j >= 0 { j = li.behind(&vs, j as usize, 1); steps += 1; assert!(steps < 10, "snapshot walk must end"); }
}
