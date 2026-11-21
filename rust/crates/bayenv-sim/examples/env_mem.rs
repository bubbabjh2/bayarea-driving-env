//! Live heap bytes of one env (beyond the shared world / population): env_mem <trace dir> [steps]
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicIsize, Ordering};
struct Count;
static LIVE: AtomicIsize = AtomicIsize::new(0);
unsafe impl GlobalAlloc for Count {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 { LIVE.fetch_add(l.size() as isize, Ordering::Relaxed); unsafe { System.alloc(l) } }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) { LIVE.fetch_sub(l.size() as isize, Ordering::Relaxed); unsafe { System.dealloc(p, l) } }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 { LIVE.fetch_add(n as isize - l.size() as isize, Ordering::Relaxed); unsafe { System.realloc(p, l, n) } }
}
#[global_allocator] static A: Count = Count;
fn mb(x: isize) -> f64 { x as f64 / 1048576.0 }
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let steps: usize = a.get(2).map(|s| s.parse().unwrap()).unwrap_or(500);
    let l0 = LIVE.load(Ordering::Relaxed);
    let (e0, run) = bayenv_sim::harness::env_from_dir(&a[1]);
    let l1 = LIVE.load(Ordering::Relaxed);
    let mut e = bayenv_sim::harness::env_from_parts(&run, e0.world.clone(), e0.pop.clone());
    let l2 = LIVE.load(Ordering::Relaxed);
    let mut p = bayenv_sim::harness::Policy::new(run["policy"].as_str().unwrap(), 1.0);
    for _ in 0..steps { p.act(&mut e); e.step(1); }
    let l3 = LIVE.load(Ordering::Relaxed);
    let w = e0.world.clone();
    let m = |f: &dyn Fn() -> Box<dyn std::any::Any>| { let a = LIVE.load(Ordering::Relaxed); let x = f(); let b = LIVE.load(Ordering::Relaxed); drop(x); mb(b - a) };
    println!("  LaneIndex {:.1} MB, SpatialHash {:.1} MB, StationNetwork {:.1} MB, VehicleStore(cap) {:.1} MB, RouteStore {:.1} MB",
        m(&|| Box::new(bayenv_sim::spatial::LaneIndex::new(w.edge_count, e.sim.vs.capacity))),
        m(&|| Box::new(bayenv_sim::spatial::SpatialHash::new(25.0, 18, e.sim.vs.capacity))),
        m(&|| Box::new(bayenv_sim::fuel::StationNetwork::new(&w, Default::default()))),
        m(&|| Box::new(bayenv_sim::vehicle::VehicleStore::new(e.sim.vs.capacity))),
        m(&|| Box::new(bayenv_sim::fuel::RouteStore::new(&e0.pop))));
    println!("  router fork {:.1} MB", m(&|| Box::new(w.nav_base.get().unwrap().fork())));
    let s = &e.sim;
    let lanes = (s.lanes.lstart.capacity() + s.lanes.lend.capacity()) * 4 * 3 / 2 + s.g.edge_count * 4 + s.lanes.order.capacity() * 4 + s.lanes.ent_pos.capacity() * 4;
    println!("world+pop+env0 {:.1} MB; one more env {:.1} MB at creation, {:.1} MB after {steps} steps (lane index ~{:.1} MB, vs capacity {})",
        mb(l1 - l0), mb(l2 - l1), mb(l3 - l1), lanes as f64 / 1048576.0, s.vs.capacity);
}
