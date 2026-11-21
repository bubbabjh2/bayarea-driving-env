//! Same-seed determinism (rust/docs/determinism.md): two sims with the same options must give bit-identical GPU state,
//! host tables and event records. The 20,000-commuter population from 07:30 has spawn-lane contention from its
//! first steps (deterministic_spawn), events of many slots per step (deterministic_events) and contacts that share a
//! car within a few hundred steps (deterministic_contacts); with any of the three off this test fails
//! (`examples/determinism.rs` for the long runs). Skips without the world data or a GPU adapter.
use bayenv_gpu::sim::{GpuOptions, GpuPopulation, GpuSim};
use std::{path::Path, sync::Arc};

const N: usize = 20000;
const CALLS: usize = 40;
const STEPS: usize = 20;

fn fnv(h: &mut u64, v: u32) { for b in v.to_le_bytes() { *h ^= b as u64; *h = h.wrapping_mul(0x100000001b3); } }
fn state_hash(s: &GpuSim) -> u64 {
    let st = s.read_state(); let mut h = 0xcbf29ce484222325u64;
    for &w in st.veh.iter().chain(&st.par) { fnv(&mut h, w); }
    for &f in &st.pose { fnv(&mut h, f.to_bits()); }
    for &x in &s.slot_trip { fnv(&mut h, x as u32); }
    for &x in &s.slot_route { fnv(&mut h, x); }
    for &x in &s.route_mirror[..s.route_words] { fnv(&mut h, x); }
    let r = &s.rec;
    for i in 0..r.len() { for v in [r.step[i] as u32, r.kind[i] as u32, r.slot[i] as u32, r.agent[i] as u32, r.other[i] as u32, r.dv_kmh[i].to_bits(), r.progress_m[i].to_bits()] { fnv(&mut h, v); } }
    h
}

#[test]
fn same_seed_runs_are_bit_identical() {
    let data = Path::new("../../../data");
    let world = match bayenv_world::World::load(data) { Ok(w) => Arc::new(w), Err(e) => { eprintln!("skip: world: {e}"); return } };
    let o = GpuOptions { capacity: (N + 1023) / 1024 * 1024, start_time: 7.5 * 3600.0, ..Default::default() };
    assert!(o.deterministic_contacts && o.deterministic_spawn && o.deterministic_events && o.deterministic_leaders && o.refuel_budget_ms.is_infinite(), "the deterministic paths are the default");
    let pop = Arc::new(GpuPopulation::generate(&world, N, 42.0, true).unwrap());
    let make = || -> Option<GpuSim> {
        let mut s = match bayenv_gpu::env::create(world.clone(), o.clone()) { Ok(s) => s, Err(e) => { eprintln!("skip: {e}"); return None } };
        s.set_population(pop.clone(), true); Some(s)
    };
    let (Some(mut x), Some(mut y)) = (make(), make()) else { return };
    for c in 0..CALLS {
        x.run_steps(STEPS); y.run_steps(STEPS); x.sync(); y.sync();
        let (hx, hy) = (state_hash(&x), state_hash(&y));
        assert_eq!(hx, hy, "runs differ after step {} (call {})", x.step, c + 1);
    }
    assert!(x.rec.len() > 1000 && x.counters.minor_contacts + x.counters.crashes > 0, "records {}, contacts {} / crashes {}", x.rec.len(), x.counters.minor_contacts, x.counters.crashes);
}
