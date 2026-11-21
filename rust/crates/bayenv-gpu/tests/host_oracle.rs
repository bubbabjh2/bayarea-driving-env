//! Native host vs the JS GPUSim mock run (`node rust/tools/oracle/gpu_mock.mjs data rust/oracle/gpu10k --n=10000 --seed=11`):
//! kernel sources, params, scatter state (veh / par / pose), route words and the dispatch lists must be identical.
//! Skips when the dump or a GPU adapter is missing. The JS-vs-Rust run in Chrome is rust/tools/oracle/compare_gpu.sh.
use bayenv_gpu::{defines::ObsMode, kernels, oracle::OracleDump, sim::GpuOptions};
use std::{path::Path, sync::Arc};

#[test]
fn host_matches_js_setup() {
    let (data, odir) = (Path::new("../../../data"), Path::new("../../oracle/gpu10k"));
    let Ok(o) = OracleDump::open(odir) else { bayenv_world::oracle::skip("host_oracle", &format!("no oracle dump at {}", odir.display())); return };
    for s in kernels::specs(true, false) {
        let f = odir.join("kernels").join(format!("{}.wgsl", s.label));
        if let Ok(js) = std::fs::read_to_string(&f) { assert_eq!(js, kernels::source(&s, ObsMode::V2, o.meta["cap"].as_u64().unwrap() as usize), "kernel source {}", s.label); }
    }
    let world = Arc::new(bayenv_world::World::load(data).expect("world"));
    let cap = o.meta["cap"].as_u64().unwrap() as usize;
    let mut opts = GpuOptions { capacity: cap, cpu_connector_join: false, contact_tangent: false, ..Default::default() };   // the JS kernels (legacy model)
    opts.features.accidents = true; opts.start_time = o.meta["time"].as_f64().unwrap() - o.meta["k"].as_f64().unwrap() * opts.dt;
    let mut sim = match bayenv_gpu::env::create(world, opts) { Ok(s) => s, Err(e) => { eprintln!("host_oracle: no GPU ({e}): not run"); return } };
    let jp: &[u32] = bytemuck::cast_slice(o.buffer("params").unwrap());
    assert_eq!(jp, &sim.params_words()[..], "params");
    assert_eq!(bytemuck::cast_slice::<u8, u32>(o.buffer("world").unwrap()), &sim.read_words("world")[..], "world blob");
    let mut acts = vec![0f32; cap * 3]; for i in 0..cap { acts[3 * i + 1] = o.meta["gas"].as_f64().unwrap() as f32; } sim.set_actions(&acts, 0);
    bayenv_gpu::env::scatter(&mut sim, o.meta["n"].as_u64().unwrap() as usize, o.meta["seed"].as_u64().unwrap() as u32);
    let st = sim.read_state();
    assert_eq!(bytemuck::cast_slice::<u8, u32>(o.buffer("veh").unwrap()), &st.veh[..], "veh");
    assert_eq!(bytemuck::cast_slice::<u8, u32>(o.buffer("par").unwrap()), &st.par[..], "par");
    assert_eq!(bytemuck::cast_slice::<u8, f32>(o.buffer("pose").unwrap()), &st.pose[..], "pose");
    let used = o.meta["scatter"]["routeWords"].as_u64().unwrap() as usize;
    assert_eq!(&bytemuck::cast_slice::<u8, u32>(o.buffer("routes").unwrap())[..used], &sim.route_mirror[..used], "route words");
    let js = o.passes();
    for (k, first) in [(1usize, true), (2, false)] {
        let rs: Vec<(String, u32)> = sim.peek_stages(first, 0).into_iter().map(|(a, b)| (a.to_string(), b)).collect();
        assert_eq!(js[k], rs, "dispatch list of step {k}");
    }
}
