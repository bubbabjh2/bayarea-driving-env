//! Compile every sim kernel through naga/wgpu and compare the generated sources with a JS oracle dump.
use bayenv_gpu::{defines::ObsMode, gpu::Gpu, kernels};
fn main() {
    let gpu = Gpu::new().expect("gpu");
    println!("adapter: {}  maxStorageBufferBindingSize {} maxBufferSize {} storage/stage {}", gpu.adapter_name(), gpu.limits.max_storage_buffer_binding_size, gpu.limits.max_buffer_size, gpu.limits.max_storage_buffers_per_shader_stage);
    let oracle = std::env::args().nth(1);
    for obs in [ObsMode::V2, ObsMode::V3, ObsMode::None] {
        let (mut ok, mut bad) = (0, 0);
        for s in kernels::specs(true, obs == ObsMode::V3) {
            let code = kernels::source(&s, obs, 131072);
            if obs == ObsMode::V2 { if let Some(o) = &oracle {
                let f = format!("{o}/kernels/{}.wgsl", s.label.replace(|c: char| !c.is_alphanumeric() && c != '.' && c != '_', "_"));
                match std::fs::read_to_string(&f) { Ok(js) => if js != code { println!("SOURCE DIFF {}", s.label); let (a, b): (Vec<_>, Vec<_>) = (js.lines().collect(), code.lines().collect());
                    for i in 0..a.len().max(b.len()) { if a.get(i) != b.get(i) { println!("  line {i}:\n   js: {:?}\n   rs: {:?}", a.get(i), b.get(i)); break; } } }, Err(_) => println!("(no oracle for {})", s.label) }
            } }
            let t = std::time::Instant::now();
            match gpu.kernel(s.label, &code, s.entry) { Ok(_) => { ok += 1; if obs == ObsMode::V2 { println!("ok   {:24} {:6.0} ms", s.label, t.elapsed().as_secs_f64() * 1e3); } }, Err(e) => { bad += 1; println!("FAIL {}: {}", s.label, &e[..e.len().min(3000)]); } }
        }
        println!("{obs:?}: {ok} ok, {bad} failed");
    }
}
