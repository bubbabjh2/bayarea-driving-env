//! World::load without data/junctions.bin: build (and bake) the junctions; compare with a reference junctions.bin.
fn main() {
    let a: Vec<String> = std::env::args().collect();
    for pass in ["cold (build + bake)", "warm (bake)"] {
        let t = std::time::Instant::now();
        let w = bayenv_world::World::load(&a[1]).unwrap();
        println!("{pass}: World::load {:.2} s", t.elapsed().as_secs_f64());
        if let Some(r) = a.get(2) { let j = w.junctions.as_ref().unwrap(); let refb = std::fs::read(r).unwrap();
            let mut b: Vec<u8> = Vec::new();
            macro_rules! put { ($($f:ident),*) => { $( b.extend_from_slice(bytemuck::cast_slice(&j.$f)); )* } }
            put!(mouth_s, gore_s, j_node, j_first_m, j_count_m, j_first_c, j_count_c, j_first_h, j_count_h, m_info, m_f, pts, conf, conf_s, hull);
            println!("  junctions {} bytes, identical to reference: {}", b.len(), b == refb); }
    }
}
