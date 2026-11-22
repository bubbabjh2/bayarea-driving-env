//! scratch: grade along edges
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let g = bayenv_sim::env::Env::load_world(std::path::Path::new(&a[1])).unwrap();
    for spec in &a[2..] {
        let mut it = spec.split(':'); let e: usize = it.next().unwrap().parse().unwrap(); let s0: f64 = it.next().unwrap().parse().unwrap(); let s1: f64 = it.next().unwrap().parse().unwrap();
        let mut s = s0; let mut out = vec![];
        while s <= s1 { out.push(format!("{s:.0}:{:.3}", g.edge_grade(e, s, -1))); s += (s1 - s0) / 8.0; }
        println!("edge {e} len {:.1} hw {} class {}: {}", g.e_length[e], g.e_hw[e], g.highway_names.get(g.e_class[e] as usize).cloned().unwrap_or_default(), out.join(" "));
    }
}
