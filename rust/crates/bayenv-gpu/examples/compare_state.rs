//! Compare two state dumps (veh cap x 40 words + pose cap x 4 floats): per-field max abs error / exact-match counts.
use bayenv_gpu::layout as L;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (x, y) = (std::fs::read(&a[1]).unwrap(), std::fs::read(&a[2]).unwrap());
    assert_eq!(x.len(), y.len(), "size");
    let cap = x.len() / 4 / (L::VEH_WORDS + 4);
    let (vx, vy): (&[u32], &[u32]) = (bytemuck::cast_slice(&x[..cap * L::VEH_WORDS * 4]), bytemuck::cast_slice(&y[..cap * L::VEH_WORDS * 4]));
    let (px, py): (&[f32], &[f32]) = (bytemuck::cast_slice(&x[cap * L::VEH_WORDS * 4..]), bytemuck::cast_slice(&y[cap * L::VEH_WORDS * 4..]));
    let active: Vec<usize> = (0..cap).filter(|&i| vx[i * L::VEH_WORDS + 7] != 0 || vy[i * L::VEH_WORDS + 7] != 0).collect();
    let mut identical_cars = 0;
    for &i in &active { if vx[i * 40..i * 40 + 40] == vy[i * 40..i * 40 + 40] && px[i * 4..i * 4 + 4] == py[i * 4..i * 4 + 4] { identical_cars += 1; } }
    println!("cars compared {}  bit-identical (all 40 fields + pose) {}", active.len(), identical_cars);
    for f in 0..L::VEH_WORDS {
        let t = L::VEH_TYPES[f]; let (mut diff, mut maxe) = (0usize, 0f64);
        for &i in &active { let (u, v) = (vx[i * 40 + f], vy[i * 40 + f]); if u != v { diff += 1;
            let e = match t { b'f' => (f32::from_bits(u) as f64 - f32::from_bits(v) as f64).abs(), b'i' => ((u as i32) as f64 - (v as i32) as f64).abs(), _ => (u as f64 - v as f64).abs() };
            if e.is_finite() && e > maxe { maxe = e; } } }
        if diff > 0 { println!("  {:14} {} differ, max |err| {:.6e}", L::VEH_FIELDS[f], diff, maxe); }
    }
    let mut maxp = 0f64; let mut dp = 0;
    for &i in &active { let d = ((px[i * 4] - py[i * 4]) as f64).hypot((px[i * 4 + 2] - py[i * 4 + 2]) as f64); if d > 0.0 { dp += 1; } if d > maxp { maxp = d; } }
    println!("  pose xz: {dp} differ, max {:.6e} m", maxp);
}
