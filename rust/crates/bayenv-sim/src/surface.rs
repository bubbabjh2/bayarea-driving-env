//! src/sim/surface.js: per-surface tyre factors and carSurface.
use crate::jm;
use crate::world::World;

// muPeakDry, muSlideDry, crr (SURFACES table; codes 12..15 = asphalt)
const SURF: [(f64, f64, f64); 12] = [
    (0.85, 0.68, 0.013), (0.85, 0.68, 0.013), (0.75, 0.60, 0.015), (0.65, 0.52, 0.020), (0.60, 0.55, 0.025), (0.65, 0.58, 0.020),
    (0.68, 0.65, 0.050), (0.45, 0.40, 0.100), (0.40, 0.35, 0.300), (0.55, 0.45, 0.013), (0.60, 0.50, 0.015), (0.85, 0.68, 0.013),
];
pub const PAVED_SHOULDER_FREEWAY: f64 = 3.0;

fn round2(x: f64) -> f64 { jm::round(x * 100.0) / 100.0 }

/// SURF_PK, SURF_SK, SURF_CRR per code.
pub fn factors() -> ([f64; 16], [f64; 16], [f64; 16]) {
    let (mut pk, mut sk, mut cr) = ([0.0; 16], [0.0; 16], [0.0; 16]);
    let r = SURF[0];
    for c in 0..16 {
        let s = if c < 12 { SURF[c] } else { r };
        let p = s.0 / r.0;
        pk[c] = p; sk[c] = p * round2(s.1 / s.0); cr[c] = s.2 / r.2;
    }
    (pk, sk, cr)
}

thread_local! { static F: ([f64; 16], [f64; 16], [f64; 16]) = factors(); }

/// carSurface -> (pk, sk, crr, codeR, codeL)
pub fn car_surface(g: &World, e: usize, s: f64, lat: f64, hw: f64, conn: i32, taper: f64) -> (f64, f64, f64, u8, u8) {
    let cs = g.e_surf[e];
    let (mut c_r, mut c_l) = (cs, cs);
    let (gr, gl) = (g.e_gnd_r[e], g.e_gnd_l[e]);
    let has_j = g.j.is_some();
    if conn < 0 && !(has_j && (s < g.e_s_exit[e] + 1.0 || s > g.e_s_stop[e])) && (gr != cs || gl != cs) {
        let w = g.lane_w(e, s);
        let lat_r = lat - hw;
        let lat_l = lat + hw;
        let nw = g.e_lanes[e] as f64 * w;
        if lat_r < 0.0 || lat_l > nw {
            let (mut pad_r, mut pad_l) = (0.0, nw);
            if let Some(lg) = &g.lg { let bd = lg.bounds(e, s); pad_r = bd.r - bd.right; pad_l = bd.r - bd.left; }
            else if taper > s { pad_r -= w; pad_l += w; }
            let sh = if g.is_shoulder_road(e) { PAVED_SHOULDER_FREEWAY } else { 0.0 };
            let left = if g.e_rev[e] >= 0 { pad_l + g.e_lanes[g.e_rev[e] as usize] as f64 * w } else { pad_l + sh };
            if lat_r < pad_r - sh { c_r = gr; }
            if lat_l > left { c_l = gl; }
        }
    }
    F.with(|(pk, sk, cr)| {
        let (a, b) = (c_r as usize & 15, c_l as usize & 15);
        (0.5 * (pk[a] + pk[b]), 0.5 * (sk[a] + sk[b]), 0.5 * (cr[a] + cr[b]), c_r, c_l)
    })
}
