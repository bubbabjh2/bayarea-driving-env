//! Port of test/sim/surface.test.js: 5 JS test cases -> 5 #[test]s (the first only partially).
//!   - "surface table": ported for the identity row and the pk/sk/crr invariants that the Rust `surface::factors()` exposes.
//!     NOT ported: the SURFACES wet/dry mu table and osmSurfaceCode (OSM tag mapping, built by tools/surfaces in JS; not in Rust).
//!   - braking distance / rolling resistance: physics level, 1:1.
//!   - carSurface: the JS test fakes edge objects; here the 'surfcar' world (motorway 2 lanes + residential pair, from
//!     tests/common/gen_worlds.mjs) is loaded and its e_surf / e_gnd_r / e_gnd_l arrays are set to the same codes.
//!   - sim: JS attachSurfaces(graph, bytes) is `World.e_surf/e_gnd_r/e_gnd_l = code; World.surfaces = true` on the loaded synthetic world.
mod common;
use bayenv_sim::jm;
use bayenv_sim::physics::*;
use bayenv_sim::population_gen::*;
use bayenv_sim::sim::*;
use bayenv_sim::surface::{car_surface, factors};
use bayenv_sim::world::World;
use std::sync::Arc;

const DT: f64 = 0.05;
// SURFACE_CODES
const ASPHALT: u8 = 0; const CONCRETE: u8 = 1; const PAVING: u8 = 2; const SETT: u8 = 3; const GRAVEL: u8 = 4; const DIRT: u8 = 6; const GRASS: u8 = 7; const SAND: u8 = 8; const METAL: u8 = 9; const CURB: u8 = 11;

fn uniform(code: u8) -> Surf { let (pk, sk, cr) = factors(); Surf { pk: pk[code as usize], sk: sk[code as usize], crr: cr[code as usize] } }
fn brake_distance(code: Option<u8>, locked: bool) -> f64 {
    let mut st = DynState { u: 100.0 / 3.6, ..Default::default() }; let mut t = 0.0;
    let sf = code.map(uniform);
    while st.u > 0.0 && t < 60.0 { step_dynamics(&mut st, 0.0, 0.0, 1.0, &VEHICLE_PARAMS[0], DT, 0.0, locked, false, sf); t += DT; }
    st.s
}

#[test]
fn surface_table_identity_and_invariants() {
    let (pk, sk, cr) = factors();
    assert_eq!(pk[ASPHALT as usize], 1.0); assert_eq!(sk[ASPHALT as usize], 0.8); assert_eq!(cr[ASPHALT as usize], 1.0);
    for c in 0..12 {
        assert!(sk[c] <= pk[c] + 1e-9, "sliding <= peak for surface {c}");
        assert!(cr[c] > 0.0 && pk[c] > 0.0, "surface {c}");
    }
    for c in 12..16 { assert_eq!((pk[c], sk[c], cr[c]), (1.0, 0.8, 1.0), "unused codes = asphalt"); }
}

#[test]
fn braking_100_to_0_asphalt_identical_lower_mu_longer_about_1_over_mu() {
    let (base, asph) = (brake_distance(None, false), brake_distance(Some(ASPHALT), false));
    assert_eq!(asph, base, "dry asphalt = previous model");
    let (pk, sk, _) = factors();
    let mut out = std::collections::HashMap::new();
    for (name, n) in [("concrete", CONCRETE), ("paving_stones", PAVING), ("sett", SETT), ("gravel", GRAVEL), ("dirt", DIRT), ("grass", GRASS), ("metal_grate", METAL)] {
        let d = brake_distance(Some(n), false);
        out.insert(name, d);
        let p = pk[n as usize];
        if p < 1.0 { assert!(d > asph * 1.05, "{name} {d:.1} m > asphalt {asph:.1} m"); }
        // tyre-limited (ABS) braking: distance ~ v^2 / (2 * 0.95 mu g); aero + rolling make it a bit shorter
        let ratio = d / asph;
        assert!(ratio > 0.85 / p && ratio < 1.05 / p, "{name}: ratio {ratio:.3} vs 1/pk {:.3}", 1.0 / p);
    }
    assert!(out["grass"] > out["gravel"] && out["gravel"] > asph);
    // sand: low mu and very high rolling resistance
    assert!(brake_distance(Some(SAND), false) > asph);
    // locked wheels (disabled car) slide on the sliding coefficient: distance ~ 1 / sk
    let (la, lg) = (brake_distance(Some(ASPHALT), true), brake_distance(Some(GRASS), true));
    let (r, want) = (lg / la, sk[ASPHALT as usize] / sk[GRASS as usize]);
    assert!(r > 0.85 * want && r < 1.05 * want, "locked grass/asphalt {r:.3} vs {want:.3}");
}

#[test]
fn rolling_resistance_follows_the_surface() {
    let (mut a, mut g) = (DynState { u: 15.0, ..Default::default() }, DynState { u: 15.0, ..Default::default() });
    for _ in 0..100 {
        step_dynamics(&mut a, 0.0, 0.0, 0.0, &VEHICLE_PARAMS[0], DT, 0.0, false, false, Some(uniform(ASPHALT)));
        step_dynamics(&mut g, 0.0, 0.0, 0.0, &VEHICLE_PARAMS[0], DT, 0.0, false, false, Some(uniform(GRASS)));
    }
    let (da, dg) = (15.0 - a.u, 15.0 - g.u);
    assert!(dg > 3.0 * da, "grass speed loss {dg:.2} vs asphalt {da:.2} m/s");
}

#[test]
fn car_surface_pavement_vs_side_ground() {
    let mut g = need!(common::load_world("surfcar"));
    let meta = need!(common::fixtures_meta());
    let edges = meta["surfcar"]["edges"].as_array().unwrap();
    assert_eq!(edges[0]["highway"], "motorway"); assert_eq!(edges[1]["reverse"], 2);
    g.e_surf[0] = CONCRETE; g.e_gnd_r[0] = GRASS; g.e_gnd_l[0] = GRAVEL;
    for e in [1, 2] { g.e_surf[e] = ASPHALT; g.e_gnd_r[e] = CURB; g.e_gnd_l[e] = CURB; }
    let (pk, _, _) = factors();
    let hw = 0.9;
    let r = car_surface(&g, 0, 100.0, 3.7, hw, -1, 0.0);
    assert_eq!((r.3, r.4, r.0), (CONCRETE, CONCRETE, 1.0));
    let r = car_surface(&g, 0, 100.0, -1.5, hw, -1, 0.0);   // on the 3 m paved shoulder: still concrete
    assert_eq!(r.3, CONCRETE);
    let r = car_surface(&g, 0, 100.0, -3.5, hw, -1, 0.0);   // right wheels past the shoulder: grass
    assert_eq!(r.3, GRASS); assert_eq!(r.4, CONCRETE);
    assert!((r.0 - 0.5 * (pk[GRASS as usize] + 1.0)).abs() < 1e-12);
    let r = car_surface(&g, 0, 100.0, 2.0 * 3.7 + 3.5, hw, -1, 0.0);   // left wheels past the left shoulder: gravel median
    assert_eq!(r.4, GRAVEL);
    let r = car_surface(&g, 0, 100.0, -3.5, hw, 0, 0.0);    // inside a junction box (connector): paved
    assert_eq!(r.3, CONCRETE);
    // two-way surface street: the left side is the opposing lanes, then that side's ground class
    let r = car_surface(&g, 1, 50.0, 3.7 + 1.5, hw, -1, 0.0);
    assert_eq!(r.4, ASPHALT);
}

#[test]
fn sim_all_asphalt_map_identical_to_no_map_all_grass_changes_dynamics() {
    let base = need!(common::shared_world("synth"));
    let mut pc = PopConfig::env_synthetic(60, 1.0); pc.return_trips = false;
    let pop = Arc::new(generate_population(&base, &pc, Some(1)).unwrap().pop);
    let run = |code: Option<u8>| {
        let mut w: World = common::load_world("synth").unwrap();
        if let Some(c) = code { for e in 0..w.edge_count { w.e_surf[e] = c; w.e_gnd_r[e] = c; w.e_gnd_l[e] = c; } w.surfaces = true; }
        let w = Arc::new(w);
        let opts = SimOpts { dt: 0.1, seed: 3.0, default_action: Action { steer: Some(0.04), steer_rate: None, gas: 0.6, brake: 0.0 }, ..SimOpts::js() };
        let mut env = common::env_with(w.clone(), pop.clone(), opts, 7.5 * 3600.0);
        for _ in 0..900 { env.step(1); }
        let v = &env.sim.vs;
        let (mut h, mut codes) = (0.0f64, std::collections::HashSet::new());
        for i in 0..v.high {
            h = jm::fmod(h * 31.0 + jm::round(v.s[i] as f64 * 1000.0) + jm::round(v.speed[i] as f64 * 1000.0), 1e9);
            if v.state[i] != 0 { codes.insert(v.surf_r[i]); }
        }
        (h, codes, v.odometer[..v.high].to_vec())
    };
    let (none, asph, grass) = (run(None), run(Some(ASPHALT)), run(Some(GRASS)));
    assert_eq!(asph.0, none.0); assert_eq!(asph.2, none.2);
    assert_ne!(grass.0, none.0);
    assert!(grass.1.contains(&GRASS));
}
