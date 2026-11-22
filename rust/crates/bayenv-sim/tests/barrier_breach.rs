//! SimOpts::barrier_breach (default off; DECISIONS.md 46, src/breach.rs): a strike on a bridge's outer rail above the
//! rail's capacity (MASH TL-3 impact severity, 156 kJ) goes through it, and the car falls to the ground or water below.
//! Worlds: "road3" (tests/common/gen_worlds.mjs; its motorway edge 0 flagged as a bridge, over flat ground 30 m down)
//! and the real Bay Bridge east span over the bay (data/ with the DEM).
mod common;
use bayenv_sim::breach::{DemGround, FallPhase, FlatGround, Ground, MASH_TL3_IS_KJ};
use bayenv_sim::population::Population;
use bayenv_sim::sim::*;
use bayenv_sim::vehicle::*;
use bayenv_sim::world::World;
use std::sync::Arc;

const SEDAN: u8 = 0;
const SUV: u8 = 1;
const TRUCK: u8 = 2;
const DROP: f64 = 30.0;

fn sim_with(g: &Arc<World>, opts: SimOpts) -> Sim {
    let opts = SimOpts { dt: 0.05, start_time: 7.0 * 3600.0, fuel: false, ..opts };
    Sim::new(g.clone(), Arc::new(Population::default()), opts, bayenv_sim::harness::router_for(g))
}
fn place(sim: &mut Sim, ty: u8, e: i32, s: f64, lane: i32, speed: f64, hr: f64, lat: Option<f64>) -> usize {
    let g = sim.g.clone();
    let (time, dt, step) = (sim.time, sim.dt, sim.step_count);
    let i = sim.vs.alloc(&g, AllocArgs { type_idx: ty, traits: DEFAULT_TRAITS, route: &[e], commuter: -1, leg: 0, time, learner: false, dt, step, lane });
    sim.vs.s[i] = s as f32; sim.vs.speed[i] = speed as f32; sim.vs.set_reaction_time(i, 0.05, dt);
    if let Some(l) = lat { sim.vs.lat[i] = l as f32; }
    sim.vs.heading_rel[i] = hr as f32;
    sim.vs.update_lanes(i, &g, e as usize); sim.vs.update_pose(i, &g); sim.vs.save_pose(i);
    if sim.drivers.len() <= i { sim.drivers.resize(i + 1, Driver::None); }
    if sim.actions.len() <= i { sim.actions.resize(i + 1, None); }
    sim.drivers[i] = Driver::External;
    i
}
fn hold(sim: &mut Sim, i: usize, gas: f64) { sim.actions[i] = Some(Action { steer: Some(0.0), steer_rate: None, gas, brake: 0.0 }); }

/// road3 with every edge flagged as a bridge (or not)
fn bridge_world(bridge: bool) -> Option<Arc<World>> {
    let mut w = common::load_world("road3")?;
    w.e_is_bridge = vec![bridge; w.edge_count];
    Some(Arc::new(w))
}

struct Run { slot: usize, state: u8, fell: bool, sim: Sim, trace: Vec<(f32, f32, f32, f32)> }
/// A car of type `ty` at `u` m/s, `deg` degrees toward the right-hand (outer) barrier (negative: toward the left), in
/// lane 0 of road3's motorway, gas holding its speed, for `secs`.
fn strike(g: &Arc<World>, opts: SimOpts, ty: u8, u: f64, deg: f64, secs: f64) -> Run { strike_on(g, opts, ty, u, deg, secs, false) }
fn strike_on(g: &Arc<World>, opts: SimOpts, ty: u8, u: f64, deg: f64, secs: f64, water: bool) -> Run {
    let mut sim = sim_with(g, opts);
    let y0 = g.edge_point(0, 500.0, 0.0, 0).y;
    sim.ground = Some(Arc::new(FlatGround { y: y0 - DROP, water }));
    let i = place(&mut sim, ty, 0, 500.0, 0, u, -deg.to_radians(), None);
    let mut trace = vec![];
    for _ in 0..(secs / sim.dt) as usize {
        if sim.vs.state[i] == STATE_DRIVING { hold(&mut sim, i, 0.3); }
        sim.step();
        trace.push((sim.vs.x[i], sim.vs.z[i], sim.vs.speed[i], sim.vs.lat[i]));
    }
    let fell = sim.falls.iter().any(|f| f.slot == i);
    Run { slot: i, state: sim.vs.state[i], fell, sim, trace }
}

#[test]
fn breach_at_30_ms_25_deg_falls_and_lands_below() {
    let Some(g) = bridge_world(true) else { return bayenv_world::oracle::skip("barrier_breach", "fixtures") };
    let on = SimOpts { barrier_breach: true, ..SimOpts::default() };
    let r = strike(&g, on.clone(), SUV, 30.0, 25.0, 8.0);
    assert!(r.fell, "an SUV at 30 m/s, 25 deg into the bridge rail goes through it");
    let fl = &r.sim.falls[0];
    let m = bayenv_sim::physics::VEHICLE_PARAMS[SUV as usize].m;
    let expect = 0.5 * m * (30.0f64 * 25f64.to_radians().sin()).powi(2) / 1000.0;
    eprintln!("SUV ({m} kg) 30 m/s 25 deg: IS {:.1} kJ (ideal {expect:.1}; TL-3 {MASH_TL3_IS_KJ}), v_normal {:.2} m/s, dv {:.1} km/h; landed after {:.2} s, {:.1} m down at {:.1} m/s; rest at ({:.1}, {:.1}, {:.1}), {:?}",
        fl.is_kj, fl.v_normal, fl.dv_kmh, fl.land_time - fl.time, fl.drop, -fl.land_vy, fl.pos[0], fl.pos[1], fl.pos[2], fl.phase);
    assert!(fl.is_kj > MASH_TL3_IS_KJ && (fl.is_kj - expect).abs() < 0.05 * expect, "impact severity {} vs {expect}", fl.is_kj);
    // the car left the store: slot freed, can never arrive; a CRASHED event and a breach crash record
    assert_eq!(r.state, STATE_FREE);
    assert!(r.sim.event_log.iter().any(|e| e.kind == EV_CRASHED && e.slot == r.slot as i32), "CRASHED event");
    assert!(r.sim.crashes.iter().any(|c| c.cause == "breach" && c.disabled == vec![r.slot]), "breach crash record");
    assert_eq!(r.sim.counters.crashed_vehicles, 1);
    // ballistic: free-fall time sqrt(2 h / g) (plus the slide over the deck edge), impact speed sqrt(2 g h)
    let t_free = (2.0 * DROP / 9.81).sqrt();
    let t_land = fl.land_time - fl.time;
    assert!(t_land > t_free && t_land < t_free + 0.6, "landing after {t_land:.2} s (free fall {t_free:.2} s)");
    assert!((-fl.land_vy - (2.0 * 9.81 * DROP).sqrt()).abs() < 0.3, "impact speed {}", -fl.land_vy);
    assert!((fl.drop - DROP).abs() < 1e-6 && !fl.water);
    // at rest on the ground below, outside the rail line, on its wheels or its roof
    assert_eq!(fl.phase, FallPhase::Rest);
    let y0 = g.edge_point(0, 500.0, 0.0, 0).y;
    assert!((fl.pos[1] - (y0 - DROP)).abs() < 1e-6, "on the ground: y {}", fl.pos[1]);
    let past = (fl.pos[0] - fl.rail.0) * fl.normal.0 + (fl.pos[2] - fl.rail.1) * fl.normal.1;
    assert!(past > 1.0, "lands outside the bridge: {past:.2} m past the rail");
    let up = bayenv_sim::breach::qrot(fl.q, [0.0, 1.0, 0.0]);
    assert!(up[1].abs() > 0.999, "settled flat: up {up:?}");
    // deterministic: the same run gives the same fall, bit for bit
    let r2 = strike(&g, on, SUV, 30.0, 25.0, 8.0);
    let f2 = &r2.sim.falls[0];
    assert_eq!((fl.pos, fl.q, fl.land_time), (f2.pos, f2.q, f2.land_time));
}

#[test]
fn shallow_scrape_and_tl3_containment_do_not_breach() {
    let Some(g) = bridge_world(true) else { return bayenv_world::oracle::skip("barrier_breach", "fixtures") };
    let on = SimOpts { barrier_breach: true, ..SimOpts::default() };
    // a shallow scrape at the same speed: IS = 1/2 2100 (30 sin 5)^2 = 7 kJ
    let r = strike(&g, on.clone(), SUV, 30.0, 5.0, 4.0);
    assert!(!r.fell && r.sim.falls.is_empty() && r.state != STATE_FREE, "a 5 deg scrape stays on the bridge");
    assert!(r.sim.event_log.iter().any(|e| (e.kind == EV_CONTACT || e.kind == EV_CRASHED) && e.slot == r.slot as i32), "it did touch the rail");
    // a 1.5 t sedan at 30 m/s, 25 deg: IS = 120 kJ, under TL-3's 156 kJ: a TL-3 rail contains it (it is crash-tested
    // with a 1100 kg car at 100 km/h, 25 deg too) ...
    let s = strike(&g, on.clone(), SEDAN, 30.0, 25.0, 4.0);
    assert!(!s.fell, "TL-3 contains a sedan at 30 m/s, 25 deg");
    // ... and a TL-2 rail (2270 kg at 70 km/h, 25 deg: 76 kJ) does not
    let t2 = strike(&g, SimOpts { breach_is_kj: 76.0, ..on.clone() }, SEDAN, 30.0, 25.0, 4.0);
    assert!(t2.fell, "a TL-2 rail lets the sedan through");
    // only the outer (right-hand) rail of a bridge: the left barrier holds, and so does a rail that is not on a bridge
    let l = strike(&g, on.clone(), SUV, 30.0, -25.0, 4.0);
    assert!(!l.fell, "left-hand barrier: no breach");
    let ground = bridge_world(false).unwrap();
    let n = strike(&ground, on, SUV, 30.0, 25.0, 4.0);
    assert!(!n.fell, "not a bridge: no breach");
}

#[test]
fn flag_off_is_unchanged() {
    let Some(g) = bridge_world(true) else { return bayenv_world::oracle::skip("barrier_breach", "fixtures") };
    assert!(!SimOpts::default().barrier_breach && !SimOpts::js().barrier_breach, "default off");
    // the strike that breaches with the flag on is an ordinary (contained) barrier crash with it off
    let off = strike(&g, SimOpts::default(), SUV, 30.0, 25.0, 4.0);
    assert!(!off.fell && off.state == STATE_CRASHED, "contained and disabled with the flag off");
    // where nothing breaches, flag on and off are the same run, step for step and bit for bit
    for (ty, deg) in [(SUV, 5.0), (SEDAN, 25.0), (SUV, -25.0)] {
        let a = strike(&g, SimOpts::default(), ty, 30.0, deg, 4.0);
        let b = strike(&g, SimOpts { barrier_breach: true, ..SimOpts::default() }, ty, 30.0, deg, 4.0);
        assert_eq!(a.trace, b.trace, "type {ty} at {deg} deg");
        assert_eq!(a.sim.event_log.len(), b.sim.event_log.len());
    }
}

#[test]
fn water_landing_ploughs_in_and_stops_within_20_m() {
    // DECISIONS.md 49: water entry from a 30 m fall kills the horizontal speed within metres (quadratic drag
    // 1/2 rho Cd A v^2), heavier vehicles a little further; it sinks to its resting depth
    let Some(g) = bridge_world(true) else { return bayenv_world::oracle::skip("barrier_breach", "fixtures") };
    let on = SimOpts { barrier_breach: true, ..SimOpts::default() };
    let mut dist = vec![];
    for (ty, deg) in [(SUV, 25.0), (TRUCK, 12.0)] {
        let r = strike_on(&g, on.clone(), ty, 30.0, deg, 12.0, true);
        assert!(r.fell, "type {ty} breaches");
        let fl = &r.sim.falls[0];
        let d = ((fl.pos[0] - fl.land_xz.0).powi(2) + (fl.pos[2] - fl.land_xz.1).powi(2)).sqrt();
        eprintln!("type {ty} ({:.0} kg, A {:.1} m^2): hits the water at {:.1} m/s horizontal, at rest {d:.1} m on, depth {:.2} m, {:?}", fl.mass, fl.area, fl.land_vh, fl.pos[1] - fl.land_y, fl.phase);
        assert!(fl.water && fl.land_vh > 20.0, "it entered the water fast: {}", fl.land_vh);
        assert!(d < 20.0, "type {ty}: came to rest {d:.1} m from water entry");
        assert_eq!(fl.phase, FallPhase::Rest);
        assert!((fl.pos[1] - (fl.land_y - bayenv_sim::breach::SINK_DEPTH)).abs() < 1e-9, "sunk to its resting depth");
        dist.push(d);
    }
    assert!(dist[1] > dist[0], "the truck ploughs further than the SUV: {dist:?}");
}

/// the real Bay Bridge east span (a 5-lane Skyway deck east of Yerba Buena Island, x > -18500, with the bay under its
/// right side; its decks are side by side, so nothing is below but water), with the DEM
#[test]
fn bay_bridge_breach_lands_in_the_bay() {
    let d = bayenv_sim::population_gen::default_data_dir();
    if !d.join("roads.meta.json").exists() { return bayenv_world::oracle::skip("barrier_breach", "no data dir"); }
    let g = bayenv_sim::env::Env::load_world(&d).expect("world");
    let ground = DemGround::load(&d).expect("data/dem/compact");
    let e = (0..g.edge_count).filter(|&e| g.e_is_bridge[e] && g.is_shoulder_road(e) && g.e_lanes[e] == 5 && g.e_length[e] > 2500.0 && {
        let p = g.edge_point(e, 0.5 * g.e_length[e], 0.0, 0);
        let w = ground.height(p.x - p.hz * 15.0, p.z + p.hx * 15.0);
        w.1 && p.y > 30.0 && p.x > -18500.0
    }).max_by(|&a, &b| g.e_length[a].partial_cmp(&g.e_length[b]).unwrap()).expect("a Bay Bridge deck");
    let mut sim = sim_with(&g, SimOpts { barrier_breach: true, ..SimOpts::default() });
    sim.ground = Some(ground);
    let s0 = 0.5 * g.e_length[e];
    let i = place(&mut sim, SUV, e as i32, s0, 0, 30.0, -25f64.to_radians(), None);
    for _ in 0..400 { if sim.vs.state[i] == STATE_DRIVING { hold(&mut sim, i, 0.3); } sim.step(); }
    assert_eq!(sim.falls.len(), 1, "breach on edge {e}");
    let fl = &sim.falls[0];
    eprintln!("Bay Bridge edge {e} ({} m): deck {:.1} m, IS {:.1} kJ, fell {:.1} m in {:.2} s, hit the water at {:.1} m/s, rest at ({:.1}, {:.2}, {:.1}) {:?}",
        g.e_length[e], fl.deck_y, fl.is_kj, fl.drop, fl.land_time - fl.time, -fl.land_vy, fl.pos[0], fl.pos[1], fl.pos[2], fl.phase);
    assert!(fl.water && fl.land_y == 0.0, "into the bay");
    assert!(fl.drop == fl.deck_y && fl.drop > 30.0, "fell from the deck to sea level");
    assert_eq!(fl.phase, FallPhase::Rest);
    assert!((fl.pos[1] + bayenv_sim::breach::SINK_DEPTH).abs() < 1e-9, "sunk to {}", fl.pos[1]);
    let d = ((fl.pos[0] - fl.land_xz.0).powi(2) + (fl.pos[2] - fl.land_xz.1).powi(2)).sqrt();
    assert!(d < 20.0, "at rest {d:.1} m from water entry");
}
