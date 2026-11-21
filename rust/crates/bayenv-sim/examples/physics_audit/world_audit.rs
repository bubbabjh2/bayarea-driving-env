//! World-geometry scan, targeted sim scenarios and rollout anomaly scan for the physics audit.
use bayenv_sim::env::{Env, EnvConfig};
use bayenv_sim::physics::{self, VEHICLE_PARAMS};
use bayenv_sim::population::Population;
use bayenv_sim::sim::*;
use bayenv_sim::vehicle::*;
use bayenv_sim::world::World;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

fn wrap(a: f64) -> f64 { a.sin().atan2(a.cos()) }

fn y_at(g: &World, e: usize, s: f64) -> f64 { g.edge_point(e, s.max(0.0).min(g.e_length[e]), 0.0, 0).y }

pub fn world(data: &str) {
    let g = Env::load_world(Path::new(data)).expect("world");
    let ne = g.edge_count;
    println!("== W1 edges: {ne} directed edges, {} nodes ==", g.node_count);
    let mut lens: Vec<f64> = g.e_length.clone(); lens.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let short = |x: f64| g.e_length.iter().filter(|&&l| l < x).count();
    let maxl = lens[ne - 1];
    println!("length min {:.3} m, < 0.5 m: {}, < 1 m: {}, < 4.5 m (a sedan): {}, < 12 m (a bus): {}; max {:.0} m (f32 ulp of s there: {:.1e} m; a car at 0.1 m/s moves {:.1e} m per 0.05 s step)",
        lens[0], short(0.5), short(1.0), short(4.5), short(12.0), maxl, (maxl as f32).to_bits() as f64 * 0.0 + f32::EPSILON as f64 * maxl, 0.1 * 0.05);
    // zero-length polyline segments
    let mut zero_seg = 0usize; let mut segs = 0usize;
    for e in 0..ne { let v0 = g.v_start[e] as usize; let n = g.v_start[e + 1] as usize - v0; for k in 0..n.saturating_sub(1) { segs += 1; if g.cum[v0 + k + 1] - g.cum[v0 + k] < 1e-6 { zero_seg += 1; } } }
    println!("polyline segments {segs}, zero-length {zero_seg}");
    let (mut wmin, mut wmax) = (1e9f64, 0.0f64);
    for e in 0..ne { for k in 0..=4 { let w = g.lane_w(e, g.e_length[e] * k as f64 / 4.0); wmin = wmin.min(w); wmax = wmax.max(w); } }
    println!("lane width range {wmin:.2} .. {wmax:.2} m");

    println!("\n== W2 grades (raw polyline slope vs the physics' edge_grade, which averages over +-6 m and clamps at 35 %) ==");
    let mut raw_len = [0.0f64; 6]; let thr = [0.05, 0.10, 0.15, 0.25, 0.35, 0.50];
    let mut total = 0.0; let mut steepest = (0.0f64, 0usize, 0.0f64);
    let mut phys_len = [0.0f64; 6];
    for e in 0..ne {
        if !g.e_has_ys[e] { total += g.e_length[e]; continue; }
        let v0 = g.v_start[e] as usize; let n = g.v_start[e + 1] as usize - v0;
        for k in 0..n.saturating_sub(1) {
            let ds = g.cum[v0 + k + 1] - g.cum[v0 + k]; if ds < 1e-6 { continue; }
            let gr = (g.ys[v0 + k + 1] - g.ys[v0 + k]).abs() / ds;
            total += ds;
            for (i, &t) in thr.iter().enumerate() { if gr > t { raw_len[i] += ds; } }
            if gr > steepest.0 && ds > 5.0 { steepest = (gr, e, g.cum[v0 + k]); }
        }
        let mut s = 0.0; while s < g.e_length[e] { let gp = g.edge_grade(e, s, 0).abs(); for (i, &t) in thr.iter().enumerate() { if gp > t { phys_len[i] += 2.0; } } s += 2.0; }
    }
    for (i, &t) in thr.iter().enumerate() { println!("road length with |grade| > {:3.0} %: raw {:8.1} km ({:.2} %), physics {:8.1} km", t * 100.0, raw_len[i] / 1e3, 100.0 * raw_len[i] / total, phys_len[i] / 1e3); }
    let e = steepest.1; let p = g.edge_point(e, steepest.2, 0.0, 0);
    println!("steepest segment > 5 m: {:.1} % on edge {e} ({}, {:.0} m long) at x {:.0} z {:.0}", steepest.0 * 100.0, g.highway(e), g.e_length[e], p.x, p.z);

    println!("\n== W3 elevation discontinuities at edge hops (y at the end of e vs the start of the next edge) ==");
    let mut bins = [0usize; 5]; let bt = [0.25, 0.5, 1.0, 2.5, 5.0]; let mut hops = 0usize; let mut worst = (0.0f64, 0usize, 0usize);
    for e in 0..ne {
        let to = g.e_to[e] as usize;
        let ye = y_at(&g, e, g.e_length[e]);
        for q in g.out_start[to]..g.out_start[to + 1] {
            let en = g.out_edges[q as usize] as usize; if en as i32 == g.e_rev[e] { continue; }
            hops += 1;
            let d = (y_at(&g, en, 0.0) - ye).abs();
            for (i, &t) in bt.iter().enumerate() { if d > t { bins[i] += 1; } }
            if d > worst.0 { worst = (d, e, en); }
        }
    }
    println!("{hops} hops; |dy| > 0.25 m: {}, > 0.5: {}, > 1: {}, > 2.5 m (layer_gap): {}, > 5 m: {}; worst {:.1} m ({} -> {}, {} -> {})", bins[0], bins[1], bins[2], bins[3], bins[4], worst.0, worst.1, worst.2, g.highway(worst.1), g.highway(worst.2));
    if let Some(j) = &g.j {
        let mut cb = [0usize; 4]; let ct = [0.5, 1.0, 2.5, 5.0]; let mut worst = 0.0f64;
        let mut rbins = [0usize; 5]; let rt = [5.0, 8.0, 12.0, 20.0, 40.0]; let mut big = 0usize;
        for m in 0..j.move_count {
            let a = j.info(m, 0) as usize; let en = j.info(m, 2) as usize;
            if a >= ne || en >= ne { continue; }
            let p0 = j.connector_point(m, 0.0); let p1 = j.connector_point(m, j.m_len[m]);
            let d0 = (p0.y - y_at(&g, a, j.s_in(m))).abs(); let d1 = (p1.y - y_at(&g, en, j.s_out(m))).abs();
            let d = d0.max(d1); worst = worst.max(d);
            for (i, &t) in ct.iter().enumerate() { if d > t { cb[i] += 1; } }
            let turn = j.turn(m).abs();
            if turn > 1.2 { big += 1; let r = j.m_len[m] / turn; for (i, &t) in rt.iter().enumerate() { if r < t { rbins[i] += 1; } } }
        }
        println!("junction connectors {}: endpoint |dy| vs edges > 0.5 m: {}, > 1: {}, > 2.5: {}, > 5: {} (worst {worst:.1} m); physics grade inside a connector is 0", j.move_count, cb[0], cb[1], cb[2], cb[3]);
        println!("turns > 69 deg: {big}; mean radius (length / turn) < 5 m: {}, < 8: {}, < 12: {}, < 20: {}, < 40: {}  (1 g on R = 8 m is {:.0} km/h, 0.3 g comfort {:.0} km/h)", rbins[0], rbins[1], rbins[2], rbins[3], rbins[4], (9.81f64 * 8.0).sqrt() * 3.6, (0.3f64 * 9.81 * 8.0).sqrt() * 3.6);
    }
    // heading kinks at hops between edges (pose heading jumps when the frame changes)
    let mut kinks = [0usize; 4]; let kt = [0.1, 0.2, 0.5, 1.0]; let mut strict_hops = 0;
    for e in 0..ne {
        let to = g.e_to[e] as usize;
        for q in g.out_start[to]..g.out_start[to + 1] {
            let en = g.out_edges[q as usize] as usize; if en as i32 == g.e_rev[e] { continue; }
            if !g.hop(e, en).strict { continue; }
            if g.j.is_some() && g.e_s_stop[e] < g.e_length[e] { continue; } // junction box: connector carries the turn
            strict_hops += 1;
            let a = g.edge_point_dir(e, g.e_length[e], 0); let b = g.edge_point_dir(en, 0.0, 0);
            let d = wrap(b.hz.atan2(b.hx) - a.hz.atan2(a.hx)).abs();
            for (i, &t) in kt.iter().enumerate() { if d > t { kinks[i] += 1; } }
        }
    }
    println!("strict hops without a junction box {strict_hops}: frame heading jump > 0.1 rad: {}, > 0.2: {}, > 0.5: {}, > 1.0: {} (a car keeps heading_rel across the hop, so its world heading jumps by this much in one step)", kinks[0], kinks[1], kinks[2], kinks[3]);
}

// ------------------------------------------------------------------------------------------------ rollouts
#[derive(Clone, Copy, Default)]
struct Snap { state: u8, x: f64, y: f64, z: f64, h: f64, u: f64, v: f64, r: f64, edge: i32, conn: i32, spawn: f32, lat: f64, s: f64, ty: u8 }

fn snap(env: &Env) -> Vec<Snap> {
    let vs = &env.sim.vs;
    (0..vs.high).map(|i| Snap { state: vs.state[i], x: vs.x[i] as f64, y: vs.y[i] as f64, z: vs.z[i] as f64, h: vs.heading[i] as f64, u: vs.speed[i] as f64, v: vs.v_lat[i] as f64, r: vs.yaw_rate[i] as f64,
        edge: vs.edge[i], conn: vs.conn[i], spawn: vs.spawn_time[i], lat: vs.lat[i] as f64, s: vs.s[i] as f64, ty: vs.type_idx[i] }).collect()
}

fn make_env(data: &str, n: usize, seed: f64, start_h: f64, mode: TrafficMode, dt: f64) -> Env {
    let world = Env::load_world(Path::new(data)).expect("world");
    let pop = Arc::new(Population::generate_like_env(&world, "real", n, seed, Some(Path::new(data))).expect("pop"));
    // AUDIT_JS=1: the JS model (all model fixes off)
    let base = if std::env::var("AUDIT_JS").is_ok() { SimOpts::js() } else { SimOpts::default() };
    let opts = SimOpts { dt, seed, traffic_mode: mode, ..base };
    let cfg = EnvConfig { opts, start_time: Some(start_h * 3600.0), static_obstacles: Vec::new(), scenario_obstacles: Vec::new(), obstacles: None };
    Env::new(world.clone(), pop, cfg, Box::new(|w| bayenv_sim::harness::router_for(&w)))
}

/// A simple physical-controls driver: lane keeping (heading + offset), speed toward the limit, brake for a leader and
/// for a red / stop control. Position-mode steer in [-1, 1].
fn keep_policy(env: &Env, i: usize, wobble: f64, t: f64) -> Action {
    let vs = &env.sim.vs; let g = &env.world;
    let e = vs.edge[i] as usize; let s = vs.s[i] as f64;
    let w = g.lane_w(e, s);
    let lane = vs.lane[i].max(0) as f64;
    let target_lane = if vs.target_lane[i] >= 0 { vs.target_lane[i] as f64 } else { lane };
    let err = vs.lat[i] as f64 - (target_lane + 0.5) * w;
    let hr = vs.heading_rel[i] as f64;
    let p = &VEHICLE_PARAMS[vs.type_idx[i] as usize];
    let delta = (-0.04 * err - 0.6 * hr).clamp(-0.15, 0.15) + wobble * (t * 0.7 + i as f64).sin();
    let u = vs.speed[i] as f64;
    let lim = g.e_speed[e].max(8.0);
    let mut gas = ((lim - u) * 0.3).clamp(0.0, 1.0);
    let mut brake: f64 = 0.0;
    let gap = vs.gap[i] as f64;
    if vs.leader[i] != -1 && gap < 2.0 * u + 6.0 { gas = 0.0; brake = ((2.0 * u + 6.0 - gap) / 15.0).clamp(0.0, 1.0); }
    let cd = vs.ctl_dist[i] as f64;
    if (vs.ctl_state[i] == 2 || vs.ctl_type[i] == 1) && cd < u * u / 5.0 + 8.0 && cd > -1.0 { gas = 0.0; brake = brake.max(if cd < 3.0 { 0.6 } else { 0.35 }); }
    if u < 0.3 && brake > 0.0 { brake = brake.max(0.3); }
    Action { steer: Some((delta / p.max_steer).clamp(-1.0, 1.0)), steer_rate: None, gas, brake }
}

pub fn sim(data: &str, n: usize, steps: usize, policy: &str) {
    let dt = 0.05;
    let mut env = make_env(data, n, 42.0, 7.0, TrafficMode::Full, dt);
    let t0 = std::time::Instant::now();
    let g = env.world.clone();
    let mut prev = snap(&env);
    let mut stopped: HashMap<(usize, u32), f64> = HashMap::new();
    #[derive(Default)] struct C { steps: u64, car_steps: u64, tele: u64, tele_edge: u64, tele_conn: u64, tele_contact: u64, tele_other: u64, tele_max: f64, tele_ex: String,
        hjump: u64, hjump_ex: String, yjump: u64, yjump_max: f64, yjump_ex: String, fast: u64, acc: u64, acc_max: f64, alat_over: u64, alat_over_kin: u64, alat_05: u64, fast_steps: u64, alat_max: f64, alat_ex: String,
        spawn: u64, spawn_overlap: u64, spawn_ex: String, overlap_pairs: u64, overlap_by: HashMap<&'static str, u64>, under: [u64; 5], under_heavy: u64,
        crash_ev: u64, ke_gain: u64, ke_gain_max: f64, contact_dy: u64, contact_dy_ex: String, nan: u64 }
    let mut c = C::default();
    let mut seen_pairs: std::collections::HashSet<(usize, usize, u32)> = Default::default();
    for step in 0..steps {
        let t = env.sim.time;
        let slots: Vec<usize> = env.driving().collect();
        for &i in &slots {
            let a = match policy {
                "const" => Action { steer: Some(0.0), steer_rate: None, gas: 0.3, brake: 0.0 },
                "wobble" => keep_policy(&env, i, 0.03, t),
                "fast" => { let mut a = keep_policy(&env, i, 0.0, t); let u = env.sim.vs.speed[i] as f64; a.gas = ((30.0 - u) * 0.3).clamp(0.0, 1.0); if a.brake > 0.0 && env.sim.vs.leader[i] == -1 { a.brake = 0.0; } a }
                _ => keep_policy(&env, i, 0.0, t),
            };
            env.act(i, a);
        }
        env.step(1);
        c.steps += 1;
        let now = snap(&env);
        let collided: std::collections::HashSet<usize> = env.sim.collided_last.iter().copied().collect();
        let evs = env.events();
        for ev in &evs {
            if ev.kind == EV_SPAWNED {
                c.spawn += 1;
                let i = ev.slot as usize; let a = &now[i];
                let pa = &VEHICLE_PARAMS[a.ty as usize];
                for (j, b) in now.iter().enumerate() {
                    if j == i || (b.state != STATE_DRIVING && b.state != STATE_CRASHED) { continue; }
                    if (a.x - b.x).hypot(a.z - b.z) > 15.0 || (a.y - b.y).abs() > 2.5 { continue; }
                    let pb = &VEHICLE_PARAMS[b.ty as usize];
                    if physics::obb_contact((a.x, a.z, a.h, pa.length / 2.0, pa.width / 2.0), (b.x, b.z, b.h, pb.length / 2.0, pb.width / 2.0)).is_some() {
                        c.spawn_overlap += 1; if c.spawn_ex.is_empty() { c.spawn_ex = format!("step {step} slot {i} edge {} overlaps slot {j} edge {} conn {} state {}", a.edge, b.edge, b.conn, b.state); } break;
                    }
                }
            }
            if (ev.kind == EV_CRASHED || ev.kind == EV_CONTACT) && ev.other >= 0 && (ev.slot as usize) < ev.other as usize {
                c.crash_ev += 1;
                let (i, j) = (ev.slot as usize, ev.other as usize);
                if i < prev.len() && j < prev.len() {
                    let ke = |s: &Snap| { let p = &VEHICLE_PARAMS[s.ty as usize]; 0.5 * p.m * (s.u * s.u + s.v * s.v) + 0.5 * p.iz * s.r * s.r };
                    let (k0, k1) = (ke(&prev[i]) + ke(&prev[j]), ke(&now[i]) + ke(&now[j]));
                    if k1 > k0 * 1.02 + 2000.0 { c.ke_gain += 1; c.ke_gain_max = c.ke_gain_max.max((k1 - k0) / k0.max(1.0)); }
                    let dy = (now[i].y - now[j].y).abs();
                    if dy > 1.0 { c.contact_dy += 1; if c.contact_dy_ex.is_empty() { c.contact_dy_ex = format!("step {step} slots {i}/{j} dy {dy:.2} m edges {} ({}) / {} ({})", now[i].edge, g.highway(now[i].edge as usize), now[j].edge, g.highway(now[j].edge as usize)); } }
                }
            }
        }
        // per-car kinematics
        for i in 0..now.len() {
            let (a, b) = (&now[i], if i < prev.len() { prev[i] } else { Snap::default() });
            if a.state != STATE_DRIVING && a.state != STATE_CRASHED { continue; }
            if !(a.x.is_finite() && a.z.is_finite() && a.u.is_finite() && a.y.is_finite()) { c.nan += 1; continue; }
            if b.state != a.state || b.spawn != a.spawn || (b.state != STATE_DRIVING && b.state != STATE_CRASHED) { continue; }
            c.car_steps += 1;
            let d = (a.x - b.x).hypot(a.z - b.z);
            let exp = a.u.max(b.u).hypot(a.v.abs().max(b.v.abs())) * dt;
            let cont = collided.contains(&i);
            if d > 1.5 * exp + 0.5 {
                c.tele += 1;
                if cont { c.tele_contact += 1 } else if a.conn != b.conn { c.tele_conn += 1 } else if a.edge != b.edge { c.tele_edge += 1 } else { c.tele_other += 1; if c.tele_other <= 6 { println!("  tele-other step {step} slot {i}: d {d:.2} exp {exp:.2} edge {} conn {} s {:.2}->{:.2} lat {:.2}->{:.2} u {:.2}->{:.2} hw {} lanes {} state {}", a.edge, a.conn, b.s, a.s, b.lat, a.lat, b.u, a.u, g.highway(a.edge as usize), g.e_lanes[a.edge as usize], a.state); } }
                if d - exp > c.tele_max { c.tele_max = d - exp; c.tele_ex = format!("step {step} slot {i}: moved {d:.2} m, speed allows {exp:.2} m; edge {}->{} conn {}->{} lat {:.2}->{:.2} contact {cont}", b.edge, a.edge, b.conn, a.conn, b.lat, a.lat); }
            }
            let dh = wrap(a.h - b.h);
            if !cont && dh.abs() > 0.3 && a.state == STATE_DRIVING { c.hjump += 1; if c.hjump_ex.is_empty() { c.hjump_ex = format!("step {step} slot {i}: heading jump {dh:.2} rad at {:.1} m/s, edge {}->{} conn {}->{}, yaw rate {:.2}", a.u, b.edge, a.edge, b.conn, a.conn, a.r); } }
            let dy = (a.y - b.y).abs();
            if dy > 0.5 + 0.4 * d { c.yjump += 1; if dy > c.yjump_max { c.yjump_max = dy; c.yjump_ex = format!("step {step} slot {i}: y {:.2} -> {:.2} over {d:.2} m, edge {}->{} ({}->{}), conn {}->{}", b.y, a.y, b.edge, a.edge, g.highway(b.edge as usize), g.highway(a.edge as usize), b.conn, a.conn); } }
            let p = &VEHICLE_PARAMS[a.ty as usize];
            if a.u > p.vmax + 0.5 { c.fast += 1; if c.fast == 1 { println!("  over-vmax step {step} slot {i} type {} u {:.1} grade {:.3} gas {:.2} brake {:.2}", a.ty, a.u, g.edge_grade(a.edge as usize, a.s, 0), env.sim.vs.gas[i], env.sim.vs.brake[i]); } }
            if !cont && a.state == STATE_DRIVING { let acc = (a.u - b.u).abs() / dt; if acc > 12.0 { c.acc += 1; c.acc_max = c.acc_max.max(acc); if c.acc <= 4 { println!("  accel step {step} slot {i}: u {:.2}->{:.2} v {:.2}->{:.2} edge {}->{} conn {}->{} lat {:.2}->{:.2} hr? state {}", b.u, a.u, b.v, a.v, b.edge, a.edge, b.conn, a.conn, b.lat, a.lat, a.state); } } }
            if a.state == STATE_DRIVING && !cont && a.u > 8.0 && dh.abs() < 0.3 {
                c.fast_steps += 1;
                let alat = (0.5 * (a.u + b.u)) * dh.abs() / dt; // world lateral acceleration (path curvature x speed^2)
                let tyre = (a.u * a.r).abs(); // what the tyre model produced
                let mu_g = p.mu * 9.81;
                if alat > 0.5 * 9.81 { c.alat_05 += 1; }
                if alat > mu_g * 1.05 { c.alat_over += 1; if tyre < 0.3 * mu_g { c.alat_over_kin += 1; } }
                if alat > c.alat_max { c.alat_max = alat; c.alat_ex = format!("step {step} slot {i} ({}): {:.1} m/s, world lateral {:.2} g, tyre-model u*r {:.2} g, edge {} conn {} ({})", ["sedan", "suv", "truck", "bus"][a.ty as usize], a.u, alat / 9.81, tyre / 9.81, a.edge, a.conn, g.highway(a.edge as usize)); }
            }
            // stuck
            if a.state == STATE_DRIVING { let k = (i, a.spawn.to_bits()); if a.u < 0.1 { stopped.entry(k).or_insert(env.sim.time); } else { stopped.remove(&k); } }
        }
        // overlapping bodies after the step (pass-through) and stacked passes
        if step % 5 == 0 {
            let mut grid: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
            for (i, a) in now.iter().enumerate() { if a.state == STATE_DRIVING || a.state == STATE_CRASHED { grid.entry(((a.x / 12.0).floor() as i64, (a.z / 12.0).floor() as i64)).or_default().push(i); } }
            for (&(cx, cz), list) in &grid {
                for &i in list {
                    for dx in -1..=1 { for dz in -1..=1 {
                        let Some(l2) = grid.get(&(cx + dx, cz + dz)) else { continue };
                        for &j in l2 {
                            if j <= i { continue; }
                            let (a, b) = (&now[i], &now[j]);
                            let (pa, pb) = (&VEHICLE_PARAMS[a.ty as usize], &VEHICLE_PARAMS[b.ty as usize]);
                            if physics::obb_contact((a.x, a.z, a.h, pa.length / 2.0 * 0.95, pa.width / 2.0 * 0.95), (b.x, b.z, b.h, pb.length / 2.0 * 0.95, pb.width / 2.0 * 0.95)).is_none() { continue; }
                            let dy = (a.y - b.y).abs();
                            if dy > env.sim.opts.layer_gap {
                                let bins = [3.5, 4.5, 5.5, 7.0, 1e9]; let k = bins.iter().position(|&x| dy < x).unwrap(); c.under[k] += 1;
                                if dy < 4.5 && (a.ty >= 2 || b.ty >= 2) { c.under_heavy += 1; }
                                continue;
                            }
                            if a.state == STATE_CRASHED && b.state == STATE_CRASHED { *c.overlap_by.entry("both wrecks").or_default() += 1; continue; }
                            let key = (i, j, a.spawn.to_bits() ^ b.spawn.to_bits());
                            if !seen_pairs.insert(key) { continue; }
                            c.overlap_pairs += 1;
                            let adj = { let (ea, eb) = (a.edge as usize, b.edge as usize); ea == eb || g.e_from[ea] == g.e_from[eb] || g.e_from[ea] == g.e_to[eb] || g.e_to[ea] == g.e_from[eb] || g.e_to[ea] == g.e_to[eb] };
                            let fw = g.is_shoulder_road(a.edge as usize) || g.is_shoulder_road(b.edge as usize);
                            let why = if a.edge == b.edge { "same edge (contact pass should separate)" } else if !adj && fw { "freeway, non-adjacent edges: contacts skipped" } else if adj { "adjacent edges" } else { "other edges" };
                            *c.overlap_by.entry(why).or_default() += 1;
                        }
                    } }
                }
            }
        }
        prev = now;
    }
    let tnow = env.sim.time;
    let stuck: Vec<_> = stopped.iter().filter(|(_, &t)| tnow - t > 300.0).collect();
    let el = t0.elapsed().as_secs_f64();
    let st: HashMap<_, _> = env.stats().into_iter().collect();
    println!("== R rollout: policy {policy}, {n} commuters, {steps} steps (dt {dt}), {:.1} s wall, {} car-steps; spawned {}, arrived {}, crash events {}, crashed vehicles {}, minor contacts {} ==",
        el, c.car_steps, st["spawned"], st["arrived"], st["crash_events"], st["crashed_vehicles"], st["minor_contacts"]);
    println!("non-finite pose/speed: {}", c.nan);
    println!("teleports (moved > 1.5 x speed x dt + 0.5 m): {} ({:.3} %): contact {}, connector change {}, edge hop {}, other {}; worst excess {:.2} m: {}", c.tele, 100.0 * c.tele as f64 / c.car_steps.max(1) as f64, c.tele_contact, c.tele_conn, c.tele_edge, c.tele_other, c.tele_max, c.tele_ex);
    println!("heading jumps > 0.3 rad in one step without contact: {}; first: {}", c.hjump, c.hjump_ex);
    println!("y jumps (> 0.5 m + 0.4 x distance): {}; worst {:.2} m: {}", c.yjump, c.yjump_max, c.yjump_ex);
    println!("speed > vmax + 0.5: {}; |accel| > 12 m/s^2 without contact: {} (max {:.1})", c.fast, c.acc, c.acc_max);
    println!("cornering (driving, > 8 m/s, no contact): {} car-steps; world lateral accel > 0.5 g: {} ({:.2} %); > mu g: {} ({:.3} %), of which the tyre model produced < 0.3 mu g: {}; max {:.2} g: {}",
        c.fast_steps, c.alat_05, 100.0 * c.alat_05 as f64 / c.fast_steps.max(1) as f64, c.alat_over, 100.0 * c.alat_over as f64 / c.fast_steps.max(1) as f64, c.alat_over_kin, c.alat_max / 9.81, c.alat_ex);
    println!("spawns {}: overlapping another body at spawn: {} ({})", c.spawn, c.spawn_overlap, c.spawn_ex);
    println!("bodies overlapping after a step (sampled every 5 steps, distinct pairs): {}; by kind {:?}", c.overlap_pairs, c.overlap_by);
    println!("xz-overlapping pairs separated vertically by > layer_gap {}: dy < 3.5: {}, < 4.5: {}, < 5.5: {}, < 7: {}, more: {}; with a truck/bus and dy < 4.5 (taller than the gap): {}", env.sim.opts.layer_gap, c.under[0], c.under[1], c.under[2], c.under[3], c.under[4], c.under_heavy);
    println!("pair contacts {}: kinetic energy up > 2 % over the step: {} (max +{:.0} %); contacts between cars > 1 m apart in y: {} ({})", c.crash_ev, c.ke_gain, 100.0 * c.ke_gain_max, c.contact_dy, c.contact_dy_ex);
    println!("cars stopped (< 0.1 m/s) for > 300 s at the end: {}", stuck.len());
    let vs = &env.sim.vs;
    let mut why: HashMap<String, usize> = HashMap::new();
    for (&(i, _), _) in &stuck {
        let l = vs.leader[i];
        let k = if vs.ctl_dist[i] < 30.0 && vs.ctl_state[i] >= 1 { "at a control".to_string() } else if l >= 0 && vs.state[l as usize] == STATE_CRASHED { "behind a wreck".into() } else if l >= 0 { "behind a stopped car".into() } else if l == -2 { "lane end".into() } else if l == -3 { "junction conflict".into() } else { format!("no leader (lat {:.1}, heading_rel {:.2})", vs.lat[i], vs.heading_rel[i]) };
        *why.entry(k).or_default() += 1;
    }
    println!("  stuck by situation: {:?}", why);
}

// ------------------------------------------------------------------------------------------------ targeted scenarios
fn spawn_some(env: &mut Env, k: usize, max_steps: usize) -> Vec<usize> {
    for _ in 0..max_steps {
        let v: Vec<usize> = env.driving().collect();
        if v.len() >= k { return v; }
        let s: Vec<usize> = env.driving().collect();
        for i in s { env.act(i, Action { steer: Some(0.0), steer_rate: None, gas: 0.0, brake: 1.0 }); }
        env.step(1);
    }
    env.driving().collect()
}

/// put slot i on its current edge at s with speed u (lane centre of `lane`), heading_rel hr
fn place(env: &mut Env, i: usize, s: f64, lane: i32, u: f64, hr: f64) {
    let g = env.world.clone();
    let vs = &mut env.sim.vs;
    let e = vs.edge[i] as usize;
    vs.conn[i] = -1; vs.s[i] = s as f32; vs.lat[i] = ((lane as f64 + 0.5) * g.lane_w(e, s)) as f32; vs.heading_rel[i] = hr as f32;
    vs.speed[i] = u as f32; vs.v_lat[i] = 0.0; vs.yaw_rate[i] = 0.0; vs.steer_angle[i] = 0.0; vs.steer_sl[i] = 0.0; vs.taper[i] = 0.0;
    vs.update_lanes(i, &g, e); vs.update_pose(i, &g); vs.save_pose(i);
}

pub fn scenarios(data: &str) {
    let dt = 0.05;
    // ---- S1 curves at speed: empty roads, every car held at 30 m/s (108 km/h) by the pedals, steer held at 0
    {
        let mut env = make_env(data, 3000, 3.0, 7.5, TrafficMode::Empty, dt);
        let g = env.world.clone();
        let cars = spawn_some(&mut env, 60, 4000);
        let mut prev;
        let mut maxlat: HashMap<usize, f64> = HashMap::new();
        let mut walls: HashMap<usize, u32> = HashMap::new();
        let mut turns: HashMap<usize, (u32, f64)> = HashMap::new(); // connectors with |turn| > 1.2 crossed at > 25 m/s
        for &i in &cars { let e = env.sim.vs.edge[i] as usize; let s = (env.sim.vs.s[i] as f64).min(g.e_length[e] - 1.0); let ln = env.sim.vs.lane[i].max(0) as i32; place(&mut env, i, s.max(0.0), ln, 30.0, 0.0); }
        prev = snap(&env);
        for _ in 0..1200 {
            for &i in &cars { if env.sim.vs.state[i] == STATE_DRIVING { let u = env.sim.vs.speed[i] as f64; env.act(i, Action { steer: Some(0.0), steer_rate: None, gas: ((30.0 - u) * 0.5).clamp(0.0, 1.0), brake: 0.0 }); } }
            env.step(1);
            let now = snap(&env);
            let evs = env.events();
            for ev in &evs { if ev.kind == EV_CRASHED || ev.kind == EV_CONTACT { *walls.entry(ev.slot as usize).or_default() += 1; } }
            for &i in &cars {
                let (a, b) = (&now[i], &prev[i]);
                if a.state != STATE_DRIVING || b.state != STATE_DRIVING || a.spawn != b.spawn { continue; }
                let dh = wrap(a.h - b.h);
                if dh.abs() < 0.3 { let al = 0.5 * (a.u + b.u) * dh.abs() / dt; let m = maxlat.entry(i).or_default(); if al > *m { *m = al; } }
                if b.conn >= 0 && a.conn != b.conn && b.u > 25.0 { if let Some(j) = &g.j { let tn = j.turn(b.conn as usize).abs(); if tn > 1.2 { let t = turns.entry(i).or_default(); t.0 += 1; t.1 = t.1.max(j.m_len[b.conn as usize] / tn); } } }
            }
            prev = now;
        }
        let mut ml: Vec<f64> = maxlat.values().map(|v| v / 9.81).collect(); ml.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let over = ml.iter().filter(|&&v| v > 1.0).count();
        let nturn: u32 = turns.values().map(|t| t.0).sum();
        let still = cars.iter().filter(|&&i| env.sim.vs.state[i] == STATE_DRIVING).count();
        println!("== S1 {} cars held at 108 km/h, steer 0, empty roads, 60 s ==", cars.len());
        println!("per-car max world lateral accel (g): median {:.2}, p90 {:.2}, max {:.2}; cars above 1 g: {over}; >69 deg junction turns taken at > 90 km/h: {nturn} (largest mean radius among them {:.1} m); cars with any contact/barrier event: {}; still driving at the end: {still}",
            ml.get(ml.len() / 2).copied().unwrap_or(0.0), ml.get(ml.len() * 9 / 10).copied().unwrap_or(0.0), ml.last().copied().unwrap_or(0.0), turns.values().map(|t| t.1).fold(0.0, f64::max), walls.len());
    }
    // ---- S2 parked obstacle rear-end at 50 km/h; S3 head-on (heading_rel = pi); S4 wreck sliding into an edge end; S5 boxed-in car (no reverse)
    {
        let mut env = make_env(data, 3000, 5.0, 7.5, TrafficMode::Full, dt);
        let g = env.world.clone();
        let cars = spawn_some(&mut env, 40, 4000);
        // choose cars on long edges (>= 200 m), one per edge
        let mut used = std::collections::HashSet::new();
        let long: Vec<usize> = cars.iter().copied().filter(|&i| { let e = env.sim.vs.edge[i] as usize; g.e_length[e] >= 250.0 && env.sim.vs.conn[i] < 0 && used.insert(e) }).collect();
        println!("\n== S2-S5 on real edges ({} candidate cars on edges >= 250 m) ==", long.len());
        if long.len() < 4 { println!("not enough long-edge cars"); return; }
        // S2: static (parked) obstacle 60 m ahead, car at 50 km/h, no brake
        let (a, b, c2, d) = (long[0], long[1], long[2], long[3]);
        let ea = env.sim.vs.edge[a] as usize; let la = env.sim.vs.lane[a].max(0) as i32;
        place(&mut env, a, 20.0, la, 50.0 / 3.6, 0.0);
        let obs = env.sim.place_static_obstacles(&[(ea as i32, 80.0, Some(la), None, 0)], true);
        // S3: car b and a twin facing it: move car c2 onto b's edge (copy route) facing backwards
        let eb = env.sim.vs.edge[b] as usize; let lb = env.sim.vs.lane[b].max(0) as i32;
        place(&mut env, b, 20.0, lb, 50.0 / 3.6, 0.0);
        { let vs = &mut env.sim.vs; vs.edge[c2] = eb as i32; vs.routes[c2] = vs.routes[b].clone(); vs.route_pos[c2] = vs.route_pos[b]; vs.route_len[c2] = vs.route_len[b]; vs.route_off[c2] = vs.route_off[b]; }
        place(&mut env, c2, 60.0, lb, 50.0 / 3.6, std::f64::consts::PI);
        // S4: car d becomes a wreck sliding at 25 m/s, 15 m before its edge end
        let ed = env.sim.vs.edge[d] as usize; let ld = env.sim.vs.lane[d].max(0) as i32;
        place(&mut env, d, g.e_length[ed] - 15.0, ld, 25.0, 0.0);
        env.sim.vs.state[d] = STATE_CRASHED; env.sim.vs.flags[d] |= FLAG_DISABLED;
        let mut log_d = Vec::new();
        let (mut s2, mut s3) = (String::new(), String::new());
        for k in 0..200 {
            for &i in &[a, b, c2] { if env.sim.vs.state[i] == STATE_DRIVING { let u = env.sim.vs.speed[i] as f64; env.act(i, Action { steer: Some(0.0), steer_rate: None, gas: ((50.0 / 3.6 - u) * 0.5).clamp(0.0, 1.0), brake: 0.0 }); } }
            let others: Vec<usize> = env.driving().filter(|i| ![a, b, c2].contains(i)).collect();
            for i in others { env.act(i, Action { steer: Some(0.0), steer_rate: None, gas: 0.0, brake: 1.0 }); }
            let pre_a = (env.sim.vs.speed[a], env.sim.vs.v_lat[a], env.sim.vs.s[a], env.sim.vs.lat[a], env.sim.vs.heading_rel[a], env.sim.vs.type_idx[a]);
            env.step(1);
            if s2.is_empty() && env.sim.collided_last.contains(&a) { println!("S2 debug pre-step a: speed {:.2} vlat {:.2} s {:.2} lat {:.2} hr {:.3} type {}; obstacle lat {:.2} type {}; after: speed {:.2} vlat {:.2} yaw {:.2}", pre_a.0, pre_a.1, pre_a.2, pre_a.3, pre_a.4, pre_a.5, env.sim.vs.lat[obs[0]], env.sim.vs.type_idx[obs[0]], env.sim.vs.speed[a], env.sim.vs.v_lat[a], env.sim.vs.yaw_rate[a]); }
            for ev in env.events() {
                if (ev.kind == EV_CRASHED || ev.kind == EV_CONTACT) && (ev.slot as usize == a) && s2.is_empty() { s2 = format!("step {k}: car dv {:.1} km/h (other {:?} dv {:.1}), kind {}", ev.dv_kmh, ev.other, ev.other_dv_kmh, if ev.kind == EV_CRASHED { "CRASHED" } else { "CONTACT" }); }
                if (ev.kind == EV_CRASHED || ev.kind == EV_CONTACT) && (ev.slot as usize == b || ev.slot as usize == c2) && s3.len() < 400 { s3 += &format!("step {k}: slot {} dv {:.1} km/h other {} dv {:.1}; ", ev.slot, ev.dv_kmh, ev.other, ev.other_dv_kmh); }
            }
            let vs = &env.sim.vs;
            if k < 40 { log_d.push(format!("{:.2}/{:.1}/{:.0}", vs.s[d] as f64 - g.e_length[ed], vs.speed[d], vs.x[d])); }
        }
        let vs = &env.sim.vs;
        println!("S2 sedan 50 km/h into a parked (static) sedan: {s2}; obstacle slot {:?} moved? s {:.2} (placed 80)", obs, vs.s[obs[0]]);
        println!("S3 head-on 50 vs 50 (same edge, one facing back; types {} vs {}): {s3}", vs.type_idx[b], vs.type_idx[c2]); println!("{s3}; after: speeds {:.2} / {:.2} m/s, states {} / {}", vs.speed[b], vs.speed[c2], vs.state[b], vs.state[c2]);
        println!("S4 wreck sliding at 90 km/h, 15 m before the end of edge {ed}: (s - length / speed / x) per step: {}", log_d.join(" "));
        // S5: stopped car 1 m behind a parked car, tries full lock + gas for 60 s
        let cands: Vec<usize> = env.driving().filter(|&i| { let e = env.sim.vs.edge[i] as usize; g.e_length[e] > 120.0 && env.sim.vs.conn[i] < 0 && i != a && i != b && i != c2 }).collect();
        if let Some(&q) = cands.first() {
            let eq = env.sim.vs.edge[q] as usize; let lq = env.sim.vs.lane[q].max(0) as i32;
            place(&mut env, q, 30.0, lq, 0.0, 0.0);
            let hl = env.sim.vs.hl[q] as f64;
            env.sim.place_static_obstacles(&[(eq as i32, 30.0 + 2.0 * hl + 1.0, Some(lq), None, 0)], true);
            let s0 = env.sim.vs.s[q];
            for k in 0..1200 {
                let st = env.sim.vs.state[q]; if st != STATE_DRIVING { println!("S5: state {st} at step {k}"); break; }
                env.act(q, Action { steer: Some(if k < 600 { 1.0 } else { -1.0 }), steer_rate: None, gas: 0.3, brake: 0.0 });
                let others: Vec<usize> = env.driving().filter(|&i| i != q).collect(); for i in others { env.act(i, Action { steer: Some(0.0), steer_rate: None, gas: 0.0, brake: 1.0 }); }
                env.step(1);
            }
            let vs = &env.sim.vs;
            println!("S5 car stopped 1 m behind a parked car, full lock + gas 0.3 for 60 s (lanes {}): moved {:.2} m along, lat {:.2}, heading_rel {:.2}, speed {:.2}, state {}, contacts so far {}",
                g.e_lanes[eq], vs.s[q] - s0, vs.lat[q], vs.heading_rel[q], vs.speed[q], vs.state[q], env.sim.counters.touches + env.sim.counters.minor_contacts);
        }
    
        // S6: barrier strike while already touching the barrier (sustained contact) vs a fresh strike
        let free: Vec<usize> = env.driving().filter(|&i| { let e = env.sim.vs.edge[i] as usize; g.e_length[e] > 300.0 && env.sim.vs.conn[i] < 0 && ![a, b, c2].contains(&i) }).collect();
        let mut used6 = std::collections::HashSet::new();
        let pick: Vec<usize> = free.into_iter().filter(|&i| used6.insert(env.sim.vs.edge[i])).take(3).collect();
        if pick.len() >= 2 {
            for (k, &q) in pick.iter().take(2).enumerate() {
                place(&mut env, q, 40.0, 0, 20.0, 0.0);
                let mut log = String::new();
                for st in 0..60 {
                    if env.sim.vs.state[q] != STATE_DRIVING { break; }
                    let hard = st >= 40;
                    // k = 0: drift into the barrier at 0.3 m/s from step 0 (sustained contact), then hit it at 6 m/s
                    // k = 1: stay off the barrier, then hit it at 6 m/s
                    let vl = if hard { -6.0 } else if k == 0 { -0.3 } else { 0.0 };
                    if !(hard && st > 40) { env.sim.vs.v_lat[q] = vl as f32; }
                    env.act(q, Action { steer: Some(0.0), steer_rate: None, gas: 0.2, brake: 0.0 });
                    let others: Vec<usize> = env.driving().filter(|&i| i != q).collect(); for i in others { env.act(i, Action { steer: Some(0.0), steer_rate: None, gas: 0.0, brake: 1.0 }); }
                    let (t0, m0, c0) = (env.sim.counters.touches, env.sim.counters.minor_contacts, env.sim.counters.crashes);
                    let vlat0 = env.sim.vs.v_lat[q];
                    env.step(1);
                    let evs: Vec<String> = env.events().iter().filter(|e| e.slot as usize == q && (e.kind == EV_CRASHED || e.kind == EV_CONTACT)).map(|e| format!("{}:{:.1}km/h", if e.kind == EV_CRASHED { "CRASHED" } else { "CONTACT" }, e.dv_kmh)).collect();
                    let dvl = (env.sim.vs.v_lat[q] - vlat0) as f64 * 3.6;
                    if st >= 38 || (env.sim.counters.touches + env.sim.counters.minor_contacts + env.sim.counters.crashes) != t0 + m0 + c0 { if log.len() < 900 { log += &format!("[{st} lat {:.2} vlat {:.1}->{:.1} (dv {:.1} km/h) ev {:?} touches+{}] ", env.sim.vs.lat[q], vlat0, env.sim.vs.v_lat[q], dvl.abs(), evs, env.sim.counters.touches - t0); } }
                }
                println!("S6 {} ({}, lanes {}): state {} | {log}", if k == 0 { "sustained barrier contact, then a 6 m/s lateral strike" } else { "fresh 6 m/s lateral strike" }, g.highway(env.sim.vs.edge[q] as usize), g.e_lanes[env.sim.vs.edge[q] as usize], env.sim.vs.state[q]);
            }
        }
}
}
