//! Physics audit scenarios (rust/docs/physics-audit.md). Measures what the sim actually does and prints it next to
//! real-world reference numbers. Read-only with respect to the sim: no model code is changed.
//!
//!   cargo run --release -p bayenv-sim --example physics_audit -- dyn                 # vehicle dynamics (no world)
//!   cargo run --release -p bayenv-sim --example physics_audit -- contact             # rigid-body contacts (no world)
//!   cargo run --release -p bayenv-sim --example physics_audit -- world <data>        # world geometry scan
//!   cargo run --release -p bayenv-sim --example physics_audit -- sim <data> [n] [steps] [policy]   # scenarios + rollout scan
use bayenv_sim::physics::*;

const KMH: f64 = 1.0 / 3.6;
const NAMES: [&str; 4] = ["sedan", "suv", "truck", "bus"];

fn step(st: &mut DynState, sr: f64, gas: f64, brake: f64, p: &VehicleParams, dt: f64, grade: f64, locked: bool, surf: Option<Surf>) {
    step_dynamics(st, sr, gas, brake, p, dt, grade, locked, true, surf);
}
fn st_u(u: f64) -> DynState { DynState { u, ..Default::default() } }

/// pedalsFor (dynamics.js), used only to hold a speed in the scenarios
fn pedals_for(p: &VehicleParams, accel: f64, u: f64, grade: f64) -> (f64, f64) {
    let need = p.m * accel + resistance(p, u, grade, 1.0);
    if need >= 0.0 { ((need / p.f0.min(p.pmax / u.max(0.1))).min(1.0), 0.0) } else { (0.0, (-need / (p.m * p.brake_decel)).min(1.0)) }
}

fn surf(name: &str) -> Option<Surf> {
    // surface.js factors: pk = muPeakDry / 0.85, sk = pk * round2(muSlide / muPeak), crr = crr / 0.013
    let (mp, ms, crr): (f64, f64, f64) = match name {
        "asphalt" => (0.85, 0.68, 0.013), "grass" => (0.45, 0.40, 0.100), "gravel" => (0.60, 0.55, 0.025),
        "wet-asphalt (doc-only)" => (0.60, 0.50, 0.013), _ => return None,
    };
    let pk = mp / 0.85;
    Some(Surf { pk, sk: pk * ((ms / mp) * 100.0).round() / 100.0, crr: crr / 0.013 })
}

fn dyn_audit() {
    let dt = 0.05;
    println!("== A1 acceleration (gas = 1, level, dry) ==");
    println!("type   0-50km/h  0-100km/h  0-60mph  vmax(km/h)  top speed reached in 120 s");
    for (k, p) in VEHICLE_PARAMS.iter().enumerate() {
        let mut st = DynState::default();
        let (mut t50, mut t100, mut t60) = (f64::NAN, f64::NAN, f64::NAN);
        let mut t = 0.0;
        while t < 120.0 {
            step(&mut st, 0.0, 1.0, 0.0, p, dt, 0.0, false, None); t += dt;
            if t50.is_nan() && st.u >= 50.0 * KMH { t50 = t; }
            if t60.is_nan() && st.u >= 26.8224 { t60 = t; }
            if t100.is_nan() && st.u >= 100.0 * KMH { t100 = t; }
        }
        println!("{:6} {:8.1}  {:9.1}  {:7.1}  {:9.0}  {:6.1}", NAMES[k], t50, t100, t60, p.vmax * 3.6, st.u * 3.6);
    }

    println!("\n== A2 braking from steady speed, brake = 1, no reaction time (dry asphalt unless stated) ==");
    println!("type   surface              mode      30km/h  60km/h  100km/h  (m)   mean decel @100 (g)");
    for (k, p) in VEHICLE_PARAMS.iter().enumerate() {
        for sname in ["asphalt", "grass", "gravel"] {
            for locked in [false, true] {
                if sname != "asphalt" && locked { continue; }
                let mut row = Vec::new();
                let mut dec100 = 0.0;
                for v in [30.0, 60.0, 100.0] {
                    let mut st = st_u(v * KMH);
                    let mut t = 0.0;
                    while st.u > 0.05 && t < 60.0 { step(&mut st, 0.0, 0.0, 1.0, p, 0.01, 0.0, locked, surf(sname)); t += 0.01; }
                    row.push(st.s);
                    if v == 100.0 { dec100 = (v * KMH) / t / 9.81; }
                }
                println!("{:6} {:20} {:8} {:7.1} {:7.1} {:8.1}        {:.2}", NAMES[k], sname, if locked { "locked" } else { "ABS" }, row[0], row[1], row[2], dec100);
            }
        }
    }

    println!("\n== A3 steady cornering (skid pad): position-mode steer held, speed held by pedals ==");
    println!("type   speed   max steady ay (g)  at steer cmd  beta(deg)  spun?   static stability factor t/2h (rollover)");
    let track = [1.58, 1.65, 2.0, 2.1];
    for (k, p) in VEHICLE_PARAMS.iter().enumerate() {
        for v in [15.0, 25.0] {
            let (mut best, mut bcmd, mut bbeta, mut spun_any) = (0.0f64, 0.0, 0.0, false);
            for i in 1..=40 {
                let cmd = i as f64 * 0.025;
                let mut st = st_u(v);
                let mut spun = false;
                let mut ay_sum = 0.0; let mut n = 0;
                for s in 0..400 { // 20 s
                    let (g, b) = pedals_for(p, (v - st.u) * 1.0, st.u, 0.0);
                    step(&mut st, cmd, g, b, p, dt, 0.0, false, None);
                    if st.v.atan2(st.u.max(0.1)).abs() > 0.35 || st.u < 0.5 * v { spun = true; }
                    if s >= 300 { ay_sum += st.u * st.r; n += 1; }
                }
                if spun { spun_any = true; continue; }
                let ay = (ay_sum / n as f64).abs() / 9.81;
                if ay > best { best = ay; bcmd = cmd; bbeta = st.v.atan2(st.u).to_degrees(); }
            }
            println!("{:6} {:4.0}m/s {:12.2}  {:14.3} {:9.1}   {:5}  {:.2}", NAMES[k], v, best, bcmd, bbeta, spun_any, track[k] / (2.0 * p.hcg));
        }
    }

    println!("\n== A4 grades (gas = 1 from rest; coast; hold) ==");
    println!("type   grade  t 0-30km/h  speed after 60 s (km/h)  terminal coast downhill (km/h)  stopped, brake 0: rolls?  stopped uphill brake 0: rolls back?");
    for (k, p) in VEHICLE_PARAMS.iter().enumerate() {
        for gr in [0.06, 0.15, 0.25, 0.35] {
            let mut st = DynState::default(); let mut t = 0.0; let mut t30 = f64::NAN;
            while t < 60.0 { step(&mut st, 0.0, 1.0, 0.0, p, dt, gr, false, None); t += dt; if t30.is_nan() && st.u >= 30.0 * KMH { t30 = t; } }
            let up60 = st.u * 3.6;
            let mut c = DynState::default(); for _ in 0..2400 { step(&mut c, 0.0, 0.0, 0.0, p, dt, -gr, false, None); }
            let mut h = DynState::default(); for _ in 0..200 { step(&mut h, 0.0, 0.0, 0.0, p, dt, -gr, false, None); }
            let mut ub = DynState::default(); for _ in 0..200 { step(&mut ub, 0.0, 0.0, 0.0, p, dt, gr, false, None); }
            println!("{:6} {:4.0}%  {:9.1}  {:15.1}  {:22.1}  {:20}  {} (u = {:.3}, s = {:.3})", NAMES[k], gr * 100.0, t30, up60, c.u * 3.6,
                if h.u > 0.1 { format!("yes, {:.1} km/h after 10 s", h.u * 3.6) } else { "no".into() }, if ub.s < -1e-6 { "yes" } else { "no" }, ub.u, ub.s);
        }
    }
    // braking downhill
    println!("type   braking 60->0 km/h on -15% (m)  on -35% (m)  on level (m)");
    for (k, p) in VEHICLE_PARAMS.iter().enumerate() {
        let mut out = Vec::new();
        for gr in [-0.15, -0.35, 0.0] {
            let mut st = st_u(60.0 * KMH); let mut t = 0.0;
            while st.u > 0.05 && t < 60.0 { step(&mut st, 0.0, 0.0, 1.0, p, 0.01, gr, false, None); t += 0.01; }
            out.push(if st.u > 0.05 { f64::INFINITY } else { st.s });
        }
        println!("{:6} {:20.1} {:12.1} {:12.1}", NAMES[k], out[0], out[1], out[2]);
    }

    println!("\n== A5 very low speed: creep and stop ==");
    for (k, p) in VEHICLE_PARAMS.iter().enumerate() {
        let mut st = DynState::default(); for _ in 0..600 { step(&mut st, 0.0, 0.0, 0.0, p, dt, 0.0, false, None); }
        let idle = st.u;
        let mut st2 = DynState::default(); for _ in 0..600 { step(&mut st2, 0.0, 0.01, 0.0, p, dt, 0.0, false, None); }
        let mut st3 = st_u(1.0); let mut t = 0.0; while st3.u > 0.0 && t < 100.0 { step(&mut st3, 0.0, 0.0, 0.0, p, dt, 0.0, false, None); t += dt; }
        println!("{:6} idle (gas 0, brake 0) after 30 s: {:.3} m/s; gas 0.01 after 30 s: {:.2} m/s; coast 1 m/s -> stop in {:.1} s / {:.2} m", NAMES[k], idle, st2.u, t, st3.s);
    }
    // minimum turning radius at walking pace (kinematic regime) and at 3 m/s (dynamic)
    for (k, p) in VEHICLE_PARAMS.iter().enumerate() {
        for v in [1.0, 3.0] {
            let mut st = st_u(v);
            for _ in 0..200 { let (g, b) = pedals_for(p, (v - st.u) * 1.0, st.u, 0.0); step(&mut st, 1.0, g, b, p, dt, 0.0, false, None); }
            let r = st.u / st.r.abs().max(1e-9);
            // curb-to-curb diameter of the outer front corner around the turn centre (rear axle line)
            let ov = (p.length - p.l) * 0.5; // assume symmetric overhang
            let d = 2.0 * ((r + p.width / 2.0).powi(2) + (p.b + p.a + ov).powi(2)).sqrt();
            println!("{:6} full lock at {:.0} m/s: path radius (CG) {:.2} m, approx. outer-corner turning circle {:.1} m", NAMES[k], v, r, d);
        }
    }

    println!("residual creep with the brake pedal held (level road): stop from v with brake b, then hold the pedal 60 s");
    for (k, p) in VEHICLE_PARAMS.iter().enumerate() {
        let mut worst = (0.0f64, 0.0f64, 0.0f64, 0.0f64); let mut creeping = 0; let mut total = 0;
        for vi in 0..200 { for &bk in &[0.3f64, 1.0] {
            let v = 2.0 + vi as f64 * 0.137; total += 1;
            let mut st = st_u(v); let mut t = 0.0;
            while st.u > 0.05 && t < 30.0 { step(&mut st, 0.0, 0.0, bk, p, dt, 0.0, false, None); t += dt; }
            for _ in 0..20 { step(&mut st, 0.0, 0.0, bk, p, dt, 0.0, false, None); }
            let s0 = st.s; for _ in 0..1200 { step(&mut st, 0.0, 0.0, bk, p, dt, 0.0, false, None); }
            if st.u > 0.0 { creeping += 1; } if st.s - s0 > worst.0 { worst = (st.s - s0, st.u, v, bk); }
        } }
        println!("{:6} {creeping}/{total} stops end with u > 0 under the held brake; worst: {:.2} m crept in 60 s at u = {:.4} m/s (from {:.1} m/s, brake {})", NAMES[k], worst.0, worst.1, worst.2, worst.3);
    }

    println!("\n== A6 high-speed stability ==");
    for (k, p) in VEHICLE_PARAMS.iter().enumerate() {
        let v = p.vmax * 0.98;
        // step steer of 2 % lock, then release
        let mut st = st_u(v); let mut rmax: f64 = 0.0; let mut rend = 0.0;
        for i in 0..400 { let cmd = if i < 40 { 0.02 } else { 0.0 }; let (g, b) = pedals_for(p, (v - st.u), st.u, 0.0); step(&mut st, cmd, g, b, p, dt, 0.0, false, None); rmax = rmax.max(st.r.abs()); if i == 399 { rend = st.r; } }
        // full-lock swerve: does it spin?
        let mut sw = st_u(v); let mut spun = false; let mut aymax: f64 = 0.0;
        for i in 0..200 { let cmd = if i < 20 { 1.0 } else if i < 40 { -1.0 } else { 0.0 }; step(&mut sw, cmd, 0.0, 0.0, p, dt, 0.0, false, None); aymax = aymax.max((sw.u * sw.r).abs()); if sw.v.atan2(sw.u.max(0.1)).abs() > 0.5 { spun = true; } }
        // lift-off + brake in a 0.5 g curve (EBD/ESC check)
        let mut cb = st_u(25.0); let mut spun2 = false;
        for i in 0..400 { let (g, b) = if i < 200 { pedals_for(p, 25.0 - cb.u, cb.u, 0.0) } else { (0.0, 1.0) }; step(&mut cb, 0.12, g, b, p, dt, 0.0, false, None); if cb.v.atan2(cb.u.max(0.1)).abs() > 0.5 { spun2 = true; } }
        println!("{:6} @{:.0} km/h: 2%-lock 2 s step -> peak r {:.3} rad/s, r after 18 s {:.2e}; full-lock swerve: peak ay {:.2} g spun={}; brake=1 in curve spun={}", NAMES[k], v * 3.6, rmax, rend, aymax / 9.81, spun, spun2);
    }

    println!("\n== A7 numerics ==");
    let p = &VEHICLE_PARAMS[0];
    let mut a = st_u(20.0); step(&mut a, 0.0, 0.0, 0.0, p, 0.0, 0.0, false, None);
    println!("dt = 0: u {} ax {} s {}", a.u, a.ax, a.s);
    let mut b = st_u(20.0); step(&mut b, f64::NAN, f64::NAN, f64::NAN, p, dt, 0.0, false, None);
    println!("NaN controls: u {:.3} steer {} (NaN -> 0)", b.u, b.steer);
    let mut c = st_u(20.0); step(&mut c, f64::INFINITY, f64::INFINITY, 0.0, p, dt, 0.0, false, None);
    println!("inf controls: u {:.3} steer {:.3}", c.u, c.steer);
    let mut d = DynState { u: f64::NAN, ..Default::default() }; step(&mut d, 0.0, 0.5, 0.0, p, dt, 0.0, false, None);
    println!("NaN state u: after step u {} s {} (sticky)", d.u, d.s);
    let mut e = st_u(20.0); let t0 = std::time::Instant::now(); step(&mut e, 0.0, 0.0, 0.0, p, 10.0, 0.0, false, None);
    println!("dt = 10 s: u {:.3} (exact coast integrates in 1000 sub-steps, {:.2} ms)", e.u, t0.elapsed().as_secs_f64() * 1e3);
    let mut f = st_u(20.0); step(&mut f, 0.0, 0.0, 0.0, p, -0.05, 0.0, false, None);
    println!("dt = -0.05: u {:.4} s {:.4} (negative dt runs backwards, no guard)", f.u, f.s);
    let mut g = st_u(20.0); step(&mut g, 0.0, 0.0, 0.0, p, dt, 1e6, false, None);
    println!("grade 1e6 (vertical): u {:.3}  (sim clamps grade at 0.35)", g.u);
    // tyre past the peak: lateral force at large slip
    println!("tyre: Fy/Fymax at slip 2, 5, 10, 20, 45 deg = {}", [2.0f64, 5.0, 10.0, 20.0, 45.0].iter().map(|deg| { let fy = (p.cf * deg.to_radians() / (p.mu * p.m * 9.81 * p.b / p.l)).tanh(); format!("{fy:.3}") }).collect::<Vec<_>>().join(", "));

    println!("\n== A8 locked-wheel wreck slide ==");
    for (k, p) in VEHICLE_PARAMS.iter().enumerate() {
        for v in [50.0, 100.0] {
            let mut st = st_u(v * KMH); let mut t = 0.0;
            while (st.u > 0.0 || st.v.abs() > 0.0) && t < 60.0 { step(&mut st, 0.0, 0.0, 0.0, p, dt, 0.0, true, surf("asphalt")); t += dt; }
            println!("{:6} locked slide from {:.0} km/h: {:.1} m in {:.1} s", NAMES[k], v, st.s, t);
        }
    }
}

// ------------------------------------------------------------------------------------------------ contacts
fn body(m: f64, iz: f64, p: &VehicleParams, x: f64, z: f64, yaw: f64, vx: f64, vz: f64, w: f64, fixed: bool) -> Body {
    Body { x, z, yaw, vx, vz, w, hl: p.length * 0.5, hw: p.width * 0.5, inv_mass: if fixed { 0.0 } else { 1.0 / m }, inv_inertia: if fixed { 0.0 } else { 1.0 / iz }, x0: x, z0: z }
}
fn ke(b: &Body, m: f64, iz: f64) -> f64 { if b.inv_mass == 0.0 { 0.0 } else { 0.5 * m * (b.vx * b.vx + b.vz * b.vz) + 0.5 * iz * b.w * b.w } }
/// body-frame forward speed after write_back (contacts.rs write_back clamps u < 0 to 0)
fn u_body(b: &Body) -> (f64, f64) { let (fx, fz) = (b.yaw.cos(), b.yaw.sin()); (b.vx * fx + b.vz * fz, b.vx * fz - b.vz * fx) }

/// Approach A toward B until the boxes touch (swept), then resolve like Sim::contacts does. Returns (dvA, dvB) km/h,
/// KE before / after (incl. the write_back clamp of backward speed), momentum before / after.
fn crash(pa: &VehicleParams, pb: &VehicleParams, mut a: Body, mut b: Body, label: &str) {
    let (ma, ia, mb, ib) = (pa.m, pa.iz, pb.m, pb.iz);
    // advance in 1 ms steps until overlap
    let mut c = None;
    for _ in 0..200000 {
        a.x += a.vx * 1e-3; a.z += a.vz * 1e-3; a.yaw += a.w * 1e-3;
        b.x += b.vx * 1e-3; b.z += b.vz * 1e-3; b.yaw += b.w * 1e-3;
        if let Some(k) = obb_contact((a.x, a.z, a.yaw, a.hl, a.hw), (b.x, b.z, b.yaw, b.hl, b.hw)) { c = Some(k); break; }
    }
    let Some(c) = c else { println!("{label}: no contact"); return };
    let e0 = ke(&a, ma, ia) + ke(&b, mb, ib);
    let p0 = (if a.inv_mass > 0.0 { ma } else { 0.0 } * a.vx + if b.inv_mass > 0.0 { mb } else { 0.0 } * b.vx, if a.inv_mass > 0.0 { ma } else { 0.0 } * a.vz + if b.inv_mass > 0.0 { mb } else { 0.0 } * b.vz);
    let r = resolve_contact(&mut a, &mut b, &c, RESTITUTION, FRICTION);
    let e1 = ke(&a, ma, ia) + ke(&b, mb, ib);
    let (ua, va) = u_body(&a); let (ub, vb) = u_body(&b);
    let clamp_loss = |u: f64, m: f64, inv: f64| if inv > 0.0 && u < 0.0 { 0.5 * m * u * u } else { 0.0 };
    let e2 = e1 - clamp_loss(ua, ma, a.inv_mass) - clamp_loss(ub, mb, b.inv_mass);
    let p1 = (if a.inv_mass > 0.0 { ma } else { 0.0 } * a.vx + if b.inv_mass > 0.0 { mb } else { 0.0 } * b.vx, if a.inv_mass > 0.0 { ma } else { 0.0 } * a.vz + if b.inv_mass > 0.0 { mb } else { 0.0 } * b.vz);
    println!("{label:52} dvA {:5.1} dvB {:5.1} km/h | KE {:7.0} -> {:7.0} kJ ({:+5.1}%) after u<0 clamp {:7.0} kJ | |p| {:6.0} -> {:6.0} | u_after A {:5.1} B {:5.1} v_lat A {:4.1} B {:4.1} m/s, yaw rate A {:5.2} B {:5.2}",
        r.dva * 3.6, r.dvb * 3.6, e0 / 1e3, e1 / 1e3, 100.0 * (e1 - e0) / e0.max(1e-9), e2 / 1e3, p0.0.hypot(p0.1), p1.0.hypot(p1.1), ua, ub, va, vb, a.w, b.w);
}

fn contact_audit() {
    let s = &VEHICLE_PARAMS[0]; let t = &VEHICLE_PARAMS[2]; let bus = &VEHICLE_PARAMS[3];
    let v50 = 50.0 * KMH;
    println!("== B1 canonical crashes (resolve_contact, e = {RESTITUTION}, mu = {FRICTION}) ==");
    println!("real reference: equal-mass collinear plastic impact (e ~ 0.1) -> dv = (1+e)/2 x closing; into a rigid barrier dv = (1+e) x v");
    crash(s, s, body(s.m, s.iz, s, -10.0, 0.0, 0.0, v50, 0.0, 0.0, false), body(s.m, s.iz, s, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, false), "sedan 50 km/h rear-ends STOPPED (driving) sedan");
    crash(s, s, body(s.m, s.iz, s, -10.0, 0.0, 0.0, v50, 0.0, 0.0, false), body(s.m, s.iz, s, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, true), "sedan 50 km/h rear-ends stopped WRECK/parked (fixed)");
    crash(s, s, body(s.m, s.iz, s, -10.0, 0.0, 0.0, v50, 0.0, 0.0, false), body(s.m, s.iz, s, 0.0, 0.0, std::f64::consts::PI, -v50, 0.0, 0.0, false), "head-on sedan 50 vs sedan 50 (aligned)");
    crash(s, s, body(s.m, s.iz, s, -10.0, 0.8, 0.0, v50, 0.0, 0.0, false), body(s.m, s.iz, s, 0.0, -0.8, std::f64::consts::PI, -v50, 0.0, 0.0, false), "head-on sedan 50 vs sedan 50, 1.6 m offset");
    crash(t, s, body(t.m, t.iz, t, -12.0, 0.0, 0.0, v50, 0.0, 0.0, false), body(s.m, s.iz, s, 0.0, 0.0, std::f64::consts::PI, -v50, 0.0, 0.0, false), "head-on truck 50 vs sedan 50");
    crash(t, s, body(t.m, t.iz, t, -12.0, 0.0, 0.0, v50, 0.0, 0.0, false), body(s.m, s.iz, s, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, false), "truck 50 rear-ends stopped sedan");
    crash(s, s, body(s.m, s.iz, s, -8.0, 0.0, 0.0, v50, 0.0, 0.0, false), body(s.m, s.iz, s, 0.0, 0.0, std::f64::consts::FRAC_PI_2, 0.0, 0.0, 0.0, false), "T-bone sedan 50 into stationary sedan side (centre)");
    crash(s, s, body(s.m, s.iz, s, -8.0, 1.5, 0.0, v50, 0.0, 0.0, false), body(s.m, s.iz, s, 0.0, 0.0, std::f64::consts::FRAC_PI_2, 0.0, 0.0, 0.0, false), "T-bone sedan 50 at the rear quarter (1.5 m off)");
    crash(s, s, body(s.m, s.iz, s, 0.0, -1.9, 0.0, 30.0, 0.0, 0.0, false), body(s.m, s.iz, s, 0.0, 0.0, 0.0, 30.0, 0.0, 0.0, false), "sideswipe (no closing)");
    crash(s, s, body(s.m, s.iz, s, -2.0, -2.0, 0.08, 30.0, 2.4, 0.0, false), body(s.m, s.iz, s, 0.0, 0.0, 0.0, 28.0, 0.0, 0.0, false), "sideswipe lane change 108 km/h, 2.4 m/s lateral");
    crash(bus, s, body(bus.m, bus.iz, bus, -12.0, 0.0, 0.0, v50, 0.0, 0.0, false), body(s.m, s.iz, s, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, true), "bus 50 into a stopped sedan WRECK (fixed)");
    crash(s, s, body(s.m, s.iz, s, -10.0, 0.0, 0.0, 100.0 * KMH, 0.0, 0.0, false), body(s.m, s.iz, s, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, true), "sedan 100 km/h into a stopped WRECK (fixed)");

    println!("\n== B2 energy check: random SHALLOW contacts (depth < 5 cm, as the swept test finds them), resolve_contact ==");
    for (label, mu) in [("mu 0.5 (sim)", FRICTION), ("mu 0 (no friction impulse)", 0.0)] {
        let mut rng = bayenv_sim::rng::Rng::new(7.0);
        let (mut gains, mut worst, mut worst_rel, mut n, mut big) = (0usize, 0.0f64, 0.0f64, 0usize, 0usize);
        let mut worst_case = String::new();
        let mut tries = 0;
        while n < 100000 && tries < 50_000_000 {
            tries += 1;
            let pa = &VEHICLE_PARAMS[(rng.random() * 4.0) as usize % 4]; let pb = &VEHICLE_PARAMS[(rng.random() * 4.0) as usize % 4];
            let mut a = body(pa.m, pa.iz, pa, 0.0, 0.0, rng.random() * 6.283, (rng.random() - 0.5) * 40.0, (rng.random() - 0.5) * 40.0, (rng.random() - 0.5) * 2.0, false);
            let ang = rng.random() * 6.283; let rr = 1.0 + rng.random() * 12.0;
            let mut b = body(pb.m, pb.iz, pb, rr * ang.cos(), rr * ang.sin(), rng.random() * 6.283, (rng.random() - 0.5) * 40.0, (rng.random() - 0.5) * 40.0, (rng.random() - 0.5) * 2.0, rng.random() < 0.1);
            if b.inv_mass == 0.0 { b.vx = 0.0; b.vz = 0.0; b.w = 0.0; } // the sim fixes only bodies at rest
            let Some(c) = obb_contact((a.x, a.z, a.yaw, a.hl, a.hw), (b.x, b.z, b.yaw, b.hl, b.hw)) else { continue };
            if c.depth > 0.05 { continue; }
            n += 1;
            let e0 = ke(&a, pa.m, pa.iz) + ke(&b, pb.m, pb.iz);
            let r = resolve_contact(&mut a, &mut b, &c, RESTITUTION, mu);
            if r.jn <= 0.0 { continue; }
            let e1 = ke(&a, pa.m, pa.iz) + ke(&b, pb.m, pb.iz);
            if e1 > e0 * (1.0 + 1e-9) + 1e-6 {
                gains += 1; if (e1 - e0) / e0 > 0.05 { big += 1; }
                if (e1 - e0) / e0 > worst_rel { worst = e1 - e0; worst_rel = (e1 - e0) / e0; worst_case = format!("jn {:.0} jt {:.0} dvA {:.1} dvB {:.1} km/h, closing {:.1} m/s", r.jn, r.jt, r.dva * 3.6, r.dvb * 3.6, r.closing); }
            }
        }
        println!("{label}: {n} shallow contacts; {gains} gained kinetic energy ({:.2} %), {big} by > 5 %; worst +{:.1} kJ (+{:.1} %): {worst_case}", 100.0 * gains as f64 / n as f64, worst / 1e3, 100.0 * worst_rel);
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    match a.get(1).map(|s| s.as_str()).unwrap_or("dyn") {
        "dyn" => dyn_audit(),
        "contact" => contact_audit(),
        "world" => world_audit::world(&a[2]),
        "sim" => world_audit::sim(&a[2], a.get(3).map(|s| s.parse().unwrap()).unwrap_or(3000), a.get(4).map(|s| s.parse().unwrap()).unwrap_or(2400), a.get(5).map(|s| s.as_str()).unwrap_or("keep")),
        "scen" => world_audit::scenarios(&a[2]),
        x => eprintln!("unknown mode {x}"),
    }
}


mod world_audit;
