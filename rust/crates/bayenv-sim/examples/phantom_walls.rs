//! Phantom-wall investigation (west SF high-speed barrier wrecks).
//!   cargo run --release -p bayenv-sim --example phantom_walls -- <data> edges
//!   cargo run --release -p bayenv-sim --example phantom_walls -- <data> run <js|default> [n] [seed] [minutes] [steer_noise]
use bayenv_sim::env::{Env, EnvConfig};
use bayenv_sim::population_gen::{generate_population, PopConfig, Region};
use bayenv_sim::sim::{Action, SimOpts, TrafficMode};
use bayenv_sim::vehicle::STATE_DRIVING;
use bayenv_sim::world::World;
use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;

const BOX: (f64, f64, f64, f64) = (-28400.0, -26200.0, -25700.0, -22400.0);
fn in_box(x: f64, z: f64) -> bool { x >= BOX.0 && x <= BOX.1 && z >= BOX.2 && z <= BOX.3 }

fn jm_min(a: f64, b: f64) -> f64 { if a < b { a } else { b } }

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = a[1].clone();
    let mode = a[2].clone();
    let bw = bayenv_world::World::load_with(Path::new(&data), &bayenv_world::LoadOptions { dem: true, ..Default::default() }).expect("world");
    let dem = bw.dem.clone();
    let world = Arc::new(World::from_bayenv(&bw));
    let g = &*world;
    match mode.as_str() {
        "edges" => {
            let dem = dem.expect("dem");
            let mut n = 0;
            for e in 0..g.edge_count {
                let v0 = g.v_start[e] as usize; let v1 = g.v_start[e + 1] as usize;
                if !(v0..v1).any(|k| in_box(g.px[k], g.pz[k])) { continue; }
                n += 1;
                let len = g.e_length[e];
                // sample bounds / heights
                let (mut min_w, mut min_w_s) = (f64::INFINITY, 0.0);
                let (mut worst_dy, mut worst_dy_s) = (0.0f64, 0.0);
                let mut s = 0.0;
                while s <= len {
                    let w = g.lane_w(e, s);
                    let (pad_r, pad_l) = if let Some(lg) = &g.lg { let bd = lg.bounds(e, s); (bd.r - bd.right, bd.r - bd.left) } else { (0.0, g.e_lanes[e] as f64 * w) };
                    let width = pad_l - pad_r;
                    if width < min_w { min_w = width; min_w_s = s; }
                    let p = g.edge_point(e, s, 0.0, 0);
                    let h = dem.height_at(p.x, p.z);
                    if (p.y - h).abs() > worst_dy.abs() { worst_dy = p.y - h; worst_dy_s = s; }
                    s += 1.0;
                }
                println!("e{e} {} lanes {} opp {} rev {} len {:.1} sExit {:.1} sStop {:.1} bridge {} ys {} minPadW {:.2}@{:.0} y-dem {:.1}@{:.0} from n{} ({:.0},{:.0}) to n{}",
                    g.highway(e), g.e_lanes[e], g.e_opp[e], g.e_rev[e], len, g.e_s_exit[e], g.e_s_stop[e], g.e_is_bridge[e], g.e_has_ys[e], min_w, min_w_s, worst_dy, worst_dy_s,
                    g.e_from[e], g.px[v0], g.pz[v0], g.e_to[e]);
            }
            eprintln!("{n} edges in box");
        }
        "run" => {
            let js = a.get(3).map(|s| s == "js").unwrap_or(true);
            let n: usize = a.get(4).map(|s| s.parse().unwrap()).unwrap_or(600);
            let seed: f64 = a.get(5).map(|s| s.parse().unwrap()).unwrap_or(777.0);
            let minutes: f64 = a.get(6).map(|s| s.parse().unwrap()).unwrap_or(40.0);
            let noise: f64 = a.get(7).map(|s| s.parse().unwrap()).unwrap_or(0.0);
            let region = a.get(8).cloned().unwrap_or("sf".into());
            let mut cfg = PopConfig::env(n, seed);
            if region == "box" { cfg.set_region(Some(Region::Bbox([BOX.0 - 1500.0, BOX.2 - 1500.0, BOX.1 + 1500.0, BOX.3 + 1500.0]))); }
            else if region != "none" { cfg.set_region(Some(Region::preset(&region))); }
            let bias: f64 = std::env::var("BIAS").ok().map(|v| v.parse().unwrap()).unwrap_or(0.0);
            let gas: f64 = std::env::var("GAS").ok().map(|v| v.parse().unwrap()).unwrap_or(1.0);
            let mut phantom = 0usize;
            let pol_name = std::env::var("POLICY").unwrap_or_default();
            let mut pol = bayenv_sim::harness::Policy::new(if pol_name.is_empty() { "cruise" } else { &pol_name }, seed);
            let o = SimOpts::default();
            let mut boxh = [0u64; 6]; let mut box_steps = 0u64; let mut box_worst = (0.0f64, 0usize, 0i32, 0.0f64);
            let hist_n: usize = std::env::var("HIST").ok().map(|v| v.parse().unwrap()).unwrap_or(25);
            let pop = Arc::new(generate_population(&world, &cfg, None).expect("pop").pop);
            let base = if js { SimOpts::js() } else { SimOpts::default() };
            let mut opts = SimOpts { dt: 0.1, seed, traffic_mode: TrafficMode::Full, ..base };
            if std::env::var("BOXOFF").is_ok() { opts.box_barrier = false; }
            if std::env::var("BOXON").is_ok() { opts.box_barrier = true; }
            let ec = EnvConfig { opts, start_time: Some(7.0 * 3600.0), static_obstacles: vec![], scenario_obstacles: vec![], obstacles: None };
            let mut env = Env::new(world.clone(), pop, ec, Box::new(|w| bayenv_sim::harness::router_for(&w)));
            env.sim.barrier_dbg = Some(Vec::new());
            let steps = (minutes * 600.0) as usize;
            let mut hist: Vec<VecDeque<String>> = Vec::new();
            let mut seen = 0usize;
            let mut rng = bayenv_sim::rng::Rng::new(seed);
            let mut big = 0usize;
            for _ in 0..steps {
                let slots: Vec<usize> = env.driving().collect();
                if !pol_name.is_empty() { pol.act(&mut env); }
                else { for &i in &slots {
                    let b = if bias > 0.0 { let h = ((i as u64).wrapping_mul(2654435761) % 1000) as f64 / 1000.0; (h * 2.0 - 1.0) * bias } else { 0.0 };
                    let st = b + if noise > 0.0 { (rng.random() * 2.0 - 1.0) * noise } else { 0.0 };
                    env.act(i, Action { steer: Some(st), steer_rate: None, gas, brake: 0.0 });
                } }
                env.step(1);
                let vs = &env.sim.vs;
                if hist.len() < vs.high { hist.resize(vs.high, VecDeque::new()); }
                for i in 0..vs.high {
                    if vs.state[i] != STATE_DRIVING && vs.state[i] != bayenv_sim::vehicle::STATE_CRASHED { hist[i].clear(); continue; }
                    let e = vs.edge[i] as usize;
                    let gy = dem.as_ref().map(|d| d.height_at(vs.x[i] as f64, vs.z[i] as f64)).unwrap_or(f64::NAN);
                    let rp = vs.route_pos[i] as usize; let rnext = vs.routes[i].get(rp + 1).copied().unwrap_or(-1);
                    hist[i].push_back(format!("  t{} st{} e{} c{} s {:.2}/{:.1} lat {:.2} hr {:.3} v {:.1} vlat {:.2} xyz ({:.1},{:.2},{:.1}) dem {:.2} lane {} lw {:.2} next e{}",
                        env.sim.step_count, vs.state[i], e, vs.conn[i], vs.s[i], g.e_length[e], vs.lat[i], vs.heading_rel[i], vs.speed[i], vs.v_lat[i], vs.x[i], vs.y[i], vs.z[i], gy, vs.lane[i], g.lane_w(e, vs.s[i] as f64), rnext));
                    if hist[i].len() > hist_n { hist[i].pop_front(); }
                }
                for i in 0..vs.high {
                    if vs.state[i] != STATE_DRIVING || vs.conn[i] >= 0 { continue; }
                    let e = vs.edge[i] as usize; let sv = vs.s[i] as f64;
                    if !(g.j.is_some() && (sv < g.e_s_exit[e] + 1.0 || sv > g.e_s_stop[e])) { continue; }
                    box_steps += 1;
                    let w = g.lane_w(e, sv);
                    let (pad_r, pad_l) = if let Some(lg) = &g.lg { let bd = lg.bounds(e, sv); (bd.r - bd.right, bd.r - bd.left) } else { (0.0, g.e_lanes[e] as f64 * w) };
                    let right = pad_r - if g.is_shoulder_road(e) { o.freeway_shoulder } else { o.shoulder };
                    let left = pad_l + if g.e_rev[e] >= 0 { (if g.e_opp[e] != 0 { g.e_opp[e] as f64 } else { g.e_lanes[e] as f64 }) * w + o.opposite_margin } else { o.shoulder };
                    let hr = vs.heading_rel[i] as f64; let ext = hr.sin().abs() * vs.hl[i] as f64 + hr.cos().abs() * vs.hw[i] as f64; let lat = vs.lat[i] as f64;
                    let x = (right - (lat - ext)).max(lat + ext - left);
                    for (k, t) in [0.0, 0.5, 1.0, 2.0, 5.0, 20.0].iter().enumerate() { if x > *t { boxh[k] += 1; } }
                    if x > box_worst.0 { box_worst = (x, i, e as i32, sv); }
                }
                let dbg = env.sim.barrier_dbg.as_mut().unwrap();
                let recs: Vec<_> = dbg.drain(..).collect();
                for d in &recs {
                    if d.dva * 3.6 >= 50.0 {
                        big += 1; if d.depth > 2.0 { phantom += 1; }
                        let tm = env.sim.time; let hh = (tm / 3600.0) as i64; let mm = ((tm % 3600.0) / 60.0) as i64; let ss = tm % 60.0;
                        println!("BIG barrier {:02}:{:02}:{:04.1} slot {} dv {:.0} km/h speed {:.0} km/h inbox {} box {} hw {} | e{} s {:.2}/{:.1} sExit {:.1} sStop {:.1} lat {:.2} hr {:.3} side {} depth {:.2} right {:.2} left {:.2} padR {:.2} padL {:.2} bounds(R {:.2} right {:.2} left {:.2}) ext {:.2} n ({:.3},{:.3}) v ({:.1},{:.1}) scrape {} xyz ({:.1},{:.2},{:.1})",
                            hh, mm, ss, d.slot, d.dva * 3.6, d.speed * 3.6, in_box(d.x, d.z), d.in_box, g.highway(d.edge as usize), d.edge, d.s, g.e_length[d.edge as usize], g.e_s_exit[d.edge as usize], g.e_s_stop[d.edge as usize],
                            d.lat, d.heading_rel, d.side, d.depth, d.right, d.left, d.pad_r, d.pad_l, d.b_r, d.b_right, d.b_left, d.ext, d.nx, d.nz, d.vx, d.vz, d.scrape, d.x, d.y, d.z);
                        if in_box(d.x, d.z) || d.depth > 2.0 || std::env::var("ALLHIST").is_ok() { for l in &hist[d.slot] { println!("{l}"); } }
                    }
                }
                for sc in &env.sim.crashes[seen..] {
                    let tm = sc.time; let hh = (tm / 3600.0) as i64; let mm = ((tm % 3600.0) / 60.0) as i64; let ss = tm % 60.0;
                    println!("SCENE {:02}:{:02}:{:04.1} {} dv {:?} edge {} {} inbox {} xz ({:.0},{:.0})", hh, mm, ss, sc.cause, sc.delta_v_kmh.iter().map(|x| x.round()).collect::<Vec<_>>(), sc.edge, g.highway(sc.edge as usize), in_box(sc.x, sc.z), sc.x, sc.z);
                }
                seen = env.sim.crashes.len();
            }
            println!("box-part car-steps {box_steps}: past the kerb line by >0 {} >0.5 {} >1 {} >2 {} >5 {} >20 m {}; worst {:.2} m (slot {} e{} s {:.1})", boxh[0], boxh[1], boxh[2], boxh[3], boxh[4], boxh[5], box_worst.0, box_worst.1, box_worst.2, box_worst.3);
            let st: std::collections::HashMap<_, _> = env.stats().into_iter().collect();
            let bsc = env.sim.crashes.iter().filter(|c| c.cause == "barrier" && !c.disabled.is_empty()).count();
            println!("barrier crash scenes (disabling) {bsc}, crashed vehicles {}, minor contacts {}", st["crashed_vehicles"], st["minor_contacts"]);
            println!("done: big barrier contacts {big} (depth > 2 m: {phantom}), crash scenes {}, arrived {}", seen, st["arrived"]);
        }
        "scan" => {
            // (1) bounds pinch: a car at a lane centre with heading 0 that barrier() would push (per lane, edge part only)
            // (2) road reference height (lane geometry, what the car uses) vs the 20 m ground grid
            let dem = dem.expect("dem");
            let o = SimOpts::default();
            let hw_max = bayenv_sim::physics::VEHICLE_PARAMS.iter().map(|p| 0.5 * p.width).fold(0.0, f64::max);
            let hw_sedan = 0.5 * bayenv_sim::physics::VEHICLE_PARAMS[0].width;
            let (mut pinch_sedan, mut pinch_any, mut nan_e, mut inv_e) = (Vec::new(), 0usize, 0usize, 0usize);
            let mut elev: Vec<(f64, f64, usize, f64, f64)> = Vec::new(); // (below m, run m, edge, s, lg-vs-edge)
            let mut lg_vs_edge: Vec<(f64, usize, f64)> = Vec::new();
            let (mut n_b3, mut n_b5, mut n_b10, mut n_a10) = (0, 0, 0, 0);
            for e in 0..g.edge_count {
                let len = g.e_length[e];
                let lanes = g.e_lanes[e] as f64;
                let lo = if g.j.is_some() { g.e_s_exit[e] + 1.0 } else { 0.0 };
                let hi = if g.j.is_some() { g.e_s_stop[e] } else { len };
                let (mut worst, mut worst_s, mut has_nan, mut has_inv, mut any) = (f64::INFINITY, 0.0, false, false, false);
                let mut s = lo;
                while s <= hi {
                    let w = g.lane_w(e, s);
                    let (pad_r, pad_l) = if let Some(lg) = &g.lg { let bd = lg.bounds(e, s); (bd.r - bd.right, bd.r - bd.left) } else { (0.0, lanes * w) };
                    let right = pad_r - if g.is_shoulder_road(e) { o.freeway_shoulder } else { o.shoulder };
                    let left = pad_l + if g.e_rev[e] >= 0 { (if g.e_opp[e] != 0 { g.e_opp[e] as f64 } else { lanes }) * w + o.opposite_margin } else { o.shoulder };
                    if !(right.is_finite() && left.is_finite() && w.is_finite()) { has_nan = true; }
                    if left < right { has_inv = true; }
                    for k in 0..g.e_lanes[e] {
                        let c = (k as f64 + 0.5) * w;
                        let m = jm_min(c - hw_sedan - right, left - c - hw_sedan);
                        if m < worst { worst = m; worst_s = s; }
                        if c - hw_max < right || c + hw_max > left { any = true; }
                    }
                    s += 1.0;
                }
                if has_nan { nan_e += 1; } if has_inv { inv_e += 1; } if any { pinch_any += 1; }
                if worst < 0.0 { pinch_sedan.push((worst, e, worst_s)); }
                // elevation
                if g.e_is_bridge[e] { continue; }
                let (mut run, mut best_run, mut best_below, mut best_s, mut max_lgd) = (0.0, 0.0, 0.0f64, 0.0, 0.0f64);
                let mut s = 0.0; let step = 2.0;
                let mut max_above = 0.0f64;
                while s <= len {
                    let p = g.edge_point(e, s, 0.0, 0);
                    let y = if let Some(lg) = &g.lg { let q = lg.offset_point(e, s, 0.0, 0.0); max_lgd = max_lgd.max((q.y - p.y).abs()); q.y } else { p.y };
                    let h = dem.height_at(p.x, p.z);
                    if h.is_finite() {
                        let below = h - y;
                        if below > 3.0 { run += step; } else { run = 0.0; }
                        if below > best_below { best_below = below; best_s = s; }
                        if run > best_run { best_run = run; }
                        max_above = max_above.max(-below);
                    }
                    s += step;
                }
                if best_below > 3.0 { n_b3 += 1; } if best_below > 5.0 { n_b5 += 1; } if best_below > 10.0 { n_b10 += 1; } if max_above > 10.0 { n_a10 += 1; }
                if best_below > 3.0 { elev.push((best_below, best_run, e, best_s, max_lgd)); }
                if max_lgd > 0.5 { lg_vs_edge.push((max_lgd, e, 0.0)); }
            }
            pinch_sedan.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            println!("PINCH: edges where a sedan at a lane centre (heading 0) is inside a barrier on the edge part: {} (any vehicle type: {pinch_any}); NaN bounds {nan_e}; inverted bounds {inv_e}", pinch_sedan.len());
            for (m, e, s) in pinch_sedan.iter().take(25) {
                let p = g.edge_point(*e, *s, 0.0, 0);
                let bd = g.lg.as_ref().map(|lg| lg.bounds(*e, *s)).unwrap_or_default();
                println!("  e{e} {} lanes {} opp {} rev {} len {:.1} s {:.1} margin {:.2} lw {:.2} bounds(R {:.2} right {:.2} left {:.2}) at ({:.0},{:.0})", g.highway(*e), g.e_lanes[*e], g.e_opp[*e], g.e_rev[*e], g.e_length[*e], s, m, g.lane_w(*e, *s), bd.r, bd.right, bd.left, p.x, p.z);
            }
            elev.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            println!("ELEV (non-bridge edges; lane-geometry reference y vs 20 m ground grid): road below ground by >3 m: {n_b3}, >5 m: {n_b5}, >10 m: {n_b10}; road above ground by >10 m: {n_a10}; |lg y - edge y| > 0.5 m: {}", lg_vs_edge.len());
            for (b, r, e, s, d) in elev.iter().take(25) {
                let p = g.edge_point(*e, *s, 0.0, 0);
                println!("  e{e} {} len {:.0} below {:.1} m at s {:.0} (run >3 m: {:.0} m) lg-edge {:.2} at ({:.0},{:.0}) inbox {}", g.highway(*e), g.e_length[*e], b, s, r, d, p.x, p.z, in_box(p.x, p.z));
            }
            let inb: Vec<_> = elev.iter().filter(|x| { let p = g.edge_point(x.2, x.3, 0.0, 0); in_box(p.x, p.z) }).collect();
            println!("  in the box: {}", inb.len());
        }
        "hops" => {
            // edge-to-edge hops without a connector (no movement for the lane): lat after mapLat (+ the lane-centre snap)
            // vs the next edge's barrier limits where its barrier first acts (s = sExit + 1 with junctions)
            let o = SimOpts::default();
            let hw = 0.5 * bayenv_sim::physics::VEHICLE_PARAMS[0].width;
            let (mut n_hops, mut n_out, mut n_strict_out) = (0usize, 0usize, 0usize);
            let mut worst: Vec<(f64, usize, usize, i32, bool, f64)> = Vec::new();
            for e in 0..g.edge_count {
                let to = g.e_to[e] as usize;
                for q in g.out_start[to]..g.out_start[to + 1] {
                    let en = g.out_edges[q as usize] as usize;
                    if en as i32 == g.e_rev[e] { continue; }
                    for k in 0..g.e_lanes[e] {
                        if let Some(j) = &g.j { if g.e_s_stop[e] < g.e_length[e] && j.movement(e, k, en as i32) >= 0 { continue; } }
                        let hop = g.hop(e, en);
                        if hop.strict {
                            let ml = g.map_lane(e, en, k);
                            if ml < 0 || ml >= g.e_lanes[en] {
                                // vehicle.js: my lane leads onto another strict out-edge -> the car goes there instead
                                let alt = (g.out_start[to]..g.out_start[to + 1]).map(|q| g.out_edges[q as usize] as usize)
                                    .any(|o2| o2 != en && o2 as i32 != g.e_rev[e] && g.hop(e, o2).strict && { let l = g.map_lane(e, o2, k); l >= 0 && l < g.e_lanes[o2] });
                                if alt { continue; }
                            }
                        }
                        n_hops += 1;
                        let mut lat = (k as f64 + 0.5) * g.lane_w(e, g.e_length[e]);
                        // at e's end the car is inside e's own barrier limits (where they act there: no junction box)
                        if !(g.j.is_some() && g.e_s_stop[e] < g.e_length[e]) {
                            let le = g.e_length[e]; let we = g.lane_w(e, le); let ne = g.e_lanes[e] as f64;
                            let (pr, pl) = if let Some(lg) = &g.lg { let bd = lg.bounds(e, le); (bd.r - bd.right, bd.r - bd.left) } else { (0.0, ne * we) };
                            let r0 = pr - if g.is_shoulder_road(e) { o.freeway_shoulder } else { o.shoulder };
                            let l0 = pl + if g.e_rev[e] >= 0 { (if g.e_opp[e] != 0 { g.e_opp[e] as f64 } else { ne }) * we + o.opposite_margin } else { o.shoulder };
                            lat = lat.max(r0 + hw).min(l0 - hw);
                        }
                        let mut l2 = g.map_lat(e, en, lat, hw);
                        let sc = if g.j.is_some() { g.e_s_exit[en] + 1.0 } else { 0.0 };
                        let sc = sc.min(g.e_length[en]);
                        let w = g.lane_w(en, 0.0); let n = g.e_lanes[en] as f64;
                        if l2 >= 0.0 && l2 <= n * w && (l2 - hw < 0.3 || l2 + hw > n * w - 0.3) && (!hop.strict || g.e_lanes[e] != g.e_lanes[en] || g.is_shoulder_road(en)) { l2 = ((l2 / w).floor().clamp(0.0, n - 1.0) + 0.5) * w; }
                        let w2 = g.lane_w(en, sc);
                        let (pad_r, pad_l) = if let Some(lg) = &g.lg { let bd = lg.bounds(en, sc); (bd.r - bd.right, bd.r - bd.left) } else { (0.0, n * w2) };
                        let right = pad_r - if g.is_shoulder_road(en) { o.freeway_shoulder } else { o.shoulder };
                        let left = pad_l + if g.e_rev[en] >= 0 { (if g.e_opp[en] != 0 { g.e_opp[en] as f64 } else { n }) * w2 + o.opposite_margin } else { o.shoulder };
                        let depth = (right - (l2 - hw)).max(l2 + hw - left);
                        if depth > 0.0 { n_out += 1; if hop.strict { n_strict_out += 1; } worst.push((depth, e, en, k, hop.strict, l2)); }
                    }
                }
            }
            worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            println!("HOPS without a connector: {n_hops} (edge, out-edge, lane); a lane-centre sedan lands past a barrier line: {n_out} (strict hops {n_strict_out}); >1 m {}, >3 m {}", worst.iter().filter(|w| w.0 > 1.0).count(), worst.iter().filter(|w| w.0 > 3.0).count());
            for (d, e, en, k, st, l2) in worst.iter().take(15) {
                let p = g.edge_point(*en, 0.0, 0.0, 0);
                println!("  e{e} ({} {} lanes) lane {k} -> e{en} ({} {} lanes) strict {st} lat {:.2} depth {:.2} at ({:.0},{:.0})", g.highway(*e), g.e_lanes[*e], g.highway(*en), g.e_lanes[*en], l2, d, p.x, p.z);
            }
        }
        "hopinfo" => {
            let e: usize = a[3].parse().unwrap();
            let to = g.e_to[e] as usize;
            println!("e{e} {} lanes {} len {:.1} sStop {:.1} rev {} to n{to} lw_end {:.2}", g.highway(e), g.e_lanes[e], g.e_length[e], g.e_s_stop[e], g.e_rev[e], g.lane_w(e, g.e_length[e]));
            for q in g.out_start[to]..g.out_start[to + 1] {
                let en = g.out_edges[q as usize] as usize;
                let h = g.hop(e, en);
                let ml: Vec<i32> = (0..g.e_lanes[e]).map(|k| g.map_lane(e, en, k)).collect();
                let lats: Vec<String> = (0..g.e_lanes[e]).map(|k| format!("{:.2}", g.map_lat(e, en, (k as f64 + 0.5) * g.lane_w(e, g.e_length[e]), 0.9))).collect();
                println!("  -> e{en} {} lanes {} strict {} shift {} mapLane {:?} mapLat {:?} sExit {:.1}", g.highway(en), g.e_lanes[en], h.strict, h.shift, ml, lats, g.e_s_exit[en]);
            }
            for q in g.in_start[g.e_from[g.out_edges[g.out_start[to] as usize] as usize] as usize]..g.in_start[g.e_from[g.out_edges[g.out_start[to] as usize] as usize] as usize + 1] { let i = g.in_edges[q as usize]; println!("  in e{i} {} lanes {}", g.highway(i as usize), g.e_lanes[i as usize]); }
        }
        "mv" => {
            let e: usize = a[3].parse().unwrap();
            let j = g.j.as_ref().unwrap();
            println!("e{e} {} lanes {} len {:.1} sExit {:.1} sStop {:.1} to n{}", g.highway(e), g.e_lanes[e], g.e_length[e], g.e_s_exit[e], g.e_s_stop[e], g.e_to[e]);
            for m in j.movements_at(g.e_to[e] as usize) {
                if j.info(m, 0) as usize != e { continue; }
                println!("  m{m} lane {} -> e{} lane {} sIn {:.2} sOut {:.2} len {:.2} turn {:.3} straight {}", j.info(m, 1), j.info(m, 2), j.info(m, 3), j.s_in(m), j.s_out(m), j.m_len[m], j.turn(m), j.info(m, 6));
            }
            let mut s = 0.0; while s <= g.e_length[e] { println!("  s {s:.0} lw {:.2}", g.lane_w(e, s)); s += 10.0; }
        }
        _ => panic!("mode"),
    }
}
