//! Port of test/sim/population.test.js (3 of 3 JS test cases -> 3 #[test]s).
//! generatePopulation(g, {n, seed, ...popOpts(g)}) -> population_gen::generate_population(PopConfig::env_synthetic + return_trips = true,
//! the JS generatePopulation default). JS `p.stats` (populationStats) is recomputed here from the Rust PopExtras (kind, crosses_bay)
//! and Population arrays with the same definitions; the JS `commuter.genome.traits.maxAccel` check becomes "the 64-float delta is
//! finite" because traits.js (delta -> traits) is not ported (the delta itself is bit-exact vs JS in tests/population_gen.rs).
//! synthAxis(g).sideOf -> Axis::synth(&World).side_of.
mod common;
use bayenv_sim::population_gen::*;
use bayenv_sim::world::World;
use std::sync::Arc;

fn cfg(n: usize, seed: f64) -> PopConfig { let mut c = PopConfig::env_synthetic(n, seed); c.return_trips = true; c }
fn gen(w: &Arc<World>, n: usize, seed: f64) -> Generated { generate_population(w, &cfg(n, seed), None).unwrap() }

#[test]
fn population_is_deterministic_for_a_seed_and_differs_across_seeds() {
    let g = need!(common::shared_world("synth"));
    let (a, b, c) = (gen(&g, 300, 7.0).pop, gen(&g, 300, 7.0).pop, gen(&g, 300, 8.0).pop);
    for i in 0..300 {
        assert_eq!(a.home[i], b.home[i]);
        assert_eq!(a.work[i], b.work[i]);
        assert_eq!(a.depart[i], b.depart[i]);
        assert_eq!(a.route(i, 0), b.route(i, 0));
    }
    assert!((0..300).any(|i| a.home[i] != c.home[i]));
}

#[test]
fn local_long_mixture_cross_bay_share_and_trip_length_histogram() {
    let g = need!(common::shared_world("synth40"));
    let Generated { pop: p, extras: x } = gen(&g, 2000, 3.0);
    let n = 2000.0;
    let local = x.kind.iter().filter(|&&k| k == 0).count() as f64;
    let (mut long_n, mut long_cross, mut cross) = (0.0, 0.0, 0.0);
    for i in 0..2000 {
        if x.crosses_bay[i] != 0 { cross += 1.0; }
        if x.kind[i] != 0 { long_n += 1.0; if x.crosses_bay[i] != 0 { long_cross += 1.0; } }
    }
    let local_share = local / n;
    let long_cross_share = if long_n > 0.0 { long_cross / long_n } else { 0.0 };
    assert!(local_share > 0.75 && local_share < 0.85, "localShare {local_share}");
    assert!(long_cross_share > 0.25 && long_cross_share < 0.6, "long cross share {long_cross_share}");
    assert!(cross / n > 0.05, "cross share {}", cross / n);
    // trip-length histogram (5 km bins) and departures per 15 min
    let max_l = p.route_length.iter().cloned().fold(0.0, f64::max);
    let mut bins = vec![0usize; (max_l / 5000.0).floor() as usize + 1];
    for &l in &p.route_length { bins[(l / 5000.0).floor() as usize] += 1; }
    assert!(bins.len() >= 2 && bins.iter().sum::<usize>() == 2000);
    let mut dep = [0usize; 96];
    for &d in &p.depart { dep[((d / 900.0).floor().max(0.0) as usize).min(95)] += 1; }
    assert!(dep.iter().filter(|&&c| c > 0).count() > 4);
    // every commuter has a route and a sane departure in the morning window
    for c in 0..2000 {
        assert!(!p.route(c, 0).is_empty());
        assert!(p.depart[c] >= 5.0 * 3600.0 && p.depart[c] <= 10.5 * 3600.0);
        assert!(p.return_depart[c] > p.depart[c] + 3600.0);
        assert!(p.deltas[c * DELTA_SIZE..(c + 1) * DELTA_SIZE].iter().all(|v| v.is_finite()));
    }
    let trips = p.trips();
    assert_eq!(trips.len(), 4000);
    for i in 1..trips.len() { assert!(trips[i].depart >= trips[i - 1].depart); }
}

#[test]
fn side_classifier_splits_the_synthetic_shores() {
    let g = need!(common::shared_world("synth"));
    let axis = Axis::synth(&g);
    let (mut east, mut west) = (0, 0);
    for i in 0..g.node_count { if axis.side_of(g.nx[i], g.nz[i]) != 0 { east += 1; } else { west += 1; } }
    assert_eq!(east, west);
}
