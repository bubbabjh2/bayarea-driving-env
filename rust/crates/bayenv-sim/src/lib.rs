//! bayenv-sim: Rust port of the JS CPU driving simulation (src/sim of bayarea-driving-env).
//! The JS sim is the oracle; see STATUS.md and BENCHMARKS.md.
pub mod jm;
pub mod pack;
pub mod world;
pub mod rng;
pub mod physics;
pub mod surface;
pub mod spatial;
pub mod vehicle;
pub mod population;
pub mod population_gen;
pub mod fuel;
pub mod sim;
pub mod leaders;
pub mod junction;
pub mod contacts;
pub mod extra;
pub mod route_stub;
pub mod env;
pub mod harness;
pub mod obs;
pub mod vec_env;
pub mod fx;
