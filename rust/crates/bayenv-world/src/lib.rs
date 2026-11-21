//! # bayenv-world
//!
//! World data of the Bay Area driving environment, ported from the JS reference (`src/roadpack.js`,
//! `src/sim/roadGraph.js`, `src/sim/data.js`, `src/geom/*`, `src/sim/controls.js`): road packs (mmap'd),
//! the sim graph, lane geometry, junctions (+ builder), traffic-control plans, surfaces, grade
//! separations, gas stations, the DEM grid and a node spatial index.
//!
//! ```no_run
//! let w = bayenv_world::World::load("data").unwrap();
//! println!("{} edges, {} junction movements", w.graph.edge_count, w.junctions.as_ref().map_or(0, |j| j.move_count()));
//! ```
//!
//! Numerics follow the JS exactly (see [`jsmath`]): ports are verified bit-for-bit against JS dumps.
pub mod bake;
pub mod blob;
pub mod controls;
pub mod events;
pub mod extras;
pub mod graph;
pub mod hops;
pub mod jsjson;
pub mod jsmath;
pub mod jsmath_x2;
pub mod junctions;
pub mod junctions_build;
pub mod lanegraph;
pub mod lanes;
pub mod lanes_build;
pub mod nav;
pub mod oracle;
pub mod roadpack;
pub mod smooth;
pub mod verify;
pub mod world;

pub use controls::TrafficControls;
pub use graph::SimGraph;
pub use junctions::JunctionSet;
pub use lanegraph::LaneGraph;
pub use lanes::LaneGeometry;
pub use roadpack::RoadSet;
pub use world::{LoadOptions, World};
