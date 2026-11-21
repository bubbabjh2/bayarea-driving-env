//! # bayenv-gpu
//!
//! Native host for the WebGPU sim: the WGSL kernels of `src/gpu/kernels/` dispatched unchanged through wgpu
//! (Metal / Vulkan / DX12), with the host logic of `src/gpu/gpuSim.js` ported (world pack, scatter, step loop,
//! events, scenes). See rust/README.md.
pub mod defines;
pub mod env;
pub use bayenv_world::events;
pub mod gpu;
pub mod kernels;
pub mod layout;
pub mod pack;
pub mod oracle;
pub mod sim;
pub mod profile;
