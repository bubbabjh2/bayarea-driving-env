// bayenv-gpu helper (not part of src/gpu/kernels): indirect dispatch size for the contact-resolve rounds
// (contacts.wgsl ownClear / ownMin / resolveRound return for g.x >= min(contacts[0], contactCap), so dispatching
// ceil(n / 128) workgroups instead of ceil(contactCap / 128) gives identical results).
struct CP { cap: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<uniform> CA: CP;
@group(0) @binding(1) var<storage, read> contacts: array<u32>;
@group(0) @binding(2) var<storage, read_write> cargs: array<u32>;
@compute @workgroup_size(1)
fn main() {
  let n = min(contacts[0], CA.cap);
  cargs[0] = max(1u, (n + 127u) / 128u); cargs[1] = 1u; cargs[2] = 1u;
}
