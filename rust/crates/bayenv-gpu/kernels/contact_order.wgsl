// bayenv-gpu patch (GpuOptions.deterministic_contacts, default on; off = the JS kernel): text replacement in the
// resolved source of src/gpu/kernels/contacts.wgsl (also inside kernels/resolve_fused.wgsl; kernels.rs `patch_text`).
// See rust/docs/determinism.md. The parallel contact-resolve rounds give each contact a priority; a contact resolves
// in a round when it holds the minimum priority on both of its cars. The JS key is random bits of the contact's LIST
// INDEX k, and k is the order in which detect threads won the atomicAdd on the contact counter, which differs from run
// to run. Two contacts that share a car (a queue pressed together, a car between a wreck and the kerb) are therefore
// resolved in a run-dependent order, and the impulses do not commute in f32. The patch derives the key from the pair
// (i, j) itself: random high bits from a hash of (i, j), low 18 bits i + j. Among the contacts that share a car c, i + j
// = c + (the other car) is distinct, so the keys stay unique where uniqueness matters (needs i + j < 2^18:
// capacity <= 131,072, asserted by the host).
//#orig prio
fn prio(k: u32) -> u32 { return (pcg(k * 2654435761u + 1u) & 0xfffc0000u) | (k & 0x3ffffu); }
//#end
//#patch prio
fn prio(k: u32) -> u32 {
  let o = 4u + k * 8u; let i = atomicLoad(&contacts[o]); let j = atomicLoad(&contacts[o + 1u]);
  return (pcg(i * 2654435761u + pcg(j + 0x9e3779b9u)) & 0xfffc0000u) | ((i + j) & 0x3ffffu);
}
//#end
