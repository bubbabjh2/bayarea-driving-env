// bayenv-gpu kernel (not part of src/gpu/kernels): the contact-resolve stage of contacts.wgsl in ONE dispatch.
// Changed vs the original (gpuSim.js: resolveRounds x [ownClear, ownMin, resolveRound] + resolve, 37 dispatches):
//  * one workgroup of RF_WG threads loops over the contact list (k = lid, lid + RF_WG, ...) and runs the
//    RF_ROUNDS rounds with storageBarrier + workgroupBarrier between the phases instead of dispatch boundaries.
//    The phase bodies are those of ownClear / ownMin / resolveRound (same isDone / prio / resolveOne), so every
//    round resolves the same set of contacts, and the contacts resolved in one round share no car.
//  * rounds stop early once every contact is done (rfLeft = contacts this round could not resolve; the later
//    rounds would skip them all), and none run without contacts;
//  * the serial tail of resolve() (compaction of the contacts left, keeping list order) is a workgroup prefix
//    sum (same order, same result); the insertion sort, the serial resolveOne loop and the diagnostics are the
//    original code, run by thread 0.
// contacts.wgsl (+ common, collidefns) is included unchanged; RF_ROUNDS is prepended by the host.
// #include contacts.wgsl

const RF_WG: u32 = 256u;
var<workgroup> rfScan: array<u32, 256>;
var<workgroup> rfM: u32;
var<workgroup> rfN: u32;
var<workgroup> rfLeft: atomic<u32>;
var<workgroup> rfLeftU: u32;

fn rfOwnClear(k: u32) {
  let o = 4u + k * 8u;
  if (isDone(o)) { return; }
  atomicStore(&contacts[ccBase() + atomicLoad(&contacts[o])], NONE);
  atomicStore(&contacts[ccBase() + atomicLoad(&contacts[o + 1u])], NONE);
}
fn rfOwnMin(k: u32) {
  let o = 4u + k * 8u;
  if (isDone(o)) { return; }
  let key = prio(k);
  atomicMin(&contacts[ccBase() + atomicLoad(&contacts[o])], key);
  atomicMin(&contacts[ccBase() + atomicLoad(&contacts[o + 1u])], key);
}
fn rfRound(k: u32) {
  let o = 4u + k * 8u;
  if (isDone(o)) { return; }
  let key = prio(k);
  if (atomicLoad(&contacts[ccBase() + atomicLoad(&contacts[o])]) != key || atomicLoad(&contacts[ccBase() + atomicLoad(&contacts[o + 1u])]) != key) { atomicAdd(&rfLeft, 1u); return; }
  resolveOne(k);
  atomicStore(&contacts[o + 2u], bitcast<u32>(2.0));
}

@compute @workgroup_size(256)
fn resolveFused(@builtin(local_invocation_index) li: u32) {
  if (li == 0u) { rfN = min(atomicLoad(&contacts[0]), P.contactCap); }
  let n = workgroupUniformLoad(&rfN);   // uniform: the barriers below sit in loops over n
  let chunks = (n + RF_WG - 1u) / RF_WG;
  for (var r = 0u; r < RF_ROUNDS; r = r + 1u) {
    if (n == 0u) { break; }
    if (li == 0u) { atomicStore(&rfLeft, 0u); }
    for (var c = 0u; c < chunks; c = c + 1u) { let k = c * RF_WG + li; if (k < n) { rfOwnClear(k); } }
    storageBarrier(); workgroupBarrier();
    for (var c = 0u; c < chunks; c = c + 1u) { let k = c * RF_WG + li; if (k < n) { rfOwnMin(k); } }
    storageBarrier(); workgroupBarrier();
    for (var c = 0u; c < chunks; c = c + 1u) { let k = c * RF_WG + li; if (k < n) { rfRound(k); } }
    storageBarrier(); workgroupBarrier();
    // every contact done: the remaining rounds would do nothing
    if (li == 0u) { rfLeftU = atomicLoad(&rfLeft); }
    if (workgroupUniformLoad(&rfLeftU) == 0u) { break; }
  }
  // order-preserving compaction of the contacts not done (resolve(): m = number kept)
  if (li == 0u) { rfM = 0u; }
  workgroupBarrier();
  for (var c = 0u; c < chunks; c = c + 1u) {
    let k = c * RF_WG + li;
    let o = 4u + k * 8u;
    let keep = k < n && !isDone(o);
    var rec: array<u32, 8>;
    if (keep) { for (var w = 0u; w < 8u; w = w + 1u) { rec[w] = atomicLoad(&contacts[o + w]); } }
    rfScan[li] = select(0u, 1u, keep);
    workgroupBarrier();
    for (var d = 1u; d < RF_WG; d = d * 2u) {
      var x = 0u; if (li >= d) { x = rfScan[li - d]; }
      workgroupBarrier();
      rfScan[li] = rfScan[li] + x;
      workgroupBarrier();
    }
    let base = rfM;
    storageBarrier(); workgroupBarrier();   // every record of this chunk is read before any is overwritten
    if (keep) { let dst = 4u + (base + rfScan[li] - 1u) * 8u; for (var w = 0u; w < 8u; w = w + 1u) { atomicStore(&contacts[dst + w], rec[w]); } }
    storageBarrier(); workgroupBarrier();
    if (li == RF_WG - 1u) { rfM = base + rfScan[li]; }
    workgroupBarrier();
  }
  if (li != 0u) { return; }
  let m = rfM;
  // from here: resolve() unchanged (CPU order sort for short lists, serial resolve, diagnostics)
  let nSort = select(0u, m, m <= 512u);
  for (var a = 1u; a < nSort; a = a + 1u) {
    var rec: array<u32, 8>;
    for (var w = 0u; w < 8u; w = w + 1u) { rec[w] = atomicLoad(&contacts[4u + a * 8u + w]); }
    var b = a;
    loop {
      if (b == 0u) { break; }
      let o = 4u + (b - 1u) * 8u;
      if (!lessPair(rec[0], rec[1], atomicLoad(&contacts[o]), atomicLoad(&contacts[o + 1u]))) { break; }
      for (var w = 0u; w < 8u; w = w + 1u) { atomicStore(&contacts[o + 8u + w], atomicLoad(&contacts[o + w])); }
      b = b - 1u;
    }
    for (var w = 0u; w < 8u; w = w + 1u) { atomicStore(&contacts[4u + b * 8u + w], rec[w]); }
  }
  for (var k = 0u; k < m; k = k + 1u) { resolveOne(k); }
  atomicStore(&io[IO_STATS + 46u], atomicLoad(&contacts[0])); atomicStore(&io[IO_STATS + 47u], m);   // diagnostics: contacts found / left to the serial pass
  atomicStore(&contacts[0], 0u);
}
