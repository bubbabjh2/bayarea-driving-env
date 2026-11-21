// bayenv-gpu helper (not part of src/gpu/kernels): one event-queue record per step, right after `tick`
// (type 254, a = the step counter), so the host can attribute every following event of a k-step batch to its step.
// Appended through common.wgsl `emit`, like every other event; nothing else reads it.
@compute @workgroup_size(1)
fn stepMark() { emit(254u, 0xFFFFFFFFu, simStep(), 0u); }
