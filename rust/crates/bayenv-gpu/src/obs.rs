//! Observation readback ring: the observation copy of a step goes into one of two MAP_READ buffers, which stays
//! mapped until the ring writes into it again. So the host can read step t (copy or zero-copy view) while the GPU
//! runs step t+1 into the other slot (lagged / async stepping), and the latest result can be exposed without a host
//! memcpy (`ready_ptr`).
//!
//! Zero-copy is only offered where a mapping pointer outlives `unmap` (`zero_copy_ok`): on Metal a mapped buffer is
//! `MTLBuffer.contents` (shared storage, unified memory) and `unmap` is a no-op, so the memory stays valid while the
//! buffer is alive. Elsewhere callers copy out of the mapped range before the slot is reused.
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Element format of an observation readback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObsFmt {
    /// the `inputs` buffer as is (bit-identical with `read_obs_into`)
    F32,
    /// binary16, converted on the GPU (kernels/obs_f16.wgsl; round to nearest even, saturating at +-65504)
    F16,
}
impl ObsFmt {
    pub fn elem_bytes(self) -> usize { match self { ObsFmt::F32 => 4, ObsFmt::F16 => 2 } }
}

/// Where a sim writes its observation rows: `rows` x obs_size elements of `fmt` at byte `offset` of `buf`.
#[derive(Clone)]
pub struct ObsDst { pub buf: Arc<wgpu::Buffer>, pub offset: u64, pub rows: usize, pub fmt: ObsFmt }

struct Slot { buf: Option<Arc<wgpu::Buffer>>, mapped: bool, done: Arc<AtomicBool>, bytes: u64, fmt: ObsFmt, tag: u64 }

/// Two MAP_READ slots: `ready` (last completed readback, mapped), one being written / in flight.
pub struct ObsRing {
    device: wgpu::Device,
    slots: [Slot; 2],
    ready: Option<usize>,
    writing: Option<usize>,
    inflight: Option<usize>,
    /// Metal: mapped pointers stay valid after unmap (see the module docs)
    pub zero_copy_ok: bool,
}

impl ObsRing {
    pub fn new(device: &wgpu::Device, backend: wgpu::Backend) -> ObsRing {
        let s = || Slot { buf: None, mapped: false, done: Arc::new(AtomicBool::new(false)), bytes: 0, fmt: ObsFmt::F32, tag: 0 };
        ObsRing { device: device.clone(), slots: [s(), s()], ready: None, writing: None, inflight: None, zero_copy_ok: backend == wgpu::Backend::Metal }
    }
    /// Pick the slot for the next readback (never the ready one; completes an in-flight readback first), unmap it,
    /// grow it to `bytes`. `tag`: caller's label of the readback (e.g. the sim step it shows). Returns the buffer to
    /// copy into; call `submitted` after the queue submit.
    pub fn begin(&mut self, bytes: usize, fmt: ObsFmt, tag: u64) -> Arc<wgpu::Buffer> {
        self.complete();
        let s = match self.ready { Some(r) => 1 - r, None => 0 };
        let bytes = ((bytes as u64 + 3) / 4 * 4).max(4);
        let sl = &mut self.slots[s];
        if sl.mapped { sl.buf.as_ref().unwrap().unmap(); sl.mapped = false; }
        if sl.buf.as_ref().map_or(true, |b| b.size() < bytes) {
            sl.buf = Some(Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("obsRing"), size: bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false })));
        }
        sl.bytes = bytes; sl.fmt = fmt; sl.tag = tag;
        self.writing = Some(s);
        sl.buf.clone().unwrap()
    }
    /// After the submit that writes the `begin` slot: request its map (completed by the next device poll).
    pub fn submitted(&mut self) {
        let s = self.writing.take().expect("ObsRing::submitted without begin");
        let sl = &mut self.slots[s];
        sl.done.store(false, Ordering::Release);
        let d = sl.done.clone();
        sl.buf.as_ref().unwrap().slice(..sl.bytes).map_async(wgpu::MapMode::Read, move |r| { if r.is_ok() { d.store(true, Ordering::Release); } });
        self.inflight = Some(s);
    }
    pub fn has_inflight(&self) -> bool { self.inflight.is_some() }
    /// Wait for the in-flight readback (polls only if its map has not completed yet); it becomes `ready`.
    pub fn complete(&mut self) {
        let Some(s) = self.inflight.take() else { return };
        if !self.slots[s].done.load(Ordering::Acquire) { self.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll"); }
        assert!(self.slots[s].done.load(Ordering::Acquire), "obs ring map failed");
        self.slots[s].mapped = true;
        self.ready = Some(s);
    }
    /// Format and tag of the ready readback.
    pub fn ready_info(&self) -> Option<(ObsFmt, u64)> { self.ready.map(|s| (self.slots[s].fmt, self.slots[s].tag)) }
    /// Forget the ready readback (the sim state changed outside a step); an in-flight one is completed first.
    pub fn invalidate(&mut self) { self.complete(); self.ready = None; }
    /// Copy `out.len()` bytes from `offset` of the ready readback.
    pub fn read_into(&self, offset: usize, out: &mut [u8]) {
        let sl = &self.slots[self.ready.expect("no ready observation")];
        let m = sl.buf.as_ref().unwrap().slice(..sl.bytes).get_mapped_range().expect("obs range");
        out.copy_from_slice(&m[offset..offset + out.len()]);
    }
    /// Pointer to the ready readback + the buffer keeping it alive (zero-copy views; `zero_copy_ok` only). The
    /// contents stay unchanged until the second `begin` after this (the first one writes the other slot).
    pub fn ready_ptr(&self) -> (*const u8, Arc<wgpu::Buffer>) {
        assert!(self.zero_copy_ok, "zero-copy observation views need unified memory (Metal)");
        let sl = &self.slots[self.ready.expect("no ready observation")];
        let b = sl.buf.clone().unwrap();
        let p = { let m = b.slice(..sl.bytes).get_mapped_range().expect("obs range"); m.as_ptr() };
        (p, b)
    }
}
