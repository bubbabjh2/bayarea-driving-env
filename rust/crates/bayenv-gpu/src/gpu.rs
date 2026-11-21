//! wgpu device bootstrap + helpers (port of `src/gpu/device.js`): named buffer registry, kernels with explicit
//! bind-group layouts parsed from the WGSL, bind groups filled by variable name.
use crate::kernels::{parse_bindings, Decl};
use std::collections::HashMap;
use std::sync::Arc;

pub struct Kernel { pub label: String, pub pipeline: wgpu::ComputePipeline, pub layouts: Vec<wgpu::BindGroupLayout>, pub groups: Vec<Vec<Decl>> }

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_info: wgpu::AdapterInfo,
    pub limits: wgpu::Limits,
    pub buffers: HashMap<String, Arc<wgpu::Buffer>>,
    /// persistent MAP_READ staging buffers per source buffer (read_into)
    staging: std::sync::Mutex<HashMap<String, wgpu::Buffer>>,
}

pub const STORAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE.union(wgpu::BufferUsages::COPY_DST).union(wgpu::BufferUsages::COPY_SRC);

impl Gpu {
    /// Highest-performance adapter (Metal / Vulkan / DX12), the adapter's own limits requested.
    pub fn new() -> Result<Gpu, String> {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
        if std::env::var("WGPU_BACKEND").is_err() { desc.backends = wgpu::Backends::PRIMARY; }
        let instance = wgpu::Instance::new(desc);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance, force_fallback_adapter: false, compatible_surface: None, ..Default::default()
        })).map_err(|e| format!("no GPU adapter: {e}"))?;
        let al = adapter.limits();
        let need = [("maxStorageBuffersPerShaderStage", al.max_storage_buffers_per_shader_stage as u64, 10u64),
            ("maxStorageBufferBindingSize", al.max_storage_buffer_binding_size as u64, 128 << 20), ("maxComputeWorkgroupsPerDimension", al.max_compute_workgroups_per_dimension as u64, 65535)];
        for (k, v, m) in need { if v < m { return Err(format!("GPU adapter too small: {k}={v} < {m}")); } }
        let mut limits = wgpu::Limits::default();
        limits.max_storage_buffer_binding_size = al.max_storage_buffer_binding_size;
        limits.max_buffer_size = al.max_buffer_size;
        limits.max_storage_buffers_per_shader_stage = al.max_storage_buffers_per_shader_stage;
        limits.max_compute_workgroups_per_dimension = al.max_compute_workgroups_per_dimension;
        limits.max_compute_invocations_per_workgroup = al.max_compute_invocations_per_workgroup;
        limits.max_compute_workgroup_size_x = al.max_compute_workgroup_size_x;
        limits.max_compute_workgroup_storage_size = al.max_compute_workgroup_storage_size;
        limits.max_bind_groups = al.max_bind_groups.min(4);
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("bayenv"), required_features: wgpu::Features::empty(), required_limits: limits.clone(),
            memory_hints: wgpu::MemoryHints::Performance, ..Default::default()
        })).map_err(|e| format!("request_device: {e}"))?;
        device.on_uncaptured_error(Arc::new(|e| eprintln!("wgpu error: {e}")));
        Ok(Gpu { adapter_info: adapter.get_info(), device, queue, limits, buffers: HashMap::new(), staging: Default::default() })
    }
    /// A second context on the same device / queue (own buffer registry): several sims on one GPU.
    pub fn share(&self) -> Gpu {
        Gpu { device: self.device.clone(), queue: self.queue.clone(), adapter_info: self.adapter_info.clone(), limits: self.limits.clone(), buffers: HashMap::new(), staging: Default::default() }
    }
    pub fn adapter_name(&self) -> String { format!("{} ({:?})", self.adapter_info.name, self.adapter_info.backend) }

    /// Create (or replace) a named buffer; size rounded up to 4 and >= 16 (as device.js).
    pub fn buffer(&mut self, name: &str, size: u64, usage: wgpu::BufferUsages, data: Option<&[u8]>) -> Arc<wgpu::Buffer> {
        let size = ((size + 3) / 4 * 4).max(16);
        let b = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some(name), size, usage, mapped_at_creation: data.is_some() });
        if let Some(d) = data {
            let n = d.len().min(size as usize);
            b.slice(..).get_mapped_range_mut().expect("mapped").slice(..n).copy_from_slice(&d[..n]);
            b.unmap();
        }
        let b = Arc::new(b);
        if let Some(old) = self.buffers.insert(name.to_string(), b.clone()) { old.destroy(); }
        b
    }
    pub fn storage(&mut self, name: &str, size: u64, data: Option<&[u8]>, extra: wgpu::BufferUsages) -> Arc<wgpu::Buffer> { self.buffer(name, size, STORAGE | extra, data) }
    pub fn uniform(&mut self, name: &str, size: u64) -> Arc<wgpu::Buffer> { self.buffer(name, size, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, None) }
    pub fn get(&self, name: &str) -> Arc<wgpu::Buffer> { self.buffers.get(name).unwrap_or_else(|| panic!("no buffer {name}")).clone() }
    pub fn write(&self, name: &str, data: &[u8], offset: u64) {
        let n = data.len() & !3; if n == 0 { return; }
        self.queue.write_buffer(&self.buffers[name], offset, &data[..n]);
    }
    pub fn memory_bytes(&self) -> u64 { self.buffers.values().map(|b| b.size()).sum() }

    /// Compile one compute pipeline with explicit layouts from the binding declarations.
    pub fn kernel(&self, label: &str, code: &str, entry: &str) -> Result<Kernel, String> {
        let decls = parse_bindings(code);
        let ng = decls.iter().map(|d| d.group + 1).max().unwrap_or(0) as usize;
        let mut groups: Vec<Vec<Decl>> = vec![Vec::new(); ng];
        for d in decls { groups[d.group as usize].push(d); }
        let layouts: Vec<wgpu::BindGroupLayout> = groups.iter().enumerate().map(|(g, ds)| {
            let entries: Vec<wgpu::BindGroupLayoutEntry> = ds.iter().map(|d| wgpu::BindGroupLayoutEntry {
                binding: d.binding, visibility: wgpu::ShaderStages::COMPUTE, count: None,
                ty: wgpu::BindingType::Buffer { ty: if d.space == "uniform" { wgpu::BufferBindingType::Uniform } else { wgpu::BufferBindingType::Storage { read_only: d.access != "read_write" } },
                    has_dynamic_offset: false, min_binding_size: None },
            }).collect();
            self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(&format!("{label}/g{g}")), entries: &entries })
        }).collect();
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(code.into()) });
        let refs: Vec<Option<&wgpu::BindGroupLayout>> = layouts.iter().map(Some).collect();
        let pl = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(label), bind_group_layouts: &refs, immediate_size: 0 });
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label), layout: Some(&pl), module: &module, entry_point: Some(entry), compilation_options: Default::default(), cache: None });
        if let Some(e) = pollster::block_on(scope.pop()) { return Err(format!("kernel {label}: {e}")); }
        Ok(Kernel { label: label.to_string(), pipeline, layouts, groups })
    }

    /// Bind group g of kernel k, buffers looked up by variable name in `reg` (then the registry).
    pub fn bind_group(&self, k: &Kernel, reg: &HashMap<&str, Arc<wgpu::Buffer>>, g: usize) -> wgpu::BindGroup {
        let bufs: Vec<Arc<wgpu::Buffer>> = k.groups[g].iter().map(|d| reg.get(d.name.as_str()).cloned().or_else(|| self.buffers.get(&d.name).cloned())
            .unwrap_or_else(|| panic!("kernel {}: no buffer for binding '{}'", k.label, d.name))).collect();
        let entries: Vec<wgpu::BindGroupEntry> = k.groups[g].iter().zip(&bufs).map(|(d, b)| wgpu::BindGroupEntry { binding: d.binding, resource: b.as_entire_binding() }).collect();
        self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some(&format!("{}/bg{g}", k.label)), layout: &k.layouts[g], entries: &entries })
    }

    /// Blocking read of a buffer range (copy to a fresh MAP_READ buffer).
    pub fn read(&self, name: &str, offset: u64, size: u64) -> Vec<u8> {
        let size = (size + 3) / 4 * 4;
        let rb = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("read"), size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.buffers[name], offset, &rb, 0, size);
        self.queue.submit([enc.finish()]);
        rb.slice(..).map_async(wgpu::MapMode::Read, |r| r.expect("map"));
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let v = rb.slice(..).get_mapped_range().expect("range").to_vec();
        rb.unmap(); rb.destroy(); v
    }

    /// Read `out.len()` bytes at `offset` of a named buffer into `out` through a persistent staging buffer
    /// (no per-call allocation; only the requested range is copied).
    pub fn read_into(&self, name: &str, offset: u64, out: &mut [u8]) {
        let size = ((out.len() as u64) + 3) / 4 * 4; if size == 0 { return; }
        let mut st = self.staging.lock().unwrap();
        let need_new = st.get(name).map_or(true, |b| b.size() < size);
        if need_new {
            let b = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("staging"), size: size.max(self.buffers[name].size().min(size * 2)), usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            st.insert(name.to_string(), b);
        }
        let rb = &st[name];
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.buffers[name], offset, rb, 0, size);
        self.queue.submit([enc.finish()]);
        let sl = rb.slice(..size);
        sl.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        { let m = sl.get_mapped_range().expect("range"); out.copy_from_slice(&m[..out.len()]); }
        rb.unmap();
    }
}
