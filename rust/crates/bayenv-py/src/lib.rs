//! Python module `bayenv` (`bayenv._bayenv`): the Bay Area driving env behind one API.
//!
//! ```python
//! import bayenv, numpy as np
//! env = bayenv.Env("data", n_cars=10000, seed=11)        # backend="gpu" (wgpu: Metal / Vulkan / DX12)
//! obs = env.observe()                                    # float32 (n_slots, obs_size), copied straight from the GPU
//! env.act(np.tile([0.0, 0.3, 0.0], (env.n_slots, 1)).astype(np.float32))
//! env.step(20)                                           # 20 steps in one submit, GIL released
//! st = env.state()                                       # dict of numpy arrays per slot
//! ```
use bayenv_gpu::defines::ObsMode;
use bayenv_gpu::layout::{self as L, flag, state as vstate, veh};
use bayenv_gpu::sim::{GpuOptions, GpuSim};
use bayenv_world::{LoadOptions, World};
use bayenv_gpu::obs::{ObsDst, ObsFmt, ObsRing};
use numpy::ndarray::{self, IxDyn};
use numpy::{PyArray1, PyArray2, PyArrayDyn, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;

mod cpu;
use cpu::{CpuBackend, CpuObs};
enum Backend { Gpu(Box<GpuSim>), Cpu(Box<std::sync::Mutex<CpuBackend>>) }

/// Multi-agent driving env (every car is an agent; slot i = car i).
#[pyclass(module = "bayenv._bayenv")]
struct Env {
    backend: Backend,
    world: Arc<World>,
    /// slots in use (scatter places cars in slots 0..n)
    rows: usize,
    obs_size: usize,
    cap: usize,
    /// observation readback ring (gpu backend)
    ring: Option<ObsRing>,
    /// bumped by calls that change the sim state outside a step (readback tags = steps + gen << 48)
    gen: u64,
}

fn gpu_mut(env: &mut Env) -> &mut GpuSim { match &mut env.backend { Backend::Gpu(s) => s, Backend::Cpu(_) => panic!("gpu-only call on the cpu backend") } }
fn gpu_ref(env: &Env) -> &GpuSim { match &env.backend { Backend::Gpu(s) => s, Backend::Cpu(_) => panic!("gpu-only call on the cpu backend") } }
fn cpu_backend(env: &Env) -> Option<std::sync::MutexGuard<'_, CpuBackend>> { match &env.backend { Backend::Cpu(e) => Some(e.lock().unwrap()), _ => None } }
/// the CPU env behind the lock
struct CpuGuard<'a>(std::sync::MutexGuard<'a, CpuBackend>);
impl<'a> std::ops::Deref for CpuGuard<'a> { type Target = bayenv_sim::env::Env; fn deref(&self) -> &Self::Target { &self.0.env } }
fn cpu_only(env: &Env) -> Option<CpuGuard<'_>> { cpu_backend(env).map(CpuGuard) }
fn py_sync(sim: &mut GpuSim) { sim.sync(); }

/// Keeps an observation ring buffer alive while a zero-copy numpy view of it exists (the view's `base`).
#[pyclass(module = "bayenv._bayenv", frozen)]
struct ObsBuffer { _buf: Arc<wgpu::Buffer> }

fn parse_fmt(dtype: &str) -> PyResult<ObsFmt> {
    match dtype { "float32" | "f4" | "f32" => Ok(ObsFmt::F32), "float16" | "f2" | "f16" | "half" => Ok(ObsFmt::F16),
        _ => Err(PyValueError::new_err("dtype must be 'float32' or 'float16'")) }
}

/// CPU observations: float32 `arr` as is, or converted to float16 (into `out` when given).
fn cpu_finish<'py>(py: Python<'py>, arr: Bound<'py, PyAny>, out: Option<Bound<'py, PyAny>>, fmt: ObsFmt) -> PyResult<Bound<'py, PyAny>> {
    if fmt == ObsFmt::F32 { return Ok(arr); }
    let h = arr.call_method1("astype", ("float16",))?;
    match out { Some(o) => { o.call_method1("__setitem__", (py.Ellipsis(), h))?; Ok(o) } None => Ok(h) }
}

/// The ready readback of `ring` as numpy (shape `shape`, dtype of the readback): a read-only zero-copy view when
/// `copy=False` (Metal; a copy elsewhere), else a copy into `out` / a new array.
fn ring_out<'py>(py: Python<'py>, ring: &ObsRing, shape: &[usize], out: Option<Bound<'py, PyAny>>, copy: bool) -> PyResult<Bound<'py, PyAny>> {
    let (fmt, _) = ring.ready_info().ok_or_else(|| PyRuntimeError::new_err("no observation readback"))?;
    let n: usize = shape.iter().product();
    let f16 = fmt == ObsFmt::F16;
    if !copy && out.is_none() && ring.zero_copy_ok {
        let (ptr, buf) = ring.ready_ptr();
        let base = Bound::new(py, ObsBuffer { _buf: buf })?.into_any();
        // SAFETY: `ptr` is the mapped (Metal: MTLBuffer.contents, page aligned) range of a buffer of >= n elements,
        // kept alive by `base`; the view is read-only. Its contents change once the ring reuses the slot (documented).
        let arr = unsafe {
            if f16 { PyArrayDyn::<u16>::borrow_from_array(&ndarray::ArrayViewD::from_shape_ptr(IxDyn(shape), ptr as *const u16), base).into_any() }
            else { PyArrayDyn::<f32>::borrow_from_array(&ndarray::ArrayViewD::from_shape_ptr(IxDyn(shape), ptr as *const f32), base).into_any() }
        };
        let kw = PyDict::new(py); kw.set_item("write", false)?;
        arr.call_method("setflags", (), Some(&kw))?;
        return if f16 { arr.call_method1("view", ("float16",)) } else { Ok(arr) };
    }
    let (arr, ret): (Bound<'py, PyAny>, Bound<'py, PyAny>) = match out {
        Some(a) => {
            let sh: Vec<usize> = a.getattr("shape")?.extract()?;
            if sh != shape { return Err(PyValueError::new_err(format!("out has the wrong shape (expected {shape:?})"))); }
            let dt: String = a.getattr("dtype")?.str()?.extract()?;
            if dt != if f16 { "float16" } else { "float32" } { return Err(PyValueError::new_err(format!("out must be {}", if f16 { "float16" } else { "float32" }))); }
            (if f16 { a.call_method1("view", ("uint16",))? } else { a.clone() }, a)
        }
        None => {
            // SAFETY: fully overwritten below before it is returned
            let a = unsafe { if f16 { PyArrayDyn::<u16>::new(py, IxDyn(shape), false).into_any() } else { PyArrayDyn::<f32>::new(py, IxDyn(shape), false).into_any() } };
            let r = if f16 { a.call_method1("view", ("float16",))? } else { a.clone() };
            (a, r)
        }
    };
    let bytes = n * fmt.elem_bytes();
    if f16 {
        let a = arr.cast::<PyArrayDyn<u16>>()?; let mut rw = a.readwrite();
        let sl = rw.as_slice_mut().map_err(|_| PyValueError::new_err("out must be C-contiguous"))?;
        py.detach(|| ring.read_into(0, &mut bytemuck::cast_slice_mut(sl)[..bytes]));
    } else {
        let a = arr.cast::<PyArrayDyn<f32>>()?; let mut rw = a.readwrite();
        let sl = rw.as_slice_mut().map_err(|_| PyValueError::new_err("out must be C-contiguous"))?;
        py.detach(|| ring.read_into(0, &mut bytemuck::cast_slice_mut(sl)[..bytes]));
    }
    Ok(ret)
}

impl Env {
    fn obs_check(&self) -> PyResult<()> {
        if self.obs_size == 0 { return Err(PyRuntimeError::new_err("created with obs='none'")); }
        Ok(())
    }
    /// CPU backend observations (n_slots, width) into `out` / a new array (float16: converted from float32 on the host).
    fn cpu_obs<'py>(&mut self, py: Python<'py>, out: Option<Bound<'py, PyAny>>, fmt: ObsFmt) -> PyResult<Bound<'py, PyAny>> {
        let Backend::Cpu(e) = &mut self.backend else { unreachable!() };
        let e = e.get_mut().unwrap(); let (n, w) = (e.n_slots(), e.obs.width());
        let f32_out = if fmt == ObsFmt::F32 { out.clone() } else { None };
        let arr = match f32_out {
            Some(a) => { let a = a.cast_into::<PyArray2<f32>>().map_err(|_| PyValueError::new_err("out must be a float32 array"))?; if a.shape() != [n, w] { return Err(PyValueError::new_err("out has the wrong shape (n_slots, obs_size)")); } a }
            None => PyArray2::<f32>::zeros(py, [n, w], false),
        };
        { let mut rw = arr.readwrite(); let sl = rw.as_slice_mut().map_err(|_| PyValueError::new_err("out must be C-contiguous"))?; py.detach(|| e.observe_into(sl)); }
        cpu_finish(py, arr.into_any(), out, fmt)
    }
    fn parts(&mut self) -> (&mut GpuSim, &mut ObsRing) { match &mut self.backend { Backend::Gpu(s) => (&mut **s, self.ring.as_mut().unwrap()), _ => unreachable!() } }
    fn tag(&self) -> u64 { gpu_ref(self).step + (self.gen << 48) }
    /// Fresh readback of the current observations into the ring (waits for any pending batch first).
    fn snapshot(&mut self, py: Python<'_>, fmt: ObsFmt) {
        let (rows, d, tag) = (self.rows, self.obs_size, self.tag());
        let (sim, ring) = self.parts();
        py.detach(|| {
            let buf = ring.begin(rows * d * fmt.elem_bytes(), fmt, tag);
            sim.submit_obs(&ObsDst { buf, offset: 0, rows, fmt });
            ring.submitted(); ring.complete();
        });
    }
    /// Wait for the in-flight batch (events ingested) and its readback.
    fn wait_inflight(&mut self, py: Python<'_>) { let (sim, ring) = self.parts(); py.detach(|| { sim.sync(); ring.complete(); }); }
    /// Make the ring's ready readback show the current state in `fmt` (reuse it when it does).
    fn ensure_current(&mut self, py: Python<'_>, fmt: ObsFmt) {
        self.wait_inflight(py);
        let tag = self.tag();
        if self.ring.as_ref().unwrap().ready_info() != Some((fmt, tag)) { self.snapshot(py, fmt); }
    }
    /// Submit k steps with their readback, left pending (`step_async`, lag=1).
    fn submit(&mut self, py: Python<'_>, k: usize, fmt: ObsFmt) {
        let (rows, d, tag) = (self.rows, self.obs_size, self.tag() + k as u64);
        let (sim, ring) = self.parts();
        py.detach(|| {
            let buf = ring.begin(rows * d * fmt.elem_bytes(), fmt, tag);
            let dst = ObsDst { buf, offset: 0, rows, fmt };
            if k == 0 { sim.submit_obs(&dst); } else { sim.run_steps_async_obs(k, &dst); }
            ring.submitted();
        });
    }
}

#[pymethods]
impl Env {
    /// Env(data_dir, n_cars=10000, seed=1, backend="gpu" | "cpu", capacity=None, obs="v2", dt=0.05, start_time=None (gpu 05:00, cpu first departure - 60 s),
    ///     accidents=True, collisions=True, fuel=True, controls=True, junctions=True, steer="position", decide_every=1,
    ///     keep_events=False, population=None, pop_seed=42, streaming=False, return_trips=True, stations=True,
    ///     refuel_budget_ms=None, cpu_connector_join=True, contact_tangent=True, route_threads=None)
    /// Loads the world (~0.1 s warm), builds the GPU buffers and scatters `n_cars` cars at rest on random routes
    /// (seeded; the JS GPUSim.scatter placement). Actions start at [0, 0, 0] for every slot.
    /// population=N (gpu): instead of the scatter, N commuters (src/sim/population.js, seed pop_seed, generated natively,
    /// bit-identical with the JS) spawn on their trip schedule (outbound + return legs; GPUSim.setPopulation), with refuel
    /// detours through the world's gas stations (stations=False: every low-fuel request fails). streaming=True: a
    /// CommuterTable whose routes are computed at departure (GPUSim.setStreaming; for very large N). Slots are recycled:
    /// n_slots = capacity (default min(N, 131072) rounded up to 1024); slot_trip() maps slots to trips.
    /// refuel_budget_ms: wall-clock budget per step() call for refuel plans (default: none, every plan at once =
    /// deterministic; the JS default is 4 ms).
    /// cpu_connector_join (gpu, default True): junction connector entry as in the CPU sim (rust/docs/kerb-investigation.md).
    /// False = legacy mode, the JS GPU kernels bit-for-bit, with their connector-entry bug (all earlier GPU checkpoints).
    /// contact_tangent (gpu, default True): f32-robust friction tangent in the contact impulse (rust/docs/crash-divergence.md).
    /// False = legacy mode, the JS GPU kernels bit-for-bit (aligned rear-end wrecks spin to the kerb).
    /// replan_at_missed_event (cpu, default True): re-plan a missed turn at the event, as the GPU does
    /// (SimOpts::replan_at_missed_event, rust/docs/crash-divergence.md cause A). False = the JS CPU sim's model.
    /// route_threads (streaming): worker threads that route departures up to 300 sim s ahead (default: the machine's
    /// cores; 0 = serial A* at departure). Every value gives the same routes and the same spawn steps.
    #[new]
    #[pyo3(signature = (data_dir, n_cars=10000, seed=1, backend="gpu", capacity=None, obs="v2", dt=0.05, start_time=None,
        accidents=true, collisions=true, fuel=true, controls=true, junctions=true, steer="position", decide_every=1, keep_events=false,
        population=None, pop_seed=42.0, streaming=false, return_trips=true, stations=true, refuel_budget_ms=None, crash_share=false, missed_at_out_edge=false, cpu_connector_join=true, contact_tangent=true, replan_at_missed_event=true, route_threads=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(py: Python<'_>, data_dir: &str, n_cars: usize, seed: u32, backend: &str, capacity: Option<usize>, obs: &str, dt: f64, start_time: Option<f64>,
           accidents: bool, collisions: bool, fuel: bool, controls: bool, junctions: bool, steer: &str, decide_every: u32, keep_events: bool,
           population: Option<usize>, pop_seed: f64, streaming: bool, return_trips: bool, stations: bool, refuel_budget_ms: Option<f64>, crash_share: bool, missed_at_out_edge: bool, cpu_connector_join: bool, contact_tangent: bool, replan_at_missed_event: bool, route_threads: Option<usize>) -> PyResult<Env> {
        if backend == "cpu" {
            let cobs = match obs { "v2" => CpuObs::V2, "v3" => CpuObs::V3, "flat" => CpuObs::Flat, "none" => CpuObs::None, _ => return Err(PyValueError::new_err("obs must be 'v2', 'v3', 'flat' or 'none'")) };
            // CPU reference sim (crate bayenv-sim): commuter population (population.js, n_cars commuters, seed),
            // cars spawn on their trip schedule; slots = sim vehicle slots (0..high)
            let data = data_dir.to_string(); let fuel_on = fuel;
            let res: Result<(Arc<World>, bayenv_sim::env::Env), String> = py.detach(move || {
                let world = Arc::new(World::load(&data).map_err(|e| format!("world: {e}"))?);
                let sw = Arc::new(bayenv_sim::world::World::from_bayenv(&world));
                let pop = Arc::new(bayenv_sim::population::Population::generate_like_env(&sw, "real", n_cars, seed as f64, Some(std::path::Path::new(&data)))?);
                let mut opts = bayenv_sim::sim::SimOpts::default(); opts.dt = dt; opts.seed = seed as f64; opts.fuel = fuel_on; opts.replan_at_missed_event = replan_at_missed_event;
                let cfg = bayenv_sim::env::EnvConfig { opts, start_time, static_obstacles: Vec::new(), scenario_obstacles: Vec::new(), obstacles: None };
                let env = bayenv_sim::env::Env::new(sw, pop, cfg, Box::new(|w| bayenv_sim::harness::router_for(&w)));
                Ok((world, env))
            });
            let (world, env) = res.map_err(PyRuntimeError::new_err)?;
            let _ = (accidents, collisions, controls, junctions, steer, decide_every, keep_events, capacity);
            return Ok(Env { backend: Backend::Cpu(Box::new(std::sync::Mutex::new(CpuBackend::new(env, cobs)))), world, rows: 0, obs_size: cobs.width(), cap: 0, ring: None, gen: 0 });
        }
        let obs_mode = match obs { "v2" => ObsMode::V2, "v3" => ObsMode::V3, "none" => ObsMode::None, _ => return Err(PyValueError::new_err("obs must be 'v2', 'v3' or 'none'")) };
        if backend != "gpu" { return Err(PyValueError::new_err("backend must be 'gpu' or 'cpu'")); }
        // scatter fills slots 0..n_cars and nothing spawns later: size the slot pool to the cars (identical results,
        // every per-slot kernel dispatch shrinks); pass capacity= for headroom
        let cap = match population {
            Some(np) => capacity.unwrap_or(((np.min(131072) + 1023) / 1024 * 1024).max(1024)),
            None => capacity.unwrap_or(((n_cars + 1023) / 1024 * 1024).max(1024)).max(n_cars),
        };
        let data = data_dir.to_string();
        let steer_rate = match steer { "position" => false, "rate" => true, _ => return Err(PyValueError::new_err("steer must be 'position' or 'rate'")) };
        let res: Result<(Arc<World>, GpuSim, usize), String> = py.detach(move || {
            let world = Arc::new(World::load_with(&data, &LoadOptions::default()).map_err(|e| format!("world: {e}"))?);
            let mut o = GpuOptions { capacity: cap, dt, start_time: start_time.unwrap_or(18000.0), seed, obs: obs_mode, steer_rate, decide_every, controls, junctions, keep_events, record_share: crash_share, missed_at_out_edge,
                return_trips, refuel_budget_ms: refuel_budget_ms.unwrap_or(f64::INFINITY), cpu_connector_join, contact_tangent, ..Default::default() };
            if let Some(t) = route_threads { o.route_threads = t; }
            o.features.accidents = accidents; o.features.collisions = collisions; o.features.fuel = fuel;
            let mut sim = bayenv_gpu::env::create(world.clone(), o)?;
            match population {
                Some(np) if streaming => { let t = bayenv_gpu::sim::CommuterTable::generate(&world, np, pop_seed)?; sim.set_streaming(Arc::new(t), stations, None); Ok((world, sim, cap)) }
                Some(np) => { let p = bayenv_gpu::sim::GpuPopulation::generate(&world, np, pop_seed, return_trips)?; sim.set_population(Arc::new(p), stations); Ok((world, sim, cap)) }
                None => { let info = bayenv_gpu::env::scatter(&mut sim, n_cars, seed); Ok((world, sim, info.placed)) }
            }
        });
        let (world, sim, placed) = res.map_err(PyRuntimeError::new_err)?;
        let obs_size = sim.obs_size;
        let ring = Some(ObsRing::new(&sim.gpu.device, sim.gpu.adapter_info.backend));
        Ok(Env { backend: Backend::Gpu(Box::new(sim)), world, rows: placed, obs_size: if obs_mode == ObsMode::None { 0 } else { obs_size }, cap, ring, gen: 0 })
    }

    /// Number of slots with a car (rows of observe() / state()).
    #[getter] fn n_slots(&self) -> usize { match cpu_only(self) { Some(e) => e.sim.vs.high, None => self.rows } }
    #[getter] fn capacity(&self) -> usize { self.cap }
    #[getter] fn obs_size(&self) -> usize { self.obs_size }
    #[getter] fn time(&self) -> f64 { match cpu_only(self) { Some(e) => e.sim.time, None => gpu_ref(self).time } }
    #[getter] fn steps(&self) -> u64 { match cpu_only(self) { Some(e) => e.sim.step_count as u64, None => gpu_ref(self).step } }
    #[getter] fn dt(&self) -> f64 { match cpu_only(self) { Some(e) => e.sim.dt, None => gpu_ref(self).dt } }
    #[getter] fn backend(&self) -> &'static str { if cpu_only(self).is_some() { "cpu" } else { "gpu" } }
    #[getter] fn adapter(&self) -> String { if cpu_only(self).is_some() { "cpu".into() } else { gpu_ref(self).gpu.adapter_name() } }

    /// Observation vectors, float32 (n_slots, obs_size), sensed in the last step (src/gpu/README.md v2 layout).
    /// `out`: optional preallocated C-contiguous array of the result dtype to fill instead (no allocation).
    /// `copy=False`: return a read-only zero-copy view of the mapped readback buffer (Metal / unified memory; a copy
    /// on other backends) that stays valid until the next observe / step* call on this env (it is reused after
    /// that); `np.array(v)` it to keep it. `dtype='float16'`: converted on the GPU (half the bytes; see
    /// rust/BENCHMARKS.md for the per-field error).
    #[pyo3(signature = (out=None, copy=true, dtype="float32"))]
    fn observe<'py>(&mut self, py: Python<'py>, out: Option<Bound<'py, PyAny>>, copy: bool, dtype: &str) -> PyResult<Bound<'py, PyAny>> {
        self.obs_check()?;
        let fmt = parse_fmt(dtype)?;
        if cpu_only(self).is_some() { return self.cpu_obs(py, out, fmt); }
        self.snapshot(py, fmt);
        ring_out(py, self.ring.as_ref().unwrap(), &[self.rows, self.obs_size], out, copy)
    }

    /// Set actions: float32 (N, 3) [steer, gas, brake] for slots offset..offset+N. steer in [-1, 1] = target wheel
    /// angle / maxSteer (steer='position') or steer rate (steer='rate'); gas, brake in [0, 1]. Held until changed.
    #[pyo3(signature = (actions, offset=0))]
    fn act(&mut self, actions: PyReadonlyArray2<'_, f32>, offset: usize) -> PyResult<()> {
        if actions.shape()[1] != 3 { return Err(PyValueError::new_err("actions must have shape (N, 3)")); }
        if let Backend::Cpu(e) = &mut self.backend {
            let e = &mut e.get_mut().unwrap().env;
            let sl = actions.as_slice().map_err(|_| PyValueError::new_err("actions must be C-contiguous float32"))?;
            let n = actions.shape()[0];
            for i in 0..n { let slot = offset + i; if slot < e.sim.vs.high && e.sim.vs.state[slot] == bayenv_sim::vehicle::STATE_DRIVING {
                e.act(slot, bayenv_sim::sim::Action { steer: Some(sl[3 * i] as f64), steer_rate: None, gas: sl[3 * i + 1] as f64, brake: sl[3 * i + 2] as f64 }); } }
            return Ok(());
        }
        if offset + actions.shape()[0] > self.cap { return Err(PyValueError::new_err("more actions than slots")); }
        let sl = actions.as_slice().map_err(|_| PyValueError::new_err("actions must be C-contiguous float32"))?;
        gpu_mut(self).set_actions(sl, offset);
        Ok(())
    }

    /// Advance k steps (dt each) in one GPU submit, without the GIL; events / scenes are processed after the batch.
    #[pyo3(signature = (k=1))]
    fn step(&mut self, py: Python<'_>, k: usize) {
        if let Backend::Cpu(e) = &mut self.backend { let e = e.get_mut().unwrap(); py.detach(|| e.step(k)); return; }
        let sim = gpu_mut(self);
        py.detach(|| sim.run_steps_async(k));   // returns after the submit; the next call waits / ingests
    }

    /// Fused RL step: optional actions (N, 3) -> k steps -> observations, in one GPU submit and one wait
    /// (actions upload, steps and the observation copy share the command stream). Returns (n_slots, obs_size);
    /// `out`, `copy`, `dtype` as in observe().
    /// lag=1 (pipelined): returns the observation as of the *start* of this call (after the previous call's steps,
    /// i.e. before `actions` act) and leaves this call's k steps running on the GPU while Python works; the next
    /// call (or any other call) waits for them. So the obs returned by call t is one action stale: it does not
    /// reflect actions_t yet, and actions_{t+1} are chosen from it. Same trajectory as lag=0 for the same actions.
    #[pyo3(signature = (actions=None, k=1, out=None, copy=true, lag=0, dtype="float32"))]
    #[allow(clippy::too_many_arguments)]
    fn step_observe<'py>(&mut self, py: Python<'py>, actions: Option<PyReadonlyArray2<'py, f32>>, k: usize, out: Option<Bound<'py, PyAny>>, copy: bool, lag: u32, dtype: &str) -> PyResult<Bound<'py, PyAny>> {
        self.obs_check()?;
        let fmt = parse_fmt(dtype)?;
        if lag > 1 { return Err(PyValueError::new_err("lag must be 0 or 1")); }
        if cpu_only(self).is_some() {
            if lag == 1 {
                let o = self.cpu_obs(py, None, fmt)?;
                if let Some(a) = actions { self.act(a, 0)?; }
                self.step(py, k);
                return match out { Some(a) => { a.call_method1("__setitem__", (py.Ellipsis(), o))?; Ok(a) } None => Ok(o) };
            }
            if let Some(a) = actions { self.act(a, 0)?; }
            self.step(py, k); return self.cpu_obs(py, out, fmt);
        }
        if let Some(a) = actions { self.act(a, 0)?; }
        if lag == 1 {
            self.ensure_current(py, fmt);
            self.submit(py, k, fmt);
        } else if k == 0 { self.snapshot(py, fmt); }
        else {
            let (rows, d, gen) = (self.rows, self.obs_size, self.gen);
            let (sim, ring) = self.parts();
            py.detach(|| {
                let buf = ring.begin(rows * d * fmt.elem_bytes(), fmt, sim.step + k as u64 + (gen << 48));
                let p = sim.submit_steps_obs(k, 0, Some(&ObsDst { buf, offset: 0, rows, fmt }), 0);
                ring.submitted();
                sim.gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
                sim.finish_steps(p, None, None);
                ring.complete();
            });
        }
        ring_out(py, self.ring.as_ref().unwrap(), &[self.rows, self.obs_size], out, copy)
    }

    /// Submit actions (optional) + k steps + their observation readback and return at once (GIL released, GPU
    /// running); collect the observations with observe_wait(). Other calls (state(), events(), ...) wait for the
    /// batch implicitly. The readback of the previous batch stays valid while this one runs.
    #[pyo3(signature = (actions=None, k=1, dtype="float32"))]
    fn step_async(&mut self, py: Python<'_>, actions: Option<PyReadonlyArray2<'_, f32>>, k: usize, dtype: &str) -> PyResult<()> {
        self.obs_check()?;
        let fmt = parse_fmt(dtype)?;
        if let Some(a) = actions { self.act(a, 0)?; }
        if cpu_only(self).is_some() { self.step(py, k); return Ok(()); }   // CPU: steps synchronously
        self.submit(py, k, fmt);
        Ok(())
    }

    /// Wait for the last step_async() batch and return its observations (dtype as submitted); without one in
    /// flight, the current observations (as observe(dtype=dtype)). `out` / `copy` as in observe().
    #[pyo3(signature = (out=None, copy=true, dtype="float32"))]
    fn observe_wait<'py>(&mut self, py: Python<'py>, out: Option<Bound<'py, PyAny>>, copy: bool, dtype: &str) -> PyResult<Bound<'py, PyAny>> {
        self.obs_check()?;
        let fmt = parse_fmt(dtype)?;
        if cpu_only(self).is_some() { return self.cpu_obs(py, out, fmt); }
        if self.ring.as_ref().unwrap().has_inflight() { self.wait_inflight(py); } else { self.ensure_current(py, fmt); }
        ring_out(py, self.ring.as_ref().unwrap(), &[self.rows, self.obs_size], out, copy)
    }

    /// True when observe(copy=False) & co. return zero-copy views (Metal / unified memory); else they copy.
    #[getter] fn zero_copy(&self) -> bool { self.ring.as_ref().is_some_and(|r| r.zero_copy_ok) }

    /// Wait for the last step() batch and process its events (other calls do this implicitly).
    fn sync(&mut self, py: Python<'_>) { if cpu_only(self).is_some() { return; } let sim = gpu_mut(self); py.detach(|| sim.sync()); }

    /// Per-slot vehicle state as numpy arrays (one GPU readback): edge, s, speed, accel, lat, heading_rel, steer,
    /// state (0 free, 1 driving, 2 crashed / stranded, 5 refuelling), flags, crashed, disabled, collided, lane,
    /// route_pos, odometer, progress_m (route metres driven), speed_limit (m/s, current edge), fuel, gas, brake, yaw_rate, v_lat,
    /// x, y, z, heading. `collided` = FLAG_COLLIDED at sense time of the last step (the JS external-protocol training
    /// signal; act.wgsl clears the flag right after), `collided_steps` = steps of the last step() call with it set.
    fn state<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        if let Some(e) = cpu_only(self) {
            let d = PyDict::new(py);
            for (name, col) in e.state_columns() {
                match name {
                    "edge" | "route_pos" | "lane" | "commuter" | "leader" => d.set_item(name, PyArray1::from_vec(py, col.iter().map(|&x| x as i32).collect::<Vec<i32>>()))?,
                    "state" | "flags" => d.set_item(name, PyArray1::from_vec(py, col.iter().map(|&x| x as u8).collect::<Vec<u8>>()))?,
                    "collided" => d.set_item(name, PyArray1::from_vec(py, col.iter().map(|&x| x != 0.0).collect::<Vec<bool>>()))?,
                    _ => d.set_item(name, PyArray1::from_vec(py, col.iter().map(|&x| x as f32).collect::<Vec<f32>>()))?,
                }
            }
            let v = &e.sim.vs; let n = v.high;
            d.set_item("crashed", PyArray1::from_vec(py, v.state[..n].iter().map(|&s| s == bayenv_sim::vehicle::STATE_CRASHED).collect::<Vec<bool>>()))?;
            d.set_item("disabled", PyArray1::from_vec(py, v.flags[..n].iter().map(|&f| f & bayenv_sim::vehicle::FLAG_DISABLED != 0).collect::<Vec<bool>>()))?;
            return Ok(d);
        }
        if let Backend::Gpu(sim) = &mut self.backend { py_sync(sim); }
        let n = self.rows; let sim = gpu_ref(self);
        let mut vehw = vec![0u32; n * L::VEH_WORDS]; let mut pose = vec![0f32; n * 4];
        py.detach(|| { sim.read_into("veh", 0, bytemuck::cast_slice_mut(&mut vehw)); sim.read_into("pose", 0, bytemuck::cast_slice_mut(&mut pose)); });
        let d = PyDict::new(py);
        let col_u = |k: usize| -> Vec<u32> { (0..n).map(|i| vehw[i * L::VEH_WORDS + k]).collect() };
        let col_f = |k: usize| -> Vec<f32> { (0..n).map(|i| f32::from_bits(vehw[i * L::VEH_WORDS + k])).collect() };
        for (name, k) in [("edge", veh::EDGE), ("state", veh::STATE), ("flags", veh::FLAGS), ("route_pos", veh::ROUTE_POS)] { d.set_item(name, PyArray1::from_vec(py, col_u(k)))?; }
        for (name, k) in [("s", veh::S), ("speed", veh::SPEED), ("accel", veh::ACCEL), ("lat", veh::LAT), ("heading_rel", veh::HEADING_REL), ("steer", veh::STEER_ANGLE),
            ("odometer", veh::ODOMETER), ("progress_m", veh::ROUTE_DIST), ("fuel", veh::FUEL), ("gas", veh::GAS), ("brake", veh::BRAKE), ("yaw_rate", veh::YAW_RATE), ("v_lat", veh::V_LAT)] {
            d.set_item(name, PyArray1::from_vec(py, col_f(k)))?;
        }
        let st = col_u(veh::STATE); let fl = col_u(veh::FLAGS);
        d.set_item("crashed", PyArray1::from_vec(py, st.iter().map(|&s| s == vstate::CRASHED).collect::<Vec<bool>>()))?;
        d.set_item("disabled", PyArray1::from_vec(py, fl.iter().map(|&f| f & flag::DISABLED != 0).collect::<Vec<bool>>()))?;
        // JS external protocol signal: FLAG_COLLIDED at sense time (kernels/collided.wgsl, between sense and act)
        let mut cf = vec![0u32; 2 * n]; if n > 0 { sim.read_into("cflag", 0, bytemuck::cast_slice_mut(&mut cf)); }
        d.set_item("collided", PyArray1::from_vec(py, (0..n).map(|i| cf[2 * i + 1] != 0).collect::<Vec<bool>>()))?;
        d.set_item("collided_steps", PyArray1::from_vec(py, (0..n).map(|i| cf[2 * i]).collect::<Vec<u32>>()))?;
        d.set_item("lane", PyArray1::from_vec(py, col_u(veh::LANES).iter().map(|&w| (w & 255) as u8).collect::<Vec<u8>>()))?;
        // CPU state() parity (rust/docs/cpu-vs-gpu.md "Crash divergence dumps"): footprint lanes, leader / gap, junction
        // connector (vConn = taper - 1 with junctions, common.wgsl), next control (ctl words 1 type, 2 light, 5 dist;
        // controls.wgsl), agent = trip id of the slot (2 * commuter + leg)
        d.set_item("lane_lo", PyArray1::from_vec(py, col_u(veh::LANES).iter().map(|&w| ((w >> 8) & 255) as f32).collect::<Vec<f32>>()))?;
        d.set_item("lane_hi", PyArray1::from_vec(py, col_u(veh::LANES).iter().map(|&w| ((w >> 16) & 255) as f32).collect::<Vec<f32>>()))?;
        d.set_item("leader", PyArray1::from_vec(py, col_u(veh::LEADER).iter().map(|&w| w as i32).collect::<Vec<i32>>()))?;
        d.set_item("gap", PyArray1::from_vec(py, col_f(veh::GAP)))?;
        let hj = sim.has_junctions();
        d.set_item("conn", PyArray1::from_vec(py, col_f(veh::TAPER).iter().map(|&t| if hj { t - 1.0 } else { -1.0 }).collect::<Vec<f32>>()))?;
        let mut ctlw = vec![0u32; n * L::CTL_WORDS]; if n > 0 { sim.read_into("ctl", 0, bytemuck::cast_slice_mut(&mut ctlw)); }
        d.set_item("ctl_type", PyArray1::from_vec(py, (0..n).map(|i| ctlw[i * L::CTL_WORDS + 1] as i32 as f32).collect::<Vec<f32>>()))?;
        d.set_item("ctl_state", PyArray1::from_vec(py, (0..n).map(|i| ctlw[i * L::CTL_WORDS + 2] as f32).collect::<Vec<f32>>()))?;
        d.set_item("ctl_dist", PyArray1::from_vec(py, (0..n).map(|i| f32::from_bits(ctlw[i * L::CTL_WORDS + 5])).collect::<Vec<f32>>()))?;
        let trip: Vec<i32> = (0..n).map(|i| sim.slot_trip.get(i).copied().unwrap_or(-1)).collect();
        d.set_item("commuter", PyArray1::from_vec(py, trip.iter().map(|&t| if t >= 0 { t >> 1 } else { -1 }).collect::<Vec<i32>>()))?;
        d.set_item("leg", PyArray1::from_vec(py, trip.iter().map(|&t| if t >= 0 { (t & 1) as f32 } else { -1.0 }).collect::<Vec<f32>>()))?;
        d.set_item("agent", PyArray1::from_vec(py, trip))?;
        // speed limit (m/s) of each car's current edge (reward.js normalises route progress by it)
        let g = &self.world.graph;
        d.set_item("speed_limit", PyArray1::from_vec(py, col_u(veh::EDGE).iter().zip(&st).map(|(&e, &s)| if s != 0 && (e as usize) < g.edge_count { g.speed_limit[e as usize] as f32 } else { f32::NAN }).collect::<Vec<f32>>()))?;
        for (name, j) in [("x", 0), ("y", 1), ("z", 2), ("heading", 3)] { d.set_item(name, PyArray1::from_vec(py, (0..n).map(|i| pose[i * 4 + j]).collect::<Vec<f32>>()))?; }
        Ok(d)
    }

    /// Crash flags per slot: bool (n_slots,) = crashed (stopped wreck) or disabled (own delta-v >= 10 km/h).
    fn crash_flags<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        if let Some(e) = cpu_only(self) { let v = &e.sim.vs; return PyArray1::from_vec(py, (0..v.high).map(|i| v.state[i] == bayenv_sim::vehicle::STATE_CRASHED || v.flags[i] & bayenv_sim::vehicle::FLAG_DISABLED != 0).collect()); }
        if let Backend::Gpu(sim) = &mut self.backend { py_sync(sim); }
        let n = self.rows; let sim = gpu_ref(self); let mut vehw = vec![0u32; n * L::VEH_WORDS];
        py.detach(|| sim.read_into("veh", 0, bytemuck::cast_slice_mut(&mut vehw)));
        PyArray1::from_vec(py, (0..n).map(|i| vehw[i * L::VEH_WORDS + veh::STATE] == vstate::CRASHED || vehw[i * L::VEH_WORDS + veh::FLAGS] & flag::DISABLED != 0).collect())
    }

    /// Raw GPU buffer as uint32 words ('veh' cap x 40, 'par' cap x 20, 'pose' cap x 4 floats, ...; src/gpu/layout.js).
    fn raw<'py>(&mut self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyArray1<u32>>> {
        if cpu_only(self).is_some() { return Err(PyNotImplementedError::new_err("gpu backend only")); }
        if let Backend::Gpu(g) = &mut self.backend { py_sync(g); } let sim = gpu_ref(self);
        if !sim.gpu.buffers.contains_key(name) { return Err(PyValueError::new_err(format!("no buffer {name}"))); }
        let w = py.detach(|| sim.read_words(name));
        Ok(PyArray1::from_vec(py, w))
    }

    /// Traffic signals / signs at sim time `time` (default now), per approach (stop line): edge, stop_at (m along
    /// the edge), control, type (0 signal, 1 stop, 2 give way, 3 ramp meter, 4 crossing), state (0 green, 1 yellow,
    /// 2 red, 3 sign, 4 dark, 255 approach not on the graph) for the through movement, time_in, time_left (s).
    #[pyo3(signature = (time=None))]
    fn signals<'py>(&self, py: Python<'py>, time: Option<f64>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        let Some(c) = self.world.controls.as_ref() else { return Ok(d) };
        let t = time.unwrap_or_else(|| match cpu_only(self) { Some(e) => e.sim.time, None => gpu_ref(self).time }); let n = c.approach_count;
        let (mut stt, mut ti, mut tl, mut ty) = (vec![0u8; n], vec![0f64; n], vec![0f64; n], vec![0i8; n]);
        for a in 0..n { if c.ap_edge[a] < 0 { stt[a] = 255; continue; } let s = c.state_of(a, 0, t); stt[a] = s.state; ti[a] = s.time_in; tl[a] = s.time_left; ty[a] = s.ty; }
        d.set_item("edge", PyArray1::from_vec(py, c.ap_edge.clone()))?; d.set_item("stop_at", PyArray1::from_vec(py, c.ap_stop_at.clone()))?;
        d.set_item("control", PyArray1::from_vec(py, c.ap_control.clone()))?; d.set_item("type", PyArray1::from_vec(py, ty))?;
        d.set_item("state", PyArray1::from_vec(py, stt))?; d.set_item("time_in", PyArray1::from_vec(py, ti))?; d.set_item("time_left", PyArray1::from_vec(py, tl))?;
        Ok(d)
    }

    /// Aggregates: GPU stats (driving, arrived, ...) and host counters (crashes, contacts, scenes, ...).
    fn stats<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        if let Some(e) = cpu_only(self) { let d = PyDict::new(py); for (k, v) in e.stats() { d.set_item(k, v)?; } return Ok(d); }
        if let Backend::Gpu(g) = &mut self.backend { py_sync(g); } let sim = gpu_ref(self); let s = sim.stats(); let c = &sim.counters; let d = PyDict::new(py);
        use L::stat as S;
        for (k, v) in [("driving", s[S::LIVE_DRIVING] as u64), ("crashed_on_scene", s[S::LIVE_CRASHED] as u64), ("arrived", s[S::ARRIVED] as u64), ("spawned", s[S::SPAWNED] as u64),
            ("stuck", s[S::STUCK] as u64), ("bumps", s[S::BUMPS] as u64), ("policy_steps", s[S::LIVE_POLICY] as u64),
            ("crash_events", c.crash_events), ("crash_scenes", c.crashes), ("crashed_vehicles", c.crashed_vehicles), ("minor_contacts", c.minor_contacts),
            ("stranded", c.stranded), ("towed", c.towed), ("cleared", c.cleared), ("missed_exits", c.missed_exits), ("reroutes", c.reroutes),
            ("signal_violations", c.signal_violations), ("stop_violations", c.stop_violations), ("events_dropped", c.events_dropped)] { d.set_item(k, v)?; }
        let drv = s[S::LIVE_DRIVING];
        d.set_item("avg_speed", if drv > 0 { s[S::LIVE_SUMSPEED] as f64 / 100.0 / drv as f64 } else { 0.0 })?;
        d.set_item("active_scenes", sim.active.len())?; d.set_item("time", sim.time)?;
        for (k, v) in [("refuel_planned", c.refuel_planned), ("refuel_plan_failed", c.refuel_plan_failed), ("refuelled", c.refuelled), ("reentered", c.reentered), ("no_route", c.no_route)] { d.set_item(k, v)?; }
        d.set_item("refueling", s[S::LIVE_DWELL])?; d.set_item("in_station_lots", sim.lot())?;
        let nc = sim.x.commuters();
        if nc > 0 {
            d.set_item("commuters", nc)?; d.set_item("trips_remaining", sim.trips_remaining())?;
            let mut fb = [0usize; 6]; for &o in &sim.x.outcome { fb[(o as usize).min(5)] += 1; }
            for (k, i) in [("outcome_arrived", 1), ("outcome_crash", 2), ("outcome_stranded", 3), ("outcome_stuck", 4), ("outcome_no_route", 5), ("outcome_unfinished", 0)] { d.set_item(k, fb[i])?; }
        }
        if let Some(r) = sim.route_stats() {
            let ss = sim.x.stream_stats;
            for (k, v) in [("route_threads", sim.opts.route_threads as u64), ("route_lookups", r.lookups), ("route_cache_hits", r.hits), ("route_sync", r.misses), ("route_prefetched", r.prefetched),
                ("route_waited", r.waited), ("stream_routed", ss.routed), ("stream_pool_full", ss.pool_full)] { d.set_item(k, v)?; }
            d.set_item("route_wait_s", r.wait_ns as f64 * 1e-9)?; d.set_item("route_sync_s", r.route_ns as f64 * 1e-9)?; d.set_item("spawn_hash", format!("{:016x}", ss.spawn_hash))?;
        }
        Ok(d)
    }

    /// Per-slot events since the last call, as a dict of equal-length numpy arrays (one row per event and car):
    /// step (global step index; rows sorted by (step, slot, kind)), time (s; arrivals: exact arrival time, others: end of
    /// their step), kind (1 arrived, 2 crashed = this car
    /// disabled, 3 contact, 4 stranded, 5 cleared, 6 spawned, 7 missed exit, 8 signal / stop violation), slot,
    /// agent (trip id 2 * commuter + leg, -1 none), other (partner slot, -1 barrier / none), dv_kmh, other_dv_kmh,
    /// share (fault share, needs crash_share=True), late_by (arrived, populations: arrival - workStart s),
    /// progress_m (route metres at the event), trip_m / trip_s (arrived). Missing values are NaN.
    fn events<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        if let Backend::Cpu(e) = &mut self.backend { let r = e.get_mut().unwrap().env.take_records(); return events_dict(py, &r, None); }
        let r = gpu_mut(self).take_records();
        events_dict(py, &r, None)
    }

    /// Host log of scene / outcome events (needs keep_events=True), drained: list of (time, kind, slot, a, b).
    fn event_log(&mut self) -> Vec<(f64, &'static str, i64, f64, f64)> {
        if cpu_only(self).is_some() { return Vec::new(); }
        if let Backend::Gpu(sim) = &mut self.backend { py_sync(sim); }
        gpu_mut(self).events.drain(..).map(|e| (e.time, e.kind, e.slot, e.a, e.b)).collect()
    }

    /// Start a new simulated day (GPUSim.resetDay): every slot freed, scenes / queues / counters / agent results cleared,
    /// clock = start_time (default: the constructor's). Populations replay their trip schedule.
    #[pyo3(signature = (start_time=None))]
    fn reset_day(&mut self, py: Python<'_>, start_time: Option<f64>) -> PyResult<()> {
        if cpu_only(self).is_some() { return Err(PyNotImplementedError::new_err("reset_day: gpu backend only")); }
        self.gen += 1; let sim = gpu_mut(self); py.detach(|| sim.reset_day(start_time)); Ok(())
    }

    /// Training obstacles (GPUSim.placeObstacles): `count` obstacles cycling through `kinds` (default all of 'midblock',
    /// 'afterCurve', 'afterMerge', 'junction', 'rampEnd', 'partial', 'slow', 'braking'), seeded; or an explicit list of
    /// dicts {edge, at, lane | lat, type} (stopped, disabled cars). Returns [(slot, kind, edge, s)].
    #[pyo3(signature = (count=40, kinds=None, seed=1.0, slow_kmh=(10.0, 20.0), brake_every=(15.0, 40.0), obstacles=None))]
    #[allow(clippy::too_many_arguments)]
    fn place_obstacles(&mut self, count: usize, kinds: Option<Vec<String>>, seed: f64, slow_kmh: (f64, f64), brake_every: (f64, f64),
                       obstacles: Option<Vec<Bound<'_, PyDict>>>) -> PyResult<Vec<(u32, &'static str, u32, f64)>> {
        if cpu_only(self).is_some() { return Err(PyNotImplementedError::new_err("place_obstacles: gpu backend only")); }
        let list = match obstacles {
            Some(list) => {
                let mut v = Vec::new();
                for d in list {
                    let g = |k: &str| -> PyResult<Option<f64>> { match d.get_item(k)? { Some(x) if !x.is_none() => Ok(Some(x.extract::<f64>()?)), _ => Ok(None) } };
                    v.push(bayenv_gpu::sim::StaticObstacle { edge: g("edge")?.ok_or_else(|| PyValueError::new_err("obstacle needs 'edge'"))? as i64, at: g("at")?.unwrap_or(0.0),
                        lane: g("lane")?, lat: g("lat")?, ty: g("type")?.unwrap_or(0.0) as u32 });
                }
                Some(v)
            }
            None => None,
        };
        let kinds = match kinds { None => bayenv_gpu::sim::ObstacleKind::ALL.to_vec(),
            Some(ks) => ks.iter().map(|k| bayenv_gpu::sim::ObstacleKind::parse(k).ok_or_else(|| PyValueError::new_err(format!("unknown obstacle kind '{k}'")))).collect::<PyResult<Vec<_>>>()? };
        let sim = gpu_mut(self); sim.sync();
        let placed = match list { Some(v) => sim.place_static_obstacles(&v),
            None => sim.place_obstacles(&bayenv_gpu::sim::ObstacleSpec { count, kinds, seed, slow_kmh: [slow_kmh.0, slow_kmh.1], brake_every: [brake_every.0, brake_every.1] }) };
        Ok(placed.iter().map(|p| (p.slot, p.kind.name(), p.edge, p.s)).collect())
    }

    /// n static wrecks on random main-tier edges for the whole day (GPUSim.placeWrecks). Returns the number placed.
    fn place_wrecks(&mut self, n: usize) -> PyResult<usize> {
        if cpu_only(self).is_some() { return Err(PyNotImplementedError::new_err("place_wrecks: gpu backend only")); }
        let sim = gpu_mut(self); sim.sync(); Ok(sim.place_wrecks(n))
    }

    /// Per commuter outcome of the outbound leg (host record): 0 unfinished, 1 arrived, 2 crash, 3 stranded, 4 stuck, 5 no route.
    fn outcomes<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
        if let Some(e) = cpu_only(self) { return PyArray1::from_slice(py, e.outcomes()); }
        let sim = gpu_mut(self); sim.sync(); PyArray1::from_slice(py, &sim.x.outcome)
    }

    /// Per-agent day results from the GPU (GPUSim.readAgents): dict of reward_sum, outcome, outcome_back, progress,
    /// trip_time, crashes (one row per agent), plus fuel_left (litres after the commuter's last finished leg, -1 none).
    fn agents<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        if cpu_only(self).is_some() { return Err(PyNotImplementedError::new_err("agents: gpu backend only")); }
        let sim = gpu_mut(self); let a = py.detach(|| sim.read_agents()); let d = PyDict::new(py);
        d.set_item("reward_sum", PyArray1::from_vec(py, a.reward_sum))?; d.set_item("outcome", PyArray1::from_vec(py, a.outcome))?;
        d.set_item("outcome_back", PyArray1::from_vec(py, a.outcome_back))?; d.set_item("progress", PyArray1::from_vec(py, a.progress))?;
        d.set_item("trip_time", PyArray1::from_vec(py, a.trip_time))?; d.set_item("crashes", PyArray1::from_vec(py, a.crashes))?;
        d.set_item("fuel_left", PyArray1::from_slice(py, &gpu_ref(self).x.fuel_left))?;
        Ok(d)
    }

    /// Slot -> trip (int32, n_slots): 2 * commuter + leg for a car on a trip, -1 never used, -2 freed, -3 obstacle / wreck
    /// (scatter cars: 2 * slot + 1).
    fn slot_trip<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i32>>> {
        if cpu_only(self).is_some() { return Err(PyNotImplementedError::new_err("slot_trip: gpu backend only")); }
        let rows = self.rows; let sim = gpu_mut(self); sim.sync(); let n = rows.min(sim.slot_trip.len()); Ok(PyArray1::from_slice(py, &sim.slot_trip[..n]))
    }

    /// Load a CPU-sim snapshot (src/sim/state.js exportState as <path>.json + <path>.bin, the format of
    /// rust/tools/oracle/gpu_parity_mock.mjs --scenario=snapshot) slot for slot and take over its clock (GPUSim.importSnapshot).
    fn import_snapshot(&mut self, path: &str) -> PyResult<()> {
        if cpu_only(self).is_some() { return Err(PyNotImplementedError::new_err("import_snapshot: gpu backend only")); }
        let snap = bayenv_gpu::sim::Snapshot::load(std::path::Path::new(path)).map_err(PyRuntimeError::new_err)?;
        self.gen += 1; let cap = self.cap; let sim = gpu_mut(self); sim.sync(); sim.import_snapshot(&snap).map_err(PyRuntimeError::new_err)?;
        self.rows = self.rows.max(snap.high.min(cap)); Ok(())
    }

    fn __repr__(&self) -> String { format!("bayenv.Env(backend='{}', n_slots={}, obs_size={}, time={:.2})", self.backend(), self.n_slots(), self.obs_size, self.time()) }
}

/// Several independent envs on one GPU device sharing one world (the world is loaded and packed once).
/// Arrays carry a leading env axis: act((E, N, 3)), step_observe -> (E, N, obs_size), crash_flags -> (E, N).
/// Env i uses seed + i. All envs are submitted together and waited on once per step call.
#[pyclass(module = "bayenv._bayenv")]
struct VecEnv { sims: Vec<GpuSim>, cpu: Vec<std::sync::Mutex<CpuBackend>>, rows: usize, obs_size: usize, ring: Option<ObsRing> }

impl VecEnv {
    fn tag(&self) -> u64 { self.sims[0].step }
    fn ring(&self) -> &ObsRing { self.ring.as_ref().unwrap() }
    fn out<'py>(&self, py: Python<'py>, out: Option<Bound<'py, PyAny>>, copy: bool) -> PyResult<Bound<'py, PyAny>> { ring_out(py, self.ring(), &[self.sims.len(), self.rows, self.obs_size], out, copy) }
    fn wait(&mut self, py: Python<'_>) { let (sims, ring) = (&mut self.sims, self.ring.as_mut().unwrap()); py.detach(|| { for s in sims.iter_mut() { s.sync(); } ring.complete(); }); }
    fn ensure_current(&mut self, py: Python<'_>, fmt: ObsFmt) {
        self.wait(py);
        if self.ring().ready_info() != Some((fmt, self.tag())) { self.submit(py, 0, fmt, false); }
    }
    /// CPU envs: observations (E, n_slots, width), rows past an env's slot count zero.
    fn cpu_obs<'py>(&mut self, py: Python<'py>, out: Option<Bound<'py, PyAny>>, fmt: ObsFmt) -> PyResult<Bound<'py, PyAny>> {
        let (e, d) = (self.cpu.len(), self.obs_size); let r = self.n_slots();
        let f32_out = if fmt == ObsFmt::F32 { out.clone() } else { None };
        let arr = match f32_out {
            Some(a) => { let a = a.cast_into::<numpy::PyArray3<f32>>().map_err(|_| PyValueError::new_err("out must be a float32 array"))?; if a.shape() != [e, r, d] { return Err(PyValueError::new_err("out has the wrong shape (n_envs, n_slots, obs_size)")); } a }
            None => numpy::PyArray3::<f32>::zeros(py, [e, r, d], false),
        };
        { let mut rw = arr.readwrite(); let sl = rw.as_slice_mut().map_err(|_| PyValueError::new_err("out must be C-contiguous"))?; let c = &mut self.cpu;
          if r > 0 { py.detach(|| { use rayon::prelude::*; sl.par_chunks_mut(r * d).zip(c.par_iter_mut()).for_each(|(o, e)| { let e = e.get_mut().unwrap(); let n = e.n_slots(); o.fill(0.0); e.observe_into(&mut o[..n * d]); }); }); } }
        cpu_finish(py, arr.into_any(), out, fmt)
    }
    /// k steps (0: readback only) in every env with the readback into the next ring slot; lazy: leave it pending.
    fn submit(&mut self, py: Python<'_>, k: usize, fmt: ObsFmt, lazy: bool) {
        let (rows, d, e, tag) = (self.rows, self.obs_size, self.sims.len(), self.tag() + k as u64);
        let (sims, ring) = (&mut self.sims, self.ring.as_mut().unwrap());
        py.detach(|| {
            let buf = ring.begin(e * rows * d * fmt.elem_bytes(), fmt, tag);
            bayenv_gpu::env::step_many_obs(sims, k, &buf, rows, fmt, lazy || k == 0, || ring.submitted());
            if !lazy { ring.complete(); }
        });
    }
}

#[pymethods]
impl VecEnv {
    #[new]
    #[pyo3(signature = (data_dir, n_envs, n_cars=1000, seed=1, capacity=None, obs="v2", dt=0.05, start_time=None, accidents=true, collisions=true, fuel=true, backend="gpu", cpu_connector_join=true, contact_tangent=true))]
    #[allow(clippy::too_many_arguments)]
    fn new(py: Python<'_>, data_dir: &str, n_envs: usize, n_cars: usize, seed: u32, capacity: Option<usize>, obs: &str, dt: f64, start_time: Option<f64>,
           accidents: bool, collisions: bool, fuel: bool, backend: &str, cpu_connector_join: bool, contact_tangent: bool) -> PyResult<VecEnv> {
        if backend == "cpu" {
            let cobs = match obs { "v2" => CpuObs::V2, "v3" => CpuObs::V3, "flat" => CpuObs::Flat, "none" => CpuObs::None, _ => return Err(PyValueError::new_err("obs must be 'v2', 'v3', 'flat' or 'none'")) };
            let data = data_dir.to_string();
            let res: Result<Vec<bayenv_sim::env::Env>, String> = py.detach(move || {
                let world = World::load(&data).map_err(|e| format!("world: {e}"))?;
                let sw = Arc::new(bayenv_sim::world::World::from_bayenv(&world));
                use rayon::prelude::*;
                (0..n_envs).into_par_iter().map(|i| make_cpu_env(&sw, &data, n_cars, (seed + i as u32) as f64, dt, start_time, fuel)).collect()
            });
            let envs = res.map_err(PyRuntimeError::new_err)?;
            let _ = (capacity, accidents, collisions);
            return Ok(VecEnv { sims: Vec::new(), cpu: envs.into_iter().map(|e| std::sync::Mutex::new(CpuBackend::new(e, cobs))).collect(), rows: 0, obs_size: cobs.width(), ring: None });
        }
        let start_time = start_time.unwrap_or(18000.0);
        let obs_mode = match obs { "v2" => ObsMode::V2, "v3" => ObsMode::V3, "none" => ObsMode::None, _ => return Err(PyValueError::new_err("obs must be 'v2', 'v3' or 'none'")) };
        let cap = capacity.unwrap_or(((n_cars + 1023) / 1024 * 1024).max(1024)).max(n_cars);
        let data = data_dir.to_string();
        let res: Result<(Vec<GpuSim>, usize), String> = py.detach(move || {
            let world = Arc::new(World::load(&data).map_err(|e| format!("world: {e}"))?);
            let mut o = GpuOptions { capacity: cap, dt, start_time, obs: obs_mode, cpu_connector_join, contact_tangent, ..Default::default() };
            o.features.accidents = accidents; o.features.collisions = collisions; o.features.fuel = fuel;
            let mut sims = bayenv_gpu::env::create_many(world, &o, n_envs)?;
            let mut placed = usize::MAX;
            for (i, s) in sims.iter_mut().enumerate() { s.opts.seed = seed + i as u32; placed = placed.min(bayenv_gpu::env::scatter(s, n_cars, seed + i as u32).placed); }
            Ok((sims, placed))
        });
        let (sims, rows) = res.map_err(PyRuntimeError::new_err)?;
        let obs_size = if obs_mode == ObsMode::None { 0 } else { sims[0].obs_size };
        let ring = ObsRing::new(&sims[0].gpu.device, sims[0].gpu.adapter_info.backend);
        Ok(VecEnv { sims, cpu: Vec::new(), rows, obs_size, ring: Some(ring) })
    }
    #[getter] fn n_envs(&self) -> usize { if self.cpu.is_empty() { self.sims.len() } else { self.cpu.len() } }
    /// rows per env (CPU: the largest env's slot count; observe / crash_flags pad the others)
    #[getter] fn n_slots(&self) -> usize { if self.cpu.is_empty() { self.rows } else { self.cpu.iter().map(|e| e.lock().unwrap().n_slots()).max().unwrap_or(0) } }
    /// CPU: slot count of each env
    fn env_slots(&self) -> Vec<usize> { if self.cpu.is_empty() { vec![self.rows; self.sims.len()] } else { self.cpu.iter().map(|e| e.lock().unwrap().n_slots()).collect() } }
    #[getter] fn obs_size(&self) -> usize { self.obs_size }
    #[getter] fn time(&self) -> f64 { if self.cpu.is_empty() { self.sims[0].time } else { self.cpu[0].lock().unwrap().env.sim.time } }

    /// actions float32 (E, N, 3)
    fn act(&mut self, actions: numpy::PyReadonlyArray3<'_, f32>) -> PyResult<()> {
        if !self.cpu.is_empty() {
            let sh = actions.shape(); if sh[0] != self.cpu.len() || sh[2] != 3 { return Err(PyValueError::new_err("actions must have shape (n_envs, N, 3)")); }
            let sl = actions.as_slice().map_err(|_| PyValueError::new_err("actions must be C-contiguous float32"))?; let per = sh[1] * 3;
            for (i, e) in self.cpu.iter_mut().enumerate() { e.get_mut().unwrap().env.act_rows(&sl[i * per..(i + 1) * per], 0); }
            return Ok(());
        }
        let sh = actions.shape();
        if sh[0] != self.sims.len() || sh[2] != 3 || sh[1] > self.rows.max(sh[1]) { return Err(PyValueError::new_err("actions must have shape (n_envs, N, 3)")); }
        let sl = actions.as_slice().map_err(|_| PyValueError::new_err("actions must be C-contiguous float32"))?;
        let per = sh[1] * 3;
        for (i, s) in self.sims.iter_mut().enumerate() { if sh[1] > s.cap { return Err(PyValueError::new_err("more actions than slots")); } s.set_actions(&sl[i * per..(i + 1) * per], 0); }
        Ok(())
    }
    /// k steps in every env (one submit per env, one wait).
    #[pyo3(signature = (k=1))]
    fn step(&mut self, py: Python<'_>, k: usize) {
        if !self.cpu.is_empty() { use rayon::prelude::*; let c = &mut self.cpu; py.detach(|| c.par_iter_mut().for_each(|e| e.get_mut().unwrap().step(k))); return; }
        let (sims, ring) = (&mut self.sims, self.ring.as_mut().unwrap()); py.detach(|| { ring.complete(); bayenv_gpu::env::step_many(sims, k, Vec::new(), Vec::new()) }); }

    /// Optional actions (E, N, 3) -> k steps in every env -> observations (E, n_slots, obs_size), one submit per env and
    /// one wait. `out`, `copy`, `dtype`, `lag` as in Env.step_observe (all envs share one readback buffer, so
    /// copy=False is a single (E, n_slots, obs_size) view).
    #[pyo3(signature = (actions=None, k=1, out=None, copy=true, lag=0, dtype="float32"))]
    #[allow(clippy::too_many_arguments)]
    fn step_observe<'py>(&mut self, py: Python<'py>, actions: Option<numpy::PyReadonlyArray3<'py, f32>>, k: usize, out: Option<Bound<'py, PyAny>>, copy: bool, lag: u32, dtype: &str) -> PyResult<Bound<'py, PyAny>> {
        if self.obs_size == 0 { return Err(PyRuntimeError::new_err("created with obs='none'")); }
        let fmt = parse_fmt(dtype)?;
        if lag > 1 { return Err(PyValueError::new_err("lag must be 0 or 1")); }
        if let Some(a) = actions { self.act(a)?; }
        if !self.cpu.is_empty() {
            let o = if lag == 1 { Some(self.cpu_obs(py, None, fmt)?) } else { None };
            self.step(py, k);
            return match o { Some(o) => match out { Some(a) => { a.call_method1("__setitem__", (py.Ellipsis(), o))?; Ok(a) } None => Ok(o) }, None => self.cpu_obs(py, out, fmt) };
        }
        if lag == 1 { self.ensure_current(py, fmt); self.submit(py, k, fmt, true); } else { self.submit(py, k, fmt, false); }
        self.out(py, out, copy)
    }
    /// Submit actions (optional) + k steps in every env + the readback, return at once; observe_wait() collects.
    #[pyo3(signature = (actions=None, k=1, dtype="float32"))]
    fn step_async(&mut self, py: Python<'_>, actions: Option<numpy::PyReadonlyArray3<'_, f32>>, k: usize, dtype: &str) -> PyResult<()> {
        if self.obs_size == 0 { return Err(PyRuntimeError::new_err("created with obs='none'")); }
        let fmt = parse_fmt(dtype)?;
        if let Some(a) = actions { self.act(a)?; }
        if !self.cpu.is_empty() { self.step(py, k); return Ok(()); }   // CPU: steps synchronously
        self.submit(py, k, fmt, true); Ok(())
    }
    /// Observations (E, n_slots, obs_size) of the last step_async() batch (waits), else the current ones.
    #[pyo3(signature = (out=None, copy=true, dtype="float32"))]
    fn observe_wait<'py>(&mut self, py: Python<'py>, out: Option<Bound<'py, PyAny>>, copy: bool, dtype: &str) -> PyResult<Bound<'py, PyAny>> {
        if self.obs_size == 0 { return Err(PyRuntimeError::new_err("created with obs='none'")); }
        let fmt = parse_fmt(dtype)?;
        if !self.cpu.is_empty() { return self.cpu_obs(py, out, fmt); }
        if self.ring().has_inflight() { self.wait(py); } else { self.ensure_current(py, fmt); }
        self.out(py, out, copy)
    }
    /// Current observations (E, n_slots, obs_size) of every env.
    #[pyo3(signature = (out=None, copy=true, dtype="float32"))]
    fn observe<'py>(&mut self, py: Python<'py>, out: Option<Bound<'py, PyAny>>, copy: bool, dtype: &str) -> PyResult<Bound<'py, PyAny>> {
        if self.obs_size == 0 { return Err(PyRuntimeError::new_err("created with obs='none'")); }
        let fmt = parse_fmt(dtype)?;
        if !self.cpu.is_empty() { return self.cpu_obs(py, out, fmt); }
        self.submit(py, 0, fmt, false);
        self.out(py, out, copy)
    }
    #[getter] fn zero_copy(&self) -> bool { self.ring.as_ref().is_some_and(|r| r.zero_copy_ok) }
    /// crash flags (E, n_slots) bool
    fn crash_flags<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray2<bool>>> {
        if !self.cpu.is_empty() { let r = self.n_slots(); let e = self.cpu.len(); let mut out = vec![false; e * r];
            for (i, c) in self.cpu.iter().enumerate() { for (j, f) in c.lock().unwrap().env.crash_flags().into_iter().enumerate() { out[i * r + j] = f; } }
            return Ok(numpy::PyArray1::from_vec(py, out).reshape([e, r])?); }
        let (e, n) = (self.sims.len(), self.rows); let mut out = vec![false; e * n];
        let sims = &self.sims;
        py.detach(|| for (i, s) in sims.iter().enumerate() { let mut w = vec![0u32; n * L::VEH_WORDS]; s.read_into("veh", 0, bytemuck::cast_slice_mut(&mut w));
            for j in 0..n { out[i * n + j] = w[j * L::VEH_WORDS + veh::STATE] == vstate::CRASHED || w[j * L::VEH_WORDS + veh::FLAGS] & flag::DISABLED != 0; } });
        Ok(numpy::PyArray1::from_vec(py, out).reshape([e, n])?)
    }
    /// Per-slot events of every env since the last call (Env.events() columns + `env`).
    fn events<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let mut all = bayenv_gpu::events::EventLog::default(); let mut envs = Vec::new();
        let logs: Vec<bayenv_gpu::events::EventLog> = if self.cpu.is_empty() { self.sims.iter_mut().map(|s| s.take_records()).collect() } else { self.cpu.iter_mut().map(|e| e.get_mut().unwrap().env.take_records()).collect() };
        for (i, r) in logs.into_iter().enumerate() { envs.extend(std::iter::repeat(i as i32).take(r.len()));
            all.step.extend(r.step); all.time.extend(r.time); all.kind.extend(r.kind); all.slot.extend(r.slot); all.agent.extend(r.agent); all.other.extend(r.other);
            all.dv_kmh.extend(r.dv_kmh); all.other_dv_kmh.extend(r.other_dv_kmh); all.share.extend(r.share); all.late_by.extend(r.late_by);
            all.progress_m.extend(r.progress_m); all.trip_m.extend(r.trip_m); all.trip_s.extend(r.trip_s);
        }
        events_dict(py, &all, Some(envs))
    }
    fn __repr__(&self) -> String { format!("bayenv.VecEnv(backend='{}', n_envs={}, n_slots={}, obs_size={})", if self.cpu.is_empty() { "gpu" } else { "cpu" }, self.n_envs(), self.n_slots(), self.obs_size) }
}

/// One CPU env on a shared (converted) world: population.js commuters (n, seed), external actions.
fn make_cpu_env(sw: &Arc<bayenv_sim::world::World>, data: &str, n_cars: usize, seed: f64, dt: f64, start_time: Option<f64>, fuel: bool) -> Result<bayenv_sim::env::Env, String> {
    let pop = Arc::new(bayenv_sim::population::Population::generate_like_env(sw, "real", n_cars, seed, Some(std::path::Path::new(data)))?);
    let mut opts = bayenv_sim::sim::SimOpts::default(); opts.dt = dt; opts.seed = seed; opts.fuel = fuel;
    let cfg = bayenv_sim::env::EnvConfig { opts, start_time, static_obstacles: Vec::new(), scenario_obstacles: Vec::new(), obstacles: None };
    Ok(bayenv_sim::env::Env::new(sw.clone(), pop, cfg, Box::new(|w| bayenv_sim::harness::router_for(&w))))
}

fn events_dict<'py>(py: Python<'py>, r: &bayenv_gpu::events::EventLog, env: Option<Vec<i32>>) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("step", PyArray1::from_vec(py, r.step.clone()))?; d.set_item("time", PyArray1::from_vec(py, r.time.clone()))?; d.set_item("kind", PyArray1::from_vec(py, r.kind.clone()))?;
    d.set_item("slot", PyArray1::from_vec(py, r.slot.clone()))?; d.set_item("agent", PyArray1::from_vec(py, r.agent.clone()))?;
    d.set_item("other", PyArray1::from_vec(py, r.other.clone()))?;
    for (k, v) in [("dv_kmh", &r.dv_kmh), ("other_dv_kmh", &r.other_dv_kmh), ("share", &r.share), ("late_by", &r.late_by), ("progress_m", &r.progress_m), ("trip_m", &r.trip_m), ("trip_s", &r.trip_s)] {
        d.set_item(k, PyArray1::from_vec(py, v.clone()))?;
    }
    if let Some(e) = env { d.set_item("env", PyArray1::from_vec(py, e))?; }
    Ok(d)
}

/// World summary without creating a sim: dict of counts.
#[pyfunction]
fn world_info(py: Python<'_>, data_dir: &str) -> PyResult<Py<PyDict>> {
    let d = data_dir.to_string();
    let w = py.detach(move || World::load(&d)).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let out = PyDict::new(py);
    out.set_item("nodes", w.graph.node_count)?; out.set_item("edges", w.graph.edge_count)?;
    out.set_item("junction_movements", w.junctions.as_ref().map_or(0, |j| j.move_count()))?;
    out.set_item("controls", w.controls.as_ref().map_or(0, |c| c.count))?; out.set_item("stations", w.stations.len())?;
    Ok(out.unbind())
}

#[pymodule]
fn _bayenv(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Env>()?;
    m.add_class::<VecEnv>()?;
    m.add_class::<ObsBuffer>()?;
    m.add_function(wrap_pyfunction!(world_info, m)?)?;
    m.add("OBS_V2_SIZE", 70)?;
    Ok(())
}
