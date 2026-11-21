//! `GpuSim`: port of `src/gpu/gpuSim.js` (host side). The kernels are the unchanged WGSL of `src/gpu/kernels/`.
//!
//! Step loop per `run_steps(k)` (one submit): spawns -> scenes -> pumps -> refuel plans -> closures / reroutes ->
//! live-traffic sample (every 60 s) -> ops -> params -> k x [tick, ops, spawn, contacts, lists, junction lists,
//! leaders, junctions, sense (+ perceive), act, integrate] -> io readback -> events ingested.
use crate::defines::{self, ObsMode, VEHICLE_PARAMS, VEHICLE_TYPE_WEIGHTS};
use crate::gpu::{Gpu, Kernel};
use crate::kernels;
use crate::obs::{ObsDst, ObsFmt};
use crate::layout::{self as L, ev, flag, io, op, state, veh, Params};
use bayenv_world::{jsmath, World};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

// host logic beyond scatter (populations, streaming, refuel plans / pumps, obstacles, snapshots, resetDay)
#[path = "host.rs"]
pub mod host;
pub use host::{CommuterTable, RouteProvider, RouteStats, GpuPopulation, ObstacleKind, ObstacleSpec, Placed, Snapshot, StaticObstacle};

/// `GPU_DEFAULTS` subset that changes behaviour (the rest are constants of the JS defaults).
#[derive(Clone, Debug)]
pub struct GpuOptions {
    pub capacity: usize,
    pub dt: f64,
    pub start_time: f64,
    pub seed: u32,
    pub obs: ObsMode,
    /// 'position' (false) or 'rate' (true) steering actions
    pub steer_rate: bool,
    pub decide_every: u32,
    pub features: Features,
    pub controls: bool,
    pub junctions: bool,
    pub table_bits: u32,
    pub contact_cap: usize,
    pub ev_cap: usize,
    pub spawn_queue: usize,
    pub op_cap: usize,
    pub max_blocks: usize,
    pub route_slack: usize,
    pub resolve_rounds: usize,
    pub progress_weight: f64,
    pub jerk_weight: f64,
    pub jerk_reward: bool,
    pub grade_obs: bool,
    pub squeeze_delay: f64,
    pub deadlock_break: u32,
    pub reroute: bool,
    /// keep a host log of events (crash / arrive / ...)
    pub keep_events: bool,
    /// per-slot event records (`events::EventLog`, drained by `take_records`)
    pub record_events: bool,
    /// crash fault shares in the records: runs contacts.wgsl's reward-v2 crash charge with zero penalties (P.crashOn;
    /// only veh.eventTime of crashing cars and the reward sums are touched) to get its per-car share (1/255 steps)
    pub record_share: bool,
    /// event records: a missed exit through a junction connector is reported when the car is on the out-edge it took
    /// (the CPU sim's `_missedExits` timing) instead of at the connector decision (the JS GPU kernels' timing)
    pub missed_at_out_edge: bool,
    /// size the contact-resolve rounds by the contacts found (indirect dispatch; identical results). JS: off
    pub indirect_resolve: bool,
    /// contact resolve in one dispatch (kernels/resolve_fused.wgsl: one workgroup runs the rounds with barriers
    /// instead of 37 dispatches; same contacts resolved per round, bit-identical state). JS: off
    pub fused_resolve: bool,
    /// zero-fill workgroup memory at the start of every workgroup (the WGSL guarantee; naga's fill is slow for
    /// arrays). Every kernel here writes its workgroup variables before reading them (scan tmp, the integrate /
    /// act counters set by thread 0), so off gives identical results. JS (Chrome): on
    pub zero_init_workgroup: bool,
    /// binary instead of linear polyline segment searches (kernels/patches.wgsl; identical when the cumulative
    /// distances are non-decreasing, checked at creation). JS: off
    pub fast_search: bool,
    /// MODEL fix (default ON): junction connector entry as in the CPU sim (kernels/cpu_connector_join.wgsl;
    /// rust/docs/kerb-investigation.md). false = legacy mode, bit-identical with the JS GPU kernels (which have the
    /// connector-entry bug; every GPU run / checkpoint before this default was trained with it). JS: off
    pub cpu_connector_join: bool,
    /// MODEL fix (default ON): f32-robust friction tangent in the contact impulse (kernels/contact_tangent.wgsl;
    /// rust/docs/crash-divergence.md). false = legacy mode, bit-identical with the JS GPU kernels (whose f32 tangent
    /// gives aligned rear-end wrecks a spurious spin). JS: off
    pub contact_tangent: bool,
    /// lane / hash-cell starts with kernels/scan4k.wgsl (4096 per workgroup, the count clears folded into the
    /// scan: 12 -> 6 dispatches; identical starts). JS: off
    pub fast_scan: bool,
    /// run the per-car kernels of `SORTED_KERNELS` in lane-list order (kernels/sorted.wgsl, indirect over the
    /// lane-list entries + cap): memory locality; bit-identical. JS: off
    pub sorted_order: bool,
    /// GPU timestamp profile of the first step of every batch (`crate::profile`; needs TIMESTAMP_QUERY). JS: opts.profile
    pub profile: bool,
    /// populations: schedule the return legs (GPU_DEFAULTS.returnTrips)
    pub return_trips: bool,
    /// wall-clock budget per frame for refuel plans (JS refuelBudgetMs = 4; f64::INFINITY = every queued plan each frame, deterministic)
    pub refuel_budget_ms: f64,
    /// streaming: route prefetch workers (0 = serial A* at departure; n = n worker threads routing departures up to
    /// `route_ahead` sim s ahead). Same routes and spawn steps for any value. Default: available cores
    pub route_threads: usize,
    /// streaming: prefetch horizon in sim seconds (JS setStreaming routeAhead = 300)
    pub route_ahead: f64,
    /// streaming: OD route cache entries (JS RouteProvider cacheSize 200000; 0 = no cache). Never changes a route
    pub route_cache: usize,
}
#[derive(Clone, Copy, Debug)]
pub struct Features { pub lane_changes: bool, pub accidents: bool, pub fuel: bool, pub collisions: bool, pub yield_: bool, pub policy: bool, pub zip_fix: bool, pub empty: bool }
impl Default for Features { fn default() -> Self { Features { lane_changes: true, accidents: false, fuel: true, collisions: true, yield_: true, policy: true, zip_fix: true, empty: false } } }
impl Features {
    pub fn bits(&self) -> u32 {
        use L::feat::*;
        (if self.lane_changes { LANECHANGE } else { 0 }) | (if self.accidents { ACCIDENTS } else { 0 }) | (if self.fuel { FUEL } else { 0 }) | (if self.collisions { COLLISIONS } else { 0 })
            | (if self.yield_ { YIELD } else { 0 }) | (if self.policy { POLICY } else { 0 }) | (if self.zip_fix { ZIPFIX } else { 0 }) | (if self.empty { EMPTY } else { 0 })
    }
}
impl GpuOptions {
    /// Set a host optimisation flag by name (indirect_resolve, fused_resolve, sorted_order, profile).
    pub fn set_flag(&mut self, name: &str, on: bool) -> Result<(), String> {
        match name { "indirect_resolve" => self.indirect_resolve = on, "fused_resolve" => self.fused_resolve = on, "sorted_order" => self.sorted_order = on, "fast_scan" => self.fast_scan = on, "fast_search" => self.fast_search = on, "zero_init_workgroup" => self.zero_init_workgroup = on, "cpu_connector_join" => self.cpu_connector_join = on, "contact_tangent" => self.contact_tangent = on,
            "profile" => self.profile = on, _ => return Err(format!("unknown GpuOptions flag {name}")) }
        Ok(())
    }
    /// Flags from `BAYENV_GPU_OPTS` ("sorted_order=1,fused_resolve=0"; examples / benchmarks).
    pub fn apply_env(&mut self) {
        if let Ok(v) = std::env::var("BAYENV_GPU_OPTS") { for kv in v.split(',').filter(|x| !x.is_empty()) {
            let (k, b) = kv.split_once('=').unwrap_or((kv, "1")); self.set_flag(k.trim(), b.trim() != "0").unwrap();
        } }
    }
    /// Optional device features these options use (timestamp queries when profiling).
    pub fn gpu_features(&self) -> wgpu::Features {
        if self.profile { wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES } else { wgpu::Features::empty() }
    }
}
impl Default for GpuOptions {
    fn default() -> Self {
        GpuOptions { capacity: 131072, dt: 0.05, start_time: 5.0 * 3600.0, seed: 1, obs: ObsMode::V2, steer_rate: false, decide_every: 1, features: Features::default(),
            controls: true, junctions: true, table_bits: 18, contact_cap: 131072, ev_cap: 16384, spawn_queue: 16384, op_cap: 16384, max_blocks: 4096, route_slack: 1 << 22,
            resolve_rounds: 12, progress_weight: 1.0, jerk_weight: 0.5, jerk_reward: false, grade_obs: true, squeeze_delay: 15.0 * 60.0, deadlock_break: 0, reroute: true, keep_events: false, record_events: true, record_share: false, missed_at_out_edge: false, indirect_resolve: true, fused_resolve: true, sorted_order: true, fast_scan: true, fast_search: true, cpu_connector_join: true, contact_tangent: true, zero_init_workgroup: false, profile: false, return_trips: true, refuel_budget_ms: 4.0,
            route_threads: std::thread::available_parallelism().map_or(1, |n| n.get()), route_ahead: 300.0, route_cache: 0 }
    }
}

/// Word offsets of the world blob sections (`GPUSim.layoutOff`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LayoutOff { pub pts: u32, pub lane_off: u32, pub adj: u32, pub eb: u32, pub blk: u32, pub geo: u32, pub lgv: u32, pub hop: u32,
    pub ctl_edge: u32, pub ctl_ap: u32, pub ctl_c: u32, pub pair: u32, pub j_mouth: u32, pub j_mov: u32, pub j_edge_mov: u32, pub j_pts: u32, pub j_conf: u32, pub j_into: u32 }

/// The packed world (`packWorld` + `packControls` + `packJunctions` + pairs, assembled by `GPUSim._init`).
#[derive(Clone, Default)]
pub struct WorldBlob {
    pub words: Vec<u32>, pub off: LayoutOff,
    pub e: usize, pub n: usize, pub p: usize, pub k: usize, pub has_lg: bool, pub pair_n: usize,
    /// junction movements (None: no junction pack)
    pub j_m: Option<usize>,
    /// controls: Some((listOff, rampStart, rampEnd))
    pub ctl: Option<(u32, f64, f64)>,
}

#[derive(Clone, Debug)]
pub struct Event { pub time: f64, pub kind: &'static str, pub slot: i64, pub a: f64, pub b: f64 }

#[derive(Clone, Copy, Debug)]
struct Wreck { lo: u32, hi: u32, ty: u32, edge: u32, s: f64 }

#[derive(Clone, Debug)]
pub struct Scene {
    pub id: usize, pub time: f64, pub cause: &'static str, pub severity: &'static str, pub vehicles: Vec<u32>, pub lanes_blocked: u32, pub lane_mask: u32,
    pub edge: u32, pub s: f64, pub x: f64, pub y: f64, pub z: f64, pub block_end: f64, pub clear_end: f64, pub squeeze_at: f64, pub squeeze: bool,
    /// 0 active, 1 shoulder, 2 cleared
    pub response: u8, pub is_crash: bool, pub block: i32, pub stranded: bool, pub gone: Vec<u32>, pub delta_v_kmh: [f64; 2],
}

#[derive(Clone, Debug, Default)]
pub struct Counters { pub minor_contacts: u64, pub crash_events: u64, pub crashes: u64, pub crashed_vehicles: u64, pub secondary_crashes: u64, pub stranded: u64, pub towed: u64,
    pub cleared: u64, pub spawned: u64, pub refuel_plan_failed: u64, pub events_dropped: u64, pub signal_violations: u64, pub stop_violations: u64, pub pull_overs: u64,
    pub towed_to_shoulder: u64, pub hauled_away: u64, pub missed_exits: u64, pub stuck_labels: u64, pub reroutes: u64, pub reroute_failed: u64, pub no_route: u64,
    pub by_severity: [u64; 4], pub refuel_planned: u64, pub refuelled: u64, pub reentered: u64 }


struct ScanLevel { n: usize, blocks: usize }

/// Kernels that can run in lane-list order with `GpuOptions.sorted_order` (all bit-identical in `opt_check`):
/// (label, sorted label, file, entry, workgroup size, used by default). Off by default where the extra threads
/// (cap + lane entries) cost more than the locality gains (M4 Max, 10k / 100k cars); `BAYENV_SORTED=a,b` overrides.
pub const SORTED_KERNELS: &[(&str, &str, &str, &str, u32, bool)] = &[
    ("leaders", "leaders.sorted", "leaders", "main", 128, true),
    ("sense", "sense.sorted", "sense", "main", 128, true),
    ("junction.yield", "junction.yield.sorted", "jyield", "jyield", 128, true),
    ("junction.perceive", "junction.perceive.sorted", "junction", "jperceive", 128, false),
    ("courtesy.propose", "courtesy.propose.sorted", "courtesy", "propose", 128, false),
    ("courtesy.apply", "courtesy.apply.sorted", "courtesy", "apply", 128, false),
    ("integrate", "integrate.sorted", "integrate", "main", 256, false),
    ("act", "act.sorted", "act", "main", 128, false),
    ("contacts.barrier", "contacts.barrier.sorted", "contacts", "barrier", 128, false),
];
/// `Dispatch.x` values >= X_LARGS: indirect sizes of the lane-list-order kernels, largs at 16 * (u32::MAX - x)
/// for 32 / 64 / 128 / 256 threads per workgroup (kernels/list_args.wgsl).
pub const X_LARGS: u32 = u32::MAX - 3;
fn x_largs(wg: u32) -> u32 { u32::MAX - match wg { 32 => 0, 64 => 1, 128 => 2, 256 => 3, _ => panic!("sorted workgroup size {wg}") } }
/// Workgroup size of a sorted kernel (`BAYENV_WG=leaders=64,...` overrides; diagnostics).
fn sorted_wg(label: &str, wg: u32) -> u32 {
    std::env::var("BAYENV_WG").ok().and_then(|v| v.split(',').find_map(|kv| kv.split_once('=').filter(|(k, _)| *k == label).map(|(_, w)| w.parse().unwrap()))).unwrap_or(wg)
}

/// One dispatch of a step: JS `_stages` stage label, pipeline, bind-group set, workgroups (0 = indirect from `cargs`).
#[derive(Clone, Debug)]
pub struct Dispatch { pub stage: &'static str, pub kernel: &'static str, pub bg: String, pub x: u32 }

pub struct GpuSim {
    pub gpu: Gpu,
    pub world: Arc<World>,
    pub opts: GpuOptions,
    pub cap: usize,
    pub dt: f64,
    pub time: f64,
    pub step: u64,
    rng_state: u32,
    pub blob: WorldBlob,
    k: HashMap<&'static str, Kernel>,
    bg: HashMap<String, Vec<wgpu::BindGroup>>,
    scan: Vec<ScanLevel>, hscan: Vec<ScanLevel>,
    /// kernels/scan4k.wgsl levels (fast_scan): lane keys, hash cells
    fscan: Vec<ScanLevel>, fhscan: Vec<ScanLevel>,
    params: Params,
    pub n_counts: usize, pub t_size: usize, w_bits: u32,
    pub io_words: usize, pub obs_size: usize,
    pub n_agents: usize,
    pub route_words: usize, route_cap: usize,
    pub route_mirror: Vec<u32>,
    edge_head: Vec<u32>, block_free: Vec<u32>, block_next: Vec<u32>,
    pub free: Vec<u32>,
    pub slot_trip: Vec<i32>,
    pub slot_route: Vec<u32>,
    slot_station: Vec<i32>,
    queue_free: Vec<u32>, queue_high: usize,
    ops_pending: Vec<u32>,
    wrecks: HashMap<u32, Wreck>,
    in_crash: HashSet<u32>,
    crash_x: HashMap<u32, CrashX>,
    refuel_queue: Vec<(u32, u32)>,
    reroute_queue: std::collections::VecDeque<Reroute>,
    last_reroute: HashMap<u32, (f64, i32)>,
    lot: u32,
    new_pose: bool,
    pub crashes: Vec<Scene>,
    pub active: Vec<usize>,
    pub counters: Counters,
    pub gpu_stats: [u32; 64],
    pub events: Vec<Event>,
    closed_key: Option<Vec<u32>>, closed_n: usize, pub closed_edges: Vec<u32>,
    edge_speed_ema: HashMap<u32, f64>, edge_speed_order: Vec<u32>,
    pub traffic_penalty: HashMap<u32, f64>, penalty_order: Vec<u32>,
    pending_penalty: Option<(HashMap<u32, f64>, Vec<u32>)>,
    next_traffic: f64,
    /// A* router (route.js makeRouter, weight 1.25) for missed-exit / closure reroutes; built on first use
    pub router: Option<bayenv_world::nav::Router>,
    /// route-word packer (packRoutes with the per-hop maneuver table)
    pub packer: Option<Arc<crate::pack::RoutePacker>>,
    staging: wgpu::Buffer,
    out_staging: Option<wgpu::Buffer>,
    /// per-slot event records (opts.record_events)
    pub rec: crate::events::EventLog,
    rec_mark: usize,
    /// missed exits via a connector waiting for the car to reach the taken edge: (slot, took edge, agent)
    missed_pending: Vec<(u32, u32, i64)>,
    /// kernels/obs_f16.wgsl + its bind group (built on the first F16 readback)
    half: Option<(Kernel, wgpu::BindGroup)>,
    /// a submitted batch whose readback / events are processed lazily (`run_steps_async`, `sync`)
    pending: Option<Pending>,
    io_host: Vec<u32>,
    /// opts.fast_search and the world's polylines allow it (checked at creation)
    fast_search_ok: bool,
    /// stages(): build the JS dispatch list (peek_stages)
    js_stages: bool,
    /// timestamp profiler (opts.profile)
    pub profiler: Option<crate::profile::Profiler>,
    /// population / streaming / stations / obstacles state (sim/host.rs)
    pub x: host::Ext,
}

#[derive(Clone, Copy, Debug, Default)]
struct CrashX { other: u32, pre_a: f64, pre_b: f64, x: f64, z: f64, nx: f64, nz: f64, ha: f64, hb: f64, closing: f64, lane_a: u32, lane_b: u32, maneuver: u32, any_crashed: bool, strike_edge: i64, has_h: bool }
#[derive(Clone, Copy, Debug)]
struct Reroute { slot: u32, edge: u32, trip: i32, missed: bool, slow: bool, rp: u32, prefix: i64, t: f64 }

const NONE: u32 = 0xFFFF_FFFF;
/// events copied with the io header in every readback (more are fetched only when written)
const EV_FAST: usize = 1024;
const LANE_W: f64 = 3.7;

fn f32bits(x: f64) -> u32 { (x as f32).to_bits() }
fn bf(u: u32) -> f64 { f32::from_bits(u) as f64 }

impl GpuSim {
    /// Build on the given world + packed blob (see `crate::pack::build_world_blob`).
    pub fn with_blob(world: Arc<World>, blob: WorldBlob, opts: GpuOptions) -> Result<GpuSim, String> { Self::with_blob_on(Gpu::new_with(opts.gpu_features())?, world, blob, opts) }

    /// As `with_blob` on a given context (e.g. `other_sim.gpu.share()`: several sims on one device).
    pub fn with_blob_on(gpu: Gpu, world: Arc<World>, blob: WorldBlob, opts: GpuOptions) -> Result<GpuSim, String> {
        let mut gpu = gpu;
        let cap = opts.capacity;
        let obs_size = opts.obs.size();
        let use_j = blob.j_m.is_some() && opts.junctions;
        gpu.storage("world", (blob.words.len() * 4) as u64, Some(bytemuck::cast_slice(&blob.words)), wgpu::BufferUsages::empty());
        gpu.storage("veh", (cap * L::VEH_WORDS * 4) as u64, None, wgpu::BufferUsages::empty());
        gpu.storage("par", (cap * L::PAR_WORDS * 4) as u64, None, wgpu::BufferUsages::empty());
        gpu.storage("ring", (cap * L::CTRL_RING * L::CTRL_STRIDE * 4) as u64, None, wgpu::BufferUsages::empty());
        gpu.storage("pose", (cap * 16) as u64, None, wgpu::BufferUsages::VERTEX);
        let n_counts = blob.k + 1;
        gpu.storage("counts", (n_counts * 4) as u64, None, wgpu::BufferUsages::empty());
        gpu.storage("items", (L::MAX_OCC * cap * 4) as u64, None, wgpu::BufferUsages::empty());
        gpu.storage("order", (L::MAX_OCC * cap * 4) as u64, None, wgpu::BufferUsages::empty());
        gpu.storage("vidx", (cap * 16) as u64, None, wgpu::BufferUsages::empty());
        gpu.storage("claim", (cap * 4) as u64, None, wgpu::BufferUsages::empty());
        let t_size = 1usize << opts.table_bits; let w_bits = (opts.table_bits + 1) / 2;
        gpu.storage("hcounts", ((t_size + 1) * 4) as u64, None, wgpu::BufferUsages::empty());
        gpu.storage("hitems", (cap * 4) as u64, None, wgpu::BufferUsages::empty());
        gpu.storage("hidx", (cap * 8) as u64, None, wgpu::BufferUsages::empty());
        gpu.storage("contacts", ((4 + opts.contact_cap * 8 + 262144 * 4 + cap) * 4) as u64, None, wgpu::BufferUsages::empty());
        gpu.storage("laneClaim", (blob.k.max(1) * 4) as u64, None, wgpu::BufferUsages::empty());
        let io_words = io::EVDATA + 4 * opts.ev_cap;
        gpu.storage("io", (io_words * 4) as u64, None, wgpu::BufferUsages::empty());
        gpu.storage("queue", (opts.spawn_queue * L::SPAWN_WORDS * 4) as u64, None, wgpu::BufferUsages::empty());
        gpu.storage("ops", (opts.op_cap * 16) as u64, None, wgpu::BufferUsages::empty());
        gpu.uniform("params", (L::PARAM_FIELDS.len() * 4) as u64);
        // scan levels (lane counts, hash cells)
        let mk_levels = |gpu: &mut Gpu, pre: &str, sums: &str, mut n: usize, src: &str| -> Vec<ScanLevel> {
            let mut lv = Vec::new(); let mut l = 0; let mut src = src.to_string();
            loop {
                let blocks = (n + 511) / 512;
                let sname = format!("{sums}{l}"); gpu.storage(&sname, (blocks.max(1) * 4) as u64, None, wgpu::BufferUsages::empty());
                let ub = format!("{pre}{l}"); gpu.uniform(&ub, 16); gpu.write(&ub, bytemuck::cast_slice(&[n as u32, 0, 0, 0]), 0);
                lv.push(ScanLevel { n, blocks }); let _ = &src;
                if blocks <= 1 { break; }
                n = blocks; src = sname; l += 1;
            }
            lv
        };
        let scan = mk_levels(&mut gpu, "scanP", "scanSums", n_counts, "counts");
        let hscan = mk_levels(&mut gpu, "hscanP", "hscanSums", t_size + 1, "hcounts");
        // fast_scan: count accumulators (zeroed by scanFirst) + 4096-element levels
        let fast_levels = |gpu: &mut Gpu, pre: &str, mut n: usize| -> Vec<ScanLevel> {
            let mut lv = Vec::new(); let mut l = 0;
            loop {
                let blocks = (n + 4095) / 4096;
                gpu.storage(&format!("{pre}S{l}"), (blocks.max(1) * 4) as u64, None, wgpu::BufferUsages::empty());
                let ub = format!("{pre}P{l}"); gpu.uniform(&ub, 16); gpu.write(&ub, bytemuck::cast_slice(&[n as u32, 0, 0, 0]), 0);
                lv.push(ScanLevel { n, blocks });
                if blocks <= 1 { break; }
                n = blocks; l += 1;
            }
            lv
        };
        let (fscan, fhscan) = if opts.fast_scan {
            gpu.storage("lacc", (n_counts * 4) as u64, None, wgpu::BufferUsages::empty());
            gpu.storage("hacc", ((t_size + 1) * 4) as u64, None, wgpu::BufferUsages::empty());
            gpu.storage("scanDummy", 16, None, wgpu::BufferUsages::empty());
            (fast_levels(&mut gpu, "fscan", n_counts), fast_levels(&mut gpu, "fhscan", t_size + 1))
        } else { (Vec::new(), Vec::new()) };
        let v3 = opts.obs == ObsMode::V3;
        gpu.storage("inputs", (cap * obs_size * 4 + if v3 { cap * L::LH_W * 4 } else { 0 }) as u64, None, wgpu::BufferUsages::empty());
        gpu.storage("actions", (cap * 3 * 4) as u64, None, wgpu::BufferUsages::empty());
        let n_agents = cap;
        gpu.storage("agents", (n_agents.max(1) * 16) as u64, None, wgpu::BufferUsages::empty());
        gpu.storage("edgeCount", (blob.e.max(1) * 4) as u64, None, wgpu::BufferUsages::empty());
        gpu.storage("edgeSpeed", (blob.e.max(1) * 4) as u64, None, wgpu::BufferUsages::empty());
        gpu.storage("telemetry", 256 * 4, None, wgpu::BufferUsages::empty());
        let jm = if use_j { blob.j_m.unwrap() } else { 0 };
        gpu.storage("ctl", ((cap * L::CTL_WORDS + if use_j { jm + blob.e + (blob.e + 31) / 32 + blob.e } else { 0 } + 1) * 4) as u64, None, wgpu::BufferUsages::empty());
        gpu.buffer("cargs", 16, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST, Some(bytemuck::cast_slice(&[1024u32, 1, 1, 0])));
        gpu.buffer("largs", 64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST, Some(bytemuck::cast_slice(&[1u32, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0])));
        gpu.uniform("listArgsP", 16); gpu.write("listArgsP", bytemuck::cast_slice(&[blob.k as u32, cap as u32, 0, 0]), 0);
        gpu.storage("cflag", (cap * 8) as u64, None, wgpu::BufferUsages::empty());
        gpu.uniform("collidedP", 16); gpu.write("collidedP", bytemuck::cast_slice(&[cap as u32, 0, 0, 0]), 0);
        gpu.uniform("contactArgsP", 16); gpu.write("contactArgsP", bytemuck::cast_slice(&[opts.contact_cap as u32, 0, 0, 0]), 0);
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("staging"), size: (io_words * 4) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let mut sim = GpuSim {
            world, cap, dt: opts.dt, time: opts.start_time, step: 0, rng_state: { let r = jsmath::to_u32(opts.seed as f64 * 2654435761.0); if r == 0 { 1 } else { r } },
            k: HashMap::new(), bg: HashMap::new(), scan, hscan, fscan, fhscan, params: Params::default(), n_counts, t_size, w_bits, io_words, obs_size, n_agents,
            route_words: 0, route_cap: 0, route_mirror: Vec::new(), edge_head: vec![0; blob.e], block_free: (0..opts.max_blocks as u32).rev().collect(), block_next: vec![0; opts.max_blocks],
            free: (0..cap as u32).rev().collect(), slot_trip: vec![-1; cap], slot_route: vec![0; cap * 2], slot_station: vec![-1; cap],
            queue_free: (0..opts.spawn_queue as u32).rev().collect(), queue_high: 0, ops_pending: Vec::new(), wrecks: HashMap::new(), in_crash: HashSet::new(), crash_x: HashMap::new(),
            refuel_queue: Vec::new(), reroute_queue: Default::default(), last_reroute: HashMap::new(), lot: 0, new_pose: false, crashes: Vec::new(), active: Vec::new(),
            counters: Counters::default(), gpu_stats: [0; 64], events: Vec::new(), closed_key: None, closed_n: 0, closed_edges: Vec::new(),
            edge_speed_ema: HashMap::new(), edge_speed_order: Vec::new(), traffic_penalty: HashMap::new(), penalty_order: Vec::new(), pending_penalty: None, next_traffic: -1e9,
            router: None, packer: None, out_staging: None, half: None, rec: Default::default(), rec_mark: 0, missed_pending: Vec::new(), pending: None, staging, io_host: vec![0; io_words], profiler: None, js_stages: false, fast_search_ok: false, x: Default::default(), blob, gpu, opts,
        };
        if !use_j { sim.blob.j_m = None; }
        if sim.opts.fast_search { sim.fast_search_ok = sim.polylines_monotone(); if !sim.fast_search_ok { eprintln!("bayenv-gpu: fast_search off (an edge / connector polyline has decreasing cumulative distance)"); } }
        sim.blob.words = Vec::new();   // uploaded: no host copy kept (block writes go straight to the GPU)
        if sim.opts.profile {
            let f = sim.gpu.device.features();
            if !f.contains(wgpu::Features::TIMESTAMP_QUERY) { return Err("profile: adapter has no TIMESTAMP_QUERY".into()); }
            sim.profiler = Some(crate::profile::Profiler::new(&sim.gpu.device, &sim.gpu.queue, f.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES)));
        }
        sim.ensure_routes(1 << 16);
        sim.compile()?;
        sim.make_bind_groups();
        sim.write_params(0);
        Ok(sim)
    }

    pub fn has_junctions(&self) -> bool { self.blob.j_m.is_some() }
    fn j_m(&self) -> usize { self.blob.j_m.unwrap_or(0) }

    fn compile(&mut self) -> Result<(), String> {
        let mut specs = kernels::specs(self.has_junctions(), self.opts.obs == ObsMode::V3);
        for sp in specs.iter_mut() { sp.file = self.kernel_file(sp.file); }
        // compile in parallel (naga + Metal pipeline creation), results in spec order
        let gpu = &self.gpu; let obs = self.opts.obs; let cap = self.cap;
        let out: Vec<Result<Kernel, String>> = std::thread::scope(|s| {
            let zi = self.opts.zero_init_workgroup; let fsr = self.fast_search_ok; let cj = self.opts.cpu_connector_join; let ct = self.opts.contact_tangent;
            let hs: Vec<_> = specs.iter().map(|sp| s.spawn(move || { let src = kernels::model_patches(kernels::source(sp, obs, cap), sp.file, cj, ct); gpu.kernel_with(sp.label, &if fsr { kernels::patch(&src) } else { src }, sp.entry, zi) })).collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for (sp, k) in specs.iter().zip(out) { self.k.insert(sp.label, k?); }
        self.k.insert("contactArgs", self.gpu.kernel("contactArgs", kernels::host_raw("contact_args"), "main")?);
        { let code = crate::defines::prefix(self.opts.obs, self.cap) + &kernels::load("common") + "\n" + kernels::host_raw("step_mark");
          self.k.insert("stepMark", self.gpu.kernel("stepMark", &code, "stepMark")?); }
        self.k.insert("collided", self.gpu.kernel("collided", kernels::host_raw("collided"), "main")?);
        if self.opts.sorted_order {
            self.k.insert("listArgs", self.gpu.kernel("listArgs", kernels::host_raw("list_args"), "main")?);
            let only = std::env::var("BAYENV_SORTED").ok();   // diagnostics: comma list of kernels to sort
            for &(l, sl, file, entry, wg, on) in SORTED_KERNELS {
                if !only.as_ref().map_or(on, |o| o.split(',').any(|x| x == l)) { continue; }
                if !self.k.contains_key(l) { continue; }
                let spec = kernels::KernelSpec { label: sl, file: self.kernel_file(file), entry, defines: true };
                let src = kernels::sorted_source(&self.patched(kernels::model_patches(kernels::source(&spec, self.opts.obs, self.cap), spec.file, self.opts.cpu_connector_join, self.opts.contact_tangent)), entry, sorted_wg(l, wg));
                self.k.insert(sl, self.gpu.kernel_with(sl, &src, &format!("{entry}_sorted"), self.opts.zero_init_workgroup)?);
            }
        }
        if self.opts.fast_scan {
            for (l, e) in [("scan4k.first", "scanFirst"), ("scan4k.inPlace", "scanInPlace"), ("scan4k.add", "addSums")] { self.k.insert(l, self.gpu.kernel_with(l, kernels::host_raw("scan4k"), e, false)?); }
        }
        if self.opts.fused_resolve {
            let src = defines::prefix(self.opts.obs, self.cap) + &format!("const RF_ROUNDS: u32 = {}u;\n", self.opts.resolve_rounds) + &self.patched(kernels::model_patches(kernels::resolve_includes(kernels::host_raw("resolve_fused")), "contacts", false, self.opts.contact_tangent));
            self.k.insert("contacts.resolveFused", self.gpu.kernel_with("contacts.resolveFused", &src, "resolveFused", self.opts.zero_init_workgroup)?);
        }
        Ok(())
    }

    fn reg(&self) -> HashMap<&'static str, Arc<wgpu::Buffer>> {
        let b = |n: &str| self.gpu.get(n);
        let mut r = HashMap::new();
        for (k, n) in [("P", "params"), ("io", "io"), ("world", "world"), ("routes", "routes"), ("veh", "veh"), ("par", "par"), ("ring", "ring"), ("pose", "pose"),
            ("counts", "counts"), ("starts", "counts"), ("items", "items"), ("order", "order"), ("vidx", "vidx"), ("claim", "claim"), ("laneClaim", "laneClaim"),
            ("queue", "queue"), ("ops", "ops"), ("inputs", "inputs"), ("actions", "actions"), ("agents", "agents"), ("edgeCount", "edgeCount"), ("edgeSpeed", "edgeSpeed"),
            ("telemetry", "telemetry"), ("ctl", "ctl"), ("CA", "contactArgsP"), ("CC", "collidedP"), ("vehw", "veh"), ("cflag", "cflag"), ("LA", "listArgsP"), ("largs", "largs"), ("cargs", "cargs"), ("hcounts", "hcounts"), ("hstarts", "hcounts"), ("hitems", "hitems"), ("hidx", "hidx"), ("contacts", "contacts")] { r.insert(k, b(n)); }
        r
    }

    fn make_bind_groups(&mut self) {
        let reg = self.reg(); let mut bg = HashMap::new();
        for (name, k) in &self.k {
            if *name == "scanBlocks" || *name == "addSums" || name.starts_with("scan4k.") { continue; }
            bg.insert(name.to_string(), (0..k.layouts.len()).map(|g| self.gpu.bind_group(k, &reg, g)).collect());
        }
        let level_bg = |kname: &str, pre: &str, sums: &str, src: &str, levels: &Vec<ScanLevel>| -> Vec<Vec<wgpu::BindGroup>> {
            (0..levels.len()).map(|l| {
                let mut r = HashMap::new();
                r.insert("SP", self.gpu.get(&format!("{pre}{l}")));
                r.insert("data", self.gpu.get(&if l == 0 { src.to_string() } else { format!("{sums}{}", l - 1) }));
                r.insert("sums", self.gpu.get(&format!("{sums}{l}")));
                vec![self.gpu.bind_group(&self.k[kname], &r, 0)]
            }).collect()
        };
        for (l, v) in level_bg("scanBlocks", "scanP", "scanSums", "counts", &self.scan).into_iter().enumerate() { bg.insert(format!("scan{l}"), v); }
        for (l, v) in level_bg("addSums", "scanP", "scanSums", "counts", &self.scan).into_iter().enumerate() { bg.insert(format!("add{l}"), v); }
        for (l, v) in level_bg("scanBlocks", "hscanP", "hscanSums", "hcounts", &self.hscan).into_iter().enumerate() { bg.insert(format!("hscanB{l}"), v); }
        for (l, v) in level_bg("addSums", "hscanP", "hscanSums", "hcounts", &self.hscan).into_iter().enumerate() { bg.insert(format!("hscanA{l}"), v); }
        if self.opts.fast_scan {
            // the JS counting kernels atomically count into the accumulators; starts stay in counts / hcounts
            let mut r = reg.clone(); r.insert("counts", self.gpu.get("lacc")); bg.insert("build.count.acc".into(), (0..self.k["build.count"].layouts.len()).map(|g| self.gpu.bind_group(&self.k["build.count"], &r, g)).collect());
            if self.k.contains_key("hash.hcount") { let mut r = reg.clone(); r.insert("hcounts", self.gpu.get("hacc")); bg.insert("hash.hcount.acc".into(), (0..self.k["hash.hcount"].layouts.len()).map(|g| self.gpu.bind_group(&self.k["hash.hcount"], &r, g)).collect()); }
            for (pre, acc, out, lv) in [("fscan", "lacc", "counts", &self.fscan), ("fhscan", "hacc", "hcounts", &self.fhscan)] {
                let sb = |l: usize| self.gpu.get(&format!("{pre}S{l}"));
                for l in 0..lv.len() {
                    let mut r = HashMap::new(); r.insert("SP", self.gpu.get(&format!("{pre}P{l}"))); r.insert("sums", sb(l));
                    if l == 0 { r.insert("src", self.gpu.get(acc)); r.insert("dst", self.gpu.get(out)); bg.insert(format!("{pre}F"), vec![self.gpu.bind_group(&self.k["scan4k.first"], &r, 0)]); }
                    else { r.insert("src", sb(l - 1)); r.insert("dst", self.gpu.get("scanDummy")); bg.insert(format!("{pre}I{l}"), vec![self.gpu.bind_group(&self.k["scan4k.inPlace"], &r, 0)]); }
                    // add level l's sums into its output (dst = starts for l = 0, else the sums of level l - 1)
                    let mut r = HashMap::new(); r.insert("SP", self.gpu.get(&format!("{pre}P{l}"))); r.insert("sums", sb(l)); r.insert("src", self.gpu.get("scanDummy"));
                    r.insert("dst", if l == 0 { self.gpu.get(out) } else { sb(l - 1) });
                    bg.insert(format!("{pre}A{l}"), vec![self.gpu.bind_group(&self.k["scan4k.add"], &r, 0)]);
                }
            }
        }
        self.bg = bg;
    }

    fn ensure_routes(&mut self, words: usize) {
        if words <= self.route_cap { return; }
        let cap = words.max((self.route_cap as f64 * 1.5).ceil() as usize);
        let old = self.gpu.buffers.get("routes").cloned();
        let nb = self.gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("routes"), size: (cap * 4) as u64, usage: crate::gpu::STORAGE, mapped_at_creation: false });
        if let (Some(o), true) = (&old, self.route_words > 0) {
            let mut enc = self.gpu.device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(o, 0, &nb, 0, (self.route_words * 4) as u64);
            self.gpu.queue.submit([enc.finish()]);
        }
        if let Some(o) = old { o.destroy(); }
        self.gpu.buffers.insert("routes".into(), Arc::new(nb));
        self.route_cap = cap;
        if !self.k.is_empty() { self.make_bind_groups(); }
    }

    /// Append packed route words to the GPU route pool (and the host mirror); returns the word offset.
    pub fn append_route(&mut self, words: &[u32]) -> usize {
        self.ensure_routes(self.route_words + words.len());
        let off = self.route_words;
        self.gpu.write("routes", bytemuck::cast_slice(words), (off * 4) as u64);
        self.route_words += words.len();
        if self.route_mirror.len() < self.route_words { self.route_mirror.resize(self.route_words + (1 << 16), 0); }
        self.route_mirror[off..off + words.len()].copy_from_slice(words);
        off
    }

    fn write_params(&mut self, op_count: usize) {
        let o = &self.opts; let w = &self.blob; let lo = w.off; let mut p = Params::default();
        let mut s = |k: &str, v: f64| p.set(k, v);
        s("cap", self.cap as f64); s("E", w.e as f64); s("N", w.n as f64); s("K", w.k as f64);
        s("T", self.t_size as f64); s("wBits", self.w_bits as f64); s("wMask", ((1u32 << self.w_bits) - 1) as f64); s("hMask", ((1u32 << (o.table_bits - self.w_bits)) - 1) as f64);
        s("ptsOff", lo.pts as f64); s("laneOffOff", lo.lane_off as f64); s("adjOff", lo.adj as f64); s("ebOff", lo.eb as f64);
        s("dt", self.dt); s("time0", self.time); s("step0", self.step as f64); s("features", o.features.bits() as f64);
        s("cellSize", 25.0); s("leaderLookahead", 250.0); s("junctionMargin", 12.0); s("collisionInterval", 1.0);
        s("s0", 2.0); s("collideShrink", 0.9); s("speedFactorMin", 0.85); s("speedFactorSpan", 0.3);
        s("laneCheckInterval", 1.0); s("yieldLook", 40.0); s("stopMargin", 1.5); s("stuckTimeout", 1800.0);
        s("spawnBlockedTimeout", 1e30); s("spawnCount", self.queue_high as f64); s("opCount", op_count as f64); s("decideEvery", o.decide_every as f64);
        s("accRate", 5e-5); s("accSpeedRef", 25.0); s("accCloseGap", 30.0); s("accMergeFactor", 3.0);
        s("accMergeDist", 60.0); s("rubberDist", 120.0); s("rubberFactor", 0.8); s("towTime", 1800.0);
        s("lowFuelFrac", 0.12); s("seed", o.seed as f64); s("evCap", o.ev_cap as f64); s("inSize", self.obs_size as f64);
        s("qrWindow", 300.0); s("qrDist", 500.0); s("qrFactor", 6.0); s("nBlocks", if self.active.is_empty() { 0.0 } else { o.max_blocks as f64 });
        s("blkOff", lo.blk as f64); s("maxCellReach", 6.0); s("fuelScale", 1.0); s("anticipation", 0.6);
        s("mergeLookahead", 80.0); s("collisionTol", 0.3); s("contactSpeed", 2.0); s("lateralTol", 0.15);
        s("offroadMargin", 1.0); s("squeezeSpeed", 3.0); s("prepBase", 250.0); s("prepPerLane", 250.0);
        s("fuelCheckInterval", 60.0); s("fuelMinFillToSkip", 0.5); s("fuelLowFraction", 0.12); s("fuelPlanFactor", 1.3);
        s("fuelReserveKm", 15.0); s("fuelPlanSpeed", 22.2); s("progressW", o.progress_weight); s("jerkW", o.jerk_weight); s("nAgents", self.n_agents as f64);
        for k in ["crashOn", "crashBase", "crashK", "crashCap", "crashFault", "crashFail"] { s(k, 0.0); }
        if o.record_share { s("crashOn", 1.0); }
        s("geoOff", lo.geo as f64); s("lgvOff", lo.lgv as f64); s("hopOff", lo.hop as f64); s("hasLG", if w.has_lg { 1.0 } else { 0.0 });
        let ctl = if o.controls { w.ctl } else { None };
        s("ctlEdgeOff", lo.ctl_edge as f64); s("ctlListOff", ctl.map_or(0.0, |c| (lo.ctl_edge + c.0) as f64)); s("ctlApOff", lo.ctl_ap as f64); s("ctlCOff", lo.ctl_c as f64);
        s("hasCtl", if ctl.is_some() { 1.0 } else { 0.0 });
        s("ctlLook", 250.0); s("rampStart", ctl.map_or(6.0 * 3600.0, |c| c.1)); s("rampEnd", ctl.map_or(10.0 * 3600.0, |c| c.2)); s("freewayShoulder", 3.0);
        s("pairOff", lo.pair as f64); s("pairN", w.pair_n as f64); s("labelMode", 0.0);
        s("jMouthOff", lo.j_mouth as f64); s("jMovOff", lo.j_mov as f64); s("jEdgeMovOff", lo.j_edge_mov as f64); s("jPtsOff", lo.j_pts as f64); s("jConfOff", lo.j_conf as f64); s("jIntoOff", lo.j_into as f64);
        s("jSight", 0.0); s("pathSight", 0.0); s("dlBreak", o.deadlock_break as f64); s("spawnLane", 0.0);
        s("rerouteSight", 2000.0); s("passWrecks", 0.0); s("trafficOn", if !self.traffic_penalty.is_empty() && o.reroute { 1.0 } else { 0.0 });
        s("steerMode", (1 | if o.steer_rate { 2 } else { 0 } | if o.grade_obs { 8 } else { 0 }) as f64);
        s("rlOn", if o.jerk_reward { 1.0 } else { 0.0 }); s("rlRec", 0.0); s("rlSeed", 1.0); s("rlChunk", 0.0); s("rlStd0", 1.0); s("rlStd1", 1.0); s("rlStd2", 1.0); s("rlRowCap", 0.0);
        s("hasJ", if w.j_m.is_some() { 1.0 } else { 0.0 }); s("jM", w.j_m.unwrap_or(0) as f64); s("jHeadOff", (self.cap * L::CTL_WORDS) as f64);
        s("layerGap", 2.5); s("pairCooldown", 7200.0); s("touchKmh", 1.0); s("shoulder", 1.0);
        s("oppositeMargin", 0.5); s("laneEndLookahead", 450.0); s("contactCap", o.contact_cap as f64); s("physDt", 0.01);
        self.gpu.write("params", bytemuck::cast_slice(&p.words), 0);
        self.params = p;
    }
    pub fn params_words(&self) -> &[u32; 128] { &self.params.words }

    // ------------------------------------------------------------------ actions / observations
    /// External actions [steer, gas, brake] per slot from `offset_slots`.
    pub fn set_actions(&mut self, actions: &[f32], offset_slots: usize) {
        assert!(offset_slots + actions.len() / 3 <= self.cap, "set_actions: more actions than slots");
        self.gpu.write("actions", bytemuck::cast_slice(actions), (offset_slots * 12) as u64);
    }
    /// Observation vectors of every slot (cap x obs_size), into `out`.
    pub fn read_obs_into(&self, out: &mut [f32]) {
        assert!(self.opts.obs != ObsMode::None, "read_obs: created with obs none");
        let n = (self.cap * self.obs_size).min(out.len());
        self.read_into("inputs", 0, bytemuck::cast_slice_mut(&mut out[..n]));
    }
    pub fn read_into(&self, name: &str, offset: u64, out: &mut [u8]) { self.gpu.read_into(name, offset, out); }
    /// Raw buffer words (veh: cap x 40, par: cap x 20, pose: cap x 4, ...).
    pub fn read_words(&self, name: &str) -> Vec<u32> { let b = self.gpu.get(name); bytemuck::cast_slice(&self.gpu.read(name, 0, b.size())).to_vec() }

    // ------------------------------------------------------------------ stepping
    fn rand(&mut self) -> f64 { let mut x = self.rng_state; x ^= x << 13; x ^= x >> 17; x ^= x << 5; self.rng_state = x; x as f64 / 4294967296.0 }
    fn gauss(&mut self) -> f64 { let mut u = 0.0; while u == 0.0 { u = self.rand(); } (-2.0 * jsmath::log(u)).sqrt() * jsmath::cos(2.0 * std::f64::consts::PI * self.rand()) }

    fn stages(&mut self, step_index: u64, first: bool, op_count: usize) -> Vec<Dispatch> {
        let cap = self.cap as u32; let f = self.opts.features; let mut d: Vec<Dispatch> = Vec::new();
        let mut stage: &'static str = "tick";
        let g = |n: usize, wg: usize| ((n + wg - 1) / wg).max(1) as u32;
        let js = self.js_stages; let so = self.opts.sorted_order; let fs = self.opts.fast_scan && !js;
        macro_rules! push { ($k:expr, $bg:expr, $x:expr) => { {
            let (mut k, mut x): (&'static str, u32) = ($k, $x);
            if so && !js { if let Some(e) = SORTED_KERNELS.iter().find(|e| e.0 == k && self.k.contains_key(e.1)) { k = e.1; x = x_largs(sorted_wg(e.0, e.4)); } }
            d.push(Dispatch { stage, kernel: k, bg: if k == $k { $bg.to_string() } else { k.to_string() }, x })
        } } }
        push!("build.tick", "build.tick", 1);
        if self.opts.record_events && !js { push!("stepMark", "stepMark", 1); }
        if first && op_count > 0 { stage = "ops"; push!("ops", "ops", g(op_count, 64)); }
        let fresh = first && self.new_pose;
        if self.queue_high > 0 || self.lot > 0 || fresh {
            stage = "spawn"; if fresh { self.new_pose = false; }
            if self.lot > 0 { push!("leave", "leave", g(cap as usize, 128)); }
            if self.queue_high > 0 { push!("spawn", "spawn", g(self.queue_high, 64)); }
            push!("integrate.newPose", "integrate.newPose", g(cap as usize, 256));
        }
        if f.collisions {
            stage = "contacts"; if so && !js && self.k.contains_key("contacts.barrier.sorted") { push!("listArgs", "listArgs", 1); }
            if fs { push!("hash.hcount", "hash.hcount.acc", g(cap as usize, 256)); self.push_fast_scan(&mut d, stage, "fhscan", true); } else {
                push!("hash.hclear", "hash.hclear", g(self.t_size + 1, 256)); push!("hash.hcount", "hash.hcount", g(cap as usize, 256));
                for l in 0..self.hscan.len() { push!("scanBlocks", &format!("hscanB{l}"), self.hscan[l].blocks as u32); }
                for l in (0..self.hscan.len().saturating_sub(1)).rev() { push!("addSums", &format!("hscanA{l}"), self.hscan[l].blocks as u32); }
            }
            push!("hash.hscatter", "hash.hscatter", g(cap as usize, 256));
            stage = "barrier"; push!("contacts.barrier", "contacts.barrier", g(cap as usize, 128));
            stage = "detect"; push!("contacts.detect", "contacts.detect", g(cap as usize, 128));
            stage = "resolve";
            if self.opts.fused_resolve && !self.js_stages { push!("contacts.resolveFused", "contacts.resolveFused", 1); } else {
                let wg = if self.opts.indirect_resolve && !self.js_stages { push!("contactArgs", "contactArgs", 1); 0 } else { g(self.opts.contact_cap, 128) };
                for _ in 0..self.opts.resolve_rounds { push!("contacts.ownClear", "contacts.ownClear", wg); push!("contacts.ownMin", "contacts.ownMin", wg); push!("contacts.resolveRound", "contacts.resolveRound", wg); }
                push!("contacts.resolve", "contacts.resolve", 1);
            }
        }
        stage = "lists";
        if fs { push!("build.count", "build.count.acc", g(cap as usize, 256)); self.push_fast_scan(&mut d, stage, "fscan", false); } else {
            push!("build.clear", "build.clear", g(self.n_counts, 256)); push!("build.count", "build.count", g(cap as usize, 256));
            for l in 0..self.scan.len() { push!("scanBlocks", &format!("scan{l}"), self.scan[l].blocks as u32); }
            for l in (0..self.scan.len().saturating_sub(1)).rev() { push!("addSums", &format!("add{l}"), self.scan[l].blocks as u32); }
        }
        push!("build.scatter", "build.scatter", g(cap as usize, 256)); push!("build.sortLanes", "build.sortLanes", g(cap as usize, 256));
        let sorted = self.opts.sorted_order && !self.js_stages;
        if sorted { push!("listArgs", "listArgs", 1); }
        let hj = self.blob.j_m.is_some() && !f.empty;
        if hj { stage = "jlists"; push!("junction.clear", "junction.clear", g(self.j_m() + self.blob.e, 256)); push!("junction.clearOff", "junction.clearOff", g(self.blob.e, 256)); push!("junction.link", "junction.link", g(cap as usize, 256)); }
        stage = "leaders"; push!("leaders", "leaders", g(cap as usize, 128));
        if hj {
            stage = "junctions";
            if f.collisions { push!("junction.perceive", "junction.perceive", g(cap as usize, 128)); }
            push!("courtesy.propose", "courtesy.propose", g(cap as usize, 128)); push!("courtesy.apply", "courtesy.apply", g(cap as usize, 128));
            if self.opts.deadlock_break != 0 { push!("junction.dlFind", "junction.dlFind", g(cap as usize, 128)); push!("junction.dlApply", "junction.dlApply", g(cap as usize, 128)); }
            push!("junction.yield", "junction.yield", g(cap as usize, 128));
            let every = jsmath::round(2.0 / self.dt).max(1.0) as u64;
            if (self.closed_n > 0 || !self.traffic_penalty.is_empty()) && self.opts.reroute && step_index % every == 0 { push!("junction.reroute", "junction.reroute", g(cap as usize, 128)); }
        }
        if f.policy {
            if self.opts.obs != ObsMode::None { stage = "sense"; push!("sense", "sense", g(cap as usize, 128)); }
            if self.opts.obs == ObsMode::V3 { stage = "perceive"; push!("perceive", "perceive", g(cap as usize, 64)); }
            if !js { stage = "collided"; push!("collided", "collided", g(cap as usize, 256)); }
            stage = "act"; push!("act", "act", g(cap as usize, 128));
        }
        stage = "integrate"; push!("integrate", "integrate", g(cap as usize, 256));
        d
    }

    fn encode(&mut self, enc: &mut wgpu::CommandEncoder, n: usize, op_count: usize) {
        use crate::profile::Profiler as P;
        let cargs = self.gpu.get("cargs"); let largs = self.gpu.get("largs");
        for k in 0..n {
            let si = self.step + k as u64 + 1;
            let st = self.stages(si, k == 0, op_count);
            let (kt, bgt) = (&self.k, &self.bg);
            let rec = |pass: &mut wgpu::ComputePass, d: &Dispatch| {
                pass.set_pipeline(&kt[d.kernel].pipeline);
                for (gi, b) in bgt[d.bg.as_str()].iter().enumerate() { pass.set_bind_group(gi as u32, b, &[]); }
                match d.x { 0 => pass.dispatch_workgroups_indirect(&cargs, 0), x if x >= X_LARGS => pass.dispatch_workgroups_indirect(&largs, 16 * (u32::MAX - x) as u64),
                    x => pass.dispatch_workgroups(x, 1, 1) }
            };
            match self.profiler.as_mut() {
                Some(p) if k == 0 && st.len() <= p.max_dispatches() => {
                    p.pending.clear();
                    if p.inside {
                        let mut pass = enc.begin_compute_pass(&Default::default());
                        for (i, d) in st.iter().enumerate() { let q = P::dispatch_index(i); pass.write_timestamp(&p.qs, q); rec(&mut pass, d); pass.write_timestamp(&p.qs, q + 1); p.pending.push((d.stage, d.kernel)); }
                    } else {
                        for (i, d) in st.iter().enumerate() {
                            let q = P::dispatch_index(i); let tw = wgpu::ComputePassTimestampWrites { query_set: &p.qs, beginning_of_pass_write_index: Some(q), end_of_pass_write_index: Some(q + 1) };
                            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(d.kernel), timestamp_writes: Some(tw) });
                            rec(&mut pass, d); p.pending.push((d.stage, d.kernel));
                        }
                    }
                }
                Some(p) if k == 1 => {
                    let i = p.step_index(); p.step_pair = true;
                    let tw = wgpu::ComputePassTimestampWrites { query_set: &p.qs, beginning_of_pass_write_index: Some(i), end_of_pass_write_index: Some(i + 1) };
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: Some(tw) });
                    for d in &st { rec(&mut pass, d); }
                }
                _ => {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                    for d in &st { rec(&mut pass, d); }
                }
            }
        }
        if let Some(p) = &self.profiler { p.resolve(enc); }
    }

    fn patched(&self, src: String) -> String { if self.fast_search_ok { kernels::patch(&src) } else { src } }
    /// fast_search precondition: every edge polyline's cumulative distance (pt.w) is non-decreasing.
    fn polylines_monotone(&self) -> bool {
        let w = &self.blob.words; let lo = self.blob.off.pts as usize;
        if w.is_empty() { return false; }
        let edges = (0..self.blob.e).all(|e| {
            let (b, n) = (w[e * 16] as usize, w[e * 16 + 1] as usize);
            (1..n).all(|k| f32::from_bits(w[lo + 4 * (b + k) + 3]) >= f32::from_bits(w[lo + 4 * (b + k - 1) + 3]))
        });
        // junction connectors: movW(m, 2) = first point, movW(m, 3) = count, point distance at jPts + 4 k + 3
        let (mo, jp) = (self.blob.off.j_mov as usize, self.blob.off.j_pts as usize);
        let conns = (0..self.blob.j_m.unwrap_or(0)).all(|m| {
            let (f, n) = (w[mo + 9 * m + 2] as usize, w[mo + 9 * m + 3] as usize);
            (1..n).all(|k| f32::from_bits(w[jp + 4 * (f + k) + 3]) >= f32::from_bits(w[jp + 4 * (f + k - 1) + 3]))
        });
        edges && conns
    }

    /// Source file of a kernel with the options (modified copies in crates/bayenv-gpu/kernels/).
    fn kernel_file(&self, file: &'static str) -> &'static str {
        file
    }

    /// scan4k levels: scanFirst, scanInPlace per upper level, then the adds top-down.
    fn push_fast_scan(&self, d: &mut Vec<Dispatch>, stage: &'static str, pre: &str, hash: bool) {
        let lv = if hash { &self.fhscan } else { &self.fscan };
        d.push(Dispatch { stage, kernel: "scan4k.first", bg: format!("{pre}F"), x: lv[0].blocks as u32 });
        for l in 1..lv.len() { d.push(Dispatch { stage, kernel: "scan4k.inPlace", bg: format!("{pre}I{l}"), x: lv[l].blocks as u32 }); }
        for l in (0..lv.len().saturating_sub(1)).rev() { d.push(Dispatch { stage, kernel: "scan4k.add", bg: format!("{pre}A{l}"), x: lv[l].blocks as u32 }); }
    }

    /// The JS dispatch list (kernel, x) of the next step, as `gpuSim.js _stages` would record it (diagnostics /
    /// oracle comparison: the host optimisations of `GpuOptions` are left out).
    pub fn peek_stages(&mut self, first: bool, op_count: usize) -> Vec<(&'static str, u32)> {
        let np = self.new_pose; self.js_stages = true;
        let v = self.stages(self.step + 1, first, op_count).into_iter().map(|d| (d.kernel, d.x)).collect(); self.new_pose = np; self.js_stages = false; v
    }
    /// The dispatch list of the next step as encoded with the current options (stage, kernel, x; x = 0: indirect).
    pub fn peek_dispatches(&mut self, first: bool, op_count: usize) -> Vec<Dispatch> {
        let np = self.new_pose; let v = self.stages(self.step + 1, first, op_count); self.new_pose = np; v
    }

    /// Run n sim steps in one submit, then read back stats + events and process them (host side).
    pub fn run_steps(&mut self, n: usize) { self.run_steps_into(n, None, None); }

    /// `run_steps(n)` plus, in the same submit, a copy of the observation rows (`obs.len() / obs_size` slots) and/or
    /// the first `veh.len() / VEH_WORDS` vehicle records into the caller's buffers: one GPU round trip per call.
    pub fn run_steps_into(&mut self, n: usize, obs: Option<&mut [f32]>, veh: Option<&mut [u32]>) {
        self.sync();
        if n == 0 { if let Some(o) = obs { self.read_into("inputs", 0, bytemuck::cast_slice_mut(o)); } if let Some(v) = veh { self.read_into("veh", 0, bytemuck::cast_slice_mut(v)); } return; }
        let p = self.submit_steps(n, obs.as_ref().map_or(0, |o| o.len()), veh.as_ref().map_or(0, |v| v.len()));
        self.gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        self.finish_steps(p, obs, veh);
    }

    /// First half of `run_steps_into`: host work, encode, submit, map requests (no wait). Call `finish_steps`
    /// after the device was polled (several sims on one device: submit all, poll once, finish all).
    pub fn submit_steps(&mut self, n: usize, obs_floats: usize, veh_words: usize) -> Pending { self.submit_steps_obs(n, obs_floats, None, veh_words) }

    /// `submit_steps` that also writes the observation rows into `dst` (an `ObsRing` slot / any COPY_DST buffer) in
    /// the same submit (F16: converted on the GPU first).
    pub fn submit_steps_obs(&mut self, n: usize, obs_floats: usize, dst: Option<&ObsDst>, veh_words: usize) -> Pending {
        self.sync();
        let t_end = self.time + n as f64 * self.dt;
        self.enqueue_spawns(t_end);
        self.rec_mark = self.rec.len();
        self.scenes();
        self.pumps();
        self.process_refuels();
        let mut sampled = false;
        if self.has_junctions() {
            self.update_closed(); self.process_reroutes();
            if self.time >= self.next_traffic { self.next_traffic = self.time + 60.0; self.traffic_sample(); sampled = true; }
        }
        let op_count = self.flush_ops();
        self.write_params(op_count);
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        if n > 0 { enc.clear_buffer(&self.gpu.get("cflag"), 0, None); self.encode(&mut enc, n, op_count); }   // per-batch collided-at-sense counts
        // io: header + the first EV_FAST events (the rest, rarely needed, is read after the map)
        let fast = (io::EVDATA + 4 * EV_FAST.min(self.opts.ev_cap)) * 4;
        enc.copy_buffer_to_buffer(&self.gpu.get("io"), 0, &self.staging, 0, fast as u64);
        enc.clear_buffer(&self.gpu.get("io"), (io::EVCOUNT * 4) as u64, Some(4));
        if let Some(d) = dst { self.encode_obs(&mut enc, d); }
        let (ob4, vb4) = (obs_floats * 4, veh_words * 4);
        if ob4 + vb4 > 0 {
            if self.out_staging.as_ref().map_or(true, |b| b.size() < (ob4 + vb4) as u64) {
                self.out_staging = Some(self.gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("outStaging"), size: (ob4 + vb4) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }));
            }
            let os = self.out_staging.as_ref().unwrap();
            if ob4 > 0 { enc.copy_buffer_to_buffer(&self.gpu.get("inputs"), 0, os, 0, ob4 as u64); }
            if vb4 > 0 { enc.copy_buffer_to_buffer(&self.gpu.get("veh"), 0, os, ob4 as u64, vb4 as u64); }
        }
        self.gpu.queue.submit([enc.finish()]);
        self.time = t_end; self.step += n as u64;
        if sampled { if let Some((p, o)) = self.pending_penalty.take() { self.traffic_penalty = p; self.penalty_order = o; } }
        self.staging.slice(..fast as u64).map_async(wgpu::MapMode::Read, |r| r.expect("map io"));
        if ob4 + vb4 > 0 { self.out_staging.as_ref().unwrap().slice(..(ob4 + vb4) as u64).map_async(wgpu::MapMode::Read, |r| r.expect("map out")); }
        Pending { fast, ob4, vb4 }
    }

    /// Submit n steps and return without waiting; the readback and events are processed by the next call that
    /// needs them (`sync`, any `run_steps*`, `submit_steps`). Same results as `run_steps(n)`.
    pub fn run_steps_async(&mut self, n: usize) {
        let p = self.submit_steps(n, 0, 0);
        self.pending = Some(p);
    }
    /// `run_steps_async` with the observation readback of `dst` in the same submit (lagged / async stepping).
    pub fn run_steps_async_obs(&mut self, n: usize, dst: &ObsDst) {
        let p = self.submit_steps_obs(n, 0, Some(dst), 0);
        self.pending = Some(p);
    }
    /// Make `p` (from `submit_steps*`) the lazily finished pending batch (`sync`), as `run_steps_async` does.
    pub fn defer(&mut self, p: Pending) { assert!(self.pending.is_none(), "defer: a batch is already pending"); self.pending = Some(p); }
    /// Copy the current observation rows into `dst` (no steps; waits for a pending batch first). Submits only:
    /// map / poll the destination afterwards.
    pub fn submit_obs(&mut self, dst: &ObsDst) {
        self.sync();
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        self.encode_obs(&mut enc, dst);
        self.gpu.queue.submit([enc.finish()]);
    }
    /// Record the observation readback of `dst.rows` slots into `dst` (F16: kernels/obs_f16.wgsl into `obsHalf`, then a copy).
    pub fn encode_obs(&mut self, enc: &mut wgpu::CommandEncoder, dst: &ObsDst) {
        assert!(self.opts.obs != ObsMode::None, "observation readback: created with obs none");
        let floats = dst.rows.min(self.cap) * self.obs_size;
        if floats == 0 { return; }
        match dst.fmt {
            ObsFmt::F32 => enc.copy_buffer_to_buffer(&self.gpu.get("inputs"), 0, &dst.buf, dst.offset, (floats * 4) as u64),
            ObsFmt::F16 => {
                if self.half.is_none() {
                    let k = self.gpu.kernel("obsF16", kernels::host_raw("obs_f16"), "main").expect("obs_f16 kernel");
                    self.gpu.storage("obsHalf", (self.cap * self.obs_size * 2) as u64, None, wgpu::BufferUsages::empty());
                    self.gpu.uniform("obsHalfP", 16);
                    let mut r = HashMap::new();
                    r.insert("HP", self.gpu.get("obsHalfP")); r.insert("inputs", self.gpu.get("inputs")); r.insert("obsHalf", self.gpu.get("obsHalf"));
                    let bg = self.gpu.bind_group(&k, &r, 0);
                    self.half = Some((k, bg));
                }
                let words = (floats + 1) / 2;
                let wgs = ((words + 255) / 256).min(65535) as u32;
                self.gpu.write("obsHalfP", bytemuck::cast_slice(&[words as u32, wgs * 256, 0, 0]), 0);
                let (k, bg) = self.half.as_ref().unwrap();
                { let mut p = enc.begin_compute_pass(&Default::default()); p.set_pipeline(&k.pipeline); p.set_bind_group(0, bg, &[]); p.dispatch_workgroups(wgs, 1, 1); }
                enc.copy_buffer_to_buffer(&self.gpu.get("obsHalf"), 0, &dst.buf, dst.offset, (words * 4) as u64);
            }
        }
    }
    /// Finish a `run_steps_async` batch (wait for it, ingest its events). No-op when nothing is pending.
    pub fn sync(&mut self) {
        if let Some(p) = self.pending.take() {
            self.gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
            self.finish_steps(p, None, None);
        }
    }

    /// Second half: read the mapped io / outputs, process events.
    pub fn finish_steps(&mut self, p: Pending, obs: Option<&mut [f32]>, veh: Option<&mut [u32]>) {
        let cnt;
        {
            let m = self.staging.slice(..p.fast as u64).get_mapped_range().expect("io range");
            let words: &[u32] = bytemuck::cast_slice(&m);
            cnt = (words[io::EVCOUNT] as usize).min(self.opts.ev_cap);
            let used = (io::EVDATA + 4 * cnt).min(p.fast / 4);
            self.io_host[..used].copy_from_slice(&words[..used]);
        }
        self.staging.unmap();
        if cnt > EV_FAST {   // event data is still in the io buffer (only the count was cleared)
            let (a, b) = (io::EVDATA + 4 * EV_FAST, io::EVDATA + 4 * cnt);
            let mut rest = vec![0u32; b - a]; self.gpu.read_into("io", (a * 4) as u64, bytemuck::cast_slice_mut(&mut rest));
            self.io_host[a..b].copy_from_slice(&rest);
        }
        if p.ob4 + p.vb4 > 0 {
            let os = self.out_staging.as_ref().unwrap();
            { let m = os.slice(..(p.ob4 + p.vb4) as u64).get_mapped_range().expect("out range");
              if let Some(o) = obs { bytemuck::cast_slice_mut::<f32, u8>(o).copy_from_slice(&m[..p.ob4]); }
              if let Some(v) = veh { bytemuck::cast_slice_mut::<u32, u8>(v).copy_from_slice(&m[p.ob4..p.ob4 + p.vb4]); } }
            os.unmap();
        }
        if let Some(p) = self.profiler.as_mut() { p.collect(&self.gpu.device); }
        self.ingest();
    }

    /// `_trafficSample`: per-edge mean speed -> EMA -> penalty (applied after this frame's encode, like the async JS).
    fn traffic_sample(&mut self) {
        let e_n = self.blob.e;
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        {
            let mut p = enc.begin_compute_pass(&Default::default());
            for (kn, x) in [("density.trafClear", ((e_n + 255) / 256) as u32), ("density.trafCount", ((self.cap + 255) / 256) as u32)] {
                p.set_pipeline(&self.k[kn].pipeline);
                for (gi, b) in self.bg[kn].iter().enumerate() { p.set_bind_group(gi as u32, b, &[]); }
                p.dispatch_workgroups(x.max(1), 1, 1);
            }
        }
        self.gpu.queue.submit([enc.finish()]);
        let cnt: Vec<u32> = bytemuck::cast_slice(&self.gpu.read("edgeCount", 0, (e_n * 4) as u64)).to_vec();
        let sum: Vec<u32> = bytemuck::cast_slice(&self.gpu.read("edgeSpeed", 0, (e_n * 4) as u64)).to_vec();
        let g = &self.world.graph;
        // EMA map in JS insertion order (Map semantics: delete + set re-appends)
        let mut ema = std::mem::take(&mut self.edge_speed_ema); let mut order = std::mem::take(&mut self.edge_speed_order);
        let mut keep = Vec::with_capacity(order.len());
        for &e in &order {
            let Some(&v) = ema.get(&e) else { continue };
            if cnt[e as usize] == 0 { let lim = g.speed_limit[e as usize]; let nv = 0.5 * v + 0.5 * lim; if nv > 0.95 * lim { ema.remove(&e); } else { ema.insert(e, nv); keep.push(e); } } else { keep.push(e); }
        }
        order = keep;
        for e in 0..e_n { if cnt[e] != 0 {
            let m = sum[e] as f64 / 100.0 / cnt[e] as f64;
            match ema.get(&(e as u32)) { None => { ema.insert(e as u32, m); order.push(e as u32); } Some(&old) => { ema.insert(e as u32, 0.5 * old + 0.5 * m); } }
        } }
        let mut pen = HashMap::new(); let mut po = Vec::new();
        for &e in &order { let v = ema[&e]; let (len, lim) = (g.length[e as usize], g.speed_limit[e as usize]);
            if v >= 0.7 * lim { continue; } pen.insert(e, len / jsmath::max(0.5, v) - len / lim); po.push(e); }
        self.edge_speed_ema = ema; self.edge_speed_order = order;
        self.pending_penalty = Some((pen, po));
    }

    fn flush_ops(&mut self) -> usize {
        let n = (self.ops_pending.len() / 4).min(self.opts.op_cap);
        if n == 0 { return 0; }
        let w: Vec<u32> = self.ops_pending.drain(..n * 4).collect();
        self.gpu.write("ops", bytemuck::cast_slice(&w), 0);
        n
    }
    fn op(&mut self, code: u32, slot: u32, a: u32, b: u32) { self.ops_pending.extend_from_slice(&[code, slot, a, b]); }

    fn event(&mut self, kind: &'static str, slot: i64, a: f64, b: f64) {
        if self.opts.keep_events { self.events.push(Event { time: self.time, kind, slot, a, b }); if self.events.len() > 20000 { self.events.drain(..10000); } }
    }

    fn ingest(&mut self) {
        let io = std::mem::take(&mut self.io_host);
        self.gpu_stats.copy_from_slice(&io[io::STATS..io::STATS + 64]);
        let n = io[io::EVCOUNT] as usize; let cap = self.opts.ev_cap;
        if n > cap { self.counters.events_dropped += (n - cap) as u64; }
        let m = n.min(cap); let base = io::EVDATA;
        for k in 0..m { let t = io[base + 4 * k];
            if t & 255 == ev::WRECK { self.wrecks.insert(io[base + 4 * k + 1], Wreck { lo: (t >> 8) & 15, hi: (t >> 12) & 15, ty: (t >> 16) & 255, edge: io[base + 4 * k + 2], s: f32::from_bits(io[base + 4 * k + 3]) as f64 }); } }
        if self.opts.record_events { self.record_batch(&io[base..base + 4 * m]); }
        for k in 0..m {
            let o = base + 4 * k; let t = io[o] & 255; let slot = io[o + 1]; let a = io[o + 2]; let b = io[o + 3];
            match t {
                ev::ARRIVE => { self.arrive(slot, f32::from_bits(b) as f64); }
                ev::CTLVIOL => { if b & 255 == 1 { self.counters.signal_violations += 1; } else { self.counters.stop_violations += 1; } self.event("signalViolation", slot as i64, a as f64, (b >> 8) as f64 / 10.0); }
                ev::TRIPX => {}
                ev::CRASHX_V => { self.crash_x.insert(slot, CrashX { other: a, pre_a: (b & 0xFFFF) as f64 / 10.0, pre_b: (b >> 16) as f64 / 10.0, strike_edge: -1, ..Default::default() }); }
                ev::MISSED => {
                    let via = (a >> 31) == 1; let took = a & 0x7fff_ffff;
                    self.counters.missed_exits += 1; self.event("missedExit", slot as i64, took as f64, via as u8 as f64);
                    self.reroute_queue.push_back(Reroute { slot, edge: took, trip: self.slot_trip[slot as usize], missed: true, slow: false, rp: 0, prefix: if via { b as i64 } else { -1 }, t: self.time });
                }
                ev::REROUTE => self.reroute_queue.push_back(Reroute { slot, edge: a, trip: self.slot_trip[slot as usize], missed: false, slow: (b >> 31) == 1, rp: b & 0x7fff_ffff, prefix: -1, t: self.time }),
                ev::CRASHX_M => { if let Some(c) = self.crash_x.get_mut(&slot) { c.maneuver = a & 255; c.any_crashed = (a & 768) != 0; c.strike_edge = b as i64; } }
                ev::CRASHX_N => { if let Some(c) = self.crash_x.get_mut(&slot) { c.nx = bf(a); c.nz = bf(b); } }
                ev::CRASHX_P => { if let Some(c) = self.crash_x.get_mut(&slot) { c.x = bf(a); c.z = bf(b); } }
                ev::CRASHX_H => { if let Some(c) = self.crash_x.get_mut(&slot) { let hh = |u: u32| u as f64 / 65536.0 * 2.0 * std::f64::consts::PI - std::f64::consts::PI;
                    c.ha = hh(a & 0xFFFF); c.hb = hh(a >> 16); c.closing = (b & 0xFFFF) as f64 / 10.0; c.lane_a = (b >> 16) & 255; c.lane_b = b >> 24; c.has_h = true; } }
                ev::STUCK => { self.counters.stuck_labels += 1; self.event("stuck", slot as i64, self.slot_trip[slot as usize] as f64, bf(b)); }
                ev::SPAWN => { self.queue_free.push(a); self.counters.spawned += 1; self.event("spawn", slot as i64, self.slot_trip[slot as usize] as f64, b as f64); }
                ev::LOWFUEL => self.refuel_queue.push((slot, a)),
                ev::STATION => self.enter_station(slot),
                ev::REENTER => { if self.x.in_lot.remove(&slot) { self.lot -= 1; } self.slot_station[slot as usize] = -1; self.counters.reentered += 1; self.event("refuelLeave", slot as i64, 0.0, 0.0); }
                ev::CRASH => { let cx = self.crash_x.remove(&slot); self.crash_v4(slot, a, (b & 0xFFFF) as f64 / 10.0, (b >> 16) as f64 / 10.0, cx); }
                ev::CONTACT => {
                    let dv = jsmath::max((b & 0xFFFF) as f64 / 10.0, (b >> 16) as f64 / 10.0);
                    if dv >= 3.0 { if io[o] & 256 != 0 { self.pull(slot); } if (io[o] & 512) != 0 && a != NONE { self.pull(a); } }
                    self.counters.minor_contacts += 1; self.counters.crash_events += 1; self.by_sev(dv);
                }
                ev::STRANDED => self.strand(slot),
                _ => {}
            }
        }
        if self.opts.record_events { let m0 = self.rec_mark.min(self.rec.len()); self.rec.sort_from(m0); self.rec_mark = self.rec.len(); }   // canonical (step, slot, kind) order
        self.io_host = io;
    }

    /// Per-slot records of one batch of GPU events (before the host handlers change slot / crash state), then one
    /// gathered readback of the involved slots' veh / par records for progress, exact arrival time and lateness.
    fn record_batch(&mut self, evs: &[u32]) {
        use crate::events::kind as K;
        let first = self.rec.len(); let t_now = self.time; let mut st: u64 = self.step; let mut t_step = t_now;
        let mut new_crash: HashSet<u32> = HashSet::new();
        let mut tripx: HashMap<u32, (f32, f32)> = HashMap::new();
        let mut share: HashMap<u32, f32> = HashMap::new();
        let agent = |st: &Vec<i32>, s: u32| -> i64 { st.get(s as usize).copied().unwrap_or(-1) as i64 };
        for e in evs.chunks_exact(4) {
            let (t, slot, a, b) = (e[0] & 255, e[1], e[2], e[3]);
            match t {
                254 => { st = a as u64; t_step = self.time - (self.step.saturating_sub(st)) as f64 * self.dt; }   // step marker (kernels/step_mark.wgsl): the following events are of step a
                ev::TRIPX => { tripx.insert(slot, (f32::from_bits(a), f32::from_bits(b))); }
                ev::CRASHPEN => { share.insert(slot, ((b >> 16) & 255) as f32 / 255.0); }
                ev::ARRIVE => {
                    self.rec.push(st, t_step, K::ARRIVED, slot, agent(&self.slot_trip, slot), -1, f32::NAN, f32::NAN, f32::NAN);
                    if let Some(&(m, d)) = tripx.get(&slot) { let i = self.rec.len() - 1; self.rec.trip_m[i] = m; self.rec.trip_s[i] = d; }
                }
                ev::CRASH | ev::CONTACT => {
                    let (di, dj) = ((b & 0xFFFF) as f32 / 10.0, (b >> 16) as f32 / 10.0);
                    let fresh = |me: &Self, s: u32, nc: &HashSet<u32>| me.wrecks.contains_key(&s) && !me.in_crash.contains(&s) && !nc.contains(&s);
                    let pair: Vec<(u32, u32, f32, f32)> = if a != NONE { vec![(slot, a, di, dj), (a, slot, dj, di)] } else { vec![(slot, NONE, di, f32::NAN)] };
                    for (me, other, d, od) in pair {
                        let k = if t == ev::CRASH && fresh(self, me, &new_crash) { new_crash.insert(me); K::CRASHED } else { K::CONTACT };
                        self.rec.push(st, t_step, k, me, agent(&self.slot_trip, me), if other == NONE { -1 } else { other as i64 }, d, od, share.remove(&me).unwrap_or(f32::NAN));
                    }
                }
                ev::STRANDED => self.rec.push(st, t_step, K::STRANDED, slot, agent(&self.slot_trip, slot), -1, f32::NAN, f32::NAN, f32::NAN),
                ev::SPAWN => self.rec.push(st, t_step, K::SPAWNED, slot, agent(&self.slot_trip, slot), -1, f32::NAN, f32::NAN, f32::NAN),
                ev::MISSED => {
                    if self.opts.missed_at_out_edge && (a >> 31) == 1 { self.missed_pending.push((slot, a & 0x7fff_ffff, agent(&self.slot_trip, slot))); }
                    else { self.rec.push(st, t_step, K::MISSED_EXIT, slot, agent(&self.slot_trip, slot), -1, f32::NAN, f32::NAN, f32::NAN); }
                }
                ev::CTLVIOL => self.rec.push(st, t_step, K::VIOLATION, slot, agent(&self.slot_trip, slot), -1, f32::NAN, f32::NAN, f32::NAN),
                _ => {}
            }
        }
        if self.rec.len() == first && self.missed_pending.is_empty() { return; }
        // gather the involved slots' records (one submit, one map)
        let mut slots: Vec<u32> = self.rec.slot[first..].iter().map(|&s| s as u32).chain(self.missed_pending.iter().map(|p| p.0)).collect(); slots.sort_unstable(); slots.dedup();
        let (vw, pw) = (L::VEH_WORDS * 4, L::PAR_WORDS * 4); let n = slots.len();
        let rb = self.gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("recGather"), size: (n * (vw + pw)) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        let (vb, pb) = (self.gpu.get("veh"), self.gpu.get("par"));
        for (i, &sl) in slots.iter().enumerate() {
            enc.copy_buffer_to_buffer(&vb, (sl as usize * vw) as u64, &rb, (i * vw) as u64, vw as u64);
            enc.copy_buffer_to_buffer(&pb, (sl as usize * pw) as u64, &rb, (n * vw + i * pw) as u64, pw as u64);
        }
        self.gpu.queue.submit([enc.finish()]);
        rb.slice(..).map_async(wgpu::MapMode::Read, |r| r.expect("map rec"));
        self.gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let words: Vec<u32> = bytemuck::cast_slice(&rb.slice(..).get_mapped_range().expect("rec")).to_vec();
        rb.unmap();
        let vrec = |i: usize, f: usize| f32::from_bits(words[i * L::VEH_WORDS + f]);
        let prec = |i: usize, f: usize| f32::from_bits(words[n * L::VEH_WORDS + i * L::PAR_WORDS + f]);
        // pending connector missed exits: report once the car is on the taken edge (step = end of this batch; exact with k = 1)
        if !self.missed_pending.is_empty() {
            let pend = std::mem::take(&mut self.missed_pending);
            for (sl, took, ag) in pend {
                let i = slots.binary_search(&sl).unwrap();
                let (vs, ve) = (words[i * L::VEH_WORDS + veh::STATE], words[i * L::VEH_WORDS + veh::EDGE]);
                if self.slot_trip[sl as usize] as i64 != ag || vs == 0 { continue; }   // trip ended / slot reused: drop
                if ve == took { self.rec.push(self.step, t_now, K::MISSED_EXIT, sl, ag, -1, f32::NAN, f32::NAN, f32::NAN); } else { self.missed_pending.push((sl, took, ag)); }
            }
        }
        for r in first..self.rec.len() {
            let i = slots.binary_search(&(self.rec.slot[r] as u32)).unwrap();
            self.rec.progress_m[r] = vrec(i, veh::ROUTE_DIST);
            if self.rec.kind[r] == K::ARRIVED && self.rec.trip_s[r].is_finite() {
                let arr = vrec(i, veh::SPAWN_TIME) as f64 + self.rec.trip_s[r] as f64;
                self.rec.time[r] = arr;
                if self.rec.agent[r] >= 0 && self.x.has_pop() { self.rec.late_by[r] = (arr - prec(i, 13) as f64) as f32; }
            }
        }
    }
    /// Take the per-slot event records collected since the last call.
    pub fn take_records(&mut self) -> crate::events::EventLog { self.sync(); self.rec.drain() }

    fn by_sev(&mut self, kmh: f64) -> &'static str {
        let (i, s) = if kmh < 10.0 { (0, "minor") } else if kmh < 25.0 { (1, "moderate") } else if kmh < 50.0 { (2, "severe") } else { (3, "very severe") };
        self.counters.by_severity[i] += 1; s
    }
    fn pull(&mut self, slot: u32) { self.counters.pull_overs += 1; let until = self.time + 300.0 + 600.0 * self.rand(); self.op(op::SET_PULL, slot, f32bits(until), 0); }

    fn release(&mut self, slot: u32, outcome: &'static str) { self.release_fuel(slot, outcome, 0.0); }
    /// `_release(slot, outcome, fuel)`: frees the slot, records the commuter's fuel / outcome (populations).
    fn release_fuel(&mut self, slot: u32, outcome: &'static str, fuel: f64) {
        if self.slot_trip[slot as usize] == -2 { return; }
        let tr = self.slot_trip[slot as usize];
        if self.x.has_pop() && tr >= 0 {
            let id = (tr >> 1) as usize;
            if outcome == "arrived" { self.x.fuel_left[id] = fuel as f32; }
            else if outcome == "towed" { let ty = self.x.vehicle_type(id); self.x.fuel_left[id] = (0.2 * VEHICLE_PARAMS[ty.min(3)].tank_l) as f32; }
            if tr & 1 == 0 && self.x.outcome[id] == 0 { self.x.outcome[id] = match outcome { "arrived" => 1, "crash" => 2, "stranded" => 3, "stuck" => 4, _ => 0 }; }
        }
        if self.opts.record_events && (outcome == "towed" || outcome == "cleared") {
            self.rec.push(self.step + 1, self.time, crate::events::kind::CLEARED, slot, tr as i64, -1, f32::NAN, f32::NAN, f32::NAN);
        }
        self.slot_trip[slot as usize] = -2;
        self.arena_free(slot);
        self.free.push(slot);
        self.event(outcome, slot as i64, tr as f64, 0.0);
    }

    /// `incidentTimeline(severity, rng)`: (notify, arrive, block, clear)
    fn incident_timeline(&mut self, severity: &str) -> (f64, f64, f64, f64) {
        let (median, sigma, lo, hi, block_frac, tow): (f64, f64, f64, f64, Option<f64>, f64) = match severity {
            "severe" => (50.0 * 60.0, 0.25, 45.0, 75.0, Some(0.8), 0.0), "very severe" => (150.0 * 60.0, 0.3, 120.0, 240.0, Some(0.9), 0.0),
            "minor" => (20.0 * 60.0, 0.4, 15.0, 30.0, Some(0.0), 0.0), _ => (20.0 * 60.0, 0.4, 10.0, 30.0, None, 5.0 * 60.0) };
        let notify = 3.4 * 60.0 * jsmath::exp(0.3 * self.gauss());
        let arrive = notify + 6.0 * 60.0 * jsmath::exp(0.3 * self.gauss());
        let clear = jsmath::min(hi * 60.0, jsmath::max(lo * 60.0, median * jsmath::exp(sigma * self.gauss())));
        let mut block = match block_frac { None => arrive + tow * jsmath::exp(0.3 * self.gauss()), Some(f) => f * clear };
        block = jsmath::min(block, clear);
        (notify, arrive, block, jsmath::max(clear, block + 60.0))
    }

    /// `poseAt(g, ei, s, lat)`: [x, y, z, heading]
    pub fn pose_at(&self, ei: usize, s: f64, lat: f64) -> [f64; 4] {
        let g = &self.world.graph; let r = g.pts(ei); let (px, pz, cum) = (&g.px[r.clone()], &g.pz[r.clone()], &g.cum[r.clone()]); let n = px.len();
        let off = g.right_edge[ei] - lat;
        let mut i = 0; while i < n - 2 && cum[i + 1] < s { i += 1; }
        let seg = cum[i + 1] - cum[i];
        let t = if seg > 0.0 { jsmath::min(1.0, jsmath::max(0.0, (s - cum[i]) / seg)) } else { 0.0 };
        let (mut dx, mut dz) = (px[i + 1] - px[i], pz[i + 1] - pz[i]); let mut ln = jsmath::hypot(dx, dz); if ln == 0.0 { ln = 1.0; } dx /= ln; dz /= ln;
        let y = if g.has(ei, bayenv_world::graph::ef::HAS_YS) { let ys = &g.ys[r]; ys[i] as f64 + (ys[i + 1] as f64 - ys[i] as f64) * t } else { g.y[ei] as f64 };
        [px[i] + dx * t * seg - dz * off, y, pz[i] + dz * t * seg + dx * off, jsmath::atan2(dz, dx)]
    }

    fn open_scene(&mut self, fresh: &[u32], cause: &'static str, severity: &'static str, lanes_blocked: u32, block_dur: f64, clear_dur: f64, is_crash: bool, all: &[u32]) -> Option<usize> {
        let first = *self.wrecks.get(&fresh[0])?;
        let e = first.edge as usize; let lanes = self.world.graph.lanes[e] as u32;
        let (mut mask, mut s_max) = (0u32, 0f64);
        for s in all { if let Some(inf) = self.wrecks.get(s) { if inf.edge == first.edge { for l in inf.lo..=inf.hi { mask |= 1 << l; } if inf.s > s_max { s_max = inf.s; } } } }
        let mut blocked = (0..lanes).filter(|l| (mask >> l) & 1 == 1).count() as u32;
        for l in 0..lanes { if blocked >= lanes_blocked { break; } if (mask >> l) & 1 == 0 { mask |= 1 << l; blocked += 1; } }
        let p = self.pose_at(e, first.s, (first.lo as f64 + 0.5) * LANE_W);
        let other_edge = all.iter().any(|q| self.wrecks.get(q).map_or(false, |w| w.edge != first.edge));
        let rec = Scene { id: self.crashes.len(), time: self.time, cause, severity, vehicles: fresh.to_vec(), lanes_blocked: blocked, lane_mask: mask, edge: first.edge, s: s_max,
            x: p[0], y: p[1], z: p[2], block_end: self.time + block_dur, clear_end: self.time + clear_dur,
            squeeze_at: self.time + jsmath::min(block_dur, (if blocked >= lanes || other_edge { 1.0 } else { 2.0 }) * self.opts.squeeze_delay),
            squeeze: false, response: 0, is_crash, block: -1, stranded: !is_crash, gone: Vec::new(), delta_v_kmh: [0.0; 2] };
        let id = rec.id; self.crashes.push(rec); self.active.push(id);
        self.block_insert(id);
        Some(id)
    }

    fn crash_v4(&mut self, slot_a: u32, slot_b: u32, dva: f64, dvb: f64, _cx: Option<CrashX>) {
        let mut slots = vec![slot_a]; if slot_b != NONE { slots.push(slot_b); }
        let severity = self.by_sev(jsmath::max(dva, dvb));
        self.counters.crash_events += 1;
        let fresh: Vec<u32> = slots.iter().cloned().filter(|s| self.wrecks.contains_key(s) && !self.in_crash.contains(s)).collect();
        if fresh.is_empty() { return; }
        for s in &fresh { self.in_crash.insert(*s); }
        let (_, _, block, clear) = self.incident_timeline(severity);
        let Some(id) = self.open_scene(&fresh, "collision", severity, 0, block, clear, true, &slots) else { return };
        self.crashes[id].delta_v_kmh = [dva, dvb];
        let secondary = slots.iter().any(|s| self.in_crash.contains(s) && !fresh.contains(s));
        self.counters.crashes += 1; self.counters.crashed_vehicles += fresh.len() as u64; if secondary { self.counters.secondary_crashes += 1; }
        for &s in &fresh { self.mark_outcome(s, 2); }
        self.event("crash", fresh[0] as i64, id as f64, jsmath::max(dva, dvb));
    }

    fn strand(&mut self, slot: u32) {
        self.in_crash.insert(slot); self.counters.stranded += 1;
        let tow = 1800.0 * jsmath::exp(0.3 * self.gauss());
        self.open_scene(&[slot], "stranded", "stranded", 1, tow, tow, false, &[slot]);
        self.mark_outcome(slot, 3);
        self.event("stranded", slot as i64, 0.0, 0.0);
    }

    fn block_write(&self, b: u32, id: usize, st: u32) {
        let r = &self.crashes[id];
        let w: [u32; 8] = [r.edge, r.lane_mask, f32bits(r.s), st, f32bits(r.time), self.block_next[b as usize], r.vehicles[0], (r.squeeze as u32) | ((r.is_crash as u32) << 1)];
        self.gpu.write("world", bytemuck::cast_slice(&w), ((self.blob.off.blk as usize + b as usize * L::BLOCK_WORDS) * 4) as u64);
    }
    fn head_write(&self, edge: u32) { self.gpu.write("world", bytemuck::cast_slice(&[self.edge_head[edge as usize]]), ((self.blob.off.eb + edge) * 4) as u64); }
    fn block_insert(&mut self, id: usize) {
        let Some(b) = self.block_free.pop() else { return };
        let e = self.crashes[id].edge;
        self.crashes[id].block = b as i32; self.block_next[b as usize] = self.edge_head[e as usize]; self.edge_head[e as usize] = b + 1;
        self.block_write(b, id, 1); self.head_write(e);
    }
    fn block_remove(&mut self, id: usize) {
        let b = self.crashes[id].block; if b < 0 { return; } let b = b as u32; let e = self.crashes[id].edge;
        let (mut prev, mut cur) = (-1i64, self.edge_head[e as usize]);
        while cur != 0 && cur - 1 != b { prev = (cur - 1) as i64; cur = self.block_next[(cur - 1) as usize]; }
        if cur != 0 {
            if prev < 0 { self.edge_head[e as usize] = self.block_next[b as usize]; self.head_write(e); }
            else { self.block_next[prev as usize] = self.block_next[b as usize];
                if let Some(&pid) = self.active.iter().find(|&&r| self.crashes[r].block == prev as i32) { let st = if self.crashes[pid].response == 1 { 2 } else { 1 }; self.block_write(prev as u32, pid, st); } }
        }
        self.block_write(b, id, 0);
        self.block_free.push(b); self.crashes[id].block = -1;
    }

    fn move_off_lanes(&mut self, id: usize) {
        let vs = self.crashes[id].vehicles.clone();
        for s in vs {
            if self.crashes[id].gone.contains(&s) || !self.wrecks.contains_key(&s) { continue; }
            let w = self.wrecks[&s]; let e = w.edge as usize; let t = &VEHICLE_PARAMS[(w.ty as usize).min(3)];
            let sh = if self.world.graph.class[e] <= 3 { 3.0 } else { 1.0 };
            if sh < t.width + 0.4 || self.world.graph.length[e] < 30.0 {
                self.crashes[id].gone.push(s); self.counters.hauled_away += 1;
                self.op(op::FREE, s, 0, 0); self.in_crash.remove(&s); self.wrecks.remove(&s);
                if self.crashes[id].stranded { self.counters.towed += 1; self.release(s, "towed"); } else { self.counters.cleared += 1; self.release(s, "cleared"); }
                continue;
            }
            self.op(op::TOW_SHOULDER, s, 0, 0); self.new_pose = true; self.counters.towed_to_shoulder += 1;
        }
    }

    fn scenes(&mut self) {
        if self.active.is_empty() { return; }
        // the JS loop runs over activeCrashes (unchanged until the end: _blockRemove searches it), keeping a new list
        let orig = self.active.clone(); let mut keep = Vec::new();
        for id in orig {
            let r = &self.crashes[id];
            if r.clear_end - r.time >= 1e8 && r.block_end - r.time >= 1e8 { keep.push(id); continue; }
            if r.response == 0 && !r.squeeze && self.time >= r.squeeze_at && self.time < r.block_end {
                self.crashes[id].squeeze = true; self.move_off_lanes(id);
                self.crashes[id].lane_mask = 0;
                if self.crashes[id].block >= 0 { self.block_write(self.crashes[id].block as u32, id, 1); }
                self.event("squeeze", self.crashes[id].vehicles[0] as i64, id as f64, 0.0);
            }
            let r = &self.crashes[id];
            if r.response == 0 && self.time >= r.block_end && self.time < r.clear_end {
                self.crashes[id].response = 1; self.move_off_lanes(id);
                if self.crashes[id].block >= 0 { self.block_write(self.crashes[id].block as u32, id, 2); }
                self.event("shoulder", self.crashes[id].vehicles[0] as i64, id as f64, 0.0);
            }
            if self.time >= self.crashes[id].clear_end {
                self.crashes[id].response = 2;
                for s in self.crashes[id].vehicles.clone() {
                    if self.crashes[id].gone.contains(&s) { continue; }
                    self.op(op::FREE, s, 0, 0); self.in_crash.remove(&s); self.wrecks.remove(&s);
                    if self.crashes[id].stranded { self.counters.towed += 1; self.release(s, "towed"); } else { self.counters.cleared += 1; self.release(s, "cleared"); }
                }
                self.block_remove(id);
                self.event("cleared", self.crashes[id].vehicles[0] as i64, id as f64, 0.0);
            } else { keep.push(id); }
        }
        self.active = keep;
    }


    fn update_closed(&mut self) {
        let g = &self.world.graph; let mut closed = Vec::new();
        for &id in &self.active { let r = &self.crashes[id]; if r.response != 0 || r.squeeze { continue; }
            let lanes = g.lanes[r.edge as usize] as u32; if (0..lanes).all(|l| (r.lane_mask >> l) & 1 == 1) { closed.push(r.edge); } }
        closed.sort();
        if self.closed_key.as_ref() == Some(&closed) { return; }
        self.closed_n = closed.len(); self.closed_key = Some(closed.clone()); self.closed_edges = closed;
        let e_n = self.blob.e; let mut bits = vec![0u32; (e_n + 31) / 32];
        for &e in &self.closed_edges { bits[(e >> 5) as usize] |= 1 << (e & 31); }
        self.gpu.write("ctl", bytemuck::cast_slice(&bits), ((self.cap * L::CTL_WORDS + self.j_m() + e_n) * 4) as u64);
    }

    fn route_of(&self, slot: u32) -> &[u32] {
        let (off, len) = (self.slot_route[2 * slot as usize] as usize, self.slot_route[2 * slot as usize + 1] as usize);
        if off + len <= self.route_mirror.len() { &self.route_mirror[off..off + len] } else { &[] }
    }

    fn process_reroutes(&mut self) {
        if self.reroute_queue.is_empty() { return; }
        if self.reroute_queue.len() > 4000 { let (st, t) = (&self.slot_trip, self.time); self.reroute_queue.retain(|r| st[r.slot as usize] == r.trip && (r.missed || t - r.t <= 60.0)); }
        let mut budget = 200;
        while budget > 0 { let Some(r) = self.reroute_queue.pop_front() else { break };
            if self.slot_trip[r.slot as usize] != r.trip || r.trip < 0 { continue; }
            if !r.missed && self.time - r.t > 60.0 { continue; }
            if !r.missed { if let Some(&(lt, ltrip)) = self.last_reroute.get(&r.slot) { if ltrip == r.trip && self.time - lt < if r.slow { 300.0 } else { 120.0 } { continue; } } }
            if !r.missed { self.last_reroute.insert(r.slot, (self.time, r.trip)); }
            budget -= 1;
            let route = self.route_of(r.slot).to_vec(); if route.is_empty() { continue; }
            let g = &self.world.graph;
            let goal = g.to[(route[route.len() - 1] & 0xFFFFF) as usize] as usize; let from = g.to[r.edge as usize] as usize;
            if from == goal { if r.missed { self.apply_reroute(r, Some(Vec::new())); } continue; }
            if self.router.is_none() { self.router = Some(bayenv_world::nav::Router::new(&self.world.graph, &bayenv_world::nav::RouterOpts { weight: 1.25, ..Default::default() })); }
            let router = self.router.as_mut().unwrap();
            let path = if r.missed || (self.closed_edges.is_empty() && self.traffic_penalty.is_empty()) { router.route(from, goal) } else {
                let pen: Vec<(u32, f64)> = self.penalty_order.iter().map(|e| (*e, self.traffic_penalty[e])).collect();
                router.route_with(from, goal, &bayenv_world::nav::RouteQuery { avoid: &self.closed_edges, penalty: &pen })
            };
            self.apply_reroute(r, path);
        }
        if self.last_reroute.len() > 300000 { self.last_reroute.clear(); }
    }
    fn apply_reroute(&mut self, r: Reroute, path: Option<Vec<u32>>) {
        if self.slot_trip[r.slot as usize] != r.trip { return; }
        let closed: HashSet<u32> = if r.missed { HashSet::new() } else { self.closed_edges.iter().cloned().collect() };
        if r.slow { if let Some(p) = &path {
            let g = &self.world.graph; let pen = &self.traffic_penalty;
            let cost = |x: &[u32]| x.iter().fold(0.0, |t, &e| { let e = (e & 0xFFFFF) as usize; t + g.length[e] / g.speed_limit[e] + pen.get(&(e as u32)).copied().unwrap_or(0.0) });
            let cur = self.route_of(r.slot); let rest: &[u32] = if (r.rp as usize + 1) <= cur.len() { &cur[r.rp as usize + 1..] } else { &[] };
            if cost(p) > cost(rest) - 60.0 { return; }
        } }
        let Some(path) = path else { self.counters.reroute_failed += 1; return };
        if (path.is_empty() && !r.missed) || path.iter().any(|x| closed.contains(x)) { self.counters.reroute_failed += 1; return; }
        let g = &self.world.graph; let pre = if r.prefix >= 0 { 1 } else { 0 };
        let mut edges: Vec<i32> = Vec::with_capacity(pre + 1 + path.len());
        if pre == 1 { edges.push(r.prefix as i32); }
        edges.push(r.edge as i32); edges.extend(path.iter().map(|&e| e as i32));
        let total = edges.iter().fold(0.0, |s, &e| s + g.length[e as usize]);
        let packer = self.packer.get_or_insert_with(|| Arc::new(crate::pack::RoutePacker::new(&self.world.graph))).clone();
        let words = packer.pack_route(&self.world.graph, &edges);
        let mut block = Vec::with_capacity(words.len() + 1); block.push(f32bits(total)); block.extend_from_slice(&words);
        let off = self.append_route(&block) + 1;
        self.op(op::SET_ROUTE, r.slot, off as u32, edges.len() as u32);
        self.slot_route[2 * r.slot as usize] = off as u32; self.slot_route[2 * r.slot as usize + 1] = edges.len() as u32;
        self.counters.reroutes += 1;
        self.event("reroute", r.slot as i64, 0.0, 0.0);
    }

    // ------------------------------------------------------------------ state readback
    pub fn read_state(&self) -> SimState {
        SimState { veh: self.read_words("veh"), par: self.read_words("par"), pose: bytemuck::cast_slice(&self.read_words("pose")).to_vec() }
    }
    pub fn stats(&self) -> &[u32; 64] { &self.gpu_stats }

    // ------------------------------------------------------------------ oracle snapshot (JS mock run)
    /// Load buffers + host state from a `rust/tools/oracle/gpu_mock.mjs` dump (world blob, scatter state, routes,
    /// actions) so the native step loop can be compared with the JS one before the native packer lands.
    pub fn load_oracle_snapshot(&mut self, dir: &std::path::Path) -> Result<(), String> {
        let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("gpu.meta.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        let bin = std::fs::read(dir.join("gpu.bin")).map_err(|e| e.to_string())?;
        for b in meta["buffers"].as_array().unwrap() {
            let name = b["name"].as_str().unwrap(); let off = b["offset"].as_i64().unwrap(); let size = b["size"].as_u64().unwrap() as usize;
            if off < 0 || name.starts_with("staging") || name == "params" { continue; }
            if name == "routes" { let words: &[u32] = bytemuck::cast_slice(&bin[off as usize..off as usize + size]); let used = meta["scatter"]["routeWords"].as_u64().unwrap() as usize;
                self.route_words = 0; self.route_mirror.clear(); self.ensure_routes(size / 4); self.append_route(&words[..used]); continue; }
            if !self.gpu.buffers.contains_key(name) { continue; }
            let have = self.gpu.get(name).size() as usize;
            self.gpu.write(name, &bin[off as usize..off as usize + size.min(have)], 0);
        }
        self.time = meta["time"].as_f64().unwrap() - meta["k"].as_f64().unwrap() * self.dt;
        self.step = 0;
        let placed = meta["scatter"]["placed"].as_u64().unwrap() as usize;
        self.free = (placed as u32..self.cap as u32).rev().collect();
        for i in 0..placed { self.slot_trip[i] = 2 * i as i32 + 1; }
        self.new_pose = true;
        Ok(())
    }

    // ------------------------------------------------------------------ benchmark population
    /// `scatter(n, {seed})`: n cars at rest on random routes at once. `pack` packs a route (edge ids) into route
    /// words (`crate::pack::pack_routes`).
    pub fn scatter(&mut self, n: usize, seed: u32, pack: &dyn Fn(&[i32]) -> Vec<u32>) -> ScatterInfo {
        let world = self.world.clone(); let g = &world.graph; let e_n = g.edge_count; let cap = self.cap;
        let n = n.min(cap); let route_edges = 160usize; let pool = n.min(100000);
        let mut rs: u32 = { let r = jsmath::to_u32(seed as f64 * 2654435761.0); if r == 0 { 1 } else { r } };
        let mut rnd = move || { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; rs as f64 / 4294967296.0 };
        let lane_m = |r: u8| { let mut m = 0.0; for e in 0..e_n { if g.rank[e] >= r { m += g.length[e] * g.lanes[e] as f64; } } m };
        let min_rank: u8 = if (n as f64) * 28.0 < 0.35 * lane_m(3) { 3 } else { 0 };
        let spacing = jsmath::max(9.0, jsmath::min(28.0, 0.35 * lane_m(min_rank) / n as f64));
        let cand: Vec<u32> = (0..e_n as u32).filter(|&e| g.rank[e as usize] >= min_rank && g.out_degree(g.to[e as usize] as usize) > 0).collect();
        let mut cum = vec![0f64; cand.len()]; let mut acc = 0.0;
        for (k, &e) in cand.iter().enumerate() { acc += g.length[e as usize] * g.lanes[e as usize] as f64; cum[k] = acc; }
        let mut pool_off = vec![0u32; pool + 1]; let mut pool_edges: Vec<Vec<i32>> = Vec::with_capacity(pool); let mut all: Vec<u32> = Vec::new();
        for r in 0..pool {
            let x = rnd() * acc; let (mut lo, mut hi) = (0usize, cand.len() - 1);
            while lo < hi { let m = (lo + hi) >> 1; if cum[m] < x { lo = m + 1; } else { hi = m; } }
            let mut route = vec![cand[lo] as i32]; let mut cur = cand[lo] as usize;
            for _ in 1..route_edges {
                let (mut best, mut best_score) = (-1i64, f64::NEG_INFINITY);
                for &eo in g.out_of(g.to[cur] as usize) {
                    if eo as i32 == g.reverse[cur] { continue; }
                    let mut dh = g.heading_start[eo as usize] - g.heading_end[cur];
                    while dh > std::f64::consts::PI { dh -= 2.0 * std::f64::consts::PI; } while dh < -std::f64::consts::PI { dh += 2.0 * std::f64::consts::PI; }
                    let score = -dh.abs() * 1.2 + g.rank[eo as usize] as f64 * 0.15 + rnd() * 1.5;
                    if score > best_score { best_score = score; best = eo as i64; }
                }
                if best < 0 { break; }
                route.push(best as i32); cur = best as usize;
            }
            let words = pack(&route);
            pool_off[r] = all.len() as u32; all.extend_from_slice(&words); pool_edges.push(route);
        }
        pool_off[pool] = all.len() as u32;
        self.route_words = 0; self.route_mirror.clear();
        self.ensure_routes(all.len() + self.opts.route_slack);
        self.append_route(&all);
        let totals: Vec<f64> = pool_edges.iter().map(|r| r.iter().fold(0.0, |s, &e| s + g.length[e as usize])).collect();
        let mut type_cum = [0f64; 4]; { let mut c = 0.0; for (i, w) in VEHICLE_TYPE_WEIGHTS.iter().enumerate() { c += w; type_cum[i] = c; } }
        let mut used: HashMap<u64, Vec<f64>> = HashMap::new();
        let mut vehw = vec![0u32; cap * L::VEH_WORDS]; let mut parw = vec![0u32; cap * L::PAR_WORDS]; let mut pose = vec![0f32; cap * 4];
        let (mut placed, mut tries) = (0usize, 0usize);
        let policy = self.opts.features.policy;
        while placed < n && tries < n * 30 {
            tries += 1;
            let r = (rnd() * pool as f64).floor() as usize; let route = &pool_edges[r];
            if route.len() < 2 { continue; }
            let k = (rnd() * (route.len() - 1) as f64).floor() as usize;
            let e0 = route[k] as usize;
            if g.length[e0] < spacing { continue; }
            let lane = (rnd() * g.lanes[e0] as f64).floor() as u32;
            let s = spacing / 2.0 + rnd() * (g.length[e0] - spacing);
            let key = e0 as u64 * 8 + lane as u64;
            let list = used.entry(key).or_default();
            if list.iter().any(|&x| (x - s).abs() < spacing) { continue; }
            list.push(s);
            let i = placed; placed += 1;
            let rr = rnd(); let ty = type_cum.iter().position(|&c| rr < c).map(|v| v as i64).unwrap_or(-1);
            let aggression = rnd(); let patience = rnd();
            let reaction = jsmath::min(1.8, jsmath::max(0.6, 0.9 * jsmath::exp(0.27 * { let mut u = 0.0; while u == 0.0 { u = rnd(); } (-2.0 * jsmath::log(u)).sqrt() * jsmath::cos(2.0 * std::f64::consts::PI * rnd()) })));
            let headway = 1.0 + rnd() * 1.2;
            let before: f64 = route[..k].iter().fold(0.0, |s, &e| s + g.length[e as usize]);
            let tyu = if ty < 0 { 0usize } else { ty as usize };
            self.par_words(&mut parw[i * L::PAR_WORDS..(i + 1) * L::PAR_WORDS], ty, aggression, patience, reaction, headway, defines::default_traits::MAX_ACCEL, defines::default_traits::MAX_DECEL,
                pool_off[r], route.len() as u32, 0.0, totals[r], 2 * i as u32 + 1);
            let lat = (lane as f64 + 0.5) * LANE_W; let tank = if ty < 0 { f64::NAN } else { VEHICLE_PARAMS[tyu].tank_l };
            let o = i * L::VEH_WORDS; let v = &mut vehw[o..o + L::VEH_WORDS];
            let fset = |v: &mut [u32], k: usize, x: f64| v[k] = f32bits(x);
            v[veh::EDGE] = e0 as u32; fset(v, veh::S, s); fset(v, veh::LAT, lat); v[veh::STATE] = state::DRIVING;
            v[veh::ROUTE_POS] = k as u32; v[veh::FLAGS] = (if policy { flag::POLICY } else { 0 }) | flag::NEW; v[veh::CMD_STEP] = self.step as u32;
            fset(v, veh::STOPPED_SINCE, -1.0); v[veh::LANES] = L::pack_lanes(lane, lane, lane, -1); fset(v, veh::LAST_DECISION, -1e9);
            fset(v, veh::FUEL, tank * (0.1 + 0.9 * rnd())); fset(v, veh::TANK, tank); fset(v, veh::SPAWN_TIME, self.time); fset(v, veh::ROUTE_DIST, before + s);
            v[veh::LEADER] = -1i32 as u32; fset(v, veh::GAP, 1e9); v[veh::STOP_POS] = -1i32 as u32; fset(v, veh::ROUTE_TOTAL, totals[r]); fset(v, veh::LAST_FUEL_CHECK, self.time); fset(v, veh::LANE_END_DIST, 1e9);
            let ps = self.pose_at(e0, s, lat);
            for j in 0..4 { pose[i * 4 + j] = ps[j] as f32; }
            fset(v, veh::PX, ps[0]); fset(v, veh::PZ, ps[2]); fset(v, veh::PYAW, ps[3]);
            self.slot_trip[i] = 2 * i as i32 + 1;
        }
        for j in placed..cap { pose[j * 4 + 1] = -1000.0; }
        self.gpu.write("veh", bytemuck::cast_slice(&vehw), 0);
        self.gpu.write("par", bytemuck::cast_slice(&parw), 0);
        self.gpu.write("pose", bytemuck::cast_slice(&pose), 0);
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        enc.clear_buffer(&self.gpu.get("ring"), 0, None);
        self.gpu.queue.submit([enc.finish()]);
        let mut iow = vec![0u32; io::EVDATA]; iow[0] = self.step as u32; self.gpu.write("io", bytemuck::cast_slice(&iow), 0);
        self.free = (placed as u32..cap as u32).rev().collect();
        self.new_pose = true;
        ScatterInfo { placed, tries, route_words: all.len(), min_rank, spacing: jsmath::round(spacing * 10.0) / 10.0, pool }
    }

    /// `_par`: VPar words (vehicle.js alloc semantics). `ty` < 0 = JS findIndex miss (VEHICLE_TYPES[-1] -> type 0 values).
    #[allow(clippy::too_many_arguments)]
    fn par_words(&self, out: &mut [u32], ty: i64, aggression: f64, patience: f64, rt: f64, headway: f64, max_accel: f64, max_decel: f64, route_off: u32, route_len: u32, work_start: f64, route_length: f64, trip: u32) {
        let t = &VEHICLE_PARAMS[if ty < 0 { 0 } else { ty as usize }];
        out[0] = route_off; out[1] = route_len; out[2] = ty as i32 as u32; out[3] = jsmath::min((L::CTRL_RING - 1) as f64, jsmath::max(0.0, jsmath::round(rt / self.dt))) as u32;
        let f = |x: f64| f32bits(x);
        out[4] = f(t.length); out[5] = f(t.width); out[6] = f(t.vmax);
        let (ma, md) = defines::driver_limits(t, max_accel, max_decel);
        out[7] = f(ma); out[8] = f(md); out[9] = f(headway); out[10] = f(aggression); out[11] = f(patience);
        out[12] = f(rt); out[13] = f(work_start); out[14] = f(route_length); out[15] = trip;
        out[16] = f(t.l); out[17] = f(t.max_steer); out[18] = f(t.max_steer_rate); out[19] = f(t.vmax);
    }
}

/// A pending `run_steps_async` batch is finished before the buffers go (a map callback of a dropped buffer errors).
impl Drop for GpuSim { fn drop(&mut self) { self.sync(); } }

/// An in-flight `submit_steps` (byte sizes of the mapped ranges).
#[derive(Clone, Copy, Debug)]
pub struct Pending { fast: usize, ob4: usize, vb4: usize }

#[derive(Clone, Debug)]
pub struct ScatterInfo { pub placed: usize, pub tries: usize, pub route_words: usize, pub min_rank: u8, pub spacing: f64, pub pool: usize }

/// Full vehicle state (cap x VEH_WORDS words, cap x PAR_WORDS, cap x 4 pose floats).
pub struct SimState { pub veh: Vec<u32>, pub par: Vec<u32>, pub pose: Vec<f32> }
impl SimState {
    pub fn u(&self, slot: usize, k: usize) -> u32 { self.veh[slot * L::VEH_WORDS + k] }
    pub fn f(&self, slot: usize, k: usize) -> f32 { f32::from_bits(self.veh[slot * L::VEH_WORDS + k]) }
}
