// bayenv-gpu model patch of the junction-yield stage (src/gpu/kernels/jyield.wgsl), with GpuOptions::junction_obstacle
// (the host prepends `const MF_JUNC_OBST: bool`): the junction-conflict obstacle of bayenv-sim's obs v2 encoder
// (obs_v2.rs: adjust.js `junctionObs` as a stopped obstacle in the own-ahead slot, inputs 8 / 9). The CPU puts
// max(0, conflict distance - 2) into the own-ahead gap when the conflicting body is stopped or reaches the conflict
// point within the car's own time to it + 2 s (my_t = (distance + length) / max(v, 1)); the JS GPU never did, so a car
// at a busy box saw a free lane ahead where the CPU saw an obstacle. junctionInputs stores that gap (raw metres, 1e9
// none) and the step it was computed in into `jobst` (2 words per slot, allocated with the flag; every ctl word is
// taken); kernels/model_sense.wgsl puts it into input 8 / 9 after the neighbours, as the CPU does.
//#orig junction-obstacle-bind
@group(1) @binding(3) var<storage, read_write> inputs: array<f32>;
//#end
//#patch junction-obstacle-bind
@group(1) @binding(3) var<storage, read_write> inputs: array<f32>;
@group(1) @binding(4) var<storage, read_write> jobst: array<u32>;
//#end
//#orig junction-obstacle
  inputs[b + 4u] = select(1.0, clamp(mouth / 60.0, 0.0, 1.0), mouth < 1e8);
}
//#end
//#patch junction-obstacle
  inputs[b + 4u] = select(1.0, clamp(mouth / 60.0, 0.0, 1.0), mouth < 1e8);
  if (MF_JUNC_OBST) {
    var jg = 1e9;
    if (dist < 1e8) { let myT = (dist + pi.length) / max(v.speed, 1.0); if (stopped || ttc < myT + 2.0) { jg = max(0.0, dist - 2.0); } }
    jobst[2u * i] = bitcast<u32>(jg); jobst[2u * i + 1u] = simStep();
  }
}
//#end
