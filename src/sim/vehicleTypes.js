// Vehicle classes. All physical values live in src/sim/physics/params.js (VEHICLE_PARAMS);
// this module adds the legacy field names other code reads (maxSpeed, tankL) and the fleet mix.
import { VEHICLE_PARAMS } from './physics/params.js';

// legacy fields (layout v3 consumers, e.g. src/gpu): comfort accel / decel and the old speed-formula fuel model
const LEGACY = [
  { maxAccel: 2.8, maxDecel: 3.5, fuelBase: 0.050, fuelAero: 5.3e-5, idleLps: 0.00025 },
  { maxAccel: 2.4, maxDecel: 3.3, fuelBase: 0.065, fuelAero: 7.0e-5, idleLps: 0.00030 },
  { maxAccel: 1.0, maxDecel: 2.0, fuelBase: 0.200, fuelAero: 1.6e-4, idleLps: 0.00080 },
  { maxAccel: 1.2, maxDecel: 2.2, fuelBase: 0.220, fuelAero: 1.7e-4, idleLps: 0.00080 },
];
export const VEHICLE_TYPES = VEHICLE_PARAMS.map((p, k) => ({
  ...LEGACY[k], ...p, maxSpeed: p.vmax, mass: p.m, wheelbase: p.L, color: k,
  emergencyDecel: Math.min(p.brakeDecel, p.mu * 9.81),   // the hardest braking the tyres allow
}));
export const VEHICLE_TYPE_WEIGHTS = [0.62, 0.30, 0.06, 0.02];
export const TYPE_SEDAN = 0, TYPE_SUV = 1, TYPE_TRUCK = 2, TYPE_BUS = 3;
