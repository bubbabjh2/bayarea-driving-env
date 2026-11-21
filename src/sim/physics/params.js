// Per-type physical vehicle parameters (real-world values; sources: typical published
// specs for a mid-size sedan (Camry/Accord class), a mid-size SUV, a loaded box truck /
// light semi, and a 40-ft transit bus). Index = typeIdx used everywhere in the sim.
//
// geometry   length, width (hitbox), wheelbase L, a / b = CG to front / rear axle, hcg CG height (m)
// mass       m (kg), Iz yaw inertia (kg m^2)
// resistance CdA (m^2), Crr
// powertrain Pmax (W, at the wheels after the driveline; power-limited CVT-equivalent),
//            F0 (N, max tractive force at low speed = peak torque x first gear / wheel radius),
//            drive 'front' | 'rear' | 'all', vmax (m/s, governor)
// brakes     brakeDecel (m/s^2 demanded at brake = 1, before the tyre limit), brakeFront (share)
// tyres      Cf, Cr cornering stiffness per axle (N/rad), mu (peak road friction, dry asphalt)
// steering   maxSteer (rad at the wheels), maxSteerRate (rad/s at the wheels, legacy 'rate' action mode)
//            'position' action mode (layout v9, default): the command is a target wheel position
//            steer in [-1,1] x maxSteer. The arms/hands follow it through a first-order lag steerLag (s),
//            then the road wheels move at most steerSlew (rad/s). Numbers: full lock -> centre in ~0.75 s
//            for cars (0.60 rad / 0.80 rad/s), ~0.95 s for heavy vehicles; full lock -> opposite lock ~1.5 s
//            (cars). Through a ~16:1 car steering ratio, 0.80 rad/s at the wheels is ~730 deg/s at the
//            hand-wheel (fast but normal human hand-over-hand steering 500-800 deg/s); trucks/buses ~20:1:
//            0.53 rad/s -> ~600 deg/s. Lag ~0.1 s (neuromuscular / arm), 0.12 s heavy vehicles.
// fuel       fuel 'gasoline' | 'diesel', bsfcMin (g/kWh at best load), idleLph (L/h), accessoryW (W), tankL

export const VEHICLE_PARAMS = [
  { name: 'sedan', length: 4.5, width: 1.85, height: 1.45, L: 2.75, a: 1.10, b: 1.65, hcg: 0.55, m: 1500, Iz: 2400,
    CdA: 0.66, Crr: 0.011, Pmax: 100e3, F0: 5600, drive: 'front', vmax: 55,
    brakeDecel: 11.0, brakeFront: 0.68, Cf: 90e3, Cr: 100e3, mu: 1.0, maxSteer: 0.60, maxSteerRate: 0.70, steerSlew: 0.80, steerLag: 0.10,
    fuel: 'gasoline', bsfcMin: 265, idleLph: 0.8, accessoryW: 1500, tankL: 50 },
  { name: 'suv', length: 4.9, width: 1.95, height: 1.75, L: 2.90, a: 1.30, b: 1.60, hcg: 0.70, m: 2100, Iz: 4100,
    CdA: 0.95, Crr: 0.012, Pmax: 150e3, F0: 8500, drive: 'all', vmax: 50,
    brakeDecel: 10.5, brakeFront: 0.65, Cf: 115e3, Cr: 125e3, mu: 1.0, maxSteer: 0.58, maxSteerRate: 0.65, steerSlew: 0.73, steerLag: 0.10,
    fuel: 'gasoline', bsfcMin: 250, idleLph: 1.1, accessoryW: 1100, tankL: 70 },
  { name: 'truck', length: 9.0, width: 2.5, height: 3.6, L: 5.5, a: 2.4, b: 3.1, hcg: 1.25, m: 15000, Iz: 1.1e5,
    CdA: 6.0, Crr: 0.0065, Pmax: 260e3, F0: 55e3, drive: 'rear', vmax: 31,
    brakeDecel: 6.5, brakeFront: 0.45, Cf: 2.6e5, Cr: 5.2e5, mu: 0.8, maxSteer: 0.50, maxSteerRate: 0.45, steerSlew: 0.53, steerLag: 0.12,
    fuel: 'diesel', bsfcMin: 200, idleLph: 2.5, accessoryW: 3000, tankL: 300 },
  { name: 'bus', length: 12.0, width: 2.55, height: 3.2, L: 6.0, a: 3.6, b: 2.4, hcg: 1.15, m: 13000, Iz: 1.6e5,
    CdA: 7.0, Crr: 0.008, Pmax: 200e3, F0: 48e3, drive: 'rear', vmax: 30,
    brakeDecel: 6.5, brakeFront: 0.45, Cf: 2.4e5, Cr: 4.4e5, mu: 0.8, maxSteer: 0.50, maxSteerRate: 0.45, steerSlew: 0.53, steerLag: 0.12,
    fuel: 'diesel', bsfcMin: 205, idleLph: 2.2, accessoryW: 3000, tankL: 250 },
];
export const TYPE_INDEX = { sedan: 0, suv: 1, truck: 2, bus: 3 };
export const RHO_AIR = 1.225, G = 9.81;
export const FUEL_DENSITY = { gasoline: 745, diesel: 832 }; // g/L
