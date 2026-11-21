//! Population (src/sim/population.js assemble() result): per-commuter arrays + packed routes + 64-float deltas.
use crate::pack::Pack;
use crate::sim::Trip;

#[derive(Clone, Debug, Default)]
pub struct Population {
    pub n: usize,
    pub vehicle_type: Vec<u8>, pub home: Vec<i32>, pub work: Vec<i32>,
    pub depart: Vec<f64>, pub return_depart: Vec<f64>, pub work_start: Vec<f64>, pub fuel_fill: Vec<f64>,
    pub route_length: Vec<f64>, pub expected_travel: Vec<f64>,
    /// route of trip k = 2 * commuter + leg: route_flat[route_offset[k] .. route_offset[k + 1]]
    pub route_flat: Vec<i32>, pub route_offset: Vec<u32>,
    pub deltas: Vec<f32>,
}

impl Population {
    pub fn load(path: &str) -> Population { Population::from_pack(&Pack::read(path).expect("population pack")) }
    pub fn from_pack(p: &Pack) -> Population {
        let n = p.meta["n"].as_u64().unwrap() as usize;
        Population { n, vehicle_type: p.u8s("vehicleType"), home: p.i32s("home"), work: p.i32s("work"), depart: p.f64s("depart"), return_depart: p.f64s("returnDepart"),
            work_start: p.f64s("workStart"), fuel_fill: p.f64s("fuelFill"), route_length: p.f64s("L"), expected_travel: p.f64s("expectedTravel"),
            route_flat: p.i32s("routeFlat"), route_offset: p.u32s("routeOffset"), deltas: p.f32s("deltas") }
    }
    pub fn route(&self, c: usize, leg: usize) -> &[i32] { let k = 2 * c + leg; &self.route_flat[self.route_offset[k] as usize..self.route_offset[k + 1] as usize] }
    /// tripsOf(commuters): outbound trips (+ return trips with a finite returnDepart and a route), sorted by depart, commuter, leg.
    pub fn trips(&self) -> Vec<Trip> {
        let mut t = Vec::with_capacity(self.n);
        for c in 0..self.n {
            t.push(Trip { commuter: c, from: self.home[c], to: self.work[c], depart: self.depart[c], leg: 0 });
            if self.return_depart[c].is_finite() && !self.route(c, 1).is_empty() { t.push(Trip { commuter: c, from: self.work[c], to: self.home[c], depart: self.return_depart[c], leg: 1 }); }
        }
        t.sort_by(|a, b| {
            let d = a.depart - b.depart;
            let v = if d != 0.0 && !d.is_nan() { d } else { let d = a.commuter as f64 - b.commuter as f64; if d != 0.0 { d } else { a.leg as f64 - b.leg as f64 } };
            crate::jm::cmp_of(v)
        });
        t
    }
}
