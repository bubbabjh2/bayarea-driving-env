//! src/sim/rng.js: mulberry32 + helpers.
use crate::jm;

#[derive(Clone, Debug)]
pub struct Rng { a: u32 }

impl Rng {
    pub fn new(seed: f64) -> Rng { let s = jm::to_u32(seed); Rng { a: if s == 0 { 0x9e3779b9 } else { s } } }
    pub fn from_u32(seed: u32) -> Rng { Rng { a: if seed == 0 { 0x9e3779b9 } else { seed } } }
    #[inline]
    pub fn random(&mut self) -> f64 {
        self.a = self.a.wrapping_add(0x6d2b79f5);
        let a = self.a;
        let mut t = (a ^ (a >> 15)).wrapping_mul(1 | a);
        t = (t.wrapping_add((t ^ (t >> 7)).wrapping_mul(61 | t))) ^ t;
        (t ^ (t >> 14)) as f64 / 4294967296.0
    }
    #[inline] pub fn int(&mut self, n: usize) -> usize { jm::floor(self.random() * n as f64) as usize }
    #[inline] pub fn range(&mut self, lo: f64, hi: f64) -> f64 { lo + (hi - lo) * self.random() }
    pub fn gaussian(&mut self) -> f64 {
        let mut u = 0.0; let mut v = 0.0;
        while u == 0.0 { u = self.random(); }
        while v == 0.0 { v = self.random(); }
        jm::sqrt(-2.0 * jm::log(u)) * jm::cos(2.0 * jm::PI * v)
    }
    pub fn weighted_index(&mut self, cum: &[f64]) -> usize {
        let total = cum[cum.len() - 1];
        let r = self.random() * total;
        let (mut lo, mut hi) = (0usize, cum.len() - 1);
        while lo < hi { let mid = (lo + hi) >> 1; if cum[mid] <= r { lo = mid + 1; } else { hi = mid; } }
        lo
    }
    pub fn exp(&mut self, m: f64) -> f64 { -jm::log(1.0 - self.random()) * m }
}

/// rng.js hashSeed (FNV-1a over UTF-16 code units).
pub fn hash_seed(s: &str) -> u32 {
    let mut h: u32 = 2166136261;
    for c in s.encode_utf16() { h ^= c as u32; h = h.wrapping_mul(16777619); }
    h
}
