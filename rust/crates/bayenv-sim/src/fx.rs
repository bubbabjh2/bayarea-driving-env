//! Fast deterministic hashing (FxHash, as rustc uses) for the sim's internal maps. No sim result depends on map
//! iteration order (JS Maps whose order matters are Vecs), so the hasher only changes speed.
use std::hash::{BuildHasherDefault, Hasher};

#[derive(Default, Clone, Copy)]
pub struct FxHasher { h: u64 }
const K: u64 = 0xf1357aea2e62a9c5;
impl Hasher for FxHasher {
    #[inline] fn write(&mut self, bytes: &[u8]) { for &b in bytes { self.h = (self.h.rotate_left(5) ^ b as u64).wrapping_mul(K); } }
    #[inline] fn write_u8(&mut self, i: u8) { self.write_u64(i as u64) }
    #[inline] fn write_u32(&mut self, i: u32) { self.write_u64(i as u64) }
    #[inline] fn write_i32(&mut self, i: i32) { self.write_u64(i as u32 as u64) }
    #[inline] fn write_u64(&mut self, i: u64) { self.h = (self.h.rotate_left(5) ^ i).wrapping_mul(K); }
    #[inline] fn write_i64(&mut self, i: i64) { self.write_u64(i as u64) }
    #[inline] fn write_usize(&mut self, i: usize) { self.write_u64(i as u64) }
    #[inline] fn finish(&self) -> u64 { self.h.rotate_left(26) }
}
pub type FxBuild = BuildHasherDefault<FxHasher>;
pub type HashMap<K, V> = std::collections::HashMap<K, V, FxBuild>;
pub type HashSet<K> = std::collections::HashSet<K, FxBuild>;

/// JS `Map(key -> slot list)` rebuilt every step: keys persist (Map.size counts them) until clear(), lists are
/// emptied each step. Only the lists filled since the last reset are touched (the JS walks every value).
pub struct ListMap<K: std::hash::Hash + Eq + Copy> { map: HashMap<K, Vec<usize>>, touched: Vec<K> }
impl<K: std::hash::Hash + Eq + Copy> Default for ListMap<K> { fn default() -> Self { ListMap { map: HashMap::default(), touched: Vec::new() } } }
impl<K: std::hash::Hash + Eq + Copy> ListMap<K> {
    #[inline] pub fn len(&self) -> usize { self.map.len() }
    #[inline] pub fn is_empty(&self) -> bool { self.map.is_empty() }
    /// no list has an entry (keys may persist with empty lists)
    pub fn is_empty_lists(&self) -> bool { self.touched.is_empty() }
    pub fn reset_lists(&mut self) { for k in self.touched.drain(..) { if let Some(l) = self.map.get_mut(&k) { l.clear(); } } }
    pub fn clear(&mut self) { self.map.clear(); self.touched.clear(); }
    #[inline] pub fn push(&mut self, k: K, v: usize) {
        let l = self.map.entry(k).or_default();
        if l.is_empty() { self.touched.push(k); }
        l.push(v);
    }
    #[inline] pub fn get(&self, k: &K) -> Option<&Vec<usize>> { self.map.get(k) }
}
