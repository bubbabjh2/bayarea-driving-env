//! Versioned, checksummed binary container for pre-baked world data (mmap'd, zero-copy).
//!
//! Layout (little-endian): magic `b"BAYBAKE\0"`, u32 format version, u32 section count, u64 source key
//! (xxh3 of the inputs it was derived from), then `count` x 64-byte entries
//! `{ name [u8; 40], ty u32, pad u32, offset u64, len u64 (elements) }`, then u64 xxh3 of all section bytes,
//! then the sections, each 64-byte aligned.
use crate::blob::{Blob, Slice};
use std::{collections::BTreeMap, io, path::Path};

pub const MAGIC: &[u8; 8] = b"BAYBAKE\0";
pub const VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Ty { U8 = 1, U16 = 2, U32 = 3, I32 = 4, F32 = 5, F64 = 6, I8 = 7, U64 = 8 }
impl Ty { fn size(self) -> usize { match self { Ty::U8 | Ty::I8 => 1, Ty::U16 => 2, Ty::U32 | Ty::I32 | Ty::F32 => 4, Ty::F64 | Ty::U64 => 8 } } }

pub trait Elem: bytemuck::Pod { const TY: Ty; }
impl Elem for u8 { const TY: Ty = Ty::U8; } impl Elem for i8 { const TY: Ty = Ty::I8; } impl Elem for u16 { const TY: Ty = Ty::U16; }
impl Elem for u32 { const TY: Ty = Ty::U32; } impl Elem for i32 { const TY: Ty = Ty::I32; } impl Elem for f32 { const TY: Ty = Ty::F32; }
impl Elem for f64 { const TY: Ty = Ty::F64; } impl Elem for u64 { const TY: Ty = Ty::U64; }

#[derive(Default)]
pub struct BakeWriter { secs: Vec<(String, Ty, Vec<u8>)> }
impl BakeWriter {
    pub fn put<T: Elem>(&mut self, name: &str, v: &[T]) -> &mut Self {
        assert!(name.len() < 40); self.secs.push((name.to_string(), T::TY, bytemuck::cast_slice(v).to_vec())); self
    }
    pub fn write(&self, path: &Path, key: u64) -> io::Result<()> {
        let head = 8 + 4 + 4 + 8 + 64 * self.secs.len() + 8;
        let mut off = (head + 63) / 64 * 64; let mut table = Vec::new(); let mut body: Vec<u8> = Vec::new();
        let base = off;
        for (name, ty, b) in &self.secs {
            let mut e = [0u8; 64]; e[..name.len()].copy_from_slice(name.as_bytes());
            e[40..44].copy_from_slice(&(*ty as u32).to_le_bytes());
            e[48..56].copy_from_slice(&(off as u64).to_le_bytes()); e[56..64].copy_from_slice(&((b.len() / ty.size()) as u64).to_le_bytes());
            table.extend_from_slice(&e);
            body.resize(off - base, 0); body.extend_from_slice(b); off = (base + body.len() + 63) / 64 * 64;
        }
        let mut out = Vec::with_capacity(base + body.len());
        out.extend_from_slice(MAGIC); out.extend_from_slice(&VERSION.to_le_bytes()); out.extend_from_slice(&(self.secs.len() as u32).to_le_bytes());
        out.extend_from_slice(&key.to_le_bytes()); out.extend_from_slice(&table);
        out.extend_from_slice(&xxhash_rust::xxh3::xxh3_64(&body).to_le_bytes());
        out.resize(base, 0); out.extend_from_slice(&body);
        let tmp = path.with_extension("tmp");
        if let Some(p) = path.parent() { std::fs::create_dir_all(p)?; }
        std::fs::write(&tmp, &out)?; std::fs::rename(&tmp, path)
    }
}

pub struct Bake { pub blob: Blob, pub key: u64, secs: BTreeMap<String, (Ty, usize, usize)> }
impl Bake {
    /// Open and validate (magic, version, bounds; `verify` also checks the body checksum, ~0.1 s/GB).
    pub fn open(path: &Path, verify: bool) -> io::Result<Bake> {
        let bad = |m: &str| io::Error::new(io::ErrorKind::InvalidData, format!("{}: {m}", path.display()));
        let blob = Blob::open(path)?; let b = blob.bytes();
        if b.len() < 24 || &b[..8] != MAGIC { return Err(bad("not a bake file")); }
        let rd32 = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap()); let rd64 = |o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
        if rd32(8) != VERSION { return Err(bad("bake version mismatch")); }
        let n = rd32(12) as usize; let key = rd64(16); let mut secs = BTreeMap::new();
        let head = 24 + 64 * n + 8; if b.len() < head { return Err(bad("truncated")); }
        let base = (head + 63) / 64 * 64;
        for k in 0..n {
            let e = &b[24 + 64 * k..24 + 64 * (k + 1)];
            let name = String::from_utf8_lossy(&e[..40]).trim_end_matches('\0').to_string();
            let ty = match u32::from_le_bytes(e[40..44].try_into().unwrap()) { 1 => Ty::U8, 2 => Ty::U16, 3 => Ty::U32, 4 => Ty::I32, 5 => Ty::F32, 6 => Ty::F64, 7 => Ty::I8, 8 => Ty::U64, _ => return Err(bad("type")) };
            let off = u64::from_le_bytes(e[48..56].try_into().unwrap()) as usize; let len = u64::from_le_bytes(e[56..64].try_into().unwrap()) as usize;
            if off + len * ty.size() > b.len() { return Err(bad("section out of range")); }
            secs.insert(name, (ty, off, len));
        }
        if verify {
            let end = secs.values().map(|&(t, o, l)| o + l * t.size()).max().unwrap_or(base).max(base);
            if xxhash_rust::xxh3::xxh3_64(&b[base..end]) != rd64(24 + 64 * n) { return Err(bad("checksum mismatch")); }
        }
        Ok(Bake { blob, key, secs })
    }
    pub fn get<T: Elem>(&self, name: &str) -> io::Result<Slice<T>> {
        let &(ty, off, len) = self.secs.get(name).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, format!("bake: no section {name}")))?;
        if ty != T::TY { return Err(io::Error::new(io::ErrorKind::InvalidData, format!("bake: section {name} type"))); }
        Ok(Slice::Mapped(self.blob.view::<T>(off, len)))
    }
    pub fn has(&self, name: &str) -> bool { self.secs.contains_key(name) }
}

/// xxh3 over the given files' bytes (bake source key).
pub fn files_key(paths: &[&Path]) -> io::Result<u64> {
    let mut h = xxhash_rust::xxh3::Xxh3::new();
    for p in paths { match Blob::open(p) { Ok(b) => h.update(b.bytes()), Err(_) => h.update(b"<missing>") } }
    h.update(&VERSION.to_le_bytes());
    Ok(h.digest())
}
