//! Memory-mapped binary packs (`*.meta.json` + `*.bin`, schema `data/roads.schema.md`) as typed zero-copy views.
use memmap2::Mmap;
use std::{fs::File, io, marker::PhantomData, ops::Deref, path::Path, sync::Arc};

/// A read-only memory-mapped file shared by its views.
#[derive(Clone)]
pub struct Blob(pub Arc<Mmap>);

impl Blob {
    pub fn open(path: &Path) -> io::Result<Blob> {
        let f = File::open(path).map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", path.display())))?;
        // SAFETY: the data files are treated as immutable while the env runs.
        let m = unsafe { Mmap::map(&f)? };
        Ok(Blob(Arc::new(m)))
    }
    pub fn bytes(&self) -> &[u8] { &self.0[..] }
    pub fn view<T: bytemuck::Pod>(&self, byte_offset: usize, len: usize) -> View<T> {
        let sz = std::mem::size_of::<T>();
        assert!(byte_offset % std::mem::align_of::<T>() == 0, "misaligned section");
        assert!(byte_offset + len * sz <= self.0.len(), "section out of range");
        View { blob: self.clone(), off: byte_offset, len, _t: PhantomData }
    }
}

/// Typed slice into a [`Blob`] (derefs to `&[T]`), or an owned vector.
pub enum Slice<T: bytemuck::Pod> { Mapped(View<T>), Owned(Vec<T>) }

pub struct View<T> { blob: Blob, off: usize, len: usize, _t: PhantomData<T> }
impl<T> Clone for View<T> { fn clone(&self) -> Self { View { blob: self.blob.clone(), off: self.off, len: self.len, _t: PhantomData } } }
impl<T: bytemuck::Pod> Deref for View<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        bytemuck::cast_slice(&self.blob.0[self.off..self.off + self.len * std::mem::size_of::<T>()])
    }
}
impl<T: bytemuck::Pod> Deref for Slice<T> {
    type Target = [T];
    fn deref(&self) -> &[T] { match self { Slice::Mapped(v) => v, Slice::Owned(v) => v } }
}
impl<T: bytemuck::Pod> Clone for Slice<T> { fn clone(&self) -> Self { match self { Slice::Mapped(v) => Slice::Mapped(v.clone()), Slice::Owned(v) => Slice::Owned(v.clone()) } } }
impl<T: bytemuck::Pod> Default for Slice<T> { fn default() -> Self { Slice::Owned(Vec::new()) } }
impl<T: bytemuck::Pod> From<Vec<T>> for Slice<T> { fn from(v: Vec<T>) -> Self { Slice::Owned(v) } }
impl<T: bytemuck::Pod + std::fmt::Debug> std::fmt::Debug for Slice<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "Slice(len={})", self.len()) }
}

/// One `sections` entry of a meta.json.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Section { pub byte_offset: usize, pub length: usize, #[serde(rename = "type")] pub ty: String }

/// A `*.meta.json` (+ its mapped `.bin`).
pub struct Pack { pub meta: serde_json::Value, pub blob: Blob, pub sections: std::collections::BTreeMap<String, Section> }

impl Pack {
    pub fn open(dir: &Path, meta_file: &str) -> io::Result<Pack> {
        let mp = dir.join(meta_file);
        let txt = std::fs::read_to_string(&mp).map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", mp.display())))?;
        let meta: serde_json::Value = serde_json::from_str(&txt).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let file = meta["file"].as_str().ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "meta.file"))?;
        let blob = Blob::open(&dir.join(file))?;
        let sections = match meta.get("sections") {
            Some(s) => serde_json::from_value(s.clone()).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?,
            None => Default::default(),
        };
        Ok(Pack { meta, blob, sections })
    }
    pub fn get<T: bytemuck::Pod>(&self, name: &str) -> Option<Slice<T>> {
        let s = self.sections.get(name)?;
        Some(Slice::Mapped(self.blob.view::<T>(s.byte_offset, s.length)))
    }
    pub fn req<T: bytemuck::Pod>(&self, name: &str) -> io::Result<Slice<T>> {
        self.get(name).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, format!("missing section {name}")))
    }
}
