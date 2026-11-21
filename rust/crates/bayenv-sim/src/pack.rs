//! BAYPACK1 container (tools/rust/pack.mjs): "BAYPACK1" | u32 LE header length | JSON header | pad to 8 | sections.
use serde_json::Value;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DType { F64, F32, I32, U32, I16, U16, I8, U8, I64 }

impl DType {
    fn parse(s: &str) -> DType {
        match s { "f64" => DType::F64, "f32" => DType::F32, "i32" => DType::I32, "u32" => DType::U32, "i16" => DType::I16,
            "u16" => DType::U16, "i8" => DType::I8, "u8" => DType::U8, "i64" => DType::I64, _ => panic!("dtype {s}") }
    }
    pub fn size(self) -> usize { match self { DType::F64 | DType::I64 => 8, DType::F32 | DType::I32 | DType::U32 => 4, DType::I16 | DType::U16 => 2, DType::I8 | DType::U8 => 1 } }
}

pub struct Section { pub dtype: DType, pub len: usize, pub bytes: Vec<u8> }

pub struct Pack { pub meta: Value, pub sections: HashMap<String, Section>, pub order: Vec<String> }

macro_rules! getter {
    ($name:ident, $t:ty, $($dt:ident => $src:ty),*) => {
        /// Section converted to the requested element type (exact for the listed widenings).
        pub fn $name(&self, key: &str) -> Vec<$t> {
            let s = self.sections.get(key).unwrap_or_else(|| panic!("pack section {key} missing"));
            let b = &s.bytes;
            match s.dtype {
                $(DType::$dt => b.chunks_exact(std::mem::size_of::<$src>()).map(|c| <$src>::from_le_bytes(c.try_into().unwrap()) as $t).collect(),)*
                d => panic!("section {key}: cannot read {:?} as {}", d, stringify!($t)),
            }
        }
    };
}

impl Pack {
    pub fn read(path: &str) -> std::io::Result<Pack> {
        let buf = std::fs::read(path)?;
        Ok(Pack::from_bytes(&buf))
    }
    pub fn from_bytes(buf: &[u8]) -> Pack {
        assert!(&buf[0..8] == b"BAYPACK1", "not a BAYPACK1 file");
        let hl = u32::from_le_bytes(buf[8..12].try_into().unwrap()) as usize;
        let hdr: Value = serde_json::from_slice(&buf[12..12 + hl]).expect("pack header");
        let data = 12 + hl + ((8 - ((12 + hl) % 8)) % 8);
        let mut sections = HashMap::new();
        let mut order = Vec::new();
        for s in hdr["sections"].as_array().unwrap() {
            let name = s["name"].as_str().unwrap().to_string();
            let dtype = DType::parse(s["dtype"].as_str().unwrap());
            let len = s["length"].as_u64().unwrap() as usize;
            let off = data + s["offset"].as_u64().unwrap() as usize;
            let bytes = buf[off..off + len * dtype.size()].to_vec();
            order.push(name.clone());
            sections.insert(name, Section { dtype, len, bytes });
        }
        Pack { meta: hdr["meta"].clone(), sections, order }
    }
    pub fn has(&self, key: &str) -> bool { self.sections.contains_key(key) }
    pub fn dtype(&self, key: &str) -> DType { self.sections[key].dtype }
    getter!(f64s, f64, F64 => f64, F32 => f32, I32 => i32, U32 => u32, I16 => i16, U16 => u16, I8 => i8, U8 => u8);
    getter!(f32s, f32, F32 => f32);
    getter!(i32s, i32, I32 => i32, U32 => u32, I16 => i16, U16 => u16, I8 => i8, U8 => u8);
    getter!(u32s, u32, U32 => u32, U16 => u16, U8 => u8);
    getter!(u8s, u8, U8 => u8);
    getter!(i8s, i8, I8 => i8);
}

/// Writer (Rust traces / exports read by the JS tools).
pub struct PackWriter { pub meta: Value, secs: Vec<(String, &'static str, usize, Vec<u8>)> }

pub trait PackElem: Copy { const DT: &'static str; fn put(self, out: &mut Vec<u8>); }
macro_rules! elem { ($t:ty, $n:expr) => { impl PackElem for $t { const DT: &'static str = $n; fn put(self, out: &mut Vec<u8>) { out.extend_from_slice(&self.to_le_bytes()); } } }; }
elem!(f64, "f64"); elem!(f32, "f32"); elem!(i32, "i32"); elem!(u32, "u32"); elem!(i16, "i16"); elem!(u16, "u16"); elem!(i8, "i8"); elem!(u8, "u8");

impl PackWriter {
    pub fn new() -> Self { PackWriter { meta: Value::Object(Default::default()), secs: Vec::new() } }
    pub fn add<T: PackElem>(&mut self, name: &str, data: &[T]) -> &mut Self {
        let mut b = Vec::with_capacity(data.len() * std::mem::size_of::<T>());
        for &x in data { x.put(&mut b); }
        self.secs.push((name.to_string(), T::DT, data.len(), b));
        self
    }
    pub fn write(&self, path: &str) -> std::io::Result<()> {
        let mut off = 0usize;
        let mut list = Vec::new();
        for (n, dt, len, b) in &self.secs {
            list.push(serde_json::json!({"name": n, "dtype": dt, "offset": off, "length": len}));
            off += b.len().div_ceil(8) * 8;
        }
        let hdr = serde_json::to_vec(&serde_json::json!({"meta": self.meta, "sections": list})).unwrap();
        let mut out = Vec::with_capacity(off + hdr.len() + 32);
        out.extend_from_slice(b"BAYPACK1");
        out.extend_from_slice(&(hdr.len() as u32).to_le_bytes());
        out.extend_from_slice(&hdr);
        while out.len() % 8 != 0 { out.push(0); }
        for (_, _, _, b) in &self.secs { out.extend_from_slice(b); while out.len() % 8 != 0 { out.push(0); } }
        std::fs::write(path, out)
    }
}
impl Default for PackWriter { fn default() -> Self { Self::new() } }
