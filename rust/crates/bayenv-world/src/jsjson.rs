//! Minimal JSON with JavaScript semantics, for files a port must rewrite byte for byte (`JSON.parse` /
//! `JSON.stringify`): object key order preserved, every number an f64 (correctly rounded parse), numbers printed
//! like JS `Number.prototype.toString`, strings escaped like `JSON.stringify`. OWNER: geom port agent.
use std::fmt::Write;

#[derive(Clone, Debug, PartialEq)]
pub enum Json { Null, Bool(bool), Num(f64), Str(String), Arr(Vec<Json>), Obj(Vec<(String, Json)>) }

impl Json {
    pub fn get(&self, k: &str) -> Option<&Json> { match self { Json::Obj(v) => v.iter().find(|(a, _)| a == k).map(|(_, b)| b), _ => None } }
    pub fn get_mut(&mut self, k: &str) -> Option<&mut Json> { match self { Json::Obj(v) => v.iter_mut().find(|(a, _)| a == k).map(|(_, b)| b), _ => None } }
    /// JS `o[k] = v`: replaces in place, or appends a new key.
    pub fn set(&mut self, k: &str, val: Json) {
        if let Json::Obj(v) = self { if let Some(p) = v.iter_mut().find(|(a, _)| a == k) { p.1 = val; } else { v.push((k.to_string(), val)); } }
    }
    pub fn as_f64(&self) -> Option<f64> { if let Json::Num(x) = self { Some(*x) } else { None } }
    pub fn as_str(&self) -> Option<&str> { if let Json::Str(s) = self { Some(s) } else { None } }
    pub fn as_arr(&self) -> Option<&Vec<Json>> { if let Json::Arr(a) = self { Some(a) } else { None } }
    pub fn as_arr_mut(&mut self) -> Option<&mut Vec<Json>> { if let Json::Arr(a) = self { Some(a) } else { None } }
    /// JS truthiness.
    pub fn truthy(&self) -> bool {
        match self { Json::Null => false, Json::Bool(b) => *b, Json::Num(x) => *x != 0.0 && !x.is_nan(), Json::Str(s) => !s.is_empty(), _ => true }
    }
    /// `JSON.stringify(v)`.
    pub fn stringify(&self) -> String { let mut s = String::new(); write_val(&mut s, self, None, 0); s }
    /// `JSON.stringify(v, null, indent)`.
    pub fn stringify_indent(&self, indent: usize) -> String { let mut s = String::new(); write_val(&mut s, self, Some(indent), 0); s }
    /// Convert from serde_json (object order as serde gives it).
    pub fn from_serde(v: &serde_json::Value) -> Json {
        match v {
            serde_json::Value::Null => Json::Null, serde_json::Value::Bool(b) => Json::Bool(*b),
            serde_json::Value::Number(n) => Json::Num(n.as_f64().unwrap_or(f64::NAN)), serde_json::Value::String(s) => Json::Str(s.clone()),
            serde_json::Value::Array(a) => Json::Arr(a.iter().map(Json::from_serde).collect()),
            serde_json::Value::Object(o) => Json::Obj(o.iter().map(|(k, v)| (k.clone(), Json::from_serde(v))).collect()),
        }
    }
}

/// JS `Number.prototype.toString()` (radix 10).
pub fn js_number(x: f64) -> String {
    if x.is_nan() { return "NaN".into(); }
    if x.is_infinite() { return if x > 0.0 { "Infinity".into() } else { "-Infinity".into() }; }
    if x == 0.0 { return "0".into(); }
    let e = format!("{:e}", x.abs());            // shortest round-trip digits: "d.ddde-7"
    let (mant, exp) = e.split_once('e').unwrap();
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let k = digits.len() as i32; let n = exp.parse::<i32>().unwrap() + 1;
    let mut s = String::new();
    if x < 0.0 { s.push('-'); }
    if k <= n && n <= 21 { s.push_str(&digits); for _ in 0..n - k { s.push('0'); } }
    else if 0 < n && n <= 21 { s.push_str(&digits[..n as usize]); s.push('.'); s.push_str(&digits[n as usize..]); }
    else if -6 < n && n <= 0 { s.push_str("0."); for _ in 0..-n { s.push('0'); } s.push_str(&digits); }
    else {
        let e1 = n - 1;
        s.push_str(&digits[..1]);
        if k > 1 { s.push('.'); s.push_str(&digits[1..]); }
        s.push('e'); s.push(if e1 >= 0 { '+' } else { '-' }); write!(s, "{}", e1.abs()).unwrap();
    }
    s
}

fn write_str(s: &mut String, v: &str) {
    s.push('"');
    for c in v.chars() {
        match c {
            '"' => s.push_str("\\\""), '\\' => s.push_str("\\\\"), '\u{8}' => s.push_str("\\b"), '\u{c}' => s.push_str("\\f"),
            '\n' => s.push_str("\\n"), '\r' => s.push_str("\\r"), '\t' => s.push_str("\\t"),
            c if (c as u32) < 0x20 => { write!(s, "\\u{:04x}", c as u32).unwrap(); }
            c => s.push(c),
        }
    }
    s.push('"');
}

fn write_val(s: &mut String, v: &Json, ind: Option<usize>, depth: usize) {
    let nl = |s: &mut String, d: usize| { if let Some(i) = ind { s.push('\n'); for _ in 0..i * d { s.push(' '); } } };
    match v {
        Json::Null => s.push_str("null"),
        Json::Bool(b) => s.push_str(if *b { "true" } else { "false" }),
        Json::Num(x) => if x.is_finite() { s.push_str(&js_number(*x)) } else { s.push_str("null") },
        Json::Str(t) => write_str(s, t),
        Json::Arr(a) => {
            if a.is_empty() { s.push_str("[]"); return; }
            s.push('[');
            for (i, x) in a.iter().enumerate() { if i > 0 { s.push(','); } nl(s, depth + 1); write_val(s, x, ind, depth + 1); }
            nl(s, depth); s.push(']');
        }
        Json::Obj(o) => {
            if o.is_empty() { s.push_str("{}"); return; }
            s.push('{');
            for (i, (k, x)) in o.iter().enumerate() {
                if i > 0 { s.push(','); }
                nl(s, depth + 1); write_str(s, k); s.push(':'); if ind.is_some() { s.push(' '); }
                write_val(s, x, ind, depth + 1);
            }
            nl(s, depth); s.push('}');
        }
    }
}

/// `JSON.parse(text)` (duplicate keys: last value wins, at the first key's position).
pub fn parse(text: &str) -> Result<Json, String> {
    let b = text.as_bytes(); let mut p = Parser { b, i: 0 };
    p.ws(); let v = p.val()?; p.ws();
    if p.i != b.len() { return Err(format!("trailing data at {}", p.i)); }
    Ok(v)
}

struct Parser<'a> { b: &'a [u8], i: usize }
impl<'a> Parser<'a> {
    fn ws(&mut self) { while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') { self.i += 1; } }
    fn err<T>(&self, m: &str) -> Result<T, String> { Err(format!("JSON: {m} at {}", self.i)) }
    fn lit(&mut self, w: &str, v: Json) -> Result<Json, String> {
        if self.b[self.i..].starts_with(w.as_bytes()) { self.i += w.len(); Ok(v) } else { self.err("bad literal") }
    }
    fn val(&mut self) -> Result<Json, String> {
        if self.i >= self.b.len() { return self.err("eof"); }
        match self.b[self.i] {
            b'{' => {
                self.i += 1; let mut o: Vec<(String, Json)> = Vec::new(); self.ws();
                if self.b.get(self.i) == Some(&b'}') { self.i += 1; return Ok(Json::Obj(o)); }
                loop {
                    self.ws(); if self.b.get(self.i) != Some(&b'"') { return self.err("key"); }
                    let k = self.string()?; self.ws();
                    if self.b.get(self.i) != Some(&b':') { return self.err(":"); }
                    self.i += 1; self.ws(); let v = self.val()?;
                    if let Some(p) = o.iter_mut().find(|(a, _)| *a == k) { p.1 = v; } else { o.push((k, v)); }
                    self.ws();
                    match self.b.get(self.i) { Some(b',') => self.i += 1, Some(b'}') => { self.i += 1; return Ok(Json::Obj(o)); } _ => return self.err("object") }
                }
            }
            b'[' => {
                self.i += 1; let mut a = Vec::new(); self.ws();
                if self.b.get(self.i) == Some(&b']') { self.i += 1; return Ok(Json::Arr(a)); }
                loop {
                    self.ws(); a.push(self.val()?); self.ws();
                    match self.b.get(self.i) { Some(b',') => self.i += 1, Some(b']') => { self.i += 1; return Ok(Json::Arr(a)); } _ => return self.err("array") }
                }
            }
            b'"' => Ok(Json::Str(self.string()?)),
            b't' => self.lit("true", Json::Bool(true)), b'f' => self.lit("false", Json::Bool(false)), b'n' => self.lit("null", Json::Null),
            _ => {
                let s = self.i;
                while self.i < self.b.len() && matches!(self.b[self.i], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9') { self.i += 1; }
                let t = std::str::from_utf8(&self.b[s..self.i]).unwrap();
                t.parse::<f64>().map(Json::Num).or_else(|_| self.err("number"))
            }
        }
    }
    fn hex4(&mut self) -> Result<u32, String> {
        if self.i + 4 > self.b.len() { return self.err("\\u"); }
        let h = u32::from_str_radix(std::str::from_utf8(&self.b[self.i..self.i + 4]).unwrap(), 16).or_else(|_| self.err("\\u"))?;
        self.i += 4; Ok(h)
    }
    fn string(&mut self) -> Result<String, String> {
        self.i += 1; let mut out: Vec<u8> = Vec::new();
        loop {
            let Some(&c) = self.b.get(self.i) else { return self.err("eof in string") };
            self.i += 1;
            match c {
                b'"' => return String::from_utf8(out).or_else(|_| self.err("utf8")),
                b'\\' => {
                    let Some(&e) = self.b.get(self.i) else { return self.err("eof") }; self.i += 1;
                    let ch: char = match e {
                        b'"' => '"', b'\\' => '\\', b'/' => '/', b'b' => '\u{8}', b'f' => '\u{c}', b'n' => '\n', b'r' => '\r', b't' => '\t',
                        b'u' => {
                            let h = self.hex4()?;
                            if (0xD800..0xDC00).contains(&h) && self.b[self.i..].starts_with(b"\\u") {
                                let save = self.i; self.i += 2; let l = self.hex4()?;
                                if (0xDC00..0xE000).contains(&l) { char::from_u32(0x10000 + ((h - 0xD800) << 10) + (l - 0xDC00)).unwrap() } else { self.i = save; '\u{FFFD}' }
                            } else { char::from_u32(h).unwrap_or('\u{FFFD}') }
                        }
                        _ => return self.err("escape"),
                    };
                    let mut buf = [0u8; 4]; out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                }
                _ => out.push(c),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn numbers_like_js() {
        for (x, s) in [(1.0, "1"), (0.1, "0.1"), (1e21, "1e+21"), (1e20, "100000000000000000000"), (1.5e-7, "1.5e-7"), (0.000001, "0.000001"),
            (-0.0, "0"), (123.456, "123.456"), (-2.5e-10, "-2.5e-10"), (5e-324, "5e-324"), (1.7976931348623157e308, "1.7976931348623157e+308")] {
            assert_eq!(js_number(x), s);
        }
        let v = parse(r#"{"a":[1,2.50,{"b":"x\"\n"}],"c":{},"d":[]}"#).unwrap();
        assert_eq!(v.stringify(), r#"{"a":[1,2.5,{"b":"x\"\n"}],"c":{},"d":[]}"#);
        assert_eq!(v.stringify_indent(1), "{\n \"a\": [\n  1,\n  2.5,\n  {\n   \"b\": \"x\\\"\\n\"\n  }\n ],\n \"c\": {},\n \"d\": []\n}");
    }
}
