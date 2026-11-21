//! `jsmath` vs V8: compares every function against the node dump written by
//! `node rust/tools/oracle/gen_math.mjs` (default `rust/oracle/math.bin`, override with `JSMATH_ORACLE=<file>`).
//! Bit-exact comparison; any NaN equals any NaN (JS cannot observe NaN payloads). Skips if the dump is missing.

use bayenv_world::jsmath;
use std::path::PathBuf;

fn eval(name: &str, a: f64, b: f64) -> Option<f64> {
    Some(match name {
        "atan2" => jsmath::atan2(a, b),
        "atan" => jsmath::atan(a),
        "sin" => jsmath::sin(a),
        "cos" => jsmath::cos(a),
        "tan" => jsmath::tan(a),
        "exp" => jsmath::exp(a),
        "log" => jsmath::log(a),
        "acos" => jsmath::acos(a),
        "asin" => jsmath::asin(a),
        "pow" => jsmath::pow(a, b),
        "cbrt" => jsmath::cbrt(a),
        "log2" => jsmath::log2(a),
        "log10" => jsmath::log10(a),
        "expm1" => jsmath::expm1(a),
        "log1p" => jsmath::log1p(a),
        "sinh" => jsmath::sinh(a),
        "cosh" => jsmath::cosh(a),
        "tanh" => jsmath::tanh(a),
        "atanh" => jsmath::atanh(a),
        "hypot" => jsmath::hypot(a, b),
        _ => return None,
    })
}

#[test]
fn jsmath_matches_v8_dump() {
    let path = std::env::var_os("JSMATH_ORACLE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../oracle/math.bin"));
    let Ok(buf) = std::fs::read(&path) else {
        bayenv_world::oracle::skip("jsmath_oracle", &format!("{} missing", path.display()));
        return;
    };
    assert_eq!(&buf[..8], b"JSMATH01", "bad magic in {}", path.display());
    let u32_at = |o: usize| u32::from_le_bytes(buf[o..o + 4].try_into().unwrap()) as usize;
    let f64_at = |o: usize| f64::from_le_bytes(buf[o..o + 8].try_into().unwrap());
    let nf = u32_at(8);
    let mut o = 12;
    let (mut total, mut bad_total) = (0usize, 0usize);
    let mut report = Vec::new();
    for _ in 0..nf {
        let len = u32_at(o);
        o += 4;
        let name = std::str::from_utf8(&buf[o..o + len]).unwrap().to_string();
        o += len;
        let n = u32_at(o);
        o += 4;
        let mut bad = 0usize;
        let mut first = None;
        for i in 0..n {
            let p = o + i * 24;
            let (a, b, want) = (f64_at(p), f64_at(p + 8), f64_at(p + 16));
            let got = eval(&name, a, b).unwrap_or_else(|| panic!("unknown function {name} in dump"));
            if got.to_bits() != want.to_bits() && !(got.is_nan() && want.is_nan()) {
                bad += 1;
                first.get_or_insert((a, b, got, want));
            }
        }
        o += n * 24;
        total += n;
        bad_total += bad;
        let mut line = format!("{name:6} {bad:>7} / {n} mismatches");
        if let Some((a, b, got, want)) = first {
            line += &format!("   first: f({a:e}, {b:e}) = {got:e}, js {want:e}");
        }
        report.push(line);
    }
    eprintln!("jsmath vs {}:\n{}", path.display(), report.join("\n"));
    assert_eq!(bad_total, 0, "{bad_total} of {total} results differ from V8");
}
