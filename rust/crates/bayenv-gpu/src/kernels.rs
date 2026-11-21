//! The WGSL kernels of `src/gpu/kernels/` (embedded unchanged at compile time) and the pipeline table of
//! `GPUSim._compile()`. `// #include x.wgsl` lines are resolved like `device.js loadKernelSource`.
use crate::defines::{prefix, ObsMode};

pub const SOURCES: &[(&str, &str)] = &[
    ("act", include_str!("../../../../src/gpu/kernels/act.wgsl")),
    ("build", include_str!("../../../../src/gpu/kernels/build.wgsl")),
    ("collidefns", include_str!("../../../../src/gpu/kernels/collidefns.wgsl")),
    ("common", include_str!("../../../../src/gpu/kernels/common.wgsl")),
    ("contacts", include_str!("../../../../src/gpu/kernels/contacts.wgsl")),
    ("controls", include_str!("../../../../src/gpu/kernels/controls.wgsl")),
    ("courtesy", include_str!("../../../../src/gpu/kernels/courtesy.wgsl")),
    ("density", include_str!("../../../../src/gpu/kernels/density.wgsl")),
    ("hash", include_str!("../../../../src/gpu/kernels/hash.wgsl")),
    ("integrate", include_str!("../../../../src/gpu/kernels/integrate.wgsl")),
    ("junction", include_str!("../../../../src/gpu/kernels/junction.wgsl")),
    ("jyield", include_str!("../../../../src/gpu/kernels/jyield.wgsl")),
    ("leaders", include_str!("../../../../src/gpu/kernels/leaders.wgsl")),
    ("lists", include_str!("../../../../src/gpu/kernels/lists.wgsl")),
    ("ops", include_str!("../../../../src/gpu/kernels/ops.wgsl")),
    ("perceive", include_str!("../../../../src/gpu/kernels/perceive.wgsl")),
    ("physics", include_str!("../../../../src/gpu/kernels/physics.wgsl")),
    ("scan", include_str!("../../../../src/gpu/kernels/scan.wgsl")),
    ("sense", include_str!("../../../../src/gpu/kernels/sense.wgsl")),
    ("spawn", include_str!("../../../../src/gpu/kernels/spawn.wgsl")),
    ("yield", include_str!("../../../../src/gpu/kernels/yield.wgsl")),
];

pub fn raw(name: &str) -> &'static str {
    let n = name.trim_end_matches(".wgsl");
    SOURCES.iter().find(|(k, _)| *k == n).map(|(_, s)| *s).unwrap_or_else(|| panic!("no kernel source {name}"))
}

/// `loadKernelSource(name)`: the file with every `// #include <file>` line replaced (recursively).
pub fn load(name: &str) -> String { resolve_includes(raw(name)) }

/// Kernels of this crate (`crates/bayenv-gpu/kernels/`): modified copies / helpers, see the header of each file.
pub const HOST_SOURCES: &[(&str, &str)] = &[
    ("contact_args", include_str!("../kernels/contact_args.wgsl")),
    ("resolve_fused", include_str!("../kernels/resolve_fused.wgsl")),
    ("sorted", include_str!("../kernels/sorted.wgsl")),
    ("list_args", include_str!("../kernels/list_args.wgsl")),
    ("scan4k", include_str!("../kernels/scan4k.wgsl")),
];

/// Lane-list-order variant of a per-car kernel (kernels/sorted.wgsl): the entry `entry` of `src` becomes the
/// function `{entry}_at` and the wrapper entry `{entry}_sorted` is appended.
pub fn sorted_source(src: &str, entry: &str, wg: u32) -> String {
    let sig_g = format!("fn {entry}(@builtin(global_invocation_id) g: vec3u)");
    let sig_li = format!("fn {entry}(@builtin(global_invocation_id) g: vec3u, @builtin(local_invocation_index) li: u32)");
    let (sig, li) = if src.contains(&sig_li) { (sig_li, true) } else { (sig_g, false) };
    let p = src.find(&sig).unwrap_or_else(|| panic!("sorted_source: no `{sig}`"));
    let attr = src[..p].rfind("@compute").expect("sorted_source: no @compute before the entry");
    let mut out = String::with_capacity(src.len() + 4096);
    out.push_str(&src[..attr]);
    out.push_str(&if li { format!("fn {entry}_at(g: vec3u, li: u32)") } else { format!("fn {entry}_at(g: vec3u)") });
    out.push_str(&src[p + sig.len()..]); out.push('\n');
    if !src.contains("var<storage, read> vidx") { out.push_str(&load("lists")); out.push('\n'); }
    // template sections: //#if G / //#if LI ... //#endif
    let mut keep = true;
    for line in host_raw("sorted").lines() {
        let t = line.trim();
        if let Some(c) = t.strip_prefix("//#if ") { keep = (c == "LI") == li; continue; }
        if t == "//#endif" { keep = true; continue; }
        if keep { out.push_str(&line.replace("ENTRY", entry).replace("WG", &wg.to_string())); out.push('\n'); }
    }
    out
}
pub fn host_raw(name: &str) -> &'static str { HOST_SOURCES.iter().find(|(k, _)| *k == name).map(|(_, s)| *s).unwrap_or_else(|| panic!("no host kernel {name}")) }

/// `// #include <file>` lines of `text` replaced by the files of `src/gpu/kernels/` (recursively).
pub fn resolve_includes(text: &str) -> String {
    text.split('\n').map(|line| {
        let t = line.trim_start();
        if let Some(rest) = t.strip_prefix("//") { let r = rest.trim_start(); if let Some(f) = r.strip_prefix("#include") {
            let f = f.trim(); if !f.is_empty() && f.split_whitespace().count() >= 1 { return load(f.split_whitespace().next().unwrap()); } } }
        line.to_string()
    }).collect::<Vec<_>>().join("\n")
}

/// One compiled pipeline of the sim: (label, source file, entry point, gets the defines prefix).
pub struct KernelSpec { pub label: &'static str, pub file: &'static str, pub entry: &'static str, pub defines: bool }

/// `GPUSim._compile()` order. `junctions` = world has a junction pack, `v3` = obs v3.
pub fn specs(junctions: bool, v3: bool) -> Vec<KernelSpec> {
    let k = |label, file, entry, defines| KernelSpec { label, file, entry, defines };
    let mut v = vec![
        k("build.tick", "build", "tick", true), k("build.clear", "build", "clear", true), k("build.count", "build", "count", true),
        k("build.scatter", "build", "scatter", true), k("build.sortLanes", "build", "sortLanes", true),
        k("scanBlocks", "scan", "scanBlocks", false), k("addSums", "scan", "addSums", false),
        k("leaders", "leaders", "main", true), k("integrate", "integrate", "main", true), k("ops", "ops", "main", true), k("act", "act", "main", true), k("sense", "sense", "main", true),
        k("hash.hclear", "hash", "hclear", true), k("hash.hcount", "hash", "hcount", true), k("hash.hscatter", "hash", "hscatter", true),
        k("contacts.barrier", "contacts", "barrier", true), k("contacts.detect", "contacts", "detect", true), k("contacts.resolve", "contacts", "resolve", true),
        k("contacts.ownClear", "contacts", "ownClear", true), k("contacts.ownMin", "contacts", "ownMin", true), k("contacts.resolveRound", "contacts", "resolveRound", true),
        k("integrate.newPose", "integrate", "newPose", true), k("spawn", "spawn", "spawn", true), k("leave", "spawn", "leave", true),
        k("density.clear", "density", "clearCounts", true), k("density.count", "density", "countCars", true),
        k("density.trafClear", "density", "clearTraffic", true), k("density.trafCount", "density", "countTraffic", true),
    ];
    if junctions {
        v.extend([k("junction.clear", "junction", "jclear", true), k("junction.clearOff", "junction", "jclearOff", true),
            k("courtesy.propose", "courtesy", "propose", true), k("courtesy.apply", "courtesy", "apply", true), k("junction.link", "junction", "jlink", true),
            k("junction.perceive", "junction", "jperceive", true), k("junction.dlFind", "junction", "jdlFind", true), k("junction.dlApply", "junction", "jdlApply", true),
            k("junction.yield", "jyield", "jyield", true), k("junction.reroute", "jyield", "rerouteScan", true)]);
    }
    if v3 { v.push(k("perceive", "perceive", "main", true)); }
    v
}

/// Final source of a kernel as compiled (defines prefix + resolved file).
pub fn source(spec: &KernelSpec, obs: ObsMode, cap: usize) -> String {
    let body = load(spec.file);
    if spec.defines { prefix(obs, cap) + &body } else { body }
}

/// `parseBindings`: `@group(g) @binding(b) var<space[, access]> name`.
#[derive(Clone, Debug, PartialEq)]
pub struct Decl { pub group: u32, pub binding: u32, pub space: String, pub access: String, pub name: String }
pub fn parse_bindings(code: &str) -> Vec<Decl> {
    let mut out = Vec::new(); let b = code.as_bytes(); let mut i = 0;
    while let Some(p) = code[i..].find("@group(") {
        let s = i + p; i = s + 7;
        let parse_num = |j: &mut usize| -> Option<u32> { let st = *j; while *j < b.len() && b[*j].is_ascii_digit() { *j += 1; } code[st..*j].parse().ok() };
        let mut j = s + 7; let Some(g) = parse_num(&mut j) else { continue }; if b.get(j) != Some(&b')') { continue; } j += 1;
        while j < b.len() && b[j].is_ascii_whitespace() { j += 1; }
        if !code[j..].starts_with("@binding(") { continue; } j += 9;
        let Some(bi) = parse_num(&mut j) else { continue }; if b.get(j) != Some(&b')') { continue; } j += 1;
        while j < b.len() && b[j].is_ascii_whitespace() { j += 1; }
        if !code[j..].starts_with("var") { continue; } j += 3;
        let (mut space, mut access) = ("handle".to_string(), "read".to_string());
        while j < b.len() && b[j].is_ascii_whitespace() { j += 1; }
        if b.get(j) == Some(&b'<') {
            let e = j + code[j..].find('>').unwrap(); let inner: Vec<&str> = code[j + 1..e].split(',').map(|x| x.trim()).collect();
            space = inner[0].to_string(); if inner.len() > 1 { access = inner[1].to_string(); } j = e + 1;
        }
        while j < b.len() && b[j].is_ascii_whitespace() { j += 1; }
        let st = j; while j < b.len() && (b[j].is_ascii_alphanumeric() || b[j] == b'_') { j += 1; }
        out.push(Decl { group: g, binding: bi, space, access, name: code[st..j].to_string() }); i = j;
    }
    out
}
