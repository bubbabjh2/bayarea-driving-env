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
    // diagnostics (kernel experiments): BAYENV_KDIR/<name>.wgsl replaces a source
    if let Ok(d) = std::env::var("BAYENV_KDIR") { if let Ok(t) = std::fs::read_to_string(std::path::Path::new(&d).join(format!("{n}.wgsl"))) { return Box::leak(t.into_boxed_str()); } }
    SOURCES.iter().chain(HOST_SOURCES).find(|(k, _)| *k == n).map(|(_, s)| *s).unwrap_or_else(|| panic!("no kernel source {name}"))
}

/// `loadKernelSource(name)`: the file with every `// #include <file>` line replaced (recursively).
pub fn load(name: &str) -> String { resolve_includes(raw(name)) }

/// Kernels of this crate (`crates/bayenv-gpu/kernels/`): modified copies / helpers, see the header of each file.
pub const HOST_SOURCES: &[(&str, &str)] = &[
    ("contact_args", include_str!("../kernels/contact_args.wgsl")),
    ("step_mark", include_str!("../kernels/step_mark.wgsl")),
    ("collided", include_str!("../kernels/collided.wgsl")),
    ("resolve_fused", include_str!("../kernels/resolve_fused.wgsl")),
    ("sorted", include_str!("../kernels/sorted.wgsl")),
    ("list_args", include_str!("../kernels/list_args.wgsl")),
    ("scan4k", include_str!("../kernels/scan4k.wgsl")),
    ("patches", include_str!("../kernels/patches.wgsl")),
    ("obs_f16", include_str!("../kernels/obs_f16.wgsl")),
    ("cpu_connector_join", include_str!("../kernels/cpu_connector_join.wgsl")),
    ("contact_tangent", include_str!("../kernels/contact_tangent.wgsl")),
    ("contact_order", include_str!("../kernels/contact_order.wgsl")),
    ("spawn_order", include_str!("../kernels/spawn_order.wgsl")),
    ("leaders_order", include_str!("../kernels/leaders_order.wgsl")),
    ("courtesy_order", include_str!("../kernels/courtesy_order.wgsl")),
    ("junction_order", include_str!("../kernels/junction_order.wgsl")),
    ("jyield_order", include_str!("../kernels/jyield_order.wgsl")),
    ("hash_order", include_str!("../kernels/hash_order.wgsl")),
    ("model_contacts", include_str!("../kernels/model_contacts.wgsl")),
    ("model_integrate", include_str!("../kernels/model_integrate.wgsl")),
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

/// Text of the function `fn name(` in `src` (to its closing brace): (start, end) byte offsets.
fn find_fn(src: &str, name: &str) -> Option<(usize, usize)> {
    let a = src.find(&format!("fn {name}("))?;
    let open = a + src[a..].find('{')?;
    let mut depth = 0i32;
    for (i, c) in src[open..].char_indices() {
        if c == '{' { depth += 1; } else if c == '}' { depth -= 1; if depth == 0 { return Some((a, open + i + 1)); } }
    }
    None
}
fn sections<'a>(text: &'a str, tag: &str) -> Vec<(&'a str, String)> {
    let mut out = Vec::new(); let mut cur: Option<(&str, Vec<&str>)> = None;
    for line in text.lines() {
        if let Some(n) = line.strip_prefix(tag) { cur = Some((n.trim(), Vec::new())); continue; }
        if line.trim() == "//#end" { if let Some((n, l)) = cur.take() { out.push((n, l.join("\n"))); } continue; }
        if let Some((_, l)) = cur.as_mut() { l.push(line); }
    }
    out
}
/// Apply a text-patch file of this crate (`//#orig id` / `//#patch id` snippet pairs, e.g. kernels/cpu_connector_join.wgsl)
/// to a resolved kernel source; panics unless every original snippet occurs exactly once (a changed upstream kernel).
pub fn patch_text(src: &str, file: &str) -> String {
    let text = host_raw(file);
    let origs = sections(text, "//#orig ");
    let mut out = src.to_string();
    for (name, body) in sections(text, "//#patch ") {
        let orig = &origs.iter().find(|(n, _)| *n == name).unwrap_or_else(|| panic!("{file} {name}: no //#orig")).1;
        assert_eq!(out.matches(orig.as_str()).count(), 1, "kernel patch {file} {name}: the original text is not in the source exactly once");
        out = out.replacen(orig.as_str(), &body, 1);
    }
    out
}
/// The model-fix flags of GpuOptions (all false = the JS kernels unchanged; see `GpuOptions` for each).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ModelFixes {
    pub cpu_connector_join: bool, pub contact_tangent: bool,
    pub wreck_real_mass: bool, pub curvature_dynamics: bool, pub wreck_end_stop: bool, pub contact_episodes: bool,
    pub connector_barrier: bool, pub brake_hold: bool, pub heavy_brakes: bool, pub barrier_soft_push: bool, pub impact_reverse: bool,
}
impl ModelFixes {
    /// every fix on (the GpuOptions default is this with curvature_dynamics off: `ModelFixes::default_model()`)
    pub fn default_model() -> ModelFixes { ModelFixes { curvature_dynamics: false, brake_hold: false, ..ModelFixes::all() } }
    pub fn all() -> ModelFixes {
        ModelFixes { cpu_connector_join: true, contact_tangent: true, wreck_real_mass: true, curvature_dynamics: true, wreck_end_stop: true, contact_episodes: true,
            connector_barrier: true, brake_hold: true, heavy_brakes: true, barrier_soft_push: true, impact_reverse: true }
    }
    /// kernels/model_integrate.wgsl is applied (any of its flags on)
    pub fn integrate_on(&self) -> bool { self.curvature_dynamics || self.wreck_end_stop || self.brake_hold || self.heavy_brakes || self.impact_reverse }
    /// kernels/model_contacts.wgsl is applied (any of its flags on)
    pub fn contacts_on(&self) -> bool { self.wreck_real_mass || self.contact_episodes || self.connector_barrier || self.barrier_soft_push || self.impact_reverse }
}
fn mf_consts(pairs: &[(&str, bool)]) -> String { pairs.iter().map(|(n, b)| format!("const MF_{n}: bool = {b};\n")).collect() }
/// Model patches selected by GpuOptions (default off: the JS kernels unchanged) for the kernel file `file`.
/// `file` "contacts" also covers kernels/resolve_fused.wgsl (contacts.wgsl + collidefns.wgsl included unchanged).
pub fn model_patches(src: String, file: &str, m: &ModelFixes) -> String {
    match file {
        "integrate" => {
            let src = if m.cpu_connector_join { patch_text(&src, "cpu_connector_join") } else { src };
            if !m.integrate_on() { return src; }
            mf_consts(&[("HEAVY_BRAKES", m.heavy_brakes), ("BRAKE_HOLD", m.brake_hold), ("IMPACT_REVERSE", m.impact_reverse), ("WRECK_END_STOP", m.wreck_end_stop), ("CURVATURE", m.curvature_dynamics)])
                + &patch_text(&src, "model_integrate")
        }
        "contacts" => {
            let src = if m.contact_tangent { patch_text(&src, "contact_tangent") } else { src };
            if !m.contacts_on() { return src; }
            mf_consts(&[("WRECK_REAL_MASS", m.wreck_real_mass), ("IMPACT_REVERSE", m.impact_reverse), ("SOFT_PUSH", m.barrier_soft_push), ("EPISODES", m.contact_episodes), ("CONN_BARRIER", m.connector_barrier)])
                + &patch_text(&src, "model_contacts")
        }
        _ => src,
    }
}
/// Determinism patches selected by GpuOptions (off: the JS kernels unchanged) for the kernel file `file` (see
/// `model_patches` for "contacts"): kernels/contact_order.wgsl (`deterministic_contacts`), kernels/spawn_order.wgsl
/// (`deterministic_spawn`), kernels/leaders_order.wgsl, courtesy_order.wgsl, junction_order.wgsl, jyield_order.wgsl, hash_order.wgsl
/// (`deterministic_leaders`).
pub fn order_patches(src: String, file: &str, o: &crate::sim::GpuOptions) -> String {
    let src = if o.deterministic_contacts && file == "contacts" { patch_text(&src, "contact_order") } else { src };
    let src = if o.deterministic_spawn && file == "spawn" { patch_text(&src, "spawn_order") } else { src };
    let src = if o.deterministic_leaders && file == "leaders" { patch_text(&src, "leaders_order") } else { src };
    let src = if o.deterministic_leaders && file == "courtesy" { patch_text(&src, "courtesy_order") } else { src };
    let src = if o.deterministic_leaders && file == "junction" { patch_text(&src, "junction_order") } else { src };
    let src = if o.deterministic_leaders && file == "jyield" { patch_text(&src, "jyield_order") } else { src };
    if o.deterministic_leaders && file == "hash" { patch_text(&src, "hash_order") } else { src }
}
/// Claim / commit rounds per step of the deterministic spawn (kernels/spawn_order.wgsl): a spawn or station re-entry
/// that loses its lane to a smaller id tries the lanes left in the next round (the JS kernel: in the same pass), and
/// after the last round in the next step.
pub const DET_ROUNDS: usize = 1;
/// Apply kernels/patches.wgsl to a resolved kernel source: each patched function present in `src` is replaced
/// (panics when its text differs from the original the patch was written for).
pub fn patch(src: &str) -> String {
    let text = host_raw("patches");
    let origs = sections(text, "//#orig ");
    let mut out = src.to_string();
    let skip = std::env::var("BAYENV_NOPATCH").unwrap_or_default();   // diagnostics: comma list of patches to leave out
    for (name, body) in sections(text, "//#patch ") {
        if skip.split(',').any(|x| x == name) { continue; }
        let Some((a, b)) = find_fn(&out, name) else { continue };
        let orig = &origs.iter().find(|(n, _)| *n == name).unwrap_or_else(|| panic!("patch {name}: no //#orig")).1;
        let norm = |x: &str| x.split_whitespace().collect::<Vec<_>>().join(" ");
        assert_eq!(norm(&out[a..b]), norm(orig), "kernel patch {name}: the original function changed upstream");
        out.replace_range(a..b, &body);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::ModelFixes;
    fn only(f: impl Fn(&mut ModelFixes)) -> ModelFixes { let mut m = ModelFixes::default(); f(&mut m); m }
    #[test]
    fn cpu_connector_join_applies() {
        let src = super::load("integrate");
        assert_eq!(super::model_patches(src.clone(), "integrate", &ModelFixes::default()), src);   // flags off: the JS kernel unchanged
        let p = super::model_patches(src.clone(), "integrate", &only(|m| m.cpu_connector_join = true));
        assert!(p.contains("if (sArr > sIn + 0.3 && P.hasLG != 0u)") && !p.contains("if (s > sIn + 0.3 && P.hasLG != 0u)"));
        assert_eq!(super::model_patches(super::load("contacts"), "contacts", &only(|m| m.cpu_connector_join = true)), super::load("contacts"));
    }
    #[test]
    fn order_patches_apply() {
        let on = crate::sim::GpuOptions::default();
        let off = crate::sim::GpuOptions { deterministic_contacts: false, deterministic_spawn: false, deterministic_leaders: false, ..Default::default() };
        for f in ["contacts", "spawn", "leaders", "courtesy", "junction", "jyield", "hash", "integrate"] {
            let src = super::load(f);
            assert_eq!(super::order_patches(src.clone(), f, &off), src, "{f}: flags off = the JS kernel");
            let p = super::order_patches(src.clone(), f, &on);   // panics unless every original snippet is there once
            assert_eq!(p == src, f == "integrate", "{f}");
        }
        let fused = super::resolve_includes(super::host_raw("resolve_fused"));
        assert!(super::order_patches(fused, "contacts", &on).contains("((i + j) & 0x3ffffu)"));
    }
    #[test]
    fn contact_tangent_applies() {
        let src = super::load("contacts");
        assert_eq!(super::model_patches(src.clone(), "contacts", &ModelFixes::default()), src);
        assert_eq!(super::model_patches(super::load("integrate"), "integrate", &only(|m| m.contact_tangent = true)), super::load("integrate"));
        let p = super::model_patches(src.clone(), "contacts", &only(|m| m.contact_tangent = true));
        assert!(p.contains("let tn = tx * c.nx + tz * c.nz;") && !p.contains("if (tl > 1e-6) {"));
        let f = super::resolve_includes(super::host_raw("resolve_fused"));
        assert!(super::model_patches(f, "contacts", &only(|m| m.contact_tangent = true)).contains("let tn = tx * c.nx + tz * c.nz;"));
    }
    #[test]
    fn model_and_order_patches_combine() {
        // the model fixes and the determinism patches touch the same kernels: every combination applies
        let det_on = crate::sim::GpuOptions::default();
        let det_off = crate::sim::GpuOptions { deterministic_contacts: false, deterministic_spawn: false, deterministic_leaders: false, ..Default::default() };
        for mf in [ModelFixes::default(), ModelFixes::default_model(), ModelFixes::all()] {
            for o in [&det_on, &det_off] {
                for f in ["contacts", "integrate", "spawn", "leaders", "courtesy", "junction", "jyield", "hash"] {
                    let _ = super::order_patches(super::model_patches(super::load(f), f, &mf), f, o);
                }
                let fused = super::resolve_includes(super::host_raw("resolve_fused"));
                let _ = super::order_patches(super::model_patches(fused, "contacts", &mf), "contacts", &crate::sim::GpuOptions { deterministic_spawn: false, deterministic_leaders: false, ..o.clone() });
            }
        }
    }
    #[test]
    fn model_fixes_apply() {
        // every patch applies to its file (and the fused resolve), each flag alone and all together
        let all = ModelFixes::all();
        let i = super::model_patches(super::load("integrate"), "integrate", &all);
        assert!(i.contains("const MF_CURVATURE: bool = true;") && i.contains("MF_HEAVY_BRAKES && m > 5000.0"));
        let c = super::model_patches(super::load("contacts"), "contacts", &all);
        assert!(c.contains("const MF_EPISODES: bool = true;") && c.contains("fn connBarrier("));
        let f = super::model_patches(super::resolve_includes(super::host_raw("resolve_fused")), "contacts", &all);
        assert!(f.contains("fn connBarrier("));
        let one: [fn(&mut ModelFixes); 9] = [|m| m.wreck_real_mass = true, |m| m.curvature_dynamics = true, |m| m.wreck_end_stop = true, |m| m.contact_episodes = true,
            |m| m.connector_barrier = true, |m| m.brake_hold = true, |m| m.heavy_brakes = true, |m| m.barrier_soft_push = true, |m| m.impact_reverse = true];
        for f in one { let m = only(f); for file in ["integrate", "contacts"] { let _ = super::model_patches(super::load(file), file, &m); } }
    }
}
