//! Conditional prompt guidance for open-ended design and refinement requests.

const OPEN_ENDED_CUES: &[&str] = &[
    "design ",
    "make it realistic",
    "make this realistic",
    "make the design",
    "improve the design",
    "refine the design",
    "be creative",
    "more creative",
    "add detail",
    "add details",
    "architectural",
    "craftsman",
    "colonial",
    "modern style",
    "house",
    "home",
    "building",
    "facade",
    "interior",
    "landscape",
    "furniture",
    "chair",
    "desk",
    "lamp",
    "robot",
    "vehicle",
    "enclosure",
];

const PRECISE_OPERATION_CUES: &[&str] = &["extrude", "revolve", "fillet", "chamfer", "hole", "exactly"];

/// Add a brief and evidence-based review checklist only for open-ended design tasks.
pub(super) fn guidance(request: &str) -> Option<String> {
    let lower = request.to_lowercase();
    if !OPEN_ENDED_CUES.iter().any(|cue| lower.contains(cue)) {
        return None;
    }
    let has_number = lower.chars().any(|ch| ch.is_ascii_digit());
    if has_number && PRECISE_OPERATION_CUES.iter().any(|cue| lower.contains(cue)) {
        return None;
    }

    let seed = stable_seed(request);
    Some(format!(
        r#"Open-ended design task guidance (request seed: {seed:016x}): establish a concise brief from the user's stated use, scale, style, dimensions, required details, and relationships. Preserve the existing design unless replacement was requested. Keep precise user dimensions unchanged. For unspecified independent elements, choose varied but coherent values using distinct deterministic sub-seeds derived from this request seed and each stable element name; keep the chosen values in the plan so a refinement can reproduce them. Check the whole design against the user's constraints after choosing element-level variation.

Before calling a result realistic or complete, review the available render and host measurements with this checklist. Answer yes/no with one short reason for each failure, or say unverified when the available view/tools cannot establish it:
1. Are visible proportions plausible for the requested scale? For full-scale residential work only, use defaults of door width 700–1200 mm, ceiling clear height 2300–3500 mm, residential wall thickness 100–400 mm, window sill 800–1100 mm above its floor, and pitched roof 15–45 degrees. Honor explicit dimensions, scale models, style exceptions, and symmetry requests.
2. Does the render show unsupported overhangs or floating geometry? Say unverified if the view cannot establish support.
3. Is there at least one intentional non-repeating/asymmetric element, unless strict symmetry was requested?
4. Does the design show the defining features of the requested style?
5. Do host measurements report zero-volume intended solids or geometry errors? Do not claim self-intersection validation unless a host check explicitly reports it.

Fix only concrete failures supported by the evidence, in one coherent batch where practical. Keep exact operations such as a specified cylinder or extrusion on the normal single-pass path; do not add planning or critique calls for them."#
    ))
}

fn stable_seed(request: &str) -> u64 {
    // FNV-1a is stable across platforms and process launches.
    request.bytes().fold(0xcbf29ce484222325, |hash, byte| (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3))
}
