"""
Macenko (2009) stain normalization for Glioma-SPARSE, organised as named,
dataset-specific PROFILES rather than a single on/off switch.

Why profiles instead of a boolean: different external-validation sources need
different corrections. Radboud and TCGA were each found to carry a different
degree of stain/scanner shift relative to the EBRAINS training data, so
normalizing everything toward one fixed reference doesn't make sense once more
than one external dataset is in play. A profile bundles the Stage-A (thumbnail)
and Stage-B (high-res patch) references fitted for ONE dataset, under one name.

Profiles live as JSON files in preprocessing/stain_profiles/, one file per
profile (id == filename stem). Built-in profiles shipped with this repo:
    RD-mrxs      -- Radboud external-validation set (.mrxs source slides)
    TCGA-Glioma  -- TCGA external-validation set

Usage (apply an existing profile):
    from glioma_sparse.preprocessing.stain_normalization import normalize_with_profile
    normalized_rgb = normalize_with_profile(rgb_uint8_array, "RD-mrxs", stage="A")

Usage (fit a new profile from a folder of images, e.g. a new site's data):
    from glioma_sparse.preprocessing.stain_normalization import fit_and_save_profile
    fit_and_save_profile(
        profile_id="MySite-2026", stageA_image_paths=[...], stageB_image_paths=[...],
        description="Whatever the user wants to remember about this dataset.")

No retraining is required for any of this -- it is a preprocessing-only correction
applied at thumbnail/patch-extraction time.
"""
import json
from pathlib import Path

import numpy as np

PROFILES_DIR = Path(__file__).with_name("stain_profiles")
_OD_THRESHOLD = 0.15   # optical density threshold below which a pixel is treated as background
_ANGLE_PERCENTILE = 1.0  # robust percentile for extreme stain angles (1st / 99th)

_profile_cache = {}


# ============================================================
# CORE MACENKO (per-image)
# ============================================================

def _rgb_to_od(rgb):
    rgb = rgb.astype(np.float64)
    return -np.log((rgb + 1.0) / 256.0)


def _od_to_rgb(od):
    rgb = 256.0 * np.exp(-od) - 1.0
    return np.clip(rgb, 0, 255).astype(np.uint8)


def get_stain_matrix(rgb, od_threshold=_OD_THRESHOLD, angle_percentile=_ANGLE_PERCENTILE):
    """
    Macenko stain separation for a single RGB uint8 image.
    Returns (stain_matrix [3x2, columns hematoxylin then eosin], max_concentrations [2,]),
    or None if the image has too little tissue (mostly background) to fit reliably.
    """
    od = _rgb_to_od(rgb).reshape(-1, 3)
    tissue = od[np.all(od > od_threshold, axis=1)]
    if tissue.shape[0] < 100:
        return None

    cov = np.cov(tissue, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    top2 = eigvecs[:, np.argsort(eigvals)[-2:]]

    proj = tissue @ top2
    angles = np.arctan2(proj[:, 1], proj[:, 0])
    min_angle = np.percentile(angles, angle_percentile)
    max_angle = np.percentile(angles, 100 - angle_percentile)

    v_min = top2 @ np.array([np.cos(min_angle), np.sin(min_angle)])
    v_max = top2 @ np.array([np.cos(max_angle), np.sin(max_angle)])

    # Order so column 0 = hematoxylin, column 1 = eosin (H has the smaller red-channel OD).
    if v_min[0] > v_max[0]:
        stain_matrix = np.stack([v_max, v_min], axis=1)
    else:
        stain_matrix = np.stack([v_min, v_max], axis=1)
    stain_matrix = stain_matrix / np.linalg.norm(stain_matrix, axis=0, keepdims=True)

    conc, *_ = np.linalg.lstsq(stain_matrix, od.T, rcond=None)
    max_conc = np.clip(np.percentile(conc, 99, axis=1), 1e-6, None)
    return stain_matrix, max_conc


def normalize_to_reference(rgb, target_stain_matrix, target_max_conc, od_threshold=_OD_THRESHOLD):
    """
    Re-stain an RGB uint8 image to match a reference stain appearance.
    Returns the input unchanged if stain separation fails (e.g. no detectable tissue),
    so this is always safe to call unconditionally on any thumbnail or patch.
    """
    result = get_stain_matrix(rgb, od_threshold=od_threshold)
    if result is None:
        return rgb
    source_matrix, source_max_conc = result

    od = _rgb_to_od(rgb).reshape(-1, 3)
    conc, *_ = np.linalg.lstsq(source_matrix, od.T, rcond=None)
    conc_norm = conc / source_max_conc[:, None] * target_max_conc[:, None]
    od_new = (np.asarray(target_stain_matrix) @ conc_norm).T
    return _od_to_rgb(od_new).reshape(rgb.shape)


def _average_stain_matrices(matrices):
    """Average a list of (3x2) unit stain matrices (already H/E-ordered) and re-normalise columns."""
    mean = np.stack(matrices, axis=0).mean(axis=0)
    return mean / np.linalg.norm(mean, axis=0, keepdims=True)


# ============================================================
# FITTING (aggregate a reference from many images)
# ============================================================

def _read_rgb(path, max_side=1024):
    """Read an image as RGB uint8, downsampling if larger than max_side on its long edge."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def fit_reference(image_paths, sample_n=300, seed=0, max_side=1024, verbose=False):
    """
    Fit an aggregate (stain_matrix, max_conc) reference from a list of image paths,
    for ONE representation level (Stage-A thumbnails, or Stage-B patches -- call
    this separately for each). Returns a dict with the reference plus fit diagnostics
    (n_used/n_failed), so a caller can judge whether the fit is trustworthy.
    """
    rng = np.random.default_rng(seed)
    image_paths = list(image_paths)
    if sample_n is not None and len(image_paths) > sample_n:
        idx = rng.choice(len(image_paths), size=sample_n, replace=False)
        image_paths = [image_paths[i] for i in idx]

    matrices, max_concs = [], []
    n_failed = 0
    for i, p in enumerate(image_paths):
        try:
            rgb = _read_rgb(p, max_side=max_side)
            result = get_stain_matrix(rgb)
        except Exception:
            result = None
        if result is None:
            n_failed += 1
            continue
        sm, mc = result
        matrices.append(sm)
        max_concs.append(mc)
        if verbose and (i + 1) % 50 == 0:
            print(f"  ...fit {i+1}/{len(image_paths)} ({n_failed} failed so far)")

    if not matrices:
        raise RuntimeError("No usable images (with enough detected tissue) to fit a reference from.")

    stain_matrix = _average_stain_matrices(matrices)
    max_conc = np.median(np.stack(max_concs, axis=0), axis=0)
    return {
        "stain_matrix": stain_matrix.tolist(),
        "max_conc": max_conc.tolist(),
        "n_used": len(matrices),
        "n_failed": n_failed,
        "n_total_considered": len(image_paths),
    }


def fit_reference_stratified(class_to_paths, min_n=10, sample_n_per_class=100, seed=0, verbose=False):
    """
    Class-balanced version of fit_reference(): fit a SEPARATE reference per
    class, then average the per-class references with EQUAL weight (plain
    mean across classes, not weighted by how many images each class has).

    Without this, a plain pooled fit_reference() over a class-imbalanced
    cohort silently weights the result by whatever happens to be the largest
    class -- in this cohort GBM/IDH-wildtype is ~40% of Stage-A slides and
    the single biggest share of Stage-B patches, so an unstratified fit is a
    "what does GBM tissue look like" reference wearing a "what does this
    cohort look like" label. Classes below min_n are skipped (too few images
    to fit a stable per-class reference) and reported separately so a caller
    can see what was excluded rather than have it silently dropped.

    Args:
        class_to_paths: dict {class_name: [image_paths]}
        min_n: classes with fewer than this many images are excluded
        sample_n_per_class: cap on how many images to sample per class
        seed: RNG seed, shared across classes for reproducibility

    Returns dict with the same shape as fit_reference() (stain_matrix,
    max_conc, n_used, n_failed, n_total_considered summed/averaged across
    included classes) plus:
        per_class: {class_name: {stain_matrix, max_conc, n_used, n_failed, n_total_considered}}
        classes_included: [class_name, ...]
        classes_excluded_too_few: {class_name: n_available}
    """
    per_class = {}
    excluded = {}
    for cls, paths in class_to_paths.items():
        paths = list(paths)
        if len(paths) < min_n:
            excluded[cls] = len(paths)
            continue
        per_class[cls] = fit_reference(paths, sample_n=sample_n_per_class, seed=seed, verbose=verbose)

    if not per_class:
        raise RuntimeError(f"No class had >= min_n={min_n} images to fit a stratified reference from.")

    matrices = [np.array(r["stain_matrix"]) for r in per_class.values()]
    max_concs = [np.array(r["max_conc"]) for r in per_class.values()]
    stain_matrix = _average_stain_matrices(matrices)          # equal weight per class
    max_conc = np.mean(np.stack(max_concs, axis=0), axis=0)   # equal weight per class

    return {
        "stain_matrix": stain_matrix.tolist(),
        "max_conc": max_conc.tolist(),
        "n_used": sum(r["n_used"] for r in per_class.values()),
        "n_failed": sum(r["n_failed"] for r in per_class.values()),
        "n_total_considered": sum(r["n_total_considered"] for r in per_class.values()),
        "per_class": per_class,
        "classes_included": sorted(per_class.keys()),
        "classes_excluded_too_few": excluded,
    }


# ============================================================
# NAMED PROFILES (registry: discover / load / apply / save)
# ============================================================

def list_profiles():
    """List available profiles as [{id, label, description, ...}], sorted by id.
    Does not include the per-stage numeric references (call load_profile for those)."""
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for p in sorted(PROFILES_DIR.glob("*.json")):
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        out.append({
            "id": data.get("id", p.stem),
            "label": data.get("label", p.stem),
            "description": data.get("description", ""),
            "has_stageA": "stageA" in data,
            "has_stageB": "stageB" in data,
        })
    return out


def load_profile(profile_id):
    """Load a full profile dict by id (filename stem), cached by id."""
    if profile_id not in _profile_cache:
        path = PROFILES_DIR / f"{profile_id}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"No stain profile named '{profile_id}' in {PROFILES_DIR}. "
                f"Available: {[p['id'] for p in list_profiles()]}")
        _profile_cache[profile_id] = json.loads(path.read_text())
    return _profile_cache[profile_id]


def get_stage_reference(profile_id, stage):
    """(stain_matrix, max_conc) numpy arrays for one stage ('A' or 'B') of a profile."""
    stage_key = f"stage{stage.upper()}"
    profile = load_profile(profile_id)
    if stage_key not in profile:
        raise KeyError(f"Profile '{profile_id}' has no {stage_key} reference fitted.")
    ref = profile[stage_key]
    return np.array(ref["stain_matrix"], dtype=np.float64), np.array(ref["max_conc"], dtype=np.float64)


def normalize_with_profile(rgb, profile_id, stage):
    """Normalize an RGB uint8 image against a named profile's reference for the
    given stage ('A' or 'B'). `profile_id=None` is a deliberate no-op passthrough,
    so call sites can do `normalize_with_profile(img, req.stain_profile, "A")`
    without a separate branch for "no normalization requested"."""
    if not profile_id:
        return rgb
    stain_matrix, max_conc = get_stage_reference(profile_id, stage)
    return normalize_to_reference(rgb, stain_matrix, max_conc)


def save_profile(profile_id, label=None, description="", stageA=None, stageB=None,
                 fitted_from_stageA=None, fitted_from_stageB=None, validated_against=None):
    """Write (or overwrite) a profile file. stageA/stageB are the dicts returned
    by fit_reference() (at least one must be given). Returns the written path."""
    if stageA is None and stageB is None:
        raise ValueError("Need at least one of stageA / stageB to save a profile.")
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    data = {"id": profile_id, "label": label or profile_id, "description": description}
    if stageA is not None:
        data["stageA"] = stageA
        if fitted_from_stageA:
            data["fitted_from_stageA"] = fitted_from_stageA
    if stageB is not None:
        data["stageB"] = stageB
        if fitted_from_stageB:
            data["fitted_from_stageB"] = fitted_from_stageB
    if validated_against:
        data["validated_against"] = validated_against
    path = PROFILES_DIR / f"{profile_id}.json"
    path.write_text(json.dumps(data, indent=2))
    _profile_cache.pop(profile_id, None)
    return path


def fit_and_save_profile(profile_id, stageA_image_paths=None, stageB_image_paths=None,
                         label=None, description="", sample_n=300, seed=0, verbose=False):
    """
    End-to-end: fit a new profile from image paths and save it under PROFILES_DIR.
    This is what backs "point at a folder of your own data and give it a name" --
    at least one of stageA_image_paths / stageB_image_paths must be provided; a
    caller pointed at raw WSI files (not pre-made thumbnails/patches) should first
    generate thumbnails via preprocessing.thumbnail_auto.create_wsi_thumbnail_auto
    (with stain_profile=None) and pass those paths in here.
    """
    stageA = (fit_reference(stageA_image_paths, sample_n=sample_n, seed=seed, verbose=verbose)
             if stageA_image_paths else None)
    stageB = (fit_reference(stageB_image_paths, sample_n=sample_n, seed=seed, verbose=verbose)
             if stageB_image_paths else None)
    return save_profile(profile_id, label=label, description=description,
                        stageA=stageA, stageB=stageB)
