"""
ResUNI: Stage A's region selection + a frozen UNI2-h foundation-model encoder + a tiny
per-fold logistic-regression head, as an alternative to the ResNet Stage B subtype classifier.

Recipe (must match scripts/analysis/resuni_predictions.py, which produced the trained heads):
  1. Take the level-0 bounding boxes of the regions Stage A selected (up to 4).
  2. Tile each region on a non-overlapping grid of 256 px tiles at a 20x-equivalent
     resolution (0.5 micron/px), matching TRIDENT's own tiling convention.
  3. Keep only tissue tiles (same saturation/brightness heuristic as the app's Stage-A
     grid-cell tissue filter), then randomly draw up to N_TILES of them.
  4. Encode each tile with the frozen UNI2-h ViT (1536-d output), mean-pool -> one slide
     embedding.
  5. Apply all 5 EBRAINS fold heads (StandardScaler + multinomial logistic regression,
     trained with n=64 tiles/slide) and average their class probabilities (the same
     "5-model ensemble" used for external validation, appropriate for a genuinely new
     slide that cannot be identified with any one training fold).

UNI2-h is a gated model (https://huggingface.co/MahmoodLab/UNI2-h) and its weights are not
redistributed with this repository. Point GLIOMA_SPARSE_UNI2H_CKPT at a local
`pytorch_model.bin` (downloaded after requesting access) to enable ResUNI; without it,
`uni2h_available()` returns False and the dashboard hides the option.

DistillPath-KS16-UNI2h (ViT-S/16, 22M parameters, 384-d, distilled from UNI2-h) is the alternative
encoder of the manuscript (Table 3). Same tiles; its own input pipeline (224 px, mean = std = 0.5,
as TRIDENT's kaiko transform used for the paper's features) and its own heads
(<config>_dp_uni2h, copied from results/resuni/rn50_f1_dp_uni2h and rn18_f1_dp_uni2h_full). Its
weights are public but under a non-commercial licence, so they are not bundled either: the file is
found locally (GLIOMA_SPARSE_DISTILLPATH_CKPT, models/DistillPath-KS16-UNI2h, the Hugging Face
cache) or downloaded from the hub on first use.
"""

from pathlib import Path
import os

import numpy as np
from PIL import Image
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
HEADS_DIR = REPO_ROOT / "data" / "resuni_heads"
SUB = ["IDH_mt", "IDH_mt_1p19q", "IDH_wt"]
N_TILES = 64
TILE_PX = 256              # UNI2-h input tile size (before the encoder's own 224 resize)
TILE_MPP_20X = 0.5         # "20x-equivalent" resolution, matching TRIDENT's own convention
MIN_TISSUE = 0.25
SEED = 42
MAX_CANDIDATE_READS = 1200  # safety cap so a very large region set stays interactive

# (Stage-A arch, Stage-A metric) -> config name, i.e. which trained ResUNI head applies.
# Only these three combinations were validated (see docs/experiments/MASTER_COMPARISON.md);
# resnet18/auc has no trained ResUNI head.
CONFIG_FOR_STAGE_A = {
    ("resnet18", "f1"): "rn18_f1",
    ("resnet50", "f1"): "rn50_f1",
    ("resnet50", "auc"): "rn50_auc",
}


# Stage-B encoders; each has its own heads in HEADS_DIR / (config + suffix)
ENCODERS = {
    "uni2h": {"label": "UNI2-h", "dim": 1536, "suffix": ""},
    "distillpath": {"label": "DistillPath", "dim": 384, "suffix": "_dp_uni2h"},
}
DISTILLPATH_HUB = "RamonK/DistillPath-KS16-UNI2h"

USER_UNI_CKPT = None          # path saved by the dashboard ("Locate UNI2-h weights"), see backend.py
EXTRA_UNI_CANDIDATES = []     # extra locations, e.g. a models/UNI2-h folder next to the packaged exe
EXTRA_DP_CANDIDATES = []      # same for DistillPath (models/DistillPath-KS16-UNI2h next to the exe)


def _hf_cache_candidates():
    """pytorch_model.bin of MahmoodLab/UNI2-h in the Hugging Face hub cache, if it was downloaded there."""
    roots = [os.environ.get("HF_HUB_CACHE"),
             str(Path(os.environ["HF_HOME"]) / "hub") if os.environ.get("HF_HOME") else None,
             str(Path.home() / ".cache" / "huggingface" / "hub")]
    found = []
    for r in filter(None, roots):
        found += sorted(Path(r).glob("models--MahmoodLab--UNI2-h/snapshots/*/pytorch_model.bin"))
    return found


def uni_checkpoint_path():
    p = os.environ.get("GLIOMA_SPARSE_UNI2H_CKPT")
    if p and Path(p).exists():
        return Path(p)
    # fallback locations, checked in order, so ResUNI works without setting the env var:
    # 1) the file the user located in the dashboard (saved in their settings);
    # 2) extra locations set by the caller (models/UNI2-h next to the packaged exe);
    # 3) inside this repo, if the user drops the checkpoint there;
    # 4) the sibling mahmoodlab/TRIDENT checkout's own local_ckpts.json location, if this
    #    machine already set TRIDENT up (see docs/experiments/EXTERNAL_VALIDATION.md);
    # 5) the Hugging Face hub cache.
    for candidate in ([Path(USER_UNI_CKPT)] if USER_UNI_CKPT else []) + list(EXTRA_UNI_CANDIDATES) + [
        REPO_ROOT / "models" / "UNI2-h" / "pytorch_model.bin",
        REPO_ROOT.parent / "mahmoodlab" / "models" / "UNI2-h" / "pytorch_model.bin",
    ] + _hf_cache_candidates():
        if candidate.exists():
            return candidate
    return None


def check_uni_checkpoint(path):
    """None if `path` holds the UNI2-h ViT-H/14 weights, else a short reason (memory-mapped, so it is quick)."""
    path = Path(path)
    if not path.is_file():
        return "file not found"
    try:
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except Exception as e:
        return f"not a PyTorch checkpoint ({type(e).__name__})"
    w = state.get("patch_embed.proj.weight") if isinstance(state, dict) else None
    if w is None or tuple(w.shape) != (1536, 3, 14, 14) or "blocks.23.norm1.weight" not in state:
        return "not the UNI2-h checkpoint (expected ViT-H/14 weights, 1536-dim, 24 blocks)"
    return None


def reset_encoder():
    """Forget the loaded encoder, e.g. after the user located a different checkpoint."""
    global _encoder, _encoder_transform
    _encoder, _encoder_transform = None, None


def uni2h_available():
    try:
        import timm  # noqa: F401
    except ImportError:
        return False
    return uni_checkpoint_path() is not None


def distillpath_checkpoint_path():
    """Local model.safetensors of DistillPath-KS16-UNI2h, or None (then it is downloaded on first use)."""
    p = os.environ.get("GLIOMA_SPARSE_DISTILLPATH_CKPT")
    if p and Path(p).exists():
        return Path(p)
    roots = [os.environ.get("HF_HUB_CACHE"),
             str(Path(os.environ["HF_HOME"]) / "hub") if os.environ.get("HF_HOME") else None,
             str(Path.home() / ".cache" / "huggingface" / "hub")]
    cache = [f for r in filter(None, roots)
             for f in sorted(Path(r).glob("models--RamonK--DistillPath-KS16-UNI2h/snapshots/*/model.safetensors"))]
    for candidate in list(EXTRA_DP_CANDIDATES) + [
        REPO_ROOT / "models" / "DistillPath-KS16-UNI2h" / "model.safetensors",
        REPO_ROOT.parent / "mahmoodlab" / "models" / "DistillPath-KS16-UNI2h" / "model.safetensors",
    ] + cache:
        if Path(candidate).exists():
            return Path(candidate)
    return None


def distillpath_available():
    """True if DistillPath can run: weights found locally, or the hub client is installed to fetch them."""
    try:
        import timm  # noqa: F401
    except ImportError:
        return False
    if distillpath_checkpoint_path() is not None:
        return True
    try:
        import huggingface_hub  # noqa: F401
        return True
    except ImportError:
        return False


def encoder_available(encoder="uni2h"):
    return uni2h_available() if encoder == "uni2h" else distillpath_available()


def resuni_config_available(archA, metric, encoder="uni2h"):
    """head config name for this Stage-A choice and encoder, or None if unsupported / heads missing."""
    cfg = CONFIG_FOR_STAGE_A.get((archA, metric))
    if cfg is None:
        return None
    cfg += ENCODERS[encoder]["suffix"]
    if not heads_available(cfg, 64):
        return None
    return cfg


TILE_OPTIONS = (8, 16, 32, 64, 128, 256, 512, 1024, "all")   # "all": every tissue tile in the regions (heads lr_nall)
ALL_TILES_CAP = 4096       # "all" keeps at most this many tiles (a random subset beyond), to bound memory and time


def norm_n(n):
    """Tile count as used in head file names: an int, or "all" (also accepted as 0 / None / "all")."""
    return "all" if n in (0, None, "all") else int(n)


def heads_available(cfg, n):
    """True if the five fold heads trained with n tiles per slide exist for this config."""
    return all((HEADS_DIR / cfg / f"lr_n{n}_fold{k}.npz").exists() for k in range(1, 6))


def tile_options(cfg):
    """Tile counts for which a matching head was trained (heads are fitted per n, see resuni_nsweep.py)."""
    return [n for n in TILE_OPTIONS if heads_available(cfg, n)]


def resuni_available_configs(encoder="uni2h"):
    return {f"{a}:{m}": resuni_config_available(a, m, encoder) is not None for a, m in CONFIG_FOR_STAGE_A}


# ============================================================
# ENCODER (lazy singleton: only loaded the first time ResUNI actually runs)
# ============================================================

_encoder = None
_encoder_transform = None
_heads = {}  # config -> list of 5 dicts {mean, scale, coef, intercept}


def get_uni_encoder(device):
    global _encoder, _encoder_transform
    if _encoder is None:
        import timm
        from torchvision import transforms
        ckpt = uni_checkpoint_path()
        if ckpt is None:
            raise RuntimeError(
                "UNI2-h checkpoint not found. Set GLIOMA_SPARSE_UNI2H_CKPT to a local "
                "pytorch_model.bin (request access at https://huggingface.co/MahmoodLab/UNI2-h).")
        timm_kwargs = dict(
            img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5,
            embed_dim=1536, mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True,
            mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU,
            reg_tokens=8, dynamic_img_size=True)
        model = timm.create_model("vit_giant_patch14_224", pretrained=False, **timm_kwargs)
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        _encoder = model
        _encoder_transform = transforms.Compose([
            transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    return _encoder, _encoder_transform


_dp = None


def get_distillpath_encoder(device):
    """DistillPath-KS16-UNI2h with the input pipeline of the paper's features (TRIDENT kaiko eval transform)."""
    global _dp
    if _dp is None:
        import timm
        from torchvision import transforms
        ckpt = distillpath_checkpoint_path()
        if ckpt is not None:
            from safetensors.torch import load_file
            model = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=0)
            model.load_state_dict(load_file(str(ckpt)), strict=True)
        else:
            try:
                model = timm.create_model(f"hf_hub:{DISTILLPATH_HUB}", pretrained=True, num_classes=0)
            except Exception as e:
                raise RuntimeError(f"DistillPath weights not found locally and the download from "
                                   f"huggingface.co/{DISTILLPATH_HUB} failed ({type(e).__name__}); "
                                   "set GLIOMA_SPARSE_DISTILLPATH_CKPT to a local model.safetensors") from e
        model.to(device).eval()
        _dp = (model, transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]))
    return _dp


def get_encoder(device, encoder="uni2h"):
    return get_uni_encoder(device) if encoder == "uni2h" else get_distillpath_encoder(device)


def get_resuni_heads(config, n=N_TILES):
    """The five fold heads for `config`, fitted on slides embedded with n tiles (default 64)."""
    if (config, n) not in _heads:
        heads = []
        for k in range(1, 6):
            z = np.load(HEADS_DIR / config / f"lr_n{n}_fold{k}.npz")
            heads.append({"mean": z["mean"], "scale": z["scale"], "coef": z["coef"],
                          "intercept": z["intercept"]})
        _heads[(config, n)] = heads
    return _heads[(config, n)]


# ============================================================
# TILE SAMPLING (mirrors TRIDENT's non-overlapping grid + a tissue filter)
# ============================================================

def _tile_size_level0(base_mpp):
    if not base_mpp:
        return 512  # fallback: typical 40x-native slide (matches TRIDENT's own value there)
    return max(1, round(TILE_PX * TILE_MPP_20X / base_mpp))


def _is_tissue(tile_img):
    arr = np.asarray(tile_img).astype(np.float32) / 255.0
    maxc, minc = arr.max(2), arr.min(2)
    sat = (maxc - minc) / (maxc + 1e-8)
    mask = (maxc < 0.9) & (sat > 0.05)
    return float(mask.mean()) >= MIN_TISSUE


def candidate_tile_coords(bboxes, tile_level0):
    """Non-overlapping (x, y) top-left level-0 coordinates of every tile inside `bboxes`."""
    coords = []
    for (bx, by, bw, bh) in bboxes:
        nx, ny = max(1, bw // tile_level0), max(1, bh // tile_level0)
        for j in range(ny):
            for i in range(nx):
                coords.append((bx + i * tile_level0, by + j * tile_level0))
    return coords


def sample_region_tiles(slide, base_mpp, bboxes, n=N_TILES, seed=SEED):
    """Random tissue tiles (PIL images) drawn from inside `bboxes`, plus how many tissue
    tiles were found in total. Reads are capped at MAX_CANDIDATE_READS candidates for
    responsiveness; a random subset of the candidate grid is read first when the grid is
    larger than that, so the result is still an unbiased sample of the region(s).
    n = "all" reads every candidate and keeps every tissue tile, up to ALL_TILES_CAP (the
    candidates are shuffled, so stopping there keeps a random subset)."""
    tile_level0 = _tile_size_level0(base_mpp)
    coords = candidate_tile_coords(bboxes, tile_level0)
    rng = np.random.default_rng(seed)
    if n == "all":
        max_reads, n, stop_at = len(coords), ALL_TILES_CAP, ALL_TILES_CAP
    else:
        max_reads, stop_at = max(MAX_CANDIDATE_READS, 4 * n), 4 * n   # larger counts need a larger read budget
    if len(coords) > max_reads:
        idx = rng.choice(len(coords), size=max_reads, replace=False)
        coords = [coords[i] for i in idx]
    rng.shuffle(coords)

    tissue_tiles = []
    for (x, y) in coords:
        tile = slide.read_region((x, y), 0, (tile_level0, tile_level0)).convert("RGB")
        if tile.size != (TILE_PX, TILE_PX):
            tile = tile.resize((TILE_PX, TILE_PX), Image.BICUBIC)
        if _is_tissue(tile):
            tissue_tiles.append(((x, y, tile_level0, tile_level0), tile))
        if len(tissue_tiles) >= stop_at:   # plenty of candidates; stop reading early
            break

    n_found = len(tissue_tiles)
    if n_found > n:
        keep = rng.choice(n_found, size=n, replace=False)
        tissue_tiles = [tissue_tiles[i] for i in keep]
    return tissue_tiles, n_found


@torch.no_grad()
def embed_tiles(tiles, device, batch_size=32, encoder="uni2h"):
    """[(bbox, PIL image), ...] -> (n, 1536) UNI2-h or (n, 384) DistillPath float32 features."""
    dim = ENCODERS[encoder]["dim"]
    model, transform = get_encoder(device, encoder)
    feats = []
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda" else _nullcontext()
    with autocast:
        for start in range(0, len(tiles), batch_size):
            batch = tiles[start:start + batch_size]
            x = torch.stack([transform(img) for _, img in batch]).to(device)
            out = model(x).float().cpu().numpy()
            feats.append(out)
    return np.concatenate(feats, axis=0) if feats else np.zeros((0, dim), np.float32)


class _nullcontext:
    def __enter__(self): return None
    def __exit__(self, *a): return False


# ============================================================
# PREDICTION
# ============================================================

def predict_from_features(feats, config, n=N_TILES):
    """(n, 1536) tile features -> ensemble subtype probs (5-fold mean) + per-fold probs
    + per-tile contribution to the ensemble's predicted class (mean over folds of each
    fold's standardized linear contribution; the head is linear, so a tile's contribution
    to the SLIDE logit is exactly this, independent of how many tiles were pooled)."""
    heads = get_resuni_heads(config, n)
    mean_feat = feats.mean(0)
    per_fold_probs = []
    for h in heads:
        z = (mean_feat - h["mean"]) / h["scale"]
        logits = h["coef"] @ z + h["intercept"]
        p = np.exp(logits - logits.max()); p /= p.sum()
        per_fold_probs.append(p)
    per_fold_probs = np.stack(per_fold_probs, 0)
    ens = per_fold_probs.mean(0)
    pred_idx = int(np.argmax(ens))

    contrib, slide_mean, bias = margin_contributions(feats, heads, pred_idx)
    return {"ens_probs": ens, "pred_idx": pred_idx, "per_fold_probs": per_fold_probs,
            "contributions": contrib, "slide_mean_contribution": slide_mean, "margin_bias": bias}


def margin_contributions(feats, heads, pred_idx):
    """Exact per-tile evidence for the predicted class, from the tiles that made the prediction.

    For each fold head f (standardisation mean_f/scale_f, weights W_f, bias b_f) the slide-level
    margin of the predicted class p over the other classes is
        m_f = (w_p - mean_{q!=p} w_q) . z_f(mean of tile features) + (b_p - mean_{q!=p} b_q),
    and because the head is linear and the slide embedding is the MEAN of the tile features,
        m_f = mean_k c_fk + bias_f,   c_fk = (w_p - mean_{q!=p} w_q) . z_f(x_k).
    c_fk is therefore the exact share of tile k in fold f's slide-level decision; it is averaged
    over the five fold heads. Positive = pushes towards the predicted class, negative = against.
    Returns (per-tile contributions (n,), their mean (= slide-level evidence), mean bias term)."""
    others = [q for q in range(heads[0]["coef"].shape[0]) if q != pred_idx]
    cs, biases = [], []
    for h in heads:
        w = h["coef"][pred_idx] - h["coef"][others].mean(0)
        z = (feats - h["mean"]) / h["scale"]
        cs.append(z @ w)
        biases.append(float(h["intercept"][pred_idx] - h["intercept"][others].mean()))
    c = np.mean(cs, axis=0)
    return c, float(c.mean()), float(np.mean(biases))


def bin_tile_to_cell(tx, ty, bbox, grid=(8, 8)):
    """Which (row, col) of an 8x8 grid over `bbox` a tile centred at (tx, ty) falls into."""
    wx, wy, ww, wh = bbox
    ch, cw = wh / grid[0], ww / grid[1]
    r = min(grid[0] - 1, max(0, int((ty - wy) // ch)))
    c = min(grid[1] - 1, max(0, int((tx - wx) // cw)))
    return r, c


def region_contribution_grid(slide, base_mpp, bbox, config, pred_idx, grid=(8, 8), n=200, encoder="uni2h"):
    """Dense per-region evidence grid (same shape as the app's classic ResNet occlusion
    grid): resamples `bbox` densely (independent of whatever tiles the slide-level n=64
    draw happened to use) and bins each tile's contribution to `pred_idx` into `grid`.
    Returns None if no tissue tiles were found."""
    tiles, _ = sample_region_tiles(slide, base_mpp, [bbox], n=n)
    if not tiles:
        return None
    feats = embed_tiles(tiles, "cuda" if torch.cuda.is_available() else "cpu", encoder=encoder)
    contrib = contributions_for_class(feats, config, pred_idx)
    imp = np.zeros(grid, np.float32)
    cnt = np.zeros(grid, np.int32)
    for (bbox_t, _tile), v in zip(tiles, contrib):
        tx, ty, tw, th = bbox_t
        r, c = bin_tile_to_cell(tx + tw / 2, ty + th / 2, bbox, grid)
        imp[r, c] += v; cnt[r, c] += 1
    imp = np.divide(imp, np.maximum(cnt, 1), out=np.full(grid, np.nan, np.float32), where=cnt > 0)
    fill = np.nanmean(imp) if np.isfinite(imp).any() else 0.0
    return np.nan_to_num(imp, nan=fill)


def contributions_for_class(feats, config, class_idx):
    """Per-tile contribution to an already-known class (no re-prediction) -- used for the
    dense, per-region interpretability view, where the class is the slide-level ResUNI call
    and `feats` are a denser, region-scoped tile sample than the one used for that call."""
    heads = get_resuni_heads(config)
    mean = np.stack([h["mean"] for h in heads]).mean(0)
    scale = np.stack([h["scale"] for h in heads]).mean(0)
    coef = np.stack([h["coef"][class_idx] for h in heads]).mean(0)
    return ((feats - mean) / scale) @ coef
