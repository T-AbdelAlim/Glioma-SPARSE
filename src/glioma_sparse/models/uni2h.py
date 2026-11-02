"""
UNI2-h encoder for Stage B (full version of Glioma-SPARSE only).

UNI2-h (ViT-H/14, 681M parameters, 1536-d) is the encoder of the main analysis of the manuscript. It is a
gated model (https://huggingface.co/MahmoodLab/UNI2-h) and its weights are not redistributed with this
repository. Point GLIOMA_SPARSE_UNI2H_CKPT at a local `pytorch_model.bin` (downloaded after requesting
access), or place it in one of the locations below; without it, `available()` returns False and the
dashboard offers to locate the file.

When this module is present, `glioma_sparse.models.resuni` registers it as the Stage-B encoder "uni2h",
whose heads are the configs without a suffix (rn18_f1, rn50_f1, rn50_auc).
"""

from pathlib import Path
import os

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]

USER_CKPT = None          # path saved by the dashboard ("Locate UNI2-h weights"), see backend.py
EXTRA_CANDIDATES = []     # extra locations, e.g. a models/UNI2-h folder next to the packaged exe

_model = None


def _hf_cache_candidates():
    """pytorch_model.bin of MahmoodLab/UNI2-h in the Hugging Face hub cache, if it was downloaded there."""
    roots = [os.environ.get("HF_HUB_CACHE"),
             str(Path(os.environ["HF_HOME"]) / "hub") if os.environ.get("HF_HOME") else None,
             str(Path.home() / ".cache" / "huggingface" / "hub")]
    found = []
    for r in filter(None, roots):
        found += sorted(Path(r).glob("models--MahmoodLab--UNI2-h/snapshots/*/pytorch_model.bin"))
    return found


def checkpoint_path():
    p = os.environ.get("GLIOMA_SPARSE_UNI2H_CKPT")
    if p and Path(p).exists():
        return Path(p)
    # fallback locations, checked in order, so UNI2-h works without setting the env var:
    # 1) the file the user located in the dashboard (saved in their settings);
    # 2) extra locations set by the caller (models/UNI2-h next to the packaged exe);
    # 3) inside this repo, if the user drops the checkpoint there;
    # 4) a sibling mahmoodlab/models folder, if this machine already set TRIDENT up;
    # 5) the Hugging Face hub cache.
    for candidate in ([Path(USER_CKPT)] if USER_CKPT else []) + list(EXTRA_CANDIDATES) + [
        REPO_ROOT / "models" / "UNI2-h" / "pytorch_model.bin",
        REPO_ROOT.parent / "mahmoodlab" / "models" / "UNI2-h" / "pytorch_model.bin",
    ] + _hf_cache_candidates():
        if candidate.exists():
            return candidate
    return None


def check_checkpoint(path):
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


def reset():
    """Forget the loaded encoder, e.g. after the user located a different checkpoint."""
    global _model
    _model = None


def available():
    try:
        import timm  # noqa: F401
    except ImportError:
        return False
    return checkpoint_path() is not None


def load(device):
    """(model, transform) of UNI2-h, loaded once."""
    global _model
    if _model is None:
        import timm
        from torchvision import transforms
        ckpt = checkpoint_path()
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
        _model = (model, transforms.Compose([
            transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]))
    return _model


ENCODER = {"label": "UNI2-h", "dim": 1536, "suffix": "", "available": available, "load": load}
