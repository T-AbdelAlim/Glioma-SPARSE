"""
Parameter counts and inference FLOPs of every component, and per-slide cost of each pipeline.

Measured, not quoted: FLOPs with torch.utils.flop_counter (counts 2 x MACs), parameters by
summing tensors. UNI2-h is instantiated from its published architecture (timm ViT, embed 1536,
depth 24, 24 heads, SwiGLU MLP, 8 register tokens, patch 14, 224 x 224 input) with random weights:
parameter count and FLOPs do not depend on the weights, so the gated weights are not needed.
CLAM-SB (gated attention, 1536 -> 512 -> 256 attention, 1 + n_classes heads) is instantiated from the
CLAM repository's model definition when available, otherwise counted analytically.

Tile counts are read from the TRIDENT feature files (number of 256 px 20x tissue tiles per slide),
for the 190 unique test slides of the five folds.

Outputs: results/master_comparison/compute_components.csv and compute_tiles.csv.
"""

import glob
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.flop_counter import FlopCounterMode

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "master_comparison"
TRIDENT = Path(r"C:\Users\TAbde\PycharmProjects\brain\mahmoodlab\trident_output")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def n_params(m):
    return sum(p.numel() for p in m.parameters()), sum(p.numel() for p in m.parameters() if p.requires_grad)


def gflops(model, x):
    model.eval()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x)
    return fc.get_total_flops() / 1e9


def latency(model, x, reps=5, half=False):
    model.eval().to(DEV)
    x = x.to(DEV)
    ctx = torch.autocast("cuda", dtype=torch.bfloat16, enabled=half)
    with torch.no_grad(), ctx:
        for _ in range(2):
            model(x)
        torch.cuda.synchronize()
        import time
        t = time.time()
        for _ in range(reps):
            model(x)
        torch.cuda.synchronize()
    return (time.time() - t) / reps


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    import timm
    from torchvision import models
    rows = []

    for name, ctor in (("ResNet-18", models.resnet18), ("ResNet-50", models.resnet50)):
        m = ctor(weights=None)
        m.fc = torch.nn.Linear(m.fc.in_features, 3)
        tot, tr = n_params(m)
        g = gflops(m, torch.randn(1, 3, 2048, 2048))
        lat = latency(m, torch.randn(1, 3, 2048, 2048))
        rows.append({"component": f"{name} (3-class head), 2048x2048 input", "params_total": tot,
                     "params_trainable_when_trained": tr, "gflops_per_forward": g,
                     "latency_s_per_forward_fp32_RTX5080": lat})

    uni = timm.create_model("vit_giant_patch14_224", img_size=224, patch_size=14, depth=24, num_heads=24,
                            init_values=1e-5, embed_dim=1536, mlp_ratio=2.66667 * 2, num_classes=0,
                            no_embed_class=True, mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU,
                            reg_tokens=8, dynamic_img_size=True)
    tot, _ = n_params(uni)
    x = torch.randn(1, 3, 224, 224)
    g = gflops(uni, x)
    lat = latency(uni, torch.randn(64, 3, 224, 224), reps=3) / 64
    lat16 = latency(uni, torch.randn(64, 3, 224, 224), reps=3, half=True) / 64
    rows.append({"component": "UNI2-h ViT (frozen encoder), 224x224 tile", "params_total": tot,
                 "params_trainable_when_trained": 0, "gflops_per_forward": g,
                 "latency_s_per_forward_fp32_RTX5080": lat, "latency_s_per_forward_bf16_RTX5080": lat16})
    uni_g = g

    # logistic regression head on 1536-d embedding, 3 classes
    rows.append({"component": "Logistic-regression subtype head (1536 -> 3)", "params_total": 1536 * 3 + 3,
                 "params_trainable_when_trained": 1536 * 3 + 3, "gflops_per_forward": 2 * 1536 * 3 / 1e9})

    # CLAM-SB: attention net 1536->512 (+ReLU), gated attention 512->256 (tanh, sigmoid), 256->1;
    # bag classifier 512 -> 6; instance classifiers 6 x (512 -> 2) for the CLAM objective
    fc1 = 1536 * 512 + 512
    att = 2 * (512 * 256 + 256) + (256 * 1 + 1)
    cls = 512 * 6 + 6
    inst = 6 * (512 * 2 + 2)
    clam_total = fc1 + att + cls + inst
    macs_per_tile = 1536 * 512 + 2 * 512 * 256 + 256
    rows.append({"component": "CLAM-SB head (gated attention MIL, 6 classes)", "params_total": clam_total,
                 "params_trainable_when_trained": clam_total,
                 "gflops_per_forward": 2 * macs_per_tile / 1e9, "note": "per tile; instance heads included in params"})
    pd.DataFrame(rows).to_csv(OUT / "compute_components.csv", index=False)
    print(pd.DataFrame(rows).to_string())

    # tile counts of the 190 unique test slides
    e = pd.read_csv(REPO / "results/e2e_multi_rn18_A0/per_slide.csv")
    ids = sorted(e.slide_id.unique())
    idx = {p.stem: p for p in TRIDENT.glob("*/20x_256px_0px_overlap/features_uni_v2/*.h5")}
    n = []
    for s in ids:
        with h5py.File(idx[s]) as h:
            n.append(int(h["coords"].shape[0]))
    n = np.array(n)
    tiles = pd.DataFrame({"slide_id": ids, "n_tissue_tiles_20x": n})
    tiles.to_csv(OUT / "compute_tiles.csv", index=False)
    print(f"tissue tiles per slide (190 test slides): mean {n.mean():.0f}, median {np.median(n):.0f}, "
          f"min {n.min()}, max {n.max()}")
    (OUT / "compute_summary.txt").write_text(
        f"uni_gflops_per_tile={uni_g}\ntiles_mean={n.mean()}\ntiles_median={np.median(n)}\n")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
