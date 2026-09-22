"""
Extended transform builders for the EBRAINS+TCGA combined training run.
data_utils/transforms.py is untouched; *_tcga_ext.py scripts opt into
stain/colour augmentation explicitly via this module instead.
"""

from torchvision import transforms

from glioma_sparse.data_utils.transforms import (
    IMAGENET_MEAN, IMAGENET_STD, build_eval_transform,
)
from glioma_sparse.data_utils.stain_augment import RandomStainJitter, build_color_jitter


def build_train_transform_ext(use_stain_jitter=True, use_color_jitter=True,
                               stain_jitter_kwargs=None,
                               color_jitter_strength="mild"):
    """Same normalize-to-ImageNet-stats tail as build_train_transform(), with
    optional stain/colour augmentation applied to the PIL image beforehand.
    Both default on; set either flag False to isolate the other's effect."""
    ops = []
    if use_stain_jitter:
        ops.append(RandomStainJitter(**(stain_jitter_kwargs or {})))
    if use_color_jitter:
        ops.append(build_color_jitter(color_jitter_strength))
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(ops)


# Re-exported for a single import site in the *_tcga_ext.py training scripts;
# identical to transforms.build_eval_transform (no augmentation at eval time).
build_eval_transform_ext = build_eval_transform
