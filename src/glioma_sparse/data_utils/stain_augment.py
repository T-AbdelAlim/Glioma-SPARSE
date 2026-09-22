"""
Training-time stain/colour augmentation, on top of (not instead of) the
preprocessing-time Macenko normalization in preprocessing/stain_normalization.py.
Reuses its stain separation (get_stain_matrix, OD<->RGB helpers) directly.
"""

import random

import numpy as np
from PIL import Image
from torchvision import transforms

from glioma_sparse.preprocessing.stain_normalization import (
    get_stain_matrix, _rgb_to_od, _od_to_rgb,
)


class RandomStainJitter:
    """Randomly rescale/shift H&E stain concentrations in OD space (Tellez et
    al. 2018-style), then reconstruct. Falls back to input unchanged if stain
    separation fails."""

    def __init__(self, alpha_range=(0.85, 1.15), beta_range=(-0.05, 0.05),
                 p=0.5, od_threshold=0.15):
        self.alpha_range = alpha_range
        self.beta_range = beta_range
        self.p = p
        self.od_threshold = od_threshold

    def __call__(self, img):
        if random.random() > self.p:
            return img
        if not isinstance(img, Image.Image):
            raise TypeError("RandomStainJitter expects a PIL Image")

        rgb = np.array(img)
        result = get_stain_matrix(rgb, od_threshold=self.od_threshold)
        if result is None:
            return img
        stain_matrix, max_conc = result

        od = _rgb_to_od(rgb).reshape(-1, 3)
        conc, *_ = np.linalg.lstsq(stain_matrix, od.T, rcond=None)  # (2, N)

        alpha = np.random.uniform(self.alpha_range[0], self.alpha_range[1], size=(2, 1))
        beta = np.random.uniform(self.beta_range[0], self.beta_range[1], size=(2, 1))
        conc_aug = conc * alpha + beta * max_conc[:, None]

        od_new = (np.asarray(stain_matrix) @ conc_aug).T
        rgb_new = _od_to_rgb(od_new).reshape(rgb.shape)
        return Image.fromarray(rgb_new)


def build_color_jitter(strength="mild"):
    """torchvision ColorJitter with H&E-appropriate defaults."""
    presets = {
        "mild": dict(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        "strong": dict(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.04),
    }
    if strength not in presets:
        raise ValueError(f"Unknown strength {strength!r}, expected one of {list(presets)}")
    return transforms.ColorJitter(**presets[strength])
