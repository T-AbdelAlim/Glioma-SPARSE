from pathlib import Path
import numpy as np
import openslide
from PIL import Image


# ============================================================
# DEFAULTS
# ============================================================

DEFAULT_TARGET_MPP = 4.0
DEFAULT_OUTPUT_SIZE = 2048
SUPPORTED_EXTS = (".svs", ".ndpi", ".tif", ".tiff", ".mrxs")


# ============================================================
# UTILITIES
# ============================================================

def safe_float(v):
    try:
        return float(str(v).strip())
    except:
        return None


def get_base_mpp(props):
    mpp_x = safe_float(props.get("openslide.mpp-x"))
    mpp_y = safe_float(props.get("openslide.mpp-y"))

    if mpp_x and mpp_y:
        return 0.5 * (mpp_x + mpp_y)

    aperio = safe_float(props.get("aperio.MPP"))
    if aperio:
        return aperio

    xres = safe_float(props.get("hamamatsu.XResolution"))
    if xres and xres > 0:
        return 10000.0 / xres

    mag = safe_float(props.get("openslide.objective-power"))
    if mag:
        return 10.0 / mag

    return None


def pick_level(slide, target_downsample):
    downsamples = slide.level_downsamples
    candidates = [i for i, d in enumerate(downsamples) if d <= target_downsample]
    return max(candidates) if candidates else 0


# ============================================================
# TISSUE MASK
# ============================================================

def create_tissue_mask(img):
    """
    Create binary tissue mask from RGB image.
    Uses brightness + saturation heuristic.
    """

    img_np = np.array(img).astype(np.float32) / 255.0

    maxc = np.max(img_np, axis=2)
    minc = np.min(img_np, axis=2)

    saturation = (maxc - minc) / (maxc + 1e-8)
    value = maxc

    tissue_mask = (value < 0.9) & (saturation > 0.05)

    return tissue_mask


def estimate_tissue_fraction(mask):
    return mask.sum() / mask.size


# ============================================================
# IMAGE POSTPROCESSING
# ============================================================

def pad_with_background_color(img):
    img_np = np.array(img)

    corners = np.concatenate([
        img_np[0:50, 0:50],
        img_np[0:50, -50:],
        img_np[-50:, 0:50],
        img_np[-50:, -50:]
    ], axis=0).reshape(-1, 3)

    valid = corners[np.any(corners > 10, axis=1)]

    if len(valid) > 0:
        bg_color = np.median(valid, axis=0).astype(np.uint8)
    else:
        bg_color = np.array([255, 255, 255], dtype=np.uint8)

    black_mask = np.all(img_np < 10, axis=-1)
    img_np[black_mask] = bg_color

    img = Image.fromarray(img_np)

    w, h = img.size
    size = max(w, h)

    padded = Image.new("RGB", (size, size), tuple(bg_color))
    padded.paste(img, ((size - w) // 2, (size - h) // 2))

    return padded


# ============================================================
# MAIN FUNCTION
# ============================================================

def create_wsi_thumbnail(
    slide_path,
    output_path=None,
    target_mpp=DEFAULT_TARGET_MPP,
    output_size=DEFAULT_OUTPUT_SIZE,
    tissue_threshold=None,
    save_mask=False,
):
    slide_path = Path(slide_path)

    if not slide_path.exists():
        raise FileNotFoundError(slide_path)

    slide = openslide.OpenSlide(str(slide_path))
    props = slide.properties
    base_mpp = get_base_mpp(props)

    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    if base_mpp is None:
        level = min(2, slide.level_count - 1)
        img = slide.read_region((0, 0), level, slide.level_dimensions[level])
    else:
        target_ds = target_mpp / base_mpp
        level = pick_level(slide, target_ds)

        ds = slide.level_downsamples[level]
        scale = target_ds / ds

        dims = slide.level_dimensions[level]
        img = slide.read_region((0, 0), level, dims)

        if abs(scale - 1.0) > 0.01:
            new_w = int(dims[0] * scale)
            new_h = int(dims[1] * scale)
            img = img.resize((new_w, new_h), Image.BICUBIC)

    img = img.convert("RGB")

    # --------------------------------------------------------
    # TISSUE MASK (BEFORE PADDING)
    # --------------------------------------------------------

    tissue_mask = create_tissue_mask(img)
    tissue_fraction = estimate_tissue_fraction(tissue_mask)

    if tissue_threshold is not None:
        if tissue_fraction < tissue_threshold:
            slide.close()
            return None, tissue_fraction

    # --------------------------------------------------------
    # POSTPROCESSING
    # --------------------------------------------------------

    img = pad_with_background_color(img)
    img = img.resize((output_size, output_size), Image.BICUBIC)

    # --------------------------------------------------------
    # SAVE OUTPUT
    # --------------------------------------------------------

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        img.save(output_path, quality=90)

        if save_mask:
            mask_img = Image.fromarray((tissue_mask * 255).astype(np.uint8))
            mask_img = mask_img.resize((output_size, output_size), Image.NEAREST)
            mask_img.save(output_path.with_name("{}_mask.png".format(output_path.stem)))

    slide.close()

    return img, tissue_fraction