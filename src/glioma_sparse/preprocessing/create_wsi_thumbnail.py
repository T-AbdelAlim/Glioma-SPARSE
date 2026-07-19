from pathlib import Path
import json
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


def parse_hex_color(s):
    if not s:
        return None
    s = str(s).strip().lstrip("#")
    if len(s) != 6:
        return None
    try:
        return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
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


def get_background_color_from_props(props):
    for key in ("openslide.background-color",
                "aperio.BackgroundColor",
                "hamamatsu.BackgroundColor"):
        rgb = parse_hex_color(props.get(key))
        if rgb is not None:
            return rgb
    return None


def pick_level(slide, target_downsample):
    downsamples = slide.level_downsamples
    candidates = [i for i, d in enumerate(downsamples) if d <= target_downsample]
    return max(candidates) if candidates else 0


# ============================================================
# TISSUE MASK
# ============================================================

def create_tissue_mask(img):
    img_np = np.array(img).astype(np.float32) / 255.0

    maxc = np.max(img_np, axis=2)
    minc = np.min(img_np, axis=2)

    saturation = (maxc - minc) / (maxc + 1e-8)
    value = maxc

    tissue_mask = (value < 0.9) & (saturation > 0.05)

    return tissue_mask


def estimate_fraction(mask):
    return mask.sum() / mask.size


# ============================================================
# IMAGE POSTPROCESSING
# ============================================================

def pad_with_background_color(img, bg_color_hint=None):
    img_np = np.array(img)

    if bg_color_hint is not None:
        bg_color = np.array(bg_color_hint, dtype=np.uint8)
    else:
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

    padding_mask = np.all(img_np == 0, axis=-1)
    img_np[padding_mask] = bg_color

    img = Image.fromarray(img_np)

    w, h = img.size
    size = max(w, h)

    padded = Image.new("RGB", (size, size), tuple(bg_color))
    padded.paste(img, ((size - w) // 2, (size - h) // 2))

    return padded


# ============================================================
# MAPPING SIDECAR
# ============================================================

def write_mapping_sidecar(output_jpg_path, payload):
    sidecar_path = Path(output_jpg_path).with_suffix(".json")
    with open(sidecar_path, "w") as f:
        json.dump(payload, f, indent=2)


# ============================================================
# MAIN FUNCTION
# ============================================================

def create_wsi_thumbnail(
    slide_path,
    output_path=None,
    target_mpp=DEFAULT_TARGET_MPP,
    output_size=DEFAULT_OUTPUT_SIZE,
    save_mask=False,
):

    slide_path = Path(slide_path)

    if not slide_path.exists():
        raise FileNotFoundError(slide_path)

    slide = openslide.OpenSlide(str(slide_path))
    props = slide.properties
    base_mpp = get_base_mpp(props)
    bg_hint = get_background_color_from_props(props)

    wsi_level0_dim = list(slide.level_dimensions[0])

    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    if base_mpp is None:
        # No MPP available; read a coarse level as-is. Mapping back to the WSI
        # is not possible without an MPP, so the sidecar will record this.
        level = min(2, slide.level_count - 1)
        img = slide.read_region((0, 0), level, slide.level_dimensions[level])
        recorded_target_mpp = None
    else:
        target_ds = target_mpp / base_mpp
        level = pick_level(slide, target_ds)

        ds = slide.level_downsamples[level]
        scale = target_ds / ds   # >= 1: extra downsample needed beyond chosen level

        dims = slide.level_dimensions[level]
        img = slide.read_region((0, 0), level, dims)

        if abs(scale - 1.0) > 0.01:
            new_w = int(dims[0] / scale)
            new_h = int(dims[1] / scale)
            img = img.resize((new_w, new_h), Image.BICUBIC)

        recorded_target_mpp = float(target_mpp)

    img = img.convert("RGB")

    tissue_w, tissue_h = img.size
    canvas_size = max(tissue_w, tissue_h)
    tissue_offset = ((canvas_size - tissue_w) // 2, (canvas_size - tissue_h) // 2)

    # --------------------------------------------------------
    # TISSUE FRACTION (BEFORE PADDING)
    # --------------------------------------------------------

    mask_before = create_tissue_mask(img)

    tissue_fraction = estimate_fraction(mask_before)

    # --------------------------------------------------------
    # POSTPROCESSING
    # --------------------------------------------------------

    img = pad_with_background_color(img, bg_color_hint=bg_hint)
    img = img.resize((output_size, output_size), Image.BICUBIC)

    # --------------------------------------------------------
    # EFFECTIVE FRACTION (AFTER RESIZE)
    # --------------------------------------------------------

    mask_after = create_tissue_mask(img)
    effective_fraction = estimate_fraction(mask_after)

    # --------------------------------------------------------
    # SAVE OUTPUT (+ MAPPING SIDECAR)
    # --------------------------------------------------------

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        img.save(output_path, quality=90)

        if save_mask:
            mask_img = Image.fromarray((mask_after * 255).astype(np.uint8))
            mask_img.save(output_path.with_name("{}_mask.png".format(output_path.stem)))

        # Mapping sidecar: enough information to convert a thumbnail-pixel
        # bounding box back to WSI level-0 pixel coordinates without re-opening
        # the slide. See glioma_sparse.interpret.wsi_mapping.
        sidecar = {
            "wsi_path": str(slide_path.resolve()),
            "wsi_level0_dim": wsi_level0_dim,
            "base_mpp": float(base_mpp) if base_mpp is not None else None,
            "target_mpp": recorded_target_mpp,
            "tissue_image_dim": [int(tissue_w), int(tissue_h)],
            "canvas_size": int(canvas_size),
            "tissue_offset_in_canvas": [int(tissue_offset[0]), int(tissue_offset[1])],
            "thumbnail_size": int(output_size),
        }
        write_mapping_sidecar(output_path, sidecar)

    slide.close()

    return img, tissue_fraction, effective_fraction
