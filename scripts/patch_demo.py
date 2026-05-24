from pathlib import Path
from PIL import Image

from glioma_sparse.data_utils.patches import Patch


# ============================================================
# PATHS
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = REPO_ROOT / "scripts" / "output_thumbnails" / "included"
OUTPUT_DIR = REPO_ROOT / "scripts" / "output_patches"


# ============================================================
# CONFIG
# ============================================================

GRID_SIZE = 8
SEED = 42          # set None for random
MAX_IMAGES = 5     # limit for quick testing


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n=== PATCH SHUFFLE DEMO (THUMBNAILS) ===\n")

    print("Input directory: {}".format(INPUT_DIR))
    print("Output directory: {}".format(OUTPUT_DIR))
    print("Grid size: {}".format(GRID_SIZE))
    print("Seed: {}\n".format(SEED))

    if not INPUT_DIR.exists():
        raise FileNotFoundError(INPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Collect images
    # --------------------------------------------------------

    image_paths = sorted(INPUT_DIR.glob("*.jpg"))

    if len(image_paths) == 0:
        raise RuntimeError("No images found in {}".format(INPUT_DIR))

    print("Found {} images\n".format(len(image_paths)))

    # limit for demo
    image_paths = image_paths[:MAX_IMAGES]

    # --------------------------------------------------------
    # Create patch transform
    # --------------------------------------------------------

    patch = Patch(
        grid_size=GRID_SIZE,
        tile_transform=None,
        shuffle=True,
        seed=SEED
    )

    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------

    for i, img_path in enumerate(image_paths):

        print("[{}/{}] {}".format(i + 1, len(image_paths), img_path.name))

        img = Image.open(img_path).convert("RGB")

        print("Size: {}".format(img.size))

        try:
            shuffled = patch(img)
        except Exception as e:
            print("Patch failed: {}\n".format(e))
            continue

        # ----------------------------------------------------
        # Save outputs
        # ----------------------------------------------------

        base_name = img_path.stem

        original_out = OUTPUT_DIR / "{}_original.jpg".format(base_name)
        shuffled_out = OUTPUT_DIR / "{}_shuffled.jpg".format(base_name)

        img.save(original_out, quality=90)
        shuffled.save(shuffled_out, quality=90)

        print("Saved:")
        print("  {}".format(original_out))
        print("  {}\n".format(shuffled_out))


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()