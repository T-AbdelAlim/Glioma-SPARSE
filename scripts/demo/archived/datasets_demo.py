from pathlib import Path

from glioma_sparse.data_utils.slide_dataset import SlideDataset
from glioma_sparse.data_utils.patches import Patch
from glioma_sparse.data_utils.transforms import build_train_transform


# ============================================================
# PATHS
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = REPO_ROOT / "scripts" / "output_thumbnails" / "included"


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n=== DATASET DEMO ===\n")

    patch = Patch(grid_size=8, shuffle=True, seed=42)

    transform = build_train_transform()

    dataset = SlideDataset(
        root_dir=DATA_DIR,
        transform=transform,
        patch_transform=patch,
        class_names=["control", "low_grade", "high_grade"]
    )

    print("Dataset size: {}".format(len(dataset)))
    print("Classes: {}\n".format(dataset.classes))

    # --------------------------------------------------------
    # Inspect a few samples
    # --------------------------------------------------------

    for i in range(min(3, len(dataset))):

        img, label = dataset[i]

        print("[{}] shape: {} label: {}".format(i, img.shape, label))


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()