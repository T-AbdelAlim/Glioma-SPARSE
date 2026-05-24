from pathlib import Path

from glioma_sparse.preprocessing.process_dataset import process_wsi_folder


# ============================================================
# PATHS
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "testdata"
OUTPUT_DIR = REPO_ROOT / "scripts" / "output_thumbnails"


# ============================================================
# CONFIG
# ============================================================

TISSUE_THRESHOLD = 0.15
THRESHOLD_METRIC = "effective"   # "tissue" or "effective"


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n=== GLIOMA-SPARSE DATASET PROCESSING DEMO ===\n")

    print("Input directory: {}".format(DATA_DIR))
    print("Output directory: {}".format(OUTPUT_DIR))
    print("Tissue threshold: {}".format(TISSUE_THRESHOLD))
    print("Threshold metric: {}\n".format(THRESHOLD_METRIC))

    if THRESHOLD_METRIC == "tissue":
        print("Filtering based on biological tissue content\n")
    else:
        print("Filtering based on effective (post-padding) content\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    process_wsi_folder(
        input_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        tissue_threshold=TISSUE_THRESHOLD,
        threshold_metric=THRESHOLD_METRIC,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()