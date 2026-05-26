from pathlib import Path
from glioma_sparse.preprocessing.process_dataset import process_wsi_folder


# ============================================================
# CONFIG
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]

BASE_INPUT = Path(
    r"D:\Thinkpad_Backup\Documents\EMC_postdoc\Virtual_Biopsy\data\WSI_ebrains\WHO2021_data"
)

OUTPUT_ROOT = REPO_ROOT / "data"

CLASS_FOLDERS = [
    "astro_IDHmt_G2",
    "astro_IDHmt_G3",
    "astro_IDHmt_G4",
    "control",
    "GBM_IDHwt",
    "oligo_IDHmt_1p19qdel_G2",
    "oligo_IDHmt_1p19qdel_G3",
]

TISSUE_THRESHOLD = 0.1
THRESHOLD_METRIC = "effective"

NUM_WORKERS = 6


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n=== GLIOMA-SPARSE BATCH PROCESSING ===\n")

    print(f"Workers per class: {NUM_WORKERS}")
    print(f"Tissue threshold:  {TISSUE_THRESHOLD} ({THRESHOLD_METRIC})")
    print("Each thumbnail is saved with a matching .json mapping sidecar\n")

    for cls in CLASS_FOLDERS:

        input_dir = BASE_INPUT / cls / cls
        output_dir = OUTPUT_ROOT / cls

        print(f"\n--- Processing: {cls} ---")
        print(f"Input:  {input_dir}")
        print(f"Output: {output_dir}")

        if not input_dir.exists():
            print("WARNING: input directory not found, skipping")
            continue

        output_dir.mkdir(parents=True, exist_ok=True)

        process_wsi_folder(
            input_dir=input_dir,
            output_dir=output_dir,
            tissue_threshold=TISSUE_THRESHOLD,
            threshold_metric=THRESHOLD_METRIC,
            num_workers=NUM_WORKERS,
        )

    print("\n=== ALL CLASSES PROCESSED ===\n")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
