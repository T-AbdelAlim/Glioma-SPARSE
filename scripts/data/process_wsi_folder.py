"""
Thumbnails for training: every whole-slide image becomes a 2048 x 2048 thumbnail at 4 um/pixel with a JSON
sidecar that maps thumbnail pixels to level-0 slide coordinates. Slides with too little tissue are set aside.

Input: one folder per class under --wsi-root, named after the WHO 2021 type (as in EBRAINS):
astro_IDHmt_G2, astro_IDHmt_G3, astro_IDHmt_G4, oligo_IDHmt_1p19qdel_G2, oligo_IDHmt_1p19qdel_G3, GBM_IDHwt and
control (a nested <class>/<class> folder is also accepted).

Output:
  data/ebrains_thumbnails/<class>/included/<slide_id>.jpg + .json   (and low_tissue/, metadata.csv)
  data/included/<control|low_grade|high_grade>/<slide_id>.jpg        (Stage A training set: grade 2 low,
                                                                       grades 3 and 4 high)

    python -m scripts.data.process_wsi_folder --wsi-root <slides>
"""

import argparse
import shutil
from pathlib import Path

from glioma_sparse.preprocessing.process_dataset import process_wsi_folder

REPO_ROOT = Path(__file__).resolve().parents[2]
CLASS_FOLDERS = ["astro_IDHmt_G2", "astro_IDHmt_G3", "astro_IDHmt_G4", "control", "GBM_IDHwt",
                 "oligo_IDHmt_1p19qdel_G2", "oligo_IDHmt_1p19qdel_G3"]


def grade_folder(cls):
    return "control" if cls == "control" else "low_grade" if cls.endswith("_G2") else "high_grade"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wsi-root", required=True, help="folder with one subfolder of slides per class")
    ap.add_argument("--tissue-threshold", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    thumbs, by_grade = REPO_ROOT / "data" / "ebrains_thumbnails", REPO_ROOT / "data" / "included"

    for cls in CLASS_FOLDERS:
        input_dir = Path(a.wsi_root) / cls
        if (input_dir / cls).is_dir():
            input_dir = input_dir / cls
        if not input_dir.is_dir():
            print(f"{cls}: {input_dir} not found, skipped")
            continue
        print(f"\n--- {cls}: {input_dir}")
        process_wsi_folder(input_dir=input_dir, output_dir=thumbs / cls, tissue_threshold=a.tissue_threshold,
                           threshold_metric="effective", num_workers=a.workers)
        dst = by_grade / grade_folder(cls)
        dst.mkdir(parents=True, exist_ok=True)
        for jpg in (thumbs / cls / "included").glob("*.jpg"):
            shutil.copy2(jpg, dst / jpg.name)

    print(f"\nthumbnails with sidecars in {thumbs}, Stage A training set in {by_grade}")


if __name__ == "__main__":
    main()
