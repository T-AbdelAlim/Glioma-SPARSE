<p align="center">
  <img src="docs/logo.png" alt="Glioma-SPARSE logo" width="220">
</p>

# Glioma-SPARSE

## A Low-Compute Deep Learning Approach for Accessible Glioma Characterization from Routine Histopathology

[App](#app) | [Versions](#versions) | [Installation](#installation) | [Retraining](#retraining) | [Cite](#citation)


## Description

Glioma-SPARSE classifies gliomas from routine H&E whole-slide images into the WHO 2021 classes (control, astrocytoma low/high grade, oligodendroglioma low/high grade, glioblastoma). It doesn't encode the whole slide at high resolution. It first looks at the slide at low resolution, finds where the signal is, and then only reads a few tiles from those regions at high resolution.

* **Stage A.** A ResNet-50 grades the slide (control, low grade, high grade) from a single 2048x2048 thumbnail.
* **Patch injection.** Each thumbnail patch is injected into a control slide to see how much it raises the predicted grade. This gives a signal map, and the top patches become the patch-regions.
* **Stage B.** A small sample of 20x tiles from the patch-regions is encoded by a frozen pathology foundation model (DistillPath, or UNI2-h in the `full` branch). A linear head predicts the molecular type (IDH-mutant, IDH-mutant 1p/19q-codeleted, IDH-wildtype). Grade and molecular type together give the integrated diagnosis.

![Overview](scripts/dashboard/explainer/overview.png)


## App

To run Glioma-SPARSE on your own slides without installing anything, download the app from the [latest release](https://github.com/T-AbdelAlim/Glioma-SPARSE/releases/latest):

| App | Runs on |
|---|---|
| `GliomaSPARSE-Dashboard-Windows.zip` | Windows 10 or 11 (64-bit) |
| `GliomaSPARSE-Dashboard-macOS.zip` | macOS on Apple silicon (M1 or later) |

Unzip it and start the app. The trained models are included. DistillPath is downloaded once on first use. A slide takes one to two minutes on a laptop. The apps are not signed, so on Windows choose *More info > Run anyway* the first time, and on macOS right-click the app and choose *Open*.


## Versions

Each branch contains the previous one.

| Branch | For | Stage B encoder |
|---|---|---|
| `light-release` | running the dashboard from source (the apps are built from it) | DistillPath |
| `main` | `light-release` plus retraining on your own slides | DistillPath |
| `full` | `main` plus UNI2-h, the encoder of the main analysis in the paper | DistillPath and UNI2-h |

The trained models are in the [release](https://github.com/T-AbdelAlim/Glioma-SPARSE/releases/latest) as `Glioma-SPARSE-models.zip`. Unzip it in the repository root.


## Installation

Requires Python 3.14. A GPU is optional.

```
git clone https://github.com/T-AbdelAlim/Glioma-SPARSE.git
cd Glioma-SPARSE
powershell -ExecutionPolicy Bypass -File setup_env.ps1     # windows
bash setup_env.sh                                          # linux
```

This creates `.venv`, installs the package and runs `python -m scripts.check_setup` to check that the models are in place. With conda: `conda env create -f environment.yml`.

Start the dashboard with:
```
python -m scripts.dashboard.run_dashboard
```
More on the dashboard in [scripts/dashboard/README.md](scripts/dashboard/README.md).


## Retraining

Branches `main` and `full`. Run everything from the repository root. The defaults reproduce the models of the paper.

**1. Thumbnails.** Put the slides in one folder per class (`astro_IDHmt_G2`, `astro_IDHmt_G3`, `astro_IDHmt_G4`, `oligo_IDHmt_1p19qdel_G2`, `oligo_IDHmt_1p19qdel_G3`, `GBM_IDHwt`, `control`).
```
python -m scripts.data.process_wsi_folder --wsi-root <slides>
```

**2. Splits.** Five cross-validation splits (the ones from the paper are in `splits/`).
```
python -m scripts.data.generate_splits
```

**3. Stage A.** One fold per run. Put the runs in `ResNet50_output/training_output/` for the dashboard.
```
python -m scripts.stage_a.train --split-csv splits/split_01.csv --model resnet50
python -m scripts.data.aggregate_splits
```

**4. Patch-regions.**
```
python -m scripts.stage_b.select_regions --model resnet50 --checkpoints-glob "ResNet50_output/training_output/*_split_0*/best_f1.pth" --out data/regions_rn50_f1.csv
```

**5. Tiles.** Segment and tile the slides with [TRIDENT](https://github.com/mahmoodlab/TRIDENT), from the TRIDENT folder:
```
python run_batch_of_slides.py --task seg --wsi_dir <slides> --job_dir <trident job> --segmenter hest
python run_batch_of_slides.py --task coords --wsi_dir <slides> --job_dir <trident job> --mag 20 --patch_size 256 --overlap 0
```
Then encode the tiles in the patch-regions with DistillPath:
```
python -m scripts.stage_b.encode_tiles --regions data/regions_rn50_f1.csv --trident-dir <trident job> --wsi-root <slides> --out data/tile_features/distillpath
```
For UNI2-h (`full` branch) use TRIDENT itself (`--task feat --patch_encoder uni_v2`).

**6. Stage B heads.**
```
python -m scripts.stage_b.fit_heads --regions data/regions_rn50_f1.csv --features-dir data/tile_features/distillpath --config rn50_f1_dp_uni2h
```
For UNI2-h: `--features-dir <trident job>/20x_256px_0px_overlap/features_uni_v2 --config rn50_f1`.


## Notes on usage

**Research use only, not a medical device.** Glioma-SPARSE is not certified (MDR, FDA or similar) and is not meant for clinical diagnosis.

UNI2-h, DistillPath and TRIDENT have their own licenses. The DistillPath and UNI2-h weights are for non-commercial use only, and UNI2-h needs an access request on [Hugging Face](https://huggingface.co/MahmoodLab/UNI2-h). The trained models are derived from the Digital Brain Tumour Atlas, so follow its terms of use. The code is under the MIT license (`LICENSE`).


## Acknowledgements and references

Please also cite these when you use the corresponding parts.

* **Data.** [Digital Brain Tumour Atlas](https://doi.org/10.1038/s41597-022-01157-0): Roetzer-Pejrimovsky T, et al. *Sci Data* 2022. External validation used slides from [TCGA](https://www.cancer.gov/tcga).
* **Stage A.** ResNet: He K, et al. *CVPR* 2016. ImageNet weights from torchvision.
* **DistillPath.** [RamonK/DistillPath-KS16-UNI2h](https://huggingface.co/RamonK/DistillPath-KS16-UNI2h): Kaspar R, et al. arXiv 2026. doi:10.48550/arXiv.2608.17872
* **UNI2-h.** [MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h): Chen RJ, et al. *Nat Med* 2024.
* **TRIDENT.** [mahmoodlab/TRIDENT](https://github.com/mahmoodlab/TRIDENT): Zhang A, et al. arXiv 2025. doi:10.48550/arXiv.2502.06750
* **OpenSlide.** Goode A, et al. *J Pathol Inform* 2013.
* **Libraries.** PyTorch, timm, scikit-learn, NumPy, pandas, SciPy, h5py, Pillow, OpenCV, Matplotlib, openpyxl, FastAPI, Uvicorn, pywebview and PyInstaller.


## Citation

Kindly consider citing the work if it supports your research:

_Abdel Alim et al. Glioma-SPARSE: A Low-Compute Deep Learning Approach for Accessible Glioma Characterization from Routine Histopathology. Preprint on bioRxiv (coming soon)._

**Github Repository:**
```
Abdel Alim, T. (2026). Glioma-SPARSE [Computer software]. https://github.com/T-AbdelAlim/Glioma-SPARSE
```
