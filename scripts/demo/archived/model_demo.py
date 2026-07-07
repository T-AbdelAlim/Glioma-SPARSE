import torch
from pathlib import Path

from glioma_sparse.data_utils.slide_dataset import SlideDataset
from glioma_sparse.data_utils.patches import Patch
from glioma_sparse.data_utils.transforms import build_eval_transform
from glioma_sparse.models.factory import build_model


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "scripts" / "output_thumbnails" / "included"


def main():

    print("\n=== MODEL DEMO ===\n")

    dataset = SlideDataset(
        root_dir=DATA_DIR,
        transform=build_eval_transform(),
        patch_transform=None,
        class_names=["control", "low_grade", "high_grade"]
    )

    img, label = dataset[0]

    print("Input shape:", img.shape)

    # add batch dimension
    x = img.unsqueeze(0)

    model = build_model(
        name="resnet18",
        num_classes=3,
        pretrained=True
    )

    model.eval()

    with torch.no_grad():
        out = model(x)

    print("Output shape:", out.shape)


if __name__ == "__main__":
    main()