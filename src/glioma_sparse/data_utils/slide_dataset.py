from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class SlideDataset(Dataset):

    SUPPORTED_EXTS = (".jpg", ".jpeg", ".png")

    def __init__(
        self,
        root_dir,
        transform=None,
        patch_transform=None,
        class_names=None
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.patch_transform = patch_transform

        if not self.root_dir.exists():
            raise FileNotFoundError(self.root_dir)

        # --------------------------------------------------
        # Classes (explicit or inferred)
        # --------------------------------------------------
        if class_names is not None:
            self.classes = class_names

            missing = [c for c in class_names if not (self.root_dir / c).exists()]
            if missing:
                raise RuntimeError(
                    "Missing class folders: {}".format(missing)
                )
        else:
            self.classes = sorted([
                d.name for d in self.root_dir.iterdir() if d.is_dir()
            ])

        if not self.classes:
            raise RuntimeError(
                "No class folders found in {}".format(self.root_dir)
            )

        # deterministic mapping
        self.class_to_idx = {
            cls_name: i for i, cls_name in enumerate(self.classes)
        }

        # --------------------------------------------------
        # Collect paths + labels
        # --------------------------------------------------
        self.paths = []
        self.labels = []

        for cls in self.classes:
            class_dir = self.root_dir / cls

            for p in class_dir.iterdir():
                if p.suffix.lower() in self.SUPPORTED_EXTS:
                    self.paths.append(p)
                    self.labels.append(self.class_to_idx[cls])

        if not self.paths:
            raise RuntimeError("No images found")

        self.class_names = self.classes

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):

        path = self.paths[idx]
        label = self.labels[idx]

        img = Image.open(path).convert("RGB")

        # --------------------------------------------------
        # Patch shuffle (train-time augmentation)
        # --------------------------------------------------
        if self.patch_transform is not None:
            img = self.patch_transform(img)

        # --------------------------------------------------
        # Normalization / tensor conversion
        # --------------------------------------------------
        if self.transform is not None:
            img = self.transform(img)

        return img, label