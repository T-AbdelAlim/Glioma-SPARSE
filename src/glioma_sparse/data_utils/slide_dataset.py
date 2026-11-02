from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class SlideDataset(Dataset):

    SUPPORTED_EXTS = (".jpg", ".jpeg", ".png")

    # --------------------------------------------------
    # CLASS ORDER
    # --------------------------------------------------
    DEFAULT_CLASS_NAMES = ["control", "low_grade", "high_grade"]

    def __init__(
        self,
        root_dir,
        transform=None,
        patch_transform=None,
        class_names=None,
        paths=None,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.patch_transform = patch_transform

        if not self.root_dir.exists():
            raise FileNotFoundError(self.root_dir)

        # --------------------------------------------------
        # CLASS ORDER (ENFORCED)
        # --------------------------------------------------
        if class_names is None:
            self.classes = self.DEFAULT_CLASS_NAMES
        else:
            self.classes = class_names

        # sanity check: folders exist
        missing = []
        for c in self.classes:
            if not (self.root_dir / c).exists():
                missing.append(c)

        if len(missing) > 0:
            raise RuntimeError(
                "Missing class folders: {}".format(missing)
            )

        # deterministic mapping (THIS is what you care about)
        self.class_to_idx = {
            cls_name: i for i, cls_name in enumerate(self.classes)
        }

        self.idx_to_class = {
            i: cls_name for cls_name, i in self.class_to_idx.items()
        }

        self.class_names = self.classes

        # --------------------------------------------------
        # COLLECT PATHS + LABELS
        # --------------------------------------------------
        self.paths = []
        self.labels = []

        if paths is not None:
            # ----------------------------------------------
            # Use predefined subset (split-safe)
            # ----------------------------------------------
            for p in paths:
                p = Path(p)

                if not p.exists():
                    raise RuntimeError(
                        "Path does not exist: {}".format(p)
                    )

                class_name = p.parent.name

                if class_name not in self.class_to_idx:
                    raise RuntimeError(
                        "Unknown class '{}' for path: {}".format(
                            class_name, p
                        )
                    )

                self.paths.append(p)
                self.labels.append(self.class_to_idx[class_name])

        else:
            # ----------------------------------------------
            # Full dataset scan
            # ----------------------------------------------
            for cls in self.classes:
                class_dir = self.root_dir / cls

                for p in class_dir.iterdir():
                    if p.suffix.lower() in self.SUPPORTED_EXTS:
                        self.paths.append(p)
                        self.labels.append(self.class_to_idx[cls])

        if len(self.paths) == 0:
            raise RuntimeError("No images found")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):

        path = self.paths[idx]
        label = self.labels[idx]

        img = Image.open(path).convert("RGB")

        # --------------------------------------------------
        # PATCH SHUFFLE
        # --------------------------------------------------
        if self.patch_transform is not None:
            img = self.patch_transform(img)

        # --------------------------------------------------
        # TRANSFORMS
        # --------------------------------------------------
        if self.transform is not None:
            img = self.transform(img)

        return img, label