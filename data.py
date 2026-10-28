"""Datasets, stratified 70/15/15 split (Step 1) and per-image preprocessing (Step 2)."""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from augment import AcquisitionAug, geometric_aug

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
NOTUMOR = CLASSES.index("notumor")
BINARY_NAMES = {"yes": 1, "tumor": 1, "no": 0, "notumor": 0}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _images(root):
    return [p for p in Path(root).rglob("*") if p.suffix.lower() in IMG_EXT]


def scan_four_class(root):
    """Kaggle/Masoud layout: <root>/**/<class>/*.jpg. Training+Testing are merged
    and re-split so the split is ours, not the dataset's."""
    items = [(str(p), CLASSES.index(p.parent.name.lower()))
             for p in _images(root) if p.parent.name.lower() in CLASSES]
    if not items:
        raise FileNotFoundError(f"No images in class folders {CLASSES} under {root}")
    return sorted(items)


def scan_external(root):
    """External set. Either 4-class folders, or binary Br35H folders (yes/no).
    Returns (items, is_binary); binary label 1 = tumor."""
    four, binary = [], []
    for p in _images(root):
        name = p.parent.name.lower()
        if name in CLASSES:
            four.append((str(p), CLASSES.index(name)))
        elif name in BINARY_NAMES:
            binary.append((str(p), BINARY_NAMES[name]))
    if binary and not four:
        return sorted(binary), True
    if four and not binary:
        return sorted(four), False
    raise FileNotFoundError(f"Could not infer 4-class or yes/no layout under {root}")


def make_splits(items, seed=42):
    labels = [y for _, y in items]
    train, rest = train_test_split(items, test_size=0.30, stratify=labels, random_state=seed)
    val, test = train_test_split(rest, test_size=0.50, stratify=[y for _, y in rest],
                                 random_state=seed)
    return {"train": train, "val": val, "test": test}


def save_splits(splits, path):
    Path(path).write_text(json.dumps(splits))


def load_splits(path):
    return {k: [tuple(i) for i in v] for k, v in json.loads(Path(path).read_text()).items()}


class MRIDataset(Dataset):
    """Grayscale -> resize -> (train: geometric + acquisition aug) -> z-score -> 3ch.

    Skull stripping / N4 correction need raw volumes or brain masks; the public 2D
    JPG datasets are already cropped, so only intensity normalisation is applied
    here. See preprocess_n4() for an optional N4 hook if SimpleITK is installed.
    """

    def __init__(self, items, train=False, size=224, acquisition=True):
        self.items, self.size = items, size
        self.geo = geometric_aug() if train else None
        self.acq = AcquisitionAug() if (train and acquisition) else None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, y = self.items[i]
        img = Image.open(path).convert("L").resize((self.size, self.size), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).unsqueeze(0)
        if self.geo is not None:
            x = self.geo(x)
        if self.acq is not None:
            x = self.acq(x)
        x = (x - x.mean()) / (x.std() + 1e-6)  # z-score
        return x.repeat(3, 1, 1), y


def preprocess_n4(gray):
    """Optional N4 bias-field correction on a 2D float array (needs SimpleITK)."""
    import SimpleITK as sitk
    img = sitk.GetImageFromArray(gray.astype(np.float32))
    mask = sitk.OtsuThreshold(img, 0, 1)
    out = sitk.N4BiasFieldCorrector().Execute(img, mask)
    return sitk.GetArrayFromImage(out)
