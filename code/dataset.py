from pathlib import Path

import pandas as pd
from monai.data.dataset import Dataset
from monai.transforms import (
    Compose,
    CropForegroundd,
    LoadImaged,
    RandFlipd,
    RandRotate90d,
    Resized,
    ScaleIntensityRangePercentilesd,
    ToTensord,
)

DATA_ROOT = Path(__file__).parent.parent / "odelia_2025_data" / "data"
SPLIT_CSV = Path(__file__).parent.parent / "odelia_2025_data" / "split_unilateral.csv"

# Extend this list to use more sequences: ["Pre", "Post_1", "Post_2", "Sub_1", "T2"]
# Pre.nii.gz    - before contrast injection                                                        
# Post_1.nii.gz - after contrast injection (early)                                                 
# Post_2.nii.gz - after contrast injection (late)                                                  
# Sub_1.nii.gz  - Post_1 minus Pre (highlights enhancement = suspicious areas)                     
# T2.nii.gz     - different contrast, shows anatomy/fluid 
SEQUENCES = ["Sub_1"]

SPATIAL_SIZE = (64, 64, 64)

# return merged annotation + metadata for each UID
# result is df with columns: UID, Fold, Split, Institution, Lesion
def load_metadata() -> pd.DataFrame:
    split_df = pd.read_csv(SPLIT_CSV)

    annotations = []
    for institution in split_df["Institution"].unique():
        ann_path = DATA_ROOT / institution / "metadata_unilateral" / "annotation.csv"
        if not ann_path.exists():
            continue
        ann_df = pd.read_csv(ann_path)
        annotations.append(ann_df)

    ann_df = pd.concat(annotations, ignore_index=True)
    return split_df.merge(ann_df[["UID", "Lesion"]], on="UID")


def make_datalist(split: str) -> list[dict]:
    df = load_metadata()
    subset = df[df["Split"] == split]

    items = []
    for _, row in subset.iterrows():
        institution = row["Institution"]
        uid = row["UID"]
        folder = DATA_ROOT / institution / "data_unilateral" / uid

        image_paths = [str(folder / f"{seq}.nii.gz") for seq in SEQUENCES]
        # Filter to sequences that actually exist on disk
        image_paths = [p for p in image_paths if Path(p).exists()]

        if not image_paths:
            continue

        items.append({"image": image_paths, "label": int(row["Lesion"])})

    return items


def get_transforms(augment: bool) -> Compose:
    base = [
        LoadImaged(keys=["image"], image_only=True, ensure_channel_first=True),
        ScaleIntensityRangePercentilesd(keys=["image"], lower=1, upper=99, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image"], source_key="image"),
        Resized(keys=["image"], spatial_size=SPATIAL_SIZE),
    ]

    aug = [
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
        RandRotate90d(keys=["image"], prob=0.5, max_k=3),
    ] if augment else []

    return Compose(base + aug + [ToTensord(keys=["image"])])


def get_dataset(split: str) -> Dataset:
    datalist = make_datalist(split)
    augment = split == "train"
    transforms = get_transforms(augment)
    return Dataset(data=datalist, transform=transforms)
