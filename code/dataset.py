from pathlib import Path

import pandas as pd
from monai.data import CacheDataset
from monai.transforms import (
    Compose,
    CropForegroundd,
    LoadImaged,
    RandFlipd,
    RandRotate90d,
    Resized,
    ScaleIntensityRangePercentilesd,
    RandGaussianNoised, 
    RandShiftIntensityd,
    RandScaleIntensityd,
    ToTensord,
)
from sklearn.model_selection import train_test_split

# DATA_ROOT = Path(__file__).parent.parent / "odelia_2025_data" / "data"
# SPLIT_CSV = Path(__file__).parent.parent / "odelia_2025_data" / "split_unilateral.csv"

# SPLIT_CSV = Path("/datasets/tdt4265/ODELIA2025/split_unilateral.csv")
DATA_ROOT = Path("/datasets/tdt4265/ODELIA2025/data")
### IGNORE SPLIT IN DATA. SPLIT FRESH ###


# Extend this list to use more sequences: ["Pre", "Post_1", "Post_2", "Sub_1", "T2"]
# Pre.nii.gz    - before contrast injection                                                        
# Post_1.nii.gz - after contrast injection (early)                                                 
# Post_2.nii.gz - after contrast injection (late)                                                  
# Sub_1.nii.gz  - Post_1 minus Pre (highlights enhancement = suspicious areas)                     
# T2.nii.gz     - different contrast, shows anatomy/fluid 
SEQUENCES = ["Pre", "Post_1", "Post_2", "Sub_1", "T2"]

SPATIAL_SIZE = (96, 96, 96)
# FOLD = 0  # which cross-validation fold to use for train/val split
### IGNORE SPLIT IN DATA. SPLIT FRESH ###

INSTITUTIONS = ["CAM", "MHA", "RUMC", "UKA"] #RHS as test set

# test split
TRAIN_INSTITUTIONS = ["CAM", "RUMC", "UKA"]
VAL_INSTITUTIONS = ["MHA"]

def load_all_annotations() -> pd.DataFrame:
    sub_tables = []
    for inst in INSTITUTIONS:
        annoted_path = DATA_ROOT / inst / "metadata_unilateral" / "annotation.csv"
        if not annoted_path.exists(): continue
        df = pd.read_csv(annoted_path)
        df["Institution"] = inst
        sub_tables.append(df)
    return pd.concat(sub_tables, ignore_index=True)

def make_datalist(split: str, seed=42) -> list[dict]:
    annotations = load_all_annotations()
    # for stratifying, at "breast level"
    train_uids, val_uids = train_test_split(
        annotations["UID"],
        test_size=0.2,
        stratify=annotations["Lesion"],
        random_state=seed
    )
    # train_uids = annotations[annotations["Institution"].isin(TRAIN_INSTITUTIONS)]["UID"]
    # val_uids = annotations[annotations["Institution"].isin(VAL_INSTITUTIONS)]["UID"]
    uid_set = set(train_uids) if split == "train" else set(val_uids)
    subset = annotations[annotations["UID"].isin(uid_set)]
    items = []
    for _, row in subset.iterrows():
        folder = DATA_ROOT / row["Institution"] / "data_unilateral" / row["UID"]
        image_paths = [str(folder / f"{seq}.nii.gz") for seq in SEQUENCES]
        if not all(Path(p).exists() for p in image_paths): continue
        items.append({"image": image_paths, "label": int(row["Lesion"])})
        
    return items
    
def get_transforms(augment: bool) -> Compose:
    base = [
        LoadImaged(keys=["image"], image_only=True, ensure_channel_first=True),
        ScaleIntensityRangePercentilesd(keys=["image"], lower=1, upper=99, b_min=0.0, b_max=1.0, clip=True, channel_wise=True),
        CropForegroundd(keys=["image"], source_key="image"),
        Resized(keys=["image"], spatial_size=SPATIAL_SIZE),
    ]

    aug = [
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
        RandRotate90d(keys=["image"], prob=0.5, max_k=3),
        RandGaussianNoised(),
        RandShiftIntensityd(),
        RandScaleIntensityd(),
    ] if augment else []

    return Compose(base + aug + [ToTensord(keys=["image"])])

def get_dataset(split: str) -> CacheDataset:
    datalist = make_datalist(split)
    augment = split == "train"
    transforms = get_transforms(augment)
    return CacheDataset(data=datalist, transform=transforms, cache_rate=1.0, num_workers=4) #change num_workers on local