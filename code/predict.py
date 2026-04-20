import argparse
import csv
from pathlib import Path

import torch
from monai.data import CacheDataset, DataLoader

from dataset import DATA_ROOT, SEQUENCES, get_transforms
from model import get_model

RSH_ROOT = DATA_ROOT / "RSH"
RSH_SPLIT = RSH_ROOT / "metadata_unilateral" / "split.csv"


def make_rsh_datalist() -> list[dict]:
    import pandas as pd
    df = pd.read_csv(RSH_SPLIT)
    items = []
    for _, row in df.iterrows():
        # if row["Split"] != "test": continue # to reduce to 34 datapoints
        uid = row["UID"]
        folder = RSH_ROOT / "data_unilateral" / uid
        image_paths = [str(folder / f"{seq}.nii.gz") for seq in SEQUENCES]
        if not all(Path(p).exists() for p in image_paths):
            print(f"Warning: skipping {uid} (missing sequences)")
            continue
        items.append({"image": image_paths, "uid": uid})
    return items


def main(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    print(f"Using sequences: {SEQUENCES}")

    datalist = make_rsh_datalist()
    print(f"RSH cases: {len(datalist)}")

    # Same deterministic transforms as validation (no augmentation)
    transforms = get_transforms(augment=False)
    ds = CacheDataset(data=datalist, transform=transforms, cache_rate=1.0, num_workers=0)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = get_model(in_channels=len(SEQUENCES), num_classes=3).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    rows = []
    uid_index = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            logits = model(images).cpu()
            probs = torch.softmax(logits, dim=1).numpy()
            for p in probs:
                uid = datalist[uid_index]["uid"]
                rows.append({
                    "ID": uid,
                    "normal": round(float(p[0]), 6),
                    "benign": round(float(p[1]), 6),
                    "malignant": round(float(p[2]), 6),
                })
                uid_index += 1

    output_path = Path(args.output)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "normal", "benign", "malignant"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} predictions to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="best_model.pth")
    parser.add_argument("--output", type=str, default="predictions.csv")
    parser.add_argument("--batch_size", type=int, default=2)
    main(parser.parse_args())
