import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
from pathlib import Path

DATA_ROOT = Path("/datasets/tdt4265/ODELIA2025/data")
SEQUENCES  = ["Pre", "Post_1", "Post_2", "Sub_1", "T2"]

# Auto-pick a malignant CAM case that has all 5 sequences
ann = pd.read_csv(DATA_ROOT / "CAM/metadata_unilateral/annotation.csv")
i = 0
m = 0
for _, row in ann[ann["Lesion"] == 2].iterrows():
    folder = DATA_ROOT / "CAM/data_unilateral" / row["UID"]
    if i >= m:
        if all((folder / f"{s}.nii.gz").exists() for s in SEQUENCES):
            break

print(f"Using: {row['UID']}")

fig, axes = plt.subplots(1, 5, figsize=(18, 4), facecolor="#0D1B2A")
titles = ["Pre", "Post_1", "Post_2", "Sub_1 ★", "T2"]

for ax, seq, title in zip(axes, SEQUENCES, titles):
    vol = nib.load(str(folder / f"{seq}.nii.gz")).get_fdata()
    sl  = vol[:, :, vol.shape[2] // 2]
    lo, hi = np.percentile(sl, 1), np.percentile(sl, 99)
    sl  = np.clip((sl - lo) / (hi - lo + 1e-8), 0, 1)

    ax.imshow(sl.T, origin="lower", cmap="inferno" if "Sub" in seq else "gray")
    ax.axis("off")
    col = "#F4A261" if "Sub" in seq else "#00B4D8"
    ax.set_title(title, color=col, fontsize=13, fontweight="bold" if "Sub" in seq else "normal")

plt.tight_layout()
plt.savefig("sequence_grid.png", dpi=180, bbox_inches="tight", facecolor="#0D1B2A")
print("Saved → sequence_grid.png")