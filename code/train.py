import argparse

import numpy as np
import torch
import torch.nn as nn
from monai.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from monai.losses import FocalLoss
from torch.optim import SGD

from dataset import get_dataset, make_datalist, SEQUENCES
from model import get_model

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].long().to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].numpy()
        logits = model(images).cpu()
        probs = torch.softmax(logits, dim=1).numpy()  # shape [B, 3]
        all_probs.extend(probs)
        all_labels.extend(labels)

    y_true = label_binarize(all_labels, classes=[0, 1, 2])  # one-hot [N, 3]
    y_pred = np.array(all_probs)                            # [N, 3]

    # Micro-averaged ROC curve (matches official evaluate.py)
    fpr, tpr, _ = roc_curve(y_true.ravel(), y_pred.ravel(), drop_intermediate=False)
    auc = roc_auc_score(y_true, y_pred, average="micro")

    sensitivity_at_90_spec = float(np.interp(0.10, fpr, tpr))       # TPR at FPR=0.10
    fpr_at_90_sens = float(np.interp(0.90, tpr, fpr))
    specificity_at_90_sens = 1.0 - fpr_at_90_sens

    return auc, specificity_at_90_sens, sensitivity_at_90_spec


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using sequences: {SEQUENCES}")

    train_ds = get_dataset("train")
    val_ds = get_dataset("val")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    train_labels = [item["label"] for item in make_datalist("train")]
    counts = np.bincount(train_labels, minlength=3).astype(float)
    class_weights = torch.tensor(len(train_labels) / (3 * counts), dtype=torch.float32).to(device)
    print(f"Class weights (normal/benign/malignant): {class_weights.tolist()}")

    model = get_model(in_channels=len(SEQUENCES), num_classes=3).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    #optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.9)
    scheduler = CosineAnnealingLR(optimizer, T_max=120)
    # criterion = nn.CrossEntropyLoss(weight=class_weights)
    # criterion = nn.CrossEntropyLoss()
    criterion = FocalLoss(weight=class_weights, gamma=1.0, to_onehot_y=True, use_softmax=True)

    best_auc = 0.0
    
    log_rows = []
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_auc, train_spec, train_sens = evaluate(model, train_loader, device)
        val_auc, val_spec, val_sens = evaluate(model, val_loader, device)
        train_score = (train_auc + train_spec + train_sens) / 3
        val_score = (val_auc + val_spec + val_sens) / 3
        scheduler.step()

        print(f"Epoch {epoch:03d} | loss: {train_loss:.4f} | AUC: {train_auc:.4f} | Val Spec@90Sens: {train_spec:.4f} | Val Sens@90Spec: {train_sens:.4f} | Val Score: {train_score:.4f}        \
              | Val AUC: {val_auc:.4f} | Val Spec@90Sens: {val_spec:.4f} | Val Sens@90Spec: {val_sens:.4f} | Val Score: {val_score:.4f}")
        
        log_rows.append({"epoch":epoch, "loss":train_loss, "train_auc":train_auc, "train_spec":train_spec, "train_sens":train_sens, "val_auc":val_auc, "val_spec":val_spec, "val_sens":val_sens})

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  -> Saved best model (score {best_auc:.4f})")
    
    import csv
    with open("training_log.csv", "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "train_auc", "train_spec", "train_sens", "val_auc", "val_spec", "val_sens"])
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"\nBest val AUC: {best_auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-5)
    main(parser.parse_args())
