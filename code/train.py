import argparse

import torch
import torch.nn as nn
from monai.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import get_dataset, SEQUENCES
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

    import numpy as np
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

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = get_model(in_channels=len(SEQUENCES), num_classes=3).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_auc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_auc, val_spec, val_sens = evaluate(model, val_loader, device)
        scheduler.step()

        print(f"Epoch {epoch:03d} | loss: {train_loss:.4f} | AUC: {val_auc:.4f} | Spec@90Sens: {val_spec:.4f} | Sens@90Spec: {val_sens:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  -> Saved best model (AUC {best_auc:.4f})")

    print(f"\nBest val AUC: {best_auc:.4f}")

    # Final test evaluation
    test_ds = get_dataset("test")
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    test_auc, test_spec, test_sens = evaluate(model, test_loader, device)
    score = (test_auc + test_spec + test_sens) / 3
    print(f"Test AUC: {test_auc:.4f} | Spec@90Sens: {test_spec:.4f} | Sens@90Spec: {test_sens:.4f} | Score: {score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    main(parser.parse_args())
