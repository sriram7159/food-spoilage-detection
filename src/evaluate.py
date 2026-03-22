import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torchvision import datasets

try:
    from src.model_utils import get_transforms, load_checkpoint
except ModuleNotFoundError:
    from model_utils import get_transforms, load_checkpoint


def save_confusion_matrix(cm: np.ndarray, class_names, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained food spoilage classifier on test split.")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--report_json", type=str, default="reports/evaluation_report.json")
    parser.add_argument("--cm_png", type=str, default="reports/confusion_matrix.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, img_size, ckpt = load_checkpoint(Path(args.checkpoint), device)

    test_ds = datasets.ImageFolder(Path(args.data_dir) / "test", transform=get_transforms(img_size, train_mode=False))
    if test_ds.classes != class_names:
        raise ValueError("Class mismatch between checkpoint and test dataset.")

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    y_true = []
    y_pred = []

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.numpy().tolist())

    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True)

    cm_out = Path(args.cm_png)
    save_confusion_matrix(cm, class_names, cm_out)

    out = {
        "checkpoint": str(args.checkpoint),
        "best_val_acc": ckpt.get("best_val_acc"),
        "test_size": len(test_ds),
        "class_names": class_names,
        "metrics": {
            "accuracy": accuracy,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
        },
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_png": str(cm_out),
    }

    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Evaluation complete")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 macro: {f1_macro:.4f}")
    print(f"Report: {report_path}")
    print(f"Confusion matrix image: {cm_out}")


if __name__ == "__main__":
    main()