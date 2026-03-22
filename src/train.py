import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

try:
    from src.model_utils import build_model, freeze_backbone, get_parameter_counts, get_transforms, set_seed
except ModuleNotFoundError:
    from model_utils import build_model, freeze_backbone, get_parameter_counts, get_transforms, set_seed


def accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()


def run_epoch(model, loader, criterion, optimizer, device, train_mode: bool, scaler, use_amp: bool):
    if train_mode:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_acc = 0.0

    progress = tqdm(loader, total=len(loader), leave=False)
    for images, labels in progress:
        images = images.to(device)
        labels = labels.to(device)

        if train_mode:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train_mode):
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            if train_mode:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_acc += accuracy(logits, labels) * batch_size
        progress.set_description(f"loss={loss.item():.4f}")

    n = len(loader.dataset)
    return running_loss / n, running_acc / n


def main():
    parser = argparse.ArgumentParser(description="Train food spoilage classifier.")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--unfreeze_backbone", action="store_true")
    parser.add_argument("--history_csv", type=str, default="reports/train_history.csv")
    parser.add_argument("--output", type=str, default="checkpoints/best_model.pt")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cpu")  # Force CPU - GPU has cuDNN issues on this system
    print(f"Using device: {device}")
    use_amp = False  # Disable AMP on CPU

    train_tfms = get_transforms(args.img_size, train_mode=True)
    eval_tfms = get_transforms(args.img_size, train_mode=False)

    data_dir = Path(args.data_dir)
    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tfms)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=eval_tfms)
    if train_ds.classes != val_ds.classes:
        raise ValueError("Class mismatch between train and val splits.")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    class_names = train_ds.classes
    n_classes = len(class_names)
    print(f"Classes: {class_names}")

    model = build_model(num_classes=n_classes, pretrained=True)

    if not args.unfreeze_backbone:
        freeze_backbone(model)

    model = model.to(device)

    total_params, trainable_params = get_parameter_counts(model)
    print(f"Parameters: total={total_params:,}, trainable={trainable_params:,}")

    labels_np = np.array(train_ds.targets)
    class_sample_count = np.array([(labels_np == t).sum() for t in range(n_classes)], dtype=np.float64)
    class_weights = class_sample_count.sum() / (n_classes * class_sample_count)
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=args.label_smoothing)
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params_to_optimize, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_acc = 0.0
    best_val_loss = float("inf")
    bad_epochs = 0
    history = []

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    history_path = Path(args.history_csv)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            train_mode=True,
            scaler=scaler,
            use_amp=use_amp,
        )
        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            train_mode=False,
            scaler=scaler,
            use_amp=use_amp,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_acc)

        print(
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | lr={current_lr:.6f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_acc": round(train_acc, 6),
                "val_loss": round(val_loss, 6),
                "val_acc": round(val_acc, 6),
                "lr": current_lr,
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "img_size": args.img_size,
                    "epoch": epoch,
                    "best_val_acc": best_val_acc,
                    "best_val_loss": best_val_loss,
                    "train_args": vars(args),
                },
                output_path,
            )
            print(f"Saved new best model to {output_path}")
        else:
            bad_epochs += 1

        improved_loss = val_loss < (best_val_loss - args.min_delta)
        if improved_loss:
            best_val_loss = val_loss
            bad_epochs = 0

        if bad_epochs >= args.patience:
            print("Early stopping triggered.")
            break

    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])
        writer.writeheader()
        writer.writerows(history)

    summary_path = history_path.with_name("training_summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_val_acc": best_val_acc,
                "best_val_loss": best_val_loss,
                "epochs_ran": len(history),
                "checkpoint": str(output_path),
            },
            f,
            indent=2,
        )

    print(f"Training complete. Best val_acc={best_val_acc:.4f}")
    print(f"Training history saved to {history_path}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
