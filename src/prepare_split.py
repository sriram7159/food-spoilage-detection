import argparse
import random
import shutil
from pathlib import Path


def collect_images(directory: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [p for p in directory.rglob("*") if p.suffix.lower() in exts]


def ensure_dirs(base_out: Path, class_names):
    for split in ["train", "val", "test"]:
        for class_name in class_names:
            (base_out / split / class_name).mkdir(parents=True, exist_ok=True)


def copy_subset(files, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        dst = out_dir / src.name
        stem = src.stem
        suffix = src.suffix
        counter = 1
        while dst.exists():
            dst = out_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        shutil.copy2(src, dst)


def split_class(images, train_ratio: float, val_ratio: float, seed: int):
    random.Random(seed).shuffle(images)
    n = len(images)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = images[:n_train]
    val = images[n_train:n_train + n_val]
    test = images[n_train + n_val:]
    return train, val, test


def main():
    parser = argparse.ArgumentParser(description="Split raw image folders into train/val/test.")
    parser.add_argument("--raw_dir", type=str, required=True, help="Raw directory with class subfolders")
    parser.add_argument("--out_dir", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    class_dirs = [d for d in raw_dir.iterdir() if d.is_dir()]
    if not class_dirs:
        raise ValueError("No class folders found in raw_dir.")

    class_names = [d.name for d in class_dirs]
    ensure_dirs(out_dir, class_names)

    print(f"Found classes: {class_names}")

    for class_dir in class_dirs:
        images = collect_images(class_dir)
        if not images:
            print(f"Skipping {class_dir.name}: no images.")
            continue

        train, val, test = split_class(images, args.train_ratio, args.val_ratio, args.seed)

        copy_subset(train, out_dir / "train" / class_dir.name)
        copy_subset(val, out_dir / "val" / class_dir.name)
        copy_subset(test, out_dir / "test" / class_dir.name)

        print(
            f"{class_dir.name}: total={len(images)}, "
            f"train={len(train)}, val={len(val)}, test={len(test)}"
        )

    print("Dataset split complete.")


if __name__ == "__main__":
    main()
