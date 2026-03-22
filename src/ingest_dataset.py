import argparse
import os
import shutil
from collections import Counter
from pathlib import Path


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images(path: Path):
    for p in path.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            yield p


def map_binary_class(name: str):
    lower = name.lower()
    if lower.startswith("fresh"):
        return "fresh"
    if lower.startswith("rotten"):
        return "spoiled"
    return None


def safe_copy(src: Path, dst_dir: Path, prefer_hardlink: bool):
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    stem = src.stem
    suffix = src.suffix
    idx = 1
    while dst.exists():
        dst = dst_dir / f"{stem}_{idx}{suffix}"
        idx += 1
    if prefer_hardlink:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass

    shutil.copy2(src, dst)


def ingest(source_dir: Path, out_dir: Path, mode: str, prefer_hardlink: bool):
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    split_dirs = [d for d in source_dir.iterdir() if d.is_dir()]
    if not split_dirs:
        raise ValueError("No split directories found in source_dir")

    counts = Counter()
    skipped = 0

    for split_dir in split_dirs:
        class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
        for class_dir in class_dirs:
            src_class_name = class_dir.name.strip().lower()

            if mode == "binary":
                dst_class_name = map_binary_class(src_class_name)
                if dst_class_name is None:
                    skipped += 1
                    continue
            else:
                dst_class_name = src_class_name

            dst_class_dir = out_dir / dst_class_name
            for img in iter_images(class_dir):
                safe_copy(img, dst_class_dir, prefer_hardlink=prefer_hardlink)
                counts[dst_class_name] += 1

    if not counts:
        raise ValueError("No images ingested. Check source path and folder names.")

    print("Ingestion complete")
    print(f"Mode: {mode}")
    for cls, cnt in sorted(counts.items()):
        print(f"{cls}: {cnt}")
    print(f"Skipped class folders: {skipped}")


def main():
    parser = argparse.ArgumentParser(description="Ingest dataset folders and create clean raw data classes.")
    parser.add_argument("--source_dir", type=str, default="dataset", help="Path that contains Train/Test folders")
    parser.add_argument("--out_dir", type=str, default="data/raw", help="Output raw data directory")
    parser.add_argument(
        "--mode",
        type=str,
        default="binary",
        choices=["binary", "multiclass"],
        help="binary maps fresh*->fresh and rotten*->spoiled; multiclass keeps folder names",
    )
    parser.add_argument(
        "--clear_out_dir",
        action="store_true",
        help="Clear output directory before ingesting",
    )
    parser.add_argument(
        "--prefer_hardlink",
        action="store_true",
        help="Use hardlinks when possible to speed up ingestion and save disk space",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    out_dir = Path(args.out_dir)

    if args.clear_out_dir and out_dir.exists():
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    ingest(
        source_dir=source_dir,
        out_dir=out_dir,
        mode=args.mode,
        prefer_hardlink=args.prefer_hardlink,
    )


if __name__ == "__main__":
    main()