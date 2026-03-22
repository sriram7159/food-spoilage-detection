import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
try:
    from src.model_utils import get_transforms, load_checkpoint
except ModuleNotFoundError:
    from model_utils import get_transforms, load_checkpoint


def predict(image_path: Path, model, class_names, preprocess, device):
    image = Image.open(image_path).convert("RGB")
    x = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).squeeze(0)

    return probs


def main():
    parser = argparse.ArgumentParser(description="Run inference for food spoilage model.")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, img_size, _ = load_checkpoint(Path(args.checkpoint), device)
    preprocess = get_transforms(img_size, train_mode=False)
    probs = predict(Path(args.image), model, class_names, preprocess, device)

    conf, idx = torch.max(probs, dim=0)
    print(f"Prediction: {class_names[idx.item()]}")
    print(f"Confidence: {conf.item():.4f}")

    k = min(args.top_k, len(class_names))
    top_probs, top_idx = torch.topk(probs, k=k)
    print("Top predictions:")
    for rank, (p, i) in enumerate(zip(top_probs.tolist(), top_idx.tolist()), start=1):
        print(f"{rank}. {class_names[i]}: {p:.4f}")


if __name__ == "__main__":
    main()
