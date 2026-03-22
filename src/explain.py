import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from src.model_utils import get_transforms, load_checkpoint
except ModuleNotFoundError:
    from model_utils import get_transforms, load_checkpoint


def overlay_heatmap(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45):
    cmap = plt.get_cmap("jet")
    heatmap_rgb = cmap(heatmap)[..., :3]
    overlay = (1 - alpha) * image_rgb + alpha * heatmap_rgb
    return np.clip(overlay, 0.0, 1.0)


def generate_saliency(image_path: Path, checkpoint_path: Path, out_path: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, img_size, _ = load_checkpoint(checkpoint_path, device)
    preprocess = get_transforms(img_size, train_mode=False)

    image = Image.open(image_path).convert("RGB")
    image_resized = image.resize((img_size, img_size))
    image_np = np.asarray(image_resized).astype(np.float32) / 255.0

    x = preprocess(image).unsqueeze(0).to(device)
    x.requires_grad_(True)

    logits = model(x)
    probs = F.softmax(logits, dim=1)
    pred_idx = int(torch.argmax(probs, dim=1).item())
    pred_label = class_names[pred_idx]
    pred_conf = float(probs[0, pred_idx].item())

    score = logits[0, pred_idx]
    model.zero_grad(set_to_none=True)
    score.backward()

    grads = x.grad.detach().abs().squeeze(0)
    saliency = torch.max(grads, dim=0).values
    saliency = saliency.cpu().numpy()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

    overlay = overlay_heatmap(image_np, saliency, alpha=0.45)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    axes[0].imshow(image_np)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(saliency, cmap="inferno")
    axes[1].set_title("Saliency Map")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay | {pred_label} ({pred_conf:.3f})")
    axes[2].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    print("Explainability artifact generated")
    print(f"Prediction: {pred_label}")
    print(f"Confidence: {pred_conf:.4f}")
    print(f"Saved saliency figure: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate saliency-map explainability artifact for a sample image.")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--output", type=str, default="reports/saliency_example.png")
    args = parser.parse_args()

    generate_saliency(Path(args.image), Path(args.checkpoint), Path(args.output))


if __name__ == "__main__":
    main()