import argparse
import json
from datetime import datetime
from pathlib import Path


def load_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_metric(metrics: dict, key: str):
    if not metrics:
        return "N/A"
    value = metrics.get(key)
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def build_markdown(training_summary, eval_report):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    metrics = (eval_report or {}).get("metrics", {})
    classes = (eval_report or {}).get("class_names", [])

    lines = [
        "# Food Spoilage Detection - Project Report",
        "",
        f"Generated: {now}",
        "",
        "## 1. Objective",
        "Build an image-only deep learning system to detect food spoilage without sensor data.",
        "",
        "## 2. System Overview",
        "1. Data collection and cleaning from Kaggle datasets",
        "2. Train/validation/test split generation",
        "3. Transfer learning with MobileNetV3-Small",
        "4. Test-set evaluation with full metrics",
        "5. Interactive Streamlit demo for real-time predictions",
        "",
        "## 3. Model Configuration",
        "- Backbone: MobileNetV3-Small (ImageNet pretrained)",
        "- Learning approach: transfer learning",
        "- Output classes: " + (", ".join(classes) if classes else "N/A"),
        "",
        "## 4. Validation Summary",
        f"- Best validation accuracy: {(training_summary or {}).get('best_val_acc', 'N/A')}",
        f"- Best validation loss: {(training_summary or {}).get('best_val_loss', 'N/A')}",
        f"- Epochs completed: {(training_summary or {}).get('epochs_ran', 'N/A')}",
        "",
        "## 5. Test Metrics",
        f"- Accuracy: {safe_metric(metrics, 'accuracy')}",
        f"- Precision (macro): {safe_metric(metrics, 'precision_macro')}",
        f"- Recall (macro): {safe_metric(metrics, 'recall_macro')}",
        f"- F1 (macro): {safe_metric(metrics, 'f1_macro')}",
        f"- F1 (weighted): {safe_metric(metrics, 'f1_weighted')}",
        "",
        "## 6. Artifacts",
        "- Model checkpoint: checkpoints/best_model.pt",
        "- Training history: reports/train_history.csv",
        "- Evaluation report: reports/evaluation_report.json",
        "- Confusion matrix: reports/confusion_matrix.png",
        "- Explainability sample: reports/saliency_example.png",
        "",
        "## 7. Professional Notes",
        "1. Use test-set metrics (not only validation metrics) in final presentation.",
        "2. Discuss failure cases using confusion matrix and low-confidence examples.",
        "3. For final improvement, expand data diversity and include hard samples.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate markdown project report from training/evaluation artifacts.")
    parser.add_argument("--training_summary", type=str, default="reports/training_summary.json")
    parser.add_argument("--evaluation_report", type=str, default="reports/evaluation_report.json")
    parser.add_argument("--output", type=str, default="reports/project_report.md")
    args = parser.parse_args()

    training_summary = load_json(Path(args.training_summary))
    eval_report = load_json(Path(args.evaluation_report))

    out_md = build_markdown(training_summary, eval_report)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out_md, encoding="utf-8")

    print(f"Project report generated: {out_path}")


if __name__ == "__main__":
    main()