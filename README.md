# Food Spoilage Detection (Image-Only, Professional Baseline)

This repository implements a vision-driven deep learning workflow for food spoilage detection using only RGB images (no sensors).

The system is built for real project delivery with:

- reproducible training
- class-imbalance handling
- early stopping and LR scheduling
- full test evaluation (accuracy, precision, recall, F1, confusion matrix)
- explainability artifact generation (saliency map)
- automatic markdown project report generation
- interactive Streamlit demo

## 1) Project Structure

- `src/prepare_split.py`: Split raw images into train/val/test
- `src/model_utils.py`: Shared model/transforms/checkpoint utilities
- `src/train.py`: Professional training pipeline (MobileNetV3 transfer learning)
- `src/evaluate.py`: Test-set evaluation and report artifact generation
- `src/explain.py`: Explainability map generation for a sample image
- `src/generate_report.py`: Auto-generate submission-ready markdown report
- `src/infer.py`: CLI inference with top-k predictions
- `app.py`: Streamlit app for demo and confidence-based decisioning
- `reports/`: Generated metrics reports and plots
- `checkpoints/`: Saved best model weights

## 2) Environment Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 3) Dataset Requirements

Expected raw layout:

```text
data/raw/
  fresh/
    img1.jpg
  spoiled/
    img2.jpg
```

You can also use multi-class layouts (for example, `fresh_apple`, `rotten_apple`, `fresh_banana`, `rotten_banana`) as long as each class is a folder.

## 4) Kaggle Dataset Suggestions

Search Kaggle with these titles:

1. Fruits Fresh and Rotten for Classification
2. Fresh and Rotten Fruits Dataset
3. Fruit and Vegetable Freshness Classification
4. Rotten vs Fresh Fruits and Vegetables

Selection guidance:

1. Prefer balanced classes.
2. Prefer real-world lighting/background variation.
3. Avoid heavy duplicates and blurry samples.
4. Keep a separate unseen test split.

## 5) End-to-End Workflow

### Step A: Ingest provided dataset folders

If your dataset is already in `dataset/Train` and `dataset/Test`, first consolidate it:

```bash
python src/ingest_dataset.py --source_dir dataset --out_dir data/raw --mode binary --clear_out_dir
```

This maps all `fresh*` folders to class `fresh` and all `rotten*` folders to class `spoiled`.

### Step B: Create train/val/test split

```bash
python src/prepare_split.py --raw_dir data/raw --out_dir data/processed --train_ratio 0.7 --val_ratio 0.2 --seed 42
```

### Step C: Train model

```bash
python src/train.py --data_dir data/processed --epochs 20 --batch_size 16 --lr 0.001 --patience 4
```

Training outputs:

- `checkpoints/best_model.pt`
- `reports/train_history.csv`
- `reports/training_summary.json`

### Step D: Evaluate on test split

```bash
python src/evaluate.py --data_dir data/processed --checkpoint checkpoints/best_model.pt
```

Evaluation outputs:

- `reports/evaluation_report.json`
- `reports/confusion_matrix.png`

### Step E: Single image inference

```bash
python src/infer.py --image path_to_image.jpg --checkpoint checkpoints/best_model.pt --top_k 3
```

### Step F: Explainability artifact

```bash
python src/explain.py --image path_to_image.jpg --checkpoint checkpoints/best_model.pt --output reports/saliency_example.png
```

### Step G: Generate final markdown project report

```bash
python src/generate_report.py --training_summary reports/training_summary.json --evaluation_report reports/evaluation_report.json --output reports/project_report.md
```

### Step H: Demo app

```bash
streamlit run app.py
```

## 10) Deploy as a Website

### Option A: Streamlit Community Cloud (fastest)

1. Push this repository to GitHub.
2. Make sure `checkpoints/best_model.pt` is committed so the app can load the model.
3. Open Streamlit Community Cloud and click **New app**.
4. Select your repo and set the main file to `app.py`.
5. Deploy.

Notes:

- Python version is pinned by `runtime.txt`.
- If `reports/evaluation_report.json` is missing in deployment, the app still runs and prediction works.

### Option B: Render (Docker deploy)

This repo includes `Dockerfile` and `render.yaml`.

1. Push repository to GitHub.
2. In Render, create a new **Web Service** from the repository.
3. Render will detect `render.yaml` and use Docker automatically.
4. Deploy and open the generated public URL.

### Local Docker Run

```bash
docker build -t food-spoilage-app .
docker run -p 8501:8501 food-spoilage-app
```

Then open `http://localhost:8501`.

## 6) Model Choice

- Backbone: MobileNetV3-Small (ImageNet pretrained)
- Approach: Transfer learning
- Fast mode: Frozen backbone (default) for quick convergence
- Optional: `--unfreeze_backbone` for full fine-tuning when you have more training time

## 7) Metrics to Report (Professional Submission)

Always report:

1. Validation accuracy (best checkpoint)
2. Test accuracy
3. Macro F1 (important when classes are imbalanced)
4. Per-class precision and recall
5. Confusion matrix analysis (failure modes)
6. Explainability example to justify model focus regions

## 8) High-Score Submission Checklist

1. Include model architecture and transfer-learning rationale.
2. Show class distribution before training.
3. Report both validation and test metrics.
4. Include confusion matrix and one explainability artifact.
5. Demonstrate the Streamlit app live with 3-5 unseen images.
6. Add a short error analysis section (false positive and false negative examples).
7. Attach generated report file `reports/project_report.md` in final submission.



For production-grade performance, plan additional work (more data, stronger augmentation strategy, calibration, and deployment monitoring).
