from pathlib import Path
import json

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from src.model_utils import get_transforms, load_checkpoint


@st.cache_resource
def load_model(checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, img_size, _ = load_checkpoint(Path(checkpoint_path), device)
    preprocess = get_transforms(img_size, train_mode=False)
    return model, class_names, preprocess, device


def load_json(path: Path):
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def to_pct(x: float) -> str:
    return f"{100.0 * float(x):.2f}%"


def confidence_level(conf: float, threshold: float):
    if conf >= max(0.9, threshold):
        return "High confidence"
    if conf >= threshold:
        return "Moderate confidence"
    return "Low confidence"


st.set_page_config(page_title="Food Spoilage Detection", page_icon="FS", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Sora:wght@600;700;800&display=swap');

:root {
    --bg: #edf4f2;
    --surface: #ffffff;
    --surface-soft: #f7fbfa;
    --ink: #0a1f1d;
    --muted: #3f5a57;
    --line: #d6e5e1;
    --brand-deep: #0b5f5a;
    --brand: #0f766e;
    --brand-bright: #22b8a5;
    --mint: #8ee9db;
}

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

h1, h2, h3, h4, .subhead {
    font-family: 'Sora', sans-serif;
}

.stApp {
    background:
      radial-gradient(1100px 560px at 14% -14%, rgba(34, 184, 165, 0.22), transparent 55%),
      radial-gradient(900px 500px at 92% -8%, rgba(11, 95, 90, 0.2), transparent 55%),
      linear-gradient(140deg, rgba(255, 255, 255, 0.72), rgba(237, 244, 242, 0.94)),
      var(--bg);
    color: var(--ink);
}

[data-testid="stAppViewContainer"] .main {
    animation: riseIn 460ms ease-out;
}

[data-testid="stAppViewContainer"] .main * {
    color: var(--ink);
}

.hero {
    position: relative;
    overflow: hidden;
    background: linear-gradient(128deg, #0b5f5a 0%, #0f766e 55%, #18a092 100%);
    color: #ffffff !important;
    border-radius: 22px;
    padding: 1.4rem 1.5rem;
    box-shadow: 0 18px 32px rgba(12, 84, 78, 0.24);
    margin-bottom: 1rem;
    border: 1px solid rgba(255, 255, 255, 0.18);
}

.hero::before,
.hero::after {
    content: "";
    position: absolute;
    border-radius: 50%;
    pointer-events: none;
}

.hero::before {
    width: 280px;
    height: 280px;
    right: -100px;
    top: -120px;
    background: radial-gradient(circle, rgba(194, 252, 243, 0.35), transparent 68%);
}

.hero::after {
    width: 220px;
    height: 220px;
    right: 200px;
    bottom: -140px;
    background: radial-gradient(circle, rgba(166, 243, 230, 0.26), transparent 68%);
}

.hero h1 {
    margin: 0;
    color: #ffffff !important;
    font-size: 2rem;
    font-weight: 800;
    line-height: 1.2;
}

.hero p {
    margin: 0.6rem 0 0 0;
    color: #d8faf4 !important;
    font-weight: 600;
    max-width: 760px;
}

.chip {
    display: inline-block;
    border: 1px solid rgba(255, 255, 255, 0.35);
    background: rgba(255, 255, 255, 0.14);
    backdrop-filter: blur(3px);
    color: #ffffff !important;
    border-radius: 999px;
    padding: 0.29rem 0.72rem;
    font-size: 0.76rem;
    font-weight: 700;
    margin-right: 0.35rem;
    margin-bottom: 0.25rem;
}

.overview-strip {
    display: grid;
    grid-template-columns: repeat(3, minmax(120px, 1fr));
    gap: 0.65rem;
    margin-bottom: 0.85rem;
}

.overview-card {
    border-radius: 12px;
    background: linear-gradient(170deg, rgba(255, 255, 255, 0.9), rgba(246, 252, 250, 0.96));
    border: 1px solid var(--line);
    padding: 0.65rem 0.72rem;
    box-shadow: 0 6px 14px rgba(13, 63, 59, 0.06);
}

.overview-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--muted);
    font-weight: 700;
}

.overview-value {
    margin-top: 0.2rem;
    font-family: 'Sora', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--ink);
}

.kpi-grid {
    display: grid;
    grid-template-columns: repeat(5, minmax(130px, 1fr));
    gap: 0.72rem;
    margin-top: 0.5rem;
    margin-bottom: 0.85rem;
}

.kpi-card {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 0.72rem 0.8rem;
    box-shadow: 0 8px 16px rgba(7, 48, 44, 0.06);
    transition: transform 180ms ease, box-shadow 180ms ease;
}

.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 20px rgba(7, 48, 44, 0.11);
}

.kpi-label {
    color: var(--muted);
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.kpi-value {
    color: var(--ink);
    font-size: 1.45rem;
    font-weight: 800;
    margin-top: 0.24rem;
}

.subhead {
    color: var(--ink);
    font-size: 1.1rem;
    font-weight: 800;
    margin-top: 0.55rem;
    margin-bottom: 0.3rem;
}

.panel {
    background: linear-gradient(172deg, #ffffff 4%, #f8fcfb 100%);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 0.85rem 0.95rem;
    box-shadow: 0 8px 18px rgba(2, 47, 43, 0.08);
}

.pred-card {
    background: linear-gradient(165deg, #f6fcfb 0%, #ffffff 100%);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 0.82rem 0.9rem;
    margin-bottom: 0.5rem;
    box-shadow: 0 8px 18px rgba(2, 47, 43, 0.08);
}

.pred-title {
    color: var(--ink);
    font-size: 1.52rem;
    font-weight: 800;
    margin: 0;
}

.pred-sub {
    color: var(--muted);
    margin: 0.24rem 0 0 0;
    font-size: 0.98rem;
    font-weight: 700;
}

.table-wrap {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 0.45rem 0.72rem;
    box-shadow: 0 8px 16px rgba(7, 48, 44, 0.06);
}

.table-wrap table {
    width: 100%;
    border-collapse: collapse;
}

.table-wrap th,
.table-wrap td {
    padding: 0.4rem 0.25rem;
    border-bottom: 1px solid #e8f1ee;
    color: var(--ink);
    text-align: left;
}

.table-wrap tr:last-child td {
    border-bottom: none;
}

div[data-testid="stAlert"] * {
    color: var(--ink) !important;
    opacity: 1 !important;
}

div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #093835, #0c5e58);
}

div[data-testid="stSidebar"] * {
    color: #f8fafc !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: #ffffff !important;
    border: 2px dashed #8aa5a0 !important;
    border-radius: 14px !important;
    box-shadow: 0 8px 20px rgba(10, 64, 59, 0.06);
}

[data-testid="stFileUploaderDropzone"] * {
    color: #0f172a !important;
    opacity: 1 !important;
}

[data-testid="stFileUploaderDropzone"] button,
.stButton > button,
button[kind="secondary"] {
    background: linear-gradient(120deg, #0d6f69, #11a08f) !important;
    color: #ffffff !important;
    border: 1px solid #0b615a !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
}

[data-testid="stFileUploaderDropzone"] button:hover,
.stButton > button:hover,
button[kind="secondary"]:hover {
    background: linear-gradient(120deg, #0f877d, #1fbda9) !important;
    color: #ffffff !important;
}

div[data-testid="stProgressBar"] > div > div > div > div {
    background: linear-gradient(90deg, #0d7f74, #1fbda9) !important;
}

@keyframes riseIn {
    from {
        opacity: 0;
        transform: translateY(8px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@media (max-width: 960px) {
    .hero h1 { font-size: 1.6rem; }
    .kpi-grid { grid-template-columns: repeat(2, minmax(130px, 1fr)); }
    .overview-strip { grid-template-columns: 1fr; }
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <span class="chip">Freshness AI</span>
    <span class="chip">Food Quality Screening</span>
    <span class="chip">Vision Model Monitoring</span>
  <h1>Food Spoilage Detection Dashboard</h1>
    <p>Fast image-based freshness prediction with confidence analytics and cleaner operational insights.</p>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Configuration")
    checkpoint = st.text_input("Checkpoint path", value="checkpoints/best_model.pt")
    report_json = st.text_input("Evaluation report path", value="reports/evaluation_report.json")
    train_summary_json = st.text_input("Training summary path", value="reports/training_summary.json")
    threshold = st.slider("Confidence threshold", min_value=0.50, max_value=0.99, value=0.70, step=0.01)
    st.caption("Higher threshold means stricter acceptance of predictions.")

if not Path(checkpoint).exists():
    st.error("Checkpoint not found. Please train or point to a valid model path.")
    st.stop()

model, class_names, preprocess, device = load_model(checkpoint)
report_data = load_json(Path(report_json))
train_summary = load_json(Path(train_summary_json))

st.markdown(
        f"""
<div class="overview-strip">
    <div class="overview-card"><div class="overview-label">Inference Device</div><div class="overview-value">{str(device).upper()}</div></div>
    <div class="overview-card"><div class="overview-label">Classes Loaded</div><div class="overview-value">{len(class_names)}</div></div>
    <div class="overview-card"><div class="overview-label">Decision Threshold</div><div class="overview-value">{int(threshold * 100)}%</div></div>
</div>
""",
        unsafe_allow_html=True,
)

st.markdown('<div class="subhead">Model Performance Snapshot</div>', unsafe_allow_html=True)

if report_data and "metrics" in report_data:
    m = report_data["metrics"]
    metrics_html = f"""
    <div class="kpi-grid">
      <div class="kpi-card"><div class="kpi-label">Test Accuracy</div><div class="kpi-value">{to_pct(m.get('accuracy', 0.0))}</div></div>
      <div class="kpi-card"><div class="kpi-label">Macro F1</div><div class="kpi-value">{to_pct(m.get('f1_macro', 0.0))}</div></div>
      <div class="kpi-card"><div class="kpi-label">Macro Precision</div><div class="kpi-value">{to_pct(m.get('precision_macro', 0.0))}</div></div>
      <div class="kpi-card"><div class="kpi-label">Macro Recall</div><div class="kpi-value">{to_pct(m.get('recall_macro', 0.0))}</div></div>
      <div class="kpi-card"><div class="kpi-label">Weighted F1</div><div class="kpi-value">{to_pct(m.get('f1_weighted', 0.0))}</div></div>
    </div>
    """
    st.markdown(metrics_html, unsafe_allow_html=True)

    if train_summary:
        train_html = f"""
        <div class="kpi-grid" style="grid-template-columns: repeat(3, minmax(150px, 1fr));">
          <div class="kpi-card"><div class="kpi-label">Best Val Accuracy</div><div class="kpi-value">{to_pct(train_summary.get('best_val_acc', 0.0))}</div></div>
          <div class="kpi-card"><div class="kpi-label">Best Val Loss</div><div class="kpi-value">{train_summary.get('best_val_loss', 0.0):.4f}</div></div>
          <div class="kpi-card"><div class="kpi-label">Epochs Run</div><div class="kpi-value">{train_summary.get('epochs_ran', '-')}</div></div>
        </div>
        """
        st.markdown(train_html, unsafe_allow_html=True)

    cls_report = report_data.get("classification_report", {})
    fresh_info = cls_report.get("fresh", {})
    spoiled_info = cls_report.get("spoiled", {})
    if fresh_info and spoiled_info:
        st.markdown('<div class="subhead">Class-wise Quality</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
<div class="table-wrap">
  <table>
    <thead>
      <tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>
    </thead>
    <tbody>
      <tr><td>fresh</td><td>{100 * fresh_info.get('precision', 0.0):.2f}%</td><td>{100 * fresh_info.get('recall', 0.0):.2f}%</td><td>{100 * fresh_info.get('f1-score', 0.0):.2f}%</td><td>{int(fresh_info.get('support', 0))}</td></tr>
      <tr><td>spoiled</td><td>{100 * spoiled_info.get('precision', 0.0):.2f}%</td><td>{100 * spoiled_info.get('recall', 0.0):.2f}%</td><td>{100 * spoiled_info.get('f1-score', 0.0):.2f}%</td><td>{int(spoiled_info.get('support', 0))}</td></tr>
    </tbody>
  </table>
</div>
""",
            unsafe_allow_html=True,
        )
else:
    st.info("Evaluation metrics not found. Uploading and prediction still work, but analytics need reports/evaluation_report.json.")

st.info("Upload an image by clicking Browse files in the box below, or drag and drop a file.")
uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp", "bmp"])

if uploaded is not None:
    col_left, col_right = st.columns([1.05, 1.0])
    image = Image.open(uploaded).convert("RGB")

    with col_left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.image(image, caption="Input image", width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    x = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1).squeeze(0)

    conf, idx = torch.max(probs, dim=0)
    label = class_names[idx.item()]
    conf_val = float(conf.item())
    c_level = confidence_level(conf_val, threshold)

    with col_right:
        st.markdown(
            f"""
<div class="pred-card">
  <p class="pred-title">Prediction: {label.title()}</p>
  <p class="pred-sub">Confidence: {to_pct(conf_val)} • {c_level}</p>
</div>
""",
            unsafe_allow_html=True,
        )

        st.progress(min(max(conf_val, 0.0), 1.0))

        if c_level == "High confidence":
            st.success("Prediction is highly stable for this image.")
        elif c_level == "Moderate confidence":
            st.info("Prediction is acceptable; verify image quality for best reliability.")
        else:
            st.warning("Low confidence: recapture with better lighting and clearer focus.")

        k = min(5, len(class_names))
        top_probs, top_idx = torch.topk(probs, k=k)

        st.markdown('<div class="subhead">Top Predictions</div>', unsafe_allow_html=True)
        for i, p in zip(top_idx.tolist(), top_probs.tolist()):
            st.write(f"- {class_names[i]}: {to_pct(p)}")

        st.markdown('<div class="subhead">Class Probability Distribution</div>', unsafe_allow_html=True)
        prob_pairs = sorted(
            zip(class_names, probs.detach().cpu().numpy().tolist()),
            key=lambda pair: pair[1],
            reverse=True,
        )
        for cname, p in prob_pairs:
            st.write(f"{cname}: {to_pct(p)}")
            st.progress(min(max(float(p), 0.0), 1.0))

        st.caption("Adjust threshold in sidebar to trade confidence strictness vs prediction coverage.")
