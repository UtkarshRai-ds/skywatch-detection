"""SkyWatch — Satellite Object Detection Portfolio Dashboard.

Five-page Streamlit application showcasing the full NWPU VHR-10 detection
pipeline: EDA, training, model comparison, live inference, and project narrative.

Run from the project root::

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import base64
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Path bootstrap ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loader import NWPU_CLASSES  # noqa: E402

# ── Design tokens ──────────────────────────────────────────────────────────────
BG        = "#0d1117"
CARD_BG   = "#161b22"
BORDER    = "#30363d"
ACCENT    = "#00d4aa"
TEXT      = "#e6edf3"
MUTED     = "#8b949e"
RED       = "#f85149"
GREEN     = "#3fb950"
BLUE      = "#58a6ff"
ORANGE    = "#f0883e"
YELLOW    = "#e3b341"

PALETTE: list[str] = [
    "#00d4aa", "#58a6ff", "#f0883e", "#f85149", "#3fb950",
    "#79c0ff", "#ffa657", "#ff7b72", "#7ee787", "#d2a8ff",
]
CLASS_COLORS: dict[str, str] = {
    name: PALETTE[i % len(PALETTE)] for i, name in enumerate(NWPU_CLASSES)
}

# Plotly layout defaults reused across all charts
_LAYOUT: dict[str, Any] = dict(
    paper_bgcolor=BG,
    plot_bgcolor=CARD_BG,
    font=dict(color=TEXT, family="Inter, sans-serif", size=12),
    margin=dict(l=12, r=12, t=40, b=12),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER),
    hoverlabel=dict(bgcolor=CARD_BG, font_color=TEXT),
)

DATA_DIR   = ROOT / "data"
MODELS_DIR = ROOT / "models"


# ── Data loading ────────────────────────────────────────────────────────────────

@st.cache_data
def load_dataset_stats() -> dict[str, Any]:
    """Load data/dataset_stats.json (loader + quality + eda results)."""
    p = DATA_DIR / "dataset_stats.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


@st.cache_data
def load_class_stats() -> dict[str, Any]:
    """Load data/class_stats.json (model results + comparison)."""
    p = DATA_DIR / "class_stats.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


@st.cache_data
def load_training_history() -> dict[str, Any]:
    """Load data/training_history.json (epoch-by-epoch metrics)."""
    p = DATA_DIR / "training_history.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


@st.cache_data
def load_image_b64(path: Path) -> str | None:
    """Read an image file and return a base64-encoded data URL."""
    if not path.exists():
        return None
    with path.open("rb") as fh:
        data = base64.b64encode(fh.read()).decode()
    suffix = path.suffix.lower().lstrip(".")
    mime   = "jpeg" if suffix in ("jpg", "jpeg") else suffix
    return f"data:image/{mime};base64,{data}"


@st.cache_resource
def load_yolo_model(model_name: str):
    """Load a YOLOv8 model with Streamlit resource caching (one instance per model)."""
    from ultralytics import YOLO
    pt = MODELS_DIR / f"{model_name}_best.pt"
    if not pt.exists():
        return None
    return YOLO(str(pt))


# ── CSS ─────────────────────────────────────────────────────────────────────────

def _inject_css() -> None:
    st.markdown(f"""
    <style>
    /* ── global background ── */
    [data-testid="stAppViewContainer"] {{ background: {BG}; }}
    [data-testid="stSidebar"]          {{ background: {CARD_BG}; border-right: 1px solid {BORDER}; }}
    [data-testid="stHeader"]           {{ background: {BG}; }}
    .block-container                   {{ padding-top: 1.5rem; padding-bottom: 2rem; }}

    /* ── text ── */
    h1, h2, h3, h4, p, li, label      {{ color: {TEXT} !important; }}
    .stMarkdown p                      {{ color: {TEXT}; }}

    /* ── KPI card ── */
    .kpi-card {{
        background: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 20px 16px;
        text-align: center;
    }}
    .kpi-value  {{ font-size: 2.1em; font-weight: 700; color: {ACCENT}; line-height: 1.1; }}
    .kpi-label  {{ font-size: 0.85em; color: {MUTED}; margin-top: 4px; }}

    /* ── tech badge ── */
    .badge {{
        display: inline-block;
        background: #1f2d3d;
        border: 1px solid #2d4a6e;
        border-radius: 4px;
        padding: 3px 10px;
        margin: 3px 2px;
        font-size: 0.8em;
        color: {BLUE};
        font-weight: 500;
    }}

    /* ── section divider ── */
    .section-title {{
        font-size: 1.25em;
        font-weight: 600;
        color: {ACCENT};
        border-bottom: 1px solid {BORDER};
        padding-bottom: 6px;
        margin-top: 28px;
        margin-bottom: 12px;
    }}

    /* ── comparison table winner ── */
    .winner {{ color: {GREEN}; font-weight: 700; }}
    .loser  {{ color: {MUTED}; }}

    /* ── sidebar nav ── */
    .stRadio > div {{ gap: 4px; }}

    /* ── hide Streamlit branding ── */
    #MainMenu, footer {{ visibility: hidden; }}
    </style>
    """, unsafe_allow_html=True)


# ── Reusable UI components ──────────────────────────────────────────────────────

def _section(title: str, subtitle: str = "") -> None:
    sub_html = f'<div style="color:{MUTED};font-size:0.88em;margin-top:2px">{subtitle}</div>' if subtitle else ""
    st.markdown(f'<div class="section-title">{title}</div>{sub_html}', unsafe_allow_html=True)


def _kpi_row(cards: list[tuple[str, str]]) -> None:
    """Render a row of KPI cards. Each card is (value, label)."""
    cols = st.columns(len(cards))
    for col, (value, label) in zip(cols, cards):
        col.markdown(
            f'<div class="kpi-card"><div class="kpi-value">{value}</div>'
            f'<div class="kpi-label">{label}</div></div>',
            unsafe_allow_html=True,
        )


def _badges(names: list[str]) -> None:
    html = " ".join(f'<span class="badge">{n}</span>' for n in names)
    st.markdown(html, unsafe_allow_html=True)


def _fig(fig: go.Figure, height: int = 380) -> None:
    """Render a Plotly figure with consistent dark settings."""
    fig.update_layout(height=height, **_LAYOUT)
    st.plotly_chart(fig, use_column_width=True, config={"displayModeBar": False})


def _img_row(image_paths: list[Path], captions: list[str] | None = None, cols: int = 3) -> None:
    """Display a grid of images loaded from disk."""
    for row_start in range(0, len(image_paths), cols):
        row_paths = image_paths[row_start : row_start + cols]
        row_caps  = (captions[row_start : row_start + cols] if captions else [None] * len(row_paths))
        c = st.columns(cols)
        for col, path, cap in zip(c, row_paths, row_caps):
            if path.exists():
                col.image(str(path), caption=cap, use_column_width=True)
            else:
                col.warning(f"Missing: {path.name}")


# ── Chart builders ──────────────────────────────────────────────────────────────

def _chart_class_dist(dist: dict) -> go.Figure:
    order  = dist["class_order"]
    counts = [dist["instances_per_class"][c] for c in order]
    colors = [CLASS_COLORS.get(c, ACCENT) for c in order]

    fig = go.Figure(go.Bar(
        x=counts, y=order, orientation="h",
        marker_color=colors,
        text=counts, textposition="outside",
        hovertemplate="%{y}: %{x} instances<extra></extra>",
    ))
    fig.update_layout(title="Instance Count per Class", xaxis_title="Annotations",
                      yaxis=dict(autorange="reversed"))
    return fig


def _chart_imbalance(imb: dict, dist: dict) -> go.Figure:
    names   = NWPU_CLASSES
    counts  = [dist["instances_per_class"][c] for c in names]
    flagged = [imb["undertrained_flags"].get(c, False) for c in names]
    colors  = [RED if f else GREEN for f in flagged]

    fig = go.Figure(go.Bar(
        x=names, y=counts,
        marker_color=colors,
        text=counts, textposition="outside",
        hovertemplate="%{x}: %{y}<extra></extra>",
    ))
    mean_val = sum(counts) / len(counts)
    fig.add_hline(y=mean_val, line_dash="dash", line_color=ORANGE,
                  annotation_text=f"mean = {mean_val:.0f}", annotation_font_color=ORANGE)
    fig.update_layout(title=f"Class Imbalance — ratio {imb['imbalance_ratio']}×  (red = undertrained)",
                      xaxis_tickangle=-30, yaxis_title="Instances")
    return fig


def _chart_nwpu_pie() -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=["Positive images (annotated)", "Background images (excluded from export)"],
        values=[650, 150],
        marker_colors=[ACCENT, BORDER],
        hole=0.55,
        textinfo="label+percent",
        hovertemplate="%{label}: %{value}<extra></extra>",
    ))
    fig.update_layout(
        title="Original NWPU VHR-10 Dataset Split",
        annotations=[dict(text="800<br>total", x=0.5, y=0.5, font_size=15,
                          font_color=TEXT, showarrow=False)],
    )
    return fig


def _chart_bbox_scatter(scatter: list[dict]) -> go.Figure:
    fig = go.Figure()
    for cls in NWPU_CLASSES:
        pts = [s for s in scatter if s["class_name"] == cls]
        if not pts:
            continue
        fig.add_trace(go.Scatter(
            x=[p["w"] for p in pts], y=[p["h"] for p in pts],
            mode="markers", name=cls,
            marker=dict(color=CLASS_COLORS[cls], size=5, opacity=0.55),
            hovertemplate=f"{cls}<br>w=%{{x:.3f}}, h=%{{y:.3f}}<extra></extra>",
        ))
    fig.add_shape(type="line", x0=0, y0=0, x1=0.6, y1=0.6,
                  line=dict(color="white", dash="dash", width=1))
    fig.update_layout(title="BBox Width vs Height (normalised, n=1 000 sample)",
                      xaxis_title="Width", yaxis_title="Height")
    return fig


def _chart_aspect_ratio(scatter: list[dict]) -> go.Figure:
    fig = go.Figure()
    for cls in NWPU_CLASSES:
        ars = [s["ar"] for s in scatter if s["class_name"] == cls]
        if not ars:
            continue
        fig.add_trace(go.Violin(
            x=[cls] * len(ars), y=ars, name=cls,
            fillcolor=CLASS_COLORS[cls], line_color=CLASS_COLORS[cls],
            opacity=0.7, box_visible=True, meanline_visible=True,
            hoverinfo="y+name",
        ))
    fig.add_hline(y=1.0, line_dash="dash", line_color="white",
                  annotation_text="w = h", annotation_font_color=MUTED)
    fig.update_layout(title="Aspect Ratio (w/h) Distribution per Class",
                      yaxis_title="w / h", showlegend=False, xaxis_tickangle=-30)
    return fig


def _chart_area_box(scatter: list[dict]) -> go.Figure:
    fig = go.Figure()
    for cls in NWPU_CLASSES:
        areas = [s["area"] * 100 for s in scatter if s["class_name"] == cls]
        if not areas:
            continue
        fig.add_trace(go.Box(
            y=areas, name=cls,
            marker_color=CLASS_COLORS[cls], line_color=CLASS_COLORS[cls],
            boxmean=True,
        ))
    fig.update_layout(title="Relative BBox Area per Class (% of image area)",
                      yaxis_title="Area (%)", showlegend=False, xaxis_tickangle=-30)
    return fig


def _chart_spatial_heatmap(spatial: dict) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("All-class centre density", "Top-4 classes per-class centres"),
        horizontal_spacing=0.10,
    )
    fig.add_trace(
        go.Histogram2d(x=spatial["cx_all"], y=spatial["cy_all"],
                       nbinsx=30, nbinsy=30, colorscale="Viridis",
                       colorbar=dict(len=0.5, x=0.45)),
        row=1, col=1,
    )
    for cls in spatial["top4_classes"]:
        c = spatial["per_class_centers"].get(cls, {})
        if not c:
            continue
        fig.add_trace(
            go.Scatter(x=c["cx_list"], y=c["cy_list"], mode="markers", name=cls,
                       marker=dict(color=CLASS_COLORS.get(cls, ACCENT), size=4, opacity=0.4)),
            row=1, col=2,
        )
    for r, c in [(1, 1), (1, 2)]:
        fig.add_shape(type="rect", x0=0, y0=0, x1=1, y1=1,
                      line=dict(color="white", width=1), row=r, col=c)
    fig.update_xaxes(range=[-0.05, 1.05], title_text="cx", row=1, col=1)
    fig.update_xaxes(range=[-0.05, 1.05], title_text="cx", row=1, col=2)
    fig.update_yaxes(range=[-0.05, 1.05], title_text="cy")
    fig.update_layout(title="Spatial Distribution of BBox Centres", height=400)
    return fig


def _chart_cooc(cooc: dict) -> go.Figure:
    names  = cooc["class_names"]
    matrix = cooc["matrix_normalized"]
    fig = go.Figure(go.Heatmap(
        z=matrix, x=names, y=names,
        colorscale="Blues",
        text=[[f"{v:.2f}" for v in row] for row in matrix],
        texttemplate="%{text}",
        textfont=dict(size=9),
        colorbar=dict(title="Fraction"),
    ))
    fig.update_layout(
        title="Row-Normalised Co-occurrence Matrix",
        xaxis_tickangle=-40,
    )
    return fig


def _chart_resolution(res: dict) -> go.Figure:
    unique = res.get("unique_resolutions", {"640x640": res.get("sample_size", 280)})
    fig = go.Figure(go.Bar(
        x=list(unique.keys()), y=list(unique.values()),
        marker_color=ACCENT,
        text=list(unique.values()), textposition="outside",
    ))
    fig.update_layout(title="Image Resolution Distribution",
                      xaxis_title="Resolution", yaxis_title="Count")
    return fig


def _draw_yolo_labels(img: np.ndarray, label_path: Path) -> np.ndarray:
    """Draw YOLO bbox annotations onto an image array."""
    canvas = img.copy()
    if not label_path.exists():
        return canvas
    h, w = canvas.shape[:2]
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cid = int(parts[0])
        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        color_hex = CLASS_COLORS.get(NWPU_CLASSES[cid], "#ffffff")
        color_bgr = tuple(int(color_hex.lstrip("#")[i:i+2], 16) for i in (4, 2, 0))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color_bgr, 2)
        label = NWPU_CLASSES[cid]
        cv2.putText(canvas, label, (x1, max(y1 - 4, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color_bgr, 1, cv2.LINE_AA)
    return canvas


@st.cache_data
def _load_test_sample_images(n: int = 6) -> list[np.ndarray]:
    """Load n test images with YOLO ground-truth overlays."""
    test_img_dir = DATA_DIR / "raw" / "test" / "images"
    test_lbl_dir = DATA_DIR / "raw" / "test" / "labels"
    if not test_img_dir.exists():
        return []
    paths = sorted(test_img_dir.glob("*.jpg"))[:n]
    result = []
    for p in paths:
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        annotated = _draw_yolo_labels(frame, test_lbl_dir / (p.stem + ".txt"))
        result.append(annotated)
    return result


def _chart_per_class_map50(cs: dict) -> go.Figure:
    n_pc = cs["models"]["yolov8n"]["per_class_mAP50"]
    s_pc = cs["models"]["yolov8s"]["per_class_mAP50"]
    # Sort by YOLOv8n performance
    order = sorted(n_pc, key=n_pc.get, reverse=True)
    labels = [c.replace("_", " ") for c in order]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="YOLOv8n", x=labels,
                         y=[n_pc[c] for c in order],
                         marker_color=BLUE, text=[f"{n_pc[c]:.3f}" for c in order],
                         textposition="outside"))
    fig.add_trace(go.Bar(name="YOLOv8s", x=labels,
                         y=[s_pc.get(c, 0) for c in order],
                         marker_color=ACCENT, text=[f"{s_pc.get(c, 0):.3f}" for c in order],
                         textposition="outside"))
    fig.add_hline(y=0.5, line_dash="dash", line_color=ORANGE,
                  annotation_text="0.5 threshold", annotation_font_color=ORANGE)
    fig.update_layout(title="Per-Class mAP50 — YOLOv8n vs YOLOv8s",
                      barmode="group", yaxis=dict(range=[0, 1.12], title="mAP50"),
                      xaxis_tickangle=-30)
    return fig


def _chart_training_loss(th: dict) -> go.Figure:
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=("Box Loss", "Class Loss", "DFL Loss"),
                        horizontal_spacing=0.08)
    style = {
        "yolov8n": dict(color=BLUE,  dash="solid"),
        "yolov8s": dict(color=ACCENT, dash="dash"),
    }
    for col_idx, loss_key in enumerate(["train_box_loss", "train_cls_loss", "train_dfl_loss"], 1):
        for mname, history in th.items():
            epochs = [r["epoch"]    for r in history["history"]]
            vals   = [r[loss_key]   for r in history["history"]]
            fig.add_trace(
                go.Scatter(x=epochs, y=vals, name=mname,
                           line=dict(**style[mname], width=2),
                           showlegend=(col_idx == 1),
                           hovertemplate=f"{mname}<br>epoch=%{{x}}<br>loss=%{{y:.4f}}<extra></extra>"),
                row=1, col=col_idx,
            )
    fig.update_yaxes(title_text="Loss", col=1)
    fig.update_xaxes(title_text="Epoch")
    fig.update_layout(title="Training Loss Curves", height=340)
    return fig


def _chart_map50_curve(th: dict) -> go.Figure:
    fig = go.Figure()
    colors = {"yolov8n": BLUE, "yolov8s": ACCENT}
    for mname, history in th.items():
        epochs = [r["epoch"]  for r in history["history"]]
        maps   = [r["mAP50"]  for r in history["history"]]
        best_e = history["best_epoch"]
        best_m = next((r["mAP50"] for r in history["history"] if r["epoch"] == best_e), None)
        fig.add_trace(go.Scatter(
            x=epochs, y=maps, name=mname,
            line=dict(color=colors[mname], width=2.5),
            hovertemplate=f"{mname} ep%{{x}} → mAP50=%{{y:.4f}}<extra></extra>",
        ))
        if best_m is not None:
            fig.add_trace(go.Scatter(
                x=[best_e], y=[best_m], mode="markers", name=f"{mname} best",
                marker=dict(symbol="star", size=14, color=colors[mname],
                            line=dict(color="white", width=1)),
                showlegend=True,
                hovertemplate=f"Best epoch {best_e}: {best_m:.4f}<extra></extra>",
            ))
    fig.update_layout(title="mAP50 Progression per Epoch",
                      xaxis_title="Epoch", yaxis_title="mAP50",
                      yaxis=dict(range=[0, 0.85]))
    return fig


def _chart_speed_accuracy(tradeoff: list[dict]) -> go.Figure:
    fig = go.Figure()
    colors = {"yolov8n": BLUE, "yolov8s": ACCENT}
    for pt in tradeoff:
        name = pt["model_name"]
        fig.add_trace(go.Scatter(
            x=[pt["total_ms"]], y=[pt["mAP50"]],
            mode="markers+text",
            name=name,
            text=[name.upper()],
            textposition="top center",
            textfont=dict(size=12, color=colors.get(name, ACCENT)),
            marker=dict(
                size=pt["params_M"] * 3.5,
                color=colors.get(name, ACCENT),
                line=dict(color="white", width=1),
                opacity=0.85,
            ),
            hovertemplate=(
                f"<b>{name}</b><br>"
                f"mAP50 = {pt['mAP50']:.3f}<br>"
                f"Latency = {pt['total_ms']:.0f} ms (CPU)<br>"
                f"Params = {pt['params_label']}<br>"
                f"FPS = {pt['fps']:.1f}<extra></extra>"
            ),
        ))
    fig.update_layout(
        title="Speed vs Accuracy Trade-off  (bubble size ∝ model size)",
        xaxis_title="Total latency per image (ms, CPU)",
        yaxis_title="mAP50 (test set)",
        showlegend=False,
    )
    return fig


# ── Page 1 — Project Overview ───────────────────────────────────────────────────

def page_overview() -> None:
    st.markdown(
        f'<h1 style="font-size:2.6em;font-weight:800;color:{ACCENT};margin-bottom:0">'
        "🛰️ SkyWatch</h1>"
        f'<p style="font-size:1.25em;color:{MUTED};margin-top:4px">'
        "Satellite Object Detection · NWPU VHR-10 · YOLOv8</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    _kpi_row([
        ("800",    "Satellite Images"),
        ("10",     "Object Classes"),
        ("0.697",  "Best mAP50"),
        ("2.3 ms", "GPU Inference"),
    ])

    st.markdown("<br>", unsafe_allow_html=True)
    _section("Tech Stack")
    _badges([
        "Python 3.12", "YOLOv8", "Ultralytics", "OpenCV",
        "Weights & Biases", "Streamlit", "Plotly", "Roboflow",
    ])

    st.markdown("<br>", unsafe_allow_html=True)
    _section("Project Description")
    st.markdown(f"""
<div style="color:{TEXT};line-height:1.75;font-size:0.97em">

**SkyWatch** is an end-to-end satellite object detection system trained on the
<a href="https://universe.roboflow.com/yolo-7muwp/nwpu-vhr-10-sgj3z" style="color:{ACCENT}">NWPU VHR-10</a>
dataset — 800 very-high-resolution (VHR) images captured at 0.5–2 m/pixel resolution,
covering 10 geospatial object categories from airplanes and ships to bridges and stadiums.

The project follows a complete MLOps pipeline:

1. **Data Foundation** — structural validation, 5 quality checks, Pascal VOC → YOLO converter
2. **Exploratory Analysis** — class imbalance (11.46× bridge/ship ratio), tiny objects
   (median bbox area = 0.61%), spatial distribution, and co-occurrence patterns
3. **Training** — YOLOv8n and YOLOv8s trained for 50 epochs on Colab T4 GPU with
   Weights & Biases tracking, identical hyperparameters for fair comparison
4. **Evaluation** — test-set mAP50, per-class AP, speed benchmarks,
   confusion matrices, and F1 curves
5. **Dashboard** — this interactive Streamlit app for portfolio presentation
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    _section("Model Architecture Comparison")

    col1, col2 = st.columns(2)
    arch_rows = [
        ("Architecture",   "YOLOv8n",        "YOLOv8s"),
        ("Parameters",     "3.0 M",           "11.1 M"),
        ("Layers",         "73",              "73"),
        ("GFLOPs",         "8.1",             "28.5"),
        ("Depth mult.",    "0.33",            "0.33"),
        ("Width mult.",    "0.25",            "0.50"),
        ("mAP50 (test)",   "0.689",           "0.697"),
        ("mAP50-95",       "0.426",           "0.437"),
        ("CPU latency",    "57 ms / img",     "139 ms / img"),
        ("GPU latency",    "~2.3 ms / img",   "~4.3 ms / img"),
        ("Best epoch",     "37 / 50",         "40 / 50"),
    ]

    header_html = (
        f'<table style="width:100%;border-collapse:collapse;font-size:0.9em;color:{TEXT}">'
        f'<thead><tr>'
        + "".join(
            f'<th style="padding:8px 12px;border-bottom:2px solid {BORDER};'
            f'text-align:left;color:{ACCENT}">{h}</th>'
            for h in arch_rows[0]
        )
        + "</tr></thead><tbody>"
    )
    body_rows = ""
    for i, row in enumerate(arch_rows[1:]):
        bg = CARD_BG if i % 2 == 0 else BG
        body_rows += f'<tr style="background:{bg}">'
        for j, cell in enumerate(row):
            body_rows += (
                f'<td style="padding:7px 12px;border-bottom:1px solid {BORDER};'
                f'font-weight:{"600" if j == 0 else "400"};color:{"#c9d1d9" if j > 0 else TEXT}">'
                f"{cell}</td>"
            )
        body_rows += "</tr>"

    st.markdown(header_html + body_rows + "</tbody></table>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    _section("Links")
    st.markdown("""
| Resource | Link |
|---|---|
| 📓 W&B Dashboard | *[Replace with your W&B run URL]* |
| 💻 GitHub Repository | *[Replace with your GitHub repo URL]* |
| 🚀 Live Demo | *This app — deployed on Streamlit Community Cloud* |
| 📊 Dataset | [NWPU VHR-10 on Roboflow Universe](https://universe.roboflow.com/yolo-7muwp/nwpu-vhr-10-sgj3z) |
""")


# ── Page 2 — Explore the Data ───────────────────────────────────────────────────

def page_eda() -> None:
    ds = load_dataset_stats()
    if not ds:
        st.warning("data/dataset_stats.json not found. Run `python run_eda.py` first.")
        return

    eda  = ds.get("eda", {})
    dist = eda.get("class_distribution", {})
    imb  = eda.get("class_imbalance", {})
    bbox = eda.get("bbox_analysis", {})
    spat = eda.get("spatial_heatmap", {})
    cooc = eda.get("cooccurrence_matrix", {})
    res  = eda.get("image_resolution", {})
    lo   = ds.get("loader", {})

    st.title("Explore the Data")
    _kpi_row([
        (str(lo.get("total_images", 800)),       "Total images"),
        (str(dist.get("total_annotations", 4811)), "Annotations"),
        (f"{imb.get('imbalance_ratio', 11.46)}×",  "Imbalance ratio"),
        ("0.61%",                                   "Median bbox area"),
    ])

    # ── Class distribution ─────────────────────────────────────────────────────
    _section("Class Distribution", "Instance counts across all splits")
    if dist:
        col1, col2 = st.columns([2, 1])
        with col1:
            _fig(_chart_class_dist(dist), height=380)
        with col2:
            _fig(_chart_nwpu_pie(), height=380)
        st.caption(
            "ℹ️ The original NWPU VHR-10 has 650 positive + 150 background images. "
            "The Roboflow export used here contains all 800 annotated images — the 150 background "
            "images were excluded. "
            f"Bridge is the dominant class ({dist['instances_per_class'].get('bridge', 0)} instances, "
            f"{dist['instances_per_class'].get('bridge', 0) / dist.get('total_annotations', 1) * 100:.1f}% of all annotations)."
        )

    # ── Class imbalance ────────────────────────────────────────────────────────
    _section("Class Imbalance", "Classes below mean − 0.5σ are flagged as undertrained")
    if imb and dist:
        _fig(_chart_imbalance(imb, dist), height=360)
        st.caption(
            f"🔴 Undertrained classes: **{', '.join(imb.get('undertrained_classes', []))}**. "
            f"Bridge ({imb.get('dominant_class', 'bridge')}) is {imb.get('imbalance_ratio', 0)}× "
            f"more common than {imb.get('rarest_class', 'ship')}. "
            "Focal loss (γ ≥ 1.5) or class-weighted sampling is recommended."
        )

    # ── BBox geometry ──────────────────────────────────────────────────────────
    _section("Bounding Box Geometry", "Normalised coordinates — all values in [0, 1]")
    scatter = bbox.get("scatter_sample", [])
    if scatter:
        col1, col2 = st.columns(2)
        with col1:
            _fig(_chart_bbox_scatter(scatter), height=380)
        with col2:
            _fig(_chart_aspect_ratio(scatter), height=380)
        _fig(_chart_area_box(scatter), height=360)

        ws = bbox.get("width_stats", {})
        hs = bbox.get("height_stats", {})
        st.caption(
            f"Median bbox: w={ws.get('median', 0):.3f}, h={hs.get('median', 0):.3f}  "
            f"(median area = {ws.get('median', 0) * hs.get('median', 0) * 100:.2f}% of image). "
            "Most objects are taller than wide (mean aspect ratio 0.757) — "
            "a small-object detection challenge requiring stride-8 feature maps."
        )

    # ── Spatial heatmap ────────────────────────────────────────────────────────
    _section("Spatial Heatmap", "Where object centres cluster in normalised image space")
    if spat and spat.get("cx_all"):
        _fig(_chart_spatial_heatmap(spat), height=420)
        st.caption(
            f"Centre mean: cx={spat.get('cx_stats', {}).get('mean', 0):.3f}, "
            f"cy={spat.get('cy_stats', {}).get('mean', 0):.3f}  "
            "(close to 0.5 — objects-of-interest framing with no strong quadrant bias)."
        )

    # ── Co-occurrence matrix ───────────────────────────────────────────────────
    _section("Co-occurrence Matrix", "Fraction of class-i annotations sharing an image with class-j")
    if cooc and cooc.get("matrix_normalized"):
        _fig(_chart_cooc(cooc), height=480)
        pairs = cooc.get("top_pairs", [])
        if pairs:
            top3 = pairs[:3]
            st.caption(
                "Strongest co-occurrence pairs: "
                + " · ".join(f"{p['class_a']} + {p['class_b']} ({p['count']})" for p in top3)
                + ". Storage tank appears almost exclusively alone."
            )

    # ── Image resolution ───────────────────────────────────────────────────────
    _section("Image Resolution", "All images are 640×640 after Roboflow's stretch resize")
    if res:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Dominant resolution", res.get("dominant_resolution", "640x640"))
            st.metric("All same resolution", "Yes" if res.get("all_same_resolution") else "No")
            st.metric("Sample size", res.get("sample_size", 280))
        with col2:
            _fig(_chart_resolution(res), height=280)
        st.caption(
            "All 640×640 (Roboflow's 'Stretch' preprocessing). Original NWPU images are variable-size "
            "(typically 1 000–1 500 px). Letterbox padding is recommended for future runs to preserve "
            "aspect ratios of elongated objects (bridges, runways)."
        )

    # ── Sample images ──────────────────────────────────────────────────────────
    _section("Sample Test Images", "Ground-truth annotations from data/raw/test/")
    images = _load_test_sample_images(6)
    if images:
        cols = st.columns(3)
        for i, img in enumerate(images):
            cols[i % 3].image(img, use_column_width=True, caption=f"Test image {i + 1}")
    else:
        st.warning("Test images not found at data/raw/test/images/.")


# ── Page 3 — Model Results ──────────────────────────────────────────────────────

def page_results() -> None:
    cs = load_class_stats()
    th = load_training_history()
    if not cs:
        st.warning("data/class_stats.json not found. Run `python run_evaluation.py` first.")
        return
    if not th:
        st.warning("data/training_history.json not found.")

    models  = cs.get("models", {})
    comp    = cs.get("comparison", {})
    tradeoff = cs.get("speed_accuracy_tradeoff", [])
    n_res   = models.get("yolov8n", {})
    s_res   = models.get("yolov8s", {})

    st.title("Model Results")

    # ── Comparison table ───────────────────────────────────────────────────────
    _section("Overall Comparison", "Winner highlighted in green")

    def _cell(val: str, is_winner: bool) -> str:
        cls = "winner" if is_winner else "loser"
        return f'<td class="{cls}" style="padding:8px 14px;border-bottom:1px solid {BORDER}">{val}</td>'

    metrics = [
        ("Metric",             "YOLOv8n",                                          "YOLOv8s"),
        ("mAP50",              f"{n_res['overall']['mAP50']:.3f}",                 f"{s_res['overall']['mAP50']:.3f}"),
        ("mAP50-95",           f"{n_res['overall']['mAP50_95']:.3f}",              f"{s_res['overall']['mAP50_95']:.3f}"),
        ("Precision",          f"{n_res['overall']['precision']:.3f}",             f"{s_res['overall']['precision']:.3f}"),
        ("Recall",             f"{n_res['overall']['recall']:.3f}",                f"{s_res['overall']['recall']:.3f}"),
        ("Parameters",         f"{n_res['params']:,}",                             f"{s_res['params']:,}"),
        ("CPU latency (avg)",  f"{n_res['speed'].get('total_ms', 57):.1f} ms",     f"{s_res['speed'].get('total_ms', 139):.1f} ms"),
        ("GPU latency (est.)", "~2.3 ms",                                          "~4.3 ms"),
        ("FPS (CPU)",          f"{n_res['speed'].get('fps', 17):.1f}",             f"{s_res['speed'].get('fps', 7):.1f}"),
        ("Best epoch",         str(n_res.get("best_epoch", 37)),                   str(s_res.get("best_epoch", 40))),
        ("Best class",         n_res.get("best_class", "vehicle").replace("_"," "), s_res.get("best_class", "vehicle").replace("_"," ")),
        ("Worst class",        n_res.get("worst_class","bridge").replace("_"," "),  s_res.get("worst_class","bridge").replace("_"," ")),
    ]

    header = (
        f'<table style="width:100%;border-collapse:collapse;font-size:0.92em;color:{TEXT}">'
        f'<thead><tr>'
        + f'<th style="padding:8px 14px;border-bottom:2px solid {BORDER};text-align:left;color:{ACCENT}">Metric</th>'
        + f'<th style="padding:8px 14px;border-bottom:2px solid {BORDER};text-align:left;color:{BLUE}">YOLOv8n</th>'
        + f'<th style="padding:8px 14px;border-bottom:2px solid {BORDER};text-align:left;color:{ACCENT}">YOLOv8s</th>'
        + "</tr></thead><tbody>"
    )
    rows_html = ""
    compare_metrics = {"mAP50", "mAP50-95", "Recall", "FPS (CPU)"}
    n_better_metrics = {"Precision", "CPU latency (avg)", "GPU latency (est.)"}
    for row in metrics[1:]:
        label, n_val, s_val = row
        if label in compare_metrics:
            n_wins = float(n_val.split()[0]) < float(s_val.split()[0])
            s_wins = not n_wins
        elif label in n_better_metrics:
            n_wins = True
            s_wins = False
        else:
            n_wins = s_wins = False
        bg = CARD_BG if metrics.index(row) % 2 == 0 else BG
        rows_html += f'<tr style="background:{bg}">'
        rows_html += f'<td style="padding:8px 14px;border-bottom:1px solid {BORDER};font-weight:500">{label}</td>'
        rows_html += _cell(n_val, n_wins)
        rows_html += _cell(s_val, s_wins)
        rows_html += "</tr>"

    st.markdown(header + rows_html + "</tbody></table>", unsafe_allow_html=True)

    if comp.get("recommendation"):
        st.markdown(
            f'<div style="margin-top:12px;padding:12px 16px;background:{CARD_BG};border-left:3px solid {ACCENT};'
            f'border-radius:4px;color:{TEXT};font-size:0.92em">'
            f'💡 {comp["recommendation"]}</div>',
            unsafe_allow_html=True,
        )

    # ── Per-class mAP50 ────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    _section("Per-Class mAP50", "Both models side by side, sorted by YOLOv8n performance")
    _fig(_chart_per_class_map50(cs), height=420)

    # ── Per-class delta table ──────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    _section("Per-Class mAP50 Delta (YOLOv8s − YOLOv8n)",
             "Note: per-class Precision and Recall were not collected during evaluation — "
             "only mAP50 is available per class. Overall P/R are in the table above.")

    pc_delta = comp.get("per_class_delta", {})
    if pc_delta:
        sorted_cls = sorted(pc_delta.keys(), key=lambda c: abs(pc_delta[c]["delta"]), reverse=True)
        delta_fig = go.Figure(go.Bar(
            x=[c.replace("_", " ") for c in sorted_cls],
            y=[pc_delta[c]["delta"] for c in sorted_cls],
            marker_color=[GREEN if pc_delta[c]["delta"] >= 0 else RED for c in sorted_cls],
            text=[f"{pc_delta[c]['delta']:+.3f}" for c in sorted_cls],
            textposition="outside",
        ))
        delta_fig.add_hline(y=0, line_color="white", line_width=1)
        delta_fig.update_layout(title="mAP50 Delta per Class (positive = s wins)",
                                 yaxis_title="Delta mAP50", xaxis_tickangle=-30)
        _fig(delta_fig, height=360)

    # ── Training curves ────────────────────────────────────────────────────────
    if th:
        st.markdown("<br>", unsafe_allow_html=True)
        _section("Training Curves", "50 epochs on Colab T4 GPU")
        _fig(_chart_training_loss(th), height=340)
        _fig(_chart_map50_curve(th), height=360)

    # ── Speed vs accuracy ──────────────────────────────────────────────────────
    if tradeoff:
        st.markdown("<br>", unsafe_allow_html=True)
        _section("Speed vs Accuracy", "Bubble size ∝ parameter count")
        _fig(_chart_speed_accuracy(tradeoff), height=380)

    # ── Confusion matrices & F1 curves ────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    _section("Confusion Matrices")
    cm_n = DATA_DIR / "yolov8n_confusion_matrix.png"
    cm_s = DATA_DIR / "yolov8s_confusion_matrix.png"
    if cm_n.exists() or cm_s.exists():
        col1, col2 = st.columns(2)
        if cm_n.exists():
            col1.image(str(cm_n), caption="YOLOv8n — Confusion Matrix", use_column_width=True)
        else:
            col1.warning("YOLOv8n confusion matrix not found.")
        if cm_s.exists():
            col2.image(str(cm_s), caption="YOLOv8s — Confusion Matrix", use_column_width=True)
        else:
            col2.warning("YOLOv8s confusion matrix not found.")
    else:
        st.warning("Confusion matrix images not found in data/.")

    st.markdown("<br>", unsafe_allow_html=True)
    _section("F1 Score Curves")
    f1_n = DATA_DIR / "yolov8n_BoxF1_curve.png"
    f1_s = DATA_DIR / "yolov8s_BoxF1_curve.png"
    if f1_n.exists() or f1_s.exists():
        col1, col2 = st.columns(2)
        if f1_n.exists():
            col1.image(str(f1_n), caption="YOLOv8n — Box F1 Curve", use_column_width=True)
        if f1_s.exists():
            col2.image(str(f1_s), caption="YOLOv8s — Box F1 Curve", use_column_width=True)
    else:
        st.warning("F1 curve images not found in data/.")

    # ── Best / worst class analysis ────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    _section("Best & Worst Class Analysis")
    for mname, res in [("YOLOv8n", n_res), ("YOLOv8s", s_res)]:
        pc = res.get("per_class_mAP50", {})
        if not pc:
            continue
        ranking = sorted(pc, key=pc.get, reverse=True)
        top3    = ranking[:3]
        bot3    = ranking[-3:]
        st.markdown(f"**{mname}**")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div style="color:{GREEN};font-weight:600">Top 3 classes</div>', unsafe_allow_html=True)
            for cls in top3:
                st.markdown(
                    f'<div style="padding:6px 12px;margin:3px 0;background:{CARD_BG};'
                    f'border-left:3px solid {GREEN};border-radius:4px;color:{TEXT}">'
                    f'<b>{cls.replace("_"," ")}</b>  —  mAP50 = <b style="color:{GREEN}">{pc[cls]:.3f}</b>'
                    "</div>",
                    unsafe_allow_html=True,
                )
        with col2:
            st.markdown(f'<div style="color:{RED};font-weight:600">Bottom 3 classes</div>', unsafe_allow_html=True)
            for cls in bot3:
                note = ""
                if cls == "bridge":
                    note = " (elongated, highly variable orientation)"
                elif cls == "basketball_court":
                    note = " (confused with tennis court)"
                elif cls == "ground_track_field":
                    note = " (extreme density, 22 instances/image)"
                st.markdown(
                    f'<div style="padding:6px 12px;margin:3px 0;background:{CARD_BG};'
                    f'border-left:3px solid {RED};border-radius:4px;color:{TEXT}">'
                    f'<b>{cls.replace("_"," ")}</b>  —  mAP50 = <b style="color:{RED}">{pc[cls]:.3f}</b>{note}'
                    "</div>",
                    unsafe_allow_html=True,
                )

    # ── Sample detections gallery ──────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    _section("Sample Detection Gallery", "data/sample_detections/ — 15 images per model")
    det_dir = DATA_DIR / "sample_detections"
    if det_dir.exists():
        tabs = st.tabs(["YOLOv8n", "YOLOv8s"])
        for tab, prefix in zip(tabs, ["yolov8n_best", "yolov8s_best"]):
            with tab:
                imgs = sorted(det_dir.glob(f"{prefix}_*.jpg"))
                if not imgs:
                    st.info(f"No detection images found for {prefix}.")
                    continue
                cols = st.columns(3)
                for i, p in enumerate(imgs[:15]):
                    cols[i % 3].image(str(p), use_column_width=True,
                                      caption=p.name.replace(f"{prefix}_", "")[:30])
    else:
        st.warning("data/sample_detections/ not found. Run `python run_evaluation.py`.")


# ── Page 4 — Live Detection ─────────────────────────────────────────────────────

def _annotate_frame(
    frame_bgr: np.ndarray,
    model: Any,
    conf: float,
) -> tuple[np.ndarray, list[dict]]:
    """Run inference on a BGR frame, return annotated RGB array and detections list."""
    results = model(frame_bgr, conf=conf, verbose=False)[0]
    canvas = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    detections: list[dict] = []

    if results.boxes is not None and len(results.boxes):
        for box in results.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            cid   = int(box.cls[0].item())
            score = float(box.conf[0].item())
            color_hex = PALETTE[cid % len(PALETTE)]
            color_rgb = tuple(int(color_hex.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color_rgb, 2)
            label = f"{NWPU_CLASSES[cid]} {score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            ly = max(y1 - 4, th + 4)
            cv2.rectangle(canvas, (x1, ly - th - 4), (x1 + tw + 4, ly), color_rgb, -1)
            cv2.putText(canvas, label, (x1 + 2, ly - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
            detections.append({
                "class": NWPU_CLASSES[cid],
                "confidence": round(score, 3),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })
    return canvas, detections


def page_live_detection() -> None:
    st.title("Live Detection")
    st.markdown(
        f'<p style="color:{MUTED}">Upload a satellite image or video to run YOLOv8 inference.</p>',
        unsafe_allow_html=True,
    )

    col_ctrl, col_info = st.columns([1, 2])
    with col_ctrl:
        model_choice = st.selectbox(
            "Model",
            ["yolov8n (faster — 57 ms CPU)", "yolov8s (more accurate — 139 ms CPU)"],
            index=0,
        )
        model_key = "yolov8n" if model_choice.startswith("yolov8n") else "yolov8s"
        conf = st.slider("Confidence threshold", 0.10, 0.90, 0.25, 0.05)
        uploaded = st.file_uploader(
            "Upload image or video",
            type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
        )

    with col_info:
        st.markdown(f"""
<div style="background:{CARD_BG};border:1px solid {BORDER};border-radius:8px;padding:16px;color:{TEXT};font-size:0.88em">
<b>Model specs</b><br><br>
{'YOLOv8n — 3.0M params, ~2.3ms GPU, best for edge' if model_key == 'yolov8n'
 else 'YOLOv8s — 11.1M params, ~4.3ms GPU, highest accuracy'}<br><br>
<b>10 detectable classes</b><br>
{'  ·  '.join(NWPU_CLASSES)}
</div>
""", unsafe_allow_html=True)

    if uploaded is None:
        st.info("Upload a file above to begin.")
        return

    with st.spinner(f"Loading {model_key} model…"):
        model = load_yolo_model(model_key)

    if model is None:
        st.error(f"Model checkpoint not found: models/{model_key}_best.pt")
        return

    file_bytes = uploaded.read()
    is_video   = uploaded.name.lower().endswith((".mp4", ".avi", ".mov"))

    # ── Image inference ────────────────────────────────────────────────────────
    if not is_video:
        img_array = np.frombuffer(file_bytes, dtype=np.uint8)
        frame_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            st.error("Could not decode image.")
            return

        t0 = time.perf_counter()
        annotated, detections = _annotate_frame(frame_bgr, model, conf)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        col_img, col_results = st.columns([2, 1])
        with col_img:
            st.image(annotated, caption=f"Detections — {len(detections)} objects", use_column_width=True)
        with col_results:
            st.metric("Inference time", f"{elapsed_ms:.1f} ms")
            st.metric("Detections", len(detections))
            st.metric("Model", model_key.upper())

            if detections:
                # Group by class
                from collections import Counter
                counts = Counter(d["class"] for d in detections)
                avg_conf = {cls: round(
                    sum(d["confidence"] for d in detections if d["class"] == cls) / counts[cls], 3
                ) for cls in counts}
                st.markdown(f'<div style="color:{ACCENT};font-weight:600;margin-top:12px">Detection summary</div>',
                            unsafe_allow_html=True)
                for cls, count in sorted(counts.items(), key=lambda x: -x[1]):
                    color = CLASS_COLORS.get(cls, ACCENT)
                    st.markdown(
                        f'<div style="padding:5px 10px;margin:2px 0;background:{CARD_BG};'
                        f'border-left:3px solid {color};border-radius:3px;font-size:0.88em;color:{TEXT}">'
                        f'<b>{cls}</b>  ×{count}  avg conf {avg_conf[cls]:.3f}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info(f"No detections above confidence {conf:.2f}.")

    # ── Video inference ────────────────────────────────────────────────────────
    else:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_in       = cap.get(cv2.CAP_PROP_FPS) or 25

        st.info(f"Video: {total_frames} frames @ {fps_in:.0f} fps. Processing every 5th frame.")

        sample_step   = 5
        max_frames    = 50
        annotated_frames: list[np.ndarray] = []
        det_counts:        list[int]        = []
        frame_idx = 0

        progress = st.progress(0.0, text="Processing frames…")
        idx = 0

        while cap.isOpened() and idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_step == 0:
                ann, dets = _annotate_frame(frame, model, conf)
                annotated_frames.append(ann)
                det_counts.append(len(dets))
                idx += 1
                progress.progress(min(idx / max_frames, 1.0), text=f"Frame {frame_idx}…")
            frame_idx += 1

        cap.release()
        progress.empty()

        if not annotated_frames:
            st.error("No frames extracted from video.")
            return

        st.success(f"Processed {len(annotated_frames)} sampled frames.")

        # Show per-frame detection count
        det_fig = go.Figure(go.Bar(
            x=list(range(1, len(det_counts) + 1)), y=det_counts,
            marker_color=ACCENT,
        ))
        det_fig.update_layout(title="Detections per Sampled Frame",
                              xaxis_title="Frame index", yaxis_title="Detections")
        _fig(det_fig, height=280)

        # Show frame gallery
        st.markdown('<div class="section-title">Annotated Frame Gallery</div>', unsafe_allow_html=True)
        n_show = min(15, len(annotated_frames))
        cols = st.columns(3)
        for i, frame in enumerate(annotated_frames[:n_show]):
            cols[i % 3].image(frame, use_column_width=True, caption=f"Frame {i + 1}")


# ── Page 5 — How I Built This ───────────────────────────────────────────────────

def page_how_i_built() -> None:
    st.title("How I Built This")

    # ── Pipeline diagram ───────────────────────────────────────────────────────
    _section("Pipeline Architecture")

    nodes = [
        "NWPU VHR-10\nRoboflow",
        "Data Loader\n& Validation",
        "EDA\n(7 analyses)",
        "Colab T4\nTraining",
        "YOLOv8n\nbest.pt",
        "YOLOv8s\nbest.pt",
        "Evaluation\n& Comparison",
        "Streamlit\nDashboard",
    ]
    x_pos = [0, 1.2, 2.4, 3.6, 4.6, 4.6, 5.8, 7.0]
    y_pos = [0.5, 0.5, 0.5, 0.5, 1.0, 0.0, 0.5, 0.5]

    edges = [(0,1),(1,2),(2,3),(3,4),(3,5),(4,6),(5,6),(6,7)]

    edge_x, edge_y = [], []
    for a, b in edges:
        edge_x += [x_pos[a], x_pos[b], None]
        edge_y += [y_pos[a], y_pos[b], None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(color=BORDER, width=2), hoverinfo="none",
    ))
    fig.add_trace(go.Scatter(
        x=x_pos, y=y_pos, mode="markers+text",
        marker=dict(size=40, color=CARD_BG, line=dict(color=ACCENT, width=2)),
        text=nodes, textposition="middle center",
        textfont=dict(size=9, color=TEXT),
        hoverinfo="text",
    ))
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False, height=220,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    _fig(fig, height=220)

    # ── Build timeline ─────────────────────────────────────────────────────────
    _section("Build Timeline")
    milestones = [
        ("M1", "Data Foundation",         "Structural validation · 5 quality checks · Pascal VOC converter"),
        ("M2", "Exploratory Analysis",     "7 EDA functions · class imbalance · bbox geometry · co-occurrence"),
        ("M3", "Data Preprocessing",       "Augmentation strategy · letterbox discussion · class weights"),
        ("M4", "Model Training",           "YOLOv8n + YOLOv8s · 50 epochs · Colab T4 · W&B tracking"),
        ("M5", "Evaluation & Comparison",  "Per-class mAP50 · speed benchmarks · confusion matrices"),
        ("M6", "Dashboard",                "5-page Streamlit app · live inference · dark theme · Plotly"),
        ("M7", "Deployment",               "Streamlit Community Cloud · GitHub · portfolio write-up"),
    ]
    for tag, title, desc in milestones:
        st.markdown(
            f'<div style="display:flex;align-items:flex-start;margin:8px 0">'
            f'<div style="min-width:40px;height:40px;border-radius:50%;background:{ACCENT};'
            f'color:#000;font-weight:700;font-size:0.8em;display:flex;align-items:center;'
            f'justify-content:center;margin-right:14px;flex-shrink:0">{tag}</div>'
            f'<div><div style="color:{TEXT};font-weight:600;font-size:0.97em">{title}</div>'
            f'<div style="color:{MUTED};font-size:0.85em">{desc}</div></div></div>',
            unsafe_allow_html=True,
        )

    # ── Key decisions ──────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    _section("Key Decisions & Trade-offs")
    decisions = [
        ("Why NWPU VHR-10 over DIOR?",
         "NWPU has 10 tightly defined classes with dense annotations (avg 6 objects/image) "
         "and a known class imbalance challenge — richer for a portfolio than DIOR's 20 sparse classes."),
        ("Why compare n vs s instead of n vs m?",
         "The nano→small jump (3M → 11M params, 3.7×) maximises the speed/accuracy contrast "
         "while keeping training time under 15 minutes each on Colab T4."),
        ("Letterbox vs stretch resize?",
         "Roboflow's default is stretch (distorts aspect ratios). Letterbox is better for "
         "elongated objects (bridges, runways). Left as-is for M1–M4 to match the Roboflow export; "
         "recommended for future iterations."),
        ("How to handle 11.46× class imbalance?",
         "Used Ultralytics default weighted loss. For production: focal loss γ=1.5 or "
         "class-balanced sampling. Bridge (mAP50=0.03) would benefit most — it's the hardest class "
         "despite being the most common."),
        ("Why mosaic=1.0?",
         "Mosaic augmentation synthesises 4-image composites, artificially creating multi-object "
         "scenes. Critical for small-object detection since it increases the effective number of "
         "objects per training step."),
    ]
    for q, a in decisions:
        with st.expander(q):
            st.markdown(f'<p style="color:{TEXT}">{a}</p>', unsafe_allow_html=True)

    # ── YOLOv8 architecture explanation ───────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    _section("YOLOv8 Architecture Explained")
    st.markdown(f"""
<div style="color:{TEXT};line-height:1.8;font-size:0.93em">

**Backbone — CSPDarknet:**
The backbone extracts hierarchical features at 4 scales (stride 8, 16, 32, and optionally 4).
Cross-Stage Partial (CSP) connections halve gradient bottlenecks by splitting feature maps across
two branches, merging with a dense block. C2f modules (the YOLOv8 refinement of C3) replace
traditional bottlenecks with a split-then-merge topology for more efficient parameter use.

**Neck — PANet (Path Aggregation Network):**
The neck fuses multi-scale features. A top-down FPN pass propagates semantics from deep layers
to shallow, followed by a bottom-up PAN path that carries fine spatial detail upward.
This bidirectional fusion lets the head detect both small objects (stride-8, P2) and large ones
(stride-32, P5) in the same forward pass.

**Head — Decoupled, Anchor-Free:**
YOLOv8 splits classification and regression into separate branches (decoupled head).
It is anchor-free: each grid cell directly regresses the distance to the 4 box edges (DFL loss)
rather than predicting offsets from preset anchors. This removes anchor hyperparameter tuning.

**Nano vs Small (depth/width multipliers):**
- YOLOv8n: depth=0.33, width=0.25 → 3.0M params, 8.1 GFLOPs
- YOLOv8s: depth=0.33, width=0.50 → 11.1M params, 28.5 GFLOPs

The width multiplier scales all channel counts; the depth multiplier scales the number of
repeated C2f blocks. Both models have 73 layers — the "nano" is thinner, not shallower.
</div>
""", unsafe_allow_html=True)

    # ── Links ──────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    _section("Links")
    st.markdown("""
| | Resource | URL |
|---|---|---|
| 📓 | Dataset | [Roboflow Universe — NWPU VHR-10](https://universe.roboflow.com/yolo-7muwp/nwpu-vhr-10-sgj3z) |
| 📄 | NWPU Paper | Cheng et al., ISPRS 2014 — Multi-class geospatial object detection |
""")


# ── Main ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="SkyWatch — Satellite Detection",
        page_icon="🛰️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_css()

    with st.sidebar:
        st.markdown(
            f'<div style="font-size:1.3em;font-weight:700;color:{ACCENT};padding:8px 0 4px">'
            "🛰️ SkyWatch</div>"
            f'<div style="font-size:0.78em;color:{MUTED};margin-bottom:20px">'
            "Satellite Object Detection</div>",
            unsafe_allow_html=True,
        )
        page = st.radio(
            "Navigation",
            options=["overview", "eda", "results", "detect", "build"],
            format_func=lambda x: {
                "overview": "🏠  Project Overview",
                "eda":      "📊  Explore the Data",
                "results":  "📈  Model Results",
                "detect":   "🔍  Live Detection",
                "build":    "🏗️  How I Built This",
            }[x],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown(
            f'<div style="font-size:0.75em;color:{MUTED}">'
            "NWPU VHR-10 · YOLOv8 · Streamlit<br>"
            "Data: Roboflow Universe</div>",
            unsafe_allow_html=True,
        )

    if page == "overview":
        page_overview()
    elif page == "eda":
        page_eda()
    elif page == "results":
        page_results()
    elif page == "detect":
        page_live_detection()
    elif page == "build":
        page_how_i_built()


if __name__ == "__main__":
    main()
