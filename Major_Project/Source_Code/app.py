"""
BrandPulse AI — Twitter Sentiment Analysis Dashboard
=====================================================
Premium SaaS Analytics Interface
Persevex NLP Internship Project

A professional brand intelligence product built on TensorFlow/scikit-learn models.

Run:
    python app.py
Then open http://127.0.0.1:8050 in your browser.
"""

from __future__ import annotations

import base64
import io
import json
import os
import random
from datetime import datetime, timedelta
from typing import Optional, Tuple

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import Input, Output, State, callback, ctx, dcc, html

from src import config
from src.model_utils import (
    load_error_analysis,
    load_logreg,
    load_lstm,
    load_model_metadata,
    predict_logreg,
    predict_lstm,
)
from src.text_utils import clean_tweet

# ============================================================================
# App initialization
# ============================================================================
app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
        {
            "name": "description",
            "content": "BrandPulse AI — Premium Sentiment Analysis Dashboard",
        },
    ],
)
app.title = "BrandPulse AI"

# ============================================================================
# Load models & data once on startup
# ============================================================================
logreg_model, tfidf_vec = load_logreg()
lstm_model, lstm_tok, lstm_cfg = load_lstm()
metadata = load_model_metadata()
error_data = load_error_analysis()

SENTIMENT_CLASSES = config.SENTIMENT_CLASSES
LOW_CONFIDENCE_THRESHOLD = config.LOW_CONFIDENCE_THRESHOLD

# Sentiment palette is owned here, not in config, so it stays locked to the
# design system's LED-style semantics (assets/style.css) regardless of what
# config.py defines. Same three keys config.SENTIMENT_COLORS used.
SENTIMENT_COLORS = {
    "positive": "#3FBF83",
    "neutral": "#D9A441",
    "negative": "#E2596E",
}

# Chart theme shared by every Plotly figure so charts sit inside a dark panel
# instead of a leftover white plotly_white box. CH1/CH2 = the two model
# accent colors used throughout the CSS (amber / signal-blue).
CHART_FONT_FAMILY = "IBM Plex Mono, ui-monospace, monospace"
CHART_TEXT_COLOR = "#9BA3A7"
CHART_GRID_COLOR = "rgba(255, 255, 255, 0.09)"
CHART_DEFAULT_COLOR = "#E3A23C"
CH1_COLOR = "#E3A23C"  # Logistic Regression
CH2_COLOR = "#4E8FE0"  # BiLSTM

CHART_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family=CHART_FONT_FAMILY, size=12, color=CHART_TEXT_COLOR),
)

available_models = []
if logreg_model is not None:
    available_models.append("TF-IDF + Logistic Regression")
if lstm_model is not None:
    available_models.append("LSTM (Deep Learning)")


def _load_overview_dataframe() -> Optional[pd.DataFrame]:
    """Read the dataset CSV once at startup for all overview visuals."""
    if not os.path.exists(config.DATA_PATH):
        return None
    try:
        return pd.read_csv(config.DATA_PATH)
    except UnicodeDecodeError:
        return pd.read_csv(config.DATA_PATH, encoding="latin-1")
    except Exception:
        return None


OVERVIEW_DF = _load_overview_dataframe()

FALLBACK_STREAM_TWEETS = [
    "Flight delayed 3 hours and no one told us why. Ridiculous.",
    "Thanks for the quick refund, appreciate the support!",
    "Boarding on time today, nice change of pace.",
    "Lost my luggage again... second time this year.",
    "Crew was super friendly on this flight, made the trip better.",
    "Anyone else stuck in the security line for an hour?",
    "Great legroom on this new plane, very comfortable.",
    "Cancelled with zero notice, missed my connection.",
    "Check-in was smooth and fast, no complaints.",
    "Why is the wifi always broken on these flights?",
    "Just landed, flight was fine, nothing special.",
    "Customer service hung up on me twice today.",
    "Loved the snacks and service, will fly again!",
    "Seat was broken and they wouldn't move me. Awful.",
    "On time departure and arrival, exactly as expected.",
]

# ============================================================================
# HELPER FUNCTIONS (all preserved from original)
# ============================================================================


def load_stream_pool(sample_size: int = 400) -> list:
    """Load real tweets from dataset to simulate live stream."""
    if os.path.exists(config.DATA_PATH):
        try:
            df = pd.read_csv(config.DATA_PATH)
        except UnicodeDecodeError:
            df = pd.read_csv(config.DATA_PATH, encoding="latin-1")
        text_col = next(
            (c for c in ["text", "tweet", "content"] if c in df.columns), None
        )
        if text_col:
            texts = df[text_col].dropna().astype(str).tolist()
            if len(texts) > sample_size:
                texts = random.sample(texts, sample_size)
            return texts
    return FALLBACK_STREAM_TWEETS


def simulated_timestamps(n: int) -> list:
    """Generate simulated timestamps across the last 24 hours."""
    now = datetime.now()
    start = now - timedelta(hours=24)
    step = timedelta(hours=24) / max(n, 1)
    out = []
    for i in range(n):
        jitter = timedelta(
            minutes=random.uniform(
                -step.total_seconds() / 120, step.total_seconds() / 120
            )
        )
        out.append(start + step * i + jitter)
    return out


def run_prediction_full(texts: list, model_choice: str) -> Tuple:
    """Run prediction with chosen model. Returns (preds, conf, proba, classes)."""
    if model_choice == "TF-IDF + Logistic Regression":
        return predict_logreg(texts, logreg_model, tfidf_vec)
    return predict_lstm(texts, lstm_model, lstm_tok, lstm_cfg)


def parse_contents(contents: str, filename: str) -> Optional[pd.DataFrame]:
    """Parse uploaded CSV file content."""
    if not contents:
        return None
    content_type, content_string = contents.split(",")
    try:
        decoded = base64.b64decode(content_string)
        try:
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        except UnicodeDecodeError:
            df = pd.read_csv(io.StringIO(decoded.decode("latin-1")))
        return df
    except Exception:
        return None


def hourly_trend(buffer_df: pd.DataFrame) -> pd.DataFrame:
    """Compute hourly sentiment breakdown as percentages."""
    if buffer_df.empty:
        return pd.DataFrame(columns=["positive", "neutral", "negative"])
    df = buffer_df.copy()
    df["hour"] = df["timestamp"].dt.floor("h")
    pivot = df.groupby(["hour", "sentiment"]).size().unstack(fill_value=0)
    for col in ["positive", "neutral", "negative"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["positive", "neutral", "negative"]]
    pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    pct.index = pct.index.strftime("%H:%M")
    return pct.round(1)


_KPI_ICON_BY_COLOR_CLASS = {
    "total": "icon-total",
    "positive": "icon-positive",
    "neutral": "icon-neutral",
    "negative": "icon-negative",
    "model-a": "icon-model",
    "model-b": "icon-model",
}

_KPI_ICON_BY_LABEL_KEYWORD = [
    ("positive", "icon-positive"),
    ("negative", "icon-negative"),
    ("neutral", "icon-neutral"),
    ("total", "icon-total"),
    ("tweets analyzed", "icon-total"),
    ("dataset", "icon-dataset"),
    ("regression", "icon-model"),
    ("lstm", "icon-model"),
    ("accuracy", "icon-metric"),
    ("f1", "icon-metric"),
    ("precision", "icon-metric"),
    ("recall", "icon-metric"),
    ("feature", "icon-metric"),
    ("date range", "icon-metric"),
]


def _infer_kpi_icon(label: str, color_class: str) -> str:
    """Pick an icon class for a KPI card from its color_class, falling back to
    matching keywords in the label. Purely cosmetic — never changes the
    card's data."""
    if color_class in _KPI_ICON_BY_COLOR_CLASS:
        return _KPI_ICON_BY_COLOR_CLASS[color_class]
    label_lower = (label or "").lower()
    for keyword, icon in _KPI_ICON_BY_LABEL_KEYWORD:
        if keyword in label_lower:
            return icon
    return "icon-metric"


def create_empty_state(icon_class: str, title: str, subtitle: str) -> html.Div:
    """A persistent, informative placeholder shown in a results panel before
    the user has triggered anything. Keeps split-layout pages visually full
    on first load instead of leaving a blank void next to the input card."""
    return html.Div(
        [
            html.Div(
                html.Span(className=f"bp-icon {icon_class}"), className="bp-empty-icon"
            ),
            html.Div(title, className="bp-empty-title"),
            html.Div(subtitle, className="bp-empty-subtitle"),
        ],
        className="bp-empty-state",
    )


def create_kpi_card_new(label: str, value: str, color_class: str = "") -> html.Div:
    """Create a premium KPI card with an icon chip."""
    classes = f"bp-kpi-card {color_class}".strip()
    icon_class = _infer_kpi_icon(label, color_class)
    return html.Div(
        [
            html.Div(
                html.Span(className=f"bp-icon {icon_class}"),
                className="bp-kpi-icon",
            ),
            html.Div(label, className="bp-kpi-label"),
            html.Div(value, className=f"bp-kpi-value {color_class}"),
        ],
        className=classes,
    )


def create_sentiment_badge_new(label: str, confidence: float) -> html.Div:
    """Create a clean sentiment prediction badge."""
    color_map = {"positive": "positive", "negative": "negative", "neutral": "neutral"}
    badge_class = color_map.get(label, "neutral")
    return html.Div(
        [
            html.Span(
                label.upper(),
                style={"fontWeight": "700", "fontSize": "18px", "marginRight": "8px"},
            ),
            html.Span(
                f"{confidence*100:.1f}% confident",
                style={"opacity": "0.8", "fontSize": "13px"},
            ),
        ],
        className=f"bp-sentiment-badge {badge_class}",
    )


def create_probability_bars_new(classes: list, proba_row: np.ndarray) -> html.Div:
    """Create professional probability bars."""
    bars = []
    for label, p in sorted(zip(classes, proba_row), key=lambda x: -x[1]):
        bars.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(label.capitalize(), style={"fontWeight": "600"}),
                            html.Span(f"{p*100:.1f}%", style={"fontWeight": "700"}),
                        ],
                        style={
                            "display": "flex",
                            "justifyContent": "space-between",
                            "marginBottom": "4px",
                            "fontSize": "13px",
                        },
                    ),
                    html.Div(
                        html.Div(
                            style={
                                "width": f"{p*100:.1f}%",
                                "background": SENTIMENT_COLORS.get(
                                    label, CHART_DEFAULT_COLOR
                                ),
                                "height": "6px",
                                "borderRadius": "3px",
                            }
                        ),
                        style={
                            "background": "var(--line-strong)",
                            "borderRadius": "3px",
                            "height": "6px",
                        },
                    ),
                ],
                style={"marginBottom": "12px"},
            )
        )

    content = [html.Div(bars)]
    if max(proba_row) < LOW_CONFIDENCE_THRESHOLD:
        content.append(
            html.Div(
                "Low confidence — model is uncertain. Consider with other signals.",
                className="bp-alert bp-alert-warning",
                style={"marginTop": "12px"},
            )
        )
    return html.Div(content)


def create_stream_feed_item(
    text: str, sentiment: str, confidence: float, timestamp
) -> html.Div:
    """One row in the Live Monitor's simulated tweet feed."""
    color_class = {
        "positive": "positive",
        "negative": "negative",
        "neutral": "neutral",
    }.get(sentiment, "neutral")
    shown_text = text if len(text) <= 110 else text[:110] + "…"
    return html.Div(
        [
            html.Div(
                timestamp.strftime("%H:%M:%S"),
                style={
                    "fontFamily": "var(--font-mono)",
                    "fontSize": "11.5px",
                    "color": "var(--text-secondary)",
                    "width": "72px",
                    "flexShrink": "0",
                },
            ),
            html.Div(shown_text, style={"flex": "1", "fontSize": "13px"}),
            html.Div(
                [
                    html.Span(
                        sentiment.upper(),
                        style={"fontWeight": "700", "fontSize": "11px"},
                    ),
                    html.Span(
                        f" {confidence*100:.0f}%",
                        style={"opacity": "0.8", "fontSize": "11px"},
                    ),
                ],
                className=f"bp-sentiment-badge {color_class}",
                style={"flexShrink": "0"},
            ),
        ],
        className="bp-card",
        style={
            "display": "flex",
            "alignItems": "center",
            "gap": "14px",
            "padding": "10px 14px",
            "marginBottom": "6px",
        },
    )


def create_sentiment_distribution_chart(sentiment_series: pd.Series) -> go.Figure:
    """Create Plotly bar chart for sentiment distribution."""
    counts = sentiment_series.value_counts()
    colors = [SENTIMENT_COLORS.get(c, CHART_DEFAULT_COLOR) for c in counts.index]
    fig = go.Figure(
        data=[
            go.Bar(
                x=counts.index,
                y=counts.values,
                marker=dict(color=colors),
                text=counts.values,
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        showlegend=False,
        height=250,
        margin=dict(l=40, r=20, t=20, b=40),
        hovermode="x unified",
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(gridcolor=CHART_GRID_COLOR),
        **CHART_THEME,
    )
    return fig


def create_confidence_histogram(confidence_series: pd.Series) -> go.Figure:
    """Create Plotly histogram for confidence distribution."""
    fig = go.Figure(
        data=[
            go.Histogram(
                x=confidence_series, nbinsx=15, marker=dict(color=CHART_DEFAULT_COLOR)
            )
        ]
    )
    fig.update_layout(
        showlegend=False,
        height=250,
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis=dict(gridcolor=CHART_GRID_COLOR),
        yaxis=dict(gridcolor=CHART_GRID_COLOR),
        **CHART_THEME,
    )
    return fig


def create_confusion_matrix_heatmap(cm: list, classes: list, title: str) -> go.Figure:
    """Create a confusion matrix heatmap."""
    cm = np.array(cm)
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=classes,
            y=classes,
            colorscale="Cividis",
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 12},
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=350,
        margin=dict(l=80, r=20, t=40, b=40),
        **CHART_THEME,
    )
    return fig


def create_trend_chart(trend_df: pd.DataFrame) -> go.Figure:
    """Create a line chart for sentiment trend."""
    if trend_df.empty:
        fig = go.Figure().add_annotation(
            text="No data available", font=dict(color=CHART_TEXT_COLOR)
        )
        fig.update_layout(**CHART_THEME)
        return fig
    fig = go.Figure()
    for sentiment in ["positive", "neutral", "negative"]:
        if sentiment in trend_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=trend_df.index,
                    y=trend_df[sentiment],
                    mode="lines+markers",
                    name=sentiment.capitalize(),
                    line=dict(
                        color=SENTIMENT_COLORS.get(sentiment, CHART_DEFAULT_COLOR),
                        width=2,
                    ),
                    marker=dict(size=5),
                )
            )
    fig.update_layout(
        showlegend=True,
        height=300,
        margin=dict(l=40, r=20, t=20, b=40),
        hovermode="x unified",
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(gridcolor=CHART_GRID_COLOR),
        **CHART_THEME,
    )
    return fig


# ----------------------------------------------------------------------------
# Model Comparison charts
# ----------------------------------------------------------------------------
# All four figures are built once, at startup, from models/model_metadata.json
# — the same numbers already shown elsewhere in the app. Nothing here re-runs
# a model or reads the dataset, so it costs nothing at request time.
#
# Sizing: every comparison chart is autosize=True with NO fixed pixel height.
# The card that holds it (.bp-compare-grid .bp-chart-card) is given one fixed
# height in CSS so all four cards — including this one and the "Accuracy /
# Macro F1 / Weighted F1" card — render at identical size, and each Plotly
# figure stretches to fill its card instead of carrying its own mismatched
# height and spilling out of the box.

_COMPARE_MODEL_COLORS = {
    "logistic_regression": CH1_COLOR,  # CH1 — amber
    "bilstm": CH2_COLOR,  # CH2 — signal-blue
}

_COMPARE_LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    autosize=True,
    margin=dict(l=48, r=16, t=16, b=40),
    font=dict(family=CHART_FONT_FAMILY, size=12, color=CHART_TEXT_COLOR),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)

# Shared config/style so every comparison dcc.Graph fills its (fixed-height)
# card via CSS instead of keeping the figure's own intrinsic pixel height.
_COMPARE_GRAPH_STYLE = {"height": "100%", "width": "100%"}
_COMPARE_GRAPH_CONFIG = {"displayModeBar": False, "responsive": True}


def _compare_model_items(models: dict) -> list:
    """Return [(key, display_name, data), ...] in a stable, display-friendly order."""
    order = ["logistic_regression", "bilstm"]
    items = []
    for key in order:
        if key in models:
            items.append(
                (
                    key,
                    models[key].get("display_name", key.replace("_", " ").title()),
                    models[key],
                )
            )
    for key, data in models.items():
        if key not in order:
            items.append(
                (key, data.get("display_name", key.replace("_", " ").title()), data)
            )
    return items


def create_metrics_grouped_bar_chart(models: dict) -> go.Figure:
    """Grouped bar chart: Accuracy vs Macro F1 vs Weighted F1, per model."""
    items = _compare_model_items(models)
    metrics = [
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro F1"),
        ("weighted_f1", "Weighted F1"),
    ]
    fig = go.Figure()
    for key, name, data in items:
        values = [round(data.get(m, 0) * 100, 1) for m, _ in metrics]
        fig.add_trace(
            go.Bar(
                name=name,
                x=[label for _, label in metrics],
                y=values,
                marker=dict(color=_COMPARE_MODEL_COLORS.get(key, CHART_DEFAULT_COLOR)),
                text=[f"{v:.1f}%" for v in values],
                textposition="outside",
            )
        )
    fig.update_layout(
        barmode="group",
        yaxis=dict(
            title="Score (%)", range=[0, 100], gridcolor="rgba(255,255,255,0.09)"
        ),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        **_COMPARE_LAYOUT_DEFAULTS,
    )
    return fig


def create_inference_speed_chart(models: dict) -> go.Figure:
    """Horizontal bar chart: total inference time on the shared test set."""
    items = _compare_model_items(models)
    names = [name for _, name, _ in items]
    speeds = [
        data.get("inference_time_seconds_for_test_set", 0) for _, _, data in items
    ]
    colors = [
        _COMPARE_MODEL_COLORS.get(key, CHART_DEFAULT_COLOR) for key, _, _ in items
    ]
    fig = go.Figure(
        go.Bar(
            x=speeds,
            y=names,
            orientation="h",
            marker=dict(color=colors),
            text=[f"{s:.2f}s" for s in speeds],
            textposition="outside",
        )
    )
    test_size = items[0][2].get("test_set_size") if items else None
    subtitle = f"Test set: {test_size:,} tweets" if test_size else ""
    fig.update_layout(
        xaxis=dict(
            title=(
                f"Inference time, seconds ({subtitle})"
                if subtitle
                else "Inference time (seconds)"
            ),
            gridcolor="rgba(255,255,255,0.09)",
        ),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        showlegend=False,
        **_COMPARE_LAYOUT_DEFAULTS,
    )
    return fig


def create_model_size_chart(models: dict) -> go.Figure:
    """Bar chart: on-disk model size in MB."""
    items = _compare_model_items(models)
    names = [name for _, name, _ in items]
    sizes = [data.get("approx_model_size_mb", 0) for _, _, data in items]
    colors = [
        _COMPARE_MODEL_COLORS.get(key, CHART_DEFAULT_COLOR) for key, _, _ in items
    ]
    fig = go.Figure(
        go.Bar(
            x=names,
            y=sizes,
            marker=dict(color=colors),
            text=[f"{s:.2f} MB" for s in sizes],
            textposition="outside",
        )
    )
    fig.update_layout(
        yaxis=dict(title="Model size (MB)", gridcolor="rgba(255,255,255,0.09)"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        showlegend=False,
        **_COMPARE_LAYOUT_DEFAULTS,
    )
    return fig


def _normalize(value: float, lo: float, hi: float, invert: bool = False) -> float:
    """Scale a value into 0-100 given a known range. invert=True means lower is better."""
    if hi == lo:
        return 100.0
    score = (value - lo) / (hi - lo) * 100
    return round(100 - score if invert else score, 1)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def create_radar_chart(models: dict) -> go.Figure:
    """Radar chart comparing Accuracy, F1, Speed, Efficiency, and Interpretability.

    Accuracy / F1 come straight from the evaluation metrics. Speed and
    Efficiency are derived by normalizing inference time and model size
    across the two models (faster / smaller -> higher score). Interpretability
    is a qualitative rating (linear TF-IDF weights are directly inspectable;
    a BiLSTM's internal representations are not) rather than a measured
    metric, and is labelled as such in the surrounding copy.
    """
    items = _compare_model_items(models)
    if not items:
        return go.Figure()

    speeds = [d.get("inference_time_seconds_for_test_set", 0) for _, _, d in items]
    sizes = [d.get("approx_model_size_mb", 0) for _, _, d in items]
    speed_lo, speed_hi = min(speeds), max(speeds)
    size_lo, size_hi = min(sizes), max(sizes)

    interpretability = {"logistic_regression": 92, "bilstm": 38}

    categories = ["Accuracy", "Macro F1", "Speed", "Efficiency", "Interpretability"]

    fig = go.Figure()
    for key, name, data in items:
        values = [
            round(data.get("accuracy", 0) * 100, 1),
            round(data.get("macro_f1", 0) * 100, 1),
            _normalize(
                data.get("inference_time_seconds_for_test_set", 0),
                speed_lo,
                speed_hi,
                invert=True,
            ),
            _normalize(
                data.get("approx_model_size_mb", 0), size_lo, size_hi, invert=True
            ),
            interpretability.get(key, 50),
        ]
        fig.add_trace(
            go.Scatterpolar(
                r=values + values[:1],
                theta=categories + categories[:1],
                fill="toself",
                name=name,
                line=dict(color=_COMPARE_MODEL_COLORS.get(key, CHART_DEFAULT_COLOR)),
                fillcolor=_hex_to_rgba(
                    _COMPARE_MODEL_COLORS.get(key, CHART_DEFAULT_COLOR), 0.18
                ),
                opacity=0.85,
            )
        )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 100], gridcolor="rgba(255,255,255,0.09)"
            ),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.09)"),
            bgcolor="rgba(0,0,0,0)",
        ),
        **{**_COMPARE_LAYOUT_DEFAULTS, "margin": dict(l=40, r=40, t=24, b=24)},
    )
    return fig


def _build_tradeoff_insight(models: dict) -> html.Div:
    """Plain-language trade-off summary computed from the live metadata
    numbers, so it never drifts out of sync with the charts above it."""
    if "logistic_regression" not in models or "bilstm" not in models:
        return html.Div()
    lr = models["logistic_regression"]
    ls = models["bilstm"]

    speed_ratio = ls.get("inference_time_seconds_for_test_set", 1) / max(
        lr.get("inference_time_seconds_for_test_set", 1), 1e-9
    )
    size_ratio = ls.get("approx_model_size_mb", 1) / max(
        lr.get("approx_model_size_mb", 1), 1e-9
    )
    acc_gain_pts = (ls.get("accuracy", 0) - lr.get("accuracy", 0)) * 100

    return html.Div(
        [
            html.Div(
                [
                    html.Span(className="bp-icon icon-comparison"),
                    html.Span("Trade-off Insight", className="bp-insight-title"),
                ],
                className="bp-insight-header",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span(className="bp-model-dot a"),
                                    html.Span(
                                        lr.get("display_name", "Logistic Regression")
                                    ),
                                ],
                                className="bp-insight-col-title",
                            ),
                            html.Ul(
                                [
                                    html.Li(
                                        f"~{speed_ratio:.1f}× faster inference on the full test set."
                                    ),
                                    html.Li(
                                        f"~{size_ratio:.1f}× smaller on disk ({lr.get('approx_model_size_mb', 0):.2f} MB vs {ls.get('approx_model_size_mb', 0):.2f} MB)."
                                    ),
                                    html.Li(
                                        "Linear TF-IDF weights are directly inspectable, so predictions are easy to explain."
                                    ),
                                    html.Li(
                                        "Best fit for real-time, low-latency, or resource-constrained deployment."
                                    ),
                                ],
                                className="bp-insight-list",
                            ),
                        ],
                        className="bp-insight-col model-a",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span(className="bp-model-dot b"),
                                    html.Span(ls.get("display_name", "BiLSTM")),
                                ],
                                className="bp-insight-col-title",
                            ),
                            html.Ul(
                                [
                                    html.Li(
                                        f"+{acc_gain_pts:.1f} points of accuracy over Logistic Regression."
                                    ),
                                    html.Li(
                                        f"Higher macro F1 ({ls.get('macro_f1', 0)*100:.1f}% vs {lr.get('macro_f1', 0)*100:.1f}%), capturing minority classes better."
                                    ),
                                    html.Li(
                                        "Sequential context lets it pick up on word order and negation TF-IDF misses."
                                    ),
                                    html.Li(
                                        "Best fit for accuracy-critical, offline / batch scenarios where latency matters less."
                                    ),
                                ],
                                className="bp-insight-list",
                            ),
                        ],
                        className="bp-insight-col model-b",
                    ),
                ],
                className="bp-insight-columns",
            ),
            html.Div(
                [
                    html.Strong("Bottom line: "),
                    f"Logistic Regression trades a few points of accuracy for roughly {speed_ratio:.0f}× the speed and a fraction of "
                    f"the footprint, so it wins when latency or deployment cost matters. BiLSTM spends that extra time and memory "
                    f"budget on {acc_gain_pts:.1f} points of accuracy and better minority-class F1, so it wins when correctness "
                    f"matters more than response time.",
                ],
                className="bp-insight-verdict",
            ),
        ],
        className="bp-insight-panel",
    )


# ============================================================================
# PRECOMPUTED STATIC CONTENT (computed once at startup)
# ----------------------------------------------------------------------------
# The Overview, Model Comparison, Error Analysis, and About tabs render data
# that never changes at runtime (dataset metadata, stored metrics, error
# summaries). Previously these were rebuilt on every dropdown/tab trigger;
# building them once here keeps tab switching instant and avoids redundant
# CSV reads / figure construction.
# ============================================================================


def _build_overview_kpis() -> list:
    if not metadata:
        return [create_kpi_card_new("No Data", "N/A")]
    kpis = [
        create_kpi_card_new(
            "Total Tweets", f"{metadata.get('dataset_size', 0):,}", "total"
        ),
        create_kpi_card_new(
            "Positive", f"{metadata.get('positive_count', 0):,}", "positive"
        ),
        create_kpi_card_new(
            "Neutral", f"{metadata.get('neutral_count', 0):,}", "neutral"
        ),
        create_kpi_card_new(
            "Negative", f"{metadata.get('negative_count', 0):,}", "negative"
        ),
    ]
    if metadata.get("models", {}).get("logistic_regression"):
        lr = metadata["models"]["logistic_regression"]
        kpis.append(
            create_kpi_card_new(
                "Logistic Regression", f"{lr['accuracy']*100:.1f}%", "model-a"
            )
        )
    if metadata.get("models", {}).get("bilstm"):
        ls = metadata["models"]["bilstm"]
        kpis.append(
            create_kpi_card_new("BiLSTM", f"{ls['accuracy']*100:.1f}%", "model-b")
        )
    return kpis


def _empty_chart(message: str = "No data") -> go.Figure:
    fig = go.Figure().add_annotation(
        text=message,
        font=dict(color=CHART_TEXT_COLOR, family=CHART_FONT_FAMILY),
        showarrow=False,
    )
    fig.update_layout(**CHART_THEME)
    return fig


def _build_overview_trend_figure() -> go.Figure:
    if OVERVIEW_DF is None:
        return _empty_chart()
    try:
        return create_trend_chart(hourly_trend(OVERVIEW_DF))
    except Exception:
        return _empty_chart()


def _build_overview_distribution_figure() -> go.Figure:
    if OVERVIEW_DF is None:
        return _empty_chart()
    try:
        sentiment_col = next(
            (c for c in ["airline_sentiment", "sentiment"] if c in OVERVIEW_DF.columns),
            None,
        )
        if sentiment_col:
            return create_sentiment_distribution_chart(OVERVIEW_DF[sentiment_col])
    except Exception:
        pass
    return _empty_chart()


def _build_comparison_content() -> html.Div:
    """Build the full Model Comparison page: a compact KPI row per model
    plus four Plotly visualizations (grouped metrics, inference speed,
    model size, radar) and a data-driven trade-off insight panel. Built
    once at startup from model_metadata.json, same as the other static
    pages, so switching to this tab is instant.

    All four chart cards use the same CSS-driven fixed height
    (.bp-compare-grid .bp-chart-card) and each dcc.Graph is set to fill
    that card (style height 100% + responsive config) instead of carrying
    its own mismatched pixel height, so every box in the grid — including
    this one and the "Accuracy / Macro F1 / Weighted F1" card — renders at
    the same size with nothing spilling outside its box.
    """
    models = metadata.get("models", {}) if metadata else {}
    if not models:
        return html.Div(
            "No comparison data available.", className="bp-alert bp-alert-info"
        )

    model_accent_cycle = ["model-a", "model-b"]
    kpi_rows = []
    for i, (_key, model_name, model_data) in enumerate(_compare_model_items(models)):
        accent_class = model_accent_cycle[i % 2]
        kpi_rows.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(className=f"bp-model-dot {accent_class}"),
                            html.Span(model_name),
                        ],
                        className="bp-model-card-title",
                    ),
                    html.Div(
                        [
                            create_kpi_card_new(
                                "Accuracy",
                                f"{model_data.get('accuracy', 0)*100:.1f}%",
                                accent_class,
                            ),
                            create_kpi_card_new(
                                "Macro F1",
                                f"{model_data.get('macro_f1', 0)*100:.1f}%",
                                accent_class,
                            ),
                            create_kpi_card_new(
                                "Weighted F1",
                                f"{model_data.get('weighted_f1', 0)*100:.1f}%",
                                accent_class,
                            ),
                            create_kpi_card_new(
                                "Size",
                                f"{model_data.get('approx_model_size_mb', 0):.2f} MB",
                                accent_class,
                            ),
                        ],
                        className="bp-kpi-grid",
                    ),
                ],
                className=f"bp-card bp-model-card {accent_class}",
            )
        )

    charts = html.Div(
        [
            html.Div(
                [
                    html.H3(
                        "Accuracy, Macro F1 & Weighted F1", className="bp-chart-title"
                    ),
                    dcc.Graph(
                        figure=create_metrics_grouped_bar_chart(models),
                        style=_COMPARE_GRAPH_STYLE,
                        config=_COMPARE_GRAPH_CONFIG,
                    ),
                ],
                className="bp-chart-card",
            ),
            html.Div(
                [
                    html.H3("Trade-off Radar", className="bp-chart-title"),
                    dcc.Graph(
                        figure=create_radar_chart(models),
                        style=_COMPARE_GRAPH_STYLE,
                        config=_COMPARE_GRAPH_CONFIG,
                    ),
                ],
                className="bp-chart-card",
            ),
            html.Div(
                [
                    html.H3("Inference Speed (test set)", className="bp-chart-title"),
                    dcc.Graph(
                        figure=create_inference_speed_chart(models),
                        style=_COMPARE_GRAPH_STYLE,
                        config=_COMPARE_GRAPH_CONFIG,
                    ),
                ],
                className="bp-chart-card",
            ),
            html.Div(
                [
                    html.H3("Model Size", className="bp-chart-title"),
                    dcc.Graph(
                        figure=create_model_size_chart(models),
                        style=_COMPARE_GRAPH_STYLE,
                        config=_COMPARE_GRAPH_CONFIG,
                    ),
                ],
                className="bp-chart-card",
            ),
        ],
        className="bp-compare-grid",
    )

    return html.Div(
        [
            html.Div(
                kpi_rows,
                className="bp-kpi-grid",
                style={
                    "gridTemplateColumns": "repeat(auto-fit, minmax(260px, 1fr))",
                    "marginBottom": "var(--space-md)",
                },
            ),
            charts,
            _build_tradeoff_insight(models),
        ]
    )


def _build_error_content() -> html.Div:
    if not error_data:
        return html.Div(
            "No error analysis data available.", className="bp-alert bp-alert-info"
        )
    return html.Div(
        [
            html.Div(
                f"Misclassifications: {error_data.get('total_misclassified', 0)}",
                className="bp-card",
                style={"marginBottom": "20px"},
            ),
            html.Div(
                "Model errors are documented and analyzed in the error analysis output.",
                className="bp-alert bp-alert-info",
            ),
        ]
    )


def _build_about_content() -> html.Div:
    if not metadata:
        return html.Div(
            "No dataset information available.", className="bp-alert bp-alert-info"
        )
    return html.Div(
        [
            html.Div(
                [
                    create_kpi_card_new(
                        "Dataset Size", f"{metadata.get('dataset_size', 0):,}"
                    ),
                    create_kpi_card_new("Features", str(metadata.get("features", 0))),
                    create_kpi_card_new(
                        "Date Range", metadata.get("date_range", "N/A")
                    ),
                ],
                className="bp-kpi-grid",
            ),
            html.Div(
                [
                    html.H3(
                        "Methodology",
                        className="bp-card-title",
                        style={"marginBottom": "12px"},
                    ),
                    html.P(
                        "Two sentiment classification approaches were trained and evaluated: "
                        "(1) TF-IDF vectorization with Logistic Regression for interpretability, "
                        "(2) Bidirectional LSTM neural network for higher accuracy.",
                        className="bp-card-content",
                    ),
                ],
                className="bp-card",
                style={"marginTop": "20px"},
            ),
        ]
    )


OVERVIEW_KPIS_CONTENT = _build_overview_kpis()
OVERVIEW_TREND_FIGURE = _build_overview_trend_figure()
OVERVIEW_DISTRIBUTION_FIGURE = _build_overview_distribution_figure()
COMPARISON_CONTENT = _build_comparison_content()
ERROR_CONTENT = _build_error_content()
ABOUT_CONTENT = _build_about_content()


NAV_ITEMS = [
    ("overview", "Overview", "icon-overview"),
    ("predict", "Predict", "icon-predict"),
    ("batch", "Batch Analysis", "icon-batch"),
    ("live", "Live Monitor", "icon-live"),
    ("comparison", "Model Comparison", "icon-comparison"),
    ("error", "Error Analysis", "icon-error"),
    ("about", "About", "icon-about"),
]


def _nav_button(
    tab_key: str, label: str, icon_class: str, active: bool = False
) -> html.Button:
    """Build a sidebar nav button with an icon + label. The button's id and
    initial className exactly match what switch_tab expects/overwrites."""
    return html.Button(
        [
            html.Span(className=f"bp-icon {icon_class}"),
            html.Span(label, className="bp-nav-label"),
        ],
        id=f"tab-{tab_key}",
        className=f"bp-nav-item{' active' if active else ''}",
        n_clicks=0,
    )


# ============================================================================
# APP LAYOUT
# ============================================================================

app.layout = html.Div(
    [
        dcc.Store(id="session-state", data={"stream_buffer": []}),
        dcc.Store(id="batch-data-store"),
        dcc.Store(id="current-tab", data="overview"),
        # Header
        html.Div(
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("Brand Intelligence", className="bp-eyebrow"),
                            html.Span("Sentiment Analytics"),
                        ],
                        className="bp-header-title",
                    ),
                    html.Div(
                        className="bp-pulse-trace",
                        title="Live signal — two channels, one system",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span(className="bp-status-dot"),
                                    html.Span("System Online"),
                                ],
                                className="bp-status-pill",
                            ),
                            html.Div(
                                dcc.Dropdown(
                                    id="model-selector",
                                    options=[
                                        {"label": m, "value": m}
                                        for m in available_models
                                    ],
                                    value=(
                                        available_models[0]
                                        if available_models
                                        else None
                                    ),
                                    clearable=False,
                                    className="bp-form-select",
                                ),
                                className="bp-model-select-wrap",
                            ),
                        ],
                        className="bp-header-right",
                    ),
                ],
                className="bp-header-content",
            ),
            className="bp-header",
        ),
        # Sidebar
        html.Div(
            [
                html.Div(
                    [
                        html.Div(className="bp-brand-mark"),
                        html.Div(
                            [
                                html.Span("BrandPulse AI", className="bp-brand-name"),
                                html.Span(
                                    "AI-Powered Sentiment Intelligence",
                                    className="bp-brand-sub",
                                ),
                            ],
                            className="bp-brand-text",
                        ),
                    ],
                    className="bp-brand",
                ),
                html.Div("Navigation", className="bp-sidebar-title"),
                html.Div(
                    [
                        _nav_button(key, label, icon, active=(key == "overview"))
                        for key, label, icon in NAV_ITEMS
                    ],
                    className="bp-nav",
                ),
                html.Div(
                    [
                        html.Div(
                            "Model Legend",
                            className="bp-sidebar-title",
                            style={"margin": "0 0 8px 6px"},
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Span(className="bp-model-dot a"),
                                        html.Span("Logistic Regression"),
                                    ],
                                    className="bp-model-tag",
                                ),
                                html.Div(
                                    [
                                        html.Span(className="bp-model-dot b"),
                                        html.Span("BiLSTM"),
                                    ],
                                    className="bp-model-tag",
                                ),
                            ],
                            className="bp-model-tags",
                        ),
                    ],
                    className="bp-sidebar-footer",
                ),
            ],
            className="bp-sidebar",
        ),
        # Main container
        html.Div(
            html.Div(
                [
                    # Content
                    html.Div(
                        [
                            # Overview Tab
                            html.Div(
                                id="page-overview",
                                children=[
                                    html.Div(
                                        [
                                            html.H2(
                                                "Overview", className="bp-page-title"
                                            ),
                                            html.P(
                                                "Sentiment analysis dashboard for brand intelligence",
                                                className="bp-page-subtitle",
                                            ),
                                        ],
                                        className="bp-page-header",
                                    ),
                                    html.Div(
                                        id="overview-kpis",
                                        className="bp-kpi-grid",
                                        children=OVERVIEW_KPIS_CONTENT,
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.H3(
                                                        "24-Hour Sentiment Trend",
                                                        className="bp-chart-title",
                                                    ),
                                                    dcc.Graph(
                                                        id="overview-trend",
                                                        figure=OVERVIEW_TREND_FIGURE,
                                                        style={"height": "280px"},
                                                    ),
                                                ],
                                                className="bp-chart-card",
                                            ),
                                            html.Div(
                                                [
                                                    html.H3(
                                                        "Sentiment Distribution",
                                                        className="bp-chart-title",
                                                    ),
                                                    dcc.Graph(
                                                        id="overview-distribution",
                                                        figure=OVERVIEW_DISTRIBUTION_FIGURE,
                                                        style={"height": "280px"},
                                                    ),
                                                ],
                                                className="bp-chart-card",
                                            ),
                                        ],
                                        className="bp-charts-grid",
                                    ),
                                ],
                                style={"display": "none"},
                            ),
                            # Predict Tab
                            html.Div(
                                id="page-predict",
                                children=[
                                    html.Div(
                                        [
                                            html.H2(
                                                "Single Prediction",
                                                className="bp-page-title",
                                            ),
                                            html.P(
                                                "Analyze the sentiment of a single tweet",
                                                className="bp-page-subtitle",
                                            ),
                                        ],
                                        className="bp-page-header",
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Enter Tweet",
                                                        className="bp-form-label",
                                                    ),
                                                    # dcc.Textarea (not html.Textarea) — dcc components are the ones
                                                    # Dash actually wires up for two-way binding, so what the user
                                                    # types is synced back to the "value" prop the callback reads via
                                                    # State. html.Textarea just renders a plain <textarea> and never
                                                    # reports typed input back to the server, which is why Analyze
                                                    # always saw an empty value and showed "Please enter a tweet."
                                                    dcc.Textarea(
                                                        id="predict-textarea",
                                                        className="bp-form-control bp-textarea",
                                                        placeholder="Paste or type a tweet here...",
                                                    ),
                                                    html.Button(
                                                        "Analyze",
                                                        id="predict-button",
                                                        className="bp-button bp-button-primary",
                                                        style={"marginTop": "12px"},
                                                    ),
                                                    html.Div(
                                                        [
                                                            html.Span(
                                                                className="bp-icon icon-model"
                                                            ),
                                                            html.Span(
                                                                "Tip: try slang, emoji, or sarcasm to see how the model handles it.",
                                                                className="bp-hint-text",
                                                            ),
                                                        ],
                                                        className="bp-hint-row",
                                                    ),
                                                ],
                                                className="bp-card",
                                            ),
                                            dcc.Loading(
                                                html.Div(
                                                    id="predict-output",
                                                    children=create_empty_state(
                                                        "icon-predict",
                                                        "No prediction yet",
                                                        "Enter a tweet on the left and click Analyze to see the predicted sentiment, class probabilities, and cleaned text here.",
                                                    ),
                                                ),
                                                type="circle",
                                                color=CHART_DEFAULT_COLOR,
                                            ),
                                        ],
                                        className="bp-split-layout",
                                    ),
                                ],
                                style={"display": "none"},
                            ),
                            # Batch Tab
                            html.Div(
                                id="page-batch",
                                children=[
                                    html.Div(
                                        [
                                            html.H2(
                                                "Batch Analysis",
                                                className="bp-page-title",
                                            ),
                                            html.P(
                                                "Upload a CSV file for bulk sentiment analysis",
                                                className="bp-page-subtitle",
                                            ),
                                        ],
                                        className="bp-page-header",
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    dcc.Upload(
                                                        id="batch-upload",
                                                        children=html.Div(
                                                            [
                                                                "Drag & drop CSV file here or click to select"
                                                            ],
                                                            className="bp-upload-inner",
                                                        ),
                                                        className="bp-upload-zone",
                                                    ),
                                                    html.Button(
                                                        "Run Analysis",
                                                        id="batch-button",
                                                        className="bp-button bp-button-primary bp-button-block",
                                                        style={"marginTop": "14px"},
                                                    ),
                                                    html.Div(
                                                        [
                                                            html.Span(
                                                                className="bp-icon icon-dataset"
                                                            ),
                                                            html.Span(
                                                                "Needs a 'text' column; each row is scored independently.",
                                                                className="bp-hint-text",
                                                            ),
                                                        ],
                                                        className="bp-hint-row",
                                                    ),
                                                ],
                                                className="bp-card",
                                            ),
                                            dcc.Loading(
                                                html.Div(
                                                    id="batch-output",
                                                    children=create_empty_state(
                                                        "icon-batch",
                                                        "No file analyzed yet",
                                                        "Upload a CSV of tweets and click Run Analysis to see the sentiment breakdown, per-row results, and downloadable output here.",
                                                    ),
                                                ),
                                                type="circle",
                                                color=CHART_DEFAULT_COLOR,
                                            ),
                                        ],
                                        className="bp-split-layout",
                                    ),
                                ],
                                style={"display": "none"},
                            ),
                            # Live Monitor Tab
                            html.Div(
                                id="page-live",
                                children=[
                                    html.Div(
                                        [
                                            html.H2(
                                                "Live Monitor",
                                                className="bp-page-title",
                                            ),
                                            html.P(
                                                "Real-time sentiment monitoring simulation",
                                                className="bp-page-subtitle",
                                            ),
                                        ],
                                        className="bp-page-header",
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Div(
                                                        [
                                                            html.Label(
                                                                "Tweets to Stream",
                                                                className="bp-form-label",
                                                            ),
                                                            dcc.Slider(
                                                                id="stream-count-slider",
                                                                min=10,
                                                                max=100,
                                                                step=5,
                                                                value=20,
                                                                marks={
                                                                    10: "10",
                                                                    50: "50",
                                                                    100: "100",
                                                                },
                                                                tooltip={
                                                                    "placement": "bottom",
                                                                    "always_visible": True,
                                                                },
                                                            ),
                                                        ],
                                                        className="bp-form-group",
                                                    ),
                                                    html.Div(
                                                        [
                                                            html.Label(
                                                                "Stream Speed",
                                                                className="bp-form-label",
                                                            ),
                                                            dcc.Slider(
                                                                id="stream-speed-slider",
                                                                min=0.05,
                                                                max=1.0,
                                                                step=0.05,
                                                                value=0.5,
                                                                marks={
                                                                    0.05: "Fast",
                                                                    0.5: "Normal",
                                                                    1.0: "Slow",
                                                                },
                                                                tooltip={
                                                                    "placement": "bottom",
                                                                    "always_visible": True,
                                                                },
                                                            ),
                                                        ],
                                                        className="bp-form-group",
                                                    ),
                                                    html.Div(
                                                        [
                                                            html.Button(
                                                                "Start",
                                                                id="stream-start-button",
                                                                className="bp-button bp-button-primary",
                                                                style={
                                                                    "marginRight": "8px"
                                                                },
                                                            ),
                                                            html.Button(
                                                                "Reset",
                                                                id="stream-reset-button",
                                                                className="bp-button bp-button-secondary",
                                                            ),
                                                        ],
                                                        style={
                                                            "display": "flex",
                                                            "gap": "8px",
                                                        },
                                                    ),
                                                ],
                                                className="bp-card",
                                            ),
                                            html.Div(
                                                id="stream-output",
                                                children=create_empty_state(
                                                    "icon-live",
                                                    "Stream not started",
                                                    "Set your stream size and speed, then click Start to watch simulated tweets get classified in real time.",
                                                ),
                                            ),
                                        ],
                                        className="bp-split-layout",
                                    ),
                                ],
                                style={"display": "none"},
                            ),
                            # Comparison Tab
                            html.Div(
                                id="page-comparison",
                                children=[
                                    html.Div(
                                        [
                                            html.H2(
                                                "Model Comparison",
                                                className="bp-page-title",
                                            ),
                                            html.P(
                                                "Compare TF-IDF Logistic Regression vs LSTM performance",
                                                className="bp-page-subtitle",
                                            ),
                                        ],
                                        className="bp-page-header",
                                    ),
                                    html.Div(
                                        id="comparison-output",
                                        children=COMPARISON_CONTENT,
                                    ),
                                ],
                                style={"display": "none"},
                            ),
                            # Error Analysis Tab
                            html.Div(
                                id="page-error",
                                children=[
                                    html.Div(
                                        [
                                            html.H2(
                                                "Error Analysis",
                                                className="bp-page-title",
                                            ),
                                            html.P(
                                                "Review model misclassifications and limitations",
                                                className="bp-page-subtitle",
                                            ),
                                        ],
                                        className="bp-page-header",
                                    ),
                                    html.Div(id="error-output", children=ERROR_CONTENT),
                                ],
                                style={"display": "none"},
                            ),
                            # About Tab
                            html.Div(
                                id="page-about",
                                children=[
                                    html.Div(
                                        [
                                            html.H2("About", className="bp-page-title"),
                                            html.P(
                                                "Dataset, methodology, and model information",
                                                className="bp-page-subtitle",
                                            ),
                                        ],
                                        className="bp-page-header",
                                    ),
                                    html.Div(id="about-output", children=ABOUT_CONTENT),
                                ],
                                style={"display": "none"},
                            ),
                        ],
                        className="bp-content",
                    ),
                ],
                className="bp-container",
            ),
            className="bp-main",
        ),
    ],
    className="bp-app",
)


# ============================================================================
# TAB NAVIGATION CALLBACK — runs entirely in the browser (clientside)
# ----------------------------------------------------------------------------
# Tab switching never needs the server: no data is fetched, no model runs,
# nothing is recomputed — it only toggles which already-rendered page div is
# visible. Running this in JS instead of a Python round-trip removes network
# latency from every click, so navigation feels instant even while the
# server is busy (e.g. mid-inference on another tab).
# ============================================================================

_TAB_KEYS = ["overview", "predict", "batch", "live", "comparison", "error", "about"]

app.clientside_callback(
    """
    function(...clicks) {
        const tabKeys = ["overview", "predict", "batch", "live", "comparison", "error", "about"];

        // Which tab is active is decided by which button actually fired
        // this callback (dash_clientside.callback_context.triggered), not
        // by comparing n_clicks totals across all seven buttons. Comparing
        // totals meant a tab only "won" once its own click count exceeded
        // every other tab's — so switching tabs could silently require
        // several clicks before the count caught up. Reading the trigger
        // directly makes every single click switch immediately, exactly
        // like a normal tab click.
        let tabIndex = 0;
        const ctx = window.dash_clientside && window.dash_clientside.callback_context;
        const triggered = ctx && ctx.triggered && ctx.triggered[0];
        if (triggered && triggered.prop_id && triggered.prop_id !== ".") {
            const triggeredId = triggered.prop_id.split(".")[0];
            const key = triggeredId.replace("tab-", "");
            const idx = tabKeys.indexOf(key);
            if (idx !== -1) {
                tabIndex = idx;
            }
        }
        const displays = clicks.map((_, i) => i === tabIndex ? {display: "block"} : {display: "none"});
        // The active page div gets a fresh enter-animation class so its
        // fade/slide-in keyframe replays on every switch, even if the user
        // returns to a tab whose className was already set this way (React
        // only restarts a CSS animation when the class VALUE changes, so we
        // alternate between two identical animation classes on every click).
        const totalClicks = clicks.reduce((a, b) => a + b, 0);
        const variant = (totalClicks % 2 === 0) ? "bp-page-enter-a" : "bp-page-enter-b";
        const pageClasses = clicks.map((_, i) => i === tabIndex ? variant : "");
        const navClasses = clicks.map((_, i) => i === tabIndex ? "bp-nav-item active" : "bp-nav-item");

        // Plotly only needs to be told to resize a chart ONCE — the moment
        // its page first becomes visible and the browser can finally
        // measure a real width instead of 0. The window itself never
        // changes size just because a tab was clicked, so re-running
        // Plotly's relayout math (bars, radar polar grid, etc.) on every
        // repeat visit to the same tab was pure wasted work — and the
        // actual source of the click-to-click lag. window.__bpResizedTabs
        // remembers which tabs were already sized once and skips them
        // after that, so switching back to an already-seen tab is instant.
        window.__bpResizedTabs = window.__bpResizedTabs || {};
        if (!window.__bpResizedTabs[tabIndex]) {
            window.__bpResizedTabs[tabIndex] = true;
            window.requestAnimationFrame(function () {
                const activePage = document.getElementById("page-" + tabKeys[tabIndex]);
                if (activePage && window.Plotly) {
                    activePage.querySelectorAll(".js-plotly-plot").forEach(function (gd) {
                        try { window.Plotly.Plots.resize(gd); } catch (e) { /* not a plot yet */ }
                    });
                }
            });
        }

        return displays.concat(pageClasses).concat(navClasses);
    }
    """,
    [Output(f"page-{tab}", "style") for tab in _TAB_KEYS]
    + [Output(f"page-{tab}", "className") for tab in _TAB_KEYS]
    + [Output(f"tab-{tab}", "className") for tab in _TAB_KEYS],
    [Input(f"tab-{tab}", "n_clicks") for tab in _TAB_KEYS],
)


# ============================================================================
# PREDICT TAB CALLBACK
# ============================================================================


@callback(
    Output("predict-output", "children"),
    Input("predict-button", "n_clicks"),
    State("predict-textarea", "value"),
    State("model-selector", "value"),
    prevent_initial_call=True,
)
def handle_predict(n_clicks, text, model_choice):
    if not model_choice or not available_models:
        return html.Div(
            "No trained model available.", className="bp-alert bp-alert-danger"
        )

    if not text or not text.strip():
        return html.Div("Please enter a tweet.", className="bp-alert bp-alert-warning")

    try:
        preds, conf, proba, classes = run_prediction_full([text], model_choice)
        sentiment = str(preds[0])
        confidence = float(conf[0])

        return html.Div(
            [
                html.Div(
                    [
                        create_sentiment_badge_new(sentiment, confidence),
                        html.Div(
                            [
                                html.H4(
                                    "Class Probabilities",
                                    style={
                                        "marginBottom": "10px",
                                        "fontSize": "12.5px",
                                        "fontWeight": "700",
                                        "textTransform": "uppercase",
                                        "letterSpacing": "0.4px",
                                        "color": "var(--text-secondary)",
                                    },
                                ),
                                create_probability_bars_new(classes, proba[0]),
                            ],
                            style={"marginTop": "16px"},
                        ),
                        html.Div(
                            [
                                html.H4(
                                    "Cleaned Text",
                                    style={
                                        "marginBottom": "10px",
                                        "fontSize": "12.5px",
                                        "fontWeight": "700",
                                        "textTransform": "uppercase",
                                        "letterSpacing": "0.4px",
                                        "color": "var(--text-secondary)",
                                    },
                                ),
                                html.Pre(
                                    clean_tweet(text) or "(empty after cleaning)",
                                    style={
                                        "background": "var(--bg)",
                                        "border": "1px solid var(--surface-border)",
                                        "padding": "12px 14px",
                                        "borderRadius": "10px",
                                        "fontSize": "12px",
                                        "fontFamily": "var(--font-mono)",
                                        "whiteSpace": "pre-wrap",
                                        "color": "var(--text-secondary)",
                                    },
                                ),
                            ],
                            style={"marginTop": "16px"},
                        ),
                    ],
                    className="bp-card",
                ),
            ],
            style={"marginTop": "18px"},
        )
    except Exception as e:
        return html.Div(f"Error: {str(e)}", className="bp-alert bp-alert-danger")


# ============================================================================
# BATCH TAB CALLBACK
# ============================================================================


@callback(
    [Output("batch-button", "disabled"), Output("batch-output", "children")],
    [Input("batch-upload", "contents"), Input("batch-button", "n_clicks")],
    [State("batch-upload", "filename"), State("model-selector", "value")],
    prevent_initial_call=True,
)
def handle_batch_analysis(contents, n_clicks, filename, model_choice):
    if not contents:
        return True, html.Div(
            "Upload a CSV file to get started.", className="bp-alert bp-alert-info"
        )

    if not model_choice or not available_models:
        return False, html.Div(
            "No trained model available.", className="bp-alert bp-alert-danger"
        )

    data = parse_contents(contents, filename or "")
    if data is None or data.empty:
        return False, html.Div(
            "Could not parse CSV file.", className="bp-alert bp-alert-danger"
        )

    text_col = next(
        (c for c in ["text", "tweet", "content", "SentimentText"] if c in data.columns),
        None,
    )
    if text_col is None:
        return False, html.Div(
            "CSV must contain: text, tweet, content, or SentimentText column.",
            className="bp-alert bp-alert-danger",
        )

    if n_clicks and n_clicks > 0:
        try:
            texts_raw = data[text_col].astype(str).tolist()
            texts = [t for t in texts_raw if t.strip()]

            if not texts:
                return False, html.Div(
                    "No non-empty text found.", className="bp-alert bp-alert-warning"
                )

            preds, conf, proba, classes = run_prediction_full(texts, model_choice)
            sentiment_series = pd.Series(preds)
            conf_series = pd.Series(conf)

            kpis = html.Div(
                [
                    create_kpi_card_new("Tweets Analyzed", f"{len(texts):,}", "total"),
                    create_kpi_card_new(
                        "Positive",
                        f"{(sentiment_series == 'positive').sum()}",
                        "positive",
                    ),
                    create_kpi_card_new(
                        "Neutral", f"{(sentiment_series == 'neutral').sum()}", "neutral"
                    ),
                    create_kpi_card_new(
                        "Negative",
                        f"{(sentiment_series == 'negative').sum()}",
                        "negative",
                    ),
                ],
                className="bp-kpi-grid",
            )

            charts = html.Div(
                [
                    html.Div(
                        [
                            html.H3("Distribution", className="bp-chart-title"),
                            dcc.Graph(
                                figure=create_sentiment_distribution_chart(
                                    sentiment_series
                                )
                            ),
                        ],
                        className="bp-chart-card",
                    ),
                    html.Div(
                        [
                            html.H3("Confidence", className="bp-chart-title"),
                            dcc.Graph(figure=create_confidence_histogram(conf_series)),
                        ],
                        className="bp-chart-card",
                    ),
                ],
                className="bp-charts-grid",
            )

            return False, html.Div([kpis, charts], style={"marginTop": "20px"})
        except Exception as e:
            return False, html.Div(
                f"Error: {str(e)}", className="bp-alert bp-alert-danger"
            )

    return False, html.Div(
        "Click 'Run Analysis' to process the file.", className="bp-alert bp-alert-info"
    )


# ============================================================================
# LIVE MONITOR CALLBACK
# ----------------------------------------------------------------------------
# Previously the Start / Reset buttons on this tab had no callback registered
# at all, so clicking them did nothing — load_stream_pool() and
# simulated_timestamps() existed but were never called. This wires them up:
# Start draws `count` real tweets from the dataset, timestamps them across a
# simulated last-24-hours window, scores them with the selected model, and
# renders a feed + 24h trend chart. This is a point-in-time simulation of a
# stream (all rows scored at once), not a truly animated tick-by-tick feed —
# the "Stream Speed" slider is kept as a label of intended pace for now
# rather than driving a real dcc.Interval animation, which would be a larger
# follow-up change.
# ============================================================================


@callback(
    Output("stream-output", "children"),
    Input("stream-start-button", "n_clicks"),
    Input("stream-reset-button", "n_clicks"),
    State("stream-count-slider", "value"),
    State("model-selector", "value"),
    prevent_initial_call=True,
)
def handle_live_stream(start_clicks, reset_clicks, stream_count, model_choice):
    triggered_id = ctx.triggered_id

    if triggered_id == "stream-reset-button":
        return create_empty_state(
            "icon-live",
            "Stream not started",
            "Set your stream size and speed, then click Start to watch simulated tweets get classified in real time.",
        )

    if not model_choice or not available_models:
        return html.Div(
            "No trained model available.", className="bp-alert bp-alert-danger"
        )

    try:
        count = int(stream_count) if stream_count else 20
        texts = load_stream_pool(sample_size=count)
        texts = texts[:count] if len(texts) > count else texts
        timestamps = simulated_timestamps(len(texts))

        preds, conf, proba, classes = run_prediction_full(texts, model_choice)

        buffer_df = pd.DataFrame(
            {
                "text": texts,
                "sentiment": [str(p) for p in preds],
                "confidence": [float(c) for c in conf],
                "timestamp": timestamps,
            }
        )

        sentiment_series = buffer_df["sentiment"]
        kpis = html.Div(
            [
                create_kpi_card_new("Tweets Streamed", f"{len(buffer_df):,}", "total"),
                create_kpi_card_new(
                    "Positive", f"{(sentiment_series == 'positive').sum()}", "positive"
                ),
                create_kpi_card_new(
                    "Neutral", f"{(sentiment_series == 'neutral').sum()}", "neutral"
                ),
                create_kpi_card_new(
                    "Negative", f"{(sentiment_series == 'negative').sum()}", "negative"
                ),
            ],
            className="bp-kpi-grid",
        )

        trend_chart = html.Div(
            [
                html.H3(
                    "24-Hour Sentiment Trend (simulated)", className="bp-chart-title"
                ),
                dcc.Graph(
                    figure=create_trend_chart(hourly_trend(buffer_df)),
                    style={"height": "260px"},
                ),
            ],
            className="bp-chart-card",
        )

        # Most recent tweets first, capped so the feed stays scannable.
        feed_rows = [
            create_stream_feed_item(
                row.text, row.sentiment, row.confidence, row.timestamp
            )
            for row in buffer_df.sort_values("timestamp", ascending=False)
            .head(30)
            .itertuples(index=False)
        ]
        feed = html.Div(
            [html.H3("Incoming Tweets", className="bp-chart-title")] + feed_rows,
            style={"marginTop": "16px"},
        )

        return html.Div([kpis, trend_chart, feed], style={"marginTop": "8px"})

    except Exception as e:
        return html.Div(f"Error: {str(e)}", className="bp-alert bp-alert-danger")


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
