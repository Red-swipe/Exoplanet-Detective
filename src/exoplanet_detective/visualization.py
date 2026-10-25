"""Plotly visualization helpers for the Streamlit dashboard.

All figure functions return plotly.graph_objects.Figure objects.
They do not use Streamlit directly, keeping the presentation layer separate
from the visualization logic.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .models import LightCurveData, ProcessedLightCurve, TransitCandidate

# ---------------------------------------------------------------------------
# Color theme
# ---------------------------------------------------------------------------

DARK_BG = "#0d1117"
PANEL_BG = "#161b22"
GRID_COLOR = "#21262d"
TEXT_COLOR = "#c9d1d9"
ACCENT_BLUE = "#58a6ff"
ACCENT_ORANGE = "#f0883e"
ACCENT_GREEN = "#3fb950"
ACCENT_RED = "#f85149"

BASE_LAYOUT = dict(
    paper_bgcolor=DARK_BG,
    plot_bgcolor=PANEL_BG,
    font=dict(color=TEXT_COLOR, family="monospace"),
    xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
    yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
    margin=dict(l=60, r=30, t=50, b=50),
)


def _apply_base_layout(fig: go.Figure, title: str) -> go.Figure:
    layout = dict(BASE_LAYOUT)
    layout["title"] = dict(text=title, font=dict(color=TEXT_COLOR, size=14))
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# Raw light curve
# ---------------------------------------------------------------------------

def plot_raw_light_curve(lc: LightCurveData) -> go.Figure:
    """Plot raw flux vs. time."""
    synthetic_note = " [SYNTHETIC]" if lc.is_synthetic else ""
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=lc.time,
            y=lc.flux,
            mode="markers",
            marker=dict(size=2, color=ACCENT_BLUE, opacity=0.6),
            name="Raw flux",
        )
    )
    _apply_base_layout(
        fig, f"Raw Light Curve — {lc.target_name}{synthetic_note}"
    )
    fig.update_xaxes(title_text="Time (BJD / BTJD)")
    fig.update_yaxes(title_text="Flux")
    return fig


# ---------------------------------------------------------------------------
# Processed vs. raw comparison
# ---------------------------------------------------------------------------

def plot_processed_comparison(result: ProcessedLightCurve) -> go.Figure:
    """Plot raw and processed light curves on separate panels."""
    synthetic_note = " [SYNTHETIC]" if result.raw.is_synthetic else ""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=["Raw", "Processed"],
        vertical_spacing=0.08,
    )
    fig.add_trace(
        go.Scattergl(
            x=result.raw.time,
            y=result.raw.flux,
            mode="markers",
            marker=dict(size=2, color=ACCENT_BLUE, opacity=0.5),
            name="Raw",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scattergl(
            x=result.processed.time,
            y=result.processed.flux,
            mode="markers",
            marker=dict(size=2, color=ACCENT_GREEN, opacity=0.5),
            name="Processed",
        ),
        row=2, col=1,
    )
    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(
            text=f"Light Curve Comparison — {result.raw.target_name}{synthetic_note}",
            font=dict(color=TEXT_COLOR, size=14),
        ),
        showlegend=True,
    )
    fig.update_xaxes(title_text="Time (BJD / BTJD)", row=2, col=1, gridcolor=GRID_COLOR)
    fig.update_yaxes(title_text="Flux", gridcolor=GRID_COLOR)
    return fig


# ---------------------------------------------------------------------------
# BLS periodogram
# ---------------------------------------------------------------------------

def plot_periodogram(
    periods: np.ndarray,
    power: np.ndarray,
    candidates: list[TransitCandidate],
    is_synthetic: bool = False,
) -> go.Figure:
    """Plot BLS periodogram with candidate peaks marked."""
    synthetic_note = " [SYNTHETIC]" if is_synthetic else ""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=periods,
            y=power,
            mode="lines",
            line=dict(color=ACCENT_BLUE, width=1),
            name="BLS power",
        )
    )
    colors = [ACCENT_ORANGE, ACCENT_RED, ACCENT_GREEN, "#a371f7", "#79c0ff"]
    for i, c in enumerate(candidates[:5]):
        color = colors[i % len(colors)]
        fig.add_vline(
            x=c.period_days,
            line=dict(color=color, dash="dash", width=1.5),
            annotation_text=f"#{c.rank} {c.period_days:.3f}d",
            annotation_font=dict(color=color, size=11),
        )
    _apply_base_layout(fig, f"BLS Periodogram{synthetic_note}")
    fig.update_xaxes(title_text="Period (days)", type="log")
    fig.update_yaxes(title_text="BLS Power (SNR)")
    return fig


# ---------------------------------------------------------------------------
# Phase-folded light curve
# ---------------------------------------------------------------------------

def plot_phase_folded(
    phase: np.ndarray,
    flux: np.ndarray,
    candidate: TransitCandidate,
    is_synthetic: bool = False,
    bin_phase: bool = True,
) -> go.Figure:
    """Plot a phase-folded light curve with optional binning."""
    synthetic_note = " [SYNTHETIC]" if is_synthetic else ""
    fig = go.Figure()

    # Raw scatter
    fig.add_trace(
        go.Scattergl(
            x=phase,
            y=flux,
            mode="markers",
            marker=dict(size=2, color=ACCENT_BLUE, opacity=0.3),
            name="Folded data",
        )
    )

    # Binned curve
    if bin_phase and len(phase) > 50:
        n_bins = min(200, len(phase) // 10)
        bin_edges = np.linspace(phase.min(), phase.max(), n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_means = []
        for j in range(n_bins):
            in_bin = (phase >= bin_edges[j]) & (phase < bin_edges[j + 1])
            if in_bin.sum() > 0:
                bin_means.append(np.nanmean(flux[in_bin]))
            else:
                bin_means.append(np.nan)
        fig.add_trace(
            go.Scatter(
                x=bin_centers,
                y=np.array(bin_means),
                mode="lines",
                line=dict(color=ACCENT_ORANGE, width=2),
                name="Binned",
            )
        )

    # Mark transit center
    fig.add_vline(
        x=0,
        line=dict(color=ACCENT_RED, dash="dot", width=1.5),
        annotation_text="Transit center",
        annotation_font=dict(color=ACCENT_RED, size=11),
    )

    _apply_base_layout(
        fig,
        f"Phase-Folded Light Curve — P={candidate.period_days:.4f}d{synthetic_note}",
    )
    fig.update_xaxes(title_text="Phase (days from transit center)")
    fig.update_yaxes(title_text="Normalized Flux")
    return fig


# ---------------------------------------------------------------------------
# Diagnostics summary figure
# ---------------------------------------------------------------------------

def plot_processing_diagnostics(result: ProcessedLightCurve) -> go.Figure:
    """Bar chart of cadences removed at each preprocessing step."""
    labels = ["NaN/Inf", "Quality flags", "Outliers", "Retained"]
    values = [
        result.n_removed_nan,
        result.n_removed_quality,
        result.n_removed_outlier,
        result.n_retained,
    ]
    colors_bar = [ACCENT_RED, ACCENT_ORANGE, "#a371f7", ACCENT_GREEN]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors_bar,
            text=values,
            textposition="auto",
        )
    )
    _apply_base_layout(fig, "Preprocessing Diagnostics")
    fig.update_yaxes(title_text="Number of cadences")
    return fig
