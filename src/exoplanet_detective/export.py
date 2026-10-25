"""Export analysis results to CSV and JSON formats."""

from __future__ import annotations

import json
import io
from typing import Optional

import numpy as np
import pandas as pd

from .models import AnalysisResult, LightCurveData


def export_light_curve_csv(lc: LightCurveData) -> str:
    """Export a LightCurveData to a CSV string."""
    data: dict = {"time": lc.time, "flux": lc.flux}
    if lc.flux_err is not None:
        data["flux_err"] = lc.flux_err
    if lc.quality is not None:
        data["quality"] = lc.quality
    df = pd.DataFrame(data)
    return df.to_csv(index=False)


def export_analysis_json(result: AnalysisResult) -> str:
    """Serialize an AnalysisResult to a JSON string.

    All numpy arrays are converted to Python lists for JSON compatibility.
    """
    d = result.to_dict()
    # Add periodogram if present
    if result.periodogram_periods is not None:
        d["periodogram_periods"] = result.periodogram_periods.tolist()
    if result.periodogram_power is not None:
        d["periodogram_power"] = result.periodogram_power.tolist()
    return json.dumps(d, indent=2, default=_json_default)


def export_phase_fold_csv(
    phase: np.ndarray,
    flux: np.ndarray,
    period_days: float,
    is_synthetic: bool,
) -> str:
    """Export phase-folded data to CSV."""
    df = pd.DataFrame({"phase_days": phase, "flux": flux})
    df["period_days"] = period_days
    df["synthetic"] = is_synthetic
    return df.to_csv(index=False)


def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
