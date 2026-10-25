"""Structured data models for light curves, analysis results, and reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# ---------------------------------------------------------------------------
# Light-curve data
# ---------------------------------------------------------------------------

@dataclass
class LightCurveData:
    """A validated, immutable container for a single light-curve observation.

    Attributes
    ----------
    time : np.ndarray
        BJD or BTJD timestamps.
    flux : np.ndarray
        Flux values (normalized or raw, depending on context).
    flux_err : np.ndarray | None
        Flux uncertainties, or None if unavailable.
    quality : np.ndarray | None
        Quality flags from the instrument pipeline.
    label : str
        Human-readable label (e.g., target name + mission).
    is_synthetic : bool
        True if the data were generated synthetically. Always displayed
        to the user when True so no synthetic results are mistaken for
        real observations.
    mission : str
        Source mission name (e.g., "Kepler", "TESS", "Synthetic").
    target_name : str
        Catalog identifier of the target.
    sector_or_quarter : str
        Sector (TESS) or Quarter (Kepler) identifier.
    exptime_seconds : float | None
        Exposure / cadence time in seconds.
    n_original : int
        Number of cadences before any processing.
    """

    time: np.ndarray
    flux: np.ndarray
    flux_err: Optional[np.ndarray]
    quality: Optional[np.ndarray]
    label: str
    is_synthetic: bool
    mission: str
    target_name: str
    sector_or_quarter: str = ""
    exptime_seconds: Optional[float] = None
    n_original: int = 0

    def __post_init__(self) -> None:
        if len(self.time) != len(self.flux):
            raise ValueError(
                f"time ({len(self.time)}) and flux ({len(self.flux)}) must have equal length."
            )
        if self.flux_err is not None and len(self.flux_err) != len(self.flux):
            raise ValueError("flux_err length must match flux length.")
        if self.quality is not None and len(self.quality) != len(self.flux):
            raise ValueError("quality length must match flux length.")
        if self.n_original == 0:
            object.__setattr__(self, "n_original", len(self.time))

    @property
    def n_cadences(self) -> int:
        """Number of cadences in this dataset."""
        return len(self.time)

    @property
    def baseline_days(self) -> float:
        """Observation baseline in days."""
        if len(self.time) < 2:
            return 0.0
        return float(np.nanmax(self.time) - np.nanmin(self.time))

    @property
    def synthetic_warning(self) -> str:
        """Warning string to prepend to any displayed result."""
        if self.is_synthetic:
            return "⚠️ SYNTHETIC DATA — not a real astronomical observation"
        return ""


# ---------------------------------------------------------------------------
# Preprocessing result
# ---------------------------------------------------------------------------

@dataclass
class ProcessedLightCurve:
    """Output of the preprocessing pipeline."""

    raw: LightCurveData
    processed: LightCurveData
    n_removed_nan: int = 0
    n_removed_quality: int = 0
    n_removed_outlier: int = 0
    n_retained: int = 0
    detrended: bool = False
    normalized: bool = False
    processing_notes: list[str] = field(default_factory=list)

    @property
    def fraction_retained(self) -> float:
        if self.raw.n_cadences == 0:
            return 0.0
        return self.n_retained / self.raw.n_cadences


# ---------------------------------------------------------------------------
# Transit candidate
# ---------------------------------------------------------------------------

@dataclass
class TransitCandidate:
    """A single transit candidate returned by the BLS search.

    All depth / radius estimates are approximations. See caveats.
    """

    period_days: float
    epoch_btjd: float
    duration_hours: float
    depth_fractional: float
    bls_power: float
    snr: float
    rank: int = 1

    # Derived (computed post-init)
    radius_ratio: Optional[float] = None  # Rp/Rs approximation
    depth_ppm: Optional[float] = None

    CAVEATS: str = (
        "This is an automated BLS candidate, not a confirmed exoplanet. "
        "Eclipsing binaries, stellar variability, and instrumental artifacts "
        "can produce similar signals. The radius ratio is an approximation "
        "assuming a central transit with no limb darkening."
    )

    def __post_init__(self) -> None:
        if self.depth_fractional > 0:
            self.radius_ratio = float(np.sqrt(self.depth_fractional))
        self.depth_ppm = float(self.depth_fractional * 1e6)


# ---------------------------------------------------------------------------
# Full analysis result
# ---------------------------------------------------------------------------

@dataclass
class AnalysisResult:
    """Complete result of a single target analysis session."""

    target_name: str
    mission: str
    sector_or_quarter: str
    is_synthetic: bool
    processed_lc: ProcessedLightCurve
    candidates: list[TransitCandidate] = field(default_factory=list)
    periodogram_periods: Optional[np.ndarray] = None
    periodogram_power: Optional[np.ndarray] = None
    search_config_dict: dict = field(default_factory=dict)
    processing_config_dict: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def best_candidate(self) -> Optional[TransitCandidate]:
        if self.candidates:
            return self.candidates[0]
        return None

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        result: dict = {
            "target_name": self.target_name,
            "mission": self.mission,
            "sector_or_quarter": self.sector_or_quarter,
            "is_synthetic": self.is_synthetic,
            "n_cadences_raw": self.processed_lc.raw.n_cadences,
            "n_cadences_processed": self.processed_lc.n_retained,
            "baseline_days": self.processed_lc.processed.baseline_days,
            "search_config": self.search_config_dict,
            "processing_config": self.processing_config_dict,
            "warnings": self.warnings,
            "errors": self.errors,
            "candidates": [],
        }
        for c in self.candidates:
            result["candidates"].append(
                {
                    "rank": c.rank,
                    "period_days": c.period_days,
                    "epoch_btjd": c.epoch_btjd,
                    "duration_hours": c.duration_hours,
                    "depth_fractional": c.depth_fractional,
                    "depth_ppm": c.depth_ppm,
                    "radius_ratio_approx": c.radius_ratio,
                    "bls_power": c.bls_power,
                    "snr": c.snr,
                    "caveats": c.CAVEATS,
                }
            )
        if self.is_synthetic:
            result["synthetic_warning"] = (
                "SYNTHETIC DATA — results are for demonstration only."
            )
        return result
