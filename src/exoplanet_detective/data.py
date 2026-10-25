"""Astronomical data retrieval via Lightkurve.

This module isolates all network I/O and Lightkurve API calls from the rest
of the application. It never silently substitutes synthetic data when real
retrieval fails — errors are surfaced to the caller.

Public API
----------
search_target(target, mission) -> list[ObservationRecord]
    Search for available observations without downloading.

download_light_curve(record) -> LightCurveData
    Download and validate a selected observation.

make_synthetic_light_curve(...) -> LightCurveData
    Generate clearly-labeled synthetic data for offline use and tests.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .models import LightCurveData

logger = logging.getLogger("exoplanet_detective.data")


# ---------------------------------------------------------------------------
# Observation record (search result)
# ---------------------------------------------------------------------------

@dataclass
class ObservationRecord:
    """Metadata for a single available observation, before download."""

    target_name: str
    mission: str
    exptime: float          # seconds
    sector_or_quarter: str
    year: Optional[int]
    description: str

    @property
    def display_label(self) -> str:
        return (
            f"{self.mission} | {self.sector_or_quarter} | "
            f"cadence {self.exptime:.0f}s | {self.description}"
        )


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

SUPPORTED_MISSIONS = ("Kepler", "K2", "TESS")


def search_target(
    target: str,
    mission: str = "Kepler",
    exptime: Optional[float] = None,
) -> list[ObservationRecord]:
    """Search for available light-curve observations for a target.

    Parameters
    ----------
    target:
        Catalog identifier or common name (e.g., "Kepler-10", "TIC 261136679").
    mission:
        One of "Kepler", "K2", "TESS".
    exptime:
        Optional exposure time filter in seconds (e.g., 1800 for long cadence).

    Returns
    -------
    list[ObservationRecord]
        Available observations, sorted by sector/quarter.

    Raises
    ------
    ValueError
        If the mission is not supported.
    RuntimeError
        If the search fails due to a network error or empty result.
    """
    if mission not in SUPPORTED_MISSIONS:
        raise ValueError(
            f"Unsupported mission {mission!r}. Choose from {SUPPORTED_MISSIONS}."
        )

    try:
        import lightkurve as lk
    except ImportError as exc:
        raise RuntimeError("lightkurve is not installed.") from exc

    logger.info("Searching %s for target %r (mission=%s)", "MAST", target, mission)
    try:
        search_result = lk.search_lightcurve(target, mission=mission, exptime=exptime)
    except Exception as exc:
        raise RuntimeError(
            f"Search failed for target {target!r}: {exc}"
        ) from exc

    if len(search_result) == 0:
        raise RuntimeError(
            f"No {mission} light curves found for {target!r}. "
            "Try a different target name or mission."
        )

    records: list[ObservationRecord] = []
    for i in range(len(search_result)):
        row = search_result[i]
        try:
            # Extract sector/quarter from description or year
            desc = str(row.description[0]) if hasattr(row, "description") else ""
            sector = str(row.mission[0]) if hasattr(row, "mission") else f"#{i}"
            yr = None
            try:
                yr = int(row.year[0]) if hasattr(row, "year") else None
            except Exception:
                pass
            exp = float(row.exptime[0]) if hasattr(row, "exptime") else 1800.0
            records.append(
                ObservationRecord(
                    target_name=target,
                    mission=mission,
                    exptime=exp,
                    sector_or_quarter=sector,
                    year=yr,
                    description=desc,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not parse search result row %d: %s", i, exc)
            continue

    if not records:
        raise RuntimeError(
            f"Search returned results but none could be parsed for {target!r}."
        )

    logger.info("Found %d observations for %r", len(records), target)
    return records


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_light_curve(
    target: str,
    mission: str,
    sector_or_quarter: Optional[str] = None,
    exptime: Optional[float] = None,
    index: int = 0,
) -> LightCurveData:
    """Download a specific light curve from MAST and return a validated model.

    Parameters
    ----------
    target:
        Target name or catalog ID.
    mission:
        Source mission.
    sector_or_quarter:
        Optional specific sector/quarter string to filter on.
    exptime:
        Optional cadence filter.
    index:
        Which result to download when multiple match (0-indexed).

    Returns
    -------
    LightCurveData
        A validated, structured light-curve container.

    Raises
    ------
    RuntimeError
        On network failure, empty result, or validation failure.
    """
    try:
        import lightkurve as lk
    except ImportError as exc:
        raise RuntimeError("lightkurve is not installed.") from exc

    logger.info("Downloading %s light curve for %r (index=%d)", mission, target, index)

    try:
        search_result = lk.search_lightcurve(target, mission=mission, exptime=exptime)
    except Exception as exc:
        raise RuntimeError(f"Download search failed: {exc}") from exc

    if len(search_result) == 0:
        raise RuntimeError(f"No {mission} data found for {target!r}.")

    if index >= len(search_result):
        index = 0
        logger.warning("Index out of range; downloading first result.")

    try:
        lc_collection = search_result[index].download()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download observation {index} for {target!r}: {exc}"
        ) from exc

    if lc_collection is None:
        raise RuntimeError(f"Download returned None for {target!r}.")

    # Lightkurve may return a LightCurveCollection or a LightCurve
    try:
        import lightkurve as lk
        if hasattr(lc_collection, "stitch"):
            lc = lc_collection.stitch()
        else:
            lc = lc_collection
    except Exception as exc:
        raise RuntimeError(f"Could not process downloaded data: {exc}") from exc

    return _lightkurve_to_model(lc, target, mission)


def _lightkurve_to_model(lc, target_name: str, mission: str) -> LightCurveData:
    """Convert a Lightkurve LightCurve object to our LightCurveData model."""
    try:
        time_arr = np.array(lc.time.value, dtype=np.float64)
        flux_arr = np.array(lc.flux.value, dtype=np.float64)
    except Exception as exc:
        raise RuntimeError(f"Could not extract time/flux arrays: {exc}") from exc

    flux_err_arr: Optional[np.ndarray] = None
    if hasattr(lc, "flux_err") and lc.flux_err is not None:
        try:
            flux_err_arr = np.array(lc.flux_err.value, dtype=np.float64)
        except Exception:
            pass

    quality_arr: Optional[np.ndarray] = None
    if hasattr(lc, "quality") and lc.quality is not None:
        try:
            quality_arr = np.array(lc.quality, dtype=np.int32)
        except Exception:
            pass

    # Sector / quarter
    sector_str = ""
    for attr in ("sector", "quarter", "campaign"):
        val = getattr(lc, attr, None)
        if val is not None:
            sector_str = f"{attr.capitalize()} {val}"
            break

    exptime_s: Optional[float] = None
    if hasattr(lc, "cadenceno") and lc.cadenceno is not None:
        pass
    try:
        exptime_s = float(lc.meta.get("TIMEDEL", 0) * 86400)
    except Exception:
        pass

    n = len(time_arr)
    if n == 0:
        raise RuntimeError(f"Downloaded light curve for {target_name!r} has 0 cadences.")

    finite_mask = np.isfinite(time_arr) & np.isfinite(flux_arr)
    n_finite = int(finite_mask.sum())
    if n_finite < 10:
        raise RuntimeError(
            f"Light curve for {target_name!r} has only {n_finite} finite cadences."
        )

    logger.info(
        "Loaded %d cadences (%d finite) for %r from %s",
        n, n_finite, target_name, mission,
    )

    return LightCurveData(
        time=time_arr,
        flux=flux_arr,
        flux_err=flux_err_arr,
        quality=quality_arr,
        label=f"{target_name} ({mission})",
        is_synthetic=False,
        mission=mission,
        target_name=target_name,
        sector_or_quarter=sector_str,
        exptime_seconds=exptime_s,
        n_original=n,
    )


# ---------------------------------------------------------------------------
# Synthetic data generator (for tests and offline demo)
# ---------------------------------------------------------------------------

def make_synthetic_light_curve(
    n_cadences: int = 35040,
    cadence_minutes: float = 30.0,
    period_days: float = 3.52,
    depth: float = 0.010,
    duration_hours: float = 2.5,
    noise_ppm: float = 500.0,
    seed: int = 42,
    target_name: str = "SYNTHETIC-DEMO-001",
) -> LightCurveData:
    """Generate a clearly labeled synthetic light curve with injected transits.

    This dataset is intended exclusively for offline demonstrations and
    deterministic unit tests. It is never passed off as real data.

    Parameters
    ----------
    n_cadences:
        Number of time steps.
    cadence_minutes:
        Spacing between cadences in minutes.
    period_days:
        Orbital period of the injected planet in days.
    depth:
        Fractional flux depth of the transit (0 < depth < 1).
    duration_hours:
        Transit duration in hours.
    noise_ppm:
        White noise amplitude in parts per million.
    seed:
        Random seed for reproducibility.
    target_name:
        Label for the synthetic target.

    Returns
    -------
    LightCurveData
        Labeled as synthetic (is_synthetic=True).
    """
    rng = np.random.default_rng(seed)

    cadence_days = cadence_minutes / 1440.0
    time = np.arange(n_cadences, dtype=np.float64) * cadence_days + 2454833.0

    flux = np.ones(n_cadences, dtype=np.float64)

    # Inject transits
    duration_days = duration_hours / 24.0
    epoch = time[0] + period_days * 0.5
    phase = ((time - epoch) % period_days)
    # Wrap phase symmetrically around 0
    phase = np.where(phase > period_days / 2, phase - period_days, phase)
    in_transit = np.abs(phase) < duration_days / 2.0
    flux[in_transit] -= depth

    # Add white noise
    noise_fraction = noise_ppm * 1e-6
    flux += rng.normal(0.0, noise_fraction, size=n_cadences)

    # Synthetic uncertainties
    flux_err = np.full(n_cadences, noise_fraction, dtype=np.float64)

    return LightCurveData(
        time=time,
        flux=flux,
        flux_err=flux_err,
        quality=None,
        label=f"[SYNTHETIC] {target_name} — P={period_days:.2f}d depth={depth:.4f}",
        is_synthetic=True,
        mission="Synthetic",
        target_name=target_name,
        sector_or_quarter="Synthetic",
        exptime_seconds=cadence_minutes * 60,
        n_original=n_cadences,
    )
