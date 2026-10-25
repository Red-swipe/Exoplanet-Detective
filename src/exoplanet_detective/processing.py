"""Light-curve preprocessing pipeline.

This module provides scientifically sensible transformations of raw light curves
while preserving originals. All operations are non-destructive — the raw
LightCurveData is always retained in the returned ProcessedLightCurve.

Assumptions and Limitations
----------------------------
- Normalization divides by the median flux, assuming a stable out-of-transit
  baseline. This can fail for heavily variable stars.
- Sigma clipping removes genuine flares and extreme outliers but may clip
  deep transits if the threshold is too low.
- Savitzky-Golay detrending removes long-term variability but can distort
  or remove transit signals if the window is shorter than the transit duration.
- Quality flag filtering uses the Lightkurve default bitmask when available.
  If the quality array is absent, no filtering is applied.
- This pipeline is not a substitute for mission-specific detrending tools
  (e.g., wotan, CBV correction).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.signal import savgol_filter

from .config import ProcessingConfig
from .models import LightCurveData, ProcessedLightCurve

logger = logging.getLogger("exoplanet_detective.processing")


# ---------------------------------------------------------------------------
# Quality flag bitmask defaults
# ---------------------------------------------------------------------------

# Kepler/K2 default quality bitmask (bits that indicate bad cadences)
KEPLER_QUALITY_BITMASK = 0b0000_0000_0001_0111_1111_1111_1111_1110
# TESS quality bitmask
TESS_QUALITY_BITMASK = 0b0000_0001_0110_0100_1000_0001_0011_1111


def get_quality_bitmask(mission: str) -> int:
    """Return the appropriate quality bitmask for a given mission."""
    if mission.upper() in ("TESS",):
        return TESS_QUALITY_BITMASK
    return KEPLER_QUALITY_BITMASK  # Kepler, K2, Synthetic → no-op if quality is None


# ---------------------------------------------------------------------------
# Main preprocessing function
# ---------------------------------------------------------------------------

def preprocess(
    raw: LightCurveData,
    config: Optional[ProcessingConfig] = None,
) -> ProcessedLightCurve:
    """Run the full preprocessing pipeline on a raw light curve.

    Parameters
    ----------
    raw:
        The input LightCurveData to process.
    config:
        Processing configuration. Uses defaults if None.

    Returns
    -------
    ProcessedLightCurve
        Contains both the original and the processed data plus diagnostics.

    Notes
    -----
    The function gracefully handles:
    - All-NaN flux arrays (returns empty processed set with a warning)
    - Datasets shorter than 10 cadences
    - Irregular cadence spacing
    """
    if config is None:
        config = ProcessingConfig()

    time = raw.time.copy()
    flux = raw.flux.copy()
    flux_err = raw.flux_err.copy() if raw.flux_err is not None else None
    quality = raw.quality.copy() if raw.quality is not None else None

    n_start = len(time)
    n_removed_nan = 0
    n_removed_quality = 0
    n_removed_outlier = 0
    notes: list[str] = []

    # -----------------------------------------------------------------------
    # 1. Remove NaN / infinite values
    # -----------------------------------------------------------------------
    if config.remove_nans:
        valid = np.isfinite(time) & np.isfinite(flux)
        if flux_err is not None:
            valid &= np.isfinite(flux_err)
        n_removed_nan = int((~valid).sum())
        time = time[valid]
        flux = flux[valid]
        if flux_err is not None:
            flux_err = flux_err[valid]
        if quality is not None:
            quality = quality[valid]
        if n_removed_nan:
            notes.append(f"Removed {n_removed_nan} NaN/inf cadences.")
        logger.debug("After NaN removal: %d cadences remain.", len(time))

    # -----------------------------------------------------------------------
    # 2. Quality flag filtering
    # -----------------------------------------------------------------------
    if config.apply_quality_mask and quality is not None:
        bitmask = get_quality_bitmask(raw.mission)
        bad = (quality & bitmask) != 0
        n_removed_quality = int(bad.sum())
        time = time[~bad]
        flux = flux[~bad]
        if flux_err is not None:
            flux_err = flux_err[~bad]
        quality = quality[~bad]
        if n_removed_quality:
            notes.append(f"Removed {n_removed_quality} quality-flagged cadences.")
        logger.debug("After quality filter: %d cadences remain.", len(time))

    # -----------------------------------------------------------------------
    # 3. Normalization
    # -----------------------------------------------------------------------
    if config.normalize and len(flux) > 0:
        median_flux = np.nanmedian(flux)
        if median_flux == 0 or not np.isfinite(median_flux):
            notes.append("WARNING: median flux is zero or non-finite; normalization skipped.")
            logger.warning("Normalization skipped (median=%.6g)", median_flux)
        else:
            flux = flux / median_flux
            if flux_err is not None:
                flux_err = flux_err / median_flux
            notes.append("Flux normalized by median.")

    # -----------------------------------------------------------------------
    # 4. Outlier sigma clipping
    # -----------------------------------------------------------------------
    # In transit astronomy, sigma clipping removes positive spikes (flares, cosmic rays)
    # and extreme non-physical drops, while carefully protecting transit dips from being
    # destroyed. Extreme drops (e.g. > 10x typical depth or > 3*sigma) can be clipped,
    # or positive outliers are clipped more aggressively than negative drops.
    if config.sigma_clip_outliers and len(flux) >= 10:
        sigma = config.sigma_clip_threshold
        median = np.nanmedian(flux)
        mad = np.nanmedian(np.abs(flux - median))
        robust_std = mad * 1.4826  # MAD to sigma conversion
        if robust_std > 0:
            # Clip positive spikes above +sigma, and only extreme negative drops above +(sigma * 10)
            # or negative values <= 0
            pos_outlier = (flux - median) > sigma * robust_std
            neg_extreme = (median - flux) > max(sigma * 10.0 * robust_std, 0.5)
            neg_nonphysical = flux <= 0.0
            outlier_mask = pos_outlier | neg_extreme | neg_nonphysical
            n_removed_outlier = int(outlier_mask.sum())
            time = time[~outlier_mask]
            flux = flux[~outlier_mask]
            if flux_err is not None:
                flux_err = flux_err[~outlier_mask]
            if quality is not None:
                quality = quality[~outlier_mask]
            if n_removed_outlier:
                notes.append(
                    f"Removed {n_removed_outlier} outliers "
                    f"(>{sigma}σ positive spikes or extreme non-physical drops using MAD estimator)."
                )
            logger.debug(
                "After sigma clip (%.1fσ): %d cadences remain.", sigma, len(time)
            )
        else:
            notes.append("Sigma clipping skipped: robust std is zero.")

    # -----------------------------------------------------------------------
    # 5. Detrending (Savitzky-Golay)
    # -----------------------------------------------------------------------
    if config.detrend and len(flux) >= 10:
        flux, notes = _apply_detrending(
            time, flux, flux_err, config.detrend_window_fraction, notes
        )
    elif config.detrend:
        notes.append("Detrending skipped: fewer than 10 cadences remain.")

    # -----------------------------------------------------------------------
    # Final checks
    # -----------------------------------------------------------------------
    n_retained = len(time)
    if n_retained == 0:
        logger.error("All cadences removed during preprocessing!")
        notes.append("ERROR: 0 cadences remain after processing.")
    elif n_retained < 50:
        notes.append(
            f"WARNING: Only {n_retained} cadences remain. "
            "Transit detection may be unreliable."
        )

    processed_lc = LightCurveData(
        time=time,
        flux=flux,
        flux_err=flux_err,
        quality=quality,
        label=f"{raw.label} [processed]",
        is_synthetic=raw.is_synthetic,
        mission=raw.mission,
        target_name=raw.target_name,
        sector_or_quarter=raw.sector_or_quarter,
        exptime_seconds=raw.exptime_seconds,
        n_original=raw.n_original,
    )

    result = ProcessedLightCurve(
        raw=raw,
        processed=processed_lc,
        n_removed_nan=n_removed_nan,
        n_removed_quality=n_removed_quality,
        n_removed_outlier=n_removed_outlier,
        n_retained=n_retained,
        detrended=config.detrend and len(flux) >= 10,
        normalized=config.normalize,
        processing_notes=notes,
    )

    logger.info(
        "Processing complete: %d → %d cadences "
        "(%d NaN, %d quality, %d outlier removed).",
        n_start, n_retained,
        n_removed_nan, n_removed_quality, n_removed_outlier,
    )
    return result


# ---------------------------------------------------------------------------
# Detrending helper
# ---------------------------------------------------------------------------

def _apply_detrending(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: Optional[np.ndarray],
    window_fraction: float,
    notes: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Apply Savitzky-Golay detrending.

    Assumptions
    -----------
    - Regular or quasi-regular cadence.
    - The trend is smooth on time scales longer than the transit duration.
    - Window fraction is relative to the total number of cadences.

    Limitations
    -----------
    - SG detrending can distort transit signals if the window is too short.
    - Gaps in the time series are not explicitly handled (data is treated
      as evenly spaced after gap removal).
    """
    n = len(flux)
    window_len = max(11, int(n * window_fraction))
    if window_len % 2 == 0:
        window_len += 1  # must be odd
    window_len = min(window_len, n - 1 if n % 2 == 0 else n)
    if window_len < 5:
        notes.append("Detrending skipped: window too short.")
        return flux, notes
    try:
        trend = savgol_filter(flux, window_length=window_len, polyorder=2)
        median_flux = np.nanmedian(flux)
        detrended = flux / trend * median_flux
        notes.append(
            f"Applied Savitzky-Golay detrending "
            f"(window={window_len} cadences, polyorder=2). "
            "Long-term trends removed; transit signals should be preserved "
            "if the window is longer than the transit duration."
        )
        logger.debug("SG detrending applied with window_len=%d.", window_len)
        return detrended, notes
    except Exception as exc:  # noqa: BLE001
        notes.append(f"Detrending failed: {exc}. Raw flux retained.")
        logger.warning("SG detrending failed: %s", exc)
        return flux, notes
