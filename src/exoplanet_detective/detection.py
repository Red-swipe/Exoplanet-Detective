"""BLS transit detection engine.

This module implements a Box Least Squares (BLS) period search using
scipy.signal.lombscargle as the frequency grid helper and a pure-NumPy
BLS implementation for full control over the algorithm.

Scientific Background
---------------------
The Box Least Squares algorithm (Kovács et al. 2002, A&A 391, 369) searches
for periodic, box-shaped flux decrements. For each trial period P and transit
duration q, it computes a signal residue (power) measuring how well a
box-shaped model fits the folded light curve.

We use astropy.timeseries.BoxLeastSquares when available (preferred), falling
back to a pure-NumPy implementation for environments without astropy.

Caveats and Limitations
-----------------------
- BLS is sensitive to red noise and correlated systematic trends.
- The highest-power peak is not necessarily a planetary transit.
- Do not interpret BLS power as a probability.
- Period aliases and harmonics may produce spurious peaks.
- Radius estimation assumes R_p/R_* = sqrt(depth) which requires:
  * central transit geometry
  * no limb darkening
  * no contaminating light
- Three or more transits are recommended for reliable period determination.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .config import SearchConfig
from .models import LightCurveData, TransitCandidate

logger = logging.getLogger("exoplanet_detective.detection")

# Minimum cadences for a meaningful BLS search
MIN_CADENCES = 50


# ---------------------------------------------------------------------------
# Main detection function
# ---------------------------------------------------------------------------

def run_bls_search(
    lc: LightCurveData,
    config: Optional[SearchConfig] = None,
) -> tuple[np.ndarray, np.ndarray, list[TransitCandidate]]:
    """Run a BLS transit search on a processed light curve.

    Parameters
    ----------
    lc:
        Processed light curve (should be normalized).
    config:
        Search configuration. Uses defaults if None.

    Returns
    -------
    periods : np.ndarray
        Trial periods in days.
    power : np.ndarray
        BLS power at each trial period.
    candidates : list[TransitCandidate]
        Ranked transit candidates, best first.

    Raises
    ------
    ValueError
        If the light curve has too few cadences for a meaningful search.
    """
    if config is None:
        config = SearchConfig()

    n = lc.n_cadences
    if n < MIN_CADENCES:
        raise ValueError(
            f"Light curve has {n} cadences; at least {MIN_CADENCES} are needed "
            "for a meaningful BLS search."
        )

    time = lc.time
    flux = lc.flux

    # Validate
    finite_mask = np.isfinite(time) & np.isfinite(flux)
    if finite_mask.sum() < MIN_CADENCES:
        raise ValueError(
            f"Fewer than {MIN_CADENCES} finite cadences available for BLS search."
        )
    time = time[finite_mask]
    flux = flux[finite_mask]

    # Subtract mean for BLS
    flux_centered = flux - np.nanmean(flux)

    logger.info(
        "Running BLS: period [%.2f, %.2f] days, %d cadences",
        config.period_min_days, config.period_max_days, len(time),
    )

    try:
        periods, power = _bls_astropy(time, flux, config)
        logger.debug("Used astropy BLS backend.")
    except Exception as exc:
        logger.warning("astropy BLS failed (%s); falling back to NumPy BLS.", exc)
        periods, power = _bls_numpy(time, flux_centered, config)

    candidates = _extract_candidates(
        time, flux, periods, power, config
    )

    logger.info(
        "BLS search complete. Peak period: %.4f days (power=%.4f)",
        periods[np.argmax(power)], np.max(power),
    )
    return periods, power, candidates


# ---------------------------------------------------------------------------
# astropy BLS backend (preferred)
# ---------------------------------------------------------------------------

def _bls_astropy(
    time: np.ndarray,
    flux: np.ndarray,
    config: SearchConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Run BLS via astropy.timeseries.BoxLeastSquares."""
    from astropy.timeseries import BoxLeastSquares
    import astropy.units as u

    model = BoxLeastSquares(time * u.day, flux)

    baseline = float(np.nanmax(time) - np.nanmin(time))
    min_dur = config.duration_hours_min / 24.0
    max_dur = config.duration_hours_max / 24.0

    minimum_period = max(config.period_min_days, 1.2 * max_dur)
    maximum_period = min(config.period_max_days, baseline / 2.0)

    if maximum_period <= minimum_period:
        maximum_period = minimum_period * 2.0

    # Ensure max duration is strictly less than minimum period
    max_dur = min(max_dur, minimum_period * 0.8)
    if min_dur >= max_dur:
        min_dur = max_dur * 0.5
    duration_days = np.linspace(min_dur, max_dur, 5)

    n_periods = int(config.frequency_factor * baseline / minimum_period)
    n_periods = max(500, min(n_periods, 5000))

    periods = np.linspace(minimum_period, maximum_period, n_periods)

    result = model.power(
        periods * u.day,
        duration_days * u.day,
        objective="snr",
    )

    periods_out = np.array(result.period.to(u.day).value)
    power_out = np.array(result.power)

    return periods_out, power_out


# ---------------------------------------------------------------------------
# NumPy BLS backend (fallback)
# ---------------------------------------------------------------------------

def _bls_numpy(
    time: np.ndarray,
    flux: np.ndarray,
    config: SearchConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Pure-NumPy BLS implementation (simplified, for fallback use).

    Uses phase-folding with bin statistics.
    """
    baseline = float(np.nanmax(time) - np.nanmin(time))
    p_min = max(config.period_min_days, config.duration_hours_min / 12.0)
    p_max = min(config.period_max_days, baseline / 2.0)
    if p_max <= p_min:
        p_max = p_min * 10.0

    n_periods = max(200, int(config.frequency_factor * baseline / p_min))
    periods = np.linspace(p_min, p_max, n_periods)
    power = np.zeros(n_periods)

    duration_days_list = np.linspace(
        config.duration_hours_min / 24.0,
        config.duration_hours_max / 24.0,
        5,
    )

    n_bins = 200
    flux_variance = np.var(flux) if np.var(flux) > 0 else 1.0

    for i, period in enumerate(periods):
        phase = (time % period) / period
        max_power_for_period = 0.0

        for dur in duration_days_list:
            dur_phase = dur / period
            if dur_phase >= 0.5:
                continue

            bin_edges = np.linspace(0, 1, n_bins + 1)
            bin_idx = np.searchsorted(bin_edges, phase, side="right") - 1
            bin_idx = np.clip(bin_idx, 0, n_bins - 1)

            bin_sum = np.bincount(bin_idx, weights=flux, minlength=n_bins)
            bin_count = np.bincount(bin_idx, minlength=n_bins).astype(float)

            with np.errstate(invalid="ignore", divide="ignore"):
                bin_mean = np.where(bin_count > 0, bin_sum / bin_count, 0.0)

            n_dur_bins = max(1, int(dur_phase * n_bins))
            kernel = np.ones(n_dur_bins) / n_dur_bins
            smoothed = np.convolve(bin_mean, kernel, mode="same")
            snr = -np.min(smoothed) / (np.std(flux) + 1e-12)
            if snr > max_power_for_period:
                max_power_for_period = snr

        power[i] = max_power_for_period

    return periods, power


# ---------------------------------------------------------------------------
# Candidate extraction
# ---------------------------------------------------------------------------

def _extract_candidates(
    time: np.ndarray,
    flux: np.ndarray,
    periods: np.ndarray,
    power: np.ndarray,
    config: SearchConfig,
) -> list[TransitCandidate]:
    """Extract top transit candidates from the BLS periodogram.

    Uses a simple peak-finding approach: mask out a window around each
    identified peak before finding the next one.
    """
    if len(power) == 0:
        return []

    candidates: list[TransitCandidate] = []
    power_work = power.copy()
    max_power = np.max(power_work)

    if max_power <= 0:
        logger.warning("BLS power is uniformly non-positive; no candidates.")
        return []

    for rank in range(1, config.max_peaks + 1):
        if np.all(power_work <= 0):
            break

        best_idx = int(np.argmax(power_work))
        best_period = float(periods[best_idx])
        best_power = float(power_work[best_idx])

        if best_power <= 0:
            break

        # Estimate transit parameters at this period
        candidate = _estimate_candidate(
            time, flux, best_period, best_power, rank, config
        )
        if candidate is not None:
            candidates.append(candidate)

        # Mask out a window of ±5% around this period to find next peak
        mask_width = best_period * 0.05
        mask = np.abs(periods - best_period) < mask_width
        # Also mask harmonics
        for harmonic in [2, 3, 0.5, 1/3]:
            harmonic_period = best_period * harmonic
            mask |= np.abs(periods - harmonic_period) < mask_width
        power_work[mask] = 0.0

    return candidates


def _estimate_candidate(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    bls_power: float,
    rank: int,
    config: SearchConfig,
) -> Optional[TransitCandidate]:
    """Estimate transit parameters at a given period by phase-folding."""
    if period <= 0:
        return None

    # Phase-fold relative to time[0]
    t0 = time[0]
    phase = (time - t0) % period
    n_bins = min(200, len(flux))
    bin_edges = np.linspace(0, period, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_idx = np.searchsorted(bin_edges, phase, side="right") - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    bin_sum = np.bincount(bin_idx, weights=flux, minlength=n_bins)
    bin_count = np.bincount(bin_idx, minlength=n_bins).astype(float)

    with np.errstate(invalid="ignore", divide="ignore"):
        bin_mean = np.where(bin_count > 0, bin_sum / bin_count, np.nan)

    if np.all(~np.isfinite(bin_mean)):
        return None

    # Find transit center and depth
    transit_bin = int(np.nanargmin(bin_mean))
    transit_phase = bin_centers[transit_bin]
    epoch = t0 + transit_phase  # Epoch centered on transit dip

    out_of_transit = bin_mean[np.isfinite(bin_mean)]
    out_of_transit_level = np.nanmedian(out_of_transit)
    transit_depth = float(out_of_transit_level - np.nanmin(bin_mean))
    if transit_depth < 0:
        transit_depth = 0.0

    # Estimate transit duration: bins below 50% of depth
    threshold = out_of_transit_level - transit_depth * 0.5
    in_transit_bins = np.isfinite(bin_mean) & (bin_mean < threshold)
    n_in_transit = int(in_transit_bins.sum())
    bin_width_days = period / n_bins
    duration_hours = max(
        config.duration_hours_min,
        min(config.duration_hours_max, n_in_transit * bin_width_days * 24.0),
    )

    # SNR estimate
    out_of_transit_std = np.nanstd(flux) + 1e-12
    snr = transit_depth / out_of_transit_std

    return TransitCandidate(
        period_days=period,
        epoch_btjd=epoch,
        duration_hours=duration_hours,
        depth_fractional=transit_depth,
        bls_power=bls_power,
        snr=float(snr),
        rank=rank,
    )


# ---------------------------------------------------------------------------
# Phase folding utility
# ---------------------------------------------------------------------------

def phase_fold(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Phase-fold a light curve at the given period and epoch.

    Parameters
    ----------
    time, flux:
        Time and flux arrays.
    period:
        Orbital period in days.
    epoch:
        Transit mid-time (same units as time).

    Returns
    -------
    phase : np.ndarray
        Phase in days, centered on transit (range: -period/2 to +period/2).
    flux : np.ndarray
        Corresponding flux values.
    """
    phase = ((time - epoch + period / 2.0) % period) - period / 2.0
    sort_idx = np.argsort(phase)
    return phase[sort_idx], flux[sort_idx]
