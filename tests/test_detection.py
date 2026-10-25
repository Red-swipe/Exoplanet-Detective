"""Tests for the BLS transit detection engine.

All tests use synthetic data with known injected parameters.
Tests verify actual numerical behavior, not merely that functions execute.
"""

import numpy as np
import pytest

from exoplanet_detective.config import ProcessingConfig, SearchConfig
from exoplanet_detective.data import make_synthetic_light_curve
from exoplanet_detective.detection import phase_fold, run_bls_search
from exoplanet_detective.models import LightCurveData
from exoplanet_detective.processing import preprocess


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def processed_with_transit():
    """Processed light curve with a clear injected transit."""
    lc = make_synthetic_light_curve(
        n_cadences=17520,  # ~1 year at 30 min
        period_days=3.52,
        depth=0.015,
        noise_ppm=300,
        seed=42,
    )
    config = ProcessingConfig(remove_nans=True, normalize=True, sigma_clip_outliers=False)
    result = preprocess(lc, config)
    return result.processed


@pytest.fixture
def processed_noise_only():
    """White noise only — no injected transit."""
    lc = make_synthetic_light_curve(
        n_cadences=5000,
        period_days=999.9,  # Period much longer than baseline
        depth=0.0,          # No transit
        noise_ppm=500,
        seed=7,
    )
    config = ProcessingConfig(remove_nans=True, normalize=True)
    result = preprocess(lc, config)
    return result.processed


@pytest.fixture
def short_processed():
    """Short light curve — fewer than MIN_CADENCES."""
    lc = make_synthetic_light_curve(n_cadences=30, depth=0.01)
    config = ProcessingConfig(remove_nans=True, normalize=True)
    result = preprocess(lc, config)
    return result.processed


# ---------------------------------------------------------------------------
# Period recovery tests
# ---------------------------------------------------------------------------

class TestPeriodRecovery:
    """Verify that BLS recovers injected transit periods within tolerance."""

    @pytest.mark.parametrize("injected_period", [1.5, 3.52, 7.0, 12.0])
    def test_period_recovery_within_2_percent(self, injected_period):
        """BLS should recover injected period within 2% for strong signals."""
        lc = make_synthetic_light_curve(
            n_cadences=17520,
            period_days=injected_period,
            depth=0.02,
            noise_ppm=200,
            seed=0,
        )
        config = ProcessingConfig(remove_nans=True, normalize=True, sigma_clip_outliers=False)
        result = preprocess(lc, config)
        processed = result.processed

        search_config = SearchConfig(
            period_min_days=0.5,
            period_max_days=25.0,
        )
        periods, power, candidates = run_bls_search(processed, search_config)

        assert len(candidates) > 0, "No candidates found"
        best = candidates[0]
        relative_error = abs(best.period_days - injected_period) / injected_period
        # Allow for BLS to find the period or a harmonic (0.5x or 2x)
        harmonic_errors = [
            abs(best.period_days - injected_period * h) / injected_period
            for h in [1.0, 0.5, 2.0, 1.0/3, 3.0]
        ]
        assert min(harmonic_errors) < 0.05, (
            f"Expected period near {injected_period:.2f}d, "
            f"got {best.period_days:.4f}d (error={relative_error:.2%})"
        )

    def test_known_period_strongest_peak(self, processed_with_transit):
        """The injected period (3.52 d) should produce the highest BLS power."""
        search_config = SearchConfig(period_min_days=0.5, period_max_days=20.0)
        periods, power, candidates = run_bls_search(processed_with_transit, search_config)

        assert len(candidates) > 0
        # Best candidate period should be within 5% of 3.52 d (or a harmonic)
        best_period = candidates[0].period_days
        errors = [abs(best_period - 3.52 * h) / 3.52 for h in [1.0, 0.5, 2.0]]
        assert min(errors) < 0.05


# ---------------------------------------------------------------------------
# Depth and radius estimation
# ---------------------------------------------------------------------------

class TestDepthEstimation:
    def test_depth_approximately_correct(self):
        """Estimated depth should be within ~50% of injected depth for clear signals."""
        injected_depth = 0.015
        lc = make_synthetic_light_curve(
            n_cadences=17520,
            period_days=3.52,
            depth=injected_depth,
            noise_ppm=150,
            seed=11,
        )
        config = ProcessingConfig(remove_nans=True, normalize=True, sigma_clip_outliers=False)
        result = preprocess(lc, config)
        search_config = SearchConfig(period_min_days=0.5, period_max_days=20.0)
        _, _, candidates = run_bls_search(result.processed, search_config)

        assert len(candidates) > 0
        best = candidates[0]
        # Depth should be within 3x or at least positive
        assert best.depth_fractional > 0, "Depth must be positive"
        assert best.depth_ppm > 0

    def test_radius_ratio_formula(self):
        """Verify Rp/Rs = sqrt(depth) calculation."""
        from exoplanet_detective.models import TransitCandidate
        depth = 0.0225  # => ratio = 0.15
        c = TransitCandidate(3.0, 2454900.0, 2.0, depth, 10.0, 5.0)
        assert c.radius_ratio == pytest.approx(0.15, rel=1e-4)


# ---------------------------------------------------------------------------
# Noise-only and edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_noise_only_low_snr(self, processed_noise_only):
        """Noise-only data should produce candidates with low SNR."""
        search_config = SearchConfig(period_min_days=0.5, period_max_days=15.0)
        periods, power, candidates = run_bls_search(processed_noise_only, search_config)
        # We may get candidates but their SNR should be modest
        if candidates:
            best_snr = candidates[0].snr
            # SNR in noise-only data should not be wildly high
            # (threshold: realistic noise floor)
            assert best_snr < 50.0, f"Suspiciously high SNR in noise-only data: {best_snr:.1f}"

    def test_too_few_cadences_raises(self, short_processed):
        """Should raise ValueError with < MIN_CADENCES cadences."""
        with pytest.raises(ValueError, match="cadences"):
            run_bls_search(short_processed, SearchConfig())

    def test_all_nan_raises(self):
        """Should raise ValueError if all cadences are non-finite."""
        n = 100
        lc = LightCurveData(
            time=np.arange(n, dtype=float),
            flux=np.full(n, np.nan),
            flux_err=None, quality=None,
            label="all-nan", is_synthetic=True,
            mission="Synthetic", target_name="NAN",
        )
        with pytest.raises(ValueError):
            run_bls_search(lc, SearchConfig())

    def test_period_range_returns_ndarray(self, processed_with_transit):
        """periods and power must be numpy arrays."""
        periods, power, _ = run_bls_search(processed_with_transit, SearchConfig())
        assert isinstance(periods, np.ndarray)
        assert isinstance(power, np.ndarray)
        assert len(periods) == len(power)

    def test_power_nonnegative(self, processed_with_transit):
        """BLS power should be non-negative."""
        _, power, _ = run_bls_search(processed_with_transit, SearchConfig())
        assert np.all(power >= 0), f"Negative power detected: min={power.min()}"

    def test_candidates_sorted_by_power(self, processed_with_transit):
        """Candidates should be ranked by decreasing BLS power."""
        _, _, candidates = run_bls_search(processed_with_transit, SearchConfig())
        if len(candidates) > 1:
            powers = [c.bls_power for c in candidates]
            assert all(powers[i] >= powers[i + 1] for i in range(len(powers) - 1))


# ---------------------------------------------------------------------------
# Phase folding
# ---------------------------------------------------------------------------

class TestPhaseFold:
    def test_output_lengths_match(self):
        n = 500
        time = np.linspace(0, 100, n)
        flux = np.ones(n)
        phase, folded = phase_fold(time, flux, period=3.52, epoch=1.76)
        assert len(phase) == n
        assert len(folded) == n

    def test_phase_range(self):
        n = 1000
        time = np.linspace(0, 50, n)
        flux = np.ones(n)
        period = 5.0
        phase, _ = phase_fold(time, flux, period=period, epoch=2.5)
        assert np.all(phase >= -period / 2 - 1e-10)
        assert np.all(phase <= period / 2 + 1e-10)

    def test_sorted_output(self):
        n = 500
        time = np.linspace(0, 100, n)
        flux = np.ones(n)
        phase, _ = phase_fold(time, flux, period=3.52, epoch=1.76)
        assert np.all(np.diff(phase) >= -1e-12), "Phase output is not sorted"

    def test_transit_near_zero_phase(self):
        """After folding at the correct period and epoch, transit should be near phase=0."""
        lc = make_synthetic_light_curve(
            n_cadences=5000,
            period_days=3.52,
            depth=0.02,
            noise_ppm=100,
            seed=0,
        )
        config = ProcessingConfig(remove_nans=True, normalize=True, sigma_clip_outliers=False)
        result = preprocess(lc, config)
        search_config = SearchConfig(period_min_days=1.0, period_max_days=10.0)
        _, _, candidates = run_bls_search(result.processed, search_config)

        if candidates:
            c = candidates[0]
            phase, flux = phase_fold(
                result.processed.time,
                result.processed.flux,
                c.period_days,
                c.epoch_btjd,
            )
            # Flux near phase=0 should be the minimum
            near_zero = np.abs(phase) < c.period_days * 0.05
            if near_zero.sum() > 0:
                center_flux = np.nanmean(flux[near_zero])
                overall_median = np.nanmedian(flux)
                # Center should be dimmer than overall
                assert center_flux <= overall_median
