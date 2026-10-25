"""Tests for the light-curve preprocessing pipeline."""

import numpy as np
import pytest

from exoplanet_detective.config import ProcessingConfig
from exoplanet_detective.data import make_synthetic_light_curve
from exoplanet_detective.models import LightCurveData
from exoplanet_detective.processing import preprocess


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_lc():
    """A clean 1000-cadence synthetic light curve with no defects."""
    return make_synthetic_light_curve(n_cadences=1000, noise_ppm=200, depth=0.01)


@pytest.fixture
def lc_with_nans():
    """Light curve with some NaN flux values injected."""
    lc = make_synthetic_light_curve(n_cadences=500)
    flux = lc.flux.copy()
    # Inject 50 NaNs
    rng = np.random.default_rng(99)
    nan_idx = rng.choice(500, size=50, replace=False)
    flux[nan_idx] = np.nan
    return LightCurveData(
        time=lc.time,
        flux=flux,
        flux_err=lc.flux_err,
        quality=lc.quality,
        label="nans",
        is_synthetic=True,
        mission="Synthetic",
        target_name="NAN-TEST",
    )


@pytest.fixture
def lc_with_outliers():
    """Light curve with extreme outlier spikes."""
    lc = make_synthetic_light_curve(n_cadences=500, noise_ppm=100)
    flux = lc.flux.copy()
    # Add 20 outlier spikes
    rng = np.random.default_rng(77)
    outlier_idx = rng.choice(500, size=20, replace=False)
    flux[outlier_idx] += 10.0  # Extreme positive spikes
    return LightCurveData(
        time=lc.time,
        flux=flux,
        flux_err=lc.flux_err,
        quality=None,
        label="outliers",
        is_synthetic=True,
        mission="Synthetic",
        target_name="OUTLIER-TEST",
    )


@pytest.fixture
def lc_with_quality():
    """Light curve with quality flags set."""
    lc = make_synthetic_light_curve(n_cadences=200)
    quality = np.zeros(200, dtype=np.int32)
    # Mark 30 cadences as bad
    quality[10:40] = 0b10  # bit 1 is set — bad for Kepler
    return LightCurveData(
        time=lc.time,
        flux=lc.flux,
        flux_err=lc.flux_err,
        quality=quality,
        label="quality",
        is_synthetic=True,
        mission="Kepler",
        target_name="QUALITY-TEST",
    )


@pytest.fixture
def tiny_lc():
    """Very short light curve (edge case)."""
    return make_synthetic_light_curve(n_cadences=5)


# ---------------------------------------------------------------------------
# Tests: clean data
# ---------------------------------------------------------------------------

class TestPreprocessClean:
    def test_retains_most_cadences(self, clean_lc):
        result = preprocess(clean_lc, ProcessingConfig(remove_nans=True, normalize=True))
        assert result.fraction_retained > 0.9

    def test_normalized_median_near_one(self, clean_lc):
        result = preprocess(clean_lc, ProcessingConfig(normalize=True))
        assert np.nanmedian(result.processed.flux) == pytest.approx(1.0, abs=0.01)

    def test_no_nans_in_output(self, clean_lc):
        result = preprocess(clean_lc, ProcessingConfig(remove_nans=True))
        assert np.all(np.isfinite(result.processed.flux))

    def test_raw_preserved(self, clean_lc):
        result = preprocess(clean_lc)
        # Raw flux must be unchanged
        np.testing.assert_array_equal(result.raw.flux, clean_lc.flux)

    def test_processing_notes_not_empty(self, clean_lc):
        result = preprocess(clean_lc, ProcessingConfig(normalize=True, remove_nans=True))
        assert len(result.processing_notes) > 0

    def test_is_synthetic_preserved(self, clean_lc):
        result = preprocess(clean_lc)
        assert result.processed.is_synthetic is True


# ---------------------------------------------------------------------------
# Tests: NaN handling
# ---------------------------------------------------------------------------

class TestPreprocessNaN:
    def test_nans_removed(self, lc_with_nans):
        result = preprocess(lc_with_nans, ProcessingConfig(remove_nans=True))
        assert result.n_removed_nan == 50
        assert np.all(np.isfinite(result.processed.flux))

    def test_nans_not_removed_when_disabled(self, lc_with_nans):
        result = preprocess(lc_with_nans, ProcessingConfig(remove_nans=False))
        assert result.n_removed_nan == 0

    def test_all_nan_flux_results_in_zero_retained(self):
        lc = make_synthetic_light_curve(n_cadences=100)
        flux = np.full(100, np.nan)
        bad_lc = LightCurveData(
            time=lc.time, flux=flux, flux_err=None,
            quality=None, label="all-nan", is_synthetic=True,
            mission="Synthetic", target_name="ALL-NAN",
        )
        result = preprocess(bad_lc, ProcessingConfig(remove_nans=True))
        assert result.n_retained == 0


# ---------------------------------------------------------------------------
# Tests: outlier clipping
# ---------------------------------------------------------------------------

class TestPreprocessOutliers:
    def test_outliers_removed(self, lc_with_outliers):
        config = ProcessingConfig(
            sigma_clip_outliers=True,
            sigma_clip_threshold=5.0,
            remove_nans=True,
            normalize=True,
        )
        result = preprocess(lc_with_outliers, config)
        assert result.n_removed_outlier >= 15  # Most of 20 outliers should be removed

    def test_no_clip_retains_all(self, lc_with_outliers):
        config = ProcessingConfig(sigma_clip_outliers=False, remove_nans=False)
        result = preprocess(lc_with_outliers, config)
        assert result.n_removed_outlier == 0

    def test_tight_sigma_removes_more(self, clean_lc):
        loose = preprocess(clean_lc, ProcessingConfig(sigma_clip_outliers=True, sigma_clip_threshold=10.0))
        tight = preprocess(clean_lc, ProcessingConfig(sigma_clip_outliers=True, sigma_clip_threshold=2.0))
        # Tighter threshold removes more or equal
        assert tight.n_removed_outlier >= loose.n_removed_outlier


# ---------------------------------------------------------------------------
# Tests: quality flags
# ---------------------------------------------------------------------------

class TestPreprocessQuality:
    def test_quality_flags_removed(self, lc_with_quality):
        config = ProcessingConfig(apply_quality_mask=True, remove_nans=False)
        result = preprocess(lc_with_quality, config)
        assert result.n_removed_quality > 0

    def test_quality_mask_disabled(self, lc_with_quality):
        config = ProcessingConfig(apply_quality_mask=False, remove_nans=False)
        result = preprocess(lc_with_quality, config)
        assert result.n_removed_quality == 0

    def test_no_quality_array_no_removal(self, clean_lc):
        config = ProcessingConfig(apply_quality_mask=True)
        result = preprocess(clean_lc, config)
        assert result.n_removed_quality == 0


# ---------------------------------------------------------------------------
# Tests: detrending
# ---------------------------------------------------------------------------

class TestPreprocessDetrending:
    def test_detrending_runs_without_crash(self, clean_lc):
        config = ProcessingConfig(detrend=True, detrend_window_fraction=0.1)
        result = preprocess(clean_lc, config)
        assert result.detrended is True

    def test_detrending_preserves_length(self, clean_lc):
        config_base = ProcessingConfig(detrend=False)
        config_det = ProcessingConfig(detrend=True, detrend_window_fraction=0.1)
        r_base = preprocess(clean_lc, config_base)
        r_det = preprocess(clean_lc, config_det)
        assert r_det.n_retained == r_base.n_retained

    def test_detrended_flux_near_one(self, clean_lc):
        """After detrending normalized data, median should still be near 1."""
        config = ProcessingConfig(normalize=True, detrend=True, detrend_window_fraction=0.1)
        result = preprocess(clean_lc, config)
        assert np.nanmedian(result.processed.flux) == pytest.approx(1.0, abs=0.05)


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------

class TestPreprocessEdgeCases:
    def test_short_lc_warning_in_notes(self, tiny_lc):
        result = preprocess(tiny_lc, ProcessingConfig(sigma_clip_outliers=False))
        notes_text = " ".join(result.processing_notes)
        # Should warn about insufficient data
        assert any("cadences" in n.lower() for n in result.processing_notes)

    def test_irregular_cadence_does_not_crash(self):
        """Non-uniform time spacing should not crash the pipeline."""
        rng = np.random.default_rng(0)
        time = np.cumsum(rng.uniform(0.01, 0.05, 200))
        flux = np.ones(200) + rng.normal(0, 0.001, 200)
        lc = LightCurveData(
            time=time, flux=flux, flux_err=None, quality=None,
            label="irregular", is_synthetic=True,
            mission="Synthetic", target_name="IRREG",
        )
        result = preprocess(lc)
        assert result.n_retained > 0

    def test_constant_flux_normalization(self):
        """Constant flux should normalize to exactly 1.0."""
        n = 100
        lc = LightCurveData(
            time=np.arange(n, dtype=float),
            flux=np.full(n, 5000.0),
            flux_err=None, quality=None,
            label="const", is_synthetic=True,
            mission="Synthetic", target_name="CONST",
        )
        result = preprocess(lc, ProcessingConfig(normalize=True, sigma_clip_outliers=False))
        np.testing.assert_allclose(result.processed.flux, 1.0, atol=1e-10)
