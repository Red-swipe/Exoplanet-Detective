"""Tests for the synthetic data generator and data validation."""

import sys
import numpy as np
import pytest

from exoplanet_detective.data import make_synthetic_light_curve, search_target, download_light_curve
from exoplanet_detective.models import LightCurveData


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

class TestMakeSyntheticLightCurve:
    def test_basic_shape(self):
        lc = make_synthetic_light_curve(n_cadences=1000)
        assert len(lc.time) == 1000
        assert len(lc.flux) == 1000

    def test_is_labeled_synthetic(self):
        lc = make_synthetic_light_curve()
        assert lc.is_synthetic is True
        assert "SYNTHETIC" in lc.label

    def test_flux_near_one(self):
        lc = make_synthetic_light_curve(depth=0.01, noise_ppm=100)
        # Most flux values should be close to 1.0
        non_transit = lc.flux[lc.flux > 0.995]
        assert len(non_transit) > len(lc.flux) * 0.8

    def test_transit_depth_visible(self):
        depth = 0.05
        lc = make_synthetic_light_curve(depth=depth, noise_ppm=50, n_cadences=10000)
        min_flux = np.min(lc.flux)
        # Minimum flux should be near 1 - depth
        assert min_flux < 1.0 - depth * 0.5

    def test_reproducible_with_seed(self):
        lc1 = make_synthetic_light_curve(seed=42)
        lc2 = make_synthetic_light_curve(seed=42)
        np.testing.assert_array_equal(lc1.flux, lc2.flux)

    def test_different_seeds_differ(self):
        lc1 = make_synthetic_light_curve(seed=1)
        lc2 = make_synthetic_light_curve(seed=2)
        assert not np.allclose(lc1.flux, lc2.flux)

    def test_flux_err_present(self):
        lc = make_synthetic_light_curve()
        assert lc.flux_err is not None
        assert len(lc.flux_err) == len(lc.flux)

    def test_no_quality_flags(self):
        lc = make_synthetic_light_curve()
        assert lc.quality is None

    def test_mission_label(self):
        lc = make_synthetic_light_curve()
        assert lc.mission == "Synthetic"

    def test_cadence_spacing(self):
        lc = make_synthetic_light_curve(cadence_minutes=30.0, n_cadences=100)
        diffs = np.diff(lc.time)
        expected_days = 30.0 / 1440.0
        np.testing.assert_allclose(diffs, expected_days, rtol=1e-6)

    def test_custom_target_name(self):
        lc = make_synthetic_light_curve(target_name="MY-TARGET")
        assert "MY-TARGET" in lc.target_name

    def test_baseline_length(self):
        lc = make_synthetic_light_curve(n_cadences=1000, cadence_minutes=30.0)
        expected_days = 999 * 30.0 / 1440.0
        assert lc.baseline_days == pytest.approx(expected_days, rel=0.01)


# ---------------------------------------------------------------------------
# LightCurveData validation (used by both real and synthetic paths)
# ---------------------------------------------------------------------------

class TestLightCurveDataValidation:
    def test_all_finite_passes(self):
        lc = make_synthetic_light_curve(n_cadences=100)
        assert np.all(np.isfinite(lc.time))
        assert np.all(np.isfinite(lc.flux))

    def test_n_cadences_matches(self):
        lc = make_synthetic_light_curve(n_cadences=500)
        assert lc.n_cadences == 500


# ---------------------------------------------------------------------------
# Search / download error handling (mocked — no network required)
# ---------------------------------------------------------------------------

class TestSearchTargetErrors:
    def test_unsupported_mission_raises(self):
        with pytest.raises(ValueError, match="Unsupported mission"):
            search_target("Kepler-10", mission="Hubble")

    def test_lightkurve_error_raises_runtime(self, monkeypatch):
        """Simulate a network/search error without importing full lightkurve internals."""
        import unittest.mock as mock
        fake_lk = mock.MagicMock()
        fake_lk.search_lightcurve.side_effect = Exception("Network timeout")
        monkeypatch.setitem(sys.modules, "lightkurve", fake_lk)

        with pytest.raises(RuntimeError, match="Search failed"):
            search_target("Kepler-10", mission="Kepler")

    def test_empty_search_result_raises(self, monkeypatch):
        """Simulate empty search result."""
        import unittest.mock as mock
        fake_lk = mock.MagicMock()
        fake_result = mock.MagicMock()
        fake_result.__len__.return_value = 0
        fake_lk.search_lightcurve.return_value = fake_result
        monkeypatch.setitem(sys.modules, "lightkurve", fake_lk)

        with pytest.raises(RuntimeError, match="No Kepler light curves found"):
            search_target("NONEXISTENT-TARGET-XYZ", mission="Kepler")
