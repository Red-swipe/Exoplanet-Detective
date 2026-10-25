"""Tests for data models."""

import numpy as np
import pytest

from exoplanet_detective.models import (
    AnalysisResult,
    LightCurveData,
    ProcessedLightCurve,
    TransitCandidate,
)


# ---------------------------------------------------------------------------
# LightCurveData
# ---------------------------------------------------------------------------

class TestLightCurveData:
    def _make(self, n=100, **kwargs):
        defaults = dict(
            time=np.linspace(0, 10, n),
            flux=np.ones(n),
            flux_err=np.ones(n) * 0.001,
            quality=None,
            label="test",
            is_synthetic=True,
            mission="Synthetic",
            target_name="TEST-001",
        )
        defaults.update(kwargs)
        return LightCurveData(**defaults)

    def test_basic_creation(self):
        lc = self._make()
        assert lc.n_cadences == 100
        assert lc.baseline_days == pytest.approx(10.0, rel=0.01)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            LightCurveData(
                time=np.arange(10),
                flux=np.arange(20),
                flux_err=None,
                quality=None,
                label="bad",
                is_synthetic=True,
                mission="Synthetic",
                target_name="T",
            )

    def test_flux_err_mismatch_raises(self):
        with pytest.raises(ValueError, match="flux_err length"):
            self._make(n=10, flux_err=np.ones(5))

    def test_quality_mismatch_raises(self):
        with pytest.raises(ValueError, match="quality length"):
            self._make(n=10, quality=np.zeros(5, dtype=int))

    def test_synthetic_warning(self):
        lc = self._make(is_synthetic=True)
        assert "SYNTHETIC" in lc.synthetic_warning

    def test_real_no_synthetic_warning(self):
        lc = self._make(is_synthetic=False)
        assert lc.synthetic_warning == ""

    def test_n_original_set_from_length(self):
        lc = self._make(n=50)
        assert lc.n_original == 50

    def test_baseline_short(self):
        lc = self._make(n=1)
        assert lc.baseline_days == 0.0

    def test_no_flux_err(self):
        lc = self._make(flux_err=None)
        assert lc.flux_err is None


# ---------------------------------------------------------------------------
# TransitCandidate
# ---------------------------------------------------------------------------

class TestTransitCandidate:
    def _make(self, depth=0.01):
        return TransitCandidate(
            period_days=3.52,
            epoch_btjd=2454900.0,
            duration_hours=2.5,
            depth_fractional=depth,
            bls_power=10.0,
            snr=8.5,
        )

    def test_radius_ratio(self):
        c = self._make(depth=0.01)
        assert c.radius_ratio == pytest.approx(0.1, rel=1e-4)

    def test_depth_ppm(self):
        c = self._make(depth=0.01)
        assert c.depth_ppm == pytest.approx(10000.0, rel=1e-4)

    def test_zero_depth(self):
        c = self._make(depth=0.0)
        assert c.radius_ratio is None

    def test_caveats_nonempty(self):
        c = self._make()
        assert len(c.CAVEATS) > 10


# ---------------------------------------------------------------------------
# AnalysisResult
# ---------------------------------------------------------------------------

class TestAnalysisResult:
    def _make_processed(self):
        n = 100
        lc = LightCurveData(
            time=np.linspace(0, 10, n),
            flux=np.ones(n),
            flux_err=None,
            quality=None,
            label="raw",
            is_synthetic=True,
            mission="Synthetic",
            target_name="T",
        )
        return ProcessedLightCurve(
            raw=lc,
            processed=lc,
            n_retained=n,
        )

    def test_best_candidate_none_when_empty(self):
        r = AnalysisResult(
            target_name="T",
            mission="S",
            sector_or_quarter="",
            is_synthetic=True,
            processed_lc=self._make_processed(),
        )
        assert r.best_candidate is None

    def test_to_dict_structure(self):
        r = AnalysisResult(
            target_name="T",
            mission="S",
            sector_or_quarter="Q1",
            is_synthetic=True,
            processed_lc=self._make_processed(),
            candidates=[
                TransitCandidate(3.5, 2454900.0, 2.5, 0.01, 10.0, 8.0)
            ],
        )
        d = r.to_dict()
        assert d["target_name"] == "T"
        assert len(d["candidates"]) == 1
        assert "caveats" in d["candidates"][0]
        assert "synthetic_warning" in d

    def test_to_dict_no_synthetic_warning_for_real(self):
        r = AnalysisResult(
            target_name="T",
            mission="Kepler",
            sector_or_quarter="Q1",
            is_synthetic=False,
            processed_lc=self._make_processed(),
        )
        d = r.to_dict()
        assert "synthetic_warning" not in d
