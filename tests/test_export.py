"""Tests for export functions."""

import csv
import io
import json

import numpy as np
import pytest

from exoplanet_detective.data import make_synthetic_light_curve
from exoplanet_detective.export import (
    export_analysis_json,
    export_light_curve_csv,
    export_phase_fold_csv,
)
from exoplanet_detective.models import (
    AnalysisResult,
    LightCurveData,
    ProcessedLightCurve,
    TransitCandidate,
)
from exoplanet_detective.processing import preprocess


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_lc():
    return make_synthetic_light_curve(n_cadences=200)


@pytest.fixture
def sample_processed(sample_lc):
    return preprocess(sample_lc)


@pytest.fixture
def sample_analysis(sample_processed):
    candidate = TransitCandidate(3.52, 2454900.0, 2.5, 0.01, 10.0, 8.5)
    return AnalysisResult(
        target_name="SYNTHETIC-DEMO-001",
        mission="Synthetic",
        sector_or_quarter="Synthetic",
        is_synthetic=True,
        processed_lc=sample_processed,
        candidates=[candidate],
        periodogram_periods=np.linspace(0.5, 30, 100),
        periodogram_power=np.random.default_rng(0).random(100),
        search_config_dict={"period_min_days": 0.5, "period_max_days": 30.0},
        processing_config_dict={"normalize": True},
        warnings=["SYNTHETIC DATA"],
    )


# ---------------------------------------------------------------------------
# CSV export tests
# ---------------------------------------------------------------------------

class TestExportLightCurveCSV:
    def test_csv_has_header(self, sample_lc):
        csv_str = export_light_curve_csv(sample_lc)
        reader = csv.DictReader(io.StringIO(csv_str))
        assert "time" in reader.fieldnames
        assert "flux" in reader.fieldnames

    def test_csv_row_count(self, sample_lc):
        csv_str = export_light_curve_csv(sample_lc)
        lines = [l for l in csv_str.strip().split("\n") if l]
        # Header + data rows
        assert len(lines) == sample_lc.n_cadences + 1

    def test_csv_includes_flux_err_when_present(self, sample_lc):
        csv_str = export_light_curve_csv(sample_lc)
        assert "flux_err" in csv_str

    def test_csv_no_flux_err_when_absent(self, sample_lc):
        lc_no_err = LightCurveData(
            time=sample_lc.time,
            flux=sample_lc.flux,
            flux_err=None, quality=None,
            label="no_err", is_synthetic=True,
            mission="Synthetic", target_name="T",
        )
        csv_str = export_light_curve_csv(lc_no_err)
        assert "flux_err" not in csv_str

    def test_csv_values_parseable(self, sample_lc):
        csv_str = export_light_curve_csv(sample_lc)
        df_lines = csv_str.strip().split("\n")
        reader = csv.DictReader(io.StringIO(csv_str))
        for i, row in enumerate(reader):
            time_val = float(row["time"])
            flux_val = float(row["flux"])
            assert np.isfinite(time_val)
            assert np.isfinite(flux_val)
            if i >= 5:
                break


# ---------------------------------------------------------------------------
# JSON export tests
# ---------------------------------------------------------------------------

class TestExportAnalysisJSON:
    def test_valid_json(self, sample_analysis):
        json_str = export_analysis_json(sample_analysis)
        d = json.loads(json_str)  # Should not raise
        assert isinstance(d, dict)

    def test_required_fields(self, sample_analysis):
        d = json.loads(export_analysis_json(sample_analysis))
        required = [
            "target_name", "mission", "is_synthetic",
            "candidates", "search_config", "warnings",
        ]
        for field in required:
            assert field in d, f"Missing field: {field}"

    def test_candidate_fields(self, sample_analysis):
        d = json.loads(export_analysis_json(sample_analysis))
        assert len(d["candidates"]) == 1
        c = d["candidates"][0]
        assert "period_days" in c
        assert "depth_fractional" in c
        assert "radius_ratio_approx" in c
        assert "caveats" in c

    def test_synthetic_warning_in_json(self, sample_analysis):
        d = json.loads(export_analysis_json(sample_analysis))
        assert "synthetic_warning" in d

    def test_periodogram_exported(self, sample_analysis):
        d = json.loads(export_analysis_json(sample_analysis))
        assert "periodogram_periods" in d
        assert "periodogram_power" in d
        assert len(d["periodogram_periods"]) == 100

    def test_json_no_numpy_types(self, sample_analysis):
        """JSON must not contain numpy scalars (not JSON-serializable by default)."""
        json_str = export_analysis_json(sample_analysis)
        # If this passes, numpy types were properly converted
        d = json.loads(json_str)
        assert isinstance(d["candidates"][0]["period_days"], float)


# ---------------------------------------------------------------------------
# Phase-fold CSV tests
# ---------------------------------------------------------------------------

class TestExportPhaseFoldCSV:
    def test_has_required_columns(self):
        phase = np.linspace(-1, 1, 100)
        flux = np.ones(100)
        csv_str = export_phase_fold_csv(phase, flux, period_days=3.52, is_synthetic=True)
        assert "phase_days" in csv_str
        assert "flux" in csv_str
        assert "period_days" in csv_str
        assert "synthetic" in csv_str

    def test_correct_row_count(self):
        n = 150
        phase = np.linspace(-1, 1, n)
        flux = np.ones(n)
        csv_str = export_phase_fold_csv(phase, flux, period_days=3.52, is_synthetic=False)
        lines = [l for l in csv_str.strip().split("\n") if l]
        assert len(lines) == n + 1  # header + data
