"""Integration tests covering the full analysis pipeline end-to-end."""

import json

import numpy as np
import pytest

from exoplanet_detective.config import ProcessingConfig, SearchConfig
from exoplanet_detective.data import make_synthetic_light_curve
from exoplanet_detective.detection import run_bls_search
from exoplanet_detective.export import export_analysis_json, export_light_curve_csv
from exoplanet_detective.models import AnalysisResult
from exoplanet_detective.processing import preprocess


class TestFullPipeline:
    """End-to-end integration tests using the synthetic demo dataset."""

    @pytest.fixture(autouse=True)
    def setup_pipeline(self):
        """Build a complete analysis result for use in all tests."""
        self.injected_period = 3.52
        self.injected_depth = 0.012

        # 1. Generate synthetic data
        self.lc = make_synthetic_light_curve(
            n_cadences=17520,
            period_days=self.injected_period,
            depth=self.injected_depth,
            noise_ppm=250,
            seed=42,
        )

        # 2. Preprocess
        proc_config = ProcessingConfig(
            remove_nans=True,
            normalize=True,
            sigma_clip_outliers=True,
            sigma_clip_threshold=5.0,
            detrend=False,
        )
        self.proc_result = preprocess(self.lc, proc_config)

        # 3. BLS search
        search_config = SearchConfig(
            period_min_days=0.5,
            period_max_days=20.0,
        )
        self.periods, self.power, self.candidates = run_bls_search(
            self.proc_result.processed, search_config
        )

        # 4. Build analysis result
        self.analysis = AnalysisResult(
            target_name=self.lc.target_name,
            mission=self.lc.mission,
            sector_or_quarter=self.lc.sector_or_quarter,
            is_synthetic=True,
            processed_lc=self.proc_result,
            candidates=self.candidates,
            periodogram_periods=self.periods,
            periodogram_power=self.power,
            search_config_dict={"period_min_days": 0.5, "period_max_days": 20.0},
            processing_config_dict={"normalize": True},
            warnings=["SYNTHETIC DATA"],
        )

    # -----------------------------------------------------------------------
    # Data integrity
    # -----------------------------------------------------------------------

    def test_lc_is_synthetic(self):
        assert self.lc.is_synthetic is True

    def test_processed_has_finite_flux(self):
        assert np.all(np.isfinite(self.proc_result.processed.flux))

    def test_most_cadences_retained(self):
        assert self.proc_result.fraction_retained > 0.95

    # -----------------------------------------------------------------------
    # BLS output structure
    # -----------------------------------------------------------------------

    def test_candidates_found(self):
        assert len(self.candidates) > 0

    def test_periodogram_shapes_match(self):
        assert len(self.periods) == len(self.power)

    def test_best_candidate_period_reasonable(self):
        best = self.candidates[0]
        # Best period should be within an order of magnitude of injected
        assert 0.1 <= best.period_days <= 100.0

    def test_depth_positive(self):
        best = self.candidates[0]
        assert best.depth_fractional >= 0

    def test_radius_ratio_present(self):
        best = self.candidates[0]
        if best.depth_fractional > 0:
            assert best.radius_ratio is not None
            assert best.radius_ratio > 0

    # -----------------------------------------------------------------------
    # Period recovery
    # -----------------------------------------------------------------------

    def test_period_or_harmonic_recovered(self):
        best_period = self.candidates[0].period_days
        harmonics = [self.injected_period * h for h in [1.0, 0.5, 2.0, 1.0/3, 3.0]]
        errors = [abs(best_period - h) / self.injected_period for h in harmonics]
        assert min(errors) < 0.05, (
            f"Best period {best_period:.4f}d not within 5% of injected "
            f"{self.injected_period:.2f}d or its harmonics"
        )

    # -----------------------------------------------------------------------
    # Export validity
    # -----------------------------------------------------------------------

    def test_csv_export_valid(self):
        csv_str = export_light_curve_csv(self.proc_result.processed)
        assert "time" in csv_str
        assert "flux" in csv_str
        lines = csv_str.strip().split("\n")
        assert len(lines) == self.proc_result.n_retained + 1

    def test_json_export_valid(self):
        json_str = export_analysis_json(self.analysis)
        d = json.loads(json_str)
        assert d["target_name"] == self.lc.target_name
        assert d["is_synthetic"] is True
        assert len(d["candidates"]) == len(self.candidates)

    def test_json_contains_periodogram(self):
        d = json.loads(export_analysis_json(self.analysis))
        assert "periodogram_periods" in d
        assert len(d["periodogram_periods"]) == len(self.periods)

    # -----------------------------------------------------------------------
    # Error handling
    # -----------------------------------------------------------------------

    def test_analysis_result_best_candidate(self):
        assert self.analysis.best_candidate is not None
        assert self.analysis.best_candidate.rank == 1

    def test_analysis_warnings_propagate(self):
        assert "SYNTHETIC DATA" in self.analysis.warnings

    def test_to_dict_complete(self):
        d = self.analysis.to_dict()
        assert "synthetic_warning" in d
        assert d["n_cadences_raw"] == self.lc.n_cadences


class TestOfflineDemoMode:
    """Tests that the offline synthetic demo works end-to-end."""

    def test_demo_generates_data(self):
        lc = make_synthetic_light_curve()
        assert lc.n_cadences > 0
        assert lc.is_synthetic

    def test_demo_pipeline_completes(self):
        lc = make_synthetic_light_curve(n_cadences=5000)
        result = preprocess(lc)
        assert result.n_retained > 0
        periods, power, candidates = run_bls_search(
            result.processed,
            SearchConfig(period_min_days=0.5, period_max_days=10.0),
        )
        assert len(periods) > 0
        assert len(power) > 0
