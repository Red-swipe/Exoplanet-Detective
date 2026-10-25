"""Exoplanet Detective — Interactive Streamlit Dashboard.

Architecture
------------
This module is the presentation layer only. It calls into:
  - data.py        → astronomical data retrieval
  - processing.py  → light-curve preprocessing
  - detection.py   → BLS transit search
  - visualization.py → Plotly figures
  - export.py      → CSV / JSON export

No analysis logic lives here. No results are manufactured.
"""

from __future__ import annotations

import json
import logging
import traceback
from typing import Optional

import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Exoplanet Detective",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark astronomical theme
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Dark backgrounds */
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .css-1d391kg, [data-testid="stSidebar"] { background-color: #161b22; }
    /* Headers */
    h1, h2, h3, h4 { color: #58a6ff; font-family: monospace; }
    /* Metric labels */
    [data-testid="stMetricLabel"] { color: #8b949e; font-size: 0.8rem; }
    [data-testid="stMetricValue"] { color: #c9d1d9; }
    /* Buttons */
    .stButton > button {
        background-color: #21262d;
        color: #c9d1d9;
        border: 1px solid #30363d;
        border-radius: 6px;
    }
    .stButton > button:hover {
        background-color: #30363d;
        border-color: #58a6ff;
        color: #58a6ff;
    }
    /* Info / warning boxes */
    .stAlert { border-radius: 6px; }
    /* Dividers */
    hr { border-color: #21262d; }
    /* Synthetic data banner */
    .synthetic-banner {
        background: #2d1117;
        border: 1px solid #f85149;
        border-radius: 6px;
        padding: 8px 16px;
        color: #f85149;
        font-weight: bold;
        margin-bottom: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

from exoplanet_detective.config import AppConfig, ProcessingConfig, SearchConfig
from exoplanet_detective.data import (
    ObservationRecord,
    download_light_curve,
    make_synthetic_light_curve,
    search_target,
)
from exoplanet_detective.detection import phase_fold, run_bls_search
from exoplanet_detective.export import (
    export_analysis_json,
    export_light_curve_csv,
    export_phase_fold_csv,
)
from exoplanet_detective.models import AnalysisResult, LightCurveData, ProcessedLightCurve
from exoplanet_detective.processing import preprocess
from exoplanet_detective.visualization import (
    plot_periodogram,
    plot_phase_folded,
    plot_processed_comparison,
    plot_processing_diagnostics,
    plot_raw_light_curve,
)

logger = logging.getLogger("exoplanet_detective.app")


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults = {
        "raw_lc": None,
        "processed_result": None,
        "analysis_result": None,
        "search_records": [],
        "selected_obs_index": 0,
        "bls_periods": None,
        "bls_power": None,
        "bls_candidates": [],
        "demo_mode": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _sidebar() -> tuple[bool, str, str, int, ProcessingConfig, SearchConfig]:
    """Render sidebar and return configuration."""
    with st.sidebar:
        st.title("🔭 Exoplanet Detective")
        st.caption("Transit detection from real stellar photometry")
        st.divider()

        demo_mode = st.checkbox(
            "Use synthetic demo data (offline mode)",
            value=st.session_state.get("demo_mode", False),
            help="Run the full pipeline on labeled synthetic data without downloading.",
        )
        st.session_state["demo_mode"] = demo_mode

        if demo_mode:
            st.info(
                "⚠️ SYNTHETIC MODE — All results use generated data. "
                "Not a real astronomical observation."
            )
            target = "SYNTHETIC-DEMO-001"
            mission = "Synthetic"
        else:
            st.subheader("Target Selection")
            target = st.text_input(
                "Target identifier",
                value="Kepler-10",
                placeholder="e.g., Kepler-10, TIC 261136679, KIC 11904151",
                help="Enter a Kepler KIC, TESS TIC, or common name.",
            )
            mission = st.selectbox(
                "Mission",
                ["Kepler", "TESS", "K2"],
                index=0,
            )

        st.divider()
        st.subheader("Preprocessing")
        normalize = st.checkbox("Normalize flux", value=True)
        sigma_clip = st.checkbox("Sigma-clip outliers", value=True)
        sigma_val = 5.0
        if sigma_clip:
            sigma_val = st.slider(
                "Sigma threshold", min_value=2.0, max_value=10.0,
                value=5.0, step=0.5,
            )
        apply_quality = st.checkbox("Apply quality mask", value=True)
        detrend = st.checkbox(
            "Detrend (Savitzky-Golay)",
            value=False,
            help="Remove long-term brightness trends. May distort transit signals if "
            "window is shorter than transit duration.",
        )
        detrend_window = 0.1
        if detrend:
            detrend_window = st.slider(
                "Detrend window fraction",
                min_value=0.02, max_value=0.5, value=0.1, step=0.01,
                help="Window as a fraction of total cadences. Larger = smoother.",
            )

        proc_config = ProcessingConfig(
            remove_nans=True,
            apply_quality_mask=apply_quality,
            normalize=normalize,
            sigma_clip_outliers=sigma_clip,
            sigma_clip_threshold=sigma_val,
            detrend=detrend,
            detrend_window_fraction=detrend_window,
        )

        st.divider()
        st.subheader("Transit Search (BLS)")
        p_min = st.number_input(
            "Min period (days)", min_value=0.1, max_value=10.0, value=0.5, step=0.1,
        )
        p_max = st.number_input(
            "Max period (days)", min_value=1.0, max_value=200.0, value=30.0, step=1.0,
        )
        dur_min = st.number_input(
            "Min transit duration (hours)", min_value=0.1, max_value=6.0, value=0.5, step=0.1,
        )
        dur_max = st.number_input(
            "Max transit duration (hours)", min_value=1.0, max_value=24.0, value=12.0, step=0.5,
        )

        search_config = SearchConfig(
            period_min_days=float(p_min),
            period_max_days=float(p_max),
            duration_hours_min=float(dur_min),
            duration_hours_max=float(dur_max),
        )

        obs_index = st.session_state.get("selected_obs_index", 0)
        return demo_mode, target, mission, obs_index, proc_config, search_config


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

def main() -> None:
    _init_state()
    demo_mode, target, mission, obs_index, proc_config, search_config = _sidebar()

    st.title("🔭 Exoplanet Detective")
    st.caption(
        "Search for periodic transit signals in real stellar photometry. "
        "All results are preliminary candidates, not confirmed discoveries."
    )

    if demo_mode:
        st.markdown(
            '<div class="synthetic-banner">'
            "⚠️ SYNTHETIC DATA MODE — Results below use generated data. "
            "This is not a real astronomical observation or discovery."
            "</div>",
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------------------------
    # SECTION 1: Data acquisition
    # -----------------------------------------------------------------------
    st.header("1 — Data Acquisition")

    if demo_mode:
        _handle_demo_data(proc_config)
    else:
        _handle_real_data(target, mission, proc_config)

    # -----------------------------------------------------------------------
    # SECTION 2: Light-curve visualization
    # -----------------------------------------------------------------------
    if st.session_state["raw_lc"] is not None:
        _section_raw_lightcurve()

    # -----------------------------------------------------------------------
    # SECTION 3: Preprocessing
    # -----------------------------------------------------------------------
    if st.session_state["raw_lc"] is not None:
        _section_preprocessing()

    # -----------------------------------------------------------------------
    # SECTION 4: Transit search
    # -----------------------------------------------------------------------
    if st.session_state["processed_result"] is not None:
        _section_transit_search(search_config)

    # -----------------------------------------------------------------------
    # SECTION 5: Results and export
    # -----------------------------------------------------------------------
    if st.session_state["analysis_result"] is not None:
        _section_results_export()

    # -----------------------------------------------------------------------
    # SECTION 6: Methodology and limitations
    # -----------------------------------------------------------------------
    _section_methodology()


# ---------------------------------------------------------------------------
# Data acquisition handlers
# ---------------------------------------------------------------------------

def _handle_demo_data(proc_config: ProcessingConfig) -> None:
    """Load synthetic demo data."""
    col1, col2 = st.columns([3, 1])
    with col1:
        period_demo = st.number_input(
            "Injected period (days)", min_value=0.5, max_value=30.0, value=3.52, step=0.01,
        )
        depth_demo = st.number_input(
            "Injected transit depth", min_value=0.001, max_value=0.1, value=0.010,
            step=0.001, format="%.4f",
        )
    with col2:
        st.write("")
        st.write("")
        if st.button("🔄 Generate Synthetic Data", use_container_width=True):
            with st.spinner("Generating synthetic light curve…"):
                lc = make_synthetic_light_curve(
                    period_days=float(period_demo),
                    depth=float(depth_demo),
                    n_cadences=17520,
                )
                st.session_state["raw_lc"] = lc
                # Auto-process
                result = preprocess(lc, proc_config)
                st.session_state["processed_result"] = result
                # Clear old search results
                st.session_state["analysis_result"] = None
                st.session_state["bls_periods"] = None
                st.session_state["bls_power"] = None
                st.session_state["bls_candidates"] = []
            st.success(
                f"Generated {lc.n_cadences} cadences | "
                f"Baseline: {lc.baseline_days:.1f} days | "
                f"[SYNTHETIC — not real data]"
            )


def _handle_real_data(target: str, mission: str, proc_config: ProcessingConfig) -> None:
    """Search and download real astronomical data."""
    col_search, col_download = st.columns(2)

    with col_search:
        if st.button("🔍 Search for Observations", use_container_width=True):
            with st.spinner(f"Searching {mission} archive for {target!r}…"):
                try:
                    records = search_target(target, mission=mission)
                    st.session_state["search_records"] = records
                    st.success(f"Found {len(records)} observation(s).")
                except Exception as exc:
                    st.error(f"Search failed: {exc}")
                    st.session_state["search_records"] = []

    records: list[ObservationRecord] = st.session_state.get("search_records", [])
    if records:
        options = [r.display_label for r in records]
        selected = st.selectbox("Select observation", options, index=0)
        obs_index = options.index(selected)
        st.session_state["selected_obs_index"] = obs_index

        with col_download:
            if st.button("⬇️ Download & Load", use_container_width=True):
                with st.spinner("Downloading from MAST archive…"):
                    try:
                        lc = download_light_curve(
                            target=target,
                            mission=mission,
                            index=obs_index,
                        )
                        st.session_state["raw_lc"] = lc
                        # Auto-process
                        result = preprocess(lc, proc_config)
                        st.session_state["processed_result"] = result
                        st.session_state["analysis_result"] = None
                        st.session_state["bls_periods"] = None
                        st.session_state["bls_power"] = None
                        st.session_state["bls_candidates"] = []
                        st.success(
                            f"Loaded {lc.n_cadences} cadences | "
                            f"Baseline: {lc.baseline_days:.1f} days | "
                            f"Mission: {lc.mission}"
                        )
                    except Exception as exc:
                        st.error(f"Download failed: {exc}")
                        logger.error("Download error: %s", traceback.format_exc())


# ---------------------------------------------------------------------------
# Section: raw light curve
# ---------------------------------------------------------------------------

def _section_raw_lightcurve() -> None:
    st.header("2 — Raw Light Curve")
    lc: LightCurveData = st.session_state["raw_lc"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Cadences", f"{lc.n_cadences:,}")
    col2.metric("Baseline", f"{lc.baseline_days:.1f} d")
    col3.metric("Mission", lc.mission)
    col4.metric("Target", lc.target_name)

    fig = plot_raw_light_curve(lc)
    st.plotly_chart(fig, use_container_width=True)

    # Download raw CSV
    raw_csv = export_light_curve_csv(lc)
    st.download_button(
        "⬇️ Download raw light curve CSV",
        data=raw_csv,
        file_name=f"{lc.target_name}_raw.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Section: preprocessing
# ---------------------------------------------------------------------------

def _section_preprocessing() -> None:
    st.header("3 — Preprocessing")
    result: Optional[ProcessedLightCurve] = st.session_state.get("processed_result")

    if result is None:
        st.info("Preprocessing will run automatically after data is loaded.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Original cadences", f"{result.raw.n_cadences:,}")
    col2.metric("Retained", f"{result.n_retained:,}")
    col3.metric("Fraction retained", f"{result.fraction_retained:.1%}")
    col4.metric("Detrended", "Yes" if result.detrended else "No")

    tab_compare, tab_diag, tab_notes = st.tabs(
        ["📊 Raw vs. Processed", "📈 Diagnostics", "📋 Processing Notes"]
    )

    with tab_compare:
        fig_cmp = plot_processed_comparison(result)
        st.plotly_chart(fig_cmp, use_container_width=True)

    with tab_diag:
        fig_diag = plot_processing_diagnostics(result)
        st.plotly_chart(fig_diag, use_container_width=True)

    with tab_notes:
        for note in result.processing_notes:
            if note.startswith("WARNING") or note.startswith("ERROR"):
                st.warning(note)
            else:
                st.write(f"• {note}")

    # Download processed CSV
    proc_csv = export_light_curve_csv(result.processed)
    st.download_button(
        "⬇️ Download processed light curve CSV",
        data=proc_csv,
        file_name=f"{result.raw.target_name}_processed.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Section: transit search
# ---------------------------------------------------------------------------

def _section_transit_search(search_config: SearchConfig) -> None:
    st.header("4 — Transit Search (BLS)")

    result: ProcessedLightCurve = st.session_state["processed_result"]

    if st.button("🔎 Run Transit Search", use_container_width=False):
        with st.spinner("Running BLS period search…"):
            try:
                periods, power, candidates = run_bls_search(
                    result.processed, search_config
                )
                st.session_state["bls_periods"] = periods
                st.session_state["bls_power"] = power
                st.session_state["bls_candidates"] = candidates

                # Build analysis result
                analysis = AnalysisResult(
                    target_name=result.raw.target_name,
                    mission=result.raw.mission,
                    sector_or_quarter=result.raw.sector_or_quarter,
                    is_synthetic=result.raw.is_synthetic,
                    processed_lc=result,
                    candidates=candidates,
                    periodogram_periods=periods,
                    periodogram_power=power,
                    search_config_dict={
                        "period_min_days": search_config.period_min_days,
                        "period_max_days": search_config.period_max_days,
                        "duration_hours_min": search_config.duration_hours_min,
                        "duration_hours_max": search_config.duration_hours_max,
                    },
                    processing_config_dict={
                        "normalize": result.normalized,
                        "detrend": result.detrended,
                    },
                )
                if result.raw.is_synthetic:
                    analysis.warnings.append(
                        "SYNTHETIC DATA — results are for demonstration only."
                    )
                st.session_state["analysis_result"] = analysis

                if candidates:
                    st.success(
                        f"Found {len(candidates)} candidate(s). "
                        f"Best period: {candidates[0].period_days:.4f} days."
                    )
                else:
                    st.warning("No convincing transit candidates detected.")
            except ValueError as exc:
                st.error(f"Search cannot proceed: {exc}")
            except Exception as exc:
                st.error(f"Transit search failed: {exc}")
                logger.error("BLS error: %s", traceback.format_exc())

    # Display results if available
    periods = st.session_state.get("bls_periods")
    power = st.session_state.get("bls_power")
    candidates = st.session_state.get("bls_candidates", [])

    if periods is not None and power is not None:
        is_syn = result.raw.is_synthetic
        tab_pgram, tab_fold = st.tabs(["📈 Periodogram", "🔄 Phase-Folded"])

        with tab_pgram:
            fig_pg = plot_periodogram(periods, power, candidates, is_synthetic=is_syn)
            st.plotly_chart(fig_pg, use_container_width=True)

        with tab_fold:
            if candidates:
                selected_rank = st.selectbox(
                    "Select candidate",
                    [f"#{c.rank} — Period {c.period_days:.4f} d" for c in candidates],
                    index=0,
                )
                rank_idx = int(selected_rank.split("#")[1].split(" ")[0]) - 1
                cand = candidates[rank_idx]

                folded_phase, folded_flux = phase_fold(
                    result.processed.time,
                    result.processed.flux,
                    cand.period_days,
                    cand.epoch_btjd,
                )
                fig_fold = plot_phase_folded(
                    folded_phase, folded_flux, cand, is_synthetic=is_syn
                )
                st.plotly_chart(fig_fold, use_container_width=True)

                # Export phase-folded
                fold_csv = export_phase_fold_csv(
                    folded_phase, folded_flux, cand.period_days, is_syn
                )
                st.download_button(
                    "⬇️ Download phase-folded CSV",
                    data=fold_csv,
                    file_name=f"{result.raw.target_name}_phase_folded.csv",
                    mime="text/csv",
                )
            else:
                st.info("No candidates to phase-fold.")


# ---------------------------------------------------------------------------
# Section: results and export
# ---------------------------------------------------------------------------

def _section_results_export() -> None:
    st.header("5 — Candidate Summary")
    analysis: AnalysisResult = st.session_state["analysis_result"]

    if analysis.is_synthetic:
        st.warning(
            "⚠️ SYNTHETIC DATA — These results are generated from an artificial "
            "light curve and do not represent a real astronomical discovery."
        )

    if not analysis.candidates:
        st.info(
            "No transit candidates were detected. This may mean:\n"
            "- No periodic transit signal is present above the noise\n"
            "- The period is outside the searched range\n"
            "- There are fewer than ~3 transits in the data\n"
            "- Detrending or normalization removed a real signal"
        )
        return

    for c in analysis.candidates:
        with st.expander(
            f"Candidate #{c.rank} — Period: {c.period_days:.4f} d "
            f"| Depth: {c.depth_ppm:.0f} ppm | SNR: {c.snr:.1f}",
            expanded=(c.rank == 1),
        ):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Period", f"{c.period_days:.4f} d")
            col2.metric("Transit depth", f"{c.depth_ppm:.0f} ppm")
            col3.metric("Duration", f"{c.duration_hours:.2f} h")
            col4.metric("Rp/Rs (approx.)", f"{c.radius_ratio:.4f}" if c.radius_ratio else "N/A")

            col5, col6 = st.columns(2)
            col5.metric("BLS power", f"{c.bls_power:.3f}")
            col6.metric("SNR", f"{c.snr:.2f}")

            st.caption(f"⚠️ {c.CAVEATS}")

    # Export
    st.subheader("Export Results")
    col_csv, col_json = st.columns(2)
    with col_csv:
        proc_csv = export_light_curve_csv(analysis.processed_lc.processed)
        st.download_button(
            "⬇️ Download processed LC (CSV)",
            data=proc_csv,
            file_name=f"{analysis.target_name}_processed.csv",
            mime="text/csv",
        )
    with col_json:
        json_str = export_analysis_json(analysis)
        st.download_button(
            "⬇️ Download analysis report (JSON)",
            data=json_str,
            file_name=f"{analysis.target_name}_analysis.json",
            mime="application/json",
        )


# ---------------------------------------------------------------------------
# Section: methodology
# ---------------------------------------------------------------------------

def _section_methodology() -> None:
    st.header("6 — Methodology & Limitations")
    with st.expander("Scientific methodology", expanded=False):
        st.markdown(
            """
### Data Source
Light curves are retrieved from NASA's MAST archive via [Lightkurve](https://lightkurve.github.io/).
Kepler and TESS provide space-based photometry with µmag-level precision.

### Preprocessing
1. **Quality filtering**: Cadences flagged by instrument pipeline are removed.
2. **NaN removal**: Missing or infinite values discarded.
3. **Normalization**: Flux divided by median → fractional flux units.
4. **Sigma clipping**: Outliers beyond Nσ (MAD estimator) removed.
5. **Detrending** (optional): Savitzky-Golay filter removes long-term variability.

### Transit Detection — BLS
The **Box Least Squares** algorithm ([Kovács et al. 2002](https://doi.org/10.1051/0004-6361:20020802))
searches for periodic box-shaped dips.

### Radius Estimation
$$R_p/R_\\star \\approx \\sqrt{\\delta}$$
where δ is the fractional transit depth.
This assumes a central, circular transit with no limb darkening or dilution.

### ⚠️ Limitations & False Positives
- **Eclipsing binaries** can mimic planetary transit depths
- **Stellar activity** (spots, flares, pulsations) produces periodic signals
- **Instrumental systematics** are common in space photometry
- **Insufficient baselines** make long-period planets undetectable
- **BLS power** is not a calibrated probability — do not interpret it as such
- This tool performs no vetting (centroid analysis, secondary eclipse, odd/even depths)
- A periodic signal does **not** confirm an exoplanet exists
"""
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
