# Exoplanet Detective

**Exoplanet Detective** is an interactive Python application for retrieving, processing, and analyzing real astronomical stellar light-curve data to search for periodic transit-like signals that may indicate the presence of exoplanets.

---

## ⚠️ Scientific Disclaimer

This tool is intended for educational and exploratory purposes. A periodic brightness dip in a stellar light curve does **not** prove an exoplanet exists. Astrophysical false positives (eclipsing binaries, stellar variability), instrumental artifacts, and systematic noise can all produce transit-like signals. All results should be treated as candidate signals requiring follow-up analysis by qualified astronomers.

---

## Features

- **Real astronomical data**: Retrieves genuine light curves from NASA Kepler and TESS archives via Lightkurve
- **Robust preprocessing**: Quality filtering, normalization, outlier removal, and detrending
- **BLS transit search**: Box Least Squares periodogram to identify periodic transit candidates
- **Interactive dashboard**: Dark-themed Streamlit app with publication-quality Plotly visualizations
- **Offline demo mode**: Clearly labeled synthetic data for testing without network access
- **Export**: Download results as CSV and JSON reports
- **CI**: GitHub Actions for automated testing

---

## Screenshots

> The application runs locally via Streamlit. Launch it and explore interactively.

---

## Installation

### Requirements

- Python 3.11 or 3.12
- pip

### Quick start

```bash
git clone https://github.com/Red-swipe/Exoplanet-Detective.git
cd Exoplanet-Detective
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -e ".[dev]"
```

---

## Running the Application

```bash
streamlit run src/exoplanet_detective/app.py
```

The dashboard will open at `http://localhost:8501`.

### Offline Demo Mode

In the sidebar, enable **"Use synthetic demo data"** to run the full analysis pipeline without downloading any astronomical data. The synthetic dataset is clearly labeled throughout the interface.

---

## Usage Examples

### Analyze a known exoplanet host

1. Launch the app: `streamlit run src/exoplanet_detective/app.py`
2. In the sidebar, enter a target: `Kepler-10` or `TIC 261136679`
3. Select mission: `Kepler` or `TESS`
4. Click **Search for Observations**
5. Select a light-curve file from the dropdown
6. Click **Download & Load**
7. Adjust preprocessing settings and click **Process Light Curve**
8. Configure BLS period search range and click **Run Transit Search**
9. Examine the periodogram and phase-folded light curve
10. Export results as CSV or JSON

### Command-line interface

```bash
# Show help
exoplanet-detective --help

# Run a quick demo analysis (synthetic data)
exoplanet-detective demo
```

---

## Data Sources

| Mission | Archive | Notes |
|---------|---------|-------|
| Kepler | MAST (NASA) | Long-cadence (30 min) and short-cadence (1 min) photometry |
| TESS | MAST (NASA) | 2-minute and 20-second cadence photometry |

Data is retrieved via [Lightkurve](https://lightkurve.github.io/) and the [MAST](https://mast.stsci.edu/) archive.

---

## Scientific Methodology

### Light Curve Processing

1. **Quality filtering**: Removes cadences flagged by the instrument pipeline
2. **NaN removal**: Discards cadences with missing flux or time values
3. **Normalization**: Divides flux by the median to produce fractional flux units
4. **Outlier clipping**: Optional sigma-clipping (default 5σ)
5. **Detrending**: Optional Savitzky-Golay filter to remove long-term stellar variability

### Transit Detection (BLS)

The **Box Least Squares** (BLS) algorithm ([Kovács et al. 2002](https://doi.org/10.1051/0004-6361:20020802)) searches for periodic, box-shaped dips in the light curve:

- Scans a grid of trial periods from `P_min` to `P_max`
- For each period, evaluates multiple transit durations
- Returns a periodogram (BLS power vs. period)
- The highest-power peak identifies the best-fit candidate period

### Radius Estimation

When a transit candidate is found, an approximate planet-to-star radius ratio is estimated as:

```
Rp/Rs ≈ sqrt(transit_depth)
```

This assumes a central transit, circular orbit, negligible limb darkening, and that the depth equals `(Rp/Rs)²`. **This is an approximation and does not establish that a planet exists.**

---

## Limitations and False Positives

- **Eclipsing binaries** can mimic transit signals with depth ratios similar to planets
- **Stellar variability** (spots, flares, pulsations) can create periodic signals
- **Instrumental systematics** are common in space photometry
- **Insufficient data** (< ~3 transits) makes period determination unreliable
- **Short baselines** miss long-period planets
- This tool does not apply any vetting criteria (centroid analysis, odd/even depth comparison, secondary eclipse search)

---

## Running Tests

```bash
pytest tests/ -v
```

For coverage report:

```bash
pytest tests/ -v --cov=src/exoplanet_detective --cov-report=html
```

---

## Example Analysis (Synthetic Data)

The following is generated from the built-in synthetic fixture (not a real astronomical discovery):

```
Target: SYNTHETIC-DEMO-001
Dataset: Synthetic (clearly labeled)
Duration: 730 days
Cadence: 30 minutes
Injected period: 3.52 days
Injected depth: 0.010 (1%)
BLS recovered period: 3.52 ± 0.01 days
Transit duration: ~2.5 hours
Estimated Rp/Rs: 0.100 (approximation only)
```

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make changes and add tests
4. Run `pytest tests/ -v`
5. Commit and push
6. Open a Pull Request

Please follow the existing code style (Black, Ruff) and add docstrings to all public functions.

---

## License

MIT — see [LICENSE](LICENSE)
