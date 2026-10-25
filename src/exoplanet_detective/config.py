"""Configuration, logging setup, and project-wide constants."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure and return the package-level logger."""
    logger = logging.getLogger("exoplanet_detective")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


logger = setup_logging(os.environ.get("EXDET_LOG_LEVEL", "INFO"))


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PACKAGE_DIR = Path(__file__).parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]
CACHE_DIR = Path.home() / ".exoplanet_detective" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Scientific constants
# ---------------------------------------------------------------------------

# Solar radius in km (IAU 2015)
R_SUN_KM = 695_700.0

# Earth radius in km (IAU 2015)
R_EARTH_KM = 6_371.0

# Jupiter radius in km
R_JUPITER_KM = 71_492.0


# ---------------------------------------------------------------------------
# Search defaults
# ---------------------------------------------------------------------------

@dataclass
class SearchConfig:
    """Configuration for a transit search run."""

    period_min_days: float = 0.5
    period_max_days: float = 30.0
    duration_hours_min: float = 0.5
    duration_hours_max: float = 12.0
    frequency_factor: int = 10
    max_peaks: int = 5


@dataclass
class ProcessingConfig:
    """Configuration for light-curve preprocessing."""

    remove_nans: bool = True
    apply_quality_mask: bool = True
    normalize: bool = True
    sigma_clip_outliers: bool = True
    sigma_clip_threshold: float = 5.0
    detrend: bool = False
    detrend_window_fraction: float = 0.1


@dataclass
class AppConfig:
    """Top-level application configuration."""

    search: SearchConfig = field(default_factory=SearchConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    offline_demo: bool = False
    log_level: str = "INFO"
