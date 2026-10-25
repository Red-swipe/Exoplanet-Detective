"""Command-line interface for Exoplanet Detective."""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    """Entry point for the exoplanet-detective CLI."""
    parser = argparse.ArgumentParser(
        prog="exoplanet-detective",
        description="Exoplanet Detective — interactive transit detection from real light curves.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # demo command
    demo_parser = subparsers.add_parser(
        "demo",
        help="Run a quick demonstration using synthetic data.",
    )
    demo_parser.add_argument(
        "--period", type=float, default=3.52,
        help="Injected period in days (default: 3.52)"
    )
    demo_parser.add_argument(
        "--depth", type=float, default=0.01,
        help="Transit depth as a fraction (default: 0.01)",
    )

    # launch command
    subparsers.add_parser(
        "launch",
        help="Launch the Streamlit dashboard (alias for: streamlit run ...)",
    )

    args = parser.parse_args()

    if args.command == "demo":
        _run_demo(period=args.period, depth=args.depth)
    elif args.command == "launch":
        _launch_dashboard()
    else:
        parser.print_help()
        return 0

    return 0


def _run_demo(period: float = 3.52, depth: float = 0.01) -> None:
    """Run a quick command-line demonstration."""
    print("\n=== Exoplanet Detective - Synthetic Demo ===")
    print(f"Injected period: {period:.2f} days | Depth: {depth:.4f} ({depth*1e6:.0f} ppm)")
    print("[WARNING] SYNTHETIC DATA - not a real astronomical observation\n")

    from .data import make_synthetic_light_curve
    from .processing import preprocess
    from .detection import run_bls_search
    from .config import ProcessingConfig, SearchConfig

    lc = make_synthetic_light_curve(
        period_days=period,
        depth=depth,
        n_cadences=17520,  # ~1 year at 30 min cadence
    )
    print(f"Generated {lc.n_cadences} cadences over {lc.baseline_days:.1f} days.")

    proc_config = ProcessingConfig(detrend=False)
    result = preprocess(lc, proc_config)
    print(f"After processing: {result.n_retained} cadences retained.")

    search_config = SearchConfig(
        period_min_days=0.5,
        period_max_days=15.0,
    )
    try:
        periods, power, candidates = run_bls_search(result.processed, search_config)
        if candidates:
            c = candidates[0]
            print(f"\nBest candidate:")
            print(f"  Period:   {c.period_days:.4f} days")
            print(f"  Depth:    {c.depth_ppm:.0f} ppm")
            print(f"  Duration: {c.duration_hours:.2f} hours")
            print(f"  Rp/Rs:    {c.radius_ratio:.4f} (approximation only)")
            print(f"  SNR:      {c.snr:.2f}")
            print(f"\n{c.CAVEATS}")
            recovered = abs(c.period_days - period) / period < 0.02
            status_str = "[OK] within 2%" if recovered else "[FAIL] outside 2%"
            print(f"\nPeriod recovery: {status_str}")
        else:
            print("No transit candidates detected.")
    except Exception as exc:
        print(f"Search failed: {exc}")

    print("\n=== Demo complete ===")
    print("Run 'streamlit run src/exoplanet_detective/app.py' for the interactive dashboard.")


def _launch_dashboard() -> None:
    """Launch the Streamlit dashboard."""
    import subprocess
    import sys
    from pathlib import Path

    app_path = Path(__file__).parent / "app.py"
    print(f"Launching dashboard: {app_path}")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path)],
        check=True,
    )


if __name__ == "__main__":
    sys.exit(main())
