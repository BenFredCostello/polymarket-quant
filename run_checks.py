"""
run_checks.py
-------------
Run all component validation checks in sequence.
Usage: python run_checks.py
"""

import sys
import traceback

CHECKS = [
    ("Black-Scholes Pricer",  "pricing.black_scholes",  "validate_pricer"),
    ("Monte Carlo Pricer",    "pricing.monte_carlo",    "validate_mc_pricer"),
    ("Calibration Module",    "analysis.calibration",   "validate_calibration"),
    ("Correlation Module",    "analysis.correlation",   "validate_correlation"),
    ("Kelly Sizer",           "sizing.kelly",           "validate_kelly"),
]

def main():
    print("\n" + "=" * 60)
    print("  POLYMARKET QUANT — COMPONENT VALIDATION")
    print("=" * 60 + "\n")

    passed = 0
    failed = 0

    for name, module_path, func_name in CHECKS:
        print(f"Testing: {name}...")
        try:
            import importlib
            mod = importlib.import_module(module_path)
            fn = getattr(mod, func_name)
            fn()
            passed += 1
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 60 + "\n")

    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
