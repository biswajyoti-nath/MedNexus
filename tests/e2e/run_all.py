#!/usr/bin/env python3
"""Automated E2E Test Runner for MedNexus AI and NeuroVision MRI Integration.

Usage:
    python tests/e2e/run_all.py
    python tests/e2e/run_all.py --tier 1
    python tests/e2e/run_all.py --tier 2
    python tests/e2e/run_all.py --tier 3
    python tests/e2e/run_all.py --tier 4
    python tests/e2e/run_all.py --mode fixture
    python tests/e2e/run_all.py --mode live --mri-url http://127.0.0.1:8080 --mednexus-url http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import unittest
import warnings
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Filter ResourceWarnings from unclosed HTTP sockets during adversarial error tests
warnings.filterwarnings("ignore", category=ResourceWarning)

from tests.e2e.test_fixtures import TestEnvironment
from tests.e2e.test_tier1_features import TestTier1FeatureCoverage
from tests.e2e.test_tier2_boundaries import TestTier2BoundaryAndCornerCases
from tests.e2e.test_tier3_combinations import TestTier3CrossFeatureCombinations
from tests.e2e.test_tier4_scenarios import TestTier4RealWorldScenarios

TIER_SUITES = {
    "1": ("Tier 1: Feature Coverage (>=5/feature)", TestTier1FeatureCoverage),
    "2": ("Tier 2: Boundary & Corner Cases (>=5/feature)", TestTier2BoundaryAndCornerCases),
    "3": ("Tier 3: Cross-Feature Combinations (Pairwise)", TestTier3CrossFeatureCombinations),
    "4": ("Tier 4: Real-World Scenarios (5 Workloads)", TestTier4RealWorldScenarios),
}


def build_suite(tier: str) -> unittest.TestSuite:
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    if tier.lower() in ("all", "1,2,3,4"):
        for _, (_, test_class) in sorted(TIER_SUITES.items()):
            suite.addTests(loader.loadTestsFromTestCase(test_class))
    else:
        for t in tier.split(","):
            t_key = t.strip()
            if t_key in TIER_SUITES:
                suite.addTests(loader.loadTestsFromTestCase(TIER_SUITES[t_key][1]))
            else:
                print(f"Warning: Unknown tier '{t_key}'. Valid tiers are 1, 2, 3, 4, or all.", file=sys.stderr)
    return suite


def main() -> int:
    parser = argparse.ArgumentParser(description="MedNexus + NeuroVision E2E Test Suite Runner")
    parser.add_argument(
        "--tier",
        default="all",
        help="Tier(s) to run: 1, 2, 3, 4, or all (comma-separated, default: all)",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "live", "fixture"],
        default="auto",
        help="Execution mode: auto (default), live, or fixture",
    )
    parser.add_argument(
        "--mri-url",
        default=os.environ.get("NEUROVISION_URL", "http://127.0.0.1:8080"),
        help="Target URL for NeuroVision MRI service (default: http://127.0.0.1:8080)",
    )
    parser.add_argument(
        "--mednexus-url",
        default=os.environ.get("MEDNEXUS_URL", "http://127.0.0.1:8000"),
        help="Target URL for MedNexus service (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose test output",
    )

    args = parser.parse_args()

    # Configure environment variables
    os.environ["TEST_MODE"] = args.mode
    os.environ["NEUROVISION_URL"] = args.mri_url
    os.environ["MEDNEXUS_URL"] = args.mednexus_url

    print("=" * 72)
    print("MEDNEXUS AI + NEUROVISION MRI — E2E TEST RUNNER")
    print("=" * 72)

    # Initialize environment
    mri_url, mednexus_url, is_live = TestEnvironment.setup(force_mode=args.mode)
    mode_desc = "LIVE SERVICES" if is_live else "CONTRACT-COMPLIANT FIXTURE SERVER"

    print(f"Mode:          {mode_desc}")
    print(f"NeuroVision:   {mri_url}")
    print(f"MedNexus AI:   {mednexus_url}")
    print(f"Target Tiers:  {args.tier}")
    print("-" * 72)

    overall_start = time.time()
    tier_results = {}
    any_failure = False

    try:
        tiers_to_run = (
            list(TIER_SUITES.keys()) if args.tier.lower() == "all" else [t.strip() for t in args.tier.split(",")]
        )

        total_run = 0
        total_failures = 0
        total_errors = 0

        for t_key in tiers_to_run:
            if t_key not in TIER_SUITES:
                continue

            tier_name, test_class = TIER_SUITES[t_key]
            print(f"\n▶ Executing {tier_name}...")

            sub_suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
            runner = unittest.TextTestRunner(
                verbosity=2 if args.verbose else 1,
                stream=sys.stdout,
            )

            t0 = time.time()
            result = runner.run(sub_suite)
            elapsed = time.time() - t0

            passed = result.testsRun - len(result.failures) - len(result.errors)
            tier_results[t_key] = {
                "name": tier_name,
                "total": result.testsRun,
                "passed": passed,
                "failures": len(result.failures),
                "errors": len(result.errors),
                "duration": elapsed,
            }

            total_run += result.testsRun
            total_failures += len(result.failures)
            total_errors += len(result.errors)

            if not result.wasSuccessful():
                any_failure = True

    finally:
        TestEnvironment.teardown()

    overall_elapsed = time.time() - overall_start

    print("\n" + "=" * 72)
    print("E2E TEST SUITE EXECUTION SUMMARY")
    print("=" * 72)
    print(f"{'Tier':<8} | {'Description':<42} | {'Pass / Total':<12} | {'Time':<7}")
    print("-" * 72)

    for t_key, r in tier_results.items():
        status_ratio = f"{r['passed']}/{r['total']}"
        print(f"Tier {t_key:<3} | {r['name'][:42]:<42} | {status_ratio:<12} | {r['duration']:.2f}s")

    print("-" * 72)
    total_passed = total_run - total_failures - total_errors
    print(f"TOTAL:    {total_passed}/{total_run} tests passed (Duration: {overall_elapsed:.2f}s)")

    if any_failure:
        print("\n❌ RESULT: TEST SUITE FAILED (Failures: {}, Errors: {})".format(total_failures, total_errors))
        return 1
    else:
        print("\n✅ RESULT: ALL E2E TESTS PASSED SUCCESSFULLY (100% PASS RATE)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
