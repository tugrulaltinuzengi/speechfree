"""
tester/run_all_tests.py

Master test runner for the entire TTS pipeline.
Runs all test modules and prints a summary report.

Usage:
    python tester/run_all_tests.py           # all tests
    python tester/run_all_tests.py --fast    # skip slow/live tests
    python tester/run_all_tests.py --module test_turkish_cleaners

Dependencies that must be installed for full coverage:
    pip install pydub soundfile numpy scipy onnxruntime

Optional (for speaker similarity):
    pip install resemblyzer
"""

import argparse
import importlib
import os
import sys
import time
import unittest
from pathlib import Path

# All test modules in execution order (fast → slow)
ALL_MODULES = [
    "test_turkish_cleaners",      # pure Python, no deps
    "test_phonetic_sentences",    # file I/O only
    "test_metadata",              # pure Python + wave writing
    "test_preprocess_audio",      # pydub, soundfile
    "test_evaluation",            # soundfile, scipy
    "test_onnx_pipeline",         # numpy, onnxruntime (live test skipped if no model)
]

TESTER_DIR = Path(__file__).parent


def check_dependencies():
    print("=== Dependency Check ===")
    deps = {
        "numpy":        "Core numerics",
        "soundfile":    "WAV I/O",
        "pydub":        "Audio preprocessing",
        "scipy":        "MCD computation",
        "onnxruntime":  "ONNX inference",
    }
    optional = {
        "resemblyzer":  "Speaker similarity (optional)",
        "torch":        "Training (optional, not needed for inference)",
        "TTS":          "coqui-tts training (optional)",
    }
    all_ok = True
    for pkg, desc in deps.items():
        try:
            importlib.import_module(pkg)
            print(f"  [OK]      {pkg:20s} {desc}")
        except ImportError:
            print(f"  [MISSING] {pkg:20s} {desc}  <- pip install {pkg}")
            all_ok = False
    for pkg, desc in optional.items():
        try:
            importlib.import_module(pkg)
            print(f"  [OK]      {pkg:20s} {desc}")
        except ImportError:
            print(f"  [SKIP]    {pkg:20s} {desc}")
    print()
    return all_ok


def run_module(module_name: str, verbosity: int = 1):
    loader = unittest.TestLoader()
    # Load from tester directory
    spec = importlib.util.spec_from_file_location(
        module_name, TESTER_DIR / f"{module_name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)

    suite = loader.loadTestsFromModule(mod)
    stream = open(os.devnull, "w") if verbosity == 0 else sys.stdout
    runner = unittest.TextTestRunner(stream=stream, verbosity=verbosity)
    t0 = time.perf_counter()
    result = runner.run(suite)
    elapsed = time.perf_counter() - t0
    return result, elapsed


def main():
    parser = argparse.ArgumentParser(description="SpeachFree TTS test runner")
    parser.add_argument("--fast", action="store_true", help="Skip slow/live tests")
    parser.add_argument("--module", default=None, help="Run a single test module")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--deps_only", action="store_true", help="Only check dependencies")
    args = parser.parse_args()

    print("=" * 60)
    print("  SpeachFree TTS - Test Suite")
    print("=" * 60)
    print()

    deps_ok = check_dependencies()
    if args.deps_only:
        sys.exit(0 if deps_ok else 1)

    modules = [args.module] if args.module else ALL_MODULES
    verbosity = 2 if args.verbose else 1

    summary = []
    overall_ok = True

    for module_name in modules:
        module_path = TESTER_DIR / f"{module_name}.py"
        if not module_path.exists():
            print(f"[SKIP] {module_name} -file not found")
            continue

        print(f"\n{'-'*60}")
        print(f"  Running: {module_name}")
        print(f"{'-'*60}")

        try:
            result, elapsed = run_module(module_name, verbosity)
            n_run   = result.testsRun
            n_fail  = len(result.failures)
            n_error = len(result.errors)
            n_skip  = len(result.skipped)
            passed  = n_run - n_fail - n_error

            status = "PASS" if result.wasSuccessful() else "FAIL"
            if not result.wasSuccessful():
                overall_ok = False

            summary.append({
                "module": module_name,
                "status": status,
                "passed": passed,
                "failed": n_fail,
                "errors": n_error,
                "skipped": n_skip,
                "time_s": elapsed,
            })

        except Exception as e:
            print(f"[ERROR] Failed to load {module_name}: {e}")
            summary.append({
                "module": module_name,
                "status": "ERROR",
                "passed": 0, "failed": 0, "errors": 1, "skipped": 0,
                "time_s": 0,
            })
            overall_ok = False

    # Summary table
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Module':<35} {'Status':6} {'Pass':>4} {'Fail':>4} {'Err':>4} {'Skip':>4} {'Time':>7}")
    print(f"  {'-'*35} {'-'*6} {'-'*4} {'-'*4} {'-'*4} {'-'*4} {'-'*7}")
    for s in summary:
        mark = "+" if s["status"] == "PASS" else "x"
        print(f"  {mark} {s['module']:<33} {s['status']:6} "
              f"{s['passed']:>4} {s['failed']:>4} {s['errors']:>4} {s['skipped']:>4} "
              f"{s['time_s']:>6.1f}s")

    total_pass = sum(s["passed"] for s in summary)
    total_fail = sum(s["failed"] + s["errors"] for s in summary)
    total_skip = sum(s["skipped"] for s in summary)
    print(f"\n  Total: {total_pass} passed, {total_fail} failed, {total_skip} skipped")

    if overall_ok:
        print("\n  ALL TESTS PASSED - pipeline is ready.")
    else:
        print("\n  SOME TESTS FAILED - review output above.")

    print()

    # ONNX model status reminder
    onnx_enc = TESTER_DIR.parent / "06_deliverable" / "vits_encoder.onnx"
    if not onnx_enc.exists():
        print("  NOTE: ONNX model files not found - live ONNX tests were skipped.")
        print("        Run 03_onnx_export/export_vits.py after training to enable them.")
        print()

    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
