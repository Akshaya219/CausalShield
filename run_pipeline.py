"""
CausalShield -- Full Pipeline Runner
=====================================
Runs Phases 2 through 5 (+ 4b/4c/4d) in sequence.

Usage:
    python run_pipeline.py              # run all
    python run_pipeline.py --from-phase 3
    python run_pipeline.py --skip-phase 4c
"""

import sys, time, argparse, subprocess
from pathlib import Path

PHASES = {
    2:   ("Phase 2 : FCI Causal Discovery",          "pipeline/phase2_causal_discovery.py"),
    3:   ("Phase 3 : ACE Feature Filtering",          "pipeline/phase3_ace_filtering.py"),
    4:   ("Phase 4 : XGBoost Binary Scoring",         "pipeline/phase4_xgboost_scoring.py"),
    "4b":("Phase 4b: Attack Category Classifier",     "pipeline/phase4b_category_classifier.py"),
    "4c":("Phase 4c: Baseline Comparisons",           "pipeline/phase4c_baselines.py"),
    "4d":("Phase 4d: Three-Tier Risk Scoring",        "pipeline/phase4d_three_tier_scoring.py"),
    5:   ("Phase 5 : SHAP Explainability",            "pipeline/phase5_shap_explainability.py"),
}
ORDER = [2, 3, 4, "4b", "4c", "4d", 5]

def run_phase(key, script_path):
    print("\n" + "=" * 65)
    print(f"  {PHASES[key][0]}")
    print("=" * 65)
    t0     = time.time()
    result = subprocess.run([sys.executable, script_path])
    elapsed= time.time() - t0
    if result.returncode != 0:
        print(f"\n  FAILED (exit {result.returncode})")
        return False
    print(f"\n  Done in {elapsed:.1f}s")
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-phase", default=None)
    parser.add_argument("--skip-phase", default="")
    args = parser.parse_args()

    skip = set(args.skip_phase.split(",")) if args.skip_phase else set()
    base = Path(__file__).resolve().parent

    started = args.from_phase is None
    for key in ORDER:
        if not started:
            if str(key) == str(args.from_phase):
                started = True
            else:
                continue
        if str(key) in skip:
            print(f"\n  Skipping Phase {key}")
            continue
        script = base / PHASES[key][1]
        if not script.exists():
            print(f"\n  Script not found: {script}"); continue
        if not run_phase(key, str(script)):
            print(f"\n  Pipeline stopped at Phase {key}.")
            print(f"  Resume with: python run_pipeline.py --from-phase {key}")
            sys.exit(1)

    print("\n" + "=" * 65)
    print("  All phases complete!")
    print(f"  Launch dashboard: streamlit run {base / 'dashboard' / 'app.py'}")
    print("=" * 65)

if __name__ == "__main__":
    main()
