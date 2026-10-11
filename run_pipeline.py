"""
CausalShield -- Full Pipeline Runner
=====================================
Runs Phases 2 through 7 (excluding 4b) in sequence.

Usage:
    python run_pipeline.py              # run all
    python run_pipeline.py --from-phase 3
    python run_pipeline.py --skip-phase 4c
"""

import sys, time, argparse, subprocess
from pathlib import Path

PHASES = {
    1:   ("Step 1 : Preprocessing & Splits",          "pipeline/01_preprocess.py"),
    2:   ("Step 2 : FCI Causal Discovery",            "pipeline/02_causal_discovery.py"),
    3:   ("Step 3 : ACE Feature Filtering",           "pipeline/03_ace_filtering.py"),
    4:   ("Step 4 : XGBoost Binary Scoring",          "pipeline/04_xgboost_training.py"),
    5:   ("Step 5 : Baseline Comparisons",            "pipeline/05_baseline_comparisons.py"),
    6:   ("Step 6 : Three-Tier Risk Scoring",         "pipeline/06_three_tier_scoring.py"),
    7:   ("Step 7 : SHAP Explainability",             "pipeline/07_shap_explainability.py"),
    8:   ("Step 8 : ToN-IoT Transfer Evaluation",     "pipeline/08_toniot_transfer.py"),
}
# Map legacy phase arguments to current steps
ALIAS_MAP = {
    "4c": 5, "4C": 5,
    "4d": 6, "4D": 6,
    "5": 7,
    "7": 8,
}
ORDER = [2, 3, 4, 5, 6, 7, 8]

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

    raw_skip = [s.strip() for s in args.skip_phase.split(",") if s.strip()]
    skip = {ALIAS_MAP.get(s, int(s) if s.isdigit() else s) for s in raw_skip}
    from_step = args.from_phase
    if from_step is not None:
        from_step = ALIAS_MAP.get(from_step, int(from_step) if str(from_step).isdigit() else from_step)

    base = Path(__file__).resolve().parent

    started = from_step is None
    for key in ORDER:
        if not started:
            if key == from_step or str(key) == str(from_step):
                started = True
            else:
                continue
        if key in skip or str(key) in skip:
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
