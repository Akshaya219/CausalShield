"""
verification/regenerate_ace_figure.py
Draws a horizontal bar chart of the signed ACE proxy values for the 7 selected features,
sorted by |ace|, with dashed lines at +/-0.05, green borders on bars with |ace|>=0.05,
and a label "retained by minimum-feature rule" on the Duration bar.

Backs up data/causal/ace_scores.png -> ace_scores.png.orig
Saves to:
  data/causal/ace_scores.png   (300 dpi)
  docs/fig3_ace_proxy.png      (300 dpi)

Run from the project root:
    python verification/regenerate_ace_figure.py
"""
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT       = Path(__file__).resolve().parents[1]
ACE_JSON   = ROOT / "data" / "causal" / "ace_scores.json"
ACE_PNG    = ROOT / "data" / "causal" / "ace_scores.png"
ACE_BACKUP = ROOT / "ace_scores.png.orig"
DOCS_DIR   = ROOT / "docs"
FIG3_OUT   = DOCS_DIR / "fig3_ace_proxy.png"

# ── backup existing figure ─────────────────────────────────────────────────
if ACE_PNG.exists():
    shutil.copy2(ACE_PNG, ACE_BACKUP)
    print(f"Backed up {ACE_PNG} -> {ACE_BACKUP}")

# ── load ACE data ──────────────────────────────────────────────────────────
ace_data = json.load(open(ACE_JSON, encoding="utf-8"))

features = list(ace_data.keys())
ace_vals = [ace_data[f]["ace"] for f in features]
abs_vals = [abs(v) for v in ace_vals]

# Sort by |ace| ascending (so highest appears at top of horizontal bar chart)
order = sorted(range(len(features)), key=lambda i: abs_vals[i])
features = [features[i] for i in order]
ace_vals = [ace_vals[i] for i in order]
abs_vals = [abs_vals[i] for i in order]

CUTOFF = 0.05
colors   = []
edgecolors = []
linewidths = []
for v in ace_vals:
    if abs(v) >= CUTOFF:
        colors.append("#2980b9" if v > 0 else "#e74c3c")
        edgecolors.append("#27ae60")
        linewidths.append(2.5)
    else:
        colors.append("#bdc3c7")
        edgecolors.append("#bdc3c7")
        linewidths.append(0.8)

# ── draw figure ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

y_pos = np.arange(len(features))
bars = ax.barh(
    y_pos, ace_vals,
    color=colors, edgecolor=edgecolors, linewidth=linewidths,
    height=0.6, zorder=3
)

# dashed cutoff lines
ax.axvline( CUTOFF, color="#27ae60", linestyle="--", linewidth=1.5, alpha=0.85,
            label=f"|ACE| = {CUTOFF} cutoff", zorder=4)
ax.axvline(-CUTOFF, color="#27ae60", linestyle="--", linewidth=1.5, alpha=0.85,
            zorder=4)
ax.axvline(0, color="black", linewidth=0.8, zorder=4)

# label "retained by minimum-feature rule" on Duration bar
if "Duration" in features:
    dur_idx = features.index("Duration")
    dur_val = ace_vals[dur_idx]
    x_text  = dur_val - 0.012 if dur_val < 0 else dur_val + 0.005
    ha_text = "right" if dur_val < 0 else "left"
    ax.text(
        dur_val, dur_idx,
        "  retained by minimum-feature rule",
        va="center", ha="left", fontsize=7.5,
        color="#8e44ad", style="italic",
    )

ax.set_yticks(y_pos)
ax.set_yticklabels(features, fontsize=10)
ax.set_xlabel("Signed ACE Proxy  (E[Y|X=1] − E[Y|X=0])", fontsize=11)
ax.set_title(
    "CausalShield — Observational ACE Proxy by Feature\n"
    "(flags & rst_count binarized at x>0; Variance & Duration at median)",
    fontsize=11, fontweight="bold"
)

# legend
green_patch = mpatches.Patch(edgecolor="#27ae60", facecolor="none",
                              linewidth=2.5, label="|ACE| ≥ 0.05 (selected)")
grey_patch  = mpatches.Patch(facecolor="#bdc3c7", label="|ACE| < 0.05 (Duration: min-feature rule)")
ax.legend(handles=[green_patch, grey_patch], fontsize=8, loc="lower right")

ax.grid(axis="x", alpha=0.3, zorder=0)
plt.tight_layout()

# ── save ───────────────────────────────────────────────────────────────────
DOCS_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(ACE_PNG,   dpi=300, bbox_inches="tight")
fig.savefig(FIG3_OUT,  dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved: {ACE_PNG}")
print(f"Saved: {FIG3_OUT}")
