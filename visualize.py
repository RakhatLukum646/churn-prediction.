"""Model visualizations built from artifacts in the repo (no raw data needed).

Outputs to assets/:
  feature_importance.png  - top-20 features by LightGBM gain
  score_progression.png   - OOF weighted F1 across model versions
  prediction_distribution.png - class distribution of submission.csv
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("assets")
OUT.mkdir(exist_ok=True)
C = {"main": "#4f46e5", "grey": "#9ca3af",
     "not_churned": "#10b981", "vol_churn": "#f59e0b", "invol_churn": "#ef4444"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})

# 1. Feature importance
fi = pd.read_csv("feature_importance.csv").sort_values("importance", ascending=False).head(20)
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(fi["feature"][::-1], fi["importance"][::-1], color=C["main"])
ax.set_xlabel("Importance (gain)")
ax.set_title("Top-20 features")
fig.tight_layout(); fig.savefig(OUT / "feature_importance.png", dpi=150); plt.close(fig)

# 2. Score progression (values from README results table)
steps = [("Baseline\nLightGBM", 0.5686), ("+ Time\nfeatures", 0.5855),
         ("+ Target enc.\n+ Cascade", 0.6011), ("+ Alpha\nsweep", 0.6129),
         ("Final\n(GPU + thresholds)", 0.6272)]
fig, ax = plt.subplots(figsize=(8, 4.5))
colors = [C["grey"]] * (len(steps) - 1) + [C["main"]]
bars = ax.bar([s[0] for s in steps], [s[1] for s in steps], color=colors)
for b, (_, v) in zip(bars, steps):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.4f}", ha="center")
ax.set_ylim(0.55, 0.64)
ax.set_ylabel("OOF weighted F1")
ax.set_title("Score progression")
fig.tight_layout(); fig.savefig(OUT / "score_progression.png", dpi=150); plt.close(fig)

# 3. Prediction distribution
sub = pd.read_csv("submission.csv")["churn_status"].value_counts()
order = [c for c in ["not_churned", "vol_churn", "invol_churn"] if c in sub.index]
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(order, [sub[c] for c in order], color=[C[c] for c in order])
total = sub.sum()
for b, c in zip(bars, order):
    ax.text(b.get_x() + b.get_width() / 2, sub[c], f"{sub[c]:,}\n({sub[c]/total:.1%})", ha="center", va="bottom")
ax.set_ylim(0, sub.max() * 1.18)
ax.set_title("Predicted class distribution (test set)")
fig.tight_layout(); fig.savefig(OUT / "prediction_distribution.png", dpi=150); plt.close(fig)
print("saved to", OUT.resolve())
