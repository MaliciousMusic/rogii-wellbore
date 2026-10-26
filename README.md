# ROGII Wellbore Geology Prediction — Kaggle 2026

**858th / 6,125 teams.** Code competition (offline notebook, 9 h budget), metric: row-pooled RMSE on
**TVT** (true vertical thickness) along horizontal wells, predicted from gamma-ray logs, well
trajectory and a reference *typewell*.

This repository keeps the research code of the competition: the sequence and alignment models, and
the measurements that decided what was worth submitting.

## Modules (`rogii/`)

| File | What it is |
|---|---|
| `seqmodel.py` | 1-D U-Net over the whole well, masked GroupNorm for variable lengths, learned correlation volume between the lateral and the typewell, distributional offset head (soft-argmin over 81 bins) |
| `seqwell.py` | One-well-one-sample autoregressive GRU trained through its own rollouts (differentiable typewell lookup, K = 16 → 64 → 256), implemented from a published specification |
| `level_scorer.py` | Listwise 1-D CNN that scores candidate level offsets; the read-out is the posterior mean, never the argmax |
| `coefnet.py` | One well in, nine numbers out: the Legendre coefficients of the residual curve, so the model never integrates a per-step error. Its failure closed the architecture search |
| `stride.py` | STRIDE: a joint posterior over a sequence of segments (length, dip), pure NumPy, no learning, implemented from a published specification |

Comments are in French. Competition data and other participants' public notebooks are not included.

## What the measurements said

- **Simplicity won, monotonically.** Six public-leaderboard submissions ranked exactly by complexity:
  LightGBM on 126 features **11.69** < 5-expert blend 12.64 < CatBoost on 210 features 12.75 <
  spatial models 15.3–18.4. Cross-validation did not predict the leaderboard (Spearman −0.24).
- **A contaminated public cluster.** The popular ~7.2 public scores rely on an override that injects a
  quantity computed from training labels on wells shared with the test set. Our 11.69 does not use it.
- **A pre-registered stop rule.** The architecture search (U-Net, correlation volume, GRU rollouts,
  listwise scoring, mixture heads) was closed by a rule written *before* the verdict: the level
  coefficient was not learnable from the available signal (R² 0.005 for the best network against
  0.088 for eight hand-built numbers).

## Author

Hadrien Lonjaret — Kaggle `agent33` · [github.com/MaliciousMusic](https://github.com/MaliciousMusic)
