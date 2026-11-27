# CT Car Crash ML Project

Machine-learning analysis of police-reported crashes in **Bridgeport, CT** (Jan 2021 – Jun 2026) from the
[Connecticut Crash Data Repository](https://www.ctcrash.uconn.edu/).

**Questions**
1. Can ML predict whether a crash is severe (fatal or suspected serious injury, KABCO K/A), and which features drive it?
2. Does ranking crash hotspots by the model's predicted severe-crash burden surface different locations than ranking by crash count?

**Methods:** time-based train/validation/test split; logistic regression, random forest and gradient boosting
(tuned on validation PR-AUC, threshold tuning, class weighting for the 1.4% severe class, probability calibration);
permutation importance; DBSCAN clustering of crash locations into hotspots.

## Files
| Path | Contents |
|---|---|
| `bridgeport_crash_severity.ipynb` | Full analysis: preprocessing, EDA, modelling, evaluation, hotspots, recommendations |
| `data/export_94993_0.csv` | CTCDR MMUCC crash-level export (query 94993) |
| `outputs/hotspots_ranked.csv` | All hotspots with observed crash and severe-crash rankings (2021 – Jun 2026) |
| `outputs/hotspots_ml_vs_count.csv` | Held-out (2025 – Jun 2026) ranking: ML expected severe crashes vs. crash count |

## Run
```
pip install -r requirements.txt
jupyter notebook bridgeport_crash_severity.ipynb
```
Run all cells from the repository root (about a minute).
