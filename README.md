# Shot Quality From Public Data

Code and data for **"Shot Quality From Public Data: Separating Shot Generation From Shot-Making in the NBA."**

The project builds a shot-quality model from free, public NBA play-by-play data. The model estimates the probability that a league-average shooter makes a given shot, using only where, when and how the shot was taken. It does not know who took the shot. Those probabilities become expected points, which support three analyses:

- **Model ceiling:** how close public data gets to tracking-data models.
- **Skill vs. luck:** whether Shooting Performance Over Expectation (SPOE) is persistent skill or noise, using exact Bernoulli confidence bands.
- **Winning:** how much of a team's record comes from shot generation (the offense's process) versus shot-making (the roster), via Pythagorean expected wins.

---

## Key results (2024–25, held-out test set of 43,906 shots)

| Result | Value |
|---|---|
| LightGBM test AUC / accuracy | **0.665** / 63.5% (always-miss baseline: 53.3%) |
| Calibration (mean absolute error) | 1.0 percentage point |
| Shot location alone (`xLegacy`, `yLegacy`, `shot_angle_rad`) | AUC 0.643 |
| Adding shooter identity (Model A, ablation only) | AUC 0.667 |
| Seven tree-ensemble models | within 0.003 AUC of each other |
| Year-to-year validation (train season X, test X+1) | AUC 0.656–0.667 |
| Year-over-year SPOE correlation (≥200 attempts in both seasons) | r = 0.570 |
| Share of between-player SPOE variance that is persistent skill | 57.5% |
| Expected wins vs. actual wins (30 teams) | r = 0.65 |

---

## Which model is which

The paper's model is **Model B**. The other two notebooks are kept for comparison.

| Model | Notebook | What it sees | Role |
|---|---|---|---|
| **B (paper)** | [`NBA_Shot_Quality_Modeling_XGB_CatBoost_NO_PRIOR.ipynb`](NBA_Shot_Quality_Modeling_XGB_CatBoost_NO_PRIOR.ipynb) | Shot context only. No shooter history, no team identity | All results in the paper |
| A | [`NBA_Shot_Quality_Modeling_XGB_CatBoost.ipynb`](NBA_Shot_Quality_Modeling_XGB_CatBoost.ipynb) | Model B's inputs plus the shooter's prior-season FG%, 3PT%, 2PT% and attempts | "+ shooter identity" ablation only |
| C | [`NBA_Shot_Quality_Modeling_XGB_CatBoost_NO_PRIOR_KEEP_TEAM.ipynb`](NBA_Shot_Quality_Modeling_XGB_CatBoost_NO_PRIOR_KEEP_TEAM.ipynb) | Model B's inputs plus team identity | Diagnostic only |

Shooter identity is excluded from Model B on purpose. Shot quality should score the shot, not the shooter, and SPOE is only meaningful if the expected points don't already encode shooter skill. See [`NO_PRIOR_ABLATION_REVIEW.md`](NO_PRIOR_ABLATION_REVIEW.md).

---

## Reproducing the paper

Notebook sections are listed by their headings as they appear in the notebook.

| Paper result | Where it is produced | Output |
|---|---|---|
| Enriched shot data (reconstructed shot clock, contest score) | [`enrich_shots.py`](enrich_shots.py) | `enriched_data/` |
| Shot-clock validation (83.8–86.4% agreement bound with NBA.com) | [`validate_shot_clock.py`](validate_shot_clock.py) | [`shot_clock_validation/summary_all_seasons.md`](shot_clock_validation/summary_all_seasons.md) |
| Model comparison (12 models + voting ensemble) | Model B, *5) Model comparison* | `model_results_no_prior.csv` |
| Year-to-year validation | [`NBA_Shot_Quality_Year_to_Year_Validation_NO_PRIOR.ipynb`](NBA_Shot_Quality_Year_to_Year_Validation_NO_PRIOR.ipynb) | in notebook |
| Feature importance | Model B, *Appendix: Feature importances* | in notebook |
| Feature-tier ablation, incl. location-only baseline | Model B, *Ablation Study: Feature Tier Contribution* | in notebook |
| Calibration curve and confusion matrix | Model B, *Appendix: Calibration curve (test set)* | in notebook |
| Player SPOE distributions and leaders | Model B, *11.5) Player Performance vs Expected* and *11.6) Key Visualizations* | in notebook |
| SPOE uncertainty and skill vs. luck (reliability, year-over-year r) | [`spoe_analysis.py`](spoe_analysis.py) | [`spoe_analysis/summary.md`](spoe_analysis/summary.md) |
| Team expected wins, team table, expected-vs-actual scatter | Model B, sections *8)* through *11)* | `season_standings_no_prior.csv` |
| "+ shooter identity" ablation (AUC 0.667) | Model A | `model_results.csv` |
| Open shots vs. wins | [`Open_Shots_vs_Wins_Analysis.ipynb`](Open_Shots_vs_Wins_Analysis.ipynb) | in notebook |
| Exploratory figures (shot charts, eFG% maps) | [`NBA_Shot_Data_Exploration_2024_25_FULL.ipynb`](NBA_Shot_Data_Exploration_2024_25_FULL.ipynb) | in notebook |

Model B trains on an 80/20 split of the 2024–25 season (`RANDOM_STATE = 42`, stratified). The fitted models are cached in `models/fitted_models_no_prior.pkl`, so re-running the notebook reloads them instead of retraining.

---

## Getting the data

### Enriched shot data and cached models (Git LFS)

`enriched_data/*.csv` (about 0.7 GB) and `models/*.pkl` (about 1.2 GB) are stored with [Git LFS](https://git-lfs.com). Install Git LFS **before** cloning, or you will get small pointer files instead of data:

```bash
git lfs install
git clone https://github.com/bnamm1/nba-shot-quality.git
```

To download only what Model B needs (the 2024–25 shots):

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/bnamm1/nba-shot-quality.git
cd nba-shot-quality
git lfs pull --include="enriched_data/nbastatsv3_2024_enriched_shots.csv"
```

`enriched_data/` is the exact input used in the paper (produced by `enrich_shots.py`).

### Raw play-by-play (not stored in this repo)

Raw season files come from [shufinskiy/nba_data](https://github.com/shufinskiy/nba_data) (NBA Stats API v3 play-by-play). They are needed to:

- regenerate `enriched_data/` with `enrich_shots.py`;
- compute actual win totals in Model B, section *8)*, which reads `raw_data/nbastatsv3_2024.csv`.

```bash
mkdir -p raw_data
for y in $(seq 2013 2024); do
  curl -L -o raw_data/nbastatsv3_$y.tar.xz \
    https://github.com/shufinskiy/nba_data/raw/main/datasets/nbastatsv3_$y.tar.xz
  tar -xJf raw_data/nbastatsv3_$y.tar.xz -C raw_data && rm raw_data/nbastatsv3_$y.tar.xz
done
```

Seasons are named by their starting year (`2024` = 2024–25). The raw files total about 7.05 million events; filtering to field-goal attempts leaves about 2.51 million shots (about 219k per season).

`player_historical_stats.csv` and `team_defensive_stats.csv` (prior-season lookups) are included in the repo.

---

## Quick start

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn lightgbm xgboost catboost optuna matplotlib jupyter requests

# Paper model (Model B)
jupyter notebook NBA_Shot_Quality_Modeling_XGB_CatBoost_NO_PRIOR.ipynb

# Supporting analyses
python spoe_analysis.py                    # skill vs. luck -> spoe_analysis/ (needs all 12 enriched seasons)
python validate_shot_clock.py --seasons 2024   # shot-clock check -> shot_clock_validation/

# Regenerate enriched data from raw play-by-play (optional; needs raw_data/)
python enrich_shots.py --seasons 2024 --out-dir enriched_data
```

---

## Features (Model B)

Model B uses 63 numeric and 7 categorical inputs, all derived from play-by-play data and prior-season team defense.

| Group | Features |
|---|---|
| Location and distance | `shotDistance`, `xLegacy`, `yLegacy`, `shot_angle_rad`, `shot_distance_log`, `shot_distance_sq`, `shotValue`, `is_paint`, `is_midrange`, `is_corner3`, `is_above_break3` |
| Game time | `period`, `ABS_TIME`, `period_sec_remaining`, `period_sec_elapsed`, `game_sec_remaining_reg` |
| Shot clock (reconstructed) | `SHOT_CLOCK_APPROX`, `is_early_clock`, `is_late_clock`, `is_heave` |
| Shot type (parsed from the play description) | 22 indicators, e.g. `is_dunk`, `is_layup`, `is_pullup`, `is_stepback`, `is_fadeaway`, `is_driving`, `is_floating`, `is_tip`, `is_putback`, `is_second_chance` |
| Contest proxy | `contest_score`, `contest_x_distance`, `contest_x_shot_clock` |
| Clock × location interactions | 13 features, e.g. `clock_x_distance`, `is_transition_paint`, `is_late_clock_midrange`, `is_desperation_3pt`, `is_rhythm_shot` |
| Prior-season opponent defense | `opp_def_fg_pct`, `opp_def_rim_pct`, `opp_def_3pt_pct`, `opp_def_mid_pct`, `opp_def_shot_type_pct` |
| Categorical | `subType`, `contest_label`, `location`, `distance_bucket`, `shot_clock_bucket`, `shot_side`, `shot_zone_simple` |

**Excluded on purpose:**
- the shooter's prior-season stats (`prior_fg_pct`, `prior_fg3_pct`, `prior_fg2_pct`, `prior_attempts`, `prior_shot_type_pct`);
- team identity (`teamTricode`).

Prior-season opponent defense is kept because it describes the shot's difficulty, not who is shooting. All prior-season values come from seasons before the training period, so nothing leaks from the future.

By total gain, the most important features are shot location (angle and court coordinates), game time and prior-season opponent defense.

---

## Models

Twelve classifiers plus a soft-voting ensemble. Cross-validation is 3-fold stratified, reported as mean ± SD.

| Group | Models |
|---|---|
| Optuna-tuned (Bayesian, TPE sampler, 10 trials, 3-fold CV on ROC AUC) | LightGBM, XGBoost, CatBoost, Gradient Boosting, HistGradientBoosting |
| Fixed hyperparameters | Logistic Regression, k-Nearest Neighbors, Random Forest, Extra Trees, AdaBoost, MLP, Gaussian Naive Bayes |
| Ensemble | Soft voting over the five tuned boosters |

LightGBM is the scoring model for every downstream analysis. The tuned parameters for each model are in `model_results_no_prior.csv` (`best_params` column), and the fixed settings are in the Model B notebook, *4) Models to try*.

Other settings: `PYTH_EXP = 14` (Pythagorean exponent), `RANDOM_STATE = 42`.

---

## Technical details

### Approximate shot clock (`SHOT_CLOCK_APPROX`)

Public play-by-play has no shot clock, so it is reconstructed from possession events (`enrich_shots.py`, `compute_shot_clock_v5`):

1. **Resets to 24 s:** made field goal, turnover, defensive rebound.
2. **Resets to 14 s:** offensive rebound and certain defensive violations. This rule is applied only from 2018–19 on, when the rule took effect. Earlier seasons reset to 24.
3. **Shot clock** = reset value − seconds elapsed since the last reset (from the `clock` field).

Validated against NBA.com's tracking dashboard, which reports attempts per shot-clock range: agreement bound 83.8–86.4% across all twelve seasons ([details](shot_clock_validation/summary_all_seasons.md)). About 2.9% of attempts have no reconstructable shot clock and are median-imputed at model time.

Known limitations:
- No inbound delay is modeled, so the 24–22 s range is under-populated.
- The "minimum 14" rule after offensive rebounds is approximated.

### Contest heuristic (`contest_score`, `contest_label`)

Defender distance is not public after 2015–16, so shot difficulty is estimated from the following:

- **Action type:**
  - +2 for pull-up, step-back, fadeaway, driving layup, hook, dunk.
  - −1 for catch-and-shoot, spot-up, generic jump shot.
- **Location:**
  - +1 at the rim (≤5 ft) and in the mid-range (8–16 ft).
  - −1 for deep threes (≥27 ft) and corner catch-and-shoot.
- **Late clock:** +1 when `SHOT_CLOCK_APPROX ≤ 5`.

Labels: `≥ 2` = likely_contested, `1` = borderline, `≤ 0` = likely_open.

### Court coordinates

`xLegacy` (−250 to +250) and `yLegacy` (0 to 470) are in tenths of feet, with the basket at (0, 0). Corner three: `|x| ≥ 220` and `y ≤ 50`.

---

## Other notebooks and scripts

| File | Purpose |
|---|---|
| [`NBA_Shot_Quality_Rolling_Temporal_Validation.ipynb`](NBA_Shot_Quality_Rolling_Temporal_Validation.ipynb), [`rerun_rolling_validation.py`](rerun_rolling_validation.py) | Expanding-window validation (train on all prior seasons, test on the next) |
| [`NBA_Shot_Quality_Year_to_Year_Validation.ipynb`](NBA_Shot_Quality_Year_to_Year_Validation.ipynb), [`..._NO_PRIOR_KEEP_TEAM.ipynb`](NBA_Shot_Quality_Year_to_Year_Validation_NO_PRIOR_KEEP_TEAM.ipynb) | Year-to-year validation for Models A and C |
| [`NBA_Shot_Quality_Temporal_Validation_2023_2024.ipynb`](NBA_Shot_Quality_Temporal_Validation_2023_2024.ipynb) | Train on 2023–24, test on 2024–25 |
| [`compare_distance_vs_proxy_2014_story_thresh.ipynb`](compare_distance_vs_proxy_2014_story_thresh.ipynb) | True defender distance vs. public proxies (2014, when tracking data was public) |
| [`feature_ablation_test.py`](feature_ablation_test.py) | Earlier feature-ablation experiments |
| [`enrich_shots_nbastatsv3_full.ipynb`](enrich_shots_nbastatsv3_full.ipynb) | Original enrichment notebook (superseded by `enrich_shots.py`) |
| [`nba_personId_fullname_2013_2025.ipynb`](nba_personId_fullname_2013_2025.ipynb) | Maps `personId` to player names via `nba_api` |
| `NBA_Shot_Data_Exploration*.ipynb`, `data-visualization.ipynb`, `enrich_savant_shots_v2.ipynb`, `nba_shot_quality_pipeline_old.ipynb`, `shot_clock_debug.ipynb` | Early exploration and legacy pipelines |

---

## License

[Apache License 2.0](LICENSE).

## Acknowledgments

- [shufinskiy/nba_data](https://github.com/shufinskiy/nba_data) for NBA Stats API play-by-play data.
- NBA.com tracking dashboards for shot-clock and closest-defender validation data.
