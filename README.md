# NBA Shot Quality Model

A machine learning project exploring the connections between **shot quality**, **luck**, and **winning** in the NBA.

## Research Goals

- How does shot quality correlate with scoring and winning?
- Which teams are getting "lucky" or "unlucky" based on expected vs actual performance?
- Can we better measure a team's true "skill" using expected points and Pythagorean winning percentage?

**Target Publication**: Wharton Sports Analytics Journal

---

## Data Description

**Source**: [shufinskiy/nba_data](https://github.com/shufinskiy/nba_data/tree/main/datasets) (play-by-play from NBA Stats API v3)

**Raw Data** (`raw_data/nbastatsv3_YYYY.csv`):
- **Rows**: ~7.05 million total across 12 season files (~550k–607k per season)
- **Features**: 24 columns
- **Time range**: 2013-14 through 2024-25 (12 seasons)
- **Key fields**: `gameId`, `personId`, `teamTricode`, `clock`, `period`, `xLegacy`, `yLegacy`, `shotDistance`, `shotResult`, `shotValue`, `actionType`, `subType`, `description`, `location`

**Enriched Data** (`enriched_data/nbastatsv3_YYYY_enriched_shots.csv`):
- **Rows**: ~2.51 million total (field goal attempts only); ~219k per season
- **Features**: 30 columns (24 original + 6 engineered)
- **Engineered columns**: `ABS_TIME`, `SHOT_CLOCK_APPROX`, `SHOT_CLOCK_SOURCE`, `contest_score`, `contest_label`, `contest_reasons`

---

## Model Performance

### Models (12 + voting ensemble)
| Category | Models |
|----------|--------|
| **Boosting (Optuna-tuned)** | LightGBM, HistGradientBoosting, GradientBoosting, XGBoost, CatBoost |
| **Ensemble** | RandomForest, ExtraTrees, AdaBoost |
| **Other** | LogisticRegression, MLP, GaussianNB, KNN |
| **Meta** | VotingEnsemble (soft voting over top 5 boosters) |

### Best Results (2024-25 season)
| Model | Test AUC | Description |
|-------|----------|-------------|
| **LightGBM** | 0.667 | Best single model (Bayesian-optimized) |
| Voting Ensemble | 0.667 | Soft voting over top 5 |
| XGBoost | 0.667 | |
| Gradient Boosting | 0.666 | |
| CatBoost | 0.666 | |

### Configuration
- `N_OPTUNA_TRIALS = 10` (Bayesian hyperparameter optimization via Optuna TPE)
- `PYTH_EXP = 14` (Pythagorean win formula exponent)
- `RANDOM_STATE = 42`
- Cross-validation: 3-fold stratified, results reported as mean ± std

### Model Details

**Baseline models** (static hyperparameters):
| Model | Key Hyperparameters |
|-------|-------------------|
| Logistic Regression | `solver="saga", max_iter=2000, C=0.5` |
| kNN | `n_neighbors=15, weights='distance'` |
| Random Forest | `n_estimators=500, max_depth=12, min_samples_leaf=5` |
| Gradient Boosting | `n_estimators=300, max_depth=5, learning_rate=0.1` |
| MLP | `hidden_layer_sizes=(128,64), max_iter=500, early_stopping=True` |
| Extra Trees | `n_estimators=500, max_depth=12, min_samples_leaf=5` |
| AdaBoost | `n_estimators=200, learning_rate=0.1` |
| Naive Bayes | Default (no tuning) |
| HistGradientBoosting | `max_iter=500, max_depth=8, learning_rate=0.05` |

**Optuna-tuned models** (Bayesian optimization, 10 trials, TPE sampler):
| Model | Tuned Hyperparameters |
|-------|----------------------|
| XGBoost | `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`, `min_child_weight` |
| CatBoost | `iterations`, `depth`, `learning_rate`, `l2_leaf_reg`, `bagging_temperature`, `random_strength` |
| LightGBM | `n_estimators`, `max_depth`, `learning_rate`, `num_leaves`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`, `min_child_samples` |
| Gradient Boosting | `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `min_samples_split`, `min_samples_leaf` |
| HistGradientBoosting | `max_iter`, `max_depth`, `learning_rate`, `max_leaf_nodes`, `min_samples_leaf`, `l2_regularization` |

### Feature Importance Insights

**Top-tier features** (shot location + player history):
- Court position (`xLegacy`, `yLegacy`) dominates
- Player skill (`prior_fg_pct`) is a top-5 predictor
- Prior attempts indicate high-volume shooter patterns

**Key finding**: Prior season features add significant predictive power — 6 of the top 20 features are `prior_*` or `opp_def_*` historical stats.

---

## Features

### Base Features (from enriched data)
| Feature | Description |
|---------|-------------|
| `shotDistance` | Distance from basket in feet (0-47 ft) |
| `SHOT_CLOCK_APPROX` | Approximate shot clock when shot taken (0-24 sec) |
| `xLegacy`, `yLegacy` | Court coordinates (tenths of feet) |
| `contest_score` | Heuristic difficulty score (-2 to +4) |
| `shotValue` | Point value (2 or 3) |

### Prior Season Features (11 features)
Player historical stats and opponent defensive ratings from the **previous season** to avoid data leakage:

| Feature | Description |
|---------|-------------|
| `prior_fg_pct` | Player's FG% from prior season |
| `prior_fg3_pct` | Player's 3PT% from prior season |
| `prior_fg2_pct` | Player's 2PT% from prior season |
| `prior_attempts` | Total shot attempts in prior season |
| `opp_def_fg_pct` | Opponent's overall FG% allowed |
| `opp_def_rim_pct` | Opponent's rim FG% allowed |
| `opp_def_3pt_pct` | Opponent's 3PT% allowed |

### Engineered Features
- **Shot Location**: `is_paint`, `is_midrange`, `is_corner3`, `is_above_break3`
- **Shot Clock Buckets**: `clock_0_4`, `clock_4_8`, ..., `clock_20_24`
- **Interactions**: `clock_x_distance`, `clock_x_paint`, `rhythm_x_distance`
- **Text-Parsed**: `is_pullup`, `is_stepback`, `is_fadeaway`, `is_driving`, `is_floating`, `is_cutting`, `is_tip`, `is_putback`, `is_second_chance`

---

## Temporal Validation

### Rolling Temporal (Train: 2014-N, Test: N+1)

Expanding window: train on all prior seasons, test on the next unseen season.

| Fold | Train Years | Test Year | AUC |
|------|-------------|-----------|-----|
| 1 | 2014-2016 | 2017 | ~0.66 |
| 2 | 2014-2017 | 2018 | ~0.66 |
| ... | ... | ... | ... |
| 8 | 2014-2023 | 2024 | ~0.67 |

**Mean AUC**: 0.662 ± 0.007

### Year-to-Year (Train: N, Test: N+1)

Single-season training to test year-over-year stability.

| Fold | Train | Test | AUC |
|------|-------|------|-----|
| 1 | 2020 | 2021 | ~0.65 |
| 2 | 2021 | 2022 | ~0.65 |
| ... | ... | ... | ... |
| 5 | 2024 | 2025 | ~0.65 |

**Mean AUC**: ~0.65

---

## Open Shots vs Wins Analysis

Comparison of estimated open shots (from our model's `contest_label`) vs actual open shots (from NBA.com closest defender data, 4+ feet) against 2024-25 season wins.

**NBA.com data**: Combined 4-6 ft (Open) + 6+ ft (Wide Open) closest defender distance.

See `Open_Shots_vs_Wins_Analysis.ipynb` for charts and correlation analysis.

---

## Project Structure

```
nba-shot-quality/
├── NBA_Shot_Quality_Modeling_XGB_CatBoost.ipynb       # Main modeling notebook
├── NBA_Shot_Quality_Rolling_Temporal_Validation.ipynb  # Expanding window validation
├── NBA_Shot_Quality_Year_to_Year_Validation.ipynb      # Year-over-year validation
├── NBA_Shot_Data_Exploration_2024_25_FULL.ipynb        # Visualization suite
├── Open_Shots_vs_Wins_Analysis.ipynb                   # Open shots vs wins charts
├── enrich_shots_nbastatsv3_full.ipynb                  # Data enrichment pipeline
├── raw_data/                                           # Raw NBA Stats API data
│   └── nbastatsv3_YYYY.csv                            #   (12 seasons, 2013-2024)
├── enriched_data/                                      # Processed shot data
│   └── nbastatsv3_YYYY_enriched_shots.csv             #   (~219k shots per season)
├── models/                                             # Cached trained models
│   ├── fitted_models.pkl
│   └── model_results.pkl
├── player_historical_stats.csv                         # Player FG% by season
├── team_defensive_stats.csv                            # Team defensive ratings
├── model_results.csv                                   # Model comparison results
├── season_standings.csv                                # Pythagorean expected wins
└── team_game_points.csv                                # Per-game expected points
```

---

## Quick Start

### 1. Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn xgboost catboost lightgbm matplotlib jupyter optuna
```

### 2. Run the Pipeline

**Enrich raw data:**
```bash
jupyter notebook enrich_shots_nbastatsv3_full.ipynb
```

**Train models:**
```bash
jupyter notebook NBA_Shot_Quality_Modeling_XGB_CatBoost.ipynb
```

### 3. Temporal Validation
- `NBA_Shot_Quality_Rolling_Temporal_Validation.ipynb` — Rolling window validation
- `NBA_Shot_Quality_Year_to_Year_Validation.ipynb` — Year-over-year generalization

---

## Technical Details

### Approximate Shot Clock (`SHOT_CLOCK_APPROX`)

The official NBA public dataset doesn't include the actual shot clock value. We **approximate** it by reconstructing possessions from play-by-play events:

1. **Reset events (new shot clock):**
   * **Made field goal** → opposing team inbounds, new 24s
   * **Turnover** → opposing team gains possession, new 24s
   * **Defensive rebound** → new 24s
   * **Offensive rebound** → 14s reset (since 2018-19 rules)
   * **Defensive violations** (kicked ball, defensive 3-second) → 14s reset

2. **Elapsed time since reset:**
   * Convert `clock` field (`PT11M43.00S`) into absolute game seconds
   * For each shot, find the last reset event and compute time difference

3. **Shot clock approximation:**
   * `SHOT_CLOCK_APPROX = reset_value - time_since_reset`
   * Example: Reset at 8:00, shot at 7:50 → `24 - 10 = 14`

**Limitations:**
- Doesn't model "minimum 14" exactly (keeps 18 when >14 remains)
- Doesn't track inbound delays
- Still effective for distinguishing early-clock (20+) vs late-clock (<=5) shots

---

### Contested/Open Heuristic (`contest_score`, `contest_label`)

Since defender distance isn't in this dataset, we estimate shot difficulty using:

1. **Action Type / SubType / Description**
   * **Contested-like** (+2): Pull-Up, Step Back, Fadeaway, Driving Layup, Hook, Dunk
   * **Open-like** (-1): Catch-and-Shoot, Spot Up, Generic Jump Shot

2. **Shot Location / Distance**
   * **At-rim (<=5 ft)** (+1) → usually more defended
   * **Mid-range (8-16 ft)** (+1) → often tightly contested
   * **Deep 3s (>=27 ft)** (-1) → more likely open

3. **Corner vs Above-the-Break**
   * Corner-3: `|xLegacy| >= 220` and `yLegacy <= 50`
   * Corner catch-and-shoot nudged more open (-1)

4. **Shot Clock Context**
   * `SHOT_CLOCK_APPROX <= 5` (+1): Late-clock = rushed/contested

5. **Labels:**
   * `>= 2` → **likely_contested**
   * `= 1` → **borderline**
   * `<= 0` → **likely_open**

---

### Court Coordinates (xLegacy / yLegacy)

NBA shot coordinates in **tenths of feet**:
- **xLegacy**: -250 to +250 (sideline to sideline)
- **yLegacy**: 0 to 470 (baseline to half-court)
- **Basket**: (0, 0)

**Zones:**
- **Paint**: ~0-8 ft from basket
- **Mid-range**: 8-22 ft
- **Corner-3**: `|x| >= 220` and `y <= 50`
- **Above-the-break 3**: All other 3PT shots

---

## Output Files

| File | Description |
|------|-------------|
| `model_results.csv` | Cross-validation (mean ± std) and test metrics for all models |
| `team_game_points.csv` | Per-game actual vs expected points by team |
| `season_standings.csv` | Pythagorean expected wins vs actual wins |
| `player_historical_stats.csv` | Player FG%, 3P%, attempts by season |
| `team_defensive_stats.csv` | Opponent FG% allowed by zone and season |

---

## License

This project is for research and educational purposes.

## Acknowledgments

- [shufinskiy/nba_data](https://github.com/shufinskiy/nba_data) for NBA Stats API play-by-play data
- NBA Stats API for closest defender distance data
- Pythagorean winning formula (exponent 14) for expected wins calculation
