# NBA Shot Quality Model

A machine learning project exploring the connections between **shot quality**, **luck**, and **winning** in the NBA.

## Research Goals

- How does shot quality correlate with scoring and winning?
- Which teams are getting "lucky" or "unlucky" based on expected vs actual performance?
- Can we better measure a team's true "skill" using expected points and Pythagorean winning percentage?

**Target Publication**: Wharton Sports Analytics Journal

---

## Model Performance

| Model | Test AUC | Description |
|-------|----------|-------------|
| **LightGBM** | 0.667 | Best single model (Bayesian-optimized) |
| Voting Ensemble | 0.665 | Combines top 5 models |
| XGBoost | 0.664 | Second-best gradient boosting |
| CatBoost | 0.663 | Handles categoricals natively |

### Feature Importance Insights

**Top-tier features** (shot location + player history):
- Court position (`xLegacy`, `yLegacy`) dominates
- Player skill (`prior_fg_pct`) is top-5 predictor
- Prior attempts indicate high-volume shooter patterns

**Key finding**: Prior season features add significant predictive power - 6 of the top 20 features are `prior_*` or `opp_def_*` historical stats.

---

## Features (62 total)

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
- **Text-Parsed**: `is_pullup`, `is_stepback`, `is_fadeaway`, `is_driving`, `is_floating`

---

## Project Structure

```
nba-shot-quality/
├── NBA_Shot_Quality_Modeling_XGB_CatBoost.ipynb  # Main modeling notebook
├── NBA_Shot_Quality_Rolling_Temporal_Validation.ipynb
├── NBA_Shot_Quality_Year_to_Year_Validation.ipynb
├── NBA_Shot_Data_Exploration_2024_25_FULL.ipynb
├── enrich_shots_nbastatsv3_full.ipynb            # Data enrichment pipeline
├── raw_data/                                      # Raw NBA Stats API data
│   └── nbastatsv3_YYYY.csv
├── enriched_data/                                 # Processed shot data
│   └── nbastatsv3_YYYY_enriched_shots.csv
├── models/                                        # Cached trained models
│   ├── fitted_models.pkl
│   └── model_results.pkl
├── player_historical_stats.csv                    # Player FG% by season
├── team_defensive_stats.csv                       # Team defensive ratings
├── model_results.csv                              # Model comparison results
├── season_standings.csv                           # Pythagorean expected wins
└── team_game_points.csv                           # Per-game expected points
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
- `NBA_Shot_Quality_Rolling_Temporal_Validation.ipynb` - Rolling window validation
- `NBA_Shot_Quality_Year_to_Year_Validation.ipynb` - Year-over-year generalization

---

## Temporal Validation Results

### Rolling Temporal (Train: 2015-N, Test: N+1)
| Fold | Train Years | Test Year | AUC |
|------|-------------|-----------|-----|
| 1 | 2015-2016 | 2017 | ~0.66 |
| 2 | 2015-2017 | 2018 | ~0.66 |
| ... | ... | ... | ... |
| 8 | 2015-2023 | 2024 | ~0.67 |

**Mean AUC**: 0.662 +/- 0.007

### Year-to-Year (Train: N, Test: N+1)
Single-year training to test year-over-year stability.

**Mean AUC**: ~0.65

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
| `model_results.csv` | Cross-validation and test metrics for all models |
| `team_game_points.csv` | Per-game actual vs expected points by team |
| `season_standings.csv` | Pythagorean expected wins vs actual wins |
| `player_historical_stats.csv` | Player FG%, 3P%, attempts by season |
| `team_defensive_stats.csv` | Opponent FG% allowed by zone and season |

---

## License

This project is for research and educational purposes.

## Acknowledgments

- NBA Stats API for play-by-play data
- Pythagorean winning formula (exponent 14) for expected wins calculation
