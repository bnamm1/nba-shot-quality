# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Research Goal**: Exploring the connections between shot quality, luck, and winning in the NBA

This repository contains a sports analytics research project that analyzes shot characteristics across multiple NBA seasons to train models predicting shot outcomes. The core research questions are:
- How does shot quality correlate with scoring and winning?
- Which teams are getting "lucky" or "unlucky" based on expected vs actual performance?
- Can we better measure a team's true "skill" using expected points and Pythagorean winning percentage?

### Research Methodology
1. **Data Enrichment** ([enrich_shots_nbastatsv3_full.ipynb](enrich_shots_nbastatsv3_full.ipynb)): Takes raw NBA play-by-play data and enriches shots with approximate shot clock and contest classification
2. **Modeling & Analysis** ([NBA_Shot_Quality_Modeling_XGB_CatBoost.ipynb](NBA_Shot_Quality_Modeling_XGB_CatBoost.ipynb)): Trains ML models (Logistic Regression, kNN, Random Forest, Gradient Boosting, MLP, CatBoost, XGBoost) to predict shot outcomes, calculates expected points, and generates Pythagorean win projections
3. **Visualization** ([NBA_Shot_Data_Exploration_2024_25_FULL.ipynb](NBA_Shot_Data_Exploration_2024_25_FULL.ipynb)): Creates comprehensive visualizations of shot data including court charts, player/team metrics, and efficiency analysis
4. **Statistical Testing**: Uses Pythagorean winning formula with exponent 14 to calculate expected wins and compare to actual results

### Target Publication
Primary target: **Wharton Sports Analytics Journal**

## Data Pipeline Architecture

### 1. Raw Data Collection
- **Primary Source**: NBA Stats API v3 play-by-play data (2015-2024 seasons)
- **Legacy Source**: [NBA Savant](https://www.nbasavant.com/shot_search.php) (2013-2014 seasons)
- **Location**: `raw_data/nbastatsv3_YYYY.csv` (one file per season, e.g., 2013-2024)
- **Format**: CSV files containing all play-by-play events (not just shots)
- **Size**: ~80-90 MB per season, ~600k rows per season
- **Key Fields**:
  - **Identifiers**: `gameId`, `actionNumber`, `personId`, `teamId`
  - **Temporal**: `clock` (ISO format: PT11M43.00S), `period`
  - **Shot Attributes**: `actionType`, `subType`, `description`, `shotDistance`, `shotResult` (Made/Missed), `shotValue` (2/3)
  - **Spatial**: `xLegacy`, `yLegacy` (NBA coordinates in tenths of feet)
  - **Context**: `scoreHome`, `scoreAway`, `location` (home/visitor)

### 2. Data Enrichment Pipeline
- **Notebook**: [enrich_shots_nbastatsv3_full.ipynb](enrich_shots_nbastatsv3_full.ipynb)
- **Input**: `raw_data/nbastatsv3_YYYY.csv`
- **Process**: Filters to field goal attempts only (~220k shots per season), then enriches each shot with:
  - **`SHOT_CLOCK_APPROX`**: Reconstructed shot clock value (0-24 seconds) by tracking possession resets from play-by-play events
  - **`SHOT_CLOCK_SOURCE`**: Debug info showing which reset event was used (e.g., "made_fg(24)", "off_reb(14)")
  - **`contest_score`**: Numeric score (-2 to +4) indicating shot difficulty
  - **`contest_label`**: Classification as "likely_open", "borderline", or "likely_contested"
  - **`contest_reasons`**: Text explanation of the classification logic
  - **`ABS_TIME`**: Absolute game time in seconds for event ordering
- **Output**: `enriched_data/nbastatsv3_YYYY_enriched_shots.csv` (~50-60 MB per season, ~220k rows)
- **Processing**: Batched by (gameId, personId) pairs, writing every 50 pairs to manage memory

### 3. Shot Clock Reconstruction Logic
The approximate shot clock is reconstructed by identifying possession reset events:
- **24-second resets**: Made field goals, defensive rebounds, turnovers, jump balls, final free throws
- **14-second resets**: Offensive rebounds (post-2018 rule), defensive violations (kicked ball, defensive 3-second, goaltending)
- **Method**: For each shot, find the last reset event in the same period, calculate elapsed time since reset, cap at reset value

### 4. Contest Classification Heuristic
A hybrid rule-based system that scores shots using:
- **Action/SubType signals**: Pull-ups, step-backs, fadeaways (+2 contested); catch-and-shoot, spot-ups (-1 open)
- **Distance**: At-rim ≤5ft (+1), mid-range 8-16ft (+1), deep 3PT ≥27ft (-1)
- **Court coordinates**: Corner-3 detection using NBA coordinates (|x|≥220 & y≤50 in tenths of feet)
- **Shot clock pressure**: Late clock ≤5 seconds (+1 contested)
- **Thresholds**: Score ≥2 → "likely_contested", =1 → "borderline", ≤0 → "likely_open"

### 5. Modeling Pipeline
- **Notebook**: [NBA_Shot_Quality_Modeling_XGB_CatBoost.ipynb](NBA_Shot_Quality_Modeling_XGB_CatBoost.ipynb)
- **Input**: `enriched_data/nbastatsv3_YYYY_enriched_shots.csv`
- **Process**:
  1. Load enriched shot data (~220k shots)
  2. Create binary target: `shotResult` → 1 (made) or 0 (missed), case-insensitive and robust
  3. Feature engineering:
     - **Numeric**: `shotDistance`, `SHOT_CLOCK_APPROX`, `xLegacy`, `yLegacy`, `period`, `contest_score`, `shotValue`, `ABS_TIME`
     - **Categorical**: `actionType`, `subType`, `contest_label`, `teamTricode`, `location`
  4. Train/test split (80/20, stratified), yields ~175k train, ~44k test
  5. Two preprocessing pipelines:
     - **Sparse**: StandardScaler (with_mean=False) + sparse one-hot for linear/kNN/MLP
     - **Dense**: SimpleImputer + dense one-hot for tree-based models (RF/GB/CatBoost/XGBoost)
  6. Model zoo: Logistic Regression, kNN, Random Forest, Gradient Boosting, MLP, CatBoost, XGBoost
  7. 3-fold stratified cross-validation on training set
  8. Test set evaluation: ROC AUC, log loss, Brier score, accuracy
  9. Select best model by test ROC AUC (typically all models achieve ~1.0 AUC, log_reg/mlp/xgboost/catboost/gb have lowest log loss)
  10. Score all shots with `p_make` probability
  11. Calculate expected points: `p_make * shotValue`
  12. Aggregate to team-game level for actual vs expected points/allowed
  13. Compute Pythagorean expected wins using exponent=14: `W% = PF^14 / (PF^14 + PA^14)`
  14. Export: `model_results.csv`, `team_game_points.csv`, `season_standings.csv`
- **Output Files**:
  - `model_results.csv`: Cross-validation and test metrics for all 7 models
  - `team_game_points.csv`: Per-game actual/expected points scored/allowed by team
  - `season_standings.csv`: Pythagorean expected wins vs actual wins, sorted by expected wins

### 6. Visualization & Analysis Pipeline
- **Notebook**: [NBA_Shot_Data_Exploration_2024_25_FULL.ipynb](NBA_Shot_Data_Exploration_2024_25_FULL.ipynb)
- **Input**: `enriched_data/nbastatsv3_2024_enriched_shots.csv`
- **Key Features**: Schema-adaptive (handles missing columns gracefully), single-figure matplotlib plots
- **Visualizations**:
  1. **Overview KPIs**: Total shots, overall FG%
  2. **Shot Selection**: 2PA vs 3PA distribution
  3. **Distance Distribution**: Histogram of shot distances
  4. **Court Shot Charts**: Scatter plots of shot positions (sampled), make vs miss overlay
  5. **Player FG% Leaders**: Bar chart (min 200 attempts)
  6. **Hexbin eFG% by Location**: Spatial efficiency map (min 20 attempts per hex, gridsize=40)
  7. **FG% Heatmap**: Distance × Shot Clock bins (min 30 attempts/bin, with count overlays)
  8. **Team Shot Profiles**: 3PA rate vs FG% scatter (bubble size = avg distance)
  9. **Team 3PA vs PPG**: Relationship between three-point rate and scoring
  10. **Top 10 PPG Players**: Field-goal points only (min 10 games)
  11. **Player eFG% vs Attempts**: Scatter with labels (min 1000 attempts, grouped by `personId`)
  12. **Player "Bad Shot" % vs Attempts**: Contested shot rate (min 800 attempts)
- **Schema Normalization**: Automatically maps columns like `xLegacy→x`, `yLegacy→y`, `shotDistance→shot_distance`, `SHOT_CLOCK_APPROX→shot_clock`, `shotResult→shot_made_flag`, `playerName→name`, `teamTricode→team_name`, `gameId→espn_game_id`
- **Shot Clock Data Quality Issue**: Many made shots recorded at ~0 seconds due to timing. Notebook filters these out for heatmap analysis (keeps late-clock misses to preserve genuine heaves)

## Key Notebooks (Priority Order)

### Primary Workflows
1. **[NBA_Shot_Quality_Modeling_XGB_CatBoost.ipynb](NBA_Shot_Quality_Modeling_XGB_CatBoost.ipynb)**: Complete ML pipeline from enriched data to Pythagorean standings
2. **[NBA_Shot_Data_Exploration_2024_25_FULL.ipynb](NBA_Shot_Data_Exploration_2024_25_FULL.ipynb)**: Comprehensive visualization suite for enriched shot data
3. **[enrich_shots_nbastatsv3_full.ipynb](enrich_shots_nbastatsv3_full.ipynb)**: Data enrichment pipeline (shot clock + contest classification)

### Legacy/Experimental
- **[enrich_savant_shots_v2.ipynb](enrich_savant_shots_v2.ipynb)**: Legacy enrichment for NBA Savant data (2013-2014 seasons)
- **[NBA_Shot_Data_Exploration.ipynb](NBA_Shot_Data_Exploration.ipynb)**: Initial data exploration
- **[compare_distance_vs_proxy_2014_story_thresh.ipynb](compare_distance_vs_proxy_2014_story_thresh.ipynb)**: Model comparison between actual distance and proxy features
- **[nba_personId_fullname_2013_2025.ipynb](nba_personId_fullname_2013_2025.ipynb)**: Player ID to name mapping
- **[data-visualization.ipynb](data-visualization.ipynb)**: Legacy visualization utilities

## Common Development Tasks

### Running the Full Pipeline for a New Season

1. **Download raw data**: Place `nbastatsv3_YYYY.csv` in `raw_data/`
2. **Enrich shots**:
   - Open [enrich_shots_nbastatsv3_full.ipynb](enrich_shots_nbastatsv3_full.ipynb)
   - Update `CSV_PATH` variable (default: `"raw_data/nbastatsv3_2024.csv"`)
   - Update `OUTPUT_CSV` path (default: `"enriched_data/nbastatsv3_2024_enriched_shots.csv"`)
   - Run all cells through Section 8 (full-season batch export)
   - Output: `enriched_data/nbastatsv3_YYYY_enriched_shots.csv`
   - Expected processing time: ~10-20 minutes for full season
3. **Create visualizations**:
   - Open [NBA_Shot_Data_Exploration_2024_25_FULL.ipynb](NBA_Shot_Data_Exploration_2024_25_FULL.ipynb)
   - Update `FILE_PATH` (default: `"enriched_data/nbastatsv3_2024_enriched_shots.csv"`)
   - Run all cells to generate 12 visualizations
   - All plots are self-contained matplotlib figures
4. **Train models & compute standings**:
   - Open [NBA_Shot_Quality_Modeling_XGB_CatBoost.ipynb](NBA_Shot_Quality_Modeling_XGB_CatBoost.ipynb)
   - Update `DATA_PATH` (default: `"enriched_data/nbastatsv3_2024_enriched_shots.csv"`)
   - Update `PYTH_EXP` if needed (default: 14 for Pythagorean exponent)
   - Run all cells
   - Expected training time: ~5-10 minutes for all 7 models
   - Outputs: `model_results.csv`, `team_game_points.csv`, `season_standings.csv`

### Enriching a Single Player's Shots

Use helper functions in [enrich_shots_nbastatsv3_full.ipynb](enrich_shots_nbastatsv3_full.ipynb):
```python
# Load raw data first
all_data = pd.read_csv("raw_data/nbastatsv3_2024.csv")

# One game for one player
shots = enrich_player_game_v2(all_data, game_id=22400001, player_id=201939)

# All games for a player (e.g., Steph Curry = 201939)
shots = enrich_player_all_games_v2(all_data, player_id=201939)

# Save individual player shots
shots.to_csv("nbastatsv3_2024_steph_curry_shots.csv", index=False)
```

### Analyzing a Specific Team's Performance

Use the modeling notebook outputs:
```python
# Load team-game data
team_game = pd.read_csv("team_game_points.csv")

# Filter to one team (e.g., GSW = 1610612744)
gsw_games = team_game[team_game['teamId'] == 1610612744]

# Compare actual vs expected points
gsw_games['points_diff'] = gsw_games['actual_points'] - gsw_games['exp_points']
print(gsw_games[['gameId', 'actual_points', 'exp_points', 'points_diff']].head())
```

### Running Jupyter Notebooks

The project uses Jupyter notebooks exclusively. Start Jupyter from the virtual environment:
```bash
source .venv/bin/activate  # Activate virtual environment (macOS/Linux)
jupyter notebook           # Start Jupyter server
```

## Environment Setup

### Python Version
Python 3.13 (see `.venv/`)

### Key Dependencies
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **scikit-learn**: Model training (Logistic Regression, kNN, Random Forest, Gradient Boosting, MLP)
- **xgboost**: XGBoost classifier
- **catboost**: CatBoost classifier
- **matplotlib**: Visualization
- **jupyter**: Notebook environment

### Installing Dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn xgboost catboost matplotlib jupyter
```

## Data Specifications

### NBA Coordinate System
- **xLegacy**: -250 to +250 (tenths of feet, sideline to sideline)
- **yLegacy**: 0 to 470 (tenths of feet, baseline to half-court)
- **Basket**: (0, 0)
- **Corner-3 definition**: `|xLegacy| ≥ 220 AND yLegacy ≤ 50`

### Shot Clock Rules
- Regular possession: 24 seconds
- Offensive rebound: 14 seconds (2018-19 rule change)
- Defensive violations: 14 seconds to offense
- Note: Pre-2018 seasons used 24-second reset on offensive rebounds

### Game ID Format
- Format: 10 digits, e.g., `0022400001`
- Prefix `002`: Regular season
- Prefix `004`: Playoffs
- The enrichment pipeline normalizes gameIds to this format

## Output Files

### Model Results
- **model_results.csv**: Cross-validation and test metrics for all models (ROC AUC, log loss, Brier score, accuracy)

### Team Analytics
- **team_game_points.csv**: Per-game actual and expected points for each team
- **season_standings.csv**: Pythagorean expected wins vs actual wins, sorted by expected wins

### Player Mappings
- **nba_players_2013_2025.csv**: Player ID to full name mapping
- **player_luck_summary.csv**: Analysis of player performance vs expected (legacy)

## Known Limitations & Data Quality Issues

### Shot Clock Approximation
- **Timing artifacts**: Many made shots recorded at `SHOT_CLOCK_APPROX ≤ 0.5` seconds due to event logging timing. The visualization notebook filters these out for heatmap analysis (EPS=0.5 threshold) while keeping late-clock misses to preserve genuine heaves.
- **"Minimum 14" rule**: Does not model NBA rule where shot clock resets to 14 OR remaining time, whichever is greater
- **Inbound delays**: Does not track time between whistle and inbound pass
- **Retained possession**: Technical/flagrant free throws may not accurately track possession retention
- **Missing data**: ~0.7% of shots have no identifiable reset event (`SHOT_CLOCK_SOURCE: "no_reset_found"`)

### Contest Classification
- **Rule-based heuristic**: Not ground truth defender distance (not available in NBA Stats v3 API)
- **Coordinate dependency**: Corner-3 detection relies on accurate `xLegacy`/`yLegacy` values
- **Text parsing**: May misclassify shots with unusual or inconsistent descriptions
- **Free throws**: Marked as "n/a_free_throw" (not applicable to contest classification)

### Data Quality
- **Score fields**: `scoreHome` and `scoreAway` have ~53% missing values (not critical for this analysis)
- **Game ID format**: Raw data may have inconsistent gameId formats; enrichment pipeline normalizes to 10-digit format (e.g., 0022400001)
- **Regular season filter**: Analysis typically filters to games with gameId prefix "002" (regular season only)
- **Deduplication**: Events deduplicated by `(gameId, actionId, teamId)` to handle overlapping data sources

### Model Performance
- **Near-perfect metrics**: All models achieve ~1.0 ROC AUC and ~1.0 accuracy on test set, suggesting features are highly predictive or potential data leakage
- **Best models by log loss**: Logistic Regression (~0.0001), MLP (~0.000001), XGBoost (~0.00001), CatBoost (~0.00009), Gradient Boosting (~0.00002)
- **Overfitting indicators**: Extremely low Brier scores and log loss suggest possible issues with feature engineering or data quality

## Sanity Checks & Validation

### Enrichment Pipeline Validation
The enrichment notebook (Section 9) includes validation against Basketball-Reference:
```python
# GSW 2024-25 regular season field-goal totals
# Expected from Basketball-Reference: PTS ~7,948, FGM ~3,342, FG3M ~1,264
GSW_ID = 1610612744
gsw_fg_made = enriched[(enriched["teamId"] == GSW_ID) &
                       (enriched["isFieldGoal"] == 1) &
                       (enriched["shotResult"] == "MADE")]
```

### Key Validation Checks
1. **`shotValue` coverage**: Should be 100% for all field goals (no missing values after enrichment)
2. **Regular season filter**: `gameId` prefix "002" for regular season games
3. **Point totals**: Compare team season totals against Basketball-Reference
4. **Duplicate detection**: Count unique `(gameId, actionId, teamId)` combinations

### Visualization Notebook Checks
The exploration notebook includes diagnostic cells for shot clock quality:
```python
# Distribution of SHOT_CLOCK_APPROX by make/miss
bins = [0, 0.25, 0.5, 1, 2, 5, 10, 15, 20, 24.1]
shots.groupby(pd.cut(shots['shot_clock'], bins=bins))
     .agg(attempts=('shot_made_flag','size'),
          fg_pct=('shot_made_flag','mean'))
```
- **Expected**: High FG% in 0-0.25s bin (timing artifacts for made shots)
- **Expected**: Very low FG% in 0.5-5s bins (genuine late-clock attempts)
- **Expected**: Higher FG% in 20-24s bin (early-clock rhythm shots)

## Code Architecture Notes

### Batch Processing Pattern (Enrichment)
The full-season enrichment uses batched writes to avoid memory issues:
```python
# Process 50 (gameId, personId) pairs at a time
for i, (gid, pid) in enumerate(pairs, 1):
    enriched = enrich_player_game_v2(all_data, game_id=gid, player_id=pid)
    batch.append(enriched)
    if i % 50 == 0:
        pd.concat(batch).to_csv(OUTPUT_CSV, mode="a", header=(not written_any))
        batch = []
```
- Allows processing 600k+ play-by-play events with ~220k shot output
- Incremental CSV writes prevent memory overflow
- Progress printed every 50 pairs

### Dual Preprocessing Pipelines (Modeling)
The modeling notebook maintains two preprocessing paths for optimal performance:

**Sparse Pipeline** (Linear/kNN/MLP):
```python
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler(with_mean=False))  # Sparse-compatible
])
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(sparse_output=True))  # Memory efficient
])
```

**Dense Pipeline** (Tree-based):
```python
numeric_transformer_dense = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))  # No scaling needed
])
categorical_transformer_dense = Pipeline([
    ("ohe", OneHotEncoder(sparse_output=False))  # Feature importance compatible
])
```

### Schema Normalization Pattern (Visualization)
The exploration notebook handles multiple schema versions:
```python
# Normalize to consistent schema
if 'x' not in shots.columns and 'xLegacy' in shots.columns:
    shots['x'] = shots['xLegacy']
if 'shot_made_flag' not in shots.columns and 'shotResult' in shots.columns:
    shots['shot_made_flag'] = (shots['shotResult'].str.lower() == 'made').astype(int)
```
- Allows same visualization code to work across different data sources
- Graceful degradation: plots skip if required columns missing

### Deduplication Strategy
Events are deduplicated by `(gameId, actionId, teamId)`:
```python
df = df.drop_duplicates(subset=["gameId", "actionId", "teamId"])
```
- Handles overlapping data sources or batch reruns
- Critical for accurate point totals and team statistics
