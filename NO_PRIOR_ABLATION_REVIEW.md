# NO_PRIOR Ablation — Review

**Companion notebooks:**
- `NBA_Shot_Quality_Modeling_XGB_CatBoost_NO_PRIOR.ipynb`
- `NBA_Shot_Quality_Year_to_Year_Validation_NO_PRIOR.ipynb`

**Branch:** `brandon/no-prior-ablation`

---

## 1. Why we ran this

The original paper defined SPOE (Shooting Performance Over Expected) as *actual points − expected points*, and framed player-level SPOE as separating **shot process** (what kind of look you generated) from **shot-making skill / luck** (who took it and whether it went in).

But Model A (the paper's model) has six shooter-identity features baked into its "expected points" baseline:

- `prior_fg_pct`, `prior_fg3_pct`, `prior_fg2_pct`, `prior_shot_type_pct`
- `prior_attempts`, `prior_high_volume`

Plus `teamTricode` as a categorical, which is a soft proxy for shooter quality (Nuggets shots skew Jokic; Warriors shots skew Curry).

That means "expected" already knows *who is shooting*, not just *what shot they took*. Consequences for SPOE:

- **Jokic's +148 SPOE** in the paper is "even better than elite-Jokic baseline predicted" — not "made shots a league-average shooter wouldn't have."
- **Buy-low interpretation breaks:** a genuinely declining shooter would also show negative SPOE because their baseline expected too much of them.
- **Year-to-year SPOE correlation gets attenuated** — this year's SPOE partly folds in last year's performance via prior_fg_pct.

The ablation removes shooter-identity features so SPOE means what the paper claims it means: over/under performance vs. a league-average shooter from the same shot context. This is the Cervone (2014) definition of shot quality.

---

## 2. What changed in Model B (NO_PRIOR)

**Removed features (7):**
- Numeric: `prior_fg_pct`, `prior_fg3_pct`, `prior_fg2_pct`, `prior_shot_type_pct`, `prior_attempts`, `prior_high_volume`
- Categorical: `teamTricode`

**Kept:**
- All shot-context features (location, distance, angle, shot clock, action type, contest score, shot mechanics)
- All opponent-defense features (`opp_def_fg_pct`, `opp_def_rim_pct`, `opp_def_3pt_pct`, `opp_def_mid_pct`, `opp_def_shot_type_pct`)
- All interaction features (`clock_x_distance`, `rhythm_x_distance`, etc.)

**Output files** are suffixed `_no_prior` so Model A results are preserved untouched:
- `model_results_no_prior.csv`
- `season_standings_no_prior.csv`
- `team_game_points_no_prior.csv`
- `models/fitted_models_no_prior.pkl`, `models/model_results_no_prior.pkl`

---

## 3. Headline: model performance barely moves

Test AUC, Model A (from paper) vs Model B (NO_PRIOR run):

| Model | A test AUC | B test AUC | Δ |
|---|---|---|---|
| voting_ensemble | 0.6670 | 0.6652 | -0.0018 |
| xgboost | 0.6666 | 0.6649 | -0.0017 |
| **lightgbm** | **0.6673** | **0.6646** | **-0.0027** |
| hist_gb | 0.6651 | 0.6645 | -0.0006 |
| gb | 0.6663 | 0.6643 | -0.0020 |
| catboost | 0.6659 | 0.6642 | -0.0017 |
| rf | 0.6637 | 0.6630 | -0.0007 |
| extra_trees | 0.6599 | 0.6620 | +0.0021 |
| log_reg | 0.6602 | 0.6600 | -0.0002 |
| mlp | 0.6568 | 0.6582 | +0.0014 |
| adaboost | 0.6401 | 0.6451 | +0.0050 |
| naive_bayes | 0.6392 | 0.6433 | +0.0041 |
| knn | 0.6284 | 0.6272 | -0.0012 |

**Takeaway:** the tuned boosters drop by 0.6–2.7 basis points. The LightGBM headline is essentially unchanged (0.667 → 0.665) — well within the CV standard deviation (~0.002).

**Substantive implication:** shooter-identity features were doing very little predictive work on top of shot context. Context features + opponent defense capture almost all of what the model was learning; the `prior_*` features were mostly memorizing who's a good shooter rather than adding signal.

**A few simple models actually improved without prior/team features** (mlp, extra_trees, adaboost, naive_bayes). The naive_bayes bump makes sense — those features violate NB's independence assumption.

---

## 4. Where Model B differs sharply: expected wins per team

Even though the model's per-shot AUC barely moves, the *team-level* implications of Model B are meaningfully different. Comparing `pyth_wins_exp` team-by-team:

### Star-heavy teams — Model B predicts fewer wins

| Team | Model A | Model B | Δ | Notable shot-makers |
|---|---:|---:|---:|---|
| PHX | 36.62 | 21.83 | **-14.79** | Booker, Durant, Beal |
| SAC | 45.10 | 34.85 | -10.25 | DeRozan, Fox, Sabonis |
| IND | 45.27 | 35.78 | -9.49 | Haliburton |
| BOS | 60.76 | 52.91 | -7.85 | Tatum, Brown, Porzingis |
| LAC | 45.41 | 39.92 | -5.49 | Kawhi, Harden, PG |
| LAL | 36.33 | 31.10 | -5.23 | LeBron, AD, Reaves |
| NYK | 51.84 | 46.79 | -5.05 | Brunson, KAT |
| MIL | 34.28 | 30.60 | -3.68 | Giannis, Dame |

### Star-lite / rebuilding teams — Model B predicts more wins

| Team | Model A | Model B | Δ |
|---|---:|---:|---:|
| TOR | 35.50 | 51.38 | **+15.88** |
| BKN | 31.31 | 42.96 | +11.65 |
| CHA | 25.32 | 34.77 | +9.45 |
| POR | 41.83 | 48.46 | +6.63 |
| HOU | 54.15 | 59.84 | +5.69 |
| ORL | 52.05 | 57.50 | +5.45 |
| PHI | 23.51 | 28.67 | +5.16 |
| WAS | 19.90 | 24.20 | +4.30 |
| ATL | 40.75 | 44.96 | +4.21 |

**The pattern is exactly what the ablation predicts.** Without `prior_fg_pct` and `prior_shot_type_pct`, the model can no longer tell "Booker at 20 ft" apart from "generic guard at 20 ft" — both get the same expected points. So:

- Teams with elite shot-makers see their expected wins **deflated** (model can no longer credit them for shot-making skill).
- Teams with poor shooters see their expected wins **inflated** (model doesn't know they miss at below-average rates).

---

## 4a. Temporal validation: 5-fold year-to-year

The `NBA_Shot_Quality_Year_to_Year_Validation_NO_PRIOR.ipynb` notebook runs LightGBM trained on season X, tested on season X+1, across 5 folds. This is a stricter test than the single-season 80/20 split because train and test come from *different* seasons — different rosters, different roles, small rule changes.

| Fold | Model A AUC | Model B AUC | Δ |
|---|---:|---:|---:|
| 2020 → 2021 | 0.6670 | 0.6560 | **-0.0110** |
| 2021 → 2022 | 0.6724 | 0.6631 | -0.0093 |
| 2022 → 2023 | 0.6752 | 0.6672 | -0.0080 |
| 2023 → 2024 | 0.6711 | 0.6639 | -0.0072 |
| 2024 → 2025 | 0.6699 | 0.6614 | -0.0085 |
| **mean** | **0.6711** | **0.6623** | **-0.0088** |

Model B: mean AUC **0.6623 ± 0.0041**, mean log loss 0.6358, mean Brier 0.2236, mean accuracy 63.22%.

**The year-to-year drop is ~3× the single-season drop** (0.009 vs. 0.003 AUC). This is itself informative:

- **Single-season:** train and test share rosters, so the model can learn shot-context patterns that generalize easily. Prior features add little.
- **Year-to-year:** rosters shift between seasons. Prior features gave Model A something *player-specific* to anchor on across the roster change; without them, Model B relies purely on shot context, which transfers less well.

This means prior-shooter features were doing **double duty**:

1. **Within-season:** memorize who's a good shooter → small AUC gain, big SPOE distortion (the original concern that motivated the ablation)
2. **Cross-season:** bridge the roster-shift gap → small but *legitimate* AUC gain from player-skill persistence

Model B loses (2) but gains a clean SPOE interpretation. That's a fair tradeoff for the paper's SPOE analysis. It also means the year-to-year AUCs shouldn't be advertised as improvements — they're roughly comparable to Model A within a wider CV noise band (0.004 std here vs 0.002 in single-season).

**Stability across folds is preserved.** The fold-to-fold spread (0.6560 – 0.6672) is similar to Model A's spread (0.6670 – 0.6752). Model B is stable, just at a slightly lower level.

---

## 5. Why this makes the paper stronger, not weaker

The residual `Model A pyth_wins_exp − Model B pyth_wins_exp` per team is essentially a measurement of that team's **shooter-skill advantage above league average**. That's a new, interpretable quantity the paper can build on:

- **Model A** measures: shot quality that includes shooter identity → predicts actual wins reasonably.
- **Model B** measures: shot quality from context only → predicts actual wins less well.
- **The gap between them** is a rough decomposition of team success into "process/context" and "roster shooting talent."

**Reframed team narratives:**
- **TOR/BKN/ORL over-perform Model B** → their raw shot process was fine, but roster shooting talent was below average. Model A "knew" this because their players had low prior FG%. → *Fair value near Model A prediction; front office should recognize the process isn't broken.*
- **PHX/IND/SAC/BOS under-perform Model B** → they won more games than pure shot context would predict. Their shooter talent was doing real work above the league baseline. → *Star-driven teams that would collapse without their scorers.*

This is a sharper paper contribution than the original "shot quality correlates with wins." It quantifies the *decomposition*, not just the correlation.

---

## 6. Important caveat: `wins_actual` column changed definition

The paper's Figure 17 shows BOS `wins_actual = 51`, but the current run shows `wins_actual = 61` (their real record was 61-21). Between the paper and now, commit `8e5f95a` ("Separate real win-loss record from field-goal-only wins") split what was one column into two:

- `wins_actual` → real regular-season wins (includes FT, OT)
- `fg_only_wins` → wins if FG-only points determine outcomes (this matches what the paper called `wins_actual`)

**For direct A vs. B comparisons, use `pyth_wins_exp` (both models compute it the same way) or `fg_only_wins` (model-agnostic). Do NOT compare `diff_vs_actual` directly across paper and current output — the denominator changed.**

The paper's discussion of "GSW overpredicted by +11.88" was measured against `fg_only_wins`, not real wins. Any rewrite should acknowledge this and probably switch to the real-wins denominator throughout, which is what a reader would expect.

---

## 7. Recommended next steps

**Analytical (do these to strengthen the resubmission):**

1. **Recompute player-level SPOE from Model B.** Expect Jokic (+148), LaVine (+133), Pritchard (+123) to grow *larger*, since their "expected" no longer assumes elite baseline. The leaderboard reshuffle is the second half of the paper's ablation story.

2. **Compute Spearman rank correlation between Model A and Model B `pyth_wins_exp` team rankings.** A low correlation (which is what the delta table above suggests) is a headline result — quantifies how much the model was "cheating" via shooter identity.

3. **Run the year-to-year `_NO_PRIOR` notebook.** That gives the temporal stability of Model B and — more importantly — clean year-to-year SPOE correlations. This is the check that originally motivated the ablation.

4. **Recompute the Pythagorean win exponent choice.** The paper uses 14; standard NBA choices are 14–16.5. Reporting sensitivity to this choice would preempt an obvious reviewer question.

**Paper-writing (once analysis is done):**

5. **Present A and B side-by-side, not as a swap.** Model A is still the "best predictive" model; Model B is the "cleanest interpretive" model. Both have a role.

6. **Reframe SPOE as Model B's residual, not Model A's.** This is the biggest change — the paper's buy-low / hot-shooting narrative only actually works with Model B.

7. **Add a robustness section** documenting: what was excluded, what was kept, why the AUC gap is small, and why the standings deltas are large despite that.

8. **Update the Discussion's team-outlier explanations** (GSW/BOS over-projection). Under Model B, GSW/BOS's gap shrinks or reverses; the free-throw and open-3 arguments in the current paper are partial explanations that Model B partially controls for.

**Repo hygiene:**

9. Preserve both Model A and Model B output CSVs in the repo — the comparison notebook will need both.
10. Consider a new comparison notebook `Model_A_vs_B_Comparison.ipynb` that loads both `season_standings*.csv` and both `model_results*.csv` and generates the delta tables and rank-correlation plots automatically.

---

## 8. What was NOT touched

- Data enrichment (`enrich_shots.py`, `enrich_shots_nbastatsv3_full.ipynb`) — same inputs.
- Feature engineering steps other than the ablated feature list — same `contest_score`, same shot-clock proxy, same interaction features.
- Model hyperparameter search spaces — same Optuna trial budgets and parameter grids.
- Pythagorean exponent — still 14.
- Free-throw handling — still excluded (this remains a paper limitation).

The ablation is a clean feature-set change; everything else in the pipeline is identical.
