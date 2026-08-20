# NO_PRIOR Ablation — Review

**Companion notebooks:**
- `NBA_Shot_Quality_Modeling_XGB_CatBoost_NO_PRIOR.ipynb` — Model B (drops shooter priors + `teamTricode`)
- `NBA_Shot_Quality_Year_to_Year_Validation_NO_PRIOR.ipynb`
- `NBA_Shot_Quality_Modeling_XGB_CatBoost_NO_PRIOR_KEEP_TEAM.ipynb` — **Model C** (drops shooter priors only; keeps `teamTricode`) — added after decomposition analysis flagged a confound in Model B
- `NBA_Shot_Quality_Year_to_Year_Validation_NO_PRIOR_KEEP_TEAM.ipynb`

**Branch:** `brandon/no-prior-ablation`

> ⚠️ **Reading order:** §1–3 are still correct. §4's team-level attribution is **partially wrong** — corrected in §4a (mechanism) and §4b (Model C team-level results). §4c updates the temporal-validation "double duty" hypothesis with Model C data. §5 is now written against Model C SPOE (no more A−B residual construction).

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

> ⚠️ **CORRECTION (see §4a):** The framing below attributes the team-level shift to shooter identity being removed. That's only partially right. The Model B ablation also removed `teamTricode`, a **team fixed effect**. Decomposition analysis (§4a) shows the team-dummy removal accounts for about two-thirds of the shift; Model C team-level results (§4b) confirm this. Read §4a and §4b before using the tables below in the paper.

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

## 4a. Correction — the team-level shift is mostly the team dummy, not shooter identity

The Model B ablation removed seven features: six `prior_*` shooter stats **and** `teamTricode`. That last one isn't a shooter feature — it's a team fixed effect. Removing it collapses every team toward the league average, which mechanically produces exactly the pattern §4 attributed to shooter identity.

**Decomposition of team-level shift (mean |Δ pyth_wins_exp|):**

| Removal | Mean shift in expected wins |
|---|---:|
| Team dummy alone | **4.82** |
| Priors alone | 2.35 |
| Both (Model B) | 8.09 (not additive — see below) |

About **two-thirds** of the team-level shift comes from dropping `teamTricode`, not from dropping shooter identity.

**Sacramento is the cleanest counter-example.** §4 attributes SAC's −10.25 to losing credit for DeRozan/Fox/Sabonis. But removing priors alone moves Sacramento **up** by +0.93. The −10.25 total is entirely the team dummy. Indiana's −9.49 is similar: shooter features account for only about 1.4 wins of it; the rest is the team dummy.

**Shifts don't add linearly.** PHX is −3.88 (priors alone) and −3.14 (team dummy alone) separately, but −13.32 together. You can't back the correct attribution out arithmetically from Model B alone — hence Model C.

### The sharpest finding: per-shot AUC doesn't validate team-level claims

`teamTricode` is worth **0.0001 AUC** — literally nothing for per-shot discrimination — yet drives two-thirds of the team-level movement. That's a general point worth putting in the paper explicitly:

> A feature can be useless for per-shot prediction and still dominate aggregate conclusions, because aggregation amplifies small systematic offsets.

This means the paper's per-shot validation (calibration curve, confusion matrix, AUC) **does not validate its team-level claims** (Pythagorean standings, expected wins). Different metrics need different validation.

### Team-level cost — all three models

Team `pyth_wins_exp` accuracy against actual outcomes:

**vs. `wins_actual` (real regular-season wins):**

| Metric | Model A | Model B | Model C |
|---|---:|---:|---:|
| Mean absolute error | **4.98** | **8.84** | **5.52** |
| Pearson r | **0.878** | **0.648** | **0.849** |

**vs. `fg_only_wins` (FG-only outcomes — model-agnostic denominator):**

| Metric | Model A | Model B | Model C |
|---|---:|---:|---:|
| Mean absolute error | **4.75** | **6.31** | **4.80** |
| Pearson r | **0.894** | **0.791** | **0.887** |

**Team `pyth_wins_exp` rank correlation (Spearman):**

| Comparison | Rank correlation |
|---|---:|
| A vs C | **0.975** |
| B vs C | 0.910 |
| A vs B | 0.852 |

**Interpretation confirmed.** Model C is essentially as accurate as Model A at team-level (r=0.849 vs 0.878 against real wins; r=0.887 vs 0.894 against fg-only). The team rank correlation is 0.975 — teams rank in almost the same order under A and C. All of Model B's team-level damage was the team-dummy removal, not the shooter-prior removal.

For the paper's abstract, the correct number to cite is **Model C: MAE 5.52, r = 0.849 against real wins**. That's an honest "substantial portion of winning" claim and preempts the reviewer's challenge that shooter identity is doing the work.

---

## 4b. Model C team-level results — the cleanly-ablated shooter-identity signal

With the confound removed, the Model A vs Model C `pyth_wins_exp` deltas are much smaller than Model B's. Mean |Δ C-A| = **2.04 wins** (compared to Model B's 5.03) — about 2.5× less movement.

**Top 8 team-level shifts under Model C (biggest |Δ C-A|):**

| Team | A | C | Δ C-A | Δ B-A (for reference) |
|---|---:|---:|---:|---:|
| BKN | 31.31 | 37.84 | **+6.53** | +11.65 |
| ATL | 40.75 | 45.38 | +4.63 | +4.21 |
| NYK | 51.84 | 48.04 | -3.80 | -5.05 |
| HOU | 54.15 | 57.77 | +3.62 | +5.69 |
| LAL | 36.33 | 32.71 | -3.62 | -5.23 |
| PHX | 36.62 | 33.66 | -2.96 | **-14.79** |
| TOR | 35.50 | 38.35 | +2.85 | **+15.88** |
| PHI | 23.51 | 26.32 | +2.81 | +5.16 |

**Compare to §4's headline outliers under Model B:**

- **PHX (Booker/KD/Beal):** Model B −14.79 → **Model C −2.96**. The star-heavy narrative was ~80% team dummy, ~20% shooter identity.
- **SAC (DeRozan/Fox/Sabonis):** Model B −10.25 → **Model C +1.20** (a sign flip!). §4a's counter-example: removing shooter priors alone actually *raises* SAC's expected wins slightly. The −10.25 was entirely the team dummy.
- **IND (Haliburton):** Model B −9.49 → **Model C −2.63**. Most of it was team dummy.
- **TOR:** Model B +15.88 → **Model C +2.85**. The rebuild-friendly narrative was mostly team-dummy artifact.
- **BOS (Tatum):** Model B −7.85 → **Model C −1.14**. Nearly all team dummy.

**BKN is the largest remaining outlier at +6.53** — even under the clean ablation, Model C over-predicts Brooklyn by ~6.5 wins. That's interesting on its own: the Nets' 26-56 record in 2024-25 was substantially worse than their shot quality (context-only) would predict. That could be a real story (roster shooting talent below what shot context implies), a coaching/defense issue, or a small-sample fluke — worth a sentence or two of discussion in the paper rather than dropping.

**What survived from §4's original narrative:**
- The general direction (star-heavy teams down, star-lite teams up) is preserved but *much smaller in magnitude*
- The interpretation as "shooter-skill contribution above league average" now holds — but the effect size is 2-3 wins for most teams, not 10-15

**What did NOT survive:**
- The PHX/SAC/IND/TOR framing as headline outliers. They shrink dramatically or reverse under Model C
- The "shooter identity dominates team wins" implication of Model B's spread

---

## 4c. Temporal validation: 5-fold year-to-year (mechanism CORRECTED)

The year-to-year notebooks run LightGBM trained on season X, tested on season X+1, across 5 folds. This is a stricter test than the single-season 80/20 split because train and test come from *different* seasons — different rosters, different roles, small rule changes.

**All three models measured:**

| Fold | Model A | Model B | Model C | C-A |
|---|---:|---:|---:|---:|
| 2020 → 2021 | 0.6670 | 0.6560 | 0.6563 | -0.0107 |
| 2021 → 2022 | 0.6724 | 0.6631 | 0.6639 | -0.0085 |
| 2022 → 2023 | 0.6752 | 0.6672 | 0.6675 | -0.0077 |
| 2023 → 2024 | 0.6711 | 0.6639 | 0.6639 | -0.0072 |
| 2024 → 2025 | 0.6699 | 0.6614 | 0.6618 | -0.0081 |
| **mean** | **0.6711** | **0.6623** | **0.6627** | **-0.0084** |

**Key finding — Model C ≈ Model B in temporal validation.** Model C's cross-season AUC drop matches Model B's almost exactly (Δ +0.0004). This is the opposite of the single-season result, where Model C ≈ Model A.

### The single-season vs. cross-season decomposition

The story splits cleanly by validation regime:

| Contribution | Single-season | Year-to-year |
|---|---:|---:|
| Removing shooter priors (A → C) | ~0 bp | **~85 bp** |
| Removing team dummy (C → B) | ~20 bp | ~0.4 bp |
| Both (A → B) | ~20 bp | ~85 bp |

**Why the divergence?** Rosters shift between seasons. `teamTricode="MIA"` in 2020 does not refer to the same players as `teamTricode="MIA"` in 2025 — so the team fixed effect has almost no cross-season transfer value. But player-level FG% history *does* transfer: a 40% three-point shooter last year is likely to be a good shooter next year, whichever jersey they wear.

### Correction to the "double duty" hypothesis

The previous version of this section (§4a in the earlier draft) claimed prior features do double duty: within-season shooter memorization *and* cross-season roster bridging. That was **half right and half wrong**:

- **Right:** priors bridge the cross-season roster-shift gap. That's exactly what the ~85 bp Model A → C gap in year-to-year shows.
- **Wrong:** priors don't do meaningful within-season memorization on top of `teamTricode`. In the single-season split, priors add essentially zero AUC (see §4a decomposition). The "small AUC gain" attributed to within-season priors was really the team dummy's contribution.

**The clean interpretation:**

- **Within-season (paper's target regime):** shooter priors are redundant with `teamTricode`. Removing them is free (Model C ≈ Model A per-shot).
- **Cross-season generalization:** shooter priors are the mechanism that transfers player skill. `teamTricode` cannot substitute because team identities are unstable.
- **For the paper's SPOE analysis (single-season):** Model C is the right ablation. The priors' cross-season utility is real but orthogonal to the SPOE claim.

Model C: mean AUC **0.6627 ± 0.0041** — statistically indistinguishable from Model B, roughly 85 bp below Model A. Fold-to-fold spread (0.6563 – 0.6675) is similar to Model A's (0.6670 – 0.6752). Stability across folds is preserved in all three models.

---

## 5. Why this makes the paper stronger, not weaker (rewritten against Model C SPOE)

The paper's SPOE analysis was intended to measure over/under-performance versus a league-average shooter from the same shot context. Model A can't provide that because its "expected" baseline includes shooter identity; Model B can't either because it strips the team fixed effect the paper's team-level analysis depends on. **Model C is the right baseline for SPOE**: it drops shooter identity while keeping the team context needed for aggregation.

### Player-level SPOE from Model C (high-volume shooters, ≥300 shots)

**Top 10 over-performers (n=276 qualifying players):**

| Player | Shots | Actual | Expected | SPOE |
|---|---:|---:|---:|---:|
| Jokić | 1,364 | 1,710 | 1,428 | **+281.7** |
| Gilgeous-Alexander | 1,656 | 1,883 | 1,717 | +166.2 |
| Pritchard | 866 | 1,073 | 923 | +149.6 |
| LaVine | 1,223 | 1,489 | 1,341 | +147.7 |
| Curry | 1,258 | 1,439 | 1,295 | +144.1 |
| Durant | 1,124 | 1,344 | 1,200 | +143.9 |
| Vučević | 1,038 | 1,229 | 1,093 | +135.5 |
| Herro | 1,378 | 1,553 | 1,419 | +134.3 |
| Brunson | 1,200 | 1,322 | 1,194 | +128.3 |
| Haliburton | 1,006 | 1,170 | 1,053 | +116.9 |

**Bottom 10 under-performers:**

| Player | Shots | Actual | Expected | SPOE |
|---|---:|---:|---:|---:|
| Castle | 988 | 941 | 1,081 | -140.4 |
| Sarr | 828 | 757 | 887 | -129.7 |
| Coulibaly | 624 | 589 | 690 | -101.4 |
| Johnson | 779 | 732 | 832 | -100.5 |
| Council IV | 463 | 409 | 498 | -88.7 |
| Mogbo | 356 | 329 | 417 | -88.4 |
| Westbrook | 831 | 840 | 927 | -87.2 |
| Rozier | 635 | 587 | 671 | -84.1 |
| Anunoby | 1,027 | 1,149 | 1,229 | -80.3 |
| Missi | 492 | 538 | 618 | -79.9 |

Median SPOE across high-volume shooters: +0.81. Mean |SPOE|: 39.5 points.

### Comparison to the paper's Model A leaderboard

The paper reported (Model A): Jokic +148.0, LaVine +133.2, Pritchard +123.0 as top-3.

**Under Model C:** Jokic **+281.7** (nearly 2× the paper's number), SGA **+166.2** (new #2), Pritchard **+149.6**, LaVine **+147.7**.

**Why the numbers grew.** Model A's "expected" for Jokic already assumed elite-Jokic baseline (via `prior_fg_pct`, `prior_shot_type_pct`), so his SPOE was measuring "even better than elite baseline predicted." Model C's expected doesn't know he's Jokic — it just assigns league-average make probability given the shot context. His SPOE now correctly measures how much better than a random shooter he'd be from those exact shots. That's what the paper's SPOE definition claimed to be doing all along; Model C actually delivers it.

### The buy-low / hot-shooting story now works cleanly

- **Positive SPOE (top 10):** Jokić, SGA, Curry, Durant, Haliburton, Brunson, Herro — a clean list of elite shot-makers. Every name is a genuine high-volume star. The interpretation "made shots a league-average shooter wouldn't have" now holds because the expected baseline actually assumes a league-average shooter.
- **Negative SPOE (bottom 10):** dominated by rookies and second-year players (Castle, Sarr, Coulibaly, Council IV, Mogbo, Missi) plus a few veterans past their prime (Westbrook, Rozier). This is exactly the "process is fine, execution isn't yet" pattern the paper's buy-low framing predicted — young players getting decent looks but converting at below-average rates.

Under Model A, the paper's bottom-3 was Grant −82.6 / Coulibaly −79.3 / Murray −76.0 — heavily influenced by those players' prior stats. Under Model C, the bottom is more consistent (all are rookies or veterans with genuinely poor recent shooting), because the expected baseline no longer discounts weak shooters ahead of time.

### Team-level residual (proper construction)

The old §5 proposed measuring "team shooter-skill advantage" as `A_pyth_wins − B_pyth_wins`. That was confounded by the team fixed effect (§4a). The **correct construction is now**:

**Team SPOE = `sum of Model C actual points − sum of Model C expected points`** at team level. One model, one subtraction, no confound, and it inherits the same error bars as the player-level SPOE.

This is directly computable from `team_game_points_no_prior_keep_team.csv` and should replace the A−B residual construction throughout the paper's team-level discussion.

### The paper's core contribution, restated

- **Model A** predicts wins well (r = 0.878 vs real wins) but its SPOE conflates shot quality with shooter identity.
- **Model C** predicts wins nearly as well (r = 0.849, MAE 5.52) and its SPOE cleanly measures over-performance vs. a league-average shooter.
- The gap between them is small at the team level (mean shift 2.04 wins) but meaningful at the player level (Jokic's SPOE nearly doubles because his personal expected baseline stops flattering him).

That's the sharper paper contribution. Not "shot quality correlates with wins" — the paper already had that. The new claim is: **using only publicly-available data, you can build a shot-quality model that predicts team wins nearly as well as one that memorizes shooter identity, and its residuals give you a clean measurement of player-level shot-making skill above expectation.**

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

1. **Compute team-level SPOE from Model C directly** — sum of actual − expected points at team level from `team_game_points_no_prior_keep_team.csv`. Replaces the A−B residual construction throughout the paper.

2. **Cross-season SPOE correlation using Model C outputs.** Year-to-year within-player SPOE correlation is a persistence test — if the paper claims SPOE reflects real skill, correlations should be substantially above chance.

3. **Investigate BKN's +6.53 Model C shift.** The largest remaining outlier under the clean ablation. Either a real story (roster shooting talent well below what shot context implies), a defense/coaching artifact, or small-sample fluke. Worth 1-2 paragraphs in the paper.

4. **Recompute the Pythagorean win exponent choice.** The paper uses 14; standard NBA choices are 14–16.5. Reporting sensitivity to this choice would preempt an obvious reviewer question.

**Paper-writing (once analysis is done):**

5. **Present A and C side-by-side, not A and B.** Model B is the confounded ablation and shouldn't be the paper's headline comparison. Model A is still the "best predictive" model; Model C is the "cleanest interpretive" model without the team-dummy noise.

6. **Reframe SPOE as Model C's residual, not Model A's or Model B's.** This is the biggest change — the paper's buy-low / hot-shooting narrative should be tested against Model C's SPOE ranking, not Model B's.

7. **Add a robustness section** documenting the two-part decomposition: shooter-identity contribution (A vs C) and team fixed-effect contribution (C vs B). The "feature with 0.0001 AUC that dominates aggregate results" insight from §4a belongs here too — it's a general point about validation-metric mismatch that reviewers will find compelling.

8. **State the r drop honestly in the abstract.** "Substantial portion of winning" now carries r = 0.648 (Model B) or an intermediate value (Model C). Report the number, don't hide it.

9. **Update the Discussion's team-outlier explanations** (GSW/BOS over-projection). Under Model C, GSW/BOS's gap should reveal what part was shooter-identity vs. free-throw exclusion vs. open-3 reliance.

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
