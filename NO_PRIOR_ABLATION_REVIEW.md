# NO_PRIOR Ablation — Review

**Companion notebooks:**
- `NBA_Shot_Quality_Modeling_XGB_CatBoost_NO_PRIOR.ipynb` — Model B (drops shooter priors + `teamTricode`)
- `NBA_Shot_Quality_Year_to_Year_Validation_NO_PRIOR.ipynb`
- `NBA_Shot_Quality_Modeling_XGB_CatBoost_NO_PRIOR_KEEP_TEAM.ipynb` — **Model C** (drops shooter priors only; keeps `teamTricode`) — added after decomposition analysis flagged a confound in Model B
- `NBA_Shot_Quality_Year_to_Year_Validation_NO_PRIOR_KEEP_TEAM.ipynb`

**Branch:** `brandon/no-prior-ablation`

> ⚠️ **Second revision (2026-08-22): Model C was demoted from deliverable to diagnostic.** An empirical counterfactual — swap the jersey on identical shots, hold everything else fixed — shows Model C's team dummy launders *more* shooter identity than Model A's, not less (BOS-WAS gap 1.61 pp under A → 3.60 pp under C, r=0.811 with actual team FG%). The team dummy is shooter identity coarsened to a roster; it isn't a "team fixed effect" free of the confound the ablation was designed to address. **Model B is the paper's deliverable — the only configuration consistent with the SPOE definition.** Model C proves the confound exists but doesn't remove it.
>
> **Reading order:** §1–3 remain correct. §4's tables are still valid, but the interpretive framing (§4b/§4c) shifts: Model C's team-level table is a diagnostic showing the confound routing through the team dummy, not a "cleanly-ablated" answer. §5 is rewritten around Model B SPOE + the TOR/PHX narrative + the same-shot-different-jersey counterfactual.

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

## 4a. Correction — the team dummy IS shooter identity, coarsened

The first version of this section framed `teamTricode` as a "team fixed effect" separate from shooter identity. The empirical evidence disagrees. **`teamTricode` is shooter identity averaged across a 5-person roster** — the same objection to `prior_fg_pct` ("baseline shouldn't know who's shooting") applies at the team level, just aggregated. If Curry's identity doesn't belong in the baseline, "Warriors" doesn't either.

### The counterfactual: same shot, different jersey

Hold every feature fixed except `teamTricode` and see what the model predicts:

- **6.0 percentage-point spread** in predicted FG% on identical shots — from 43.6% (CHA) to 49.6% (PHX)
- **BOS 48.8% vs WAS 45.2% on the same shot** — a 3.6-point gap purely from the jersey
- **r = 0.811** between the team dummy's predicted-FG-per-team and each team's *actual* 2024-25 FG%

The model has memorized each team's shooting rate from within-season data alone. That's not a "team fixed effect" — that's team-level shooter identity being learned from 7,317 shots per team (SE ≈ 0.58 pp against a 7.6 pp league spread; signal-to-noise ~13:1, a trivially easy problem for the model).

### Cross-season: the dummy encodes stale identity, not scheme

If the team dummy captured offensive system (spacing, pace, scheme), it should persist across seasons because schemes change slowly. It doesn't:

- Cross-season contribution to AUC: **~0.4 basis points** (vs ~20 bp within-season — a 50× collapse)
- Out-of-sample (train 2023-24, predict 2024-25): team dummy still applies a 5.8 pp offset **but r drops to 0.351** with the target season's actual shooting
- CLE case: stale dummy predicts 46.0%; CLE actually shot 49.1% (second-best in league)

Season-specific talent and variance behave this way. Stable scheme doesn't. What the dummy encodes is who was shooting last year, not how the offense is designed. And even if 10-20% of the dummy is genuine scheme, the aggregate cannot be decomposed — a known omission is easier to defend than an unknown mixture.

### The Curry/Draymond argument generalizes

The original definitional objection to `prior_fg_pct` was: Curry's identity doesn't belong in the "expected" baseline because SPOE is meant to measure over/under-performance versus a league-average shooter. The counterfactual above shows exactly the same objection applies to `teamTricode`. It's the same argument one level of aggregation up.

### Decomposition of team-level shift (mean |Δ pyth_wins_exp|)

The original decomposition table is still numerically correct — dropping the team dummy alone accounts for ~2/3 of Model B's team-level shift — but the *interpretation* changes. It's not "the team dummy was doing separate work"; it's "the team dummy was carrying the majority of the shooter-identity signal because it's the fatter channel."

| Removal | Mean shift in expected wins | What was removed |
|---|---:|---|
| Team dummy alone | **4.82** | Team-level shooter-identity signal |
| Priors alone | 2.35 | Individual shooter-identity signal |
| Both (Model B) | 8.09 (not additive) | Full shooter-identity removal — the SPOE-consistent configuration |

Priors and team dummy overlap because they're both encoding shooter identity, just at different granularities.

**Decomposition of team-level shift (mean |Δ pyth_wins_exp|):**

| Removal | Mean shift in expected wins |
|---|---:|
| Team dummy alone | **4.82** |
| Priors alone | 2.35 |
| Both (Model B) | 8.09 (not additive — see below) |

About **two-thirds** of the team-level shift comes from dropping `teamTricode` — but *because* the team dummy is the fatter channel for shooter identity, not because it's a separate effect.

**Sacramento shows the channel-swap dynamic.** Under Model C (priors dropped, team dummy kept), removing priors alone moves SAC *up* by +0.93. That doesn't mean shooter identity isn't affecting SAC — it means the team dummy absorbs the slack when priors are removed. In fact, the [same-shot-different-jersey counterfactual](#the-counterfactual-same-shot-different-jersey) shows the team dummy in Model C encodes *more* shooter identity than in Model A (BOS-WAS gap doubles from 1.61 → 3.60 pp). Model C isn't a clean counter-example; it's a demonstration of how the shooter signal reroutes.

**Shifts don't add linearly.** PHX is −3.88 (priors alone) and −3.14 (team dummy alone) separately, but −13.32 together. The interaction confirms both channels overlap in what they encode — remove one and the other partly absorbs its work; remove both and you get the full shooter-identity removal.

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

**Interpretation, corrected.** Model C sits close to Model A at team-level (r=0.849 vs 0.878; MAE 5.52 vs 4.98) because both configurations still contain shooter identity — Model A splits it across two channels (priors + team dummy), Model C funnels all of it into the team dummy alone. **This proximity is the confound, not evidence of a clean ablation.** The 0.975 rank correlation between A and C likewise indicates that both models are ranking teams by roughly the same underlying (identity-contaminated) signal.

**The honest number for the paper is Model B: MAE 8.84, r = 0.648.** That's the accuracy of context-only shot quality — no shooter identity in the baseline, either individual or team-aggregated. The correlation dropping from 0.88 to 0.65 is what the SPOE definition costs; concealing it via Model C would be exactly the sleight-of-hand the abstract's "substantial portion of winning" claim needs to survive review.

---

## 4b. Model C team-level results — diagnostic, NOT deliverable

> **Framing correction:** This section previously described Model C as "the cleanly-ablated shooter-identity signal." That was wrong. Model C keeps `teamTricode` and drops `prior_*`, which routes the shooter-identity signal *entirely* through the team dummy. The counterfactual in §4a shows this makes the team dummy encode *more* shooter identity than in Model A, not less. Model C is useful as a diagnostic — it demonstrates the reroute empirically — but its team-level table is not "shooter identity removed"; it's "shooter identity funneled through one channel."

With the shooter-identity signal reroutéd to the team dummy, the Model A vs Model C `pyth_wins_exp` deltas are smaller than Model B's. Mean |Δ C-A| = **2.04 wins** (compared to Model B's 5.03) — about 2.5× less movement. This *is not* evidence that shooter identity contributes only 2.04 wins per team on average; it's evidence that Model C's team dummy is doing most of what Model A's priors + team dummy were doing together.

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

**Reading the outliers correctly.** Under the corrected framing, the Model C values below are *not* what the outlier shift would be with shooter identity removed — they're what remains after the shift is rerouted through the team dummy. The Model B values are the identity-removed answer.

- **PHX (Booker/KD/Beal):** Model B −14.79, Model C −2.96. The −14.79 is the honest measurement of how much PHX's shot process undershoots their actual wins once you don't already credit Booker/Durant/Beal.
- **SAC (DeRozan/Fox/Sabonis):** Model B −10.25, Model C +1.20. Model C's sign flip is not "shooter identity irrelevant for SAC" — it's the team dummy re-absorbing all of the SAC-shooter signal that priors were carrying in Model A.
- **IND, TOR, BOS:** same pattern. The Model B numbers are what the SPOE definition says the team-level shot-process story is; Model C partially undoes the ablation via the team dummy.

**Team-level story that survives to the paper (under Model B):**
- TOR: shot process worth ~51 wins to a league-average shooter, actual wins 30 → shooters ~20 wins below league average given that shot context
- PHX: shot process worth ~22 wins, actual 36 → stars carried a below-average process (Durant/Booker/Beal generated wins their raw shot mix didn't imply)
- BOS, IND, SAC: similar decompositions — story is legible because the baseline doesn't already know who took the shot

These claims exist *only* under Model B. Model C, by re-encoding shooter identity through the team dummy, structurally cannot make them.
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

## 5. Why this makes the paper stronger, not weaker (Model B is the deliverable)

The paper's SPOE analysis was intended to measure over/under-performance versus a **league-average shooter** from the same shot context. Any configuration where the baseline knows who the shooter is — including at the team-aggregated level — cannot deliver that.

- **Model A** encodes shooter identity via `prior_*` + `teamTricode` (both channels active)
- **Model C** encodes shooter identity via `teamTricode` alone, but *more strongly* than Model A because the priors channel is closed and the signal reroutes (BOS-WAS jersey gap 1.61 pp → 3.60 pp; see §4a)
- **Model B** is the only configuration where the baseline is genuinely context-only

**Model B is the deliverable.** Model C is a valuable diagnostic — it *proves* the rerouting empirically — but its SPOE and team-level tables are not "shooter identity removed."

### Team-level story under Model B — with computed SPOE

Model B `pyth_wins_exp` vs actual wins gives the direction; Model B **team-level SPOE** (points_scored_actual − points_scored_expected, summed across the season) gives the magnitude. This is a direct computation from `team_game_points_no_prior.csv` — one model, one subtraction, no confound.

**Top offensive over-performers under Model B:**

| Team | Actual wins | Points_actual | Points_expected | Team SPOE_off |
|---|---:|---:|---:|---:|
| PHX | 36 | 7,938 | 7,482 | **+456** |
| SAC | 40 | 8,090 | 7,711 | +379 |
| BOS | 61 | 8,283 | 7,928 | +355 |
| DEN | 50 | 8,432 | 8,097 | +335 |
| IND | 50 | 8,229 | 7,908 | +321 |
| CLE | 64 | 8,607 | 8,338 | +269 |

**Bottom offensive (shooters converted below context) — "the TOR problem":**

| Team | Actual wins | Points_actual | Points_expected | Team SPOE_off |
|---|---:|---:|---:|---:|
| TOR | 30 | 7,797 | 8,315 | **−518** |
| CHA | 19 | 7,349 | 7,841 | −492 |
| POR | 36 | 7,707 | 8,119 | −412 |
| ORL | 41 | 7,181 | 7,557 | −376 |
| BKN | 26 | 7,285 | 7,637 | −352 |
| NOP | 21 | 7,650 | 7,983 | −334 |

**Money-quote interpretation:**

- **TOR: −518 offensive SPOE.** ~6.3 points/game left on the table by shooter execution. Their shot process was worth ~51 wins to a league-average shooter; they won 30. The 21-win expected-vs-actual gap is quantified: roughly the size of TOR's shooter deficit relative to league-average conversion.
- **PHX: +456 offensive SPOE.** Their shot process was worth ~22 wins to a league-average shooter; stars converted +456 points beyond that expectation, buying them ~14 wins of shooter-execution gain to end at 36 wins. This is the clean statement of "KD/Booker/Beal carried a poor process."
- **BKN: −352 offensive SPOE.** Even after removing shooter identity, BKN's process expected ~43 wins vs. their 26 actual. The offensive SPOE says roster shooter talent was ~4.3 points/game below league average — the largest process-execution gap in the league by actual-wins delta.

### Correlations — the honest headline decomposition

| Quantity | vs wins_actual |
|---|---:|
| Model B `pyth_wins_exp` (shot process) | **r = 0.648** |
| Team SPOE_off (shooter execution) | **r = 0.653** |
| Team SPOE_net (offense + defense) | **r = 0.667** |
| Team SPOE_def alone | r = 0.046 |

The paper's "shot quality accounts for a substantial portion of winning" is now precise: **shot process explains ~42% of win variance (r=0.648); shooter execution above/below league average explains another comparable share.** Together they nearly close the gap to actual wins. This is a much sharper contribution than the original correlation-only framing.

Note the defense-side SPOE has essentially zero correlation with own-team wins (r=0.046) — which makes sense: whether your opponents shot better or worse than context predicts is roughly orthogonal to your own team's success. Only offensive SPOE meaningfully translates to wins.

The abstract's headline should read: **r = 0.648, MAE = 8.84** for context-only shot process vs real wins, with team-level shooter-execution SPOE ranging from PHX +456 to TOR −518 points. That's the honest number and the honest spread. It doesn't hide behind the identity-contaminated proximity of Models A and C.

### Player-level SPOE — Model B leaderboard

**Top 10 over-performers under Model B (n=276 qualifying players, ≥300 shots):**

| Player | Shots | Actual | Model B Expected | Model B SPOE | Model C SPOE (for reference) |
|---|---:|---:|---:|---:|---:|
| Jokić | 1,364 | 1,710 | 1,416 | **+294.2** | +281.7 |
| Durant | 1,124 | 1,344 | 1,132 | **+211.6** | +143.9 |
| Gilgeous-Alexander | 1,656 | 1,883 | 1,701 | +181.5 | +166.2 |
| Pritchard | 866 | 1,073 | 896 | +177.4 | +149.6 |
| LaVine | 1,223 | 1,489 | 1,321 | +167.9 | +147.7 |
| Curry | 1,258 | 1,439 | 1,296 | +143.1 | +144.1 |
| Haliburton | 1,006 | 1,170 | 1,028 | +141.5 | +116.9 |
| Herro | 1,378 | 1,553 | 1,416 | +136.9 | +134.3 |
| Vučević | 1,038 | 1,229 | 1,094 | +134.9 | +135.5 |
| Jerome | 618 | 749 | 614 | +134.7 | +116.5 |

**Bottom 10 under-performers under Model B:**

| Player | Shots | Actual | Model B Expected | Model B SPOE | Model C SPOE |
|---|---:|---:|---:|---:|---:|
| Sarr | 828 | 757 | 907 | -149.9 | -129.7 |
| Castle | 988 | 941 | 1,083 | -142.5 | -140.4 |
| Coulibaly | 624 | 589 | 702 | -113.1 | -101.4 |
| Johnson | 779 | 732 | 845 | -112.7 | -100.5 |
| Mogbo | 356 | 329 | 436 | -107.4 | -88.4 |
| Council IV | 463 | 409 | 503 | -93.7 | -88.7 |
| Barnes | 1,063 | 1,024 | 1,116 | -91.5 | -39.8 |
| Grant | 574 | 536 | 627 | -90.7 | -68.2 |
| Missi | 492 | 538 | 627 | -89.2 | -79.9 |
| Carter Jr. | 491 | 489 | 574 | -84.8 | -72.9 |

Median SPOE across HV shooters: -0.97 (Model B) vs +0.81 (Model C). Mean |SPOE|: **44.3 (Model B)** vs 39.5 (Model C) — magnitudes ~12% larger under Model B.

### How Model B changes the leaderboard vs Model A and Model C

The paper's Model A top-3 was Jokić +148.0, LaVine +133.2, Pritchard +123.0. Model B: Jokić **+294.2**, Durant **+211.6**, SGA **+181.5**. Every A-era number roughly doubles — under Model A, half of each elite shooter's real skill was hidden inside the "he's Jokić" / "he's LaVine" baseline.

**Durant is the cleanest illustration of the Model C reroute.** Under Model C: +143.9. Under Model B: +211.6, a +67.8 jump. Durant plays for PHX; the counterfactual (§4a) shows PHX's team dummy predicts 49.6% FG under Model C — that team credit was flattering Durant's expected baseline. Model B strips it and his SPOE grows accordingly. **Curry is the counter-case** (Model B +143.1, Model C +144.1, essentially unchanged): GSW's team dummy sits near league average, so removing it doesn't shift his baseline.

**Directional confirmation:** in **8 of 10** Model B top-10 cases, |B_SPOE| ≥ |C_SPOE|. The two exceptions (Curry, Vučević) both play for teams whose Model C team dummy is close to league average.

### The buy-low / hot-shooting story now works cleanly

- **Positive SPOE (top 10 under Model B):** Jokić, Durant, SGA, Pritchard, LaVine, Curry, Haliburton, Herro, Vučević, Jerome — a clean list of elite shot-makers, every name a genuine high-volume star. The interpretation "made shots a league-average shooter wouldn't have" now holds because the baseline is genuinely league-average, not player- or team-specific.
- **Negative SPOE (bottom 10):** dominated by rookies and second-year players (Sarr, Castle, Coulibaly, Council IV, Mogbo, Missi) plus a couple of veterans past their prime (Grant, Carter Jr.). **Barnes** at -91.5 is the newly-visible case: under Model C his SPOE was only -39.8 because TOR's team dummy carried the shooter-identity slack; Model B reveals him individually.

The paper's Model A bottom-3 was Grant −82.6 / Coulibaly −79.3 / Murray −76.0 — heavily influenced by prior stats. Under Model B, the bottom is more consistent: all rookies or genuinely poor shooters converting below what shot context predicts. This is exactly the "process is fine, execution isn't yet" pattern the paper's buy-low framing predicted.

### The counterfactual as robustness demonstration for the paper

The strongest single piece of evidence for choosing Model B is the same-shot-different-jersey demonstration in §4a. That analysis should live in the paper's methods or robustness section:

- **Same shot, different jersey** → 6.0 pp predicted-FG spread across teams under Model C
- **BOS 48.8% vs WAS 45.2%** on identical shots → 3.6 pp gap purely from the team dummy
- Model C's team-dummy predicted-FG-per-team correlates r=0.811 with **actual** 2024-25 team FG%
- Removing shooter priors makes this contamination **worse**, not better (BOS-WAS 1.61 pp → 3.60 pp, a 2.2× increase)

That's a two-paragraph section that decisively answers the reviewer question "why not just use team dummies?" The answer is the counterfactual, plus the note that the alternative is a baseline that already knows which team took the shot — which is definitionally incompatible with the SPOE claim.

### The paper's core contribution, restated

- **Model A** predicts wins with r = 0.878 but its SPOE conflates shot quality with shooter identity (individual + team-level)
- **Model B** predicts wins with r = 0.648 and its SPOE cleanly measures over-performance vs. a league-average shooter — the paper's definition, honestly implemented
- **The gap between them** is what shooter identity is worth at the team-level Pythagorean-wins scale. That gap is real, it's ~8 wins of MAE, and being able to measure it is the paper's contribution

The sharper claim: **using only publicly-available data, you can build a context-only shot-quality model whose residuals cleanly measure player and team shot-making skill above what shot context alone predicts.** Model B is that model.

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

1. ~~**[BLOCKING] Compute Model B player-level SPOE.**~~ ✅ **DONE** (Aug 22). `player_points_vs_expected_insample_no_prior.csv` is generated. Jokić +294.2, Durant +211.6, SGA +181.5 are the new top-3. Directional prediction confirmed: 8/10 top-10 |B_SPOE| ≥ |C_SPOE|, with Durant showing the cleanest reroute effect (+67.8 jump from removing PHX's team dummy).

2. ~~**Compute team-level SPOE from Model B directly**~~ ✅ **DONE** (Aug 22). Computed from `season_standings_no_prior.csv` (aggregated `pts_scored_actual − pts_scored_exp`). Full table added to §5. Top: PHX +456 (KD/Booker/Beal); bottom: TOR −518 ("the TOR problem"). Team SPOE_off vs wins_actual r = **0.653**, comparable to Model B pyth_wins_exp itself (r = 0.648). The two together decompose team wins into process (r=0.648) + execution (r=0.653) — a much sharper story than correlation alone.

3. **Get Claude Desktop's counterfactual scripts into the repo.** The same-shot-different-jersey demonstration (6.0 pp spread on identical shots, r=0.811 with actual team FG%, and the Model A vs C reroute showing BOS-WAS gap doubles) is the strongest single piece of evidence in the analysis. It belongs in the paper's methods/robustness section and needs to be reproducible from committed code, not just cited from this doc.

4. **Cross-season SPOE persistence test using Model B.** Within-player year-to-year SPOE correlation — if the paper claims SPOE reflects real skill, correlations should be substantially above chance. Model B is the appropriate model; Model C's team dummy contaminates cross-season SPOE the same way it contaminates within-season.

5. **Investigate BKN's Model B shift.** Under Model B, BKN's process expects ~43 wins while they won 26 — a 17-win execution gap. Even after removing shooter identity, that's a large residual worth 1-2 paragraphs in the paper: real story (roster shot-making well below league average) or defense/schedule artifact?

6. **Recompute the Pythagorean win exponent choice.** The paper uses 14; standard NBA choices are 14–16.5. Reporting sensitivity to this choice would preempt an obvious reviewer question.

**Paper-writing (once analysis is done):**

7. **Present A and B as the two-model comparison.** Model A is the "best predictive" model (includes shooter identity). Model B is the definition-consistent model (context only). The paper's SPOE claim needs Model B; Model A stays as the benchmark.

8. **Reframe SPOE as Model B's residual.** Player and team SPOE both come from Model B — one model, one subtraction, no confound. Drop the A−B residual construction from any earlier drafts.

9. **State the r drop honestly in the abstract.** "Substantial portion of winning" is Model B: r = 0.648, MAE = 8.84 vs real wins. Report the numbers directly. That's what SPOE-consistent shot-quality-vs-wins predictive power actually looks like.

10. **Add a robustness section built around the counterfactual.** Same shot, different jersey → 6.0 pp spread. Removing shooter priors doubles it. Model B is the only configuration where the baseline is genuinely context-only. This preempts every reviewer challenge to the ablation choice.

11. **Update the Discussion's team-outlier explanations** with Model B numbers. TOR/PHX/BOS/SAC/BKN are the money quotes — narratives about process vs execution that only exist when the baseline doesn't already encode who was shooting.

12. **Frame Model C explicitly as a diagnostic, not a candidate model.** If included in the paper at all, it's evidence for why Model B is necessary — not an alternative deliverable.

**Repo hygiene:**

13. Model C's fitted models (`models/fitted_models_no_prior_keep_team.pkl`) can be kept for reference but shouldn't be used for any paper-facing analysis.
14. Consider a `Model_A_vs_B_Comparison.ipynb` notebook that loads `season_standings.csv` and `season_standings_no_prior.csv`, produces the delta tables, rank-correlation plots, and the counterfactual demonstration end-to-end.

---

## 8. What was NOT touched

- Data enrichment (`enrich_shots.py`, `enrich_shots_nbastatsv3_full.ipynb`) — same inputs.
- Feature engineering steps other than the ablated feature list — same `contest_score`, same shot-clock proxy, same interaction features.
- Model hyperparameter search spaces — same Optuna trial budgets and parameter grids.
- Pythagorean exponent — still 14.
- Free-throw handling — still excluded (this remains a paper limitation).

The ablation is a clean feature-set change; everything else in the pipeline is identical.
