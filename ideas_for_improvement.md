Current Ceiling Analysis
The theoretical ceiling for shot prediction is probably ~72-75% AUC because:
Even perfect information can't predict random variation (shooter's touch that day)
NBA players are skilled - good shooters make tough shots, bad shooters miss open ones
High-Impact Ideas (Likely to Help)
1. Assisted vs Unassisted Shots
Parse play-by-play to identify if shot was assisted. Assisted shots are typically higher quality.

# Could parse from description or look at next event
df['is_assisted'] = desc_lower.str.contains('assist').astype(int)
2. In-Game Shooter Momentum (Hot Hand)
Rolling average of player's last N shots in the game. Research shows hot hand is real but small.

# Careful with leakage - only use PRIOR shots
df['player_game_fg_pct_prior'] = df.groupby(['gameId', 'personId'])['shotResult'].transform(
    lambda x: x.shift().expanding().mean()
)
3. Team Pace Factor
Fast-paced teams create more transition opportunities (higher FG%). Use prior season pace.
4. Rest Days / Back-to-Back
Players shoot worse on back-to-backs. Could add days_rest feature from schedule.
5. Quarter-End Heaves
Separate feature for end-of-quarter situations (different from just late shot clock).

df['is_quarter_end'] = (df['period_sec_remaining'] <= 3).astype(int)
6. Shot Sequence / Offensive Rebound
Is this a putback after offensive rebound? Second-chance points have different FG%.
Medium-Impact Ideas
7. Better Contest Estimation
Use defender positions if available from tracking data
Parse "contested" from description more carefully
8. Player Archetype
Cluster players by shooting style (spot-up shooter, slasher, post player) and use as feature.
9. Matchup History
How does this shooter perform against this specific team historically?
10. Time of Season
Early season (rust), mid-season (rhythm), late season (fatigue), playoffs (intensity).
Model Architecture Ideas
11. Stacking/Ensemble
Combine predictions from multiple models (LR, RF, XGB, CatBoost) via meta-learner.
12. Feature Selection
Remove noisy features. Use SHAP or permutation importance to identify low-value features.
13. Calibration
Use Platt scaling or isotonic regression to improve probability calibration.
Which of these would you like to pursue? I'd recommend:
Assisted shots - Easy to implement, clear signal
Quarter-end heaves - Easy, should help separate true desperation shots
Stacking ensemble - Can squeeze out extra 0.5-1% AUC

---

# Shot Clock Proxy — Validated Findings

Measured against NBA.com ground truth by `validate_shot_clock.py`. Ground truth
is range-level (FGA/FGM per shot-clock range per player), so no per-shot accuracy
figure appears anywhere below.

## DONE — season-conditional reset (was the larger of the two bugs)

**Fixed in `enrich_shots.py` (`compute_shot_clock_v5`).** v4 applied the 14-second
reset unconditionally, but all three of its 14s branches (`off_reb`,
`def_violation`, `def_foul`) are 2018-19 rules. v5 derives the season from
`gameId` and uses 24s before then.

Result — every pre-2018 season improved on every metric:

| Season | FG% err (pp) | TVD | Bound | curve r |
|---|---|---|---|---|
| 2013-14 | 6.79 → 2.90 | 0.141 → 0.070 | 78.3 → 84.1% | 0.006 → 0.911 |
| 2014-15 | 6.82 → 2.95 | 0.147 → 0.081 | 78.0 → 84.3% | 0.053 → 0.955 |
| 2015-16 | 6.24 → 2.76 | 0.140 → 0.050 | 79.0 → 84.9% | 0.079 → 0.908 |
| 2016-17 | 6.81 → 2.93 | 0.171 → 0.087 | 76.6 → 83.8% | −0.060 → 0.911 |
| 2017-18 | 4.20 → 3.36 | 0.144 → 0.070 | 79.8 → 86.4% | 0.698 → 0.972 |

The headline: the FG%/shot-clock relationship was **absent** in 2013-14 through
2016-17 (r ≈ 0) and is recovered in all of them. All seven post-2018 seasons were
verified byte-identical to v4 (~1.47M shots, zero mismatches), confirming the
change is inert where the rule already applied.

## The fix did NOT improve model performance (important for framing)

Rolling temporal validation re-run on v5 data (`rerun_rolling_validation.py`):
AUC moved by between **+0.0001 and −0.0007** across all eight folds, and accuracy
by ≤0.0002. Effectively zero, even though five of the training seasons went from a
shot-clock/outcome correlation of ~0 to ~0.91.

**Why this is not a contradiction.** Shot clock is a weak predictor of shot outcome
once distance, shot type and shooter quality are known. The real FG%-by-range curve
runs 59.2% → 35.9%, but the bulk of shots (47% of attempts, the 15-7 range) sit at
46.4%, essentially league average. Only the tails discriminate, and they are ~12%
of attempts. With 62 features available, gradient boosting recovers most of that
signal from correlated features anyway.

**What to claim in the paper.** The fix establishes `SHOT_CLOCK_APPROX` as a
*valid measurement* — validated against NBA.com ground truth, TVD halved, curve
correlation recovered. It does **not** improve prediction, and claiming otherwise
would not survive review. The honest framing is measurement validity, plus the
finding that shot clock carries little marginal predictive power in this model.
That second result is publishable in its own right.

## Where the proxy stands now (2024-25, representative)

**Strong in the middle, still weak at the extremes:**

| Range | Proxy share | True share | Proxy FG% | True FG% |
|---|---|---|---|---|
| 24-22 | 0.7% | 3.2% | 47.5% | 59.2% |
| 22-18 → 7-4 | close | close | within 0.3–1.5pp | — |
| 4-0 | 13.5% | 9.1% | 42.0% | 35.9% |

FG% mean absolute deviation 3.51pp, TVD 0.111, range agreement bounded above by
85.7%. The middle four ranges cover ~86% of all shots and match well, which is
what supports the paper's claim.

## 14. Model inbound delay on dead-ball resets (now the top remaining item)

**The dominant error, with a known mechanism.** Reset rules split by whether a
possession needs an inbound pass:

- Dead-ball (`made_fg`, `final_ft`): mean **8.7s** remaining, 20.8% of shots ≤4s
- Live-ball (`def_reb`, `off_reb`): mean **13.3s** remaining, 6.2% of shots ≤4s

A **4.6s gap in the wrong direction** — teams push in transition after a made
basket, so those possessions should show *more* clock left, not less. The cause:
the reconstruction starts the 24s clock at the made-basket event timestamp, but
the real clock starts when a player legally touches the ball inbounds. Every
after-a-basket possession is charged for inbound time that never ran off. This is
**unchanged in v5** and affects every season equally.

Consequence: no reset rule can currently produce a full-clock shot (max 2% of any
rule's shots land in 24-22), and the 4-0 bucket is diluted with ordinary shots
rather than the desperation heaves it should isolate — which is why its FG% reads
42.0% instead of 35.9%.

**Do NOT fix with a constant offset.** Tested: +4.5s on dead-ball resets halves
TVD (0.111 → 0.055) and lifts the bound to 91.4%, but *degrades* the FG% curve
correlation (0.71 → 0.33). It games the marginal distribution while moving shots
into ranges whose scoring profile they do not share. The fix belongs inside the
reconstruction, estimating inbound time per possession.

## 15. Sub-second clock precision

`_abs_time()` parses `PT11M43.00S` with `int()`, discarding decimals. This makes
`SHOT_CLOCK_APPROX` strictly integer-valued (25 distinct values) and floors every
estimate. Side effect: range-edge convention moves whole integer classes between
buckets, swinging TVD from 0.111 (right-closed) to 0.071 (left-closed).

## 16. Review the `def_foul` rule (partly explained, still open)

Suspicious in v4: 40% of its shots landed in 4-0 and 34% in 7-4, far more
expiring-clock mass than any other rule. Part of this was the season bug — pre-2018
it was capping at 14s when the real reset was 24s — and v5 fixes that portion. It
is still worth re-checking post-2018, where the 14s cap is correct but the mass
still looks heavy. Note the rule is named `def_foul` in v5; the old `_14` suffix
was false for pre-2018 seasons.

## 17. Per-shot ground truth (if a hard accuracy number is needed)

NBA.com only publishes shot-clock *ranges*, so no confusion matrix or agreement
rate is derivable from it — the ~86% figure is a bound, not an accuracy. The
2014-15 SportVU shot logs carry per-shot `SHOT_CLOCK` stamps and would join to
`raw_data/nbastatsv3_2014.csv` on player/period/game-clock/distance. Availability
unverified.

## 18. NBA.com ground truth is incomplete pre-2018 (caveat, not a fix)

Our play-by-play FGA exceeds NBA.com's tracking FGA by **+6.3% to +6.5%** for
2013-14 through 2016-17, against **+0.19% to +0.70%** from 2018-19 onward —
roughly 12,000 shots per season with no SportVU-era tracking record.

This does not explain the v5 improvement (both versions faced the same
denominator), but it does mean pre-2018 validation compares our full shot set
against a ~94% sample of it, so those metrics carry more uncertainty. It is also a
candidate explanation for why pre-2018 TVD now *beats* post-2018 (2015-16 at 0.050
vs a post-2018 best of 0.092): a partial ground truth may simply be easier to
match. Worth stating in the paper rather than letting a reviewer find it.
