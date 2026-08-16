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

# Shot Clock Proxy — Validated Findings (deferred, not yet fixed)

Measured against NBA.com ground truth by `validate_shot_clock.py`; full numbers
in `shot_clock_validation/summary_2024-25.md`. Nothing below has been changed in
`compute_shot_clock_v4` — this is the to-do list.

## What validation showed (2024-25)

The proxy is **strong in the middle, broken at the extremes**:

| Range | Proxy share | True share | Proxy FG% | True FG% |
|---|---|---|---|---|
| 24-22 | 0.7% | 3.2% | 47.5% | 59.2% |
| 22-18 → 7-4 | close | close | within 0.3–1.5pp | — |
| 4-0 | 13.5% | 9.1% | 42.0% | 35.9% |

Headline: FG% mean absolute deviation 3.51pp, distribution TVD 0.111, per-shot
range agreement bounded above by 85.7%. The middle four ranges cover ~86% of all
shots and match well, which is what currently supports the paper's claim.

## 14. Model inbound delay on dead-ball resets (highest value)

**The dominant error, with a known mechanism.** Reset rules split by whether a
possession needs an inbound pass:

- Dead-ball (`made_fg`, `final_ft`): mean **8.7s** remaining, 20.8% of shots ≤4s
- Live-ball (`def_reb`, `off_reb`): mean **13.3s** remaining, 6.2% of shots ≤4s

A **4.6s gap in the wrong direction** — teams push in transition after a made
basket, so those possessions should show *more* clock left, not less. The cause:
`compute_shot_clock_v4` starts the 24s clock at the made-basket event timestamp,
but the real clock starts when a player legally touches the ball inbounds. Every
after-a-basket possession is charged for inbound time that never ran off.

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

## 16. Review the `def_foul_14` rule

Independently suspicious: 40% of its shots land in 4-0 and 34% in 7-4 — far more
expiring-clock mass than any other rule. Worth checking separately from the
inbound issue.

## 17. Correct the documented no-reset rate

`CLAUDE.md` states ~0.7% of shots have no identifiable reset. Measured for
2024-25 it is **2.92%** (6,409 of 219,528).

## 18. Per-shot ground truth (if a hard accuracy number is needed)

NBA.com only publishes shot-clock *ranges*, so no confusion matrix or agreement
rate is derivable from it — 85.7% is a bound, not an accuracy. The 2014-15
SportVU shot logs carry per-shot `SHOT_CLOCK` stamps and would join to
`raw_data/nbastatsv3_2014.csv` on player/period/game-clock/distance. Availability
unverified, and it validates only pre-2018 rules (no 14s offensive-rebound reset).
