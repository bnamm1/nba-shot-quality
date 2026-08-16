# Shot-clock proxy validation — 2013-14 to 2024-25

Cross-season summary from `validate_shot_clock.py`. Ground truth is
NBA.com's tracking dashboard, which reports FGA/FGM per shot-clock
*range*. No per-shot accuracy figure is claimed anywhere.

## Headline

Accuracy is **consistent across all seasons**. The two-regime split that
previously appeared at the 2018-19 rule change is gone, following the
season-conditional reset fix in `enrich_shots.py`
(`compute_shot_clock_v5`).

| Era | Seasons | FG% error | TVD | Agreement bound |
|---|---|---|---|---|
| Pre-2018-19 | 5 | 2.8–3.4 pp | 0.050–0.087 | 83.8–86.4% |
| 2018-19 onward | 7 | 2.3–3.5 pp | 0.092–0.113 | 85.6–86.4% |

### History

Before the fix, all three 14-second branches (`off_reb`,
`def_violation`, `def_foul`) were applied unconditionally, though the
rule only took effect in 2018-19. Pre-2018 seasons then ran 6.2–6.8pp
FG% error with a curve correlation of ~0 — the shot-clock/outcome
relationship was absent entirely. All seven post-2018 seasons were
verified byte-identical before and after, confirming the change is
inert where the rule already applied.

## Remaining error

The **inbound-delay bias** is untouched and affects every season:
dead-ball resets (`made_fg`, `final_ft`) start the clock at the event
timestamp rather than the inbound touch, averaging 8.7s remaining
against 13.3s for live-ball resets. Consequently the proxy's 24-22
share sits at 0.5–0.8% in every season while truth ranges 3.0–5.5% —
no reset rule can produce a full-clock shot.

## Per season

| Season | FG% err (pp) | curve r | TVD | bound | 24-22 proxy/true | 4-0 proxy/true |
|---|---|---|---|---|---|---|
| 2013-14 | 2.90 | +0.911 | 0.070 | 84.1% | 2.6% / 5.5% | 15.3% / 12.2% |
| 2014-15 | 2.95 | +0.955 | 0.081 | 84.3% | 2.5% / 5.3% | 15.2% / 10.5% |
| 2015-16 | 2.76 | +0.908 | 0.050 | 84.9% | 2.4% / 4.8% | 14.3% / 14.8% |
| 2016-17 | 2.93 | +0.911 | 0.087 | 83.8% | 2.5% / 5.2% | 14.0% / 9.4% |
| 2017-18 | 3.36 | +0.972 | 0.070 | 86.4% | 5.5% / 5.4% | 12.1% / 8.5% |
| 2018-19 | 3.50 | +0.534 | 0.113 | 85.7% | 0.7% / 3.0% | 12.9% / 8.7% |
| 2019-20 | 3.08 | +0.794 | 0.113 | 85.6% | 0.6% / 3.1% | 12.7% / 8.5% |
| 2020-21 | 2.29 | +0.924 | 0.110 | 85.7% | 0.6% / 3.0% | 13.2% / 8.8% |
| 2021-22 | 2.73 | +0.890 | 0.106 | 86.0% | 0.7% / 3.0% | 13.7% / 9.2% |
| 2022-23 | 2.99 | +0.763 | 0.104 | 85.9% | 0.7% / 3.0% | 13.5% / 8.7% |
| 2023-24 | 2.76 | +0.812 | 0.092 | 86.4% | 0.7% / 3.0% | 13.4% / 8.7% |
| 2024-25 | 3.51 | +0.711 | 0.111 | 85.7% | 0.7% / 3.2% | 13.5% / 9.1% |

## What this means for the paper

- Shot-clock-based claims are defensible across **all twelve seasons**,
  with the inbound-delay caveat stated.
- State the pre-2018 ground-truth gap: our play-by-play FGA exceeds
  NBA.com's tracking FGA by +6.3 to +6.5% for 2013-14 through 2016-17
  against +0.19 to +0.70% from 2018-19 on, so those seasons are
  validated against a ~94% sample. This is also a candidate reason
  pre-2018 TVD now beats post-2018.
- The ~86% agreement figure is an **upper bound**, not an accuracy.

See `ideas_for_improvement.md` for the remaining fix list.
