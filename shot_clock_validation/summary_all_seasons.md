# Shot-clock proxy validation — 2013-14 to 2024-25

Cross-season summary from `validate_shot_clock.py`. Ground truth is
NBA.com's tracking dashboard, which reports FGA/FGM per shot-clock
*range*. No per-shot accuracy figure is claimed anywhere.

## Headline

The proxy splits into **two regimes** at the 2018-19 rule change:

| Era | Seasons | FG% error | TVD | Agreement bound |
|---|---|---|---|---|
| Pre-2018-19 | 5 | 4.2–6.8 pp | 0.140–0.171 | 76.6–79.8% |
| 2018-19 onward | 7 | 2.3–3.5 pp | 0.092–0.113 | 85.6–86.4% |

**The pre-2018 seasons are not merely worse — the FG%/shot-clock
relationship is absent.** Curve correlation across the six ranges is
≈ 0 for 2013-14 through 2016-17 (0.006, 0.053, 0.079, −0.060). In those
seasons the proxy carries essentially no valid shot-clock signal, so any
model consuming `SHOT_CLOCK_APPROX` there is consuming noise.

## Cause

`compute_shot_clock_v4` applies the 14-second offensive-rebound reset
unconditionally: `add_reset(per, t, 14, "off_reb", ...)`. That rule only
took effect in **2018-19**; before it, an offensive rebound reset to 24.
Verified directly — in 2015-16 every `off_reb` shot is capped at 14.0s
(n=18,342, max 14.0s), roughly 10 seconds too low. `CLAUDE.md` lists this
as a known limitation, but the code does not branch on season.

This is **separate from** the inbound-delay bias, which affects all
seasons: the proxy's 24-22 share sits at 0.5–0.8% in every season while
truth ranges 3.0–5.5%. No reset rule can produce a full-clock shot.

## Per season

| Season | FG% err (pp) | curve r | TVD | bound | 24-22 proxy/true | 4-0 proxy/true |
|---|---|---|---|---|---|---|
| 2013-14 | 6.79 | +0.006 | 0.141 | 78.3% | 0.5% / 5.5% | 18.4% / 12.2% |
| 2014-15 | 6.82 | +0.053 | 0.147 | 78.0% | 0.5% / 5.3% | 18.4% / 10.5% |
| 2015-16 | 6.24 | +0.079 | 0.140 | 79.0% | 0.5% / 4.8% | 17.4% / 14.8% |
| 2016-17 | 6.81 | -0.060 | 0.171 | 76.6% | 0.5% / 5.2% | 16.9% / 9.4% |
| 2017-18 | 4.20 | +0.698 | 0.144 | 79.8% | 0.8% / 5.4% | 14.5% / 8.5% |
| 2018-19 | 3.50 | +0.534 | 0.113 | 85.7% | 0.7% / 3.0% | 12.9% / 8.7% |
| 2019-20 | 3.08 | +0.794 | 0.113 | 85.6% | 0.6% / 3.1% | 12.7% / 8.5% |
| 2020-21 | 2.29 | +0.924 | 0.110 | 85.7% | 0.6% / 3.0% | 13.2% / 8.8% |
| 2021-22 | 2.73 | +0.890 | 0.106 | 86.0% | 0.7% / 3.0% | 13.7% / 9.2% |
| 2022-23 | 2.99 | +0.763 | 0.104 | 85.9% | 0.7% / 3.0% | 13.5% / 8.7% |
| 2023-24 | 2.76 | +0.812 | 0.092 | 86.4% | 0.7% / 3.0% | 13.4% / 8.7% |
| 2024-25 | 3.51 | +0.711 | 0.111 | 85.7% | 0.7% / 3.2% | 13.5% / 9.1% |

## What this means for the paper

- Shot-clock-based claims are defensible from **2018-19 onward**
  (FG% error 2.3–3.5pp across seven seasons), with the inbound-delay
  caveat stated.
- Pre-2018 seasons should be excluded, or the season-conditional reset
  fixed and the enrichment re-run, before any shot-clock claim covering
  them is made.
- Note 2015-16 ground truth looks anomalous (true 4-0 share 14.8% vs
  ~9-12% either side); worth a sanity check before citing that season.

See `ideas_for_improvement.md` items 14-18 for the fix list.
