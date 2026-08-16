# Shot-clock proxy validation — 2013-14 to 2017-18

Cross-season summary from `validate_shot_clock.py`. Ground truth is
NBA.com's tracking dashboard, which reports FGA/FGM per shot-clock
*range*. No per-shot accuracy figure is claimed anywhere.

## Headline

The proxy splits into **two regimes** at the 2018-19 rule change:

| Era | Seasons | FG% error | TVD | Agreement bound |
|---|---|---|---|---|
| Pre-2018-19 | 5 | 2.8–3.4 pp | 0.050–0.087 | 83.8–86.4% |

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
| 2013-14 | 2.90 | +0.911 | 0.070 | 84.1% | 2.6% / 5.5% | 15.3% / 12.2% |
| 2014-15 | 2.95 | +0.955 | 0.081 | 84.3% | 2.5% / 5.3% | 15.2% / 10.5% |
| 2015-16 | 2.76 | +0.908 | 0.050 | 84.9% | 2.4% / 4.8% | 14.3% / 14.8% |
| 2016-17 | 2.93 | +0.911 | 0.087 | 83.8% | 2.5% / 5.2% | 14.0% / 9.4% |
| 2017-18 | 3.36 | +0.972 | 0.070 | 86.4% | 5.5% / 5.4% | 12.1% / 8.5% |

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
