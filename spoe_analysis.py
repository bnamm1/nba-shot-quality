#!/usr/bin/env python3
"""
Is SPOE skill or luck? Year-over-year stability + Bernoulli uncertainty.

SPOE (Shooting Points Over Expected) = actual points - expected points, where
expected points for shot i is v_i * p_i (shot value x modelled make probability).

TWO METHODOLOGICAL FIXES vs the main modelling notebook
-------------------------------------------------------
1. NO PLAYER-SKILL FEATURES. The main notebook feeds `prior_fg_pct`,
   `prior_shot_type_pct` and `prior_attempts` to the model, so its expected
   points already encode how good the shooter is. SPOE built on that measures
   deviation from a *skill-adjusted* baseline, which removes precisely the
   signal this analysis is testing for. Here the model sees shot context only
   (where, when, what kind of shot, how contested) and never who is shooting.

2. OUT-OF-FOLD SCORING. The main notebook trains on 80% then scores 100%,
   so most shots are scored by a model that memorised them, biasing SPOE toward
   zero. Here each season is scored by a model trained on the other eleven
   (leave-one-season-out), so no shot informs its own expectation.

THE STATISTICS
--------------
Each shot is a Bernoulli trial: points_i = v_i * Bernoulli(p_i). Hence

    Var(points_i) = v_i^2 * p_i * (1 - p_i)
    Var(SPOE)     = sum_i v_i^2 * p_i * (1 - p_i)      (expectation is fixed)
    SE(SPOE)      = sqrt(Var(SPOE))

giving a Wald interval SPOE +/- 1.96*SE and z = SPOE / SE. This assumes shots
are independent given the model, which ignores hot-hand and within-game
correlation; see the caveat printed in the report.

Splitting skill from luck, two independent ways:

  (a) Year-over-year correlation of per-shot SPOE. If it is skill it persists
      across seasons; if it is variance it does not.

  (b) Empirical Bayes. Observed rate r_i has known sampling variance s_i^2 from
      the Bernoulli sum above. Method-of-moments on the between-player variance:

          tau^2 = max(0, Var(r_i) - mean(s_i^2))

      Reliability = tau^2 / (tau^2 + mean(s_i^2)) is the share of observed
      spread that is real between-player difference rather than noise, and
      B_i = tau^2 / (tau^2 + s_i^2) shrinks each player toward the league mean.

(a) and (b) are computed independently and should roughly agree -- they are a
check on each other.

Usage:
    python spoe_analysis.py
    python spoe_analysis.py --min-shots 200 --seasons 2018 2019 2020
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from lightgbm import LGBMClassifier
    HAVE_LGBM = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAVE_LGBM = False

REPO = Path(__file__).resolve().parent
ENRICHED = REPO / "enriched_data"
OUT_DIR = REPO / "spoe_analysis"
RANDOM_STATE = 42

LGBM_PARAMS = {
    "n_estimators": 563, "max_depth": 8, "learning_rate": 0.0117,
    "num_leaves": 84, "subsample": 0.668, "colsample_bytree": 0.626,
    "reg_alpha": 3.47, "reg_lambda": 4.91, "min_child_samples": 42,
    "random_state": RANDOM_STATE, "n_jobs": -1, "verbose": -1,
}

# Shot context only. personId, playerName and every prior_* player statistic are
# deliberately absent -- including them would let the model absorb the shooting
# skill this analysis exists to measure.
NUMERIC = [
    "shotDistance", "SHOT_CLOCK_APPROX", "xLegacy", "yLegacy", "period",
    "contest_score", "shotValue", "ABS_TIME",
    "shot_distance_log", "shot_distance_sq", "shot_angle_rad",
    "clock_x_distance", "clock_x_paint",
]
BINARY = [
    "is_paint", "is_midrange", "is_corner3", "is_above_break3",
    "is_late_clock", "is_early_clock", "is_heave", "is_rhythm_shot",
    "is_pullup", "is_stepback", "is_fadeaway", "is_turnaround",
    "is_driving", "is_floating", "is_cutting", "is_tip", "is_putback",
]
CATEGORICAL = ["subType", "contest_label", "location"]

READ_COLS = [
    "gameId", "personId", "playerName", "teamTricode", "location", "period",
    "shotDistance", "shotValue", "shotResult", "xLegacy", "yLegacy",
    "ABS_TIME", "SHOT_CLOCK_APPROX", "contest_score", "contest_label",
    "actionType", "subType", "description",
]

C_MAIN = "#2a78d6"
C_ALT = "#eb6834"
INK, INK2, MUTED, GRID, AXIS, SURFACE = ("#0b0b0b", "#52514e", "#898781",
                                         "#e1e0d9", "#c3c2b7", "#fcfcfb")
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.edgecolor": AXIS, "grid.color": GRID,
    "axes.grid": True, "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def season_label(year: int) -> str:
    return f"{year}-{str(year + 1)[-2:]}"


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    dist = pd.to_numeric(df["shotDistance"], errors="coerce").fillna(0)
    sc = pd.to_numeric(df["SHOT_CLOCK_APPROX"], errors="coerce").fillna(12)
    x = pd.to_numeric(df["xLegacy"], errors="coerce").fillna(0) / 10.0
    y = pd.to_numeric(df["yLegacy"], errors="coerce").fillna(0) / 10.0
    sv = pd.to_numeric(df["shotValue"], errors="coerce").fillna(2)

    df["shot_distance_log"] = np.log1p(dist)
    df["shot_distance_sq"] = dist ** 2
    df["shot_angle_rad"] = np.arctan2(y, np.abs(x))
    df["is_paint"] = (dist <= 8).astype(int)
    df["is_midrange"] = ((dist > 8) & (dist <= 22)).astype(int)
    df["is_corner3"] = ((sv == 3) & (x.abs() >= 22.0) & (y <= 5.0)).astype(int)
    df["is_above_break3"] = ((sv == 3) & (df["is_corner3"] == 0)).astype(int)
    df["is_late_clock"] = (sc <= 5).astype(int)
    df["is_early_clock"] = (sc >= 18).astype(int)
    df["is_heave"] = (sc <= 2).astype(int)
    df["is_rhythm_shot"] = ((sc >= 8) & (sc <= 16)).astype(int)
    df["clock_x_distance"] = sc * dist
    df["clock_x_paint"] = sc * df["is_paint"]

    combined = (df["actionType"].fillna("").str.lower() + " "
                + df["subType"].fillna("").str.lower() + " "
                + df["description"].fillna("").str.lower())
    for name, pat in [("is_pullup", "pullup|pull-up|pull up"),
                      ("is_stepback", "step back|stepback"),
                      ("is_fadeaway", "fadeaway|fade away"),
                      ("is_driving", "driving"),
                      ("is_floating", "floating|floater|runner"),
                      ("is_cutting", "cutting|cut"), ("is_tip", "tip"),
                      ("is_putback", "putback|put-back|put back")]:
        df[name] = combined.str.contains(pat, regex=True).astype(int)
    df["is_turnaround"] = (combined.str.contains("turnaround")
                           & ~combined.str.contains("fadeaway")).astype(int)
    return df


def load_all(years) -> pd.DataFrame:
    frames = []
    for y in years:
        path = ENRICHED / f"nbastatsv3_{y}_enriched_shots.csv"
        if not path.exists():
            print(f"  WARNING: missing {path.name}", file=sys.stderr)
            continue
        df = pd.read_csv(path, low_memory=False, usecols=lambda c: c in READ_COLS)
        df["season"] = season_label(y)
        df["season_year"] = y
        df["made"] = (df["shotResult"].astype(str).str.strip().str.lower()
                      == "made").astype(int)
        df = df[pd.to_numeric(df["shotValue"], errors="coerce").isin([2, 3])]
        frames.append(engineer(df))
        print(f"  {season_label(y)}: {len(frames[-1]):,} shots", flush=True)
    return pd.concat(frames, ignore_index=True)


def oof_predict(df: pd.DataFrame) -> np.ndarray:
    """Leave-one-season-out p(make). No shot contributes to its own expectation."""
    feats = [c for c in NUMERIC + BINARY if c in df.columns]
    cats = [c for c in CATEGORICAL if c in df.columns]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), feats),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore",
                                                sparse_output=False))]), cats),
    ], remainder="drop")

    p = np.full(len(df), np.nan)
    for yr in sorted(df["season_year"].unique()):
        t0 = time.time()
        te = df["season_year"] == yr
        tr = ~te
        clf = (LGBMClassifier(**LGBM_PARAMS) if HAVE_LGBM
               else HistGradientBoostingClassifier(random_state=RANDOM_STATE))
        model = Pipeline([("pre", pre), ("clf", clf)])
        model.fit(df.loc[tr, feats + cats], df.loc[tr, "made"])
        p[te.values] = model.predict_proba(df.loc[te, feats + cats])[:, 1]
        print(f"    {season_label(yr)}: trained on {tr.sum():,}, "
              f"scored {te.sum():,} ({time.time() - t0:.0f}s)", flush=True)
    return p


def player_seasons(df: pd.DataFrame, min_shots: int) -> pd.DataFrame:
    """Per player-season SPOE with Bernoulli standard errors."""
    v = pd.to_numeric(df["shotValue"], errors="coerce")
    df = df.assign(
        exp_points=df["p_make"] * v,
        act_points=df["made"] * v,
        # Var(v * Bernoulli(p)) = v^2 p (1-p)
        shot_var=(v ** 2) * df["p_make"] * (1 - df["p_make"]),
    )
    g = (df.groupby(["season_year", "season", "personId", "playerName"])
         .agg(n_shots=("made", "size"), actual=("act_points", "sum"),
              expected=("exp_points", "sum"), var=("shot_var", "sum"))
         .reset_index())
    g = g[g["n_shots"] >= min_shots].copy()
    g["spoe"] = g["actual"] - g["expected"]
    g["se"] = np.sqrt(g["var"])
    g["spoe_per_shot"] = g["spoe"] / g["n_shots"]
    g["se_per_shot"] = g["se"] / g["n_shots"]
    g["z"] = g["spoe"] / g["se"]
    g["ci_lo"] = g["spoe"] - 1.96 * g["se"]
    g["ci_hi"] = g["spoe"] + 1.96 * g["se"]
    g["significant"] = np.where(g["ci_lo"] > 0, "above",
                                np.where(g["ci_hi"] < 0, "below", "not distinguishable"))
    return g.sort_values(["season_year", "spoe"], ascending=[True, False])


def year_over_year(ps: pd.DataFrame) -> pd.DataFrame:
    """Pair each player's season X with X+1."""
    a = ps[["personId", "playerName", "season_year", "spoe_per_shot",
            "n_shots", "se_per_shot"]].copy()
    b = a.copy()
    b["season_year"] = b["season_year"] - 1
    m = a.merge(b, on=["personId", "season_year"], suffixes=("_x", "_x1"))
    m["pair"] = (m["season_year"].map(season_label) + " -> "
                 + (m["season_year"] + 1).map(season_label))
    return m


def empirical_bayes(ps: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Method-of-moments shrinkage of per-shot SPOE toward the league mean.

    Observed r_i = theta_i + noise, Var(noise) = s_i^2 known from the Bernoulli
    sum. Total observed variance = tau^2 + mean(s_i^2), so tau^2 is what remains
    after removing sampling noise -- the real between-player spread.
    """
    r = ps["spoe_per_shot"].to_numpy()
    s2 = ps["se_per_shot"].to_numpy() ** 2
    mu = float(np.average(r, weights=ps["n_shots"]))
    observed_var = float(np.var(r, ddof=1))
    mean_s2 = float(np.mean(s2))
    tau2 = max(0.0, observed_var - mean_s2)

    B = tau2 / (tau2 + s2) if tau2 > 0 else np.zeros_like(s2)
    out = ps.copy()
    out["shrinkage"] = B
    out["spoe_per_shot_shrunk"] = mu + B * (r - mu)
    out["spoe_shrunk_total"] = out["spoe_per_shot_shrunk"] * out["n_shots"]

    stats = {
        "league_mean_rate": mu,
        "observed_var": observed_var,
        "mean_sampling_var": mean_s2,
        "tau2": tau2,
        "tau": float(np.sqrt(tau2)),
        "reliability": tau2 / (tau2 + mean_s2) if (tau2 + mean_s2) > 0 else 0.0,
        "n_player_seasons": len(ps),
    }
    return out, stats


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def _clean(ax, title, subtitle=None, xlabel=None, ylabel=None):
    ax.set_title(title, color=INK, fontsize=13, fontweight="600", loc="left",
                 pad=24 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=MUTED,
                fontsize=9.5, va="bottom")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_yoy(yoy: pd.DataFrame, r: float, path: Path):
    fig, ax = plt.subplots(figsize=(7.5, 7))
    x = yoy["spoe_per_shot_x"] * 100
    y = yoy["spoe_per_shot_x1"] * 100
    ax.axhline(0, color=AXIS, lw=1)
    ax.axvline(0, color=AXIS, lw=1)
    ax.scatter(x, y, s=14, alpha=0.35, color=C_MAIN, edgecolors="none")
    if len(yoy) > 2:
        b, a = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, a + b * xs, color=C_ALT, lw=2,
                label=f"fit: slope {b:.2f}")
        ax.legend(frameon=False, fontsize=10, loc="upper left")
    lim = max(abs(x).max(), abs(y).max()) * 1.05
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    _clean(ax, "Does SPOE persist from one season to the next?",
           f"Each point is one player in consecutive seasons.  r = {r:.3f}  "
           f"(n = {len(yoy):,} pairs)",
           "SPOE per 100 shots, season X (points)",
           "SPOE per 100 shots, season X+1 (points)")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_reliability_by_volume(tiers: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = np.arange(len(tiers))
    ax.bar(xs, tiers["r"], color=C_MAIN, width=0.6)
    for i, (rv, n) in enumerate(zip(tiers["r"], tiers["n_pairs"])):
        ax.annotate(f"r = {rv:.3f}\nn = {n:,}", (i, rv),
                    textcoords="offset points", xytext=(0, 5), ha="center",
                    fontsize=9, color=INK2)
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_xticks(xs, tiers["tier"])
    ax.margins(y=0.20)
    ax.grid(axis="x", visible=False)
    _clean(ax, "Year-over-year SPOE correlation by shot volume",
           "More shots means less sampling noise, so a real skill signal should strengthen.",
           "Minimum shots in both seasons", "Correlation r")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_caterpillar(ps: pd.DataFrame, season: str, path: Path, top: int = 30):
    sub = ps[ps["season"] == season].nlargest(top, "n_shots").sort_values("spoe")
    fig, ax = plt.subplots(figsize=(8.5, 9))
    ys = np.arange(len(sub))
    colors = [C_ALT if lo > 0 or hi < 0 else MUTED
              for lo, hi in zip(sub["ci_lo"], sub["ci_hi"])]
    ax.hlines(ys, sub["ci_lo"], sub["ci_hi"], color=colors, lw=2.2)
    ax.plot(sub["spoe"], ys, "o", ms=6, color=INK, mec=SURFACE, mew=1.5,
            linestyle="none")
    ax.axvline(0, color=AXIS, lw=1.4)
    ax.set_yticks(ys, [f"{n}  ({s:,})" for n, s in
                       zip(sub["playerName"], sub["n_shots"])], fontsize=8.5)
    ax.grid(axis="y", visible=False)
    _clean(ax, f"SPOE with 95% intervals — {season}",
           "Highest-volume shooters. Orange = interval excludes zero.",
           "Points over expected (season total)")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", type=int, nargs="+",
                    default=list(range(2013, 2025)))
    ap.add_argument("--min-shots", type=int, default=100,
                    help="minimum shots for a player-season (default 100)")
    ap.add_argument("--write-player-csv", action="store_true",
                    help="also regenerate player_performance_vs_expected.csv for "
                         "the latest season using this corrected methodology "
                         "(context-only model, out-of-fold scoring), keeping the "
                         "original columns and adding uncertainty")
    args = ap.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"loading {len(args.seasons)} seasons ...")
    df = load_all(args.seasons)
    print(f"  total {len(df):,} shots\n")

    print("leave-one-season-out scoring (no shot informs its own expectation):")
    df["p_make"] = oof_predict(df)

    ps = player_seasons(df, args.min_shots)
    print(f"\n{len(ps):,} player-seasons with >= {args.min_shots} shots")

    yoy = year_over_year(ps)
    r_all = (yoy["spoe_per_shot_x"].corr(yoy["spoe_per_shot_x1"])
             if len(yoy) > 2 else float("nan"))

    tiers = []
    # Skip tiers at or below --min-shots: they would duplicate the base set.
    for lo in [v for v in (100, 200, 400, 600, 800) if v >= args.min_shots]:
        s = yoy[(yoy["n_shots_x"] >= lo) & (yoy["n_shots_x1"] >= lo)]
        if len(s) > 10:
            tiers.append({"tier": f">={lo}", "n_pairs": len(s),
                          "r": s["spoe_per_shot_x"].corr(s["spoe_per_shot_x1"])})
    tiers = pd.DataFrame(tiers)

    ps_eb, eb = empirical_bayes(ps)

    ps_eb.to_csv(OUT_DIR / "player_season_spoe.csv", index=False)
    yoy.to_csv(OUT_DIR / "spoe_year_over_year.csv", index=False)
    if not tiers.empty:
        tiers.to_csv(OUT_DIR / "spoe_reliability_by_volume.csv", index=False)

    fig_yoy(yoy, r_all, OUT_DIR / "fig1_spoe_year_over_year.png")
    if not tiers.empty:
        fig_reliability_by_volume(tiers, OUT_DIR / "fig2_reliability_by_volume.png")
    latest = ps["season"].max()
    fig_caterpillar(ps, latest, OUT_DIR / f"fig3_spoe_intervals_{latest}.png")

    sig = ps["significant"].value_counts()
    n_above = int(sig.get("above", 0)); n_below = int(sig.get("below", 0))
    n_null = int(sig.get("not distinguishable", 0))

    lines = [
        "# Is SPOE skill or luck?",
        "",
        "Generated by `spoe_analysis.py`. Expected points come from a shot-context",
        "model that never sees player identity or prior player performance, scored",
        "leave-one-season-out so no shot informs its own expectation.",
        "",
        "## Answer",
        "",
        f"- Year-over-year correlation of per-shot SPOE: **r = {r_all:.3f}** "
        f"({len(yoy):,} player pairs)",
        f"- Share of observed SPOE spread that is real between-player difference "
        f"(empirical-Bayes reliability): **{eb['reliability'] * 100:.1f}%**",
        f"- Between-player SD of true SPOE rate: **{eb['tau'] * 100:.2f}** "
        f"points per 100 shots ({eb['tau']:.5f} per shot)",
        "",
        "These two are computed independently -- a correlation and a "
        "variance decomposition --",
        "so their agreement (or not) is a check on both.",
        "",
        "## Year-over-year correlation by volume",
        "",
        "| Min shots in both seasons | Pairs | r |",
        "|---|---|---|",
    ]
    for _, t in tiers.iterrows():
        lines.append(f"| {t['tier']} | {t['n_pairs']:,} | {t['r']:.3f} |")

    lines += [
        "",
        "Sampling noise falls as volume rises, so a real skill signal should",
        "strengthen down this table. A flat or falling column is evidence against",
        "skill.",
        "",
        "## Uncertainty on individual players",
        "",
        "Each shot is a Bernoulli trial, so a player's season SPOE has variance",
        "`sum v_i^2 p_i (1 - p_i)`. Using 95% Wald intervals across "
        f"{len(ps):,} player-seasons",
        f"(>= {args.min_shots} shots):",
        "",
        f"- **{n_above:,}** are significantly above expectation",
        f"- **{n_below:,}** are significantly below",
        f"- **{n_null:,}** ({n_null / max(len(ps), 1) * 100:.1f}%) are **not "
        f"distinguishable from zero**",
        "",
        "## Empirical Bayes",
        "",
        f"- League mean SPOE rate: {eb['league_mean_rate']:.5f} points/shot",
        f"- Observed variance of player rates: {eb['observed_var']:.6f}",
        f"- Mean sampling variance: {eb['mean_sampling_var']:.6f}",
        f"- Between-player variance tau^2: {eb['tau2']:.6f}",
        f"- Reliability tau^2 / (tau^2 + s^2): **{eb['reliability']:.3f}**",
        "",
        "`player_season_spoe.csv` carries a shrunk estimate per player-season;",
        "shrinking toward the league mean in proportion to each player's noise is",
        "a better forecast of next season than the raw figure.",
        "",
        "## Caveats",
        "",
        "- Variance assumes shots are independent given the model. Within-game and",
        "  hot-hand correlation would widen the true intervals, so these are a",
        "  lower bound on uncertainty.",
        "- SPOE inherits any bias in the expectation model. Systematic",
        "  under-modelling of shot difficulty for a player type shows up as SPOE.",
        "- Players are included at different volumes; the empirical-Bayes",
        "  reliability is an average across a heterogeneous set.",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n")

    if args.write_player_csv:
        # Same columns as the original so existing consumers keep working, plus
        # the uncertainty the original could not express.
        latest_ps = ps_eb[ps_eb["season"] == latest].copy()
        out = pd.DataFrame({
            "personId": latest_ps["personId"],
            "playerName": latest_ps["playerName"],
            "total_shots": latest_ps["n_shots"],
            "actual_points": latest_ps["actual"],
            "exp_points": latest_ps["expected"],
            "points_diff": latest_ps["spoe"],
            "pct_diff": latest_ps["spoe"] / latest_ps["expected"] * 100,
            "se": latest_ps["se"],
            "z": latest_ps["z"],
            "ci_lo": latest_ps["ci_lo"],
            "ci_hi": latest_ps["ci_hi"],
            "significant": latest_ps["significant"],
            "shrinkage": latest_ps["shrinkage"],
            "points_diff_shrunk": latest_ps["spoe_shrunk_total"],
        }).sort_values("points_diff", ascending=False)

        target = REPO / "player_performance_vs_expected.csv"
        backup = REPO / "player_performance_vs_expected_v4_insample.csv"
        if target.exists() and not backup.exists():
            target.rename(backup)
            print(f"\npreserved previous version -> {backup.name}")
        out.to_csv(target, index=False)
        print(f"wrote {target.name} ({len(out):,} players, {latest})")

    print(f"\nyear-over-year r = {r_all:.3f} ({len(yoy):,} pairs)")
    print(f"reliability      = {eb['reliability']:.3f}  "
          f"(tau = {eb['tau'] * 100:.2f} pts per 100 shots)")
    print(f"significant      : {n_above:,} above / {n_below:,} below / "
          f"{n_null:,} indistinguishable")
    print(f"-> {OUT_DIR}/summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
