#!/usr/bin/env python3
"""
Re-run rolling temporal validation on the v5-corrected enrichment.

Extracted from NBA_Shot_Quality_Rolling_Temporal_Validation.ipynb. The logic
(features, folds, hyperparameters, leakage exclusions) is unchanged -- only the
underlying enriched data has moved from v4 to v5.

WHY ONLY THIS ONE
-----------------
Of the downstream outputs, this is the only one whose inputs actually changed.
v5 differs from v4 only for 2013-14 through 2017-18; all seven post-2018 seasons
were verified byte-identical. So:

  rolling_temporal_validation_results.csv   STALE -- every fold trains from 2014
  model_results.csv                         fine  -- 2024-25 only
  team_game_points.csv, season_standings    fine  -- 2024-25 only
  year_to_year_temporal_validation_results  fine  -- folds are 2020->2021 onward
  temporal_validation_summary.csv           fine  -- 2022-23 -> 2024-25
  model_comparison_2014_distance_vs_proxy   fine  -- reads the SAVANT file
                                                     (nba_savant_2014_*), which
                                                     v5 does not regenerate

The previous results are preserved as *_v4.csv before anything is overwritten,
so the before/after comparison survives.

Expect the early folds to move most: they train on seasons where
SHOT_CLOCK_APPROX went from noise (curve correlation ~0) to signal (~0.91), and
the model uses shot-clock features heavily.

Usage:
    python rerun_rolling_validation.py
    python rerun_rolling_validation.py --folds 2   # quick smoke test
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, brier_score_loss, log_loss,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    LIGHTGBM_AVAILABLE = False

REPO = Path(__file__).resolve().parent
ENRICHED = REPO / "enriched_data"
OUT_CSV = REPO / "rolling_temporal_validation_results.csv"
RANDOM_STATE = 42

BEST_LGBM_PARAMS = {
    "n_estimators": 563, "max_depth": 8, "learning_rate": 0.0117,
    "num_leaves": 84, "subsample": 0.668, "colsample_bytree": 0.626,
    "reg_alpha": 3.47, "reg_lambda": 4.91, "min_child_samples": 42,
    "random_state": RANDOM_STATE, "n_jobs": -1, "verbose": -1,
}
BEST_HISTGB_PARAMS = {
    "max_iter": 282, "max_depth": 10, "learning_rate": 0.0197,
    "max_leaf_nodes": 74, "min_samples_leaf": 32, "random_state": RANDOM_STATE,
}

LEAKAGE_COLUMNS = ["scoreHome", "scoreAway", "pointsTotal", "shotResult", "target"]

# Label year -> enriched file year (the notebook's off-by-one naming: label 2014
# means the 2013-14 season, stored as nbastatsv3_2013).
SEASONS = {label: ENRICHED / f"nbastatsv3_{label - 1}_enriched_shots.csv"
           for label in range(2014, 2026)}

FOLDS = [{"train_years": list(range(2014, t)), "test_year": t,
          "name": f"2014-{str(t - 1)[-2:]} -> {t}"} for t in range(2017, 2025)]

NUMERIC_FEATURES = [
    "shotDistance", "SHOT_CLOCK_APPROX", "xLegacy", "yLegacy", "period",
    "contest_score", "shotValue", "ABS_TIME",
    "shot_distance_log", "shot_distance_sq", "shot_angle_rad",
    "clock_x_distance", "clock_x_paint", "clock_x_corner3",
    "clock_x_above_break3", "rhythm_x_distance",
]
PRIOR_FEATURES = [
    "prior_fg_pct", "prior_fg3_pct", "prior_fg2_pct", "prior_shot_type_pct",
    "prior_attempts", "prior_high_volume",
    "opp_def_fg_pct", "opp_def_rim_pct", "opp_def_3pt_pct", "opp_def_mid_pct",
    "opp_def_shot_type_pct",
]
BINARY_FEATURES = [
    "is_paint", "is_midrange", "is_corner3", "is_above_break3",
    "is_late_clock", "is_early_clock", "is_heave",
    "clock_0_4", "clock_4_8", "clock_8_12", "clock_12_16", "clock_16_20",
    "clock_20_24", "is_transition_paint", "is_transition_midrange",
    "is_transition_3pt", "is_late_clock_paint", "is_late_clock_midrange",
    "is_desperation_3pt", "is_heave_shot", "is_rhythm_shot",
    "is_pullup", "is_stepback", "is_fadeaway", "is_turnaround",
    "is_driving", "is_floating", "is_cutting", "is_tip", "is_putback",
    "is_second_chance",
]
CATEGORICAL_FEATURES = ["subType", "contest_label", "teamTricode", "location"]

# Only the columns the pipeline actually needs, so 12 seasons fit in memory.
READ_COLS = [
    "gameId", "personId", "teamId", "teamTricode", "location", "period",
    "shotDistance", "shotValue", "shotResult", "xLegacy", "yLegacy",
    "ABS_TIME", "SHOT_CLOCK_APPROX", "contest_score", "contest_label",
    "actionType", "subType", "description",
]

player_hist = (pd.read_csv(REPO / "player_historical_stats.csv")
               if (REPO / "player_historical_stats.csv").exists() else None)
team_def_hist = (pd.read_csv(REPO / "team_defensive_stats.csv")
                 if (REPO / "team_defensive_stats.csv").exists() else None)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dist = pd.to_numeric(df.get("shotDistance", 0), errors="coerce").fillna(0)
    sc = pd.to_numeric(df.get("SHOT_CLOCK_APPROX", 12), errors="coerce").fillna(12)
    x_raw = pd.to_numeric(df.get("xLegacy", 0), errors="coerce").fillna(0)
    y_raw = pd.to_numeric(df.get("yLegacy", 0), errors="coerce").fillna(0)
    shot_value = pd.to_numeric(df.get("shotValue", 2), errors="coerce").fillna(2)

    df["shot_distance_log"] = np.log1p(dist)
    df["shot_distance_sq"] = dist ** 2
    df["is_paint"] = (dist <= 8).astype(int)
    df["is_midrange"] = ((dist > 8) & (dist <= 22)).astype(int)

    x_ft, y_ft = x_raw / 10.0, y_raw / 10.0
    df["shot_angle_rad"] = np.arctan2(y_ft, np.abs(x_ft))
    df["is_corner3"] = ((shot_value == 3) & (x_ft.abs() >= 22.0)
                        & (y_ft <= 5.0)).astype(int)
    df["is_above_break3"] = ((shot_value == 3) & (df["is_corner3"] == 0)).astype(int)

    df["is_late_clock"] = (sc <= 5).astype(int)
    df["is_early_clock"] = (sc >= 18).astype(int)
    df["is_heave"] = (sc <= 2).astype(int)
    for lo, hi, name in [(0, 4, "clock_0_4"), (4, 8, "clock_4_8"),
                         (8, 12, "clock_8_12"), (12, 16, "clock_12_16"),
                         (16, 20, "clock_16_20")]:
        df[name] = ((sc >= lo) & (sc < hi)).astype(int)
    df["clock_20_24"] = ((sc >= 20) & (sc <= 24)).astype(int)

    df["clock_x_distance"] = sc * dist
    df["is_transition_paint"] = ((sc >= 18) & (dist <= 8)).astype(int)
    df["is_transition_midrange"] = ((sc >= 18) & (dist > 8) & (dist <= 22)).astype(int)
    df["is_transition_3pt"] = ((sc >= 18) & (shot_value == 3)).astype(int)
    df["is_late_clock_paint"] = ((sc <= 5) & (dist <= 8)).astype(int)
    df["is_late_clock_midrange"] = ((sc <= 5) & (dist > 8) & (dist <= 22)).astype(int)
    df["is_desperation_3pt"] = ((sc <= 5) & (shot_value == 3)).astype(int)
    df["is_heave_shot"] = ((sc <= 2) & (dist >= 28)).astype(int)
    df["clock_x_paint"] = sc * df["is_paint"]
    df["clock_x_corner3"] = sc * df["is_corner3"]
    df["clock_x_above_break3"] = sc * df["is_above_break3"]
    df["is_rhythm_shot"] = ((sc >= 8) & (sc <= 16)).astype(int)
    df["rhythm_x_distance"] = df["is_rhythm_shot"] * dist

    blank = pd.Series([""] * len(df), index=df.index)
    combined = (df.get("actionType", blank).fillna("").str.lower() + " "
                + df.get("subType", blank).fillna("").str.lower() + " "
                + df.get("description", blank).fillna("").str.lower())
    df["is_pullup"] = combined.str.contains("pullup|pull-up|pull up", regex=True).astype(int)
    df["is_stepback"] = combined.str.contains("step back|stepback", regex=True).astype(int)
    df["is_fadeaway"] = combined.str.contains("fadeaway|fade away", regex=True).astype(int)
    df["is_turnaround"] = (combined.str.contains("turnaround")
                           & ~combined.str.contains("fadeaway")).astype(int)
    df["is_driving"] = combined.str.contains("driving").astype(int)
    df["is_floating"] = combined.str.contains("floating|floater|runner", regex=True).astype(int)
    df["is_cutting"] = combined.str.contains("cutting|cut", regex=True).astype(int)
    df["is_tip"] = combined.str.contains("tip").astype(int)
    df["is_putback"] = combined.str.contains("putback|put-back|put back", regex=True).astype(int)
    df["is_second_chance"] = (df["is_putback"] | df["is_tip"]).astype(int)
    return df


def merge_prior_stats(df: pd.DataFrame, prior_season_year: int) -> pd.DataFrame:
    df = df.copy()
    if player_hist is not None and "personId" in df.columns:
        ps = player_hist[player_hist["season_year"] == prior_season_year][
            ["personId", "fg_pct", "fg3_pct", "fg2_pct", "total_attempts"]].copy()
        ps.columns = ["personId", "prior_fg_pct", "prior_fg3_pct",
                      "prior_fg2_pct", "prior_attempts"]
        df = df.merge(ps, on="personId", how="left")
        defaults = {
            "prior_fg_pct": ps["prior_fg_pct"].mean() if len(ps) else 0.45,
            "prior_fg3_pct": ps["prior_fg3_pct"].mean() if len(ps) else 0.35,
            "prior_fg2_pct": ps["prior_fg2_pct"].mean() if len(ps) else 0.50,
            "prior_attempts": ps["prior_attempts"].median() if len(ps) else 200,
        }
        for col, val in defaults.items():
            df[col] = df[col].fillna(val)
        df["prior_shot_type_pct"] = np.where(df["shotValue"] == 3,
                                             df["prior_fg3_pct"], df["prior_fg2_pct"])
        df["prior_high_volume"] = (df["prior_attempts"] >= 300).astype(int)

    if team_def_hist is not None and {"teamId", "gameId"} <= set(df.columns):
        gt = df[["gameId", "teamId"]].drop_duplicates()
        gto = gt.merge(gt, on="gameId", suffixes=("", "_opp"))
        gto = gto[gto["teamId"] != gto["teamId_opp"]].rename(
            columns={"teamId_opp": "opponentId"})
        df = df.merge(gto[["gameId", "teamId", "opponentId"]],
                      on=["gameId", "teamId"], how="left")

        pd_ = team_def_hist[team_def_hist["season_year"] == prior_season_year][
            ["teamId", "opp_fg_pct_allowed", "opp_rim_fg_pct_allowed",
             "opp_3pt_pct_allowed", "opp_mid_fg_pct_allowed"]].copy()
        pd_.columns = ["opponentId", "opp_def_fg_pct", "opp_def_rim_pct",
                       "opp_def_3pt_pct", "opp_def_mid_pct"]
        df = df.merge(pd_, on="opponentId", how="left")
        for col in ["opp_def_fg_pct", "opp_def_rim_pct", "opp_def_3pt_pct",
                    "opp_def_mid_pct"]:
            df[col] = df[col].fillna(pd_[col].mean() if len(pd_) else 0.46)
        df["opp_def_shot_type_pct"] = np.where(
            df["shotValue"] == 3, df["opp_def_3pt_pct"],
            np.where(df["shotDistance"] <= 5, df["opp_def_rim_pct"],
                     df["opp_def_mid_pct"]))
    return df


def load_seasons(needed: set[int]) -> dict:
    data = {}
    for label in sorted(needed):
        path = SEASONS[label]
        if not path.exists():
            print(f"  WARNING {label}: missing {path.name}", file=sys.stderr)
            continue
        df = pd.read_csv(path, low_memory=False,
                         usecols=lambda c: c in READ_COLS)
        df["target"] = (df["shotResult"].astype(str).str.strip().str.lower()
                        == "made").astype(int)
        data[label] = engineer_features(df)
        print(f"  {label}: {len(df):,} shots, FG% {df['target'].mean():.1%}",
              flush=True)
    return data


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", type=int, default=len(FOLDS),
                    help="run only the first N folds (smoke test)")
    args = ap.parse_args(argv)

    folds = FOLDS[:args.folds]
    needed = {y for f in folds for y in f["train_years"]} | {
        f["test_year"] for f in folds}

    print(f"loading {len(needed)} seasons from {ENRICHED.name}/ ...")
    data = load_seasons(needed)

    numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    results = []
    for fold in folds:
        name, test_year = fold["name"], fold["test_year"]
        if test_year not in data or any(y not in data for y in fold["train_years"]):
            print(f"\n{name}: SKIPPED (missing seasons)", file=sys.stderr)
            continue

        t0 = time.time()
        train_df = pd.concat(
            [merge_prior_stats(data[y], y - 1) for y in fold["train_years"]],
            ignore_index=True)
        test_df = merge_prior_stats(data[test_year], test_year - 1)

        all_numeric = NUMERIC_FEATURES + PRIOR_FEATURES + BINARY_FEATURES
        num_features = [f for f in all_numeric
                        if f in train_df.columns and f not in LEAKAGE_COLUMNS]
        cat_features = [f for f in CATEGORICAL_FEATURES
                        if f in train_df.columns and f not in LEAKAGE_COLUMNS]
        all_features = num_features + cat_features
        prior_count = len([f for f in PRIOR_FEATURES if f in train_df.columns])

        preprocessor = ColumnTransformer([
            ("num", numeric_transformer, num_features),
            ("cat", categorical_transformer, cat_features)], remainder="drop")

        if LIGHTGBM_AVAILABLE:
            clf, model_name = LGBMClassifier(**BEST_LGBM_PARAMS), "LightGBM"
        else:
            clf, model_name = (HistGradientBoostingClassifier(**BEST_HISTGB_PARAMS),
                               "HistGradientBoosting")
        model = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])

        print(f"\n{name}: train {len(train_df):,} / test {len(test_df):,}, "
              f"{len(all_features)} features ({prior_count} prior)", flush=True)
        model.fit(train_df[all_features], train_df["target"])

        y_test = test_df["target"]
        y_proba = model.predict_proba(test_df[all_features])[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        best_thresh, best_acc = 0.5, accuracy_score(y_test, (y_proba >= 0.5).astype(int))
        for t in np.arange(0.4, 0.6, 0.01):
            a = accuracy_score(y_test, (y_proba >= t).astype(int))
            if a > best_acc:
                best_acc, best_thresh = a, t

        results.append({
            "fold": name, "train_years": str(fold["train_years"]),
            "test_year": test_year, "model": model_name,
            "n_train": len(train_df), "n_test": len(test_df),
            "n_features": len(all_features), "n_prior_features": prior_count,
            "auc": auc, "log_loss": log_loss(y_test, y_proba),
            "brier": brier_score_loss(y_test, y_proba),
            "accuracy": best_acc, "threshold": best_thresh,
            "train_fg_pct": train_df["target"].mean(),
            "test_fg_pct": y_test.mean(),
        })
        print(f"  AUC {auc:.4f}  acc {best_acc*100:.2f}%  "
              f"({time.time() - t0:.0f}s)", flush=True)

    if not results:
        print("no folds completed", file=sys.stderr)
        return 1

    new = pd.DataFrame(results)

    # Preserve the v4 baseline once, so before/after survives a re-run.
    baseline = OUT_CSV.with_name(OUT_CSV.stem + "_v4.csv")
    if OUT_CSV.exists() and not baseline.exists():
        old = pd.read_csv(OUT_CSV)
        old.to_csv(baseline, index=False)
        print(f"\npreserved previous results -> {baseline.name}")

    if baseline.exists():
        old = pd.read_csv(baseline)[["fold", "auc", "accuracy"]]
        cmp = new[["fold", "auc", "accuracy"]].merge(
            old, on="fold", suffixes=("_v5", "_v4"), how="left")
        cmp["auc_delta"] = cmp["auc_v5"] - cmp["auc_v4"]
        print("\nBEFORE / AFTER")
        print(cmp.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if len(results) < len(FOLDS):
        # A partial run must not overwrite a complete result set.
        partial = OUT_CSV.with_name(OUT_CSV.stem + "_partial.csv")
        new.to_csv(partial, index=False)
        print(f"\npartial run ({len(new)}/{len(FOLDS)} folds) -> {partial.name}")
        print(f"{OUT_CSV.name} left untouched; re-run without --folds to replace it")
    else:
        new.to_csv(OUT_CSV, index=False)
        print(f"\nwrote {OUT_CSV.name} ({len(new)} folds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
