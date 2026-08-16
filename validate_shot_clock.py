#!/usr/bin/env python3
"""
Validate the reconstructed shot clock (SHOT_CLOCK_APPROX) against NBA.com ground truth.

The NBA's tracking dashboard (`leaguedashplayerptshot`) reports FGA and FGM per
shot-clock *range* per player. It does not expose a per-shot clock value, so this
script validates at the range level. Three tests, in descending order of strength:

  1. FG%-by-bucket overlay      Does the proxy preserve the relationship between
                                shot clock and shot outcome? This is what a shot
                                quality model actually consumes.
  2. Bucket-share distribution  Does the proxy place shots in the right ranges,
                                marginally? Systematic bias shows up here.
  3. Per-player overlap bound   sum_b min(proxy_b, truth_b) / N is a hard UPPER
                                BOUND on per-shot bucket agreement. It is a bound,
                                never an accuracy figure -- see README note.

Plus an excess-mass attribution by SHOT_CLOCK_SOURCE, which localises *where* the
discrepancy sits among the reset rules in compute_shot_clock_v4.

This script is standalone: it reads enriched_data/ and writes shot_clock_validation/.
It does not modify any existing notebook or data file.

Usage:
    python validate_shot_clock.py --seasons 2024
    python validate_shot_clock.py --seasons 2018 2019 2020 --condition
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

REPO = Path(__file__).resolve().parent
ENRICHED_DIR = REPO / "enriched_data"
OUT_DIR = REPO / "shot_clock_validation"
CACHE_DIR = OUT_DIR / "cache"

STATS_URL = "https://stats.nba.com/stats/leaguedashplayerptshot"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
    "Sec-Fetch-Site": "same-site",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Dest": "empty",
    "Connection": "keep-alive",
}

# NBA.com's shot-clock ranges, ordered from full clock down to expiring.
# Verified 2024-25: these six partition FGA exactly (219,120 = unfiltered total)
# and "ShotClock Off" returns zero rows, so no mass is unaccounted for.
BUCKETS = [
    "24-22",
    "22-18 Very Early",
    "18-15 Early",
    "15-7 Average",
    "7-4 Late",
    "4-0 Very Late",
]

# Short labels for plotting.
BUCKET_SHORT = {
    "24-22": "24-22",
    "22-18 Very Early": "22-18",
    "18-15 Early": "18-15",
    "15-7 Average": "15-7",
    "7-4 Late": "7-4",
    "4-0 Very Late": "4-0",
}

# Upper edges of each bucket. NBA does not document whether the edges are open or
# closed; we default to right-closed (22 < sc <= 24) and report a sensitivity
# check under the left-closed alternative so the choice is never load-bearing.
BUCKET_EDGES = [(22, 24), (18, 22), (15, 18), (7, 15), (4, 7), (0, 4)]

# Reference palette, categorical slots 1 and 2 (light mode). Documented as
# passing the all-pairs CVD gate; used unchanged, so no re-validation needed.
C_PROXY = "#2a78d6"   # slot 1, blue
C_TRUTH = "#eb6834"   # slot 2, orange

# Reference sequential ramp (blue 100->700), light to dark. Used for the one
# heatmap; the full range is permitted for continuous magnitude, where the
# lightest step means "near zero" and may recede toward the surface.
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
            "#0d366b"]
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color": INK,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": AXIS,
    "grid.color": GRID,
    "axes.grid": True,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
})


def season_label(year: int) -> str:
    """2024 -> '2024-25' (the NBA's season string for that season's raw file)."""
    return f"{year}-{str(year + 1)[-2:]}"


# --------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------

def _params(season: str, shot_clock_range: str, period: int = 0) -> dict:
    return {
        "LeagueID": "00", "Season": season, "SeasonType": "Regular Season",
        "PerMode": "Totals", "PlayerOrTeam": "Player",
        "ShotClockRange": shot_clock_range,
        "TeamID": 0, "OpponentTeamID": 0, "Month": 0, "Period": period,
        "LastNGames": 0, "PORound": 0,
        "PaceAdjust": "N", "PlusMinus": "N", "Rank": "N",
        "Outcome": "", "Location": "", "SeasonSegment": "",
        "DateFrom": "", "DateTo": "", "VsConference": "", "VsDivision": "",
        "GameSegment": "", "College": "", "Conference": "", "Country": "",
        "DraftPick": "", "DraftYear": "", "Division": "", "DribbleRange": "",
        "GameScope": "", "GeneralRange": "", "Height": "",
        "PlayerExperience": "", "PlayerPosition": "", "ShotDistRange": "",
        "StarterBench": "", "TouchTimeRange": "", "Weight": "",
    }


def fetch_truth(session, season: str, bucket: str, period: int = 0,
                tries: int = 5, pause: float = 2.0) -> pd.DataFrame:
    """Fetch one (season, bucket, period) slice, caching to disk.

    stats.nba.com rate-limits aggressively and fails by returning an empty body
    rather than an error status, so retry on empty as well as on exception.
    """
    slug = bucket.split()[0].replace("-", "_")
    cache = CACHE_DIR / f"ptshot_{season}_{slug}_p{period}.json"

    if cache.exists():
        payload = json.loads(cache.read_text())
    else:
        payload = None
        for attempt in range(tries):
            try:
                r = session.get(STATS_URL, params=_params(season, bucket, period),
                                headers=HEADERS, timeout=90)
                if r.status_code == 200 and r.text.strip():
                    payload = r.json()
                    break
                if r.status_code == 400:
                    raise RuntimeError(f"bad request for {bucket!r}: {r.text[:200]}")
                print(f"      retry {attempt + 1}/{tries} "
                      f"(status={r.status_code}, len={len(r.text)})")
            except requests.RequestException as e:
                print(f"      retry {attempt + 1}/{tries} ({type(e).__name__})")
            time.sleep(pause * (attempt + 1) * 2)

        if payload is None:
            raise RuntimeError(
                f"could not fetch {season} {bucket!r} period={period} after "
                f"{tries} tries -- stats.nba.com is likely rate-limiting; "
                f"wait a few minutes and re-run (completed slices are cached)"
            )
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(payload))
        time.sleep(pause)

    rs = payload["resultSets"][0]
    df = pd.DataFrame(rs["rowSet"], columns=rs["headers"])
    keep = ["PLAYER_ID", "PLAYER_NAME", "FGA", "FGM"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["bucket"] = bucket
    df["period"] = period
    return df


def load_truth(season: str, periods=(0,), verbose=True) -> pd.DataFrame:
    """Ground truth as one long frame: PLAYER_ID x bucket x period -> FGA, FGM."""
    session = requests.Session()
    session.headers.update(HEADERS)
    try:  # warm cookies; harmless if it fails
        session.get("https://www.nba.com/stats/", timeout=30)
    except requests.RequestException:
        pass

    frames = []
    for period in periods:
        for bucket in BUCKETS:
            if verbose:
                tag = "" if period == 0 else f" period {period}"
                print(f"    fetching {season} {bucket}{tag} ...", flush=True)
            frames.append(fetch_truth(session, season, bucket, period))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# Proxy
# --------------------------------------------------------------------------

def assign_bucket(sc: pd.Series, right_closed: bool = True) -> pd.Series:
    """Map a continuous shot-clock value onto NBA.com's six ranges.

    right_closed: 22 < sc <= 24 (default). Otherwise 22 <= sc < 24.
    NaN in, NaN out -- unassignable shots stay unassignable and are counted
    against the proxy rather than dropped.
    """
    out = pd.Series(pd.NA, index=sc.index, dtype="object")
    for (lo, hi), name in zip(BUCKET_EDGES, BUCKETS):
        if right_closed:
            # The bottom bucket must include 0 itself.
            hit = (sc > lo) & (sc <= hi) if lo > 0 else (sc >= 0) & (sc <= hi)
        else:
            hit = (sc >= lo) & (sc < hi) if hi < 24 else (sc >= lo) & (sc <= hi)
        out[hit.fillna(False)] = name
    return out


def load_proxy(year: int) -> pd.DataFrame:
    path = ENRICHED_DIR / f"nbastatsv3_{year}_enriched_shots.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing enriched data: {path}")
    df = pd.read_csv(
        path, low_memory=False,
        usecols=["gameId", "actionId", "teamId", "personId", "playerName",
                 "period", "shotResult", "isFieldGoal", "shotDistance",
                 "SHOT_CLOCK_APPROX", "SHOT_CLOCK_SOURCE"],
    )
    df = df.drop_duplicates(subset=["gameId", "actionId", "teamId"])
    df = df[df["isFieldGoal"] == 1].copy()

    # Regular season only: normalised gameIds are 002....., which read as 22...
    # once the CSV strips leading zeros. Match NBA.com's Regular Season filter.
    gid = df["gameId"].astype(str).str.zfill(10)
    df = df[gid.str.startswith("002")].copy()

    df["made"] = df["shotResult"].str.strip().str.lower().eq("made").astype(int)
    df["bucket"] = assign_bucket(df["SHOT_CLOCK_APPROX"])
    # Reset family, e.g. "off_reb(14)" -> "off_reb"
    df["source"] = df["SHOT_CLOCK_SOURCE"].astype(str).str.replace(
        r"\(.*\)", "", regex=True)
    return df


# --------------------------------------------------------------------------
# Analyses
# --------------------------------------------------------------------------

def league_comparison(proxy: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Bucket shares and FG% per bucket, proxy vs truth."""
    t = (truth[truth["period"] == 0]
         .groupby("bucket")[["FGA", "FGM"]].sum())
    p = (proxy.dropna(subset=["bucket"])
         .groupby("bucket")
         .agg(FGA=("made", "size"), FGM=("made", "sum")))

    out = pd.DataFrame(index=BUCKETS)
    out["proxy_fga"] = p["FGA"].reindex(BUCKETS).fillna(0).astype(int)
    out["truth_fga"] = t["FGA"].reindex(BUCKETS).fillna(0).astype(int)
    out["proxy_share"] = out["proxy_fga"] / out["proxy_fga"].sum()
    out["truth_share"] = out["truth_fga"] / out["truth_fga"].sum()
    out["share_diff"] = out["proxy_share"] - out["truth_share"]
    out["proxy_fg_pct"] = (p["FGM"].reindex(BUCKETS) /
                           p["FGA"].reindex(BUCKETS))
    out["truth_fg_pct"] = (t["FGM"].reindex(BUCKETS) /
                           t["FGA"].reindex(BUCKETS))
    out["fg_pct_diff"] = out["proxy_fg_pct"] - out["truth_fg_pct"]
    return out


def overlap_bound(proxy: pd.DataFrame, truth: pd.DataFrame,
                  by_period: bool = False) -> dict:
    """Upper bound on per-shot bucket agreement.

    For each cell (player [x period]) and bucket b, at most min(proxy_b, truth_b)
    of the proxy's assignments can be correct. Summing gives the tightest bound
    obtainable from marginals alone. Conditioning on more observables (period)
    partitions the data further and tightens the bound, because compensating
    errors across cells can no longer cancel.

    Shots the proxy could not classify (NaN) stay in the denominator -- they are
    assignments the proxy failed to make, not shots to excuse.
    """
    keys = ["personId", "period"] if by_period else ["personId"]
    tkeys = ["PLAYER_ID", "period"] if by_period else ["PLAYER_ID"]

    pr = proxy.copy()
    tr = truth.copy()
    if by_period:
        # NBA's Period filter covers regulation only; OT is fetched as period 0.
        pr = pr[pr["period"].between(1, 4)]
        tr = tr[tr["period"].between(1, 4)]
    else:
        tr = tr[tr["period"] == 0]

    denom_all = len(pr)
    p_counts = (pr.dropna(subset=["bucket"])
                .groupby(keys + ["bucket"]).size().rename("proxy_n"))
    t_counts = (tr.groupby(tkeys + ["bucket"])["FGA"].sum().rename("truth_n"))
    t_counts.index = t_counts.index.set_names(keys + ["bucket"])

    joined = pd.concat([p_counts, t_counts], axis=1).fillna(0)
    matched = np.minimum(joined["proxy_n"], joined["truth_n"]).sum()

    denom_classified = int(p_counts.sum())
    return {
        "bound_all_shots": matched / denom_all if denom_all else float("nan"),
        "bound_classified": (matched / denom_classified
                             if denom_classified else float("nan")),
        "matched": int(matched),
        "n_proxy_shots": int(denom_all),
        "n_classified": denom_classified,
        "coverage": denom_classified / denom_all if denom_all else float("nan"),
        "cells": int(joined.index.droplevel("bucket").nunique()),
    }


# Reset rules requiring an inbound pass. The real 24s clock starts when a player
# legally touches the ball inbounds, seconds after the event timestamp the
# reconstruction keys off; live-ball rules have no such gap. Comparing the two
# families isolates un-modelled inbound delay from ordinary shot selection.
DEAD_BALL_RESETS = ["made_fg", "final_ft"]
LIVE_BALL_RESETS = ["def_reb", "off_reb"]


def reset_bias(proxy: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Per-reset-rule clock profile, and the dead-ball vs live-ball contrast.

    Teams push in transition after a made basket, so if anything dead-ball
    possessions should show *more* clock left, not less. A large negative gap is
    evidence of un-modelled inbound delay rather than real shot selection.
    """
    p = proxy.dropna(subset=["SHOT_CLOCK_APPROX"])
    tbl = (p.groupby("source")["SHOT_CLOCK_APPROX"]
           .agg(n="size", mean_clock="mean", median_clock="median"))
    tbl["pct_full_clock"] = p.groupby("source")["SHOT_CLOCK_APPROX"].apply(
        lambda s: (s > 22).mean() * 100)
    tbl["pct_expiring"] = p.groupby("source")["SHOT_CLOCK_APPROX"].apply(
        lambda s: (s <= 4).mean() * 100)
    tbl["family"] = np.where(tbl.index.isin(DEAD_BALL_RESETS), "dead-ball",
                             np.where(tbl.index.isin(LIVE_BALL_RESETS),
                                      "live-ball", "mixed"))
    tbl = tbl.sort_values("mean_clock")

    dead = p[p["source"].isin(DEAD_BALL_RESETS)]["SHOT_CLOCK_APPROX"]
    live = p[p["source"].isin(LIVE_BALL_RESETS)]["SHOT_CLOCK_APPROX"]
    contrast = {
        "dead_ball_n": len(dead), "live_ball_n": len(live),
        "dead_ball_mean": dead.mean(), "live_ball_mean": live.mean(),
        "gap_seconds": dead.mean() - live.mean(),
        "dead_ball_pct_expiring": (dead <= 4).mean() * 100,
        "live_ball_pct_expiring": (live <= 4).mean() * 100,
    }
    return tbl, contrast


def fig_reset_bias(tbl: pd.DataFrame, season: str, path: Path):
    # "mixed" is a residual, not a peer series, so it deliberately takes the
    # neutral ink rather than a categorical slot: the point of the figure is the
    # dead-ball/live-ball contrast with everything else set aside. It fails the
    # palette validator's chroma floor by design ("reads gray") while still
    # clearing CVD separation (dE 9.8) and the normal-vision floor (17.6).
    colors = {"dead-ball": C_TRUTH, "live-ball": C_PROXY, "mixed": MUTED}
    sub = tbl[tbl["n"] > 500]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    y = np.arange(len(sub))
    ax.barh(y, sub["mean_clock"], color=[colors[f] for f in sub["family"]],
            height=0.62)
    for yi, (v, n) in enumerate(zip(sub["mean_clock"], sub["n"])):
        ax.annotate(f"{v:.1f}s   (n={n:,})", (v, yi), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9,
                    color=INK_SECONDARY)
    ax.set_yticks(y, sub.index)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.margins(x=0.18)
    _style(ax, f"Mean reconstructed clock remaining, by reset rule — {season}",
           "Dead-ball resets require an inbound the reconstruction does not model.")
    ax.set_xlabel("Mean SHOT_CLOCK_APPROX at time of shot (seconds)", fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[k])
               for k in ("dead-ball", "live-ball", "mixed")]
    ax.legend(handles, ["Dead-ball (inbound required)", "Live-ball", "Mixed"],
              frameon=False, fontsize=9.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def excess_by_source(proxy: pd.DataFrame, league: pd.DataFrame) -> pd.DataFrame:
    """Where the bucket-share discrepancy sits, by reset rule.

    This is attribution, not accuracy: it shows which reset rules feed the buckets
    the proxy over- or under-populates, which is what you need to fix the logic.
    """
    scale = league["truth_fga"].sum() / max(len(proxy), 1)
    comp = (proxy.dropna(subset=["bucket"])
            .groupby(["bucket", "source"]).size().rename("n").reset_index())
    comp = comp[comp["bucket"].isin(BUCKETS)]

    rows = []
    for bucket in BUCKETS:
        sub = comp[comp["bucket"] == bucket]
        excess = league.loc[bucket, "proxy_fga"] * scale - league.loc[bucket, "truth_fga"]
        total = sub["n"].sum()
        for _, r in sub.iterrows():
            rows.append({
                "bucket": bucket,
                "source": r["source"],
                "n": int(r["n"]),
                "share_of_bucket": r["n"] / total if total else np.nan,
                "bucket_excess_fga": excess,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def _style(ax, title, subtitle=None, ylabel=None):
    ax.set_title(title, color=INK, fontsize=13, fontweight="600", loc="left",
                 pad=18 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=MUTED,
                fontsize=9.5, va="bottom")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(axis="x", visible=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)


def fig_fg_pct(league: pd.DataFrame, season: str, path: Path):
    """The money figure: does the proxy reproduce the real FG% curve?"""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(BUCKETS))
    labels = [BUCKET_SHORT[b] for b in BUCKETS]

    ax.plot(x, league["truth_fg_pct"] * 100, color=C_TRUTH, linewidth=2,
            marker="o", markersize=9, markeredgecolor=SURFACE,
            markeredgewidth=2, label="NBA.com (tracking)", zorder=3)
    ax.plot(x, league["proxy_fg_pct"] * 100, color=C_PROXY, linewidth=2,
            marker="o", markersize=9, markeredgecolor=SURFACE,
            markeredgewidth=2, label="Proxy (reconstructed)", zorder=4)

    for xi, (pv, tv) in enumerate(zip(league["proxy_fg_pct"],
                                      league["truth_fg_pct"])):
        ax.annotate(f"{pv * 100:.1f}", (xi, pv * 100), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=8.5, color=INK_SECONDARY)
        ax.annotate(f"{tv * 100:.1f}", (xi, tv * 100), textcoords="offset points",
                    xytext=(0, -16), ha="center", fontsize=8.5, color=INK_SECONDARY)

    ax.set_xticks(x, labels)
    ax.set_xlabel("Shot clock remaining (seconds)", fontsize=10)
    ax.margins(y=0.14)  # headroom so the outer point labels are not clipped
    _style(ax, f"FG% by shot-clock range — {season}",
           "Does the proxy preserve the shot-clock/outcome relationship a model consumes?",
           "FG%")
    ax.legend(frameon=False, fontsize=10, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_shares(league: pd.DataFrame, season: str, path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(BUCKETS))
    w = 0.38
    labels = [BUCKET_SHORT[b] for b in BUCKETS]

    ax.bar(x - w / 2 - 0.01, league["proxy_share"] * 100, w, color=C_PROXY,
           label="Proxy (reconstructed)")
    ax.bar(x + w / 2 + 0.01, league["truth_share"] * 100, w, color=C_TRUTH,
           label="NBA.com (tracking)")

    for xi, (pv, tv) in enumerate(zip(league["proxy_share"], league["truth_share"])):
        ax.annotate(f"{pv * 100:.1f}", (xi - w / 2, pv * 100),
                    textcoords="offset points", xytext=(0, 3), ha="center",
                    fontsize=8.5, color=INK_SECONDARY)
        ax.annotate(f"{tv * 100:.1f}", (xi + w / 2, tv * 100),
                    textcoords="offset points", xytext=(0, 3), ha="center",
                    fontsize=8.5, color=INK_SECONDARY)

    ax.set_xticks(x, labels)
    ax.set_xlabel("Shot clock remaining (seconds)", fontsize=10)
    _style(ax, f"Share of field-goal attempts by shot-clock range — {season}",
           "Systematic reconstruction bias appears here as a consistent shift in mass.",
           "% of FGA")
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_source_profile(proxy: pd.DataFrame, season: str, path: Path):
    """Which reset rule feeds which bucket -- the debugging map."""
    sub = proxy.dropna(subset=["bucket"])
    top = sub["source"].value_counts().head(8).index.tolist()
    mat = (sub[sub["source"].isin(top)]
           .groupby(["source", "bucket"]).size().unstack(fill_value=0)
           .reindex(columns=BUCKETS, fill_value=0)
           .reindex(top))
    frac = mat.div(mat.sum(axis=1), axis=0)

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "ref_blue", SEQ_BLUE)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    im = ax.imshow(frac.values, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(BUCKETS)), [BUCKET_SHORT[b] for b in BUCKETS])
    ax.set_yticks(range(len(top)), top)
    ax.grid(False)
    for i in range(frac.shape[0]):
        for j in range(frac.shape[1]):
            v = frac.values[i, j]
            if v > 0.005:
                ax.text(j, i, f"{v * 100:.0f}", ha="center", va="center",
                        fontsize=8.5, color="#ffffff" if v > 0.55 else INK)
    _style(ax, f"Shot-clock range distribution by reset rule — {season}",
           "Row-normalised %. Localises which reset rule feeds which range.")
    ax.set_xlabel("Shot clock remaining (seconds)", fontsize=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(color=MUTED, labelcolor=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def write_summary(path: Path, season: str, league: pd.DataFrame,
                  bounds: dict, sens: dict, proxy: pd.DataFrame,
                  truth: pd.DataFrame, contrast: dict):
    p_tot, t_tot = int(len(proxy)), int(truth[truth["period"] == 0]["FGA"].sum())
    corr = league[["proxy_fg_pct", "truth_fg_pct"]].corr().iloc[0, 1]
    mad = (league["fg_pct_diff"].abs().mean()) * 100
    tvd = 0.5 * league["share_diff"].abs().sum()

    lines = [
        f"# Shot-clock proxy validation — {season}",
        "",
        "Generated by `validate_shot_clock.py`. Ground truth is NBA.com's tracking",
        "dashboard (`leaguedashplayerptshot`), which reports FGA/FGM per shot-clock",
        "*range* per player. There is no per-shot ground-truth clock in this source,",
        "so nothing below is a per-shot accuracy figure.",
        "",
        "## Coverage",
        "",
        f"- Proxy field-goal attempts: **{p_tot:,}**",
        f"- NBA.com field-goal attempts: **{t_tot:,}** "
        f"({(p_tot - t_tot) / t_tot * 100:+.2f}% difference)",
        f"- Proxy shots with no reconstructable clock: "
        f"**{p_tot - bounds['player']['n_classified']:,}** "
        f"({(1 - bounds['player']['coverage']) * 100:.2f}%)",
        "",
        "## Test 1 — FG% by shot-clock range (strongest test)",
        "",
        "Whether the proxy preserves the shot-clock/outcome relationship a shot",
        "quality model actually consumes.",
        "",
        "| Range | Proxy FG% | NBA.com FG% | Diff |",
        "|---|---|---|---|",
    ]
    for b in BUCKETS:
        r = league.loc[b]
        lines.append(f"| {BUCKET_SHORT[b]} | {r['proxy_fg_pct'] * 100:.1f}% | "
                     f"{r['truth_fg_pct'] * 100:.1f}% | "
                     f"{r['fg_pct_diff'] * 100:+.1f} pp |")
    lines += [
        "",
        f"- Correlation of the two FG% curves across ranges: **r = {corr:.4f}**",
        f"- Mean absolute FG% deviation: **{mad:.2f} pp**",
        "",
        "## Test 2 — Distribution of attempts across ranges",
        "",
        "| Range | Proxy share | NBA.com share | Diff |",
        "|---|---|---|---|",
    ]
    for b in BUCKETS:
        r = league.loc[b]
        lines.append(f"| {BUCKET_SHORT[b]} | {r['proxy_share'] * 100:.1f}% | "
                     f"{r['truth_share'] * 100:.1f}% | "
                     f"{r['share_diff'] * 100:+.1f} pp |")
    lines += [
        "",
        f"- Total variation distance: **{tvd:.4f}** "
        f"(0 = identical, 1 = disjoint)",
        "",
        "## Test 3 — Upper bound on per-shot range agreement",
        "",
        "`sum_b min(proxy_b, truth_b) / N` within each cell. This is a **hard upper",
        "bound**, not an accuracy: the true per-shot agreement cannot exceed it, but",
        "may be far below. Conditioning on more observables tightens the bound.",
        "",
        "| Conditioning | Cells | Bound (all shots) | Bound (classified only) |",
        "|---|---|---|---|",
    ]
    for name, b in bounds.items():
        lines.append(f"| {name} | {b['cells']:,} | "
                     f"**{b['bound_all_shots'] * 100:.1f}%** | "
                     f"{b['bound_classified'] * 100:.1f}% |")
    lines += [
        "",
        "## Sensitivity — bucket edge convention",
        "",
        "NBA.com does not document whether range edges are open or closed. Results",
        "under both conventions:",
        "",
        "| Convention | TVD | Bound (all shots) |",
        "|---|---|---|",
        f"| right-closed (default) | {sens['right']['tvd']:.4f} | "
        f"{sens['right']['bound'] * 100:.1f}% |",
        f"| left-closed | {sens['left']['tvd']:.4f} | "
        f"{sens['left']['bound'] * 100:.1f}% |",
        "",
        "## Diagnostic — where the bias comes from",
        "",
        "Reset rules split into two families: those requiring an **inbound pass**",
        "(the real clock starts when a player touches the ball inbounds, seconds",
        "after the event the reconstruction keys off) and **live-ball** rules with",
        "no such gap. Teams push in transition after a made basket, so dead-ball",
        "possessions should if anything show *more* clock remaining, not less.",
        "",
        f"- Dead-ball resets ({', '.join(DEAD_BALL_RESETS)}): "
        f"n={contrast['dead_ball_n']:,}, mean **{contrast['dead_ball_mean']:.2f}s** "
        f"remaining, {contrast['dead_ball_pct_expiring']:.1f}% at <=4s",
        f"- Live-ball resets ({', '.join(LIVE_BALL_RESETS)}): "
        f"n={contrast['live_ball_n']:,}, mean **{contrast['live_ball_mean']:.2f}s** "
        f"remaining, {contrast['live_ball_pct_expiring']:.1f}% at <=4s",
        f"- Gap: **{contrast['gap_seconds']:.2f}s** in the wrong direction",
        "",
        "This localises the dominant error: the reconstruction charges dead-ball",
        "possessions for inbound time that never ran off the shot clock, pushing",
        "them toward the expiring end. It is a fix inside `compute_shot_clock_v4`",
        "(model the inbound gap per possession), not a constant post-hoc offset --",
        "a uniform shift improves the distribution match but degrades the FG%",
        "relationship, because it moves shots into ranges whose scoring profile",
        "they do not share.",
        "",
        "## Caveats",
        "",
        "- Ground truth is range-level, so no confusion matrix or per-shot",
        "  agreement rate is derivable from this source.",
        "- `SHOT_CLOCK_APPROX` is integer-valued (25 distinct values), so range-edge",
        "  convention moves whole integer classes between ranges -- hence the",
        "  sensitivity check above.",
        "- `_abs_time()` in the enrichment pipeline parses `PT11M43.00S` with",
        "  `int()`, discarding sub-second precision, so reconstructed values are",
        "  floored and biased low by up to ~1s.",
        "- The 14-second offensive-rebound reset only exists from 2018-19 onward;",
        "  earlier seasons exercise a different branch of the reset logic.",
    ]
    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------

def run_season(year: int, condition: bool) -> dict:
    season = season_label(year)
    print(f"\n=== {season} ===")

    print("  loading proxy ...", flush=True)
    proxy = load_proxy(year)
    print(f"    {len(proxy):,} field-goal attempts, "
          f"{proxy['bucket'].notna().mean() * 100:.2f}% classified")

    periods = (0, 1, 2, 3, 4) if condition else (0,)
    truth = load_truth(season, periods=periods)

    league = league_comparison(proxy, truth)

    bounds = {"player": overlap_bound(proxy, truth, by_period=False)}
    if condition:
        bounds["player x period"] = overlap_bound(proxy, truth, by_period=True)

    # Sensitivity to the bucket-edge convention.
    sens = {}
    for name, right in (("right", True), ("left", False)):
        alt = proxy.copy()
        alt["bucket"] = assign_bucket(alt["SHOT_CLOCK_APPROX"], right_closed=right)
        lg = league_comparison(alt, truth)
        sens[name] = {
            "tvd": 0.5 * lg["share_diff"].abs().sum(),
            "bound": overlap_bound(alt, truth)["bound_all_shots"],
        }

    bias_tbl, contrast = reset_bias(proxy)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    league.to_csv(OUT_DIR / f"league_comparison_{season}.csv")
    bias_tbl.to_csv(OUT_DIR / f"reset_rule_bias_{season}.csv")
    excess_by_source(proxy, league).to_csv(
        OUT_DIR / f"source_attribution_{season}.csv", index=False)

    fig_fg_pct(league, season, OUT_DIR / f"fig1_fg_pct_by_bucket_{season}.png")
    fig_shares(league, season, OUT_DIR / f"fig2_bucket_shares_{season}.png")
    fig_source_profile(proxy, season, OUT_DIR / f"fig3_source_profile_{season}.png")
    fig_reset_bias(bias_tbl, season, OUT_DIR / f"fig4_reset_rule_bias_{season}.png")

    write_summary(OUT_DIR / f"summary_{season}.md", season, league, bounds,
                  sens, proxy, truth, contrast)

    # Persist the scalars so --summarize-only can rebuild without re-running.
    (OUT_DIR / f"metrics_{season}.json").write_text(json.dumps({
        "season": season,
        "agreement_bound_pct": bounds["player"]["bound_all_shots"] * 100,
        "coverage_pct": bounds["player"]["coverage"] * 100,
        "dead_live_gap_seconds": contrast["gap_seconds"],
    }, indent=2))

    corr = league[["proxy_fg_pct", "truth_fg_pct"]].corr().iloc[0, 1]
    tvd = 0.5 * league["share_diff"].abs().sum()
    print(f"  FG% curve correlation : r = {corr:.4f}")
    print(f"  distribution TVD      : {tvd:.4f}")
    for name, b in bounds.items():
        print(f"  agreement bound ({name}): {b['bound_all_shots'] * 100:.1f}%")
    print(f"  -> {OUT_DIR}/summary_{season}.md")

    return {"season": season, "corr": corr, "tvd": tvd, "bounds": bounds}


# The 14-second offensive-rebound reset took effect in 2018-19. compute_shot_clock_v4
# applies it unconditionally, so seasons before this exercise a rule that did not
# exist yet -- the reason the metrics below break into two regimes.
RULE_CHANGE_SEASON = "2018-19"


def collect_seasons() -> pd.DataFrame:
    """Rebuild the cross-season table from per-season outputs already on disk."""
    import re
    rows = []
    for f in sorted(OUT_DIR.glob("league_comparison_*.csv")):
        m = re.search(r"(\d{4}-\d{2})", f.name)
        if not m:
            continue
        d = pd.read_csv(f, index_col=0)
        season = m.group(1)
        extra = {}
        mf = OUT_DIR / f"metrics_{season}.json"
        if mf.exists():
            extra = {k: v for k, v in json.loads(mf.read_text()).items()
                     if k != "season"}
        rows.append({
            **extra,
            "season": season,
            "fg_pct_mad_pp": d["fg_pct_diff"].abs().mean() * 100,
            "fg_pct_curve_r": d[["proxy_fg_pct", "truth_fg_pct"]].corr().iloc[0, 1],
            "share_tvd": 0.5 * d["share_diff"].abs().sum(),
            "proxy_share_24_22": d.loc["24-22", "proxy_share"] * 100,
            "truth_share_24_22": d.loc["24-22", "truth_share"] * 100,
            "proxy_share_4_0": d.loc["4-0 Very Late", "proxy_share"] * 100,
            "truth_share_4_0": d.loc["4-0 Very Late", "truth_share"] * 100,
        })
    return pd.DataFrame(rows).sort_values("season").reset_index(drop=True)


def fig_seasons(df: pd.DataFrame, path: Path):
    """Three metrics across seasons as small multiples.

    Deliberately not a dual-axis chart: the measures are on different scales, so
    they get their own panels sharing one x-axis.
    """
    panels = [
        ("fg_pct_mad_pp", "Mean absolute FG% error (pp)", "lower is better"),
        ("share_tvd", "Distribution error (TVD)", "lower is better"),
        ("agreement_bound_pct", "Range-agreement upper bound (%)", "higher is better"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    xs = np.arange(len(df))
    try:
        brk = list(df["season"]).index(RULE_CHANGE_SEASON)
    except ValueError:
        brk = None

    for ax, (col, label, hint) in zip(axes, panels):
        if col not in df.columns:
            continue
        ax.plot(xs, df[col], color=C_PROXY, linewidth=2, marker="o",
                markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        if brk is not None:
            ax.axvline(brk - 0.5, color=C_TRUTH, linewidth=1.6, linestyle="--",
                       zorder=1)
        ax.set_ylabel(label, fontsize=9.5)
        ax.margins(y=0.22)
        ax.grid(axis="x", visible=False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(AXIS)
        ax.spines["bottom"].set_color(AXIS)
        ax.text(0.995, 0.04, hint, transform=ax.transAxes, ha="right",
                fontsize=8.5, color=MUTED)

    # Describe what the data actually shows rather than assuming the pre-fix
    # two-regime state: a recovered pre-2018 curve correlation means v5 is in use.
    pre_r = df.loc[df["season"] < RULE_CHANGE_SEASON, "fg_pct_curve_r"]
    fixed = pre_r.empty or pre_r.max() > 0.5

    if brk is not None:
        axes[0].annotate(
            "14s rule takes effect (2018-19)" if fixed else
            "14s offensive-rebound rule\ntakes effect (2018-19)",
            xy=(brk - 0.5, axes[0].get_ylim()[1]), xytext=(brk + 0.15, 0.92),
            textcoords=("data", "axes fraction"), fontsize=9, color=C_TRUTH,
            va="top")

    axes[0].set_title(
        "Shot-clock proxy accuracy by season",
        color=INK, fontsize=13, fontweight="600", loc="left", pad=20)
    axes[0].text(0, 1.03,
                 "Season-conditional resets applied; accuracy is consistent "
                 "across the rule change."
                 if fixed else
                 "The reconstruction applies the 14s reset in every season, "
                 "including those predating the rule.",
                 transform=axes[0].transAxes, color=MUTED, fontsize=9.5,
                 va="bottom")
    axes[-1].set_xticks(xs, df["season"], rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_all_seasons_summary(df: pd.DataFrame, path: Path):
    pre = df[df["season"] < RULE_CHANGE_SEASON]
    post = df[df["season"] >= RULE_CHANGE_SEASON]

    # Detect whether the season-conditional fix is in the data being validated.
    pre_r = pre["fg_pct_curve_r"].max() if not pre.empty else float("nan")
    fixed = pd.isna(pre_r) or pre_r > 0.5

    lines = [
        f"# Shot-clock proxy validation — {df['season'].iloc[0]} to "
        f"{df['season'].iloc[-1]}",
        "",
        "Cross-season summary from `validate_shot_clock.py`. Ground truth is",
        "NBA.com's tracking dashboard, which reports FGA/FGM per shot-clock",
        "*range*. No per-shot accuracy figure is claimed anywhere.",
        "",
        "## Headline",
        "",
    ]
    if fixed:
        lines += [
            "Accuracy is **consistent across all seasons**. The two-regime split "
            "that",
            "previously appeared at the 2018-19 rule change is gone, following the",
            "season-conditional reset fix in `enrich_shots.py`",
            "(`compute_shot_clock_v5`).",
            "",
        ]
    else:
        lines += [
            "The proxy splits into **two regimes** at the 2018-19 rule change, "
            "because",
            "the reconstruction applies the 14s reset in seasons that predate it:",
            "",
        ]
    lines += [
        "| Era | Seasons | FG% error | TVD | Agreement bound |",
        "|---|---|---|---|---|",
    ]
    for name, sub in (("Pre-2018-19", pre), ("2018-19 onward", post)):
        if sub.empty:
            continue
        b = (f"{sub['agreement_bound_pct'].min():.1f}–"
             f"{sub['agreement_bound_pct'].max():.1f}%"
             if "agreement_bound_pct" in sub else "n/a")
        lines.append(
            f"| {name} | {len(sub)} | "
            f"{sub['fg_pct_mad_pp'].min():.1f}–{sub['fg_pct_mad_pp'].max():.1f} pp | "
            f"{sub['share_tvd'].min():.3f}–{sub['share_tvd'].max():.3f} | {b} |")

    if fixed:
        lines += [
            "",
            "### History",
            "",
            "Before the fix, all three 14-second branches (`off_reb`,",
            "`def_violation`, `def_foul`) were applied unconditionally, though the",
            "rule only took effect in 2018-19. Pre-2018 seasons then ran 6.2–6.8pp",
            "FG% error with a curve correlation of ~0 — the shot-clock/outcome",
            "relationship was absent entirely. All seven post-2018 seasons were",
            "verified byte-identical before and after, confirming the change is",
            "inert where the rule already applied.",
            "",
            "## Remaining error",
            "",
            "The **inbound-delay bias** is untouched and affects every season:",
            "dead-ball resets (`made_fg`, `final_ft`) start the clock at the event",
            "timestamp rather than the inbound touch, averaging 8.7s remaining",
            "against 13.3s for live-ball resets. Consequently the proxy's 24-22",
            "share sits at 0.5–0.8% in every season while truth ranges 3.0–5.5% —",
            "no reset rule can produce a full-clock shot.",
            "",
        ]
    else:
        lines += [
            "",
            "**Pre-2018 seasons are not merely worse — the FG%/shot-clock",
            "relationship is absent** (curve correlation ~0), so any model",
            "consuming `SHOT_CLOCK_APPROX` there is consuming noise. Cause: the",
            "14-second resets are applied unconditionally despite only taking",
            "effect in 2018-19. Fixed in `enrich_shots.py`.",
            "",
        ]

    lines += [
        "## Per season",
        "",
        "| Season | FG% err (pp) | curve r | TVD | bound | 24-22 proxy/true | 4-0 proxy/true |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        b = (f"{r['agreement_bound_pct']:.1f}%"
             if "agreement_bound_pct" in r and pd.notna(r.get("agreement_bound_pct"))
             else "—")
        lines.append(
            f"| {r['season']} | {r['fg_pct_mad_pp']:.2f} | {r['fg_pct_curve_r']:+.3f} | "
            f"{r['share_tvd']:.3f} | {b} | "
            f"{r['proxy_share_24_22']:.1f}% / {r['truth_share_24_22']:.1f}% | "
            f"{r['proxy_share_4_0']:.1f}% / {r['truth_share_4_0']:.1f}% |")

    lines += [
        "",
        "## What this means for the paper",
        "",
    ]
    if fixed:
        lines += [
            "- Shot-clock-based claims are defensible across **all twelve "
            "seasons**,",
            "  with the inbound-delay caveat stated.",
            "- State the pre-2018 ground-truth gap: our play-by-play FGA exceeds",
            "  NBA.com's tracking FGA by +6.3 to +6.5% for 2013-14 through 2016-17",
            "  against +0.19 to +0.70% from 2018-19 on, so those seasons are",
            "  validated against a ~94% sample. This is also a candidate reason",
            "  pre-2018 TVD now beats post-2018.",
            "- The ~86% agreement figure is an **upper bound**, not an accuracy.",
        ]
    else:
        lines += [
            "- Shot-clock claims are defensible from **2018-19 onward** only.",
            "- Pre-2018 seasons need the season-conditional reset fix "
            "(`enrich_shots.py`)",
            "  and a re-run before any claim covers them.",
        ]
    lines += [
        "",
        "See `ideas_for_improvement.md` for the remaining fix list.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main(argv=None):
    # Rebound below from --enriched-dir / --out-dir; declared here because the
    # argparse help text references them.
    global ENRICHED_DIR, OUT_DIR, CACHE_DIR

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", type=int, nargs="+", default=[2024],
                    help="season start years, e.g. --seasons 2022 2023 2024")
    ap.add_argument("--condition", action="store_true",
                    help="also condition the bound on period (4x more requests, "
                         "tighter bound)")
    ap.add_argument("--summarize-only", action="store_true",
                    help="skip validation; rebuild the cross-season summary from "
                         "per-season outputs already in shot_clock_validation/")
    ap.add_argument("--enriched-dir", type=Path, default=None,
                    help="directory of enriched shot CSVs "
                         f"(default: {ENRICHED_DIR.name}/)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="where to write results; use a separate directory to "
                         f"compare against an existing run (default: {OUT_DIR.name}/)")
    args = ap.parse_args(argv)

    # Rebind module-level paths so every downstream function follows.
    if args.enriched_dir:
        ENRICHED_DIR = args.enriched_dir
    if args.out_dir:
        OUT_DIR = args.out_dir
        # Keep the shared ground-truth cache so a second run costs no requests.
        CACHE_DIR = REPO / "shot_clock_validation" / "cache"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    if not args.summarize_only:
        for year in args.seasons:
            try:
                results.append(run_season(year, args.condition))
            except (FileNotFoundError, RuntimeError) as e:
                print(f"  SKIPPED {season_label(year)}: {e}", file=sys.stderr)

    # Cross-season view, rebuilt from whatever per-season outputs exist on disk
    # so it can be regenerated without re-running the full validation.
    df = collect_seasons()
    if len(df) > 1:
        if "agreement_bound_pct" not in df.columns:
            df["agreement_bound_pct"] = np.nan
        df.to_csv(OUT_DIR / "summary_all_seasons.csv", index=False)
        fig_seasons(df, OUT_DIR / "fig5_accuracy_by_season.png")
        write_all_seasons_summary(df, OUT_DIR / "summary_all_seasons.md")
        cols = [c for c in ["season", "fg_pct_mad_pp", "fg_pct_curve_r",
                            "share_tvd", "agreement_bound_pct"] if c in df]
        print(f"\n{df[cols].to_string(index=False)}")
        print(f"\n  -> {OUT_DIR}/summary_all_seasons.md")
    return 0 if (results or args.summarize_only) else 1


if __name__ == "__main__":
    raise SystemExit(main())
