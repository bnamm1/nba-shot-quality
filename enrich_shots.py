#!/usr/bin/env python3
"""
Shot enrichment with season-aware shot-clock reconstruction (v5).

Extracted from enrich_shots_nbastatsv3_full.ipynb so the pipeline can run over
many seasons without hand-editing CSV_PATH/OUTPUT_CSV per run.

WHAT CHANGED FROM v4
--------------------
v4 applied the 14-second reset unconditionally. All three of its 14s branches are
post-2018 rules:

    off_reb            offensive rebound          -> 14s from 2018-19, else 24s
    def_violation_14   kicked ball / def 3 sec    -> 14s from 2018-19, else 24s
    def_foul_14        defensive foul, offense    -> 14s from 2018-19, else 24s
                       retains possession

Validation against NBA.com ground truth (see validate_shot_clock.py) showed this
made every pre-2018 season substantially wrong: FG% error 6.2-6.8pp vs 2.3-3.5pp
after the rule change, and a FG%/shot-clock curve correlation of ~0 for 2013-14
through 2016-17. In 2015-16 this touched 25,028 shots, 12.03% of the season.

v5 derives the season from gameId and applies the era-correct reset value.

STILL KNOWN-WRONG IN v5 (deliberately unchanged, see ideas_for_improvement.md)
-----------------------------------------------------------------------------
Dead-ball resets (made_fg, final_ft) start the clock at the event timestamp, but
the real clock starts when a player touches the ball inbounds. This costs every
season roughly 4.6s on those possessions and is why no reset rule can produce a
full-clock shot. Fixing it is a separate change; v5 does not attempt it.

_abs_time() still truncates sub-second precision, so values remain integers.

Usage:
    python enrich_shots.py --seasons 2015
    python enrich_shots.py --seasons 2013 2014 2015 2016 2017
    python enrich_shots.py --seasons 2015 --out-dir enriched_data   # overwrite
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
RAW_DIR = REPO / "raw_data"
# Defaults to a NEW directory so an existing enriched_data/ is never clobbered
# before the output has been validated. Pass --out-dir enriched_data to replace.
DEFAULT_OUT_DIR = REPO / "enriched_data_v5"

REG_PERIOD_LEN = 12 * 60
OT_PERIOD_LEN = 5 * 60

# The 14-second reset rules took effect in the 2018-19 season.
FOURTEEN_SECOND_RULE_FROM = 2018


def _period_len_s(period: int) -> int:
    return REG_PERIOD_LEN if period <= 4 else OT_PERIOD_LEN


def _abs_time(period: int, clock_iso: str):
    # clock like 'PT11M43.00S'. NOTE: sub-second precision is discarded here,
    # which floors every reconstructed value. Carried over from v4 unchanged.
    m = re.match(r"PT(\d+)M(\d+)\.\d+S", str(clock_iso))
    if not m:
        return None
    rem = int(m.group(1)) * 60 + int(m.group(2))
    return sum(_period_len_s(p) for p in range(1, period)) + (
        _period_len_s(period) - rem)


def season_start_year(game_id) -> int | None:
    """Season start year from a gameId: 0021500001 -> 2015.

    Digits 3-4 hold the two-digit season code. Verified against the local data:
    every 2015-16 game reads '15'.
    """
    s = str(game_id).strip()
    if s.endswith(".0"):
        s = s[:-2]
    s = s.zfill(10)
    if len(s) < 5 or not s[3:5].isdigit():
        return None
    code = int(s[3:5])
    return 2000 + code if code < 50 else 1900 + code


def uses_14_second_rule(game_id) -> bool:
    yr = season_start_year(game_id)
    # Unknown season: assume the modern rule rather than silently reverting.
    return True if yr is None else yr >= FOURTEEN_SECOND_RULE_FROM


# --------------------------------------------------------------------------
# Event vocabularies (unchanged from v4)
# --------------------------------------------------------------------------

_FINAL_FT = {
    "free throw 1 of 1", "free throw 2 of 2", "free throw 3 of 3",
    "free throw technical", "free throw technical 2 of 2",
    "free throw clear path 2 of 2",
    "free throw flagrant 1 of 1", "free throw flagrant 2 of 2",
    "free throw flagrant 3 of 3",
}

_TURNOVER_POSSESSION = {
    "bad pass", "lost ball", "out of bounds - bad pass turnover",
    "out of bounds lost ball turnover", "offensive foul turnover",
    "offensive charge", "shot clock turnover", "traveling", "backcourt turnover",
    "step out of bounds turnover", "double dribble", "palming turnover",
    "8 second violation", "5 second violation", "illegal assist turnover",
    "excess timeout turnover", "basket from below turnover",
    "too many players turnover", "punched ball turnover", "inbound turnover",
    "illegal screen turnover", "offensive goaltending",
}

_DEFENSE_KEEPS_POSSESSION = {
    "kicked ball", "kicked ball violation", "defense 3 second",
    "defensive goaltending",
}

TEAM_NAME_TO_TRICODE = {
    "ATLANTA HAWKS": "ATL", "HAWKS": "ATL",
    "BOSTON CELTICS": "BOS", "CELTICS": "BOS",
    "BROOKLYN NETS": "BKN", "NETS": "BKN",
    "CHARLOTTE HORNETS": "CHA", "HORNETS": "CHA",
    "CHICAGO BULLS": "CHI", "BULLS": "CHI",
    "CLEVELAND CAVALIERS": "CLE", "CAVALIERS": "CLE", "CAVS": "CLE",
    "DALLAS MAVERICKS": "DAL", "MAVERICKS": "DAL", "MAVS": "DAL",
    "DENVER NUGGETS": "DEN", "NUGGETS": "DEN",
    "DETROIT PISTONS": "DET", "PISTONS": "DET",
    "GOLDEN STATE WARRIORS": "GSW", "WARRIORS": "GSW",
    "HOUSTON ROCKETS": "HOU", "ROCKETS": "HOU",
    "INDIANA PACERS": "IND", "PACERS": "IND",
    "LA CLIPPERS": "LAC", "LOS ANGELES CLIPPERS": "LAC", "CLIPPERS": "LAC",
    "LA LAKERS": "LAL", "LOS ANGELES LAKERS": "LAL", "LAKERS": "LAL",
    "MEMPHIS GRIZZLIES": "MEM", "GRIZZLIES": "MEM",
    "MIAMI HEAT": "MIA", "HEAT": "MIA",
    "MILWAUKEE BUCKS": "MIL", "BUCKS": "MIL",
    "MINNESOTA TIMBERWOLVES": "MIN", "TIMBERWOLVES": "MIN", "WOLVES": "MIN",
    "NEW ORLEANS PELICANS": "NOP", "PELICANS": "NOP",
    "NEW YORK KNICKS": "NYK", "KNICKS": "NYK",
    "OKLAHOMA CITY THUNDER": "OKC", "THUNDER": "OKC",
    "ORLANDO MAGIC": "ORL", "MAGIC": "ORL",
    "PHILADELPHIA 76ERS": "PHI", "76ERS": "PHI", "SIXERS": "PHI",
    "PHOENIX SUNS": "PHX", "SUNS": "PHX",
    "PORTLAND TRAIL BLAZERS": "POR", "TRAIL BLAZERS": "POR", "BLAZERS": "POR",
    "SACRAMENTO KINGS": "SAC", "KINGS": "SAC",
    "SAN ANTONIO SPURS": "SAS", "SPURS": "SAS",
    "TORONTO RAPTORS": "TOR", "RAPTORS": "TOR",
    "UTAH JAZZ": "UTA", "JAZZ": "UTA",
    "WASHINGTON WIZARDS": "WAS", "WIZARDS": "WAS",
}

_OFF_FOUL_KEYWORDS = [
    "offensive foul", "offensive", "charge", "charging", "illegal screen",
    "moving screen", "player control", "offensive charge",
]


# --------------------------------------------------------------------------
# Shot clock v5
# --------------------------------------------------------------------------

def compute_game_resets(g: pd.DataFrame, reset_short: int) -> tuple:
    """Build the possession-reset table for ONE game.

    v4 recomputed this for every (game, player) pair, re-filtering the whole
    season each time. It depends only on the game, so it is computed once here
    and shared across every shooter in that game -- identical output, far less
    work.

    reset_short is 14 from 2018-19 onward and 24 before, covering offensive
    rebounds and defensive violations/fouls where the offense retains the ball.
    """
    teams_in_game = [t for t in g[g["teamId"].notna()]["teamId"].unique() if t != 0]

    tricode_to_id = {}
    if "teamTricode" in g.columns:
        for tid, tri in g[["teamId", "teamTricode"]].dropna().values.tolist():
            if tid != 0 and tri:
                tricode_to_id[str(tri).upper()] = tid

    team_name_to_id = {name: tricode_to_id[tri]
                       for name, tri in TEAM_NAME_TO_TRICODE.items()
                       if tri in tricode_to_id}

    def get_other_team(team):
        for t_id in teams_in_game:
            if t_id != team:
                return t_id
        return None

    def infer_team_from_desc(desc):
        if not desc:
            return None
        desc_u = str(desc).upper()
        for tri, tid in tricode_to_id.items():
            if tri and re.search(r"\b" + re.escape(tri) + r"\b", desc_u):
                return tid
        for name, tid in team_name_to_id.items():
            if name and re.search(r"\b" + re.escape(name) + r"\b", desc_u):
                return tid
        return None

    def infer_team_from_event(team, team_tricode, desc):
        if pd.notna(team) and team != 0:
            return team
        if pd.notna(team_tricode):
            tri = str(team_tricode).upper()
            if tri in tricode_to_id:
                return tricode_to_id[tri]
        return infer_team_from_desc(desc)

    def is_team_rebound_desc(desc):
        if not desc:
            return False
        desc_l = str(desc).lower()
        if "rebound" not in desc_l:
            return False
        if "team rebound" in desc_l:
            return True
        desc_u = desc_l.upper()
        return any(tri and re.search(r"\b" + re.escape(tri) + r"\b", desc_u)
                   for tri in tricode_to_id)

    resets = []
    last_shot_team = {p: None for p in range(1, 12)}
    last_reset_by_team = {}

    def add_reset(per, t, cap, note, action_num, team_id):
        if team_id is None or pd.isna(team_id) or team_id == 0:
            return
        resets.append((per, t, cap, note, action_num, team_id))
        last_reset_by_team[(per, team_id)] = (t, cap)

    for _, ev in g.iterrows():
        per = int(ev["period"])
        t = ev["ABS_TIME"]
        aType = str(ev["actionType"]).strip().lower()
        sType = (str(ev["subType"]).strip().lower()
                 if pd.notna(ev.get("subType")) else "")
        team = ev.get("teamId", np.nan)
        team_tri = ev.get("teamTricode", np.nan)
        action_num = ev.get("actionNumber", np.nan)
        desc = str(ev.get("description", ""))
        desc_l = desc.lower()

        if pd.isna(team) or team == 0:
            if aType not in ["period"]:
                inferred = infer_team_from_event(team, team_tri, desc)
                if inferred is not None:
                    team = inferred

        if aType == "period" and sType == "start":
            last_shot_team[per] = None
            continue

        if aType == "jump ball":
            if pd.notna(team) and team != 0:
                add_reset(per, t, 24, "jumpball", action_num, team)
            continue

        if aType == "timeout":
            continue

        if aType in ["made shot", "missed shot", "free throw"]:
            if pd.notna(team) and team != 0:
                last_shot_team[per] = team

        is_team_reb = (aType != "rebound") and is_team_rebound_desc(desc)

        if aType == "rebound" or is_team_reb:
            reb_team = team
            shooter = last_shot_team.get(per)
            if pd.isna(reb_team) or reb_team == 0:
                if shooter is not None:
                    reb_team = (shooter if "offensive" in sType
                                else get_other_team(shooter))

            if "offensive" in sType:
                is_off = True
            elif "defensive" in sType:
                is_off = False
            elif pd.notna(reb_team) and pd.notna(shooter):
                is_off = (reb_team == shooter)
            else:
                is_off = False

            if pd.notna(reb_team) and reb_team != 0:
                if is_off:
                    add_reset(per, t, reset_short, "off_reb", action_num, reb_team)
                else:
                    add_reset(per, t, 24, "def_reb", action_num, reb_team)
            continue

        if aType == "made shot":
            shot_team = infer_team_from_event(team, team_tri, desc)
            other_team = get_other_team(shot_team) if shot_team is not None else None
            if other_team is not None:
                add_reset(per, t, 24, "made_fg", action_num, other_team)
            continue

        if aType == "turnover":
            off_team = (infer_team_from_event(team, team_tri, desc)
                        or last_shot_team.get(per))
            if sType in _TURNOVER_POSSESSION or sType in ("", "regular"):
                other_team = get_other_team(off_team) if off_team is not None else None
                if other_team is not None:
                    add_reset(per, t, 24, "turnover", action_num, other_team)
            continue

        if aType == "violation" and sType in _DEFENSE_KEEPS_POSSESSION:
            poss_team = (get_other_team(team) if (pd.notna(team) and team != 0)
                         else last_shot_team.get(per))
            if poss_team is not None:
                add_reset(per, t, reset_short, "def_violation", action_num, poss_team)
            continue

        if aType == "foul":
            sdesc = (sType + " " + desc_l).strip()
            if any(k in sdesc for k in _OFF_FOUL_KEYWORDS):
                offender = (infer_team_from_event(team, team_tri, desc)
                            or last_shot_team.get(per))
                other_team = get_other_team(offender) if offender is not None else None
                if other_team is not None:
                    add_reset(per, t, 24, "off_foul", action_num, other_team)
            else:
                poss_team = (get_other_team(team) if (pd.notna(team) and team != 0)
                             else last_shot_team.get(per))
                if poss_team is not None:
                    rem, last_cap = None, None
                    last = last_reset_by_team.get((per, poss_team))
                    if last is not None and pd.notna(t):
                        last_t, last_cap = last
                        if pd.notna(last_t):
                            rem = last_cap - (t - last_t)
                            if rem > last_cap or rem <= 0:
                                rem = None
                    # Pre-2018 reset_short is 24, so this always resets to 24 --
                    # which is what the pre-rule-change behaviour should be.
                    if rem is None or rem <= reset_short:
                        add_reset(per, t, reset_short, "def_foul", action_num,
                                  poss_team)
                    else:
                        cap = (int(min(last_cap, rem)) if last_cap is not None
                               else int(rem))
                        add_reset(per, t, cap, "keep_clock", action_num, poss_team)
            continue

        if aType == "free throw":
            if sType in _FINAL_FT:
                ft_team = (infer_team_from_event(team, team_tri, desc)
                           or last_shot_team.get(per))
                other_team = get_other_team(ft_team) if ft_team is not None else None
                if other_team is not None:
                    add_reset(per, t, 24, "final_ft", action_num, other_team)
            continue

    resets_df = pd.DataFrame(
        resets,
        columns=["PERIOD", "ABS_TIME", "RESET_VAL", "RESET_NOTE",
                 "ACTION_NUMBER", "POSS_TEAM"])
    resets_df = resets_df.sort_values(
        ["PERIOD", "ABS_TIME", "ACTION_NUMBER"]).reset_index(drop=True)
    return resets_df, infer_team_from_event


def attach_shot_clock(shots: pd.DataFrame, resets_df: pd.DataFrame,
                      infer_team_from_event) -> pd.DataFrame:
    """Attach SHOT_CLOCK_APPROX / SHOT_CLOCK_SOURCE to a game's shots."""
    sc_vals, sc_notes = [], []
    for _, sh in shots.iterrows():
        per = int(sh["period"])
        t = sh["ABS_TIME"]
        shot_action_num = sh.get("actionNumber", np.nan)
        shot_team = infer_team_from_event(
            sh.get("teamId", np.nan), sh.get("teamTricode", np.nan),
            sh.get("description", ""))

        prior = resets_df[
            (resets_df["PERIOD"] == per)
            & (resets_df["POSS_TEAM"] == shot_team)
            & ((resets_df["ABS_TIME"] < t)
               | ((resets_df["ABS_TIME"] == t)
                  & (resets_df["ACTION_NUMBER"] < shot_action_num)))
        ].sort_values(["ABS_TIME", "ACTION_NUMBER"])

        if prior.empty:
            period_start_time = sum(_period_len_s(p) for p in range(1, per))
            delta = t - period_start_time
            if delta <= 24:
                sc_vals.append(max(0, 24 - delta))
                sc_notes.append("period_start(24)")
            else:
                sc_vals.append(np.nan)
                sc_notes.append("no_reset_found")
        else:
            last = prior.iloc[-1]
            delta = max(0, t - int(last["ABS_TIME"]))
            cap = int(last["RESET_VAL"])
            if delta > cap:
                sc_vals.append(np.nan)
                sc_notes.append(f"stale_reset({cap})")
            else:
                sc_vals.append(max(0, cap - delta))
                sc_notes.append(f"{last['RESET_NOTE']}({cap})")

    shots = shots.copy()
    shots["SHOT_CLOCK_APPROX"] = sc_vals
    shots["SHOT_CLOCK_SOURCE"] = sc_notes
    return shots


def compute_shot_clock_v5(df: pd.DataFrame, game_id, player_id=None) -> pd.DataFrame:
    """Season-aware shot-clock reconstruction for one game.

    player_id=None returns every shooter in the game (what the season export
    wants); passing one keeps the v4 single-player behaviour.
    """
    g = df[df["gameId"] == game_id].copy()
    if g.empty:
        return g
    g = g.sort_values(["period", "actionNumber"]).reset_index(drop=True)
    g["ABS_TIME"] = g.apply(
        lambda r: _abs_time(int(r["period"]), r["clock"]), axis=1)

    reset_short = 14 if uses_14_second_rule(game_id) else 24
    resets_df, infer_team = compute_game_resets(g, reset_short)

    shots = g[g["actionType"].str.contains("Shot", case=False, na=False)]
    if player_id is not None:
        shots = shots[shots["personId"] == player_id]
    if shots.empty:
        return shots.copy()
    return attach_shot_clock(shots, resets_df, infer_team)


# --------------------------------------------------------------------------
# Contest classifier (logic unchanged from v3 -- only its shot-clock input moves)
# --------------------------------------------------------------------------

_SHOT_3PT_PAT = re.compile(
    r"\b3\s*pt\b|\b3pt\b|\b3-?point\b|\b3 pointer\b|\bthree point\b", re.IGNORECASE)
_SHOT_2PT_PAT = re.compile(
    r"\b2\s*pt\b|\b2pt\b|\b2-?point\b|\b2 pointer\b", re.IGNORECASE)
_FT_PAT = re.compile(r"\bfree throw\b", re.IGNORECASE)

_CONTESTED_LIKE = {
    "pullup jump shot", "running pull-up jump shot", "step back jump shot",
    "turnaround fadeaway shot", "fadeaway jump shot",
    "driving floating jump shot", "floating jump shot",
    "driving floating bank jump shot", "driving reverse layup shot",
    "reverse layup shot", "driving layup shot", "running layup shot",
    "tip layup shot", "putback layup shot", "driving dunk shot",
    "running dunk shot", "tip dunk shot", "putback dunk shot",
    "alley oop dunk shot", "alley oop layup shot", "turnaround hook shot",
    "driving hook shot", "hook shot", "hook bank shot",
    "turnaround bank hook shot", "turnaround fadeaway bank jump shot",
    "fadeaway bank shot", "turnaround bank shot", "jump bank shot",
}
_OPEN_LIKE = {"jump shot", "running jump shot", "spot up", "catch and shoot",
              "catch-and-shoot", "regular"}


def infer_shot_value_from_text(description, action_type, sub_type):
    text = f"{description or ''} {action_type or ''} {sub_type or ''}".lower()
    if _FT_PAT.search(text):
        return (1, False, True)
    if _SHOT_3PT_PAT.search(text):
        return (3, True, False)
    if _SHOT_2PT_PAT.search(text):
        return (2, True, False)
    fg_cues = ("made shot", "missed shot", "layup", "dunk", "jumper",
               "jump shot", "hook", "putback", "tip", "bank", "fadeaway")
    if any(cue in text for cue in fg_cues):
        return (2, True, False)
    return (np.nan, False, False)


def classify_contest_level(df: pd.DataFrame) -> pd.DataFrame:
    def nstr(x):
        return str(x).strip().lower() if pd.notna(x) else ""

    has_coords = ("xLegacy" in df.columns) and ("yLegacy" in df.columns)
    out_rows = []
    for _, r in df.iterrows():
        desc, a, s = nstr(r.get("description")), nstr(r.get("actionType")), nstr(r.get("subType"))

        shotv = r.get("shotValue", np.nan)
        if pd.isna(shotv):
            shotv, is_fg, is_ft = infer_shot_value_from_text(
                r.get("description", ""), r.get("actionType", ""),
                r.get("subType", ""))
        else:
            is_fg, is_ft = True, False

        if is_ft or not is_fg:
            new = r.copy()
            new["contest_score"] = np.nan
            new["contest_label"] = "n/a_free_throw" if is_ft else "n/a_non_fg"
            new["contest_reasons"] = "free throw" if is_ft else "not a field goal"
            out_rows.append(new)
            continue

        try:
            dist = r.get("shotDistance", r.get("SHOT_DISTANCE", np.nan))
            dist = float(dist) if pd.notna(dist) else np.nan
        except (TypeError, ValueError):
            dist = np.nan
        try:
            rx = float(r.get("xLegacy")) if has_coords else np.nan
            ry = float(r.get("yLegacy")) if has_coords else np.nan
        except (TypeError, ValueError):
            rx, ry = np.nan, np.nan

        is_3 = (shotv == 3)
        is_corner = bool(is_3 and pd.notna(rx) and pd.notna(ry)
                         and abs(rx) >= 220 and ry <= 50)
        is_above_break = bool(is_3 and not is_corner)

        score, why = 0, []
        if s in _CONTESTED_LIKE:
            score += 2; why.append("subType: contested-like")
        if s in _OPEN_LIKE:
            score -= 1; why.append("subType: open-like")

        text = f"{desc} {a}".lower()
        if any(k in text for k in ("step back", "pull-up", "pullup", "fadeaway")):
            score += 1; why.append("text: pressure-move")
        if any(k in text for k in ("catch and shoot", "catch-and-shoot",
                                   "spot up", "spot-up")):
            score -= 1; why.append("text: catch/spot")

        if pd.notna(dist):
            if dist <= 5:
                score += 1; why.append("distance: at-rim/very close")
            if 8 <= dist <= 16:
                score += 1; why.append("distance: mid-range")
            if is_above_break and dist >= 27:
                score -= 1; why.append("distance: deep above-break 3 (>=27ft)")

        if is_corner:
            why.append("coords: corner-3")
            if any(k in text for k in ("catch and shoot", "catch-and-shoot",
                                       "spot up", "spot-up")) or s in {
                    "jump shot", "spot up", "catch and shoot", "catch-and-shoot"}:
                score -= 1; why.append("corner: catch/spot")

        try:
            sc = r.get("SHOT_CLOCK_APPROX", np.nan)
            sc = float(sc) if pd.notna(sc) else np.nan
        except (TypeError, ValueError):
            sc = np.nan
        if pd.notna(sc) and sc <= 5:
            score += 1; why.append("clock: late (<=5s)")

        new = r.copy()
        new["shotValue"] = shotv
        new["contest_score"] = score
        new["contest_label"] = ("likely_contested" if score >= 2
                                else "borderline" if score == 1 else "likely_open")
        new["contest_reasons"] = ", ".join(why)
        out_rows.append(new)

    return pd.DataFrame(out_rows).reset_index(drop=True)


def enrich_game(df: pd.DataFrame, game_id, player_id=None) -> pd.DataFrame:
    shots = compute_shot_clock_v5(df, game_id, player_id)
    return classify_contest_level(shots) if not shots.empty else shots


# --------------------------------------------------------------------------
# Season export
# --------------------------------------------------------------------------

def enrich_season(year: int, out_dir: Path, flush_every: int = 50) -> Path:
    raw = RAW_DIR / f"nbastatsv3_{year}.csv"
    if not raw.exists():
        raise FileNotFoundError(f"missing raw data: {raw}")

    out_path = out_dir / f"nbastatsv3_{year}_enriched_shots.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  loading {raw.name} ...", flush=True)
    all_data = pd.read_csv(raw, low_memory=False)

    game_ids = sorted(
        all_data.loc[all_data["actionType"].str.contains("Shot", case=False,
                                                         na=False), "gameId"].unique())
    era = "14s rule" if uses_14_second_rule(game_ids[0]) else "pre-2018 (24s)"
    print(f"  {len(all_data):,} events, {len(game_ids):,} games, era: {era}")

    batch, written, n_shots = [], False, 0
    t0 = time.time()
    for i, gid in enumerate(game_ids, 1):
        enriched = enrich_game(all_data, gid)
        if not enriched.empty:
            batch.append(enriched)
            n_shots += len(enriched)
        if i % flush_every == 0 or i == len(game_ids):
            if batch:
                pd.concat(batch, ignore_index=True).to_csv(
                    out_path, mode="w" if not written else "a",
                    header=not written, index=False)
                written = True
                batch = []
            rate = i / max(time.time() - t0, 1e-9)
            eta = (len(game_ids) - i) / max(rate, 1e-9)
            print(f"    {i}/{len(game_ids)} games  {n_shots:,} shots  "
                  f"eta {eta/60:.1f}m", flush=True)

    print(f"  -> {out_path}  ({n_shots:,} shots, "
          f"{(time.time() - t0)/60:.1f} min)")
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", type=int, nargs="+", required=True,
                    help="season start years, e.g. --seasons 2013 2014 2015")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                    help=f"output directory (default: {DEFAULT_OUT_DIR.name}/ -- "
                         f"pass 'enriched_data' to replace the originals)")
    args = ap.parse_args(argv)

    ok = []
    for year in args.seasons:
        print(f"\n=== {year}-{str(year+1)[-2:]} ===")
        try:
            ok.append(enrich_season(year, args.out_dir))
        except (FileNotFoundError, KeyError) as e:
            print(f"  SKIPPED {year}: {e}", file=sys.stderr)

    print(f"\nenriched {len(ok)}/{len(args.seasons)} seasons -> {args.out_dir}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
