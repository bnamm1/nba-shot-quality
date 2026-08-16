#!/usr/bin/env python3
"""
Promote enriched_data_v5/ to be the canonical enriched_data/.

Dry-run by default. Nothing is moved, deleted, or staged unless --apply is given,
and even then the result is left STAGED but UNCOMMITTED so it can be reviewed.

WHY THIS EXISTS
---------------
Carrying both directories means ~1.3GB in LFS against a 1GB free tier, and leaves
a split-brain dataset where some seasons come from v4 and some from v5. This
promotes one canonical set.

WHAT IT PROTECTS
----------------
enriched_data/ also holds nba_savant_2013/2014_enriched_shots_v2.csv, produced by
a completely different pipeline (enrich_savant_shots_v2.ipynb). v5 does NOT
regenerate those. They are explicitly preserved; a naive "rm enriched_data/*.csv"
would destroy them with no way to regenerate from this repo's tooling.

WHAT IT DOES NOT DO
-------------------
LFS objects already committed stay in git history. This frees working-tree space
and future push/pull bandwidth, but does not shrink what is already stored. Only
a history rewrite would do that, which is not worth it here.

Usage:
    python swap_to_v5.py             # report what would happen
    python swap_to_v5.py --apply     # do it, leaving changes staged
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
SRC = REPO / "enriched_data_v5"
DST = REPO / "enriched_data"

EXPECTED_SEASONS = list(range(2013, 2025))  # 2013-14 .. 2024-25

# v5 should not change how many shots exist, only their values. Anything beyond
# this is treated as a truncated / still-being-written file, not a real change.
ROW_COUNT_TOLERANCE = 0.01

# Produced by a different pipeline; v5 cannot regenerate these.
PROTECTED = ["nba_savant_2013_enriched_shots_v2.csv",
             "nba_savant_2014_enriched_shots_v2.csv"]


def count_rows(path: Path) -> int:
    """Line count minus header, read in blocks to stay cheap on 50MB files."""
    n = 0
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            n += chunk.count(b"\n")
    return max(n - 1, 0)


def header_of(path: Path) -> str:
    with path.open("r") as f:
        return f.readline().strip()


def looks_like_v5(path: Path, sample_lines: int = 200_000) -> bool | None:
    """v5 renamed def_foul_14 -> def_foul / def_violation_14 -> def_violation.

    Returns True/False, or None if neither marker appears in the sample.
    """
    old = new = 0
    with path.open("r") as f:
        f.readline()
        for i, line in enumerate(f):
            if i >= sample_lines:
                break
            if "def_foul_14(" in line or "def_violation_14(" in line:
                old += 1
            elif "def_foul(" in line or "def_violation(" in line:
                new += 1
    if old == 0 and new == 0:
        return None
    return new > old


def preflight() -> tuple[list, list]:
    """Returns (moves, problems). Any problem aborts the swap."""
    moves, problems = [], []

    if not SRC.is_dir():
        return [], [f"missing source directory: {SRC}"]

    for name in PROTECTED:
        if not (DST / name).exists():
            problems.append(f"protected file already missing: {name}")

    for year in EXPECTED_SEASONS:
        fname = f"nbastatsv3_{year}_enriched_shots.csv"
        src, dst = SRC / fname, DST / fname

        if not src.exists():
            problems.append(f"{year}: not yet enriched in {SRC.name}/ "
                            f"-- run enrich_shots.py --seasons {year}")
            continue
        if src.stat().st_size == 0:
            problems.append(f"{year}: v5 file is empty")
            continue

        v5_rows = count_rows(src)
        note = ""
        if dst.exists():
            v4_rows = count_rows(dst)
            if header_of(src) != header_of(dst):
                problems.append(f"{year}: column header differs from v4")
                continue
            if v5_rows != v4_rows:
                delta = abs(v5_rows - v4_rows) / max(v4_rows, 1)
                # A large gap almost always means the file is still being
                # written by enrich_shots.py, which appends in batches. Refuse
                # rather than promote a truncated season.
                if delta > ROW_COUNT_TOLERANCE:
                    problems.append(
                        f"{year}: row count {v4_rows:,} -> {v5_rows:,} "
                        f"({delta * 100:.1f}% off) -- enrichment still running, "
                        f"or the file is truncated")
                    continue
                note = f"row count {v4_rows:,} -> {v5_rows:,}  <-- CHANGED"
            else:
                note = f"{v5_rows:,} rows (unchanged)"
        else:
            note = f"{v5_rows:,} rows (no v4 counterpart)"

        prov = looks_like_v5(src)
        if prov is False:
            problems.append(f"{year}: v5 file still contains v4-era "
                            f"def_foul_14 markers -- is it really v5?")
            continue
        if prov is None:
            note += "  [provenance unverifiable: no foul/violation markers]"

        moves.append((src, dst, note))

    return moves, problems


def git(*args, check=True):
    return subprocess.run(["git", "-C", str(REPO), *args],
                          capture_output=True, text=True, check=check)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true",
                    help="actually perform the swap (default is a dry run)")
    args = ap.parse_args(argv)

    moves, problems = preflight()

    print(f"protected (kept as-is): {', '.join(PROTECTED)}\n")
    print(f"{len(moves)}/{len(EXPECTED_SEASONS)} seasons ready to promote:")
    for src, dst, note in moves:
        print(f"  {src.name:42s} {note}")

    if problems:
        print(f"\n{len(problems)} problem(s) -- nothing will be changed:")
        for p in problems:
            print(f"  ! {p}")
        return 1

    if not args.apply:
        print("\nDRY RUN. Re-run with --apply to perform the swap.")
        print("Changes will be left staged but uncommitted for review.")
        return 0

    print("\napplying ...")
    for src, dst, _ in moves:
        os.replace(src, dst)          # atomic within the same filesystem
        print(f"  moved {src.name}")

    # Stage: .gitattributes already matches enriched_data/*.csv, so the LFS
    # filter applies on add.
    git("add", "enriched_data")
    r = git("status", "--porcelain", "enriched_data")
    print(f"\nstaged:\n{r.stdout.rstrip()}")

    # Confirm the staged blobs really are LFS pointers, not raw CSV.
    bad = []
    for _, dst, _ in moves:
        rel = dst.relative_to(REPO)
        out = git("cat-file", "-p", f":{rel}", check=False).stdout[:40]
        if not out.startswith("version https://git-lfs"):
            bad.append(str(rel))
    if bad:
        print(f"\n! NOT LFS pointers ({len(bad)}): {bad[:3]}", file=sys.stderr)
        print("  Unstage with 'git reset' before committing.", file=sys.stderr)
        return 1
    print(f"\nverified: all {len(moves)} staged files are LFS pointers")

    try:
        SRC.rmdir()
        print(f"removed empty {SRC.name}/")
    except OSError:
        left = [p.name for p in SRC.iterdir()]
        print(f"{SRC.name}/ not empty, left in place: {left}")

    print("\nStaged, NOT committed. Review with 'git status', then commit.")
    print("Downstream code reading enriched_data/ now picks up v5 automatically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
