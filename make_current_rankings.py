#!/usr/bin/env python3
"""
Build reports/atp_rankings_current.csv - the file courtiq_engine actually reads.

THE GAP THIS CLOSES
-------------------
The workflow produces atp_ranks_<tournament>.csv per event. The engine reads
reports/atp_rankings_current.csv (RANKINGS_CURRENT, line ~53). Nothing ever
connected the two, so every per-tournament rank file built since Washington was
ignored while the engine kept reading a Washington-scoped 32-player snapshot
from 2026-07-21.

The failure is silent: a player absent from the file gets MISSING_RANK (2000),
so delta_rank becomes fabricated - a huge fake gap when one side is missing, or
exactly 0 when both are, which looks like a normal value. US Open R128 ran with
41 of 64 matches silently zeroed and only 3 clean. rank_diff is the model's #2
feature by importance.

WHAT THIS DOES
--------------
Unions every atp_ranks_*.csv found, keeping each player's rank from the LATEST
tournament they appear in, ordered by TOURNEY_ORDER. That covers more players
than any single draw, including entrants who were in an earlier field but not
the current one.

TIME FAITHFULNESS
-----------------
--as-of <tournament> excludes every event at or after that point in
TOURNEY_ORDER, so a snapshot rebuilt for a past tournament cannot borrow ranks
from matches that had not been played. Use it whenever regenerating history.
Without it, the build is "as of now".

Profiles are NOT a source: player_profiles_latest.csv has no rank column.

Usage:
    python make_current_rankings.py
    python make_current_rankings.py --as-of beijing2026
    python make_current_rankings.py --dry-run
    python make_current_rankings.py --out reports/atp_rankings_current.csv
"""
import sys, os, glob, re, shutil
from datetime import datetime
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from courtiq_engine import TOURNEY_ORDER, alias
except Exception as e:
    print(f"ABORT: could not import from courtiq_engine ({e})")
    sys.exit(1)

OUT_DEFAULT = os.path.join("reports", "atp_rankings_current.csv")
SEARCH = ["atp_ranks_*.csv", os.path.join("reports", "atp_ranks_*.csv")]


def tourney_of(path):
    m = re.match(r"^atp_ranks_(.+?)\.csv$", os.path.basename(path))
    return m.group(1) if m else None


def order_index(t):
    """Position in TOURNEY_ORDER.

    Unknown names sort FIRST (-1), not last. Files like
    atp_ranks_wimbledon2025_onward.csv are historical aggregates, not events;
    sorting them last would let them win the groupby .last() and silently
    override every real tournament's ranks. Sorting them first means they only
    supply players no real tournament covers, and --as-of can exclude them
    explicitly rather than by accident.
    """
    try:
        return TOURNEY_ORDER.index(t)
    except ValueError:
        return -1


def read_ranks(path):
    """Accept rank + (name|player). Returns DataFrame[name, rank] or None."""
    try:
        d = pd.read_csv(path)
    except Exception as e:
        print(f"  unreadable: {os.path.basename(path)} -> {e}")
        return None
    namecol = next((c for c in ("name", "player") if c in d.columns), None)
    if namecol is None or "rank" not in d.columns:
        print(f"  skip (needs 'rank' + 'name'/'player'): {os.path.basename(path)} "
              f"has {list(d.columns)}")
        return None
    out = d[[namecol, "rank"]].rename(columns={namecol: "name"})
    out["name"] = out["name"].astype(str).str.strip()
    out = out[out["name"].ne("") & out["name"].str.lower().ne("nan")]
    out["rank"] = pd.to_numeric(out["rank"], errors="coerce")
    return out.dropna(subset=["rank"])


def main():
    as_of = None
    if "--as-of" in sys.argv:
        as_of = sys.argv[sys.argv.index("--as-of") + 1]
    out_path = OUT_DEFAULT
    if "--out" in sys.argv:
        out_path = sys.argv[sys.argv.index("--out") + 1]
    dry = "--dry-run" in sys.argv

    files = []
    for pat in SEARCH:
        files += glob.glob(pat)
    files = sorted(set(files))
    if not files:
        print("ABORT: no atp_ranks_*.csv found in . or reports/")
        sys.exit(1)

    cutoff = None
    if as_of is not None:
        if as_of not in TOURNEY_ORDER:
            print(f"ABORT: {as_of!r} not in TOURNEY_ORDER. Known tail: "
                  f"{TOURNEY_ORDER[-6:]}")
            sys.exit(1)
        cutoff = TOURNEY_ORDER.index(as_of)
        print(f"as-of {as_of} - excluding tournaments at or after that point\n")

    rows, used, skipped = [], [], []
    for path in files:
        t = tourney_of(path)
        if t is None:
            continue
        idx = order_index(t)
        # A historical aggregate has no place in the order, so it cannot be
        # shown to predate the cutoff. Exclude it from any as-of rebuild.
        if cutoff is not None and (idx == -1 or idx >= cutoff):
            skipped.append(t if idx != -1 else f"{t} (not in TOURNEY_ORDER)")
            continue
        d = read_ranks(path)
        if d is None or d.empty:
            continue
        d["tourney"] = t
        d["order"] = idx
        rows.append(d)
        used.append((idx, t if idx != -1 else f"{t}  (not in TOURNEY_ORDER)", len(d)))

    if not rows:
        print("ABORT: no usable rank files after filtering.")
        sys.exit(1)

    all_ranks = pd.concat(rows, ignore_index=True)
    all_ranks["key"] = all_ranks["name"].map(alias)

    # latest tournament wins for each player
    all_ranks = all_ranks.sort_values(["key", "order"])
    latest = all_ranks.groupby("key", as_index=False).last()
    final = (latest[["rank", "name", "tourney"]]
             .rename(columns={"tourney": "source_tourney"})
             .sort_values("rank")
             .reset_index(drop=True))
    final["rank"] = final["rank"].astype(int)

    print(f"{len(used)} rank file(s) used:")
    for idx, t, n in sorted(used):
        print(f"    {t:<24}{n:>5} players")
    if skipped:
        print(f"\n  excluded by --as-of: {', '.join(sorted(skipped))}")
    unknown = [t for idx, t, _ in used if idx == -1]
    if unknown:
        print(f"\n  !! {len(unknown)} source(s) not in TOURNEY_ORDER - treated as")
        print(f"     historical aggregates, lowest priority, never overriding a")
        print(f"     real tournament: {', '.join(t.split('  ')[0] for t in unknown)}")

    print(f"\n{len(final)} unique players | rank range "
          f"{final['rank'].min()}-{final['rank'].max()}")
    depth = (final['rank'] <= 300).sum()
    print(f"  {depth} inside the top 300")
    contrib = final.source_tourney.value_counts()
    print("\n  most recent source per player:")
    for t, n in contrib.items():
        print(f"    {t:<24}{n:>5}")

    if dry:
        print(f"\nDRY RUN - would write {out_path}")
        return

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if os.path.exists(out_path):
        bak = f"{out_path}.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(out_path, bak)
        print(f"\nbacked up existing -> {bak}")
    final.to_csv(out_path, index=False)
    print(f"WROTE {out_path}")
    print("\nNEXT:  python check_rank_coverage.py reports/<tournament>_<round>_draw.csv")


if __name__ == "__main__":
    main()
