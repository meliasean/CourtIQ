#!/usr/bin/env python3
"""
Verify every player in a draw resolves in the rankings file BEFORE predicting.

WHY
---
courtiq_engine reads reports/atp_rankings_current.csv. Any player absent from
it is filled with MISSING_RANK (2000), so delta_rank becomes fabricated:

    one player missing  -> a huge fake gap   (Zverev/Shelton: 1994)
    both players missing -> exactly 0        (Zverev/Khachanov: -0.0)

The second is invisible in a spreadsheet, which is why this went unnoticed for
four tournaments. rank_diff is the model's #2 feature by importance.

The file was a Washington-scoped 32-player snapshot under a name implying a
tour-wide list. Munich ran 100% fabricated, Canada 41%, Cincinnati 47%,
US Open 55%. Washington itself read 0% only because the file was its own field.

RUN THIS BEFORE EVERY predict. Exit code 1 if any player would impute.

Usage:
    python check_rank_coverage.py reports/beijing2026_R32_draw.csv
    python check_rank_coverage.py <draw.csv> --rankings reports/atp_rankings_current.csv
"""
import sys, os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from courtiq_engine import alias
except Exception:
    def alias(n): return str(n).strip()

def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    draw_path = sys.argv[1]
    ranks_path = "reports/atp_rankings_current.csv"
    if "--rankings" in sys.argv:
        ranks_path = sys.argv[sys.argv.index("--rankings") + 1]

    for p in (draw_path, ranks_path):
        if not os.path.exists(p):
            print(f"ABORT: {p} not found"); sys.exit(1)

    draw = pd.read_csv(draw_path)
    ranks = pd.read_csv(ranks_path)

    namecol = next((c for c in ("name", "player") if c in ranks.columns), None)
    if namecol is None or "rank" not in ranks.columns:
        print(f"ABORT: {ranks_path} needs a 'rank' column plus 'name' or 'player'.")
        print(f"       found: {list(ranks.columns)}")
        sys.exit(1)

    known = {alias(n) for n in ranks[namecol].astype(str)}
    print(f"rankings : {ranks_path}  ({len(ranks)} players)")
    if "tourney_name" in ranks.columns:
        t = ranks.tourney_name.dropna().unique()
        print(f"  !! this file is TOURNAMENT-SCOPED: {', '.join(map(str, t[:3]))}")
        print(f"     a per-event snapshot under a name implying a tour-wide list")
    print(f"draw     : {draw_path}  ({len(draw)} matches)")

    players = sorted(set(draw.player_a.astype(str)) | set(draw.player_b.astype(str)))
    missing = [p for p in players if alias(p) not in known]

    print(f"\n{len(players) - len(missing)}/{len(players)} players resolve")

    if not missing:
        print("\nOK - no player will impute. Safe to predict.")
        return

    print(f"\n{len(missing)} player(s) would be filled with MISSING_RANK (2000):")
    for p in missing:
        print(f"    {p}")

    # which matches are affected, and how
    both = one = 0
    print("\naffected matches:")
    for _, r in draw.iterrows():
        a_miss = alias(str(r.player_a)) not in known
        b_miss = alias(str(r.player_b)) not in known
        if not (a_miss or b_miss):
            continue
        kind = "BOTH -> delta_rank = 0 (silent)" if a_miss and b_miss else "ONE  -> huge fake gap"
        both += a_miss and b_miss
        one += (a_miss or b_miss) and not (a_miss and b_miss)
        print(f"    {r.player_a} vs {r.player_b}   {kind}")

    print(f"\n{one} match(es) with a fabricated gap, {both} silently zeroed.")
    print("\nFIX: copy this tournament's rank file into place, then re-run:")
    print("     cp atp_ranks_<tournament>.csv reports/atp_rankings_current.csv")
    sys.exit(1)

if __name__ == "__main__":
    main()
