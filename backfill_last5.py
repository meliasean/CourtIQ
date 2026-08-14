#!/usr/bin/env python3
"""
Rebuild the missing last5 column and correct the streak column in
player_profiles_latest.csv.

WHY
---
The players-page Form column renders W/L badges from profiles["last5"].
That column does not exist in the current profile schema, so the column
renders as an em-dash. Nothing in the pipeline writes it.

HOW
---
Actual winners are derivable from the scored prediction files:
    correct_prediction == 1  ->  winner is pred_winner
    correct_prediction == 0  ->  winner is the other player
That yields a dated W/L record per player, which is exactly what last5 needs.

STREAK
------
update_profiles() reset streak to 0 each run and counted only the current
event, making it a per-tournament streak rather than a career one. Stored
values therefore disagree with the last5 badges. Streak is recomputed here
from the SAME sorted history that produces last5, inside the same loop, so
the two columns cannot fall out of agreement. It is computed over FULL
history, not last5 - a streak can be longer than five.

DUPLICATE SAFETY
----------------
Each round exists as both *_predictions_complete.csv and
*_predictions_cck_complete.csv holding the SAME matches. Counting both would
double every result - the same class of bug that inflated book accuracy.
So exactly ONE file is used per (tournament, round), preferring the cck file.
Rows are then deduped again on (date, player_a, player_b) as a backstop.

DRY RUN by default.

Usage:
    python backfill_last5.py
    python backfill_last5.py --commit
    python backfill_last5.py --n 10 --commit      # keep last 10 instead of 5
"""
import sys, os, re, glob, json, shutil
from collections import defaultdict
from datetime import datetime
import pandas as pd
from courtiq_engine import TOURNEY_ORDER, ROUND_ORDER

PROFILES = "reports/player_profiles_latest.csv"
KEEP = 5


def parse_date(date_str):
    """Source CSVs mix ISO (YYYY-MM-DD) and US (M/D/YYYY) date strings across
    (and even within) tournaments. Raw string comparison sorts them wrong -
    e.g. '4/22/2026' > '2026-07-08' lexically despite being chronologically
    earlier. Parse to a real date so the sort key is actually chronological."""
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    return None  # unparseable/missing - sorts first, see chrono_key


def chrono_key(date_str, tourney, rnd):
    """Sort key: date first (parsed, not raw string), then true tournament/
    round progression as a tiebreak for rounds that share a date (several
    tournaments only stamp one date across all rounds, so alphabetical
    round-name order would otherwise scramble same-date results, e.g. F
    before R32)."""
    d = parse_date(date_str)
    d_key = (0, d) if d is not None else (-1, None)  # unparseable sorts first, not crashes
    t_idx = TOURNEY_ORDER.index(tourney) if tourney in TOURNEY_ORDER else len(TOURNEY_ORDER)
    r_idx = ROUND_ORDER.index(rnd) if rnd in ROUND_ORDER else len(ROUND_ORDER)
    return (d_key, t_idx, r_idx)


def round_key(path):
    """(tournament, round) from e.g. washington2026_SF_predictions_cck_complete.csv"""
    b = os.path.basename(path)
    m = re.match(r"^(.+?)_([A-Za-z0-9]+)_predictions", b)
    return (m.group(1), m.group(2)) if m else (b, "")


def main():
    commit = "--commit" in sys.argv
    keep = KEEP
    if "--n" in sys.argv:
        keep = int(sys.argv[sys.argv.index("--n") + 1])
    prof_path = PROFILES
    if "--profiles" in sys.argv:
        prof_path = sys.argv[sys.argv.index("--profiles") + 1]

    if not os.path.exists(prof_path):
        print(f"ABORT: {prof_path} not found"); sys.exit(1)

    # ---- pick exactly one file per (tournament, round) -------------------
    cands = glob.glob("reports/*_predictions*complete*.csv")
    if not cands:
        print("ABORT: no scored prediction files found in reports/"); sys.exit(1)

    chosen = {}
    for p in cands:
        k = round_key(p)
        # prefer the cck file; it carries the same matches plus book columns
        if k not in chosen or ("_cck_" in os.path.basename(p)
                               and "_cck_" not in os.path.basename(chosen[k])):
            chosen[k] = p

    print(f"{len(cands)} scored file(s) on disk -> {len(chosen)} unique (tournament, round)")
    print(f"  {len(cands) - len(chosen)} skipped as duplicate views of the same matches\n")

    # ---- derive winners --------------------------------------------------
    seen, results, skipped = set(), [], 0
    for k, path in sorted(chosen.items()):
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"  unreadable: {os.path.basename(path)} -> {e}"); continue
        need = {"player_a", "player_b", "pred_winner", "correct_prediction"}
        if not need.issubset(df.columns):
            skipped += 1; continue

        for _, r in df.iterrows():
            cp = r["correct_prediction"]
            if pd.isna(cp) or pd.isna(r.get("pred_winner")):
                continue
            a, b = str(r["player_a"]).strip(), str(r["player_b"]).strip()
            pw = str(r["pred_winner"]).strip()
            date = str(r.get("date", "")).strip()[:10]

            dedupe = (date, a, b)
            if dedupe in seen:
                continue
            seen.add(dedupe)

            if pw == a:
                win, lose = (a, b) if int(cp) == 1 else (b, a)
            elif pw == b:
                win, lose = (b, a) if int(cp) == 1 else (a, b)
            else:
                continue  # pred_winner matches neither player - leave it alone

            results.append((date, win, "W", k))
            results.append((date, lose, "L", k))

    print(f"Derived {len(results)//2} unique match result(s) "
          f"covering {len(set(p for _, p, _, _ in results))} player(s)")
    if skipped:
        print(f"  ({skipped} file(s) lacked the required columns)")

    # ---- build last5 per player -----------------------------------------
    hist = defaultdict(list)
    for date, player, res, k in results:
        hist[player].append((date, res, k))

    last5 = {}
    streaks = {}
    for player, rows in hist.items():
        rows.sort(key=lambda x: chrono_key(x[0], x[2][0], x[2][1]))  # chronological
        last5[player] = json.dumps(
            [{"result": r, "date": d} for d, r, _k in rows[-keep:]]
        )
        # Trailing streak, same ordering as last5 above.
        # The derived history is sparse, so a missing loss would otherwise let
        # a streak run straight through a tournament exit. In a knockout draw a
        # player either takes the title or loses exactly one match, which gives
        # a structural test for the gap.
        last_in_tourney = {}
        n_in_tourney = {}
        for _d, _res, (t, rnd) in rows:
            n_in_tourney[t] = n_in_tourney.get(t, 0) + 1
        for _d, res, (t, rnd) in rows:
            last_in_tourney[t] = (rnd, res)

        last_res = rows[-1][1]
        cur_t = rows[-1][2][0]
        n = 0
        for _d, res, (t, _rnd) in reversed(rows):
            if res != last_res:
                break
            if t != cur_t:
                if last_res == "W":
                    # may only continue if the earlier event ended in a title
                    prev_rnd, prev_res = last_in_tourney[t]
                    if not (prev_rnd == "F" and prev_res == "W"):
                        break
                else:
                    # may only continue if the later event was a lone loss,
                    # i.e. no unrecorded wins are implied before it
                    if n_in_tourney[cur_t] != 1:
                        break
                cur_t = t
            n += 1
        streaks[player] = float(n if last_res == "W" else -n)

    # ---- attach to profiles ---------------------------------------------
    prof = pd.read_csv(prof_path)
    prof["name"] = prof["name"].astype(str).str.strip()
    matched = prof["name"].isin(last5)

    print(f"\nProfiles: {len(prof)} row(s)")
    print(f"  {int(matched.sum())} matched to derived history")
    print(f"  {int((~matched).sum())} unmatched (no scored matches on file)")

    unmatched_hist = sorted(set(last5) - set(prof["name"]))
    if unmatched_hist:
        print(f"\n  {len(unmatched_hist)} name(s) have history but NO profile row.")
        print("  These are probably alias splits - worth comparing to the alias audit:")
        for n in unmatched_hist[:15]:
            print(f"      {n}")
        if len(unmatched_hist) > 15:
            print(f"      ... and {len(unmatched_hist)-15} more")

    if not commit:
        print("\nDRY RUN - nothing written. Re-run with --commit.")
        sample = [n for n in prof["name"] if n in last5][:3]
        for n in sample:
            print(f"  sample  {n}: {last5[n]}")

        if "streak" in prof.columns:
            cmp_rows = []
            for _, pr in prof.iterrows():
                nm = pr["name"]
                if nm in streaks:
                    try:
                        old_s = float(pr["streak"] or 0)
                    except (TypeError, ValueError):
                        old_s = 0.0
                    new_s = streaks[nm]
                    if old_s != new_s:
                        cmp_rows.append((abs(new_s - old_s), nm, old_s, new_s))
            cmp_rows.sort(reverse=True)
            print(f"\n  {len(cmp_rows)} streak value(s) would change. Largest shifts:")
            for _, nm, old_s, new_s in cmp_rows[:12]:
                print(f"      {nm:32} {old_s:+.0f}  ->  {new_s:+.0f}")
            print("      Stored values were per-tournament, so most will grow.")
            print("      Sanity-check one you know before committing.")
        return

    prof["last5"] = prof["name"].map(last5).fillna("[]")
    if "streak" in prof.columns:
        # fillna keeps existing values for players with no derived history
        prof["streak"] = prof["name"].map(streaks).fillna(prof["streak"]).astype(float)
    else:
        prof["streak"] = prof["name"].map(streaks).fillna(0.0).astype(float)
    bak = f"{prof_path}.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(prof_path, bak)
    prof.to_csv(prof_path, index=False)
    print(f"\nWROTE {prof_path}  (backup: {bak})")
    print("  added column: last5")
    print("  recomputed column: streak")
    print("\nNEXT:  python courtiq_engine.py site --output docs/index.html")


if __name__ == "__main__":
    main()
