#!/usr/bin/env python3
"""
Why are some streaks too long?

HYPOTHESIS
----------
backfill_last5.py derives history from scored prediction files only. That
history is SPARSE - any match CourtIQ never predicted, or predicted in a round
that was never scored, is absent. A missing LOSS does not break a streak: the
backward walk skips it and keeps counting wins.

STRUCTURAL TEST
---------------
In a knockout draw a player either wins the title or loses exactly one match.
So if a player's last appearance in a tournament is a win in a round other than
the Final, a loss is missing from the data, and any streak that crosses that
tournament boundary is inflated.

This script is READ ONLY. It reuses backfill_last5's own loaders so it sees
exactly what the backfill saw.

Usage:
    python diagnose_streak.py "Carlos Alcaraz"
    python diagnose_streak.py --all          # every affected profile
"""
import sys, os, re, glob
from collections import defaultdict
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backfill_last5 import chrono_key, round_key

PROFILES = "reports/player_profiles_latest.csv"


def load_history(reports="reports"):
    """Rebuild the same hist dict backfill_last5 builds."""
    cands = glob.glob(os.path.join(reports, "*_predictions*complete*.csv"))
    chosen = {}
    for p in cands:
        k = round_key(p)
        if not k:
            continue
        if k not in chosen or ("_cck_" in os.path.basename(p)
                               and "_cck_" not in os.path.basename(chosen[k])):
            chosen[k] = p

    seen, hist = set(), defaultdict(list)
    for k, path in sorted(chosen.items()):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if not {"player_a", "player_b", "pred_winner", "correct_prediction"}.issubset(df.columns):
            continue
        for _, r in df.iterrows():
            cp = r["correct_prediction"]
            if pd.isna(cp) or pd.isna(r.get("pred_winner")):
                continue
            a, b = str(r["player_a"]).strip(), str(r["player_b"]).strip()
            pw = str(r["pred_winner"]).strip()
            date = str(r.get("date", "")).strip()[:10]
            if (date, a, b) in seen:
                continue
            seen.add((date, a, b))
            if pw == a:
                win, lose = (a, b) if int(cp) == 1 else (b, a)
            elif pw == b:
                win, lose = (b, a) if int(cp) == 1 else (a, b)
            else:
                continue
            hist[win].append((date, "W", k))
            hist[lose].append((date, "L", k))
    return hist, chosen


def streaks(rows):
    """Return (naive, corrected, break_reason). rows must be chrono-sorted."""
    seq = list(rows)
    if not seq:
        return 0, 0, None

    last_res = seq[-1][1]
    naive = 0
    for _d, res, _k in reversed(seq):
        if res != last_res:
            break
        naive += 1
    naive = naive if last_res == "W" else -naive

    # corrected: a streak may cross a tournament boundary only if the player
    # WON that earlier tournament (last match there was a Final they won).
    last_in_tourney = {}
    for d, res, (t, rnd) in seq:
        last_in_tourney[t] = (rnd, res)

    corrected, reason = 0, None
    cur_t = seq[-1][2][0]
    for d, res, (t, rnd) in reversed(seq):
        if res != last_res:
            break
        if t != cur_t:
            prev_rnd, prev_res = last_in_tourney[t]
            if not (prev_rnd == "F" and prev_res == "W"):
                reason = (f"streak crossed into {t}, where the last recorded match "
                          f"was {prev_rnd} ({prev_res}) - not a title, so a loss is missing")
                break
            cur_t = t
        corrected += 1
    corrected = corrected if last_res == "W" else -corrected
    return naive, corrected, reason


def main():
    hist, chosen = load_history()
    for p in hist:
        hist[p].sort(key=lambda x: chrono_key(x[0], x[2][0], x[2][1]))

    prof = pd.read_csv(PROFILES) if os.path.exists(PROFILES) else None
    stored = {}
    if prof is not None:
        prof["name"] = prof["name"].astype(str).str.strip()
        stored = dict(zip(prof["name"], prof.get("streak", pd.Series(dtype=float))))

    if "--all" in sys.argv:
        bad = []
        for player, rows in hist.items():
            n, c, why = streaks(rows)
            if n != c:
                bad.append((abs(n - c), player, stored.get(player), n, c))
        bad.sort(reverse=True)
        print(f"{len(bad)} profile(s) have an inflated streak "
              f"out of {len(hist)} with derived history\n")
        print(f"  {'player':<30}{'stored':>8}{'naive':>8}{'corrected':>11}")
        for _, player, st, n, c in bad[:40]:
            st_s = f"{st:+.0f}" if st is not None and pd.notna(st) else "--"
            print(f"  {player:<30}{st_s:>8}{n:>+8}{c:>+11}")
        if len(bad) > 40:
            print(f"  ... and {len(bad)-40} more")
        return

    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    player = sys.argv[1]
    if player not in hist:
        near = [p for p in hist if player.lower() in p.lower()]
        print(f"No derived history for {player!r}."
              + (f" Did you mean: {near}" if near else ""))
        sys.exit(1)

    rows = hist[player]
    print(f"{player} - {len(rows)} match(es) in the derived history\n")
    print(f"  {'date':<12}{'tournament':<20}{'round':<7}{'result'}")
    for d, res, (t, rnd) in rows:
        print(f"  {d:<12}{t:<20}{rnd:<7}{res}")

    n, c, why = streaks(rows)
    st = stored.get(player)
    print(f"\n  stored in profile : {st:+.0f}" if st is not None and pd.notna(st)
          else "\n  stored in profile : --")
    print(f"  naive streak      : {n:+d}   (what backfill_last5 computes)")
    print(f"  corrected streak  : {c:+d}   (breaking on missing exit losses)")
    if why:
        print(f"\n  {why}")

    # which rounds of the player's tournaments exist as scored files at all
    ts = sorted({t for _d, _r, (t, _rn) in rows})
    print(f"\n  Scored rounds on disk for this player's tournaments:")
    for t in ts:
        rounds = sorted({rnd for (tt, rnd) in chosen if tt == t})
        mine = sorted({rnd for _d, _r, (tt, rnd) in rows if tt == t})
        print(f"    {t:<20} files: {','.join(rounds) or '(none)'}")
        print(f"    {'':<20} this player appears in: {','.join(mine)}")


if __name__ == "__main__":
    main()
