#!/usr/bin/env python3
"""
Score Cincinnati 2026 SF - both matches.

PROVENANCE
----------
Winners DERIVED from the Final pairing (DraftKings, 2026-08-23): Fils vs
Tiafoe. Each finalist maps to exactly one SF, so the mapping is unambiguous -
but a derived winner cannot tell a match that was PLAYED from one that was
awarded. No walkover or retirement is known; if one occurred it will be scored
here as a normal result.

Tiafoe won as the +105 underdog; Nakashima was -128. Fils held at -286.
Book 1/2 this round.

Usage:  python score_cincinnati2026_sf.py [reports_dir]   (default: reports)
"""
import sys, os
import pandas as pd
import unicodedata

def al(n):
    return unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode("ascii").strip().lower().replace("-", " ")

reports = sys.argv[1] if len(sys.argv) > 1 else "reports"

RESULTS = {
    ("Flavio Cobolli", "Arthur Fils"):          "Arthur Fils",      # 1
    ("Brandon Nakashima", "Frances Tiafoe"):    "Frances Tiafoe",   # 2
}

RES = {frozenset([al(a), al(b)]): al(w) for (a, b), w in RESULTS.items()}
assert len(RES) == 2, f"expected 2 unique matches, built {len(RES)}"

def score_file(path, write_complete_to):
    if not os.path.exists(path):
        print(f"  skip (not found): {path}"); return None
    df = pd.read_csv(path)
    scored, seen = 0, set()
    for i, r in df.iterrows():
        key = frozenset([al(r["player_a"]), al(r["player_b"])])
        if key not in RES: continue
        winner = RES[key]
        if pd.isna(r.get("pred_winner")): continue
        df.at[i, "correct_prediction"] = 1 if al(r["pred_winner"]) == winner else 0
        oa, ob = r.get("odds_player_a"), r.get("odds_player_b")
        if pd.notna(oa) and pd.notna(ob):
            book_pick = al(r["player_a"]) if oa < ob else al(r["player_b"])
            df.at[i, "correct_prediction_book"] = 1 if book_pick == winner else 0
        seen.add(key); scored += 1

    missing = set(RES) - seen
    if missing:
        print(f"  !! {len(missing)} result(s) had no matching row in {os.path.basename(path)}:")
        for k in missing:
            print(f"       {' vs '.join(sorted(k))}")
        print("     -> spelling mismatch. Check against the CSV before trusting the numbers.")

    if scored == 0:
        print(f"  ABORT: nothing matched in {os.path.basename(path)}; not writing.")
        return None

    df.to_csv(write_complete_to, index=False)
    sc = df[df["correct_prediction"].notna()]
    acc = sc["correct_prediction"].mean() * 100 if len(sc) else 0
    print(f"  {os.path.basename(write_complete_to)}: {scored} scored, "
          f"model {int(sc['correct_prediction'].sum())}/{len(sc)} = {acc:.0f}%")
    return df

print("Scoring Cincinnati 2026 SF (2 matches)...")
score_file(os.path.join(reports, "cincinnati2026_SF_predictions.csv"),
           os.path.join(reports, "cincinnati2026_SF_predictions_complete.csv"))
cck = score_file(os.path.join(reports, "cincinnati2026_SF_predictions_cck.csv"),
                 os.path.join(reports, "cincinnati2026_SF_predictions_cck_complete.csv"))
if cck is not None:
    bk = cck[cck["correct_prediction_book"].notna()]
    if len(bk):
        print(f"\nBook: {int(bk['correct_prediction_book'].sum())}/{len(bk)}")
    print("\nNOTE: winners derived from the Final pairing, not score lines.")
