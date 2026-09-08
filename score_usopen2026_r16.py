#!/usr/bin/env python3
"""
Score US Open 2026 R16 - all 8 matches.

PROVENANCE
----------
Winners DERIVED from the QF pairings (DraftKings, 2026-09-08), not read from
R16 score lines. All 4 QF slots resolve to consecutive R16 bracket positions,
so the mapping is unambiguous - but a derived winner cannot tell a match that
was PLAYED from one that was awarded. No walkover or retirement is known this
round; if one occurred it will be scored here as a normal result.

Two favourites beaten: Tiafoe over Medvedev (-165), Khachanov over Tien (-191).
Book 6/8.

Usage:  python score_usopen2026_r16.py [reports_dir]   (default: reports)
"""
import sys, os
import pandas as pd
import unicodedata

def al(n):
    return unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode("ascii").strip().lower().replace("-", " ")

reports = sys.argv[1] if len(sys.argv) > 1 else "reports"

RESULTS = {
    ("Alexander Zverev", "Luciano Darderi"):                   "Alexander Zverev",          # 1
    ("Arthur Gea", "Botic van de Zandschulp"):                 "Botic van de Zandschulp",   # 2
    ("Karen Khachanov", "Learner Tien"):                       "Karen Khachanov",           # 3
    ("Francisco Cerundolo", "Alexander Blockx"):               "Alexander Blockx",          # 4
    ("Daniil Medvedev", "Frances Tiafoe"):                     "Frances Tiafoe",            # 5
    ("Alex Michelsen", "Tomas Martin Etcheverry"):             "Alex Michelsen",            # 6
    ("Ben Shelton", "Stefanos Tsitsipas"):                     "Ben Shelton",               # 7
    ("Tommy Paul", "Carlos Alcaraz"):                          "Carlos Alcaraz",            # 8
}

RES = {frozenset([al(a), al(b)]): al(w) for (a, b), w in RESULTS.items()}
assert len(RES) == 8, f"expected 8 unique matches, built {len(RES)}"

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

print("Scoring US Open 2026 R16 (8 matches)...")
score_file(os.path.join(reports, "usopen2026_R16_predictions.csv"),
           os.path.join(reports, "usopen2026_R16_predictions_complete.csv"))
cck = score_file(os.path.join(reports, "usopen2026_R16_predictions_cck.csv"),
                 os.path.join(reports, "usopen2026_R16_predictions_cck_complete.csv"))
if cck is not None:
    bk = cck[cck["correct_prediction_book"].notna()]
    if len(bk):
        print(f"\nBook: {int(bk['correct_prediction_book'].sum())}/{len(bk)} = "
              f"{bk['correct_prediction_book'].mean()*100:.0f}%")
    print("\nNOTE: winners derived from QF pairings, not score lines.")
