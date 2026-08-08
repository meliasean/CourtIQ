#!/usr/bin/env python3
"""
Score Canada Masters 2026 R32 - all 16 matches.

PROVENANCE
----------
Winners DERIVED from the R16 pairings (DraftKings, 2026-08-08), not read from
R32 score lines. All 8 R16 slots resolve to consecutive R32 bracket positions,
so the mapping is unambiguous - but a derived winner cannot tell a match that
was PLAYED from one that was awarded. No walkover or retirement is known this
round; if one occurred it will be scored here as a normal result. Spot-check.

Usage:  python score_canada2026_r32.py [reports_dir]   (default: reports)
"""
import sys, os
import pandas as pd
import unicodedata

def al(n):
    return unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode("ascii").strip().lower().replace("-", " ")

reports = sys.argv[1] if len(sys.argv) > 1 else "reports"

RESULTS = {
    ("Tallon Griekspoor", "Matteo Arnaldi"):        "Tallon Griekspoor",        # 1
    ("Daniel Merida", "Alex Michelsen"):            "Daniel Merida",            # 2
    ("Learner Tien", "Tommy Paul"):                 "Learner Tien",             # 3
    ("Alexei Popyrin", "Thiago Agustin Tirante"):   "Thiago Agustin Tirante",   # 4
    ("Botic van de Zandschulp", "Hubert Hurkacz"):  "Botic van de Zandschulp",  # 5
    ("Terence Atmane", "Jakub Mensik"):             "Jakub Mensik",             # 6
    ("Casper Ruud", "Joao Fonseca"):                "Joao Fonseca",             # 7
    ("Zizou Bergs", "Ben Shelton"):                 "Ben Shelton",              # 8
    ("Jiri Lehecka", "Alexander Blockx"):           "Jiri Lehecka",             # 9
    ("Rafael Jodar", "Lorenzo Musetti"):            "Rafael Jodar",             # 10
    ("Mariano Navone", "Arthur Fils"):              "Arthur Fils",              # 11
    ("Cameron Norrie", "Alex De Minaur"):           "Cameron Norrie",           # 12
    ("Yannick Hanfmann", "Nuno Borges"):            "Nuno Borges",              # 13
    ("Luciano Darderi", "Juncheng Shang"):          "Luciano Darderi",          # 14
    ("Frances Tiafoe", "Arthur Rinderknech"):       "Arthur Rinderknech",       # 15
    ("Brandon Nakashima", "Titouan Droguet"):       "Brandon Nakashima",        # 16
}

RES = {frozenset([al(a), al(b)]): al(w) for (a, b), w in RESULTS.items()}
assert len(RES) == 16, f"expected 16 unique matches, built {len(RES)}"

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

print("Scoring Canada Masters 2026 R32 (16 matches)...")
score_file(os.path.join(reports, "canada2026_R32_predictions.csv"),
           os.path.join(reports, "canada2026_R32_predictions_complete.csv"))
cck = score_file(os.path.join(reports, "canada2026_R32_predictions_cck.csv"),
                 os.path.join(reports, "canada2026_R32_predictions_cck_complete.csv"))
if cck is not None:
    bk = cck[cck["correct_prediction_book"].notna()]
    if len(bk):
        print(f"\nBook: {int(bk['correct_prediction_book'].sum())}/{len(bk)} = "
              f"{bk['correct_prediction_book'].mean()*100:.0f}%")
    print("\nNOTE: winners derived from R16 pairings, not score lines.")
