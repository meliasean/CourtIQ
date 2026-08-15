#!/usr/bin/env python3
"""
Score Cincinnati 2026 R128 - all 32 matches.

PROVENANCE
----------
Winners DERIVED from the R64 pairings (DraftKings, 2026-08-15), not read from
R128 score lines. All 32 R64 slots resolve to consecutive R128 bracket
positions, so the mapping is unambiguous - but a derived winner cannot tell a
match that was PLAYED from one that was awarded. No walkover or retirement is
known this round; if one occurred it will be scored here as a normal result.

Note match 19: Griekspoor withdrew PRE-match and was replaced by Shevchenko,
so that match was played and is scored normally.

Usage:  python score_cincinnati2026_r128.py [reports_dir]   (default: reports)
"""
import sys, os
import pandas as pd
import unicodedata

def al(n):
    return unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode("ascii").strip().lower().replace("-", " ")

reports = sys.argv[1] if len(sys.argv) > 1 else "reports"

RESULTS = {
    ("Cameron Norrie", "Dino Prizmic"):                   "Cameron Norrie",           # 1
    ("Marton Fucsovics", "Terence Atmane"):               "Terence Atmane",           # 2
    ("Sho Shimabukuro", "Hubert Hurkacz"):                "Hubert Hurkacz",           # 3
    ("Kyrian Jacquet", "Adolfo Daniel Vallejo"):          "Adolfo Daniel Vallejo",    # 4
    ("Denis Shapovalov", "Adrian Mannarino"):             "Denis Shapovalov",         # 5
    ("Roman Andres Burruchaga", "Jan-Lennard Struff"):    "Jan-Lennard Struff",       # 6
    ("Mariano Navone", "Raphael Collignon"):              "Mariano Navone",           # 7
    ("Camilo Ugo Carabelli", "Miomir Kecmanovic"):        "Miomir Kecmanovic",        # 8
    ("Thiago Agustin Tirante", "Jan Choinski"):           "Thiago Agustin Tirante",   # 9
    ("Martin Landaluce", "Jack Draper"):                  "Martin Landaluce",         # 10
    ("Rinky Hijikata", "Gael Monfils"):                   "Rinky Hijikata",           # 11
    ("Zachary Svajda", "Mattia Bellucci"):                "Mattia Bellucci",          # 12
    ("Titouan Droguet", "Matteo Berrettini"):             "Matteo Berrettini",        # 13
    ("Yannick Hanfmann", "Luca Van Assche"):              "Yannick Hanfmann",         # 14
    ("J.J. Wolf", "James Duckworth"):                     "James Duckworth",          # 15
    ("Vit Kopriva", "Quentin Halys"):                     "Quentin Halys",            # 16
    ("Alex Michelsen", "Jesper de Jong"):                 "Alex Michelsen",           # 17
    ("Daniel Merida", "Marin Cilic"):                     "Daniel Merida",            # 18
    ("Botic van de Zandschulp", "Aleksandr Shevchenko"):  "Botic van de Zandschulp",  # 19
    ("Christopher O'Connell", "Kamil Majchrzak"):         "Christopher O'Connell",    # 20
    ("Tomas Machac", "Pablo Carreno Busta"):              "Pablo Carreno Busta",      # 21
    ("Nuno Borges", "Thanasi Kokkinakis"):                "Nuno Borges",              # 22
    ("Aleksandar Kovacevic", "Karen Khachanov"):          "Aleksandar Kovacevic",     # 23
    ("Hamad Medjedovic", "Marco Trungelliti"):            "Marco Trungelliti",        # 24
    ("Jaime Faria", "Jenson Brooksby"):                   "Jaime Faria",              # 25
    ("Adam Walton", "Nicolas Mejia"):                     "Adam Walton",              # 26
    ("Fabian Marozsan", "Michael Zheng"):                 "Michael Zheng",            # 27
    ("Coleman Wong", "Daniel Altmaier"):                  "Daniel Altmaier",          # 28
    ("Sebastian Baez", "Grigor Dimitrov"):                "Sebastian Baez",           # 29
    ("Juncheng Shang", "Lorenzo Sonego"):                 "Lorenzo Sonego",           # 30
    ("Mark Lajal", "Juan Manuel Cerundolo"):              "Juan Manuel Cerundolo",    # 31
    ("Valentin Royer", "Stefanos Tsitsipas"):             "Stefanos Tsitsipas",       # 32
}

RES = {frozenset([al(a), al(b)]): al(w) for (a, b), w in RESULTS.items()}
assert len(RES) == 32, f"expected 32 unique matches, built {len(RES)}"

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

print("Scoring Cincinnati 2026 R128 (32 matches)...")
score_file(os.path.join(reports, "cincinnati2026_R128_predictions.csv"),
           os.path.join(reports, "cincinnati2026_R128_predictions_complete.csv"))
cck = score_file(os.path.join(reports, "cincinnati2026_R128_predictions_cck.csv"),
                 os.path.join(reports, "cincinnati2026_R128_predictions_cck_complete.csv"))
if cck is not None:
    bk = cck[cck["correct_prediction_book"].notna()]
    if len(bk):
        print(f"\nBook: {int(bk['correct_prediction_book'].sum())}/{len(bk)} = "
              f"{bk['correct_prediction_book'].mean()*100:.0f}%")
    print("\nNOTE: winners derived from R64 pairings, not score lines.")
