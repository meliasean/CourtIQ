#!/usr/bin/env python3
"""
Score Cincinnati 2026 R64 - all 32 matches.

PROVENANCE
----------
Winners DERIVED from the R32 pairings (DraftKings, 2026-08-17), not read from
R64 score lines. All 16 R32 slots resolve to consecutive R64 bracket positions,
so the mapping is unambiguous - but a derived winner cannot tell a match that
was PLAYED from one that was awarded. No walkover or retirement is known this
round; if one occurred it will be scored here as a normal result.

Usage:  python score_cincinnati2026_r64.py [reports_dir]   (default: reports)
"""
import sys, os
import pandas as pd
import unicodedata

def al(n):
    return unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode("ascii").strip().lower().replace("-", " ")

reports = sys.argv[1] if len(sys.argv) > 1 else "reports"

RESULTS = {
    ("Alexander Zverev", "Cameron Norrie"):                "Alexander Zverev",         # 1
    ("Terence Atmane", "Tomas Martin Etcheverry"):         "Terence Atmane",           # 2
    ("Tommy Paul", "Hubert Hurkacz"):                      "Tommy Paul",               # 3
    ("Adolfo Daniel Vallejo", "Valentin Vacherot"):        "Adolfo Daniel Vallejo",    # 4
    ("Rafael Jodar", "Denis Shapovalov"):                  "Rafael Jodar",             # 5
    ("Jan-Lennard Struff", "Alejandro Tabilo"):            "Alejandro Tabilo",         # 6
    ("Alexander Blockx", "Mariano Navone"):                "Alexander Blockx",         # 7
    ("Miomir Kecmanovic", "Flavio Cobolli"):               "Flavio Cobolli",           # 8
    ("Novak Djokovic", "Thiago Agustin Tirante"):          "Thiago Agustin Tirante",   # 9
    ("Martin Landaluce", "Matteo Arnaldi"):                "Martin Landaluce",         # 10
    ("Luciano Darderi", "Rinky Hijikata"):                 "Rinky Hijikata",           # 11
    ("Mattia Bellucci", "Jakub Mensik"):                   "Jakub Mensik",             # 12
    ("Jiri Lehecka", "Matteo Berrettini"):                 "Jiri Lehecka",             # 13
    ("Yannick Hanfmann", "Arthur Fils"):                   "Arthur Fils",              # 14
    ("Arthur Fery", "James Duckworth"):                    "Arthur Fery",              # 15
    ("Quentin Halys", "Alex De Minaur"):                   "Alex De Minaur",           # 16
    ("Taylor Fritz", "Alex Michelsen"):                    "Taylor Fritz",             # 17
    ("Daniel Merida", "Zizou Bergs"):                      "Daniel Merida",            # 18
    ("Joao Fonseca", "Botic van de Zandschulp"):           "Joao Fonseca",             # 19
    ("Christopher O'Connell", "Casper Ruud"):              "Christopher O'Connell",    # 20
    ("Andrey Rublev", "Pablo Carreno Busta"):              "Andrey Rublev",            # 21
    ("Nuno Borges", "Francisco Cerundolo"):                "Nuno Borges",              # 22
    ("Brandon Nakashima", "Aleksandar Kovacevic"):         "Brandon Nakashima",        # 23
    ("Marco Trungelliti", "Daniil Medvedev"):              "Daniil Medvedev",          # 24
    ("Ben Shelton", "Jaime Faria"):                        "Jaime Faria",              # 25
    ("Adam Walton", "Ignacio Buse"):                       "Adam Walton",              # 26
    ("Ugo Humbert", "Michael Zheng"):                      "Michael Zheng",            # 27
    ("Daniel Altmaier", "Lorenzo Musetti"):                "Lorenzo Musetti",          # 28
    ("Learner Tien", "Sebastian Baez"):                    "Learner Tien",             # 29
    ("Lorenzo Sonego", "Frances Tiafoe"):                  "Frances Tiafoe",           # 30
    ("Arthur Rinderknech", "Juan Manuel Cerundolo"):       "Juan Manuel Cerundolo",    # 31
    ("Stefanos Tsitsipas", "Felix Auger Aliassime"):       "Felix Auger Aliassime",    # 32
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

print("Scoring Cincinnati 2026 R64 (32 matches)...")
score_file(os.path.join(reports, "cincinnati2026_R64_predictions.csv"),
           os.path.join(reports, "cincinnati2026_R64_predictions_complete.csv"))
cck = score_file(os.path.join(reports, "cincinnati2026_R64_predictions_cck.csv"),
                 os.path.join(reports, "cincinnati2026_R64_predictions_cck_complete.csv"))
if cck is not None:
    bk = cck[cck["correct_prediction_book"].notna()]
    if len(bk):
        print(f"\nBook: {int(bk['correct_prediction_book'].sum())}/{len(bk)} = "
              f"{bk['correct_prediction_book'].mean()*100:.0f}%")
    print("\nNOTE: winners derived from R32 pairings, not score lines.")
