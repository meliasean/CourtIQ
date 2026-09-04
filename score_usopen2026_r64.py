#!/usr/bin/env python3
"""
Score US Open 2026 R64 - all 32 matches.

PROVENANCE
----------
Winners DERIVED from the R32 pairings (DraftKings, 2026-09-04), not read from
R64 score lines. All 16 R32 slots resolve to consecutive R64 bracket positions,
so the mapping is unambiguous - but a derived winner cannot tell a match that
was PLAYED from one that was awarded. No walkover or retirement is known this
round; if one occurred it will be scored here as a normal result.

Eight favourites beaten. Sweeny over Musetti (-3900) is the largest upset by
price in the CourtIQ record - roughly 97% devigged. Also van de Zandschulp
over de Minaur, Khachanov over Auger-Aliassime, Navone over Berrettini,
Michelsen over Nakashima, Merida over Rublev, Gea over Svajda, Bonzi over Buse.
Book 24/32.

Usage:  python score_usopen2026_r64.py [reports_dir]   (default: reports)
"""
import sys, os
import pandas as pd
import unicodedata

def al(n):
    return unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode("ascii").strip().lower().replace("-", " ")

reports = sys.argv[1] if len(sys.argv) > 1 else "reports"

RESULTS = {
    ("Alexander Zverev", "Quentin Halys"):                 "Alexander Zverev",          # 1
    ("Alexei Popyrin", "Alejandro Tabilo"):                "Alejandro Tabilo",          # 2
    ("Luciano Darderi", "Dalibor Svrcina"):                "Luciano Darderi",           # 3
    ("Dane Sweeny", "Lorenzo Musetti"):                    "Dane Sweeny",               # 4
    ("Yunchaokete Bu", "Michael Zheng"):                   "Michael Zheng",             # 5
    ("Zachary Svajda", "Arthur Gea"):                      "Arthur Gea",                # 6
    ("Zizou Bergs", "Jesper de Jong"):                     "Zizou Bergs",               # 7
    ("Botic van de Zandschulp", "Alex De Minaur"):         "Botic van de Zandschulp",   # 8
    ("Felix Auger Aliassime", "Karen Khachanov"):          "Karen Khachanov",           # 9
    ("Benjamin Bonzi", "Ignacio Buse"):                    "Benjamin Bonzi",            # 10
    ("Jakub Mensik", "Jurij Rodionov"):                    "Jakub Mensik",              # 11
    ("Gael Monfils", "Learner Tien"):                      "Learner Tien",              # 12
    ("Taylor Fritz", "Mattia Bellucci"):                   "Taylor Fritz",              # 13
    ("Jan-Lennard Struff", "Francisco Cerundolo"):         "Francisco Cerundolo",       # 14
    ("Alexander Blockx", "Marco Trungelliti"):             "Alexander Blockx",          # 15
    ("Tristan Schoolkate", "Flavio Cobolli"):              "Flavio Cobolli",            # 16
    ("Daniil Medvedev", "Sebastian Gorzny"):               "Daniil Medvedev",           # 17
    ("Jaume Munar", "Arthur Rinderknech"):                 "Arthur Rinderknech",        # 18
    ("Valentin Vacherot", "Kamil Majchrzak"):              "Valentin Vacherot",         # 19
    ("Rei Sakamoto", "Frances Tiafoe"):                    "Frances Tiafoe",            # 20
    ("Brandon Nakashima", "Alex Michelsen"):               "Alex Michelsen",            # 21
    ("Daniel Merida", "Andrey Rublev"):                    "Daniel Merida",             # 22
    ("Tomas Martin Etcheverry", "Jacob Fearnley"):         "Tomas Martin Etcheverry",   # 23
    ("Matteo Berrettini", "Mariano Navone"):               "Mariano Navone",            # 24
    ("Ben Shelton", "Hubert Hurkacz"):                     "Ben Shelton",               # 25
    ("Denis Shapovalov", "Luca Van Assche"):               "Denis Shapovalov",          # 26
    ("Jiri Lehecka", "Toby Samuel"):                       "Jiri Lehecka",              # 27
    ("Lloyd Harris", "Stefanos Tsitsipas"):                "Stefanos Tsitsipas",        # 28
    ("Alexander Bublik", "Adrian Mannarino"):              "Alexander Bublik",          # 29
    ("Dino Prizmic", "Tommy Paul"):                        "Tommy Paul",                # 30
    ("James Duckworth", "Yibing Wu"):                      "Yibing Wu",                 # 31
    ("Jaime Faria", "Carlos Alcaraz"):                     "Carlos Alcaraz",            # 32
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

print("Scoring US Open 2026 R64 (32 matches)...")
score_file(os.path.join(reports, "usopen2026_R64_predictions.csv"),
           os.path.join(reports, "usopen2026_R64_predictions_complete.csv"))
cck = score_file(os.path.join(reports, "usopen2026_R64_predictions_cck.csv"),
                 os.path.join(reports, "usopen2026_R64_predictions_cck_complete.csv"))
if cck is not None:
    bk = cck[cck["correct_prediction_book"].notna()]
    if len(bk):
        print(f"\nBook: {int(bk['correct_prediction_book'].sum())}/{len(bk)} = "
              f"{bk['correct_prediction_book'].mean()*100:.0f}%")
    print("\nNOTE: winners derived from R32 pairings, not score lines.")
