#!/usr/bin/env python3
"""
Score US Open 2026 R128 - 54 of 64 matches.

PROVENANCE
----------
Winners DERIVED from the R64 pairings (DraftKings, 2026-09-02), not read from
R128 score lines. Every available R64 slot resolves to consecutive R128 bracket
positions, so the mapping is unambiguous - but a derived winner cannot tell a
match that was PLAYED from one that was awarded.

NOT YET RESOLVED (10 matches, still in progress - they feed R64 slots 5,6,7,8,10):
    9  Jodar / Bu                14  de Jong / Passaro
    10 Marozsan / Zheng          15  Choinski / van de Zandschulp
    11 Svajda / Altmaier         16  Guerrieri / de Minaur
    12 Juan Manuel Cerundolo / Gea   19  Molcan / Bonzi
    13 Bergs / Taberner          20  Giron / Buse

Book went 38/54 this round - 16 favourites beaten, including Navone over
Djokovic (-588), Tsitsipas over Fils (-435) and Gorzny over Collignon (-769).

Usage:  python score_usopen2026_r128.py [reports_dir]   (default: reports)
"""
import sys, os
import pandas as pd
import unicodedata

def al(n):
    return unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode("ascii").strip().lower().replace("-", " ")

reports = sys.argv[1] if len(sys.argv) > 1 else "reports"

RESULTS = {
    ("Alexander Zverev", "Lorenzo Sonego"):                    "Alexander Zverev",          # 1
    ("Quentin Halys", "Facundo Diaz Acosta"):                  "Quentin Halys",             # 2
    ("Grigor Dimitrov", "Alexei Popyrin"):                     "Alexei Popyrin",            # 3
    ("Yannick Hanfmann", "Alejandro Tabilo"):                  "Alejandro Tabilo",          # 4
    ("Luciano Darderi", "Harry Wendelken"):                    "Luciano Darderi",           # 5
    ("Dalibor Svrcina", "Valentin Royer"):                     "Dalibor Svrcina",           # 6
    ("Dane Sweeny", "Corentin Moutet"):                        "Dane Sweeny",               # 7
    ("Arthur Fery", "Lorenzo Musetti"):                        "Lorenzo Musetti",           # 8
    ("Felix Auger Aliassime", "Rinky Hijikata"):               "Felix Auger Aliassime",     # 17
    ("Roman Andres Burruchaga", "Karen Khachanov"):            "Karen Khachanov",           # 18
    ("Jakub Mensik", "Shintaro Mochizuki"):                    "Jakub Mensik",              # 21
    ("Jurij Rodionov", "Giovanni Mpetshi Perricard"):          "Jurij Rodionov",            # 22
    ("Adolfo Daniel Vallejo", "Gael Monfils"):                 "Gael Monfils",              # 23
    ("Nuno Borges", "Learner Tien"):                           "Learner Tien",              # 24
    ("Taylor Fritz", "Darwin Blanch"):                         "Taylor Fritz",              # 25
    ("Mattia Bellucci", "Zsombor Piros"):                      "Mattia Bellucci",           # 26
    ("Camilo Ugo Carabelli", "Jan-Lennard Struff"):            "Jan-Lennard Struff",        # 27
    ("Filip Misolic", "Francisco Cerundolo"):                  "Francisco Cerundolo",       # 28
    ("Alexander Blockx", "Marcelo Tomas Barrios Vera"):        "Alexander Blockx",          # 29
    ("Juncheng Shang", "Marco Trungelliti"):                   "Marco Trungelliti",         # 30
    ("Nishesh Basavareddy", "Tristan Schoolkate"):             "Tristan Schoolkate",        # 31
    ("Francisco Comesana", "Flavio Cobolli"):                  "Flavio Cobolli",            # 32
    ("Daniil Medvedev", "Hugo Gaston"):                        "Daniil Medvedev",           # 33
    ("Sebastian Gorzny", "Raphael Collignon"):                 "Sebastian Gorzny",          # 34
    ("Jaume Munar", "Terence Atmane"):                         "Jaume Munar",               # 35
    ("Sho Shimabukuro", "Arthur Rinderknech"):                 "Arthur Rinderknech",        # 36
    ("Valentin Vacherot", "Aleksandar Kovacevic"):             "Valentin Vacherot",         # 37
    ("Kamil Majchrzak", "Hamad Medjedovic"):                   "Kamil Majchrzak",           # 38
    ("Aleksandar Vukic", "Rei Sakamoto"):                      "Rei Sakamoto",              # 39
    ("Martin Damm", "Frances Tiafoe"):                         "Frances Tiafoe",            # 40
    ("Brandon Nakashima", "Sebastian Baez"):                   "Brandon Nakashima",         # 41
    ("Alex Michelsen", "Federico Cina"):                       "Alex Michelsen",            # 42
    ("Daniel Merida", "Marton Fucsovics"):                     "Daniel Merida",             # 43
    ("Otto Virtanen", "Andrey Rublev"):                        "Andrey Rublev",             # 44
    ("Tomas Martin Etcheverry", "Vit Kopriva"):                "Tomas Martin Etcheverry",   # 45
    ("Martin Landaluce", "Jacob Fearnley"):                    "Jacob Fearnley",            # 46
    ("Matteo Berrettini", "Stanislas Wawrinka"):               "Matteo Berrettini",         # 47
    ("Mariano Navone", "Novak Djokovic"):                      "Mariano Navone",            # 48
    ("Ben Shelton", "Tallon Griekspoor"):                      "Ben Shelton",               # 49
    ("Damir Dzumhur", "Hubert Hurkacz"):                       "Hubert Hurkacz",            # 50
    ("Miomir Kecmanovic", "Denis Shapovalov"):                 "Denis Shapovalov",          # 51
    ("Luca Van Assche", "Cameron Norrie"):                     "Luca Van Assche",           # 52
    ("Jiri Lehecka", "Pablo Carreno Busta"):                   "Jiri Lehecka",              # 53
    ("Toby Samuel", "Tomas Machac"):                           "Toby Samuel",               # 54
    ("Lloyd Harris", "Jack Kennedy"):                          "Lloyd Harris",              # 55
    ("Stefanos Tsitsipas", "Arthur Fils"):                     "Stefanos Tsitsipas",        # 56
    ("Alexander Bublik", "J.J. Wolf"):                         "Alexander Bublik",          # 57
    ("Thiago Agustin Tirante", "Adrian Mannarino"):            "Adrian Mannarino",          # 58
    ("Dino Prizmic", "Aleksandr Shevchenko"):                  "Dino Prizmic",              # 59
    ("Coleman Wong", "Tommy Paul"):                            "Tommy Paul",                # 60
    ("Matteo Arnaldi", "James Duckworth"):                     "James Duckworth",           # 61
    ("Yibing Wu", "Adam Walton"):                              "Yibing Wu",                 # 62
    ("Jaime Faria", "Jenson Brooksby"):                        "Jaime Faria",               # 63
    ("Roman Safiullin", "Carlos Alcaraz"):                     "Carlos Alcaraz",            # 64
}

RES = {frozenset([al(a), al(b)]): al(w) for (a, b), w in RESULTS.items()}
assert len(RES) == 54, f"expected 54 unique matches, built {len(RES)}"

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

print("Scoring US Open 2026 R128 (54 of 64; 10 still in progress)...")
score_file(os.path.join(reports, "usopen2026_R128_predictions.csv"),
           os.path.join(reports, "usopen2026_R128_predictions_complete.csv"))
cck = score_file(os.path.join(reports, "usopen2026_R128_predictions_cck.csv"),
                 os.path.join(reports, "usopen2026_R128_predictions_cck_complete.csv"))
if cck is not None:
    bk = cck[cck["correct_prediction_book"].notna()]
    if len(bk):
        print(f"\nBook: {int(bk['correct_prediction_book'].sum())}/{len(bk)} = "
              f"{bk['correct_prediction_book'].mean()*100:.0f}%")
    print("\nNOTE: winners derived from R64 pairings, not score lines.")
    print("      10 matches remain unscored; re-run once they resolve.")
