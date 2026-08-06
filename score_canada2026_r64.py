#!/usr/bin/env python3
"""
Score Canada Masters 2026 R64 - 31 of 32 matches.

PROVENANCE - READ THIS
----------------------
Winners are DERIVED from the R32 pairings (odds feed, 2026-08-06), not read
from R64 score lines. Every R32 player maps to exactly one R64 match and all
16 R32 slots resolve to consecutive R64 bracket positions, so the mapping is
unambiguous. But a derived winner cannot distinguish a match that was PLAYED
from one that was awarded.

Independently confirmed from score lines:
    Shang 2-1 Rublev        (R64 28)
    Lehecka 2-0 Kopriva     (R64 17)
    Nakashima 2-0 Altmaier  (R64 31)
    Munar - Blockx          (R64 18)  WALKOVER
    Droguet 2-0 Faria       (R64 32)

MATCH 32 SUBSTITUTION:
    Auger-Aliassime withdrew injured. Droguet played Jaime Faria (lucky
    loser) instead and won 2-0 as the +129 underdog, so the book was wrong
    here. The original Droguet/Auger-Aliassime pick had no action and is
    voided. The replacement matchup was predicted only after it had been
    played, so any pick logged for it is post-hoc and must also be voided -
    the prediction is kept for accuracy, the pick is not counted.

EXCLUDED:
    R64 18 Munar / Blockx. Munar withdrew pre-match; Blockx advanced by
    walkover. No match was played, so it must not enter model or book
    accuracy. The logged pick on it is already voided.

If any OTHER match ended in a walkover or retirement, this script cannot
see it and will score it as a normal result. Spot-check before trusting
the numbers, or re-run against a score feed when one is available.

PICKS RESOLVING THIS ROUND (the two known live ones):
    Borges over Etcheverry .... Borges advanced       WIN
    Cobolli over Hanfmann ..... Hanfmann advanced     LOSS

Usage:  python score_canada2026_r64.py [reports_dir]   (default: reports)
"""
import sys, os
import pandas as pd
import unicodedata

def al(n):
    return unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode("ascii").strip().lower().replace("-", " ")

reports = sys.argv[1] if len(sys.argv) > 1 else "reports"

# (player_a, player_b): winner        # bracket slot
RESULTS = {
    ("Alexander Zverev", "Tallon Griekspoor"):          "Tallon Griekspoor",       # 1
    ("Fabian Marozsan", "Matteo Arnaldi"):              "Matteo Arnaldi",          # 2
    ("Ugo Humbert", "Daniel Merida"):                   "Daniel Merida",           # 3
    ("Alex Michelsen", "Francisco Cerundolo"):          "Alex Michelsen",          # 4
    ("Learner Tien", "Gael Monfils"):                   "Learner Tien",            # 5
    ("Valentin Royer", "Tommy Paul"):                   "Tommy Paul",              # 6
    ("Raphael Collignon", "Alexei Popyrin"):            "Alexei Popyrin",          # 7
    ("Thiago Agustin Tirante", "Taylor Fritz"):         "Thiago Agustin Tirante",  # 8
    ("Daniil Medvedev", "Botic van de Zandschulp"):     "Botic van de Zandschulp", # 9
    ("Hubert Hurkacz", "Alejandro Tabilo"):             "Hubert Hurkacz",          # 10
    ("Karen Khachanov", "Terence Atmane"):              "Terence Atmane",          # 11
    ("Jacob Fearnley", "Jakub Mensik"):                 "Jakub Mensik",            # 12
    ("Casper Ruud", "Juan Manuel Cerundolo"):           "Casper Ruud",             # 13
    ("Stefanos Tsitsipas", "Joao Fonseca"):             "Joao Fonseca",            # 14
    ("Zizou Bergs", "Sebastian Baez"):                  "Zizou Bergs",             # 15
    ("Jenson Brooksby", "Ben Shelton"):                 "Ben Shelton",             # 16
    ("Jiri Lehecka", "Vit Kopriva"):                    "Jiri Lehecka",            # 17  confirmed 2-0
    # 18 Munar / Blockx  -- WALKOVER, deliberately absent
    ("Rafael Jodar", "Corentin Moutet"):                "Rafael Jodar",            # 19
    ("Nicolas Mejia", "Lorenzo Musetti"):               "Lorenzo Musetti",         # 20
    ("Valentin Vacherot", "Mariano Navone"):            "Mariano Navone",          # 21
    ("Zachary Svajda", "Arthur Fils"):                  "Arthur Fils",             # 22
    ("Ignacio Buse", "Cameron Norrie"):                 "Cameron Norrie",          # 23
    ("James Duckworth", "Alex De Minaur"):              "Alex De Minaur",          # 24
    ("Flavio Cobolli", "Yannick Hanfmann"):             "Yannick Hanfmann",        # 25
    ("Nuno Borges", "Tomas Martin Etcheverry"):         "Nuno Borges",             # 26
    ("Luciano Darderi", "Gabriel Diallo"):              "Luciano Darderi",         # 27
    ("Juncheng Shang", "Andrey Rublev"):                "Juncheng Shang",          # 28  confirmed 2-1
    ("Frances Tiafoe", "Marin Cilic"):                  "Frances Tiafoe",          # 29
    ("Miomir Kecmanovic", "Arthur Rinderknech"):        "Arthur Rinderknech",      # 30
    ("Brandon Nakashima", "Daniel Altmaier"):           "Brandon Nakashima",       # 31  confirmed 2-0
    ("Titouan Droguet", "Jaime Faria"):                 "Titouan Droguet",         # 32  see note
}

RES = {frozenset([al(a), al(b)]): al(w) for (a, b), w in RESULTS.items()}
assert len(RES) == 31, f"expected 31 unique matches (32 minus the walkover), built {len(RES)}"

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
        oa, ob = r.get("odds_player_a", r.get("odds_a")), r.get("odds_player_b", r.get("odds_b"))
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

print("Scoring Canada Masters 2026 R64 (31 of 32; Munar/Blockx walkover excluded)...")
score_file(os.path.join(reports, "canada2026_R64_predictions.csv"),
           os.path.join(reports, "canada2026_R64_predictions_complete.csv"))
cck = score_file(os.path.join(reports, "canada2026_R64_predictions_cck.csv"),
                 os.path.join(reports, "canada2026_R64_predictions_cck_complete.csv"))
if cck is not None:
    bk = cck[cck["correct_prediction_book"].notna()]
    if len(bk):
        print(f"\nBook: {int(bk['correct_prediction_book'].sum())}/{len(bk)} = "
              f"{bk['correct_prediction_book'].mean()*100:.0f}%")
    print("\nNOTE: winners derived from R32 pairings, not score lines.")
    print("      A walkover or retirement in any match other than Munar/Blockx")
    print("      would be scored here as a normal result. Spot-check.")
