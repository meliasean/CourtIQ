#!/usr/bin/env python3
"""
Score Canada Masters 2026 R128 - COMPLETE (all 32 matches).

Supersedes the earlier partial version (20 matches). Winners read literally
from the ATP draw score lines supplied 2026-08-04, not inferred from R64.

RECONCILIATION: 32 matches, matching a 96-draw Masters R128 exactly.

RETIREMENT - verified, not inferred:
    Svajda / Shapovalov read 4-6, 6-4, 3-0. The third set is incomplete, so
    Shapovalov retired trailing 3-0 in the decider. Svajda appears vs Fils
    in R64, confirming the direction downstream.

PICKS THIS ROUND (7 logged, 6-1):
    Griekspoor over Sonego .................. WIN
    van de Zandschulp over Mpetshi Perricard  WIN
    Kecmanovic over Zheng ................... WIN
    Cilic over Shimabukuro .................. WIN
    Moutet over Fucsovics ................... WIN
    Kopriva over Galarneau .................. WIN
    Mannarino over Fearnley ................. LOSS

Usage:  python score_canada2026_r128.py [reports_dir]   (default: reports)
"""
import sys, os
import pandas as pd
import unicodedata

def al(n):
    return unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode("ascii").strip().lower().replace("-", " ")

reports = sys.argv[1] if len(sys.argv) > 1 else "reports"

RESULTS = {
    # --- newly resolved -------------------------------------------------
    ("Lorenzo Sonego", "Tallon Griekspoor"):          "Tallon Griekspoor",
    ("Shintaro Mochizuki", "Fabian Marozsan"):        "Fabian Marozsan",
    ("Daniel Merida", "Liam Draxl"):                  "Daniel Merida",
    ("Pablo Carreno Busta", "Valentin Royer"):        "Valentin Royer",
    ("Alexei Popyrin", "Roman Andres Burruchaga"):    "Alexei Popyrin",
    ("Duncan Chan", "Thiago Agustin Tirante"):        "Thiago Agustin Tirante",
    ("Giovanni Mpetshi Perricard", "Botic van de Zandschulp"): "Botic van de Zandschulp",
    ("Hubert Hurkacz", "Marcos Giron"):               "Hubert Hurkacz",
    ("Terence Atmane", "Jack Draper"):                "Terence Atmane",
    ("Jacob Fearnley", "Adrian Mannarino"):           "Jacob Fearnley",
    ("Juan Manuel Cerundolo", "Hamad Medjedovic"):    "Juan Manuel Cerundolo",
    ("Zachary Svajda", "Denis Shapovalov"):           "Zachary Svajda",   # Shapovalov ret. 0-3 in 3rd

    # --- already scored in the earlier pass (idempotent) ----------------
    ("Sebastian Baez", "Mattia Bellucci"):            "Sebastian Baez",
    ("Adam Walton", "Jenson Brooksby"):               "Jenson Brooksby",
    ("Martin Damm", "Stefanos Tsitsipas"):            "Stefanos Tsitsipas",
    ("Kamil Majchrzak", "Gael Monfils"):              "Gael Monfils",
    ("Camilo Ugo Carabelli", "Cameron Norrie"):       "Cameron Norrie",
    ("Marton Fucsovics", "Corentin Moutet"):          "Corentin Moutet",
    ("Alex Michelsen", "Jan-Lennard Struff"):         "Alex Michelsen",
    ("Aleksandar Vukic", "Daniel Altmaier"):          "Daniel Altmaier",
    ("Sho Shimabukuro", "Marin Cilic"):               "Marin Cilic",
    ("Michael Zheng", "Miomir Kecmanovic"):           "Miomir Kecmanovic",
    ("Gabriel Diallo", "Kyrian Jacquet"):             "Gabriel Diallo",
    ("Aleksandar Kovacevic", "Nuno Borges"):          "Nuno Borges",
    ("Benjamin Bonzi", "Yannick Hanfmann"):           "Yannick Hanfmann",
    ("James Duckworth", "Christopher O'Connell"):     "James Duckworth",
    ("Luca Van Assche", "Titouan Droguet"):           "Titouan Droguet",
    ("Nicolas Mejia", "Martin Landaluce"):            "Nicolas Mejia",
    ("Matteo Berrettini", "Mariano Navone"):          "Mariano Navone",
    ("Alexis Galarneau", "Vit Kopriva"):              "Vit Kopriva",
    ("Jaume Munar", "Rinky Hijikata"):                "Jaume Munar",
    ("Adolfo Daniel Vallejo", "Juncheng Shang"):      "Juncheng Shang",
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
        print("     -> spelling mismatch. Check these against the CSV before trusting the numbers.")

    if scored == 0:
        print(f"  ABORT: nothing matched in {os.path.basename(path)}; not writing.")
        return None

    df.to_csv(write_complete_to, index=False)
    sc = df[df["correct_prediction"].notna()]
    acc = sc["correct_prediction"].mean() * 100 if len(sc) else 0
    print(f"  {os.path.basename(write_complete_to)}: {scored} scored, "
          f"model {int(sc['correct_prediction'].sum())}/{len(sc)} = {acc:.0f}%")
    return df

print("Scoring Canada Masters 2026 R128 (complete, 32 matches)...")
score_file(os.path.join(reports, "canada2026_R128_predictions.csv"),
           os.path.join(reports, "canada2026_R128_predictions_complete.csv"))
cck = score_file(os.path.join(reports, "canada2026_R128_predictions_cck.csv"),
                 os.path.join(reports, "canada2026_R128_predictions_cck_complete.csv"))
if cck is not None:
    bk = cck[cck["correct_prediction_book"].notna()]
    if len(bk):
        print(f"\nBook: {int(bk['correct_prediction_book'].sum())}/{len(bk)} = "
              f"{bk['correct_prediction_book'].mean()*100:.0f}%")
    print("\nNOTE: Svajda advanced on Shapovalov's retirement (0-3 in the 3rd).")
    print("      Direction confirmed against R64: Svajda vs Fils.")
