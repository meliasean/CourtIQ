#!/usr/bin/env python3
"""
Re-score Queens 2026 R32 after regenerating predictions.

Usage:  python score_queens_r32.py [reports_dir]   (default: reports)
"""
import sys, os
import pandas as pd
import unicodedata

def al(n):
    return unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode("ascii").strip().replace("-", " ")

reports = sys.argv[1] if len(sys.argv) > 1 else "reports"

# ---- KNOWN RESULTS ----
RESULTS = {
    ("Alex De Minaur", "Gabriel Diallo"): "Alex De Minaur",
    ("Jack Pinnington Jones", "Denis Shapovalov"): "Denis Shapovalov",
    ("Brandon Nakashima", "Marton Fucsovics"): "Brandon Nakashima",
    ("Ignacio Buse", "Marcos Giron"): "Ignacio Buse",                          # Giron replaced Jodar in draw
    ("Jakub Mensik", "Adrian Mannarino"): "Adrian Mannarino",                  # UPSET (Mensik -229)
    ("Arthur Fery", "Toby Samuel"): "Arthur Fery",
    ("Jenson Brooksby", "Martin Damm"): "Jenson Brooksby",
    ("Aleksandar Kovacevic", "Francisco Cerundolo"): "Francisco Cerundolo",
    ("Tommy Paul", "Zachary Svajda"): "Tommy Paul",
    ("Botic van de Zandschulp", "Harry Wendelken"): "Botic van de Zandschulp",
    ("Giovanni Mpetshi Perricard", "Corentin Moutet"): "Corentin Moutet",
    ("Cameron Norrie", "Alejandro Davidovich Fokina"): "Alejandro Davidovich Fokina",
    ("Arthur Rinderknech", "Hamad Medjedovic"): "Hamad Medjedovic",            # mild upset (Rinderknech -133)
    ("Ugo Humbert", "Marin Cilic"): "Ugo Humbert",
    ("Rinky Hijikata", "Alejandro Tabilo"): "Rinky Hijikata",
    ("Kamil Majchrzak", "Jiri Lehecka"): "Jiri Lehecka",
}

RES = {frozenset([al(a), al(b)]): al(w) for (a, b), w in RESULTS.items()}

def score_file(path, write_complete_to):
    if not os.path.exists(path):
        print(f"  skip (not found): {path}"); return None
    df = pd.read_csv(path)
    scored = 0
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
        scored += 1
    df.to_csv(write_complete_to, index=False)
    sc = df[df["correct_prediction"].notna()]
    acc = sc["correct_prediction"].mean() * 100 if len(sc) else 0
    print(f"  {os.path.basename(write_complete_to)}: {scored} scored, model {int(sc['correct_prediction'].sum())}/{len(sc)} = {acc:.0f}%")
    return df

print("Scoring Queens 2026 R32...")
std = score_file(os.path.join(reports, "queens2026_R32_predictions.csv"),
                 os.path.join(reports, "queens2026_R32_predictions_complete.csv"))
cck = score_file(os.path.join(reports, "queens2026_R32_predictions_cck.csv"),
                 os.path.join(reports, "queens2026_R32_predictions_cck_complete.csv"))

if cck is not None:
    bk = cck[cck["correct_prediction_book"].notna()]
    if len(bk):
        print(f"\nBook: {int(bk['correct_prediction_book'].sum())}/{len(bk)} = {bk['correct_prediction_book'].mean()*100:.0f}%")
print(f"\nTotal results defined: {len(RESULTS)}")
