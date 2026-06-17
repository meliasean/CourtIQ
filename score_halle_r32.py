#!/usr/bin/env python3
"""
Re-score Halle 2026 R32 after regenerating predictions.

Usage:  python score_halle_r32.py [reports_dir]   (default: reports)
"""
import sys, os
import pandas as pd
import unicodedata

def al(n):
    return unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode("ascii").strip().replace("-", " ")

reports = sys.argv[1] if len(sys.argv) > 1 else "reports"

# ---- KNOWN RESULTS ----
RESULTS = {
    ("Alexander Zverev", "Vit Kopriva"): "Alexander Zverev",
    ("Joao Fonseca", "Yannick Hanfmann"): "Yannick Hanfmann",                  # UPSET (Fonseca -251)
    ("Alexei Popyrin", "Raphael Collignon"): "Raphael Collignon",
    ("Mattia Bellucci", "Alexander Bublik"): "Mattia Bellucci",                # UPSET (Bublik -340)
    ("Ben Shelton", "Lorenzo Sonego"): "Ben Shelton",
    ("Karen Khachanov", "Ethan Quinn"): "Ethan Quinn",                         # UPSET (Khachanov -306)
    ("Fabian Marozsan", "Miomir Kecmanovic"): "Fabian Marozsan",
    ("Zizou Bergs", "Taylor Fritz"): "Taylor Fritz",
    ("Andrey Rublev", "Hubert Hurkacz"): "Hubert Hurkacz",                     # mild upset (Rublev -128)
    ("Nikoloz Basilashvili", "Daniel Altmaier"): "Daniel Altmaier",
    ("Terence Atmane", "Martin Landaluce"): "Terence Atmane",
    ("Tomas Martin Etcheverry", "Daniil Medvedev"): "Daniil Medvedev",
    ("Flavio Cobolli", "Frances Tiafoe"): "Frances Tiafoe",
    ("Tallon Griekspoor", "Sho Shimabukuro"): "Sho Shimabukuro",               # mild upset (Griekspoor -164)
    ("Max Schoenhaus", "Learner Tien"): "Learner Tien",
    ("Nuno Borges", "Felix Auger-Aliassime"): "Felix Auger-Aliassime",
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

print("Scoring Halle 2026 R32...")
std = score_file(os.path.join(reports, "halle2026_R32_predictions.csv"),
                 os.path.join(reports, "halle2026_R32_predictions_complete.csv"))
cck = score_file(os.path.join(reports, "halle2026_R32_predictions_cck.csv"),
                 os.path.join(reports, "halle2026_R32_predictions_cck_complete.csv"))

if cck is not None:
    bk = cck[cck["correct_prediction_book"].notna()]
    if len(bk):
        print(f"\nBook: {int(bk['correct_prediction_book'].sum())}/{len(bk)} = {bk['correct_prediction_book'].mean()*100:.0f}%")
print(f"\nTotal results defined: {len(RESULTS)}")
