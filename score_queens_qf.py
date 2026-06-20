#!/usr/bin/env python3
"""
Score Queens 2026 QF. All 4 results inferred from SF matchups.
Usage:  python score_queens_qf.py [reports_dir]   (default: reports)
"""
import sys, os
import pandas as pd
import unicodedata

def al(n):
    return unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode("ascii").strip().replace("-", " ")

reports = sys.argv[1] if len(sys.argv) > 1 else "reports"

RESULTS = {
    ("Alex De Minaur", "Brandon Nakashima"): "Brandon Nakashima",          # UPSET (de Minaur -282)
    ("Arthur Fery", "Francisco Cerundolo"): "Francisco Cerundolo",
    ("Tommy Paul", "Alejandro Davidovich Fokina"): "Tommy Paul",
    ("Ugo Humbert", "Rinky Hijikata"): "Ugo Humbert",                       # Humbert -204
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

print("Scoring Queens 2026 QF...")
score_file(os.path.join(reports, "queens2026_QF_predictions.csv"),
           os.path.join(reports, "queens2026_QF_predictions_complete.csv"))
cck = score_file(os.path.join(reports, "queens2026_QF_predictions_cck.csv"),
                 os.path.join(reports, "queens2026_QF_predictions_cck_complete.csv"))
if cck is not None:
    bk = cck[cck["correct_prediction_book"].notna()]
    if len(bk):
        print(f"\nBook: {int(bk['correct_prediction_book'].sum())}/{len(bk)} = {bk['correct_prediction_book'].mean()*100:.0f}%")
print(f"\nTotal results defined: {len(RESULTS)}")
