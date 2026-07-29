#!/usr/bin/env python3
"""
Score Washington 2026 R32. 15 of 16 results resolved; De Minaur vs Tsitsipas pending.
Uncomment the De Minaur/Tsitsipas entry when it finishes.
Usage:  python score_washington_r32.py [reports_dir]   (default: reports)
"""
import sys, os
import pandas as pd
import unicodedata

def al(n):
    return unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode("ascii").strip().lower().replace("-", " ")

reports = sys.argv[1] if len(sys.argv) > 1 else "reports"

RESULTS = {
    ("Brandon Nakashima", "Tomas Martin Etcheverry"): "Brandon Nakashima",
    ("Lorenzo Musetti", "Matteo Arnaldi"): "Lorenzo Musetti",
    ("Trevor Svajda", "Jakub Mensik"): "Jakub Mensik",
    ("Taylor Fritz", "Zizou Bergs"): "Taylor Fritz",
    ("Kamil Majchrzak", "Tommy Paul"): "Kamil Majchrzak",
    ("Alex Michelsen", "Mackenzie McDonald"): "Alex Michelsen",
    ("Adrian Mannarino", "Learner Tien"): "Adrian Mannarino",
    ("Arthur Fils", "Rafael Jodar"): "Rafael Jodar",
    ("Kei Nishikori", "Juncheng Shang"): "Kei Nishikori",
    ("Aleksandar Vukic", "Zachary Svajda"): "Aleksandar Vukic",
    ("Frances Tiafoe", "Terence Atmane"): "Terence Atmane",
    ("Alejandro Tabilo", "Tallon Griekspoor"): "Alejandro Tabilo",
    ("Ugo Humbert", "Andres Martin"): "Ugo Humbert",
    ("Martin Damm", "Ben Shelton"): "Ben Shelton",
    ("Marcos Giron", "Cruz Hewitt"): "Cruz Hewitt",
    # ── PENDING (uncomment when finished) ──
    # ("Alex de Minaur", "Stefanos Tsitsipas"): "?",
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

print("Scoring Washington 2026 R32...")
score_file(os.path.join(reports, "washington2026_R32_predictions.csv"),
           os.path.join(reports, "washington2026_R32_predictions_complete.csv"))
cck = score_file(os.path.join(reports, "washington2026_R32_predictions_cck.csv"),
                 os.path.join(reports, "washington2026_R32_predictions_cck_complete.csv"))
if cck is not None:
    bk = cck[cck["correct_prediction_book"].notna()]
    if len(bk):
        print(f"\nBook: {int(bk['correct_prediction_book'].sum())}/{len(bk)} = {bk['correct_prediction_book'].mean()*100:.0f}%")
