#!/usr/bin/env python3
"""
Score Cincinnati 2026 Final.

RESULT (reported directly by Sean, 2026-08-23):
    Arthur Fils def. Frances Tiafoe  -  Fils takes the title.

Fils was the -210 favourite, so the book was right here.

Usage:  python score_cincinnati2026_f.py [reports_dir]   (default: reports)
"""
import sys, os
import pandas as pd
import unicodedata

def al(n):
    return unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode("ascii").strip().lower().replace("-", " ")

reports = sys.argv[1] if len(sys.argv) > 1 else "reports"

RESULTS = {
    ("Arthur Fils", "Frances Tiafoe"): "Arthur Fils",
}

RES = {frozenset([al(a), al(b)]): al(w) for (a, b), w in RESULTS.items()}
assert len(RES) == 1

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

    if scored == 0:
        print(f"  WARNING: no rows matched in {os.path.basename(path)} - check spellings:")
        for _, r in df.iterrows():
            print(f"    {r['player_a']} vs {r['player_b']}")
        return None

    df.to_csv(write_complete_to, index=False)
    sc = df[df["correct_prediction"].notna()]
    acc = sc["correct_prediction"].mean() * 100 if len(sc) else 0
    print(f"  {os.path.basename(write_complete_to)}: {scored} scored, "
          f"model {int(sc['correct_prediction'].sum())}/{len(sc)} = {acc:.0f}%")
    return df

print("Scoring Cincinnati 2026 Final...")
score_file(os.path.join(reports, "cincinnati2026_F_predictions.csv"),
           os.path.join(reports, "cincinnati2026_F_predictions_complete.csv"))
cck = score_file(os.path.join(reports, "cincinnati2026_F_predictions_cck.csv"),
                 os.path.join(reports, "cincinnati2026_F_predictions_cck_complete.csv"))
if cck is not None:
    bk = cck[cck["correct_prediction_book"].notna()]
    if len(bk):
        print(f"\nBook: {int(bk['correct_prediction_book'].sum())}/{len(bk)}")
