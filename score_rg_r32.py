#!/usr/bin/env python3
"""
Re-score RG 2026 R32 after regenerating predictions.

The `predict` command writes BLANK prediction files (no results). This script
re-applies all known match results so you don't lose scoring after a regen.

Usage:
    python score_rg_r32.py [reports_dir]   (default: reports)

It reads  rg2026_R32_predictions.csv  and  rg2026_R32_predictions_cck.csv,
fills correct_prediction (model/cck) and correct_prediction_book for every
finished match below, and writes the *_complete.csv variants.

To add results as more matches finish: append (player_a, player_b): winner
to RESULTS and re-run.
"""
import sys, os
import pandas as pd
import numpy as np
import unicodedata

def al(n):
    return unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode("ascii").strip().replace("-", " ")

reports = sys.argv[1] if len(sys.argv) > 1 else "reports"

# ---- KNOWN RESULTS (winner of each finished match) -------------------------
RESULTS = {
    # === R32 — Day 1 (May 29) ===
    ("Juan Manuel Cerundolo", "Martin Landaluce"): "Juan Manuel Cerundolo",
    ("Matteo Berrettini", "Francisco Comesana"): "Matteo Berrettini",
    ("Felix Auger Aliassime", "Brandon Nakashima"): "Felix Auger Aliassime",
    ("Flavio Cobolli", "Learner Tien"): "Flavio Cobolli",
    ("Francisco Cerundolo", "Zachary Svajda"): "Zachary Svajda",        # UPSET (F.Cerundolo -909)
    ("Alex De Minaur", "Jakub Mensik"): "Jakub Mensik",                  # UPSET (De Minaur -400)
    ("Nuno Borges", "Andrey Rublev"): "Andrey Rublev",
    ("Casper Ruud", "Tommy Paul"): "Casper Ruud",
    ("Joao Fonseca", "Novak Djokovic"): "Joao Fonseca",                  # UPSET (Djokovic -217)
    ("Alex Michelsen", "Rafael Jodar"): "Rafael Jodar",
    ("Thiago Agustin Tirante", "Pablo Carreno Busta"): "Pablo Carreno Busta",
    ("Karen Khachanov", "Jesper de Jong"): "Jesper de Jong",
    ("Quentin Halys", "Alexander Zverev"): "Alexander Zverev",
    # === R32 — Day 2 (May 30) ===
    ("Jaime Faria", "Frances Tiafoe"): "Frances Tiafoe",
    ("Matteo Arnaldi", "Raphael Collignon"): "Matteo Arnaldi",
    ("Moise Kouame", "Alejandro Tabilo"): "Alejandro Tabilo",
}
# ---------------------------------------------------------------------------

RES = {frozenset([al(a), al(b)]): al(w) for (a, b), w in RESULTS.items()}

def score_file(path, write_complete_to):
    if not os.path.exists(path):
        print(f"  skip (not found): {path}")
        return None
    df = pd.read_csv(path)
    scored = 0
    unmatched_results = set(RES.keys())
    for i, r in df.iterrows():
        key = frozenset([al(r["player_a"]), al(r["player_b"])])
        if key not in RES:
            continue
        unmatched_results.discard(key)
        winner = RES[key]
        if pd.isna(r.get("pred_winner")):
            print(f"  WARN row {i} {r['player_a']} vs {r['player_b']}: result known but no prediction — skipped")
            continue
        df.at[i, "correct_prediction"] = 1 if al(r["pred_winner"]) == winner else 0
        oa, ob = r.get("odds_player_a"), r.get("odds_player_b")
        if pd.notna(oa) and pd.notna(ob):
            book_pick = al(r["player_a"]) if oa < ob else al(r["player_b"])
            df.at[i, "correct_prediction_book"] = 1 if book_pick == winner else 0
        scored += 1
    df.to_csv(write_complete_to, index=False)
    if unmatched_results:
        print(f"  NOTE: {len(unmatched_results)} results had no matching row in {os.path.basename(path)} "
              f"(opponent mismatch?)")
    sc = df[df["correct_prediction"].notna()]
    acc = sc["correct_prediction"].mean() * 100 if len(sc) else 0
    print(f"  {os.path.basename(write_complete_to)}: {scored} scored, model {int(sc['correct_prediction'].sum())}/{len(sc)} = {acc:.0f}%")
    return df

print("Scoring RG 2026 R32...")
std = score_file(os.path.join(reports, "rg2026_R32_predictions.csv"),
                 os.path.join(reports, "rg2026_R32_predictions_complete.csv"))
cck = score_file(os.path.join(reports, "rg2026_R32_predictions_cck.csv"),
                 os.path.join(reports, "rg2026_R32_predictions_cck_complete.csv"))

if cck is not None:
    bk = cck[cck["correct_prediction_book"].notna()]
    if len(bk):
        print(f"\nBook (from cck file): {int(bk['correct_prediction_book'].sum())}/{len(bk)} = {bk['correct_prediction_book'].mean()*100:.0f}%")
print(f"\nTotal results defined: {len(RESULTS)}")
print("Done. Complete files written. Verify, then regenerate site + push.")
