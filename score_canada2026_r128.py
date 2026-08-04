#!/usr/bin/env python3
"""
Score Canada Masters 2026 R128 (partial - 20 of 32 completed).
Results transcribed from odds-feed paste supplied by Sean, 2026-08-04.
Winner read literally from displayed score (first player -> first score).

NOT INCLUDED (still in play at time of transcription):
    Duncan Chan vs Thiago Agustin Tirante   (1-1)
    Hubert Hurkacz vs Marcos Giron          (1-1)

Usage:  python score_canada2026_r128.py [reports_dir]   (default: reports)
"""
import sys, os
import pandas as pd
import unicodedata

def al(n):
    return unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode("ascii").strip().lower().replace("-", " ")

reports = sys.argv[1] if len(sys.argv) > 1 else "reports"

# (player_a, player_b): winner
RESULTS = {
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
        print("     -> check spelling against the CSV before trusting these numbers.")

    if scored == 0:
        print(f"  ABORT: nothing matched in {os.path.basename(path)}; not writing.")
        return None

    df.to_csv(write_complete_to, index=False)
    sc = df[df["correct_prediction"].notna()]
    acc = sc["correct_prediction"].mean() * 100 if len(sc) else 0
    print(f"  {os.path.basename(write_complete_to)}: {scored} scored, model {int(sc['correct_prediction'].sum())}/{len(sc)} = {acc:.0f}%")
    return df

print("Scoring Canada Masters 2026 R128 (partial)...")
score_file(os.path.join(reports, "canada2026_R128_predictions.csv"),
           os.path.join(reports, "canada2026_R128_predictions_complete.csv"))
cck = score_file(os.path.join(reports, "canada2026_R128_predictions_cck.csv"),
                 os.path.join(reports, "canada2026_R128_predictions_cck_complete.csv"))
if cck is not None:
    bk = cck[cck["correct_prediction_book"].notna()]
    if len(bk):
        print(f"\nBook: {int(bk['correct_prediction_book'].sum())}/{len(bk)} = {bk['correct_prediction_book'].mean()*100:.0f}%")
    print("\nReminder: Chan/Tirante and Hurkacz/Giron still unscored.")
