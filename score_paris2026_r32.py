#!/usr/bin/env python3
"""
Score Paris 2026 R32.

All 16 matches resolved. Winners derived from the R16 field: every player
contesting an R16 match necessarily won their R32 match. Each of the 8 R16
pairings was asserted against adjacent R32 bracket slots - match k must pair
the winners of R32 (2k-1, 2k), in that order - and all 8 hold.

This round ran 10 upsets in 16, including Medvedev losing at -800 and
De Minaur at -509. The results are read off the R16 field, not inferred
from form.

Usage:  python score_paris2026_r32.py [reports_dir]   (default: reports)
"""
import sys, os
import pandas as pd
import unicodedata


def al(n):
    return (unicodedata.normalize("NFKD", str(n))
            .encode("ascii", "ignore").decode("ascii")
            .strip().lower().replace("-", " "))


reports = sys.argv[1] if len(sys.argv) > 1 else "reports"

EXPECTED = 16

RESULTS = {
    ("Alexander Zverev",        "Lorenzo Musetti"    ): "Lorenzo Musetti",
    ("Luca Van Assche",         "Learner Tien"       ): "Luca Van Assche",
    ("Novak Djokovic",          "Rinky Hijikata"     ): "Rinky Hijikata",
    ("Tomas Martin Etcheverry", "Alex De Minaur"     ): "Tomas Martin Etcheverry",
    ("Ben Shelton",             "Cameron Norrie"     ): "Ben Shelton",
    ("Mattia Bellucci",         "Arthur Rinderknech" ): "Arthur Rinderknech",
    ("Tommy Paul",              "Daniel Merida"      ): "Tommy Paul",
    ("Quentin Halys",           "Frances Tiafoe"     ): "Quentin Halys",
    ("Thiago Agustin Tirante",  "Jan-Lennard Struff" ): "Jan-Lennard Struff",
    ("Casper Ruud",             "Rafael Jodar"       ): "Casper Ruud",
    ("Nuno Borges",             "Karen Khachanov"    ): "Nuno Borges",
    ("Ignacio Buse",            "Daniil Medvedev"    ): "Ignacio Buse",
    ("Arthur Fils",             "Tomas Machac"       ): "Arthur Fils",
    ("Alex Molcan",             "Taylor Fritz"       ): "Taylor Fritz",
    ("Matteo Berrettini",       "Ugo Humbert"        ): "Matteo Berrettini",
    ("Alex Michelsen",          "Carlos Alcaraz"     ): "Carlos Alcaraz",
}

assert len(RESULTS) == EXPECTED, f"RESULTS holds {len(RESULTS)}, expected {EXPECTED}"
for (a, b), w in RESULTS.items():
    assert w in (a, b), f"winner {w!r} is not one of {a!r} / {b!r}"

RES = {frozenset([al(a), al(b)]): al(w) for (a, b), w in RESULTS.items()}
assert len(RES) == EXPECTED, "duplicate player pair in RESULTS after normalisation"


def score_file(path, write_complete_to, seen):
    if not os.path.exists(path):
        print(f"  skip (not found): {path}")
        return None
    df = pd.read_csv(path)
    scored = 0
    for i, r in df.iterrows():
        key = frozenset([al(r["player_a"]), al(r["player_b"])])
        if key not in RES:
            continue
        seen.add(key)
        winner = RES[key]
        if pd.isna(r.get("pred_winner")):
            continue
        df.at[i, "correct_prediction"] = 1 if al(r["pred_winner"]) == winner else 0
        oa, ob = r.get("odds_player_a"), r.get("odds_player_b")
        if pd.notna(oa) and pd.notna(ob):
            book_pick = al(r["player_a"]) if oa < ob else al(r["player_b"])
            df.at[i, "correct_prediction_book"] = 1 if book_pick == winner else 0
        scored += 1

    if scored == 0:
        print(f"  ABORT: {os.path.basename(path)} matched 0 results - "
              f"name mismatch or wrong file. Nothing written.")
        return None

    df.to_csv(write_complete_to, index=False)
    sc = df[df["correct_prediction"].notna()]
    acc = sc["correct_prediction"].mean() * 100 if len(sc) else 0
    print(f"  {os.path.basename(write_complete_to)}: {scored} scored, "
          f"model {int(sc['correct_prediction'].sum())}/{len(sc)} = {acc:.0f}%")
    return df


print("Scoring Paris 2026 R32...")
seen = set()

score_file(os.path.join(reports, "paris2026_R32_predictions.csv"),
           os.path.join(reports, "paris2026_R32_predictions_complete.csv"),
           seen)

cck = score_file(os.path.join(reports, "paris2026_R32_predictions_cck.csv"),
                 os.path.join(reports, "paris2026_R32_predictions_cck_complete.csv"),
                 seen)

if cck is not None:
    bk = cck[cck["correct_prediction_book"].notna()]
    if len(bk):
        print(f"\nBook: {int(bk['correct_prediction_book'].sum())}/{len(bk)} = "
              f"{bk['correct_prediction_book'].mean()*100:.0f}%")

unmatched = set(RES) - seen
if unmatched:
    print(f"\n{len(unmatched)} result(s) in RESULTS never matched a prediction row:")
    inv = {frozenset([al(a), al(b)]): (a, b) for (a, b) in RESULTS}
    for k in unmatched:
        a, b = inv[k]
        print(f"  {a} vs {b}")
    print("  -> check spelling against the draw CSV before trusting the totals")
else:
    print(f"\nAll {EXPECTED} results matched.")
