#!/usr/bin/env python3
"""
Score Paris 2026 R64.

PARTIAL ROUND - only 3 of 24 matches resolved.

Paris has 64 bracket slots, 8 of them byes for seeds 1-8, so 24 matches were
played in the R64. Winners here are derived from the two published R32
pairings: Shelton vs Norrie fixes Norrie, and Ruud vs Jodar fixes both Ruud
and Jodar. Each was asserted against its R64 slot-pair before being accepted.

STILL OPEN - deliberately absent from RESULTS:
    match 1  Lorenzo Musetti vs Hubert Hurkacz
    match 2  Luca Van Assche vs Mariano Navone
    match 3  Adolfo Daniel Vallejo vs Learner Tien
    match 4  Novak Djokovic vs Valentin Vacherot
    match 5  Luciano Darderi vs Rinky Hijikata
    match 6  Andrey Rublev vs Tomas Martin Etcheverry
    match 8  Mattia Bellucci vs Lorenzo Sonego
    match 9  Arthur Rinderknech vs Francisco Cerundolo
    match 10 Tommy Paul vs Holger Rune
    match 11 Daniel Merida vs Sebastian Baez
    match 12 Quentin Halys vs Arthur Fery
    match 13 Jan-Lennard Struff vs Arthur Gea
    match 16 Nuno Borges vs Moise Kouame
    match 17 Denis Shapovalov vs Karen Khachanov
    match 18 Ignacio Buse vs Zizou Bergs
    match 19 Kyrian Jacquet vs Tomas Machac
    match 20 Alex Molcan vs Titouan Droguet
    match 21 Benjamin Bonzi vs Taylor Fritz
    match 22 Brandon Nakashima vs Matteo Berrettini
    match 23 Ugo Humbert vs Botic van de Zandschulp
    match 24 Alex Michelsen vs Jiri Lehecka

Re-run this script as those land; it is safe to run repeatedly. Note that
re-running `predict` on a fuller R32 draw shifts every row's positional
match_no, so the earlier R32 log rows must be dropped and re-logged rather
than appended to.

Usage:  python score_paris2026_r64.py [reports_dir]   (default: reports)
"""
import sys, os
import pandas as pd
import unicodedata


def al(n):
    return (unicodedata.normalize("NFKD", str(n))
            .encode("ascii", "ignore").decode("ascii")
            .strip().lower().replace("-", " "))


reports = sys.argv[1] if len(sys.argv) > 1 else "reports"

EXPECTED = 3

RESULTS = {
    ("Cameron Norrie",   "Matteo Arnaldi"        ): "Cameron Norrie",
    ("Casper Ruud",      "Juan Manuel Cerundolo" ): "Casper Ruud",
    ("Alejandro Tabilo", "Rafael Jodar"          ): "Rafael Jodar",
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


print("Scoring Paris 2026 R64...")
seen = set()

score_file(os.path.join(reports, "paris2026_R64_predictions.csv"),
           os.path.join(reports, "paris2026_R64_predictions_complete.csv"),
           seen)

cck = score_file(os.path.join(reports, "paris2026_R64_predictions_cck.csv"),
                 os.path.join(reports, "paris2026_R64_predictions_cck_complete.csv"),
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
