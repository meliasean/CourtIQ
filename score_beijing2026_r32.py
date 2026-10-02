#!/usr/bin/env python3
"""
Score Beijing 2026 R32.

14 of 16 matches resolved. Winners derived from the R16 field: every player
contesting an R16 match necessarily won their R32 match, and each R16 pairing
was checked against adjacent R32 bracket slots before being accepted.

STILL OPEN - deliberately absent from RESULTS:
    match 5  Daniil Medvedev vs Pablo Carreno Busta
    match 6  Jan-Lennard Struff vs Thiago Agustin Tirante

The R16 odds list re-priced Medvedev/Carreno Busta (-500/+366) rather than
pairing either of them against a Struff/Tirante winner, which means that
quarter of the draw had not resolved. Neither Struff nor Tirante appears in
the R16 field, so match 6 cannot be read either. Re-run this script once
both land; it is safe to run repeatedly.

Usage:  python score_beijing2026_r32.py [reports_dir]   (default: reports)
"""
import sys, os
import pandas as pd
import unicodedata


def al(n):
    return (unicodedata.normalize("NFKD", str(n))
            .encode("ascii", "ignore").decode("ascii")
            .strip().lower().replace("-", " "))


reports = sys.argv[1] if len(sys.argv) > 1 else "reports"

EXPECTED = 14

RESULTS = {
    ("Alexander Zverev",        "Cameron Norrie"):            "Alexander Zverev",
    ("Juncheng Shang",          "Sebastian Baez"):             "Juncheng Shang",
    ("Juan Manuel Cerundolo",   "Yunchaokete Bu"):             "Yunchaokete Bu",
    ("Nuno Borges",             "Novak Djokovic"):             "Novak Djokovic",
    ("Botic van de Zandschulp", "Francisco Cerundolo"):        "Francisco Cerundolo",
    ("Alexander Bublik",        "Jakub Mensik"):               "Jakub Mensik",
    ("Alex De Minaur",          "Mariano Navone"):             "Alex De Minaur",
    ("Ignacio Buse",            "Quentin Halys"):              "Quentin Halys",
    ("Tallon Griekspoor",       "Andrey Rublev"):              "Andrey Rublev",
    ("Roman Safiullin",         "Flavio Cobolli"):             "Roman Safiullin",
    ("Learner Tien",            "Hubert Hurkacz"):             "Hubert Hurkacz",
    ("Zhizhen Zhang",           "Arthur Gea"):                 "Arthur Gea",
    ("Alex Molcan",             "Tomas Machac"):               "Alex Molcan",
    ("Karen Khachanov",         "Felix Auger Aliassime"):      "Karen Khachanov",
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


print("Scoring Beijing 2026 R32...")
seen = set()

score_file(os.path.join(reports, "beijing2026_R32_predictions.csv"),
           os.path.join(reports, "beijing2026_R32_predictions_complete.csv"),
           seen)

cck = score_file(os.path.join(reports, "beijing2026_R32_predictions_cck.csv"),
                 os.path.join(reports, "beijing2026_R32_predictions_cck_complete.csv"),
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

print("\nStill open: match 5 Medvedev/Carreno Busta, match 6 Struff/Tirante.")
print("Re-run this script once those resolve.")
