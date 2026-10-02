#!/usr/bin/env python3
"""
Score Tokyo 2026 R32.

All 16 matches resolved. Winners derived from the R16 field: every player
contesting an R16 match necessarily won their R32 match. All eight R16
pairings were checked against adjacent R32 bracket slots - (1,2), (3,4),
(5,6), (7,8), (9,10), (11,12), (13,14), (15,16) - and every one lines up,
which is what confirms the round is complete rather than partially read.

Usage:  python score_tokyo2026_r32.py [reports_dir]   (default: reports)
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
    ("Carlos Alcaraz",               "Alex Michelsen"):       "Carlos Alcaraz",
    ("Matteo Arnaldi",               "Rei Sakamoto"):         "Matteo Arnaldi",
    ("Denis Shapovalov",             "Sho Shimabukuro"):      "Denis Shapovalov",
    ("Alejandro Tabilo",             "Tommy Paul"):           "Alejandro Tabilo",
    ("Taylor Fritz",                 "Jaume Munar"):          "Jaume Munar",
    ("Jaime Faria",                  "Arthur Fery"):          "Jaime Faria",
    ("Holger Rune",                  "Kyrian Jacquet"):       "Kyrian Jacquet",
    ("Luciano Darderi",              "Casper Ruud"):          "Luciano Darderi",
    ("Brandon Nakashima",            "Ugo Humbert"):          "Ugo Humbert",
    ("Jiri Lehecka",                 "Zizou Bergs"):          "Jiri Lehecka",
    ("Alejandro Davidovich Fokina",  "Matteo Berrettini"):    "Matteo Berrettini",
    ("Adolfo Daniel Vallejo",        "Rafael Jodar"):         "Adolfo Daniel Vallejo",
    ("Valentin Vacherot",            "Alexander Blockx"):     "Valentin Vacherot",
    ("Tomas Martin Etcheverry",      "Stefanos Tsitsipas"):   "Stefanos Tsitsipas",
    ("Arthur Fils",                  "Luca Van Assche"):      "Arthur Fils",
    ("Kei Nishikori",                "Frances Tiafoe"):       "Frances Tiafoe",
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


print("Scoring Tokyo 2026 R32...")
seen = set()

score_file(os.path.join(reports, "tokyo2026_R32_predictions.csv"),
           os.path.join(reports, "tokyo2026_R32_predictions_complete.csv"),
           seen)

cck = score_file(os.path.join(reports, "tokyo2026_R32_predictions_cck.csv"),
                 os.path.join(reports, "tokyo2026_R32_predictions_cck_complete.csv"),
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
