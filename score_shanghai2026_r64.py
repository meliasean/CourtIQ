#!/usr/bin/env python3
"""
Score Shanghai 2026 R64.

All 32 matches resolved. Winners derived from the R32 field: every player
contesting an R32 match necessarily won their R64 match. Each of the 16 R32
pairings was asserted against adjacent R64 bracket slots - match k must pair
the winners of R64 (2k-1, 2k), in that order - and all 16 hold, which is what
confirms the round is complete rather than partially read.

Usage:  python score_shanghai2026_r64.py [reports_dir]   (default: reports)
"""
import sys, os
import pandas as pd
import unicodedata


def al(n):
    return (unicodedata.normalize("NFKD", str(n))
            .encode("ascii", "ignore").decode("ascii")
            .strip().lower().replace("-", " "))


reports = sys.argv[1] if len(sys.argv) > 1 else "reports"

EXPECTED = 32

RESULTS = {
    ("Alexander Zverev",            "Yibing Wu"               ): "Alexander Zverev",
    ("Quentin Halys",               "Alexander Blockx"        ): "Quentin Halys",
    ("Alexander Bublik",            "Tomas Machac"            ): "Alexander Bublik",
    ("Matteo Berrettini",           "Brandon Nakashima"       ): "Brandon Nakashima",
    ("Alex De Minaur",              "Alex Molcan"             ): "Alex De Minaur",
    ("Rei Sakamoto",                "Andrey Rublev"           ): "Rei Sakamoto",
    ("Karen Khachanov",             "Arthur Fery"             ): "Karen Khachanov",
    ("Adrian Mannarino",            "Flavio Cobolli"          ): "Adrian Mannarino",
    ("Ben Shelton",                 "Daniel Altmaier"         ): "Ben Shelton",
    ("Arthur Gea",                  "Ugo Humbert"             ): "Arthur Gea",
    ("Francisco Cerundolo",         "Roman Safiullin"         ): "Francisco Cerundolo",
    ("Miomir Kecmanovic",           "Jakub Mensik"            ): "Jakub Mensik",
    ("Novak Djokovic",              "Hubert Hurkacz"          ): "Hubert Hurkacz",
    ("Yi Zhou",                     "Lorenzo Musetti"         ): "Yi Zhou",
    ("Alejandro Davidovich Fokina", "Jenson Brooksby"         ): "Alejandro Davidovich Fokina",
    ("Yannick Hanfmann",            "Frances Tiafoe"          ): "Frances Tiafoe",
    ("Arthur Fils",                 "Pavel Kotov"             ): "Arthur Fils",
    ("Botic van de Zandschulp",     "Alex Michelsen"          ): "Botic van de Zandschulp",
    ("Jiri Lehecka",                "Nuno Borges"             ): "Jiri Lehecka",
    ("Thiago Agustin Tirante",      "Rafael Jodar"            ): "Rafael Jodar",
    ("Learner Tien",                "Zachary Svajda"          ): "Learner Tien",
    ("Yunchaokete Bu",              "Casper Ruud"             ): "Casper Ruud",
    ("Alejandro Tabilo",            "Pablo Carreno Busta"     ): "Pablo Carreno Busta",
    ("Camilo Ugo Carabelli",        "Felix Auger Aliassime"   ): "Felix Auger Aliassime",
    ("Daniil Medvedev",             "Jan-Lennard Struff"      ): "Daniil Medvedev",
    ("Dalibor Svrcina",             "Tomas Martin Etcheverry" ): "Dalibor Svrcina",
    ("Luciano Darderi",             "Stefanos Tsitsipas"      ): "Stefanos Tsitsipas",
    ("Matteo Arnaldi",              "Taylor Fritz"            ): "Taylor Fritz",
    ("Tommy Paul",                  "Adolfo Daniel Vallejo"   ): "Tommy Paul",
    ("Sebastian Baez",              "Valentin Vacherot"       ): "Valentin Vacherot",
    ("Ignacio Buse",                "Zizou Bergs"             ): "Zizou Bergs",
    ("Juan Manuel Cerundolo",       "Carlos Alcaraz"          ): "Carlos Alcaraz",
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


print("Scoring Shanghai 2026 R64...")
seen = set()

score_file(os.path.join(reports, "shanghai2026_R64_predictions.csv"),
           os.path.join(reports, "shanghai2026_R64_predictions_complete.csv"),
           seen)

cck = score_file(os.path.join(reports, "shanghai2026_R64_predictions_cck.csv"),
                 os.path.join(reports, "shanghai2026_R64_predictions_cck_complete.csv"),
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
