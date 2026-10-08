#!/usr/bin/env python3
"""
Score Shanghai 2026 R128.

All 32 matches resolved. Winners derived from the R64 field: every player
contesting an R64 match necessarily won their R128 match. All 32 R64 pairings
were checked against adjacent R128 bracket slots, every R128 winner appears in
exactly one R64 match, and no R64 player who entered the R128 is absent from
the winner list - that three-way closure is what confirms the round is complete.

Shanghai is a 96 draw, so the 32 seeds sat on byes and only these 32 matches
were played in the R128.

Usage:  python score_shanghai2026_r128.py [reports_dir]   (default: reports)
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
    ("Yibing Wu",               "Michael Zheng"        ): "Yibing Wu",
    ("Quentin Halys",           "Coleman Wong"         ): "Quentin Halys",
    ("Zhizhen Zhang",           "Tomas Machac"         ): "Tomas Machac",
    ("Aleksandar Kovacevic",    "Matteo Berrettini"    ): "Matteo Berrettini",
    ("Alex Molcan",             "Federico Cina"        ): "Alex Molcan",
    ("Marco Trungelliti",       "Rei Sakamoto"         ): "Rei Sakamoto",
    ("Arthur Fery",             "Marin Cilic"          ): "Arthur Fery",
    ("Adrian Mannarino",        "Nikoloz Basilashvili" ): "Adrian Mannarino",
    ("Holger Rune",             "Daniel Altmaier"      ): "Daniel Altmaier",
    ("Arthur Gea",              "Jaime Faria"          ): "Arthur Gea",
    ("Rinky Hijikata",          "Roman Safiullin"      ): "Roman Safiullin",
    ("Sho Shimabukuro",         "Miomir Kecmanovic"    ): "Miomir Kecmanovic",
    ("Hubert Hurkacz",          "James Duckworth"      ): "Hubert Hurkacz",
    ("Mattia Bellucci",         "Yi Zhou"              ): "Yi Zhou",
    ("Jaume Munar",             "Jenson Brooksby"      ): "Jenson Brooksby",
    ("Yannick Hanfmann",        "Kamil Majchrzak"      ): "Yannick Hanfmann",
    ("Pavel Kotov",             "Tallon Griekspoor"    ): "Pavel Kotov",
    ("Botic van de Zandschulp", "Daniel Merida"        ): "Botic van de Zandschulp",
    ("Nuno Borges",             "Facundo Diaz Acosta"  ): "Nuno Borges",
    ("Thiago Agustin Tirante",  "Hamad Medjedovic"     ): "Thiago Agustin Tirante",
    ("Fabian Marozsan",         "Zachary Svajda"       ): "Zachary Svajda",
    ("Luca Van Assche",         "Yunchaokete Bu"       ): "Yunchaokete Bu",
    ("Mariano Navone",          "Pablo Carreno Busta"  ): "Pablo Carreno Busta",
    ("Camilo Ugo Carabelli",    "Ilia Simakin"         ): "Camilo Ugo Carabelli",
    ("Martin Landaluce",        "Jan-Lennard Struff"   ): "Jan-Lennard Struff",
    ("Cameron Norrie",          "Dalibor Svrcina"      ): "Dalibor Svrcina",
    ("Kimmer Coppejans",        "Stefanos Tsitsipas"   ): "Stefanos Tsitsipas",
    ("Bernard Tomic",           "Matteo Arnaldi"       ): "Matteo Arnaldi",
    ("Adolfo Daniel Vallejo",   "Valentin Royer"       ): "Adolfo Daniel Vallejo",
    ("Marcos Giron",            "Sebastian Baez"       ): "Sebastian Baez",
    ("Vit Kopriva",             "Zizou Bergs"          ): "Zizou Bergs",
    ("Juan Manuel Cerundolo",   "Nicolas Mejia"        ): "Juan Manuel Cerundolo",
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


print("Scoring Shanghai 2026 R128...")
seen = set()

score_file(os.path.join(reports, "shanghai2026_R128_predictions.csv"),
           os.path.join(reports, "shanghai2026_R128_predictions_complete.csv"),
           seen)

cck = score_file(os.path.join(reports, "shanghai2026_R128_predictions_cck.csv"),
                 os.path.join(reports, "shanghai2026_R128_predictions_cck_complete.csv"),
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
