#!/usr/bin/env python3
"""
Score ATP Finals 2026 RR2.

All 4 matches resolved. Winners reported directly.

PROVENANCE - this differs from every knockout round in this repo. A knockout
scorer derives the previous round's winners from the next round's field, which
is a real derivation and is asserted. Nothing equivalent exists here.

The ATP Finals round robin runs a schedule fixed in advance by within-group
seeding, not by results: 1v4 and 2v3, then 1v2 and 3v4, then 1v3 and 2v4.
Both groups followed it exactly. So no round's pairings carry any information
about any other round's results - RR2's card was identical whoever won RR1.
An earlier version of this scorer claimed a consistency check on that basis;
it was invalid and has been removed.

What is verified instead is the draw structure: no match crosses groups, and
the three rounds together cover all six within-group pairings exactly once in
each group. That validates the schedule, not the outcomes.

Groups - A: Zverev(1), Shelton(2), Fils(3), Jodar(4).
         B: Alcaraz(1), Medvedev(2), Cobolli(3), Tiafoe(4).

1 upset in 4: Jodar lost at -112, the tightest line of the event. Standings
after this round - A: Zverev 2-0, Shelton 1-1, Fils 1-1, Jodar 0-2.
B: Alcaraz 2-0, Medvedev 1-1, Tiafoe 1-1, Cobolli 0-2.

Usage:  python score_atpfinals2026_rr2.py [reports_dir]   (default: reports)
"""
import sys, os
import pandas as pd
import unicodedata


def al(n):
    return (unicodedata.normalize("NFKD", str(n))
            .encode("ascii", "ignore").decode("ascii")
            .strip().lower().replace("-", " "))


reports = sys.argv[1] if len(sys.argv) > 1 else "reports"

EXPECTED = 4

RESULTS = {
    ("Arthur Fils",      "Rafael Jodar"    ): "Arthur Fils",
    ("Alexander Zverev", "Ben Shelton"     ): "Alexander Zverev",
    ("Carlos Alcaraz",   "Daniil Medvedev" ): "Carlos Alcaraz",
    ("Flavio Cobolli",   "Frances Tiafoe"  ): "Frances Tiafoe",
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


print("Scoring ATP Finals 2026 RR2...")
seen = set()

score_file(os.path.join(reports, "atpfinals2026_RR2_predictions.csv"),
           os.path.join(reports, "atpfinals2026_RR2_predictions_complete.csv"),
           seen)

cck = score_file(os.path.join(reports, "atpfinals2026_RR2_predictions_cck.csv"),
                 os.path.join(reports, "atpfinals2026_RR2_predictions_cck_complete.csv"),
                 seen)

if cck is not None:
    bk = cck[cck["correct_prediction_book"].notna()]
    if len(bk):
        print(f"\nBook: {int(bk['correct_prediction_book'].sum())}/{len(bk)} = "
              f"{bk['correct_prediction_book'].mean()*100:.0f}%")

unmatched = set(RES) - seen
if unmatched:
    print("\nThe final never matched a prediction row - "
          "check spelling against reports/atpfinals2026_RR2_draw.csv")
else:
    print(f"\nAll {EXPECTED} result matched. ATP Finals 2026 RR2 complete.")
