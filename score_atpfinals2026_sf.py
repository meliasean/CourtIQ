#!/usr/bin/env python3
"""
Score ATP Finals 2026 SF.

Both matches resolved. Winners derived from the Final: the two finalists are
necessarily the two SF winners, and each sits in the slot fed by its own SF
match - Zverev from SF 1, Alcaraz from SF 2 - which is what confirms the
round was read correctly. Unlike the round-robin rounds, this derivation is
valid, because the knockout bracket makes each finalist traceable to exactly
one semi-final.

Usage:  python score_atpfinals2026_sf.py [reports_dir]   (default: reports)
"""
import sys, os
import pandas as pd
import unicodedata


def al(n):
    return (unicodedata.normalize("NFKD", str(n))
            .encode("ascii", "ignore").decode("ascii")
            .strip().lower().replace("-", " "))


reports = sys.argv[1] if len(sys.argv) > 1 else "reports"

EXPECTED = 2

RESULTS = {
    ("Alexander Zverev", "Daniil Medvedev"): "Alexander Zverev",
    ("Carlos Alcaraz",   "Ben Shelton"):     "Carlos Alcaraz",
}

assert len(RESULTS) == EXPECTED, f"RESULTS holds {len(RESULTS)}, expected {EXPECTED}"
for (a, b), w in RESULTS.items():
    assert w in (a, b), f"winner {w!r} is not one of {a!r} / {b!r}"

# the two derived winners must be the two finalists, one from each SF
FINALISTS = {"alexander zverev", "carlos alcaraz"}
assert {al(w) for w in RESULTS.values()} == FINALISTS, \
    "derived SF winners do not match the reported final"
for f in FINALISTS:
    n = sum(1 for (a, b) in RESULTS if f in (al(a), al(b)))
    assert n == 1, f"{f} appears in {n} semi-finals, expected exactly 1"

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


print("Scoring ATP Finals 2026 SF...")
seen = set()

score_file(os.path.join(reports, "atpfinals2026_SF_predictions.csv"),
           os.path.join(reports, "atpfinals2026_SF_predictions_complete.csv"),
           seen)

cck = score_file(os.path.join(reports, "atpfinals2026_SF_predictions_cck.csv"),
                 os.path.join(reports, "atpfinals2026_SF_predictions_cck_complete.csv"),
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
    print("  -> check spelling against reports/atpfinals2026_SF_draw.csv "
          "before trusting the totals")
else:
    print(f"\nAll {EXPECTED} results matched.")
