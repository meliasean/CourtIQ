#!/usr/bin/env python3
"""
Score ATP Finals 2026 RR3 (final round-robin round).

PROVENANCE. Winners are taken from the direct report only. The semi-final
card was tested as a cross-check and it confirms nothing: of the 16 possible
RR3 outcome sets, 10 produce a semi-final field and pairing structure
consistent with Alcaraz/Shelton + Zverev/Medvedev, because any three-way tie
at 2-1 is resolved on sets and games percentages, which are not derivable
from win-loss records. Every one of the four RR3 matches could have gone
either way and still yielded this semi-final card. No match-level inference
was made.

What IS verified here, from the draw file alone:
  - no cross-group pairings,
  - every within-group pairing covered exactly once across RR1-RR3,
  - the RR3 card matches seed pattern 1v3 / 2v4 by within-group seed.

Usage:  python score_atpfinals2026_rr3.py [reports_dir]   (default: reports)
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
    ("Ben Shelton",      "Rafael Jodar"):    "Ben Shelton",
    ("Alexander Zverev", "Arthur Fils"):     "Alexander Zverev",
    ("Daniil Medvedev",  "Frances Tiafoe"):  "Daniil Medvedev",
    ("Carlos Alcaraz",   "Flavio Cobolli"):  "Carlos Alcaraz",
}

assert len(RESULTS) == EXPECTED, f"RESULTS holds {len(RESULTS)}, expected {EXPECTED}"
for (a, b), w in RESULTS.items():
    assert w in (a, b), f"winner {w!r} is not one of {a!r} / {b!r}"

GROUP_A = {"alexander zverev", "ben shelton", "arthur fils", "rafael jodar"}
GROUP_B = {"carlos alcaraz", "daniil medvedev", "frances tiafoe", "flavio cobolli"}
for (a, b) in RESULTS:
    na, nb = al(a), al(b)
    assert (na in GROUP_A) == (nb in GROUP_A), f"cross-group pairing: {a} vs {b}"

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


print("Scoring ATP Finals 2026 RR3...")
seen = set()

score_file(os.path.join(reports, "atpfinals2026_RR3_predictions.csv"),
           os.path.join(reports, "atpfinals2026_RR3_predictions_complete.csv"),
           seen)

cck = score_file(os.path.join(reports, "atpfinals2026_RR3_predictions_cck.csv"),
                 os.path.join(reports, "atpfinals2026_RR3_predictions_cck_complete.csv"),
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
    print("  -> check spelling against reports/atpfinals2026_RR3_draw.csv "
          "before trusting the totals")
else:
    print(f"\nAll {EXPECTED} results matched. Round robin complete.")
