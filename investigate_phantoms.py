#!/usr/bin/env python3
"""
Identify the source match behind each suspected phantom player name.

CONTEXT
-------
Several names show the signature: 1 draw row + 4 prediction rows (std/cck x
pre-match/complete) = ONE real match occurrence, plus many profile rows. A
rebuild deriving profiles from match history will faithfully mint a profile
for a name that appears once in a draw, so these are typos in source data
rather than stale profile rows.

THE DECISIVE CHECK
------------------
For each suspect, find its match row(s), then ask whether the CANONICAL name
also appears in that same tournament and round.

  canonical ABSENT from that round  -> the suspect is almost certainly a
                                       misspelling of the canonical player
  canonical PRESENT in that round   -> they are different entries; renaming
                                       would merge two distinct matches, so
                                       investigate before touching it

READ ONLY. Prints full row context so the call is yours, not the script's.

Usage:
    python investigate_phantoms.py
    python investigate_phantoms.py "Some Name" "Canonical Name"
"""
import sys, os, glob, re
import pandas as pd
import unicodedata

def al(n):
    return unicodedata.normalize("NFKD", str(n)).encode("ascii","ignore").decode("ascii").strip().lower().replace("-"," ")

# suspect -> canonical it is probably a typo of
SUSPECTS = {
    "Aleksander Vukic":      "Aleksandar Vukic",
    "Guy Gen Ouden":         "Guy Den Ouden",
    "Jan Engel":             "Justin Engel",
    "Lorenzo Darderi":       "Luciano Darderi",
    "Luca Darderi":          "Luciano Darderi",
    "Ming Zheng":            "Michael Zheng",
    "Miroslav Topo":         "Marko Topo",
    "Nikolai Budov Kjaer":   "Nicolai Budkov Kjaer",
    "Oscar Virtanen":        "Otto Virtanen",
    "Alexander Shevchenko":  "Aleksandr Shevchenko",
}
if len(sys.argv) == 3:
    SUSPECTS = {sys.argv[1]: sys.argv[2]}

NAME_COLS = ["player_a","player_b","pred_winner","actual_winner","winner",
             "loser","player","player_name","name","opponent","pick_player"]
SHOW = ["date","tourney","tournament","round","match_no","player_a","player_b",
        "odds_a","odds_b","odds_player_a","odds_player_b","pred_winner",
        "correct_prediction","pick_player","bucket"]

def round_key(path):
    m = re.match(r"^(.+?)_([A-Za-z0-9]+)_predictions", os.path.basename(path))
    return (m.group(1), m.group(2)) if m else None

files = []
for root in ("reports", "live_log", "."):
    files += glob.glob(os.path.join(root, "*.csv"))
files = sorted(set(f for f in files if "player_profiles" not in os.path.basename(f).lower()))

print(f"Scanning {len(files)} non-profile CSV(s)\n" + "=" * 74)

for suspect, canon in SUSPECTS.items():
    print(f"\n{suspect!r}   (suspected typo of {canon!r})")
    hits, rounds = [], set()
    for path in files:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        cols = [c for c in NAME_COLS if c in df.columns]
        if not cols:
            continue
        m = pd.Series(False, index=df.index)
        for c in cols:
            m |= df[c].astype(str).map(al) == al(suspect)
        if not m.any():
            continue
        k = round_key(path)
        if k:
            rounds.add(k)
        for _, r in df[m].iterrows():
            hits.append((path, r))

    if not hits:
        print("   no rows outside profile files - nothing in match history to rename.")
        print("   If this name is in player_profiles_latest.csv it cannot have been")
        print("   derived from match data. Check whether a rebuild clears it.")
        continue

    seen = set()
    for path, r in hits:
        sig = tuple(str(r.get(c, "")) for c in ("date","player_a","player_b"))
        if sig in seen:
            continue
        seen.add(sig)
        bits = [f"{c}={r[c]}" for c in SHOW if c in r.index and pd.notna(r[c])]
        print(f"   [{os.path.basename(path)}]")
        print(f"     {' | '.join(bits)[:180]}")

    # decisive check
    print(f"   -- is {canon!r} also in those rounds? --")
    for (t, rnd) in sorted(rounds):
        found = False
        for path in files:
            if round_key(path) != (t, rnd):
                continue
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            for c in [c for c in NAME_COLS if c in df.columns]:
                if (df[c].astype(str).map(al) == al(canon)).any():
                    found = True
        verdict = ("PRESENT - different entries, do NOT merge without checking"
                   if found else
                   "ABSENT - consistent with a misspelling of this player")
        print(f"     {t} {rnd}: {verdict}")

print("\n" + "=" * 74)
print("Where the canonical name is ABSENT from the round, add the suspect to")
print("ALIASES in normalize_player_aliases.py. Where it is PRESENT, inspect the")
print("row first - it may be a field-alignment error like the Jiri Zhang case,")
print("in which case the opponent, date or result may also be wrong.")
