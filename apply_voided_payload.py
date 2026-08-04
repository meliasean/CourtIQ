#!/usr/bin/env python3
"""
Pass `voided` through to the site payload.

The match payload is built from an explicit whitelist at line ~1504:

    match_keys = ["match_no", "player_a", "player_b", ...]
    matches = json.loads(merged[[c for c in match_keys if c in merged.columns]]
                         .to_json(orient="records"))

`voided` was not on that list, so the column was dropped before serialization.
Every JS void check therefore read `undefined` and passed through - the picks
page kept rendering the card and the badge kept saying "pending". The stats
card worked because compute_pick_stats() reads the CSVs directly and never
goes through this payload.

Idempotent. DRY RUN by default.

Usage:
    python apply_voided_payload.py
    python apply_voided_payload.py --commit
"""
import sys, os, shutil, ast
from datetime import datetime

ANCHOR = '"p_elo_a", "p_temp_a"]'
REPLACE = '"p_elo_a", "p_temp_a", "voided"]'


def main():
    commit = "--commit" in sys.argv
    path = "courtiq_engine.py"
    if "--file" in sys.argv:
        path = sys.argv[sys.argv.index("--file") + 1]
    if not os.path.exists(path):
        print(f"ABORT: {path} not found"); sys.exit(1)

    with open(path, "r", encoding="utf-8", newline="") as fh:
        raw = fh.read()
    nl = "\r\n" if "\r\n" in raw else "\n"
    lines = raw.split(nl)

    if '"voided"]' in raw or '"voided",' in raw:
        print("SKIP: voided already in match_keys."); return

    hits = [i for i, L in enumerate(lines) if ANCHOR in L]
    if len(hits) != 1:
        print(f"ABORT: expected 1 match_keys terminator, found {len(hits)}")
        for i, L in enumerate(lines):
            if "match_keys" in L or "p_temp_a" in L:
                print(f"    line {i+1}: {L.strip()[:110]}")
        sys.exit(1)

    i = hits[0]
    lines[i] = lines[i].replace(ANCHOR, REPLACE)

    out = nl.join(lines)
    try:
        ast.parse(out)
    except SyntaxError as e:
        print(f"ABORT: patched file fails to parse -> {e}"); sys.exit(1)

    print(f"APPLY  'voided' added to match_keys (line {i+1})")
    print("       the whitelist is guarded by `if c in merged.columns`, so")
    print("       rounds whose CSVs have no voided column are unaffected")
    print("Syntax check on patched source: OK")

    if not commit:
        print("\nDRY RUN - nothing written. Re-run with --commit.")
        return

    bak = f"{path}.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(path, bak)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(out)
    print(f"\nWROTE {path}  (backup: {bak})")
    print("\nNEXT:  python courtiq_engine.py site --output docs/index.html")
    print('       grep -c \'"voided"\' docs/index.html   # should be > 0')


if __name__ == "__main__":
    main()
