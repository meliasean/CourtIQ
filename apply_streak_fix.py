#!/usr/bin/env python3
"""
Fix streak in update_profiles() so it carries across tournaments.

ROOT CAUSE
----------
    streak = 0
    for _, r in sub.sort_values("date").iterrows():
        ...

`sub` holds only the CURRENT event's matches and streak resets to 0 each run,
so the stored value is a per-tournament streak, not a career streak. A player
on a 7-match win run who wins 2 more is written as +2 instead of +9.

This seeds the loop from the player's existing stored streak, so an unbeaten
run continues and a loss correctly flips it.

Idempotent. DRY RUN by default.

Usage:
    python apply_streak_fix.py
    python apply_streak_fix.py --commit
"""
import sys, os, shutil, ast
from datetime import datetime

SEEDED = '''        # Seed from the player's stored streak so runs continue across
        # tournaments. Resetting to 0 here made this a per-event streak.
        streak = 0
        if not seed_row.empty:
            try:
                streak = int(float(seed_row.iloc[-1].get("streak") or 0))
            except Exception:
                streak = 0'''


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

    anchors = [i for i, L in enumerate(lines)
               if L.strip() == 'for _, r in sub.sort_values("date").iterrows():']
    if not anchors:
        print("ABORT: streak loop anchor not found"); sys.exit(1)
    if len(anchors) > 1:
        print(f"ABORT: {len(anchors)} anchors matched; expected 1"); sys.exit(1)

    i = anchors[0]

    # Idempotence check FIRST: after patching, the line above the loop is still
    # "streak = 0" (the except body), so shape alone cannot tell us if we ran.
    if 'seed_row.iloc[-1].get("streak"' in raw:
        print("SKIP: streak already seeded. Nothing to do."); return

    if i == 0 or lines[i-1].strip() != "streak = 0":
        print(f"ABORT: expected 'streak = 0' immediately above line {i+1}, found:")
        print(f"       {lines[i-1]!r}")
        sys.exit(1)

    # verify seed_row is defined before this point in the function
    ctx = nl.join(lines[max(0, i-15):i])
    if "seed_row" not in ctx:
        print("ABORT: seed_row not defined above the streak loop; "
              "cannot seed safely."); sys.exit(1)

    lines[i-1:i] = SEEDED.split("\n")
    out = nl.join(lines)

    try:
        ast.parse(out)
    except SyntaxError as e:
        print(f"ABORT: patched file fails to parse -> {e}"); sys.exit(1)

    print(f"APPLY  seed streak from stored profile value (line {i})")
    print("Syntax check on patched source: OK")

    if not commit:
        print("\nDRY RUN - nothing written. Re-run with --commit.")
        return

    bak = f"{path}.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(path, bak)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(out)
    print(f"\nWROTE {path}  (backup: {bak})")


if __name__ == "__main__":
    main()
