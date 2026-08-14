#!/usr/bin/env python3
"""
Fix inflated streaks in backfill_last5.py.

BUG
---
Streaks are derived from scored prediction files, which are SPARSE. A match
CourtIQ never predicted, or predicted in a round never scored, is absent - and
a missing LOSS does not break a streak, so the backward walk counts through it.

Alcaraz: Monte Carlo is a 56 draw, so he had a bye and won R32/R16/QF/SF before
losing the final to Sinner. The final is not in the scored files. Add his
Barcelona R32 win and the walk counts 4 + 1 = 5. True streak is +1.

FIX
---
In a knockout draw a player either wins the title or loses exactly one match.
So a WIN streak may cross a tournament boundary only if the player won that
earlier event - if their last recorded match there is a win in any round but
the Final, an exit loss is missing and the streak breaks.

LOSS streaks are treated separately and more conservatively. Losses are the
matches that DO get recorded (they are the player's exit), so the failure mode
is the reverse: unrecorded WINS earlier in the same event. A loss streak may
therefore only cross a boundary when the loss is the player's ONLY recorded
match in the later event, i.e. plausibly a first-round exit.

LIMITATION: neither rule can see a first-round exit whose match is missing
entirely. This makes streaks conservative, not perfect.

Idempotent. DRY RUN by default.

Usage:
    python apply_streak_boundary_fix.py
    python apply_streak_boundary_fix.py --commit
"""
import sys, os, shutil, ast
from datetime import datetime

OLD_START = "seq = [r for _d, r, _k in rows]"
OLD_END   = 'streaks[player] = float(n if last_res == "W" else -n)'

NEW = '''        # Trailing streak, same ordering as last5 above.
        # The derived history is sparse, so a missing loss would otherwise let
        # a streak run straight through a tournament exit. In a knockout draw a
        # player either takes the title or loses exactly one match, which gives
        # a structural test for the gap.
        last_in_tourney = {}
        n_in_tourney = {}
        for _d, _res, (t, rnd) in rows:
            n_in_tourney[t] = n_in_tourney.get(t, 0) + 1
        for _d, res, (t, rnd) in rows:
            last_in_tourney[t] = (rnd, res)

        last_res = rows[-1][1]
        cur_t = rows[-1][2][0]
        n = 0
        for _d, res, (t, _rnd) in reversed(rows):
            if res != last_res:
                break
            if t != cur_t:
                if last_res == "W":
                    # may only continue if the earlier event ended in a title
                    prev_rnd, prev_res = last_in_tourney[t]
                    if not (prev_rnd == "F" and prev_res == "W"):
                        break
                else:
                    # may only continue if the later event was a lone loss,
                    # i.e. no unrecorded wins are implied before it
                    if n_in_tourney[cur_t] != 1:
                        break
                cur_t = t
            n += 1
        streaks[player] = float(n if last_res == "W" else -n)'''


def main():
    commit = "--commit" in sys.argv
    path = "backfill_last5.py"
    if "--file" in sys.argv:
        path = sys.argv[sys.argv.index("--file") + 1]
    if not os.path.exists(path):
        print(f"ABORT: {path} not found"); sys.exit(1)

    with open(path, "r", encoding="utf-8", newline="") as fh:
        raw = fh.read()
    nl = "\r\n" if "\r\n" in raw else "\n"
    lines = raw.split(nl)

    if "last_in_tourney" in raw:
        print("SKIP: boundary rule already present."); return

    starts = [i for i, L in enumerate(lines) if L.strip() == OLD_START]
    ends   = [i for i, L in enumerate(lines) if L.strip() == OLD_END]
    if len(starts) != 1 or len(ends) != 1:
        print(f"ABORT: expected 1 streak block, found {len(starts)} start / {len(ends)} end.")
        print("       Your file differs from the expected shape - send it over.")
        sys.exit(1)
    i, j = starts[0], ends[0]
    if j <= i:
        print("ABORT: streak block end precedes its start."); sys.exit(1)

    # keep the comment line above the block if present
    k = i - 1 if i > 0 and lines[i-1].strip().startswith("#") else i
    before = nl.join(lines[k:j+1])
    lines[k:j+1] = NEW.split("\n")

    out = nl.join(lines)
    if out == raw:
        print("ABORT: edit produced no change."); sys.exit(1)
    try:
        ast.parse(out)
    except SyntaxError as e:
        print(f"ABORT: patched file fails to parse -> {e}"); sys.exit(1)

    print(f"APPLY  streak boundary rule (lines {k+1}-{j+1})")
    print("\n--- replaced ---")
    print(before)
    print("\nSyntax check on patched source: OK")

    if not commit:
        print("\nDRY RUN - nothing written. Re-run with --commit.")
        return

    bak = f"{path}.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(path, bak)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(out)
    print(f"\nWROTE {path}  (backup: {bak})")
    print("\nNEXT:")
    print("  python backfill_last5.py            # check the diff table")
    print("  python backfill_last5.py --commit")
    print("  python courtiq_engine.py site --output docs/index.html")


if __name__ == "__main__":
    main()
